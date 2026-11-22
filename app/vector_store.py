"""向量 RAG 完整版：法条切块 -> 向量入库 -> 向量检索 -> 大模型照着法条作答。

整个文件只有两个动作：
    python -m app.vector_store            增量入库（data/raw 里所有法律 md）
    python -m app.vector_store 你的问题    法律咨询：检索法条 + 大模型作答
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
import uuid

# 本机网络设置（必须放在建立连接之前）：
# Clash 系统代理会劫持发往 localhost 的请求（回 502），本机地址要直连；
# HuggingFace 国内直连超时，下载模型走国内镜像站。
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,::1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from pathlib import Path

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.config import (
    COLLECTION_NAME,
    DATA_DIR,
    EMBED_DIM,
    EMBED_MODEL,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    QDRANT_URL,
)

# 两个全局对象，全文件共用：
model = TextEmbedding(EMBED_MODEL)                            # 向量化模型（首次运行自动下载）
client = QdrantClient(QDRANT_URL, check_compatibility=False)  # 连接 Docker 里的 Qdrant


def split_blocks(path: Path) -> list[dict]:
    """读一部法律的 md，按 '### 第X条' 切成一块一块，每块带法律名。

    每块长这样：
        {"text": "### 第十九条\n\n劳动合同期限三个月以上...", "law": "劳动合同法"}

    law 取自文件名（"劳动合同法.md" -> "劳动合同法"）。
    为什么要有它：两部法律都有"第十九条"，只靠条文号分不清是谁的，
    出处必须长到 "法律名 > 条文号" 两级。
    """
    law_name = path.stem             # 文件名去掉 .md，就是法律名
    blocks = []
    current: list[str] = []          # 攒"当前这一条"的所有行
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("### "):  # 看到 ### = 新的一条开始了
            if current:              # 先把上一条收尾
                text = "\n".join(current).strip()
                if text:
                    blocks.append({"text": text, "law": law_name})
            current = [line]         # 新的一条从这行开始攒
        else:
            current.append(line)
    if current:                      # 最后一条后面没有 ### 帮它收尾，手动补
        text = "\n".join(current).strip()
        if text:
            blocks.append({"text": text, "law": law_name})
    return blocks


def embed(texts: list[str]) -> list[list[float]]:
    """把一批文本各变成一个 512 维的数字列表（向量）。

    例子：embed(["试用期不得超过六个月"]) -> [[0.021, -0.113, ...]]
    意思相近的文本，向量就挨得近——"按意思找"的原理全在这。
    """
    return [list(v) for v in model.embed(texts)]


def point_id(law: str, text: str) -> str:
    """由'法律名+条文号'算出稳定 id：同一条法条永远算出同一个 id。

    例子："劳动法 > 第二十一条" 不管入库多少次、排在第几个入库，
    算出的 id 都相同——id 成了法条的身份证。
    有了它才能做增量更新：库里认得出"这条我已经存过、没变过"。
    """
    article = text.splitlines()[0].removeprefix("### ")
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{law} > {article}"))


def ingest() -> int:
    """增量入库：对比本地 md 与库里的现状，只处理有变化的部分。

    每块先算稳定 id，然后分四种情况：
    - 新条（库里没这个 id）        -> 向量化 + 存入
    - 没变的条（id 有、文本相同）  -> 跳过（不重复向量化，这是省时间的核心）
    - 改过的条（id 有、文本不同）  -> 重新向量化，upsert 按 id 覆盖旧向量
    - 已删的条（库里有、本地 md 没有）-> 从库里删除
    返回本次实际向量化写入的块数。
    """
    # 1. 本地：切出所有块，按稳定 id 归档（同一条永远算出同一个 id）
    blocks = []
    for path in sorted(Path(DATA_DIR).glob("*.md")):
        blocks.extend(split_blocks(path))
    blocks_by_id = {point_id(c["law"], c["text"]): c for c in blocks}

    # 2. 建集合（只在第一次运行时；已存在则不动，里面旧数据保留）
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )

    # 3. 库里：翻出现有全部点的 id 和文本，作为对比基准
    existing: dict[str, str] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=256,
            offset=offset,
            with_payload=True,
        )
        for p in points:
            existing[p.id] = p.payload["text"]
        if offset is None:
            break

    # 4. 对比出"要写的"（新条+改条）和"要删的"（本地已没有的条文）
    to_upsert = [
        (pid, c) for pid, c in blocks_by_id.items()
        if pid not in existing or existing[pid] != c["text"]
    ]
    removed = [pid for pid in existing if pid not in blocks_by_id]
    if removed:
        client.delete(COLLECTION_NAME, points_selector=removed)

    # 5. 只向量化要写的块，upsert 按 id 存（改条自动覆盖旧向量）
    vectors = embed([c["text"] for _, c in to_upsert])
    points = [
        PointStruct(id=pid, vector=v, payload={"text": c["text"], "law": c["law"]})
        for (pid, c), v in zip(to_upsert, vectors)
    ]
    if points:
        client.upsert(COLLECTION_NAME, points=points)
    return len(points)


def search(question: str, top_k: int = 3) -> list[dict]:
    """检索：问题变向量 -> 找最近的 top_k 块 -> 返回结果列表（不打印）。

    每条结果长这样：
        {"law": "劳动合同法", "article": "第八十七条", "text": "### 第八十七条...", "score": 0.72}
    供 answer() 拼提示词用；想自己看原始命中结果，调它后自己打印。
    """
    qvec = embed([question])[0]
    hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=qvec,
        limit=top_k,
        with_payload=True,   # 把入库时存的原文和法律名带回来
    )
    results = []
    for h in hits.points:
        text = h.payload["text"]
        results.append({
            "law": h.payload["law"],
            "article": text.splitlines()[0].removeprefix("### "),
            "text": text,
            "score": h.score,
        })
    return results


def ask_llm(prompt: str) -> str:
    """调 OpenAI 兼容接口：把提示词发给大模型，返回回答文本。

    所谓"调大模型"就是一个 HTTP POST 三件套：
    发到哪：{base_url}/chat/completions
    发什么：{"model": 模型名, "messages": [{"role": "user", "content": 提示词}]}
    敲门砖：请求头 Authorization: Bearer <key>
    回答藏在 resp["choices"][0]["message"]["content"]。
    """
    req = urllib.request.Request(
        f"{LLM_BASE_URL.rstrip('/')}/chat/completions",
        data=json.dumps({
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
        }).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read())
    return body["choices"][0]["message"]["content"]


PROMPT_TEMPLATE = """你是一个法律咨询助手。请只依据下面的法律条文回答问题。

要求：
1. 引用条文时注明出处（如：劳动合同法第八十七条）
2. 检索到的条文中没有依据的内容，如实说明，不要编造
3. 用通俗的话给出结论，再列依据

法律条文材料：
{context}

问题：{question}
"""


def answer(question: str) -> None:
    """法律咨询问答：检索 -> 拼提示词 -> 调大模型 -> 打印依据和回答。

    RAG 的最后一块拼图：search 负责"找到法条"，answer 负责
    "让大模型照着法条说话"——模型没变，变的是卷子上夹了什么材料。
    没配 LLM_API_KEY 时降级：只打印拼好的提示词，手动粘给任意 AI 效果相同。
    """
    # 1. 检索：找最相关的几条法条
    hits = search(question)

    # 2. 把法条拼成材料段，出处用【法律名 条文号】标注
    context = "\n\n".join(f"【{h['law']} {h['article']}】\n{h['text']}" for h in hits)

    # 3. 按模板拼提示词（出一张"开卷考试"的卷子）
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    # 4. 配了 key 就真调接口，没配就降级打印卷子
    print("检索到的依据：")
    for h in hits:
        print(f"  【{h['law']} > {h['article']}】相似度 {h['score']:.4f}")
    if not LLM_API_KEY:
        print("\n【未配置 LLM_API_KEY，降级模式】把下面的提示词手动粘给任意 AI 即可：\n")
        print(prompt)
        return
    reply = ask_llm(prompt)
    print(f"\n===== 咨询回答 =====\n{reply}")


if __name__ == "__main__":
    if len(sys.argv) > 1:    # 带了问题 -> 法律咨询问答
        answer("".join(sys.argv[1:]))
    else:                    # 不带参数 -> 入库
        written = ingest()
        print(f"入库完成：本次向量化写入 {written} 块（没变的块原地不动，跳过向量化）。")
