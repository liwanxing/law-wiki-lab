"""最简版向量 RAG：法条切块 -> 变向量 -> 存进 Qdrant -> 查向量。

整个文件只有两个动作：
    python -m app.vector_store            把 data/raw 里所有法律 md 入库
    python -m app.vector_store 你的问题    查最相关的法条
"""

from __future__ import annotations

import os
import sys

# 本机网络设置（必须放在建立连接之前）：
# Clash 系统代理会劫持发往 localhost 的请求（回 502），本机地址要直连；
# HuggingFace 国内直连超时，下载模型走国内镜像站。
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,::1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from pathlib import Path

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.config import COLLECTION_NAME, DATA_DIR, EMBED_DIM, EMBED_MODEL, QDRANT_URL

# 两个全局对象，全文件共用：
model = TextEmbedding(EMBED_MODEL)                            # 向量化模型（首次运行自动下载）
client = QdrantClient(QDRANT_URL, check_compatibility=False)  # 连接 Docker 里的 Qdrant


def split_blocks(text: str) -> list[str]:
    """把法律 md 按 '### 第X条' 切成一块一块，一块就是一条法条。

    例子：下面这种 md 会切成两个块——
        ### 第十九条
        劳动合同期限三个月以上不满一年的，试用期不得超过一个月...

        ### 第二十条
        劳动者在试用期的工资不得低于...
    """
    blocks = []
    current: list[str] = []          # 攒"当前这一条"的所有行
    for line in text.splitlines():
        if line.startswith("### "):  # 看到 ### = 新的一条开始了
            if current:              # 先把上一条收尾
                blocks.append("\n".join(current).strip())
            current = [line]         # 新的一条从这行开始攒
        else:
            current.append(line)
    if current:                      # 最后一条后面没有 ### 帮它收尾，手动补
        blocks.append("\n".join(current).strip())
    return [b for b in blocks if b]


def embed(texts: list[str]) -> list[list[float]]:
    """把一批文本各变成一个 512 维的数字列表（向量）。

    例子：embed(["试用期不得超过六个月"]) -> [[0.021, -0.113, ...]]
    意思相近的文本，向量就挨得近——"按意思找"的原理全在这。
    """
    return [list(v) for v in model.embed(texts)]


def ingest() -> int:
    """入库三步：读文件切块 -> 每块变向量 -> 存进 Qdrant。返回入库块数。"""
    # 1. 读 data/raw 下所有法律 md，切成块（以后加新法律，把 md 放进目录就行）
    blocks = []
    for path in sorted(Path(DATA_DIR).glob("*.md")):
        blocks.extend(split_blocks(path.read_text(encoding="utf-8")))

    # 2. 建集合：先删旧的再建，保证每次入库从零开始（学习期可复现）
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )

    # 3. 每块变向量，连同原文一起存进 Qdrant
    vectors = embed(blocks)
    points = [
        PointStruct(id=i, vector=v, payload={"text": t})
        for i, (t, v) in enumerate(zip(blocks, vectors), start=1)
    ]
    client.upsert(COLLECTION_NAME, points=points)
    return len(points)


def search(question: str, top_k: int = 3) -> None:
    """查询三步：问题变向量 -> 找最近的 top_k 块 -> 打印出处、相似度和原文。"""
    qvec = embed([question])[0]
    hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=qvec,
        limit=top_k,
        with_payload=True,   # 把入库时存的原文带回来
    )
    for h in hits.points:
        text = h.payload["text"]
        # 出处就藏在块的首行（入库时每个块都以 "### 第X条" 开头），
        # 去掉 "### " 前缀得到干净的条文号，如 "### 第十九条" -> "第十九条"
        source = text.splitlines()[0].removeprefix("### ")
        print(f"\n【{source}】相似度 {h.score:.4f}")
        print(text)


if __name__ == "__main__":
    if len(sys.argv) > 1:    # 带了问题 -> 查询
        search("".join(sys.argv[1:]))
    else:                    # 不带参数 -> 入库
        print(f"入库完成：{ingest()} 个条文块已存进 Qdrant。")
