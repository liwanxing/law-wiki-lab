"""法律咨询问答：向量检索法条 -> 拼提示词 -> 大模型照着法条作答。

    python -m app.ask 你的问题
"""

from __future__ import annotations

import json
import sys
import urllib.request

from app.common import client, embed
from app.config import COLLECTION_NAME, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, RELEVANCE_THRESHOLD

PROMPT_TEMPLATE = """你是一个法律咨询助手。请严格只依据下面的法律条文材料回答问题。

硬性约束：
1. 只准引用材料中【】标注的条文；材料里没有的条文一律不得提及——
   哪怕你自己的知识里知道，也不准用
2. 如果材料不足以回答问题，明确说"检索到的条文中没有回答这个问题所需的依据"
3. 引用格式照抄【】里的出处（如：劳动合同法第八十七条）
4. 用通俗的话给出结论，再列依据

法律条文材料：
{context}

问题：{question}
"""


def search(question: str, top_k: int = 3) -> list[dict]:
    """检索：问题变向量 -> 找最近的 top_k 块 -> 返回结果列表（不打印）。

    每条结果长这样：
        {"law": "劳动合同法", "article": "第八十七条", "text": "### 第八十七条...", "score": 0.72}
    供 answer() 拼提示词用。
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


def answer(question: str) -> None:
    """法律咨询问答：检索 -> 拼提示词 -> 调大模型 -> 打印依据和回答。

    RAG 的最后一块拼图：search 负责"找到法条"，answer 负责
    "让大模型照着法条说话"——模型没变，变的是卷子上夹了什么材料。
    """
    # 1. 检索：找最相关的几条法条
    hits = search(question)

    # 2. 阈值闸门：最高分低于阈值 = 库里没有相关法条。
    #    不把弱相关材料喂给模型——省一次调用，也杜绝模型硬答。
    #    hits 按 score 从高到低排列，所以看第一个就知道整体质量。
    if not hits or hits[0]["score"] < RELEVANCE_THRESHOLD:
        print("库里没有找到与这个问题相关的法条，无法回答。")
        print("可以换个说法重试，或往 data/raw/ 里扩充法律库。")
        return

    # 3. 把法条拼成材料段，出处用【法律名 条文号】标注
    context = "\n\n".join(f"【{h['law']} {h['article']}】\n{h['text']}" for h in hits)

    # 4. 按模板拼提示词（出一张"开卷考试"的卷子）
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    # 5. 调大模型作答，先打印检索依据（让人能核对答案是不是照材料说的）
    print("检索到的依据：")
    for h in hits:
        print(f"  【{h['law']} > {h['article']}】相似度 {h['score']:.4f}")
    reply = ask_llm(prompt)
    print(f"\n===== 咨询回答 =====\n{reply}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        answer("".join(sys.argv[1:]))
    else:
        print("用法：python -m app.ask 你的问题")
