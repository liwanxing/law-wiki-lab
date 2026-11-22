"""法律咨询问答：向量检索法条 -> 拼提示词 -> 大模型照着法条作答。

    python -m app.ask 你的问题
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request

from app.common import client, embed, fingerprint
from app.config import (COLLECTION_NAME, LLM_API_KEY, LLM_BASE_URL,
                        LLM_MODEL, NOTES_DIR, RELEVANCE_THRESHOLD)
from app.ingest import ingest

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
        results.append({
            "breadcrumb": h.payload["breadcrumb"],   # 如：劳动法 > 第五章 工资 > 第四十八条
            "text": h.payload["text"],
            "links": h.payload.get("links", []),      # 这条引用了哪些别的条文
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


def expand_links(hits: list[dict]) -> list[dict]:
    """链接扩展：顺着检索结果的双链，把被引用条文的原文捞回来。

    为什么值得做：87 条说"依照本法第四十七条标准的二倍"——
    47 条（经济补偿计算标准）是理解 87 条的前提，但 47 条
    和"开除赔偿"字面不相似，向量检索很可能捞不到它。
    双链在入库时已存好，这里只需按目标地址去库里捞原文。

    匹配逻辑：双链是短地址（法名 > 条文号），面包屑是完整地址
    （法名 > 章 > 条文号）——首段和末段都对上就是同一条文。
    返回扩展块列表（score 记 0.0，表示不是直接命中而是顺链来的）。
    """
    # 1. 收集所有双链目标（多命中指向同一目标时自动去重）
    targets = set()
    for h in hits:
        targets.update(h["links"])
    if not targets:
        return []

    # 2. 翻一遍库，把命中的目标条文捞出来（数据在 payload 里，不用读原文）
    extra = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=256,
            offset=offset,
            with_payload=True,
        )
        for p in points:
            bc = p.payload["breadcrumb"]
            short = bc.split(" > ")[0] + " > " + bc.split(" > ")[-1]  # 法名 + 条文号
            if short in targets:
                extra.append({
                    "breadcrumb": bc,
                    "text": p.payload["text"],
                    "links": p.payload.get("links", []),
                    "score": 0.0,
                })
                targets.discard(short)   # 已找到，不再重复捞
        if offset is None or not targets:
            break
    return extra


def generate_answer(question: str, verbose: bool = False,
                    exclude_notes: bool = False) -> tuple[str, list[dict]]:
    """生成回答（不交互）：检索 -> 闸门 -> 扩展 -> 拼提示词 -> 调模型。

    返回 (回答文本, 依据列表)。answer()（交互问答）和
    sync.py（法条更新后的批量重生成）共用同一套生成逻辑。

    exclude_notes=True 时剔除笔记块（一级面包屑，如"违法辞退赔偿"）：
    同步重生成必须只用原材料（法条），不能拿旧笔记当依据——
    否则检索第一名永远是旧笔记自己，生成出来还是旧笔记的复述，
    永远吸收不到法条的更新。
    """
    hits = search(question)
    if exclude_notes:
        hits = [h for h in hits if " > " in h["breadcrumb"]]

    # 阈值闸门：最高分低于阈值 = 库里没有相关法条
    if not hits or hits[0]["score"] < RELEVANCE_THRESHOLD:
        if verbose:
            print("库里没有找到与这个问题相关的法条，无法回答。")
            print("可以换个说法重试，或往 data/raw/ 里扩充法律库。")
        return "", []

    # 链接扩展：顺着双链把被引用的条文也拉进卷子
    linked = expand_links(hits)
    context = "\n\n".join(f"【{h['breadcrumb']}】\n{h['text']}" for h in hits + linked)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    if verbose:
        print("检索到的依据：")
        for h in hits:
            line = f"  【{h['breadcrumb']}】相似度 {h['score']:.4f}"
            if h["links"]:
                line += f"  双链→ {'、'.join(h['links'])}"
            print(line)
        if linked:
            print("顺双链扩展进材料的条文：")
            for h in linked:
                print(f"  【{h['breadcrumb']}】")

    reply = ask_llm(prompt)
    return reply, hits + linked


def save_note(question: str, reply: str, sources: list[dict]) -> None:
    """知识回流：把整理出的回答和它的依据一起存成笔记文件，并入库。

    "加"的原理：一个新标题 = 一个新文件 = 一级面包屑 = 一个新稳定 id，
    对增量入库来说就是一条"新条"——ingest() 认得它，自动向量化写入。
    （同标题重存就是"更新"：id 不变、文本变了，增量机制按"改条"覆盖。）

    sources 是本次生成依赖的条文，存成文件末尾一行：
        （依据条文：面包屑@指纹、面包屑@指纹）
    指纹 = 条文当时的文本哈希，等于"我依据的条文当时长什么样"。
    法条以后变了，sync.py 靠这行记录反查"哪些笔记该重做"。
    """
    try:
        if input("\n把这条整理存进知识库？(y = 存，直接回车 = 跳过)：").strip().lower() != "y":
            return
        title = input("笔记标题（如：违法辞退赔偿）：").strip()
    except EOFError:             # stdin 已关闭（脚本化管道等场景），静默跳过
        return
    title = re.sub(r'[\\/:*?"<>|]', "-", title)   # Windows 文件名非法字符换成 -
    if not title:
        print("未提供标题，跳过。")
        return
    NOTES_DIR.mkdir(exist_ok=True)
    path = NOTES_DIR / f"{title}.md"
    # 标题就是知识的主键：同名文件已存在 = 同一主题的知识被修订 = "更新"；
    # 新标题 = 新主题 = "加"。加/更新不是两个功能，是同一个写文件动作的两种结果。
    action = "更新已有笔记" if path.exists() else "新增笔记"
    lines = [f"# {title}", "", reply, "", f"（来源问题：{question}）"]
    # 依据只记法条（多级面包屑）。笔记自己也可能被检索命中（一级面包屑），
    # 把自己记成依据会让 sync 永远判"过期"——它不在法条指纹表里。
    law_sources = [s for s in sources if " > " in s["breadcrumb"]]
    if law_sources:
        refs = "、".join(f"{s['breadcrumb']}@{fingerprint(s['text'])}" for s in law_sources)
        lines.append(f"（依据条文：{refs}）")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    written = ingest()   # 复用入库流水线：新标题=新id=新增；同标题=同id+新文本=覆盖更新
    print(f"已{action} {path.name}，入库完成：本次向量化写入 {written} 块。")


def answer(question: str) -> None:
    """法律咨询问答：生成回答 -> 打印 -> 询问是否回流知识库。"""
    reply, sources = generate_answer(question, verbose=True)
    if not reply:
        return
    print(f"\n===== 咨询回答 =====\n{reply}")

    # 知识回流：答得好就存回知识库（连同依据条文一起记下，供同步用）
    save_note(question, reply, sources)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        answer("".join(sys.argv[1:]))
    else:
        print("用法：python -m app.ask 你的问题")
