"""入库流水线：读法律 md -> 切块 -> 向量化 -> 增量存进 Qdrant。

    python -m app.ingest
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path

from qdrant_client.models import Distance, PointStruct, VectorParams

from app.common import client, embed
from app.config import COLLECTION_NAME, DATA_DIR, EMBED_DIM, NOTES_DIR

# 双链提取：条文里的“本法第X条”引用。中文数字条文号，一条法内唯一。
LINK_PATTERN = re.compile(r"本法(第[一二三四五六七八九十百零]+条)")


def extract_links(text: str, law_name: str) -> list[str]:
    """提取条文里的双链：“本法第X条”引用 → 目标地址（法名 > 条文号）。

    法律条文天生是一张引用网：第八十七条写着“依照本法第四十七条”，
    问 87 条时自然该想到把 47 条也带上——双链就是把这种引用关系
    显式存下来，供以后的检索扩展用（链接扩展层）。
    返回去重后的目标列表，短格式（不含章）——条文号在法内唯一，够定位。
    """
    links = []
    for article in LINK_PATTERN.findall(text):
        target = f"{law_name} > {article}"
        if target not in links:
            links.append(target)
    return links


def split_blocks(path: Path, with_links: bool = True) -> list[dict]:
    """读一部法律的 md，按 '### 第X条' 切成一块一块，每块带面包屑。

    面包屑 = 这条法条在文档里的位置路径，由它头顶上的标题拼成：
        "劳动法 > 第七章 女职工和未成年工特殊保护 > 第六十二条"

    原理还是逐行扫描，只是多了一份"当前位置"的记录：
    见到 # 换法名、见到 ## 换章名、见到 ### 收一条——
    每条收尾时，面包屑 = 头顶的 # ## 标题 + 这条的条文号（### 行）。
    面包屑必须唯一到条文级，否则同章各条算出同一个 id，会互相覆盖。

    with_links=False 时不提双链（整理笔记里的"本法"没有明确指向，
    提出来的链会指错地方，所以笔记块一律 links 为空）。
    """
    law_name = path.stem
    breadcrumb = [law_name]            # 当前位置：第 0 层固定用文件名（法名）
    blocks = []
    current: list[str] = []            # 攒"当前这一条"的所有行
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("### "):    # 看到 ### = 新的一条开始
            if current:                # 收上一条
                text = "\n".join(current).strip()
                if text:
                    parts = list(breadcrumb)
                    if current[0].startswith("### "):
                        parts.append(current[0][4:].strip())  # 条文号补进面包屑末级
                    blocks.append({"text": text, "breadcrumb": " > ".join(parts),
                                   "links": extract_links(text, law_name) if with_links else []})
            current = [line]
        elif line.startswith("## "):   # 换章：更新位置第 1 层（章标题不进条文内容）
            breadcrumb = [law_name, line[3:].strip()]
        elif line.startswith("# "):    # 换法名：重置位置（第 0 层仍用文件名，保持简洁）
            breadcrumb = [law_name]
        else:
            current.append(line)
    if current:                        # 最后一条后面没有 ### 帮它收尾，手动补
        text = "\n".join(current).strip()
        if text:
            parts = list(breadcrumb)
            if current[0].startswith("### "):
                parts.append(current[0][4:].strip())      # 条文号补进面包屑末级
            blocks.append({"text": text, "breadcrumb": " > ".join(parts),
                           "links": extract_links(text, law_name) if with_links else []})
    return blocks


def point_id(breadcrumb: str) -> str:
    """由面包屑算出稳定 id：同一位置的法条永远算出同一个 id。

    面包屑（法名 > 章 > 条）就是法条的唯一地址——地址不变 id 不变。
    例子："劳动法 > 第五章 工资 > 第四十八条" 不管入库多少次都算出同一个 id。
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, breadcrumb))


def ingest() -> int:
    """增量入库：对比本地 md 与库里的现状，只处理有变化的部分。

    每块先算稳定 id，然后分四种情况：
    - 新条（库里没这个 id）        -> 向量化 + 存入
    - 没变的条（id 有、要存的数据全相同）-> 跳过（不重复向量化，这是省时间的核心）
    - 改过的条（id 有、数据不同）  -> 重新向量化，upsert 按 id 覆盖旧向量
    - 已删的条（库里有、本地 md 没有）-> 从库里删除
    返回本次实际向量化写入的块数。
    """
    # 1. 本地：切出所有块（法条 + 整理笔记），按稳定 id 归档
    blocks = []
    for path in sorted(Path(DATA_DIR).glob("*.md")):
        blocks.extend(split_blocks(path))                    # 法条："本法"有明确指向
    NOTES_DIR.mkdir(exist_ok=True)                           # 顺手建目录，首次运行也不报错
    for path in sorted(Path(NOTES_DIR).glob("*.md")):
        blocks.extend(split_blocks(path, with_links=False))  # 笔记："本法"无指向，不提双链
    blocks_by_id = {point_id(c["breadcrumb"]): c for c in blocks}

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
            # 对比基准不只是文本，而是要存的全部数据（文本/面包屑/双链）——
            # 否则块文本没变但双链变了时，增量机制发现不了，库就旧了
            existing[p.id] = (p.payload["text"], p.payload["breadcrumb"],
                              tuple(p.payload.get("links", [])))
        if offset is None:
            break

    # 4. 对比出"要写的"（新条+改条）和"要删的"（本地已没有的条文）
    to_upsert = [
        (pid, c) for pid, c in blocks_by_id.items()
        if pid not in existing
        or existing[pid] != (c["text"], c["breadcrumb"], tuple(c["links"]))
    ]
    removed = [pid for pid in existing if pid not in blocks_by_id]
    if removed:
        client.delete(COLLECTION_NAME, points_selector=removed)

    # 5. 只向量化要写的块，upsert 按 id 存（改条自动覆盖旧向量）
    vectors = embed([c["text"] for _, c in to_upsert])
    points = [
        PointStruct(id=pid, vector=v,
                    payload={"text": c["text"], "breadcrumb": c["breadcrumb"],
                             "links": c["links"]})
        for (pid, c), v in zip(to_upsert, vectors)
    ]
    if points:
        client.upsert(COLLECTION_NAME, points=points)
    return len(points)


if __name__ == "__main__":
    written = ingest()
    print(f"入库完成：本次向量化写入 {written} 块（没变的块原地不动，跳过向量化）。")
