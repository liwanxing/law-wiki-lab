"""入库流水线：读法律 md -> 切块 -> 向量化 -> 增量存进 Qdrant。

    python -m app.ingest
"""

from __future__ import annotations

import uuid
from pathlib import Path

from qdrant_client.models import Distance, PointStruct, VectorParams

from app.common import client, embed
from app.config import COLLECTION_NAME, DATA_DIR, EMBED_DIM


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


if __name__ == "__main__":
    written = ingest()
    print(f"入库完成：本次向量化写入 {written} 块（没变的块原地不动，跳过向量化）。")
