"""知识一致性维护：法条更新后，自动重生成受影响的笔记。

    python -m app.sync

背景：笔记是模型从多条法条整合出来的"新知识"，和法条之间没有
一对一的副本关系；但它的内容依赖法条——法条改了，旧笔记就过时了。
所以保存笔记时在文件末尾记一行：

    （依据条文：面包屑@指纹、面包屑@指纹）

指纹是条文文本的短哈希，等于"我依据的条文当时长什么样"。
有了这行记录，本模块就能做后台的批量同步：

1. 重新切块 data/raw/，算出每条法条的当前指纹
2. 逐条检查笔记：依据里任一指纹对不上（条文改过）或条文已不在
   （被删）→ 该笔记过期
3. 对过期笔记：用它记录的来源问题重新检索作答（只用原材料，
   见 generate_answer 的 exclude_notes），覆盖笔记文件
4. 最后跑一次入库，让向量库与笔记文件重新一致

法条变 → 反查依据记录找到受影响笔记 → 重新生成覆盖。
这就是"知识库更新，之前的答案也跟着更新"的闭环。
"""

from __future__ import annotations

import re
from pathlib import Path

from app.ask import generate_answer
from app.common import fingerprint
from app.config import DATA_DIR, NOTES_DIR
from app.ingest import ingest, split_blocks

REF_PATTERN = re.compile(r"（依据条文：([^）]*)）")
QUESTION_PATTERN = re.compile(r"（来源问题：([^）]*)）")


def parse_note(path: Path) -> dict | None:
    """解析一条笔记：标题、来源问题、依据列表 [(面包屑, 当时指纹)]。

    没有"依据条文"行的老笔记无法判断过期，返回 None（跳过）。
    """
    text = path.read_text(encoding="utf-8")
    q = QUESTION_PATTERN.search(text)
    refs = REF_PATTERN.search(text)
    if not q or not refs:
        return None
    deps = []
    for item in refs.group(1).split("、"):
        item = item.strip()
        if "@" in item:
            breadcrumb, fp = item.rsplit("@", 1)
            deps.append((breadcrumb.strip(), fp.strip()))
    return {"path": path, "title": path.stem,
            "question": q.group(1).strip(), "deps": deps}


def current_fingerprints() -> dict[str, str]:
    """重新切块 data/raw/，返回 {面包屑: 当前文本指纹}。

    和入库同一个切分函数——法条在库里的身份是面包屑，
    这里算出的指纹必须和入库时写进 payload 的文本对得上。
    """
    result = {}
    for path in sorted(Path(DATA_DIR).glob("*.md")):
        for block in split_blocks(path):
            result[block["breadcrumb"]] = fingerprint(block["text"])
    return result


def main() -> None:
    current = current_fingerprints()

    # 1. 逐条笔记对账：依据的条文，现在还是不是原来的样子
    stale = []
    for path in sorted(Path(NOTES_DIR).glob("*.md")):
        note = parse_note(path)
        if note is None:
            print(f"- {path.stem}：没有依据记录（旧格式笔记），跳过")
            continue
        changed = [bc for bc, fp in note["deps"] if current.get(bc) != fp]
        if changed:
            stale.append(note)
            print(f"- {path.stem}：依据已变化 {', '.join(changed)}")
        else:
            print(f"- {path.stem}：依据未变化")

    if not stale:
        print("同步完成：没有过期的笔记。")
        return

    # 2. 重新生成过期笔记：用来源问题重新问一遍，只用原材料法条
    for note in stale:
        reply, sources = generate_answer(note["question"], exclude_notes=True)
        if not reply:
            print(f"  {note['title']}：重新生成失败，保留原笔记")
            continue
        refs = "、".join(f"{s['breadcrumb']}@{fingerprint(s['text'])}" for s in sources)
        note["path"].write_text(
            f"# {note['title']}\n\n{reply}\n\n（来源问题：{note['question']}）\n"
            f"（依据条文：{refs}）\n",
            encoding="utf-8",
        )
        print(f"  已重新生成：{note['title']}（新依据 {len(sources)} 条）")

    # 3. 入库：让向量库里这条笔记的向量跟上新文本
    written = ingest()
    print(f"同步完成：重新生成 {len(stale)} 条笔记，本次向量化写入 {written} 块。")


if __name__ == "__main__":
    main()