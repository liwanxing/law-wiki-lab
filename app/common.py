"""共享资源：向量化模型、Qdrant 连接、embed 函数。

入库（ingest.py）和问答（ask.py）都要用的东西放这里，
两个模块都从这 import——模型实例和数据库连接全项目只建一份。
"""

from __future__ import annotations

import os

# 本机网络设置（必须放在建立连接之前）：
# Clash 系统代理会劫持发往 localhost 的请求（回 502），本机地址要直连；
# HuggingFace 国内直连超时，下载模型走国内镜像站。
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,::1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from fastembed import TextEmbedding
from qdrant_client import QdrantClient

from app.config import EMBED_MODEL, QDRANT_URL

# 两个全局对象，全项目共用：
model = TextEmbedding(EMBED_MODEL)                            # 向量化模型（首次运行自动下载）
client = QdrantClient(QDRANT_URL, check_compatibility=False)  # 连接 Docker 里的 Qdrant


def embed(texts: list[str]) -> list[list[float]]:
    """把一批文本各变成一个 512 维的数字列表（向量）。

    例子：embed(["试用期不得超过六个月"]) -> [[0.021, -0.113, ...]]
    意思相近的文本，向量就挨得近——"按意思找"的原理全在这。
    """
    return [list(v) for v in model.embed(texts)]
