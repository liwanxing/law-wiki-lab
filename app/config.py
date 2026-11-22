"""全局配置：项目里所有"路径"都集中在这里定义。

为什么要单独一个文件：路径一旦散落在各个模块里，以后挪目录就要全局搜代码逐个改；
集中在这里，调整目录结构只动这一个文件，其他模块一律从这里 import。

路径是怎么推出来的（拿本文件举例）：
    __file__        = <项目根>/app/config.py      # 本文件的完整路径（Python 自带变量）
    .parent         = <项目根>/app                # 往上一级：app 目录
    .parent.parent  = <项目根>                    # 再往上一级：项目根目录
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: 原始法律资料目录。vector_store.py 读这里所有 .md 文件，
#: 每个 '### 第X条' 小节会被切成一个块。
DATA_DIR = PROJECT_ROOT / "data" / "raw"


# ============ 向量检索（第一步：Qdrant 跑通）============

#: Qdrant 服务地址。Docker 容器 law-wiki-qdrant 把容器内的 6333 端口
#: 映射到了本机 6333（6333 是 Qdrant 的标准 HTTP 端口，另一个 6334
#: 是 gRPC 端口，本项目用不到）。
QDRANT_URL = "http://localhost:6333"

#: 向量集合（collection）名。可以理解成"一张专门存向量的表"，
#: 同一个 Qdrant 服务里不同项目用不同集合名，数据互不干扰。
COLLECTION_NAME = "law_wiki"

#: 向量化模型：BAAI 出品的中文小模型，本地运行、免费、无需 API key。
#: 它把一段文字变成一个 512 维的数字列表（"向量"）。
#: 首次运行会自动下载模型文件（约 95MB）到本地缓存目录。
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"

#: 上面这个模型输出的向量维度，创建集合时要用它。
EMBED_DIM = 512
