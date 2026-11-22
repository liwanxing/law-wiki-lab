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


# ============ 大模型问答（第 5 层：法律咨询）============

# 调大模型 = 发一个 HTTP POST 到 OpenAI 兼容接口的 /chat/completions，
# 三要素：接口地址 + 模型名 + API key。

#: 接口地址（OpenAI 兼容）。智谱 GLM 免费模型 glm-4-flash，2026-11 验证连通。
LLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

#: 模型名：glm-4-flash 是智谱的免费对话模型。
LLM_MODEL = "glm-4-flash"

#: API key：优先读项目根下的 api_key.txt（不进 git，想换 key 就新建这个文件）；
#: 文件不存在就直接用内置 key——智谱 glm-4-flash 是免费模型，学习项目内置无妨。
def _load_api_key() -> str:
    key_file = PROJECT_ROOT / "api_key.txt"
    if key_file.exists():
        return key_file.read_text(encoding="utf-8").strip()
    return "2f78b5cfd72d45c5995189703ddab45a.OPWz6mBEHSux44rP"


LLM_API_KEY = _load_api_key()
