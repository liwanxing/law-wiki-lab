# law-wiki-lab —— 法律向量知识库（RAG 学习项目）

做一个法律库 + 法律咨询功能：把真实法律条文向量化存进 Qdrant，
提问时按"意思"检索出最相关的法条。

数据：`data/raw/` 下两部真实法律：劳动合同法（2012 修正版 98 条）、劳动法（2018 修正版 107 条）。

## 怎么跑

```powershell
# 1. 启动 Qdrant（向量数据库，Docker 容器，HTTP 端口 6333）
docker run -d --name law-wiki-qdrant -p 6333:6333 -v law-wiki-qdrant-data:/qdrant/storage qdrant/qdrant:v1.19.2

# 2. 安装依赖（两个包，模型本地跑，无需任何 API key）
python -m pip install qdrant-client fastembed

# 3. 法条入库：切块 -> 向量化 -> 存进 Qdrant（首次会下载 embedding 模型约 95MB）
python -m app.vector_store

# 4. 法律咨询问答：向量检索最相关法条 + 大模型照着法条作答
python -m app.vector_store "试用期最长多久"
python -m app.vector_store "被公司开除了怎么赔偿"   # 全文没有"开除"二字，照样命中第 87 条

# 问答需要 LLM_API_KEY 环境变量（智谱 glm-4-flash 免费模型）；
# 没配置时自动降级：只打印拼好的提示词，可手动粘给任意 AI
```

## 文件结构

```
app/
  vector_store.py  全部代码：切块 -> 向量入库 -> 向量检索 -> 大模型问答
  config.py        配置：Qdrant/embedding/大模型接口（key 走环境变量 LLM_API_KEY）
data/raw/          法律原文 markdown（### 第X条 = 一块，加法律直接放新 md）
```

学习方式：只保留最简基线代码，每看懂一层、提交一版，再追加下一层功能，
让每一版代码都只包含已看懂的东西。

## 本机网络注意事项

- Clash 系统代理会劫持发往 localhost 的请求（回 502）→ `vector_store.py` 已设 `NO_PROXY`
- HuggingFace 国内直连超时 → 已设 `HF_ENDPOINT=https://hf-mirror.com`
- embedding 模型 `bge-small-zh-v1.5` 输出 **512 维**向量（`config.py` 的 `EMBED_DIM` 要一致）

## 路线（逐层演进，全部完成）

- [x] 第 1 层：向量 RAG 基线 —— 法条切块、向量化入库 Qdrant、向量检索
- [x] 第 2 层：查询带出处 —— 条文号从块首行提取，答案可溯源
- [x] 第 3 层：多法律混合检索 —— 劳动法入库零代码扩展，payload 加法律名
- [x] 第 4 层：稳定 id 增量更新 —— uuid5 按内容生成身份证号，只重算改动的条文
- [x] 第 5 层：大模型问答 —— 检索到的法条拼进提示词，glm-4-flash 照着法条作答
