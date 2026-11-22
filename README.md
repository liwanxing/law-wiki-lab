# law-wiki-lab —— 法律向量知识库（RAG 学习项目）

做一个法律库 + 法律咨询功能：把真实法律条文向量化存进 Qdrant，
提问时按"意思"检索出最相关的法条。

数据：`data/raw/劳动合同法.md`（2012 修正版全文 98 条）。

## 怎么跑

```powershell
# 1. 启动 Qdrant（向量数据库，Docker 容器，HTTP 端口 6333）
docker run -d --name law-wiki-qdrant -p 6333:6333 -v law-wiki-qdrant-data:/qdrant/storage qdrant/qdrant:v1.19.2

# 2. 安装依赖（两个包，模型本地跑，无需任何 API key）
python -m pip install qdrant-client fastembed

# 3. 法条入库：切块 -> 向量化 -> 存进 Qdrant（首次会下载 embedding 模型约 95MB）
python -m app.vector_store

# 4. 问问题 -> 向量检索最相关的法条
python -m app.vector_store "试用期最长多久"
python -m app.vector_store "被公司开除了怎么赔偿"   # 全文没有"开除"二字，照样命中第 87 条
```

## 文件结构

```
app/
  vector_store.py  全部代码：切块 -> 向量化入库 -> 向量查询（现阶段唯一要看的文件）
  config.py        配置：Qdrant 地址、embedding 模型、维度
data/raw/          法律原文 markdown（### 第X条 = 一块，加法律直接放新 md）
```

学习方式：只保留最简基线代码，每看懂一层、提交一版，再追加下一层功能，
让每一版代码都只包含已看懂的东西。

## 本机网络注意事项

- Clash 系统代理会劫持发往 localhost 的请求（回 502）→ `vector_store.py` 已设 `NO_PROXY`
- HuggingFace 国内直连超时 → 已设 `HF_ENDPOINT=https://hf-mirror.com`
- embedding 模型 `bge-small-zh-v1.5` 输出 **512 维**向量（`config.py` 的 `EMBED_DIM` 要一致）

## 路线（四步，一步一个可跑的小目标）

- [x] 第一步：向量 RAG 跑通 —— 真实法条切块、向量化入库 Qdrant、向量检索 ← 已完成
- [ ] 第二步：扩展法律库 —— 加入更多法律（劳动法、民法典节选等），验证"入库即扩展"
- [ ] 第三步：Wiki 切分与面包屑检索 —— 切分粒度优化、面包屑怎么更好地参与检索
- [ ] 第四步：更新机制 —— 法条追加/修订时增量更新，知识库越用越准
