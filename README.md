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

# 3. 法条入库：切块 -> 向量化 -> 增量存储（首次会下载 embedding 模型约 95MB）
python -m app.ingest

# 4. 法律咨询问答：向量检索最相关法条 + 大模型照着法条作答
python -m app.ask "试用期最长多久"
python -m app.ask "被公司开除了怎么赔偿"   # 全文没有"开除"二字，照样命中第 87 条

# 5. 法条更新后同步笔记：笔记里记着"依据条文@指纹"，指纹对不上就自动重生成
python -m app.ingest   # 先让改过的法条入库
python -m app.sync     # 再对账：找出依据已变化的笔记，重新检索作答并覆盖、入库

# 问答用的是智谱 glm-4-flash 免费模型，key 已内置（也可在项目根建 api_key.txt 换自己的）
```

## 文件结构

```
app/
  config.py  配置：Qdrant/embedding/大模型接口（key 内置智谱免费模型，可用 api_key.txt 覆盖）
  common.py  共享资源：向量化模型、Qdrant 连接（入库和问答共用，只建一份）
  ingest.py  入库流水线：切块（面包屑+双链） -> 向量化 -> 稳定 id 增量存储
  ask.py     问答：向量检索 -> 链接扩展 -> 大模型作答 -> 知识回流（笔记带依据条文）
  sync.py    一致性维护：笔记的依据指纹对不上（法条改了）就自动重生成该笔记
data/raw/          法律原文 markdown（### 第X条 = 一块，加法律直接放新 md）
data/notes/       整理笔记 markdown（问答确认后存回，一份知识一个文件，与法条同库检索）
```

学习方式：只保留最简基线代码，每看懂一层、提交一版，再追加下一层功能，
让每一版代码都只包含已看懂的东西。

## 本机网络注意事项

- Clash 系统代理会劫持发往 localhost 的请求（回 502）→ `common.py` 已设 `NO_PROXY`
- HuggingFace 国内直连超时 → 已设 `HF_ENDPOINT=https://hf-mirror.com`
- embedding 模型 `bge-small-zh-v1.5` 输出 **512 维**向量（`config.py` 的 `EMBED_DIM` 要一致）

## 路线（逐层演进，全部完成）

- [x] 第 1 层：向量 RAG 基线 —— 法条切块、向量化入库 Qdrant、向量检索
- [x] 第 2 层：查询带出处 —— 条文号从块首行提取，答案可溯源
- [x] 第 3 层：多法律混合检索 —— 劳动法入库零代码扩展，payload 加法律名
- [x] 第 4 层：稳定 id 增量更新 —— uuid5 按内容生成身份证号，只重算改动的条文
- [x] 第 5 层：大模型问答 —— 检索到的法条拼进提示词，glm-4-flash 照着法条作答
- [x] 第 6 层：四模块拆分 —— config/common/ingest/ask，key 本地文件优先
- [x] 第 7 层：提示词硬性约束 —— 只准引用材料里的条文
- [x] 第 8 层：相似度阈值闸门 —— 弱相关直接拒答
- [x] 第 9 层：面包屑 —— 每块带章节地址，面包屑同时是稳定 id
- [x] 第 10 层：双链 —— 正则提取"本法第X条"，块间关系一并入库
- [x] 第 11 层：链接扩展 —— 命中法条后顺双链把被引用条文拉进材料
- [x] 第 12 层：知识回流之加 —— 问答整理存为笔记，复用增量入库
- [x] 第 13 层：知识回流之更新 —— 同标题覆盖修订，标题即知识主键
- [x] 第 14 层：知识一致性维护 —— 笔记记"依据条文@指纹"，sync 发现法条变更自动重生成笔记
