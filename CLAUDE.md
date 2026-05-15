# CLAUDE.md — 项目指令

## 项目定位
企业级 Agent/RAG 文档问答系统。目标：可解释、可测试、可扩展、可治理幻觉、可适配扫描PDF的最小闭环系统。这不是 Demo，是工程能力展示。


## 技术栈约束
- Python 3.11+
- 依赖管理: pyproject.toml
- Backend: FastAPI
- Framework: LlamaIndex (核心编排)
- OCR: PaddleOCR（中文OCR，ppstructure 提取表格）
- PDF Parser: pymupdf / pymupdf4llm（PDF解析）
- Vector Store: FAISS(本地轻量)
- Embedding：bge-m3 
- Rerank：bge-reranker
- OpenAI API 兼容接口（LLM 调用）


## 代码工程规范

### 目录结构规范
project/
├── app/
│ ├── agents/ # 各类 Agent（路由、证据判断、自检等）
│ ├── parsers/ # PDF分类器、OCR、结构恢复、表格提取
│ ├── retrievers/ # 混合检索、重排序
│ ├── selfcheck/ # 幻觉检测、拒答判断
│ ├── workflows/ # 完整问答流水线编排
│ ├── schemas/ # Pydantic 模型
│ ├── prompts/ # 所有提示词（独立文件）
│ ├── services/ # LLM调用、Embedding等基础服务
│ └── api/ # FastAPI 接口
├── tests/ # unit / integration / regression
├── data/ # raw / parsed / chunks / vector_store
├── scripts/ # 启动、评估脚本
├── configs/ # 配置驱动
├── .env.example # 环境变量模板（可提交）
├── .gitignore # 确保 .env 不上传
├── pyproject.toml
└── README.md

### 核心管线

PDF → 类型识别 → OCR/解析 → OCR后处理 → 结构恢复 → 表格提取→ Chunk构建 → Embedding → Hybrid Retrieval(BM25+Dense) → Rerank→ LLM生成(带引用) → SelfCheck → 最终响应

每个阶段必须：独立模块、独立可测、独立可替换。

### 必须遵守

- 所有函数/类有 type hints 和 docstring
- 日志用 Python logging，不用 print
- 异常不吞掉，至少记录日志并抛出
- 配置外置到 configs/ 目录（YAML），不在代码中硬编码
- Prompt 模板独立管理在 app/prompts/ 目录，不内嵌代码中
- LLM 输出做 JSON schema 校验 + 重试 + 超时控制
- 使用 dataclass 或 Pydantic 定义数据结构

### 代码风格

- 类型提示: 所有函数/类必须有 type hints 和 docstring。
- 日志: 使用 Python logging，关键节点(OCR/检索/生成)必须打印耗时和关键信息。
- 异常处理: 禁止裸 except，必须捕获具体异常并记录日志。
- 配置外置: 敏感信息走 .env，参数配置走 configs/。
- Prompt 管理: 禁止代码内嵌超长 Prompt，必须放在 app/prompts/ 下。

### README 必须包含

1. 项目架构（含流程图）
2. 模块说明与设计取舍
3. 技术选型原因
4. OCR/表格处理说明
5. 检索与幻觉治理设计
6. 测试方案与评估结果
7. 启动方式（从零到运行）
8. 不同业务场景的扩展方案（金融/合规/客户交付/多语言）


## 关键设计要求

### PDF解析与OCR

- 不信 OCR: OCR 结果不可信。必须实现 OCRNormalizer 处理 O/0, l/1 混淆、空格丢失、编号错乱。
- 表格处理: 表格禁止打平为纯文本。必须提取为 Markdown 格式，保留表头与行列关系，作为独立 Chunk 处理。
- 智能分类: 需实现 PDFClassifier 判断扫描件/原生 PDF，并自动选择解析策略。

### 分块策略

- 按文档结构分块（章节标题 > 条款编号 > 语义段落），禁止固定字数粗暴切分
- 表格独立成 chunk，保留表头、行列关系、表格标题
- Chunk metadata 必须包含：page, chunk_id, section, source, is_table
- 支持相邻 chunk 的 overlap，但不破坏条款/章节边界

### 检索策略

- 必须使用 Hybrid Retrieval：BM25（关键词精确匹配）+ Dense Retrieval（语义匹配）
- 必须有 Reranker 层：OCR 场景下 Dense 容易召回噪声文本
- 检索输出包含：score, source_page, chunk_id, snippet
- 根据问题特征（如包含条款编号 vs 语义问题）动态调整检索权重

### 幻觉治理
这是本项目的核心。必须实现以下机制：

- Citation Validation: 每个回答必须附带 source page / chunk id。
- No-Answer Detection: 若 Rerank Score < Threshold，必须触发拒答逻辑(返回“文档中未提及”)。
- SelfCheck Agent: 生成答案后，必须检查 Answer 是否由 Evidence 支撑。输出 grounded, hallucination_risk 字段。

## 测试与验证
### 预设测试集
请在 tests/test_cases.json 中准备以下 5 类问题：

- 正文查询: "键的技术条件包含哪些内容？"
- 表格查询: "平键的键宽是多少？" (验证表格解析)
- 无答案问题: "文档中提到的汽车维修流程是什么？" (验证拒答机制)
- OCR容错: 包含轻微错别字的查询 (验证模糊匹配)
- 模糊问题: "包装要求"

### 评估脚本
实现 scripts/evaluate.py:

- 批量运行测试集。
- 输出 Markdown 报告，包含：问题、答案、来源页码、是否拒答、耗时。
- 统计指标：准确率、拒答率、幻觉率、引用率



## 开发工作流

使用 Claude Code 辅助开发，但遵循：

- 每个模块先设计再编码，编码后立即验证
- AI 生成的代码必须人工校验：架构合理性、异常处理、幻觉风险、可测试性
- 不直接提交未经验证的 AI 生成代码
- 在 README 中记录 AI 工具的使用过程和验证方法