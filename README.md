# JZ-RAG-Agent

企业级 Agent/RAG 文档问答系统。目标：可解释、可测试、可扩展、可治理幻觉、可适配扫描PDF的最小闭环系统。

## 技术栈

- **Python**: 3.10+
- **Backend**: FastAPI
- **Framework**: LlamaIndex
- **OCR**: PaddleOCR (ppstructure 表格提取), rapidocr-onnxruntime
- **PDF Parser**: pymupdf, pymupdf4llm
- **Vector Store**: FAISS
- **Embedding**: bge-m3
- **Rerank**: bge-reranker
- **LLM**: OpenAI API 兼容接口

## 项目结构

```
jz-rag-agent/
├── app/
│   ├── agents/        # 各类 Agent
│   ├── parsers/       # PDF 分类、OCR、表格提取
│   ├── retrievers/    # 混合检索、重排序
│   ├── selfcheck/     # 幻觉检测
│   ├── workflows/     # 流水线编排
│   ├── schemas/       # Pydantic 模型
│   ├── prompts/       # 提示词模板
│   ├── services/       # LLM/Embedding 服务
│   └── api/           # FastAPI 接口
├── configs/           # YAML 配置
├── data/              # raw / parsed / chunks / vector_store
├── scripts/            # 启动、评估脚本
├── tests/             # 单元/集成/回归测试
└── pyproject.toml
```

## 快速启动

### 1. 环境准备

```bash
# 确认 uv 已安装
which uv  # 或: pip install uv

# 创建虚拟环境
uv venv .venv

# 激活虚拟环境 (Windows)
.venv\Scripts\activate

# 激活虚拟环境 (Linux/Mac)
# source .venv/bin/activate
```

### 2. 安装依赖

```bash
# 安装项目及所有依赖
uv pip install -e ".[dev]"

# 或仅安装运行时依赖
uv pip install -e .
```

### 3. 配置环境变量

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 填入你的 API Key 等配置
```

### 4. 运行测试

```bash
# 运行所有测试
uv run python -m pytest tests/ -v

# 运行特定测试
uv run python -m pytest tests/unit/test_ocr_normalizer.py -v

# 运行测试脚本 (解析器验证)
uv run python scripts/test_parsers.py
```

### 5. 启动 API 服务

```bash
uv run python -m app.api.main
# 或
uv run uvicorn app.api.main:app --reload --port 8000
```

## 核心管线

```
PDF → 类型识别 → OCR/解析 → OCR后处理 → 结构恢复 → 表格提取 → Chunk构建
→ Embedding → Hybrid Retrieval → Rerank → LLM生成 → SelfCheck → 最终响应
```

## 模块说明

### app/parsers

| 模块 | 说明 |
|------|------|
| `PDFClassifier` | 判断 PDF 是扫描件还是原生 PDF |
| `OCRParser` | OCR 解析器 (rapidocr / paddleocr) |
| `OCRNormalizer` | 清洗 OCR 错误 (l/1, O/0, 空格) |
| `TableExtractor` | 表格提取并转为 Markdown 格式 |
| `PDFParser` | 统一解析入口 |

### app/retrievers

| 模块 | 说明 |
|------|------|
| `BM25Retriever` | 关键词精确匹配 |
| `DenseRetriever` | 语义匹配 |
| `HybridRetriever` | BM25 + Dense 混合检索 |
| `Reranker` | 重排序 |

### app/selfcheck

| 模块 | 说明 |
|------|------|
| `SelfCheckAgent` | 幻觉检测 |
| `NoAnswerDetector` | 拒答判断 |

## OCR 错误纠正规则

| 场景 | 示例 | 结果 |
|------|------|------|
| l 在两个数字之间 | `12l3` | `1213` |
| l 在数字串开头 | `l23` | `l23` (不变) |
| O 在两个数字之间 | `12O3` | `1203` |
| O 在数字串开头 | `O123` | `O123` (不变) |
| 多余空格 | `键  宽` | `键 宽` |
| 尾部空格 | `GB/T 1568  ` | `GB/T 1568` |

## 开发指南

### 添加新模块

1. 在对应目录下创建 `*.py` 文件
2. 添加 `__init__.py` 导出 (如需要)
3. 编写单元测试 `tests/unit/test_*.py`
4. 更新本 README

### 代码规范

- 类型提示: 所有函数/类必须有 type hints
- 日志: 使用 `logging`，关键节点打印耗时
- 异常: 禁止裸 `except`，捕获具体异常并记录
- 配置: 敏感信息走 `.env`，参数走 `configs/`

## 评估

```bash
# 运行评估脚本
uv run python scripts/evaluate.py
```

输出 Markdown 报告，包含：问题、答案、来源页码、是否拒答、耗时。