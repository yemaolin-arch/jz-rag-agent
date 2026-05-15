"""RAG 系统主入口

支持以下命令：
- python main.py --build-index: 构建向量索引
- python main.py --query "问题": 执行检索（待实现）
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# 添加项目根目录到 sys.path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config_loader import get_full_config, load_yaml_config
from app.parsers.pdf_parser import PDFParser
from app.retrievers.vector_store import VectorStore
from app.retrievers.hybrid_retriever import HybridRetriever
from app.agents import QAAgent

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def build_index():
    """构建索引"""
    logger.info("=" * 60)
    logger.info("Building RAG Index")
    logger.info("=" * 60)

    start_time = time.time()

    # 加载配置
    config = get_full_config()
    paths_config = config.get("paths", {})
    data_dir = paths_config.get("data_dir", "./data")
    raw_dir = paths_config.get("raw_dir", "./data/raw")

    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Raw PDF directory: {raw_dir}")

    # 解析 PDF
    logger.info("-" * 40)
    logger.info("Step 1: Parsing PDFs")
    parse_start = time.time()

    parser = PDFParser(data_dir=Path(data_dir))
    pdf_dir = Path(raw_dir)

    all_chunks = []
    pdf_files = list(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDF files found in {pdf_dir}")
        return

    for pdf_file in pdf_files:
        logger.info(f"Parsing: {pdf_file.name}")
        try:
            chunks = parser.parse(pdf_file, use_cache=True)
            all_chunks.extend(chunks)
            logger.info(f"  -> {len(chunks)} chunks (cache hit allowed)")
        except Exception as e:
            logger.error(f"Failed to parse {pdf_file.name}: {e}")

    parse_time = time.time() - parse_start
    logger.info(f"PDF parsing completed in {parse_time:.2f}s")
    logger.info(f"Total chunks: {len(all_chunks)}")
    table_chunks = sum(1 for c in all_chunks if c.is_table)
    logger.info(f"Table chunks: {table_chunks}")

    if not all_chunks:
        logger.error("No chunks generated, aborting")
        return

    # 构建向量索引
    logger.info("-" * 40)
    logger.info("Step 2: Building Vector Index")
    vector_start = time.time()

    vector_store = VectorStore()
    vector_store.build_index(all_chunks, index_name="default", save=True)

    vector_time = time.time() - vector_start
    logger.info(f"Vector index built in {vector_time:.2f}s")

    # 构建混合检索
    logger.info("-" * 40)
    logger.info("Step 3: Building Hybrid Retriever")
    hybrid_start = time.time()

    retriever = HybridRetriever(vector_store=vector_store)
    retriever.build(all_chunks, index_name="default", save=True)

    hybrid_time = time.time() - hybrid_start
    logger.info(f"Hybrid retriever built in {hybrid_time:.2f}s")

    # 统计信息
    total_time = time.time() - start_time

    logger.info("-" * 40)
    logger.info("Index Building Summary")
    logger.info("-" * 40)
    logger.info(f"Total PDFs processed: {len(pdf_files)}")
    logger.info(f"Total chunks: {len(all_chunks)}")
    logger.info(f"Table chunks: {table_chunks}")
    logger.info(f"Text chunks: {len(all_chunks) - table_chunks}")
    logger.info(f"Parse time: {parse_time:.2f}s")
    logger.info(f"Vector index time: {vector_time:.2f}s")
    logger.info(f"BM25 index time: {hybrid_time - vector_time:.2f}s")
    logger.info(f"Total time: {total_time:.2f}s")

    # 保存统计
    stats_file = Path(data_dir) / "index_stats.json"
    import json
    stats = {
        "pdf_count": len(pdf_files),
        "total_chunks": len(all_chunks),
        "table_chunks": table_chunks,
        "text_chunks": len(all_chunks) - table_chunks,
        "parse_time": parse_time,
        "vector_index_time": vector_time,
        "hybrid_index_time": hybrid_time,
        "total_time": total_time,
    }
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info(f"Stats saved to {stats_file}")
    logger.info("=" * 60)
    logger.info("Index building completed successfully!")
    logger.info("=" * 60)


def query(query_text: str, top_k: int = 5):
    """执行问答查询"""
    logger.info(f"Query: {query_text}")

    # 加载检索器
    retriever = HybridRetriever()
    if not retriever.load("default"):
        logger.error("Index not found. Run --build-index first.")
        return

    # 初始化 QA Agent
    qa_agent = QAAgent(retriever=retriever)

    # 执行问答
    result = qa_agent.answer(query_text, top_k=top_k)

    # 打印格式化结果
    print("\n" + "=" * 60)
    print("问答结果")
    print("=" * 60)

    if result["rejected"]:
        print(f"\n[拒答] {result.get('reject_reason', '未知原因')}")
        print(f"\n建议：请尝试重新描述问题，或检查是否使用了正确的术语。")
    else:
        print(f"\n回答:\n{result['answer']}")
        print(f"\n来源: {', '.join(result['sources'])}")

        self_check = result.get("self_check", {})
        is_grounded = self_check.get("is_grounded", False)
        confidence = self_check.get("confidence", 0.0)

        if is_grounded:
            status = "通过"
            conf_level = "High" if confidence >= 0.7 else "Medium"
        else:
            status = "疑似幻觉"
            conf_level = "Low" if confidence < 0.5 else "Medium"

        print(f"\n自检结果: {status}")
        print(f"置信度: {conf_level} ({confidence:.2f})")

        if result.get("has_warning"):
            print(f"\n[警告] 答案中部分内容未经完全确认，请谨慎参考。")

    print("\n" + "=" * 60)


def main():
    """主入口"""
    parser = argparse.ArgumentParser(description="RAG System Main")
    parser.add_argument("--build-index", action="store_true", help="Build vector index")
    parser.add_argument("--query", type=str, help="Query to search")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")

    args = parser.parse_args()

    if args.build_index:
        build_index()
    elif args.query:
        query(args.query, args.top_k)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()