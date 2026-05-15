"""评估脚本

内置测试集，运行 QAAgent 执行问答，输出 Markdown 格式报告。
"""

import sys
import time
from pathlib import Path
from datetime import datetime

# 添加项目根目录到 sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# 内置测试集
TEST_CASES = [
    {
        "id": 1,
        "type": "正文查询",
        "query": "键的技术条件包含哪些内容？",
        "description": "验证正文解析完整性"
    },
    {
        "id": 2,
        "type": "表格查询",
        "query": "平键的键宽 AQL 值是多少？",
        "description": "验证表格内容提取"
    },
    {
        "id": 3,
        "type": "无答案问题",
        "query": "汽车维修流程是什么？",
        "description": "验证拒答机制有效"
    },
    {
        "id": 4,
        "type": "OCR 容错",
        "query": "键的技术条件包含哪些内溶？",
        "description": "验证模糊匹配能力（故意包含错别字）"
    },
    {
        "id": 5,
        "type": "模糊问题",
        "query": "包装要求",
        "description": "验证语义理解能力"
    },
]


def run_evaluation():
    """运行评估"""
    print("=" * 70)
    print("RAG 系统评估报告")
    print("=" * 70)
    print(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"测试用例数: {len(TEST_CASES)}")
    print("=" * 70)

    # 初始化组件
    try:
        from app.retrievers.hybrid_retriever import HybridRetriever
        from app.agents import QAAgent

        logger.info("Loading hybrid retriever...")
        retriever = HybridRetriever()
        if not retriever.load("default"):
            logger.error("Failed to load index. Run --build-index first.")
            print("错误：索引未构建。请先运行: python main.py --build-index")
            return

        qa_agent = QAAgent(retriever=retriever)
        logger.info("QAAgent initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        print(f"错误：初始化失败 - {e}")
        return

    # 存储结果
    results = []

    # 遍历测试用例
    for test_case in TEST_CASES:
        case_id = test_case["id"]
        case_type = test_case["type"]
        query = test_case["query"]
        description = test_case["description"]

        print(f"\n[{case_id}/{len(TEST_CASES)}] {case_type}: {query}")

        start_time = time.time()
        try:
            result = qa_agent.answer(query, top_k=5)
            elapsed = time.time() - start_time

            # 解析结果
            answer = result.get("answer", "")
            sources = result.get("sources", [])
            rejected = result.get("rejected", False)
            self_check = result.get("self_check", {})
            is_grounded = self_check.get("is_grounded", False)
            confidence = self_check.get("confidence", 0.0)
            reject_reason = result.get("reject_reason", "")

            # 截断过长答案
            if len(answer) > 200:
                answer_preview = answer[:200] + "..."
            else:
                answer_preview = answer

            # 记录结果
            results.append({
                "id": case_id,
                "type": case_type,
                "query": query,
                "answer": answer_preview,
                "answer_full": answer,
                "sources": ", ".join(sources) if sources else "-",
                "rejected": "是" if rejected else "否",
                "elapsed": f"{elapsed:.2f}s",
                "is_grounded": "通过" if is_grounded else "失败",
                "confidence": f"{confidence:.2f}",
                "reject_reason": reject_reason or "-",
                "status": "success"
            })

            # 打印结果
            status_icon = "[OK]" if not rejected else "[X]"
            status_text = "拒答" if rejected else "回答"
            print(f"  状态: {status_icon} {status_text}")
            print(f"  来源: {', '.join(sources) if sources else '-'}")
            print(f"  自检: {'通过' if is_grounded else '失败'} (置信度 {confidence:.2f})")
            print(f"  耗时: {elapsed:.2f}s")
            if rejected:
                print(f"  原因: {reject_reason}")

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Test case {case_id} failed: {e}")

            results.append({
                "id": case_id,
                "type": case_type,
                "query": query,
                "answer": f"[错误] {str(e)}",
                "answer_full": f"[错误] {str(e)}",
                "sources": "-",
                "rejected": "错误",
                "elapsed": f"{elapsed:.2f}s",
                "is_grounded": "-",
                "confidence": "-",
                "reject_reason": str(e),
                "status": "error"
            })

            print(f"  状态: [X] 错误")
            print(f"  错误: {e}")

    # 统计
    total = len(results)
    success = sum(1 for r in results if r["status"] == "success" and r["rejected"] != "是")
    rejected_count = sum(1 for r in results if r["rejected"] == "是")
    errors = sum(1 for r in results if r["status"] == "error")

    # 生成 Markdown 报告
    report = generate_markdown_report(results, total, success, rejected_count, errors)

    # 保存报告
    report_path = PROJECT_ROOT / "evaluation_report.md"
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n报告已保存: {report_path}")
    except Exception as e:
        logger.error(f"Failed to save report: {e}")

    # 打印汇总
    print("\n" + "=" * 70)
    print("评估汇总")
    print("=" * 70)
    print(f"总测试数: {total}")
    print(f"成功回答: {success}")
    print(f"拒答数: {rejected_count}")
    print(f"错误数: {errors}")
    print("=" * 70)

    return results


def generate_markdown_report(results: list, total: int, success: int, rejected_count: int, errors: int) -> str:
    """生成 Markdown 格式报告"""

    lines = [
        "# RAG 系统评估报告",
        "",
        f"**评估时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 汇总统计",
        "",
        f"| 指标 | 值 |",
        f"|------|-----|",
        f"| 总测试数 | {total} |",
        f"| 成功回答 | {success} |",
        f"| 拒答数 | {rejected_count} |",
        f"| 错误数 | {errors} |",
        "",
        "## 详细结果",
        "",
        "| # | 类型 | 问题 | 答案 | 来源 | 拒答 | 自检 | 置信度 | 耗时 |",
        "|---|------|------|------|------|------|------|--------|------|",
    ]

    for r in results:
        # 转义 Markdown 特殊字符
        query_escaped = r["query"].replace("|", "\\|").replace("\n", " ")
        answer_escaped = r["answer"].replace("|", "\\|").replace("\n", " ")

        lines.append(
            f"| {r['id']} | {r['type']} | {query_escaped} | {answer_escaped} | "
            f"{r['sources']} | {r['rejected']} | {r['is_grounded']} | "
            f"{r['confidence']} | {r['elapsed']} |"
        )

    lines.append("")
    lines.append("## 说明")
    lines.append("")
    lines.append("- **自检**: SelfCheckAgent 评估答案是否由参考资料支撑")
    lines.append("- **置信度**: SelfCheck 返回的置信度 (0-1)")
    lines.append("- **拒答**: 当检索分数过低或自检失败时触发")
    lines.append("")
    lines.append("## 测试用例说明")
    lines.append("")
    lines.append("| # | 类型 | 说明 |")
    lines.append("|---|------|------|")
    for tc in TEST_CASES:
        lines.append(f"| {tc['id']} | {tc['type']} | {tc['description']} |")

    return "\n".join(lines)


def main():
    """主入口"""
    import argparse

    parser = argparse.ArgumentParser(description="RAG 系统评估脚本")
    parser.add_argument("--force-reparse", action="store_true", help="强制重新解析 PDF")
    parser.add_argument("--top-k", type=int, default=5, help="检索返回数量")

    args = parser.parse_args()

    if args.force_reparse:
        print("注意: 强制重新解析需要运行: python main.py --build-index --force-reparse")
        print("当前将使用现有索引...")

    run_evaluation()


if __name__ == "__main__":
    main()