"""解析器测试脚本

测试 PDFClassifier、PDFParser、OCRParser、TableExtractor、OCRNormalizer
"""

import argparse
import logging
import sys
from pathlib import Path

# 添加项目根目录到 sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
PDF_PATH = PROJECT_ROOT / "data" / "raw" / "GBT 1568-2008 键 技术条件.pdf"


def test_pdf_classifier():
    """测试 PDFClassifier"""
    print("\n" + "=" * 60)
    print("1. PDFClassifier 测试")
    print("=" * 60)

    from app.parsers import PDFClassifier

    classifier = PDFClassifier()

    try:
        import pymupdf

        with pymupdf.open(PDF_PATH) as doc:
            total_pages = len(doc)
            text_pages = 0
            image_pages = 0

            for page_num in range(min(2, total_pages)):
                page = doc[page_num]
                text = page.get_text().strip()
                images = page.get_images()

                has_text = bool(text and len(text) > 50)
                has_images = bool(images)

                if has_text:
                    text_pages += 1
                if has_images:
                    image_pages += 1

                print(f"\n--- 第 {page_num + 1} 页详情 ---")
                print(f"  是否有文本层: {has_text}")
                print(f"  文本长度: {len(text) if text else 0} 字符")
                print(f"  是否有图片: {has_images}")
                print(f"  图片数量: {len(images) if images else 0}")

                if text:
                    preview = text[:300].replace("\n", " | ")
                    print(f"  文本预览: {preview}...")

            print(f"\n汇总: 总页数={total_pages}, 文本页={text_pages}, 图片页={image_pages}")

        pdf_type = classifier.classify(PDF_PATH)
        print(f"\n分类结果: {pdf_type}")
        print(f"  -> {'扫描件 (需要 OCR)' if pdf_type == 'scanned' else '原生 PDF (直接提取)'}")

    except Exception as e:
        logger.error(f"PDFClassifier 测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_pdf_parser(force_reparse: bool = False):
    """测试 PDFParser"""
    print("\n" + "=" * 60)
    print("2. PDFParser 测试 (解析前2页)")
    print("=" * 60)

    from app.parsers import PDFParser

    parser = PDFParser()

    try:
        chunks = parser.parse(PDF_PATH, force_reparse=force_reparse)

        print(f"\n总 chunk 数: {len(chunks)}")

        # 按页码筛选前2页
        page1_chunks = [c for c in chunks if c.page == 1]
        page2_chunks = [c for c in chunks if c.page == 2]

        for page_num, page_chunks in [(1, page1_chunks), (2, page2_chunks)]:
            if not page_chunks:
                print(f"\n--- 第 {page_num} 页: 无 chunk")
                continue

            print(f"\n--- 第 {page_num} 页 ({len(page_chunks)} chunks) ---")

            for i, chunk in enumerate(page_chunks[:3]):  # 最多3个chunk
                content = chunk.content[:500] if len(chunk.content) > 500 else chunk.content
                content = content.replace("\n", " | ")
                print(f"  Chunk {i+1}: is_table={chunk.is_table}, len={len(chunk.content)}")
                print(f"    内容: {content}...")

        # 统计
        total_text = sum(len(c.content) for c in chunks)
        table_chunks = [c for c in chunks if c.is_table]
        print(f"\n统计: 总字数={total_text}, 表格chunk数={len(table_chunks)}")

    except Exception as e:
        logger.error(f"PDFParser 测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_table_extractor():
    """测试 TableExtractor"""
    print("\n" + "=" * 60)
    print("3. TableExtractor 测试")
    print("=" * 60)

    from app.parsers import TableExtractor

    extractor = TableExtractor()

    try:
        tables = extractor.extract_tables_pymupdf(PDF_PATH, page_numbers=[1, 2])

        print(f"\n检测到 {len(tables)} 个表格")

        for i, table in enumerate(tables):
            markdown = table["markdown"]
            rows = markdown.count("\n") - 1  # 减去表头行
            cols = markdown.split("|")[1].count(" ") if markdown else 0

            print(f"\n--- 表格 {i+1} ---")
            print(f"  页码: {table['page']}")
            print(f"  行列数: {rows} 行 x {cols} 列")
            print(f"  Markdown 预览:")
            print(markdown[:500] if len(markdown) > 500 else markdown)

    except Exception as e:
        logger.error(f"TableExtractor 测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_ocr_normalizer():
    """测试 OCRNormalizer"""
    print("\n" + "=" * 60)
    print("4. OCRNormalizer 测试")
    print("=" * 60)

    from app.parsers import OCRNormalizer

    # 模拟 OCR 错误样本
    test_cases = [
        ("11llOO0O", "l/1 和 O/0 混淆"),
        ("技 术 条 件", "多余空格"),
        ("键  宽   尺寸", "多个空格"),
        ("1.2.3.4.5", "编号格式"),
        ("GB/T 1568-2008  键", "尾部多余空格"),
        ("l2O0l2O0", "混合错误"),
    ]

    print("\nOCR 错误样本标准化测试:")
    print("-" * 60)

    for original, description in test_cases:
        normalized = OCRNormalizer.normalize(original)
        status = "[*]" if original != normalized else "[~]"
        print(f"\n{status} {description}")
        print(f"  INPUT:  '{original}'")
        print(f"  OUTPUT:  '{normalized}'")

    # 真实 OCR 样本测试（如果有的话）
    print("\n" + "-" * 60)
    print("真实样本测试:")


def test_ocr_parser():
    """测试 OCRParser (仅扫描件)"""
    print("\n" + "=" * 60)
    print("5. OCRParser 测试 (如果是扫描件)")
    print("=" * 60)

    from app.parsers import PDFClassifier, OCRParser

    classifier = PDFClassifier()
    pdf_type = classifier.classify(PDF_PATH)

    if pdf_type == "native":
        print("原生 PDF，跳过 OCR 测试")
        return

    try:
        ocr = OCRParser(use_rapidocr=True)
        results = ocr.parse(PDF_PATH)

        print(f"\nOCR 结果: {len(results)} 个文本块")

        # 统计
        total_chars = sum(len(r["text"]) for r in results)
        avg_score = sum(r.get("score", 0) for r in results) / len(results) if results else 0

        print(f"总字符数: {total_chars}")
        print(f"平均置信度: {avg_score:.2f}")

        # 前500字
        all_text = " ".join(r["text"] for r in results)
        print(f"\n前500字预览:")
        print(all_text[:500].replace("\n", " | "))

    except Exception as e:
        logger.error(f"OCRParser 测试失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="PDF 解析器测试套件")
    parser.add_argument("--force-reparse", action="store_true", help="强制重新解析，忽略缓存")
    args = parser.parse_args()

    print("=" * 60)
    print("PDF 解析器测试套件")
    print(f"测试文件: {PDF_PATH}")
    if args.force_reparse:
        print("模式: 强制重新解析 (忽略缓存)")
    print("=" * 60)

    if not PDF_PATH.exists():
        print(f"错误: 文件不存在 {PDF_PATH}")
        sys.exit(1)

    test_pdf_classifier()
    test_pdf_parser(force_reparse=args.force_reparse)
    test_table_extractor()
    test_ocr_normalizer()
    test_ocr_parser()

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()