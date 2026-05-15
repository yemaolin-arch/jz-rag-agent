"""PDF 解析器模块

包含：
- PDFClassifier: 判断 PDF 是扫描件还是原生 PDF
- TableExtractor: 从 PDF 中提取表格并转换为 Markdown
- OCRParser: OCR 解析器（支持 rapidocr 和 paddleocr）
- OCRNormalizer: OCR 结果标准化
- PDFParser: 统一解析入口
"""

from .pdf_classifier import PDFClassifier
from .table_extractor import TableExtractor
from .ocr_parser import OCRParser, OCRNormalizer, TableReconstructor, merge_ocr_blocks
from .pdf_parser import PDFParser

__all__ = [
    "PDFClassifier",
    "TableExtractor",
    "OCRParser",
    "OCRNormalizer",
    "TableReconstructor",
    "merge_ocr_blocks",
    "PDFParser",
]