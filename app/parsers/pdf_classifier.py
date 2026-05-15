"""PDF 分类器模块

判断 PDF 是扫描件还是原生 PDF，指导后续解析策略。
"""

import logging
from pathlib import Path
from typing import Literal

import pymupdf

logger = logging.getLogger(__name__)


class PDFClassifier:
    """PDF 类型分类器

    通过检测 PDF 中是否包含嵌入文本或大量图片来判断类型。
    扫描件通常只有图片层，原生 PDF 包含可提取的文本层。
    """

    def classify(self, file_path: str | Path) -> Literal["scanned", "native"]:
        """判断 PDF 类型

        Args:
            file_path: PDF 文件路径

        Returns:
            "scanned": 扫描件，需要 OCR
            "native": 原生 PDF，可直接提取文本
        """
        file_path = Path(file_path)
        logger.info(f"Classifying PDF: {file_path.name}")

        try:
            with pymupdf.open(file_path) as doc:
                text_pages = 0
                image_pages = 0

                for page_num in range(len(doc)):
                    page = doc[page_num]
                    text = page.get_text().strip()
                    images = page.get_images()

                    if text and len(text) > 50:
                        text_pages += 1
                    if images:
                        image_pages += 1

                total_pages = len(doc)
                text_ratio = text_pages / total_pages if total_pages > 0 else 0

                logger.info(
                    f"PDF {file_path.name}: pages={total_pages}, "
                    f"text_pages={text_pages}, image_pages={image_pages}, "
                    f"text_ratio={text_ratio:.2f}"
                )

                if text_ratio < 0.3 and image_pages > total_pages * 0.5:
                    logger.info(f"Classified as scanned PDF: {file_path.name}")
                    return "scanned"
                else:
                    logger.info(f"Classified as native PDF: {file_path.name}")
                    return "native"

        except Exception as e:
            logger.error(f"Failed to classify PDF {file_path}: {e}")
            return "native"