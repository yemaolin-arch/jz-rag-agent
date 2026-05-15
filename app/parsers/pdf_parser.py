"""PDF 解析器模块

统一入口，根据 PDF 类型选择直接提取还是 OCR。
返回原始 OCR 结果（不含分块逻辑）。
分块由 SemanticChunker 负责。
支持缓存：解析结果保存到 data/parsed/{filename}.json
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pymupdf

from app.schemas.common import DocumentChunk

from .pdf_classifier import PDFClassifier
from .table_extractor import TableExtractor
from .ocr_parser import OCRParser

logger = logging.getLogger(__name__)

# 缓存版本号，修改时递增
CACHE_VERSION = 3

# 默认数据目录（可通过配置覆盖）
DEFAULT_DATA_DIR = Path("data")


class PDFParser:
    """PDF 解析器

    统一入口，根据 PDF 类型选择解析策略：
    - 原生 PDF: 直接提取文本 + 表格
    - 扫描件: OCR
    不负责分块，分块由 SemanticChunker 负责。
    支持缓存：parsed 结果保存到 data/parsed/
    """

    def __init__(self, data_dir: Path | None = None):
        """初始化 PDFParser

        Args:
            data_dir: 数据目录路径，默认为 data/
        """
        self.data_dir = data_dir or DEFAULT_DATA_DIR
        self.parsed_dir = self.data_dir / "parsed"
        self.classifier = PDFClassifier()
        self.table_extractor = TableExtractor()
        self.ocr_parser: OCRParser | None = None

    def parse(
        self,
        file_path: str | Path,
        use_cache: bool = True,
        force_reparse: bool = False,
    ) -> list[DocumentChunk]:
        """解析 PDF 文件（兼容旧接口，返回 DocumentChunk）

        Args:
            file_path: PDF 文件路径
            use_cache: 是否使用缓存（默认 True）
            force_reparse: 是否强制重新解析（忽略缓存）

        Returns:
            DocumentChunk 列表（由 SemanticChunker 生成）
        """
        from app.chunkers import SemanticChunker

        file_path = Path(file_path)
        logger.info(f"Parsing PDF: {file_path.name}")

        # 获取原始数据
        raw_data = self.parse_raw(file_path, use_cache, force_reparse)

        # 如果缓存命中，直接返回（缓存中存的是最终 chunks）
        if isinstance(raw_data, list) and len(raw_data) > 0:
            if hasattr(raw_data[0], 'content'):  # DocumentChunk 对象
                logger.info(f"Loaded from cache: {file_path.name}")
                return raw_data

        # 否则，调用 SemanticChunker 生成分块
        if isinstance(raw_data, dict) and 'ocr_results' in raw_data:
            chunker = SemanticChunker()
            chunks = chunker.chunk(
                raw_data['ocr_results'],
                source_path=str(file_path),
            )
            return chunks

        return []

    def parse_raw(
        self,
        file_path: str | Path,
        use_cache: bool = True,
        force_reparse: bool = False,
    ) -> dict[str, Any]:
        """解析 PDF 文件，返回原始数据（供 SemanticChunker 使用）

        Args:
            file_path: PDF 文件路径
            use_cache: 是否使用缓存（默认 True）
            force_reparse: 是否强制重新解析（忽略缓存）

        Returns:
            原始解析数据，包含：
            - ocr_results: OCR 结果列表（扫描件）
            - native_text: 原生 PDF 文本（原生 PDF）
            - pdf_type: PDF 类型
            - page_count: 页数
        """
        file_path = Path(file_path)
        logger.info(f"Parsing PDF (raw): {file_path.name}")

        # 检查缓存
        if use_cache and not force_reparse:
            cached = self._load_raw_from_cache(file_path)
            if cached:
                logger.info(f"Loaded raw from cache: {file_path.name}")
                return cached

        # 判断 PDF 类型
        pdf_type = self.classifier.classify(file_path)
        logger.info(f"PDF type: {pdf_type}")

        raw_data: dict[str, Any] = {
            "pdf_type": pdf_type,
            "file_path": str(file_path),
        }

        try:
            if pdf_type == "scanned":
                raw_data.update(self._parse_scanned_raw(file_path))
            else:
                raw_data.update(self._parse_native_raw(file_path))

            logger.info(f"Extracted raw data from {file_path.name}")

            # 保存缓存
            if use_cache:
                self._save_raw_to_cache(file_path, raw_data)

            return raw_data

        except Exception as e:
            logger.error(f"Failed to parse PDF {file_path}: {e}")
            raise

    def _load_raw_from_cache(self, file_path: Path) -> dict[str, Any] | None:
        """从缓存加载原始解析结果"""
        cache_file = self._get_cache_path(file_path)
        if not cache_file.exists():
            return None

        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            cache_version = data.get("cache_version", 0)
            if cache_version < CACHE_VERSION:
                logger.info(f"Cache version mismatch: {cache_version} < {CACHE_VERSION}, reparsing")
                return None

            logger.info(f"Raw cache hit: {cache_file.name}")
            return data

        except Exception as e:
            logger.warning(f"Failed to load raw cache {cache_file}: {e}")
            return None

    def _save_raw_to_cache(self, file_path: Path, raw_data: dict[str, Any]) -> None:
        """保存原始解析结果到缓存"""
        cache_file = self._get_cache_path(file_path)
        self.parsed_dir.mkdir(parents=True, exist_ok=True)

        data = {
            **raw_data,
            "cache_version": CACHE_VERSION,
            "cached_at": datetime.now().isoformat(),
        }

        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved raw to cache: {cache_file.name}")
        except Exception as e:
            logger.warning(f"Failed to save raw cache {cache_file}: {e}")

    def _get_cache_path(self, file_path: Path) -> Path:
        """获取缓存文件路径"""
        return self.parsed_dir / f"{file_path.stem}.json"

    def _parse_native_raw(self, file_path: Path) -> dict[str, Any]:
        """解析原生 PDF，返回原始数据"""
        with pymupdf.open(file_path) as doc:
            page_count = len(doc)
            pages_text = []
            tables = []

            for page_num in range(page_count):
                page = doc[page_num]
                text = page.get_text()
                pages_text.append({"page": page_num + 1, "text": text})

            # 提取表格
            extracted_tables = self.table_extractor.extract_tables_pymupdf(file_path)
            for table in extracted_tables:
                tables.append(table)

        return {
            "page_count": page_count,
            "pages_text": pages_text,
            "tables": tables,
        }

    def _parse_scanned_raw(self, file_path: Path) -> dict[str, Any]:
        """解析扫描 PDF，返回原始 OCR 结果"""
        if self.ocr_parser is None:
            self.ocr_parser = OCRParser(use_rapidocr=True)

        ocr_results = self.ocr_parser.parse(file_path)

        with pymupdf.open(file_path) as doc:
            page_count = len(doc)

        return {
            "page_count": page_count,
            "ocr_results": ocr_results,
        }