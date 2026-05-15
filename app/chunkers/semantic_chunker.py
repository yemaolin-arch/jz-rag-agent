"""语义分块器

将 OCR 结果或原生 PDF 解析结果转换为语义完整的 DocumentChunk 列表。
负责：
1. 表格区域检测与质量评分
2. 劣质表格降级为文本
3. OCR 块的语义合并
"""

import logging
import re
from pathlib import Path
from typing import Any

import pymupdf

from app.schemas.common import DocumentChunk
from app.parsers.ocr_parser import TableReconstructor, merge_ocr_blocks

logger = logging.getLogger(__name__)


class SemanticChunker:
    """语义分块器

    将解析器返回的原始 OCR 结果转换为语义完整的 DocumentChunk。
    包含表格质量评估和降级策略。
    """

    def __init__(self):
        """初始化语义分块器"""
        self.table_reconstructor = TableReconstructor()

    def chunk(
        self,
        ocr_results: list[dict[str, Any]],
        source_path: str,
        page_width: float | None = None,
        page_height: float | None = None,
    ) -> list[DocumentChunk]:
        """将 OCR 结果分块为语义完整的 DocumentChunk

        Args:
            ocr_results: OCR 解析结果列表，每项包含 text, bbox, page
            source_path: PDF 文件路径（用于生成 chunk_id）
            page_width: 页面宽度（可选，默认从 PDF 读取）
            page_height: 页面高度（可选，默认从 PDF 读取）

        Returns:
            DocumentChunk 列表
        """
        if not ocr_results:
            return []

        source_name = str(source_path)
        file_stem = Path(source_path).stem

        chunks = []

        # 按页面处理
        page_results: dict[int, list[dict]] = {}
        for item in ocr_results:
            page = item.get("page", 1)
            if page not in page_results:
                page_results[page] = []
            page_results[page].append(item)

        for page, items in page_results.items():
            # 获取页面尺寸
            if page_width is None or page_height is None:
                try:
                    with pymupdf.open(source_path) as doc:
                        page_obj = doc[page - 1] if page <= len(doc) else doc[0]
                        pw = page_obj.rect.width
                        ph = page_obj.rect.height
                except Exception:
                    pw = 800  # 默认值
                    ph = 600
            else:
                pw = page_width
                ph = page_height

            # 检测表格
            tables = self.table_reconstructor.reconstruct_from_ocr_results(
                items, pw, ph
            )

            # 处理表格：质量评估与降级
            for table_idx, table in enumerate(tables):
                chunk = self._process_table(table, file_stem, page, table_idx, source_name)
                if chunk:
                    chunks.append(chunk)

            # 语义合并非表格文本
            merged_texts = merge_ocr_blocks(items, pw, ph)

            for para_idx, para in enumerate(merged_texts):
                if para.strip() and len(para.strip()) > 5:
                    chunk = DocumentChunk(
                        content=para.strip(),
                        page=page,
                        chunk_id=f"{file_stem}_p{page}_para{para_idx}",
                        is_table=False,
                        section=self._detect_section(para),
                        source=source_name,
                    )
                    chunks.append(chunk)

        return chunks

    def _process_table(
        self,
        table: dict[str, Any],
        file_stem: str,
        page: int,
        table_idx: int,
        source: str,
    ) -> DocumentChunk | None:
        """处理单个表格：质量评估与降级

        Args:
            table: 表格数据，包含 markdown, cells, bbox
            file_stem: 文件名（不含扩展名）
            page: 页码
            table_idx: 表格索引
            source: 源文件路径

        Returns:
            DocumentChunk 或 None（如果表格无效）
        """
        markdown = table.get("markdown", "")
        cells = table.get("cells", [])

        # 计算非空单元格比例
        non_empty = sum(1 for c in cells if c.get("text", "").strip())
        total = len(cells)
        valid_cell_ratio = non_empty / total if total > 0 else 0

        # 计算列数（通过 markdown 第二行，即表头分隔线行）
        col_count = 1
        lines = markdown.split("\n")
        if len(lines) >= 2:
            header_line = lines[1]
            col_count = max(1, header_line.count("|") - 1)
        elif "|" in markdown:
            col_count = max(1, markdown.count("|") // 2)

        # 计算每列平均单元格数（判断是否过于稀疏）
        avg_cells_per_col = total / col_count if col_count > 0 else 0

        # 计算有效列数（至少有2个非空单元格的列）
        effective_cols = non_empty / 2 if non_empty > 0 else 0

        # 降级策略：
        # 1. 非空比例 < 0.5 -> 降级
        # 2. 列数 > 10 -> 降级
        # 3. 每列平均 < 2.5 单元格 -> 降级
        # 4. 有效列数 < 3 -> 降级
        is_sparse = valid_cell_ratio < 0.5
        is_too_wide = col_count > 10
        is_sparse_cols = avg_cells_per_col < 2.5
        is_narrow_table = effective_cols < 3

        logger.info(f"Table check: total={total}, cols={col_count}, eff_cols={effective_cols:.0f}, "
                    f"non_empty={non_empty}, ratio={valid_cell_ratio:.2f}, avg_per_col={avg_cells_per_col:.1f}, "
                    f"sparse={is_sparse}, wide={is_too_wide}, sparse_cols={is_sparse_cols}, narrow={is_narrow_table}")

        if is_sparse or is_narrow_table:
            # 严重问题：表格太稀疏或太小，降级为正文
            logger.info(f"Degraded sparse/narrow table to text: ratio={valid_cell_ratio:.2f}, narrow={is_narrow_table}")
            cell_texts = [c.get("text", "").strip() for c in cells if c.get("text", "").strip()]
            combined_text = "\n".join(cell_texts)
            if combined_text and len(combined_text) > 10:
                return DocumentChunk(
                    content=combined_text,
                    page=page,
                    chunk_id=f"{file_stem}_p{page}_table{table_idx}_text",
                    is_table=False,
                    section="table_degraded",
                    source=source,
                )
        elif is_too_wide or is_sparse_cols:
            # 表格列数过多或每列过疏，保留为文本
            logger.info(f"Keeping table as text (wide/sparse): cols={col_count}, avg={avg_cells_per_col:.1f}")
            cell_texts = [c.get("text", "").strip() for c in cells if c.get("text", "").strip()]
            combined_text = " ".join(cell_texts)
            if combined_text and len(combined_text) > 10:
                return DocumentChunk(
                    content=combined_text,
                    page=page,
                    chunk_id=f"{file_stem}_p{page}_table{table_idx}_text",
                    is_table=False,
                    section="table_as_text",
                    source=source,
                )
        else:
            # 有效表格，保留 markdown
            return DocumentChunk(
                content=markdown,
                page=page,
                chunk_id=f"{file_stem}_p{page}_table{table_idx}",
                is_table=True,
                section="table",
                source=source,
            )

        return None

    def _detect_section(self, text: str) -> str | None:
        """检测文本所属章节"""
        text = text.strip()
        section_patterns = [
            r"^第[一二三四五六七八九十]+[章节]",
            r"^\d+\.\d+",
            r"^\d+\.",
            r"^[A-Z]\.",
        ]

        for pattern in section_patterns:
            if re.match(pattern, text):
                return text[:50]

        return None