"""表格提取模块

将 PDF 中的表格提取为 Markdown 格式。
- 扫描件：使用 PaddleOCR table=True 进行表格识别
- 原生 PDF：使用 pymupdf / pdfplumber 提取
"""

import logging
from pathlib import Path
from typing import Any

import fitz

logger = logging.getLogger(__name__)


class TableExtractor:
    """表格提取器，将 PDF 表格转换为 Markdown 格式"""

    @staticmethod
    def extract_tables_pymupdf(
        file_path: str | Path, page_numbers: list[int] | None = None
    ) -> list[dict[str, Any]]:
        """使用 pymupdf 提取表格（适合原生 PDF）

        Args:
            file_path: PDF 文件路径
            page_numbers: 指定页面列表，None 表示所有页面

        Returns:
            表格列表，每项包含 page, table_index, markdown, bbox
        """
        file_path = Path(file_path)
        tables = []

        try:
            with fitz.open(file_path) as doc:
                pages_to_check = page_numbers or range(len(doc))

                for page_num in pages_to_check:
                    page = doc[page_num]

                    # 获取页面的表格检测结果
                    tabs = page.find_tables()
                    if tabs is None:
                        continue

                    for table_idx, tab in enumerate(tabs):
                        try:
                            # 提取表格数据
                            data = tab.extract()
                            if not data or not data[0]:
                                continue

                            markdown = TableExtractor._to_markdown(data)
                            tables.append({
                                "page": page_num + 1,
                                "table_index": table_idx,
                                "markdown": markdown,
                                "bbox": tab.bbox,
                            })
                            logger.info(
                                f"Extracted table from page {page_num + 1}, "
                                f"table {table_idx}: {len(data)} rows"
                            )
                        except Exception as e:
                            logger.warning(
                                f"Failed to extract table {table_idx} on page {page_num + 1}: {e}"
                            )
                            continue

        except Exception as e:
            logger.error(f"Failed to extract tables with pymupdf: {e}")

        return tables

    @staticmethod
    def extract_tables_paddleocr(
        file_path: str | Path, page_numbers: list[int] | None = None
    ) -> list[dict[str, Any]]:
        """使用 PaddleOCR 提取表格（适合扫描件）

        Args:
            file_path: PDF 文件路径
            page_numbers: 指定页面列表，None 表示所有页面

        Returns:
            表格列表，每项包含 page, table_index, markdown, bbox
        """
        file_path = Path(file_path)
        tables = []

        try:
            from paddleocr import PaddleOCR

            ocr = PaddleOCR(
                use_angle_cls=True,
                lang="ch",
                use_gpu=False,
                table=True,  # 启用表格识别
            )
            logger.info("Using PaddleOCR table mode for scanned PDF")

            with fitz.open(file_path) as doc:
                pages_to_check = page_numbers or range(len(doc))

                for page_num in pages_to_check:
                    page = doc[page_num]
                    # 渲染为高分辨率图像
                    mat = fitz.Matrix(2.0, 2.0)
                    pix = page.get_pixmap(matrix=mat)
                    img_data = pix.tobytes("png")

                    # 使用 PaddleOCR 表格识别
                    result = ocr.ocr(img_data, table=True)

                    if result and result[0]:
                        for table_idx, table_result in enumerate(result[0]):
                            if not table_result:
                                continue

                            # table_result 格式: (bbox, {html: str})
                            if isinstance(table_result, list) and len(table_result) >= 2:
                                bbox = table_result[0]
                                html_content = table_result[1].get("html", "")

                                # 将 HTML 表格转换为 Markdown
                                markdown = TableExtractor._html_table_to_markdown(html_content)

                                if markdown:
                                    tables.append({
                                        "page": page_num + 1,
                                        "table_index": table_idx,
                                        "markdown": markdown,
                                        "bbox": bbox,
                                    })
                                    logger.info(
                                        f"PaddleOCR extracted table from page {page_num + 1}, "
                                        f"table {table_idx}"
                                    )

        except ImportError as e:
            logger.error(f"PaddleOCR not available: {e}")
        except Exception as e:
            logger.error(f"Failed to extract tables with PaddleOCR: {e}")

        return tables

    @staticmethod
    def _html_table_to_markdown(html_table: str) -> str:
        """将 HTML 表格转换为 Markdown 格式

        Args:
            html_table: HTML 表格字符串

        Returns:
            Markdown 格式字符串
        """
        import re

        if not html_table:
            return ""

        try:
            from bs4 import BeautifulSoup
        except ImportError:
            # 如果没有 bs4，使用正则表达式解析
            return TableExtractor._html_table_to_markdown_regex(html_table)

        try:
            soup = BeautifulSoup(html_table, "html.parser")
            table = soup.find("table")
            if not table:
                return ""

            rows = table.find_all("tr")
            if not rows:
                return ""

            markdown_rows = []

            for row_idx, row in enumerate(rows):
                cells = row.find_all(["th", "td"])
                if not cells:
                    continue

                cell_contents = []
                for cell in cells:
                    text = cell.get_text(strip=True)
                    cell_contents.append(text)

                # 第一行作为表头
                if row_idx == 0:
                    header_line = "| " + " | ".join(cell_contents) + " |"
                    markdown_rows.append(header_line)

                    # 分隔线
                    sep_line = "|" + "|".join(["---"] * len(cell_contents)) + "|"
                    markdown_rows.append(sep_line)
                else:
                    row_line = "| " + " | ".join(cell_contents) + " |"
                    markdown_rows.append(row_line)

            return "\n".join(markdown_rows)

        except Exception as e:
            logger.warning(f"Failed to parse HTML table: {e}")
            return TableExtractor._html_table_to_markdown_regex(html_table)

    @staticmethod
    def _html_table_to_markdown_regex(html_table: str) -> str:
        """使用正则表达式解析 HTML 表格（备用方案）"""
        import re

        if not html_table:
            return ""

        # 提取所有 <tr> 内容
        row_pattern = re.compile(r"<tr[^>]*>(.*?)</tr>", re.DOTALL | re.IGNORECASE)
        cell_pattern = re.compile(r"<t[hd][^>]*>(.*?)</t[hd]>", re.DOTALL | re.IGNORECASE)

        rows = row_pattern.findall(html_table)
        if not rows:
            return ""

        markdown_rows = []

        for row_idx, row_html in enumerate(rows):
            cells = cell_pattern.findall(row_html)
            if not cells:
                continue

            # 清理单元格内容
            cell_contents = []
            for cell in cells:
                text = re.sub(r"<[^>]+>", "", cell).strip()
                text = text.replace("&nbsp;", " ").replace("&amp;", "&")
                cell_contents.append(text)

            if row_idx == 0:
                header_line = "| " + " | ".join(cell_contents) + " |"
                markdown_rows.append(header_line)
                sep_line = "|" + "|".join(["---"] * len(cell_contents)) + "|"
                markdown_rows.append(sep_line)
            else:
                row_line = "| " + " | ".join(cell_contents) + " |"
                markdown_rows.append(row_line)

        return "\n".join(markdown_rows)

    @staticmethod
    def _to_markdown(table_data: list[list[str]]) -> str:
        """将表格数据（2D list）转换为 Markdown 格式

        Args:
            table_data: 2D 列表 [[cell1, cell2], [cell3, cell4], ...]

        Returns:
            Markdown 格式字符串，如 | 列1 | 列2 |
        """
        if not table_data:
            return ""

        # 计算每列最大宽度
        col_widths = []
        for row in table_data:
            for col_idx, cell in enumerate(row):
                cell_len = len(str(cell).strip())
                if col_idx >= len(col_widths):
                    col_widths.append(cell_len)
                else:
                    col_widths[col_idx] = max(col_widths[col_idx], cell_len)

        lines = []

        # 表头
        header = table_data[0]
        header_line = "| " + " | ".join(
            str(cell).strip().ljust(col_widths[i])
            for i, cell in enumerate(header)
        ) + " |"
        lines.append(header_line)

        # 分隔线
        sep_line = "|" + "|".join(
            "-" * (col_widths[i] + 2) for i in range(len(col_widths))
        ) + "|"
        lines.append(sep_line)

        # 数据行
        for row in table_data[1:]:
            row_line = "| " + " | ".join(
                str(cell).strip().ljust(col_widths[i]) if i < len(row) else ""
                for i in range(len(col_widths))
            ) + " |"
            lines.append(row_line)

        return "\n".join(lines)

    @staticmethod
    def extract_table_as_chunk(
        file_path: str | Path,
        page: int,
        markdown: str,
        table_caption: str | None = None,
    ) -> dict[str, Any]:
        """将表格转换为文档 chunk 格式

        Args:
            file_path: PDF 文件路径
            page: 页码
            markdown: Markdown 格式的表格
            table_caption: 表格标题（可选）

        Returns:
            包含表格内容和元数据的字典
        """
        file_path = Path(file_path)

        content = markdown
        if table_caption:
            content = f"**表格 {table_caption}**\n\n{markdown}"

        return {
            "content": content,
            "metadata": {
                "source": str(file_path),
                "page": page,
                "is_table": True,
                "section": "table",
            },
        }

    @staticmethod
    def extract(
        file_path: str | Path,
        page_numbers: list[int] | None = None,
        is_scanned: bool = False,
    ) -> list[dict[str, Any]]:
        """统一的表格提取入口

        Args:
            file_path: PDF 文件路径
            page_numbers: 指定页面列表，None 表示所有页面
            is_scanned: 是否为扫描件

        Returns:
            表格列表，每项包含 page, table_index, markdown, bbox
        """
        if is_scanned:
            return TableExtractor.extract_tables_paddleocr(file_path, page_numbers)
        else:
            return TableExtractor.extract_tables_pymupdf(file_path, page_numbers)