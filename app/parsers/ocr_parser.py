"""OCR 解析器模块

使用 rapidocr_onnxruntime 或 paddleocr 对扫描 PDF 进行 OCR。
包含 OCRNormalizer 用于清洗常见的 OCR 错误。
"""

import logging
import re
from pathlib import Path
from typing import Optional, Any

logger = logging.getLogger(__name__)


class OCRNormalizer:
    """OCR 结果标准化器

    清洗常见的 OCR 错误，包括：
    - l/1 混淆：仅在纯数字串中 l 和 1 混淆才修正（如 "l23" → "123"，"12l3" → "1213"）
    - O/0 混淆：仅在数字上下文中的 O 才转为 0
    - 空格乱码：多余空格、丢失空格
    - 编号错乱：连续空格、Tab
    """

    @classmethod
    def normalize(cls, text: str) -> str:
        """标准化 OCR 文本

        Args:
            text: 原始 OCR 文本

        Returns:
            清洗后的文本
        """
        if not text:
            return ""

        result = text

        # 修复空格问题（先做，避免干扰后面的模式匹配）
        result = re.sub(r"[ \t]{2,}", " ", result)  # 多个空格/Tab 合并
        result = re.sub(r"\n[ \t]+", "\n", result)  # 行首多余空格/tab
        result = re.sub(r"[ \t]+\n", "\n", result)  # 行尾多余空格/tab
        result = re.sub(r"^[ \t]+|[ \t]+$", "", result, flags=re.MULTILINE)  # 去除首尾空格/tab（非换行）

        # 修复 l/1 混淆：仅在纯数字字符串中转换
        # 例如 "l23" → "123", "12l3" → "1213", 但 "label" 不变
        result = cls._fix_l_1_in_digits(result)

        # 修复 O/0 混淆：仅当 O 在数字之间或数字与 O 连续时转换
        result = cls._fix_O_0_in_digits(result)

        return result

    @classmethod
    def _fix_l_1_in_digits(cls, text: str) -> str:
        """修复连续数字串中的 l/1 混淆

        只有当 l 前后都是数字时，才将 l 转为 1。
        例如：12l3 → 1213, 但 l23、l2l 保持不变
        """
        return re.sub(r"(?<=\d)l(?=\d)", "1", text)

    @classmethod
    def _fix_O_0_in_digits(cls, text: str) -> str:
        """修复连续数字串中的 O/0 混淆

        只有当 O 前后都是数字时，才将 O 转为 0。
        例如：12O3 → 1203, 但 O123、O0O 保持不变
        """
        return re.sub(r"(?<=\d)O(?=\d)", "0", text)


class OCRParser:
    """OCR 解析器

    支持 rapidocr_onnxruntime（轻量级）和 paddleocr 两种引擎。
    优先使用 rapidocr，如果不可用则回退到 paddleocr。
    """
    # 保留在类中供 TableReconstructor 使用
    ResultType = list[dict[str, any]]

    def __init__(self, use_rapidocr: bool = True):
        """初始化 OCR 解析器

        Args:
            use_rapidocr: 是否优先使用 rapidocr，False 则使用 paddleocr
        """
        self._use_rapidocr = use_rapidocr
        self._engine = None
        self._engine_type: Optional[str] = None
        self._init_engine()

    def _init_engine(self) -> None:
        """初始化 OCR 引擎"""
        if self._use_rapidocr:
            try:
                from rapidocr_onnxruntime import RapidOCR
                self._engine = RapidOCR()
                self._engine_type = "rapidocr"
                logger.info("OCR engine: rapidocr_onnxruntime")
            except ImportError:
                logger.warning("rapidocr_onnxruntime not available, falling back to paddleocr")
                self._use_rapidocr = False

        if not self._use_rapidocr or self._engine is None:
            try:
                from paddleocr import PaddleOCR
                self._engine = PaddleOCR(
                    use_angle_cls=True,
                    lang="ch",
                    use_gpu=False,
                    show_log=False,
                )
                self._engine_type = "paddleocr"
                logger.info("OCR engine: paddleocr")
            except ImportError:
                logger.error("Neither rapidocr_onnxruntime nor paddleocr is available")
                raise ImportError("No OCR engine available. Install rapidocr-onnxruntime or paddleocr.")

    def parse(self, file_path: str | Path) -> list[dict[str, any]]:
        """对 PDF 进行 OCR

        Args:
            file_path: PDF 文件路径

        Returns:
            OCR 结果列表，每项包含 text, bbox, page
        """
        file_path = Path(file_path)
        logger.info(f"Starting OCR for: {file_path.name}")

        try:
            import fitz  # pymupdf

            # 将 PDF 转换为图片
            images = []
            with fitz.open(file_path) as doc:
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    # 渲染为高分辨率图像
                    mat = fitz.Matrix(2.0, 2.0)  # 2x zoom
                    pix = page.get_pixmap(matrix=mat)
                    img_data = pix.tobytes("png")
                    images.append((page_num, img_data))

            results = []
            for page_num, img_data in images:
                if self._engine_type == "rapidocr":
                    ocr_result, elapse = self._engine(img_data)
                    if ocr_result:
                        for item in ocr_result:
                            bbox, text, score = item
                            normalized_text = OCRNormalizer.normalize(text)
                            results.append({
                                "text": normalized_text,
                                "bbox": bbox,
                                "page": page_num + 1,
                                "score": score,
                            })
                else:  # paddleocr
                    result = self._engine.ocr(img_data, cls=True)
                    if result and result[0]:
                        for line in result[0]:
                            if line:
                                bbox = line[0]
                                text = line[1][0]
                                score = line[1][1]
                                normalized_text = OCRNormalizer.normalize(text)
                                results.append({
                                    "text": normalized_text,
                                    "bbox": bbox,
                                    "page": page_num + 1,
                                    "score": score,
                                })

                logger.info(f"OCR page {page_num + 1}: {len(results) if results else 0} text blocks")

            logger.info(f"OCR completed: {len(results)} text blocks extracted from {file_path.name}")
            return results

        except Exception as e:
            logger.error(f"OCR failed for {file_path}: {e}")
            raise


def merge_ocr_blocks(
    ocr_results: list[dict[str, Any]],
    page_width: float,
    page_height: float,
    line_height_ratio: float = 1.5,
) -> list[str]:
    """合并 OCR 文本块为语义段落

    策略：
    1. 按 Y 坐标（行）排序文本块
    2. 同一行内的块按 X 坐标排序并合并
    3. 行间距离小于 line_height * line_height_ratio 的行视为同一段落
    4. 不同段落的行之间插入换行符

    Args:
        ocr_results: OCR 结果列表
        page_width: 页面宽度
        page_height: 页面高度
        line_height_ratio: 行高倍数阈值（超过此值则断开段落）

    Returns:
        合并后的段落列表
    """
    if not ocr_results:
        return []

    # 转换为统一格式
    blocks = []
    for item in ocr_results:
        bbox = item.get("bbox", [])
        if not bbox or len(bbox) < 4:
            continue
        text = item.get("text", "").strip()
        if not text:
            continue

        # bbox: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        x1, y1 = bbox[0]
        x2, y2 = bbox[2]
        blocks.append({
            "text": text,
            "x1": x1, "y1": y1,
            "x2": x2, "y2": y2,
            "cx": (x1 + x2) / 2,
            "cy": (y1 + y2) / 2,
            "width": x2 - x1,
            "height": y2 - y1,
        })

    if not blocks:
        return []

    # 按 Y 坐标（行）排序
    blocks.sort(key=lambda b: (b["cy"], b["x1"]))

    # 计算平均行高
    heights = [b["height"] for b in blocks if b["height"] > 0]
    avg_line_height = sum(heights) / len(heights) if heights else 20

    # 合并同一行的文本块
    rows: list[list[dict]] = []
    current_row: list[dict] = []
    current_row_y = None

    for block in blocks:
        if not current_row:
            current_row.append(block)
            current_row_y = block["cy"]
        else:
            # 检查是否与当前行在同一水平线上
            y_diff = abs(block["cy"] - current_row_y)
            if y_diff <= avg_line_height * 0.5:  # 行高的一半作为同行的阈值
                current_row.append(block)
            else:
                # 新行
                rows.append(current_row)
                current_row = [block]
                current_row_y = block["cy"]

    if current_row:
        rows.append(current_row)

    # 按 X 坐标排序每行，然后合并
    merged_rows = []
    for row in rows:
        row.sort(key=lambda b: b["x1"])
        merged_text = " ".join(b["text"] for b in row)
        merged_rows.append(merged_text)

    # 按段落合并（行间距超过阈值则断开）
    paragraphs: list[str] = []
    current_para_lines: list[str] = []
    last_row_y = None

    for i, row_text in enumerate(merged_rows):
        row_block = rows[i][0]  # 取该行第一个块的位置信息
        row_y = row_block["cy"]

        if last_row_y is None:
            current_para_lines.append(row_text)
        else:
            # 检查行间距
            row_spacing = row_y - last_row_y
            if row_spacing <= avg_line_height * line_height_ratio:
                # 同一段落
                current_para_lines.append(row_text)
            else:
                # 新的段落
                if current_para_lines:
                    paragraphs.append("\n".join(current_para_lines))
                current_para_lines = [row_text]

        last_row_y = row_y

    if current_para_lines:
        paragraphs.append("\n".join(current_para_lines))

    return paragraphs


class TableReconstructor:
    """基于 OCR 坐标的表格区域检测与重建

    通过分析 OCR 文本块的坐标分布，检测网格状排列的区域，
    将这些文本块重新拼接为 Markdown 表格。
    """

    def __init__(
        self,
        col_threshold: float = 100.0,
        row_threshold: float = 50.0,
        min_cols: int = 2,
        min_rows: int = 2,
        min_table_height: float = 0.03,
        min_cells_for_table: int = 6,
        x_cluster_tolerance: float = 20.0,
    ):
        """初始化表格重建器

        Args:
            col_threshold: 列间最小间距（像素）
            row_threshold: 行间最小间距（像素）
            min_cols: 最小列数
            min_rows: 最小行数
            min_table_height: 最小表格高度（相对于页面高度）
            min_cells_for_table: 最小单元格数量才能构成表格
            x_cluster_tolerance: x 坐标聚类容差（像素）
        """
        self.col_threshold = col_threshold
        self.row_threshold = row_threshold
        self.min_cols = min_cols
        self.min_rows = min_rows
        self.min_table_height = min_table_height
        self.min_cells_for_table = min_cells_for_table
        self.x_cluster_tolerance = x_cluster_tolerance

    def reconstruct_from_ocr_results(
        self,
        ocr_results: list[dict[str, any]],
        page_width: float,
        page_height: float,
    ) -> list[dict[str, any]]:
        """从 OCR 结果中重建表格

        Args:
            ocr_results: OCR 解析结果列表，每项包含 text, bbox, page
            bbox: [x1, y1, x2, y2] 坐标
            page_width: 页面宽度
            page_height: 页面高度

        Returns:
            表格列表，每项包含 page, table_index, markdown, cells
        """
        if not ocr_results:
            return []

        # 按页面分组
        page_results: dict[int, list[dict]] = {}
        for item in ocr_results:
            page = item.get("page", 1)
            if page not in page_results:
                page_results[page] = []
            page_results[page].append(item)

        tables = []
        table_idx = 0

        for page, items in page_results.items():
            page_tables = self._detect_tables_on_page(
                items, page_width, page_height
            )
            for table in page_tables:
                table["page"] = page
                table["table_index"] = table_idx
                tables.append(table)
                table_idx += 1

        return tables

    def _detect_tables_on_page(
        self,
        items: list[dict],
        page_width: float,
        page_height: float,
    ) -> list[dict[str, any]]:
        """在单个页面上检测表格区域"""
        if len(items) < self.min_rows * self.min_cols:
            return []

        # 提取所有文本块的边界框
        cells = []
        for item in items:
            bbox = item.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            text = item.get("text", "")
            if not text.strip():
                continue

            # bbox 格式: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            x1, y1 = bbox[0]
            x2, y2 = bbox[2]
            cells.append({
                "text": text.strip(),
                "x1": x1, "y1": y1,
                "x2": x2, "y2": y2,
                "cx": (x1 + x2) / 2,
                "cy": (y1 + y2) / 2,
                "width": x2 - x1,
                "height": y2 - y1,
            })

        if len(cells) < self.min_rows * self.min_cols:
            return []

        # 分析垂直分布找列边界
        col_boundaries = self._find_vertical_boundaries(cells, page_width)

        # 分析水平分布找行边界
        row_boundaries = self._find_horizontal_boundaries(cells, page_height)

        # 如果列数和行数都足够多，认为是表格
        if len(col_boundaries) >= self.min_cols + 1 and len(row_boundaries) >= self.min_rows + 1:
            table_height = row_boundaries[-1] - row_boundaries[0]
            if table_height >= page_height * self.min_table_height:
                # 构建表格结构
                markdown = self._build_table_markdown(
                    cells, col_boundaries, row_boundaries
                )
                if markdown:
                    return [{
                        "markdown": markdown,
                        "cells": cells,
                        "bbox": self._compute_table_bbox(cells),
                    }]

        return []

    def _find_vertical_boundaries(
        self, cells: list[dict], page_width: float
    ) -> list[float]:
        """通过文本块 x 坐标聚类找列边界

        策略：收集所有单元格的中心 x 坐标，用聚类方法找列，
        然后取每列的边界（最左和最右）。
        """
        if len(cells) < 2:
            return [0.0, page_width]

        # 收集所有 x 中心坐标
        x_centers = sorted([cell["cx"] for cell in cells])

        # 聚类：x 坐标差值小于 tolerance 的归为一类
        clusters: list[list[float]] = []
        for x in x_centers:
            if not clusters:
                clusters.append([x])
            elif x - clusters[-1][-1] <= self.x_cluster_tolerance:
                clusters[-1].append(x)
            else:
                clusters.append([x])

        if len(clusters) < 2:
            return [0.0, page_width]

        # 计算每个聚类的边界
        col_boundaries = []
        for i, cluster in enumerate(clusters):
            col_min = min(c["x1"] for c in cells if abs(c["cx"] - cluster[0]) <= self.x_cluster_tolerance)
            col_max = max(c["x2"] for c in cells if abs(c["cx"] - cluster[0]) <= self.x_cluster_tolerance)
            if i == 0:
                col_boundaries.append(0.0)
                col_boundaries.append(col_min)
            else:
                col_boundaries.append(col_min)
                col_boundaries.append(col_max)

        col_boundaries.append(page_width)

        # 去重并排序
        col_boundaries = sorted(set(col_boundaries))

        return col_boundaries

    def _find_horizontal_boundaries(
        self, cells: list[dict], page_height: float
    ) -> list[float]:
        """通过文本块 y 坐标聚类找行边界"""
        if len(cells) < 2:
            return [0.0, page_height]

        y_edges = []
        for cell in cells:
            y_edges.append(cell["y1"])
            y_edges.append(cell["y2"])

        y_edges.sort()

        row_boundaries = [0.0]  # 页面上边缘

        for i in range(1, len(y_edges)):
            gap = y_edges[i] - y_edges[i - 1]
            if gap > self.row_threshold:
                boundary = (y_edges[i] + y_edges[i - 1]) / 2
                row_boundaries.append(boundary)

        row_boundaries.append(page_height)  # 页面下边缘
        return row_boundaries

        row_boundaries.append(page_height)  # 页面下边缘
        return row_boundaries

    def _build_table_markdown(
        self,
        cells: list[dict],
        col_boundaries: list[float],
        row_boundaries: list[float],
    ) -> str:
        """根据行列边界构建 Markdown 表格"""
        # 将每个单元格分配到行列
        rows_data: list[list[str]] = []

        for row_idx in range(len(row_boundaries) - 1):
            row_cells = ["" for _ in range(len(col_boundaries) - 1)]
            rows_data.append(row_cells)

        for cell in cells:
            # 找到所属列
            col_idx = None
            for i in range(len(col_boundaries) - 1):
                if col_boundaries[i] <= cell["cx"] < col_boundaries[i + 1]:
                    col_idx = i
                    break

            # 找到所属行
            row_idx = None
            for i in range(len(row_boundaries) - 1):
                if row_boundaries[i] <= cell["cy"] < row_boundaries[i + 1]:
                    row_idx = i
                    break

            if col_idx is not None and row_idx is not None:
                rows_data[row_idx][col_idx] = cell["text"]

        # 构建 Markdown
        if not rows_data:
            return ""

        # 表头
        lines = []
        header = rows_data[0]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")

        # 数据行
        for row in rows_data[1:]:
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)

    def _compute_table_bbox(self, cells: list[dict]) -> list[float]:
        """计算表格的边界框"""
        if not cells:
            return [0, 0, 0, 0]

        min_x = min(c["x1"] for c in cells)
        min_y = min(c["y1"] for c in cells)
        max_x = max(c["x2"] for c in cells)
        max_y = max(c["y2"] for c in cells)

        return [min_x, min_y, max_x, max_y]