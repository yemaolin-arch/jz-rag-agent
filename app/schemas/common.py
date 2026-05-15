"""通用数据模型

包含 DocumentChunk、TableChunk 等数据结构。
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentChunk:
    """文档块数据结构

    Attributes:
        content: 文本内容
        page: 页码
        chunk_id: 唯一标识
        is_table: 是否为表格块
        section: 章节标识（如有）
        source: 来源文件路径
    """
    content: str
    page: int
    chunk_id: str
    is_table: bool = False
    section: str | None = None
    source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式"""
        return {
            "content": self.content,
            "metadata": {
                "page": self.page,
                "chunk_id": self.chunk_id,
                "is_table": self.is_table,
                "section": self.section,
                "source": self.source,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DocumentChunk":
        """从字典创建实例"""
        metadata = data.get("metadata", {})
        return cls(
            content=data.get("content", ""),
            page=metadata.get("page", 1),
            chunk_id=metadata.get("chunk_id", ""),
            is_table=metadata.get("is_table", False),
            section=metadata.get("section"),
            source=metadata.get("source"),
        )


@dataclass
class ParsedPDF:
    """解析后的 PDF 数据结构

    Attributes:
        file_path: PDF 文件路径
        chunks: 文档块列表
        parsed_at: 解析时间
        pdf_type: PDF 类型 (scanned/native)
    """
    file_path: str
    chunks: list[DocumentChunk] = field(default_factory=list)
    parsed_at: str | None = None
    pdf_type: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式"""
        return {
            "file_path": self.file_path,
            "chunks": [c.to_dict() for c in self.chunks],
            "parsed_at": self.parsed_at,
            "pdf_type": self.pdf_type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ParsedPDF":
        """从字典创建实例"""
        return cls(
            file_path=data.get("file_path", ""),
            chunks=[DocumentChunk.from_dict(c) for c in data.get("chunks", [])],
            parsed_at=data.get("parsed_at"),
            pdf_type=data.get("pdf_type", "unknown"),
        )


@dataclass
class TableChunk:
    """表格块数据结构

    Attributes:
        markdown: Markdown 格式的表格
        page: 页码
        rows: 行数
        cols: 列数
        source: 来源文件路径
    """
    markdown: str
    page: int
    rows: int = 0
    cols: int = 0
    source: str | None = None

    @property
    def is_table(self) -> bool:
        return True

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式"""
        return {
            "markdown": self.markdown,
            "page": self.page,
            "rows": self.rows,
            "cols": self.cols,
            "source": self.source,
        }