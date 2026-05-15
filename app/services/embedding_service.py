"""Embedding 服务模块

单例模式加载 HuggingFaceEmbeddings 模型。
首次运行需要联网下载模型（约 2GB）。
"""

import logging
from pathlib import Path
from typing import Any

from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config_loader import get_full_config

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Embedding 服务单例

    使用 LangChain 的 HuggingFaceEmbeddings 加载本地模型。
    模型配置从 configs/config.yaml 读取。
    """

    _instance: "EmbeddingService | None" = None
    _embeddings: HuggingFaceEmbeddings | None = None

    def __new__(cls) -> "EmbeddingService":
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """初始化（仅首次）"""
        if self._embeddings is None:
            self._initialize()

    def _initialize(self) -> None:
        """初始化 embedding 模型"""
        config = get_full_config()
        embedding_config = config.get("embedding", {})
        model_name = embedding_config.get("model", "BAAI/bge-m3")
        device = embedding_config.get("device", "cpu")
        batch_size = embedding_config.get("batch_size", 32)

        logger.info(f"Initializing EmbeddingService with model: {model_name}")

        try:
            self._embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": device},
                encode_kwargs={"batch_size": batch_size, "normalize_embeddings": True},
            )
            logger.info(f"Embedding model loaded successfully on {device}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """获取 embeddings 实例"""
        if self._embeddings is None:
            self._initialize()
        return self._embeddings

    def embed_query(self, text: str) -> list[float]:
        """对单个查询进行 embedding

        Args:
            text: 查询文本

        Returns:
            embedding 向量
        """
        return self.embeddings.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """对多个文档进行 embedding

        Args:
            texts: 文档列表

        Returns:
            embedding 向量列表
        """
        return self.embeddings.embed_documents(texts)

    def embed_chunks(self, chunks: list[dict[str, Any]]) -> list[list[float]]:
        """对 DocumentChunk 列表进行 embedding

        Args:
            chunks: DocumentChunk 列表（来自 app.schemas.common.DocumentChunk.to_dict()）

        Returns:
            embedding 向量列表
        """
        texts = [c.get("content", "") for c in chunks]
        return self.embed_documents(texts)


def get_embedding_service() -> EmbeddingService:
    """获取 EmbeddingService 单例

    Returns:
        EmbeddingService 实例
    """
    return EmbeddingService()