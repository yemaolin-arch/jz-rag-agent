"""向量存储模块

封装 FAISS 操作，支持构建索引和检索。
"""

import logging
import os
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from app.core.config_loader import get_full_config
from app.schemas.common import DocumentChunk
from app.services.embedding_service import get_embedding_service

logger = logging.getLogger(__name__)


class VectorStore:
    """FAISS 向量存储

    支持：
    - build_index: 从 DocumentChunk 列表构建索引
    - load_index: 加载已有索引
    - search: 向量检索
    """

    def __init__(self, index_path: str | Path | None = None):
        """初始化 VectorStore

        Args:
            index_path: 索引文件路径，默认从配置读取
        """
        config = get_full_config()
        paths_config = config.get("paths", {})
        self.index_dir = Path(index_path or paths_config.get("vector_store_dir", "./data/vector_store"))
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self._index: faiss.Index | None = None
        self._chunks: list[dict[str, Any]] = []
        self._embedding_service = get_embedding_service()

    @property
    def index(self) -> faiss.Index:
        """获取 FAISS 索引"""
        if self._index is None:
            raise RuntimeError("Index not loaded. Call load_index() first.")
        return self._index

    @property
    def chunks(self) -> list[dict[str, Any]]:
        """获取 chunks 列表"""
        return self._chunks

    def build_index(
        self,
        chunks: list[DocumentChunk],
        index_name: str = "default",
        save: bool = True,
    ) -> None:
        """构建 FAISS 索引

        Args:
            chunks: DocumentChunk 列表
            index_name: 索引名称（用于文件名）
            save: 是否保存索引到磁盘
        """
        if not chunks:
            logger.warning("No chunks provided for indexing")
            return

        logger.info(f"Building index for {len(chunks)} chunks")

        # 提取文本内容
        texts = [c.content for c in chunks]
        chunk_dicts = [c.to_dict() for c in chunks]

        # 生成 embeddings
        logger.info("Generating embeddings...")
        embeddings = self._embedding_service.embed_documents(texts)
        embeddings = np.array(embeddings, dtype=np.float32)

        # 归一化（FAISS L2 需要）
        faiss.normalize_L2(embeddings)

        # 构建索引
        dim = embeddings.shape[1]
        logger.info(f"Embedding dimension: {dim}")

        self._index = faiss.IndexFlatIP(dim)  # Inner Product (cosine sim)
        self._index.add(embeddings)

        self._chunks = chunk_dicts

        logger.info(f"Index built with {self._index.ntotal} vectors")

        if save:
            self.save_index(index_name)

    def save_index(self, index_name: str = "default") -> None:
        """保存索引到磁盘

        Args:
            index_name: 索引名称
        """
        if self._index is None:
            raise RuntimeError("No index to save")

        index_file = self.index_dir / f"{index_name}.index"
        chunk_file = self.index_dir / f"{index_name}_chunks.json"

        logger.info(f"Saving index to {index_file}")

        # 保存 FAISS 索引
        faiss.write_index(self._index, str(index_file))

        # 保存 chunks 元数据
        import json
        with open(chunk_file, "w", encoding="utf-8") as f:
            json.dump(self._chunks, f, ensure_ascii=False, indent=2)

        logger.info(f"Index saved: {self._index.ntotal} vectors")

    def load_index(self, index_name: str = "default") -> bool:
        """加载已有索引

        Args:
            index_name: 索引名称

        Returns:
            是否加载成功
        """
        index_file = self.index_dir / f"{index_name}.index"
        chunk_file = self.index_dir / f"{index_name}_chunks.json"

        if not index_file.exists() or not chunk_file.exists():
            logger.warning(f"Index files not found: {index_name}")
            return False

        try:
            logger.info(f"Loading index from {index_file}")
            self._index = faiss.read_index(str(index_file))

            import json
            with open(chunk_file, "r", encoding="utf-8") as f:
                self._chunks = json.load(f)

            logger.info(f"Index loaded: {self._index.ntotal} vectors, {len(self._chunks)} chunks")
            return True

        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """执行向量检索

        Args:
            query: 查询文本
            k: 返回 top-k 结果

        Returns:
            检索结果列表（包含 chunk 和 score）
        """
        if self._index is None:
            raise RuntimeError("Index not loaded")

        # 生成 query embedding
        query_vec = self._embedding_service.embed_query(query)
        query_vec = np.array([query_vec], dtype=np.float32)
        faiss.normalize_L2(query_vec)

        # 检索
        k = min(k, self._index.ntotal)
        scores, indices = self._index.search(query_vec, k)

        # 组装结果
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self._chunks):
                result = {
                    "rank": i + 1,
                    "score": float(score),
                    "chunk": self._chunks[int(idx)],
                }
                results.append(result)

        return results

    def exists(self, index_name: str = "default") -> bool:
        """检查索引是否存在

        Args:
            index_name: 索引名称

        Returns:
            是否存在
        """
        index_file = self.index_dir / f"{index_name}.index"
        chunk_file = self.index_dir / f"{index_name}_chunks.json"
        return index_file.exists() and chunk_file.exists()

    def get_stats(self) -> dict[str, Any]:
        """获取索引统计信息

        Returns:
            统计信息字典
        """
        if self._index is None:
            return {"loaded": False}

        return {
            "loaded": True,
            "vector_count": self._index.ntotal,
            "chunk_count": len(self._chunks),
            "index_dir": str(self.index_dir),
        }


def build_vector_index(
    chunks: list[DocumentChunk],
    index_name: str = "default",
    index_path: str | Path | None = None,
) -> VectorStore:
    """构建向量索引的便捷函数

    Args:
        chunks: DocumentChunk 列表
        index_name: 索引名称
        index_path: 索引目录

    Returns:
        VectorStore 实例
    """
    store = VectorStore(index_path=index_path)
    store.build_index(chunks, index_name=index_name)
    return store