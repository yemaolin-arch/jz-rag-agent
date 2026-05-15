"""混合检索模块

实现 BM25 + FAISS 混合检索，使用 RRF 算法融合结果。
"""

import json
import logging
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from app.core.config_loader import get_full_config
from app.retrievers.vector_store import VectorStore
from app.schemas.common import DocumentChunk

logger = logging.getLogger(__name__)

# RRF 参数：k 值越大，排名差异的影响越小
RRF_K = 60


class HybridRetriever:
    """混合检索器

    结合 BM25（关键词精确匹配）和 FAISS（语义向量匹配），
    使用 RRF (Reciprocal Rank Fusion) 算法融合结果。
    """

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        bm25_path: str | Path | None = None,
    ):
        """初始化混合检索器

        Args:
            vector_store: VectorStore 实例，默认创建新实例
            bm25_path: BM25 索引文件路径
        """
        config = get_full_config()
        paths_config = config.get("paths", {})
        self.bm25_dir = Path(bm25_path or paths_config.get("chunks_dir", "./data/chunks"))
        self.bm25_dir.mkdir(parents=True, exist_ok=True)

        self._vector_store = vector_store or VectorStore()
        self._bm25: BM25Okapi | None = None
        self._chunks: list[dict[str, Any]] = []
        self._loaded = False

    @property
    def vector_store(self) -> VectorStore:
        """获取 VectorStore"""
        return self._vector_store

    def load(self, index_name: str = "default") -> bool:
        """加载向量索引和 BM25 索引

        Args:
            index_name: 索引名称

        Returns:
            是否加载成功
        """
        # 加载向量索引
        vs_loaded = self._vector_store.load_index(index_name)
        if not vs_loaded:
            logger.error("Failed to load vector store")
            return False

        # 加载 BM25
        bm25_file = self.bm25_dir / f"{index_name}_bm25.json"
        if bm25_file.exists():
            try:
                with open(bm25_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    corpus = data.get("corpus", [])
                    self._chunks = data.get("chunks", [])

                if corpus:
                    logger.info(f"Loading BM25 with {len(corpus)} documents")
                    self._bm25 = BM25Okapi(corpus)
                    self._loaded = True
                    logger.info("Both vector and BM25 indexes loaded")
                    return True
            except Exception as e:
                logger.error(f"Failed to load BM25: {e}")

        # 如果没有 BM25 文件，使用向量索引的 chunks
        self._chunks = self._vector_store.chunks
        logger.warning("BM25 index not found, using vector store only")
        return vs_loaded

    def build(
        self,
        chunks: list[DocumentChunk],
        index_name: str = "default",
        save: bool = True,
    ) -> None:
        """构建混合索引

        Args:
            chunks: DocumentChunk 列表
            index_name: 索引名称
            save: 是否保存
        """
        # 构建向量索引
        self._vector_store.build_index(chunks, index_name, save=save)
        self._chunks = [c.to_dict() if isinstance(c, DocumentChunk) else c for c in chunks]

        # 构建 BM25 索引
        corpus = [c.get("content", "") for c in self._chunks]
        if corpus:
            logger.info(f"Building BM25 with {len(corpus)} documents")
            self._bm25 = BM25Okapi(corpus)

            if save:
                bm25_file = self.bm25_dir / f"{index_name}_bm25.json"
                data = {
                    "corpus": corpus,
                    "chunks": self._chunks,
                }
                with open(bm25_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                logger.info(f"BM25 index saved to {bm25_file}")

        self._loaded = True

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
    ) -> list[dict[str, Any]]:
        """执行混合检索

        Args:
            query: 查询文本
            top_k: 返回 top-k 结果
            vector_weight: 向量检索权重
            bm25_weight: BM25 权重

        Returns:
            融合后的检索结果列表
        """
        # 向量检索
        vector_results = self._vector_store.search(query, k=top_k * 2)

        # BM25 检索
        bm25_results = []
        if self._bm25 is not None:
            bm25_scores = self._bm25.get_scores(query.split())
            top_indices = bm25_scores.argsort()[::-1][:top_k * 2]
            bm25_results = [
                {
                    "rank": i + 1,
                    "score": float(bm25_scores[idx]),
                    "chunk": self._chunks[idx],
                }
                for i, idx in enumerate(top_indices)
                if idx < len(self._chunks)
            ]

        # RRF 融合
        fused = self._rrf_fusion(vector_results, bm25_results, top_k)

        logger.info(
            f"Retrieved {len(fused)} results (vector: {len(vector_results)}, "
            f"bm25: {len(bm25_results)}, weights: v={vector_weight}, bm25={bm25_weight})"
        )

        return fused

    def _rrf_fusion(
        self,
        vector_results: list[dict[str, Any]],
        bm25_results: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """RRF (Reciprocal Rank Fusion) 算法融合结果

        RRF score = weight_v / (k + rank_v) + weight_bm25 / (k + rank_bm25)

        Args:
            vector_results: 向量检索结果
            bm25_results: BM25 检索结果
            top_k: 返回数量

        Returns:
            融合后的结果
        """
        # 构建 chunk_id 到结果的映射
        fused_scores: dict[str, float] = {}

        # 添加向量检索分数
        for result in vector_results:
            chunk_id = result["chunk"].get("metadata", {}).get("chunk_id", "")
            if chunk_id:
                # RRF 公式: weight / (k + rank)
                score = 0.7 / (RRF_K + result["rank"])
                fused_scores[chunk_id] = fused_scores.get(chunk_id, 0) + score

        # 添加 BM25 分数
        for result in bm25_results:
            chunk_id = result["chunk"].get("metadata", {}).get("chunk_id", "")
            if chunk_id:
                score = 0.3 / (RRF_K + result["rank"])
                fused_scores[chunk_id] = fused_scores.get(chunk_id, 0) + score

        # 按分数排序
        sorted_chunks = sorted(
            fused_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        # 构建最终结果
        chunk_map = {c.get("metadata", {}).get("chunk_id", ""): c for c in self._chunks}
        results = []
        for rank, (chunk_id, score) in enumerate(sorted_chunks, 1):
            if chunk_id in chunk_map:
                results.append({
                    "rank": rank,
                    "score": score,
                    "chunk": chunk_map[chunk_id],
                })

        return results

    def get_stats(self) -> dict[str, Any]:
        """获取检索器统计信息"""
        vs_stats = self._vector_store.get_stats()
        return {
            "loaded": self._loaded,
            "vector_store": vs_stats,
            "bm25_loaded": self._bm25 is not None,
            "chunks_count": len(self._chunks),
        }


def create_hybrid_retriever(
    chunks: list[DocumentChunk] | None = None,
    index_name: str = "default",
) -> HybridRetriever:
    """创建混合检索器的便捷函数

    Args:
        chunks: 可选，预构建 chunks
        index_name: 索引名称

    Returns:
        HybridRetriever 实例
    """
    retriever = HybridRetriever()

    if chunks:
        retriever.build(chunks, index_name, save=True)
    else:
        retriever.load(index_name)

    return retriever