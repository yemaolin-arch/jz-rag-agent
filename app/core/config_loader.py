"""配置加载模块

从 YAML 文件和环境变量加载配置，支持配置合并和默认值。
"""

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


def load_env() -> None:
    """加载 .env 文件到环境变量"""
    project_root = Path(__file__).parent.parent.parent
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    else:
        load_dotenv()


class PathSettings(BaseSettings):
    """路径配置"""
    data_dir: str = Field(default="./data", alias="DATA_DIR")
    raw_dir: str = Field(default="./data/raw", alias="RAW_DIR")
    parsed_dir: str = Field(default="./data/parsed", alias="PARSED_DIR")
    chunks_dir: str = Field(default="./data/chunks", alias="CHUNKS_DIR")
    vector_store_dir: str = Field(default="./data/vector_store", alias="VECTOR_STORE_DIR")


class PDFSettings(BaseSettings):
    """PDF 解析配置"""
    classify_enabled: bool = True
    force_ocr_for_scanned: bool = True
    cache_enabled: bool = True


class RetrievalSettings(BaseSettings):
    """检索配置"""
    top_k: int = 10
    bm25_weight: float = 0.3
    dense_weight: float = 0.7
    min_score: float = 0.3


class RerankSettings(BaseSettings):
    """Rerank 配置"""
    top_k: int = 5
    model: str = "BAAI/bge-reranker-base"
    batch_size: int = 32


class SelfCheckSettings(BaseSettings):
    """SelfCheck 配置"""
    threshold: float = 0.7
    max_retries: int = 2


class GenerationSettings(BaseSettings):
    """Generation 配置"""
    temperature: float = 0.1
    max_tokens: int = 2048
    timeout: int = 60


class OCRSettings(BaseSettings):
    """OCR 配置"""
    use_angle_cls: bool = True
    lang: str = "ch"
    use_gpu: bool = False
    engine: str = "rapidocr"


class TableSettings(BaseSettings):
    """表格提取配置"""
    min_cols: int = 2
    min_rows: int = 2
    col_threshold: float = 100.0
    row_threshold: float = 50.0
    x_cluster_tolerance: float = 20.0


class ChunkingSettings(BaseSettings):
    """分块配置"""
    max_chunk_size: int = 512
    overlap_size: int = 50
    table_as_chunk: bool = True


class EmbeddingSettings(BaseSettings):
    """Embedding 配置"""
    model: str = Field(default="BAAI/bge-m3", alias="EMBEDDING_MODEL")
    device: str = Field(default="cpu", alias="EMBEDDING_DEVICE")
    batch_size: int = 32


class LoggingSettings(BaseSettings):
    """日志配置"""
    level: str = Field(default="INFO", alias="LOG_LEVEL")
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class AppSettings(BaseSettings):
    """应用完整配置"""
    paths: PathSettings = Field(default_factory=PathSettings)
    pdf: PDFSettings = Field(default_factory=PDFSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    rerank: RerankSettings = Field(default_factory=RerankSettings)
    selfcheck: SelfCheckSettings = Field(default_factory=SelfCheckSettings)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    ocr: OCRSettings = Field(default_factory=OCRSettings)
    table: TableSettings = Field(default_factory=TableSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)


def load_yaml_config(config_path: Path | None = None) -> dict[str, Any]:
    """从 YAML 文件加载配置"""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "configs" / "config.yaml"

    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}")
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache
def get_settings() -> AppSettings:
    """获取应用配置（单例）"""
    load_env()
    return AppSettings()


@lru_cache
def get_yaml_config() -> dict[str, Any]:
    """获取 YAML 配置（单例）"""
    load_env()
    return load_yaml_config()


def get_full_config() -> dict[str, Any]:
    """获取完整配置（YAML + 环境变量）"""
    yaml_config = get_yaml_config()
    settings = get_settings()

    # 合并配置（YAML 优先级低于环境变量）
    merged = {
        "paths": yaml_config.get("paths", {}),
        "pdf": yaml_config.get("pdf", {}),
        "retrieval": yaml_config.get("retrieval", {}),
        "rerank": yaml_config.get("rerank", {}),
        "selfcheck": yaml_config.get("selfcheck", {}),
        "generation": yaml_config.get("generation", {}),
        "ocr": yaml_config.get("ocr", {}),
        "table": yaml_config.get("table", {}),
        "chunking": yaml_config.get("chunking", {}),
        "embedding": yaml_config.get("embedding", {}),
        "logging": yaml_config.get("logging", {}),
    }

    return merged