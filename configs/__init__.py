"""配置加载模块 (兼容层)

功能已迁移到 app.core.config_loader，
此文件保留用于向后兼容。
"""

from app.core.config_loader import (
    get_settings,
    get_yaml_config,
    get_full_config,
    load_yaml_config,
    AppSettings,
)

__all__ = [
    "get_settings",
    "get_yaml_config",
    "get_full_config",
    "load_yaml_config",
    "AppSettings",
]