"""LLM 服务模块

使用 OpenAI SDK 兼容 MiniMax API。
单例模式避免重复初始化。
"""

import logging
import os
from typing import Any

from openai import OpenAI

from app.core.config_loader import get_full_config

logger = logging.getLogger(__name__)


class LLMService:
    """LLM 服务单例

    使用 OpenAI SDK 连接 MiniMax API。
    配置从环境变量或 configs/config.yaml 读取。
    """

    _instance: "LLMService | None" = None
    _client: OpenAI | None = None

    def __new__(cls) -> "LLMService":
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """初始化（仅首次）"""
        if self._client is None:
            self._initialize()

    def _initialize(self) -> None:
        """初始化 OpenAI 客户端"""
        # 优先使用环境变量（MiniMax 配置）
        api_key = os.getenv("MINIMAX_API_KEY") or os.getenv("LLM_API_KEY", "")
        base_url = os.getenv("MINIMAX_BASE_URL") or os.getenv("LLM_BASE_URL", "")
        model = os.getenv("MINIMAX_MODEL_NAME") or os.getenv("LLM_MODEL", "")

        # 如果环境变量为空，尝试从配置文件读取
        if not api_key or not base_url:
            config = get_full_config()
            llm_config = config.get("llm", {})
            api_key = api_key or llm_config.get("api_key", "")
            base_url = base_url or llm_config.get("base_url", "https://api.minimax.chat/v1")
            model = model or llm_config.get("model", "abab6.5s-chat")

        if not api_key:
            logger.warning("LLM API key not configured. Set MINIMAX_API_KEY in .env")

        logger.info(f"Initializing LLMService with base_url: {base_url}")

        self._client = OpenAI(
            api_key=api_key or "dummy",
            base_url=base_url,
            timeout=60,
        )
        self._model = model
        self._configured = bool(api_key)
        logger.info(f"LLM client initialized with model: {self._model}, configured: {self._configured}")

    @property
    def client(self) -> OpenAI:
        """获取 OpenAI 客户端"""
        if self._client is None:
            self._initialize()
        return self._client

    @property
    def model(self) -> str:
        """获取模型名称"""
        return self._model

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        """生成文本

        Args:
            prompt: 用户 prompt
            system_prompt: 系统 prompt（可选）
            temperature: 温度参数
            max_tokens: 最大 token 数

        Returns:
            生成的文本
        """
        if not getattr(self, "_configured", True):
            raise RuntimeError("LLM API not configured. Please set MINIMAX_API_KEY in .env")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise

    def generate_with_context(
        self,
        query: str,
        context: str,
        system_prompt: str | None = None,
    ) -> str:
        """基于上下文生成答案

        Args:
            query: 用户问题
            context: 检索到的上下文
            system_prompt: 系统 prompt

        Returns:
            生成的答案
        """
        prompt = f"""基于以下参考资料回答问题。

参考资料:
{context}

问题: {query}

请根据参考资料回答，如果资料中没有相关信息，请说明"资料中未提及"。

回答:"""

        return self.generate(prompt, system_prompt=system_prompt)


def get_llm_service() -> LLMService:
    """获取 LLMService 单例

    Returns:
        LLMService 实例
    """
    return LLMService()