"""SelfCheck Agent - 幻觉检测 Agent"""

import json
import logging
import re
from typing import Any

from app.services.llm_service import LLMService
from app.prompts.qa_prompt import SELFCHECK_PROMPT

logger = logging.getLogger(__name__)


class SelfCheckAgent:
    """自检 Agent

    检查生成的答案是否由参考资料支撑，检测幻觉。
    """

    def __init__(self, llm_service: LLMService | None = None):
        """初始化

        Args:
            llm_service: LLM 服务实例
        """
        self.llm = llm_service or LLMService()

    def check(self, answer: str, evidence: str) -> dict[str, Any]:
        """检查答案是否由证据支撑

        Args:
            answer: 生成的答案
            evidence: 参考资料

        Returns:
            {
                "is_grounded": bool,  # 是否由证据支撑
                "confidence": float,    # 置信度 0-1
                "reasoning": str        # 判断理由
            }
        """
        if not answer or not answer.strip():
            return {
                "is_grounded": False,
                "confidence": 1.0,
                "reasoning": "答案为空"
            }

        if not evidence or not evidence.strip():
            return {
                "is_grounded": False,
                "confidence": 1.0,
                "reasoning": "参考资料为空"
            }

        # 拼接 Prompt
        prompt = SELFCHECK_PROMPT.format(context=evidence, answer=answer)

        try:
            response = self.llm.generate(
                prompt=prompt,
                system_prompt="你是一个严谨的答案质量评估助手。",
                temperature=0.1,
                max_tokens=512,
            )

            # 解析 JSON 响应
            result = self._parse_response(response)
            logger.info(f"SelfCheck result: is_grounded={result.get('is_grounded')}, confidence={result.get('confidence')}")
            return result

        except Exception as e:
            logger.error(f"SelfCheck failed: {e}")
            return {
                "is_grounded": False,
                "confidence": 0.0,
                "reasoning": f"自检过程出错: {str(e)}"
            }

    def _parse_response(self, response: str) -> dict[str, Any]:
        """解析 LLM 返回的 JSON

        Args:
            response: LLM 原始响应

        Returns:
            解析后的 dict
        """
        # 尝试提取 JSON（可能包含在 markdown 代码块中）
        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            json_str = json_match.group()
            try:
                data = json.loads(json_str)
                # 验证字段
                if "is_grounded" in data and "reasoning" in data:
                    return {
                        "is_grounded": bool(data["is_grounded"]),
                        "confidence": float(data.get("confidence", 0.5)),
                        "reasoning": str(data["reasoning"])
                    }
            except json.JSONDecodeError:
                pass

        # JSON 解析失败，尝试文本匹配
        response_lower = response.lower()
        if "is_grounded" in response_lower and "true" in response_lower:
            is_grounded = True
        elif "is_grounded" in response_lower and "false" in response_lower:
            is_grounded = False
        elif "grounded" in response_lower and "hallucination" not in response_lower:
            is_grounded = True
        else:
            is_grounded = False

        # 提取 confidence
        conf_match = re.search(r'"confidence"\s*:\s*([0-9.]+)', response)
        confidence = float(conf_match.group(1)) if conf_match else 0.5

        # 提取 reasoning
        reason_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', response)
        reasoning = reason_match.group(1) if reason_match else response[:200]

        return {
            "is_grounded": is_grounded,
            "confidence": confidence,
            "reasoning": reasoning
        }