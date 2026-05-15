"""QA Agent - 问答 Agent"""

import logging
from typing import Any

from app.services.llm_service import LLMService
from app.retrievers.hybrid_retriever import HybridRetriever
from app.agents.self_check_agent import SelfCheckAgent
from app.prompts.qa_prompt import QA_SYSTEM_PROMPT, QA_PROMPT, REFUSE_PROMPT

logger = logging.getLogger(__name__)


class QAAgent:
    """问答 Agent

    完整的问答流程：
    1. 检索相关文档块
    2. 判空（无结果或低分拒答）
    3. 基于上下文生成答案
    4. 自检查找幻觉
    5. 返回答案或拒答
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        llm_service: LLMService | None = None,
        self_check_agent: SelfCheckAgent | None = None,
        score_threshold: float = 0.0001,
    ):
        """初始化

        Args:
            retriever: 混合检索器
            llm_service: LLM 服务实例
            self_check_agent: 自检 Agent 实例
            score_threshold: 检索分数阈值，低于此值拒答
        """
        self.retriever = retriever
        self.llm = llm_service or LLMService()
        self.self_check = self_check_agent or SelfCheckAgent(self.llm)
        self.score_threshold = score_threshold

    def answer(self, query: str, top_k: int = 5) -> dict[str, Any]:
        """回答用户问题

        Args:
            query: 用户问题
            top_k: 检索返回的数量

        Returns:
            {
                "answer": str,           # 答案文本
                "sources": list[str],     # 来源页码列表
                "self_check": dict,       # 自检结果
                "rejected": bool,         # 是否拒答
                "retrieval_results": list # 检索结果（用于调试）
            }
        """
        # Step 1: 检索
        logger.info(f"Retrieving for query: {query}")
        retrieval_results = self.retriever.retrieve(query, top_k=top_k)

        # Step 2: 判空
        if not retrieval_results:
            logger.info("No retrieval results")
            return self._build_reject_response(
                reason="未找到相关文档",
                retrieval_results=[]
            )

        top_score = retrieval_results[0].get("score", 0)
        if top_score < self.score_threshold:
            logger.info(f"Top score {top_score} below threshold {self.score_threshold}")
            return self._build_reject_response(
                reason=f"检索分数过低 ({top_score:.4f})",
                retrieval_results=retrieval_results
            )

        # Step 3: 构建上下文
        context, sources = self._build_context(retrieval_results)
        logger.info(f"Built context from {len(sources)} sources: {sources}")

        # Step 4: 生成答案
        try:
            answer = self._generate_answer(query, context)
        except RuntimeError as e:
            # LLM 未配置
            logger.warning(f"LLM not configured: {e}")
            return {
                "answer": "[LLM 未配置，无法生成答案]",
                "sources": sources,
                "self_check": {"is_grounded": False, "confidence": 0.0, "reasoning": "LLM未配置"},
                "rejected": True,
                "reject_reason": "LLM未配置",
                "retrieval_results": retrieval_results
            }
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return self._build_reject_response(
                reason=f"生成答案失败: {str(e)}",
                retrieval_results=retrieval_results
            )

        # Step 5: 自检
        self_check_result = self.self_check.check(answer, context)

        # Step 6: 决策
        if self_check_result["is_grounded"]:
            # 自检通过
            logger.info("SelfCheck passed")
            return {
                "answer": answer,
                "sources": sources,
                "self_check": self_check_result,
                "rejected": False,
                "retrieval_results": retrieval_results
            }
        else:
            # 自检失败 - 返回带警告的答案或拒答
            confidence = self_check_result.get("confidence", 0.5)
            if confidence >= 0.7:
                # 高置信度幻觉，直接拒答
                logger.warning(f"High confidence hallucination detected: {confidence}")
                return self._build_reject_response(
                    reason="答案存在幻觉风险，已拒绝",
                    retrieval_results=retrieval_results,
                    self_check_result=self_check_result
                )
            else:
                # 低置信度，返回带警告的答案
                logger.warning(f"Low confidence hallucination risk: {confidence}")
                warning_answer = f"{answer}\n\n[警告：部分内容未能在参考资料中完全确认]"
                return {
                    "answer": warning_answer,
                    "sources": sources,
                    "self_check": self_check_result,
                    "rejected": False,
                    "has_warning": True,
                    "retrieval_results": retrieval_results
                }

    def _build_context(self, retrieval_results: list[dict]) -> tuple[str, list[str]]:
        """构建上下文字符串

        Args:
            retrieval_results: 检索结果列表

        Returns:
            (context_str, sources_list)
        """
        contexts = []
        sources = []

        for i, result in enumerate(retrieval_results):
            chunk = result.get("chunk", {})
            metadata = chunk.get("metadata", {})
            content = chunk.get("content", "")
            page = metadata.get("page", "?")

            if content:
                contexts.append(f"[来源 {i+1} - Page {page}]\n{content}")
                sources.append(f"Page {page}")

        return "\n\n".join(contexts), sources

    def _generate_answer(self, query: str, context: str) -> str:
        """生成答案

        Args:
            query: 用户问题
            context: 上下文

        Returns:
            生成的答案
        """
        prompt = QA_PROMPT.format(context=context, query=query)

        answer = self.llm.generate(
            prompt=prompt,
            system_prompt=QA_SYSTEM_PROMPT,
            temperature=0.1,
            max_tokens=2048,
        )

        return answer.strip()

    def _build_reject_response(
        self,
        reason: str,
        retrieval_results: list,
        self_check_result: dict | None = None
    ) -> dict[str, Any]:
        """构建拒答响应

        Args:
            reason: 拒答原因
            retrieval_results: 检索结果
            self_check_result: 自检结果（可选）

        Returns:
            拒答响应 dict
        """
        response = {
            "answer": REFUSE_PROMPT.strip(),
            "sources": [],
            "self_check": self_check_result or {
                "is_grounded": False,
                "confidence": 1.0,
                "reasoning": reason
            },
            "rejected": True,
            "reject_reason": reason,
            "retrieval_results": retrieval_results
        }
        return response