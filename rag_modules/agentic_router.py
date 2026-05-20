"""
Agentic RAG 路由器
将被动检索升级为主动探索式检索
支持反思评估、查询改写、迭代检索循环
"""

import json
import logging
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class RetrievalStrategy(Enum):
    """检索策略枚举"""
    HYBRID_TRADITIONAL = "hybrid_traditional"
    GRAPH_RAG = "graph_rag"
    COMBINED = "combined"


@dataclass
class QualityAssessment:
    """检索结果质量评估"""
    score: float  # 0-1 质量分数
    is_satisfactory: bool  # 是否满意
    feedback: str  # 改进建议
    missing_aspects: List[str] = field(default_factory=list)  # 缺失的信息维度
    relevant_aspects: List[str] = field(default_factory=list)  # 已覆盖的信息维度


@dataclass
class AgentTrace:
    """Agent执行轨迹记录"""
    iterations: int  # 实际迭代次数
    final_quality: float  # 最终质量分数
    query_evolution: List[str] = field(default_factory=list)  # 查询演变历史
    strategies_used: List[str] = field(default_factory=list)  # 各轮使用的策略
    quality_scores: List[float] = field(default_factory=list)  # 各轮质量分数
    reasoning_log: List[str] = field(default_factory=list)  # 决策推理日志


class AgenticQueryRouter:
    """
    Agentic RAG 路由器 - 主动探索式检索

    核心能力：
    1. 反思评估：使用LLM评估检索结果质量
    2. 查询改写：基于反馈自动改写查询
    3. 迭代检索：多轮循环直到满意或达到上限
    4. 累积上下文：保留各轮有价值的结果

    使用方式：
    - 可选启用，通过参数控制
    - 适用于复杂查询场景
    - 内部调用现有检索模块
    """

    def __init__(
        self,
        traditional_retrieval,  # HybridRetrievalModule
        graph_rag_retrieval,    # GraphRAGRetrieval
        intelligent_router,     # IntelligentQueryRouter
        llm_client,             # OpenAI client
        config
    ):
        self.traditional_retrieval = traditional_retrieval
        self.graph_rag_retrieval = graph_rag_retrieval
        self.intelligent_router = intelligent_router
        self.llm_client = llm_client
        self.config = config

        # Agentic 配置
        self.max_iterations = getattr(config, 'agentic_max_iterations', 3)
        self.quality_threshold = getattr(config, 'agentic_quality_threshold', 0.7)

        # 统计信息
        self.stats = {
            "total_agentic_queries": 0,
            "avg_iterations": 0.0,
            "avg_final_quality": 0.0,
            "successful_rate": 0.0
        }
        self._iteration_history = []

    def agentic_search(
        self,
        query: str,
        top_k: int = 5,
        strategy: RetrievalStrategy = RetrievalStrategy.COMBINED
    ) -> Tuple[List[Document], AgentTrace]:
        """
        主入口：带反思循环的主动探索式检索

        Args:
            query: 用户查询
            top_k: 返回结果数量
            strategy: 检索策略

        Returns:
            documents: 累积的检索结果
            trace: Agent执行轨迹
        """
        logger.info(f"启动Agentic检索: {query}")

        # 初始化轨迹
        trace = AgentTrace(
            iterations=0,
            final_quality=0.0,
            query_evolution=[query],
            strategies_used=[],
            quality_scores=[],
            reasoning_log=[]
        )

        accumulated_docs: List[Document] = []
        current_query = query

        # 迭代检索循环
        for iteration in range(self.max_iterations):
            trace.iterations = iteration + 1
            logger.info(f"第 {iteration + 1} 轮迭代，当前查询: {current_query}")
            trace.reasoning_log.append(f"迭代 {iteration + 1}: 执行检索 '{current_query}'")

            # 1. 执行检索
            docs = self._execute_retrieval(current_query, top_k, strategy)
            trace.strategies_used.append(strategy.value)

            # 2. 累积并去重结果
            accumulated_docs = self._merge_and_deduplicate(accumulated_docs, docs)
            logger.info(f"累积文档数: {len(accumulated_docs)}")

            # 3. 评估质量
            assessment = self.evaluate_retrieval_quality(current_query, accumulated_docs)
            trace.quality_scores.append(assessment.score)
            trace.reasoning_log.append(
                f"质量评估: 分数={assessment.score:.2f}, 满意={assessment.is_satisfactory}"
            )

            logger.info(f"质量评估: {assessment.score:.2f} (阈值: {self.quality_threshold})")

            # 4. 判断是否满意
            if assessment.is_satisfactory or assessment.score >= self.quality_threshold:
                trace.final_quality = assessment.score
                trace.reasoning_log.append(f"检索满意，结束迭代 (分数={assessment.score:.2f})")
                logger.info(f"检索满意，结束迭代")
                break

            # 5. 最后一轮不再改写
            if iteration == self.max_iterations - 1:
                trace.final_quality = assessment.score
                trace.reasoning_log.append(f"达到最大迭代次数，结束 (分数={assessment.score:.2f})")
                logger.info(f"达到最大迭代次数 {self.max_iterations}")
                break

            # 6. 改写查询
            rewritten_query = self.rewrite_query(
                current_query,
                accumulated_docs,
                assessment.feedback,
                assessment.missing_aspects
            )

            if rewritten_query and rewritten_query != current_query:
                current_query = rewritten_query
                trace.query_evolution.append(current_query)
                trace.reasoning_log.append(f"查询改写: '{current_query}'")
                logger.info(f"查询改写: {current_query}")
            else:
                # 改写失败，尝试切换策略
                strategy = self._switch_strategy(strategy)
                trace.reasoning_log.append(f"改写无效，切换策略: {strategy.value}")
                logger.info(f"改写无效，切换策略: {strategy.value}")

        # 更新统计
        self._update_stats(trace)

        # 最终排序
        final_docs = self._rank_final_results(accumulated_docs, query)[:top_k]

        logger.info(f"Agentic检索完成: {trace.iterations}轮, 最终质量={trace.final_quality:.2f}")
        return final_docs, trace

    def _execute_retrieval(
        self,
        query: str,
        top_k: int,
        strategy: RetrievalStrategy
    ) -> List[Document]:
        """
        执行检索：调用现有检索模块
        """
        try:
            if strategy == RetrievalStrategy.HYBRID_TRADITIONAL:
                return self.traditional_retrieval.hybrid_search(query, top_k)
            elif strategy == RetrievalStrategy.GRAPH_RAG:
                return self.graph_rag_retrieval.graph_rag_search(query, top_k)
            else:  # COMBINED
                return self._combined_retrieval(query, top_k)
        except Exception as e:
            logger.error(f"检索执行失败: {e}")
            return []

    def _combined_retrieval(self, query: str, top_k: int) -> List[Document]:
        """
        组合检索：结合传统和图RAG
        """
        half_k = max(1, top_k // 2)

        # 并行执行两种检索
        traditional_docs = self.traditional_retrieval.hybrid_search(query, half_k)
        graph_docs = self.graph_rag_retrieval.graph_rag_search(query, top_k - half_k)

        # 合并去重
        combined = []
        seen_ids = set()

        for doc in graph_docs + traditional_docs:
            doc_id = doc.metadata.get("node_id", hash(doc.page_content))
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                combined.append(doc)

        return combined[:top_k]

    def evaluate_retrieval_quality(
        self,
        query: str,
        documents: List[Document]
    ) -> QualityAssessment:
        """
        反思评估：使用LLM评估检索结果质量

        评估维度：
        1. 相关性：结果是否直接回答问题
        2. 完整性：是否覆盖问题的主要方面
        3. 准确性：信息是否准确可信
        """
        logger.info(f"评估检索质量: {query}")

        if not documents:
            return QualityAssessment(
                score=0.0,
                is_satisfactory=False,
                feedback="未检索到任何结果",
                missing_aspects=["所有相关信息"]
            )

        # 构建结果摘要
        result_summary = self._summarize_documents(documents)

        prompt = f"""
        作为检索质量评估专家，请评估以下检索结果是否能充分回答用户问题。

        用户问题：{query}

        检索结果摘要：
        {result_summary}

        请从以下维度评估：
        1. 相关性：结果是否直接回答了用户的问题？（0-0.5分）
        2. 完整性：是否覆盖了问题涉及的主要方面？（0-0.3分）
        3. 可信度：信息来源是否可靠，内容是否具体？（0-0.2分）

        请同时分析：
        - 结果中已经涵盖的信息维度（relevant_aspects）
        - 结果中缺失或不足的信息维度（missing_aspects）

        评分标准：
        - 0.0-0.4: 结果明显不足，需要大幅改进
        - 0.5-0.6: 结果部分相关，但有明显缺失
        - 0.7-0.8: 结果较为满意，可接受
        - 0.9-1.0: 结果非常满意，充分回答问题

        请严格返回JSON格式，不要包含其他文字：
        {{
            "score": 0.75,
            "is_satisfactory": true,
            "feedback": "结果质量分析...",
            "missing_aspects": ["缺失维度1", "缺失维度2"],
            "relevant_aspects": ["已覆盖维度1", "已覆盖维度2"]
        }}
        """

        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500,
                timeout=30
            )

            result = json.loads(response.choices[0].message.content.strip())

            assessment = QualityAssessment(
                score=result.get("score", 0.5),
                is_satisfactory=result.get("is_satisfactory", False),
                feedback=result.get("feedback", ""),
                missing_aspects=result.get("missing_aspects", []),
                relevant_aspects=result.get("relevant_aspects", [])
            )

            logger.info(f"评估完成: score={assessment.score:.2f}, satisfied={assessment.is_satisfactory}")
            return assessment

        except Exception as e:
            logger.error(f"质量评估失败: {e}")
            # 降级方案：基于结果数量的简单评估
            score = min(len(documents) / 5.0, 1.0) * 0.6  # 简单估算
            return QualityAssessment(
                score=score,
                is_satisfactory=len(documents) >= 3,
                feedback="降级评估：基于结果数量",
                missing_aspects=[]
            )

    def rewrite_query(
        self,
        original_query: str,
        documents: List[Document],
        feedback: str,
        missing_aspects: List[str]
    ) -> str:
        """
        查询改写：基于评估反馈改写查询以获得更好结果

        改写策略：
        1. 添加缺失的关键词
        2. 调整查询角度
        3. 分解复杂问题
        """
        logger.info(f"改写查询: {original_query}")

        result_summary = self._summarize_documents(documents[:3])  # 只用前3个

        prompt = f"""
        作为查询优化专家，请改写用户查询以获得更好的检索结果。

        原始查询：{original_query}

        上次检索结果摘要：
        {result_summary}

        评估反馈：{feedback}
        缺失的信息维度：{', '.join(missing_aspects) if missing_aspects else '无'}

        改写策略（选择最合适的）：
        1. 添加缺失关键词：将缺失维度转化为具体关键词加入查询
        2. 调整查询角度：从不同视角重新表述问题
        3. 问题分解：将复杂问题拆解为更具体的子问题
        4. 扩展实体：添加相关实体或分类信息

        改写要求：
        - 保持查询的原意和目标
        - 改写后的查询应更具体、更易检索
        - 避免过度改写导致偏离原意

        请严格返回JSON格式：
        {{
            "rewritten_query": "改写后的查询",
            "strategy_used": "使用的策略名称",
            "added_keywords": ["新增关键词1", "关键词2"],
            "reasoning": "改写理由"
        }}
        """

        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,  # 稍高温度增加多样性
                max_tokens=300,
                timeout=30
            )

            result = json.loads(response.choices[0].message.content.strip())
            rewritten = result.get("rewritten_query", original_query)

            if rewritten and rewritten.strip() and rewritten != original_query:
                logger.info(f"查询改写成功: {rewritten}")
                logger.info(f"策略: {result.get('strategy_used')}, 理由: {result.get('reasoning')}")
                return rewritten.strip()
            else:
                logger.warning("改写结果与原查询相同或为空")
                return original_query

        except Exception as e:
            logger.error(f"查询改写失败: {e}")
            # 降级方案：简单添加缺失关键词
            if missing_aspects:
                keywords = " ".join(missing_aspects[:2])
                return f"{original_query} {keywords}"
            return original_query

    def _summarize_documents(self, documents: List[Document], max_length: int = 500) -> str:
        """
        生成文档摘要，用于评估和改写prompt
        """
        if not documents:
            return "无检索结果"

        summaries = []
        for i, doc in enumerate(documents[:5]):  # 最多5个
            content = doc.page_content[:200]  # 每个截取200字符
            recipe_name = doc.metadata.get("recipe_name", "未知")
            search_type = doc.metadata.get("search_type", doc.metadata.get("search_method", "unknown"))
            score = doc.metadata.get("final_score", doc.metadata.get("relevance_score", 0))

            summaries.append(
                f"[{i+1}] {recipe_name} (类型:{search_type}, 相关度:{score:.2f})\n"
                f"    内容摘要: {content}..."
            )

        summary = "\n".join(summaries)
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."

        return summary

    def _merge_and_deduplicate(
        self,
        accumulated: List[Document],
        new_docs: List[Document]
    ) -> List[Document]:
        """
        合并并去重文档结果
        保留各轮有价值的结果
        """
        merged = list(accumulated)
        seen_ids = {doc.metadata.get("node_id", hash(doc.page_content)) for doc in accumulated}
        seen_content_hashes = {hash(doc.page_content[:100]) for doc in accumulated}

        for doc in new_docs:
            doc_id = doc.metadata.get("node_id", hash(doc.page_content))
            content_hash = hash(doc.page_content[:100])

            # 按ID和内容双重去重
            if doc_id not in seen_ids and content_hash not in seen_content_hashes:
                seen_ids.add(doc_id)
                seen_content_hashes.add(content_hash)
                merged.append(doc)

        # 按相关度排序
        merged.sort(
            key=lambda x: x.metadata.get("final_score", x.metadata.get("relevance_score", 0)),
            reverse=True
        )

        return merged

    def _switch_strategy(self, current: RetrievalStrategy) -> RetrievalStrategy:
        """
        切换检索策略：当改写无效时尝试不同策略
        """
        if current == RetrievalStrategy.HYBRID_TRADITIONAL:
            return RetrievalStrategy.GRAPH_RAG
        elif current == RetrievalStrategy.GRAPH_RAG:
            return RetrievalStrategy.COMBINED
        else:
            return RetrievalStrategy.HYBRID_TRADITIONAL

    def _rank_final_results(
        self,
        documents: List[Document],
        query: str
    ) -> List[Document]:
        """
        最终结果排序：综合考虑相关度和来源多样性
        """
        # 已经在merge时排序，这里可以做额外调整
        # 确保不同来源的结果混合分布

        if len(documents) <= 5:
            return documents

        # Round-robin方式混合不同来源
        source_groups = {
            "graph": [],
            "traditional": [],
            "other": []
        }

        for doc in documents:
            search_type = doc.metadata.get("search_type", doc.metadata.get("search_method", ""))
            if "graph" in search_type.lower():
                source_groups["graph"].append(doc)
            elif "traditional" in search_type.lower() or "dual" in search_type.lower():
                source_groups["traditional"].append(doc)
            else:
                source_groups["other"].append(doc)

        # 交替添加
        final = []
        max_len = max(len(g) for g in source_groups.values())
        for i in range(max_len):
            for source in ["graph", "traditional", "other"]:
                if i < len(source_groups[source]):
                    final.append(source_groups[source][i])

        return final

    def _update_stats(self, trace: AgentTrace):
        """
        更新统计信息
        """
        self.stats["total_agentic_queries"] += 1
        self._iteration_history.append(trace.iterations)

        # 计算平均值
        total = self.stats["total_agentic_queries"]
        self.stats["avg_iterations"] = sum(self._iteration_history) / total
        self.stats["avg_final_quality"] = (
            self.stats["avg_final_quality"] * (total - 1) + trace.final_quality
        ) / total
        self.stats["successful_rate"] = (
            self.stats["successful_rate"] * (total - 1) +
            (1.0 if trace.final_quality >= self.quality_threshold else 0.0)
        ) / total

    def get_agentic_statistics(self) -> Dict[str, Any]:
        """
        获取Agentic检索统计
        """
        return {
            **self.stats,
            "config": {
                "max_iterations": self.max_iterations,
                "quality_threshold": self.quality_threshold
            }
        }

    def explain_agentic_process(self, trace: AgentTrace) -> str:
        """
        解释Agentic检索过程
        """
        explanation = f"""
        Agentic检索执行报告

        原始查询：{trace.query_evolution[0] if trace.query_evolution else 'N/A'}

        执行轨迹：
        - 迭代次数：{trace.iterations}
        - 最终质量分数：{trace.final_quality:.2f}
        - 质量阈值：{self.quality_threshold}

        查询演变：
        """

        for i, q in enumerate(trace.query_evolution):
            prefix = "→ " if i > 0 else "  "
            explanation += f"{prefix}第{i+1}轮: '{q}'\n"

        explanation += "\n质量变化：\n"
        for i, score in enumerate(trace.quality_scores):
            explanation += f"  第{i+1}轮: {score:.2f}\n"

        explanation += "\n决策日志：\n"
        for log in trace.reasoning_log[-5:]:  # 显示最后5条
            explanation += f"  - {log}\n"

        return explanation