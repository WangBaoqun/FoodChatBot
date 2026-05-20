"""
OrchestratorAgent（协调器）
负责分析用户意图、分派任务、整合结果
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# 意图分类枚举
class IntentType(Enum):
    """用户意图类型"""
    RECIPE_SEARCH = "recipe_search"        # 菜谱查找
    INGREDIENT_PAIRING = "ingredient_pairing"  # 食材搭配
    INGREDIENT_SUBSTITUTE = "ingredient_substitute"  # 食材替代
    HEALTH_ADVICE = "health_advice"        # 健康饮食建议
    COMPLEX_QUERY = "complex_query"        # 复合查询
    GENERAL = "general"                    # 一般查询


ORCHESTRATOR_PROMPT = """
你是Multi-Agent系统的协调器。你的任务是：

1. 分析用户查询的意图
2. 决定调用哪些专门化Agent
3. 协调多Agent的执行顺序（并行或串行）
4. 整合各Agent的结果，生成统一回答

可用的专门化Agent：
- RecipeFinderAgent: 菜谱查找，适合查询菜谱做法、菜系、分类等
- IngredientExpertAgent: 食材专家，适合食材搭配、替代、关联查询
- HealthAdvisorAgent: 健康顾问，适合健康饮食建议、禁忌检查

意图分析规则：
- 包含"怎么做"、"做法"、"菜谱" → recipe_search（菜谱查找）
- 包含"搭配"、"配什么"、"可以和什么" → ingredient_pairing（食材搭配）
- 包含"替代"、"换成"、"用什么代替" → ingredient_substitute（食材替代）
- 包含"减肥"、"糖尿病"、"高血压"、"老人"等健康关键词 → health_advice（健康建议）
- 多个意图同时存在 → complex_query（复合查询）

请分析以下查询并返回JSON格式的决策（intent字段请使用英文）：

{
    "intent": "recipe_search 或 ingredient_pairing 或 ingredient_substitute 或 health_advice 或 complex_query 或 general",
    "agents_to_call": ["Agent1", "Agent2"],
    "execution_mode": "parallel 或 sequential",
    "reasoning": "决策理由"
}
"""


class OrchestratorAgent:
    """
    OrchestratorAgent（协调器）

    职责：
    - 接收用户查询，分析意图
    - 决定调用哪些专门化Agent
    - 协调多Agent并行/串行执行
    - 整合各Agent结果，生成统一回答
    """

    def __init__(self, llm, specialized_agents: Dict[str, Any], config):
        """
        初始化协调器

        Args:
            llm: LangChain LLM实例
            specialized_agents: 专门化Agent字典
            config: 配置对象
        """
        self.llm = llm
        self.specialized_agents = specialized_agents
        self.config = config
        self.system_prompt = SystemMessage(content=ORCHESTRATOR_PROMPT)

        logger.info(f"OrchestratorAgent初始化完成，管理Agent: {list(specialized_agents.keys())}")

    def analyze_intent(self, query: str) -> Tuple[IntentType, List[str], str]:
        """
        分析用户查询意图

        Args:
            query: 用户查询

        Returns:
            intent: 意图类型
            agents_to_call: 需要调用的Agent列表
            reasoning: 决策理由
        """
        logger.info(f"分析意图: {query}")

        try:
            messages = [
                self.system_prompt,
                HumanMessage(content=f"分析查询: {query}")
            ]

            response = self.llm.invoke(messages)
            result_text = response.content.strip()

            # 解析JSON（可能需要提取）
            try:
                # 尝试直接解析
                result = json.loads(result_text)
            except json.JSONDecodeError:
                # 尝试提取JSON块
                if "{" in result_text and "}" in result_text:
                    start = result_text.find("{")
                    end = result_text.rfind("}") + 1
                    result = json.loads(result_text[start:end])
                else:
                    # 降级：基于规则的意图分析
                    return self._rule_based_intent_analysis(query)

            intent_str = result.get("intent", "general")
            agents = result.get("agents_to_call", [])
            reasoning = result.get("reasoning", "")

            # 映射意图类型（支持中英文）
            intent_map = {
                # 英文映射
                "recipe_search": IntentType.RECIPE_SEARCH,
                "ingredient_pairing": IntentType.INGREDIENT_PAIRING,
                "ingredient_substitute": IntentType.INGREDIENT_SUBSTITUTE,
                "health_advice": IntentType.HEALTH_ADVICE,
                "complex_query": IntentType.COMPLEX_QUERY,
                "general": IntentType.GENERAL
            }

            intent = intent_map.get(intent_str, IntentType.GENERAL)

            logger.info(f"意图分析结果: {intent.value}, Agent: {agents}")
            return intent, agents, reasoning

        except Exception as e:
            logger.error(f"意图分析失败: {e}")
            return self._rule_based_intent_analysis(query)

    def _rule_based_intent_analysis(self, query: str) -> Tuple[IntentType, List[str], str]:
        """基于规则的降级意图分析"""
        agents = []
        intent = IntentType.GENERAL

        # 关键词规则
        recipe_keywords = ["怎么做", "做法", "菜谱", "食谱", "制作"]
        ingredient_keywords = ["搭配", "配什么", "可以和什么", "替代", "换成"]
        health_keywords = ["减肥", "糖尿病", "高血压", "老人", "孕妇", "儿童", "健康", "低脂", "低糖"]

        if any(kw in query for kw in recipe_keywords):
            agents.append("RecipeFinderAgent")
            intent = IntentType.RECIPE_SEARCH

        if any(kw in query for kw in ingredient_keywords):
            agents.append("IngredientExpertAgent")
            intent = IntentType.INGREDIENT_PAIRING

        if any(kw in query for kw in health_keywords):
            agents.append("HealthAdvisorAgent")
            intent = IntentType.HEALTH_ADVICE

        # 如果匹配多个，则为复合查询
        if len(agents) > 1:
            intent = IntentType.COMPLEX_QUERY

        # 如果没有匹配任何，使用通用处理（RecipeFinder）
        if not agents:
            agents.append("RecipeFinderAgent")

        reasoning = f"基于关键词规则的意图分析"
        return intent, agents, reasoning

    def dispatch_to_agent(self, agent_name: str, query: str) -> str:
        """
        分派任务给专门化Agent

        Args:
            agent_name: Agent名称
            query: 查询内容

        Returns:
            Agent执行结果
        """
        logger.info(f"分派任务给 {agent_name}")

        agent = self.specialized_agents.get(agent_name)
        if agent is None:
            logger.error(f"Agent {agent_name} 不存在")
            return f"Agent {agent_name} 未初始化"

        try:
            result = agent.invoke(query)
            return result
        except Exception as e:
            logger.error(f"Agent {agent_name} 执行失败: {e}")
            return f"执行出错: {str(e)}"

    def synthesize_results(
        self,
        query: str,
        agent_responses: Dict[str, str],
        intent: IntentType
    ) -> str:
        """
        整合各Agent结果，生成最终回答

        Args:
            query: 原始查询
            agent_responses: 各Agent的响应
            intent: 意图类型

        Returns:
            整合后的最终回答
        """
        logger.info(f"整合结果，Agent数: {len(agent_responses)}")

        if len(agent_responses) == 1:
            # 单Agent，直接返回结果
            return list(agent_responses.values())[0]

        # 多Agent，需要整合
        synthesis_prompt = f"""
        用户查询：{query}

        各专家Agent的回答：
        """

        for agent_name, response in agent_responses.items():
            synthesis_prompt += f"\n\n【{agent_name}】\n{response[:500]}..."

        synthesis_prompt += """
        请整合以上各专家的建议，生成一个完整、连贯的回答。
        注意：
        1. 保留各专家的核心观点
        2. 避免重复内容
        3. 添加必要的衔接和总结
        """

        try:
            messages = [
                SystemMessage(content="你是回答整合专家，擅长整合多个来源的信息。"),
                HumanMessage(content=synthesis_prompt)
            ]

            response = self.llm.invoke(messages)
            return response.content

        except Exception as e:
            logger.error(f"结果整合失败: {e}")
            # 降级：简单拼接
            combined = "\n\n".join([
                f"【{name}】\n{response}"
                for name, response in agent_responses.items()
            ])
            return combined

    def get_info(self) -> Dict[str, Any]:
        """获取协调器信息"""
        return {
            "name": "OrchestratorAgent",
            "role": "协调器",
            "managed_agents": list(self.specialized_agents.keys()),
            "description": "负责分析意图、分派任务、整合结果"
        }