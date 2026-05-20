"""
LangGraph工作流定义
实现Multi-Agent协作的工作流编排
"""

import logging
from typing import TypedDict, Dict, Any, List, Optional, Annotated
from enum import Enum

from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)


# 定义状态类型
class AgentState(TypedDict):
    """Agent工作流状态"""
    query: str                                      # 用户查询
    intent: str                                     # 意图类型
    agents_to_call: List[str]                       # 需要调用的Agent
    execution_mode: str                             # 执行模式
    reasoning: str                                  # 决策理由
    agent_responses: Dict[str, str]                 # 各Agent响应
    final_answer: str                               # 最终回答
    visualization_steps: List[Dict[str, Any]]       # 可视化步骤记录


def create_agent_workflow(
    orchestrator,
    specialized_agents: Dict[str, Any],
    visualizer,
    config
) -> StateGraph:
    """
    创建Agent工作流

    Args:
        orchestrator: OrchestratorAgent实例
        specialized_agents: 专门化Agent字典
        visualizer: ExecutionVisualizer实例
        config: 配置对象

    Returns:
        LangGraph StateGraph工作流
    """
    logger.info("创建LangGraph工作流")

    # 定义节点函数

    def orchestrator_node(state: AgentState) -> AgentState:
        """协调器节点：分析意图"""
        query = state["query"]
        visualizer.start_visualization(query)

        # 分析意图
        intent, agents, reasoning = orchestrator.analyze_intent(query)
        visualizer.record_intent_analysis(intent.value, agents)

        return {
            "intent": intent.value,
            "agents_to_call": agents,
            "reasoning": reasoning,
            "visualization_steps": visualizer.steps.copy()
        }

    def recipe_finder_node(state: AgentState) -> AgentState:
        """菜谱查找节点"""
        query = state["query"]
        agent_name = "RecipeFinderAgent"

        if agent_name not in specialized_agents:
            return {"agent_responses": {agent_name: "Agent未初始化"}}

        result = orchestrator.dispatch_to_agent(agent_name, query)

        # 简化结果用于可视化
        result_summary = result[:50] + "..." if len(result) > 50 else result
        visualizer.record_agent_execution(agent_name, ["graph_rag_search", "hybrid_search"], result_summary)

        return {
            "agent_responses": {agent_name: result},
            "visualization_steps": visualizer.steps.copy()
        }

    def ingredient_expert_node(state: AgentState) -> AgentState:
        """食材专家节点"""
        query = state["query"]
        agent_name = "IngredientExpertAgent"

        if agent_name not in specialized_agents:
            return {"agent_responses": {agent_name: "Agent未初始化"}}

        result = orchestrator.dispatch_to_agent(agent_name, query)

        result_summary = result[:50] + "..." if len(result) > 50 else result
        visualizer.record_agent_execution(agent_name, ["ingredient_pairing", "graph_neighbor_search"], result_summary)

        return {
            "agent_responses": {agent_name: result},
            "visualization_steps": visualizer.steps.copy()
        }

    def health_advisor_node(state: AgentState) -> AgentState:
        """健康顾问节点"""
        query = state["query"]
        agent_name = "HealthAdvisorAgent"

        if agent_name not in specialized_agents:
            return {"agent_responses": {agent_name: "Agent未初始化"}}

        result = orchestrator.dispatch_to_agent(agent_name, query)

        result_summary = result[:50] + "..." if len(result) > 50 else result
        visualizer.record_agent_execution(agent_name, ["health_recipe_search", "dietary_check"], result_summary)

        return {
            "agent_responses": {agent_name: result},
            "visualization_steps": visualizer.steps.copy()
        }

    def synthesizer_node(state: AgentState) -> AgentState:
        """整合节点：生成最终回答"""
        query = state["query"]
        agent_responses = state["agent_responses"]
        intent_str = state["intent"]

        # 整合结果
        from .orchestrator import IntentType
        intent = IntentType(intent_str)
        final_answer = orchestrator.synthesize_results(query, agent_responses, intent)

        # 记录可视化
        answer_summary = final_answer[:50] + "..." if len(final_answer) > 50 else final_answer
        visualizer.record_synthesis(answer_summary)

        return {
            "final_answer": final_answer,
            "visualization_steps": visualizer.steps.copy()
        }

    # 条件路由函数
    def route_by_intent(state: AgentState) -> List[str]:
        """根据意图决定调用哪些Agent"""
        return state["agents_to_call"]

    # 构建工作流图
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("recipe_finder", recipe_finder_node)
    workflow.add_node("ingredient_expert", ingredient_expert_node)
    workflow.add_node("health_advisor", health_advisor_node)
    workflow.add_node("synthesizer", synthesizer_node)

    # 设置入口
    workflow.set_entry_point("orchestrator")

    # 添加条件边（根据意图分派）
    workflow.add_conditional_edges(
        "orchestrator",
        route_by_intent,
        {
            "RecipeFinderAgent": "recipe_finder",
            "IngredientExpertAgent": "ingredient_expert",
            "HealthAdvisorAgent": "health_advisor"
        }
    )

    # 添加合并边（各Agent → 整合节点）
    workflow.add_edge("recipe_finder", "synthesizer")
    workflow.add_edge("ingredient_expert", "synthesizer")
    workflow.add_edge("health_advisor", "synthesizer")

    # 结束
    workflow.add_edge("synthesizer", END)

    logger.info("LangGraph工作流创建完成")
    return workflow.compile()


class SimpleMultiAgentWorkflow:
    """
    简化版Multi-Agent工作流
    不使用LangGraph的复杂条件分支，采用简单的顺序执行
    """

    def __init__(
        self,
        orchestrator,
        specialized_agents: Dict[str, Any],
        visualizer,
        config
    ):
        self.orchestrator = orchestrator
        self.specialized_agents = specialized_agents
        self.visualizer = visualizer
        self.config = config

        logger.info("SimpleMultiAgentWorkflow初始化完成")

    def run(self, query: str) -> Dict[str, Any]:
        """
        执行工作流

        Args:
            query: 用户查询

        Returns:
            执行结果
        """
        # 1. 开始可视化
        self.visualizer.start_visualization(query)

        # 2. 分析意图
        from .orchestrator import IntentType
        intent, agents, reasoning = self.orchestrator.analyze_intent(query)
        self.visualizer.record_intent_analysis(intent.value, agents)

        # logger.info(f"意图: {intent.value}, Agent: {agents}")

        # 3. 执行Agent调用（简化版不支持并行）
        agent_responses = {}
        for agent_name in agents:
            if agent_name in self.specialized_agents:
                result = self.orchestrator.dispatch_to_agent(agent_name, query)
                agent_responses[agent_name] = result

                # 记录可视化
                result_summary = result[:50] + "..." if len(result) > 50 else result
                self.visualizer.record_agent_execution(
                    agent_name,
                    self._get_agent_tools(agent_name),
                    result_summary
                )
            else:
                agent_responses[agent_name] = f"Agent {agent_name} 未初始化"

        # 4. 整合结果
        final_answer = self.orchestrator.synthesize_results(query, agent_responses, intent)

        # 记录可视化
        answer_summary = final_answer[:50] + "..." if len(final_answer) > 50 else final_answer
        self.visualizer.record_synthesis(answer_summary)

        return {
            "query": query,
            "intent": intent.value,
            "agents_called": agents,
            "agent_responses": agent_responses,
            "final_answer": final_answer,
            "visualization": self.visualizer.render(),
            "visualization_compact": self.visualizer.render_compact()
        }

    def _get_agent_tools(self, agent_name: str) -> List[str]:
        """获取Agent的工具列表"""
        tool_map = {
            "RecipeFinderAgent": ["graph_rag_search", "hybrid_search"],
            "IngredientExpertAgent": ["ingredient_pairing", "graph_neighbor_search"],
            "HealthAdvisorAgent": ["health_recipe_search", "dietary_check"]
        }
        return tool_map.get(agent_name, [])