"""
食材专家Agent
专门负责食材搭配、替代等查询
"""

import json
import logging
from typing import List, Dict, Any, Optional

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

INGREDIENT_EXPERT_PROMPT = """
你是食材搭配专家。你的任务是：

1. 分析食材间的搭配关系
2. 提供食材替代建议
3. 查询食材相关信息和关联网络

可用工具：
- ingredient_pairing: 食材搭配查询，查找与指定食材经常搭配的其他食材
- ingredient_substitute: 食材替代推荐，当需要替代某种食材时提供建议
- graph_neighbor_search: 图邻居查询，探索食材或菜谱的关联网络

工作流程：
1. 分析查询类型（搭配、替代、关联查询）
2. 选择合适的工具执行查询
3. 解释搭配或替代的原因
4. 提供实用的建议

搭配建议原则：
- 考虑口味互补性
- 考虑营养均衡
- 考虑烹饪方式匹配
- 参考传统菜谱中的经典搭配

返回格式示例：
鸡肉搭配建议：
- 西兰花（高搭配频率） - 口味清爽，营养互补
- 胡萝卜（高搭配频率） - 增加甜味，丰富口感
- 土豆（中等搭配频率） - 经典组合，饱腹感强

建议烹饪方式：炒、炖、烤
"""


class IngredientExpertAgent:
    """
    食材专家Agent

    职责：
    - 食材搭配推荐
    - 食材替代建议
    - 食材关联网络探索
    """

    def __init__(self, llm, tools: List[tool]):
        """
        初始化食材专家Agent

        Args:
            llm: LangChain LLM实例
            tools: 可用的工具列表
        """
        self.llm = llm
        self.tools = tools
        self.system_prompt = SystemMessage(content=INGREDIENT_EXPERT_PROMPT)

        # 绑定工具到LLM
        self.llm_with_tools = llm.bind_tools(tools)

        logger.info(f"IngredientExpertAgent初始化完成，可用工具: {[t.name for t in tools]}")

    def invoke(self, query: str) -> str:
        """
        执行食材相关查询

        Args:
            query: 用户查询

        Returns:
            食材查询结果
        """
        logger.info(f"IngredientExpertAgent处理查询: {query}")

        messages = [
            self.system_prompt,
            HumanMessage(content=f"用户查询: {query}\n请分析食材关系并提供建议。")
        ]

        try:
            # 调用LLM执行
            response = self.llm_with_tools.invoke(messages)
            messages.append(response)

            # 处理工具调用
            if response.tool_calls:
                tool_results = self._execute_tools(response.tool_calls)
                for result in tool_results:
                    messages.append(result)

                final_response = self.llm.invoke(messages)
                return final_response.content
            else:
                return response.content

        except Exception as e:
            logger.error(f"IngredientExpertAgent执行失败: {e}")
            return f"食材查询出错: {str(e)}"

    def _execute_tools(self, tool_calls: List[Dict]) -> List[AIMessage]:
        """执行工具调用"""
        results = []

        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})

            tool_func = None
            for t in self.tools:
                if t.name == tool_name:
                    tool_func = t
                    break

            if tool_func:
                try:
                    result = tool_func.invoke(tool_args)
                    results.append(AIMessage(content=f"工具{tool_name}结果: {result}"))
                except Exception as e:
                    logger.error(f"工具{tool_name}执行失败: {e}")
                    results.append(AIMessage(content=f"工具{tool_name}执行出错: {str(e)}"))

        return results

    def get_info(self) -> Dict[str, Any]:
        """获取Agent信息"""
        return {
            "name": "IngredientExpertAgent",
            "role": "食材专家",
            "tools": [t.name for t in self.tools],
            "description": "专门负责食材搭配、替代等查询"
        }