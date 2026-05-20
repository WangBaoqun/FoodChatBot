"""
菜谱查找Agent
专门负责菜谱检索和筛选
"""

import json
import logging
from typing import List, Dict, Any, Optional

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

RECIPE_FINDER_PROMPT = """
你是菜谱查找专家。你的任务是：

1. 根据用户需求检索合适的菜谱
2. 支持按分类、菜系、难度筛选
3. 返回清晰、有用的菜谱信息

可用工具：
- graph_rag_search: 图结构检索，适合涉及食材关系、菜系分类的查询
- hybrid_search: 混合检索，适合通用菜谱查询
- agentic_search: Agentic迭代检索，适合复杂或多约束条件的查询
- filter_by_category: 分类筛选，按菜系/分类/难度精确筛选

工作流程：
1. 分析查询内容，判断是否需要筛选条件
2. 选择合适的检索工具执行查询
3. 整理检索结果，提取关键信息
4. 以清晰的格式返回菜谱推荐

注意事项：
- 如果用户指定了菜系或分类，优先使用filter_by_category
- 如果查询涉及食材搭配或关系，使用graph_rag_search
- 如果查询复杂或有多个约束，使用agentic_search
- 简单查询使用hybrid_search

返回格式示例：
找到3道相关菜谱：
1. 宫保鸡丁（川菜，中等难度） - 经典川菜，鸡肉配花生...
2. 麻婆豆腐（川菜，简单） - 家常川菜，豆腐配豆瓣酱...
"""


class RecipeFinderAgent:
    """
    菜谱查找Agent

    职责：
    - 执行菜谱检索
    - 支持多维度筛选
    - 返回格式化的菜谱信息
    """

    def __init__(self, llm, tools: List[tool]):
        """
        初始化菜谱查找Agent

        Args:
            llm: LangChain LLM实例
            tools: 可用的工具列表
        """
        self.llm = llm
        self.tools = tools
        self.system_prompt = SystemMessage(content=RECIPE_FINDER_PROMPT)

        # 绑定工具到LLM
        self.llm_with_tools = llm.bind_tools(tools)

        logger.info(f"RecipeFinderAgent初始化完成，可用工具: {[t.name for t in tools]}")

    def invoke(self, query: str) -> str:
        """
        执行菜谱查找

        Args:
            query: 用户查询

        Returns:
            菜谱查找结果
        """
        logger.info(f"RecipeFinderAgent处理查询: {query}")

        messages = [
            self.system_prompt,
            HumanMessage(content=f"用户查询: {query}\n请检索并返回相关菜谱信息。")
        ]

        try:
            # 调用LLM执行
            response = self.llm_with_tools.invoke(messages)

            # 处理工具调用
            if response.tool_calls:
                tool_results = self._execute_tools(response.tool_calls)
                # 将工具结果加入消息
                messages.append(response)
                for result in tool_results:
                    messages.append(result)

                # 再次调用LLM生成最终回答
                final_response = self.llm.invoke(messages)
                return final_response.content
            else:
                return response.content

        except Exception as e:
            logger.error(f"RecipeFinderAgent执行失败: {e}")
            return f"菜谱查找出错: {str(e)}"

    def _execute_tools(self, tool_calls: List[Dict]) -> List[AIMessage]:
        """执行工具调用"""
        results = []

        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})

            # 找到对应的工具
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
            "name": "RecipeFinderAgent",
            "role": "菜谱查找",
            "tools": [t.name for t in self.tools],
            "description": "专门负责菜谱检索和筛选"
        }