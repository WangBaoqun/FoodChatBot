"""
健康顾问Agent
专门负责健康饮食建议和禁忌检查
"""

import json
import logging
from typing import List, Dict, Any, Optional

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

HEALTH_ADVISOR_PROMPT = """
你是健康饮食顾问。你的任务是：

1. 根据用户健康状况推荐适合的菜品
2. 检查菜品是否符合特定饮食要求
3. 提供健康饮食建议和禁忌提醒

可用工具：
- health_recipe_search: 健康菜谱检索，根据健康需求（减肥、糖尿病等）检索适合的菜品
- dietary_check: 饮食禁忌检查，检查某道菜是否适合特定健康状况
- category_filter: 分类筛选，按分类筛选菜品

常见健康需求：
- 减肥：推荐低热量、低脂、清淡菜品
- 糖尿病：推荐低糖、少油、清淡菜品
- 高血压：推荐低盐、清淡菜品
- 老人饮食：推荐易消化、营养、温和菜品
- 儿童饮食：推荐营养、温和菜品

工作流程：
1. 分析用户的健康状况或饮食需求
2. 使用health_recipe_search检索适合的菜品
3. 使用dietary_check检查具体菜品是否合适
4. 提供健康饮食建议

注意事项：
- 始终提醒用户注意个体差异
- 提供具体的饮食建议
- 标注禁忌或需谨慎的食材/菜品

返回格式示例：
为减肥人群推荐：

推荐菜品：
1. 清蒸鱼 - 低脂高蛋白，热量约150大卡/100g
2. 西兰花炒鸡胸肉 - 营养均衡，热量约200大卡
3. 凉拌黄瓜 - 清爽低热量

饮食建议：
- 优先选择蒸、煮、凉拌等烹饪方式
- 避免油炸、重油菜品
- 注意控制主食摄入量

禁忌提醒：
- 避免红烧肉等高脂肪菜品
- 避免糖醋类菜品
"""


class HealthAdvisorAgent:
    """
    健康顾问Agent

    职责：
    - 健康饮食推荐
    - 饮食禁忌检查
    - 健康饮食建议
    """

    def __init__(self, llm, tools: List[tool]):
        """
        初始化健康顾问Agent

        Args:
            llm: LangChain LLM实例
            tools: 可用的工具列表
        """
        self.llm = llm
        self.tools = tools
        self.system_prompt = SystemMessage(content=HEALTH_ADVISOR_PROMPT)

        # 绑定工具到LLM
        self.llm_with_tools = llm.bind_tools(tools)

        logger.info(f"HealthAdvisorAgent初始化完成，可用工具: {[t.name for t in tools]}")

    def invoke(self, query: str) -> str:
        """
        执行健康饮食查询

        Args:
            query: 用户查询

        Returns:
            健康饮食建议
        """
        logger.info(f"HealthAdvisorAgent处理查询: {query}")

        messages = [
            self.system_prompt,
            HumanMessage(content=f"用户查询: {query}\n请提供健康饮食建议。")
        ]

        try:
            # 调用LLM执行
            response = self.llm_with_tools.invoke(messages)

            # 处理工具调用
            if response.tool_calls:
                tool_results = self._execute_tools(response.tool_calls)
                messages.append(response)
                for result in tool_results:
                    messages.append(result)

                final_response = self.llm.invoke(messages)
                return final_response.content
            else:
                return response.content

        except Exception as e:
            logger.error(f"HealthAdvisorAgent执行失败: {e}")
            return f"健康饮食查询出错: {str(e)}"

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
            "name": "HealthAdvisorAgent",
            "role": "健康顾问",
            "tools": [t.name for t in self.tools],
            "description": "专门负责健康饮食建议和禁忌检查"
        }