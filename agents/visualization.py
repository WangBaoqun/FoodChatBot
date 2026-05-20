"""
执行流程可视化模块
提供Agent执行过程的文本可视化
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ExecutionStep:
    """执行步骤记录"""
    agent_name: str
    action: str
    timestamp: str
    result_summary: str = ""
    tools_used: List[str] = field(default_factory=list)


class ExecutionVisualizer:
    """
    执行流程可视化器

    提供文本可视化功能，显示Agent执行路径：
    - Orchestrator决策过程
    - 各Agent执行情况
    - 工具调用记录
    - 最终整合过程
    """

    def __init__(self):
        self.steps: List[ExecutionStep] = []
        self.current_query: str = ""
        self.intent_analysis: Dict[str, Any] = {}

    def start_visualization(self, query: str):
        """开始可视化记录"""
        self.steps = []
        self.current_query = query
        self.intent_analysis = {}

        self._add_step(
            agent_name="OrchestratorAgent",
            action="接收查询",
            result_summary=f"查询: {query}"
        )

    def record_intent_analysis(self, intent: str, agents_to_call: List[str]):
        """记录意图分析结果"""
        self.intent_analysis = {
            "intent": intent,
            "agents": agents_to_call
        }

        self._add_step(
            agent_name="OrchestratorAgent",
            action="分析意图",
            result_summary=f"意图: {intent}，分派Agent: {', '.join(agents_to_call)}"
        )

    def record_agent_execution(
        self,
        agent_name: str,
        tools_used: List[str],
        result_summary: str
    ):
        """记录Agent执行"""
        self._add_step(
            agent_name=agent_name,
            action="执行查询",
            tools_used=tools_used,
            result_summary=result_summary
        )

    def record_synthesis(self, final_answer_summary: str):
        """记录结果整合"""
        self._add_step(
            agent_name="Synthesizer",
            action="整合结果",
            result_summary=final_answer_summary
        )

    def _add_step(
        self,
        agent_name: str,
        action: str,
        result_summary: str = "",
        tools_used: List[str] = None
    ):
        """添加执行步骤"""
        step = ExecutionStep(
            agent_name=agent_name,
            action=action,
            timestamp=datetime.now().strftime("%H:%M:%S"),
            result_summary=result_summary,
            tools_used=tools_used or []
        )
        self.steps.append(step)

    def render(self) -> str:
        """渲染可视化文本"""
        if not self.steps:
            return "无执行记录"

        lines = []
        lines.append("┌" + "─" * 50 + "┐")

        # 第一步：Orchestrator接收查询
        first_step = self.steps[0]
        lines.append(f"│ 🔍 OrchestratorAgent 分析意图")
        lines.append(f"│    查询: {self.current_query[:40]}{'...' if len(self.current_query) > 40 else ''}")

        if self.intent_analysis:
            lines.append(f"│    意图: {self.intent_analysis.get('intent', '未知')}")

        lines.append("└" + "─" * 50 + "┘")

        # Agent执行部分
        agent_steps = [s for s in self.steps if s.agent_name not in ["OrchestratorAgent", "Synthesizer"]]

        if agent_steps:
            lines.append("    ↓ 分派")

            # 渲染各Agent执行
            agent_count = len(agent_steps)
            if agent_count == 1:
                # 单Agent
                step = agent_steps[0]
                lines.append("┌" + "─" * 50 + "┐")
                lines.append(f"│ 🤖 {step.agent_name}")
                if step.tools_used:
                    lines.append(f"│    工具: [{', '.join(step.tools_used)}]")
                lines.append(f"│    {step.result_summary[:45]}{'...' if len(step.result_summary) > 45 else ''}")
                lines.append("└" + "─" * 50 + "┘")
            else:
                # 多Agent并行
                lines.append("┌" + "─" * 20 + "┬" + "─" * 30 + "┐")

                for i, step in enumerate(agent_steps):
                    prefix = "│" if i == 0 else "│"
                    lines.append(f"{prefix} {step.agent_name[:18]:<18} │ {step.tools_used[0] if step.tools_used else '执行':<28}")
                    if step.result_summary:
                        summary_short = step.result_summary[:25]
                        lines.append(f"{prefix} → {summary_short:<15} │ ...")

                lines.append("└" + "─" * 20 + "┴" + "─" * 30 + "┘")

        # 最终整合
        synth_steps = [s for s in self.steps if s.agent_name == "Synthesizer"]
        if synth_steps:
            lines.append("    ↓ 整合")
            lines.append("┌" + "─" * 50 + "┐")
            lines.append("│ 📝 Synthesizer 生成回答")
            synth = synth_steps[0]
            lines.append(f"│    {synth.result_summary[:45]}{'...' if len(synth.result_summary) > 45 else ''}")
            lines.append("└" + "─" * 50 + "┘")

        return "\n".join(lines)

    def render_compact(self) -> str:
        """渲染紧凑版可视化"""
        if not self.steps:
            return ""

        agents_used = [s.agent_name for s in self.steps if s.agent_name not in ["OrchestratorAgent", "Synthesizer"]]
        tools_all = []
        for s in self.steps:
            tools_all.extend(s.tools_used)

        return f"[流程] Orchestrator → {', '.join(agents_used)} → Synthesizer | 工具: [{', '.join(set(tools_all))}]"

    def get_execution_summary(self) -> Dict[str, Any]:
        """获取执行摘要"""
        return {
            "query": self.current_query,
            "intent": self.intent_analysis.get("intent", ""),
            "agents_used": [s.agent_name for s in self.steps if s.agent_name not in ["OrchestratorAgent", "Synthesizer"]],
            "tools_used": list(set(t for s in self.steps for t in s.tools_used)),
            "total_steps": len(self.steps)
        }