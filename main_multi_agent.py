"""
Multi-Agent协作系统主程序
集成各Agent组件，提供交互式问答功能
"""

import os
import sys
import time
import logging
from typing import Dict, Any, Optional

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from config import DEFAULT_CONFIG, GraphRAGConfig

# 导入RAG模块
from rag_modules import (
    GraphDataPreparationModule,
    MilvusIndexConstructionModule,
    GenerationIntegrationModule,
    AgenticQueryRouter
)
from rag_modules.hybrid_retrieval import HybridRetrievalModule
from rag_modules.graph_rag_retrieval import GraphRAGRetrieval
from rag_modules.intelligent_query_router import IntelligentQueryRouter

# 导入Agent模块
from agents import (
    OrchestratorAgent,
    RecipeFinderAgent,
    IngredientExpertAgent,
    HealthAdvisorAgent,
    ExecutionVisualizer
)
from agents.graph_workflow import SimpleMultiAgentWorkflow
from agents.tools.rag_tools import initialize_retrievers as init_rag_tools
from agents.tools.ingredient_tools import initialize_retrievers as init_ingredient_tools
from agents.tools.health_tools import initialize_retrievers as init_health_tools

# 加载环境变量
load_dotenv()


class MultiAgentCookingSystem:
    """
    Multi-Agent烹饪助手系统

    核心组件：
    1. OrchestratorAgent：协调器，分析意图、分派任务
    2. RecipeFinderAgent：菜谱查找
    3. IngredientExpertAgent：食材专家
    4. HealthAdvisorAgent：健康顾问
    5. ExecutionVisualizer：执行流程可视化
    """

    def __init__(self, config: Optional[GraphRAGConfig] = None):
        self.config = config or DEFAULT_CONFIG

        # RAG组件
        self.data_module = None
        self.index_module = None
        self.generation_module = None
        self.traditional_retrieval = None
        self.graph_rag_retrieval = None
        self.agentic_router = None

        # Agent组件
        self.orchestrator = None
        self.recipe_finder = None
        self.ingredient_expert = None
        self.health_advisor = None
        self.visualizer = None
        self.workflow = None

        # 系统状态
        self.system_ready = False

    def initialize(self):
        """初始化系统"""
        logger.info("初始化Multi-Agent系统...")

        try:
            # 1. 初始化RAG组件（复用现有）
            print("初始化数据模块...")
            self._init_rag_components()

            # 2. 初始化Agent组件
            print("初始化Agent组件...")
            self._init_agent_components()

            # 3. 初始化工具
            print("初始化工具层...")
            self._init_tools()

            # 4. 创建工作流
            print("创建工作流...")
            self._init_workflow()

            self.system_ready = True
            print("✅ Multi-Agent系统初始化完成！")

        except Exception as e:
            logger.error(f"系统初始化失败: {e}")
            raise

    def _init_rag_components(self):
        """初始化RAG组件"""
        # 数据模块
        self.data_module = GraphDataPreparationModule(
            uri=self.config.neo4j_uri,
            user=self.config.neo4j_user,
            password=self.config.neo4j_password,
            database=self.config.neo4j_database
        )

        # 向量索引模块
        self.index_module = MilvusIndexConstructionModule(
            host=self.config.milvus_host,
            port=self.config.milvus_port,
            collection_name=self.config.milvus_collection_name,
            dimension=self.config.milvus_dimension,
            model_name=self.config.embedding_model
        )

        # 生成模块
        self.generation_module = GenerationIntegrationModule(
            model_name=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )

    def _load_knowledge_base(self):
        """加载知识库"""
        logger.info("加载知识库...")

        if self.index_module.has_collection():
            self.index_module.load_collection()
            self.data_module.load_graph_data()
            self.data_module.build_recipe_documents()
            chunks = self.data_module.chunk_documents(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )

            # 初始化检索器
            self.traditional_retrieval = HybridRetrievalModule(
                config=self.config,
                milvus_module=self.index_module,
                data_module=self.data_module,
                llm_client=self.generation_module.client
            )
            self.traditional_retrieval.initialize(chunks)

            self.graph_rag_retrieval = GraphRAGRetrieval(
                config=self.config,
                llm_client=self.generation_module.client
            )
            self.graph_rag_retrieval.initialize()

            # Agentic路由器
            intelligent_router = IntelligentQueryRouter(
                traditional_retrieval=self.traditional_retrieval,
                graph_rag_retrieval=self.graph_rag_retrieval,
                llm_client=self.generation_module.client,
                config=self.config
            )

            self.agentic_router = AgenticQueryRouter(
                traditional_retrieval=self.traditional_retrieval,
                graph_rag_retrieval=self.graph_rag_retrieval,
                intelligent_router=intelligent_router,
                llm_client=self.generation_module.client,
                config=self.config
            )

            logger.info("知识库加载完成")
            return True
        else:
            logger.error("知识库不存在，请先运行main.py构建")
            return False

    def _init_agent_components(self):
        """初始化Agent组件"""
        # 确保RAG组件已初始化
        if not self._load_knowledge_base():
            raise RuntimeError("知识库加载失败")

        # 创建LLM实例（用于Agent）
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=self.config.llm_model,
            temperature=0.1,
            api_key=os.getenv("ANTHROPIC_AUTH_TOKEN"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        # 导入工具
        from agents.tools import (
            graph_rag_search, hybrid_search, agentic_search, filter_by_category,
            ingredient_pairing, ingredient_substitute, graph_neighbor_search,
            health_recipe_search, dietary_check, category_filter
        )

        # 创建专门化Agent
        self.recipe_finder = RecipeFinderAgent(
            llm=llm,
            tools=[graph_rag_search, hybrid_search, agentic_search, filter_by_category]
        )

        self.ingredient_expert = IngredientExpertAgent(
            llm=llm,
            tools=[ingredient_pairing, ingredient_substitute, graph_neighbor_search]
        )

        self.health_advisor = HealthAdvisorAgent(
            llm=llm,
            tools=[health_recipe_search, dietary_check, category_filter]
        )

        # 创建协调器
        specialized_agents = {
            "RecipeFinderAgent": self.recipe_finder,
            "IngredientExpertAgent": self.ingredient_expert,
            "HealthAdvisorAgent": self.health_advisor
        }

        self.orchestrator = OrchestratorAgent(
            llm=llm,
            specialized_agents=specialized_agents,
            config=self.config
        )

        # 创建可视化器
        self.visualizer = ExecutionVisualizer()

        logger.info("Agent组件初始化完成")

    def _init_tools(self):
        """初始化工具层"""
        # 获取Neo4j driver
        neo4j_driver = self.traditional_retrieval.driver

        # 初始化各工具模块的检索器
        init_rag_tools(
            graph_rag_retrieval=self.graph_rag_retrieval,
            hybrid_retrieval=self.traditional_retrieval,
            agentic_router=self.agentic_router,
            neo4j_driver=neo4j_driver
        )

        init_ingredient_tools(
            graph_rag_retrieval=self.graph_rag_retrieval,
            neo4j_driver=neo4j_driver
        )

        init_health_tools(
            graph_rag_retrieval=self.graph_rag_retrieval,
            neo4j_driver=neo4j_driver
        )

        logger.info("工具层初始化完成")

    def _init_workflow(self):
        """初始化工作流"""
        specialized_agents = {
            "RecipeFinderAgent": self.recipe_finder,
            "IngredientExpertAgent": self.ingredient_expert,
            "HealthAdvisorAgent": self.health_advisor
        }

        self.workflow = SimpleMultiAgentWorkflow(
            orchestrator=self.orchestrator,
            specialized_agents=specialized_agents,
            visualizer=self.visualizer,
            config=self.config
        )

        logger.info("工作流初始化完成")

    def ask(self, query: str, show_visualization: bool = True) -> str:
        """
        执行Multi-Agent问答

        Args:
            query: 用户查询
            show_visualization: 是否显示执行流程可视化

        Returns:
            回答结果
        """
        if not self.system_ready:
            raise RuntimeError("系统未就绪")

        print(f"\n❓ 用户问题: {query}")

        start_time = time.time()

        # 执行工作流
        result = self.workflow.run(query)

        # 显示可视化
        if show_visualization:
            print("\n" + result["visualization"])

        # 显示执行信息
        print(f"\n📋 执行信息:")
        print(f"   意图: {result['intent']}")
        print(f"   调用Agent: {', '.join(result['agents_called'])}")

        # 显示回答
        print("\n📝 回答:")
        print(result["final_answer"])

        # 性能统计
        end_time = time.time()
        print(f"\n⏱️ 执行耗时: {end_time - start_time:.2f}秒")

        return result["final_answer"]

    def run_interactive(self):
        """运行交互式问答"""
        if not self.system_ready:
            print("❌ 系统未就绪")
            return

        print("\n欢迎使用Multi-Agent烹饪助手！")
        print("系统包含以下专门化Agent:")
        print("   - RecipeFinderAgent: 菜谱查找")
        print("   - IngredientExpertAgent: 食材搭配")
        print("   - HealthAdvisorAgent: 健康饮食建议")
        print("\n可用命令:")
        print("   - 'hide' : 关闭流程可视化")
        print("   - 'show' : 显示流程可视化")
        print("   - 'stats' : 查看系统状态")
        print("   - 'quit' : 退出系统")
        print("\n" + "=" * 60)

        show_viz = True

        while True:
            try:
                user_input = input("\n您的问题: ").strip()

                if not user_input:
                    continue

                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'hide':
                    show_viz = False
                    print("✅ 流程可视化已关闭")
                    continue
                elif user_input.lower() == 'show':
                    show_viz = True
                    print("✅ 流程可视化已开启")
                    continue
                elif user_input.lower() == 'stats':
                    self._show_stats()
                    continue

                # 执行问答
                self.ask(user_input, show_visualization=show_viz)

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"处理出错: {e}")
                logger.error(f"问答处理失败: {e}")

        print("\n👋 感谢使用Multi-Agent烹饪助手！")
        self._cleanup()

    def _show_stats(self):
        """显示系统统计"""
        print("\n系统状态:")
        print("-" * 40)
        print(f"   RAG系统: ✅ 就绪")
        print(f"   Orchestrator: ✅ 就绪")
        print(f"   RecipeFinder: ✅ 就绪")
        print(f"   IngredientExpert: ✅ 就绪")
        print(f"   HealthAdvisor: ✅ 就绪")

        # RAG统计
        if self.agentic_router:
            stats = self.agentic_router.get_agentic_statistics()
            print(f"\n   Agentic统计:")
            print(f"      总查询: {stats['total_agentic_queries']}")
            print(f"      平均迭代: {stats['avg_iterations']:.1f}")

    def _cleanup(self):
        """清理资源"""
        if self.data_module:
            self.data_module.close()
        if self.traditional_retrieval:
            self.traditional_retrieval.close()
        if self.graph_rag_retrieval:
            self.graph_rag_retrieval.close()
        if self.index_module:
            self.index_module.close()


def main():
    """主函数"""
    print("启动Multi-Agent烹饪助手系统...")

    try:
        # 创建系统
        system = MultiAgentCookingSystem()

        # 初始化
        system.initialize()

        # 运行交互模式
        system.run_interactive()

    except Exception as e:
        logger.error(f"系统运行失败: {e}")
        import traceback
        traceback.print_exc()
        print(f"\n❌ 系统错误: {e}")


if __name__ == "__main__":
    main()