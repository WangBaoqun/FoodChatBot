"""
Agentic RAG 功能测试脚本
测试主动探索式检索的核心功能
"""

import os
import sys

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from config import DEFAULT_CONFIG
from rag_modules import (
    GraphDataPreparationModule,
    MilvusIndexConstructionModule,
    GenerationIntegrationModule,
    AgenticQueryRouter
)
from rag_modules.hybrid_retrieval import HybridRetrievalModule
from rag_modules.graph_rag_retrieval import GraphRAGRetrieval
from rag_modules.intelligent_query_router import IntelligentQueryRouter


def test_agentic_router():
    """测试 AgenticQueryRouter 核心功能"""

    print("=" * 60)
    print("Agentic RAG 功能测试")
    print("=" * 60)

    # 1. 初始化各模块
    print("\n[1] 初始化模块...")

    try:
        # 数据模块
        data_module = GraphDataPreparationModule(
            uri=DEFAULT_CONFIG.neo4j_uri,
            user=DEFAULT_CONFIG.neo4j_user,
            password=DEFAULT_CONFIG.neo4j_password,
            database=DEFAULT_CONFIG.neo4j_database
        )

        # 向量索引模块
        index_module = MilvusIndexConstructionModule(
            host=DEFAULT_CONFIG.milvus_host,
            port=DEFAULT_CONFIG.milvus_port,
            collection_name=DEFAULT_CONFIG.milvus_collection_name,
            dimension=DEFAULT_CONFIG.milvus_dimension,
            model_name=DEFAULT_CONFIG.embedding_model
        )

        # 生成模块
        generation_module = GenerationIntegrationModule(
            model_name=DEFAULT_CONFIG.llm_model,
            temperature=DEFAULT_CONFIG.temperature,
            max_tokens=DEFAULT_CONFIG.max_tokens
        )

        print("✅ 模块初始化成功")

    except Exception as e:
        print(f"❌ 模块初始化失败: {e}")
        return

    # 2. 加载知识库
    print("\n[2] 加载知识库...")

    try:
        if index_module.has_collection():
            index_module.load_collection()
            data_module.load_graph_data()
            data_module.build_recipe_documents()
            chunks = data_module.chunk_documents(
                chunk_size=DEFAULT_CONFIG.chunk_size,
                chunk_overlap=DEFAULT_CONFIG.chunk_overlap
            )
            print(f"✅ 知识库加载成功，文档块数: {len(chunks)}")
        else:
            print("❌ 知识库不存在，请先运行 main.py 构建知识库")
            return

    except Exception as e:
        print(f"❌ 知识库加载失败: {e}")
        return

    # 3. 初始化检索模块
    print("\n[3] 初始化检索引擎...")

    try:
        # 传统检索
        traditional_retrieval = HybridRetrievalModule(
            config=DEFAULT_CONFIG,
            milvus_module=index_module,
            data_module=data_module,
            llm_client=generation_module.client
        )
        traditional_retrieval.initialize(chunks)

        # 图RAG检索
        graph_rag_retrieval = GraphRAGRetrieval(
            config=DEFAULT_CONFIG,
            llm_client=generation_module.client
        )
        graph_rag_retrieval.initialize()

        # 智能路由器
        intelligent_router = IntelligentQueryRouter(
            traditional_retrieval=traditional_retrieval,
            graph_rag_retrieval=graph_rag_retrieval,
            llm_client=generation_module.client,
            config=DEFAULT_CONFIG
        )

        # Agentic路由器
        agentic_router = AgenticQueryRouter(
            traditional_retrieval=traditional_retrieval,
            graph_rag_retrieval=graph_rag_retrieval,
            intelligent_router=intelligent_router,
            llm_client=generation_module.client,
            config=DEFAULT_CONFIG
        )

        print("✅ 检索引擎初始化成功")

    except Exception as e:
        print(f"❌ 检索引擎初始化失败: {e}")
        return

    # 4. 测试Agentic检索
    print("\n[4] 测试Agentic检索功能")
    print("-" * 40)

    # 测试用例
    test_queries = [
        # 简单查询 - 期望1轮完成
        "红烧肉怎么做",

        # 中等复杂度 - 可能需要2轮
        "适合减肥的低热量菜品有哪些",

        # 复杂查询 - 可能需要多轮迭代
        "糖尿病人可以吃的川菜，制作时间不要太长"
    ]

    for i, query in enumerate(test_queries):
        print(f"\n--- 测试 {i+1}: {query} ---")

        try:
            # 执行Agentic检索
            docs, trace = agentic_router.agentic_search(
                query=query,
                top_k=5
            )

            # 显示结果
            print(f"迭代次数: {trace.iterations}")
            print(f"最终质量分数: {trace.final_quality:.2f}")
            print(f"查询演变: {trace.query_evolution}")

            if docs:
                print(f"检索结果数: {len(docs)}")
                # 显示前3个结果
                for j, doc in enumerate(docs[:3]):
                    recipe_name = doc.metadata.get('recipe_name', '未知')
                    score = doc.metadata.get('final_score', doc.metadata.get('relevance_score', 0))
                    print(f"  [{j+1}] {recipe_name} (分数: {score:.3f})")
            else:
                print("未检索到结果")

            # 显示质量变化
            print(f"质量变化轨迹: {trace.quality_scores}")

        except Exception as e:
            print(f"❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()

    # 5. 显示统计信息
    print("\n[5] Agentic检索统计")
    print("-" * 40)

    stats = agentic_router.get_agentic_statistics()
    print(f"总Agentic查询次数: {stats['total_agentic_queries']}")
    print(f"平均迭代次数: {stats['avg_iterations']:.1f}")
    print(f"平均最终质量: {stats['avg_final_quality']:.2f}")
    print(f"成功达标率: {stats['successful_rate']:.1%}")
    print(f"配置: 最大迭代={stats['config']['max_iterations']}, 阈值={stats['config']['quality_threshold']}")

    # 6. 清理资源
    print("\n[6] 清理资源...")
    data_module.close()
    traditional_retrieval.close()
    graph_rag_retrieval.close()
    index_module.close()
    print("✅ 测试完成")


def test_quality_assessment():
    """单独测试质量评估功能"""

    print("\n" + "=" * 60)
    print("测试质量评估功能 (单独)")
    print("=" * 60)

    # 模拟文档
    from langchain_core.documents import Document

    mock_docs = [
        Document(
            page_content="菜品名称: 红烧肉\n描述: 经典家常菜，肉质软嫩，色泽红亮...",
            metadata={"recipe_name": "红烧肉", "relevance_score": 0.9}
        ),
        Document(
            page_content="菜品名称: 清蒸鱼\n描述: 清淡健康，适合减肥人群...",
            metadata={"recipe_name": "清蒸鱼", "relevance_score": 0.7}
        )
    ]

    # 初始化LLM客户端
    try:
        generation_module = GenerationIntegrationModule()
        from rag_modules.agentic_router import AgenticQueryRouter

        # 创建简化版路由器（只测试评估功能）
        class MockRouter:
            def __init__(self, llm_client, config):
                self.llm_client = llm_client
                self.config = config

            def evaluate_retrieval_quality(self, query, docs):
                # 直接调用AgenticQueryRouter的评估方法
                from rag_modules.agentic_router import AgenticQueryRouter
                router = AgenticQueryRouter(
                    None, None, None, self.llm_client, self.config
                )
                return router.evaluate_retrieval_quality(query, docs)

        mock_router = MockRouter(generation_module.client, DEFAULT_CONFIG)

        # 测试评估
        query = "适合减肥的菜品"
        assessment = mock_router.evaluate_retrieval_quality(query, mock_docs)

        print(f"查询: {query}")
        print(f"评估分数: {assessment.score:.2f}")
        print(f"是否满意: {assessment.is_satisfactory}")
        print(f"反馈: {assessment.feedback}")
        print(f"缺失维度: {assessment.missing_aspects}")
        print(f"已覆盖维度: {assessment.relevant_aspects}")

    except Exception as e:
        print(f"❌ 质量评估测试失败: {e}")


if __name__ == "__main__":
    # 运行完整测试
    test_agentic_router()

    # 可选：单独测试质量评估
    # test_quality_assessment()