"""
RAG检索工具封装
将现有RAG模块封装为LangChain工具
"""

import json
import logging
from typing import List, Dict, Any, Optional

from langchain_core.tools import tool
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# JSON序列化辅助函数（确保中文正常显示）
def _json_dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)

# 全局检索器实例（由主程序初始化时注入）
_retrievers = {
    "graph_rag": None,
    "hybrid": None,
    "agentic": None,
    "neo4j_driver": None
}


def initialize_retrievers(
    graph_rag_retrieval=None,
    hybrid_retrieval=None,
    agentic_router=None,
    neo4j_driver=None
):
    """初始化检索器实例"""
    _retrievers["graph_rag"] = graph_rag_retrieval
    _retrievers["hybrid"] = hybrid_retrieval
    _retrievers["agentic"] = agentic_router
    _retrievers["neo4j_driver"] = neo4j_driver
    logger.info("RAG检索工具初始化完成")


@tool
def graph_rag_search(query: str, top_k: int = 5) -> str:
    """
    使用图RAG检索菜谱和食材关系信息。

    适合查询涉及：
    - 食材搭配关系
    - 菜谱分类和菜系
    - 多跳关系推理

    Args:
        query: 用户查询内容
        top_k: 返回结果数量

    Returns:
        检索结果的JSON字符串
    """
    logger.info(f"graph_rag_search: {query}")

    if _retrievers["graph_rag"] is None:
        return _json_dumps({"error": "图RAG检索器未初始化", "results": []})

    try:
        documents = _retrievers["graph_rag"].graph_rag_search(query, top_k)
        results = _format_documents(documents)
        return _json_dumps({"query": query, "results": results, "count": len(results)})
    except Exception as e:
        logger.error(f"图RAG检索失败: {e}")
        return _json_dumps({"error": str(e), "results": []})


@tool
def hybrid_search(query: str, top_k: int = 5) -> str:
    """
    使用混合检索查找菜谱。

    结合向量检索和图结构检索，适合：
    - 通用菜谱查询
    - 关键词匹配检索
    - 简单信息查找

    Args:
        query: 用户查询内容
        top_k: 返回结果数量

    Returns:
        检索结果的JSON字符串
    """
    logger.info(f"hybrid_search: {query}")

    if _retrievers["hybrid"] is None:
        return _json_dumps({"error": "混合检索器未初始化", "results": []})

    try:
        documents = _retrievers["hybrid"].hybrid_search(query, top_k)
        results = _format_documents(documents)
        return _json_dumps({"query": query, "results": results, "count": len(results)})
    except Exception as e:
        logger.error(f"混合检索失败: {e}")
        return _json_dumps({"error": str(e), "results": []})


@tool
def agentic_search(query: str, top_k: int = 5) -> str:
    """
    使用Agentic迭代检索，支持反思和查询改写。

    适合复杂查询场景：
    - 多约束条件查询
    - 需要多次尝试优化的查询
    - 需要高质量结果的查询

    Args:
        query: 用户查询内容
        top_k: 返回结果数量

    Returns:
        检索结果和执行轨迹的JSON字符串
    """
    logger.info(f"agentic_search: {query}")

    if _retrievers["agentic"] is None:
        return _json_dumps({"error": "Agentic检索器未初始化", "results": []})

    try:
        documents, trace = _retrievers["agentic"].agentic_search(query, top_k)
        results = _format_documents(documents)
        trace_info = {
            "iterations": trace.iterations,
            "final_quality": trace.final_quality,
            "query_evolution": trace.query_evolution
        }
        return _json_dumps({
            "query": query,
            "results": results,
            "count": len(results),
            "trace": trace_info
        })
    except Exception as e:
        logger.error(f"Agentic检索失败: {e}")
        return _json_dumps({"error": str(e), "results": []})


@tool
def filter_by_category(category: str, cuisine_type: str = "", difficulty: str = "", top_k: int = 10) -> str:
    """
    按分类、菜系、难度筛选菜谱。

    Args:
        category: 菜品分类（如：家常菜、川菜、粤菜）
        cuisine_type: 菜系类型（可选）
        difficulty: 难度等级（可选：简单、中等、困难）
        top_k: 返回结果数量

    Returns:
        筛选结果的JSON字符串
    """
    logger.info(f"filter_by_category: category={category}, cuisine={cuisine_type}")

    if _retrievers["neo4j_driver"] is None:
        return _json_dumps({"error": "Neo4j连接未初始化", "results": []})

    try:
        driver = _retrievers["neo4j_driver"]

        with driver.session() as session:
            # 构建Cypher查询
            conditions = ["r.category IS NOT NULL"]
            if category:
                conditions.append(f"r.category CONTAINS '{category}'")
            if cuisine_type:
                conditions.append(f"r.cuisineType CONTAINS '{cuisine_type}'")
            if difficulty:
                conditions.append(f"r.difficulty = '{difficulty}'")

            cypher = f"""
            MATCH (r:Recipe)
            WHERE { ' AND '.join(conditions) }
            RETURN r.nodeId as id, r.name as name, r.category as category,
                   r.cuisineType as cuisine_type, r.difficulty as difficulty
            LIMIT {top_k}
            """

            result = session.run(cypher)
            recipes = []
            for record in result:
                recipes.append({
                    "id": record["id"],
                    "name": record["name"],
                    "category": record["category"],
                    "cuisine_type": record["cuisine_type"],
                    "difficulty": record["difficulty"]
                })

            return _json_dumps({
                "category": category,
                "cuisine_type": cuisine_type,
                "difficulty": difficulty,
                "results": recipes,
                "count": len(recipes)
            })
    except Exception as e:
        logger.error(f"分类筛选失败: {e}")
        return _json_dumps({"error": str(e), "results": []})


def _format_documents(documents: List[Document]) -> List[Dict[str, Any]]:
    """将Document对象格式化为字典列表"""
    results = []
    for doc in documents:
        results.append({
            "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
            "recipe_name": doc.metadata.get("recipe_name", "未知"),
            "search_type": doc.metadata.get("search_type", doc.metadata.get("search_method", "unknown")),
            "score": doc.metadata.get("final_score", doc.metadata.get("relevance_score", 0)),
            "node_id": doc.metadata.get("node_id", "")
        })
    return results