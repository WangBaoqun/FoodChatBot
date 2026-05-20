"""
食材相关工具
提供食材搭配、替代等功能
"""

import json
import logging
from typing import List, Dict, Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# JSON序列化辅助函数（确保中文正常显示）
def _json_dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)

# 全局检索器实例
_retrievers = {
    "graph_rag": None,
    "neo4j_driver": None
}


def initialize_retrievers(graph_rag_retrieval=None, neo4j_driver=None):
    """初始化检索器实例"""
    _retrievers["graph_rag"] = graph_rag_retrieval
    _retrievers["neo4j_driver"] = neo4j_driver
    logger.info("食材工具初始化完成")


@tool
def ingredient_pairing(ingredient: str, target_type: str = "", top_k: int = 10) -> str:
    """
    查询食材的搭配推荐。

    通过图遍历查找与指定食材经常一起使用的其他食材。

    Args:
        ingredient: 食材名称（如：鸡肉、牛肉、西红柿）
        target_type: 目标食材类型（可选：蔬菜、肉类、调味料）
        top_k: 返回结果数量

    Returns:
        搭配推荐的JSON字符串
    """
    logger.info(f"ingredient_pairing: {ingredient}, target={target_type}")

    if _retrievers["neo4j_driver"] is None:
        return _json_dumps({"error": "Neo4j连接未初始化", "pairings": []})

    try:
        driver = _retrievers["neo4j_driver"]

        with driver.session() as session:
            # 图遍历查询：食材 → 菜谱 → 其他食材
            target_filter = ""
            if target_type:
                target_filter = f"AND other.category CONTAINS '{target_type}'"

            cypher = f"""
            // 查找使用该食材的菜谱（注意关系方向：Recipe -> Ingredient）
            MATCH (r:Recipe)-[:REQUIRES]->(i:Ingredient)
            WHERE i.name CONTAINS '{ingredient}'

            // 找出这些菜谱使用的其他食材
            MATCH (r)-[:REQUIRES]->(other:Ingredient)
            WHERE other.name <> i.name {target_filter}

            // 统计搭配频率
            WITH other, count(r) as pairing_count
            ORDER BY pairing_count DESC
            LIMIT {top_k}

            RETURN other.name as name, other.category as category, pairing_count
            """

            result = session.run(cypher)
            pairings = []
            for record in result:
                pairings.append({
                    "ingredient": record["name"],
                    "category": record["category"],
                    "pairing_count": record["pairing_count"],
                    "recommendation_level": "高" if record["pairing_count"] > 10 else "中"
                })

            return _json_dumps({
                "base_ingredient": ingredient,
                "target_type": target_type,
                "pairings": pairings,
                "count": len(pairings),
                "summary": f"找到{len(pairings)}种与{ingredient}搭配的食材"
            })
    except Exception as e:
        logger.error(f"食材搭配查询失败: {e}")
        return _json_dumps({"error": str(e), "pairings": []})


@tool
def ingredient_substitute(ingredient: str, reason: str = "", top_k: int = 5) -> str:
    """
    查询食材替代推荐。

    根据食材类型和特性推荐替代食材。

    Args:
        ingredient: 需要替代的食材名称
        reason: 替代原因（可选：过敏、口味、成本）
        top_k: 返回结果数量

    Returns:
        替代推荐的JSON字符串
    """
    logger.info(f"ingredient_substitute: {ingredient}, reason={reason}")

    if _retrievers["neo4j_driver"] is None:
        return _json_dumps({"error": "Neo4j连接未初始化", "substitutes": []})

    try:
        driver = _retrievers["neo4j_driver"]

        with driver.session() as session:
            # 查询同类型食材作为替代
            cypher = f"""
            // 查找目标食材的类型
            MATCH (i:Ingredient)
            WHERE i.name CONTAINS '{ingredient}'
            WITH i.category as target_category

            // 查找同类型食材
            MATCH (other:Ingredient)
            WHERE other.category = target_category AND other.name <> '{ingredient}'
            RETURN other.name as name, other.category as category
            LIMIT {top_k}
            """

            result = session.run(cypher)
            substitutes = []
            for record in result:
                substitutes.append({
                    "ingredient": record["name"],
                    "category": record["category"],
                    "substitution_ratio": "相似"
                })

            # 如果没有结果，提供通用建议
            if not substitutes:
                substitutes = [
                    {"ingredient": "请查询具体食材", "category": "通用", "note": "无匹配替代"}
                ]

            return _json_dumps({
                "original_ingredient": ingredient,
                "reason": reason,
                "substitutes": substitutes,
                "count": len(substitutes),
                "usage_tip": "替代后可能需要调整烹饪时间和用量"
            })
    except Exception as e:
        logger.error(f"食材替代查询失败: {e}")
        return _json_dumps({"error": str(e), "substitutes": []})


@tool
def graph_neighbor_search(node_name: str, depth: int = 1, top_k: int = 10) -> str:
    """
    图邻居节点查询。

    查询指定节点的邻居关系，用于探索食材或菜谱的关联网络。

    Args:
        node_name: 节点名称（食材或菜谱）
        depth: 遍历深度（1或2）
        top_k: 返回结果数量

    Returns:
        邻居节点的JSON字符串
    """
    logger.info(f"graph_neighbor_search: {node_name}, depth={depth}")

    if _retrievers["neo4j_driver"] is None:
        return _json_dumps({"error": "Neo4j连接未初始化", "neighbors": []})

    try:
        driver = _retrievers["neo4j_driver"]

        with driver.session() as session:
            cypher = f"""
            MATCH (n)
            WHERE n.name CONTAINS '{node_name}'

            MATCH (n)-[r*1..{depth}]-(neighbor)
            WHERE NOT neighbor = n

            WITH distinct neighbor, type(last(r)) as relation_type
            RETURN neighbor.name as name,
                   labels(neighbor)[0] as node_type,
                   relation_type
            LIMIT {top_k}
            """

            result = session.run(cypher)
            neighbors = []
            for record in result:
                neighbors.append({
                    "name": record["name"],
                    "node_type": record["node_type"],
                    "relation_type": record["relation_type"]
                })

            return _json_dumps({
                "center_node": node_name,
                "depth": depth,
                "neighbors": neighbors,
                "count": len(neighbors)
            })
    except Exception as e:
        logger.error(f"图邻居查询失败: {e}")
        return _json_dumps({"error": str(e), "neighbors": []})