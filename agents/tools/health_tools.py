"""
健康饮食相关工具
提供健康菜谱推荐和饮食禁忌检查
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

# 常见健康需求关键词映射
HEALTH_KEYWORDS = {
    "减肥": ["清淡", "低脂", "蔬菜", "低热量"],
    "低热量": ["蔬菜", "清淡", "蒸", "凉拌"],
    "糖尿病": ["低糖", "清淡", "蔬菜", "少油"],
    "高血压": ["低盐", "清淡", "蔬菜"],
    "高血脂": ["低脂", "蔬菜", "清淡"],
    "孕妇": ["营养", "温和", "清淡"],
    "老人": ["清淡", "易消化", "营养"],
    "儿童": ["营养", "温和", "蔬菜"]
}

# 饮食禁忌提示
DIETARY_WARNINGS = {
    "糖尿病": "注意控制糖分摄入，避免高糖菜品",
    "高血压": "注意控制盐分摄入，避免重口味菜品",
    "减肥": "注意热量摄入，避免油炸和重油菜品",
    "老人": "注意消化能力，避免过于辛辣或油腻的菜品"
}


def initialize_retrievers(graph_rag_retrieval=None, neo4j_driver=None):
    """初始化检索器实例"""
    _retrievers["graph_rag"] = graph_rag_retrieval
    _retrievers["neo4j_driver"] = neo4j_driver
    logger.info("健康工具初始化完成")


@tool
def health_recipe_search(health_need: str, category: str = "", top_k: int = 10) -> str:
    """
    根据健康需求检索适合的菜谱。

    Args:
        health_need: 健康需求（如：减肥、糖尿病、高血压、老人饮食）
        category: 菜品分类限制（可选）
        top_k: 返回结果数量

    Returns:
        健康菜谱推荐的JSON字符串
    """
    logger.info(f"health_recipe_search: {health_need}, category={category}")

    if _retrievers["neo4j_driver"] is None:
        return _json_dumps({"error": "Neo4j连接未初始化", "recipes": []})

    try:
        driver = _retrievers["neo4j_driver"]

        # 获取健康关键词
        keywords = HEALTH_KEYWORDS.get(health_need, ["清淡", "营养"])

        with driver.session() as session:
            # 改进查询：使用Category节点和更宽松的匹配
            # 方式1：通过Category节点匹配健康相关分类
            # 方式2：匹配difficulty（简单难度的通常更清淡）
            # 方式3：匹配tags字段

            cypher = f"""
            MATCH (r:Recipe)
            WHERE r.nodeId >= '200000000'

            // 尝试多种匹配方式
            OPTIONAL MATCH (r)-[:BELONGS_TO_CATEGORY]->(c:Category)
            WHERE c.name IN ['素菜', '清淡', '凉菜', '蒸菜', '汤', '蔬菜']

            OPTIONAL MATCH (r)-[:REQUIRES]->(i:Ingredient)
            WHERE i.category IN ['蔬菜', '蛋白质']

            WITH r, c,
                 CASE
                     WHEN c.name IS NOT NULL THEN true
                     WHEN r.difficulty <= 2 THEN true
                     WHEN r.tags IS NOT NULL AND (
                         r.tags CONTAINS '清淡' OR
                         r.tags CONTAINS '低脂' OR
                         r.tags CONTAINS '健康' OR
                         r.tags CONTAINS '减肥'
                     ) THEN true
                     ELSE false
                 END as isHealthy,
                 collect(DISTINCT i.name) as healthyIngredients

            WHERE isHealthy = true
            {f"AND r.category CONTAINS '{category}'" if category else ""}

            RETURN r.nodeId as id, r.name as name, r.category as category,
                   r.cuisineType as cuisine_type, r.difficulty as difficulty,
                   r.description as description, healthyIngredients
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
                    "difficulty": record["difficulty"],
                    "description": record["description"][:100] if record["description"] else "",
                    "healthy_ingredients": record["healthyIngredients"][:3] if record["healthyIngredients"] else [],
                    "health_match": health_need
                })

            # 获取饮食提示
            warning = DIETARY_WARNINGS.get(health_need, "")

            return _json_dumps({
                "health_need": health_need,
                "keywords_used": keywords,
                "recipes": recipes,
                "count": len(recipes),
                "dietary_warning": warning,
                "summary": f"为{health_need}需求推荐{len(recipes)}道菜品"
            })
    except Exception as e:
        logger.error(f"健康菜谱检索失败: {e}")
        return _json_dumps({"error": str(e), "recipes": []})


@tool
def dietary_check(recipe_name: str, health_condition: str) -> str:
    """
    检查菜品是否符合特定健康条件的饮食要求。

    Args:
        recipe_name: 菜品名称
        health_condition: 健康状况（如：糖尿病、高血压、减肥）

    Returns:
        饮食禁忌检查结果的JSON字符串
    """
    logger.info(f"dietary_check: {recipe_name} for {health_condition}")

    try:
        # 基于健康关键词的简单检查
        keywords_to_avoid = {
            "糖尿病": ["糖", "甜", "蜜", "炸"],
            "高血压": ["盐", "咸", "辣", "腌制"],
            "减肥": ["炸", "油", "肥肉", "高热量"],
            "老人": ["辣", "炸", "硬", "生"]
        }

        avoid_keywords = keywords_to_avoid.get(health_condition, [])

        # 简单检查菜品名称中的禁忌词
        warnings = []
        for kw in avoid_keywords:
            if kw in recipe_name:
                warnings.append(f"菜品名称含'{kw}'，可能不适合{health_condition}人群")

        is_suitable = len(warnings) == 0

        recommendation = ""
        if not is_suitable:
            recommendation = f"建议{health_condition}人群谨慎食用或选择替代菜品"
        else:
            recommendation = f"该菜品适合{health_condition}人群食用"

        return _json_dumps({
            "recipe_name": recipe_name,
            "health_condition": health_condition,
            "is_suitable": is_suitable,
            "warnings": warnings,
            "recommendation": recommendation
        })
    except Exception as e:
        logger.error(f"饮食禁忌检查失败: {e}")
        return _json_dumps({"error": str(e), "is_suitable": True})


@tool
def category_filter(categories: str, exclude_categories: str = "", top_k: int = 20) -> str:
    """
    按分类筛选或排除菜谱。

    Args:
        categories: 要包含的分类（逗号分隔）
        exclude_categories: 要排除的分类（逗号分隔）
        top_k: 返回结果数量

    Returns:
        分类筛选结果的JSON字符串
    """
    logger.info(f"category_filter: include={categories}, exclude={exclude_categories}")

    if _retrievers["neo4j_driver"] is None:
        return _json_dumps({"error": "Neo4j连接未初始化", "recipes": []})

    try:
        driver = _retrievers["neo4j_driver"]

        include_list = [c.strip() for c in categories.split(",") if c.strip()]
        exclude_list = [c.strip() for c in exclude_categories.split(",") if c.strip()]

        with driver.session() as session:
            # 构建查询
            include_conditions = [f"r.category CONTAINS '{c}'" for c in include_list]
            exclude_conditions = [f"NOT r.category CONTAINS '{c}'" for c in exclude_list]

            where_clause = " AND ".join(include_conditions + exclude_conditions) if (include_conditions or exclude_conditions) else "r.category IS NOT NULL"

            cypher = f"""
            MATCH (r:Recipe)
            WHERE {where_clause}
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
                "include_categories": include_list,
                "exclude_categories": exclude_list,
                "recipes": recipes,
                "count": len(recipes)
            })
    except Exception as e:
        logger.error(f"分类筛选失败: {e}")
        return _json_dumps({"error": str(e), "recipes": []})