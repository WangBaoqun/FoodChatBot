"""
自动生成 RAGAS 测试数据集
从菜谱数据自动生成测试问题和 ground truth
"""

import os
import json
import csv
import random
import logging
from typing import List, Dict, Any
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestDataGenerator:
    """测试数据生成器"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.nodes_file = self.data_dir / "cypher" / "nodes.csv"
        self.relationships_file = self.data_dir / "cypher" / "relationships.csv"
        self.cook_dir = self.data_dir / "cook" / "dishes"

        # 数据存储
        self.recipes: List[Dict] = []
        self.ingredients: List[Dict] = []
        self.cooking_steps: List[Dict] = []
        self.categories: List[Dict] = []
        self.nodes_by_id: Dict[str, Dict] = {}

        # 关系存储
        self.recipe_ingredients: Dict[str, List[Dict]] = {}  # recipeId -> [ingredient details]
        self.recipe_steps: Dict[str, List[Dict]] = {}  # recipeId -> [step details]
        self.recipe_category: Dict[str, str] = {}  # recipeId -> category
        self.recipe_difficulty: Dict[str, str] = {}  # recipeId -> difficulty

        self._load_nodes()
        self._load_relationships()

    def _load_nodes(self):
        """加载 CSV 节点数据"""
        if not self.nodes_file.exists():
            logger.error(f"节点文件不存在: {self.nodes_file}")
            return

        with open(self.nodes_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                node_id = row.get('nodeId', '')
                labels = row.get('labels', '')
                self.nodes_by_id[node_id] = row

                if labels == 'Recipe':
                    self.recipes.append(row)
                elif labels == 'Ingredient':
                    self.ingredients.append(row)
                elif labels == 'CookingStep':
                    self.cooking_steps.append(row)
                elif labels == 'RecipeCategory':
                    self.categories.append(row)

        logger.info(f"加载节点: {len(self.recipes)} 菜谱, {len(self.ingredients)} 食材, {len(self.cooking_steps)} 步骤, {len(self.categories)} 分类")

    def _load_relationships(self):
        """加载关系数据，建立菜谱与食材/步骤的关联"""
        if not self.relationships_file.exists():
            logger.error(f"关系文件不存在: {self.relationships_file}")
            return

        with open(self.relationships_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                start_id = row.get('startNodeId', '')
                end_id = row.get('endNodeId', '')
                rel_type = row.get('relationshipType', '')
                amount = row.get('amount', '')
                unit = row.get('unit', '')
                step_order = row.get('step_order', '')

                # 801000001 = REQUIRES (recipe -> ingredient)
                # 801000003 = CONTAINS_STEP (recipe -> step)
                # 801000004 = BELONGS_TO_CATEGORY (recipe -> category)
                # 801000005 = HAS_DIFFICULTY (recipe -> difficulty)

                if rel_type == '801000001':  # REQUIRES
                    if start_id not in self.recipe_ingredients:
                        self.recipe_ingredients[start_id] = []
                    end_node = self.nodes_by_id.get(end_id, {})
                    self.recipe_ingredients[start_id].append({
                        'name': end_node.get('name', ''),
                        'amount': amount,
                        'unit': unit,
                        'isMain': end_node.get('isMain', 'False')
                    })

                elif rel_type == '801000003':  # CONTAINS_STEP
                    if start_id not in self.recipe_steps:
                        self.recipe_steps[start_id] = []
                    end_node = self.nodes_by_id.get(end_id, {})
                    self.recipe_steps[start_id].append({
                        'description': end_node.get('description', ''),
                        'step_order': float(step_order) if step_order else 0,
                        'methods': end_node.get('methods', ''),
                        'tools': end_node.get('tools', ''),
                        'timeEstimate': end_node.get('timeEstimate', '')
                    })

                elif rel_type == '801000004':  # BELONGS_TO_CATEGORY
                    end_node = self.nodes_by_id.get(end_id, {})
                    self.recipe_category[start_id] = end_node.get('name', '')

                elif rel_type == '801000005':  # HAS_DIFFICULTY
                    end_node = self.nodes_by_id.get(end_id, {})
                    self.recipe_difficulty[start_id] = end_node.get('name', '')

        logger.info(f"加载关系: {len(self.recipe_ingredients)} 个菜谱有食材, {len(self.recipe_steps)} 个菜谱有步骤")

    def _read_recipe_markdown(self, file_path: str) -> str:
        """读取菜谱 markdown 文件内容"""
        if not file_path:
            return ""
        full_path = self.cook_dir / file_path
        if full_path.exists():
            with open(full_path, 'r', encoding='utf-8') as f:
                return f.read()
        return ""

    def _parse_markdown_content(self, content: str) -> Dict:
        """解析 markdown 内容提取关键信息"""
        result = {'ingredients': [], 'steps': [], 'tips': []}
        if not content:
            return result

        lines = content.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()
            if '## 必备原料和工具' in line or '## 计算' in line:
                current_section = 'ingredients'
            elif '## 操作' in line:
                current_section = 'steps'
            elif '## 附加内容' in line or '## 技术总结' in line:
                current_section = 'tips'
            elif line.startswith('- ') or line.startswith('* '):
                item = line[2:].strip()
                if current_section == 'ingredients':
                    result['ingredients'].append(item)
                elif current_section == 'steps':
                    result['steps'].append(item)
                elif current_section == 'tips':
                    result['tips'].append(item)

        return result

    def generate_single_recipe_questions(self, count: int = 10) -> List[Dict]:
        """生成单菜谱查询问题"""
        questions = []

        # 选择有完整数据的菜谱
        valid_recipes = [r for r in self.recipes if r.get('nodeId') in self.recipe_steps]
        selected = random.sample(valid_recipes, min(count, len(valid_recipes)))

        for recipe in selected:
            name = recipe.get('name', '')
            node_id = recipe.get('nodeId', '')
            file_path = recipe.get('filePath', '')

            # 获取步骤
            steps = self.recipe_steps.get(node_id, [])
            steps_sorted = sorted(steps, key=lambda x: x.get('step_order', 0))

            if steps_sorted:
                step_descs = [s.get('description', '') for s in steps_sorted[:6] if s.get('description')]
                ground_truth = f"制作{name}的主要步骤：\n" + "\n".join(step_descs)

                questions.append({
                    "question": f"如何制作{name}？",
                    "ground_truth": ground_truth,
                    "query_type": "single_recipe"
                })

            # 获取食材
            ingredients = self.recipe_ingredients.get(node_id, [])
            if ingredients:
                main_ings = [i for i in ingredients if i.get('isMain') == 'True']
                all_ings = [f"{i.get('name', '')} {i.get('amount', '')}{i.get('unit', '')}" for i in ingredients[:8]]
                ground_truth = f"制作{name}需要的食材：\n" + "\n".join(all_ings)

                questions.append({
                    "question": f"制作{name}需要哪些食材？",
                    "ground_truth": ground_truth,
                    "query_type": "ingredients_query"
                })

        return questions

    def generate_category_questions(self) -> List[Dict]:
        """生成分类查询问题"""
        questions = []

        # 按分类收集菜谱
        category_recipes: Dict[str, List[str]] = {}
        for recipe in self.recipes:
            node_id = recipe.get('nodeId', '')
            cat = self.recipe_category.get(node_id, recipe.get('category', ''))
            if cat:
                if cat not in category_recipes:
                    category_recipes[cat] = []
                category_recipes[cat].append(recipe.get('name', ''))

        for category, recipe_names in category_recipes.items():
            if len(recipe_names) >= 3:
                questions.append({
                    "question": f"有哪些{category}菜品？",
                    "ground_truth": f"{category}菜品包括：" + "、".join(recipe_names[:10]),
                    "query_type": "category_query"
                })

        return questions

    def generate_difficulty_questions(self) -> List[Dict]:
        """生成难度查询问题"""
        questions = []

        # 按难度收集菜谱
        difficulty_recipes: Dict[str, List[str]] = {}
        for recipe in self.recipes:
            node_id = recipe.get('nodeId', '')
            diff = self.recipe_difficulty.get(node_id, '')
            if diff:
                if diff not in difficulty_recipes:
                    difficulty_recipes[diff] = []
                difficulty_recipes[diff].append(recipe.get('name', ''))

        for difficulty, recipe_names in difficulty_recipes.items():
            if len(recipe_names) >= 3:
                questions.append({
                    "question": f"有哪些{difficulty}难度的菜品？",
                    "ground_truth": f"{difficulty}难度的菜品包括：" + "、".join(recipe_names[:10]),
                    "query_type": "difficulty_query"
                })

        return questions

    def generate_step_detail_questions(self, count: int = 5) -> List[Dict]:
        """生成步骤详情查询问题"""
        questions = []

        # 选择有详细步骤的菜谱
        valid_recipes = [(r, self.recipe_steps.get(r.get('nodeId', ''), []))
                        for r in self.recipes if len(self.recipe_steps.get(r.get('nodeId', ''), [])) >= 3]
        selected = random.sample(valid_recipes, min(count, len(valid_recipes)))

        for recipe, steps in selected:
            name = recipe.get('name', '')
            steps_sorted = sorted(steps, key=lambda x: x.get('step_order', 0))

            if steps_sorted:
                first_step = steps_sorted[0]
                questions.append({
                    "question": f"制作{name}的第一步是什么？",
                    "ground_truth": f"制作{name}的第一步：{first_step.get('description', '')}",
                    "query_type": "step_query"
                })

        return questions

    def generate_time_questions(self, count: int = 5) -> List[Dict]:
        """生成时间相关问题"""
        questions = []

        # 选择有时间信息的菜谱
        valid_recipes = [r for r in self.recipes if r.get('prepTime') or r.get('cookTime')]
        selected = random.sample(valid_recipes, min(count, len(valid_recipes)))

        for recipe in selected:
            name = recipe.get('name', '')
            prep_time = recipe.get('prepTime', '')
            cook_time = recipe.get('cookTime', '')

            if prep_time or cook_time:
                ground_truth = f"制作{name}"
                if prep_time:
                    ground_truth += f"准备时间约{prep_time}"
                if cook_time:
                    ground_truth += f"，烹饪时间约{cook_time}"

                questions.append({
                    "question": f"制作{name}需要多长时间？",
                    "ground_truth": ground_truth,
                    "query_type": "time_query"
                })

        return questions

    def generate_method_questions(self) -> List[Dict]:
        """生成烹饪方法相关问题"""
        questions = []

        # 按烹饪方法关键词分类
        method_keywords = ['清蒸', '红烧', '煎', '炒', '煮', '炖', '烤', '炸']

        for method in method_keywords:
            matching = [r.get('name', '') for r in self.recipes if method in r.get('name', '')]
            if len(matching) >= 2:
                questions.append({
                    "question": f"有哪些{method}类菜品？",
                    "ground_truth": f"{method}类菜品包括：" + "、".join(matching[:10]),
                    "query_type": "method_query"
                })

        return questions

    def generate_tool_questions(self) -> List[Dict]:
        """生成工具相关问题"""
        questions = []

        # 从步骤中提取工具信息
        tool_recipes: Dict[str, List[str]] = {}

        for recipe in self.recipes:
            node_id = recipe.get('nodeId', '')
            steps = self.recipe_steps.get(node_id, [])
            for step in steps:
                tools = step.get('tools', '')
                if tools:
                    for tool in tools.split(','):
                        tool = tool.strip()
                        if tool and len(tool) < 10:  # 过滤过长的工具名
                            if tool not in tool_recipes:
                                tool_recipes[tool] = []
                            if recipe.get('name', '') not in tool_recipes[tool]:
                                tool_recipes[tool].append(recipe.get('name', ''))

        # 生成常见工具的问题
        common_tools = ['炒锅', '蒸锅', '微波炉', '烤箱', '空气炸锅', '平底锅']
        for tool in common_tools:
            if tool in tool_recipes and len(tool_recipes[tool]) >= 2:
                questions.append({
                    "question": f"哪些菜品需要使用{tool}？",
                    "ground_truth": f"需要使用{tool}的菜品：" + "、".join(tool_recipes[tool][:8]),
                    "query_type": "tool_query"
                })

        return questions

    def generate_all_questions(self) -> List[Dict]:
        """生成所有类型的测试问题"""
        all_questions = []

        # 各类型问题
        all_questions.extend(self.generate_single_recipe_questions(count=12))
        all_questions.extend(self.generate_category_questions())
        all_questions.extend(self.generate_difficulty_questions())
        all_questions.extend(self.generate_step_detail_questions(count=4))
        all_questions.extend(self.generate_time_questions(count=4))
        all_questions.extend(self.generate_method_questions())
        all_questions.extend(self.generate_tool_questions())

        logger.info(f"共生成 {len(all_questions)} 个测试问题")
        return all_questions

    def save_to_json(self, questions: List[Dict], output_file: str = "evaluation/test_dataset.json"):
        """保存测试数据集到 JSON 文件"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(questions, f, ensure_ascii=False, indent=2)

        logger.info(f"测试数据集已保存到: {output_path}")
        return output_path


def main():
    """主函数"""
    print("=" * 50)
    print("开始生成 RAGAS 测试数据集")
    print("=" * 50)

    generator = TestDataGenerator()

    if not generator.recipes:
        print("[错误] 未加载到菜谱数据，请检查 nodes.csv 文件")
        return

    print(f"\n已加载 {len(generator.recipes)} 个菜谱")

    # 生成测试问题
    questions = generator.generate_all_questions()

    # 打印问题类型分布
    type_counts = {}
    for q in questions:
        t = q.get('query_type', 'unknown')
        type_counts[t] = type_counts.get(t, 0) + 1

    print("\n问题类型分布:")
    for t, c in type_counts.items():
        print(f"  {t}: {c} 个")

    # 保存数据集
    output_file = generator.save_to_json(questions)

    print(f"\n[完成] 测试数据集生成完成！")
    print(f"   文件路径: {output_file}")
    print(f"   问题数量: {len(questions)}")

    # 显示部分示例
    print("\n示例问题:")
    for i, q in enumerate(questions[:5]):
        print(f"\n问题 {i+1}: {q['question']}")
        gt_preview = q['ground_truth'][:80] + "..." if len(q['ground_truth']) > 80 else q['ground_truth']
        print(f"标准答案: {gt_preview}")


if __name__ == "__main__":
    main()