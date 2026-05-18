"""
评估执行脚本
执行完整的 RAGAS 评估流程
"""

import os
import sys
import json
import logging
import argparse
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from main import AdvancedGraphRAGSystem
from config import DEFAULT_CONFIG
from rag_modules.ragas_evaluation import RagasEvaluator, load_test_dataset, save_rag_responses

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RAGEvaluationRunner:
    """RAG 评估执行器"""

    def __init__(self, config=None):
        self.config = config or DEFAULT_CONFIG
        self.rag_system = None
        self.evaluator = None

    def initialize(self):
        """初始化 RAG 系统和评估器"""
        logger.info("初始化 RAG 系统...")

        # 初始化 RAG 系统
        self.rag_system = AdvancedGraphRAGSystem(self.config)
        self.rag_system.initialize_system()
        self.rag_system.build_knowledge_base()

        # 初始化评估器
        logger.info("初始化 RAGAS 评估器...")
        self.evaluator = RagasEvaluator(
            llm_model=self.config.llm_model,
            embedding_model=self.config.embedding_model
        )

        logger.info("初始化完成")

    def collect_rag_responses(self, test_data: List[Dict], stream: bool = False) -> List[Dict]:
        """
        执行 RAG 查询并收集响应

        Args:
            test_data: 测试数据集
            stream: 是否使用流式输出

        Returns:
            RAG 响应列表
        """
        responses = []

        logger.info(f"开始收集 RAG 响应，共 {len(test_data)} 个问题...")

        for i, item in enumerate(test_data):
            question = item.get('question', '')
            logger.info(f"处理问题 {i+1}/{len(test_data)}: {question[:50]}...")

            try:
                # 先执行检索获取上下文
                docs, analysis = self.rag_system.query_router.route_query(question, self.config.top_k)
                contexts = [doc.page_content for doc in docs]

                # 执行完整问答获取答案
                result, _ = self.rag_system.ask_question_with_routing(
                    question,
                    stream=False,  # 评估时不使用流式
                    explain_routing=False
                )

                response = {
                    'question': question,
                    'answer': result if isinstance(result, str) else str(result),
                    'contexts': contexts[:5]  # 最多取 5 个上下文片段
                }

                responses.append(response)

                # 显示进度
                print(f"  [{i+1}/{len(test_data)}] 已完成")

            except Exception as e:
                logger.error(f"处理问题 {i+1} 失败: {e}")
                responses.append({
                    'question': question,
                    'answer': f"处理失败: {str(e)}",
                    'contexts': []
                })

        return responses

    def run_evaluation(self, test_data_path: str = "evaluation/test_dataset.json",
                       quick_mode: bool = False,
                       output_dir: str = "evaluation/results"):
        """
        运行完整评估流程

        Args:
            test_data_path: 测试数据集路径
            quick_mode: 快速模式（只评估少量样本）
            output_dir: 结果输出目录
        """
        # 加载测试数据
        test_data = load_test_dataset(test_data_path)

        if quick_mode:
            test_data = test_data[:5]
            logger.info(f"快速模式：只评估 {len(test_data)} 个样本")

        # 收集 RAG 响应
        responses = self.collect_rag_responses(test_data)

        # 保存 RAG 响应
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        response_file = Path(output_dir) / f"rag_responses_{timestamp}.json"
        save_rag_responses(responses, str(response_file))

        # 准备评估数据集
        dataset = self.evaluator.prepare_evaluation_dataset(test_data, responses)

        # 运行评估
        results = self.evaluator.run_evaluation(dataset)
        results['timestamp'] = timestamp

        # 保存结果
        results_file = Path(output_dir) / f"evaluation_results_{timestamp}.json"
        self.evaluator.save_results(results, str(results_file))

        # 生成报告
        report_file = Path(output_dir) / f"evaluation_report_{timestamp}.txt"
        report = self.evaluator.generate_report(results, str(report_file))

        # 显示报告
        print("\n" + report)

        return results

    def cleanup(self):
        """清理资源"""
        if self.rag_system:
            self.rag_system._cleanup()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="RAGAS RAG 评估")
    parser.add_argument('--quick', action='store_true', help='快速评估模式（5个样本）')
    parser.add_argument('--test-data', default='evaluation/test_dataset.json', help='测试数据集路径')
    parser.add_argument('--output-dir', default='evaluation/results', help='结果输出目录')

    args = parser.parse_args()

    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    runner = RAGEvaluationRunner()

    try:
        # 初始化
        runner.initialize()

        # 运行评估
        results = runner.run_evaluation(
            test_data_path=args.test_data,
            quick_mode=args.quick,
            output_dir=args.output_dir
        )

        print("\n评估完成!")

    except Exception as e:
        logger.error(f"评估失败: {e}")
        import traceback
        traceback.print_exc()
        print(f"\n评估失败: {e}")

    finally:
        runner.cleanup()


if __name__ == "__main__":
    main()