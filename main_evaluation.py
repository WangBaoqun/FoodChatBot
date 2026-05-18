"""
RAGAS RAG 评估入口程序
支持完整评估和快速评估模式
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from evaluation.run_evaluation import RAGEvaluationRunner
from evaluation.generate_test_data import TestDataGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="RAGAS RAG 系统评估工具")

    parser.add_argument('--quick', action='store_true',
                        help='快速评估模式（仅评估5个样本）')

    parser.add_argument('--full', action='store_true',
                        help='完整评估模式（评估全部样本）')

    parser.add_argument('--generate-data', action='store_true',
                        help='重新生成测试数据集')

    parser.add_argument('--test-data', default='evaluation/test_dataset.json',
                        help='测试数据集路径')

    parser.add_argument('--output-dir', default='evaluation/results',
                        help='评估结果输出目录')

    parser.add_argument('--num-samples', type=int, default=None,
                        help='指定评估样本数量')

    args = parser.parse_args()

    print("=" * 60)
    print("RAGAS RAG 系统评估工具")
    print("=" * 60)

    # 生成测试数据
    if args.generate_data or not Path(args.test_data).exists():
        print("\n正在生成测试数据集...")
        generator = TestDataGenerator()
        questions = generator.generate_all_questions()
        generator.save_to_json(questions, args.test_data)
        print(f"测试数据集已生成: {args.test_data}")
        print(f"问题数量: {len(questions)}")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 确定评估模式
    quick_mode = args.quick or (not args.full and args.num_samples is None)

    # 如果指定了样本数量
    if args.num_samples:
        quick_mode = args.num_samples <= 10
        print(f"\n将评估 {args.num_samples} 个样本")

    print(f"\n评估模式: {'快速模式' if quick_mode else '完整模式'}")
    print(f"测试数据: {args.test_data}")
    print(f"输出目录: {args.output_dir}")

    # 运行评估
    runner = RAGEvaluationRunner()

    try:
        print("\n正在初始化 RAG 系统...")
        runner.initialize()

        print("\n开始评估...")
        results = runner.run_evaluation(
            test_data_path=args.test_data,
            quick_mode=quick_mode,
            output_dir=str(output_dir)
        )

        # 显示最终结果
        print("\n" + "=" * 60)
        print("评估完成!")
        print("=" * 60)

        scores = results.get('scores', {})
        print("\n各指标得分:")
        for metric, score in scores.items():
            if metric != 'average':
                print(f"  {metric}: {score:.4f}")

        print(f"\n综合分数: {scores.get('average', 0):.4f}")

        # 性能建议
        avg_score = scores.get('average', 0)
        if avg_score >= 0.7:
            print("\n[优秀] RAG 系统表现良好!")
        elif avg_score >= 0.5:
            print("\n[中等] 系统表现一般，建议优化检索或生成模块")
        else:
            print("\n[待改进] 系统需要进一步优化")

            # 针对性建议
            if scores.get('faithfulness', 0) < 0.5:
                print("  - 忠实度低：检查答案是否偏离检索内容")
            if scores.get('answer_relevance', 0) < 0.5:
                print("  - 相关性低：优化答案与问题的匹配度")
            if scores.get('context_precision', 0) < 0.4:
                print("  - 上下文精确度低：减少检索噪声")
            if scores.get('context_recall', 0) < 0.3:
                print("  - 上下文召回率低：增加检索覆盖范围")

    except KeyboardInterrupt:
        print("\n\n评估被用户中断")
        return

    except Exception as e:
        logger.error(f"评估失败: {e}")
        import traceback
        traceback.print_exc()
        print(f"\n评估失败: {e}")
        print("请检查：")
        print("  1. Neo4j 和 Milvus 服务是否正常运行")
        print("  2. API 密钥配置是否正确")
        print("  3. 测试数据集是否存在")

    finally:
        runner.cleanup()
        print("\n资源已释放")


if __name__ == "__main__":
    main()