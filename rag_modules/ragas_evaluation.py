"""
RAGAS 评估封装模块
封装 RAGAS 框架，配置百炼 API 作为评估 LLM
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from ragas import EvaluationDataset, evaluate
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._context_precision import ContextPrecision
from ragas.metrics._context_recall import ContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class RagasEvaluator:
    """RAGAS 评估封装类"""

    def __init__(self,
                 llm_model: str = "qwen-plus",
                 embedding_model: str = "BAAI/bge-small-zh-v1.5",
                 api_key: Optional[str] = None,
                 base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"):
        """
        初始化 RAGAS 评估器

        Args:
            llm_model: 评估使用的 LLM 模型名称
            embedding_model: 嵌入模型名称
            api_key: API 密钥（默认从环境变量读取）
            base_url: API 基础 URL
        """
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.base_url = base_url  # 保存 base_url 参数

        # 获取 API 密钥
        self.api_key = api_key or os.getenv("ANTHROPIC_AUTH_TOKEN")
        if not self.api_key:
            raise ValueError("请设置 ANTHROPIC_AUTH_TOKEN 环境变量或传入 api_key 参数")

        # 初始化评估 LLM
        self._init_llm()

        # 初始化嵌入模型
        self._init_embeddings()

        # 选择评估指标（使用实例化的类）
        self.metrics = [Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()]

        logger.info(f"RagasEvaluator 初始化完成，LLM: {llm_model}, Embedding: {embedding_model}")

    def _init_llm(self):
        """初始化评估用的 LLM"""
        chat_model = ChatOpenAI(
            model=self.llm_model,
            base_url=self.base_url,
            api_key=self.api_key,
            temperature=0.0  # 评估时使用确定性输出
        )
        self.evaluator_llm = LangchainLLMWrapper(chat_model)

    def _init_embeddings(self):
        """初始化嵌入模型"""
        embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings)  # RAGAS 期望的接口（内部调用，这样设计解耦，不依赖具体的langchain或者其他框架）

    def prepare_evaluation_dataset(self,
                                    test_data: List[Dict],
                                    rag_responses: List[Dict]) -> EvaluationDataset:
        """
        准备 RAGAS 评估数据集

        Args:
            test_data: 测试数据列表，包含 question 和 ground_truth
            rag_responses: RAG 系统响应列表，包含 contexts 和 answer

        Returns:
            RAGAS EvaluationDataset 对象
        """
        from ragas.dataset_schema import SingleTurnSample

        samples = []

        for i, (test_item, response) in enumerate(zip(test_data, rag_responses)):
            try:
                sample = SingleTurnSample(
                    user_input=test_item.get('question', ''),
                    reference=test_item.get('ground_truth', ''),
                    response=response.get('answer', ''),
                    retrieved_contexts=response.get('contexts', [])
                )
                samples.append(sample)
            except Exception as e:
                logger.warning(f"样本 {i} 数据准备失败: {e}")

        return EvaluationDataset(samples=samples)

    def run_evaluation(self, dataset: EvaluationDataset) -> Dict[str, Any]:
        """
        运行 RAGAS 评估

        Args:
            dataset: RAGAS 评估数据集

        Returns:
            评估结果字典
        """
        logger.info(f"开始 RAGAS 评估，样本数量: {len(dataset.samples)}")

        try:
            results = evaluate(
                dataset=dataset,
                metrics=self.metrics,
                llm=self.evaluator_llm,
                embeddings=self.evaluator_embeddings
            )

            # 提取分数 - 使用 to_pandas() 方法
            scores = {}
            try:
                df = results.to_pandas()
                # 从 DataFrame 中计算各指标的平均分数
                for metric in self.metrics:
                    metric_name = metric.name
                    if metric_name in df.columns:
                        # 计算平均值，排除 NaN
                        valid_values = df[metric_name].dropna()
                        if len(valid_values) > 0:
                            scores[metric_name] = float(valid_values.mean())
                        else:
                            scores[metric_name] = 0.0
                    else:
                        scores[metric_name] = 0.0
            except Exception as e:
                logger.warning(f"从 DataFrame 提取分数失败: {e}")
                # 备用方案：从 results 对象直接提取
                for metric in self.metrics:
                    metric_name = metric.name
                    scores[metric_name] = 0.0

            # 计算综合分数
            if scores:
                avg_score = sum(scores.values()) / len(scores)
                scores['average'] = avg_score

            logger.info(f"RAGAS 评估完成，综合分数: {scores.get('average', 0):.4f}")

            return {
                'scores': scores,
                'details': results,
                'num_samples': len(dataset.samples)
            }

        except Exception as e:
            logger.error(f"RAGAS 评估失败: {e}")
            raise

    def generate_report(self, results: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """
        生成评估报告

        Args:
            results: 评估结果
            output_path: 报告输出路径（可选）

        Returns:
            报告文本
        """
        scores = results.get('scores', {})
        num_samples = results.get('num_samples', 0)

        report_lines = [
            "=" * 50,
            "RAGAS 评估报告",
            "=" * 50,
            f"评估样本数量: {num_samples}",
            "",
            "各指标得分:",
            "-" * 30,
        ]

        # 指标说明和分数
        metric_explanations = {
            'faithfulness': '答案忠实度 - 答案是否忠于检索上下文',
            'answer_relevance': '答案相关性 - 答案是否回应了用户问题',
            'context_precision': '上下文精确度 - 检索内容是否精简无噪声',
            'context_recall': '上下文召回率 - 检索内容是否覆盖 ground truth',
        }

        for metric_name, score in scores.items():
            if metric_name != 'average':
                explanation = metric_explanations.get(metric_name, metric_name)
                report_lines.append(f"{metric_name}: {score:.4f} ({explanation})")

        report_lines.extend([
            "",
            "-" * 30,
            f"综合平均分数: {scores.get('average', 0):.4f}",
            "",
            "评估标准参考:",
            "- Faithfulness > 0.7: 良好",
            "- Answer Relevance > 0.8: 良好",
            "- Context Precision > 0.6: 可接受",
            "- Context Recall > 0.5: 基础目标",
            "",
            "=" * 50,
        ])

        report = "\n".join(report_lines)

        # 保存报告
        if output_path:
            Path(output_path).write_text(report, encoding='utf-8')
            logger.info(f"评估报告已保存到: {output_path}")

        return report

    def save_results(self, results: Dict[str, Any], output_path: str):
        """保存评估结果到 JSON 文件"""
        output_data = {
            'scores': results.get('scores', {}),
            'num_samples': results.get('num_samples', 0),
            'timestamp': results.get('timestamp', ''),
        }

        Path(output_path).write_text(
            json.dumps(output_data, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )
        logger.info(f"评估结果已保存到: {output_path}")


def load_test_dataset(file_path: str = "evaluation/test_dataset.json") -> List[Dict]:
    """加载测试数据集"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f"加载测试数据集: {len(data)} 个问题")
    return data


def save_rag_responses(responses: List[Dict], output_path: str = "evaluation/rag_responses.json"):
    """保存 RAG 系统响应"""
    Path(output_path).write_text(
        json.dumps(responses, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )
    logger.info(f"RAG 响应已保存到: {output_path}")