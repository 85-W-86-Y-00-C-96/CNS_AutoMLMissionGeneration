# scoring_judge.py (升级版)

import pandas as pd
from pathlib import Path
import importlib.util
from abc import ABC, abstractmethod  # [关键改动] 我们在这里也导入 ABC


class ScoringJudge:
    """
    一个通用的“评分裁判”。
    负责加载指定任务的评估指标并为提交文件打分。
    """

    def __init__(self):
        print("ScoringJudge 初始化成功。")

    def score(self, comp_path: Path, submission_path: Path) -> float:
        """
        为给定的提交文件打分。
        """
        print(f"\n>>> ScoringJudge: 开始为 {submission_path.name} 评分...")

        solution_path = comp_path / "data" / "private" / "test_answer.csv"
        metric_script_path = comp_path / "metric.py"

        if not solution_path.exists() or not metric_script_path.exists():
            print("!!! ScoringJudge: 错误，找不到答案文件或评估脚本。")
            return -1.0

        try:
            solution_df = pd.read_csv(solution_path)
            submission_df = pd.read_csv(submission_path)

            # 动态加载 metric.py 模块
            spec = importlib.util.spec_from_file_location("metric_module", metric_script_path)
            metric_module = importlib.util.module_from_spec(spec)

            class BaseMetric(ABC):
                @abstractmethod
                def evaluate(self, solution: pd.DataFrame, submission: pd.DataFrame) -> float:
                    raise NotImplementedError

            metric_module.BaseMetric = BaseMetric
            spec.loader.exec_module(metric_module)

            # 找到并实例化继承自 BaseMetric 的类
            MetricClass = None
            for name, obj in metric_module.__dict__.items():
                if isinstance(obj, type) and issubclass(obj,
                                                        metric_module.BaseMetric) and obj is not metric_module.BaseMetric:
                    MetricClass = obj
                    break

            if not MetricClass:
                print("!!! ScoringJudge: 错误，在 metric.py 中找不到继承自 BaseMetric 的类。")
                return -1.0

            metric_instance = MetricClass()

            # 计算分数
            score = metric_instance.evaluate(solution_df, submission_df)

            print(f">>> ScoringJudge: 评分完成。最终分数为: {score:.6f}")
            return score

        except Exception as e:
            print(f"!!! ScoringJudge: 评分过程中发生错误: {e}")
            return -1.0