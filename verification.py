# verification.py

import ast
from pathlib import Path
from providers import DeepSeekLLMProvider
from execution_agent import ExecutionAgent
from scoring_judge import ScoringJudge
import shutil
import pandas as pd


class Verifier:
    def __init__(self, llm_provider: DeepSeekLLMProvider):
        print("Verifier 初始化成功")
        self.llm = llm_provider
        self.execution_agent = ExecutionAgent(llm_provider)
        self.scoring_judge = ScoringJudge()
        self.errors = []
        self.warnings = []

    def _reset(self):
        self.errors = []
        self.warnings = []

    # --- 第一阶段: 断言检查 (Assertions) ---
    def _check_directory_layout(self, comp_path: Path):
        required_dirs = [comp_path / "data", comp_path / "data" / "raw", comp_path / "data" / "public",
                         comp_path / "data" / "private"]
        for d in required_dirs:
            if not d.is_dir(): self.errors.append(f"结构错误：缺少目录 {d.relative_to(comp_path)}")

    def _check_file_existence(self, comp_path: Path):
        required_files = [comp_path / "prepare.py", comp_path / "metric.py", comp_path / "description.txt",
                          comp_path / "data" / "public" / "train.csv", comp_path / "data" / "public" / "test.csv",
                          comp_path / "data" / "public" / "sample_submission.csv",
                          comp_path / "data" / "private" / "test_answer.csv"]
        for f in required_files:
            if not f.is_file(): self.errors.append(f"文件错误：缺少文件 {f.relative_to(comp_path)}")

    def _check_prepare_py_interface(self, comp_path: Path):
        prepare_script_path = comp_path / "prepare.py"
        if not prepare_script_path.exists(): return
        try:
            with open(prepare_script_path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read())
            prepare_found = False
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == "prepare":
                    prepare_found = True
                    if len(node.args.args) != 3: self.errors.append(
                        f"接口错误 (prepare.py): 'prepare' 函数应有 3 个参数，但找到了 {len(node.args.args)} 个。")
                    break
            if not prepare_found: self.errors.append("接口错误 (prepare.py): 找不到名为 'prepare' 的函数。")
        except Exception as e:
            self.errors.append(f"AST分析错误 (prepare.py): {e}")

    def _check_metric_py_interface(self, comp_path: Path):
        metric_script_path = comp_path / "metric.py"
        if not metric_script_path.exists(): return
        try:
            with open(metric_script_path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read())
            inherits_base_metric = False
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    if any(isinstance(base, ast.Name) and base.id == 'BaseMetric' for base in node.bases):
                        inherits_base_metric = True
                        break
            if not inherits_base_metric: self.errors.append("接口错误 (metric.py): 找不到任何继承自 'BaseMetric' 的类。")
        except Exception as e:
            self.errors.append(f"AST分析错误 (metric.py): {e}")

    def run_assertions(self, refactored_comp_path: Path) -> bool:
        self._reset()
        print(f"\n>>> Verifier (Stage 1): 开始对 {refactored_comp_path.name} 进行断言检查...")
        if not refactored_comp_path.is_dir(): return False
        self._check_directory_layout(refactored_comp_path)
        self._check_file_existence(refactored_comp_path)
        self._check_prepare_py_interface(refactored_comp_path)
        self._check_metric_py_interface(refactored_comp_path)
        if not self.errors:
            print(f">>> Verifier (Stage 1): [成功] 所有断言检查通过！")
            return True
        else:
            print(f"!!! Verifier (Stage 1): [失败] 发现 {len(self.errors)} 个断言错误:")
            for error in self.errors: print(f"  - {error}")
            return False

    # --- 第二阶段: 语义审查 (Semantic Reviews) - Prompt 优化 ---

    def _review_description(self, comp_path: Path):
        description_path = comp_path / "description.txt"
        if not description_path.exists(): return
        description_content = description_path.read_text(encoding='utf-8')

        prompt = """
        你是一名经验丰富的 Kaggle 竞赛评审员。你的任务是审查以下任务描述的语义质量。

        请检查以下几点，并给出评级：
        1.  **清晰度与完整性**：描述是否清晰、完整？如果缺少一些次要信息，评级为 WARN。
        2.  **特征合理性**：描述中提到的特征是否可能导致“捷径解”？例如，使用高度相关的特征（如车库面积预测车库容量）会让任务变得简单，这应该评级为 WARN，因为它不是一个致命错误。
        3.  **答案泄露 (致命错误)**：描述中是否【明确】泄露了测试集的答案或可以【直接】推导出答案的信息（例如，将测试集的目标值本身作为特征）？这是致命错误，必须评级为 FAIL。

        审查完成后，你必须严格按照以下格式之一进行回复：
        - 如果没有任何问题，只回复：`OK`
        - 如果存在一些非致命的问题（如指标不完美、特征高度相关），只回复：`WARN: [这里是你发现的具体问题]`
        - 如果存在致命的答案泄露问题，只回复：`FAIL: [这里是你发现的具体问题]`
        """

        response = self.llm.query(prompt, description_content).strip()
        if response.upper().startswith("FAIL"):
            self.errors.append(f"语义审查-致命错误 (description.txt): {response}")
        elif response.upper().startswith("WARN"):
            self.warnings.append(f"语义审查-警告 (description.txt): {response}")

    def _review_metric_appropriateness(self, comp_path: Path):
        metric_script_path = comp_path / "metric.py"
        description_path = comp_path / "description.txt"
        if not metric_script_path.exists() or not description_path.exists(): return

        context = f"任务描述:\n---\n{description_path.read_text(encoding='utf-8')}\n---\n\n评估指标代码:\n---\n{metric_script_path.read_text(encoding='utf-8')}\n---"

        # [Prompt 优化] 同样引入三级评级，放宽对指标选择的要求
        prompt = """
        你是一名数据科学专家。你的任务是判断下面提供的评估指标，是否适合其对应的任务。

        请遵循以下原则：
        - **合适的指标**：如果指标是行业标准且没有明显缺陷，评级为 OK。
        - **不够完美的指标**：如果指标能用，但不是最优选择（例如，在不均衡分类任务中使用 Accuracy），这是一个常见的设计缺陷，但不致命。请评级为 WARN。
        - **完全错误的指标**：如果指标与任务类型完全不匹配（例如，为分类任务使用 RMSE），这是一个致命错误。请评级为 FAIL。

        审查完成后，你必须严格按照以下格式之一进行回复：
        - 如果你认为指标是合适的，只回复：`OK`
        - 如果你认为指标不够完美但可用，只回复：`WARN: [这里是你认为不够完美的理由]`
        - 如果你认为指标完全错误，只回复：`FAIL: [这里是你认为完全错误的理由]`
        """

        response = self.llm.query(prompt, context).strip()
        if response.upper().startswith("FAIL"):
            self.errors.append(f"语义审查-致命错误 (Metric): {response}")
        elif response.upper().startswith("WARN"):
            self.warnings.append(f"语义审查-警告 (Metric): {response}")

    def run_semantic_reviews(self, refactored_comp_path: Path) -> bool:
        self._reset()
        print(f"\n>>> Verifier (Stage 2): 开始对 {refactored_comp_path.name} 进行语义审查...")

        self._review_description(refactored_comp_path)
        self._review_metric_appropriateness(refactored_comp_path)

        # 只有当存在 errors 时，才算失败
        if self.warnings:
            print(f">>> Verifier (Stage 2): [警告] 发现 {len(self.warnings)} 个非致命问题:")
            for warning in self.warnings:
                print(f"  - {warning}")

        if not self.errors:
            print(f">>> Verifier (Stage 2): [通过] 没有发现致命的语义错误。")
            return True
        else:
            print(f"!!! Verifier (Stage 2): [失败] 发现 {len(self.errors)} 个致命语义错误:")
            for error in self.errors:
                print(f"  - {error}")
            return False

    def run_execution_validation(self, refactored_comp_path: Path) -> bool:
            """
            [第三阶段] 模拟 AI 参赛者解决任务，并进行评分，以验证任务的可解性。
            """
            self._reset()
            print(f"\n>>> Verifier (Stage 3): 开始对 {refactored_comp_path.name} 进行执行验证...")

            # 1. 流程验证：让 ExecutionAgent 尝试解决任务并生成提交文件
            submission_output_dir = refactored_comp_path / "temp_submission"
            submission_output_dir.mkdir(exist_ok=True)
            submission_file = submission_output_dir / "submission.csv"

            solve_success = self.execution_agent.solve_task(refactored_comp_path, submission_file)

            if not solve_success or not submission_file.exists():
                self.errors.append("执行验证-流程失败: ExecutionAgent 未能成功生成提交文件。")

            else:
                # 2. 性能验证：为 Agent 的提交评分
                agent_score = self.scoring_judge.score(refactored_comp_path, submission_file)

                if agent_score == -1.0:  # 表示评分过程出错
                    self.errors.append("执行验证-流程失败: ScoringJudge 在评分过程中发生错误。")
                else:
                    # 在这里，我们可以实现一个与“随机猜测”基线的对比
                    # 为了简化，我们暂时只检查分数是否是一个有效的数值
                    # (一个更复杂的实现会计算一个 baseline_score)
                    if pd.isna(agent_score):
                        self.errors.append(f"执行验证-性能失败: Agent 得分无效 (NaN)。")
                    else:
                        print(f">>> Verifier (Stage 3): 虚拟参赛者得分为 {agent_score:.6f}。")

            # 清理临时文件
            shutil.rmtree(submission_output_dir)

            if not self.errors:
                print(f">>> Verifier (Stage 3): [成功] 执行验证通过！")
                return True
            else:
                print(f"!!! Verifier (Stage 3): [失败] 发现 {len(self.errors)} 个执行问题:")
                for error in self.errors:
                    print(f"  - {error}")
                return False