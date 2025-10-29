import os
import io
import json
import shutil
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import List
import shutil
import sys
import subprocess
import pdfplumber
from providers import DeepSeekLLMProvider
from verification import Verifier
from HarvesterAgent import HarvesterAgent


# ==========================================================
# 1. 结构化容器
# ==========================================================
@dataclass
class TaskProposal:

    prediction_target: str
    evaluation_metric: str
    data_utilization: str
    justification: str
    difficulty: str = "中等"
    tags: List[str] = None
    feature_columns: List[str] = None
    task_type: str = "回归"

    def pretty_print(self):
        print(f"--- 任务构想: 预测 '{self.prediction_target}' ({self.task_type}) ---")


# ==========================================================
# 2. LLM Provider
# ==========================================================



# ==========================================================
# 3. Brainstormer Agent
# ==========================================================
class BrainstormerAgent:
    def __init__(self, llm_provider: DeepSeekLLMProvider):
        self.llm = llm_provider

    def explore_dataset(self, dataset_path: Path) -> str:
        print(f"\n>>> Brainstormer: 正在探索数据集文件: {dataset_path}...")
        try:
            df = pd.read_csv(dataset_path)
            columns_list = df.columns.tolist()
            buffer = io.StringIO()
            df.info(buf=buffer)
            info_str = buffer.getvalue()
            summary = (f"数据集探索报告\n=======================\n所有列名: {columns_list}\n"
                       f"数据维度: {df.shape[0]} 行, {df.shape[1]} 列\n\n--- 数据样本 (前5行) ---\n{df.head().to_string()}\n\n"
                       f"--- 列信息 ---\n{info_str}\n\n--- 数值列统计摘要 ---\n{df.describe().to_string()}")
            print(">>> Brainstormer: 数据集探索完成。")
            return summary
        except Exception as e:
            return f"探索数据集时出错: {e}"

    def brainstorm_tasks(self, dataset_path: Path) -> List[TaskProposal]:
        dataset_summary = self.explore_dataset(dataset_path)
        if dataset_summary.startswith("错误"): return []
        prompt = ("你是一位顶级的 Kaggle 竞赛设计大师。基于用户提供的数据集探索报告，你的任务是构思出 3 到 5 个多样化且高质量的机器学习任务。\n\n"
                  "你必须严格遵循以下规则：\n1. 你必须以一个包含 JSON 对象的列表形式返回你的构思，不要有任何其他多余的文字或解释。\n"
                  "2. 每个 JSON 对象必须包含以下键：'prediction_target', 'evaluation_metric', 'task_type', 'feature_columns', 'data_utilization', 'justification'。\n"
                  "3. 'prediction_target' 必须是数据集中的一个列名。\n4. 'task_type' 必须是 '回归' 或 '分类'。\n"
                  "5. 'feature_columns' 必须是一个包含数据集中真实列名的列表。")
        llm_response = self.llm.query(prompt, dataset_summary)
        try:
            json_start = llm_response.find('[')
            json_end = llm_response.rfind(']') + 1
            if json_start == -1 or json_end == 0: raise json.JSONDecodeError("Not a valid JSON list", llm_response, 0)
            proposals_json = json.loads(llm_response[json_start:json_end])
            task_proposals = [TaskProposal(**p) for p in proposals_json]
            print(f"\n>>> Brainstormer: 成功解析出 {len(task_proposals)} 个任务构思。")
            return task_proposals
        except (json.JSONDecodeError, TypeError) as e:
            print(
                f"!!! Brainstormer 解析 JSON 时出错: {e} !!!\n--- 原始 LLM 响应 ---\n{llm_response}\n----------------------")
            return []


# ==========================================================
# 4. Designer Agent
# ==========================================================
class DesignerAgent:
    def __init__(self, llm_provider: DeepSeekLLMProvider):
        self.llm = llm_provider

    def design_task_package(self, proposal: TaskProposal, task_package_path: Path):
        print(f"\n>>> Designer: 开始为任务 '{proposal.prediction_target}' 创建临时任务包...")
        task_package_path.mkdir(exist_ok=True)
        context = (
            f"任务构思详情:\n- 预测目标 (Target): {proposal.prediction_target}\n- 任务类型 (Task Type): {proposal.task_type}\n"
            f"- 评估指标 (Metric): {proposal.evaluation_metric}\n- 使用的特征 (Features): {proposal.feature_columns}")
        prompt = """
                你是一个全栈机器学习工程师。基于用户提供的任务构思详情，生成一个 Kaggle 风格的任务文件包。

                你必须严格以一个 JSON 对象的格式返回所有文件的内容，不要有任何其他多余的文字或解释。
                这个 JSON 对象必须包含以下四个键：'description.txt', 'prepare.py', 'metric.py', 'sample_submission.csv'。

                详细要求:
                1.  `description.txt`: 撰写专业的任务描述。
                2.  `prepare.py`:
                    - 必须 `import os` 和 `pandas as pd`。
                    - 必须实现 `prepare_data(raw_data_path, output_dir)` 函数。
                    - [重要!] 在函数内，必须使用 `os.path.join(raw_data_path, 'train.csv')` 来构建完整的文件路径进行读取。
                    - 使用 `train_test_split` (random_state=42) 切分数据。
                    - 将 'train.csv', 'test.csv', 'test_solution.csv' 保存到 `output_dir`。'test_solution.csv' 必须包含 'Id' 列。
                3.  `metric.py`: 实现 `calculate_metric(solution_path, submission_path)` 函数。
                4.  `sample_submission.csv`: 创建包含 'Id' 和目标列的示例文件。
                """
        llm_response = self.llm.query(prompt, context)
        try:
            json_start = llm_response.find('{')
            json_end = llm_response.rfind('}') + 1
            if json_start == -1 or json_end == 0: raise json.JSONDecodeError("Not a valid JSON object", llm_response, 0)
            files_content = json.loads(llm_response[json_start:json_end])
            for filename, content in files_content.items():
                (task_package_path / filename).write_text(content, encoding='utf-8')
            print(f">>> Designer: 临时任务包创建成功: {task_package_path}")
            return True
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            print(
                f"!!! Designer 解析或写入文件时出错: {e} !!!\n--- 原始 LLM 响应 ---\n{llm_response}\n----------------------")
            return False


# ==========================================================
# 5. Refactor Agent
# ==========================================================
class RefactorAgent:
    def __init__(self, llm_provider: DeepSeekLLMProvider, templates_path: Path):
        self.llm = llm_provider
        self.templates_path = templates_path

    def _extract_python_code(self, raw_response: str) -> str:
        """
        从 LLM 可能返回的 Markdown 格式响应中，提取纯净的 Python 代码。
        """
        # 寻找 python 代码块的开始标记
        code_block_start = raw_response.find("```python")
        if code_block_start != -1:
            # 如果找到了，再寻找结束标记
            code_block_end = raw_response.find("```", code_block_start + len("```python"))
            if code_block_end != -1:
                # 提取并返回代码块内部的内容，并去除首尾空白
                code = raw_response[code_block_start + len("```python"):code_block_end].strip()
                return code

        # 如果没有找到 ```python 块，尝试寻找普通的 ``` 块
        code_block_start = raw_response.find("```")
        if code_block_start != -1:
            code_block_end = raw_response.find("```", code_block_start + 3)
            if code_block_end != -1:
                code = raw_response[code_block_start + 3:code_block_end].strip()
                return code

        # 如果连 ``` 都没有找到，我们假设整个响应就是代码 (作为最后的备用方案)
        return raw_response.strip()
    def refactor_task_package(self, designer_package_path: Path, refactored_comp_path: Path, raw_dataset_path: Path):
        print(f"\n>>> Refactor: 开始重构任务包: {designer_package_path.name}")

        # 1. 创建标准的竞赛目录结构
        data_dir = refactored_comp_path / "data"
        raw_dir = data_dir / "raw"
        public_dir = data_dir / "public"
        private_dir = data_dir / "private"
        for d in [refactored_comp_path, data_dir, raw_dir, public_dir, private_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # 2. 将原始数据集复制到 raw/ 目录
        shutil.copy(raw_dataset_path, raw_dir / raw_dataset_path.name)
        print(f">>> Refactor: 已将原始数据复制到 {raw_dir}")

        # 3.  执行 Designer 生成的 prepare.py 来生成切分好的数据
        designer_prepare_script = designer_package_path / "prepare.py"
        temp_output_dir = designer_package_path / "temp_output"
        temp_output_dir.mkdir(exist_ok=True)
        python_executable = sys.executable
        try:
            prep_code = designer_prepare_script.read_text(encoding='utf-8')
            exec_code = (f"{prep_code}\n"
                         f"prepare_data(r'{raw_dir}', r'{temp_output_dir}')")

            # 使用我们虚拟环境的 python.exe 来执行代码
            subprocess.run([python_executable, "-c", exec_code], check=True, capture_output=True, text=True)

            # 4. 将生成的文件移动到标准目录
            shutil.move(temp_output_dir / "train.csv", public_dir / "train.csv")
            shutil.move(temp_output_dir / "test.csv", public_dir / "test.csv")
            shutil.move(temp_output_dir / "test_solution.csv", private_dir / "test_answer.csv")
            shutil.move(designer_package_path / "sample_submission.csv", public_dir / "sample_submission.csv")
            shutil.move(designer_package_path / "description.txt", refactored_comp_path / "description.txt")
            print(f">>> Refactor: 已成功执行数据切分并移动文件到标准目录。")

        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"!!! Refactor: 执行或移动 Designer 的 prepare.py 时出错: {e}")
            if isinstance(e, subprocess.CalledProcessError): print(f"  Stderr: {e.stderr}")
            shutil.rmtree(refactored_comp_path)  # 出错则清理
            return False
        finally:
            shutil.rmtree(temp_output_dir)  # 清理临时输出目录

        # 重写 prepare.py
        original_prepare_code = designer_prepare_script.read_text(encoding='utf-8')
        prepare_template = (self.templates_path / "prepare_template.py").read_text(encoding='utf-8')
        prepare_rewrite_prompt = f"""
        你是一个代码重构专家。你的任务是将用户提供的 Python 脚本重构成符合指定模板和函数签名的标准格式。

        这是标准的模板，你必须遵循它的结构和函数签名：
        --- TEMPLATE START ---
        {prepare_template}
        --- TEMPLATE END ---

        这是用户提供的原始代码，你需要理解其核心逻辑（如读取什么文件，如何切分等），然后将其逻辑填充到标准模板中：
        --- ORIGINAL CODE START ---
        {original_prepare_code}
        --- ORIGINAL CODE END ---

        请直接返回重构后的、完整的、可直接运行的 Python 代码，不要有任何其他解释。
        """
        raw_refactored_prepare_code = self.llm.query(prepare_rewrite_prompt)
        # [关键改动] 在写入文件前，先调用净化函数
        refactored_prepare_code = self._extract_python_code(raw_refactored_prepare_code)
        (refactored_comp_path / "prepare.py").write_text(refactored_prepare_code, encoding='utf-8')

        # 重写 metric.py
        original_metric_code = (designer_package_path / "metric.py").read_text(encoding='utf-8')
        base_metric_template = (self.templates_path / "base_metric.py").read_text(encoding='utf-8')
        metric_rewrite_prompt = f"""
        你是一个代码重构专家。你的任务是将用户提供的 Python 脚本重构成一个继承自指定基类的标准类。

        这是标准的基类模板，你必须继承它并实现其抽象方法：
        --- TEMPLATE START ---
        {base_metric_template}
        --- TEMPLATE END ---

        这是用户提供的原始代码，你需要理解其核心评估逻辑（如使用了什么 sklearn 函数），然后将其逻辑整合到标准模板的 `evaluate` 方法中：
        --- ORIGINAL CODE START ---
        {original_metric_code}
        --- ORIGINAL CODE END ---

        请直接返回重构后的、完整的、可直接运行的 Python 代码，不要有任何其他解释。确保类名是唯一的，例如 'MyCompetitionMetric'。
        """
        raw_refactored_metric_code = self.llm.query(metric_rewrite_prompt)
        refactored_metric_code = self._extract_python_code(raw_refactored_metric_code)
        (refactored_comp_path / "metric.py").write_text(refactored_metric_code, encoding='utf-8')

        print(f">>> Refactor: 已成功重写 prepare.py 和 metric.py。")
        print(f">>> Refactor: 重构完成！标准化的竞赛包位于: {refactored_comp_path}")
        return True


# ==========================================================
# 6. 主流水线
# ==========================================================
class MLESmithPipeline:
    def __init__(self, workspace_dir: str = "./mle_smith_workspace"):
        self.workspace_path = Path(workspace_dir)
        self.raw_data_path = self.workspace_path / "raw_data"
        self.designer_output_path = self.workspace_path / "designer_output"
        self.refactor_output_path = self.workspace_path / "refactor_output"  # 最终输出
        self.templates_path = self.workspace_path / "templates"

        # 初始化 Agents
        self.llm_provider = DeepSeekLLMProvider(model="deepseek-chat")
        self.harvester = HarvesterAgent(self.llm_provider, self.raw_data_path)
        self.brainstormer = BrainstormerAgent(self.llm_provider)
        self.designer = DesignerAgent(self.llm_provider)
        self.refactor = RefactorAgent(self.llm_provider, self.templates_path)
        self.llm_provider = DeepSeekLLMProvider(model="deepseek-chat")
        self.verifier = Verifier(self.llm_provider)

        self.setup_workspace()

    def setup_workspace(self):
        """初始化所有需要的工作区目录和模板文件"""
        for p in [self.raw_data_path, self.templates_path]:
            p.mkdir(parents=True, exist_ok=True)

        # 清理之前运行的输出
        for p in [self.designer_output_path, self.refactor_output_path]:
            if p.exists(): shutil.rmtree(p)
            p.mkdir(parents=True, exist_ok=True)

        # 写入 Refactor 需要的模板文件
        (self.templates_path / "prepare_template.py").write_text("""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare(raw_dir: Path, public_dir: Path, private_dir: Path):
    # TODO: 实现核心的数据读取和切分逻辑
    # 例如:
    # raw_df = pd.read_csv(raw_dir / 'train.csv')
    # train_df, test_df = train_test_split(raw_df, test_size=0.2, random_state=42)
    # solution = test_df[['Id', 'TargetColumn']].copy()
    # test_df = test_df.drop(columns=['TargetColumn'])
    #
    # train_df.to_csv(public_dir / 'train.csv', index=False)
    # test_df.to_csv(public_dir / 'test.csv', index=False)
    # solution.to_csv(private_dir / 'test_answer.csv', index=False)
    print("Data preparation complete.")
""", encoding='utf-8')

        (self.templates_path / "base_metric.py").write_text("""
from abc import ABC, abstractmethod
import pandas as pd

class BaseMetric(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def evaluate(self, solution: pd.DataFrame, submission: pd.DataFrame) -> float:
        raise NotImplementedError
""", encoding='utf-8')
        print("工作区和模板文件初始化完成。")

    def run(self, source: str):
        print(f"\n===== 开始运行 MLE-Smith 完整流水线 (Brainstormer -> Designer -> Refactor) =====\n")
        dataset_file_path = self.harvester.harvest_from_source(source)
        if not dataset_file_path or not dataset_file_path.exists():
            print("Harvester 未能成功生成数据集，流水线终止。")
            return

        # --- 第一步: Brainstormer ---
        proposals = self.brainstormer.brainstorm_tasks(dataset_file_path)
        if not proposals:
            print("\nBrainstormer 未能生成任何任务构思，流水线终止。")
            return

        # --- 第二步: Designer ---
        for i, proposal in enumerate(proposals):
            proposal.pretty_print()
            designer_package_name = f"designer_task_{i + 1}"
            designer_package_path = self.designer_output_path / designer_package_name

            success = self.designer.design_task_package(proposal, designer_package_path)
            if not success:
                print(f"Designer 未能为构思 {i + 1} 创建任务包，跳过此构思。")
                continue

            # --- 第三步: Refactor ---
            refactored_comp_name = f"competition_{i + 1}_{proposal.prediction_target.lower()}"
            refactored_comp_path = self.refactor_output_path / refactored_comp_name
            self.refactor.refactor_task_package(designer_package_path, refactored_comp_path, dataset_file_path)

            # --- 第四步: Verifier
            assertion_passed = self.verifier.run_assertions(refactored_comp_path)

            if assertion_passed:
                print(f"--- 任务 '{refactored_comp_name}' 已通过断言检查。 ---")

                # 语义审查
                review_passed = self.verifier.run_semantic_reviews(refactored_comp_path)

                if review_passed:
                    print(f"--- 任务 '{refactored_comp_name}' 已通过语义审查。 ---")

                    # 执行验证
                    execution_passed = self.verifier.run_execution_validation(refactored_comp_path)

                    if execution_passed:
                        print(f"\n========================================================")
                        print(f" 任务 '{refactored_comp_name}' 已通过所有验证！")
                        print(f"========================================================")
                    else:
                        print(f"--- 任务 '{refactored_comp_name}' 未能通过执行验证，被视为无效任务。 ---")
                else:
                    print(f"--- 任务 '{refactored_comp_name}' 未能通过语义审查，被视为无效任务。 ---")

        print(f"\n===== MLE-Smith 流水线执行完毕！ =====")
        print(f"所有标准化的竞赛包都位于: {self.refactor_output_path}")


# ==========================================================
# 7. 主程序
# ==========================================================
if __name__ == "__main__":
    # 确保 API 密钥已设置
    os.environ["DEEPSEEK_API_KEY"] = "sk-353a88a777bd4c598f17b2923677e100"

    if "YOUR_API_KEY_HERE" in os.environ.get("DEEPSEEK_API_KEY", ""):
        print("\n警告：设置您的 DEEPSEEK_API_KEY。")
    else:
        try:
            pipeline = MLESmithPipeline()
            source_url = "../CNS/Towards large-scale quantum optimization solvers with few qubits.pdf"

            # 如果您已经下载了PDF，可以提供本地路径
            # source_pdf = "path/to/s41467-024-55346-z.pdf"

            pipeline.run(source=source_url)

        except Exception as e:
            print(f"\n程序运行时发生意外错误: {e}")
            import traceback

            traceback.print_exc()