# execution_agent.py

from pathlib import Path
from providers import DeepSeekLLMProvider
import subprocess
import sys


class ExecutionAgent:
    """
    一个简化的“虚拟 AI 参赛者”。
    它的任务是为给定的 MLE 任务生成一段解题代码并执行它。
    """

    def __init__(self, llm_provider: DeepSeekLLMProvider):
        self.llm = llm_provider
        print("ExecutionAgent 初始化成功。")

    def _extract_python_code(self, raw_response: str) -> str:
        # (这个辅助函数与 RefactorAgent 中的相同)
        code_block_start = raw_response.find("```python")
        if code_block_start != -1:
            code_block_end = raw_response.find("```", code_block_start + len("```python"))
            if code_block_end != -1:
                return raw_response[code_block_start + len("```python"):code_block_end].strip()
        code_block_start = raw_response.find("```")
        if code_block_start != -1:
            code_block_end = raw_response.find("```", code_block_start + 3)
            if code_block_end != -1:
                return raw_response[code_block_start + 3:code_block_end].strip()
        return raw_response.strip()

    def solve_task(self, comp_path: Path, submission_output_path: Path) -> bool:
        """
        读取任务描述，生成并执行解题代码，最终产出 submission.csv。
        """
        print(f"\n>>> ExecutionAgent: 开始尝试解决任务: {comp_path.name}")

        description = (comp_path / "description.txt").read_text(encoding='utf-8')
        # 为了让 Agent 知道数据在哪里，我们需要提供 public 文件夹的路径
        public_data_path = comp_path / "data" / "public"

        context = f"""
        任务描述:
        ---
        {description}
        ---

        可用的数据文件位于以下目录中 (代码中请直接使用这些相对路径):
        - 训练数据: 'train.csv'
        - 需要预测的数据: 'test.csv'
        - 提交格式示例: 'sample_submission.csv'

        所有数据文件都在这个路径下：'{public_data_path.resolve()}'
        """

        prompt = """
               你是一名初级数据科学家。你的任务是编写一段 Python 脚本的核心逻辑来解决一个 Kaggle 竞赛。

               你的脚本必须严格遵循以下规则：
               1.  使用 `pandas` 读取位于当前工作目录下的 `'train.csv'` 和 `'test.csv'`。
               2.  使用 `sklearn` 的基础模型和预处理器（`SimpleImputer`, `OneHotEncoder`）。
               3.  [重要!] 你的代码**不应该**包含文件保存的逻辑 (`to_csv`)。
               4.  你的代码的最后一步，应该是**生成一个名为 `submission_df` 的最终 Pandas DataFrame**。这个 DataFrame 必须包含 'Id' 列和预测结果列。

               你的脚本逻辑应该是：
               - 导入库。
               - 读取 `train.csv`, `test.csv`。
               - 预处理数据，生成干净的 `X_train` 和 `X_test`。
               - 训练模型。
               - 预测。
               - `submission_df = pd.DataFrame({'Id': test_df['Id'], 'SalePrice': predictions})` (这里的 'SalePrice' 应根据任务动态变化)

               请直接返回完整的 Python 代码块，不要有任何其他解释。
               """

        raw_code = self.llm.query(prompt, context)
        solve_script_core_logic = self._extract_python_code(raw_code)

        # 准备执行代码
        full_exec_code = (
            f"{solve_script_core_logic}\n\n"
            f"# --- 由系统注入的文件保存逻辑 ---\n"
            f"output_path = r'{submission_output_path.resolve()}'\n"
            f"submission_df.to_csv(output_path, index=False)\n"
            f"print(f'Submission file created successfully at {{output_path}}')\n"
        )

        print(">>> ExecutionAgent: 已生成解题代码，准备执行...")

        try:
            python_executable = sys.executable
            result = subprocess.run(
                [python_executable, "-c", full_exec_code],
                check=True,
                capture_output=True,
                text=True,
                cwd=public_data_path
            )
            print(f">>> ExecutionAgent: 解题代码执行成功！")
            print(f"  Stdout: {result.stdout.strip()}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"!!! ExecutionAgent: 解题代码执行失败: {e}")
            print(f"  Stderr: {e.stderr}")
            return False