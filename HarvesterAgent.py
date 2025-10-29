

import json
import pandas as pd
from pathlib import Path
from typing import List
import requests
import pdfplumber

# 从项目模块中导入
from providers import DeepSeekLLMProvider


class HarvesterAgent:
    """
    一个先进的代理，能够从URL或PDF文件中自动提取、筛选和清洗表格数据，
    为下游的机器学习任务生成做准备。
    """

    def __init__(self, llm_provider: DeepSeekLLMProvider, output_dir: Path):
        self.llm = llm_provider
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _extract_tables_from_url(self, url: str) -> List[pd.DataFrame]:
        try:
            print(f">>> Harvester: 正在从 URL 提取表格: {url}")
            # 添加 User-Agent 头来模拟浏览器，避免被一些网站阻止
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
            tables = pd.read_html(requests.get(url, headers=headers).text)
            print(f">>> Harvester: 成功提取到 {len(tables)} 个表格。")
            return [tbl.head(5) for tbl in tables if not tbl.empty]
        except Exception as e:
            print(f"!!! Harvester: 从 URL 提取表格失败: {e}")
            return []

    def _extract_tables_from_pdf(self, pdf_path: Path) -> List[pd.DataFrame]:
        try:
            print(f">>> Harvester: 正在从 PDF 提取表格: {pdf_path}")
            tables = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_tables()
                    for tbl_data in extracted:
                        if tbl_data and len(tbl_data) > 1:
                            df = pd.DataFrame(tbl_data[1:], columns=tbl_data[0])
                            tables.append(df.head(5))
            print(f">>> Harvester: 成功提取到 {len(tables)} 个表格。")
            return tables
        except Exception as e:
            print(f"!!! Harvester: 从 PDF 提取表格失败: {e}")
            return []

    def harvest_from_source(self, source: str) -> Path:
        """
        主入口函数，处理URL或本地PDF文件。
        返回生成的 train.csv 文件路径，如果失败则返回 None。
        """
        if source.startswith('http'):
            raw_tables = self._extract_tables_from_url(source)
        elif Path(source).is_file() and Path(source).suffix.lower() == '.pdf':
            raw_tables = self._extract_tables_from_pdf(Path(source))
        else:
            print(f"!!! Harvester: 不支持的源格式: {source}")
            return None

        if not raw_tables:
            print("!!! Harvester: 未能从源中提取到任何表格。")
            return None

        print("\n>>> Harvester: 请求 LLM 筛选最有价值的表格...")
        tables_as_markdown = "\n\n".join(
            [f"--- 表格 {i + 1} ---\n{tbl.to_markdown()}" for i, tbl in enumerate(raw_tables)])
        selection_prompt = (
            "你是一名顶尖的数据科学家顾问。你的任务是从用户提供的多个原始表格中，"
            "挑选出唯一一个最适合用于创建有意义的机器学习任务的表格。\n\n"
            "你必须严格遵循以下JSON格式返回你的决策，不要有任何其他解释：\n"
            '{"best_table_index": [一个整数, 代表你选择的表格索引, 从1开始计数], '
            '"justification": "[一段简短的文字, 解释你为什么选择这个表格，以及它可以用来做什么样的ML任务]"}'
        )
        context = f"以下是从一份科学文档中提取的所有表格：\n{tables_as_markdown}"
        llm_response = self.llm.query(selection_prompt, context)

        try:
            selection = json.loads(llm_response)
            best_table_index = selection['best_table_index'] - 1
            print(f">>> Harvester: LLM 决策完成。选择了表格 {best_table_index + 1}。")
            print(f"    理由: {selection['justification']}")
            selected_table = raw_tables[best_table_index]
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"!!! Harvester: 解析LLM的表格选择时出错: {e}\n--- 原始响应 ---\n{llm_response}")
            return None

        print("\n>>> Harvester: 请求 LLM 清洗并重构选定的表格...")
        cleaning_prompt = (
            "你是一名专业的数据工程师。你的任务是接收一个原始的、可能很凌乱的表格数据，"
            "然后执行以下操作：\n"
            "1.  **清理数据**: 移除任何空值、注释符号或不相关的文本。\n"
            "2.  **重构列名**: 将原始的列名重命名为对机器学习友好的、具有描述性的蛇形命名法（snake_case）名称。\n"
            "3.  **格式化输出**: 必须以一个包含JSON对象的列表形式返回清洗后的数据。不要返回任何其他解释或文字。\n"
        )
        context = f"请处理以下表格：\n{selected_table.to_markdown()}"
        llm_response_clean = self.llm.query(cleaning_prompt, context)

        try:
            json_start = llm_response_clean.find('[')
            json_end = llm_response_clean.rfind(']') + 1
            cleaned_data = json.loads(llm_response_clean[json_start:json_end])
            final_df = pd.DataFrame(cleaned_data)
        except (json.JSONDecodeError, TypeError) as e:
            print(f"!!! Harvester: 解析LLM的清洗结果时出错: {e}\n--- 原始响应 ---\n{llm_response_clean}")
            return None

        final_df.insert(0, 'Id', range(len(final_df)))
        output_file = self.output_dir / "train.csv"
        final_df.to_csv(output_file, index=False)
        print(f"\n>>> Harvester: 采集和处理完成！最终的干净数据已保存到: {output_file}")
        print("--- 生成的CSV内容 ---")
        print(final_df.to_string())
        print("--------------------")

        return output_file