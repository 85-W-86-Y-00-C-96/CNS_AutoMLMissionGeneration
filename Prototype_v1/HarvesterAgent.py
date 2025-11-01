# File: harvester_agent.py
import json
import os

import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import requests
import pdfplumber
import re
import datetime
import zipfile # <--- 新增 import
import io      # <--- 新增 import

from providers import DeepSeekLLMProvider


class HarvesterAgent:
    """
    升级版代理，能够从URL或PDF中自动提取、评估和结构化多种资产，
    包括多个表格、图表上下文和代码片段。
    """

    def __init__(self, llm_provider: DeepSeekLLMProvider, base_output_dir: Path):
        self.llm = llm_provider
        self.base_output_dir = base_output_dir
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. 原始数据提取工具 ---
    def _extract_from_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        print(f">>> Harvester: 正在从 PDF 中提取所有原始资产: {pdf_path}")
        assets = {"tables": [], "images": [], "full_text": ""}
        try:
            with pdfplumber.open(pdf_path) as pdf:
                full_text_list = []
                for i, page in enumerate(pdf.pages):
                    full_text_list.append(page.extract_text() or "")
                    # 提取表格
                    for tbl_data in (page.extract_tables() or []):
                        if tbl_data and len(tbl_data) > 1:
                            df = pd.DataFrame(tbl_data[1:], columns=tbl_data[0])
                            assets["tables"].append(df)
                    # 提取图片 (及其位置)
                    for img in (page.images or []):
                        img['page_number'] = i + 1
                        assets["images"].append(img)
                assets["full_text"] = "\n".join(full_text_list)
            print(f">>> Harvester: 提取完成。发现 {len(assets['tables'])} 个表格, {len(assets['images'])} 张图片。")
            return assets
        except Exception as e:
            print(f"!!! Harvester: 从 PDF 提取时失败: {e}")
            return None

    # --- 2. 针对不同资产的处理模块 ---
    def _process_tables(self, raw_tables: List[pd.DataFrame], output_path: Path) -> List[Dict]:
        if not raw_tables: return []
        print("\n>>> Harvester [Tables]: 请求 LLM 评估所有表格...")
        tables_preview = "\n\n".join(
            [f"--- 表格 {i + 1} ---\n{tbl.head(5).to_markdown()}" for i, tbl in enumerate(raw_tables)])

        prompt = (
            "你是一名数据科学家顾问。评估以下所有表格。对于每一个适合创建机器学习任务的表格，"
            "以JSON列表的形式返回其信息。忽略那些不适合的表格。\n"
            "每个JSON对象必须包含: "
            "'table_index' (整数, 从1开始), "
            "'justification' (字符串, 为什么它适合ML), "
            "'suggested_filename' (字符串, 一个描述性的snake_case文件名, 以.csv结尾)"
        )
        context = f"以下是提取的表格预览：\n{tables_preview}"
        llm_response = self.llm.query(prompt, context)

        try:
            # 清理和解析LLM返回的JSON
            json_str = re.search(r'\[.*\]', llm_response, re.DOTALL).group(0)
            valuable_tables_info = json.loads(json_str)
        except (AttributeError, json.JSONDecodeError) as e:
            print(f"!!! Harvester [Tables]: 解析LLM评估结果失败: {e}\n--- 原始响应 ---\n{llm_response}")
            return []

        processed_tables = []
        for info in valuable_tables_info:
            try:
                idx = info['table_index'] - 1
                original_table_full = raw_tables[idx]

                cleaning_prompt = "..."  # (your cleaning prompt)
                context = f"请清洗并重构以下表格：\n{original_table_full.to_markdown()}"
                clean_llm_response = self.llm.query(cleaning_prompt, context)

                # --- START OF FIX ---
                match = re.search(r'\[.*\]', clean_llm_response, re.DOTALL)
                if not match:
                    print(f"!!! Harvester [Tables]: LLM清洗响应中未找到有效的JSON列表。跳过表格 {idx + 1}。")
                    print(f"--- 原始响应 ---\n{clean_llm_response}")
                    continue  # 跳到下一个循环

                json_clean_str = match.group(0)
                # --- END OF FIX ---

                cleaned_data = json.loads(json_clean_str)
                df = pd.DataFrame(cleaned_data)
                df.insert(0, 'Id', range(len(df)))

                table_path = output_path / "tables"
                table_path.mkdir(exist_ok=True)
                file_path = table_path / info['suggested_filename']
                df.to_csv(file_path, index=False)

                info['path'] = file_path
                processed_tables.append(info)
                print(f">>> Harvester [Tables]: 已成功处理并保存: {file_path}")

            except Exception as e:
                print(f"!!! Harvester [Tables]: 处理表格索引 {info.get('table_index')} 时出错: {e}")
                continue

        return processed_tables

    def _process_charts(self, raw_images: List[Dict], full_text: str, output_path: Path) -> List[Dict]:
        if not raw_images: return []
        print("\n>>> Harvester [Charts]: 开始处理提取的图片...")
        processed_charts = []
        chart_path = output_path / "charts"
        chart_path.mkdir(exist_ok=True)

        # 这是一个简化的上下文提取，可以进一步优化
        figure_captions = re.findall(r'(Fig\.|Figure)\s(\d+)\s*\|(.*)', full_text)

        for i, img in enumerate(raw_images):
            # 保存图片
            img_filename = f"chart_{i + 1}_page_{img['page_number']}.png"

            # 查找图片的上下文
            context = "未找到明确的图表标题。"
            # 这是一个简单的匹配逻辑，可以根据需要变得更复杂
            # 暂时假设图片顺序与标题顺序一致
            if i < len(figure_captions):
                context = f"Fig. {figure_captions[i][1]}: {figure_captions[i][2].strip()}"

            prompt = (
                "你是一位科学分析师。基于以下图表的标题/上下文，请用一句话总结该图表的内容，"
                "并评估其是否包含可用于机器学习的结构化数据（例如，通过OCR或视觉模型提取）。"
            )
            llm_response = self.llm.query(prompt, context)

            desc_filename = chart_path / f"chart_{i + 1}_description.txt"
            desc_filename.write_text(f"--- 图表上下文 ---\n{context}\n\n--- LLM 分析 ---\n{llm_response}",
                                     encoding='utf-8')

            processed_charts.append({
                "image_path_placeholder": chart_path / img_filename,
                "description_path": desc_filename,
                "context": context
            })
            print(f">>> Harvester [Charts]: 已分析并保存图表描述: {desc_filename}")

        return processed_charts

    def _process_code(self, full_text: str, output_path: Path) -> List[Dict]:
        if not full_text: return []
        print("\n>>> Harvester [Code]: 请求 LLM 提取所有代码片段...")
        prompt = (
            "你是一名代码分析师。扫描以下整篇文档的文本，提取出所有代码块、伪代码或命令行片段。"
            "以JSON列表的形式返回。每个对象应包含: "
            "'code' (字符串, 包含完整的代码片段), "
            "'description' (字符串, 简要说明这段代码的作用或上下文)"
        )
        llm_response = self.llm.query(prompt, full_text)

        try:
            json_str = re.search(r'\[.*\]', llm_response, re.DOTALL).group(0)
            code_snippets = json.loads(json_str)
        except (AttributeError, json.JSONDecodeError):
            print("!!! Harvester [Code]: 未能从文本中解析出代码片段。")
            return []

        processed_code = []
        code_path = output_path / "code"
        code_path.mkdir(exist_ok=True)
        for i, snippet in enumerate(code_snippets):
            file_extension = ".py" if "python" in snippet.get('description', '').lower() else ".txt"
            file_path = code_path / f"code_snippet_{i + 1}{file_extension}"
            file_path.write_text(snippet['code'], encoding='utf-8')
            processed_code.append({
                "path": file_path,
                "description": snippet['description']
            })
            print(f">>> Harvester [Code]: 已提取并保存代码片段: {file_path}")

        return processed_code

    def _handle_zenodo_source(self, url: str, output_path: Path) -> Dict[str, Any]:
        print(f">>> Harvester [Zenodo]: 已识别 Zenodo 仓库链接: {url}")
        processed_assets = {"tables": [], "code_snippets": [], "charts": []}

        try:
            # 从 URL 获取 Zenodo 记录 ID
            record_id = url.split('/')[-1]
            api_url = f"https://zenodo.org/api/records/{record_id}"

            # 使用 API 获取文件元数据
            print(f">>> Harvester [Zenodo]: 正在查询 Zenodo API: {api_url}")
            response = requests.get(api_url)
            response.raise_for_status()
            metadata = response.json()

            files = metadata.get('files', [])
            if not files:
                print("!!! Harvester [Zenodo]: API 响应中未找到文件。")
                return {}

            # 寻找 .zip 压缩包
            zip_file_info = next((f for f in files if f['key'].endswith('.zip')), None)
            if not zip_file_info:
                print("!!! Harvester [Zenodo]: 未在此仓库中找到 .zip 压缩包。")
                return {}

            # 下载并解压
            zip_url = zip_file_info['links']['self']
            print(f">>> Harvester [Zenodo]: 正在下载压缩包: {zip_url}")
            zip_response = requests.get(zip_url)
            zip_response.raise_for_status()

            unzip_path = output_path / "unzipped_content"
            unzip_path.mkdir(exist_ok=True)
            print(f">>> Harvester [Zenodo]: 正在解压到: {unzip_path}")
            with zipfile.ZipFile(io.BytesIO(zip_response.content)) as z:
                z.extractall(unzip_path)

            # 探索解压后的文件
            file_manifest = []
            for root, _, filenames in os.walk(unzip_path):
                for filename in filenames:
                    file_manifest.append(os.path.relpath(os.path.join(root, filename), unzip_path))

            manifest_str = "\n".join(file_manifest)
            print(">>> Harvester [Zenodo]: 解压内容清单:\n" + manifest_str)

            prompt = (
                "你是一位顶尖的数据科学家。以下是一个科研项目仓库的文件清单。请分析并识别出数据文件和代码文件。\n\n"
                "**规则:**\n"
                "1.  **数据文件识别**: 将 `.csv`, `.json`, `.txt` 文件视为数据文件。同时，**特别注意**那些**没有文件扩展名**的文件，它们很可能也是重要的文本数据文件，请将它们也包含在 'data_files' 列表中。\n"
                "2.  **忽略二进制文件**: 完全忽略 `.pkl`, `.db`, `.zip` 等二进制文件。\n"
                "3.  **代码文件识别**: 将 `.py` 和 `.ipynb` 文件视为 'code_files'。\n"
                "4.  **严格的JSON输出**: 你的回复必须且只能是一个JSON对象，不包含任何解释性文字。结构如下:\n"
                "{'data_files': ['path/to/data1', ...], 'code_files': ['path/to/script1.py', ...]}\n"
                "5.  **空列表**: 如果找不到任何符合条件的文本数据文件，你必须返回一个空的 'data_files' 列表，但依然保持完整的JSON结构。\n\n"
                "**示例:**\n"
                "输入清单:\n"
                "project/data/G1\n"
                "project/run.py\n"
                "project/model.pkl\n"
                "输出JSON:\n"
                "{\n"
                "  \"data_files\": [\"project/data/G1\"],\n"
                "  \"code_files\": [\"project/run.py\"]\n"
                "}"
            )
            llm_response = self.llm.query(prompt, manifest_str)

            #print("\n--- RAW LLM RESPONSE (for file identification) ---")
            #print(llm_response)
            #print("--------------------------------------------------\n")
            match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if not match:
                print("!!! Harvester [Zenodo]: LLM响应中未找到有效的JSON对象。")
                print(f"--- 原始响应 ---\n{llm_response}")
                return {}  # 返回空字典，表示失败但程序不崩溃

            json_str = match.group(0)
            identified_files = json.loads(json_str)

            #  处理识别出的数据文件
            data_files_path = output_path / "tables"
            data_files_path.mkdir(exist_ok=True)
            for data_file_rel_path in identified_files.get('data_files', []):
                full_path = unzip_path / data_file_rel_path

                try:
                    if not full_path.exists():
                        print(f"!!! Harvester [Zenodo]: 文件 '{data_file_rel_path}' 在清单中但实际不存在，跳过。")
                        continue

                    # sep=r'\s+': 使用一个或多个空白字符作为分隔符
                    df = pd.read_csv(full_path, header=None, sep=r'\s+')

                    final_path = data_files_path / Path(data_file_rel_path).name

                    df.columns = [f'col_{i}' for i in range(len(df.columns))]
                    df.to_csv(final_path, index=False)

                    processed_assets["tables"].append({
                        "path": final_path,
                        "justification": f"从 Zenodo 仓库 {record_id} 中自动识别并成功读取的数据文件。",
                        "suggested_filename": Path(data_file_rel_path).name
                    })
                    print(f">>> Harvester [Zenodo]: 已成功处理并保存数据文件: {final_path}")

                except Exception as e:
                    # 捕获所有可能的读取错误 (UnicodeDecodeError, ParserError, etc.)
                    print(f"!!! Harvester [Zenodo]: 无法读取或解析文件 '{data_file_rel_path}'。")
                    print(f"    错误类型: {type(e).__name__}")
                    print("    此文件可能为二进制格式或非标准文本。将跳过此文件。")
                    continue


            return processed_assets

        except Exception as e:
            print(f"!!! Harvester [Zenodo]: 处理 Zenodo 仓库时出错: {e}")
            return {}
    # --- 3. 主协调方法 ---
    def harvest_from_source(self, source: str) -> Dict[str, List]:

        source_name = Path(source).stem
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_output_path = self.base_output_dir / f"harvest_{source_name}_{timestamp}"
        run_output_path.mkdir(parents=True, exist_ok=True)
        print(f"--- Harvester Run Initialized. Outputting to: {run_output_path} ---")
        if 'zenodo.org' in source or 'doi.org/10.5281/zenodo' in source:
            # 解析DOI链接
            if 'doi.org' in source:
                try:
                    # 获取 DOI 指向的最终 URL
                    response = requests.head(source, allow_redirects=True)
                    source = response.url
                except requests.RequestException as e:
                    print(f"!!! Harvester: 解析 DOI 链接失败: {e}")
                    return {}

            return self._handle_zenodo_source(source, run_output_path)
        # 统一提取
        elif source.startswith('http'):
            print("!!! Harvester: URL处理暂未实现，请提供本地PDF路径。")
            return {}
        elif Path(source).is_file() and Path(source).suffix.lower() == '.pdf':
            raw_assets = self._extract_from_pdf(Path(source))
        else:
            print(f"!!! Harvester: 不支持的源格式: {source}")
            return {}

        if not raw_assets: return {}

        # 分别处理各类资产
        final_assets = {
            "tables": self._process_tables(raw_assets["tables"], run_output_path),
            "charts": self._process_charts(raw_assets["images"], raw_assets["full_text"], run_output_path),
            "code_snippets": self._process_code(raw_assets["full_text"], run_output_path)
        }

        print(f"\n--- Harvester Run Finished ---")
        return final_assets