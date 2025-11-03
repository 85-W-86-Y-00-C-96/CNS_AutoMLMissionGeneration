# 文件名: Auto_ML.py
import os
import re
import json
import git
import requests
import zipfile
import sys
import glob
from io import BytesIO
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from io import StringIO
from deepseek_ai import DeepSeekAI
from tqdm import tqdm


#  LLM Provider
class DeepSeekLLMProvider:
    def __init__(self, model="deepseek-chat"):
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key: raise ValueError("错误：请先设置 DEEPSEEK_API_KEY 环境变量。")
        self.client = DeepSeekAI(api_key=api_key)
        self.model = model
        print(f"DeepSeekLLMProvider 初始化成功！模型: {self.model}")

    def query(self, system_prompt: str, user_context: str) -> str:
        print(f"\n--- 正在向 DeepSeek API (模型: {self.model}) 发送请求... ---")
        try:
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_context}]
            response = self.client.chat.completions.create(model=self.model, messages=messages,
                                                           response_format={"type": "json_object"})
            result = response.choices[0].message.content
            print("--- 成功接收到 API 响应 ---")
            return result
        except Exception as e:
            print(f"!!! 调用 DeepSeek API 时出错: {e} !!!")
            return f'{{"error": "API call failed: {e}"}}'



def _extract_json_from_response(raw_text: str) -> str:
    match = re.search(r'```(json)?\s*(\{.*?\})\s*```', raw_text, re.DOTALL)
    if match: return match.group(2)
    match = re.search(r'\{.*\}', raw_text, re.DOTALL)
    if match: return match.group(0)
    return raw_text


# 信息提取
def extract_key_sections_from_html(url: str) -> dict:
    print(f"--- 正在从URL提取信息: {url} ---")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    try:
        response = requests.get(url, headers=headers, timeout=45);
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'lxml')
        title = soup.find('h1').get_text(strip=True) if soup.find('h1') else "No title found"
        abstract_tag = soup.find('section', attrs={'aria-labelledby': re.compile(r'Abs', re.I)})
        abstract = abstract_tag.get_text(strip=True, separator='\n') if abstract_tag else "No abstract found"

        def get_section_text(section_title_regex):
            header = soup.find(['h2', 'h3'], string=re.compile(section_title_regex, re.I))
            if not header: return ""
            content = []
            for sibling in header.find_next_siblings():
                if sibling.name in ['h2', 'h3']: break
                content.append(sibling.get_text(strip=True, separator='\n'))
            return "\n".join(content)

        methods_text = get_section_text(r'Methods|Methodology|Materials and Methods')
        results_text = get_section_text(r'Results')
        captions = []
        for fig_tag in soup.find_all('figure'):
            caption = fig_tag.find('figcaption') or fig_tag.find(class_=re.compile(r'caption', re.I))
            if caption: captions.append(caption.get_text(strip=True))
        links = set()
        repo_pattern = r'https?://(github\.com|gitlab\.com|figshare\.com|zenodo\.org|doi\.org)[\w\-\./%]+'
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            if re.match(repo_pattern, href): links.add(href)
        print("--- 网页信息提取完成 ---")
        return {"title": title, "url": url, "abstract": abstract, "methods": methods_text, "results": results_text,
                "captions": "\n".join(captions), "availability_links": sorted(list(links))}
    except requests.exceptions.RequestException as e:
        print(f"请求网页时出错: {e}");
        return None


# LLM 驱动的详细任务解析
def generate_task_with_llm(paper_data: dict, llm_provider: DeepSeekLLMProvider) -> dict:
    system_prompt = """
    你是一位顶尖的机器学习科学家和软件工程师。你的任务是深入阅读科学论文的摘要、方法和结果，然后提取并生成一个详细的、结构化的JSON对象，用于自动化地创建机器学习基准项目。

    你的回复必须且只能是一个格式正确的JSON对象，不包含任何Markdown标记或额外文本。
    JSON的结构必须如下：
    {
      "ml_task": {
        "task_type": "机器学习任务类型，例如：Image Classification, Time Series Forecasting, Text Generation, Object Detection。",
        "problem_statement": "用一句话简洁地描述核心问题。",
        "input_description": "描述模型需要接收的输入数据是什么，包括格式、维度等关键信息。",
        "output_description": "描述模型需要预测的输出是什么，包括格式、含义等。"
      },
      "evaluation_criteria": {
        "primary_metric": "论文中用于衡量模型性能最主要的指标，例如：Accuracy, F1-Score, Mean Absolute Error, BLEU Score。",
        "secondary_metrics": "一个包含论文中提到的其他次要评估指标的字符串列表。",
        "reported_performance": "引用论文中关于模型在主要指标上达到的关键性能结果的一句话描述。如果找不到，请填写 'Not specified'。"
      },
      "data_details": {
        "format": "数据的核心格式，例如：'CSV', 'Image files (JPEG/PNG)', 'Text files (.txt)', 'JSONL', 'HDF5'。",
        "structure_description": "描述数据是如何组织的。例如：'一个名为data.csv的单一CSV文件'，或者 '图像存储在以类别命名的子文件夹中'，或者 '训练数据位于train.txt，标签位于labels.txt'。",
        "target_column_or_logic": "如果数据是表格，请指明目标列的名称。如果是其他格式，请描述如何确定标签。例如：'label'，或者 '文件名本身就是标签'，或者 '子文件夹的名称是标签'。如果无法确定，请填写 'Undetermined'。"
      },
      "model_details": {
        "architecture_family": "论文中使用的模型架构属于哪个家族？例如：'Convolutional Neural Network (CNN)', 'Transformer', 'Gradient Boosting Tree', 'Recurrent Neural Network (RNN)'。如果不清楚，请填写 'General Deep Learning Model'。",
        "key_libraries_mentioned": "一个字符串列表，包含论文中明确提到或强烈暗示使用的关键Python库。例如：['PyTorch', 'TensorFlow', 'scikit-learn', 'Hugging Face Transformers', 'Pandas']。如果未提及，返回空列表 []。"
      }
    }
    """
    user_context = f"""
    Title: {paper_data['title']}
    Abstract: {paper_data['abstract']}
    Methods Section Summary: {paper_data['methods'][:4000]}
    Results Section Summary: {paper_data['results'][:4000]}
    Figure/Table Captions: {paper_data['captions']}
    """
    response_raw = llm_provider.query(system_prompt, user_context)
    response_clean = _extract_json_from_response(response_raw)
    try:
        detailed_task_data = json.loads(response_clean)
        return detailed_task_data
    except json.JSONDecodeError as e:
        print(f"!!! 解析详细任务定义JSON时出错: {e}。原始响应: '{response_raw}' !!!")
        return {"error": f"Failed to parse LLM response. Raw response: {response_raw}"}


# 自动化数据处理与分析
def _handle_zenodo_link(link: str, download_dir: str) -> list:
    log_messages = []
    try:
        record_id_match = re.search(r'(\d+)$', link)
        if not record_id_match: raise ValueError("无法从链接中提取有效的Zenodo记录ID。")
        record_id = record_id_match.group(1)
        print(f"  > 检测到Zenodo链接，提取到记录ID: {record_id}")
        api_url = f"https://zenodo.org/api/records/{record_id}"
        response = requests.get(api_url);
        response.raise_for_status()
        data = response.json()
        files_to_download = data.get('files', [])
        if not files_to_download:
            message = f"  > Zenodo记录 {record_id} 中未找到文件。";
            print(message);
            log_messages.append(message);
            return log_messages
        print(f"  > 找到 {len(files_to_download)} 个文件，开始下载... (可按 Ctrl+C 中断)")
        for file_info in files_to_download:
            local_filepath = os.path.join(download_dir, file_info['key'])
            try:
                with requests.get(file_info['links']['self'], stream=True) as r:
                    r.raise_for_status()
                    with open(local_filepath, 'wb') as f, tqdm(
                            desc=f"    L 下载 {file_info['key']}", total=file_info['size'], unit='B', unit_scale=True,
                            unit_divisor=1024
                    ) as bar:
                        for chunk in r.iter_content(chunk_size=8192): f.write(chunk); bar.update(len(chunk))
                log_messages.append(f"Successfully downloaded: {local_filepath}")
                if file_info['key'].endswith('.zip'):
                    print(f"    L 检测到ZIP文件，正在解压: {file_info['key']}")
                    with zipfile.ZipFile(local_filepath, 'r') as zip_ref: zip_ref.extractall(download_dir)
                    log_messages.append(f"Automatically extracted ZIP: {local_filepath}")
            except KeyboardInterrupt:
                print(f"\n中断了 {file_info['key']} 的下载")
                if os.path.exists(local_filepath): os.remove(local_filepath); print(
                    f"  > 已删除不完整的文件: {local_filepath}")
                raise
    except Exception as e:
        if not isinstance(e, KeyboardInterrupt):
            error_message = f"处理Zenodo链接 {link} 时失败: {e}";
            print(f"!!! {error_message} !!!");
            log_messages.append(error_message)
        raise
    return log_messages


def _profile_data_directory(directory: str) -> dict:
    print(f"  > 正在分析目录中的数据: {directory}")
    profile = {'file_formats': {}, 'primary_format': 'unknown', 'file_count': 0}
    DATA_EXTENSIONS = ['.csv', '.nc', '.h5', '.hdf5', '.jsonl', '.json', '.parquet', '.tsv', '.dat', '.grib', '.grib2']
    all_files = glob.glob(os.path.join(directory, '**', '*'), recursive=True)
    for file_path in all_files:
        if os.path.isfile(file_path):
            profile['file_count'] += 1
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext:
                profile['file_formats'][file_ext] = profile['file_formats'].get(file_ext, 0) + 1
    data_format_counts = {ext: count for ext, count in profile['file_formats'].items() if ext in DATA_EXTENSIONS}
    if data_format_counts:
        profile['primary_format'] = max(data_format_counts, key=data_format_counts.get)
    print(f"  > 分析完成: 共 {profile['file_count']} 个文件, 主要数据格式为 '{profile['primary_format']}'")
    return profile


def setup_data_source(links: list, download_dir="data"):
    print(f"--- 正在处理数据源链接，目标文件夹: {download_dir} ---")
    if not links: return ["No data links provided."], {}
    os.makedirs(download_dir, exist_ok=True)
    local_setup_log = []

    try:
        for link in links:
            if "github.com" in link:
                repo_name = urlparse(link).path.split('/')[-1].replace('.git', '')
                repo_path = os.path.join(download_dir, repo_name)
                if not os.path.exists(repo_path):
                    print(f"正在克隆GitHub仓库: {link} -> {repo_path}")
                    try:
                        git.Repo.clone_from(link, repo_path)
                        local_setup_log.append(f"Successfully cloned to {repo_path}")
                    except git.exc.GitCommandError as e:
                        if "terminated" in str(e).lower():
                            raise KeyboardInterrupt
                        else:
                            raise e
                else:
                    print(f"仓库已存在于: {repo_path}")
                    local_setup_log.append(f"Repository already exists at {repo_path}")
            elif "zenodo" in link:
                zenodo_logs = _handle_zenodo_link(link, download_dir)
                local_setup_log.extend(zenodo_logs)
            else:
                message = f"无法自动处理的链接 (请手动操作): {link}";
                print(message);
                local_setup_log.append(message)
    except KeyboardInterrupt:
        print("数据下载/克隆过程已被手动终止。")
        sys.exit(0)
    except Exception as e:
        error_message = f"处理链接时发生未知错误: {e}";
        print(f"!!! {error_message} !!!");
        local_setup_log.append(error_message)

    # 在所有下载完成后，对整个数据目录进行一次总分析
    final_profile = _profile_data_directory(download_dir)
    return local_setup_log, final_profile


def main(url: str):
    try:
        llm_provider = DeepSeekLLMProvider(model="deepseek-chat")
    except ValueError as e:
        print(e);
        return
    paper_data = extract_key_sections_from_html(url)
    if not paper_data:
        print("无法从URL提取有效信息，流程终止。");
        return
    detailed_task_definition = generate_task_with_llm(paper_data, llm_provider)
    if "error" in detailed_task_definition:
        print("!!! LLM未能成功生成任务定义，流程终止。 !!!");
        print(f"错误详情: {detailed_task_definition['error']}");
        return

    setup_logs, data_profile = setup_data_source(paper_data['availability_links'])

    print("\n" + "=" * 50)
    print("---            整合最终任务定义            ---")
    print("=" * 50)

    final_task_output = {
        "source_paper": {
            "title": paper_data.get('title', 'N/A'),
            "url": paper_data.get('url', 'N/A')
        },
        **detailed_task_definition,
        "data_profile": data_profile,
        "data_sources": {
            "raw_data_path": "data",
            "original_links": paper_data.get('availability_links', []),
            "local_setup_log": setup_logs
        }
    }

    final_json_output = json.dumps(final_task_output, indent=2, ensure_ascii=False)
    print(final_json_output)
    output_filename = "generated_mle_task.json"
    with open(output_filename, "w", encoding='utf-8') as f:
        f.write(final_json_output)
    print(f"\n详细任务定义(含数据画像)已成功保存到: {output_filename}")


# 启动入口
if __name__ == "__main__":
    os.environ["DEEPSEEK_API_KEY"] = "sk-353a88a777bd4c598f17b2923677e100"
    paper_url = "https://www.nature.com/articles/s41467-025-61087-4"
    main(paper_url)