# 文件名: Auto_ML.py
# (此为完整文件，支持从urls.txt批量处理，并生成全英文JSON)

import os
import re
import json
import time
import random

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


class DeepSeekLLMProvider:
    # ★ MODIFIED: __init__ 方法现在接收一个 api_key 参数
    def __init__(self, api_key: str, model="deepseek-chat"):
        if not api_key:
            raise ValueError("Error: API key was not provided.")
        self.client = DeepSeekAI(api_key=api_key)
        self.model = model
        print(f"DeepSeekLLMProvider initialized successfully! Model: {self.model}")

    def query(self, system_prompt: str, user_context: str) -> str:
        print(f"\n--- Sending request to DeepSeek API (model: {self.model})... ---")
        try:
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_context}]
            response = self.client.chat.completions.create(model=self.model, messages=messages,
                                                           response_format={"type": "json_object"})
            result = response.choices[0].message.content
            print("--- API response received successfully. ---")
            return result
        except Exception as e:
            print(f"!!! Error calling DeepSeek API: {e} !!!")
            return f'{{"error": "API call failed: {e}"}}'


def _extract_json_from_response(raw_text: str) -> str:
    match = re.search(r'```(json)?\s*(\{.*?\})\s*```', raw_text, re.DOTALL)
    if match: return match.group(2)
    match = re.search(r'\{.*\}', raw_text, re.DOTALL)
    if match: return match.group(0)
    return raw_text


def extract_key_sections_from_html(url: str) -> dict:
    print(f"--- Extracting information from URL: {url} ---")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    try:
        response = requests.get(url, headers=headers, timeout=45);
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'lxml')
        title = soup.find('h1').get_text(strip=True) if soup.find('h1') else "Untitled Paper"
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
        print("--- Web information extraction complete. ---")
        return {"title": title, "url": url, "abstract": abstract, "methods": methods_text, "results": results_text,
                "captions": "\n".join(captions), "availability_links": sorted(list(links))}
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}");
        return None


def generate_task_with_llm(paper_data: dict, llm_provider: DeepSeekLLMProvider) -> dict:
    system_prompt = """
    You are a top-tier interdisciplinary scientist, skilled in both machine learning and specific scientific domains. Your task is to deeply analyze a scientific paper and output two core components in a highly structured JSON format: 1. Details for a reproducible machine learning task. 2. A summary of the paper's scientific contributions.

    Your response MUST be a single, well-formed JSON object, without any Markdown formatting or explanatory text.
    The JSON structure MUST be as follows:
    {
      "scientific_summary": {
        "background": "A 2-3 sentence summary of the research field's background and existing challenges.",
        "hypothesis_or_goal": "A clear statement of the paper's core research goal, scientific hypothesis, or the key problem it aims to solve.",
        "methodology_summary": "A bulleted list (as a single string separated by '\\n-') summarizing the key methods, techniques, or experimental designs used to achieve the goal.",
        "key_results": "A bulleted list (as a single string separated by '\\n-') summarizing the most important scientific findings or conclusions of the paper."
      },
      "ml_task": {
        "task_type": "The type of machine learning task, e.g., 'Image Classification', 'Time Series Forecasting'.",
        "problem_statement": "A concise, one-sentence description of the core problem.",
        "input_description": "Description of the model's input data, including format, dimensions, etc.",
        "output_description": "Description of the model's predicted output, including format, meaning, etc."
      },
      "evaluation_criteria": {
        "primary_metric": "The main metric used to evaluate model performance, e.g., 'Accuracy', 'F1-Score'.",
        "secondary_metrics": "A list of other evaluation metrics mentioned.",
        "reported_performance": "A quote of the key performance result reported in the paper."
      },
      "data_details": {
        "format": "The core format of the data, e.g., 'CSV', 'Image files (JPEG/PNG)', 'NetCDF'.",
        "structure_description": "How the data is organized, e.g., 'A single CSV file named data.csv'.",
        "target_column_or_logic": "The name of the target column or a description of how to determine labels."
      },
      "model_details": {
        "architecture_family": "The family of the model architecture, e.g., 'Convolutional Neural Network (CNN)'.",
        "key_libraries_mentioned": "A list of key Python libraries mentioned or strongly implied."
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
        print(f"!!! Error parsing detailed task definition JSON: {e}. Raw response: '{response_raw}' !!!")
        return {"error": f"Failed to parse LLM response. Raw response: {response_raw}"}


def _handle_zenodo_link(link: str, download_dir: str) -> list:
    log_messages = []
    try:
        record_id_match = re.search(r'(\d+)$', link)
        if not record_id_match: raise ValueError("Could not extract a valid Zenodo record ID.")
        record_id = record_id_match.group(1)
        print(f"  > Detected Zenodo link, Record ID: {record_id}")
        api_url = f"https://zenodo.org/api/records/{record_id}"
        response = requests.get(api_url);
        response.raise_for_status()
        data = response.json()
        files_to_download = data.get('files', [])
        if not files_to_download:
            message = f"  > No files found in Zenodo record {record_id}.";
            print(message);
            log_messages.append(message);
            return log_messages
        print(f"  > Found {len(files_to_download)} files. Starting download... (Press Ctrl+C to interrupt)")
        for file_info in files_to_download:
            original_filename = file_info['key']
            sanitized_filename = original_filename.replace('/', '_').replace('\\', '_')
            if original_filename != sanitized_filename:
                print(f"    L Sanitizing filename: '{original_filename}' -> '{sanitized_filename}'")

            local_filepath = os.path.join(download_dir, sanitized_filename)
            try:
                with requests.get(file_info['links']['self'], stream=True) as r:
                    r.raise_for_status()
                    with open(local_filepath, 'wb') as f, tqdm(desc=f"    L Downloading {file_info['key']}",
                                                               total=file_info['size'], unit='B', unit_scale=True,
                                                               unit_divisor=1024) as bar:
                        for chunk in r.iter_content(chunk_size=8192): f.write(chunk); bar.update(len(chunk))
                log_messages.append(f"Successfully downloaded: {local_filepath}")
                if file_info['key'].endswith('.zip'):
                    print(f"    L Detected ZIP file, extracting: {file_info['key']}")
                    zip_filename_no_ext = os.path.splitext(file_info['key'])[0]
                    extract_path = os.path.join(download_dir, zip_filename_no_ext)
                    os.makedirs(extract_path, exist_ok=True)
                    with zipfile.ZipFile(local_filepath, 'r') as zip_ref:
                        zip_ref.extractall(extract_path)
                    success_message = f"Successfully extracted ZIP to: {extract_path}"
                    print(f"    L {success_message}")
                    log_messages.append(success_message)
                    try:
                        os.remove(local_filepath)
                        delete_message = f"Original ZIP file deleted: {local_filepath}"
                        print(f"    L Original ZIP file deleted: {file_info['key']}")
                        log_messages.append(delete_message)
                    except OSError as e:
                        error_message = f"Could not delete ZIP file {local_filepath}: {e}";
                        print(f"    L !!! {error_message} !!!");
                        log_messages.append(error_message)
            except KeyboardInterrupt:
                print(f"\n!!! User interrupted download of {file_info['key']}. !!!")
                if os.path.exists(local_filepath): os.remove(local_filepath); print(
                    f"  > Incomplete file deleted: {local_filepath}")
                raise
    except Exception as e:
        if not isinstance(e, KeyboardInterrupt):
            error_message = f"Failed to process Zenodo link {link}: {e}";
            print(f"!!! {error_message} !!!");
            log_messages.append(error_message)
        raise
    return log_messages


def _profile_data_directory(directory: str) -> dict:
    print(f"  > Profiling data in directory: {directory}")
    profile = {'file_formats': {}, 'primary_format': 'unknown', 'file_count': 0}
    DATA_EXTENSIONS = ['.csv', '.nc', '.h5', '.hdf5', '.jsonl', '.json', '.parquet', '.tsv', '.dat', '.grib', '.grib2']
    all_files = glob.glob(os.path.join(directory, '**', '*'), recursive=True)
    for file_path in all_files:
        if os.path.isfile(file_path):
            profile['file_count'] += 1
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext: profile['file_formats'][file_ext] = profile['file_formats'].get(file_ext, 0) + 1
    data_format_counts = {ext: count for ext, count in profile['file_formats'].items() if ext in DATA_EXTENSIONS}
    if data_format_counts: profile['primary_format'] = max(data_format_counts, key=data_format_counts.get)
    print(
        f"  > Profiling complete: {profile['file_count']} total files, primary data format is '{profile['primary_format']}'")
    return profile


def setup_data_source(links: list, download_dir: str):
    print(f"--- Processing data sources, target directory: {download_dir} ---")
    if not links: return ["No data links provided."], {}
    os.makedirs(download_dir, exist_ok=True)
    local_setup_log = []
    try:
        for link in links:
            if "github.com" in link:
                repo_name = urlparse(link).path.split('/')[-1].replace('.git', '')
                repo_path = os.path.join(download_dir, repo_name)
                if not os.path.exists(repo_path):
                    print(f"Cloning GitHub repository: {link} -> {repo_path} (Press Ctrl+C to interrupt)")
                    try:
                        git.Repo.clone_from(link, repo_path)
                        local_setup_log.append(f"Successfully cloned to {repo_path}")
                    except git.exc.GitCommandError as e:
                        if "terminated" in str(e).lower():
                            raise KeyboardInterrupt
                        else:
                            raise e
                else:
                    print(f"Repository already exists at: {repo_path}")
                    local_setup_log.append(f"Repository already exists at {repo_path}")
            elif "zenodo" in link:
                zenodo_logs = _handle_zenodo_link(link, download_dir)
                local_setup_log.extend(zenodo_logs)
            else:
                message = f"Link cannot be processed automatically (manual handling required): {link}";
                print(message);
                local_setup_log.append(message)
    except KeyboardInterrupt:
        print("\n========================================================")
        print(" Data download/clone process terminated by user.")
        print("========================================================")
        sys.exit(0)
    except Exception as e:
        error_message = f"An unknown error occurred while processing links: {e}";
        print(f"!!! {error_message} !!!");
        local_setup_log.append(error_message)
    final_profile = _profile_data_directory(download_dir)
    return local_setup_log, final_profile




def process_single_paper(url: str, output_base_dir: str, llm_provider: DeepSeekLLMProvider):

    print(f"\n{'=' * 80}\nProcessing paper: {url}\n{'=' * 80}")

    paper_data = extract_key_sections_from_html(url)
    if not paper_data:
        raise Exception("Failed to extract web data from URL.")

    sanitized_title = re.sub(r'[\s:/\\]+', '_', paper_data['title'].lower())[:80]
    paper_output_dir = os.path.join(output_base_dir, f"paper_{sanitized_title}")
    os.makedirs(paper_output_dir, exist_ok=True)
    print(f"--- Output directory for this paper: {paper_output_dir} ---")

    detailed_task_definition = generate_task_with_llm(paper_data, llm_provider)
    if "error" in detailed_task_definition:
        raise Exception(f"LLM failed to generate task definition. Details: {detailed_task_definition['error']}")

    paper_data_dir = os.path.join(paper_output_dir, "data")
    setup_logs, data_profile = setup_data_source(paper_data['availability_links'], paper_data_dir)


    final_task_output = {
        "source_paper": {"title": paper_data.get('title', 'N/A'), "url": paper_data.get('url', 'N/A')},
        **detailed_task_definition,
        "data_profile": data_profile,
        "data_sources": {
            "raw_data_path": paper_data_dir,  # ★ 使用绝对或相对清晰的路径
            "original_links": paper_data.get('availability_links', []),
            "local_setup_log": setup_logs
        }
    }
    output_filename = os.path.join(paper_output_dir, "generated_mle_task.json")
    with open(output_filename, "w", encoding='utf-8') as f:
        f.write(json.dumps(final_task_output, indent=2, ensure_ascii=False))

    print(f"\n✅ Detailed task definition saved to: {output_filename}")
    print(f"Next step: Run 'python benchmark_generator_v2.py \"{output_filename}\"' to generate the project.")


if __name__ == "__main__":
    # 定义输入文件和主输出目录
    DEEPSEEK_API_KEY = "sk-353a88a777bd4c598f17b2923677e100"
    if not DEEPSEEK_API_KEY:
        print("Error: Please set your DEEPSEEK_API_KEY in the script.")
        sys.exit(1)
    URL_FILE = "urls.txt"
    MAIN_OUTPUT_DIR = "batch_output"


    if not os.path.exists(URL_FILE):
        print(f"Error: Input file '{URL_FILE}' not found.")
        print("Please create a file named 'urls.txt' and add one paper URL per line.")
        sys.exit(1)

    with open(URL_FILE, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip()]

    if not urls:
        print(f"Error: No URLs found in '{URL_FILE}'.")
        sys.exit(1)

    os.makedirs(MAIN_OUTPUT_DIR, exist_ok=True)

    try:
        llm_provider = DeepSeekLLMProvider(api_key=DEEPSEEK_API_KEY, model="deepseek-chat")
    except ValueError as e:
        print(e);
        sys.exit(1)

    success_count = 0
    failure_count = 0

    print(f"\nFound {len(urls)} papers to process. Starting batch job...")

    for i, url in enumerate(urls):
        try:
            process_single_paper(url, MAIN_OUTPUT_DIR, llm_provider)
            success_count += 1
        except Exception as e:
            print(f"\n{'!' * 80}\nFAILED to process paper: {url}\nReason: {e}\n{'!' * 80}")
            failure_count += 1
        time.sleep(random.uniform(2, 5))

    print("\n" + "=" * 80)
    print("Batch processing complete.")
    print(f"  - Success: {success_count}")
    print(f"  - Failed:  {failure_count}")
    print(f"Results are saved in the '{MAIN_OUTPUT_DIR}' directory.")
    print("=" * 80)
