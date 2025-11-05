
import os
import json
import re
import shutil
import argparse
import sys

import pandas as pd
from sklearn.model_selection import train_test_split
import glob

try:
    from Auto_ML import DeepSeekLLMProvider
except ImportError:
    print("Error: Could not import DeepSeekLLMProvider from 'Auto_ML.py'.")
    print("Please ensure 'Auto_ML.py' and 'benchmark_generator_v2.py' are in the same folder.")
    exit(1)

def generate_code_with_llm(task_data: dict, llm_provider: DeepSeekLLMProvider, file_type: str) -> str:

    data_profile = task_data.get('data_profile', {})
    context = f"""
    # Task Context:
    - Task Type: {task_data.get('ml_task', {}).get('task_type', 'N/A')}
    - Problem Statement: {task_data.get('ml_task', {}).get('problem_statement', 'N/A')}
    # ... (omitted for brevity) ...
    # Detected Data Profile:
    - Primary Data Format: {data_profile.get('primary_format', 'unknown')}
    - All Detected File Formats: {list(data_profile.get('file_formats', {}).keys())}
    - Raw data is located in the 'data/' subdirectory of the paper's output folder.
    """
    if file_type == 'train':
        system_prompt = f"""
        You are a senior Python machine learning engineer. Based on the provided task context and data profile, write a complete, runnable `train.py` script.
        Requirements:
        1. You **MUST** write data loading code that can natively handle the '{data_profile.get('primary_format', 'unknown')}' format.
           - For '.nc', use the `xarray` library.
           - For '.h5' or '.hdf5', use `h5py` or `pandas.HDFStore`.
           - For '.csv', use `pandas.read_csv`.
           - For other formats, choose the most appropriate library.
        2. Select a simple but suitable model as a starting point based on the 'architecture_family' and 'key_libraries_mentioned'.
        3. Include complete model training and saving logic.
        4. Leave clear # TODO: comments where the user needs to specify exact filenames or variable names.
        Your output must be pure Python code, without any Markdown formatting (like ```python) or explanatory text.
        """
    elif file_type == 'score':
        system_prompt = f"""
        You are a senior Python machine learning engineer. Based on the provided task context and data profile, write a complete, runnable `score.py` script.
        Requirements:
        1. Include logic to load the saved model and test data. The test data loading logic **MUST** be able to handle the '{data_profile.get('primary_format', 'unknown')}' format.
        2. Use the loaded model to make predictions on the test data.
        3. Calculate the score using an appropriate library (e.g., scikit-learn.metrics) based on the specified 'primary_metric' ('{task_data.get('evaluation_criteria', {}).get('primary_metric', 'N/A')}').
        4. Print the final scoring result in a clear JSON format.
        5. Include command-line argument parsing to allow the user to specify the model path and test data path.
        Your output must be pure Python code, without any Markdown formatting (like ```python) or explanatory text.
        """
    else:
        return "# Invalid file type requested"
    print(f"  > Requesting code generation from LLM for {file_type}.py...")
    generated_code = llm_provider.query(system_prompt, context)
    clean_code = generated_code.strip()
    if clean_code.startswith("```python"): clean_code = clean_code[len("```python"):].strip()
    if clean_code.endswith("```"): clean_code = clean_code[:-len("```")].strip()
    return clean_code

def generate_description_txt(task_data: dict, file_path: str):
    print("  > Generating description.txt...")
    summary = task_data.get('scientific_summary', {})
    background = summary.get('background', 'N/A')
    hypothesis = summary.get('hypothesis_or_goal', 'N/A')
    methodology = summary.get('methodology_summary', 'N/A').replace('\n', '\n  ')
    results = summary.get('key_results', 'N/A').replace('\n', '\n  ')
    content = f"""
# ==============================================================================
# AI-Generated Machine Learning Task Benchmark
# ==============================================================================

## 1. Source Paper Information
- **Title:** {task_data.get('source_paper', {}).get('title', 'N/A')}
- **URL:** {task_data.get('source_paper', {}).get('url', 'N/A')}

## 2. Scientific Context (AI-Summarized)
### Background
{background}

### Research Goal or Hypothesis
{hypothesis}

### Core Methodology
  {methodology}

### Key Results
  {results}

## 3. Machine Learning Task Definition
- **Task Type:** {task_data.get('ml_task', {}).get('task_type', 'N/A')}
- **Problem Statement:** {task_data.get('ml_task', {}).get('problem_statement', 'N/A')}
- **Primary Metric:** {task_data.get('evaluation_criteria', {}).get('primary_metric', 'N/A')}

## 4. Data Profile (Auto-Analyzed)
- **Raw Data Path:** {task_data.get('data_sources', {}).get('raw_data_path', 'N/A')}
- **Primary Data Format:** {task_data.get('data_profile', {}).get('primary_format', 'N/A')}
- **File Format Statistics:** {json.dumps(task_data.get('data_profile', {}).get('file_formats', {}), indent=2)}

## 5. How to Get Started
1.  **Install Dependencies**: `requirements.txt` has been intelligently generated based on the detected data formats. Run `pip install -r requirements.txt`.
2.  **Review AI-Generated Code**: Check `train.py` and `score.py`. They are custom-generated to handle the '{task_data.get('data_profile', {}).get('primary_format', 'N/A')}' format. Pay special attention to `# TODO:` comments.
3.  **Train the Model**: `python train.py`
4.  **Evaluate the Model**: `python score.py`
"""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)


def generate_templates(benchmark_path: str, task_data: dict):

    print("  > Intelligently generating requirements.txt...")
    base_reqs = {"pandas", "scikit-learn", "joblib"}
    data_profile = task_data.get('data_profile', {})
    detected_formats = data_profile.get('file_formats', {}).keys()
    FORMAT_TO_LIBS = {'.nc': ['xarray', 'netCDF4'], '.h5': ['h5py'], '.hdf5': ['h5py'], '.parquet': ['pyarrow'], '.grib': ['cfgrib', 'eccodes'], '.grib2': ['cfgrib', 'eccodes']}
    for fmt, libs in FORMAT_TO_LIBS.items():
        if fmt in detected_formats:
            for lib in libs: base_reqs.add(lib)
    key_libs = task_data.get('model_details', {}).get('key_libraries_mentioned', [])
    for lib in key_libs:
        lib_lower = lib.lower()
        if 'torch' in lib_lower: base_reqs.add('torch'); base_reqs.add('torchvision')
        elif 'tensorflow' in lib_lower or 'keras' in lib_lower: base_reqs.add('tensorflow')
        elif 'hugging' in lib_lower: base_reqs.add('transformers'); base_reqs.add('datasets')
        else: base_reqs.add(lib_lower)
    req_content = "\n".join(sorted(list(base_reqs)))
    with open(os.path.join(benchmark_path, 'requirements.txt'), 'w', encoding='utf-8') as f:
        f.write(req_content)

def main(json_file_path: str):

    print(f"--- Reading detailed task definition file: {json_file_path} ---")
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            task_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{json_file_path}' not found. Please run Auto_ML.py first."); return
    except json.JSONDecodeError:
        print(f"Error: File '{json_file_path}' is not a valid JSON file."); return
    DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-353a88a777bd4c598f17b2923677e100")
    if not DEEPSEEK_API_KEY:
        print(
            "Error: API key not found. Please set the DEEPSEEK_API_KEY environment variable or hardcode it in the script.")
        sys.exit(1)

    try:

        llm_provider = DeepSeekLLMProvider(api_key=DEEPSEEK_API_KEY, model="deepseek-chat")
    except ValueError as e:
        print(e);
        return
    title = task_data.get('source_paper', {}).get('title', 'untitled')
    sanitized_title = re.sub(r'[\s:/\\]+', '_', title.lower())
    benchmark_path = f"benchmark_{sanitized_title[:50]}"
    if os.path.exists(benchmark_path):
        print(f"Warning: Directory '{benchmark_path}' already exists. Its contents will be overwritten.")
    else:
        print(f"--- Creating benchmark directory: {benchmark_path} ---")
    os.makedirs(benchmark_path, exist_ok=True)
    print("--- Dynamically generating project scripts... ---")
    train_py_content = generate_code_with_llm(task_data, llm_provider, 'train')
    with open(os.path.join(benchmark_path, 'train.py'), 'w', encoding='utf-8') as f:
        f.write(train_py_content)
    score_py_content = generate_code_with_llm(task_data, llm_provider, 'score')
    with open(os.path.join(benchmark_path, 'score.py'), 'w', encoding='utf-8') as f:
        f.write(score_py_content)
    generate_description_txt(task_data, os.path.join(benchmark_path, 'description.txt'))
    generate_templates(benchmark_path, task_data)
    shutil.copy(json_file_path, os.path.join(benchmark_path, 'source_task_definition.json'))
    print("\n" + "="*50)
    print("Dynamically generated benchmark project created successfully!")
    print(f"Project Path: {benchmark_path}")


if __name__ == "__main__":
    os.environ["DEEPSEEK_API_KEY"] = "sk-353a88a777bd4c598f17b2923677e100"
    parser = argparse.ArgumentParser(description="Dynamically generates an ML benchmark project from a detailed task definition JSON.")
    parser.add_argument("json_file", type=str, nargs='?', default=None, help="Path to the task definition JSON file generated by Auto_ML.py.")
    args = parser.parse_args()
    if not args.json_file:
        print("Error: Please provide the path to a 'generated_mle_task.json' file.")
        print("Example: python benchmark_generator_v2.py batch_output/paper_some_title/generated_mle_task.json")
        sys.exit(1)
    main(args.json_file)