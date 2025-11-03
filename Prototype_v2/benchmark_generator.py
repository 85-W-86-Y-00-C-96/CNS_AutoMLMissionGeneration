# 文件名: benchmark_generator_v2.py
import os
import json
import re
import shutil
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import glob
from Auto_ML import DeepSeekLLMProvider



# ==============================================================================
# 核心功能：LLM驱动的代码生成
# ==============================================================================
def generate_code_with_llm(task_data: dict, llm_provider: DeepSeekLLMProvider, file_type: str) -> str:
    data_profile = task_data.get('data_profile', {})
    context = f"""
    # 任务背景:
    - 任务类型: {task_data.get('ml_task', {}).get('task_type', 'N/A')}
    - 问题陈述: {task_data.get('ml_task', {}).get('problem_statement', 'N/A')}
    - 模型输入: {task_data.get('ml_task', {}).get('input_description', 'N/A')}
    - 模型输出: {task_data.get('ml_task', {}).get('output_description', 'N/A')}

    # 数据细节:
    - 格式: {task_data.get('data_details', {}).get('format', 'N/A')}
    - 结构: {task_data.get('data_details', {}).get('structure_description', 'N/A')}
    - 标签逻辑: {task_data.get('data_details', {}).get('target_column_or_logic', 'N/A')}

    # 模型细节:
    - 架构家族: {task_data.get('model_details', {}).get('architecture_family', 'N/A')}
    - 关键库: {', '.join(task_data.get('model_details', {}).get('key_libraries_mentioned', []))}

    # 评估标准:
    - 主要指标: {task_data.get('evaluation_criteria', {}).get('primary_metric', 'N/A')}

    # 已检测的数据画像:
    - 主要数据格式: {data_profile.get('primary_format', 'unknown')}
    - 所有检测到的文件格式: {list(data_profile.get('file_formats', {}).keys())}
    - 原始数据位于 'data/' 目录。
    """

    if file_type == 'train':
        system_prompt = f"""
        你是一位资深的Python机器学习工程师。根据下面提供的任务背景和数据画像，编写一个完整、可运行的 `train.py` 脚本。
        要求：
        1. 你 **必须** 编写能够原生处理 '{data_profile.get('primary_format', 'unknown')}' 格式的数据加载代码。
           - 如果是 '.nc'，使用 `xarray` 库。
           - 如果是 '.h5' 或 '.hdf5'，使用 `h5py` 或 `pandas.HDFStore`。
           - 如果是 '.csv'，使用 `pandas.read_csv`。
           - 对于其他格式，请选择最合适的库。
        2. 根据“模型架构家族”和“关键库”，选择一个简单但非常合适的模型作为起点（例如，如果库是scikit-learn且任务是分类，使用LogisticRegression或RandomForestClassifier；如果库是PyTorch且任务是图像分类，使用一个简单的CNN）。
        3. 包含完整的模型训练和保存逻辑。
        4. 在需要用户指定确切文件名或变量名的地方，留下清晰的 # TODO: 注释。
        你的输出必须是纯Python代码，不包含任何Markdown标记（例如 ```python）或任何解释性文字。
        """
    elif file_type == 'score':
        system_prompt = f"""
        你是一位资深的Python机器学习工程师。根据下面提供的任务背景和数据画像，编写一个完整、可运行的 `score.py` 脚本。
        要求：
        1. 包含加载已保存模型和测试数据的逻辑。测试数据加载逻辑 **必须** 能够处理 '{data_profile.get('primary_format', 'unknown')}' 格式。
        2. 根据指定的“主要指标”('{task_data.get('evaluation_criteria', {}).get('primary_metric', 'N/A')}')，使用合适的库（例如 scikit-learn.metrics）来计算分数。
        3. 以清晰的JSON格式打印最终的评分结果。
        4. 包含命令行参数解析，允许用户指定模型路径和测试数据路径。
        你的输出必须是纯Python代码，不包含任何Markdown标记（例如 ```python）或任何解释性文字。
        """
    else:
        return "# Invalid file type requested"

    print(f"  > 正在为 {file_type}.py 向LLM请求生成代码...")
    generated_code = llm_provider.query(system_prompt, context)
    clean_code = generated_code.strip()
    if clean_code.startswith("```python"):
        clean_code = clean_code[len("```python"):].strip()
    if clean_code.endswith("```"):
        clean_code = clean_code[:-len("```")].strip()
    return clean_code


def generate_description_txt(task_data: dict, file_path: str):
    print("  > 生成 description.txt...")
    content = f"""
# ==============================================================================
# 自动生成的机器学习任务基准 (AI-Generated)
# ==============================================================================

## 源论文信息
- **标题:** {task_data.get('source_paper', {}).get('title', 'N/A')}
- **URL:** {task_data.get('source_paper', {}).get('url', 'N/A')}

## 机器学习任务定义
- **任务类型:** {task_data.get('ml_task', {}).get('task_type', 'N/A')}
- **问题陈述:** {task_data.get('ml_task', {}).get('problem_statement', 'N/A')}

## 数据画像 (自动分析)
- **原始数据路径:** {task_data.get('data_sources', {}).get('raw_data_path', 'N/A')}
- **主要数据格式:** {task_data.get('data_profile', {}).get('primary_format', 'N/A')}
- **文件格式统计:** {json.dumps(task_data.get('data_profile', {}).get('file_formats', {}), indent=2)}

## 如何开始
1.  **检查并安装依赖**: `requirements.txt` 已根据检测到的数据格式智能生成。运行 `pip install -r requirements.txt`。
2.  **检查AI生成的代码**: 查看 `train.py` 和 `score.py`。它们是为处理 '{task_data.get('data_profile', {}).get('primary_format', 'N/A')}' 格式而定制的。特别注意标记为 # TODO: 的部分。
3.  **训练模型**: `python train.py`
4.  **评估模型**: `python score.py`
"""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)


def generate_templates(benchmark_path: str, task_data: dict):
    print("  > 生成 requirements.txt...")
    base_reqs = {"pandas", "scikit-learn", "joblib"}
    data_profile = task_data.get('data_profile', {})
    detected_formats = data_profile.get('file_formats', {}).keys()

    FORMAT_TO_LIBS = {
        '.nc': ['xarray', 'netCDF4'],
        '.h5': ['h5py'], '.hdf5': ['h5py'],
        '.parquet': ['pyarrow'],
        '.grib': ['cfgrib', 'eccodes'], '.grib2': ['cfgrib', 'eccodes']
    }
    for fmt, libs in FORMAT_TO_LIBS.items():
        if fmt in detected_formats:
            for lib in libs: base_reqs.add(lib)

    key_libs = task_data.get('model_details', {}).get('key_libraries_mentioned', [])
    for lib in key_libs:
        lib_lower = lib.lower()
        if 'torch' in lib_lower:
            base_reqs.add('torch'); base_reqs.add('torchvision')
        elif 'tensorflow' in lib_lower or 'keras' in lib_lower:
            base_reqs.add('tensorflow')
        elif 'hugging' in lib_lower:
            base_reqs.add('transformers'); base_reqs.add('datasets')
        else:
            base_reqs.add(lib_lower)

    req_content = "\n".join(sorted(list(base_reqs)))
    with open(os.path.join(benchmark_path, 'requirements.txt'), 'w', encoding='utf-8') as f:
        f.write(req_content)


def main(json_file_path: str):
    print(f"--- 读取详细任务定义文件: {json_file_path} ---")
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            task_data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 文件 '{json_file_path}' 未找到。请先运行 Auto_ML.py 生成该文件。");
        return
    except json.JSONDecodeError:
        print(f"错误: 文件 '{json_file_path}' 不是一个有效的JSON文件。");
        return

    try:
        llm_provider = DeepSeekLLMProvider(model="deepseek-chat")
    except ValueError as e:
        print(e);
        return

    title = task_data.get('source_paper', {}).get('title', 'untitled')
    sanitized_title = re.sub(r'[\s:/\\]+', '_', title.lower())
    benchmark_path = f"benchmark_{sanitized_title[:50]}"
    if os.path.exists(benchmark_path):
        print(f"警告: 文件夹 '{benchmark_path}' 已存在。其中的内容将被覆盖。")
    else:
        print(f"--- 创建基准文件夹: {benchmark_path} ---")
    os.makedirs(benchmark_path, exist_ok=True)

    print("--- 正在动态生成项目脚本 ---")

    train_py_content = generate_code_with_llm(task_data, llm_provider, 'train')
    with open(os.path.join(benchmark_path, 'train.py'), 'w', encoding='utf-8') as f:
        f.write(train_py_content)

    score_py_content = generate_code_with_llm(task_data, llm_provider, 'score')
    with open(os.path.join(benchmark_path, 'score.py'), 'w', encoding='utf-8') as f:
        f.write(score_py_content)

    generate_description_txt(task_data, os.path.join(benchmark_path, 'description.txt'))
    generate_templates(benchmark_path, task_data)

    shutil.copy(json_file_path, os.path.join(benchmark_path, 'source_task_definition.json'))

    print("\n" + "=" * 50)
    print("动态生成的基准项目已创建成功!")
    print(f"项目路径: {benchmark_path}")




if __name__ == "__main__":
    os.environ["DEEPSEEK_API_KEY"] = "sk-353a88a777bd4c598f17b2923677e100"
    parser = argparse.ArgumentParser(description="根据详细任务定义JSON，动态生成一个ML基准项目。")
    parser.add_argument("json_file", type=str, default="generated_mle_task.json", nargs='?',
                        help="任务定义JSON文件路径。")
    args = parser.parse_args()
    main(args.json_file)