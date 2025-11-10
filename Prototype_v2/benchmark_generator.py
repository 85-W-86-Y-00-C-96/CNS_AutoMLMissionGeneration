import os
import json
import re
import shutil
import argparse
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
import glob
from tqdm import tqdm

try:
    from Auto_ML import DeepSeekLLMProvider
except ImportError:
    print("Error: Could not import DeepSeekLLMProvider from 'Auto_ML.py'.")
    print("Please ensure 'Auto_ML.py' and 'benchmark_generator_v2.py' are in the same folder.")
    exit(1)

try:
    import xarray as xr

    XARRAY_AVAILABLE = True
except ImportError:
    XARRAY_AVAILABLE = False

try:
    import pyarrow

    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False


def generate_code_with_llm(task_data: dict, llm_provider: DeepSeekLLMProvider, file_type: str) -> str:
    data_profile = task_data.get('data_profile', {})
    context = f"""
    # Task Context:
    - Task Type: {task_data.get('ml_task', {}).get('task_type', 'N/A')}
    - Problem Statement: {task_data.get('ml_task', {}).get('problem_statement', 'N/A')}
    - Input Description: {task_data.get('ml_task', {}).get('input_description', 'N/A')}
    - Output Description: {task_data.get('ml_task', {}).get('output_description', 'N/A')}
    # Data Details:
    - Format: {task_data.get('data_details', {}).get('format', 'N/A')}
    - Structure: {task_data.get('data_details', {}).get('structure_description', 'N/A')}
    - Label Logic: {task_data.get('data_details', {}).get('target_column_or_logic', 'N/A')}
    # Model Details:
    - Architecture Family: {task_data.get('model_details', {}).get('architecture_family', 'N/A')}
    - Key Libraries: {', '.join(task_data.get('model_details', {}).get('key_libraries_mentioned', []))}
    # Evaluation Criteria:
    - Primary Metric: {task_data.get('evaluation_criteria', {}).get('primary_metric', 'N/A')}
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
           - For '.nc', use the `xarray` library, and show how to load multiple files from a directory using `xr.open_mfdataset`.
           - For '.h5' or '.hdf5', use `h5py` or `pandas.HDFStore`.
           - For '.csv', use `pandas.read_csv`.
           - For image formats, suggest using `torchvision.datasets.ImageFolder` or `tf.keras.utils.image_dataset_from_directory`.
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
        1. Include logic to load the saved model and test data. The test data loading logic **MUST** be able to handle the '{data_profile.get('primary_format', 'unknown')}' format, likely by loading multiple files from the test directory.
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


def _split_image_data(source_dir, train_path, test_path):
    print("  > Applying stratified split for image data...")
    image_paths = []
    labels = []
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    for root, _, files in os.walk(source_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                image_path = os.path.join(root, file)
                label = os.path.basename(root)
                image_paths.append(image_path)
                labels.append(label)
    if not image_paths:
        raise ValueError("No image files found in the source directory.")
    train_files, test_files, _, _ = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    def copy_files(file_list, dest_root):
        for file_path in tqdm(file_list, desc=f"Copying to {os.path.basename(dest_root)}"):
            label = os.path.basename(os.path.dirname(file_path))
            dest_dir = os.path.join(dest_root, label)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy(file_path, dest_dir)

    copy_files(train_files, train_path)
    copy_files(test_files, test_path)
    print(
        f"  > Successfully split {len(image_paths)} images into {len(train_files)} training and {len(test_files)} testing samples.")


def split_and_organize_data(task_data: dict, benchmark_path: str):
    print("--- Attempting to intelligently split and organize data... ---")

    source_data_dir = task_data.get('data_sources', {}).get('raw_data_path')
    if not source_data_dir or not os.path.exists(source_data_dir):
        print("  > Warning: Source data directory not found. Skipping data splitting.")
        return

    train_path = os.path.join(benchmark_path, 'data', 'train')
    test_path = os.path.join(benchmark_path, 'data', 'test')
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    train_dirs = [d for d in ['train', 'training'] if os.path.isdir(os.path.join(source_data_dir, d))]
    test_dirs = [d for d in ['test', 'testing', 'validation', 'val'] if os.path.isdir(os.path.join(source_data_dir, d))]
    if train_dirs and test_dirs:
        print(f"  > Detected pre-split directories: '{train_dirs[0]}' and '{test_dirs[0]}'. Copying files...")
        try:
            shutil.copytree(os.path.join(source_data_dir, train_dirs[0]), train_path, dirs_exist_ok=True)
            shutil.copytree(os.path.join(source_data_dir, test_dirs[0]), test_path, dirs_exist_ok=True)
            print("  > Successfully copied pre-split data.")
            return
        except Exception as e:
            print(f"  > Error copying pre-split directories: {e}")

    primary_format = task_data.get('data_profile', {}).get('primary_format')
    image_formats = ['.jpg', '.jpeg', '.png', '.bmp']
    table_formats = ['.csv', '.parquet', '.tsv']
    is_image_dataset = primary_format in image_formats

    print(f"  > Primary data format detected: '{primary_format}'. Applying specific strategy...")

    try:
        files_of_primary_format = glob.glob(os.path.join(source_data_dir, '**', f'*{primary_format}'), recursive=True)
        if not files_of_primary_format:
            raise FileNotFoundError(f"No '{primary_format}' files found in the source directory.")

        MULTI_FILE_THRESHOLD = 50
        is_multi_file_dataset = len(files_of_primary_format) > MULTI_FILE_THRESHOLD

        if is_multi_file_dataset:
            print(
                f"  > Detected multi-file dataset ({len(files_of_primary_format)} files). Splitting file list chronologically.")
            files_of_primary_format.sort()
            split_index = int(len(files_of_primary_format) * 0.8)
            train_files = files_of_primary_format[:split_index]
            test_files = files_of_primary_format[split_index:]

            for file_path in tqdm(train_files, desc="Copying train files"):
                shutil.copy(file_path, train_path)
            for file_path in tqdm(test_files, desc="Copying test files"):
                shutil.copy(file_path, test_path)
            print(f"  > Successfully split files: {len(train_files)} for training, {len(test_files)} for testing.")
            return

        else:
            print(
                f"  > Detected few-files dataset ({len(files_of_primary_format)} files). Merging up to 10 largest files before splitting.")

            top_files = sorted(files_of_primary_format, key=os.path.getsize, reverse=True)[:10]
            print(f"  > Identified {len(top_files)} largest files for merging.")

            if primary_format in table_formats:
                TABLE_READERS = {'.csv': pd.read_csv, '.parquet': pd.read_parquet,
                                 '.tsv': lambda f: pd.read_csv(f, sep='\t')}
                if primary_format == '.parquet' and not PYARROW_AVAILABLE:
                    raise ImportError("pyarrow is required to read .parquet files. Please run: pip install pyarrow")

                dfs = [TABLE_READERS[primary_format](f) for f in top_files]
                merged_df = pd.concat(dfs, ignore_index=True)

                train_df, test_df = train_test_split(merged_df, test_size=0.2, random_state=42)

                output_train_path = os.path.join(train_path, f'train{primary_format}')
                output_test_path = os.path.join(test_path, f'test{primary_format}')
                if primary_format == '.csv':
                    train_df.to_csv(output_train_path, index=False)
                    test_df.to_csv(output_test_path, index=False)
                elif primary_format == '.parquet':
                    train_df.to_parquet(output_train_path, index=False)
                    test_df.to_parquet(output_test_path, index=False)
                elif primary_format == '.tsv':
                    train_df.to_csv(output_train_path, sep='\t', index=False)
                    test_df.to_csv(output_test_path, sep='\t', index=False)
                print(f"  > Successfully merged and performed random split.")
                return

            elif primary_format == '.nc':
                if not XARRAY_AVAILABLE: raise ImportError(
                    "xarray and netCDF4 are required to process .nc files. Please run: pip install xarray netCDF4")
                with xr.open_mfdataset(top_files) as merged_ds:
                    if 'time' in merged_ds.dims and len(merged_ds['time']) > 1:
                        print("  > Performing time-based split (80/20) on merged dataset...")
                        split_index = int(len(merged_ds['time']) * 0.8)
                        train_ds = merged_ds.isel(time=slice(0, split_index))
                        test_ds = merged_ds.isel(time=slice(split_index, None))
                    else:
                        print(
                            "  > No 'time' dimension found. Performing random split on first dimension of merged dataset...")
                        first_dim = list(merged_ds.dims)[0]
                        idx = list(range(merged_ds.dims[first_dim]))
                        train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42)
                        train_ds = merged_ds.isel({first_dim: sorted(train_idx)})
                        test_ds = merged_ds.isel({first_dim: sorted(test_idx)})
                train_ds.to_netcdf(os.path.join(train_path, 'train.nc'))
                test_ds.to_netcdf(os.path.join(test_path, 'test.nc'))
                print("  > Successfully merged and split NetCDF into train.nc and test.nc.")
                return

            elif is_image_dataset:
                _split_image_data(source_data_dir, train_path, test_path)
                return

    except Exception as e:
        print(f"  > Failed to apply automatic splitting strategy: {e}")

    print("  > Could not apply any automatic splitting strategy.")
    print("  > Empty 'train' and 'test' directories are created. Please organize your data manually.")


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
    base_reqs = {"pandas", "scikit-learn", "joblib", "tqdm"}
    data_profile = task_data.get('data_profile', {})
    detected_formats = data_profile.get('file_formats', {}).keys()
    FORMAT_TO_LIBS = {'.nc': ['xarray', 'netCDF4'], '.h5': ['h5py'], '.hdf5': ['h5py'], '.parquet': ['pyarrow'],
                      '.grib': ['cfgrib', 'eccodes'], '.grib2': ['cfgrib', 'eccodes']}
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
    print(f"--- Reading detailed task definition file: {json_file_path} ---")
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            task_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{json_file_path}' not found. Please run Auto_ML.py first.");
        return
    except json.JSONDecodeError:
        print(f"Error: File '{json_file_path}' is not a valid JSON file.");
        return

    DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
    if not DEEPSEEK_API_KEY:
        print("Error: API key not found. Please set the DEEPSEEK_API_KEY environment variable.")
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

    split_and_organize_data(task_data, benchmark_path)

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

    print("\n" + "=" * 50)
    print("Dynamically generated benchmark project created successfully!")



if __name__ == "__main__":
    os.environ["DEEPSEEK_API_KEY"] = "sk-353a88a777bd4c598f17b2923677e100"
    parser = argparse.ArgumentParser(
        description="Dynamically generates an ML benchmark project from a detailed task definition JSON.")
    parser.add_argument("json_file", type=str, nargs='?', default=None,
                        help="Path to the task definition JSON file generated by Auto_ML.py.")
    args = parser.parse_args()
    if not args.json_file:
        print("Error: Please provide the path to a 'generated_mle_task.json' file.")
        print("Example: python benchmark_generator_v2.py batch_output/paper_some_title/generated_mle_task.json")
        sys.exit(1)
    main(args.json_file)

