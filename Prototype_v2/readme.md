## 使用指南



### 步骤 1: 分析论文 (`Auto_ML.py`)

此脚本负责从URL到生成结构化分析报告（`.json`文件）和下载数据的全过程。

1.  **创建 `urls.txt` 文件**
    在项目根目录下，创建一个名为 `urls.txt` 的文本文件。

2.  **添加论文URL**
    在 `urls.txt` 文件中，每行粘贴一个希望分析的论文的URL。例如：
    ```
    https://www.nature.com/articles/s41467-023-44101-5
    https://www.nature.com/articles/s41586-024-07113-6
    ```

3.  **运行分析脚本**
    在终端中运行：
    ```bash
    python Auto_ML.py
    ```
    脚本将开始批量处理 `urls.txt` 中的每个链接。处理完成后会得到一个 `batch_output/` 文件夹，其中包含了每篇论文的独立分析结果。

    **输出结构示例:**
    ```
    batch_output/
    └── paper_title_of_paper_one/
        ├── data/                  <-- 下载的数据
        └── generated_mle_task.json  <-- 详细的分析报告
    ```
    在脚本的输出日志中，会清晰地打印出每个生成的 `.json` 文件的路径。

### 步骤 2: 生成Benchmark项目 (`benchmark_generator.py`)

此脚本读取上一步生成的 `.json` 文件，并动态生成一个完整的、可运行的ML项目文件夹。

1.  **选择一个分析报告**
    从 `batch_output/` 文件夹中，选择一个您想要生成代码的 `generated_mle_task.json` 文件，并**复制其完整路径**。

2.  **运行生成器脚本**
    在终端中，运行 `benchmark_generator.py`，并将复制的路径作为命令行参数传入。

    **示例:**
    ```bash
    python benchmark_generator_v2.py batch_output/paper_benchmark_dataset_and_deep_learning_method_for_/generated_mle_task.json
    ```
    *提示：如果路径中包含空格，请用双引号将其包裹起来。*

3.  **探索生成的项目**
    脚本执行完毕后，会在项目根目录创建一个新的 `benchmark_.../` 文件夹。进入该文件夹，您会看到：
    ```
    benchmark_.../
    ├── data/               <-- 原始数据的符号链接或副本 (待实现)
    ├── description.txt     <-- 包含科学摘要和任务详情的说明文档
    ├── requirements.txt    <-- 智能生成的环境依赖
    ├── score.py            <-- 评估脚本
    ├── train.py            <-- 训练脚本
    └── source_task_definition.json
    ```

---

## 10.10benchmark_generator更新数据集划分逻辑

训练集和测试集的生成采用了一种多策略决策流程。它首先会优先检查数据源中是否已存在作者预先划分好的train和test文件夹，并直接复制。如果没有预划分，它会根据数据文件的数量进行判断：对于包含大量文件（>50）的数据集，它会划分文件列表本身，通常按文件名排序后将前80%的文件复制到训练集，后20%到测试集；对于文件较少的数据集，它则会合并最多10个最大文件的内容，然后对合并后的数据项（如行或时间步）进行80/20的划分。
