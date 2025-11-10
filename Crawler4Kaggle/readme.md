### 安装需要的库
```
pip install selenium webdriver-manager beautifulsoup4
```

### 作为主程序直接运行
脚本会使用预设的竞赛ID执行抓取，并将结果保存为 `competition_data.json`。

1.  **设置目标竞赛**:
    打开 `Crawler1.py` 文件，在文件末尾找到并修改 `competition_id` 变量的值。
    ```python
    # ...
    if __name__ == "__main__":
        # 在这里更改为您想爬取的任何Kaggle竞赛ID
        competition_id = "titanic"
        # ...
    ```

2.  **运行脚本**:
    在项目文件夹的终端中，执行以下命令：
    ```bash
    python Crawler1.py
    ```
    执行完毕后，会在当前目录下生成 `competition_data.json` 文件。

### 作为函数导入并调用

可以将此脚本作为一个模块导入项目中，以编程方式控制抓取过程。

1.  **导入函数**:
    从 `Crawler1.py` 导入核心函数。
    ```python
    from Crawler1 import extract_and_save_competition_data
    ```

2.  **调用函数**:
    您可以指定任何竞赛ID和输出文件名来调用该函数，两个参数都接受string格式输入。它会自动执行抓取和保存，并返回一个包含结果的字典。
    ```python

    # 定义目标竞赛
    target_competition = "titanic"
    output_file = f"{target_competition}_data.json"

    # 调用函数进行抓取
    extracted_data = extract_and_save_competition_data(
        competition_name=target_competition,
        output_filename=output_file
    )

    # 接着处理返回的数据
    if "error" not in extracted_data:
        print(f"成功提取竞赛 '{target_competition}' 的数据，已保存至 {output_file}。")
        # print(extracted_data)
    else:
        print(f"提取过程中发生错误: {extracted_data['error']}")
    ```

### 说明

由于Kaggle网站似乎是动态加载的，直接用requests只能获得一些占位符，所以使用selenium来直接实例化一个浏览器页面来加载渲染结果，目前的反反爬设置如下：
```python
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    #以“无头模式”运行浏览器,加快运行速度
    options.add_argument("--headless")
    #手动设置User-Agent，可以扩展成User-Agent池
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36")
    #关闭 enable-automation 这个开关
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    #关闭浏览器自动化扩展
    options.add_experimental_option('useAutomationExtension', False)
```

对于关键信息的抓取，目前是提取成纯文本，然后用keyword来切分，这可能鲁棒性不够强，如果正文中出现了关键词可能导致切分失误，如果需要可能还需要优化。
