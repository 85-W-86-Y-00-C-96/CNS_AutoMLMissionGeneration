import time
import json
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup


def extract_and_save_competition_data(competition_name: str, output_filename="competition_data.json"):
    base_url = "https://www.kaggle.com/competitions/"
    overview_url = f"{base_url}{competition_name}"

    data_to_save = {
        "description": "Extraction failed.",
        "background": "Extraction failed.",
        "evaluation": "Extraction failed."
    }
    #都是一些反反爬的设置选项，目前我试了一下应该已经够用了，如果之后遇到更多问题可以加一点类似time.sleep(random.uniform(1.5, 4.5))还有模拟鼠标移动之类的，目前好像不需要
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    options.add_argument("--headless")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)

    driver = None
    try:
        driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
        wait = WebDriverWait(driver, 20)

        print(f"Navigating to Overview page: {overview_url}")
        driver.get(overview_url)

        #这是我发现kaggle页面底部有Cookie按钮，可能会干扰浏览，所以让脚本自动点一下，不过好像没影响哈哈哈哈
        try:
            cookie_button = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'OK, Got it.')]"))
            )
            cookie_button.click()
            print("Cookie banner handled.")
            time.sleep(2)
        except TimeoutException:
            print("Cookie banner not found.")

        print("Waiting for the main content area to load...")
        main_content_container = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "main, [role='main']"))
        )
        print("Main content area loaded.")

        html_content = main_content_container.get_attribute('innerHTML')
        soup = BeautifulSoup(html_content, 'html.parser')
        full_text = soup.get_text(separator=' ', strip=True)

        desc_kw = "Description"
        back_kw = "Background"
        eval_kw = "Evaluation"
        #因为不同的竞赛后面一个section可能不同，我就随便列了一些，如果有的竞赛后面是其他的关键词，可以在这里添加
        end_keywords = ["Timeline", "Prizes & Awards", "Code", "Data"]
        unwanted_prefix = "link keyboard_arrow_up"

        # 提取并清洗 Description
        try:
            start_index = full_text.index(desc_kw) + len(desc_kw)
            end_index = full_text.index(back_kw, start_index)
            raw_text = full_text[start_index:end_index].strip()
            cleaned_text = raw_text.removeprefix(unwanted_prefix).strip()
            data_to_save["description"] = cleaned_text
            print("Successfully extracted and cleaned 'Description'.")
        except ValueError:
            data_to_save["description"] = "Could not find markers to slice 'Description'."
            print("Failed to extract 'Description'.")

        # 提取并清洗 Background
        try:
            start_index = full_text.index(back_kw) + len(back_kw)
            end_index = full_text.index(eval_kw, start_index)
            raw_text = full_text[start_index:end_index].strip()
            cleaned_text = raw_text.removeprefix(unwanted_prefix).strip()
            data_to_save["background"] = cleaned_text
            print("Successfully extracted and cleaned 'Background'.")
        except ValueError:
            data_to_save["background"] = "Could not find markers to slice 'Background'."
            print("Failed to extract 'Background'.")

        # 提取并清洗 Evaluation
        try:
            start_index = full_text.index(eval_kw) + len(eval_kw)
            end_index = -1
            for keyword in end_keywords:
                try:
                    idx = full_text.find(keyword, start_index)
                    if idx != -1 and (end_index == -1 or idx < end_index):
                        end_index = idx
                except ValueError:
                    continue

            if end_index != -1:
                raw_text = full_text[start_index:end_index].strip()
            else:
                raw_text = full_text[start_index:].strip()
            cleaned_text = raw_text.removeprefix(unwanted_prefix).strip()
            data_to_save["evaluation"] = cleaned_text
            print("Successfully extracted and cleaned 'Evaluation'.")
        except ValueError:
            data_to_save["evaluation"] = "Could not find markers to slice 'Evaluation'."
            print("Failed to extract 'Evaluation'.")
    #上面三个部分的抓取基本上是复制粘贴，所以如果需要其它信息扩展性还是比较强的
    #不过目前可能鲁棒性一般，因为是纯文本切割的
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        data_to_save["error"] = str(e)
    finally:
        if driver:
            print("Task finished. Browser will close.")
            driver.quit()

    try:
        with open(output_filename, 'w', encoding='utf-8') as json_file:
            json.dump(data_to_save, json_file, ensure_ascii=False, indent=4)
        print(f"\nClean data successfully saved to '{output_filename}'")
    except Exception as e:
        print(f"\nFailed to save data to JSON file. Error: {e}")

    return data_to_save


if __name__ == "__main__":
    competition_id = "asap-aes"
    print(f"--- Starting Final Extractor for: '{competition_id}' ---")

    extracted_data = extract_and_save_competition_data(competition_id, "competition_data.json")

    print("\n--- Final Cleaned Data ---")
    print(json.dumps(extracted_data, indent=2))