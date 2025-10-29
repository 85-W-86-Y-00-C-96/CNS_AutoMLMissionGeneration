import os
from deepseek_ai import DeepSeekAI

class DeepSeekLLMProvider:
    def __init__(self, model="deepseek-chat"):
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key: raise ValueError("错误：请先设置 DEEPSEEK_API_KEY 环境变量。")
        self.client = DeepSeekAI(api_key=api_key)
        self.model = model
        print(f"DeepSeekLLMProvider (v0.0.1) 初始化成功！")

    def query(self, prompt: str, context: str = "") -> str:
        print(f"\n--- 正在向 DeepSeek API (模型: {self.model}) 发送请求... ---")
        try:
            messages = [{"role": "system", "content": prompt}, {"role": "user", "content": context}]
            response = self.client.chat.completions.create(model=self.model, messages=messages)
            result = response.choices[0].message.content
            print("--- 成功接收到 API 响应 ---")
            return result
        except Exception as e:
            print(f"!!! 调用 DeepSeek API 时出错: {e} !!!")
            return f"Error: {e}"