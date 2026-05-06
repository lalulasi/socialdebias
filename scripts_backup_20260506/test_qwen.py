import os
import dashscope
from dashscope import Generation

def test_qwen_api_intl():
    """
    测试调用阿里云百炼 (DashScope) 国际站 (新加坡) 节点的 Qwen 模型。
    """
    # 1. 获取环境变量中的 API Key
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    
    if not api_key:
        print("错误：未找到 DASHSCOPE_API_KEY 环境变量！")
        return

    # 设置 DashScope API Key
    dashscope.api_key = api_key
    
    # 2. 【核心修复】将网关地址指向新加坡节点
    # 根据你提供的截图，将 Base URL 改为国际站地址
    dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'
    
    print(f"成功加载 API Key (前 5 位): {api_key[:5]}...")
    print(f"当前请求网关: {dashscope.base_http_api_url}")

    # 3. 准备测试问题 (模拟第四章的提示词工程)
    prompt_text = (
        "你是一个专业的新闻编辑。请将下面这句带有强烈煽动性情绪的假新闻，"
        "改写成一句语气中立、客观的陈述句，不要改变其核心事实。\n\n"
        "原始新闻：气死我了！那个无耻的政客昨天晚上居然偷偷把国家的钱都卷走跑路了！"
    )

    print("\n--- 发送请求给 Qwen ---")
    print(f"使用的模型: qwen-plus")
    print("-" * 25)

    try:
        # 4. 调用 API
        response = Generation.call(
            model='qwen-plus', 
            prompt=prompt_text,
            result_format='message'
        )

        # 5. 处理返回结果
        if response.status_code == 200:
            print("\n--- Qwen 返回结果 ---")
            generated_text = response.output.choices[0].message.content
            print(generated_text)
            print("-" * 25)
            print("测试成功！API 调用一切正常。")
        else:
            print("\n--- API 调用失败 ---")
            print(f"Request ID: {response.request_id}")
            print(f"Status Code: {response.status_code}")
            print(f"Error Code: {response.code}")
            print(f"Error Message: {response.message}")

    except Exception as e:
        print(f"\n发生异常: {e}")

if __name__ == "__main__":
    test_qwen_api_intl()