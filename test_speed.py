import time
from openai import OpenAI
from sympy import true

# ========= 配置 =========
QWEN_API_KEY = "sk-cce714d1a39f43edad80e58a5e066973"
DEEPSEEK_API_KEY = "sk-65825f6e05a0409c907548359045ef54"

# DashScope（Qwen）
qwen_client = OpenAI(
    api_key=QWEN_API_KEY,
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)

# DeepSeek 官方
deepseek_client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com/v1"
)

# ========= 测试 prompt =========
PROMPT = """
请将以下新闻改写成更简洁版本：

日本东京近日宣布，将进一步推动AI产业发展，
政府计划增加对初创企业的资金支持，并加强与高校的合作，
以提升全球竞争力。
"""

# ========= 测试函数 =========
def test_model(client, model_name, extra_body=None):
    start = time.time()

    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": PROMPT}],
        temperature=0.7,
        max_tokens=300,
        extra_body=extra_body or {}
    )

    end = time.time()

    print(f"\n=== {model_name} ===")
    print("耗时: {:.2f}s".format(end - start))
    print("输出:", resp.choices[0].message.content.strip())


# ========= 执行 =========
if __name__ == "__main__":

    # Qwen（关闭 thinking）
    test_model(
        qwen_client,
        "qwen3.5-35b-a3b",
        extra_body={"enable_thinking": True}  # ⭐关键
    )

    # DeepSeek（默认无 thinking）
    test_model(
        deepseek_client,
        "deepseek-chat"  # 或 deepseek-coder / 你的模型名
    )