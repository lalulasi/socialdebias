"""
对抗改写提示词模板（意见 12：结构化提示词工程）

四段式结构：角色设定 + 硬约束 + 风格指令 + 格式要求
"""

# ============== 共享模块 ==============

SYSTEM_HARD_CONSTRAINT = """You are a professional editor performing news rewriting. Follow these ABSOLUTE rules:

ENTITY PRESERVATION (HARD CONSTRAINT):
- Preserve ALL person names, place names, organization names EXACTLY as they appear
- Preserve ALL dates, times, numbers, monetary amounts EXACTLY
- Preserve ALL quoted statements and their attribution
- Do NOT add information not in the original
- Do NOT remove factual claims

OUTPUT FORMAT:
- Output ONLY the rewritten news text
- Do NOT include explanations, notes, or meta-commentary
- Do NOT use markdown formatting
- Length should be approximately the same as the original
"""

# ============== 4 种风格模板（方案 β）==============

STYLE_PROMPTS = {
    "neutral": {
        "role": "You are a neutral news editor for a non-partisan wire service like Reuters.",
        "instruction": """Rewrite the following news in a NEUTRAL, FACTUAL tone:
- Remove all emotional language (e.g., "shocking", "amazing", "devastating")
- Remove exclamation marks and emphatic punctuation
- Replace opinionated adjectives with neutral descriptors
- Use objective voice throughout
- Preserve all entities, quotes, and factual claims""",
    },

    "academic": {
        "role": "You are an academic researcher writing for a peer-reviewed publication.",
        "instruction": """Rewrite the following news in an ACADEMIC, ANALYTICAL tone:
- Use formal, precise vocabulary
- Prefer passive voice where appropriate
- Add appropriate hedging language ("reportedly", "according to", "it was indicated")
- Restructure sentences to be more complex
- Preserve all entities, quotes, and factual claims""",
    },

    "report": {
        "role": "You are a senior correspondent for The Associated Press.",
        "instruction": """Rewrite the following news in a STANDARD NEWS REPORT style:
- Use the inverted pyramid structure
- Start with the most important information
- Use clear, concise sentences
- Avoid editorial language and opinion
- Preserve all entities, quotes, and factual claims""",
    },

    "simplified": {
        "role": "You are writing for a general audience with simple, clear language.",
        "instruction": """Rewrite the following news in a SIMPLIFIED, ACCESSIBLE tone:
- Use short, simple sentences
- Replace complex words with common ones (but keep all proper names)
- Break long paragraphs into shorter ones
- Explain technical terms briefly
- Preserve all entities, quotes, and factual claims""",
    },
}


def build_prompt(original_text: str, style: str, max_input_chars: int = 6000) -> str:
    """
    构造完整 prompt。
    Qwen3 用 /no_think 前缀关闭思考模式。

    Args:
        original_text: 原始新闻文本
        style: 风格 key (neutral/academic/report/simplified)
        max_input_chars: 输入截断（防止过长，~800 词）
    """
    if style not in STYLE_PROMPTS:
        raise ValueError(f"Unknown style: {style}. Available: {list(STYLE_PROMPTS.keys())}")

    # 截断超长输入（PolitiFact 有的新闻 3000 词+，截断到 ~800 词）
    truncated = original_text[:max_input_chars]
    if len(original_text) > max_input_chars:
        truncated += " [...]"

    style_cfg = STYLE_PROMPTS[style]

    prompt = f"""/no_think

{style_cfg['role']}

{SYSTEM_HARD_CONSTRAINT}

{style_cfg['instruction']}

ORIGINAL NEWS:
{truncated}

REWRITTEN NEWS (start directly, no preamble):
"""
    return prompt


if __name__ == "__main__":
    # 测试
    sample = "President Trump announced new tariffs on Chinese imports today, marking a significant escalation in trade tensions."
    for style in STYLE_PROMPTS.keys():
        print(f"=== {style} ===")
        print(build_prompt(sample, style))
        print()

# ========== 中文版（用于 Weibo21）==========

STYLE_PROMPTS_ZH = {
    "neutral": {
        "role": "你是一名专业的新闻编辑，工作于一家中立、客观的新闻机构。",
        "instruction": "请将以下中文新闻改写为中立、客观、事实性的语气：\n- 删除所有情绪化表达\n- 删除感叹号和过度强调的标点\n- 用客观描述替代带情绪的形容词\n- 保持中立陈述视角\n- 必须保留所有人名、地名、机构名、日期、数字等关键事实",
    },
}


def build_prompt_zh(original_text, style="neutral", max_chars=2000):
    """构造中文 prompt，要求 LLM 输出中文。"""
    if style not in STYLE_PROMPTS_ZH:
        raise ValueError(f"未知风格: {style}")
    truncated = original_text[:max_chars]
    if len(original_text) > max_chars:
        truncated += " [...]"
    style_cfg = STYLE_PROMPTS_ZH[style]
    parts = [
        style_cfg["role"],
        "",
        "【硬约束】",
        "- 必须保留所有人名、地名、机构名称",
        "- 必须保留所有日期、时间、数字、金额",
        "- 必须保留所有引用语句",
        "- 不得添加原文中没有的信息",
        "",
        "【输出要求】",
        "- 仅输出改写后的中文文本（绝对不要翻译为英文）",
        "- 长度与原文相近",
        "- 不要添加解释或注释",
        "",
        "【风格指令】",
        style_cfg["instruction"],
        "",
        "【原始新闻】",
        truncated,
        "",
        "【改写后的中文新闻】（用中文输出，禁止英文）：",
    ]
    return "\n".join(parts)
