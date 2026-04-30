"""
多维表层风格特征提取器（意见 8 实现）
对应论文 3.2 节"多维表层特征定义"
输出 17 维向量：情绪 8 + 句法 5 + 词汇 4
"""
import re
import numpy as np


class SurfaceFeatureExtractor:
    """提取 17 维表层风格特征。"""

    EMOTION_WORDS = {
        "anger": {
            "angry", "anger", "rage", "furious", "fury", "outrage", "outraged",
            "hatred", "hate", "hated", "hating", "mad", "madness",
            "irritate", "irritated", "irate", "indignant", "infuriate", "infuriated",
            "wrath", "vengeful", "hostile", "aggressive", "bitter", "resentful",
            "annoy", "annoyed", "annoying", "frustrated", "frustrating", "愤怒", "生气", "气愤", "恼火", "怒", "怒火", "暴怒", "气", "厌恶", "怨", "怨恨", "仇恨", "讨厌",
        },
        "joy": {
            "happy", "happiness", "joy", "joyful", "delight", "delighted", "delightful",
            "wonderful", "pleased", "pleasure", "celebrate", "celebrated", "celebration",
            "glad", "gleeful", "cheerful", "thrilled", "thrilling", "ecstatic",
            "euphoric", "elated", "jubilant", "merry", "blissful", "content",
            "smile", "laugh", "laughter", "amusing", "fun", "enjoy", "enjoyed",
            "exciting", "excited", "wonderful", "fantastic", "great", "awesome", "开心", "高兴", "快乐", "幸福", "欣喜", "喜悦", "欢乐", "欢喜", "愉快", "欣慰", "欢迎", "庆祝", "祝贺", "笑", "好棒", "棒", "厉害", "赞",
        },
        "fear": {
            "afraid", "fear", "feared", "fearful", "scared", "scary", "terrified", "terrifying",
            "panic", "panicked", "danger", "dangerous", "threat", "threatening",
            "worry", "worried", "worrying", "anxious", "anxiety", "nervous",
            "alarm", "alarmed", "alarming", "frighten", "frightened", "frightening",
            "horrified", "horrifying", "dread", "dreadful", "creepy", "spooky",
            "intimidating", "menacing", "ominous", "害怕", "恐惧", "惊恐", "担心", "担忧", "忧虑", "焦虑", "紧张", "不安", "惊", "恐", "怕", "危险", "可怕", "吓人",
        },
        "disgust": {
            "disgust", "disgusted", "disgusting", "horrible", "awful", "nasty",
            "vile", "gross", "repulsive", "revolting", "sickening", "sick",
            "loathsome", "abhorrent", "appalling", "atrocious", "despicable",
            "detestable", "obnoxious", "offensive", "repugnant", "filthy",
            "dirty", "rotten", "yuck", "ugh", "outrageous", "恶心", "讨厌", "厌恶", "可恶", "可耻", "无耻", "卑鄙", "肮脏", "恶劣", "丑陋",
        },
        "sadness": {
            "sad", "sadness", "tragic", "tragedy", "mourn", "mourning", "grief",
            "grieve", "grieving", "sorrow", "sorrowful", "depressed", "depression",
            "melancholy", "miserable", "misery", "unhappy", "gloomy", "gloom",
            "despair", "despairing", "heartbroken", "lament", "lamenting",
            "weep", "wept", "cry", "cried", "crying", "tear", "tears",
            "lonely", "loneliness", "regret", "regretful", "悲伤", "难过", "悲", "哭", "泪", "心痛", "痛心", "伤心", "失望", "绝望", "悲哀", "哀", "凄惨", "可怜", "无奈",
        },
        "surprise": {
            "shock", "shocked", "shocking", "astonish", "astonished", "astonishing",
            "amaze", "amazed", "amazing", "stunned", "stunning", "incredible", "unbelievable",
            "surprised", "surprising", "startle", "startled", "startling",
            "bewildered", "bewildering", "stupefied", "wonder", "wondered",
            "awestruck", "speechless", "dumbfounded", "flabbergasted",
            "remarkable", "extraordinary", "phenomenal", "spectacular", "震惊", "惊", "惊讶", "意外", "不可思议", "难以置信", "不敢相信", "竟然", "居然", "突然", "惊人",
        },
        "trust": {
            "trust", "trusted", "trustworthy", "honest", "honesty", "reliable",
            "credible", "credibility", "faithful", "faith", "genuine", "sincere",
            "sincerity", "loyal", "loyalty", "dependable", "respected", "respectable",
            "authentic", "authentic", "valid", "verified", "confirmed",
            "proven", "established", "reputable", "legitimate", "transparent", "相信", "信任", "可信", "靠谱", "可靠", "诚实", "真实", "证实", "确认", "权威", "官方",
        },
        "anticipation": {
            "expect", "expected", "expecting", "expectation", "anticipate", "anticipated",
            "anticipating", "anticipation", "hopeful", "hope", "hoped", "hoping",
            "future", "soon", "upcoming", "forthcoming", "imminent", "approaching",
            "eager", "eagerly", "looking", "predict", "predicted", "predicting",
            "foresee", "foresaw", "preview", "ahead", "tomorrow", "outlook", "期待", "希望", "愿望", "盼望", "等待", "未来", "将", "即将", "马上", "很快", "预测", "预计",
        },
    }
    SUPERLATIVES = {"best", "worst", "greatest", "biggest", "most", "least",
                    "perfect", "ultimate", "extreme", "absolutely", "totally", "completely"}
    MODALS = {"must", "should", "could", "would", "might", "may",
              "shall", "ought", "needs"}
    FEATURE_DIM = 8

    def __init__(self, spacy_model="en_core_web_sm"):
        import spacy
        self.nlp = spacy.load(spacy_model, disable=["ner", "lemmatizer"])

    def extract(self, text: str, max_chars: int = 10000) -> np.ndarray:
        """返回 17 维特征向量。"""
        text = text[:max_chars] if text else ""
        if not text.strip():
            return np.zeros(self.FEATURE_DIM, dtype=np.float32)

        doc = self.nlp(text)
        tokens = [t.text.lower() for t in doc if t.is_alpha]
        n_tok = max(len(tokens), 1)
        n_chars = max(len(text), 1)

        features = []
        # ===== 情绪 8 维（每类的词频比例）=====
        for emo in ["anger", "joy", "fear", "disgust",
                    "sadness", "surprise", "trust", "anticipation"]:
            words = self.EMOTION_WORDS[emo]
            # 子字符串匹配，中英文通用
            text_lower = text.lower()
            count = sum(1 for w in words if w.lower() in text_lower)
            # 归一化：按文本长度（每 100 字符的命中率）
            norm = max(len(text), 100)
            features.append(count / norm * 100)
        # [B 实验：暂时只保留情绪 8 维，其他维度禁用]


        feat = np.array(features, dtype=np.float32)
        feat = np.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=0.0)
        assert len(feat) == self.FEATURE_DIM, f"特征维度错误: {len(feat)}"
        return feat


FEATURE_NAMES = [
    "emo_anger", "emo_joy", "emo_fear", "emo_disgust",
    "emo_sadness", "emo_surprise", "emo_trust", "emo_anticipation",
    "syn_avg_slen", "syn_max_slen", "syn_qmark_per1k", "syn_excl_per1k", "syn_passive_per_sent",
    "lex_ttr", "lex_superlative", "lex_modal", "lex_flesch",
]


if __name__ == "__main__":
    # smoke test
    extractor = SurfaceFeatureExtractor()
    sample_strong = "SHOCKING!! Trump WINS in a HISTORIC victory! Supporters are absolutely THRILLED!"
    sample_neutral = "President Trump won the election. Multiple sources reported the outcome."

    print("=== 强烈风格 ===")
    feat1 = extractor.extract(sample_strong)
    for n, v in zip(FEATURE_NAMES, feat1):
        print(f"  {n:25s}: {v:.4f}")

    print("\n=== 中立风格 ===")
    feat2 = extractor.extract(sample_neutral)
    for n, v in zip(FEATURE_NAMES, feat2):
        print(f"  {n:25s}: {v:.4f}")

    print(f"\n两个样本欧氏距离: {np.linalg.norm(feat1 - feat2):.4f}")