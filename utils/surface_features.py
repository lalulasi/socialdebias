"""
Surface-style feature extractor.

输出维度可选：
  dim=8  情绪特征
  dim=17 情绪 8 + 句法 5 + 词汇 4

17 维顺序严格对应 FEATURE_NAMES。
"""
import numpy as np


class SurfaceFeatureExtractor:
    """提取表层风格特征。dim=8 或 17。"""

    EMOTION_WORDS = {
        "anger":       ["angry", "rage", "furious", "outrage", "hate", "hostile"],
        "joy":         ["happy", "joy", "delight", "glad", "cheer", "celebrate"],
        "fear":        ["fear", "afraid", "scared", "terror", "panic", "dread"],
        "disgust":     ["disgust", "gross", "revolting", "sick", "nasty"],
        "sadness":     ["sad", "grief", "sorrow", "mourn", "despair", "cry"],
        "surprise":    ["surprise", "shock", "astonish", "stun", "unexpected"],
        "trust":       ["trust", "reliable", "honest", "credible", "faithful"],
        "anticipation":["expect", "anticipate", "await", "hope", "forecast"],
    }

    EMOTION_ORDER = ["anger", "joy", "fear", "disgust",
                     "sadness", "surprise", "trust", "anticipation"]

    FEATURE_DIM = 17

    def __init__(self, dim=8, spacy_model="en_core_web_sm"):
        assert dim in (8, 17), f"dim 仅支持 8 或 17，收到 {dim}"
        self.dim = dim
        import spacy
        self.nlp = spacy.load(spacy_model, disable=["ner", "lemmatizer"])

    @staticmethod
    def _syllables(word):
        word = word.lower()
        vowels = "aeiouy"
        count, prev_v = 0, False
        for ch in word:
            is_v = ch in vowels
            if is_v and not prev_v:
                count += 1
            prev_v = is_v
        if word.endswith("e"):
            count = max(1, count - 1)
        return max(1, count)

    def _flesch(self, tokens, n_sent):
        n_words = max(len(tokens), 1)
        n_sent = max(n_sent, 1)
        n_syl = sum(self._syllables(w) for w in tokens)
        return 206.835 - 1.015 * (n_words / n_sent) - 84.6 * (n_syl / n_words)

    def extract(self, text, max_chars=10000):
        """返回 self.dim 维特征向量。"""
        text = text[:max_chars] if text else ""
        if not text.strip():
            return np.zeros(self.dim, dtype=np.float32)

        doc = self.nlp(text)
        tokens = [t.text.lower() for t in doc if t.is_alpha]
        n_tok = max(len(tokens), 1)

        features = []
        text_lower = text.lower()
        for emo in self.EMOTION_ORDER:
            words = self.EMOTION_WORDS[emo]
            count = sum(1 for w in words if w.lower() in text_lower)
            norm = max(len(text), 100)
            features.append(count / norm * 100)

        if self.dim >= 17:
            sents = list(doc.sents)
            n_sent = max(len(sents), 1)
            slens = [sum(1 for t in s if t.is_alpha) for s in sents] or [0]
            features.append(float(np.mean(slens)))                       # syn_avg_slen
            features.append(float(np.max(slens)))                        # syn_max_slen
            features.append(text.count("?") / max(len(text), 1) * 1000)  # syn_qmark_per1k
            features.append(text.count("!") / max(len(text), 1) * 1000)  # syn_excl_per1k
            n_pass = sum(1 for t in doc if t.dep_ in ("nsubjpass", "auxpass", "csubjpass"))
            features.append(n_pass / n_sent)                             # syn_passive_per_sent
            features.append(len(set(tokens)) / n_tok)                    # lex_ttr
            n_sup = sum(1 for t in doc if t.tag_ in ("JJS", "RBS"))
            features.append(n_sup / n_tok * 100)                         # lex_superlative /100tok
            n_mod = sum(1 for t in doc if t.tag_ == "MD")
            features.append(n_mod / n_tok * 100)                         # lex_modal /100tok
            features.append(self._flesch(tokens, n_sent))               # lex_flesch

        feat = np.array(features, dtype=np.float32)
        feat = np.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=0.0)
        assert len(feat) == self.dim, f"特征维度错误: {len(feat)} != {self.dim}"
        return feat


FEATURE_NAMES = [
    "emo_anger", "emo_joy", "emo_fear", "emo_disgust",
    "emo_sadness", "emo_surprise", "emo_trust", "emo_anticipation",
    "syn_avg_slen", "syn_max_slen", "syn_qmark_per1k", "syn_excl_per1k", "syn_passive_per_sent",
    "lex_ttr", "lex_superlative", "lex_modal", "lex_flesch",
]


if __name__ == "__main__":
    for dim in (8, 17):
        ex = SurfaceFeatureExtractor(dim=dim)
        f_strong = ex.extract("SHOCKING!! Trump WINS in a HISTORIC victory! Supporters are absolutely THRILLED!")
        f_neutral = ex.extract("President Trump won the election. Multiple sources reported the outcome.")
        print(f"\n=== dim={dim} ===  len(strong)={len(f_strong)}")
        for n, v in zip(FEATURE_NAMES[:dim], f_strong):
            print(f"  {n:22s}: {v:.4f}")
        print(f"  欧氏距离(强 vs 中立): {np.linalg.norm(f_strong - f_neutral):.4f}")
