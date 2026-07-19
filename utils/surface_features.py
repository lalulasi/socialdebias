"""Surface-style features used by the SocialDebias bias branch.

The paper configuration uses the official NRC Emotion Lexicon for the first
eight dimensions.  The old hand-written seed-word implementation remains
available only as ``feature_version="legacy_seed_v0"`` so historical
checkpoints can still be evaluated explicitly.
"""
import csv
import json
import os
import re
from pathlib import Path

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

    def __init__(
            self,
            dim=8,
            spacy_model="en_core_web_sm",
            lexicon_path=None,
            language="en",
            feature_version="nrc_emolex_v1",
            stopwords_path=None,
    ):
        assert dim in (8, 17), f"dim 仅支持 8 或 17，收到 {dim}"
        self.dim = dim
        self.language = language
        self.feature_version = feature_version
        self.stopwords = set()
        if stopwords_path:
            self.stopwords = {
                line.strip().lower()
                for line in Path(stopwords_path).read_text(encoding="utf-8").splitlines()
                if line.strip()
            }

        self.nlp = None
        if language == "en" or dim == 17:
            import spacy
            self.nlp = spacy.load(spacy_model, disable=["ner", "lemmatizer"])

        self.lexicon = None
        if feature_version == "nrc_emolex_v1":
            lexicon_path = lexicon_path or os.environ.get("NRC_EMOLEX_PATH")
            if not lexicon_path:
                candidates = [
                    Path("data/lexicons/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"),
                    Path("data/lexicons/nrc_emolex.tsv"),
                ]
                lexicon_path = next((str(p) for p in candidates if p.exists()), None)
            if not lexicon_path:
                raise FileNotFoundError(
                    "缺少 NRC-EmoLex 词典。请用 --surface_lexicon_path 指定官方词典，"
                    "或设置 NRC_EMOLEX_PATH；不能用少量手写关键词替代论文中的 NRC 特征。"
                )
            self.lexicon = self._load_lexicon(lexicon_path)
        elif feature_version != "legacy_seed_v0":
            raise ValueError(f"未知表层特征版本: {feature_version}")

    @classmethod
    def _load_lexicon(cls, path):
        """Load NRC long TSV, a wide TSV/CSV, or a JSON word mapping."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"NRC-EmoLex 文件不存在: {path}")
        if path.suffix.lower() == ".json":
            raw = json.loads(path.read_text(encoding="utf-8"))
            return {
                str(word).lower(): {
                    emo for emo, value in values.items()
                    if emo in cls.EMOTION_ORDER and float(value) > 0
                }
                for word, values in raw.items()
            }

        with path.open("r", encoding="utf-8-sig", newline="") as f:
            sample = f.read(4096)
            f.seek(0)
            delimiter = "," if path.suffix.lower() == ".csv" else "\t"
            rows = list(csv.reader(f, delimiter=delimiter))
        if not rows:
            raise ValueError(f"空 NRC-EmoLex 文件: {path}")

        lexicon = {}
        header = [c.strip().lower() for c in rows[0]]
        emotion_columns = {
            emo: header.index(emo) for emo in cls.EMOTION_ORDER if emo in header
        }
        has_word_header = any(
            name in header for name in ("word", "term", "english word")
        )
        if emotion_columns and has_word_header:
            word_col = next(
                (header.index(name) for name in ("word", "term", "english word") if name in header),
                0,
            )
            for row in rows[1:]:
                if len(row) <= word_col:
                    continue
                word = row[word_col].strip().lower()
                labels = {
                    emo for emo, col in emotion_columns.items()
                    if col < len(row) and row[col].strip() not in ("", "0", "0.0")
                }
                if word and labels:
                    lexicon[word] = labels
        else:
            # Official word-level format: word<TAB>emotion<TAB>association.
            for row in rows:
                if len(row) < 3:
                    continue
                word, emotion, association = row[0].strip().lower(), row[1].strip().lower(), row[2].strip()
                if emotion in cls.EMOTION_ORDER and association not in ("", "0", "0.0"):
                    lexicon.setdefault(word, set()).add(emotion)
        if not lexicon:
            raise ValueError(f"未从词典解析到 NRC 八类情绪词: {path}")
        return lexicon

    def _tokens(self, text):
        if self.language == "zh":
            import jieba
            return [
                token.strip().lower() for token in jieba.lcut(text)
                if token.strip() and not re.fullmatch(r"\W+", token)
                and token.strip().lower() not in self.stopwords
            ]
        doc = self.nlp(text)
        return [
            token.text.lower() for token in doc
            if token.is_alpha and not token.is_stop
            and token.text.lower() not in self.stopwords
        ]

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

        doc = self.nlp(text) if self.nlp is not None else None
        tokens = self._tokens(text)
        n_tok = max(len(tokens), 1)

        emotion_counts = np.zeros(8, dtype=np.float32)
        if self.feature_version == "legacy_seed_v0":
            text_lower = text.lower()
            for i, emo in enumerate(self.EMOTION_ORDER):
                emotion_counts[i] = sum(
                    1 for word in self.EMOTION_WORDS[emo] if word in text_lower
                )
            features = (emotion_counts / max(len(text), 100) * 100).tolist()
        else:
            emotion_to_idx = {emo: i for i, emo in enumerate(self.EMOTION_ORDER)}
            for token in tokens:
                for emotion in self.lexicon.get(token, ()):
                    emotion_counts[emotion_to_idx[emotion]] += 1.0

            # 论文 §3.7.1：累计强度后执行 L2 归一化。
            norm = float(np.linalg.norm(emotion_counts))
            features = (emotion_counts / norm if norm > 0 else emotion_counts).tolist()

        if self.dim >= 17:
            if doc is None:
                raise ValueError("17 维句法/词汇特征目前仅支持 spaCy 可处理的语言")
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
