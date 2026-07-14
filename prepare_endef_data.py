"""
prepare_endef_data.py — 把 SocialDebias 的 pkl 数据转成 ENDEF 输入格式

ENDEF 每条样本需要:
  - content    : 新闻正文
  - label      : 0(real) / 1(fake)
  - time       : "YYYY-MM-DD ..." 字符串（ENDEF 两个 BERT 模型 forward 不读 year，填占位）
  - entity_list: [{"entity": "xxx"}, ...]（bert_endef 真用，需 NER 提取）
另需 *_emo.npy：shape (n, 38)，ENDEF 两个 BERT 模型 forward 不读 emotion，填全 0 占位

输出目录结构:
  ENDEF_en/data_politifact/{train,val,test}.json + {train,val,test}_emo.npy
  ENDEF_en/data_gossipcop/  同上
  ENDEF_ch/data_weibo21/    同上

用法（在 socialdebias 项目根目录运行，需要 spaCy / jieba）:
  python prepare_endef_data.py                      # 跑全部 3 数据集
  python prepare_endef_data.py --dataset politifact # 只跑一个
"""
import argparse
import json
import os
import pickle
import random
from pathlib import Path

import numpy as np

EMO_DIM = 38                       # 原始 emo.npy 是 (n, 38)
PLACEHOLDER_TIME = "2016-01-01 00:00:00"   # 两个 BERT 模型 forward 不读 year，占位无害
VAL_RATIO = 0.1                    # PolitiFact / GossipCop 从 train 切 val 的比例
SPLIT_SEED = 42                    # 切 val 的固定种子（保证可复现）

def build_datasets(socialdebias_root, endef_en_root, endef_ch_root):
    return {
        "politifact": {
            "lang": "en",
            "train_pkl": socialdebias_root / "data/sheepdog/news_articles/politifact_train.pkl",
            "test_pkl": socialdebias_root / "data/sheepdog/news_articles/politifact_test.pkl",
            "val_pkl": None,
            "out_dir": endef_en_root / "data_politifact",
            "format": "dict",
        },
        "gossipcop": {
            "lang": "en",
            "train_pkl": socialdebias_root / "data/sheepdog/news_articles/gossipcop_train.pkl",
            "test_pkl": socialdebias_root / "data/sheepdog/news_articles/gossipcop_test.pkl",
            "val_pkl": None,
            "out_dir": endef_en_root / "data_gossipcop",
            "format": "dict",
        },
        "weibo21": {
            "lang": "zh",
            "train_pkl": socialdebias_root / "data/weibo21_repo/data/train.pkl",
            "val_pkl": socialdebias_root / "data/weibo21_repo/data/val.pkl",
            "test_pkl": socialdebias_root / "data/weibo21_repo/data/test.pkl",
            "out_dir": endef_ch_root / "data_weibo21",
            "format": "dataframe",
        },
    }


# ============ 数据读取 ============
def load_pkl_split(path, fmt):
    """返回 (texts: list[str], labels: list[int])"""
    with open(path, "rb") as f:
        d = pickle.load(f)
    if fmt == "dict":
        texts = list(d["news"])
        labels = [int(x) for x in d["labels"]]
    elif fmt == "dataframe":
        texts = d["content"].astype(str).tolist()
        labels = [int(x) for x in d["label"].tolist()]
    else:
        raise ValueError(f"未知格式: {fmt}")
    return texts, labels


# ============ 实体提取 ============
_spacy_nlp = None
def extract_entities_en(text):
    """英文用 spaCy en_core_web_sm 提取 PERSON/ORG/GPE/...，返回 [{'entity': str}, ...]"""
    global _spacy_nlp
    if _spacy_nlp is None:
        import spacy
        _spacy_nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    try:
        doc = _spacy_nlp(text[:5000])   # 截断防超长文本拖慢
        seen = set()
        ents = []
        for e in doc.ents:
            name = e.text.strip()
            if name and name.lower() not in seen and len(name) > 1:
                seen.add(name.lower())
                ents.append({"entity": name})
        return ents
    except Exception:
        return []


def extract_entities_zh(text):
    """中文用 jieba.posseg 提取 nr/ns/nt（人名/地名/机构名），返回 [{'entity': str}, ...]"""
    import jieba.posseg as pseg
    try:
        seen = set()
        ents = []
        for word, flag in pseg.cut(text[:5000]):
            if flag in ("nr", "ns", "nt") and len(word) > 1 and word not in seen:
                seen.add(word)
                ents.append({"entity": word})
        return ents
    except Exception:
        return []


# ============ 单 split 转换 ============
def convert_split(texts, labels, lang, split_name):
    """texts/labels -> ENDEF json list + emo 占位数组"""
    extract_fn = extract_entities_en if lang == "en" else extract_entities_zh
    records = []
    n = len(texts)
    for i, (text, label) in enumerate(zip(texts, labels)):
        if (i + 1) % 200 == 0 or i == 0:
            print(f"    [{split_name}] {i+1}/{n} 实体提取中...")
        ents = extract_fn(str(text))
        records.append({
            "content": str(text),
            "label": int(label),
            "time": PLACEHOLDER_TIME,
            "entity_list": ents,
        })
    emo = np.zeros((n, EMO_DIM), dtype=np.float64)
    return records, emo


# ============ 单数据集处理 ============
def process_dataset(name, cfg):
    print(f"\n{'='*60}")
    print(f"处理数据集: {name} (lang={cfg['lang']})")
    print(f"{'='*60}")
    os.makedirs(cfg["out_dir"], exist_ok=True)

    # ---- 读 train / test，并准备 val ----
    train_texts, train_labels = load_pkl_split(cfg["train_pkl"], cfg["format"])
    test_texts, test_labels = load_pkl_split(cfg["test_pkl"], cfg["format"])

    if cfg["val_pkl"] is not None:
        # Weibo21: 有现成 val
        val_texts, val_labels = load_pkl_split(cfg["val_pkl"], cfg["format"])
        print(f"  使用现成 val split")
    else:
        # PolitiFact / GossipCop: 从 train 切 9:1
        idx = list(range(len(train_texts)))
        random.Random(SPLIT_SEED).shuffle(idx)
        n_val = max(1, int(len(idx) * VAL_RATIO))
        val_idx = set(idx[:n_val])
        new_train_texts = [train_texts[i] for i in idx[n_val:]]
        new_train_labels = [train_labels[i] for i in idx[n_val:]]
        val_texts = [train_texts[i] for i in idx[:n_val]]
        val_labels = [train_labels[i] for i in idx[:n_val]]
        train_texts, train_labels = new_train_texts, new_train_labels
        print(f"  从 train 切 9:1 → train={len(train_texts)}, val={len(val_texts)}")

    print(f"  规模: train={len(train_texts)}, val={len(val_texts)}, test={len(test_texts)}")

    # ---- 三个 split 各自转换 ----
    for split_name, (texts, labels) in [
        ("train", (train_texts, train_labels)),
        ("val",   (val_texts, val_labels)),
        ("test",  (test_texts, test_labels)),
    ]:
        records, emo = convert_split(texts, labels, cfg["lang"], split_name)
        json_path = os.path.join(cfg["out_dir"], f"{split_name}.json")
        emo_path = os.path.join(cfg["out_dir"], f"{split_name}_emo.npy")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        np.save(emo_path, emo)
        # 统计实体数
        avg_ents = sum(len(r["entity_list"]) for r in records) / max(1, len(records))
        n_fake = sum(1 for r in records if r["label"] == 1)
        print(f"  ✓ {split_name}: {len(records)} 条 (fake={n_fake}), "
              f"平均实体数={avg_ents:.1f} → {json_path}")

    print(f"  完成 → {cfg['out_dir']}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all",
                        choices=["all", "politifact", "gossipcop", "weibo21"],
                        help="要转换的数据集")
    parser.add_argument("--socialdebias_root", type=Path,
                        default=Path(__file__).resolve().parent,
                        help="SocialDebias 项目根目录")
    parser.add_argument("--endef_en_root", type=Path,
                        default=os.environ.get("ENDEF_EN_ROOT"),
                        help="英文 ENDEF 仓库根目录，也可通过 ENDEF_EN_ROOT 设置")
    parser.add_argument("--endef_ch_root", type=Path,
                        default=os.environ.get("ENDEF_CH_ROOT"),
                        help="中文 ENDEF 仓库根目录，也可通过 ENDEF_CH_ROOT 设置")
    args = parser.parse_args()

    targets = ["politifact", "gossipcop", "weibo21"] if args.dataset == "all" else [args.dataset]
    if any(name in {"politifact", "gossipcop"} for name in targets) and args.endef_en_root is None:
        parser.error("英文数据转换需要 --endef_en_root 或 ENDEF_EN_ROOT")
    if "weibo21" in targets and args.endef_ch_root is None:
        parser.error("Weibo21 转换需要 --endef_ch_root 或 ENDEF_CH_ROOT")

    datasets = build_datasets(
        args.socialdebias_root,
        args.endef_en_root,
        args.endef_ch_root,
    )
    for name in targets:
        process_dataset(name, datasets[name])

    print(f"\n{'='*60}")
    print("全部完成。下一步：把各 data_xxx 目录作为 ENDEF 的 --root_path")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
