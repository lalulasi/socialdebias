"""Batch explanation-consistency evaluation for PolitiFact adversarial variants."""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modeling.attributor import BertAttributor
from modeling.social_debias import SocialDebiasModel, infer_bottleneck_dim
from scripts.train_bert_baseline import BertBaselineModel
from utils.explanation_metrics import compute_all_metrics


class BertZeroBaseline(torch.nn.Module):
    def __init__(self, model_name="bert-base-uncased", num_classes=2):
        super().__init__()
        torch.manual_seed(42)
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_classes)
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        torch.nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(out.last_hidden_state[:, 0, :])


def load_texts(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        texts = data.get("news") or data.get("rewritten")
        labels = data.get("labels") or data.get("label")
        return list(texts), [int(x) for x in labels]
    text_col = "news" if "news" in data.columns else "content"
    label_col = "labels" if "labels" in data.columns else "label"
    return data[text_col].tolist(), [int(x) for x in data[label_col].tolist()]


def load_socialdebias(ckpt_path, model_name, device, surface_feat_dim):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    model = SocialDebiasModel(
        model_name=model_name,
        num_classes=2,
        hidden_dim=config.get("hidden_dim", 384),
        bottleneck_dim=infer_bottleneck_dim(state, config),
        use_frozen_bert=False,
        surface_feat_dim=config.get("surface_feat_dim", surface_feat_dim),
    ).to(device)
    # frozen_bert.* is expected to be absent because attribution does not use
    # the semantic anchor; all trainable branch weights must still match.
    incompatible = model.load_state_dict(state, strict=False)
    unexpected = [k for k in incompatible.unexpected_keys if not k.startswith("frozen_bert.")]
    if incompatible.missing_keys or unexpected:
        raise ValueError(
            f"解释模型 checkpoint 架构不匹配: missing={incompatible.missing_keys}, "
            f"unexpected={unexpected}"
        )
    return model.eval()


def load_bert_baseline(ckpt_path, model_name, device):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"BERT checkpoint format is invalid: {ckpt_path}")
    model = BertBaselineModel(model_name=model_name).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.eval()


def summarize(rows, prefix):
    out = {}
    for key in ("top_k_overlap", "spearman", "js_divergence"):
        vals = [r[prefix][key] for r in rows if r[prefix].get(key) is not None]
        vals = [v for v in vals if not np.isnan(v)]
        out[key] = {
            "mean": float(np.mean(vals)) if vals else float("nan"),
            "std": float(np.std(vals)) if vals else float("nan"),
        }
    return out


def top_tokens(tokens, scores, k):
    ranked = sorted(
        zip(tokens, scores), key=lambda item: abs(float(item[1])), reverse=True
    )
    return [token for token, _ in ranked[:k]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="politifact", choices=["politifact"])
    parser.add_argument("--language", default="en")
    parser.add_argument("--variant", default="C")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--bert_ckpt", default=None,
                        help="same-seed trained BERT baseline checkpoint")
    parser.add_argument("--baseline_zero", action="store_true",
                        help="legacy random-head control; not for paper tables")
    parser.add_argument("--topk", "--top_k", dest="topk", type=int, default=10)
    parser.add_argument("--n_steps", type=int, default=50)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--surface_feat_dim", type=int, default=8)
    parser.add_argument("--output", default="results/explanation/politifact_surface_seed42.json")
    args = parser.parse_args()

    model_name = "bert-base-chinese" if args.language == "zh" else "bert-base-uncased"
    clean_path = f"data/sheepdog/news_articles/{args.dataset}_test.pkl"
    adv_path = f"data/sheepdog/adversarial_test/{args.dataset}_test_adv_{args.variant}.pkl"
    clean_texts, clean_labels = load_texts(clean_path)
    adv_texts, adv_labels = load_texts(adv_path)
    if clean_labels != adv_labels:
        raise ValueError("clean 与 adversarial 标签顺序不一致")

    if args.max_samples > 0:
        clean_texts = clean_texts[:args.max_samples]
        adv_texts = adv_texts[:args.max_samples]
        clean_labels = clean_labels[:args.max_samples]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sd_model = load_socialdebias(args.ckpt, model_name, device, args.surface_feat_dim)
    sd_attr = BertAttributor(sd_model, tokenizer, device, n_steps=args.n_steps)

    bert_attr = None
    bert_key = None
    if args.bert_ckpt:
        bert_model = load_bert_baseline(args.bert_ckpt, model_name, device)
        bert_attr = BertAttributor(bert_model, tokenizer, device, n_steps=args.n_steps)
        bert_key = "bert"
    elif args.baseline_zero:
        bert_model = BertZeroBaseline(model_name).to(device).eval()
        bert_attr = BertAttributor(bert_model, tokenizer, device, n_steps=args.n_steps)
        bert_key = "zero_bert"

    rows = []
    for i, (clean, adv, label) in enumerate(zip(clean_texts, adv_texts, clean_labels)):
        if i == 0 or (i + 1) % 5 == 0:
            print(f"[{i + 1}/{len(clean_texts)}] label={label}")

        sd_clean = sd_attr.attribute(clean, target_class=label, max_length=args.max_length)
        sd_adv = sd_attr.attribute(adv, target_class=label, max_length=args.max_length)
        row = {
            "idx": i,
            "id": f"pf_test_{i:03d}",
            "label": label,
            "orig_topk_tokens": top_tokens(
                sd_clean["tokens"], sd_clean["scores"], args.topk
            ),
            "adv_topk_tokens": top_tokens(
                sd_adv["tokens"], sd_adv["scores"], args.topk
            ),
            "socialdebias": compute_all_metrics(
                sd_clean["tokens"], sd_clean["scores"],
                sd_adv["tokens"], sd_adv["scores"],
                k=args.topk,
            ),
        }

        if bert_attr is not None:
            b_clean = bert_attr.attribute(clean, target_class=label, max_length=args.max_length)
            b_adv = bert_attr.attribute(adv, target_class=label, max_length=args.max_length)
            row[bert_key] = compute_all_metrics(
                b_clean["tokens"], b_clean["scores"],
                b_adv["tokens"], b_adv["scores"],
                k=args.topk,
            )
        rows.append(row)

        if (i + 1) % 10 == 0:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump({"args": vars(args), "rows": rows}, f, indent=2)

    result = {
        "args": vars(args),
        "summary": {"socialdebias": summarize(rows, "socialdebias")},
        "rows": rows,
    }
    if bert_attr is not None:
        result["summary"][bert_key] = summarize(rows, bert_key)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
    print(f"结果: {args.output}")


if __name__ == "__main__":
    main()
