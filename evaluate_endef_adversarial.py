"""Evaluate ENDEF checkpoints on paired SheepDog style attacks.

Copy this file into ``ENDEF_en/`` (as described by the reproduction guide), or
run it there directly.  Paths are resolved from ``SOCIALDEBIAS_ROOT`` and
``ENDEF_EN_ROOT`` instead of machine-specific absolute paths.
"""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from transformers import BertTokenizer


SOCIALDEBIAS_ROOT = Path(os.environ.get("SOCIALDEBIAS_ROOT", ""))
ENDEF_EN_ROOT = Path(os.environ.get("ENDEF_EN_ROOT", Path(__file__).resolve().parent))
MAX_LEN = 170
ENTITY_MAX_LEN = 50


def require_paths():
    if not SOCIALDEBIAS_ROOT.is_dir():
        raise RuntimeError("请设置 SOCIALDEBIAS_ROOT 为 socialdebias 仓库绝对路径")
    if not (ENDEF_EN_ROOT / "models").is_dir():
        raise RuntimeError("请设置 ENDEF_EN_ROOT，或把本脚本复制到 ENDEF_en 根目录")


def import_models():
    sys.path.insert(0, str(ENDEF_EN_ROOT))
    from models.bert import BERTFENDModel
    from models.bertendef import BERT_ENDEFModel
    return BERTFENDModel, BERT_ENDEFModel


def load_adv(path):
    with path.open("rb") as f:
        data = pickle.load(f)
    return list(data["news"]), [int(x) for x in data["labels"]]


def load_clean(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    labels = [
        int(row["label"]) if isinstance(row["label"], int)
        else (1 if row["label"] == "fake" else 0)
        for row in data
    ]
    return [row["content"] for row in data], labels


class Runner:
    def __init__(self, batch_size=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.BERTFENDModel, self.BERT_ENDEFModel = import_models()
        self._nlp = None

    def entities(self, text):
        if self._nlp is None:
            import spacy
            self._nlp = spacy.load(
                "en_core_web_sm", disable=["parser", "lemmatizer"]
            )
        seen, values = set(), []
        for entity in self._nlp(text[:5000]).ents:
            value = entity.text.strip()
            if len(value) > 1 and value.lower() not in seen:
                seen.add(value.lower()); values.append(value)
        return " [SEP] ".join(values)

    def tokenize(self, texts, max_length):
        encoded = self.tokenizer(
            texts, max_length=max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        return encoded["input_ids"], encoded["attention_mask"]

    def build_model(self, model_type):
        if model_type == "bert":
            return self.BERTFENDModel(
                emb_dim=768, mlp_dims=[384], dropout=0.2
            ).to(self.device)
        return self.BERT_ENDEFModel(
            emb_dim=768, mlp_dims=[384], dropout=0.2
        ).to(self.device)

    @torch.no_grad()
    def infer(self, model, texts, labels, model_type):
        model.eval()
        content_ids, content_masks = self.tokenize(texts, MAX_LEN)
        if model_type == "bert_endef":
            entity_ids, entity_masks = self.tokenize(
                [self.entities(text) for text in texts], ENTITY_MAX_LEN
            )
        predictions = []
        for start in range(0, len(texts), self.batch_size):
            batch = {
                "content": content_ids[start:start + self.batch_size].to(self.device),
                "content_masks": content_masks[start:start + self.batch_size].to(self.device),
            }
            if model_type == "bert_endef":
                batch["entity"] = entity_ids[start:start + self.batch_size].to(self.device)
                batch["entity_masks"] = entity_masks[start:start + self.batch_size].to(self.device)
                _, _, output = model(**batch)
            else:
                output = model(**batch)
            predictions.extend(output.detach().cpu().reshape(-1).tolist())
        probs = np.asarray(predictions)
        pred = (probs >= 0.5).astype(np.int64)
        labels_np = np.asarray(labels)
        try:
            auc = roc_auc_score(labels_np, probs)
        except ValueError:
            auc = float("nan")
        return {
            "accuracy": accuracy_score(labels_np, pred),
            "f1": f1_score(labels_np, pred, average="macro"),
            "auc": auc,
        }, pred, labels_np

    def evaluate(self, dataset, model_type, seed, ckpt_path):
        model = self.build_model(model_type)
        state = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        model.load_state_dict(state, strict=True)
        clean_path = ENDEF_EN_ROOT / f"data_{dataset}" / "test.json"
        clean_texts, clean_labels = load_clean(clean_path)
        clean_metrics, clean_pred, clean_labels_np = self.infer(
            model, clean_texts, clean_labels, model_type
        )
        results = {"clean": clean_metrics}
        asrs = []
        for variant in "ABCD":
            path = (
                SOCIALDEBIAS_ROOT / "data/sheepdog/adversarial_test"
                / f"{dataset}_test_adv_{variant}.pkl"
            )
            texts, labels = load_adv(path)
            metrics, pred, labels_np = self.infer(model, texts, labels, model_type)
            if not np.array_equal(clean_labels_np, labels_np):
                raise ValueError(
                    f"{dataset} adv_{variant} 标签顺序与 clean 不一致，不能计算 paired ASR"
                )
            clean_correct = clean_pred == clean_labels_np
            denominator = int(clean_correct.sum())
            asr = (
                float((clean_correct & (pred != labels_np)).sum()) / denominator
                if denominator else float("nan")
            )
            metrics["asr"] = asr
            results[f"adv_{variant}"] = metrics
            asrs.append(asr)
        adv_f1 = float(np.mean([results[f"adv_{v}"]["f1"] for v in "ABCD"]))
        results["summary"] = {
            "clean_f1": clean_metrics["f1"],
            "avg_adv_f1": adv_f1,
            "f1_drop": clean_metrics["f1"] - adv_f1,
            "avg_asr": float(np.mean(asrs)),
            "asr_per_variant": {v: results[f"adv_{v}"]["asr"] for v in "ABCD"},
        }
        return {
            "dataset": dataset, "model": model_type, "seed": seed,
            "ckpt": str(ckpt_path), "results": results,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["politifact", "gossipcop"])
    parser.add_argument("--model", choices=["bert", "bert_endef"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--ckpt")
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()
    require_paths()
    output_dir = Path(args.output_dir) if args.output_dir else ENDEF_EN_ROOT / "results_endef_adv"
    output_dir.mkdir(parents=True, exist_ok=True)
    runner = Runner(args.batch_size)
    jobs = []
    if args.batch:
        for dataset in ("politifact", "gossipcop"):
            for model in ("bert", "bert_endef"):
                for seed in (42, 2024, 3407):
                    ckpt = ENDEF_EN_ROOT / "backup_ckpt" / f"{dataset}_{model}_seed{seed}.pkl"
                    if ckpt.exists():
                        jobs.append((dataset, model, seed, ckpt))
    else:
        if None in (args.dataset, args.model, args.seed, args.ckpt):
            parser.error("非 --batch 模式需要 --dataset/--model/--seed/--ckpt")
        jobs.append((args.dataset, args.model, args.seed, Path(args.ckpt)))
    for dataset, model, seed, ckpt in jobs:
        result = runner.evaluate(dataset, model, seed, ckpt)
        output = output_dir / f"endef_adv_{dataset}_seed{seed}_{model}.json"
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        summary = result["results"]["summary"]
        print(
            f"{dataset} {model} seed={seed}: clean={summary['clean_f1']:.4f} "
            f"adv={summary['avg_adv_f1']:.4f} drop={summary['f1_drop']*100:.2f}pp "
            f"ASR={summary['avg_asr']*100:.2f}%"
        )


if __name__ == "__main__":
    main()
