"""
消融实验对抗集评估脚本。
给定一个消融 ckpt，在 clean + 4 个对抗集（A/B/C/D）上评估，
输出 F1/Acc/AUC/ASR 到 JSON。

用法:
    python scripts/evaluate_ablation_adv.py \
        --ckpt results/ablation/ablation_politifact_full_seed42.pt \
        --dataset politifact \
        --variant full \
        --seed 42

保存: results/ablation_adv/ablation_adv_{dataset}_{variant}_seed{seed}.json
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# 复用老脚本的两个核心函数（别的一概不碰）
from scripts.evaluate_adversarial import evaluate_model, load_adversarial_test
from utils.real_dataloader import load_dataset
from utils.dataloader import FakeNewsDataset
from utils.device import get_device
from modeling.social_debias import SocialDebiasModel


def build_loader(samples, tokenizer, batch_size=16, max_length=512):
    ds = FakeNewsDataset(samples, tokenizer, max_length)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def compute_asr(clean_acc, adv_correct_on_clean_correct_total, adv_correct_ids):
    """
    占位：老脚本里的 ASR 计算应该是在 evaluate_model 里一起算的。
    如果 evaluate_model 返回的 dict 里直接有 ASR 就不用这个函数。
    """
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="消融 ckpt 路径")
    parser.add_argument("--dataset", type=str, required=True, choices=["politifact", "gossipcop"])
    parser.add_argument("--variant", type=str, required=True,
                        choices=["full", "no_grl", "no_consist", "no_both"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--bert_name", type=str, default="bert-base-uncased")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--variants_adv", type=str, default="A,B,C,D",
                        help="要评估的对抗变体（逗号分隔）")
    parser.add_argument("--output_dir", type=str, default="results/ablation_adv")
    args = parser.parse_args()

    device = get_device()
    print(f"[AblAdv] device={device}")
    print(f"[AblAdv] dataset={args.dataset} variant={args.variant} seed={args.seed}")
    print(f"[AblAdv] ckpt: {args.ckpt}")

    # ==== 构造模型并加载 ckpt ====
    model = SocialDebiasModel(
        model_name=args.bert_name,
        num_classes=2,
        hidden_dim=384,
        dropout=0.1,
        grl_lambda=1.0,
        use_frozen_bert=True,
    ).to(device)

    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    # 裸 state_dict，直接加载；strict=False 兼容 frozen_bert 权重差异
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[AblAdv] missing keys: {len(missing)}（前3个: {missing[:3]}）")
    if unexpected:
        print(f"[AblAdv] unexpected keys: {len(unexpected)}（前3个: {unexpected[:3]}）")
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.bert_name)

    # ==== 加载干净测试集 ====
    print(f"\n[AblAdv] 加载干净测试集 {args.dataset}")
    _, _, clean_samples = load_dataset(dataset_name=args.dataset, seed=42)
    clean_loader = build_loader(clean_samples, tokenizer, args.batch_size, args.max_length)

    # ==== 评估干净集 ====
    results = OrderedDict()
    clean_metrics = evaluate_model(model, clean_loader, device, is_social_debias=True)
    results["clean"] = clean_metrics
    # 兼容老脚本返回 dict 的各种命名，尝试多个 key
    clean_acc = clean_metrics.get("accuracy", clean_metrics.get("acc"))
    clean_f1 = clean_metrics.get("f1")
    clean_auc = clean_metrics.get("auc")
    print(f"[AblAdv] 干净集: Acc={clean_acc:.4f} F1={clean_f1:.4f} AUC={clean_auc if clean_auc else 'N/A'}")

    # ==== 评估 4 个对抗集 ====
    variants_adv = args.variants_adv.split(",")
    for v in variants_adv:
        v = v.strip()
        print(f"\n[AblAdv] 加载对抗集 {v}...")
        try:
            adv_samples = load_adversarial_test(args.dataset, v)
            adv_loader = build_loader(adv_samples, tokenizer, args.batch_size, args.max_length)
            adv_metrics = evaluate_model(model, adv_loader, device, is_social_debias=True)
            results[f"adv_{v}"] = adv_metrics
            adv_acc = adv_metrics.get("accuracy", adv_metrics.get("acc"))
            adv_f1 = adv_metrics.get("f1")
            print(f"[AblAdv] 对抗 {v}: Acc={adv_acc:.4f} F1={adv_f1:.4f}")
        except FileNotFoundError as e:
            print(f"[AblAdv] 对抗集 {v} 文件不存在：{e}")
            results[f"adv_{v}"] = {"error": str(e)}

    # ==== 计算平均 F1 降幅 ====
    adv_f1s = []
    for v in variants_adv:
        v = v.strip()
        m = results.get(f"adv_{v}", {})
        if "f1" in m:
            adv_f1s.append(m["f1"])
    if adv_f1s:
        avg_adv_f1 = sum(adv_f1s) / len(adv_f1s)
        f1_drop = clean_f1 - avg_adv_f1
        results["summary"] = {
            "clean_f1": clean_f1,
            "avg_adv_f1": avg_adv_f1,
            "f1_drop": f1_drop,
            "retention_rate": avg_adv_f1 / clean_f1 if clean_f1 > 0 else 0,
        }
        print(f"\n[AblAdv] 平均对抗 F1={avg_adv_f1:.4f}, F1 降幅={f1_drop:.4f} ({f1_drop*100:.2f}pp)")

    # ==== 保存 ====
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ablation_adv_{args.dataset}_{args.variant}_seed{args.seed}.json"

    result_full = {
        "dataset": args.dataset,
        "variant": args.variant,
        "seed": args.seed,
        "ckpt": args.ckpt,
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(result_full, f, indent=2, default=str)
    print(f"\n[AblAdv] 结果已保存: {out_path}")


if __name__ == "__main__":
    main()