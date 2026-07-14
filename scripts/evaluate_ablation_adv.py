"""
消融实验对抗集评估脚本。
给定一个消融 ckpt，在 clean + 4 个对抗集（A/B/C/D）上评估，
输出 F1/Acc/AUC/ASR 到 JSON。

用法:
    python scripts/evaluate_ablation_adv.py --dataset gossipcop --seed 42 --save_suffix abl_full

保存: results/ablation_adv/ablation_adv_{dataset}_{save_suffix}_seed{seed}.json
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

from scripts.evaluate_adversarial import load_adversarial_test
from scripts.evaluate_surface_adv import (
    SurfaceTestDataset,
    compute_paired_asr,
    evaluate_model,
)
from utils.real_dataloader import load_dataset
from utils.device import get_device
from utils.surface_features import SurfaceFeatureExtractor
from modeling.social_debias import SocialDebiasModel


def build_loader(samples, tokenizer, extractor, feat_mean, feat_std,
                 batch_size=16, max_length=512):
    ds = SurfaceTestDataset(
        samples, tokenizer, max_length, extractor, feat_mean, feat_std
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None, help="显式指定消融 ckpt 路径")
    parser.add_argument("--dataset", type=str, required=True, choices=["politifact", "gossipcop"])
    parser.add_argument("--variant", type=str, default=None,
                        help="旧参数兼容；未传 save_suffix 时映射成 abl_{variant}")
    parser.add_argument("--save_suffix", type=str, default=None)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--bert_name", type=str, default="bert-base-uncased")
    parser.add_argument("--surface_feat_dim", type=int, default=None,
                        help="仅用于校验检查点的表层特征维度")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--variants_adv", type=str, default="A,B,C,D",
                        help="要评估的对抗变体（逗号分隔）")
    parser.add_argument("--output_dir", type=str, default="results/ablation_adv")
    args = parser.parse_args()
    if args.save_suffix is None:
        args.save_suffix = f"abl_{args.variant}" if args.variant else "abl_full"
    if args.ckpt is None:
        args.ckpt = (
            f"results/models/socialdebias_{args.dataset}_{args.language}"
            f"_seed{args.seed}_{args.save_suffix}.pt"
        )

    device = get_device()
    print(f"[AblAdv] device={device}")
    print(f"[AblAdv] dataset={args.dataset} suffix={args.save_suffix} seed={args.seed}")
    print(f"[AblAdv] ckpt: {args.ckpt}")

    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"检查点格式不正确：{args.ckpt}")

    config = checkpoint.get("config", {})
    surface_dim = config.get("surface_feat_dim")
    if surface_dim is None:
        raise ValueError("检查点未保存 surface_feat_dim，无法保证评测配置一致")
    if args.surface_feat_dim is not None and args.surface_feat_dim != surface_dim:
        raise ValueError(
            f"指定的 surface_feat_dim={args.surface_feat_dim} 与检查点的 {surface_dim} 不一致"
        )

    feat_mean = checkpoint.get("feat_mean")
    feat_std = checkpoint.get("feat_std")
    if surface_dim > 0 and (feat_mean is None or feat_std is None):
        raise ValueError("检查点缺少表层特征的均值或标准差")

    # 根据检查点配置重建模型
    model = SocialDebiasModel(
        model_name=args.bert_name,
        num_classes=2,
        hidden_dim=384,
        dropout=0.1,
        grl_lambda=1.0,
        use_frozen_bert=True,
        surface_feat_dim=surface_dim,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.bert_name)
    extractor = SurfaceFeatureExtractor(dim=surface_dim) if surface_dim > 0 else None

    # ==== 加载干净测试集 ====
    print(f"\n[AblAdv] 加载干净测试集 {args.dataset}")
    _, _, clean_samples = load_dataset(dataset_name=args.dataset, seed=args.seed)
    clean_loader = build_loader(
        clean_samples, tokenizer, extractor, feat_mean, feat_std,
        args.batch_size, args.max_length,
    )

    # ==== 评估干净集 ====
    results = OrderedDict()
    clean_metrics, clean_pred, clean_labels = evaluate_model(model, clean_loader, device)
    results["clean"] = clean_metrics
    clean_acc = clean_metrics["accuracy"]
    clean_f1 = clean_metrics.get("f1")
    clean_auc = clean_metrics.get("auc")
    print(f"[AblAdv] 干净集: Acc={clean_acc:.4f} F1={clean_f1:.4f} AUC={clean_auc if clean_auc else 'N/A'}")

    # ==== 评估 4 个对抗集 ====
    variants_adv = args.variants_adv.split(",")
    asr_per_variant = {}
    for v in variants_adv:
        v = v.strip()
        print(f"\n[AblAdv] 加载对抗集 {v}...")
        try:
            adv_samples = load_adversarial_test(args.dataset, v)
            adv_loader = build_loader(
                adv_samples, tokenizer, extractor, feat_mean, feat_std,
                args.batch_size, args.max_length,
            )
            adv_metrics, adv_pred, adv_labels = evaluate_model(model, adv_loader, device)
            if not torch.equal(
                torch.as_tensor(clean_labels), torch.as_tensor(adv_labels)
            ):
                raise ValueError(f"对抗变体 {v} 的标签顺序与干净测试集不一致")
            asr, attacked, clean_correct = compute_paired_asr(
                clean_pred, clean_labels, adv_pred
            )
            asr_per_variant[f"adv_{v}"] = {
                "asr": asr,
                "attacked": attacked,
                "clean_correct": clean_correct,
            }
            results[f"adv_{v}"] = adv_metrics
            adv_acc = adv_metrics["accuracy"]
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
            "asr_per_variant": asr_per_variant,
            "avg_asr": sum(item["asr"] for item in asr_per_variant.values()) / len(asr_per_variant),
        }
        print(f"\n[AblAdv] 平均对抗 F1={avg_adv_f1:.4f}, F1 降幅={f1_drop:.4f} ({f1_drop*100:.2f}pp)")

    # ==== 保存 ====
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ablation_adv_{args.dataset}_{args.save_suffix}_seed{args.seed}.json"

    result_full = {
        "dataset": args.dataset,
        "variant": args.save_suffix,
        "seed": args.seed,
        "ckpt": args.ckpt,
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(result_full, f, indent=2, default=str)
    print(f"\n[AblAdv] 结果已保存: {out_path}")


if __name__ == "__main__":
    main()
