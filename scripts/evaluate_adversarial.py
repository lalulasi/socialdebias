"""
对抗鲁棒性评估脚本

同时加载基线 BERT 和 SocialDebias 两个 checkpoint，
在干净测试集和 4 个对抗测试集（A/B/C/D）上评估，
输出对比表。

对应老师意见 18：鲁棒性指标体系（增加 ASR、性能保持率等）
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm
from collections import OrderedDict

from utils.device import get_device
from utils.dataloader import FakeNewsDataset
from utils.real_dataloader import load_sheepdog_pkl
from modeling.bert_classifier import BertClassifier
from modeling.social_debias import SocialDebiasModel
from configs.base_config import get_config


def load_adversarial_test(dataset: str, variant: str) -> list:
    """加载一个对抗测试集（A/B/C/D 变体之一）"""
    path = f"./data/sheepdog/adversarial_test/{dataset}_test_adv_{variant}.pkl"
    return load_sheepdog_pkl(path)


def evaluate_model(model, loader, device, is_social_debias=False):
    """在 loader 上评估模型"""
    model.eval()
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            if is_social_debias:
                logits = model.predict(input_ids, attention_mask)
            else:
                logits = model(input_ids, attention_mask)

            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=-1).cpu().numpy()

            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, average="macro"),
        "auc": roc_auc_score(all_labels, all_probs) if len(set(all_labels.tolist())) > 1 else float("nan"),
        "preds": all_preds,
        "labels": all_labels,
    }


def compute_asr(clean_result, adv_result):
    """
    计算攻击成功率（Attack Success Rate）
    ASR = 在干净集上预测正确但在对抗集上预测错误的样本比例

    前提：两组样本顺序一致（都是对同一批新闻的原始/改写版）
    """
    clean_correct = (clean_result["preds"] == clean_result["labels"])
    adv_wrong = (adv_result["preds"] != adv_result["labels"])

    # 在"干净集上正确"的样本中，"对抗集上错误"的比例
    flipped = clean_correct & adv_wrong
    n_correct_on_clean = clean_correct.sum()

    if n_correct_on_clean == 0:
        return 0.0
    return flipped.sum() / n_correct_on_clean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="politifact")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--variants", type=str, default="A,B,C,D",
                        help="要评估的对抗变体，逗号分隔")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    print("=" * 80)
    print("对抗鲁棒性评估")
    print("=" * 80)

    device = get_device()
    print(f"设备: {device}")

    config = get_config(mode="dev_real", language=args.language, dataset=args.dataset)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # 加载干净测试集
    print(f"\n加载干净测试集: {args.dataset}")
    from utils.real_dataloader import load_dataset
    _, _, test_data = load_dataset(dataset_name=args.dataset, seed=42)

    # 加载对抗测试集
    variants = args.variants.split(",")
    adv_datasets = {}
    for v in variants:
        print(f"加载对抗测试集 (变体 {v})")
        adv_data = load_adversarial_test(args.dataset, v)
        adv_datasets[v] = adv_data
        print(f"  变体 {v}: {len(adv_data)} 条样本")

    # 构建 DataLoader
    def make_loader(data):
        ds = FakeNewsDataset(data, tokenizer, config.max_length)
        return DataLoader(ds, batch_size=config.batch_size, shuffle=False, num_workers=0)

    clean_loader = make_loader(test_data)
    adv_loaders = {v: make_loader(d) for v, d in adv_datasets.items()}

    # ============ 加载并评估基线 BERT ============
    print("\n" + "=" * 80)
    print("【1/2】评估基线 BERT")
    print("=" * 80)

    baseline_ckpt = f"./results/models/baseline_{args.dataset}_{args.language}_seed{args.seed}.pt"
    if not os.path.exists(baseline_ckpt):
        print(f"⚠️  未找到基线 checkpoint: {baseline_ckpt}")
        print("请先运行 train_baseline.py")
        return

    baseline_model = BertClassifier(model_name=config.model_name).to(device)
    ckpt = torch.load(baseline_ckpt, map_location=device)
    baseline_model.load_state_dict(ckpt["model_state_dict"])
    print(f"加载 baseline checkpoint (val F1={ckpt.get('val_f1', 'N/A'):.4f})")

    baseline_results = OrderedDict()
    baseline_results["clean"] = evaluate_model(baseline_model, clean_loader, device, is_social_debias=False)
    print(f"  干净集: Acc={baseline_results['clean']['accuracy']:.4f} | F1={baseline_results['clean']['f1']:.4f}")

    for v in variants:
        baseline_results[v] = evaluate_model(baseline_model, adv_loaders[v], device, is_social_debias=False)
        print(f"  对抗{v}: Acc={baseline_results[v]['accuracy']:.4f} | F1={baseline_results[v]['f1']:.4f}")

    del baseline_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ============ 加载并评估 SocialDebias ============
    print("\n" + "=" * 80)
    print("【2/2】评估 SocialDebias")
    print("=" * 80)

    sd_ckpt = f"./results/models/socialdebias_{args.dataset}_{args.language}_seed{args.seed}.pt"
    if not os.path.exists(sd_ckpt):
        print(f"⚠️  未找到 SocialDebias checkpoint: {sd_ckpt}")
        print("请先运行 train_socialdebias.py")
        return

    sd_model = SocialDebiasModel(
        model_name=config.model_name,
        use_frozen_bert=False,  # 推理不需要冻结 BERT，省内存
    ).to(device)
    ckpt = torch.load(sd_ckpt, map_location=device)
    sd_model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"加载 SocialDebias checkpoint (val F1={ckpt.get('val_f1', 'N/A'):.4f})")

    sd_results = OrderedDict()
    sd_results["clean"] = evaluate_model(sd_model, clean_loader, device, is_social_debias=True)
    print(f"  干净集: Acc={sd_results['clean']['accuracy']:.4f} | F1={sd_results['clean']['f1']:.4f}")

    for v in variants:
        sd_results[v] = evaluate_model(sd_model, adv_loaders[v], device, is_social_debias=True)
        print(f"  对抗{v}: Acc={sd_results[v]['accuracy']:.4f} | F1={sd_results[v]['f1']:.4f}")

    # ============ 汇总输出对比表 ============
    print("\n" + "=" * 80)
    print("鲁棒性对比表")
    print("=" * 80)

    # 表头（加入 Acc 和 AUC）
    header = f"\n{'测试集':<12} | {'基线 Acc':>9} | {'基线 F1':>9} | {'基线 AUC':>9} | {'Ours Acc':>9} | {'Ours F1':>9} | {'Ours AUC':>9} | {'基线 ASR':>9} | {'Ours ASR':>9}"
    print(header)
    print("-" * len(header))

    def fmt_row(name, baseline, ours, baseline_asr=None, ours_asr=None):
        asr_b = f"{baseline_asr:>9.4f}" if baseline_asr is not None else f"{'N/A':>9}"
        asr_o = f"{ours_asr:>9.4f}" if ours_asr is not None else f"{'N/A':>9}"
        return (f"{name:<12} | "
                f"{baseline['accuracy']:>9.4f} | {baseline['f1']:>9.4f} | {baseline['auc']:>9.4f} | "
                f"{ours['accuracy']:>9.4f} | {ours['f1']:>9.4f} | {ours['auc']:>9.4f} | "
                f"{asr_b} | {asr_o}")

    # 干净集
    print(fmt_row('干净 Clean', baseline_results['clean'], sd_results['clean']))

    # 各对抗变体
    baseline_avg_f1_drop = 0
    sd_avg_f1_drop = 0
    baseline_avg_auc_drop = 0
    sd_avg_auc_drop = 0
    baseline_avg_asr = 0
    sd_avg_asr = 0

    for v in variants:
        baseline_asr = compute_asr(baseline_results["clean"], baseline_results[v])
        sd_asr = compute_asr(sd_results["clean"], sd_results[v])

        baseline_drop = baseline_results["clean"]["f1"] - baseline_results[v]["f1"]
        sd_drop = sd_results["clean"]["f1"] - sd_results[v]["f1"]
        baseline_auc_drop = baseline_results["clean"]["auc"] - baseline_results[v]["auc"]
        sd_auc_drop = sd_results["clean"]["auc"] - sd_results[v]["auc"]

        print(fmt_row(f'对抗 {v}', baseline_results[v], sd_results[v], baseline_asr, sd_asr))

        baseline_avg_f1_drop += baseline_drop
        sd_avg_f1_drop += sd_drop
        baseline_avg_auc_drop += baseline_auc_drop
        sd_avg_auc_drop += sd_auc_drop
        baseline_avg_asr += baseline_asr
        sd_avg_asr += sd_asr

    n_variants = len(variants)
    print("-" * len(header))
    print(f"{'平均 F1 降幅':<12} | {'':<9} | {baseline_avg_f1_drop / n_variants:>9.4f} | {'':<9} | "
          f"{'':<9} | {sd_avg_f1_drop / n_variants:>9.4f} | {'':<9} | {'':<9} | {'':<9}")
    print(f"{'平均 AUC 降幅':<12}| {'':<9} | {'':<9} | {baseline_avg_auc_drop / n_variants:>9.4f} | "
          f"{'':<9} | {'':<9} | {sd_avg_auc_drop / n_variants:>9.4f} | {'':<9} | {'':<9}")
    print(f"{'平均 ASR':<12} | {'':<9} | {'':<9} | {'':<9} | "
          f"{'':<9} | {'':<9} | {'':<9} | {baseline_avg_asr / n_variants:>9.4f} | {sd_avg_asr / n_variants:>9.4f}")

    # 鲁棒性改善
    print("\n" + "=" * 80)
    print("鲁棒性改善")
    print("=" * 80)
    f1_drop_improvement = baseline_avg_f1_drop / n_variants - sd_avg_f1_drop / n_variants
    auc_drop_improvement = baseline_avg_auc_drop / n_variants - sd_avg_auc_drop / n_variants
    asr_improvement = baseline_avg_asr / n_variants - sd_avg_asr / n_variants
    print(f"平均 F1 下降减少:  {f1_drop_improvement * 100:+.2f} 百分点 (越大越好)")
    print(f"平均 AUC 下降减少: {auc_drop_improvement * 100:+.2f} 百分点 (越大越好)")
    print(f"平均 ASR 降低:    {asr_improvement * 100:+.2f} 百分点 (越大越好)")

    if f1_drop_improvement > 0:
        print(f"\n✅ SocialDebias 比基线更鲁棒（F1 下降减少 {f1_drop_improvement * 100:.2f}pp）")
    else:
        print(f"\n⚠️  SocialDebias 在该数据上未显示出鲁棒性优势")
        print("   可能原因：单次实验随机性 / lambda 参数需要调整 / 训练 epoch 不够")

    print("\n评估完成")


if __name__ == "__main__":
    main()