"""
对比学习增强的 SocialDebias 训练（意见 5 深做版）

在 train_socialdebias.py 基础上加：
- 对比学习数据加载器（原文 + Qwen 对抗版本）
- InfoNCE 损失
- 总损失 = λ_fact * L_fact + λ_bias * L_bias + λ_consist * L_consist + λ_contrast * L_infonce

使用:
    python scripts/train_socialdebias_contrastive.py \\
        --dataset politifact --language en --seed 42 \\
        --lambda_fact 1.0 --lambda_bias 0.5 --lambda_consist 0.3 \\
        --lambda_contrast 0.3 --temperature 0.07 \\
        --adv_pkl data/qwen_adv/politifact_train_adv_filtered_v2.pkl
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from modeling.social_debias import SocialDebiasModel
from modeling.infonce import info_nce_loss
from utils.contrastive_dataloader import ContrastiveFakeNewsDataset
from utils.dataloader import FakeNewsDataset
from utils.real_dataloader import load_dataset
from utils.device import get_device


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"]
            
            outputs = model(input_ids, attention_mask)
            logits = outputs["fact_logits"]
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")
    return {"acc": acc, "f1": f1, "auc": auc}


def main():
    parser = argparse.ArgumentParser()
    # 基本参数
    parser.add_argument("--dataset", default="politifact",
                        choices=["politifact", "gossipcop", "lun"])
    parser.add_argument("--language", default="en", choices=["en", "zh"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=512)
    
    # SocialDebias λ
    parser.add_argument("--lambda_fact", type=float, default=1.0)
    parser.add_argument("--lambda_bias", type=float, default=0.5)
    parser.add_argument("--lambda_consist", type=float, default=0.3)
    
    # 对比学习参数
    parser.add_argument("--lambda_contrast", type=float, default=0.3,
                        help="InfoNCE 损失权重")
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="InfoNCE 温度系数")
    parser.add_argument("--adv_pkl", required=True,
                        help="Qwen 对抗数据 pkl 路径")
    # 在 --adv_pkl 那一行下面加：
    parser.add_argument("--orig_pkl", default=None,
                        help="原数据 pkl 路径（默认 data/sheepdog/news_articles/{dataset}_train.pkl）")
    parser.add_argument("--styles", nargs="+", default=None,
                        help="要使用的风格列表（默认全部）")
    
    # 输出
    parser.add_argument("--save_dir", default="./results/models")
    parser.add_argument("--save_suffix", default="contrastive",
                        help="ckpt 文件名后缀")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"[Contrastive] device={device}, seed={args.seed}")
    print(f"[Contrastive] λ_fact={args.lambda_fact}, λ_bias={args.lambda_bias}, "
          f"λ_consist={args.lambda_consist}, λ_contrast={args.lambda_contrast}, "
          f"τ={args.temperature}")

    # ==== Tokenizer ====
    bert_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(bert_name)

    # ==== 数据 ====
    # train: 用对比学习版（读 Qwen 对抗数据）
    #orig_pkl = f"data/sheepdog/news_articles/{args.dataset}_train.pkl"
    orig_pkl = args.orig_pkl or f"data/sheepdog/news_articles/{args.dataset}_train.pkl"
    train_set = ContrastiveFakeNewsDataset.from_pkl(
        original_pkl_path=orig_pkl,
        adversarial_pkl_path=args.adv_pkl,
        tokenizer=tokenizer,
        max_length=args.max_length,
        styles=args.styles,
    )
    
    # val/test: 用普通数据（load_dataset 切的 val/test）
    _, val_data, test_data = load_dataset(
        dataset_name=args.dataset, seed=args.seed,
    )
    val_set = FakeNewsDataset(val_data, tokenizer, args.max_length)
    test_set = FakeNewsDataset(test_data, tokenizer, args.max_length)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    print(f"[Contrastive] train batches={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")

    # ==== 模型 ====
    model = SocialDebiasModel(
        model_name=bert_name,
        num_classes=2,
        hidden_dim=384,
        dropout=0.1,
        grl_lambda=1.0,
        use_frozen_bert=True,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ==== 训练 ====
    save_path = (f"{args.save_dir}/socialdebias_{args.dataset}_{args.language}"
                 f"_seed{args.seed}_{args.save_suffix}.pt")
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
    best_val_f1 = -1.0
    best_test = None
    history = []
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        loss_components = {"total": [], "fact": [], "bias": [], "consist": [], "contrast": []}
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            # 原文 forward
            orig_ids = batch["orig_input_ids"].to(device)
            orig_mask = batch["orig_attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs_orig = model(orig_ids, orig_mask)
            
            # 对抗 forward（只算 fact_repr，不算其他损失）
            adv_ids = batch["adv_input_ids"].to(device)
            adv_mask = batch["adv_attention_mask"].to(device)
            outputs_adv = model(adv_ids, adv_mask)
            
            # 损失
            # L_fact, L_bias, L_consist 用原文（避免对抗版本污染分类）
            loss_fact = criterion(outputs_orig["fact_logits"], labels)
            
            loss_bias = torch.tensor(0.0, device=device)
            if args.lambda_bias > 0 and "bias_logits" in outputs_orig:
                loss_bias = criterion(outputs_orig["bias_logits"], labels)
            
            loss_consist = torch.tensor(0.0, device=device)
            if args.lambda_consist > 0 and "frozen_repr" in outputs_orig:
                cos = torch.nn.functional.cosine_similarity(
                    outputs_orig["shared_repr"], outputs_orig["frozen_repr"], dim=-1
                )
                loss_consist = (1.0 - cos).mean()
            
            # InfoNCE：原文 fact_repr 和对抗 fact_repr 的对比
            loss_contrast = torch.tensor(0.0, device=device)
            if args.lambda_contrast > 0:
                loss_contrast = info_nce_loss(
                    outputs_orig["fact_repr"],
                    outputs_adv["fact_repr"],
                    temperature=args.temperature,
                )
            
            # 总损失
            loss = (args.lambda_fact * loss_fact
                    + args.lambda_bias * loss_bias
                    + args.lambda_consist * loss_consist
                    + args.lambda_contrast * loss_contrast)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            loss_components["total"].append(loss.item())
            loss_components["fact"].append(loss_fact.item())
            loss_components["bias"].append(loss_bias.item() if torch.is_tensor(loss_bias) else 0.0)
            loss_components["consist"].append(loss_consist.item() if torch.is_tensor(loss_consist) else 0.0)
            loss_components["contrast"].append(loss_contrast.item() if torch.is_tensor(loss_contrast) else 0.0)
        
        # epoch 汇总
        epoch_time = time.time() - epoch_start
        avg_total = np.mean(loss_components["total"])
        avg_fact = np.mean(loss_components["fact"])
        avg_bias = np.mean(loss_components["bias"])
        avg_consist = np.mean(loss_components["consist"])
        avg_contrast = np.mean(loss_components["contrast"])
        
        val_m = evaluate(model, val_loader, device)
        history.append({
            "epoch": epoch, "epoch_time": epoch_time,
            "loss_total": avg_total, "loss_fact": avg_fact,
            "loss_bias": avg_bias, "loss_consist": avg_consist, "loss_contrast": avg_contrast,
            **{f"val_{k}": v for k, v in val_m.items()},
        })
        
        print(f"[E{epoch}] {epoch_time:.0f}s | total={avg_total:.4f} fact={avg_fact:.4f} "
              f"bias={avg_bias:.4f} consist={avg_consist:.4f} contrast={avg_contrast:.4f}")
        print(f"        val acc={val_m['acc']:.4f} f1={val_m['f1']:.4f} auc={val_m['auc']:.4f}")
        
        # 保存最佳
        if val_m["f1"] > best_val_f1:
            best_val_f1 = val_m["f1"]
            test_m = evaluate(model, test_loader, device)
            best_test = {"epoch": epoch, **test_m}
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_f1": val_m["f1"],
                "val_acc": val_m["acc"],
                "val_auc": val_m["auc"],
                "config": {
                    "model_name": bert_name,
                    "language": args.language,
                    "dataset": args.dataset,
                    "lambda_fact": args.lambda_fact,
                    "lambda_bias": args.lambda_bias,
                    "lambda_consist": args.lambda_consist,
                    "lambda_contrast": args.lambda_contrast,
                    "temperature": args.temperature,
                },
            }, save_path)
            print(f"        ⭐ 新最佳！test acc={test_m['acc']:.4f} f1={test_m['f1']:.4f}")
    
    # 保存训练历史 JSON
    import json
    history_path = save_path.replace(".pt", "_history.json")
    with open(history_path, "w") as f:
        json.dump({
            "args": vars(args),
            "best_val_f1": best_val_f1,
            "best_test": best_test,
            "history": history,
        }, f, indent=2, default=str)
    print(f"\n[Contrastive] 训练完成")
    print(f"  ckpt: {save_path}")
    print(f"  history: {history_path}")
    print(f"  best val F1: {best_val_f1:.4f}")
    if best_test:
        print(f"  best test: acc={best_test['acc']:.4f} f1={best_test['f1']:.4f}")


if __name__ == "__main__":
    main()