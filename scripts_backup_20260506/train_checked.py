"""
CHECKED 中文虚假新闻检测训练脚本
对比：纯文本 SocialDebias vs SocialDebias + 评论编码

使用:
  # baseline（无评论）
  python scripts/train_checked.py --seed 42 --save_suffix checked_baseline
  
  # 带评论编码
  python scripts/train_checked.py --seed 42 --use_comment --save_suffix checked_comment
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
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
from utils.checked_dataloader import load_checked_dataset
from utils.checked_torch_dataset import CheckedDataset
from utils.device import get_device


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device, use_comment):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            kwargs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            if use_comment:
                kwargs["comment_input_ids"] = batch["comment_input_ids"].to(device)
                kwargs["comment_attention_mask"] = batch["comment_attention_mask"].to(device)
                kwargs["comment_mask"] = batch["comment_mask"].to(device)
            
            out = model(**kwargs)
            logits = out["fact_logits"]
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["label"].numpy())
            all_probs.extend(probs)
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    f1_fake = f1_score(all_labels, all_preds, pos_label=1)  # 假新闻类 F1
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")
    return {"acc": acc, "f1": f1, "f1_fake": f1_fake, "auc": auc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--max_comments", type=int, default=5)
    parser.add_argument("--max_comment_length", type=int, default=64)
    parser.add_argument("--lambda_fact", type=float, default=1.0)
    parser.add_argument("--lambda_bias", type=float, default=0.5)
    parser.add_argument("--lambda_consist", type=float, default=0.3)
    parser.add_argument("--use_comment", action="store_true",
                        help="是否使用评论编码")
    parser.add_argument("--save_suffix", default="checked")
    parser.add_argument("--save_dir", default="./results/models")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"[CHECKED] device={device}, seed={args.seed}, use_comment={args.use_comment}")

    bert_name = "bert-base-chinese"
    tokenizer = AutoTokenizer.from_pretrained(bert_name)

    # 数据
    train_data, val_data, test_data = load_checked_dataset(seed=args.seed)
    
    train_set = CheckedDataset(train_data, tokenizer, args.max_length, args.max_comments, args.max_comment_length)
    val_set = CheckedDataset(val_data, tokenizer, args.max_length, args.max_comments, args.max_comment_length)
    test_set = CheckedDataset(test_data, tokenizer, args.max_length, args.max_comments, args.max_comment_length)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # 模型
    model = SocialDebiasModel(
        model_name=bert_name,
        num_classes=2,
        hidden_dim=384,
        dropout=0.1,
        grl_lambda=1.0,
        use_frozen_bert=True,
        use_comment_encoder=args.use_comment,
        comment_model_name=bert_name,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # 类别加权交叉熵（处理 1:5 不均衡）
    n_fake = sum(1 for s in train_data if s["label"] == 1)
    n_real = sum(1 for s in train_data if s["label"] == 0)
    weight = torch.tensor([1.0, n_real / max(n_fake, 1)], dtype=torch.float32).to(device)
    print(f"[CHECKED] 类别权重: real=1.0, fake={weight[1].item():.2f}")
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=0.1)

    # 训练
    save_path = f"{args.save_dir}/socialdebias_zh_seed{args.seed}_{args.save_suffix}.pt"
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
    best_val_f1 = -1.0
    best_test = None
    history = []
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        ll = {"total": [], "fact": [], "bias": [], "consist": []}
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            kwargs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            if args.use_comment:
                kwargs["comment_input_ids"] = batch["comment_input_ids"].to(device)
                kwargs["comment_attention_mask"] = batch["comment_attention_mask"].to(device)
                kwargs["comment_mask"] = batch["comment_mask"].to(device)
            
            labels = batch["label"].to(device)
            outputs = model(**kwargs)
            
            loss_fact = criterion(outputs["fact_logits"], labels)
            loss_bias = torch.tensor(0.0, device=device)
            loss_consist = torch.tensor(0.0, device=device)
            
            if args.lambda_bias > 0:
                loss_bias = criterion(outputs["bias_logits"], labels)
            if args.lambda_consist > 0 and "frozen_repr" in outputs:
                cos = torch.nn.functional.cosine_similarity(
                    outputs["shared_repr"], outputs["frozen_repr"], dim=-1
                )
                loss_consist = (1.0 - cos).mean()
            
            loss = (args.lambda_fact * loss_fact +
                    args.lambda_bias * loss_bias +
                    args.lambda_consist * loss_consist)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            ll["total"].append(loss.item())
            ll["fact"].append(loss_fact.item())
            ll["bias"].append(loss_bias.item() if torch.is_tensor(loss_bias) else 0.0)
            ll["consist"].append(loss_consist.item() if torch.is_tensor(loss_consist) else 0.0)
        
        epoch_time = time.time() - t0
        means = {k: np.mean(v) if v else 0.0 for k, v in ll.items()}
        val_m = evaluate(model, val_loader, device, args.use_comment)
        history.append({"epoch": epoch, **{f"loss_{k}": v for k,v in means.items()},
                        **{f"val_{k}": v for k,v in val_m.items()}})
        
        print(f"[E{epoch}] {epoch_time:.0f}s | total={means['total']:.4f} fact={means['fact']:.4f}")
        print(f"        val acc={val_m['acc']:.4f} f1={val_m['f1']:.4f} f1_fake={val_m['f1_fake']:.4f} auc={val_m['auc']:.4f}")
        
        if val_m["f1"] > best_val_f1:
            best_val_f1 = val_m["f1"]
            test_m = evaluate(model, test_loader, device, args.use_comment)
            best_test = {"epoch": epoch, **test_m}
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_f1": val_m["f1"], "val_acc": val_m["acc"], "val_auc": val_m["auc"],
                "config": {
                    "model_name": bert_name, "language": "zh", "dataset": "checked",
                    "use_comment": args.use_comment,
                    "lambda_fact": args.lambda_fact, "lambda_bias": args.lambda_bias,
                    "lambda_consist": args.lambda_consist,
                },
            }, save_path)
            print(f"        ⭐ test acc={test_m['acc']:.4f} f1={test_m['f1']:.4f} f1_fake={test_m['f1_fake']:.4f}")

    # 保存历史
    history_path = save_path.replace(".pt", "_history.json")
    with open(history_path, "w") as f:
        json.dump({"args": vars(args), "best_val_f1": best_val_f1,
                   "best_test": best_test, "history": history}, f, indent=2, default=str)
    print(f"\n[CHECKED] 完成。ckpt: {save_path}")
    print(f"          best test: {best_test}")


if __name__ == "__main__":
    main()