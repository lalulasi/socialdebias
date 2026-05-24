"""
========================================================================
  scripts/aggregate_surface_3seeds.py
  PolitiFact 3 seed 主表汇总脚本（支持 surface / surface_adaptive / surface_all 等）

  论文表 5-3 默认使用 surface_adaptive 配置（与表 5-12 自适应缩放章节叙事一致）
========================================================================

  用途：批量评估 3 个 seed 的 ckpt（PolitiFact 90 条测试集 × clean + adv_A/B/C/D
        4 变体），汇总 Macro-F1 / F1 Drop / ASR 的均值与标准差，输出 CSV 与
        可直接粘贴到论文表 5-3 的 Markdown 片段。

  评估口径：
    - F1：sklearn macro F1
    - F1 Drop = Clean F1 − mean(Adv_A/B/C/D F1)
    - ASR     = N_flipped / N_correct_clean
        其中 N_correct_clean = clean 阶段预测正确的样本数
              N_flipped       = 这些样本中在 adv 上翻转为错误的数量

  使用：
    # 论文表 5-3 主结果（surface_adaptive 配置，3 seed）— 默认就是这个
    python scripts/aggregate_surface_3seeds.py

    # 等价于显式写出全部参数：
    python scripts/aggregate_surface_3seeds.py \\
        --ckpt_dir /socialdebias/results/models \\
        --config surface_adaptive \\
        --dataset politifact \\
        --seeds 42 2024 3407

    # 同时跑其他配置（消融用）
    python scripts/aggregate_surface_3seeds.py --config surface
    python scripts/aggregate_surface_3seeds.py --config surface_all
    python scripts/aggregate_surface_3seeds.py --config nli_soft14

  ckpt 命名约定（用户项目，位于 /socialdebias/results/models/）：
    socialdebias_politifact_en_seed42_surface_adaptive.pt
    socialdebias_politifact_en_seed2024_surface_adaptive.pt
    socialdebias_politifact_en_seed3407_surface_adaptive.pt

  对抗集 pkl 路径（请按你项目实际路径修改 _build_adv_pkl_paths）：
    data/SheepDog/politifact_test.pkl
    data/SheepDog/politifact_test_adv_A.pkl
    ...
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

# ====== 项目模块导入（按你项目实际结构调整）======
# 若以下 import 失败，请把脚本放到 socialdebias/ 根目录下运行
try:
    from modeling.social_debias import SocialDebiasModel
    from utils.real_dataloader import load_sheepdog_pkl, build_dataloader
    from utils.surface_features import extract_surface_features
except ImportError as e:
    print(f"[WARN] 项目模块导入失败：{e}")
    print("[WARN] 请确认从 socialdebias/ 根目录运行此脚本")
    print("[WARN] 或使用 --consume_csv 模式直接消费 evaluate_adversarial.py 的输出")


# =========================================================================
# Section 1：单 seed × 单变体的评估（核心函数）
# =========================================================================
def evaluate_one_variant(
    model: torch.nn.Module,
    pkl_path: str,
    device: torch.device,
    batch_size: int = 16,
    max_length: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    在某个 pkl（clean 或 adv_X）上跑一次推理，返回 (y_true, y_pred)
    """
    samples = load_sheepdog_pkl(pkl_path)
    dataloader = build_dataloader(
        samples,
        batch_size=batch_size,
        max_length=max_length,
        shuffle=False,
        surface_extractor=extract_surface_features,
    )

    model.eval()
    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"  {Path(pkl_path).stem:30s}", leave=False):
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            surface  = batch["surface"].to(device)
            labels   = batch["label"]

            logits = model(input_ids, attn_mask, surface_features=surface)["fact_logits"]
            preds = logits.argmax(dim=-1).cpu().numpy()

            y_true_all.append(labels.numpy())
            y_pred_all.append(preds)

    return np.concatenate(y_true_all), np.concatenate(y_pred_all)


# =========================================================================
# Section 2：单 seed 的完整评估（clean + 4 adv 变体）+ ASR 计算
# =========================================================================
def evaluate_one_seed(
    seed: int,
    ckpt_path: str,
    adv_pkl_paths: Dict[str, str],   # {"clean": "...", "adv_A": "...", ...}
    device: torch.device,
    surface_feat_dim: int = 8,
) -> Dict:
    """
    返回字典：
        {
          "seed": 42,
          "clean_acc": ..., "clean_f1": ...,
          "adv_A_f1": ..., ..., "adv_D_f1": ...,
          "avg_adv_f1": ...,
          "f1_drop_pp": ...,           # 单位：百分点
          "asr": ...,                  # 0-1 范围
          "n_clean_correct": ...,
          "n_flipped": ...,
        }
    """
    print(f"\n[Seed {seed}] 加载 ckpt: {ckpt_path}")
    model = SocialDebiasModel(surface_feat_dim=surface_feat_dim).to(device)
    state = torch.load(ckpt_path, map_location=device)
    # 兼容两种存储格式
    state_dict = state.get("model_state_dict", state) if isinstance(state, dict) else state
    model.load_state_dict(state_dict)

    result = {"seed": seed}

    # ---- Clean ----
    y_true_clean, y_pred_clean = evaluate_one_variant(
        model, adv_pkl_paths["clean"], device
    )
    clean_acc = accuracy_score(y_true_clean, y_pred_clean)
    clean_f1  = f1_score(y_true_clean, y_pred_clean, average="macro")
    result["clean_acc"] = clean_acc
    result["clean_f1"]  = clean_f1
    print(f"  Clean: Acc={clean_acc:.4f} F1={clean_f1:.4f}")

    # clean 阶段预测正确的 mask（ASR 的分母）
    correct_mask = (y_true_clean == y_pred_clean)
    n_correct_clean = int(correct_mask.sum())
    result["n_clean_correct"] = n_correct_clean

    # ---- 4 个对抗变体 ----
    adv_f1s = []
    n_flipped_total = 0
    n_compare_total = 0

    for variant in ["adv_A", "adv_B", "adv_C", "adv_D"]:
        y_true_adv, y_pred_adv = evaluate_one_variant(
            model, adv_pkl_paths[variant], device
        )
        adv_f1 = f1_score(y_true_adv, y_pred_adv, average="macro")
        adv_acc = accuracy_score(y_true_adv, y_pred_adv)
        result[f"{variant}_f1"]  = adv_f1
        result[f"{variant}_acc"] = adv_acc
        adv_f1s.append(adv_f1)

        # ASR 计算（按变体）
        # 注意：clean 与 adv 是 1-1 对应的（同一 90 条样本的改写）
        if len(y_true_adv) == len(y_true_clean):
            # 在 clean 正确的样本子集里，统计 adv 是否翻转为错误
            flipped = correct_mask & (y_true_adv != y_pred_adv)
            n_flipped_v = int(flipped.sum())
            asr_v = n_flipped_v / max(n_correct_clean, 1)
            result[f"{variant}_asr"] = asr_v
            n_flipped_total += n_flipped_v
            n_compare_total += n_correct_clean
            print(f"  {variant}: Acc={adv_acc:.4f} F1={adv_f1:.4f} ASR={asr_v:.4f}")
        else:
            print(f"  [WARN] {variant} 样本数与 clean 不一致，跳过 ASR 计算")
            result[f"{variant}_asr"] = float("nan")

    # ---- 汇总 ----
    avg_adv_f1 = float(np.mean(adv_f1s))
    f1_drop_pp = (clean_f1 - avg_adv_f1) * 100  # 百分点

    # Avg ASR（按变体取平均，与论文 5.3.1 表口径一致）
    asr_per_variant = [result.get(f"adv_{v}_asr", np.nan) for v in ["A", "B", "C", "D"]]
    avg_asr = float(np.nanmean(asr_per_variant))

    result["avg_adv_f1"]  = avg_adv_f1
    result["f1_drop_pp"]  = f1_drop_pp
    result["asr"]         = avg_asr
    result["n_flipped"]   = n_flipped_total

    print(f"  [Summary] Clean F1={clean_f1:.4f}  Avg Adv F1={avg_adv_f1:.4f}  "
          f"F1 Drop={f1_drop_pp:.2f}pp  ASR={avg_asr*100:.2f}%")

    return result


# =========================================================================
# Section 3：CSV 消费模式 —— 如果已有 evaluate_adversarial.py 单 seed 输出
# =========================================================================
def consume_existing_csv(csv_paths: List[str]) -> List[Dict]:
    """
    直接消费 evaluate_adversarial.py 输出的单 seed CSV
    每个 CSV 应包含列：variant (clean/adv_A/.../adv_D), acc, f1, asr
    返回与 evaluate_one_seed 同样格式的 dict 列表
    """
    rows = []
    for csv_path in csv_paths:
        seed = _infer_seed_from_filename(csv_path)
        df = pd.read_csv(csv_path)
        result = {"seed": seed}
        for _, r in df.iterrows():
            v = r["variant"]
            if v == "clean":
                result["clean_acc"] = float(r["acc"])
                result["clean_f1"]  = float(r["f1"])
            else:
                result[f"{v}_acc"] = float(r["acc"])
                result[f"{v}_f1"]  = float(r["f1"])
                if "asr" in df.columns:
                    result[f"{v}_asr"] = float(r["asr"])
        adv_f1s = [result.get(f"adv_{x}_f1") for x in ["A", "B", "C", "D"]]
        adv_f1s = [x for x in adv_f1s if x is not None]
        result["avg_adv_f1"] = float(np.mean(adv_f1s))
        result["f1_drop_pp"] = (result["clean_f1"] - result["avg_adv_f1"]) * 100
        if all(f"adv_{x}_asr" in result for x in ["A", "B", "C", "D"]):
            result["asr"] = float(np.mean(
                [result[f"adv_{x}_asr"] for x in ["A", "B", "C", "D"]]
            ))
        rows.append(result)
    return rows


def _infer_seed_from_filename(path: str) -> int:
    """从文件名 surface_politifact_seed42.csv 中提取 seed"""
    import re
    m = re.search(r"seed[_-]?(\d+)", Path(path).stem)
    return int(m.group(1)) if m else -1


# =========================================================================
# Section 4：3 seed 跨 seed 聚合 + 论文行片段
# =========================================================================
def aggregate_seeds(seed_results: List[Dict]) -> Dict:
    """
    将 3 个 seed 的字典结果聚合为均值 ± std
    """
    df = pd.DataFrame(seed_results)
    agg = {}
    for col in ["clean_acc", "clean_f1",
                "adv_A_f1", "adv_B_f1", "adv_C_f1", "adv_D_f1",
                "avg_adv_f1", "f1_drop_pp", "asr"]:
        if col in df.columns:
            agg[f"{col}_mean"] = float(df[col].mean())
            agg[f"{col}_std"]  = float(df[col].std(ddof=1)) if len(df) > 1 else 0.0
    agg["n_seeds"]   = len(seed_results)
    agg["seed_list"] = sorted([r["seed"] for r in seed_results])
    return agg


def format_paper_table_row(agg: Dict, config_name: str = "surface") -> str:
    """
    生成可粘贴到论文表 5-3 的 Markdown 行（也可手抄到 Word 表格里）
    """
    md = []
    md.append(f"## 论文表 5-3 SocialDebias 行（{config_name} 配置, "
              f"{agg['n_seeds']} seed: {agg['seed_list']}）\n")
    md.append("| 模型配置 | Clean Macro-F1 | Adv 平均 Macro-F1 | F1 Drop | ASR |")
    md.append("|---|---|---|---|---|")
    line = (
        f"| SocialDebias (本文方法, {config_name} 配置, 8 维门控) "
        f"| {agg['clean_f1_mean']:.4f} ± {agg['clean_f1_std']:.4f} "
        f"| {agg['avg_adv_f1_mean']:.4f} ± {agg['avg_adv_f1_std']:.4f} "
        f"| **{agg['f1_drop_pp_mean']:.2f}pp** "
        f"| {agg['asr_mean']*100:.2f}% |"
    )
    md.append(line)
    md.append("")
    md.append(f"**算术校验**：Clean F1 − Avg Adv F1 = "
              f"{agg['clean_f1_mean']:.4f} − {agg['avg_adv_f1_mean']:.4f} = "
              f"{(agg['clean_f1_mean']-agg['avg_adv_f1_mean'])*100:.2f}pp ≈ "
              f"F1 Drop {agg['f1_drop_pp_mean']:.2f}pp ✓")
    md.append("")
    md.append("**逐 seed 数据**：")
    return "\n".join(md)


def format_per_seed_table(seed_results: List[Dict]) -> str:
    """逐 seed 明细表"""
    md = ["", "| Seed | Clean F1 | adv_A | adv_B | adv_C | adv_D | Avg Adv | F1 Drop | ASR |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(seed_results, key=lambda x: x["seed"]):
        md.append(
            f"| {r['seed']} | {r['clean_f1']:.4f} "
            f"| {r.get('adv_A_f1', 0):.4f} | {r.get('adv_B_f1', 0):.4f} "
            f"| {r.get('adv_C_f1', 0):.4f} | {r.get('adv_D_f1', 0):.4f} "
            f"| {r['avg_adv_f1']:.4f} | {r['f1_drop_pp']:.2f}pp "
            f"| {r.get('asr', 0)*100:.2f}% |"
        )
    return "\n".join(md)


# =========================================================================
# Section 5：路径构造（按你项目实际命名调整）
# =========================================================================
def _build_ckpt_path(ckpt_dir: str, dataset: str, seed: int,
                     config: str = "surface", language: str = "en") -> str:
    """
    用户项目实际命名约定（/socialdebias/results/models/ 下）：
      socialdebias_politifact_en_seed42_surface.pt
      socialdebias_politifact_en_seed2024_surface_adaptive.pt
      socialdebias_politifact_en_seed3407_surface_all.pt
    """
    name = f"socialdebias_{dataset}_{language}_seed{seed}_{config}.pt"
    path = Path(ckpt_dir) / name
    if path.exists():
        return str(path)
    # 备用候选（兼容其他命名格式）
    candidates = [
        Path(ckpt_dir) / f"socialdebias_{dataset}_seed{seed}_{config}.pt",
        Path(ckpt_dir) / config / f"socialdebias_{dataset}_{language}_seed{seed}.pt",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        f"未找到 ckpt: {path}\n候选：{candidates}"
    )


def _build_adv_pkl_paths(dataset: str, data_root: str = "data/SheepDog") -> Dict[str, str]:
    """SheepDog 对抗集 pkl 约定路径"""
    return {
        "clean": f"{data_root}/{dataset}_test.pkl",
        "adv_A": f"{data_root}/{dataset}_test_adv_A.pkl",
        "adv_B": f"{data_root}/{dataset}_test_adv_B.pkl",
        "adv_C": f"{data_root}/{dataset}_test_adv_C.pkl",
        "adv_D": f"{data_root}/{dataset}_test_adv_D.pkl",
    }


# =========================================================================
# Section 6：主入口
# =========================================================================
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt_dir", type=str, default="./results/models",
                        help="ckpt 所在目录（默认 /socialdebias/results/models）")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 2024, 3407],
                        help="要评估的 seed 列表（默认 42 2024 3407，与基线 3 seed 口径一致）")
    parser.add_argument("--config", type=str, default="surface_adaptive",
                        choices=["surface", "surface_adaptive", "surface_all",
                                 "nli_soft14", "nli_hard", "nli_soft",
                                 "surface_contrast", "liar_speaker"],
                        help="ckpt 配置名（决定文件名后缀，默认 surface_adaptive 用于论文表 5-3）")
    parser.add_argument("--dataset", type=str, default="politifact",
                        choices=["politifact", "gossipcop", "weibo21"])
    parser.add_argument("--language", type=str, default="en",
                        choices=["en", "zh"],
                        help="数据集语言后缀（PolitiFact/GossipCop=en, Weibo21=zh）")
    parser.add_argument("--data_root", type=str, default="data/SheepDog/adversarial_test",
                        help="SheepDog 对抗集 pkl 根目录")
    parser.add_argument("--surface_feat_dim", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto",
                        help="auto / cpu / cuda / mps")
    parser.add_argument("--output_csv", type=str,
                        default=None,
                        help="逐 seed 明细 CSV（默认根据 config 自动命名）")
    parser.add_argument("--output_table_md", type=str,
                        default=None,
                        help="论文表行 Markdown（默认根据 config 自动命名）")
    parser.add_argument("--consume_csv", type=str, nargs="+", default=None,
                        help="若提供，则直接消费 evaluate_adversarial.py 的 CSV 输出")
    args = parser.parse_args()

    # 默认输出文件名（按 config 区分，避免覆盖）
    if args.output_csv is None:
        args.output_csv = f"results/main/{args.config}_{args.dataset}_3seed.csv"
    if args.output_table_md is None:
        args.output_table_md = f"results/main/{args.config}_{args.dataset}_table_row.md"

    # 设备
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"[Device] {device}")

    # ---- 收集每 seed 结果 ----
    if args.consume_csv:
        print(f"[Mode] 消费现有 CSV：{args.consume_csv}")
        seed_results = consume_existing_csv(args.consume_csv)
    else:
        print(f"[Mode] 评估 ckpt 模式")
        print(f"[Config] {args.config}  |  Dataset: {args.dataset}_{args.language}")
        adv_pkl_paths = _build_adv_pkl_paths(args.dataset, args.data_root)
        seed_results = []
        for seed in args.seeds:
            try:
                ckpt_path = _build_ckpt_path(
                    args.ckpt_dir, args.dataset, seed,
                    config=args.config, language=args.language
                )
            except FileNotFoundError as e:
                print(f"[ERROR] {e}")
                continue
            result = evaluate_one_seed(
                seed=seed,
                ckpt_path=ckpt_path,
                adv_pkl_paths=adv_pkl_paths,
                device=device,
                surface_feat_dim=args.surface_feat_dim,
            )
            seed_results.append(result)

    if not seed_results:
        print("[ERROR] 没有任何 seed 评估成功，终止")
        sys.exit(1)

    # ---- 聚合 ----
    agg = aggregate_seeds(seed_results)
    print("\n" + "=" * 70)
    print(f"3 Seed 聚合结果（{args.dataset}, {args.config} 配置）")
    print("=" * 70)
    print(json.dumps(agg, indent=2, ensure_ascii=False))

    # ---- 保存 CSV ----
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(seed_results).to_csv(args.output_csv, index=False, float_format="%.4f")
    print(f"\n[Save] 逐 seed 明细 → {args.output_csv}")

    # ---- 保存论文行 Markdown ----
    Path(args.output_table_md).parent.mkdir(parents=True, exist_ok=True)
    md_content = (
        format_paper_table_row(agg, config_name=args.config)
        + format_per_seed_table(seed_results)
        + "\n\n---\n\n"
        + f"**直接替换表行用以下值（{args.config} 配置）：**\n\n"
        + f"- Clean Macro-F1：**{agg['clean_f1_mean']:.4f}** ± {agg['clean_f1_std']:.4f}\n"
        + f"- Adv 平均 Macro-F1：**{agg['avg_adv_f1_mean']:.4f}** ± {agg['avg_adv_f1_std']:.4f}\n"
        + f"- F1 Drop：**{agg['f1_drop_pp_mean']:.2f}pp**（std {agg['f1_drop_pp_std']:.2f}）\n"
        + f"- ASR：**{agg['asr_mean']*100:.2f}%**（std {agg['asr_std']*100:.2f}）\n"
    )
    with open(args.output_table_md, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[Save] 论文表行 Markdown → {args.output_table_md}")

    print("\n✓ Done.")


if __name__ == "__main__":
    main()
