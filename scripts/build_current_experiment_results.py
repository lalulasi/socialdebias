#!/usr/bin/env python3
"""Build a current-run-only thesis experiment data book.

Unlike build_paper_results_report.py, this document deliberately contains no
legacy paper values and no new-vs-old deltas. It is intended to be downloaded
and compared with the thesis manually.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import build_paper_results_report as common


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def value_text(value, column=""):
    parsed = common.number(value)
    if parsed is None:
        return "—" if value in (None, "") else str(value)
    if column in ("n", "n_seeds", "seed") and float(parsed).is_integer():
        return str(int(parsed))
    return f"{parsed:.4f}"


def add_table(lines, headers, rows):
    if rows:
        lines.extend(common.markdown_table(headers, rows))
    else:
        lines.append("> 本轮没有可用数据。")
    lines.append("")


def add_csv(lines, title, path, columns, predicate=None):
    lines.extend([f"### {title}", ""])
    rows = common.read_csv(path)
    if predicate:
        rows = [row for row in rows if predicate(row)]
    if not rows:
        lines.extend([f"> 缺失或无可用记录：`{path}`", ""])
        return []
    available = [column for column in columns if any(column in row for row in rows)]
    add_table(
        lines,
        available,
        [[value_text(row.get(column), column) for column in available] for row in rows],
    )
    lines.extend([f"来源：`{path}`", ""])
    return rows


def filter_summaries(filter_rows):
    datasets = defaultdict(lambda: [0, 0])
    quality = defaultdict(lambda: [0, 0])
    tones = ("emotionally_triggering", "sensational", "objective", "neutral")
    for row in filter_rows:
        name = row.get("file", "")
        dataset = next((x for x in ("politifact", "gossipcop", "weibo21") if x in name), "unknown")
        source = "qwen" if "qwen" in name else "deepseek" if "deepseek" in name else "unknown"
        tone = next((x for x in tones if x in name), "unknown")
        n_in = int(float(row.get("n_in", 0)))
        n_out = int(float(row.get("n_out", 0)))
        datasets[dataset][0] += n_in
        datasets[dataset][1] += n_out
        quality[("LLM来源", source)][0] += n_in
        quality[("改写风格", tone)][0] += n_in
        quality[("LLM来源", source)][1] += n_out
        quality[("改写风格", tone)][1] += n_out
    return datasets, quality


def add_current_main_tables(lines, results_root, histories):
    main_rows = common.current_main_rows(
        results_root, common.summarize_histories(histories), legacy={}
    )
    for dataset, table in (
        ("politifact", "表5-3"),
        ("gossipcop", "表5-4"),
        ("weibo21", "表5-5"),
    ):
        lines.extend([f"### §5.3.1 / {table}：{dataset} 本轮主结果", ""])
        rendered = []
        for row in main_rows:
            if row.get("dataset") != dataset:
                continue
            rendered.append([
                row.get("method"), row.get("n", "—"),
                common.fmt_ms(row.get("clean_f1_mean"), row.get("clean_f1_std")),
                common.fmt_ms(row.get("avg_adv_f1_mean"), row.get("avg_adv_f1_std")),
                common.fmt_pp(row.get("f1_drop_mean"), row.get("f1_drop_std")),
                common.fmt_pct(row.get("avg_asr_mean")),
                common.fmt_pct(row.get("parse_fail_rate")),
            ])
        add_table(
            lines,
            ["方法", "n", "Clean Macro-F1", "Avg Adv Macro-F1", "F1 Drop", "ASR", "API解析失败率"],
            rendered,
        )
    return main_rows


def build_document(project_root, results_root, endef_root):
    histories, unreadable = common.collect_histories(results_root / "models")
    checks = common.build_checks(project_root, results_root, endef_root)
    filter_path = project_root / "data/socialdebias_adv/filtered/filter_report.csv"
    social_path = results_root / "socialdebias_adv_eval.csv"
    filter_rows = common.read_csv(filter_path)
    social_rows = common.read_csv(social_path)

    lines = [
        "# SocialDebias 本轮全部实验结果数据总册",
        "",
        f"生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}  ",
        f"项目目录：`{project_root}`  ",
        f"本轮结果目录：`{results_root}`  ",
        "",
        "> 本文档只包含 paper-v2 本轮实际实验数据，不包含旧论文数值，不计算新旧差值。请下载后按节号和表号与论文逐项人工比较。",
        "",
        "## 一、数据总览与论文位置",
        "",
    ]
    mapping = [
        row for row in common.build_artifact_mapping(project_root, results_root)
        if row[0] != "P12"
    ]
    add_table(
        lines,
        ["阶段", "论文章节", "表格/图", "本轮实验内容", "数据来源"],
        [[row[0], row[1], row[2], row[3], f"`{row[4]}`"] for row in mapping],
    )

    lines.extend(["## 二、本轮实验完整性", ""])
    add_table(
        lines,
        ["阶段", "产物", "实际", "期望", "状态", "路径"],
        [[item["stage"], item["item"], item["actual"], item["expected"],
          "✅" if item["ok"] else "❌", f"`{item['path']}`"] for item in checks],
    )
    if unreadable:
        lines.extend(["无法读取的 history：", ""])
        lines.extend(f"- `{item}`" for item in unreadable)
        lines.append("")

    lines.extend(["## 三、第4章：对抗数据构造与增强", ""])

    lines.extend(["### P1 / §4.3–§4.4：训练对抗数据、NLI与词典审计", ""])
    audit_path = results_root / "data_preparation/after.json"
    audit_rows = []
    if audit_path.exists():
        audit = common.read_json(audit_path)
        for item in audit.get("assets", []):
            path = item.get("path", item.get("dataset", ""))
            if not any(token in str(path) for token in ("qwen_adv", "lexicons", "adversarial_test")):
                continue
            audit_rows.append([
                item.get("kind", "—"), path,
                item.get("rows", item.get("clean_rows", "—")),
                item.get("success_rows", "—"),
                item.get("success_unique_orig", "—"),
                item.get("schema_ok", False),
            ])
        audit_rows.append([
            "ready", "paper_v2_data_ready / training_ready", "—", "—", "—",
            f"{audit.get('paper_v2_data_ready')} / {audit.get('training_ready')}",
        ])
    add_table(
        lines,
        ["类型", "文件/数据集", "行数", "成功行", "唯一原样本", "schema/ready"],
        audit_rows,
    )
    lines.extend([f"来源：`{audit_path}`", ""])

    lines.extend(["### §4.2.1 / 表4-1：不同 LLM 攻击源下的 SocialDebias Accuracy", ""])
    source_acc = common.summarize_social_adv_detail(
        social_rows, ("dataset", "source"), "acc", weighted=True
    )
    add_table(
        lines,
        ["数据集", "攻击源", "随机种子数", "Accuracy（均值±std）"],
        [[dataset, source, item.get("n_seeds", 0),
          common.fmt_ms(item.get("mean"), item.get("std"))]
         for dataset in ("politifact", "gossipcop", "weibo21")
         for source in ("qwen", "deepseek")
         for item in [source_acc.get((dataset, source), {})]],
    )

    add_csv(
        lines, "§4.4.1 / 表4-3：NLI 双软标签机制本轮结果",
        results_root / "nli_mechanism_summary.csv",
        ["dataset", "suffix", "n", "test_f1_mean", "test_f1_std", "test_bias_acc_mean"],
    )

    datasets, quality = filter_summaries(filter_rows)
    lines.extend(["### §4.5.1 / 表4-4：SocialDebias-Adv 样本规模与保留率", ""])
    dataset_rows = []
    total_in = total_out = 0
    for dataset in ("politifact", "gossipcop", "weibo21"):
        n_in, n_out = datasets.get(dataset, (0, 0))
        total_in += n_in
        total_out += n_out
        dataset_rows.append([dataset, n_in, n_out, common.fmt_pct(n_out / n_in if n_in else None)])
    dataset_rows.append(["合计", total_in, total_out, common.fmt_pct(total_out / total_in if total_in else None)])
    add_table(lines, ["数据集", "生成量", "过滤后保留量", "保留率"], dataset_rows)
    lines.extend([f"来源：`{filter_path}`", ""])

    lines.extend(["### §4.5.2 / 表4-5：不同来源与风格的质量保留率", ""])
    quality_rows = []
    for dimension, categories in (
        ("LLM来源", ("deepseek", "qwen")),
        ("改写风格", ("neutral", "objective", "sensational", "emotionally_triggering")),
    ):
        for category in categories:
            n_in, n_out = quality.get((dimension, category), (0, 0))
            quality_rows.append([dimension, category, n_in, n_out, common.fmt_pct(n_out / n_in if n_in else None)])
    add_table(lines, ["分析维度", "类别", "生成量", "保留量", "保留率"], quality_rows)

    add_csv(
        lines, "§4.5：24个对抗文件逐文件过滤明细",
        filter_path,
        ["file", "lang", "n_in", "n_out", "keep_rate", "fail_entity", "fail_semantic", "fail_nli", "fail_error"],
    )

    lines.extend(["## 四、第5章：模型鲁棒性与解释一致性", ""])
    main_rows = add_current_main_tables(lines, results_root, histories)

    lines.extend(["### §5.3.2 / 表5-6：四种改写风格的 SocialDebias ASR", ""])
    tone_asr = common.summarize_social_adv_detail(
        social_rows, ("dataset", "tone"), "asr", weighted=False
    )
    tone_rows = []
    for dataset in ("politifact", "gossipcop", "weibo21"):
        values = []
        for tone in ("neutral", "objective", "sensational", "emotionally_triggering"):
            item = tone_asr.get((dataset, tone), {})
            mean = item.get("mean")
            if mean is not None:
                values.append((tone, mean))
            tone_rows.append([dataset, tone, item.get("n_seeds", 0), common.fmt_pct(mean), common.fmt_pct(item.get("std"))])
        if values:
            tone_rows.append([dataset, "最高ASR风格", "—", max(values, key=lambda x: x[1])[0], "—"])
    add_table(lines, ["数据集", "改写风格", "随机种子数", "ASR均值", "ASR std"], tone_rows)

    lines.extend(["### §5.3.3 / 表5-7：SocialDebias-Adv 三方评估", ""])
    summary_rows = [
        row for row in social_rows
        if row.get("file") == "__dataset_summary__" and row.get("seed") in ("mean", "single")
    ]
    add_table(
        lines,
        ["数据集", "方法", "seed口径", "Accuracy", "Acc std", "Macro-F1", "F1 std", "ASR", "ASR std"],
        [[row.get("dataset"), row.get("method"), row.get("seed"),
          value_text(row.get("acc")), value_text(row.get("acc_std")),
          value_text(row.get("f1")), value_text(row.get("f1_std")),
          value_text(row.get("asr")), value_text(row.get("asr_std"))]
         for row in sorted(summary_rows, key=lambda r: (r.get("dataset", ""), r.get("method", "")))],
    )

    add_csv(
        lines, "§5.4.3 / 表5-8：IG 解释一致性三随机种子汇总",
        results_root / "explanation/explanation_3seed_summary.csv",
        ["model", "n_seeds", "top_k_overlap_mean", "top_k_overlap_std", "spearman_mean", "spearman_std", "js_divergence_mean", "js_divergence_std"],
    )

    lines.extend(["### §5.4.3：IG 每个随机种子的90对样本汇总", ""])
    explanation_seed_rows = []
    for path in sorted((results_root / "explanation").glob("politifact_surface_all_seed*.json")):
        try:
            payload = common.read_json(path)
            seed_match = re.search(r"seed(\d+)", path.name)
            for model, metrics in payload.get("summary", {}).items():
                explanation_seed_rows.append([
                    seed_match.group(1) if seed_match else "—", model,
                    common.fmt(metrics.get("top_k_overlap", {}).get("mean")),
                    common.fmt(metrics.get("spearman", {}).get("mean")),
                    common.fmt(metrics.get("js_divergence", {}).get("mean")),
                    len(payload.get("rows", [])),
                ])
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    add_table(lines, ["seed", "模型", "Top-K", "Spearman", "JS", "样本对数"], explanation_seed_rows)

    lines.extend(["### §5.4.4 / 表5-9：50对人工评价", ""])
    human_path = results_root / "human_eval/final/human_eval_metrics.json"
    human_rows = []
    if human_path.exists():
        human = common.read_json(human_path).get("human_metrics", {})
        for label, key, pct in (
            ("样本对数", "n_samples", False),
            ("Clean Accuracy", "human_acc_clean", True),
            ("Adv Accuracy", "human_acc_adv", True),
            ("Accuracy Drop", "human_acc_drop", True),
            ("Human ASR", "human_asr", True),
            ("关键词Jaccard均值", "jaccard_orig_adv_mean", False),
            ("关键词Jaccard std", "jaccard_orig_adv_std", False),
        ):
            value = human.get(key)
            human_rows.append([label, common.fmt_pct(value) if pct else value_text(value)])
    add_table(lines, ["指标", "本轮结果"], human_rows)
    lines.extend([f"来源：`{human_path}`", ""])

    add_csv(
        lines, "§5.5.1 / 表5-10：组件消融 Clean",
        results_root / "ablation_clean_summary.csv",
        ["dataset", "suffix", "n", "test_f1_mean", "test_f1_std", "test_bias_acc_mean"],
    )
    add_csv(
        lines, "§5.5.1 / 表5-10：组件消融对抗结果",
        results_root / "ablation_adv/ablation_adv_summary.csv",
        ["dataset", "variant", "n", "clean_f1_mean", "avg_adv_f1_mean", "f1_drop_mean", "avg_asr_mean"],
    )
    add_csv(
        lines, "§3.3.1、§5.5.1 / 表5-11：8维与17维表层特征",
        results_root / "surface_8_vs_17_clean_summary.csv",
        ["dataset", "suffix", "n", "test_f1_mean", "test_f1_std"],
    )
    add_csv(
        lines, "§5.5.2 / 表5-12：NLI机制完整结果",
        results_root / "nli_mechanism_summary.csv",
        ["dataset", "suffix", "n", "test_f1_mean", "test_f1_std", "test_bias_acc_mean"],
    )
    add_csv(
        lines, "§5.5.3 / 表5-13：固定λ与自适应λ Clean结果",
        results_root / "adaptive_clean_summary.csv",
        ["dataset", "suffix", "n", "test_f1_mean", "test_f1_std", "test_bias_acc_mean"],
    )
    add_csv(
        lines, "§5.5.3 / 表5-13：固定λ对抗结果",
        results_root / "surface_fixed_adv_summary.csv",
        ["dataset", "model", "split", "f1_mean", "f1_std", "asr_mean", "asr_std", "n"],
    )
    add_csv(
        lines, "§5.5.3 / 表5-13：自适应λ对抗结果",
        results_root / "surface_adaptive_adv_summary.csv",
        ["dataset", "model", "split", "f1_mean", "f1_std", "asr_mean", "asr_std", "n"],
    )

    lines.extend(["## 五、其它本轮基线与逐变体数据", ""])
    add_csv(
        lines, "P2：BiLSTM 三随机种子汇总",
        results_root / "lstm_summary.csv",
        ["dataset", "seed", "test_acc", "test_f1", "test_auc"],
    )
    add_csv(
        lines, "P2：DeepSeek Clean及adv_A/B/C/D逐变体结果",
        results_root / "llm_baseline/summary.csv",
        ["dataset", "variant", "n", "accuracy", "f1_macro", "asr", "parse_fail_rate"],
    )
    add_csv(
        lines, "P3：surface_all三数据集Clean汇总",
        results_root / "surface_all_clean_summary.csv",
        ["dataset", "suffix", "n", "test_acc_mean", "test_acc_std", "test_f1_mean", "test_f1_std", "test_auc_mean", "test_auc_std", "test_bias_acc_mean"],
    )
    add_csv(
        lines, "P4：BERT与SocialDebias逐对抗变体汇总",
        results_root / "main_summary.csv",
        ["dataset", "model", "split", "f1_mean", "f1_std", "asr_mean", "asr_std", "n"],
    )
    add_csv(
        lines, "P5：24组tone/source/method评估明细（跨seed行）",
        social_path,
        ["file", "dataset", "tone", "source", "method", "seed", "n", "acc", "acc_std", "f1", "f1_std", "asr", "asr_std"],
        predicate=lambda row: row.get("seed") in ("mean", "single"),
    )

    lines.extend([
        "## 六、使用说明",
        "",
        "1. 本文档所有数值均来自本轮结果目录；没有引用旧论文数值。",
        "2. 神经网络主结果以三个随机种子的均值±总体标准差为准。",
        "3. DeepSeek 为单次零样本 API 评估，应明确标注 `single`。",
        "4. P5 的原始生成文件未随公开包保留时，以24个过滤后发布集、filter_report、Git commit和SHA-256作为数据来源证明。",
        "5. 修改论文时逐表人工对照；不要把本轮和旧论文随机种子合并计算。",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--results_root", type=Path,
        default=PROJECT_ROOT / "results/paper_v2_20260719",
    )
    parser.add_argument("--endef_root", type=Path, default=Path("/autodl-fs/data/ENDEF-SIGIR2022"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    results_root = args.results_root.resolve()
    endef_root = args.endef_root.resolve() if args.endef_root.exists() else None
    output = args.output or results_root / "本轮全部实验结果_按论文章节表格汇总.md"
    document = build_document(project_root, results_root, endef_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document + "\n", encoding="utf-8")
    print(f"Current experiment data book: {output}")


if __name__ == "__main__":
    main()
