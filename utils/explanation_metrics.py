"""
解释一致性量化指标（回应意见 15）

三个互补的指标：
1. Top-K 重合度：两组归因在最重要的 K 个词上的重合比例
2. Spearman 秩相关：两组归因的整体排序相关性
3. Jensen-Shannon 散度：两组归因分布的对称距离

注意：两段文本的分词结果可能不同，需要先做词元对齐。
对于对齐，我们采用"共同出现的词元"子集做比较。
"""
import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import jensenshannon
from typing import List, Dict, Tuple


def align_tokens(
        tokens_a: List[str],
        scores_a: List[float],
        tokens_b: List[str],
        scores_b: List[float],
) -> Tuple[List[str], List[float], List[float]]:
    """
    对齐两组词元：取两者共同出现的词元，按 a 的顺序排列。

    这是最简单的对齐策略，适合处理"改写前后有部分词相同"的场景。
    更复杂的对齐（如词向量匹配）留待后续优化。

    Returns:
        common_tokens: 共同词元列表
        aligned_scores_a: a 在这些词元上的分数
        aligned_scores_b: b 在这些词元上的分数
    """
    # 用 dict 查找 b 中每个词元的分数（若重复出现，取第一次）
    b_dict = {}
    for tok, score in zip(tokens_b, scores_b):
        if tok not in b_dict:
            b_dict[tok] = score

    common_tokens, aligned_a, aligned_b = [], [], []
    seen = set()
    for tok, score_a in zip(tokens_a, scores_a):
        if tok in b_dict and tok not in seen:
            common_tokens.append(tok)
            aligned_a.append(score_a)
            aligned_b.append(b_dict[tok])
            seen.add(tok)

    return common_tokens, aligned_a, aligned_b


def top_k_overlap(
        tokens_a: List[str],
        scores_a: List[float],
        tokens_b: List[str],
        scores_b: List[float],
        k: int = 10,
        use_abs: bool = True,
) -> float:
    """
    指标 1：Top-K 重合度

    取两组归因中最重要的 K 个词元，计算交集占比。

    Args:
        use_abs: True 表示按分数绝对值排序（包含正负两边最重要的词）
                 False 表示只看正向归因

    Returns:
        overlap_ratio ∈ [0, 1]，越高表示两组关注点越一致
    """

    def top_k_tokens(tokens, scores, k):
        key = (lambda s: abs(s)) if use_abs else (lambda s: s)
        indexed = sorted(enumerate(scores), key=lambda x: key(x[1]), reverse=True)
        top_indices = [i for i, _ in indexed[:k]]
        return set(tokens[i] for i in top_indices)

    top_a = top_k_tokens(tokens_a, scores_a, k)
    top_b = top_k_tokens(tokens_b, scores_b, k)

    if len(top_a) == 0 or len(top_b) == 0:
        return 0.0

    intersection = top_a & top_b
    union = top_a | top_b
    return len(intersection) / len(union)  # Jaccard 相似度


def spearman_correlation(
        tokens_a: List[str],
        scores_a: List[float],
        tokens_b: List[str],
        scores_b: List[float],
) -> float:
    """
    指标 2：Spearman 秩相关

    在共同词元上，计算两组归因排序的相关性。

    Returns:
        correlation ∈ [-1, 1]，越接近 1 越一致
    """
    _, aligned_a, aligned_b = align_tokens(tokens_a, scores_a, tokens_b, scores_b)

    if len(aligned_a) < 3:
        # 共同词元太少，无法计算可靠的相关性
        return float("nan")

    corr, _ = spearmanr(aligned_a, aligned_b)
    if np.isnan(corr):
        return 0.0
    return float(corr)


def js_divergence(
        tokens_a: List[str],
        scores_a: List[float],
        tokens_b: List[str],
        scores_b: List[float],
) -> float:
    """
    指标 3：Jensen-Shannon 散度（在归一化归因分布上）

    把归因分数通过 softmax 转成概率分布，再算 JS 散度。
    使用绝对值，避免负数无法归一化。

    Returns:
        divergence ∈ [0, 1]，0 表示完全一致，1 表示完全不同。
        注意：这个越小越好，和上面两个指标相反。
    """
    _, aligned_a, aligned_b = align_tokens(tokens_a, scores_a, tokens_b, scores_b)

    if len(aligned_a) < 3:
        return float("nan")

    # 取绝对值 + 归一化成概率分布
    p = np.abs(np.array(aligned_a))
    q = np.abs(np.array(aligned_b))

    # 避免全零（退化分布）
    if p.sum() == 0 or q.sum() == 0:
        return 1.0

    p = p / p.sum()
    q = q / q.sum()

    # scipy 的 jensenshannon 返回 JS 距离（平方根），我们算 JS 散度（平方）
    # 且它的底是 e，我们改成底 2 以获得 [0, 1] 区间
    js_dist = jensenshannon(p, q, base=2)
    return float(js_dist ** 2)  # JS 距离平方 = JS 散度


def compute_all_metrics(
        tokens_a: List[str],
        scores_a: List[float],
        tokens_b: List[str],
        scores_b: List[float],
        k: int = 10,
) -> Dict[str, float]:
    """
    一次性计算所有指标。

    Returns:
        {
            'top_k_overlap':  Jaccard 相似度 ∈ [0, 1]（越高越好）
            'spearman':       秩相关 ∈ [-1, 1]（越高越好）
            'js_divergence':  JS 散度 ∈ [0, 1]（越低越好）
            'common_tokens_count': 共同词元数量（诊断用）
        }
    """
    common_tokens, _, _ = align_tokens(tokens_a, scores_a, tokens_b, scores_b)

    return {
        "top_k_overlap": top_k_overlap(tokens_a, scores_a, tokens_b, scores_b, k=k),
        "spearman": spearman_correlation(tokens_a, scores_a, tokens_b, scores_b),
        "js_divergence": js_divergence(tokens_a, scores_a, tokens_b, scores_b),
        "common_tokens_count": len(common_tokens),
    }


if __name__ == "__main__":
    # 构造几个测试案例，验证指标的正确性
    print("=" * 60)
    print("指标正确性测试")
    print("=" * 60)

    # 测试 1：完全相同 → 所有指标应达到最优值
    tokens = ["the", "shocking", "news", "about", "economy"]
    scores_same = [0.1, 0.8, 0.2, 0.05, 0.3]

    print("\n测试 1: 归因完全相同")
    m = compute_all_metrics(tokens, scores_same, tokens, scores_same)
    print(f"  Top-K 重合: {m['top_k_overlap']:.3f}  (期望 1.0)")
    print(f"  Spearman:  {m['spearman']:.3f}  (期望 1.0)")
    print(f"  JS 散度:   {m['js_divergence']:.4f}  (期望 ~0)")

    # 测试 2：完全反向
    scores_reversed = [-s for s in scores_same]
    print("\n测试 2: 归因符号完全相反")
    m = compute_all_metrics(tokens, scores_same, tokens, scores_reversed)
    print(f"  Top-K 重合: {m['top_k_overlap']:.3f}  (期望 1.0，因为用了绝对值)")
    print(f"  Spearman:  {m['spearman']:.3f}  (期望 -1.0)")
    print(f"  JS 散度:   {m['js_divergence']:.4f}  (期望 ~0，因为|abs|相同)")

    # 测试 3：完全不相关
    scores_random = [0.5, 0.1, -0.3, 0.9, 0.05]
    print("\n测试 3: 归因分数重排")
    m = compute_all_metrics(tokens, scores_same, tokens, scores_random)
    print(f"  Top-K 重合: {m['top_k_overlap']:.3f}")
    print(f"  Spearman:  {m['spearman']:.3f}")
    print(f"  JS 散度:   {m['js_divergence']:.4f}")

    print("\n✅ 指标实现正确")