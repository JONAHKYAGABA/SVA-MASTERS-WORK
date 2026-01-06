"""
Statistical Significance Testing Module

Implements tests from methodology Section 15:
- Paired t-tests for continuous metrics (IoU, BERTScore, BLEU)
- McNemar's test for binary metrics (Exact Match, Pointing Accuracy)
- Bootstrap confidence intervals (10,000 resamples)
- Cohen's d effect size calculation
- Bonferroni correction for multiple comparisons

All pairwise model comparisons undergo rigorous statistical evaluation
to establish whether observed performance differences reflect genuine
model capabilities rather than random variation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

# Statistical imports
try:
    from scipy import stats
    from scipy.stats import ttest_rel, ttest_ind, chi2_contingency
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available - some statistical tests will be disabled")


@dataclass
class StatisticalTestResult:
    """Container for statistical test results."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    is_significant: bool
    interpretation: str
    
    def __str__(self):
        sig_marker = "***" if self.p_value < 0.001 else "**" if self.p_value < 0.01 else "*" if self.p_value < 0.05 else ""
        return (f"{self.test_name}: stat={self.statistic:.4f}, p={self.p_value:.4f}{sig_marker}, "
                f"d={self.effect_size:.4f}, CI=[{self.confidence_interval[0]:.4f}, {self.confidence_interval[1]:.4f}]")


@dataclass  
class ComparisonResults:
    """Container for comprehensive comparison between two models."""
    model_a_name: str
    model_b_name: str
    metric_name: str
    model_a_scores: np.ndarray
    model_b_scores: np.ndarray
    mean_difference: float
    test_result: StatisticalTestResult
    bonferroni_significant: bool = False


# =============================================================================
# Core Statistical Tests
# =============================================================================

def paired_t_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alpha: float = 0.05
) -> StatisticalTestResult:
    """
    Paired t-test for continuous metrics.
    
    From methodology Section 15.1:
    t = d̄ / (sd / √n)
    
    Applied to: IoU, BERTScore, BLEU, ROUGE-L, F1 scores
    
    Args:
        scores_a: Scores from model A (per-sample)
        scores_b: Scores from model B (per-sample)
        alpha: Significance threshold
    
    Returns:
        StatisticalTestResult with test statistics
    """
    if not SCIPY_AVAILABLE:
        return StatisticalTestResult(
            test_name="Paired t-test",
            statistic=0.0, p_value=1.0, effect_size=0.0,
            confidence_interval=(0.0, 0.0),
            is_significant=False,
            interpretation="scipy not available"
        )
    
    # Ensure arrays
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)
    
    # Compute differences
    differences = scores_a - scores_b
    mean_diff = np.mean(differences)
    
    # Perform paired t-test
    t_stat, p_value = ttest_rel(scores_a, scores_b)
    
    # Compute Cohen's d effect size
    # d = d̄ / sd
    effect_size = cohens_d_paired(scores_a, scores_b)
    
    # Bootstrap confidence interval for mean difference
    ci_low, ci_high = bootstrap_confidence_interval(differences)
    
    # Determine significance
    is_significant = p_value < alpha
    
    # Interpretation
    interpretation = interpret_effect_size(effect_size)
    if is_significant:
        direction = "higher" if mean_diff > 0 else "lower"
        interpretation += f" Model A is significantly {direction} (p={p_value:.4f})"
    else:
        interpretation += f" No significant difference (p={p_value:.4f})"
    
    return StatisticalTestResult(
        test_name="Paired t-test",
        statistic=t_stat,
        p_value=p_value,
        effect_size=effect_size,
        confidence_interval=(ci_low, ci_high),
        is_significant=is_significant,
        interpretation=interpretation
    )


def mcnemar_test(
    correct_a: np.ndarray,
    correct_b: np.ndarray,
    alpha: float = 0.05
) -> StatisticalTestResult:
    """
    McNemar's test for binary metrics.
    
    From methodology Section 15.1:
    χ² = (n₁₀ - n₀₁)² / (n₁₀ + n₀₁)
    
    Where:
    - n₁₀ = cases where Model A correct, Model B incorrect
    - n₀₁ = cases where Model B correct, Model A incorrect
    
    Applied to: Exact Match, Pointing Accuracy, Binary classification
    
    Args:
        correct_a: Binary correctness array for model A (1=correct, 0=incorrect)
        correct_b: Binary correctness array for model B
        alpha: Significance threshold
    
    Returns:
        StatisticalTestResult with test statistics
    """
    if not SCIPY_AVAILABLE:
        return StatisticalTestResult(
            test_name="McNemar's test",
            statistic=0.0, p_value=1.0, effect_size=0.0,
            confidence_interval=(0.0, 0.0),
            is_significant=False,
            interpretation="scipy not available"
        )
    
    # Ensure binary arrays
    correct_a = np.asarray(correct_a).astype(int)
    correct_b = np.asarray(correct_b).astype(int)
    
    # Build contingency table
    # n₁₁: both correct, n₁₀: A correct B wrong, n₀₁: A wrong B correct, n₀₀: both wrong
    n11 = np.sum((correct_a == 1) & (correct_b == 1))
    n10 = np.sum((correct_a == 1) & (correct_b == 0))
    n01 = np.sum((correct_a == 0) & (correct_b == 1))
    n00 = np.sum((correct_a == 0) & (correct_b == 0))
    
    # McNemar's test statistic
    # χ² = (n₁₀ - n₀₁)² / (n₁₀ + n₀₁)
    if n10 + n01 == 0:
        chi2_stat = 0.0
        p_value = 1.0
    else:
        chi2_stat = (n10 - n01) ** 2 / (n10 + n01)
        p_value = 1 - stats.chi2.cdf(chi2_stat, df=1)
    
    # Effect size (odds ratio)
    odds_ratio = (n10 + 0.5) / (n01 + 0.5)
    effect_size = np.log(odds_ratio)  # Log odds ratio
    
    # Confidence interval for proportion difference
    prop_diff = (n10 - n01) / len(correct_a)
    se = np.sqrt((n10 + n01) / len(correct_a) ** 2)
    ci_low = prop_diff - 1.96 * se
    ci_high = prop_diff + 1.96 * se
    
    is_significant = p_value < alpha
    
    # Interpretation
    if is_significant:
        better = "A" if n10 > n01 else "B"
        interpretation = f"Model {better} significantly better (p={p_value:.4f})"
    else:
        interpretation = f"No significant difference (p={p_value:.4f})"
    
    return StatisticalTestResult(
        test_name="McNemar's test",
        statistic=chi2_stat,
        p_value=p_value,
        effect_size=effect_size,
        confidence_interval=(ci_low, ci_high),
        is_significant=is_significant,
        interpretation=interpretation
    )


# =============================================================================
# Effect Size Calculations
# =============================================================================

def cohens_d_paired(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """
    Cohen's d for paired samples.
    
    From methodology Section 15.2:
    d = (x̄₁ - x̄₂) / √((s₁² + s₂²) / 2)
    
    Interpretation:
    - |d| < 0.2: Negligible effect
    - 0.2 ≤ |d| < 0.5: Small effect
    - 0.5 ≤ |d| < 0.8: Medium effect
    - |d| ≥ 0.8: Large effect
    """
    mean_diff = np.mean(scores_a) - np.mean(scores_b)
    pooled_std = np.sqrt((np.var(scores_a, ddof=1) + np.var(scores_b, ddof=1)) / 2)
    
    if pooled_std == 0:
        return 0.0
    
    return mean_diff / pooled_std


def cohens_d_independent(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """Cohen's d for independent samples."""
    n1, n2 = len(scores_a), len(scores_b)
    var1, var2 = np.var(scores_a, ddof=1), np.var(scores_b, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return (np.mean(scores_a) - np.mean(scores_b)) / pooled_std


def interpret_effect_size(d: float) -> str:
    """
    Interpret Cohen's d effect size.
    
    From methodology Section 15.2:
    - |d| < 0.2: Negligible - Not practically meaningful
    - 0.2-0.5: Small - Detectable but minor improvement
    - 0.5-0.8: Medium - Meaningful improvement worth pursuing
    - > 0.8: Large - Substantial improvement, highly impactful
    """
    abs_d = abs(d)
    
    if abs_d < 0.2:
        return "Negligible effect (|d|<0.2)"
    elif abs_d < 0.5:
        return "Small effect (0.2≤|d|<0.5)"
    elif abs_d < 0.8:
        return "Medium effect (0.5≤|d|<0.8)"
    else:
        return "Large effect (|d|≥0.8)"


# =============================================================================
# Bootstrap Confidence Intervals
# =============================================================================

def bootstrap_confidence_interval(
    data: np.ndarray,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    statistic: str = 'mean'
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval.
    
    From methodology Section 15.1:
    CI₉₅% = [Percentile₂.₅(θ*), Percentile₉₇.₅(θ*)]
    
    where θ* represents metric computed on bootstrap sample
    
    Args:
        data: Sample data
        n_bootstrap: Number of bootstrap resamples (default: 10,000)
        confidence_level: Confidence level (default: 0.95)
        statistic: 'mean', 'median', or callable
    
    Returns:
        (lower, upper) confidence interval bounds
    """
    data = np.asarray(data)
    n = len(data)
    
    # Statistic function
    if statistic == 'mean':
        stat_func = np.mean
    elif statistic == 'median':
        stat_func = np.median
    elif callable(statistic):
        stat_func = statistic
    else:
        stat_func = np.mean
    
    # Generate bootstrap samples
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        resample_idx = np.random.randint(0, n, size=n)
        resample = data[resample_idx]
        bootstrap_stats.append(stat_func(resample))
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    # Compute percentiles
    alpha = (1 - confidence_level) / 2
    lower = np.percentile(bootstrap_stats, alpha * 100)
    upper = np.percentile(bootstrap_stats, (1 - alpha) * 100)
    
    return (lower, upper)


def bootstrap_difference_ci(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95
) -> Tuple[float, float]:
    """
    Bootstrap confidence interval for difference between two groups.
    
    Args:
        scores_a: Scores from model A
        scores_b: Scores from model B
        n_bootstrap: Number of bootstrap resamples
        confidence_level: Confidence level
    
    Returns:
        (lower, upper) CI for mean difference (A - B)
    """
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)
    n = len(scores_a)
    
    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        idx = np.random.randint(0, n, size=n)
        diff = np.mean(scores_a[idx]) - np.mean(scores_b[idx])
        bootstrap_diffs.append(diff)
    
    bootstrap_diffs = np.array(bootstrap_diffs)
    
    alpha = (1 - confidence_level) / 2
    lower = np.percentile(bootstrap_diffs, alpha * 100)
    upper = np.percentile(bootstrap_diffs, (1 - alpha) * 100)
    
    return (lower, upper)


# =============================================================================
# Multiple Comparisons Correction
# =============================================================================

def bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05
) -> Tuple[float, List[bool]]:
    """
    Bonferroni correction for multiple comparisons.
    
    From methodology Section 15.1:
    α_corrected = α / k
    
    where k = number of simultaneous comparisons
    
    This conservative correction ensures probability of any false positive
    across all tests remains below α.
    
    Args:
        p_values: List of p-values from multiple tests
        alpha: Original significance threshold
    
    Returns:
        (corrected_alpha, list of bools indicating significance)
    """
    k = len(p_values)
    alpha_corrected = alpha / k
    
    significant = [p < alpha_corrected for p in p_values]
    
    return alpha_corrected, significant


def holm_bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05
) -> List[bool]:
    """
    Holm-Bonferroni step-down correction (less conservative).
    
    Args:
        p_values: List of p-values
        alpha: Significance threshold
    
    Returns:
        List of bools indicating significance after correction
    """
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_idx]
    
    significant = [False] * n
    
    for i, (idx, p) in enumerate(zip(sorted_idx, sorted_p)):
        threshold = alpha / (n - i)
        if p <= threshold:
            significant[idx] = True
        else:
            break  # Stop at first non-significant
    
    return significant


# =============================================================================
# Comprehensive Comparison Functions
# =============================================================================

def compare_models_on_metric(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    model_a_name: str,
    model_b_name: str,
    metric_name: str,
    is_binary: bool = False,
    alpha: float = 0.05
) -> ComparisonResults:
    """
    Comprehensive comparison of two models on a single metric.
    
    Args:
        scores_a: Per-sample scores for model A
        scores_b: Per-sample scores for model B
        model_a_name: Name of model A
        model_b_name: Name of model B
        metric_name: Name of the metric
        is_binary: Whether metric is binary (use McNemar's) or continuous (use t-test)
        alpha: Significance threshold
    
    Returns:
        ComparisonResults with full statistics
    """
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)
    
    mean_diff = np.mean(scores_a) - np.mean(scores_b)
    
    if is_binary:
        test_result = mcnemar_test(scores_a, scores_b, alpha)
    else:
        test_result = paired_t_test(scores_a, scores_b, alpha)
    
    return ComparisonResults(
        model_a_name=model_a_name,
        model_b_name=model_b_name,
        metric_name=metric_name,
        model_a_scores=scores_a,
        model_b_scores=scores_b,
        mean_difference=mean_diff,
        test_result=test_result
    )


def multi_metric_comparison(
    results_a: Dict[str, np.ndarray],
    results_b: Dict[str, np.ndarray],
    model_a_name: str,
    model_b_name: str,
    binary_metrics: Optional[List[str]] = None,
    alpha: float = 0.05
) -> Dict[str, ComparisonResults]:
    """
    Compare two models across multiple metrics with Bonferroni correction.
    
    Args:
        results_a: Dict of metric_name -> per-sample scores for model A
        results_b: Dict of metric_name -> per-sample scores for model B
        model_a_name: Name of model A
        model_b_name: Name of model B
        binary_metrics: List of metric names that are binary
        alpha: Significance threshold
    
    Returns:
        Dict of metric_name -> ComparisonResults
    """
    binary_metrics = binary_metrics or ['exact_match', 'pointing_accuracy']
    
    comparisons = {}
    all_p_values = []
    
    # Run all comparisons
    for metric_name in results_a.keys():
        if metric_name not in results_b:
            continue
        
        is_binary = metric_name in binary_metrics
        
        comparison = compare_models_on_metric(
            results_a[metric_name],
            results_b[metric_name],
            model_a_name,
            model_b_name,
            metric_name,
            is_binary,
            alpha
        )
        comparisons[metric_name] = comparison
        all_p_values.append(comparison.test_result.p_value)
    
    # Apply Bonferroni correction
    alpha_corrected, bonferroni_sig = bonferroni_correction(all_p_values, alpha)
    
    # Update results with Bonferroni significance
    for i, (metric_name, comparison) in enumerate(comparisons.items()):
        comparison.bonferroni_significant = bonferroni_sig[i]
    
    return comparisons


# =============================================================================
# Reporting Functions
# =============================================================================

def generate_comparison_report(
    comparisons: Dict[str, ComparisonResults],
    alpha: float = 0.05
) -> str:
    """
    Generate formatted statistical comparison report.
    
    Args:
        comparisons: Dict from multi_metric_comparison
        alpha: Original significance threshold
    
    Returns:
        Formatted report string
    """
    if not comparisons:
        return "No comparisons available"
    
    first = next(iter(comparisons.values()))
    model_a = first.model_a_name
    model_b = first.model_b_name
    
    lines = [
        "=" * 80,
        f"STATISTICAL COMPARISON: {model_a} vs {model_b}",
        "=" * 80,
        "",
        f"Number of comparisons: {len(comparisons)}",
        f"Bonferroni-corrected α: {alpha / len(comparisons):.4f}",
        "",
        "-" * 80,
        f"{'Metric':<25} {'Δ Mean':>10} {'t/χ²':>10} {'p-value':>12} {'Effect':>10} {'Sig':>5}",
        "-" * 80,
    ]
    
    for metric_name, comp in comparisons.items():
        result = comp.test_result
        sig_marker = "***" if comp.bonferroni_significant else ("*" if result.is_significant else "")
        
        lines.append(
            f"{metric_name:<25} {comp.mean_difference:>+10.4f} "
            f"{result.statistic:>10.3f} {result.p_value:>12.4f} "
            f"{result.effect_size:>10.3f} {sig_marker:>5}"
        )
    
    lines.extend([
        "-" * 80,
        "",
        "Legend: *** significant after Bonferroni, * significant at α=0.05",
        "",
        "EFFECT SIZE INTERPRETATION:",
        "  |d| < 0.2: Negligible",
        "  0.2 ≤ |d| < 0.5: Small",
        "  0.5 ≤ |d| < 0.8: Medium",
        "  |d| ≥ 0.8: Large",
        "=" * 80,
    ])
    
    return "\n".join(lines)


def generate_latex_table(
    comparisons: Dict[str, ComparisonResults],
    caption: str = "Model Comparison Results"
) -> str:
    """
    Generate LaTeX table for paper inclusion.
    
    Args:
        comparisons: Dict from multi_metric_comparison
        caption: Table caption
    
    Returns:
        LaTeX table string
    """
    if not comparisons:
        return ""
    
    first = next(iter(comparisons.values()))
    
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Metric & $\\Delta$ Mean & Statistic & $p$-value & Cohen's $d$ & 95\\% CI & Sig. \\\\",
        "\\midrule",
    ]
    
    for metric_name, comp in comparisons.items():
        result = comp.test_result
        ci_low, ci_high = result.confidence_interval
        sig = "$^{***}$" if comp.bonferroni_significant else ("$^{*}$" if result.is_significant else "")
        
        lines.append(
            f"{metric_name} & {comp.mean_difference:+.3f} & {result.statistic:.3f} & "
            f"{result.p_value:.4f} & {result.effect_size:.3f} & "
            f"[{ci_low:.3f}, {ci_high:.3f}] & {sig} \\\\"
        )
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\label{tab:model_comparison}",
        "\\end{table}",
    ])
    
    return "\n".join(lines)

