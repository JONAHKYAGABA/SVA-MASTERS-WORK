"""
MIMIC-CXR VQA Utilities Module

Provides helper functions for:
- Training utilities
- Hardware detection and optimization
- Explainability and attention analysis
- Statistical significance testing
- NLP evaluation metrics (BLEU, ROUGE, BERTScore)
"""

from .utils import (
    AverageMeter,
    seed_everything,
    save_checkpoint,
    load_checkpoint,
    setup_logging,
    get_lr,
)

from .hardware_utils import (
    detect_hardware,
    print_hardware_info,
    get_optimal_config_overrides,
    optimize_for_hardware,
    set_optimal_environment,
    get_deepspeed_config_for_hardware,
    HardwareInfo,
    GPUInfo,
)

from .explainability import (
    AttentionExtractor,
    AttentionAnalysis,
    compute_attention_entropy,
    compute_plausibility,
    analyze_attention,
    create_attention_heatmap,
    save_attention_visualization,
    batch_attention_analysis,
    interpret_attention_metrics,
)

from .statistical_tests import (
    StatisticalTestResult,
    ComparisonResults,
    paired_t_test,
    mcnemar_test,
    cohens_d_paired,
    interpret_effect_size,
    bootstrap_confidence_interval,
    bootstrap_difference_ci,
    bonferroni_correction,
    holm_bonferroni_correction,
    compare_models_on_metric,
    multi_metric_comparison,
    generate_comparison_report,
    generate_latex_table,
)

from .nlp_metrics import (
    compute_bleu4,
    compute_bleu4_batch,
    compute_rouge_l,
    compute_rouge_l_batch,
    compute_bert_score,
    compute_clinical_term_f1,
    compute_clinical_term_f1_batch,
    compute_semantic_type_accuracy,
    compute_token_f1,
    compute_exact_match,
    compute_all_nlp_metrics,
    format_nlp_metrics_report,
    extract_medical_terms,
    get_answer_type,
    MEDICAL_TERMS,
)

__all__ = [
    # General utils
    'AverageMeter',
    'seed_everything',
    'save_checkpoint',
    'load_checkpoint',
    'setup_logging',
    'get_lr',
    
    # Hardware utils
    'detect_hardware',
    'print_hardware_info',
    'get_optimal_config_overrides',
    'optimize_for_hardware',
    'set_optimal_environment',
    'get_deepspeed_config_for_hardware',
    'HardwareInfo',
    'GPUInfo',
    
    # Explainability (methodology Section 14)
    'AttentionExtractor',
    'AttentionAnalysis',
    'compute_attention_entropy',
    'compute_plausibility',
    'analyze_attention',
    'create_attention_heatmap',
    'save_attention_visualization',
    'batch_attention_analysis',
    'interpret_attention_metrics',
    
    # Statistical Tests (methodology Section 15)
    'StatisticalTestResult',
    'ComparisonResults',
    'paired_t_test',
    'mcnemar_test',
    'cohens_d_paired',
    'interpret_effect_size',
    'bootstrap_confidence_interval',
    'bootstrap_difference_ci',
    'bonferroni_correction',
    'holm_bonferroni_correction',
    'compare_models_on_metric',
    'multi_metric_comparison',
    'generate_comparison_report',
    'generate_latex_table',
    
    # NLP Metrics (methodology Section 13.1)
    'compute_bleu4',
    'compute_bleu4_batch',
    'compute_rouge_l',
    'compute_rouge_l_batch',
    'compute_bert_score',
    'compute_clinical_term_f1',
    'compute_clinical_term_f1_batch',
    'compute_semantic_type_accuracy',
    'compute_token_f1',
    'compute_exact_match',
    'compute_all_nlp_metrics',
    'format_nlp_metrics_report',
    'extract_medical_terms',
    'get_answer_type',
    'MEDICAL_TERMS',
]

