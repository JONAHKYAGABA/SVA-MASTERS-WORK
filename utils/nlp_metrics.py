"""
NLP Evaluation Metrics for Medical VQA

Implements metrics from methodology Section 13.1:
- BLEU-4: N-gram precision for fluency
- ROUGE-L: Longest common subsequence
- BERTScore: Semantic similarity via contextual embeddings
- Clinical Term F1: Medical terminology accuracy
- Semantic Answer Type Accuracy

These metrics assess correctness, fluency, and clinical appropriateness
of generated answers across question types.
"""

import numpy as np
import re
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter
import logging

logger = logging.getLogger(__name__)

# Optional imports with fallbacks
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    import nltk
    # Ensure punkt is available
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("NLTK not available - BLEU calculation will use simple tokenization")

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    logger.warning("rouge_score not available - ROUGE-L will use fallback implementation")

try:
    from bert_score import score as bert_score_fn
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    logger.warning("bert_score not available - BERTScore will be skipped")


# =============================================================================
# Medical Terminology
# =============================================================================

# Common medical terms for Clinical Term F1
MEDICAL_TERMS = {
    # CheXpert categories
    'atelectasis', 'cardiomegaly', 'consolidation', 'edema',
    'enlarged', 'cardiomediastinum', 'fracture', 'lesion',
    'opacity', 'effusion', 'pneumonia', 'pneumothorax',
    'pleural', 'devices', 'finding',
    
    # Anatomical regions
    'lung', 'lungs', 'heart', 'aorta', 'trachea', 'mediastinum',
    'diaphragm', 'spine', 'rib', 'ribs', 'clavicle', 'hilum',
    'cardiac', 'pulmonary', 'thoracic', 'chest', 'pleura',
    
    # Medical descriptors
    'normal', 'abnormal', 'bilateral', 'unilateral',
    'left', 'right', 'upper', 'lower', 'middle',
    'mild', 'moderate', 'severe', 'acute', 'chronic',
    'enlarged', 'decreased', 'increased', 'prominent',
    
    # Findings
    'infiltrate', 'mass', 'nodule', 'calcification',
    'thickening', 'widening', 'congestion', 'hemorrhage',
    'fibrosis', 'emphysema', 'hyperinflation',
    
    # Devices
    'tube', 'catheter', 'pacemaker', 'wire', 'line',
    'stent', 'prosthesis', 'implant',
    
    # Measurements/positions
    'anterior', 'posterior', 'lateral', 'medial',
    'superior', 'inferior', 'proximal', 'distal',
}

# Answer type categories for Semantic Answer Type Accuracy
ANSWER_TYPE_PATTERNS = {
    'anatomical': r'\b(lung|heart|aorta|trachea|mediastinum|diaphragm|spine|rib|clavicle|hilum|chest|thorax|pleura)\b',
    'disease': r'\b(atelectasis|cardiomegaly|consolidation|edema|effusion|pneumonia|pneumothorax|fracture|lesion|nodule|mass|fibrosis|emphysema)\b',
    'measurement': r'\b(\d+\.?\d*\s*(mm|cm|ml|%|degrees?))\b',
    'device': r'\b(tube|catheter|pacemaker|wire|line|stent|prosthesis|implant)\b',
    'binary': r'\b(yes|no|present|absent|normal|abnormal)\b',
    'severity': r'\b(none|mild|moderate|severe|minimal|significant)\b',
    'position': r'\b(left|right|bilateral|upper|lower|middle|anterior|posterior|lateral|medial)\b',
}


# =============================================================================
# Tokenization
# =============================================================================

def tokenize(text: str, use_nltk: bool = True) -> List[str]:
    """
    Tokenize text for metric computation.
    
    Args:
        text: Input text string
        use_nltk: Whether to use NLTK tokenizer
    
    Returns:
        List of tokens
    """
    text = text.lower().strip()
    
    if use_nltk and NLTK_AVAILABLE:
        try:
            return word_tokenize(text)
        except:
            pass
    
    # Fallback: simple whitespace + punctuation tokenization
    tokens = re.findall(r'\b\w+\b', text)
    return tokens


def extract_medical_terms(text: str) -> List[str]:
    """
    Extract medical terminology from text.
    
    Args:
        text: Input text
    
    Returns:
        List of medical terms found
    """
    tokens = tokenize(text)
    return [t for t in tokens if t in MEDICAL_TERMS]


def get_answer_type(text: str) -> str:
    """
    Determine semantic type of an answer.
    
    Args:
        text: Answer text
    
    Returns:
        Answer type category
    """
    text_lower = text.lower()
    
    # Check each pattern
    for answer_type, pattern in ANSWER_TYPE_PATTERNS.items():
        if re.search(pattern, text_lower):
            return answer_type
    
    return 'other'


# =============================================================================
# BLEU Score
# =============================================================================

def compute_bleu4(
    prediction: str,
    reference: str,
    smoothing: bool = True
) -> float:
    """
    Compute BLEU-4 score for a single prediction.
    
    From methodology Section 13.1:
    BLEU-4 = BP × exp(Σₙ wₙ log pₙ)
    
    where pₙ = n-gram precision, wₙ = 1/4, BP = brevity penalty
    
    Args:
        prediction: Predicted answer
        reference: Ground truth answer
        smoothing: Whether to use smoothing (avoids 0 for missing n-grams)
    
    Returns:
        BLEU-4 score in [0, 1]
    """
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0
    
    if NLTK_AVAILABLE:
        smoothing_fn = SmoothingFunction().method1 if smoothing else None
        try:
            score = sentence_bleu(
                [ref_tokens],
                pred_tokens,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=smoothing_fn
            )
            return score
        except:
            pass
    
    # Fallback implementation
    return _compute_bleu_fallback(pred_tokens, ref_tokens)


def _compute_bleu_fallback(pred_tokens: List[str], ref_tokens: List[str]) -> float:
    """Simple BLEU-4 fallback implementation."""
    
    def get_ngrams(tokens: List[str], n: int) -> Counter:
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))
    
    # Compute n-gram precisions
    precisions = []
    for n in range(1, 5):
        pred_ngrams = get_ngrams(pred_tokens, n)
        ref_ngrams = get_ngrams(ref_tokens, n)
        
        if not pred_ngrams:
            precisions.append(0.0)
            continue
        
        matches = sum(min(pred_ngrams[ng], ref_ngrams[ng]) for ng in pred_ngrams)
        total = sum(pred_ngrams.values())
        
        precisions.append(matches / total if total > 0 else 0.0)
    
    # Geometric mean
    if 0 in precisions:
        # Smoothing: add small value to avoid log(0)
        precisions = [p + 1e-10 for p in precisions]
    
    log_precision = sum(np.log(p) for p in precisions) / 4
    
    # Brevity penalty
    bp = min(1.0, np.exp(1 - len(ref_tokens) / max(len(pred_tokens), 1)))
    
    return bp * np.exp(log_precision)


def compute_bleu4_batch(
    predictions: List[str],
    references: List[str]
) -> Dict[str, float]:
    """
    Compute BLEU-4 for a batch of predictions.
    
    Returns:
        Dict with 'bleu4' mean and per-sample scores
    """
    scores = [compute_bleu4(p, r) for p, r in zip(predictions, references)]
    
    return {
        'bleu4': np.mean(scores),
        'bleu4_std': np.std(scores),
        'bleu4_scores': scores
    }


# =============================================================================
# ROUGE-L Score
# =============================================================================

def compute_rouge_l(prediction: str, reference: str) -> float:
    """
    Compute ROUGE-L (Longest Common Subsequence) score.
    
    From methodology Section 13.1:
    ROUGE-L = LCS(X, Y) / length(Y)
    
    where LCS = longest common subsequence
    
    Args:
        prediction: Predicted answer
        reference: Ground truth answer
    
    Returns:
        ROUGE-L F1 score in [0, 1]
    """
    if ROUGE_AVAILABLE:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(reference, prediction)
        return scores['rougeL'].fmeasure
    
    # Fallback implementation
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0
    
    lcs_length = _lcs_length(pred_tokens, ref_tokens)
    
    precision = lcs_length / len(pred_tokens) if len(pred_tokens) > 0 else 0
    recall = lcs_length / len(ref_tokens) if len(ref_tokens) > 0 else 0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def _lcs_length(x: List[str], y: List[str]) -> int:
    """Compute length of longest common subsequence."""
    m, n = len(x), len(y)
    
    # DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i-1] == y[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]


def compute_rouge_l_batch(
    predictions: List[str],
    references: List[str]
) -> Dict[str, float]:
    """Compute ROUGE-L for a batch."""
    scores = [compute_rouge_l(p, r) for p, r in zip(predictions, references)]
    
    return {
        'rouge_l': np.mean(scores),
        'rouge_l_std': np.std(scores),
        'rouge_l_scores': scores
    }


# =============================================================================
# BERTScore
# =============================================================================

def compute_bert_score(
    predictions: List[str],
    references: List[str],
    model_type: str = 'microsoft/deberta-xlarge-mnli',
    lang: str = 'en',
    verbose: bool = False
) -> Dict[str, float]:
    """
    Compute BERTScore for semantic similarity.
    
    From methodology Section 13.1:
    BERTScore_F1 = 2 × (P_BERT × R_BERT) / (P_BERT + R_BERT)
    
    Uses contextual embeddings for robust paraphrase handling.
    
    Args:
        predictions: List of predicted answers
        references: List of reference answers
        model_type: BERT model for embeddings
        lang: Language code
        verbose: Show progress
    
    Returns:
        Dict with precision, recall, F1 scores
    """
    if not BERTSCORE_AVAILABLE:
        logger.warning("BERTScore not available, returning zeros")
        return {
            'bert_score_precision': 0.0,
            'bert_score_recall': 0.0,
            'bert_score_f1': 0.0,
        }
    
    try:
        P, R, F1 = bert_score_fn(
            predictions, references,
            model_type=model_type,
            lang=lang,
            verbose=verbose,
            rescale_with_baseline=True
        )
        
        return {
            'bert_score_precision': P.mean().item(),
            'bert_score_recall': R.mean().item(),
            'bert_score_f1': F1.mean().item(),
            'bert_score_f1_scores': F1.tolist(),
        }
    except Exception as e:
        logger.error(f"BERTScore computation failed: {e}")
        return {
            'bert_score_precision': 0.0,
            'bert_score_recall': 0.0,
            'bert_score_f1': 0.0,
        }


# =============================================================================
# Clinical Term F1
# =============================================================================

def compute_clinical_term_f1(
    prediction: str,
    reference: str
) -> float:
    """
    Compute F1 score specifically over medical terminology.
    
    From methodology Section 13.1:
    Clinical Term F1 calculates F1 over medical terminology
    extracted using MetaMap, measuring clinical accuracy
    independent of general language fluency.
    
    Args:
        prediction: Predicted answer
        reference: Reference answer
    
    Returns:
        F1 score for medical terms
    """
    pred_terms = set(extract_medical_terms(prediction))
    ref_terms = set(extract_medical_terms(reference))
    
    if len(pred_terms) == 0 and len(ref_terms) == 0:
        return 1.0  # Both empty = match
    
    if len(pred_terms) == 0 or len(ref_terms) == 0:
        return 0.0
    
    # Compute precision, recall, F1
    intersection = pred_terms & ref_terms
    
    precision = len(intersection) / len(pred_terms)
    recall = len(intersection) / len(ref_terms)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_clinical_term_f1_batch(
    predictions: List[str],
    references: List[str]
) -> Dict[str, float]:
    """Compute Clinical Term F1 for a batch."""
    scores = [compute_clinical_term_f1(p, r) for p, r in zip(predictions, references)]
    
    return {
        'clinical_term_f1': np.mean(scores),
        'clinical_term_f1_std': np.std(scores),
        'clinical_term_f1_scores': scores
    }


# =============================================================================
# Semantic Answer Type Accuracy
# =============================================================================

def compute_semantic_type_accuracy(
    predictions: List[str],
    references: List[str]
) -> float:
    """
    Compute accuracy of answer semantic type classification.
    
    From methodology Section 13.1:
    Assesses whether predicted answer belongs to correct semantic
    category (anatomical location, disease name, measurement)
    even when exact wording differs.
    
    Args:
        predictions: List of predictions
        references: List of references
    
    Returns:
        Accuracy of semantic type matching
    """
    correct = 0
    total = 0
    
    for pred, ref in zip(predictions, references):
        pred_type = get_answer_type(pred)
        ref_type = get_answer_type(ref)
        
        if pred_type == ref_type:
            correct += 1
        total += 1
    
    return correct / total if total > 0 else 0.0


# =============================================================================
# Token-Level F1 Score
# =============================================================================

def compute_token_f1(prediction: str, reference: str) -> float:
    """
    Compute token-level F1 score.
    
    From methodology Section 13.1:
    F1 = 2 × (Precision × Recall) / (Precision + Recall)
    
    where precision/recall computed at token level.
    
    Args:
        prediction: Predicted answer
        reference: Reference answer
    
    Returns:
        Token F1 score
    """
    pred_tokens = set(tokenize(prediction))
    ref_tokens = set(tokenize(reference))
    
    if len(pred_tokens) == 0 and len(ref_tokens) == 0:
        return 1.0
    
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0
    
    intersection = pred_tokens & ref_tokens
    
    precision = len(intersection) / len(pred_tokens)
    recall = len(intersection) / len(ref_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)


# =============================================================================
# Exact Match
# =============================================================================

def compute_exact_match(prediction: str, reference: str) -> bool:
    """
    Compute exact match between prediction and reference.
    
    From methodology Section 13.1:
    EM measures percentage of answers matching ground truth exactly.
    
    Normalization: lowercase, strip whitespace, remove punctuation
    
    Args:
        prediction: Predicted answer
        reference: Reference answer
    
    Returns:
        True if exact match after normalization
    """
    def normalize(text):
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)
        text = ' '.join(text.split())
        return text
    
    return normalize(prediction) == normalize(reference)


# =============================================================================
# Comprehensive NLP Metrics
# =============================================================================

def compute_all_nlp_metrics(
    predictions: List[str],
    references: List[str],
    compute_bertscore: bool = True,
    bertscore_model: str = 'microsoft/deberta-xlarge-mnli'
) -> Dict[str, Any]:
    """
    Compute all NLP metrics for VQA evaluation.
    
    Args:
        predictions: List of predicted answers
        references: List of reference answers
        compute_bertscore: Whether to compute BERTScore (slow)
        bertscore_model: Model for BERTScore
    
    Returns:
        Dict with all metrics
    """
    results = {}
    
    # Exact Match
    em_scores = [compute_exact_match(p, r) for p, r in zip(predictions, references)]
    results['exact_match'] = np.mean(em_scores)
    results['exact_match_scores'] = em_scores
    
    # Token F1
    f1_scores = [compute_token_f1(p, r) for p, r in zip(predictions, references)]
    results['token_f1'] = np.mean(f1_scores)
    results['token_f1_std'] = np.std(f1_scores)
    results['token_f1_scores'] = f1_scores
    
    # BLEU-4
    bleu_results = compute_bleu4_batch(predictions, references)
    results.update(bleu_results)
    
    # ROUGE-L
    rouge_results = compute_rouge_l_batch(predictions, references)
    results.update(rouge_results)
    
    # Clinical Term F1
    clinical_results = compute_clinical_term_f1_batch(predictions, references)
    results.update(clinical_results)
    
    # Semantic Type Accuracy
    results['semantic_type_accuracy'] = compute_semantic_type_accuracy(predictions, references)
    
    # BERTScore (optional - expensive)
    if compute_bertscore and BERTSCORE_AVAILABLE:
        bert_results = compute_bert_score(predictions, references, model_type=bertscore_model)
        results.update(bert_results)
    
    return results


def format_nlp_metrics_report(metrics: Dict[str, Any]) -> str:
    """
    Format NLP metrics as readable report.
    
    Args:
        metrics: Dict from compute_all_nlp_metrics
    
    Returns:
        Formatted string report
    """
    lines = [
        "=" * 60,
        "NLP EVALUATION METRICS",
        "=" * 60,
        "",
        "ANSWER ACCURACY METRICS:",
        f"  Exact Match:           {metrics.get('exact_match', 0):.2%}",
        f"  Token F1:              {metrics.get('token_f1', 0):.4f} ± {metrics.get('token_f1_std', 0):.4f}",
        f"  BLEU-4:                {metrics.get('bleu4', 0):.4f} ± {metrics.get('bleu4_std', 0):.4f}",
        f"  ROUGE-L:               {metrics.get('rouge_l', 0):.4f} ± {metrics.get('rouge_l_std', 0):.4f}",
        "",
        "CLINICAL RELEVANCE:",
        f"  Clinical Term F1:      {metrics.get('clinical_term_f1', 0):.4f}",
        f"  Semantic Type Acc:     {metrics.get('semantic_type_accuracy', 0):.2%}",
    ]
    
    if 'bert_score_f1' in metrics:
        lines.extend([
            "",
            "SEMANTIC SIMILARITY (BERTScore):",
            f"  Precision:             {metrics.get('bert_score_precision', 0):.4f}",
            f"  Recall:                {metrics.get('bert_score_recall', 0):.4f}",
            f"  F1:                    {metrics.get('bert_score_f1', 0):.4f}",
        ])
    
    lines.append("=" * 60)
    
    return "\n".join(lines)

