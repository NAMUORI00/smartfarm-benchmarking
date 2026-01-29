"""Domain-specific evaluation metrics for agricultural RAG.

Specialized metrics for wasabi/smartfarm domain:
- Ontology Coverage: How well answer covers domain ontology terms
- Term Accuracy: Correctness of technical terms used
- Causal Chain Match: Whether cause-effect relationships are captured
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


# Default ontology path
DEFAULT_ONTOLOGY_PATH = Path(__file__).parent.parent.parent / "data" / "ontology" / "wasabi_ontology.json"


def _load_ontology(path: Optional[Path] = None) -> Dict[str, Dict[str, List[str]]]:
    """Load domain ontology from JSON file."""
    if path is None:
        path = DEFAULT_ONTOLOGY_PATH
    
    if not path.exists():
        return {}
    
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_text(text: str) -> str:
    """Normalize text for matching."""
    return text.lower().strip()


def _extract_ontology_terms(ontology: Dict[str, Dict[str, List[str]]]) -> Dict[str, Set[str]]:
    """Extract all terms from ontology, grouped by category."""
    terms_by_category = {}
    for category, concepts in ontology.items():
        terms = set()
        for concept, aliases in concepts.items():
            terms.add(_normalize_text(concept))
            for alias in aliases:
                terms.add(_normalize_text(alias))
        terms_by_category[category] = terms
    return terms_by_category


def ontology_coverage(
    text: str,
    ontology: Optional[Dict[str, Dict[str, List[str]]]] = None,
    categories: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Measure how well text covers domain ontology terms.
    
    Args:
        text: Text to analyze (answer or source)
        ontology: Domain ontology dict (loaded from file if None)
        categories: Optional list of categories to check (all if None)
    
    Returns:
        Dict with coverage metrics per category and overall
    """
    if ontology is None:
        ontology = _load_ontology()
    
    terms_by_category = _extract_ontology_terms(ontology)
    text_lower = _normalize_text(text)
    
    results = {
        "per_category": {},
        "total_terms_found": 0,
        "total_terms_possible": 0,
    }
    
    for category, terms in terms_by_category.items():
        if categories and category not in categories:
            continue
        
        found = [t for t in terms if t in text_lower]
        results["per_category"][category] = {
            "found": found,
            "count": len(found),
            "total": len(terms),
            "coverage": len(found) / len(terms) if terms else 0.0,
        }
        results["total_terms_found"] += len(found)
        results["total_terms_possible"] += len(terms)
    
    results["overall_coverage"] = (
        results["total_terms_found"] / results["total_terms_possible"]
        if results["total_terms_possible"] > 0 else 0.0
    )
    
    return results


def term_accuracy(
    prediction: str,
    reference: str,
    ontology: Optional[Dict[str, Dict[str, List[str]]]] = None,
) -> Dict[str, float]:
    """Measure accuracy of technical terms in prediction vs reference.
    
    Args:
        prediction: Model's answer
        reference: Ground truth answer
        ontology: Domain ontology for term extraction
    
    Returns:
        Dict with precision, recall, f1 for term usage
    """
    if ontology is None:
        ontology = _load_ontology()
    
    terms_by_category = _extract_ontology_terms(ontology)
    all_terms = set()
    for terms in terms_by_category.values():
        all_terms.update(terms)
    
    pred_lower = _normalize_text(prediction)
    ref_lower = _normalize_text(reference)
    
    # Find terms in each text
    pred_terms = {t for t in all_terms if t in pred_lower}
    ref_terms = {t for t in all_terms if t in ref_lower}
    
    if not pred_terms and not ref_terms:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}  # Both empty = perfect
    
    common = pred_terms & ref_terms
    
    precision = len(common) / len(pred_terms) if pred_terms else 0.0
    recall = len(common) / len(ref_terms) if ref_terms else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pred_terms": list(pred_terms),
        "ref_terms": list(ref_terms),
        "common_terms": list(common),
    }


# Causal patterns for agricultural domain
CAUSAL_PATTERNS = [
    # Korean patterns
    (r"(.+)(?:하면|되면|일 때)\s*(.+)(?:된다|한다|발생한다)", "condition_result"),
    (r"(.+)(?:때문에|으로 인해)\s*(.+)", "cause_effect"),
    (r"(.+)(?:위해서는|위해)\s*(.+)(?:해야|필요)", "goal_action"),
    # English patterns
    (r"if\s+(.+?),?\s+then\s+(.+)", "condition_result"),
    (r"(.+)\s+(?:causes?|leads? to|results? in)\s+(.+)", "cause_effect"),
    (r"to\s+(.+?),?\s+(?:you should|we need to)\s+(.+)", "goal_action"),
]


def _extract_causal_chains(text: str) -> List[Dict[str, str]]:
    """Extract causal relationships from text."""
    chains = []
    text_lower = text.lower()
    
    for pattern, chain_type in CAUSAL_PATTERNS:
        matches = re.finditer(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            chains.append({
                "type": chain_type,
                "antecedent": match.group(1).strip(),
                "consequent": match.group(2).strip(),
                "full_match": match.group(0),
            })
    
    return chains


def causal_chain_match(
    prediction: str,
    reference: str,
) -> Dict[str, Any]:
    """Measure how well prediction captures causal relationships from reference.
    
    Args:
        prediction: Model's answer
        reference: Ground truth or source text
    
    Returns:
        Dict with causal chain matching metrics
    """
    pred_chains = _extract_causal_chains(prediction)
    ref_chains = _extract_causal_chains(reference)
    
    if not ref_chains:
        # No causal chains in reference - check if prediction also has none
        return {
            "ref_chain_count": 0,
            "pred_chain_count": len(pred_chains),
            "matched": 0,
            "recall": 1.0 if not pred_chains else 0.5,  # Neutral if no chains expected
            "precision": 1.0 if not pred_chains else 0.5,
        }
    
    # Simple matching: check if consequent keywords appear
    matched = 0
    for ref_chain in ref_chains:
        ref_consequent_words = set(ref_chain["consequent"].split())
        for pred_chain in pred_chains:
            pred_consequent_words = set(pred_chain["consequent"].split())
            # Check word overlap
            if len(ref_consequent_words & pred_consequent_words) >= 2:
                matched += 1
                break
    
    return {
        "ref_chain_count": len(ref_chains),
        "pred_chain_count": len(pred_chains),
        "matched": matched,
        "recall": matched / len(ref_chains) if ref_chains else 0.0,
        "precision": matched / len(pred_chains) if pred_chains else 0.0,
        "ref_chains": ref_chains,
        "pred_chains": pred_chains,
    }


@dataclass
class DomainMetrics:
    """Container for computing domain-specific metrics."""
    
    ontology_path: Optional[Path] = None
    
    def __post_init__(self):
        self.ontology = _load_ontology(self.ontology_path)
    
    def compute_all(
        self,
        prediction: str,
        reference: str,
        check_causal: bool = True,
    ) -> Dict[str, Any]:
        """Compute all domain metrics.
        
        Args:
            prediction: Model's answer
            reference: Ground truth answer
            check_causal: Whether to check causal chain matching
        
        Returns:
            Dict of metric results
        """
        results = {}
        
        # Ontology coverage for prediction
        coverage = ontology_coverage(prediction, self.ontology)
        results["ontology_coverage"] = coverage["overall_coverage"]
        results["ontology_terms_found"] = coverage["total_terms_found"]
        
        # Term accuracy
        term_acc = term_accuracy(prediction, reference, self.ontology)
        results["term_precision"] = term_acc["precision"]
        results["term_recall"] = term_acc["recall"]
        results["term_f1"] = term_acc["f1"]
        
        # Causal chain matching
        if check_causal:
            causal = causal_chain_match(prediction, reference)
            results["causal_recall"] = causal["recall"]
            results["causal_precision"] = causal["precision"]
        
        return results
    
    def aggregate(self, all_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate metrics across multiple examples."""
        if not all_results:
            return {}
        
        # Get numeric metrics only
        numeric_keys = [k for k in all_results[0].keys() 
                       if isinstance(all_results[0][k], (int, float))]
        
        aggregated = {}
        for key in numeric_keys:
            values = [r.get(key, 0.0) for r in all_results]
            aggregated[f"mean_{key}"] = sum(values) / len(values)
        
        return aggregated
