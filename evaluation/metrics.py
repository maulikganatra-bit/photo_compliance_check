"""
Metrics and agreement computation for license plate detection experiments.
"""
from typing import List, Dict, Any
from collections import Counter

def compute_agreement(yolo_results: List[Dict[str, Any]], llm_results: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Compute agreement statistics between YOLO and LLMs.
    Args:
        yolo_results: List of YOLO result dicts.
        llm_results: List of lists of LLM result dicts (one list per LLM).
    Returns:
        Dict with agreement rates, majority vote, and disagreement cases.
    """
    n = len(yolo_results)
    agreement_counts = [0] * len(llm_results)
    yolo_flags = 0
    llm_flags = [0] * len(llm_results)
    disagreement_cases = []
    majority_votes = []
    for i in range(n):
        yolo_flag = bool(yolo_results[i]["license_plate_detected"])
        yolo_flags += yolo_flag
        llm_flags_i = []
        for j, llm_list in enumerate(llm_results):
            llm_flag = bool(llm_list[i]["license_plate_detected"])
            llm_flags[j] += llm_flag
            llm_flags_i.append(llm_flag)
            if yolo_flag == llm_flag:
                agreement_counts[j] += 1
        # Majority vote
        votes = [yolo_flag] + llm_flags_i
        vote = Counter(votes).most_common(1)[0][0]
        majority_votes.append(vote)
        if not all(v == votes[0] for v in votes):
            disagreement_cases.append(i)
    agreement_rates = [c / n for c in agreement_counts]
    yolo_flagged_pct = yolo_flags / n
    llm_flagged_pct = [f / n for f in llm_flags]
    return {
        "agreement_rates": agreement_rates,
        "yolo_flagged_pct": yolo_flagged_pct,
        "llm_flagged_pct": llm_flagged_pct,
        "majority_votes": majority_votes,
        "disagreement_cases": disagreement_cases
    }
