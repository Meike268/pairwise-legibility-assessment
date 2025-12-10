#!/usr/bin/env python3
"""
Ranking Evaluation Script

This script creates rankings from ground-truth standard pairwise annotations and model predictions,
then evaluates how well the model ranking correlates with the ground-truth standard using Kendall's tau.

Usage:
    python ranking_main.py --gold-path /path/to/gold_pairwise_comparisons.csv --model-path /path/to/model_pairwise_comparisons.csv [--gold-output-dir /path/to/save_dir.csv] 
        [--model-output-dir /path/to/save_dir.csv] [--use-binary-pred <true|false>]

Notes:
    - Default paths are set for the gold path and the model path. 
    - The output directories (gold-output-dir, model-output-dir) are optional. If set, the rankings are saved under the specified path. 
    - The binary prediction should only be set to true, if the model predictions are binary (0 or 1) and not available as predictions. 
    - The create_ranking.Ranker takes the column names for id1, id2, and label as input. Their values need to be adapted to the
        column names used in the respective files containing the pairwise comparisons.
"""

import os
import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import src.ranking.create_ranking as create_ranking
import src.ranking.evaluate_ranking as evaluate_ranking
from src.config.paths import ANNOTATIONS_DIR, RESULTS_DIR


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate model rankings against ground-truth standard rankings"
    )
    
    parser.add_argument(
        "--gold-path",
        type=str,
        default=os.path.join(ANNOTATIONS_DIR, "samples_test_pairs_with_scores.csv"),
        help="Path to gold standard pairwise comparison CSV file"
    )
    
    parser.add_argument(
        "--model-path", 
        type=str,
        default=os.path.join(RESULTS_DIR, "collages_pixtral_medium_crops_224", "experiment", "collages_pixtral_medium_crops_224_test_pred.csv"),
        help="Path to model predictions CSV file"
    )
    
    parser.add_argument(
        "--gold-output-dir",
        type=str,
        default="",
        help="Directory to save gold ranking results (optional)"
    )

    parser.add_argument(
        "--model-output-dir",
        type=str,
        default="",
        help="Directory to save model ranking results (optional)"
    )
    
    parser.add_argument(
        "--use-binary-pred",
        action="store_true",
        help="Use binary predictions instead of probabilities for model ranking (optional)"
    )
    
    return parser.parse_args()


def validate_file_paths(gold_path: str, model_path: str):
    """Validate that input files exist."""
    if not Path(gold_path).exists():
        raise FileNotFoundError(f"Gold standard file not found: {gold_path}")
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model predictions file not found: {model_path}")


def create_gold_ranking(gold_path: str, gold_output_dir: str):
    """Create ranking from gold standard annotations."""
    print("Creating gold standard ranking...")
    
    ranker_gold = create_ranking.Ranker(
        comparisons_file=gold_path,
        id1="sampleId1",
        id2="sampleId2", 
        label="score",
        save_path=gold_output_dir
    )
    
    return ranker_gold.rank()


def create_model_ranking(model_path: str, model_output_dir: str, use_binary_pred: bool = False):
    """Create ranking from model predictions."""
    print("Creating model ranking...")
    
    label_column = "pred" if use_binary_pred else "prob"
    print(f"Using '{label_column}' column for model ranking")
    
    ranker_model = create_ranking.Ranker(
        comparisons_file=model_path,
        id1="id1",
        id2="id2",
        label=label_column,
        save_path=model_output_dir
    )
    
    return ranker_model.rank()


def evaluate_rankings(gold_ranking: List, model_ranking: List, gold_ids: set, model_ids: set):
    """Evaluate model ranking against gold standard using Kendall's tau."""
    # Find common items for fair comparison
    common_ids = list(set(gold_ids) & set(model_ids))
    
    print(f"Gold ranking has {len(gold_ids)} items")
    print(f"Model ranking has {len(model_ids)} items")
    print(f"Common items for evaluation: {len(common_ids)}")
    
    if len(common_ids) == 0:
        raise ValueError("No common items found between gold and model rankings!")
    
    if len(common_ids) < 10:
        print(f"WARNING: Only {len(common_ids)} common items found. Results may be unreliable.")
    
    # Evaluate rankings
    evaluator = evaluate_ranking.Evaluator(gold_ranking, model_ranking, common_ids)
    tau, p_value = evaluator.kendall_tau_from_rankings()
    
    return tau, p_value


def print_results(tau: float, p_value: float) -> None:
    """Print evaluation results."""
    print(f"\n{'='*50}")
    print("RANKING EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Kendall's tau: {tau:.4f}")
    print(f"p-value:       {p_value:.4f}")
    
    # Interpret correlation strength
    if abs(tau) >= 0.7:
        strength = "strong"
    elif abs(tau) >= 0.4:
        strength = "moderate"
    elif abs(tau) >= 0.2:
        strength = "weak"
    else:
        strength = "very weak"
    
    direction = "positive" if tau > 0 else "negative"
    
    print(f"Correlation:   {strength} {direction} correlation")
    
    # Statistical significance
    if p_value < 0.001:
        significance = "highly significant (p < 0.001)"
    elif p_value < 0.01:
        significance = "very significant (p < 0.01)"
    elif p_value < 0.05:
        significance = "significant (p < 0.05)"
    else:
        significance = "not significant (p >= 0.05)"
    
    print(f"Significance:  {significance}")
    print(f"{'='*50}")


def main() -> None:
    """Main function to orchestrate the ranking evaluation."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Validate inputs
        validate_file_paths(args.gold_path, args.model_path)
        
        # Create gold ranking
        gold_ranking, theta_gold, ids_gold, trace_gold = create_gold_ranking(
            args.gold_path, args.gold_output_dir
        )
        
        # Create model ranking
        model_ranking, theta_model, ids_model, trace_model = create_model_ranking(
            args.model_path, args.model_output_dir, args.use_binary_pred
        )
        
        # Evaluate model ranking agaings gold ranking
        tau, p_value = evaluate_rankings(gold_ranking, model_ranking, ids_gold, ids_model)
        
        # Print results
        print_results(tau, p_value)
        
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()