#!/usr/bin/env python3
"""
Evaluation Script

Run evaluation for a trained model.

Usage:
    python evaluation_main.py --experiment <experiment_name> --evaluation-method <method> --dataset <validation|test>

Example:
    python evaluation_main.py --experiment baselineResnet224 --evaluation-method baseline --dataset validation

Valid evaluation methods:
    - baseline
    - all_baseline (runs baseline and baseline_assymmetry_averaging)
    - all_collage (runs collage, collage_averaging, collage_asymmetry_averaging, and collage_averaging_asymmetry_averaging)
    - baseline_asymmetry_averaging
    - collage
    - collage_averaging
    - collage_asymmetry_averaging
    - collage_averaging_asymmetry_averaging

Note:
    - Experiment definitions live in `src/training/experiment.py` and available experiment names
      are the same as those accepted by `training_main.py`.
    - Model file locations are taken from `src/config/paths.py`.
"""
import os
import argparse
import sys
import random
import numpy as np
import torch

import src.training.experiment as experiment
import src.training.evaluate as evaluate
from src.config.paths import MODELS_DIR

# Set seeds for reproducibility
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    
    parser = argparse.ArgumentParser(
        description="Evaluate model predictions against gold predictions"
    )

    parser.add_argument(
        "--experiment",
        type=str,
        help="Name of the experiment to be conducted"
    )

    parser.add_argument(
        "--evaluation-method",
        type=str,
        help=(
            "Evaluation method to apply. Options: 'baseline', 'all_baseline', 'all_collage', "
            "'baseline_asymmetry_averaging', 'collage', 'collage_averaging', "
            "'collage_asymmetry_averaging', 'collage_averaging_asymmetry_averaging'"
        ),
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["validation", "test"],
        help="Dataset on which the model shall be evaluated ('validation' or 'test').",
    )
    
    return parser.parse_args()

def validate_arguments(args):
    """Validate that the provided experiment name is one of the known experiments.

    Raises:
        ValueError: If the experiment name is not recognized.
    """
    valid = {
        "baselineResnet224",
        "baselinePixtral224",
        "baselinePixtral336",
        "smallCollagesResnet224",
        "smallCollagesPixtral224",
        "smallCollagesPixtral336",
        "largeCollagesResnet224",
        "largeCollagesPixtral224",
        "largeCollagesPixtral336",
    }
    if args.experiment not in valid:
        raise ValueError("Please select a valid experiment name.")


def main():
    try:
        # Set seed for reproducibility
        set_seed(42)
        
        # Parse the input arguments
        args = parse_arguments()

        # Validate the input arguments
        validate_arguments(args)

        # Instantiate the correct experiment based on the input argument
        if args.experiment=="baselineResnet224":
            exp = experiment.ExperimentBaselineResnet224()
            path = os.path.join(MODELS_DIR, "trained", "baseline_resnet_none_crops_224_model_best.pt")

        elif args.experiment=="baselinePixtral224":
            exp = experiment.ExperimentBaselinePixtral224()
            path = os.path.join(MODELS_DIR, "trained", "baseline_pixtral_none_crops_224_model_best.pt")

        elif args.experiment=="baselinePixtral336":
            exp= experiment.ExperimentBaselinePixtral336()
            path = os.path.join(MODELS_DIR, "trained", "baseline_pixtral_none_crops_336_model_best.pt")

        elif args.experiment=="smallCollagesResnet224":
            exp = experiment.ExperimentSmallCollagesResnet224()
            path = os.path.join(MODELS_DIR, "trained", "collages_resnet_small_crops_224_model_best.pt")

        elif args.experiment=="smallCollagesPixtral224":
            exp = experiment.ExperimentSmallCollagesPixtral224()
            path = os.path.join(MODELS_DIR, "trained", "collages_pixtral_small_crops_224_model_best.pt")

        elif args.experiment=="smallCollagesPixtral336":
            exp = experiment.ExperimentSmallCollagesPixtral336()
            path = os.path.join(MODELS_DIR, "trained", "collages_pixtral_small_crops_336_model_best.pt")
        
        elif args.experiment=="largeCollagesResnet224":
            exp = experiment.ExperimentLargeCollagesResnet224()
            path = os.path.join(MODELS_DIR, "trained", "collages_resnet_large_crops_224_model_best.pt")

        elif args.experiment=="largeCollagesPixtral224":
            exp = experiment.ExperimentLargeCollagesPixtral224()
            path = os.path.join(MODELS_DIR, "trained", "collages_pixtral_large_crops_224_model_best.pt")

        elif args.experiment=="largeCollagesPixtral336":
            exp = experiment.ExperimentLargeCollagesPixtral336()
            path = os.path.join(MODELS_DIR, "trained", "collages_pixtral_large_crops_336_model_best.pt")
        
        # Setup the experiment
        exp.setup_experiment()

        # Instantiate the evaluator
        eval = evaluate.Evaluator(exp.model, path, exp.model_abbr)

        # Run evaluation on validation dataset
        if args.dataset == "validation":
            status = eval.evaluate(exp.val_loader, args.dataset, args.evaluation_method)
            print(f"Evaluation on validation set completed with status {status}")

        # Run evaluation on test dataset
        elif args.dataset == "test":
            status = eval.evaluate(exp.test_loader, args.dataset, args.evaluation_method)
            print(f"Evaluation on test set completed with status {status}")

        # Invalid dataset definition
        else:
            raise ValueError("Please use a valid dataset (validation or test)!")
        

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
   

    
