#!/usr/bin/env python3
"""
Training Script

This script runs a training experiment defined in `src.training.experiment`.

Usage:
    python training_main.py --experiment <experiment_name> [--epochs N] [--evaluate-every-n-batches M]

Example:
    python training_main.py --experiment baselineResnet224 --epochs 10 --evaluate-every-n-batches 100

Valid experiment names (as of this code):
    - baselineResnet224
    - baselinePixtral224
    - baselinePixtral336
    - smallCollagesResnet224
    - smallCollagesPixtral224
    - smallCollagesPixtral336
    - largeCollagesResnet224
    - largeCollagesPixtral224
    - largeCollagesPixtral336

Notes:
    - Model, dataset and output paths are configured centrally in `src/config/paths.py`.
    - Set seeds are applied for reproducibility; deterministic CuDNN behavior is enabled
      which may affect runtime performance.
"""
import argparse
import sys
import random
import numpy as np
import torch

import src.training.experiment as experiment
import src.training.train as train

# Set seeds for reproducibility
def set_seed(seed=42):
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
        description="Conduct experiment"
    )

    parser.add_argument(
        "--experiment",
        type=str,
        help="Name of the experiment to be conducted"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Set the amount of epochs for training"
    )

    parser.add_argument(
        "--evaluate-every-n-batches",
        type=int,
        default=100,
        help="Number of batches after which evaluation on val dataset should be made"
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

        elif args.experiment=="baselinePixtral224":
            exp = experiment.ExperimentBaselinePixtral224()

        elif args.experiment=="baselinePixtral336":
            exp= experiment.ExperimentBaselinePixtral336()

        elif args.experiment=="smallCollagesResnet224":
            exp = experiment.ExperimentSmallCollagesResnet224()

        elif args.experiment=="smallCollagesPixtral224":
            exp = experiment.ExperimentSmallCollagesPixtral224()

        elif args.experiment=="smallCollagesPixtral336":
            exp = experiment.ExperimentSmallCollagesPixtral336()
        
        elif args.experiment=="largeCollagesResnet224":
            exp = experiment.ExperimentLargeCollagesResnet224()

        elif args.experiment=="largeCollagesPixtral224":
            exp = experiment.ExperimentLargeCollagesPixtral224()

        elif args.experiment=="largeCollagesPixtral336":
            exp = experiment.ExperimentLargeCollagesPixtral336()
        
        # Setup the experiment
        exp.setup_experiment()

        # Instantiate the evaluator
        trainer = train.Trainer(exp.model, exp.optimizer, exp.loss_function, args.epochs, exp.train_loader, exp.val_loader, exp.model_abbr, args.evaluate_every_n_batches, exp.study_type)

        # Train the model configured by the experiment
        trainer.train()
     

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
   

    
