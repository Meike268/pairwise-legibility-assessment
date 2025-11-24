#!/usr/bin/env python3
"""
Optuna Hyperparameter Tuning Script

This script performs hyperparameter tuning using the Optuna Framework. 

Usage:
    python optuna_tuning.py

Notes:
    - An Optuna objective is defined around an experiment/trainer pair.
    - The results are saved under the project `RESULTS_DIR/optuna` directory.
    - As of the current state, the experimental configurations for which hyperparameter tuning
        shall be performed are hardcoded.
"""

import os
import optuna
import json

import src.training.train as train
import src.models.preprocessor as preprocessor
import src.training.experiment as experiment
from src.config.paths import RESULTS_DIR, IMAGES_COLLAGES

class OptunaTuning:
    """
    Helper class to run Optuna studies for a specific experiment setup.

    Args:
        model_name: Name of the model ('resnet' or 'pixtral').
        image_size: Image size used for transforms (224 or 336).
        transform: Torchvision transform to apply to images.
        crop_size: Crop size label used by experiments ('small', 'large').
        input_directory: Directory containing input images (IMAGES_COLLAGES  or IMAGES_PADDED).
        study_type: Type of study ('collages' or 'baseline').
    """

    def __init__(self, model_name, image_size, transform, crop_size, input_directory, study_type):
        self.model_name = model_name 
        self.image_size = image_size
        self.transform = transform
        self.crop_size = crop_size
        self.input_directory = input_directory
        self.study_type = study_type
        self.epochs = 3
        self.trials = 7

        # Prepare results directory for optuna outputs
        self.optuna_dir = os.path.join(RESULTS_DIR, "optuna")
        os.makedirs(self.optuna_dir, exist_ok=True)

    
    def objective(self, trial):

        # Optimizer type
        optimizer_name = trial.suggest_categorical("optimizer", ["SGD", "Adam"])

        # Hyperparameter Ranges
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)

        # Momentum only for SGD
        if optimizer_name == "SGD":
            momentum = trial.suggest_float("momentum", 0.7, 0.99)
        else:
            momentum = None  # Not used for Adam

        # NOTE: current implementation uses a specific Experiment class.
        # Depending on `model_name`/`crop_size` you may want to instantiate
        # a different Experiment subclass here.
        exp = experiment.ExperimentLargeCollagesResnet224(
            momentum=momentum,
            lr=lr,
            weight_decay=weight_decay,
            dropout=dropout,
            optimizer_name=optimizer_name,
        )
        exp.setup_experiment()

        trainer = train.Trainer(
            exp.model,
            exp.optimizer,
            exp.loss_function,
            self.epochs,
            exp.train_loader,
            exp.val_loader,
            exp.model_abbr,
            355,
            exp.study_type,
        )

        best_roc_auc_val = trainer.train()

        return best_roc_auc_val

    def run_study(self):
        study_name = f"{self.study_type}_{self.model_name}_{self.crop_size}_crops_{self.image_size}"

        # Use a dedicated optuna directory under RESULTS_DIR
        db_name = f"{study_name}.db"
        db_path = os.path.join(self.optuna_dir, db_name)

        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage=f"sqlite:///{db_path}",
            load_if_exists=True,
        )

        print(f"Starting study: {study_name} with {self.trials} trials and {self.epochs} epochs.")

        # Run optimization
        study.optimize(self.objective, n_trials=self.trials)

        # Print best trial
        print("\n Best trial:")
        print(f"  Value: {study.best_trial.value}")
        print("  Params:")
        for key, value in study.best_trial.params.items():
            print(f"    {key}: {value}")

        # Save all trial results to CSV
        df = study.trials_dataframe()
        csv_name = f"optuna_results_{study_name}.csv"
        csv_path = os.path.join(self.optuna_dir, csv_name)
        df.to_csv(csv_path, index=False)
        print(f"\n📄 All trial results saved to: {csv_path}")

        # Save best parameters to JSON
        json_name = f"best_params_{study_name}.json"
        json_path = os.path.join(self.optuna_dir, json_name)
        with open(json_path, "w") as f:
            json.dump(study.best_trial.params, f, indent=4)
        print(f"📌 Best hyperparameters saved to: {json_path}")




if __name__ == "__main__":
    # This script can be used for different experimental configurations. You may want to adapt the values set here to your use case.

    model_name = 'resnet'  # 'resnet' or 'pixtral'
    image_size = 224 # 224 or 336
    transform = preprocessor.get_final_transforms_resnet(image_size) # 'get_final_transforms_resnet' or 'get_final_transforms_pixtral'
    crop_size = 'large'  # 'small', 'large' or None
    input_directory = IMAGES_COLLAGES # IMAGES_COLLAGES or IMAGES_PADDED
    study_type = 'collages' # 'collages' or 'baseline'

    # Initialize Optuna study
    optuna_study = OptunaTuning(model_name, image_size, transform, crop_size, input_directory, study_type)

    # Run Optuna study
    optuna_study.run_study()






