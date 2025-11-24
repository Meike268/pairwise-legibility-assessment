import sys
sys.path.append('../src')

import torch
import time
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import os
import csv
import pandas as pd
from src.config.paths import RESULTS_DIR, MODELS_DIR

def move_to_device(*args, device):
    """Move one or more tensors to the target device."""

    return [x.to(device) for x in args]

class Trainer:
    """Encapsulates training loop, evaluation and logging for a model.

    The Trainer manages device placement, epoch/batch training, periodic
    validation, plotting of results, checkpoint saving and CSV logging.

    Args:
        model (torch.nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): Optimizer for parameter updates.
        loss_function (callable): Loss function taking (preds, targets).
        epochs (int): Number of epochs to train.
        train_loader (Iterable): DataLoader for training data.
        val_loader (Iterable): DataLoader for validation data.
        model_abbr (str): Short name used to build output filenames.
        evaluate_every_n (int): Evaluate and log every N batches.
        study_type (str): Either 'baseline' or 'collages' to switch validation logic.
    """

    def __init__(self, model, optimizer, loss_function, epochs, train_loader, val_loader, model_abbr, evaluate_every_n, study_type):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.epochs = epochs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = None
        self.model_abbr = model_abbr
        self.evaluate_every_n = evaluate_every_n
        self.study_type = study_type
        self.best_roc_auc = 0.0
        self.train_losses, self.val_losses, self.train_accuracies, self.val_accuracies = [], [], [], []


    # -----------------------------------------HELPER METHODS-------------------------------------------
    def setup_model(self):
        """Configure device and move the model to the chosen device.

        Sets ``self.device`` to CUDA when available and transfers the model
        parameters to that device.
        """
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        # Initialize model
        self.model = self.model.to(self.device)
        print("Model is on device:", next(self.model.parameters()).device)
    
    def save_model(self, epoch: int, batch: int, path: str):
        """Save a training checkpoint to disk.

        The checkpoint contains the epoch, batch index, model and optimizer
        state dicts.
        """
        torch.save({
            "epoch": epoch,
            "batch": batch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)
        print(f"Model saved to {sys.path}")

    
    def print_results_training(self, path: str):
        """Print a short summary after training completes."""
        print(f"\n{'='*50}")
        print(f"TRAINING COMPLETED AFTER {self.epochs}")
        print(f"{'='*50}")
        print(f"Best AUC Value on val: {self.best_roc_auc:.4f}")
        print(f"Path to best model:    {path}")
        print(f"{'='*50}")

    def plot_training_results(self, epoch: int, batch: int, train_losses: list, val_losses: list, train_accuracies: list, val_accuracies: list, roc_auc: float, fpr, tpr):
        """Plot training/validation metrics and save figures."""

        # Create x-axis labels that match the actual number of data points
        num_evaluations = len(train_losses)
        x_list = [f"Eval_{i}" for i in range(num_evaluations)]

        plt.figure(figsize=(8, 5))
        plt.plot(x_list, train_losses, label='Train Loss')
        plt.plot(x_list, val_losses, label='Val Loss')
        plt.xlabel(f'Evaluation Points (every {self.evaluate_every_n} batches)')
        plt.ylabel('Value')
        plt.title(f'Model Loss {self.model_abbr}')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(RESULTS_DIR, f"{self.model_abbr}_train_loss_epoch_{epoch}_batch_{batch}.png"))

        plt.figure(figsize=(8, 5))
        plt.plot(x_list, train_accuracies, label='Train Accuracy')
        plt.plot(x_list, val_accuracies, label='Val Accuracy')
        plt.xlabel(f'Evaluation Points (every {self.evaluate_every_n} batches)')
        plt.ylabel('Value')
        plt.title(f'Model Accuracy {self.model_abbr}')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(RESULTS_DIR, f"{self.model_abbr}_train_accuracy_epoch_{epoch}_batch_{batch}.png"))

        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve Val (Epoch {epoch}, Batch {batch})")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(os.path.join(RESULTS_DIR, f"{self.model_abbr}_val_roc_curve_epoch_{epoch}_batch_{batch}.png"))

    def log_metrics_to_csv(self, metrics: dict):
        """Append a metrics row to a CSV file named after the model."""

        filepath = os.path.join(RESULTS_DIR, f"{self.model_abbr}_train_metrics.csv")
        file_exists = os.path.isfile(filepath)
        with open(filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics)

    
    #------------------------------------MAIN TRAIN AND EVALUATION LOGIC-------------------------------------

    def train_epoch(self, train_loader, epoch, eval_every_n_batches):

        """Run training for a single epoch, evaluating periodically.

        This method iterates over ``train_loader`` and performs forward and
        backward passes. Every ``eval_every_n_batches`` it runs validation
        and logs/plots metrics.

        Args:
            train_loader (Iterable): DataLoader for training data.
            epoch (int): Current epoch index used for logging/filenames.
            eval_every_n_batches (int): Frequency (in batches) to run validation.
        """

        # Initialize loss and accuracy counters
        train_loss = 0.0  # Total training loss for the current epoch
        train_count_correct = 0  # Total number of correct predictions in the current epoch
        train_count_total = 0  # Total number of samples seen in the current epoch
        t1 = time.time()

        try:
            # Iterate over batches: each batch returns
            # - image1, image2: torch tensors of shape [batch_size, channels, height, width]
            # - target: tensor of -1 or 1 of shape [batch_size]
            print(f"Starting model training for epoch {epoch}")
            for batch, (image1, image2, target, _, _) in enumerate(tqdm(train_loader, desc=f"Training")):
                self.model.train()
                image1, image2, target = move_to_device(image1, image2, target, device=self.device)
                #print("image1 batch shape:", image1.shape)  # Should be (32, 3, H, W)
                #print("image2 batch shape:", image2.shape)
                target = (target == 1).float().unsqueeze(1)  # Converts each value in target to 1.0 if it is 1, and to 0.0 if it is -1 -> Shape becomes [batch_size, 1]
                self.optimizer.zero_grad()
                score = self.model(image1, image2)  # Forward pass, returns one logit per input pair -> shape: [batch_size, 1]
                loss = self.loss_function(score, target) # Loss computation 
                loss.backward() # Backpropagation
                self.optimizer.step() # Backpropagation 
                train_loss += loss.item() * target.size(0) # Accumulate loss (loss.item() holds loss for curent batch) over current batch

                # Accuracy Tracking: torch.sign(score) gives -1, 0 or 1 -> compare predictions with the true target to compute training accuracy
                prob = torch.sigmoid(score) # results in probabilities between 0 and 1
                pred = (prob >= 0.5).int() # Convert probabilities to binary predictions (1 or 0)
                train_count_correct += (pred == target.int()).sum().item() # Count correct predictions
                train_count_total += target.size(0) # Adds batch size to total number of examples seen so far 

                # Evaluate train and val every N batches
                if eval_every_n_batches is not None and (batch + 1) % eval_every_n_batches == 0:
                    print(f"Evaluating after batch {batch}")
                    t2 = time.time() - t1 
                    
                    # Calculate train loss and train accuracy for this n batches
                    train_loss = train_loss / train_count_total
                    train_acc = train_count_correct / train_count_total if train_count_total > 0 else 0

                    # Validate on val dataset
                    if self.study_type == "baseline":
                        val_loss, val_acc, precision, recall, f1, roc_auc, fpr, tpr = self.evaluate_on_val_baseline(self.val_loader)
                    else:
                        val_loss, val_acc, precision, recall, f1, roc_auc, fpr, tpr = self.evaluate_on_val_collage(self.val_loader)

                    self.train_accuracies.append(train_acc)
                    self.train_losses.append(train_loss)
                    self.val_accuracies.append(val_acc)
                    self.val_losses.append(val_loss)

                    # Plot training results
                    self.plot_training_results(epoch, batch, self.train_losses, self.val_losses, self.train_accuracies, self.val_accuracies, roc_auc, fpr, tpr)

                    # Save results to csv
                    metrics = {"epoch": epoch, "batch": batch, "train_loss": train_loss, "train_accuracy": train_acc,
                       "val_loss": val_loss, "val_accuracy": val_acc, "val_precision": precision,
                       "val_recall": recall, "val_f1_score": f1, "val_roc_auc": roc_auc, "time_s": t2}
                    self.log_metrics_to_csv(metrics)

                    # Save best model
                    if roc_auc > self.best_roc_auc:
                        self.best_roc_auc = roc_auc
                        self.save_model(epoch, batch, path=os.path.join(MODELS_DIR, f"{self.model_abbr}_model_best.pt"))
                        print("Saved new best model!")

                    # Reset counters
                    train_loss = 0.0
                    train_count_correct = 0
                    train_count_total = 0
                    t1 = time.time()

        except Exception as e:
            print(f"Error in training loop: {e}")
            raise
     
    def evaluate_on_val_baseline(self, val_loader):
        """Validate the model on baseline (non-collage) data.

        The function expects batches of images shaped [batch_size, C, H, W]
        and returns the aggregated loss, accuracy and classification metrics
        along with ROC curve points.

        Args:
            val_loader (Iterable): DataLoader yielding validation batches.

        Returns:
            tuple: (val_loss, val_acc, precision, recall, f1, roc_auc, fpr, tpr)
        """
        self.model.eval()
        val_loss = 0.0
        val_count_correct = 0
        val_count_total = 0

        all_preds, all_probs, all_labels = [], [], []

        for image1, image2, target, _, _ in tqdm(val_loader, desc="Validation"):
            image1, image2, target = move_to_device(image1, image2, target, device=self.device)
            # image1, image2: [batch_size, C, H, W]
            # target: [batch_size]
            target = (target == 1).float().unsqueeze(1)  # [batch_size, 1]

            with torch.no_grad():
                score = self.model(image1, image2)  # [batch_size, 1]
                loss = self.loss_function(score, target)
                val_loss += loss.item() * target.size(0)

                prob = torch.sigmoid(score)
                pred = (prob >= 0.5).int()

                val_count_correct += (pred == target.int()).sum().item()
                val_count_total += target.size(0)

                all_preds.extend(pred.view(-1).cpu().numpy())
                all_probs.extend(prob.view(-1).cpu().numpy())
                all_labels.extend(target.view(-1).cpu().numpy())

        val_acc = val_count_correct / val_count_total if val_count_total > 0 else 0
        val_loss /= val_count_total

        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)

        fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)

        return val_loss, val_acc, precision, recall, f1, roc_auc, fpr, tpr
    

    def evaluate_on_val_collage(self, val_loader):
        """Validate the model on collage inputs (multiple crops).

        This expects each batch to contain tensors with shape
        [batch_size, num_crops, C, H, W]. The method flattens crops for model
        inference, computes per-crop metrics and then aggregates results to
        return loss, accuracy, precision, recall, f1 and ROC data.

        Args:
            val_loader (Iterable): DataLoader yielding collage validation batches.

        Returns:
            tuple: (val_loss, val_acc, precision, recall, f1, roc_auc, fpr, tpr)
        """
        self.model.eval()
        # Initialize loss and accuracy counters
        val_loss = 0.0 # Validation loss
        val_count_correct = 0 # Total number of correct predictions 
        val_count_total = 0 # Total number of samples seen 

        all_preds, all_probs, all_labels = [], [], []
        all_ids = []

        with torch.no_grad():
            for image1, image2, target, id1_batch, id2_batch in tqdm(val_loader, desc="Validation"):
                image1, image2, target = move_to_device(image1, image2, target, device=self.device)
                #print("image1 val shape:", image1.shape, flush=True)
                #print("image2 val shape:", image2.shape, flush=True)

                batch_size, num_crops, C, H, W = image1.shape
                
                # Flatten batch and crops to pass through model
                image1_flat = image1.view(batch_size * num_crops, C, H, W)
                image2_flat = image2.view(batch_size * num_crops, C, H, W)

                #print("image1 val flattened shape:", image1_flat.shape, flush=True)
                #print("image2 val flattened shape:", image2_flat.shape, flush=True)
                
                score_flat = self.model(image1_flat, image2_flat)  # shape: [batch_size * num_crops, 1]
                #print("score val flattened shape:", score_flat.shape, flush=True)

                # Use individual crop scores instead of aggregating
                # Expand targets to match number of crops per sample
                target = (target == 1).float()  # shape: [batch_size]
                target_expanded = target.unsqueeze(1).expand(-1, num_crops).reshape(-1, 1)  # shape: [batch_size * num_crops, 1]
                #print("target expanded shape:", target_expanded.shape, flush=True)
                
                # Calculate loss using all individual crop predictions
                loss = self.loss_function(score_flat, target_expanded)
                val_loss += loss.item() * target_expanded.size(0)
                
                # Convert scores to probabilities and predictions
                prob_flat = torch.sigmoid(score_flat)  # shape: [batch_size * num_crops, 1]
                pred_flat = (prob_flat >= 0.5).int()   # shape: [batch_size * num_crops, 1]
                
                # Count correct predictions (each crop is evaluated individually)
                val_count_correct += (pred_flat == target_expanded.int()).sum().item()
                val_count_total += target_expanded.size(0)

                # Store individual crop predictions for metrics calculation
                all_preds.extend(pred_flat.view(-1).cpu().numpy())
                all_probs.extend(prob_flat.view(-1).cpu().numpy())
                all_labels.extend(target_expanded.view(-1).cpu().numpy())
                
                # Store IDs for each crop (each crop gets the same ID pair but with crop index)
                if id1_batch is not None and id2_batch is not None:
                    for i in range(batch_size):
                        for crop_idx in range(num_crops):
                            all_ids.append((f"{id1_batch[i]}_crop{crop_idx}", f"{id2_batch[i]}_crop{crop_idx}"))
                else:
                    current_id_count = len(all_ids)
                    for i in range(batch_size):
                        for crop_idx in range(num_crops):
                            all_ids.append((f"unknown_{current_id_count + i * num_crops + crop_idx}_1_crop{crop_idx}", 
                                          f"unknown_{current_id_count + i * num_crops + crop_idx}_2_crop{crop_idx}"))



        # Calculate validation accuracy and loss 
        val_acc = val_count_correct / val_count_total if val_count_total > 0 else 0
        val_loss /= val_count_total

        # Calculate precision, recall and f1 
        print(f"Debug: len(all_labels)={len(all_labels)}, len(all_preds)={len(all_preds)}, len(all_probs)={len(all_probs)}", flush=True)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)

        # Compute ROC for the epoch
        print(f"Debug: About to compute ROC curve with {len(all_labels)} labels and {len(all_probs)} probs", flush=True)
        fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)

        # Create and save predictions DataFrame with individual crop predictions
        df = pd.DataFrame({
            'id1': [ids[0] for ids in all_ids],
            'id2': [ids[1] for ids in all_ids],
            'pred': all_preds,
            'prob': all_probs,
            'label': all_labels
        })
        
        return val_loss, val_acc, precision, recall, f1, roc_auc, fpr, tpr   
    
    # MAIN FUNCTION TO CALL FOR MODEL TRAINING
    def train(self):
        """Run the full training loop across epochs.

        This is the main entry point to start training. It calls
        ``setup_model``, iterates over epochs invoking ``train_epoch``, saves
        per-epoch checkpoints and prints a final summary. The method returns
        the best validation ROC AUC observed during training.

        Returns:
            float: Best validation ROC AUC value observed.
        """

        try:
            print("Setting up the model...")
            self.setup_model()
            print("Successfully set up the model!")
        except Exception as e:
            print(f"Error setting up the model {self.model_abbr}: {e}")
            raise

        num_batches = len(self.train_loader) // 32
        remaining_samples = len(self.train_loader) % 32
        if remaining_samples != 0:
            num_batches = num_batches + 1

        print(f"Starting training for model {self.model_abbr} with {self.epochs} epochs and evaluation every {self.evaluate_every_n} batches.")
        for epoch in range(self.epochs):
            print("Starting model training...")
            self.train_epoch(self.train_loader, epoch, self.evaluate_every_n)

            print("Saving model for epoch...")
            self.save_model(epoch, num_batches, os.path.join(MODELS_DIR, f"{self.model_abbr}_model_epoch_{epoch}.pt"))

        path_to_model = os.path.join(MODELS_DIR, f"{self.model_abbr}_model_best.pt")
        self.print_results_training(path_to_model)

        return self.best_roc_auc


