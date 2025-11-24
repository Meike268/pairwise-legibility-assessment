import sys
sys.path.append('../src')

import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import csv
import os
import pandas as pd

from src.config.paths import RESULTS_DIR

def move_to_device(*args, device):
    """Move a collection of tensors to the given device."""

    return [x.to(device) for x in args]


class Evaluator:
    """
    Evaluator for running model inference and recording evaluation outputs.

    Provides methods to load a checkpoint, prepare the model device,
    run several inference variants, compute basic metrics and ROC curves, plot results, and
    log metrics/predictions to CSV files.

    Args:
        model (torch.nn.Module): The PyTorch model to evaluate.
        path_to_model (str): Path to a checkpoint file (used by ``load_model``).
        model_abbr (str): Short abbreviation used to name output files.
    """

    def __init__(self, model, path_to_model, model_abbr):
        self.model = model
        self.path_to_model = path_to_model
        self.model_abbr = model_abbr

#-----------------------------------------------HELPER METHODS----------------------------------------------------

    def load_model(self):
        """Load model weights and checkpoint metadata from disk."""
        checkpoint = torch.load(self.path_to_model)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.start_epoch = checkpoint["epoch"]

    def setup_model(self):
        """Configure device and load a checkpoint."""
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        # Initialize model
        self.model.to(self.device)
        print("Model is on device:", next(self.model.parameters()).device)

        # Load checkpoint if available
        try:
            self.load_model()
            print(f"Model loaded from {self.path_to_model} at epoch {self.start_epoch}")
        except Exception as e:
            self.start_epoch = 0
            print(f"Failed to load model from {self.path_to_model}. Error: {e}")

    def compute_metrics(self, all_labels, all_preds):
        """Compute simple classification metrics from labels and predictions.

        Args:
            all_labels (Sequence[int] or np.ndarray): Ground-truth binary labels.
            all_preds (Sequence[int] or np.ndarray): Predicted binary labels.

        Returns:
            float: Accuracy score.
        """

        acc = accuracy_score(all_labels, all_preds)
        return acc
    
    def compute_roc(self, all_labels, all_probs):
        """Compute ROC curve points and area-under-curve from probabilities.

        Args:
            all_labels (Sequence[int] or np.ndarray): Ground-truth binary labels.
            all_probs (Sequence[float] or np.ndarray): Predicted positive-class probabilities.

        Returns:
            tuple: (fpr, tpr, roc_auc) where fpr/tpr are arrays and roc_auc is a float.
        """

        fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, roc_auc

    def plot_evaluation_results(self, fpr, tpr, roc_auc, dataset):
        """Plot and save an ROC curve to the results directory.

        Args:
            fpr (array-like): False positive rates.
            tpr (array-like): True positive rates.
            roc_auc (float): Area under the ROC curve.
            dataset (str): Dataset name used in the output filename.
        """

        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve Test")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(os.path.join(RESULTS_DIR, f"{self.model_abbr}_{dataset}_roc_curve.png"))

    def log_metrics_to_csv(self, metrics: dict, dataset):
        """Append evaluation metrics to a CSV file, preserving header stability.

        If the file already exists and contains a header, that header will be
        reused so that subsequent rows have a stable column order. Missing
        keys are written as empty strings.

        Args:
            metrics (dict): Mapping of metric names to values.
            dataset (str): Dataset identifier used to name the output file.
        """

        filepath = os.path.join(RESULTS_DIR, f"{self.model_abbr}_{dataset}_evaluation_metrics.csv")

        # Consider file exists only if it has content (non-zero size)
        file_exists = os.path.isfile(filepath) and os.path.getsize(filepath) > 0

        # Determine stable fieldnames: reuse existing header if present, else use metrics keys
        if file_exists:
            with open(filepath, newline='') as f_read:
                reader = csv.reader(f_read)
                existing_header = next(reader, None)
            fieldnames = existing_header if existing_header else list(metrics.keys())
        else:
            fieldnames = list(metrics.keys())

        # Write/appended row using stable fieldnames (fill missing keys with empty string)
        with open(filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            row = {k: metrics.get(k, "") for k in fieldnames}
            writer.writerow(row)

#-----------------------------------------------INFERENCE METHODS----------------------------------------------------


    def inference_baseline(self, data_loader, type):
        """Evaluate the model on baseline-preprocessed images.

        This performs a forward pass without any crop aggregation: each sample
        in the batch is processed and a single probability/prediction is
        returned per sample.

        Args:
            data_loader (Iterable): DataLoader that yields
                (image1, image2, target, id1_batch, id2_batch).
            type (str): Dataset name used for progress display and output naming.

        Returns:
            tuple: (all_preds, all_probs, all_labels, all_ids)
        """
        all_preds, all_labels, all_probs = [], [], []
        all_ids = []  # Store (id1, id2) pairs for each sample

        with torch.no_grad():
            for image1, image2, target, id1_batch, id2_batch in tqdm(data_loader, desc=f"Validation on {type} set"):
                image1, image2, target = move_to_device(image1, image2, target, device=self.device)
                batch_size, C, H, W = image1.shape

                target = (target == 1).float().unsqueeze(1) 
                
                # Forward pass
                score = self.model(image1, image2)  # logits, shape: [batch_size * num_crops, 1]
                
                # Convert logits to probabilities
                prob = torch.sigmoid(score)
                pred = (prob >= 0.5).int()

                # Convert to numpy for processing
                pred_numpy = pred.view(-1).cpu().numpy()
                prob_numpy = prob.view(-1).cpu().numpy()
                target_numpy = target.view(-1).cpu().numpy()
                
                # Store results for each sample in the batch
                for i in range(batch_size):
                    all_preds.append(pred_numpy[i])
                    all_probs.append(prob_numpy[i])
                    all_labels.append(target_numpy[i])
                    
                    # Store IDs if available
                    if id1_batch is not None and id2_batch is not None:
                        all_ids.append((id1_batch[i], id2_batch[i]))
                    else:
                        all_ids.append((f"unknown_{len(all_ids)}_1", f"unknown_{len(all_ids)}_2"))
        
        return all_preds, all_probs, all_labels, all_ids

        
    def inference_baseline_assymmetry_averaging(self, data_loader, type):
        """Evaluate baseline inputs using asymmetry averaging.

        The method computes predictions for (A,B) and (B,A) and combines them
        using the asymmetry-averaging method implemented in the project.

        Args:
            data_loader (Iterable): DataLoader yielding batched samples.
            type (str): Dataset name used for progress and output.

        Returns:
            tuple: (all_preds, all_probs, all_labels, all_ids)
        """
        all_preds, all_labels, all_probs = [], [], []
        all_ids = []  # Store (id1, id2) pairs for each sample

        with torch.no_grad():
            for image1, image2, target, id1_batch, id2_batch in tqdm(data_loader, desc=f"Validation on {type} set"):
                image1, image2, target = move_to_device(image1, image2, target, device=self.device)
                batch_size, C, H, W = image1.shape

                target = (target == 1).float().unsqueeze(1) 
                
                # Forward pass
                score1 = self.model(image1, image2)  # logits, shape: [batch_size * num_crops, 1]
                score2 = self.model(image2, image1)
                
                # Convert logits to probabilities
                prob1 = torch.sigmoid(score1)
                prob2 = torch.sigmoid(score2)
                prob = 0.5 * (prob1 + (1-prob2))
                pred = (prob >= 0.5).int()

                # Convert to numpy for processing
                pred_numpy = pred.view(-1).cpu().numpy()
                prob_numpy = prob.view(-1).cpu().numpy()
                target_numpy = target.view(-1).cpu().numpy()
                
                # Store results for each sample in the batch
                for i in range(batch_size):
                    all_preds.append(pred_numpy[i])
                    all_probs.append(prob_numpy[i])
                    all_labels.append(target_numpy[i])
                    
                    # Store IDs if available
                    if id1_batch is not None and id2_batch is not None:
                        all_ids.append((id1_batch[i], id2_batch[i]))
                    else:
                        all_ids.append((f"unknown_{len(all_ids)}_1", f"unknown_{len(all_ids)}_2"))
        
        return all_preds, all_probs, all_labels, all_ids
    

    def inference_collage(self, data_loader, type):
        """Evaluate the model on collage-preprocessed images.

        This method handles inputs where each sample contains multiple crops
        (``num_crops``). It flattens crops for model execution and returns
        per-crop predictions and probabilities.

        Args:
            data_loader (Iterable): DataLoader that yields
                (image1, image2, target, id1_batch, id2_batch) where image tensors
                have shape [batch_size, num_crops, C, H, W].
            type (str): Dataset name used for progress display and output naming.

        Returns:
            tuple: (all_preds, all_probs, all_labels, all_ids)
        """
        all_preds, all_labels, all_probs = [], [], []
        all_ids = []  # Store (id1, id2) pairs for each sample

        with torch.no_grad():
            for image1, image2, target, id1_batch, id2_batch in tqdm(data_loader, desc=f"Validation on {type} set"):
                image1, image2, target = move_to_device(image1, image2, target, device=self.device)
                
                batch_size, num_crops, C, H, W = image1.shape
                
                # Flatten batch and crops
                image1_flat = image1.view(batch_size * num_crops, C, H, W)
                image2_flat = image2.view(batch_size * num_crops, C, H, W)
                
                # Forward pass
                score_flat = self.model(image1_flat, image2_flat)  # logits, shape: [batch_size * num_crops, 1]
                
                
                target = (target == 1).float()  # shape: [batch_size]
                target_expanded = target.unsqueeze(1).expand(-1, num_crops).reshape(-1, 1)  # shape: [batch_size * num_crops, 1]
                             
                # Convert aggregated logits to probabilities
                prob_flat = torch.sigmoid(score_flat)
                pred_flat = (prob_flat >= 0.5).int()

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
                    for i in range(batch_size):
                        for crop_idx in range(num_crops):
                            all_ids.append((f"unknown_{len(all_ids)}_1_crop{crop_idx}", f"unknown_{len(all_ids)}_2_crop{crop_idx}"))

        
        return all_preds, all_probs, all_labels, all_ids

    def inference_collage_asymmetry_averaging(self, data_loader, type):
        """Evaluate collage inputs using asymmetry averaging.

        Computes predictions for both crop-sets (A vs B and B vs A), aggregates
        per-crop logits into per-sample logits and applies the asymmetry
        averaging heuristic.

        Args:
            data_loader (Iterable): Batched data with multiple crops per sample.
            type (str): Dataset identifier used for progress messages.

        Returns:
            tuple: (all_preds, all_probs, all_labels, all_ids)
        """
        all_preds, all_labels, all_probs = [], [], []
        all_ids = []  # Store (id1, id2) pairs for each sample

        with torch.no_grad():
            for image1, image2, target, id1_batch, id2_batch in tqdm(data_loader, desc=f"Validation on {type} set"):
                image1, image2, target = move_to_device(image1, image2, target, device=self.device)
                
                batch_size, num_crops, C, H, W = image1.shape
                
                # Flatten batch and crops
                image1_flat = image1.view(batch_size * num_crops, C, H, W)
                image2_flat = image2.view(batch_size * num_crops, C, H, W)
                
                # Forward pass
                score_flat_1 = self.model(image1_flat, image2_flat)  # logits, shape: [batch_size * num_crops, 1]
                score_flat_2 = self.model(image2_flat, image1_flat)
                
                
                target = (target == 1).float()  # shape: [batch_size]
                target_expanded = target.unsqueeze(1).expand(-1, num_crops).reshape(-1, 1)  # shape: [batch_size * num_crops, 1]
                             
                # Convert aggregated logits to probabilities
                prob_flat_1 = torch.sigmoid(score_flat_1)
                prob_flat_2 = torch.sigmoid(score_flat_2)
                prob_flat = 0.5 * (prob_flat_1 + (1-prob_flat_2))
                pred_flat = (prob_flat >= 0.5).int()

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
                    for i in range(batch_size):
                        for crop_idx in range(num_crops):
                            all_ids.append((f"unknown_{len(all_ids)}_1_crop{crop_idx}", f"unknown_{len(all_ids)}_2_crop{crop_idx}"))

        
        return all_preds, all_probs, all_labels, all_ids



    def inference_collage_averaging(self, data_loader, type):
        """Evaluate collage inputs and aggregate crop logits by mean per sample.

        This flattens the crop dimension for model inference, reshapes the
        returned logits to [batch_size, num_crops, ...], and aggregates via
        mean to produce a single prediction per sample.

        Args:
            data_loader (Iterable): Batched data containing multiple crops.
            type (str): Dataset name for logging/output filenames.

        Returns:
            tuple: (all_preds, all_probs, all_labels, all_ids)
        """
        all_preds, all_labels, all_probs = [], [], []
        all_ids = []  # Store (id1, id2) pairs for each sample

        with torch.no_grad():
            for image1, image2, target, id1_batch, id2_batch in tqdm(data_loader, desc=f"Validation on {type} set"):
                image1, image2, target = move_to_device(image1, image2, target, device=self.device)
                
                batch_size, num_crops, C, H, W = image1.shape
                
                # Flatten batch and crops
                image1_flat = image1.view(batch_size * num_crops, C, H, W)
                image2_flat = image2.view(batch_size * num_crops, C, H, W)
                
                # Forward pass
                score_flat = self.model(image1_flat, image2_flat)  # logits, shape: [batch_size * num_crops, 1]
                
                # Reshape and aggregate logits by mean
                score = score_flat.view(batch_size, num_crops, -1)  # [batch_size, 9, 1]
                score_agg = score.mean(dim=1)                        # [batch_size, 1]
                
                target = (target == 1).float().unsqueeze(1)         # [batch_size, 1]
                
                # Convert aggregated logits to probabilities
                prob = torch.sigmoid(score_agg)
                pred = (prob >= 0.5).int()

                # Convert to numpy for processing
                pred_numpy = pred.view(-1).cpu().numpy()
                prob_numpy = prob.view(-1).cpu().numpy()
                target_numpy = target.view(-1).cpu().numpy()
                
                # Store results for each sample in the batch
                for i in range(batch_size):
                    all_preds.append(pred_numpy[i])
                    all_probs.append(prob_numpy[i])
                    all_labels.append(target_numpy[i])
                    
                    # Store IDs if available
                    if id1_batch is not None and id2_batch is not None:
                        all_ids.append((id1_batch[i], id2_batch[i]))
                    else:
                        all_ids.append((f"unknown_{len(all_ids)}_1", f"unknown_{len(all_ids)}_2"))

        return all_preds, all_probs, all_labels, all_ids

    def inference_collage_averaging_asymmetry_averaging(self, data_loader, type):
        """Combine collage averaging and asymmetry averaging for collage inputs.

        This method performs A vs B and B vs A per-crop inference, aggregates
        each direction over crops and then combines both directions via the
        asymmetry averaging heuristic.

        Args:
            data_loader (Iterable): Batched collage data.
            type (str): Dataset name used for progress output and filenames.

        Returns:
            tuple: (all_preds, all_probs, all_labels, all_ids)
        """
        all_preds, all_labels, all_probs = [], [], []
        all_ids = []  # Store (id1, id2) pairs for each sample

        with torch.no_grad():
            for image1, image2, target, id1_batch, id2_batch in tqdm(data_loader, desc=f"Validation on {type} set"):
                image1, image2, target = move_to_device(image1, image2, target, device=self.device)
                
                batch_size, num_crops, C, H, W = image1.shape
                
                # Flatten batch and crops
                image1_flat = image1.view(batch_size * num_crops, C, H, W)
                image2_flat = image2.view(batch_size * num_crops, C, H, W)
                
                # Forward pass
                score_flat_1 = self.model(image1_flat, image2_flat)  # logits, shape: [batch_size * num_crops, 1]
                score_flat_2 = self.model(image2_flat, image1_flat)
                
                # Reshape and aggregate logits by mean
                score_1 = score_flat_1.view(batch_size, num_crops, -1)  # [batch_size, 9, 1]
                score_agg_1 = score_1.mean(dim=1)                        # [batch_size, 1]
                score_2 = score_flat_2.view(batch_size, num_crops, -1)
                score_agg_2 = score_2.mean(dim=1)
                
                target = (target == 1).float().unsqueeze(1)         # [batch_size, 1]
                
                # Convert aggregated logits to probabilities
                prob1 = torch.sigmoid(score_agg_1)
                prob2 = torch.sigmoid(score_agg_2)
                prob = 0.5 * (prob1 + (1-prob2))
                pred = (prob >= 0.5).int()

                # Convert to numpy for processing
                pred_numpy = pred.view(-1).cpu().numpy()
                prob_numpy = prob.view(-1).cpu().numpy()
                target_numpy = target.view(-1).cpu().numpy()
                
                # Store results for each sample in the batch
                for i in range(batch_size):
                    all_preds.append(pred_numpy[i])
                    all_probs.append(prob_numpy[i])
                    all_labels.append(target_numpy[i])
                    
                    # Store IDs if available
                    if id1_batch is not None and id2_batch is not None:
                        all_ids.append((id1_batch[i], id2_batch[i]))
                    else:
                        all_ids.append((f"unknown_{len(all_ids)}_1", f"unknown_{len(all_ids)}_2"))

        return all_preds, all_probs, all_labels, all_ids

    
#-----------------------------------------------EVALUATE METHOD (ENTRY)----------------------------------------------------
    def evaluate(self, data_loader, dataset, evaluation_method):
        """Entry point to evaluate a model using a chosen evaluation method.

        The method prepares the model (device/checkpoint), runs the selected
        inference routine(s) and computes/logs metrics and predictions. The
        supported ``evaluation_method`` values control which inference
        variants are executed (baseline/collage with different aggregation
        and symmetry heuristics).

        Args:
            data_loader (Iterable): DataLoader for the dataset to evaluate.
            dataset (str): Dataset name used for progress display and output files.
            evaluation_method (str): One of the supported evaluation method
                identifiers (e.g., "baseline", "collage", "all_baseline", ...).

        Returns:
            str: Always returns "Success!" on completion.

        Raises:
            ValueError: If an unsupported ``evaluation_method`` is provided.
        """

        try:
            self.setup_model()
        except Exception as e:
            print(f"Error setting up the model: {e}")
            raise

        self.model.eval()

        if evaluation_method == "all_baseline":
            all_preds, all_probs, all_labels, all_ids = self.inference_baseline(data_loader, dataset)
            accuracy = self.compute_metrics(all_labels, all_preds)
            fpr, tpr, roc_auc = self.compute_roc(all_labels, all_probs)
            metrics = {"method": "baseline", "accuracy": accuracy, "roc_auc": roc_auc}
            self.log_metrics_to_csv(metrics, dataset)
            df = pd.DataFrame({
                'id1': [ids[0] for ids in all_ids],
                'id2': [ids[1] for ids in all_ids],
                'pred': all_preds,
                'prob': all_probs,
                'label': all_labels
            })
            df.to_csv(os.path.join(RESULTS_DIR, f"{self.model_abbr}_{dataset}_predictions_baseline.csv"), index=False)

            all_preds, all_probs, all_labels, all_ids = self.inference_baseline_assymmetry_averaging(data_loader, dataset)
            accuracy = self.compute_metrics(all_labels, all_preds)
            fpr, tpr, roc_auc = self.compute_roc(all_labels, all_probs)
            metrics = {"method": "baseline_assymetry_averaging", "accuracy": accuracy, "roc_auc": roc_auc}
            self.log_metrics_to_csv(metrics, dataset)
            df = pd.DataFrame({
                'id1': [ids[0] for ids in all_ids],
                'id2': [ids[1] for ids in all_ids],
                'pred': all_preds,
                'prob': all_probs,
                'label': all_labels
            })
            df.to_csv(os.path.join(RESULTS_DIR, f"{self.model_abbr}_{dataset}_predictions_baseline_assymetry_averaging.csv"), index=False)


        elif evaluation_method == "baseline":
            all_preds, all_probs, all_labels, all_ids = self.inference_baseline(data_loader, dataset)
            accuracy = self.compute_metrics(all_labels, all_preds)
            fpr, tpr, roc_auc = self.compute_roc(all_labels, all_probs)
            metrics = {"method": evaluation_method, "accuracy": accuracy, "roc_auc": roc_auc}
            self.log_metrics_to_csv(metrics, dataset)
            df = pd.DataFrame({
                'id1': [ids[0] for ids in all_ids],
                'id2': [ids[1] for ids in all_ids],
                'pred': all_preds,
                'prob': all_probs,
                'label': all_labels
            })
            df.to_csv(os.path.join(RESULTS_DIR, f"{self.model_abbr}_{dataset}_predictions_{evaluation_method}.csv"), index=False)


        elif evaluation_method == "baseline_asymmetry_averaging":
            all_preds, all_probs, all_labels, all_ids = self.inference_baseline_assymmetry_averaging(data_loader, dataset)
            accuracy = self.compute_metrics(all_labels, all_preds)
            fpr, tpr, roc_auc = self.compute_roc(all_labels, all_probs)
            metrics = {"method": evaluation_method, "accuracy": accuracy, "roc_auc": roc_auc}
            self.log_metrics_to_csv(metrics, dataset)
            df = pd.DataFrame({
                'id1': [ids[0] for ids in all_ids],
                'id2': [ids[1] for ids in all_ids],
                'pred': all_preds,
                'prob': all_probs,
                'label': all_labels
            })
            df.to_csv(os.path.join(RESULTS_DIR, f"{self.model_abbr}_{dataset}_predictions_{evaluation_method}.csv"), index=False)


        elif evaluation_method == "all_collage":
            all_preds, all_probs, all_labels, all_ids = self.inference_collage(data_loader, dataset)
            accuracy = self.compute_metrics(all_labels, all_preds)
            fpr, tpr, roc_auc = self.compute_roc(all_labels, all_probs)
            metrics = {"method": "collage", "accuracy": accuracy, "roc_auc": roc_auc}
            self.log_metrics_to_csv(metrics, dataset)
            
            all_preds, all_probs, all_labels, all_ids = self.inference_collage_averaging(data_loader, dataset)
            accuracy = self.compute_metrics(all_labels, all_preds)
            fpr, tpr, roc_auc = self.compute_roc(all_labels, all_probs)
            metrics = {"method": "collage_averaging", "accuracy": accuracy, "roc_auc": roc_auc}
            self.log_metrics_to_csv(metrics, dataset)
            df = pd.DataFrame({
                'id1': [ids[0] for ids in all_ids],
                'id2': [ids[1] for ids in all_ids],
                'pred': all_preds,
                'prob': all_probs,
                'label': all_labels
            })
            df.to_csv(os.path.join(RESULTS_DIR, f"{self.model_abbr}_{dataset}_predictions_collage_averaging.csv"), index=False)

            all_preds, all_probs, all_labels, all_ids = self.inference_collage_asymmetry_averaging(data_loader, dataset)
            accuracy = self.compute_metrics(all_labels, all_preds)
            fpr, tpr, roc_auc = self.compute_roc(all_labels, all_probs)
            metrics = {"method": "assymetry_averaging", "accuracy": accuracy, "roc_auc": roc_auc}
            self.log_metrics_to_csv(metrics, dataset)
            
            all_preds, all_probs, all_labels, all_ids = self.inference_collage_averaging_asymmetry_averaging(data_loader, dataset)
            accuracy = self.compute_metrics(all_labels, all_preds)
            fpr, tpr, roc_auc = self.compute_roc(all_labels, all_probs)
            metrics = {"method": "collage_averaging_assymetry_averaging", "accuracy": accuracy, "roc_auc": roc_auc}
            self.log_metrics_to_csv(metrics, dataset)
            df = pd.DataFrame({
                'id1': [ids[0] for ids in all_ids],
                'id2': [ids[1] for ids in all_ids],
                'pred': all_preds,
                'prob': all_probs,
                'label': all_labels
            })
            df.to_csv(os.path.join(RESULTS_DIR, f"{self.model_abbr}_{dataset}_predictions_collage_averaging_assymetry_averaging.csv"), index=False)

        elif evaluation_method == "collage":
            all_preds, all_probs, all_labels, all_ids = self.inference_collage(data_loader, dataset)
            accuracy = self.compute_metrics(all_labels, all_preds)
            fpr, tpr, roc_auc = self.compute_roc(all_labels, all_probs)
            metrics = {"method": evaluation_method, "accuracy": accuracy, "roc_auc": roc_auc}
            self.log_metrics_to_csv(metrics, dataset)

        elif evaluation_method == "collage_averaging":
            all_preds, all_probs, all_labels, all_ids = self.inference_collage_averaging(data_loader, dataset)
            accuracy = self.compute_metrics(all_labels, all_preds)
            fpr, tpr, roc_auc = self.compute_roc(all_labels, all_probs)
            metrics = {"method": evaluation_method, "accuracy": accuracy, "roc_auc": roc_auc}
            self.log_metrics_to_csv(metrics, dataset)
            df = pd.DataFrame({
                'id1': [ids[0] for ids in all_ids],
                'id2': [ids[1] for ids in all_ids],
                'pred': all_preds,
                'prob': all_probs,
                'label': all_labels
            })
            df.to_csv(os.path.join(RESULTS_DIR, f"{self.model_abbr}_{dataset}_predictions_{evaluation_method}.csv"), index=False)

        elif evaluation_method == "collage_asymmetry_averaging":
            all_preds, all_probs, all_labels, all_ids = self.inference_collage_asymmetry_averaging(data_loader, dataset)
            accuracy = self.compute_metrics(all_labels, all_preds)
            fpr, tpr, roc_auc = self.compute_roc(all_labels, all_probs)
            metrics = {"method": evaluation_method, "accuracy": accuracy, "roc_auc": roc_auc}
            self.log_metrics_to_csv(metrics, dataset)
            
        elif evaluation_method == "collage_averaging_assymetry_averaging":
            all_preds, all_probs, all_labels, all_ids = self.inference_collage_averaging_asymmetry_averaging(data_loader, dataset)
            accuracy = self.compute_metrics(all_labels, all_preds)
            fpr, tpr, roc_auc = self.compute_roc(all_labels, all_probs)
            metrics = {"method": evaluation_method, "accuracy": accuracy, "roc_auc": roc_auc}
            self.log_metrics_to_csv(metrics, dataset)
            df = pd.DataFrame({
                'id1': [ids[0] for ids in all_ids],
                'id2': [ids[1] for ids in all_ids],
                'pred': all_preds,
                'prob': all_probs,
                'label': all_labels
            })
            df.to_csv(os.path.join(RESULTS_DIR, f"{self.model_abbr}_{dataset}_predictions_{evaluation_method}.csv"), index=False)


        else:
            raise ValueError("Please use a valid evaluation_method (all_baseline, baseline, baseline_symmetry_averaging, all_collage, collage, collage_crop_aggregation, collage_symmetry_averaging, collage_crop_aggregation_symmetry_averaging)!")

        return "Success!"
    
    
    

    