#!/usr/bin/env python3
"""
Scoring Script 

Infer a relative legibility score using a binary prediction model and ranking. 

Usage: 
    python scoring_main.py --reference-samples-count <reference_samples_count> --ratings-count <ratings_count>

Example:
    python scoring_main.py --reference-samples-count 100 --ratings-count 5

Notes:
    - This script currently employs one specific experimental configuration (collages_resnet_small_crops_224) 
        using the respective trained model. If another model shall be used, the code must be changed accordingly.
    - The code samples reference and input samples from a ground-truth ranking of samples, runs model inference to 
        generate pairwise predictions for one input sample, builds a global ranking of reference samples
        including the input sample, and derives a numeric score for the input sample using a scoring function. 
    - This script impelements the asymmetry_averaging (AA) inference method with collage-preprocessed images.
    - One score is inferred for each input sample separately.
    - The reference and input samples are chosen randomly, but uniformly distributed over the ranking sections. 
"""
import numpy as np
import os
import pandas as pd
import torch
import scipy.stats as stats
import math
import argparse
import sys

import src.ranking.create_ranking as create_ranking
from src.data.loader import ImagePairDatasetValTest, load_image_by_id
import src.models.preprocessor as preprocessor
from src.models.feature_extractor import FeatureExtractorResNet
from src.models.siamese_network import SiameseNeuralNetwork
from src.models.classifier import ClassifierResNet
from src.config.paths import SCORING_DIR, IMAGES_COLLAGES, MODELS_DIR, ANNOTATIONS_DIR



def move_to_device(*args, device=None):
    """Move tensors/objects to a device if possible. Returns a tuple with moved objects.

    If device is None the original objects are returned unchanged.
    """
    moved = []
    for x in args:
        try:
            if device is not None and hasattr(x, "to"):
                moved.append(x.to(device))
            else:
                moved.append(x)
        except Exception:
            moved.append(x)
    return tuple(moved)


class Scorer:
    """
    Run inference with a trained model and convert pairwise predictions into rank-based scores.

    Attributes:
        ratings (int): Number of discrete rating bins used to split CSVs.
        reference_sample_count (int): Number of reference samples to draw.
        input_sample_count (int): Number of input samples to draw.
        model (torch.nn.Module): Loaded siamese model used for inference.
        device (torch.device): Torch device used for inference.
        dataset_reference_samples (Dataset): Validation/reference dataset wrapper.
        dataset_input_samples (Dataset): Test/input dataset wrapper.
    """

    def __init__(self, ratings: int, reference_sample_count: int, input_sample_count: int):
        """
        Args:
            ratings: Number of rating sections to split CSVs into (default 5).
            reference_sample_count: How many reference items to sample (default 100).
            input_sample_count: How many input items to sample (default 100).
        """
        self.ratings = ratings
        self.reference_sample_count = reference_sample_count
        self.input_sample_count = input_sample_count

        # Load and setup model        
        try:
            self.setup_model()
        except Exception as e:
            print(f"Error setting up the model: {e}")
            raise

        self.model.eval()

        # Load data - this needs to be adapted to the specific experimental configuration if a different one is used
        self.transform = preprocessor.get_final_transforms_resnet(224) 
        self.dataset_reference_samples = ImagePairDatasetValTest(
                                annotations_file=os.path.join(ANNOTATIONS_DIR, "samples_val_pairs_with_scores.csv"),
                                root_dir=IMAGES_COLLAGES,
                                transform=self.transform,
                                crop_size="small"
                            )
        self.dataset_input_samples = ImagePairDatasetValTest(
                                annotations_file=os.path.join(ANNOTATIONS_DIR, "samples_test_pairs_with_scores.csv"),
                                root_dir=IMAGES_COLLAGES,
                                transform=self.transform,
                                crop_size="small"
                            )

    def load_model(self):
        """Load model state from a checkpoint file.

        Expects the checkpoint to contain a `model_state_dict` key and an
        `epoch` key. On success the model weights are loaded and
        `self.start_epoch` is set. Any exception will propagate to the
        caller.
        """
        checkpoint = torch.load(self.path_to_model)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.start_epoch = checkpoint["epoch"]

    def setup_model(self):
        """Instantiate the model, move it to the configured device and
        attempt to load pre-trained weights.

        This method sets `self.model`, `self.device`, and `self.path_to_model`.
        It will try to call `self.load_model()`; on failure it falls back to
        leaving the freshly initialized model as-is and sets `self.start_epoch` to 0.
        """
        # Initialize model - this nees to be adapted to the specific experimental configuration if a different one is used
        self.model = SiameseNeuralNetwork(
            FeatureExtractorResNet(), 
            ClassifierResNet(dropout=0.0550)  
        )
        self.path_to_model = os.path.join(MODELS_DIR, "trained", "collages_resnet_small_crops_224_model_best.pt")

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def get_input_samples(self, path_ranking_file, random_state=42):
        """Sample input rows from CSV and return their sample_ids in original CSV order.

        random_state: int or None. When provided, sampling will be reproducible. A different
        per-section seed is generated from this value so sections don't pick identical rows.
        """
        df = pd.read_csv(path_ranking_file)
        sections = np.array_split(df, self.ratings)
        count_rows = int(self.input_sample_count / self.ratings)

        # Randomly select rows from each section. Use per-section seeds when random_state is set
        if random_state is not None:
            rng = np.random.RandomState(random_state)
            seeds = rng.randint(0, 2**31 - 1, size=len(sections))
            selected_rows = [section.sample(n=count_rows, random_state=int(seeds[i])) for i, section in enumerate(sections)]
        else:
            selected_rows = [section.sample(n=count_rows) for section in sections]

        # Concatenate the selected rows into a new DataFrame but keep original indices.
        result_df = pd.concat(selected_rows, ignore_index=False)

        # Sort by original CSV index to preserve the original ordering of rows
        result_df = result_df.sort_index()

        return result_df["sample_id"].tolist()

    def get_reference_samples(self, path_ranking_file, random_state=42):
        """Sample reference rows from CSV and return their sample_ids in original CSV order.

        random_state: int or None. When provided, sampling will be reproducible. A different
        per-section seed is generated from this value so sections don't pick identical rows.
        """
        df = pd.read_csv(path_ranking_file)
        sections = np.array_split(df, self.ratings)
        count_rows = int(self.reference_sample_count / self.ratings)

        # Randomly select rows from each section. Use per-section seeds when random_state is set
        if random_state is not None:
            rng = np.random.RandomState(random_state)
            seeds = rng.randint(0, 2**31 - 1, size=len(sections))
            selected_rows = [section.sample(n=count_rows, random_state=int(seeds[i])) for i, section in enumerate(sections)]
        else:
            selected_rows = [section.sample(n=count_rows) for section in sections]

        # Concatenate the selected rows into a new DataFrame but keep original indices
        result_df = pd.concat(selected_rows, ignore_index=False)

        # Sort by original CSV index to preserve the original ordering of rows
        result_df = result_df.sort_index()

        return result_df["sample_id"].tolist()


    def model_inference(self, reference_sample_ids: list, sample_id: str):
        """Infer pairwise predictions between an input sample and the reference samples.

        The method loads the input sample tensor and each reference tensor,
        moves them to the model device, and performs a forward pass.
        It aggregates predictions across multiple crops (if present) and
        returns a DataFrame with pairwise predictions `pred` for each
        (reference, input) pair.

        Args:
            reference_sample_ids: iterable of reference sample ids to compare.
            sample_id: id of the input sample to score against references.

        Returns:
            pandas.DataFrame with columns `id1`, `id2`, and `pred`.
        """

        # Load the input image from the test dataset
        input_sample_tensor = load_image_by_id(self.dataset_input_samples, sample_id)
        
        all_preds, all_probs = [], []
        all_ids = []  
        
        # Do model inference using collage averaging and symmetry averaging methods
        with torch.no_grad():
            for ref_id in reference_sample_ids:

                # Load the reference image from the validation dataset
                reference_sample_tensor = load_image_by_id(self.dataset_reference_samples, ref_id)

                # move tensors to the selected device
                image1, image2 = move_to_device(reference_sample_tensor, input_sample_tensor, device=self.device)

                if not hasattr(image1, 'shape') or len(image1.shape) != 4:
                    raise ValueError(f"Unexpected image tensor shape for ref_id={ref_id}: {getattr(image1, 'shape', None)}")

                num_crops, C, H, W = image1.shape

                # Forward pass (model expected to accept tensors shaped (num_crops, C, H, W))
                try:
                    score_flat_1 = self.model(image1, image2)
                    score_flat_2 = self.model(image2, image1)
                except Exception as e:
                    raise RuntimeError(f"Model forward failed for ref_id={ref_id} with shapes {image1.shape}, {image2.shape}: {e}") from e

                # Reshape and aggregate logits by mean across crops
                # score_flat_* expected shape: (num_crops, 1) or (num_crops,)
                score_1 = score_flat_1.view(num_crops, -1)
                score_agg_1 = score_1.mean(dim=0)
                score_2 = score_flat_2.view(num_crops, -1)
                score_agg_2 = score_2.mean(dim=0)

                # Convert aggregated logits to probabilities
                prob1 = torch.sigmoid(score_agg_1)
                prob2 = torch.sigmoid(score_agg_2)
                prob = 0.5 * (prob1 + (1 - prob2))
                pred = (prob >= 0.5).int()

                # Convert to numpy for processing 
                pred_numpy = pred.view(-1).cpu().numpy()
                prob_numpy = prob.view(-1).cpu().numpy()

                # Append each element (if multiple) to results lists, duplicating ids accordingly
                for p_val, pr_val in zip(pred_numpy, prob_numpy):
                    all_preds.append(int(p_val))
                    all_probs.append(float(pr_val))
                    all_ids.append((str(ref_id), str(sample_id)))

        # Create and save predictions DataFrame once after inference loop
        df = pd.DataFrame({
            'id1': [ids[0] for ids in all_ids],
            'id2': [ids[1] for ids in all_ids],
            'pred': all_preds
        })

        os.makedirs(SCORING_DIR, exist_ok=True)
        df.to_csv(os.path.join(SCORING_DIR, "model_inference_predictions.csv"), index=False)

        return df
    

    def create_reference_sample_pairs(self, reference_sample_ids: list):
        """Create all ordered pairwise combinations for reference samples.

        Each ordered pair (i, j) receives a synthetic `pred` label: 1 if the
        first element precedes the second in the list, 0 otherwise. Pairs with
        identical indices are omitted.

        Args:
            reference_sample_ids: iterable of reference sample ids.

        Returns:
            pandas.DataFrame with columns `id1`, `id2`, `pred`.
        """

        rows = []
        n = len(reference_sample_ids)
        for i, id1 in enumerate(reference_sample_ids):
            for j, id2 in enumerate(reference_sample_ids):
                if i == j:
                    continue
                pred = 1 if i < j else 0
                rows.append({"id1": id1, "id2": id2, "pred": pred})

        df_pairs = pd.DataFrame(rows)

        return df_pairs


    def create_ranking(self, file_path: str):
        """
        Create a global ranking from a CSV file of pairwise comparisons.

        Args:
            file_path: Path to a CSV file containing columns `id1`, `id2`, `pred`.

        Returns:
            A ranking object (format provided by `create_ranking.Ranker.rank`).
        """
        ranker = create_ranking.Ranker(
            comparisons_file=file_path,
            id1="id1",
            id2="id2", 
            label="pred",
            save_path=None
        )
        ranking, theta, all_ids, trace = ranker.rank()

        return ranking

    def get_score_from_ranking(self, ranking: list, sample_id):
        """Derive a score from a ranking for a single sample.

        The score is computed as: (position * ratings) / max(1, n)
        where `position` is the index of the sample in the ranking (0-based)
        and `n` is the length of the ranking. This maps better positions to
        lower numeric scores.

        Args:
            ranking: Iterable or array-like representing the ranking order of ids.
            sample_id: The sample id to locate within the ranking.

        Returns:
            float: Derived score for the input sample.
        """
        
        try:
            if hasattr(ranking, "tolist"):
                ranking_list = ranking.tolist()
            else:
                ranking_list = list(ranking)
        except Exception:
            ranking_list = [ranking]

        # Try to convert entries to ints (handles np.int64, np.array scalars, etc.)
        normalized = []
        for x in ranking_list:
            try:
                normalized.append(int(x))
            except Exception:
                normalized.append(x)

        # Normalize input id
        try:
            input_id = int(sample_id)
        except Exception:
            input_id = sample_id


        # Find index using normalized ints first, then fall back to string match
        if input_id in normalized:
            position_input_sample = normalized.index(input_id)
        else:
            normalized_str = [str(x) for x in normalized]
            if str(sample_id) in normalized_str:
                position_input_sample = normalized_str.index(str(sample_id))
            else:
                raise ValueError(f"Input sample id {sample_id} not found in ranking (checked normalized types)")

        score = (position_input_sample * self.ratings) / max(1, len(normalized))
        return score
    
    def evaluate_scoring(self, input_file: str = "final_scores.csv"):
        """Compute per-class and overall Spearman correlations between
        predicted and gold scores and save the results to CSV.

        Args:
            input_file: CSV filename (inside `SCORING_DIR`) containing at least
                        `score_gold` and `score_pred` columns.
        """

        df = pd.read_csv(os.path.join(SCORING_DIR, input_file))
        df['score_gold_classes'] = df['score_gold'].apply(math.floor)

        classes_list = []
        spearman_list = []
        p_list = []
        for i in range(0, 5):
            filtered_df = df[df['score_gold_classes'] == i]
            preds_array = filtered_df['score_pred'].values
            golds_array = filtered_df['score_gold'].values

            rho, p_value = stats.spearmanr(golds_array, preds_array)

            classes_list.append(f"{i}")
            spearman_list.append(rho)
            p_list.append(p_value)

        preds_array = df['score_pred'].values
        golds_array = df['score_gold'].values
        rho, p_value = stats.spearmanr(golds_array, preds_array)
        
        classes_list.append("overall")
        spearman_list.append(rho)
        p_list.append(p_value)

        df_results = pd.DataFrame({
                'class': classes_list,
                'spearman_rho': spearman_list,
                'p_value': p_list
            })
        os.makedirs(SCORING_DIR, exist_ok=True)
        df_results.to_csv(os.path.join(SCORING_DIR, "scores_evaluation.csv"), index=False)

    def main(self):
        """Main function to orchestrate the scoring.

        This method performs the following steps:
        - sample input and reference items from CSV files
        - create pairwise comparisons and run model inference
        - build a ranking for each input and derive a predicted score
        - compare predicted scores to gold scores and save results
        """
        # Sample instances from the test dataset as input samples
        input_sample_ids = self.get_input_samples(os.path.join(ANNOTATIONS_DIR, "samples_train_ranking.csv"))

        # Sample instances from the validation dataset as reference samples
        reference_sample_ids = self.get_reference_samples(os.path.join(ANNOTATIONS_DIR, "samples_val_ranking.csv"))

        # Create pairs with gold labels for reference samples
        reference_comparisons = self.create_reference_sample_pairs(reference_sample_ids)

        score_predicted_list = []
        score_gold_list = []
        sample_id_list = []

        for sample_id in input_sample_ids:

            # Get predictions for pairwise comparisons between each reference sample and input sample
            prediction_df = self.model_inference(reference_sample_ids, sample_id)

            # Concatenate the predicted pairwise comparisons with the comparisons coming from the reference samples
            all_pairs = pd.concat([prediction_df, reference_comparisons], ignore_index=True)
            os.makedirs(SCORING_DIR, exist_ok=True)
            all_pairs.to_csv(os.path.join(SCORING_DIR, "pairwise_comparisons.csv"), index=False)

            # Create a ranking containing the reference samples and the input sample and predict a score based on that
            global_ranking = self.create_ranking(os.path.join(SCORING_DIR, "pairwise_comparisons.csv"))
            score_predicted = self.get_score_from_ranking(global_ranking, sample_id)

            # Get the gold score based on the test ranking
            df = pd.read_csv(os.path.join(ANNOTATIONS_DIR, "samples_train_ranking.csv"))
            gold_ranking = df["sample_id"].tolist()
            score_gold = self.get_score_from_ranking(gold_ranking, sample_id)

            # Save predicted score, gold score and sample ID in lists
            score_predicted_list.append(round(score_predicted, 4))
            score_gold_list.append(round(score_gold, 4))
            sample_id_list.append(sample_id)

            print(f"Successfully predicted score {score_predicted:.4f} for sampleId {sample_id}! The true score is {score_gold:.4f}.")

        df = pd.DataFrame({
            'sampleId': sample_id_list,
            'score_gold': score_gold_list,
            'score_pred': score_predicted_list
        })
        os.makedirs(SCORING_DIR, exist_ok=True)
        df.to_csv(os.path.join(SCORING_DIR, "final_scores.csv"), index=False)

        self.evaluate_scoring()





if __name__ == "__main__":
    
    try:
        parser = argparse.ArgumentParser(
            description="Infer scoring from ranking"
        )

        parser.add_argument(
            "--reference-samples-count",
            type=int,
            default=100,
            help="Set the amount of reference samples"
        )

        parser.add_argument(
            "--ratings-count",
            type=int,
            default=5,
            help="Set the amount of ratings"
        )

        args = parser.parse_args()

        scorer = Scorer(ratings=args.ratings_count, reference_sample_count=args.reference_samples_count, input_sample_count=100)
        scorer.main()

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)