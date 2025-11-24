import pandas as pd
import numpy as np
import pymc as pm
import os

from src.config.paths import ORIGINAL_DIR


class Ranker:
    def __init__(self, comparisons_file, id1, id2, label, save_path):
        self.comparisons_file = comparisons_file
        self.save_path = save_path
        self.id1 = id1
        self.id2 = id2
        self.label = label

    def load_comparisons_from_csv(self):
        """
        Read pairwise comparisons from the configured CSV file and normalize
        them to an internal representation.

        Behavior:
          - Integer labels (e.g. -1, 0, 1) are interpreted as hard outcomes
            and converted to (winner, loser) tuples.
          - Float labels in [0, 1] are treated as probabilities and mapped to
            weighted comparisons (winner, loser, weight) with weight in [0,1]
            encoding confidence.

        Returns:
            tuple: (comparisons, all_ids, use_weights)
                - comparisons: list of (winner, loser) or (winner, loser, weight)
                - all_ids: set of all unique item ids found in the CSV
                - use_weights: bool indicating whether returned comparisons include weights

        Raises:
            ValueError: if an integer label outside the expected set is encountered.
        """
        df = pd.read_csv(self.comparisons_file)
        df[self.id1] = df[self.id1].astype(int)
        df[self.id2] = df[self.id2].astype(int)
        
        comparisons = []
        all_ids = set()
        use_weights = False

        for _, row in df.iterrows():
            i, j, label = row[self.id1], row[self.id2], row[self.label]
            all_ids.update([i, j])

            # Handle both integer labels (for gold standard) and float probabilities (for model)
            if isinstance(label, (int, np.integer)):
                # Integer labels: 1, 0, -1 (no weights)
                if label == 1:
                    comparisons.append((i, j))
                elif label == -1:
                    comparisons.append((j, i))
                elif label == 0: 
                    comparisons.append((j, i))
                else:
                    raise ValueError(f"Unsupported integer label {label}; must be 1 or -1")
            else:
                # Float probabilities: use actual probability values as weights
                use_weights = True
                label = float(label)
                if label > 0.5:
                    # i beats j with confidence (probability - 0.5) * 2
                    # This maps 0.5-1.0 to 0.0-1.0 confidence
                    weight = (label - 0.5) * 2
                    comparisons.append((i, j, weight))
                elif label < 0.5:
                    # j beats i with confidence (0.5 - probability) * 2  
                    # This maps 0.0-0.5 to 1.0-0.0 confidence
                    weight = (0.5 - label) * 2
                    comparisons.append((j, i, weight))
                else:
                    # Skip exact ties (probability = 0.5)
                    continue
            
        return comparisons, all_ids, use_weights


    def normalize_ids(self, comparisons, all_ids):
        """
        Map arbitrary item IDs to contiguous integer indices in [0, n_items).

        This returns a list of comparisons where each original id is replaced
        by an integer index suitable for array indexing in the model, and a
        dictionary mapping original id -> index.

        Supports both unweighted comparisons (winner, loser) and weighted
        comparisons (winner, loser, weight).

        Args:
            comparisons (list): List of comparisons using original ids.
            all_ids (set): Set of all unique ids present in comparisons.

        Returns:
            tuple: (normalized_comparisons, id_to_idx)
        """
        id_list = sorted(all_ids)
        id_to_idx = {id_: idx for idx, id_ in enumerate(id_list)}
        
        # Check if comparisons include weights
        if len(comparisons) > 0 and len(comparisons[0]) == 3:
            # Format: (winner, loser, weight)
            normalized_comparisons = [(id_to_idx[i], id_to_idx[j], w) for (i, j, w) in comparisons]
        else:
            # Format: (winner, loser)
            normalized_comparisons = [(id_to_idx[i], id_to_idx[j]) for (i, j) in comparisons]
            
        return normalized_comparisons, id_to_idx


    def bayesian_bradley_terry(self, comparisons, n_items, use_weights=False, draws=2000, tune=1000, random_seed=42):
        """
        Fit a Bayesian Bradley–Terry model with PyMC to estimate latent skills.

        The model places a Normal(0,1) prior on each item's skill and models
        each observed comparison as Bernoulli(sigmoid(skill_winner - skill_loser)).
        When ``use_weights`` is True, comparisons must be tuples
        (winner, loser, weight) and weights are added to the log-likelihood
        through a potential term to modulate influence.

        Args:
            comparisons (list): Normalized comparisons (indices), either
                (winner, loser) or (winner, loser, weight).
            n_items (int): Number of distinct items.
            use_weights (bool): Whether comparisons include a weight value.
            draws (int): Number of posterior draws to collect.
            tune (int): Number of tuning steps for the sampler.
            random_seed (int): RNG seed for reproducibility.

        Returns:
            InferenceData: Posterior samples containing the ``skill`` parameter.
        """
        if use_weights:
            # Extract winners, losers, and weights
            winners = np.array([comp[0] for comp in comparisons])
            losers = np.array([comp[1] for comp in comparisons])
            weights = np.array([comp[2] for comp in comparisons])
        else:
            # Standard binary comparisons
            winners = np.array([i for i, j in comparisons])
            losers = np.array([j for i, j in comparisons])
            weights = np.ones(len(winners))  # All weights = 1

        with pm.Model() as model:
            skill = pm.Normal("skill", mu=0, sigma=1, shape=n_items)
            
            # Probability that winner beats loser
            p = pm.math.sigmoid(skill[winners] - skill[losers])
            
            if use_weights:
                # Weighted likelihood - higher weights give more influence
                # Use weights to adjust the "effective number of observations"
                outcome = pm.Bernoulli("outcome", p, observed=np.ones(len(winners)))
                # Add weight factor to the log-likelihood
                pm.Potential("weight_factor", pm.math.sum(weights * pm.math.log(p)))
            else:
                # Standard unweighted likelihood
                outcome = pm.Bernoulli("outcome", p, observed=np.ones(len(winners)))

            trace = pm.sample(
                draws=draws, 
                tune=tune,
                chains=4,
                random_seed=random_seed,
                idata_backend="numpyro",  
            )

        return trace
    
    def merge_with_additional_information(self, ranking):
        """
        Merge the ranking (best-to-worst list of sample ids) with metadata and
        save the result to ``self.save_path``.

        Expects a metadata CSV at ``{ORIGINAL_DIR}/xai_dataset.csv``
        keyed by ``sample_id``. The function retains a selected set of
        columns and writes the merged table to disk.

        Args:
            ranking (list): Ordered list of sample ids (best first).

        Returns:
            pandas.DataFrame: The merged DataFrame written to ``self.save_path``.
        """

        rank_df = pd.DataFrame({'sample_id': ranking})
        rank_df['rank'] = rank_df.index + 1

        all_data_df = pd.read_csv(os.path.join(ORIGINAL_DIR, "/xai_dataset.csv"))

        merged_df = pd.merge(rank_df, all_data_df, on='sample_id', how='left')

        columns_to_keep = ['sample_id', 'rank', 'recording_dir_name', 'text', 'reference_sentence_id', 'is_written_in_pure_cursive', 'is_stroke_thin', 'contains_typo', 'contains_correction']
        final_df = merged_df[columns_to_keep]

        final_df.to_csv(self.save_path, index=False)

        return final_df


    def rank(self):
        """
        Execute the full ranking pipeline and return results.

        Steps performed:
          1. Load comparisons from CSV.
          2. Normalize item ids to integer indices.
          3. Fit the Bayesian Bradley–Terry model.
          4. Compute posterior mean skills and produce a ranking.
          5. Optionally merge with metadata and save to CSV.

        Returns:
            tuple: (final_ranking, theta, all_ids, trace)
        """
        result = self.load_comparisons_from_csv()
        comparisons, all_ids, use_weights = result
        
        normalized_comparisons, id_to_idx = self.normalize_ids(comparisons, all_ids)

        n_items = len(id_to_idx)
        
        # Pass use_weights flag to Bradley-Terry model
        trace = self.bayesian_bradley_terry(normalized_comparisons, n_items, use_weights=use_weights)

        # Posterior mean of skill parameters: theta[i] is the average (expected) skill of item i according to the posterior
        theta = trace.posterior["skill"].mean(dim=("chain", "draw")).values

        # Sorts the item indices in descending order of skill (highest skill first).
        ranking = sorted(range(n_items), key=lambda i: theta[i], reverse=True)

        # Converts the ranked indices back to the original item IDs.
        idx_to_id = {idx: id_ for id_, idx in id_to_idx.items()}
        final_ranking = [idx_to_id[i] for i in ranking]

        print("Ranking (best to worst):", final_ranking)
        print("Scores:", theta)

        if self.save_path:
            self.merge_with_additional_information(final_ranking)
        
        return final_ranking, theta, all_ids, trace

