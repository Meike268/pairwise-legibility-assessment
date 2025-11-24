from collections import defaultdict
import itertools
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import random
from src.config.paths import ANNOTATIONS_DIR


def split_by_recording_dirs(path_to_dataset: str, train_ratio: float, val_ratio: float, test_ratio: float, seed: int = 42):
    """
    Split a dataset CSV into train/validation/test sets by recording directory.

    This function ensures that all samples originating from the same
    ``recording_dir_name`` are assigned to the same split, avoiding leakage
    between splits. The split is performed randomly using the provided seed.

    Args:
        path_to_dataset (str): Path to the full dataset CSV file. The file
            must contain a ``recording_dir_name`` column.
        train_ratio (float): Fraction of unique recording directories to use
            for training.
        val_ratio (float): Fraction for validation.
        test_ratio (float): Fraction for testing.
        seed (int): Random seed used for reproducible splits (default 42).

    Returns:
        tuple: Paths to the created CSV files for (train_path, val_path, test_path).

    Raises:
        ValueError: If the provided ratios do not sum to a positive value or are invalid.
    """

    df = pd.read_csv(path_to_dataset)

    # Get unique group keys
    unique_dirs = df['recording_dir_name'].unique()

    # Split the directory names randomly
    train_dirs, valtest_dirs = train_test_split(
        unique_dirs,
        test_size=(val_ratio + test_ratio),
        random_state=seed
    )

    # Further split valtest_dirs into validation and test sets
    val_dirs, test_dirs = train_test_split(
        valtest_dirs,
        test_size=test_ratio / (val_ratio + test_ratio),
        random_state=seed
    )

    # Filter original DataFrame based on group keys
    train_df = df[df['recording_dir_name'].isin(train_dirs)]
    val_df = df[df['recording_dir_name'].isin(val_dirs)]
    test_df = df[df['recording_dir_name'].isin(test_dirs)]

    # Create splits directory under centralized data directory
    val_path = os.path.join(ANNOTATIONS_DIR, "samples_val.csv")
    train_path = os.path.join(ANNOTATIONS_DIR, "samples_train.csv")
    test_path = os.path.join(ANNOTATIONS_DIR, "samples_test.csv")

    # Save the filtered DataFrames to CSV files
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    val_df.to_csv(val_path, index=False)


    return train_path, val_path, test_path

def generate_unique_random_pairs(path_to_dataset, output_path, num_pairs=10):
    """
    Produce a CSV of unique unordered random pairs for each sample.

    For every sample in the input dataset this function selects up to
    ``num_pairs`` distinct opponents sampled at random. Self-pairings are
    excluded. Pairs are treated as unordered (``(A,B)`` == ``(B,A)``) and
    duplicates across the entire output are prevented.

    Args:
        path_to_dataset (str): Path to a CSV file containing a ``sample_id`` column.
        output_path (str): Path to the CSV file to create with pair columns
            ``sample1_id`` and ``sample2_id``.
        num_pairs (int): Maximum number of opponents to select per sample.

    Returns:
        None. The resulting pairs are saved to ``output_path``.
    """
    df = pd.read_csv(path_to_dataset)
    sample_ids = df['sample_id'].tolist()
    seen_pairs = set()
    pair_list = []

    for sample_id in sample_ids:
        available_ids = set(sample_ids) - {sample_id}
        selected_ids = set()

        while len(selected_ids) < num_pairs and available_ids:
            candidate = random.choice(list(available_ids))
            # Create a sorted tuple to ensure (A, B) == (B, A)
            pair = tuple(sorted((sample_id, candidate)))
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                selected_ids.add(candidate)
                pair_list.append({'sample1_id': pair[0], 'sample2_id': pair[1]})
            available_ids -= {candidate}
    
    result_df = pd.DataFrame(pair_list)
    result_df.to_csv(output_path, index=False)
    print(f"Saved unique pairs from {path_to_dataset}to {output_path}")


def concatenate_csv_files(input_files, output_file):
    """
    Concatenate multiple CSV files into a single CSV file preserving row order.

    Args:
        input_files (Iterable[str]): List of input CSV file paths to concatenate.
        output_file (str): Destination path for the concatenated CSV.

    Returns:
        None. The concatenated CSV is written to ``output_file``.
    """
    dataframes = []
    for file in input_files:
        df = pd.read_csv(file)
        dataframes.append(df)

    concatenated_df = pd.concat(dataframes, ignore_index=True)
    concatenated_df.to_csv(output_file, index=False)
    print(f"Concatenated {len(input_files)} files into {output_file}")

