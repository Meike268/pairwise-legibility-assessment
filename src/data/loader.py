import sys
sys.path.append('..')

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import pandas as pd
import glob
import torch
import random
import math

from src.config.paths import ANNOTATIONS_DIR



class ImagePairDataset(Dataset):
    """
    Creates a Dataset of images pairs and their annotation.
    Transforms the images according to a specified transformation.
    """

    def __init__(self, annotations_file: str, root_dir: str, transform, crop_size: str):
        """
        :param annotations_file: path to csv-file containing the pairwise annotations
        :param root_dir: path to directory containing the preprocessed images
        :param transform: transformation function
        :param crop_size: size of crop ('small', 'large', or None)
        """
        self.pairs_df = pd.read_csv(annotations_file)
        self.root_dir = root_dir
        self.transform = transform
        self.image_index = self._index_images()

        if crop_size == 'small':
            self.crop_size = 800
        elif crop_size == 'large':
            self.crop_size = 1000
        else:
            self.crop_size = None

    def _index_images(self):
        """
        Index all images by ID for fast lookup of filepath.
        """
        index = {}
        pattern = os.path.join(self.root_dir, "**", f"*.png")
        for path in glob.glob(pattern, recursive=True):
            file_id = os.path.splitext(os.path.basename(path))[0]
            index[file_id] = path
        return index

    def __len__(self):
        """Return the number of annotated pairs available in this dataset."""
        return len(self.pairs_df)

    
class ImagePairDatasetBaseline(ImagePairDataset):
    """
    Dataset for baseline images (no additional collage cropping).

    Each item returned by this dataset is a full-image pair and a label:
        (img1, img2, label_tensor, sample_id1_str, sample_id2_str)

    The images are opened and converted to RGB; `self.transform` is applied if present.
    """
    def __init__(self, annotations_file, root_dir, transform, crop_size):
        super().__init__(annotations_file, root_dir, transform, crop_size)
    
    def __getitem__(self, idx):
        """
        Retrieve the pair at index ``idx`` for baseline evaluation/training.

        Args:
            idx (int): index of the row in the annotations CSV.

        Returns:
            tuple: (img1, img2, label_tensor, id1_str, id2_str)
                - img1/img2: PIL image or transformed tensor if `self.transform` is set
                - label_tensor: `torch.LongTensor` representing the pair score
                - id1_str/id2_str: string identifiers of the samples

        Raises:
            FileNotFoundError: if either image id cannot be found in `self.image_index`.
        """
        row = self.pairs_df.iloc[idx]
        id1, id2 = row['sampleId1'], row['sampleId2']
        label = row['score']

        try:
            img1_path = self.image_index[str(id1)]
            img2_path = self.image_index[str(id2)]
        except KeyError as e:
            raise FileNotFoundError(f"Image ID '{e.args[0]}' not found in folder structure.")

        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB") 

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.long), str(id1), str(id2)
    
class ImagePairDatasetTrain(ImagePairDataset):
    def __init__(self, annotations_file, root_dir, transform, crop_size):
        super().__init__(annotations_file, root_dir, transform, crop_size)

    def __getitem__(self, idx):
        """
        Retrieve the pair at index ``idx`` for training.

        Behavior:
            - Loads both images corresponding to the pair.
            - Performs a random square crop of size ``self.crop_size`` on each image.
            - Applies ``self.transform`` to the cropped patches when provided.

        Args:
            idx (int): index of the row in the annotations CSV.

        Returns:
            tuple: (crop1, crop2, label_tensor, id1_str, id2_str)
                - crop1/crop2: transformed crop tensors (or raw PIL crops if no transform)
                - label_tensor: `torch.LongTensor` representing the pair score
                - id strings: string identifiers for the samples

        Raises:
            FileNotFoundError: if either image id cannot be found in `self.image_index`.
            ValueError: if ``self.crop_size`` is larger than the image dimensions.
        """
        row = self.pairs_df.iloc[idx]
        id1, id2 = row['sampleId1'], row['sampleId2']
        label = row['score']

        try:
            img1_path = self.image_index[str(id1)]
            img2_path = self.image_index[str(id2)]
        except KeyError as e:
            raise FileNotFoundError(f"Image ID '{e.args[0]}' not found in folder structure.")

        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")
        
        w, h = img1.size  

        if self.crop_size > h or self.crop_size > w:
            raise ValueError(f"Crop size {self.crop_size} exceeds image dimensions ({h}, {w})")

        # Random crop on PIL image
        left1 = random.randint(0, w - self.crop_size)
        top1 = random.randint(0, h - self.crop_size)
        crop1 = img1.crop((left1, top1, left1 + self.crop_size, top1 + self.crop_size))

        left2 = random.randint(0, w - self.crop_size)
        top2 = random.randint(0, h - self.crop_size)
        crop2 = img2.crop((left2, top2, left2 + self.crop_size, top2 + self.crop_size))

        if self.transform:
            crop1 = self.transform(crop1)
            crop2 = self.transform(crop2)

        return crop1, crop2, torch.tensor(label, dtype=torch.long), str(id1), str(id2)
    
class ImagePairDatasetValTest(ImagePairDataset):
    clock_angles = [math.radians(a) for a in [0, 45, 90, 135, 180, 225, 270, 315]]
    
    """
    Dataset for validation and testing that returns multiple crops per image.

    For each image pair this dataset generates a set of square crops taken from
    the image center and a set of clock-position offsets around the center
    (center + 8 positions). Each item returned is a stacked tensor of crops
    for image1, a stacked tensor of crops for image2, and the label.
    """
    def __init__(self, annotations_file, root_dir, transform, crop_size):
        super().__init__(annotations_file, root_dir, transform, crop_size)

    def __getitem__(self, idx):
        """
        Retrieve the pair at index ``idx`` for validation/testing.

        Behavior:
            - Loads both images.
            - Constructs a list of centers (center + 8 clock positions) and
              extracts square crops of size ``self.crop_size`` at those centers.
            - Applies ``self.transform`` to each crop when provided and returns
              stacked tensors (num_crops x C x H x W).

        Args:
            idx (int): index of the row in the annotations CSV.

        Returns:
            tuple: (crops1_tensor, crops2_tensor, label_tensor, id1_str, id2_str)

        Raises:
            FileNotFoundError: if either image id cannot be found in `self.image_index`.
            ValueError: if ``self.crop_size`` is larger than the image dimensions.
        """
        row = self.pairs_df.iloc[idx]
        id1, id2 = row['sampleId1'], row['sampleId2']
        label = row['score']

        try:
            img1_path = self.image_index[str(id1)]
            img2_path = self.image_index[str(id2)]
        except KeyError as e:
            raise FileNotFoundError(f"Image ID '{e.args[0]}' not found in image index.")

        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

        w, h = img1.size

        if self.crop_size > w or self.crop_size > h:
            raise ValueError(f"Crop size {self.crop_size} exceeds image dimensions ({w}, {h})")

        cx, cy = w // 2, h // 2
        r = int(self.crop_size * 1.5)

        # Define centers: 1 center + 8 clock positions around center
        centers = [(cx, cy)]
        centers.extend((
            (int(cx + r * math.cos(angle)), int(cy - r * math.sin(angle)))
            for angle in self.clock_angles
        ))

        crops1 = self.crop_at_centers(img1, centers, self.crop_size)
        crops2 = self.crop_at_centers(img2, centers, self.crop_size)

        return torch.stack(crops1), torch.stack(crops2), torch.tensor(label, dtype=torch.long), str(id1), str(id2)


    def crop_at_centers(self, image, centers, crop_size):
        """
        Crop square patches centered at specified coordinates.
        If crop exceeds image boundary, adjust the box to fit inside.
        Apply self.transform.
        """
        crops = []
        half = crop_size // 2
        w, h = image.size

        for (x, y) in centers:
            # Calculate crop box boundaries
            left = max(0, x - half)
            top = max(0, y - half)
            right = left + crop_size
            bottom = top + crop_size

            # Adjust crop if it goes beyond right or bottom edge
            if right > w:
                right = w
                left = right - crop_size
            if bottom > h:
                bottom = h
                top = bottom - crop_size

            crop = image.crop((left, top, right, bottom))
            if self.transform:
                crop = self.transform(crop)
            crops.append(crop)

        return crops


def load_pair_by_ids(dataset, id1, id2):
    """
    Helper to load an image pair from a dataset by sample IDs.

    Args:
        dataset: an instance of ImagePairDataset* (e.g. ImagePairDatasetBaseline / Train / ValTest)
        id1: first sample id 
        id2: second sample id 

    Returns:
        img1, img2, label, sample_id1, sample_id2

    Functionality:
        - Searches the dataset.pairs_df for a row where sampleId1==id1 and sampleId2==id2.
        - If not found, searches for the reversed order (sampleId1==id2 and sampleId2==id1).
          In that case, the returned images are reordered so that the first corresponds to the
          requested id1 and the second to id2. The label is adjusted so that it always encodes the relation "does id1 beat id2?":
            * if label is in {-1, 1} it is multiplied by -1 when reversed
            * if label is in {0, 1} it is mapped to 1-label when reversed
        - If no matching row is found, raises ValueError.

    Note: the helper uses the dataset's __getitem__ implementation so transforms/cropping
    will be applied the same way as when used during training/evaluation.
    """
    id1s, id2s = str(id1), str(id2)
    df = dataset.pairs_df

    # Ensure comparison as strings to match how dataset indexes/returns ids
    mask = (df['sampleId1'].astype(str) == id1s) & (df['sampleId2'].astype(str) == id2s)
    if mask.any():
        row_idx = df.index[mask][0]
        img1, img2, label, r1, r2 = dataset[row_idx]
        return img1, img2, label, r1, r2

    # Try reversed ordering
    mask_rev = (df['sampleId1'].astype(str) == id2s) & (df['sampleId2'].astype(str) == id1s)
    if mask_rev.any():
        row_idx = df.index[mask_rev][0]
        img_a, img_b, label, r1, r2 = dataset[row_idx]

        # dataset returned images in order (r1==id2s, r2==id1s)
        # Reorder images so the returned pair corresponds to (id1s, id2s)
        # Also adjust label when it's a simple numeric indicator.
        # Handle torch tensors for label
        try:
            is_tensor = torch.is_tensor(label)
        except Exception:
            is_tensor = False

        raw_label = None
        if is_tensor:
            try:
                raw_label = int(label.item())
            except Exception:
                raw_label = None
        else:
            try:
                raw_label = int(label)
            except Exception:
                raw_label = None

        new_label = label
        if raw_label is not None:
            if raw_label in (-1, 1):
                new_val = -raw_label
            elif raw_label in (0, 1):
                new_val = 1 - raw_label
            else:
                new_val = raw_label

            if is_tensor:
                new_label = torch.tensor(new_val, dtype=label.dtype)
            else:
                new_label = new_val

        # Return images in requested order (id1s, id2s)
        return img_b, img_a, new_label, id1s, id2s

    raise ValueError(f"No pair found for ids {id1s} and {id2s} in dataset annotations.")


def load_image_by_id(dataset, sample_id, deterministic: bool = True, seed: int = 42):
    """
    Helper to load a single image from `dataset` by its sample id and apply the exact same
    preprocessing that the dataset's `__getitem__` would apply for that image.

    Functionality per dataset type:
      - ImagePairDatasetBaseline: open full image, apply `dataset.transform` and return transformed image
      - ImagePairDatasetTrain: perform the same random crop used in `__getitem__`, apply `dataset.transform` and return the cropped, transformed image
      - ImagePairDatasetValTest: produce the same center + clock-position crops, apply transforms and return a tensor of stacked crops (num_crops x C x H x W)

    Args:
        dataset: an instance of one of the ImagePairDataset* classes
        sample_id: id of the sample 
        deterministic: if True, make Train's random crop deterministic using `seed`
        seed: integer seed used when `deterministic=True`

    Returns:
        Depending on dataset: a transformed torch.Tensor (baseline/train) or a stacked tensor of crops (val/test)
    """
    sid = str(sample_id)
    img_path = dataset.image_index.get(sid)
    if img_path is None:
        raise FileNotFoundError(f"Sample ID {sid} not found in dataset image index.")

    img = Image.open(img_path).convert("RGB")

    # Baseline: full image + transform
    if isinstance(dataset, ImagePairDatasetBaseline):
        if dataset.transform:
            return dataset.transform(img)
        return img

    # Train: random crop then transform
    if isinstance(dataset, ImagePairDatasetTrain):
        if dataset.crop_size is None:
            raise ValueError("Dataset crop_size is None; cannot perform train cropping")

        w, h = img.size
        if dataset.crop_size > w or dataset.crop_size > h:
            raise ValueError(f"Crop size {dataset.crop_size} exceeds image dimensions ({w}, {h})")

        # Deterministic behavior for reproducibility
        if deterministic:
            rnd = random.Random(seed)
            left = rnd.randint(0, w - dataset.crop_size)
            top = rnd.randint(0, h - dataset.crop_size)
        else:
            left = random.randint(0, w - dataset.crop_size)
            top = random.randint(0, h - dataset.crop_size)

        crop = img.crop((left, top, left + dataset.crop_size, top + dataset.crop_size))
        if dataset.transform:
            crop = dataset.transform(crop)
        return crop

    # Val/Test: produce multiple center + clock-position crops
    if isinstance(dataset, ImagePairDatasetValTest):
        if dataset.crop_size is None:
            raise ValueError("Dataset crop_size is None; cannot perform val/test cropping")

        w, h = img.size
        if dataset.crop_size > w or dataset.crop_size > h:
            raise ValueError(f"Crop size {dataset.crop_size} exceeds image dimensions ({w}, {h})")

        cx, cy = w // 2, h // 2
        r = int(dataset.crop_size * 1.5)
        centers = [(cx, cy)]
        centers.extend((
            (int(cx + r * math.cos(angle)), int(cy - r * math.sin(angle)))
            for angle in dataset.clock_angles
        ))

        crops = dataset.crop_at_centers(img, centers, dataset.crop_size)
        # crop_at_centers already applies dataset.transform when present and returns a list of tensors
        # Stack into a single tensor (num_crops x C x H x W)
        try:
            return torch.stack(crops)
        except Exception:
            return crops

    



def get_data_loaders(input_directory: str, crop_size: str, transform: object, baseline: bool):
    """
    Helper to get dataloaders for all three datasets (train, val, test).

    Args:
        input_directory: path to directory containing preprocessed images
        crop_size: for the collage preprocessing method - defines the size of crops taken out of collages
        tranform: function defining the transformations that shall be applied to the images (depends on the neural network)
        baseline: defines whether the baseline preprocessing method (True) or the collage preprocessing method (False) should be used 
    
    Returns: train loader, test loader, and val loader
    """

    if baseline:
        train_dataset = ImagePairDatasetBaseline(
            annotations_file=os.path.join(ANNOTATIONS_DIR, f"samples_train_pairs_with_scores.csv"),
            root_dir=input_directory,
            transform=transform,
            crop_size=None
        )
        test_dataset = ImagePairDatasetBaseline(
            annotations_file=os.path.join(ANNOTATIONS_DIR, f"samples_test_pairs_with_scores.csv"),
            root_dir=input_directory,
            transform=transform,
            crop_size=None
        )
        val_dataset = ImagePairDatasetBaseline(
            annotations_file=os.path.join(ANNOTATIONS_DIR, f"samples_val_pairs_with_scores.csv"),
            root_dir=input_directory,
            transform=transform,
            crop_size=None
        )
        print(f"DataLoader: Using baseline dataset.")
    else: 
        # Create Datasets for each Subset
        train_dataset = ImagePairDatasetTrain(
            annotations_file=os.path.join(ANNOTATIONS_DIR, f"samples_train_pairs_with_scores.csv"),
            root_dir=input_directory,
            transform=transform,
            crop_size=crop_size
        )
        test_dataset = ImagePairDatasetValTest(
            annotations_file=os.path.join(ANNOTATIONS_DIR, f"samples_test_pairs_with_scores.csv"),
            root_dir=input_directory,
            transform=transform,
            crop_size=crop_size
        )
        val_dataset = ImagePairDatasetValTest(
            annotations_file=os.path.join(ANNOTATIONS_DIR, f"samples_val_pairs_with_scores.csv"),
            root_dir=input_directory,
            transform=transform,
            crop_size=crop_size
        )
        print("DataLoader: Using collages dataset.")

    # Crate DataLoader for each Subset
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        generator=torch.Generator().manual_seed(42)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False
    )

    return train_loader, val_loader, test_loader
