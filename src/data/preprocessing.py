from torchvision.transforms import Compose
import torchvision.transforms.functional as F
import os
import random
import cv2
import numpy as np
from PIL import Image

def preprocess_directory(input_dir: str, output_dir: str, collage: bool):
    """
    Preprocess all images in a directory and write results to an output directory.

    Two preprocessing modes are supported:
      - collage=True: creates a large collage canvas from each input image.
      - collage=False: pads the image to a square and center-crops a fixed-size patch.

    Note: This function performs filesystem traversal, reads images with OpenCV,
    applies the selected preprocessing, and writes results to `output_dir`.

    Args:
        input_dir (str): Path to the directory containing the raw images.
        output_dir (str): Path where preprocessed images will be saved. If
            ``None`` the output filename is derived from the input path.
        collage (bool): If True use the collage method; otherwise use the
            padding + center-crop baseline method.

    Returns:
        None. Processed images are written to `output_dir`.

    Raises:
        OSError: if writing files to disk fails (propagates from underlying IO calls).
    """
    if not collage:
        # Find maximum and minimum width in original images
        max_width = find_max_width(input_dir) # 2926 pixels
        min_width = find_min_width(input_dir) # 931 pixels
        

    for dirpath, dirnames, filenames in os.walk(input_dir):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                in_path = os.path.join(dirpath, filename)

                # Load image
                img = cv2.imread(in_path)
                if img is None:
                    continue

                # Convert to grayscale if needed
                if len(img.shape) == 3 and img.shape[2] == 3:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    gray = img
                
                # Preprocess the image
                otsu = apply_otsu_threshold(gray)
                if collage:
                    output = create_collage(otsu, target_height=6920, target_width=5852)
                else:
                    padded = pad_to_square(otsu, max_width) # Pad to maximum width
                    output = center_crop_pil(padded, min_width) # Crop the center image with size of min_width
                    output = np.array(output)
                    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
                if output_dir:
                    rel_path = os.path.relpath(dirpath, input_dir)
                    out_dir = os.path.join(output_dir, rel_path)
                    os.makedirs(out_dir, exist_ok=True)
                    out_path = os.path.join(out_dir, filename)
                else:
                    out_path = os.path.splitext(in_path)[0] + "_otsu.png"
                cv2.imwrite(out_path, output)
    print(f"Preprocessing of directory {input_dir} completed. Results were saved to {output_dir}.")

def apply_otsu_threshold(image):
    """
    Binarize an image using Otsu's thresholding.

    The function converts color images to grayscale, applies a small Gaussian
    blur and then Otsu's threshold to produce a binary image.

    Args:
        image (numpy.ndarray): Input image in BGR (OpenCV) or grayscale format.

    Returns:
        numpy.ndarray: Binary image (dtype uint8) with values {0, 255}.
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, binary_img = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_img

def create_collage(image, target_height: int, target_width: int):
    """
    Tile an input image into a larger canvas of a given size.

    The input image is repeatedly pasted row-by-row into a new empty canvas
    of size (target_height, target_width). For rows where the last tile does
    not fully fit, a cropped slice from the original image is used.

    Args:
        image (numpy.ndarray): Source image array (grayscale or color) in
            OpenCV format (H x W) or (H x W x C).
        target_height (int): Height of the output collage canvas in pixels.
        target_width (int): Width of the output collage canvas in pixels.

    Returns:
        numpy.ndarray: New image array of shape (target_height, target_width[, C]).
    """
    # Get height and width of original image
    original_height, original_width = image.shape[:2]

    # Create blank output image (black background)
    if len(image.shape) == 3:
        new_image = np.zeros((target_height, target_width, image.shape[2]), dtype=image.dtype)
    else:
        new_image = np.zeros((target_height, target_width), dtype=image.dtype)

    # Fill image row by row
    for y_offset in range(0, target_height, original_height):
        row_height = min(original_height, target_height - y_offset)
        start_x = random.randint(0, original_width - 1)
        current_x = 0
        remaining = target_width

        # Paste until the row is fully filled horizontally
        while remaining > 0:
            crop_x_start = start_x % original_width
            slice_width = min(original_width - crop_x_start, remaining)

            if len(image.shape) == 2:
                # Grayscale
                cropped = image[:row_height, crop_x_start:crop_x_start + slice_width]
                new_image[y_offset:y_offset + row_height, current_x:current_x + slice_width] = cropped
            else:
                # Color
                cropped = image[:row_height, crop_x_start:crop_x_start + slice_width, :]
                new_image[y_offset:y_offset + row_height, current_x:current_x + slice_width, :] = cropped

            current_x += slice_width
            remaining -= slice_width
            start_x += slice_width  
            
    return new_image


def pad_to_square(image, size: int, fill: int =255):
    """
    Pad an image (OpenCV array) to a square of the requested size using a
    constant fill color, and return a PIL.Image instance.

    Args:
        image (numpy.ndarray): Input image in BGR color order (OpenCV format).
        size (int): Target square side length in pixels.
        fill (int): Grayscale fill value for padding (default 255 = white).

    Returns:
        PIL.Image.Image: Padded image in RGB mode (PIL Image).
    """
    # Convert BGR to RGB
    cv2_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(cv2_rgb)
    w, h = image.size
    if w < size:
        diff_width = size - w
        padding_width = (diff_width // 2, 0, diff_width - diff_width // 2, 0)
        image = F.pad(image, padding_width, fill=fill)
    if h < size:
        diff_height = size - h
        padding_height = (0, diff_height // 2, 0, diff_height - diff_height // 2)
        image = F.pad(image, padding_height, fill=fill)

    return image
    

def center_crop_pil(image, crop_size: int):
    """
    Return a centered square crop from a PIL image.

    Args:
        image (PIL.Image.Image): Input image.
        crop_size (int): Side length of the square crop in pixels.

    Returns:
        PIL.Image.Image: Centered square crop (may be smaller if image is
            smaller than requested crop size in either dimension).
    """
    width, height = image.size
    center_x, center_y = width // 2, height // 2

    half_crop = crop_size // 2

    # Compute bounding box for the crop
    left = max(center_x - half_crop, 0)
    upper = max(center_y - half_crop, 0)
    right = min(center_x + half_crop, width)
    lower = min(center_y + half_crop, height)

    return image.crop((left, upper, right, lower))


def find_max_width(root_dir: str):
    """
    Compute the maximum image width (in pixels) for PNG images under a
    directory tree.

    Args:
        root_dir (str): Root directory to search for PNG images.

    Returns:
        int: Maximum width in pixels found among PNG images. Returns 0 if no
            readable PNG images are found.
    """
    max_width = 0
    max_file = None

    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(".png"):
                file_path = os.path.join(subdir, file)
                try:
                    image = cv2.imread(file_path)
                    if image is None:
                        print(f"Warning: Failed to read {file_path}")
                        continue
                    height, width = image.shape[:2]
                    if width > max_width:
                        max_width = width
                        max_file = file_path
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    print(f"Maximum width: {max_width} pixels")
    print(f"File with max width: {max_file}")

    return max_width

def find_min_width(root_dir: str):
    """
    Compute the minimum image width (in pixels) for PNG images under a
    directory tree.

    Args:
        root_dir (str): Root directory to search for PNG images.

    Returns:
        int: Minimum width in pixels found among PNG images. If no images are
            found this function currently returns a high default (2926).
    """
    min_width = 2926
    min_file = None

    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(".png"):
                file_path = os.path.join(subdir, file)
                try:
                    image = cv2.imread(file_path)
                    if image is None:
                        print(f"Warning: Failed to read {file_path}")
                        continue
                    height, width = image.shape[:2]
                    if width < min_width:
                        min_width = width
                        min_file = file_path
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    print(f"Maximum width: {min_width} pixels")
    print(f"File with max width: {min_file}")

    return min_width

def find_max_height(root_dir: str):
    """
    Compute the maximum image height (in pixels) for PNG images under a
    directory tree.

    Args:
        root_dir (str): Root directory to search for PNG images.

    Returns:
        int: Maximum height in pixels found among PNG images. Returns 0 if no
            readable PNG images are found.
    """
    max_height = 0
    max_file = None

    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(".png"):
                file_path = os.path.join(subdir, file)
                try:
                    with Image.open(file_path) as img:
                        width, height = img.size
                        if height > max_height:
                            max_height = height
                            max_file = file_path
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    print(f"Maximum height: {max_height} pixels")
    print(f"File with max height: {max_file}")

    return max_height

def transform_directory(input_path: str, output_path: str, transform: Compose):
    """
    Apply a torchvision `transform` to every image in a directory tree and
    save the transformed outputs preserving the folder structure.

    The function reads images with OpenCV in grayscale mode, applies the
    provided `transform`, converts results back to OpenCV-compatible numpy
    arrays when necessary, and writes the output images.

    Args:
        input_path (str): Directory containing input images.
        output_path (str): Directory where transformed images will be saved.
        transform (torchvision.transforms.Compose): Transformation pipeline to
            apply to each input image. The transform should accept a numpy
            array or PIL image depending on how it was constructed.

    Returns:
        None. Processed images are saved to disk.
    """
    transform = transform
    input_folder = input_path
    output_folder = output_path

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            # Skip hidden/system files
            if filename.startswith('.'):
                continue

            # Full path to the input image
            input_path = os.path.join(root, filename)

            # Compute relative path to maintain folder structure in output
            relative_path = os.path.relpath(root, input_folder)
            output_dir = os.path.join(output_folder, relative_path)
            os.makedirs(output_dir, exist_ok=True)

            # Full path to the output image
            output_path = os.path.join(output_dir, filename)

            # Read the image
            image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                print(f"Warning: Failed to read {input_path}")
                continue

            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            result = transform(image)

            # Process output depending on its type
            """
            if isinstance(result, torch.Tensor):
                result = result.permute(1, 2, 0).detach().cpu().numpy()

                if result.max() <= 1.0:
                    result = (result * 255).astype(np.uint8)
                else:
                    result = np.clip(result, 0, 255).astype(np.uint8)

                if result.shape[-1] == 3:
                    # Convert RGB to BGR for OpenCV
                    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            """

            if isinstance(result, np.ndarray):
                if result.ndim == 2:  # grayscale
                    pass
                elif result.shape[-1] == 3:
                    # Assume it's still RGB and convert to BGR if needed
                    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

            else:
                print(f"Unsupported result type for {input_path}")
                continue

            cv2.imwrite(output_path, result)
            print(f"Processed and saved: {output_path}")








