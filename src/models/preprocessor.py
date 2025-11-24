from torchvision import transforms
from transformers import PixtralImageProcessor


class PixtralPreprocessorTransform:
    """
    Callable wrapper around ``PixtralImageProcessor`` that produces a tensor.

    This wrapper adapts the Hugging Face Pixtral image processor so it can be
    used as a torchvision-style transform (callable). The processor expects a
    PIL image or numpy array and returns a PyTorch tensor suitable for
    passing to the Pixtral vision encoder.
    """
    def __init__(self, model_name):
        """
        Initialize the Pixtral image processor.

        Args:
            model_name (str): Hugging Face model identifier to load the image
                processor from (e.g. "mistral-community/pixtral-12b").
        """
        self.processor = PixtralImageProcessor.from_pretrained(model_name)

    def __call__(self, image):
        """
        Process a single image and return pixel values tensor.

        Args:
            image (PIL.Image.Image or numpy.ndarray): Input image.

        Returns:
            torch.Tensor: Pixel values tensor with shape (3, H, W) (first
                batch dimension removed).
        """
        inputs = self.processor(images=image, return_tensors="pt")
        return inputs["pixel_values"][0]  # shape: (3, H, W)

    

def get_final_transforms_resnet(image_size):
    """
    Return a torchvision ``Compose`` pipeline for ResNet-style inputs.

    The pipeline resizes the input to ``(image_size, image_size)``, converts
    it to a tensor and normalizes using ImageNet statistics.

    Args:
        image_size (int): Target side length (pixels) for the square resize.

    Returns:
        torchvision.transforms.Compose: Transform pipeline producing a
            normalized tensor ready for ResNet encoders.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])  # ImageNet stats
    ])

def get_final_transforms_pixtral(image_size):
    """
    Return a transform pipeline that prepares images for the Pixtral encoder.

    The pipeline resizes the image and then applies the
    ``PixtralPreprocessorTransform`` which returns pixel values tensors
    expected by the Pixtral vision model.

    Args:
        image_size (int): Target side length (pixels) for the square resize.

    Returns:
        torchvision.transforms.Compose: Transform pipeline that yields tensors
            compatible with the Pixtral model's input requirements.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        PixtralPreprocessorTransform(model_name="mistral-community/pixtral-12b")
    ])
