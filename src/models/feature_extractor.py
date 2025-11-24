import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from transformers import PixtralVisionModel

    


class FeatureExtractorResNet(nn.Module):
    """
    Pretrained ResNet18 for extracting image features.
    """

    def __init__(self):
        super().__init__()

        # Initialize the ResNet18 network (without fc layer)
        self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()

        # Freeze all ResNet parameters
        for param in self.resnet.parameters():
           param.requires_grad = False
        

    def forward(self, x):
        """
        Extract ResNet features for a batch of images.

        The ResNet backbone is used without the final fully-connected layer
        and runs in a no-grad context so the pretrained weights remain frozen.

        Args:
            x (torch.Tensor): Input batch tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Feature tensor of shape (B, 512) produced by the
                ResNet backbone (the final classification layer has been
                replaced by an identity).
        """
        with torch.no_grad():
            x = self.resnet(x)
        return x


class FeatureExtractorPixtral(nn.Module):
    """
    Pretrained Pixtral Vision Encoder for extracting image features.
    """

    def __init__(self):
        super().__init__()

        # Initialize the Pixtral network
        self.pixtral_vision_encoder = PixtralVisionModel.from_pretrained("Prarabdha/pixtral-12b-vision-model")

    
    def forward(self, x):    
        """
        Extract Pixtral vision features for a batch of images.

        The Pixtral encoder is called per-sample inside a no-grad context and
        the pooled output embeddings are concatenated to form a batch tensor.

        Args:
            x (torch.Tensor): Input batch tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Feature tensor of shape (B, 1024) where each row is
                the pooled embedding produced by the Pixtral encoder.
        """
        batch_size = x.shape[0]
        embeddings = []
        with torch.no_grad():
            for i in range(batch_size):
                single_image = x[i].unsqueeze(0)  # (1, 3, H, W)
                output = self.pixtral_vision_encoder(pixel_values=single_image)
                emb = output.last_hidden_state.mean(dim=1)  # (1, 1024)
                embeddings.append(emb)
        return torch.cat(embeddings, dim=0)  # (batch_size, 1024)
        





