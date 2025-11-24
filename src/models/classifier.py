import torch.nn as nn


class ClassifierResNet(nn.Module):
    """
    Classifier head for ResNet feature embeddings.

    The network expects input feature vectors formed by concatenating two
    ResNet embeddings (each with 512 dimensions), resulting in an input
    dimensionality of ``512 * 2``. The head maps that vector through a small
    MLP to a single scalar output per example.

    Args:
        dropout (float): Dropout probability used in the hidden layer.
    """
    def __init__(self, dropout):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        """
        Forward pass through the classifier head.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 512*2).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1) containing a
                scalar score/logit per input example.
        """
        return self.classifier(x)
    
class ClassifierPixtral(nn.Module):
    """
    Classifier head for Pixtral feature embeddings.

    This variant assumes each image is represented by a 1024-dim embedding
    and concatenates two such embeddings (input dim ``1024 * 2``). The head
    reduces the concatenated vector to a single scalar output.

    Args:
        dropout (float): Dropout probability used in the hidden layer.
    """
    def __init__(self, dropout):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(1024 * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        """
        Forward pass through the Pixtral classifier head.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1024*2).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1) containing a
                scalar score/logit per input example.
        """
        return self.classifier(x)
    