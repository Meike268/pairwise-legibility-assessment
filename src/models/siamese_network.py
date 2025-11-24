import torch
import torch.nn as nn


class SiameseNeuralNetwork(nn.Module):
    """
    Simple Siamese wrapper combining a feature extractor and a classifier.

    The `feature_extractor` should map a batch of images to a per-image
    embedding tensor of shape (batch_size, D). The `classifier` should accept
    the concatenated embedding of shape (batch_size, 2*D) and return a
    scalar output per example (e.g., a logit or regression value).

    Args:
        feature_extractor (nn.Module): Pretrained backbone that produces
            per-image embeddings.
        classifier (nn.Module): Small head that maps concatenated embeddings
            to a scalar output.
    """
    def __init__(self, feature_extractor, classifier):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier

    def forward(self, image1, image2):
        """
        Forward pass for a pair of images.

        Each image is passed independently through the shared
        ``feature_extractor``. The two resulting embeddings are concatenated
        along the feature dimension and the concatenated tensor is passed to
        the ``classifier``.

        Args:
            image1 (torch.Tensor): Batch of images with shape (B, C, H, W).
            image2 (torch.Tensor): Batch of images with shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor from the classifier, typically of
                shape (B, 1) containing a scalar score/logit per pair.
        """
        feature_representation_image1 = self.feature_extractor(image1)
        feature_representation_image2 = self.feature_extractor(image2)

        #print("→ feature1 shape:", feature_representation_image1.shape)
        #print("→ feature2 shape:", feature_representation_image2.shape)

        combined = torch.cat((feature_representation_image1, feature_representation_image2), dim=1)
        #print("→ combined shape:", combined.shape)

        out = self.classifier(combined)
        #print("→ output shape:", out.shape)
        return out
