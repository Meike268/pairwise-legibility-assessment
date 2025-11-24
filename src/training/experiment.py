import sys
sys.path.append('../src')

import torch
from torch.nn import BCEWithLogitsLoss
from src.models.feature_extractor import FeatureExtractorResNet, FeatureExtractorPixtral
from src.models.siamese_network import SiameseNeuralNetwork
from src.models.classifier import ClassifierResNet, ClassifierPixtral
import src.data.loader as loader
import src.models.preprocessor as preprocessor
from src.config.paths import IMAGES_PADDED, IMAGES_COLLAGES


"""Experiment configuration classes for model setup and training.

This module provides a base ``Experiment`` class that centralizes model
setup (backbone selection, optimizer, loss, data loaders) and several
concrete experiment configurations (preset hyperparameters and transforms)
for different backbones and dataset variants used in the project.

Each concrete experiment class sets attributes such as ``model_name``,
``image_size``, ``transform``, ``crop_size``, ``input_directory``, and
training hyperparameters (learning rate, weight decay, dropout, optimizer
choice). The base class exposes ``setup_experiment`` which instantiates the
model, optimizer, loss and prepares data loaders.
"""


class Experiment:
    def __init__(self):
        """Base experiment container.

        The base class is intentionally lightweight: concrete experiment
        subclasses initialize specific attributes (model/backbone name,
        image and crop sizes, input directories, data transforms and
        optimizer/hyperparameter defaults). 
        Concrete classes should call ``setup_experiment`` to instantiate the
        model, optimizer and data loaders after their attributes are set.
        """
    
    
    def setup_experiment(self):
        """Instantiates and defines all necessary aspects of the model

            Instantiates the model, sets up the optimizer, defines the loss function, defines the model abbreviation, 
            loads the data, and instantiates the trainer. 
        
        """
        # Instantiate the model
        if self.model_name == "resnet":
            self.model = SiameseNeuralNetwork(
                FeatureExtractorResNet(), 
                ClassifierResNet(dropout=self.dropout)  
            )
            print(f"Using ResNet.")
        elif self.model_name == "pixtral":
            self.model = SiameseNeuralNetwork(
                FeatureExtractorPixtral(), 
                ClassifierPixtral(dropout=self.dropout)  
            )
            print(f"Using PixTral.")
        else:
            self.model = None
            raise ValueError(f"Unknown model name: {self.model_name}")

        # Optimizer setup
        if self.optimizer_name == "SGD":
            self.optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
                nesterov=True,
            )
            print("Using SGD.")
        else:  # Adam
            self.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
            print("Using Adam.")

        # Loss function
        self.loss_function = BCEWithLogitsLoss()

        # Model abbreviation
        self.model_abbr = f"{self.study_type}_{self.model_name}_{self.crop_size}_crops_{self.image_size}"

        # Data
        if self.study_type == 'baseline':
            self.train_loader, self.val_loader, self.test_loader = loader.get_data_loaders(
                input_directory=self.input_directory,
                crop_size=self.crop_size,
                transform=self.transform,
                baseline=True
            )
            print("Using baseline images.")

        else:
            self.train_loader, self.val_loader, self.test_loader = loader.get_data_loaders(
                input_directory=self.input_directory,
                crop_size=self.crop_size,
                transform=self.transform,
                baseline=False
            )
            print("Using collage images.")
    


class ExperimentBaselineResnet224(Experiment):
    """Baseline experiment using ResNet backbone with 224px inputs.

    Args:
        momentum (float, optional): SGD momentum. Defaults to None.
        lr (float, optional): Learning rate. Defaults to 0.0001.
        weight_decay (float, optional): Weight decay. Defaults to 0.0019.
        dropout (float, optional): Dropout probability for the classifier. Defaults to 0.0732.
        optimizer_name (str, optional): Optimizer identifier ("Adam" or "SGD"). Defaults to "Adam".
    """

    def __init__(self, momentum=None, lr=0.0001, weight_decay=0.0019, dropout=0.0732, optimizer_name="Adam"):
        super().__init__()

        self.model_name = "resnet"
        self.image_size = 224
        self.transform = preprocessor.get_final_transforms_resnet(self.image_size)
        self.crop_size = "none"
        self.input_directory = IMAGES_PADDED
        self.study_type = "baseline"
        self.optimizer_name = optimizer_name

        self.momentum = momentum
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout


class ExperimentBaselinePixtral224(Experiment):
    """Baseline experiment using PixTral backbone with 224px inputs.

    Args:
        momentum (float, optional): SGD momentum. Defaults to None.
        lr (float, optional): Learning rate. Defaults to 0.0006.
        weight_decay (float, optional): Weight decay. Defaults to 0.0004.
        dropout (float, optional): Dropout probability for the classifier. Defaults to 0.1738.
        optimizer_name (str, optional): Optimizer identifier. Defaults to "Adam".
    """

    def __init__(self, momentum=None, lr=0.0006, weight_decay=0.0004, dropout=0.1738, optimizer_name="Adam"):
        super().__init__()

        self.model_name = "pixtral"
        self.image_size = 224
        self.transform = preprocessor.get_final_transforms_pixtral(self.image_size)
        self.crop_size = "none"
        self.input_directory = IMAGES_PADDED
        self.study_type = "baseline"
        self.optimizer_name = optimizer_name

        self.momentum = momentum
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout


class ExperimentBaselinePixtral336(Experiment):
    """Baseline experiment using PixTral backbone with 336px inputs.

    Args:
        momentum (float, optional): SGD momentum. Defaults to None.
        lr (float, optional): Learning rate. Defaults to 0.0006.
        weight_decay (float, optional): Weight decay. Defaults to 0.0004.
        dropout (float, optional): Dropout probability for the classifier. Defaults to 0.0640.
        optimizer_name (str, optional): Optimizer identifier. Defaults to "Adam".
    """

    def __init__(self, momentum=None, lr=0.0006, weight_decay=0.0004, dropout=0.0640, optimizer_name="Adam"):
        super().__init__()

        self.model_name = "pixtral"
        self.image_size = 336
        self.transform = preprocessor.get_final_transforms_pixtral(self.image_size)
        self.crop_size = "none"
        self.input_directory = IMAGES_PADDED
        self.study_type = "baseline"
        self.optimizer_name = optimizer_name

        self.momentum = momentum
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout


class ExperimentSmallCollagesResnet224(Experiment):
    """Small-collage experiment using ResNet and 224px inputs.

    Args:
        momentum (float, optional): SGD momentum. Defaults to None.
        lr (float, optional): Learning rate. Defaults to 0.0003.
        weight_decay (float, optional): Weight decay. Defaults to 7.75e-06.
        dropout (float, optional): Dropout probability for the classifier. Defaults to 0.0550.
        optimizer_name (str, optional): Optimizer identifier. Defaults to "Adam".
    """

    def __init__(self, momentum=None, lr=0.0003, weight_decay=7.75e-06, dropout=0.0550, optimizer_name="Adam"):
        super().__init__()

        self.model_name = "resnet"
        self.image_size = 224
        self.transform = preprocessor.get_final_transforms_resnet(self.image_size)
        self.crop_size = "small"
        self.input_directory = IMAGES_COLLAGES
        self.study_type = "collages"
        self.optimizer_name = optimizer_name

        self.momentum = momentum
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout


class ExperimentSmallCollagesPixtral224(Experiment):
    """Small-collage experiment using PixTral and 224px inputs.

    Args:
        momentum (float, optional): SGD momentum. Defaults to None.
        lr (float, optional): Learning rate. Defaults to 0.0014.
        weight_decay (float, optional): Weight decay. Defaults to 0.0002.
        dropout (float, optional): Dropout probability for the classifier. Defaults to 0.2388.
    """

    def __init__(self, momentum=None, lr=0.0014, weight_decay=0.0002, dropout=0.2388):
        super().__init__()

        self.model_name = "pixtral"
        self.image_size = 224
        self.transform = preprocessor.get_final_transforms_pixtral(self.image_size)
        self.crop_size = "small"
        self.input_directory = IMAGES_COLLAGES
        self.study_type = "collages"
        self.optimizer_name = "Adam"

        self.momentum = momentum
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout


class ExperimentSmallCollagesPixtral336(Experiment):
    """Small-collage experiment using PixTral and 336px inputs.

    Args:
        momentum (float, optional): SGD momentum. Defaults to None.
        lr (float, optional): Learning rate. Defaults to 0.0005.
        weight_decay (float, optional): Weight decay. Defaults to 6.44e-05.
        dropout (float, optional): Dropout probability for the classifier. Defaults to 0.0252.
    """

    def __init__(self, momentum=None, lr=0.0005, weight_decay=6.44e-05, dropout=0.0252):
        super().__init__()

        self.model_name = "pixtral"
        self.image_size = 336
        self.transform = preprocessor.get_final_transforms_pixtral(self.image_size)
        self.crop_size = "small"
        self.input_directory = IMAGES_COLLAGES
        self.study_type = "collages"
        self.optimizer_name = "Adam"

        self.momentum = momentum
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout


class ExperimentLargeCollagesResnet224(Experiment):
    """Large-collage experiment using ResNet with 224px inputs.

    Args:
        momentum (float, optional): SGD momentum. Defaults to None.
        lr (float, optional): Learning rate. Defaults to 0.0022.
        weight_decay (float, optional): Weight decay. Defaults to 0.0018.
        dropout (float, optional): Dropout probability for the classifier. Defaults to 0.0271.
        optimizer_name (str, optional): Optimizer identifier. Defaults to "Adam".
    """

    def __init__(self, momentum=None, lr=0.0022, weight_decay=0.0018, dropout=0.0271, optimizer_name="Adam"):
        super().__init__()

        self.model_name = "resnet"
        self.image_size = 224
        self.transform = preprocessor.get_final_transforms_resnet(self.image_size)
        self.crop_size = "large"
        self.input_directory = IMAGES_COLLAGES
        self.study_type = "collages"
        self.optimizer_name = optimizer_name

        self.momentum = momentum
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout

class ExperimentLargeCollagesPixtral224(Experiment):
    """Large-collage experiment using PixTral with 224px inputs.

    Args:
        momentum (float, optional): SGD momentum. Defaults to None.
        lr (float, optional): Learning rate. Defaults to 0.0010.
        weight_decay (float, optional): Weight decay. Defaults to 2.5965e-06.
        dropout (float, optional): Dropout probability for the classifier. Defaults to 0.2595.
        optimizer_name (str, optional): Optimizer identifier. Defaults to "Adam".
    """

    def __init__(self, momentum=None, lr=0.0010, weight_decay=2.5965e-06, dropout=0.2595, optimizer_name="Adam"):
        super().__init__()

        self.model_name = "pixtral"
        self.image_size = 224
        self.transform = preprocessor.get_final_transforms_pixtral(self.image_size)
        self.crop_size = "large"
        self.input_directory = IMAGES_COLLAGES
        self.study_type = "collages"
        self.optimizer_name = optimizer_name

        self.momentum = momentum
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout


class ExperimentLargeCollagesPixtral336(Experiment):
    """Large-collage experiment using PixTral with 336px inputs.

    Args:
        momentum (float, optional): SGD momentum. Defaults to None.
        lr (float, optional): Learning rate. Defaults to 0.0004.
        weight_decay (float, optional): Weight decay. Defaults to 4.6182e-06.
        dropout (float, optional): Dropout probability for the classifier. Defaults to 0.2129.
        optimizer_name (str, optional): Optimizer identifier. Defaults to "Adam".
    """

    def __init__(self, momentum=None, lr=0.0004, weight_decay=4.6182e-06, dropout=0.2129, optimizer_name="Adam"):
        super().__init__()

        self.model_name = "pixtral"
        self.image_size = 336
        self.transform = preprocessor.get_final_transforms_pixtral(self.image_size)
        self.crop_size = "large"
        self.input_directory = IMAGES_COLLAGES
        self.study_type = "collages"
        self.optimizer_name = optimizer_name

        self.momentum = momentum
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
