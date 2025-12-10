# Pairwise Legibility Assessment

This project was developed as part of my Master thesis with the title "Deep Learning Based Assessment of Handwriting Legibility via Pairwise Image Comparisons and Ranking" at the Chair of Explainable Machine Learning (University of Bamberg). Use it for experimentation, reproduction of results, or as a reference. If you reuse or adapt parts of the code, please cite the thesis or contact the author for details.

**Abstract**: 

Handwriting legibility is closely associated with other cognitive processes and can
be an indicator of general learning difficulties and developmental disorders. Early
detection of handwriting problems is therefore crucial for enabling targeted sup-
port. Existing manual and automated approaches to assess handwriting legibility
are time consuming, subjective, or reliant on a specialized technical infrastructure.
More research is needed to develop an accessible, reliable, and objective method for
assessing handwriting legibility.

This work introduces a methodological framework that applies deep learning tech-
niques to images of handwriting samples. The framework comprises (a) the col-
lection of legibility annotations through pairwise comparisons, (b) the training of
deep learning based binary classification models to identify the more legible sample
of two input samples, (c) the development and evaluation of a novel image pre-
processing technique for creating square sized input images, and (d) the inferral of
individual legibility scores using a global ranking approach. A small preliminary an-
notation study demonstrated high inter-rater (Kendall’s τ = 0.7398) and intra-rater
(Kendall’s τ = 0.8529) agreement for the pairwise comparison method. The main
annotation study yielded a balanced distribution of binary labels. The binary classi-
fication models achieved strong performance (ACC = 0.8205, ROC AUC = 0.8994,
Kendall’s τ = 0.6638), which was further improved by applying the novel image
preprocessing method, yielding a maximum ACC of 0.8368, a ROC AUC of 0.9208
and a Kendall’s τ of 0.7032. Deriving sample-specific legibility scores via a reference
ranking proved effective, as indicated by a high correlation with ground-truth scores
(Spearman’s ρ = 0.8847).

The results of this work offer theoretical and practical implications. The method-
ological framework establishes a foundation for automated handwriting legibility
assessment and a new direction for image preprocessing methods. The newly intro-
duced image preprocessing method proved effective in this work and shows potential
for use in other scenarios where images must be reformatted prior to neural network
processing. A refinement of the proposed methodological framework for legibility
assessment could further enable its integration into educational applications that
offer children accessible, reliable, and objective feedback on handwriting legibility.

**Data:** 
- As of current state, the repository does not contain annotation files, sample images, training results, and trained models. 

**Where to start:**
- Training: run `model_training_main.py` to train experiments defined in `src/training/experiment.py`.
- Evaluation: run `model_evaluation_main.py` to evaluate a trained model on `validation` or `test` datasets.
- Ranking: run `ranking_main.py` to create rankings from pairwise comparisons and compare model ranking to ground-truth ranking.
- Scoring: use `scoring_main.py` to derive a legibility score for an individual handwriting sample using model inference and ranking. 
- Hyperparameter tuning: `optuna_tuning.py` provides an Optuna-based helper to tune hyperparameters; results are saved under the configured results directory.

(Each script has in-file documentation and command-line argument parsing; run with `-h` for help.)

**Experiment classes:**
- Location: `src/training/experiment.py`.
- Purpose: Each `Experiment` subclass encapsulates the configuration for a training run:
  - Model architecture and backbone selection (feature extractor + classifier)
  - Image transforms and crop sizes
  - Data loaders for `train`, `val`, and `test`
  - Optimizer and loss function setup
  - Short names/identifiers used for saving checkpoints and results
- Typical usage flow:
  1. Instantiate the desired `Experiment` subclass (e.g. `ExperimentBaselineResnet224`).
  2. Call `setup_experiment()` to create model, optimizer, data loaders and any other objects.
  3. Pass `model`, `optimizer`, `loss_function`, `model_abbr`, `study_type` and data loaders to the `Trainer` instance to run training.

**Image preprocessing and transforms:**
- The code supports two image preprocessing strategies, the *baseline* preprocessing and the *collage* preprocessing, that are implemented in `src/data/preprocessing.py`. Both use model-specific transform pipelines (resize, normalize) that are implemented in `src/models/preprocessor.py`. The images were preprocessed and saved prior to model training and are loaded via the DataLoader (`src/data/loader.py`).

- Baseline preprocessing: The baseline preprocessing method prepares a single square image per sample using white padding and center cropping. An image tensor with shape `(batch_size, channels, height, width)` is fed into the model.

- Collage preprocessing: The collage preprocessing creates a large collage per sample and randomly takes multiple crops per collage for training. An image tensor with shape `(batch_size, num_crops, channels, height, width)` is fed into the model. The code averages logits across crops at inference. 

- Each `Experiment.setup_experiment()` selects appropriate transform functions and assigns them to the dataset constructors (`train_loader`, `val_loader`, `test_loader`). If you change transforms in `src/models/preprocessor.py`, experiments that use those helpers will pick up the changes automatically.

**Centralized paths:**
- The repository centralizes filesystem paths in `src/config/paths.py`.
- Prefer using those constants instead of hard-coded paths. Typical constants include:
  - `BASE_DIR` — project data root
  - `IMAGES_PADDED`, `IMAGES_COLLAGES` — image directories
  - `RESULTS_DIR`, `MODELS_DIR` — where training results and model checkpoints are written
  - `ANNOTATIONS_DIR` — where CSV annotation/split files are stored
  - `SCORING_DIR` — folder for scoring/evaluation outputs
  - `ORIGINAL_DIR` — folder where the data from the previous project is saved

**Environment:**
- The code uses Python 3.x and common ML packages: `torch`/`torchvision`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `scipy`, `optuna`. Check `environment.yml` in the repository root for precise versions.



**Extending:**
- To add a new experiment, add a subclass in `src/training/experiment.py`.
- To change where outputs are written, update `src/config/paths.py` and ensure directories exist (scripts try to create directories where appropriate).



