# Pairwise Legibility Assessment

This project was developed as part of a Master's thesis. It is research code and was not developed, hardened, or tested as production code in an employment setting. Use it for experimentation, reproduction of results, or as a reference.

**Abstract**: 
Handwriting is taught in the early school years and is crucial for keeping up with schoolwork and for acquiring competencies connected with handwriting such as text comprehension. Handwriting legibility is closely associated with other cognitive and developmental processes and can be an indicator of general learning difficulties and serious conditions. Early detection of handwriting problems is therefore crucial for enabling targeted support. Existing manual and automated approaches to assess handwriting legibility are often time-consuming, subjective, or reliant on specialized technical infrastructure. Currently, further research is needed to develop an accessible, reliable and objective method for assessing handwriting legibility. 

This work introduces a methodological framework that applies deep-learning techniques to images of handwriting samples. The framework comprises (a) the collection of legibility annotations through pairwise comparisons, (b) the training of binary classification models to identify the more legible sample of two input samples, (c) the evaluation of a novel image preprocessing technique called collage method, and (d) the derivation of legibility scores using a global ranking approach. A small preliminary annotation study demonstrated high inter-rater (τ = 0.7398) and intra-rater (τ = 0.8529) agreement for the pairwise comparison method. The main annotation study yielded a balanced distribution of binary labels. The binary classification models achieved strong performance (ACC = 0.8205, ROC AUC = 0.8994, τ = 0.6638), which was further improved by applying the collage preprocessing method, yielding a maximum ACC of 0.8368, a ROC AUC of 0.9208 and a maximum τ of 0.7032. Deriving sample-specific legibility scores via a reference ranking proved effective, as indicated by a high correlation with ground-truth scores (ρ = 0.8847).

The results of this work offer several theoretical and practical implications. The methodological framework provides new directions for automated handwriting legibility assessment and establishes a foundation for future research. The proposed collage preprocessing method is broadly applicable and enables the conversion of images into square formats while minimizing information loss. Furthermore, the methodological framework could be refined for integration into educational applications, offering children accessible, reliable and objective feedback on handwriting legibility. Because the methods operate directly on image data, no specialized technical infrastructure is required beyond a camera and the respective assessment application.

**Quick Overview**
- **Purpose**: Train and evaluate pairwise (siamese-like) models to assess the legibility of image samples by comparing image pairs and deriving global rankings and scores.
- **Structure**: The repository contains training code, evaluation utilities, ranking helpers, and small scripts to run experiments and inference. 
- **Data**: As of current state, the repository does not contain annotation files, sample images, training results, and trained models. 

**Where to start**
- Training: run `model_training_main.py` to train experiments defined in `src/training/experiment.py`.
- Evaluation: run `model_evaluation_main.py` to evaluate a trained model on `validation` or `test` datasets.
- Ranking: run `ranking_main.py` to create rankings from pairwise CSVs and compare model ranking to gold standard.
- Scoring / Inference: use `score_inferral.py` to derive a legibility score for an individual handwriting sample using model inference and ranking. 
- Hyperparameter tuning: `optuna_tuning.py` provides an Optuna-based helper to tune hyperparameters; results are saved under the configured results directory.

**How to run (examples)**
- Train an experiment (example):

```
python model_training_main.py --experiment baselineResnet224 --epochs 10 --evaluate-every-n-batches 100
```

- Evaluate a trained model (example):

```
python model_evaluation_main.py --experiment baselineResnet224 --evaluation-method baseline --dataset validation
```

- Create rankings and evaluate correlation:

```
python ranking_main.py --gold-path /path/to/gold.csv --model-path /path/to/model_predictions.csv
```

- Run Optuna tuning (example entry at bottom of `optuna_tuning.py`): edit parameters there or create a small launcher script and run:

```
python optuna_tuning.py
```

- Infer scores for new images using the scorer helper in `score_inferral.py`:

```
python score_inferral.py --reference-samples-count 100 --ratings-count 5
```

(Each script has in-file documentation and command-line argument parsing; run with `-h` for help.)

**Experiment classes (high-level)**
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
  3. Pass `exp.model`, `exp.optimizer`, `exp.loss_function`, `exp.model_abbr`, `exp.study_type` and loaders to the `Trainer` to run training.

**Centralized paths/config**
- The repository centralizes filesystem paths in `src/config/paths.py`.
- Prefer using those constants instead of hard-coded paths. Typical constants include:
  - `BASE_DIR` — project data root
  - `IMAGES_PADDED`, `IMAGES_COLLAGES` — image directories
  - `RESULTS_DIR`, `MODELS_DIR` — where training results and model checkpoints are written
  - `ANNOTATIONS_DIR` — where CSV annotation/split files are stored
  - `SCORING_DIR` — folder for scoring/evaluation outputs
  - `ORIGINAL_DIR` — folder where the data from the previous project is saved

**Environment & dependencies**
- The code uses Python 3.x and common ML packages: `torch`/`torchvision`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `scipy`, `optuna` (for tuning). Check `environment.yml` in the repository root for precise versions.

**Caveats & notes**
- This is thesis/research code: quality-of-life features and edge-case handling are not guaranteed. 
- File/directory constants are centralized in `src/config/paths.py`; if your environment uses different locations, edit that file rather than patching individual scripts.

**Contributing / Extending**
- To add a new experiment, add a subclass in `src/training/experiment.py`.
- To change where outputs are written, update `src/config/paths.py` and ensure directories exist (scripts try to create directories where appropriate).

**Contact / provenance**
- This project was created for a Master's thesis; it documents the experiments and tools used by the author. If you reuse or adapt parts of the code, please cite/refer to the thesis or contact the author for details.


