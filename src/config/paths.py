# Base directory for all data, models and results
BASE_DIR = ""

# Common paths used across the project
IMAGES_PADDED = BASE_DIR + "/images/padded" # Directory where baseline-preprocessed images are saved
IMAGES_COLLAGES = BASE_DIR + "/images/collages" # Directory where collage-preprocessed images are saved
RESULTS_DIR = BASE_DIR + "/results" # Directory where training results are saved
MODELS_DIR = BASE_DIR + "/models" # Directory where models are saved
ORIGINAL_DIR = BASE_DIR + "/old_data" # Directory containing the original images and annotations (from the previous project)
SCORING_DIR = BASE_DIR + "/scoring" # Directory where the scoring results are be saved
ANNOTATIONS_DIR = BASE_DIR + "/data/splits" # Directory where the annotation files are saved
TUNING_DIR = BASE_DIR + "/hyperparameter_tuning" # Directory where the hyperparameter tuning results are saved