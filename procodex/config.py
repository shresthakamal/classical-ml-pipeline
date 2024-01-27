BASE_DIR = "procodex/"
DATA_DIR = "data/raw/20230821_full_data.xlsx"
TEST_DIR = "data/raw/test_ids.xlsx"

TRAIN = "data/processed/train.pkl"
TEST = "data/processed/test.pkl"

LOG_DIR = BASE_DIR + "/logs"
LOG_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> >> <level>{message}</level>"

VECTORIZER = {
    "ckpth": "checkpoints/vectorizers.pkl",
    "params": {
        "ngram_range": (1, 2),
        "max_features": 3475,
    },
}

MODELS_PARAMETERS = {
    "logisticregression": {
        "ckpth": "checkpoints/logisticregression.pkl",
        "params": {
            "C": 1.0,
            "penalty": "l2",
            "solver": "liblinear",
        },
    },
    "knn": {
        "ckpth": "checkpoints/knn.pkl",
        "params": {
            "n_neighbors": 5,
            "weights": "distance",
        },
    },
    "naivebayes": {
        "ckpth": "checkpoints/naivebayes.pkl",
        "params": {},
    },
    "randomforest": {
        "ckpth": "checkpoints/randomforest.pkl",
        "params": {
            "n_estimators": 100,
        },
    },
    "decisiontree": {
        "ckpth": "checkpoints/decisiontree.pkl",
        "params": {
            "criterion": "gini",
            "splitter": "best",
            "max_depth": None,
        },
    },
    "gradientboost": {
        "ckpth": "checkpoints/gradientboost.pkl",
        "params": {
            "n_estimators": 100,
            "learning_rate": 1.0,
            "max_depth": 1,
            "random_state": 0,
        },
    },
}
