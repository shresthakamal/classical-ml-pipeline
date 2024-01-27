from procodex.utils.logger import logger
from procodex.config import *

from procodex.features.build_features import *
from procodex.utils.metrics import *
from procodex.utils.searilizer import serializer
from procodex.architectures.dispatcher import MODELS


def train_pipeline(model_name, train_X, train_y, test_X, test_y):
    logger.info("Initiating Model Training Pipeline")

    # INITIALIZING MODEL PARAMETERS
    model_params = MODELS_PARAMETERS[model_name]["params"]
    model = MODELS[model_name]
    logger.info(f"{model}::{model_params}")

    # FITTING THE MODEL
    classifier = model(**model_params).fit(train_X, train_y)
    serializer(classifier, MODELS_PARAMETERS[model_name]["ckpth"], "save")

    # TRAINING AND VALIDATION SCORE
    logger.info(f"Training Accuracy Score: {classifier.score(train_X, train_y)}")

    predictions = classifier.predict(test_X)
    logger.info(f"Validation Accuracy Score: {model_accuracy(predictions, test_y)}")


def testing_pipeline(model_name="logisticregression", test_query=None, processor=None):
    logger.info(f"Initiating Inference Pipeline...")

    # RUNNING INFERENCE ON USER INPUT
    test_X = processor.inference(test_query)

    # LOADING THE SAVED MODEL
    model = serializer(path=MODELS_PARAMETERS[model_name]["ckpth"], mode="load")
    logger.info(f"Model Name: {model, model.get_params()}, User Query: {test_query}")

    prediction = model.predict(test_X)
    probas = model.predict_proba(test_X)

    logger.info(f"Inference Result: Prediction>>{prediction}, Probas>>{probas}")


def user_model_selection():
    """Function to take user selection, can be CLI or dashboard

    Returns:
        str: User Selected Model
    """
    return "naivebayes"


def get_user_query():
    return {
        "title": "This is a sentence",
        "keywords": "This is a sentence",
    }


if __name__ == "__main__":
    user_selected_model = user_model_selection()

    train_X, train_y, test_X, test_y, processor = build_features(
        DATA_DIR, TEST_DIR, VECTORIZER
    )

    # TRAINING AND VALIDATION PIPELINE

    train_pipeline(
        user_selected_model,
        train_X,
        train_y,
        test_X,
        test_y,
    )

    # INFERENCE PIPELINE

    testing_pipeline(
        user_selected_model,
        get_user_query(),
        processor=build_features(DATA_DIR, TEST_DIR, VECTORIZER, only_processor=True),
    )
