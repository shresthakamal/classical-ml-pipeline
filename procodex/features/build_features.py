import os
import itertools
import numpy as np
from procodex.utils.searilizer import serializer
from sklearn.feature_extraction.text import TfidfVectorizer
from procodex.data.make_data import *

from procodex.utils.utils import *


def process_title(text):
    text = remove_punctuation(text)
    text = remove_stop_words(text)
    text = lower_case(text)
    text = lemmatize(text)
    text = remove_extra_non_breaking_spaces(text)

    return text


def process_keywords(text):
    # split the keywords based on ";"
    tokens = text.split(";")
    tokens = [remove_stop_words(t) for t in tokens]
    tokens = [remove_non_ascii(t) for t in tokens]

    tokens = [t.split("/") for t in tokens]
    tokens = list(itertools.chain.from_iterable(tokens))

    tokens = [remove_extra_non_breaking_spaces(t) for t in tokens]
    tokens = [lemmatize(t) for t in tokens]

    tokens = [lower_case(t) for t in tokens]

    return " ".join(tokens)


def process_target(label):
    label_map = {"Relevant": 1, "Not relevant": 0}
    return label_map[label]


class TFIDFVectorizer:
    def __init__(self, **params):
        self.params = params

    def vectorize(self, sentences):
        vectorizer = TfidfVectorizer(**self.params)
        vectors = vectorizer.fit_transform(sentences)

        return vectors, vectorizer


class PreProcessor:
    def __init__(self, **params):
        self.params = params
        self.vectorizer = TFIDFVectorizer(**params["vectorizer"]["params"])

        self.train_transform = self.params["train_transform"]
        self.test_transform = self.params["test_transform"]

        assert len(self.train_transform.keys()) == len(
            self.test_transform.keys()
        ), "[ERROR]: Different Training and Testing Transformations"

    def __dimension_check(self, ndim):
        if ndim < self.params["vectorizer"]["params"]["max_features"]:
            raise Exception(
                f"""Maximum features is more than number of TFIDF tokens, {ndim} < {self.params["vectorizer"]["max_features"]}"""
            )

    def process(self, df, mode):
        # PRE-PROCESSING MODES
        # TRAINING MODE
        #   APPLIES TRAINING TRANSFORM
        #   VECTORIZES TRAINING DATA
        #   SAVES TFIDF VECTORIZERS
        #   SAVES TRAINING DATA
        # TESTING MODE
        ##  APPLIES TESTING TRANSFORM
        ##  LOADS TFIDF VECTORIZERS- FROM TRAINING DATA
        ##  LOADS TRAINING DATA

        feature_vectors = []
        vectorizer_chpkt = self.params["vectorizer"]["ckpth"]

        if mode == "train":
            feature_vectorizers = {}

            for column, transform in self.train_transform.items():
                df[column] = df[column].apply(transform)

                if column != "target":
                    vecs, vectorizer = self.vectorizer.vectorize(df[column].values)
                    self.__dimension_check(vecs.toarray().shape[1])

                    feature_vectors.append(vecs.toarray())
                    feature_vectorizers[column] = vectorizer
                else:
                    y = df[column].values

            if serializer(feature_vectorizers, vectorizer_chpkt, "save"):
                logger.info(
                    f"Train TFIDF Vectorizers successfully saved at {vectorizer_chpkt}"
                )

        elif mode == "test":
            logger.info(f"Loading saved train vectorizers from {vectorizer_chpkt}")

            if not os.path.exists(vectorizer_chpkt):
                raise Exception("Vectorizers missing for test")

            feature_vectorizers = serializer(path=vectorizer_chpkt, mode="load")

            for column, transform in self.test_transform.items():
                df[column] = df[column].apply(transform)

                if column != "target":
                    vecs = feature_vectorizers[column].transform(df[column].values)
                    self.__dimension_check(vecs.toarray().shape[1])

                    feature_vectors.append(vecs.toarray())
                else:
                    y = df[column].values
        else:
            raise Exception("Preprocessing mode is not identified")

        # FEATURES ARE CONCATENATED at axis 1
        # so (-1, X) and (-1, X) becomes (-1, 2, X) reshaped to (-1, 2X)

        X = np.stack(feature_vectors, axis=1)
        X = np.reshape(X, (X.shape[0], -1))

        return X, y

    def inference(self, query):
        """User Query based inference"""

        feature_vectors = []
        feature_vectorizers = serializer(
            path=self.params["vectorizer"]["ckpth"], mode="load"
        )

        for column, value in query.items():
            value = self.test_transform[column](value)

            vecs = feature_vectorizers[column].transform([value])
            feature_vectors.append(vecs.toarray())

        X = np.stack(feature_vectors, axis=1)
        X = np.reshape(X, (X.shape[0], -1))

        return X


def build_features(DATA_DIR=None, TEST_DIR=None, VECTORIZER=None, only_processor=False):
    # INITIALIZING PRE-PROCESSING PIPELINE

    train_transform = {
        "title": process_title,
        "keywords": process_keywords,
        "target": process_target,
    }
    test_transform = {
        "title": process_title,
        "keywords": process_keywords,
        "target": process_target,
    }

    processor = PreProcessor(
        **{
            "vectorizer": VECTORIZER,
            "train_transform": train_transform,
            "test_transform": test_transform,
        }
    )

    # FLAG FOR INFERENCE
    if only_processor:
        return processor

    # GET TRAIN AND TEST DATAFRAMES
    logger.info("Building train and test features...")
    train, test = get_train_test_dataframe(DATA_DIR, TEST_DIR)

    # GENERATE TRAINING AND VALIDATION DATA
    train_X, train_y = processor.process(train, mode="train")
    test_X, test_y = processor.process(test, mode="test")

    return train_X, train_y, test_X, test_y, processor


if __name__ == "__main__":
    pass
