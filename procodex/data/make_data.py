import pandas as pd
from procodex import config
from procodex.utils.logger import logger
from procodex.utils.searilizer import serializer


def get_train_test_dataframe(DATA_DIR, TEST_DIR):
    logger.info(f"Loading data from: {DATA_DIR}, {TEST_DIR}")

    data = pd.read_excel(DATA_DIR)
    test = pd.read_excel(TEST_DIR)

    if "id" not in test.columns or "id" not in data.columns:
        raise Exception("Missing 'id' column in dataframes")

    test_ids = list(test["id"].values)

    # FILTERING TRAIN AND TEST BASED ON THE TEST IDs
    train = data[~data["id"].isin(test_ids)]
    test = data[data["id"].isin(test_ids)]

    serializer(train, config.TRAIN, mode="save")
    serializer(test, config.TEST, mode="save")

    return train, test


if __name__ == "__main__":
    pass
