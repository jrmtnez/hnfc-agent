import logging
import re
import pandas as pd

from sklearn.model_selection import train_test_split

from agent.data.entities.config import ROOT_LOGGER_ID
from agent.data.entities.config import TRAIN_ITEM_FILTER, TRAIN_LABEL
from agent.data.entities.config import DEV_ITEM_FILTER, DEV_LABEL
from agent.data.entities.config import TEST_ITEM_FILTER, TEST_LABEL
from agent.data.entities.config import FD_CLASS_VALUES, ITEMS_TABLE

from agent.data.entities.config import SENTENCES_TABLE, FC_TRAIN_TEST_SENTENCE_FIELDS
from agent.data.entities.config import FC_TRAIN_SENTENCES_FILTER, FC_TEST_SENTENCES_FILTER
from agent.data.entities.config import FC_CLASS_VALUES

from agent.data.sql.sql_mgmt import get_connection, select_fields_where
from agent.classifiers.utils.class_mgmt import get_numeric_rating, get_number_of_classes

logger = logging.getLogger(ROOT_LOGGER_ID)


TRAIN_DEV_ITEM_FIELDS = "text, item_class_4"
ITEM_LIWC_EXPORT_FILE_TRAIN = "data/baselines/liwc_item_train.csv"
ITEM_LIWC_EXPORT_FILE_DEV = "data/baselines/liwc_item_dev.csv"
ITEM_LIWC_EXPORT_FILE_TEST = "data/baselines/liwc_item_test.csv"
ITEM_LIWC_FEATURES_FILE_TRAIN = "data/baselines/LIWC2015 Results (liwc_item_train).txt"
ITEM_LIWC_FEATURES_FILE_DEV = "data/baselines/LIWC2015 Results (liwc_item_dev).txt"
ITEM_LIWC_FEATURES_FILE_TEST = "data/baselines/LIWC2015 Results (liwc_item_test).txt"
SENTENCE_LIWC_EXPORT_FILE_TRAIN = "data/baselines/liwc_sentence_train.csv"
SENTENCE_LIWC_EXPORT_FILE_TEST = "data/baselines/liwc_sentence_test.csv"
SENTENCE_LIWC_FEATURES_FILE_TRAIN = "data/baselines/LIWC2015 Results (liwc_sentence_train).txt"
SENTENCE_LIWC_FEATURES_FILE_TEST = "data/baselines/LIWC2015 Results (liwc_sentence_test).txt"


def export_item_dataset(dataset_label, item_filter, text_file):
    connection = get_connection()
    logging.info("Loading %s item dataset...", dataset_label)
    annotate_items = select_fields_where(connection, ITEMS_TABLE, TRAIN_DEV_ITEM_FIELDS, item_filter)
    logging.info("Item %s dataset size: %s", dataset_label, len(annotate_items))

    with open(text_file, "w", encoding="utf-8") as file:
        file.write("document\tclass\n")
        for item in annotate_items:
            text = item[0]
            text = re.sub(r"\s\s+", " ", text.replace("\r", " ").replace("\n", " ")).strip()
            file.write(text.replace("\t", " ") + "\t" + item[1] + "\n")
    connection.close()

    logging.info("Item %s dataset exported to: %s", dataset_label, text_file)


def export_item_data_for_liwc_features():
    export_item_dataset(TRAIN_LABEL, TRAIN_ITEM_FILTER, ITEM_LIWC_EXPORT_FILE_TRAIN)
    export_item_dataset(DEV_LABEL, DEV_ITEM_FILTER, ITEM_LIWC_EXPORT_FILE_DEV)
    export_item_dataset(TEST_LABEL, TEST_ITEM_FILTER, ITEM_LIWC_EXPORT_FILE_TEST)


def get_item_liwc_features(binary_classifier=False, dev_dataset=True):
    # process LIWC files with:
    # (\t\d+),(?=\d+)
    # $1.

    train_df = pd.read_csv(ITEM_LIWC_FEATURES_FILE_TRAIN, sep="\t")
    if dev_dataset:
        test_df = pd.read_csv(ITEM_LIWC_FEATURES_FILE_DEV, sep="\t")
    else:
        test_df = pd.read_csv(ITEM_LIWC_FEATURES_FILE_TEST, sep="\t")

    y_train = []
    for item_class in train_df["Source (B)"].values:
        y_train.append(get_numeric_rating(FD_CLASS_VALUES, item_class, binary_classifier=binary_classifier))

    y_test = []
    for item_class in test_df["Source (B)"].values:
        y_test.append(get_numeric_rating(FD_CLASS_VALUES, item_class, binary_classifier=binary_classifier))

    x_train = train_df.drop(["Source (A)", "Source (B)"], axis=1)
    x_test = test_df.drop(["Source (A)", "Source (B)"], axis=1)

    number_of_classes = get_number_of_classes(binary_classifier, FD_CLASS_VALUES)
    input_size = x_train[:].shape[1]

    logging.info("Input size: %s", input_size)
    logging.info("Number of classes: %s", number_of_classes)

    logging.debug("LIWC columns: %s", x_test.columns.values.tolist())
    logging.info("LIWC train dataset size: %s", len(x_train))
    logging.info("LIWC test dataset size: %s", len(x_test))

    return x_train, y_train, x_test, y_test


def export_sentence_dataset(dataset_label, sentence_filter, text_file):
    connection = get_connection()
    logging.info("Loading %s item dataset...", dataset_label)
    annotate_sentences = select_fields_where(connection, SENTENCES_TABLE, FC_TRAIN_TEST_SENTENCE_FIELDS, sentence_filter)
    logging.info("Item %s dataset size: %s", dataset_label, len(annotate_sentences))

    with open(text_file, "w", encoding="utf-8") as file:
        file.write("document\tclass\n")
        for sentence in annotate_sentences:
            text = sentence[1]
            text = re.sub(r"\s\s+", " ", text.replace("\r", " ").replace("\n", " ")).strip()
            file.write(text.replace("\t", " ") + "\t" + sentence[2] + "\n")
    connection.close()

    logging.info("Sentence %s dataset exported to: %s", dataset_label, text_file)


def export_sentence_data_for_liwc_features():
    export_sentence_dataset(TRAIN_LABEL, FC_TRAIN_SENTENCES_FILTER, SENTENCE_LIWC_EXPORT_FILE_TRAIN)
    export_sentence_dataset(TEST_LABEL, FC_TEST_SENTENCES_FILTER, SENTENCE_LIWC_EXPORT_FILE_TEST)


def get_sentence_liwc_features(binary_classifier=False, dev_dataset=False, val_split=0.8):
    # process LIWC files with:
    # (\t\d+),(?=\d+)
    # $1.

    # train_df = pd.read_csv(SENTENCE_LIWC_FEATURES_FILE_TRAIN, sep="\t")

    # force same train number of instances than other sentence fc classifiers
    big_train_df = pd.read_csv(SENTENCE_LIWC_FEATURES_FILE_TRAIN, sep="\t")
    dev_size = 1 - val_split
    train_df, _ = train_test_split(big_train_df, test_size=dev_size, random_state=25, shuffle=True)

    test_df = pd.read_csv(SENTENCE_LIWC_FEATURES_FILE_TEST, sep="\t")

    y_train = []

    for sentence_class in train_df["Source (B)"].values:
        y_train.append(get_numeric_rating(FC_CLASS_VALUES, sentence_class, binary_classifier=binary_classifier))

    y_test = []
    for sentence_class in test_df["Source (B)"].values:
        y_test.append(get_numeric_rating(FC_CLASS_VALUES, sentence_class, binary_classifier=binary_classifier))

    x_train = train_df.drop(["Source (A)", "Source (B)"], axis=1)
    x_test = test_df.drop(["Source (A)", "Source (B)"], axis=1)

    number_of_classes = get_number_of_classes(binary_classifier, FC_CLASS_VALUES)
    input_size = x_train[:].shape[1]

    logging.info("Input size: %s", input_size)
    logging.info("Number of classes: %s", number_of_classes)

    logging.debug("LIWC columns: %s", x_test.columns.values.tolist())
    logging.info("LIWC train dataset size: %s", len(x_train))
    logging.info("LIWC test dataset size: %s", len(x_test))

    return x_train, y_train, x_test, y_test