import torch
import logging

from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from agent.data.entities.config import FC_CLASS_VALUES
from agent.classifiers.utils.class_mgmt import get_number_of_classes, get_numeric_rating
from agent.classifiers.data.transformer_text_triple_fc_dataset_mgmt  import get_raw_datasets
from agent.data.sql.sql_mgmt import get_connection, select_fields_where
from agent.data.entities.config import FD_CLASS_VALUES, ITEMS_TABLE, TRAIN_ITEM_FILTER, DEV_ITEM_FILTER

ITEM_FIELDS = "text, item_class_4"


def get_tfidf_inputs(x_train_text, x_test_text, lowercase=True):
    vectorizer = TfidfVectorizer(ngram_range=(1, 1), lowercase=lowercase)
    train_text_fitted_vectorizer = vectorizer.fit(x_train_text)
    train_text_tiv_fit_transform = train_text_fitted_vectorizer.transform(x_train_text)
    x_train_tfidf = train_text_tiv_fit_transform.toarray()
    x_test_tfidf = train_text_fitted_vectorizer.transform(x_test_text).toarray()

    return x_train_tfidf, x_test_tfidf


def get_datasets(val_split=0.8, lowercase=True, binary_classifier=False, cuis=False, unk_tokens=False, items_dataset=False):
    logging.info("Loading datasets...")

    if items_dataset:
        x_train, y_train, x_test, y_test = get_raw_text_item_datasets(binary_classifier)
    else:
        x_train, y_train, x_test, y_test, _, _ = get_raw_datasets(binary_classifier=binary_classifier, cuis=cuis, unk_tokens=unk_tokens)

    x_train_tfidf, x_test_tfidf = get_tfidf_inputs(x_train, x_test, lowercase=lowercase)

    x_train_tfidf = torch.tensor(x_train_tfidf, dtype=torch.float)
    x_test_tfidf = torch.tensor(x_test_tfidf, dtype=torch.float)
    y_train = torch.tensor(y_train)  # multiclass needs longs instead of floats
    y_test = torch.tensor(y_test)

    dataset = TensorDataset(x_train_tfidf, y_train)
    dev_size = 1 - val_split
    train_ds, dev_ds = train_test_split(dataset, test_size=dev_size, random_state=25, shuffle=True)

    test_ds = TensorDataset(x_test_tfidf, y_test)

    logging.info("Train dataset size: %s", len(train_ds))
    logging.info("Dev dataset size: %s", len(dev_ds))
    logging.info("Test dataset size: %s", len(test_ds))

    input_size = x_train_tfidf[:].shape[1]
    number_of_classes = get_number_of_classes(binary_classifier, FC_CLASS_VALUES)

    logging.info("Input size: %s", input_size)
    logging.info("Number of classes: %s", number_of_classes)

    return train_ds, dev_ds, test_ds, input_size, number_of_classes


def get_raw_text_item_datasets(binary_classifier=False):
    connection = get_connection()

    train_annotate_items = select_fields_where(connection, ITEMS_TABLE, ITEM_FIELDS, TRAIN_ITEM_FILTER)
    x_train = []
    y_train = []
    for item in train_annotate_items:
        item_text = item[0]
        item_class = item[1]

        x_train.append(item_text)
        y_train.append(get_numeric_rating(FD_CLASS_VALUES, item_class, binary_classifier=binary_classifier))

    test_annotate_items = select_fields_where(connection, ITEMS_TABLE, ITEM_FIELDS, DEV_ITEM_FILTER)
    x_test = []
    y_test = []
    for item in test_annotate_items:
        item_text = item[0]
        item_class = item[1]

        x_test.append(item_text)
        y_test.append(get_numeric_rating(FD_CLASS_VALUES, item_class, binary_classifier=binary_classifier))

    connection.close()

    return x_train, y_train, x_test, y_test