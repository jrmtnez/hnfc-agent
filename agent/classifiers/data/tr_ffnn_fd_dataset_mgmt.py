import torch
import logging

from torch.utils.data import TensorDataset

from agent.data.entities.config import FD_CLASS_VALUES, ITEMS_TABLE
from agent.data.entities.config import TRAIN_ITEM_FILTER, TRAIN_LABEL
from agent.classifiers.utils.class_mgmt import get_numeric_rating, get_number_of_classes
from agent.data.sql.sql_mgmt import get_connection, select_fields_where
from agent.classifiers.transformer_classifier_text_input_mgmt import get_transformer_inputs
from agent.classifiers.data.count_features_fd_dataset_mgmt import get_raw_datasets_from_filter as get_count_features_datasets_from_filter

TRAIN_DEV_ITEM_FIELDS = "text, item_class_4"


def get_datasets_from_filter(pretrained_model, dev_item_filter, dev_label, max_lenght, binary_classifier=False):

    logging.info("Loading %s item dataset...", TRAIN_LABEL)

    # transformer text dataset
    connection = get_connection()
    train_annotate_items = select_fields_where(connection, ITEMS_TABLE, TRAIN_DEV_ITEM_FIELDS, TRAIN_ITEM_FILTER)
    x_train = []
    y_train = []
    for item in train_annotate_items:
        item_text = item[0]
        item_class_4 = item[1]
        x_train.append(item_text)
        y_train.append(get_numeric_rating(FD_CLASS_VALUES, item_class_4, binary_classifier=binary_classifier))

    items_max_lenght = [len(x.split()) for x in x_train]
    items_max_lenght = max(items_max_lenght)

    if items_max_lenght < max_lenght:
        max_lenght = items_max_lenght

    logging.info("Max. sequence lenght: %s", max_lenght)

    input_ids_train, attention_masks_train = get_transformer_inputs(x_train, pretrained_model, max_lenght)
    y_train = torch.tensor(y_train)  # multiclass needs longs instead of floats

    # ffnn count features dataset
    x_train_ffnn, _, x_test_ffnn, _ = get_count_features_datasets_from_filter(TRAIN_ITEM_FILTER,
                                                                              dev_item_filter,
                                                                              TRAIN_LABEL,
                                                                              dev_label,
                                                                              review_level=9,
                                                                              binary_classifier=binary_classifier)

    x_train_ffnn = torch.tensor(x_train_ffnn.to_numpy(), dtype=torch.float)
    x_test_ffnn = torch.tensor(x_test_ffnn.to_numpy(), dtype=torch.float)

    ffnn_input_size = x_train_ffnn[:].shape[1]

    logging.info("FFNN input size: %s", ffnn_input_size)

    # combined dataset
    dataset = TensorDataset(input_ids_train, attention_masks_train, x_train_ffnn, y_train)

    logging.info("Loading %s item dataset...", dev_label)

    dev_annotate_items = select_fields_where(connection, ITEMS_TABLE, TRAIN_DEV_ITEM_FIELDS, dev_item_filter)
    x_dev = []
    y_dev = []
    for item in dev_annotate_items:
        item_text = item[0]
        item_class_4 = item[1]
        x_dev.append(item_text)
        y_dev.append(get_numeric_rating(FD_CLASS_VALUES, item_class_4, binary_classifier=binary_classifier))

    input_ids_dev, attention_masks_dev = get_transformer_inputs(x_dev, pretrained_model, max_lenght)

    y_dev = torch.tensor(y_dev)

    train_ds = dataset

    dev_ds = TensorDataset(input_ids_dev, attention_masks_dev, x_test_ffnn, y_dev)

    logging.info("%s dataset size: %s", TRAIN_LABEL,  len(train_ds))
    logging.info("%s dataset size: %s", dev_label, len(dev_ds))

    number_of_classes = get_number_of_classes(binary_classifier, FD_CLASS_VALUES)

    logging.info("Number of classes: %s", number_of_classes)

    return train_ds, dev_ds, None, ffnn_input_size, number_of_classes, None
