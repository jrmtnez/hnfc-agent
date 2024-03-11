import torch
import logging
import re

from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split

from agent.data.entities.config import FD_CLASS_VALUES, ITEMS_TABLE
from agent.data.entities.config import TRAIN_ITEM_FILTER, TRAIN_LABEL, TEST_ITEM_FILTER
from agent.data.entities.config import DATA_CACHE_PATH, EXPORT_CHECK_INPUT_DATA_RESULTS_FILES
from agent.classifiers.utils.class_mgmt import get_numeric_rating, get_number_of_classes
from agent.data.sql.sql_mgmt import get_connection, select_fields_where
from agent.classifiers.transformer_classifier_text_input_mgmt import get_transformer_inputs

TRAIN_DEV_ITEM_FIELDS = "text, item_class_4"


def get_datasets_from_filter(pretrained_model, test_item_filter, test_label, max_lenght,
                             val_split=0.8, binary_classifier=False):

    logging.info("Loading %s item dataset...", TRAIN_LABEL)

    val_split=0.8
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

    dataset = TensorDataset(input_ids_train, attention_masks_train, y_train)

    dev_size = 1 - val_split
    train_ds, dev_ds = train_test_split(dataset, test_size=dev_size, random_state=25, shuffle=True)


    pretrained_model_label = pretrained_model.replace("/", "-")


    logging.info("Loading %s item dataset...", test_label)

    # if EXPORT_CHECK_INPUT_DATA_RESULTS_FILES:
    #     logging.info("Exporting check input data file...")

    #     if binary_classifier:
    #         binary_label = "bin"
    #     else:
    #         binary_label = "mc"

    #     file_name = DATA_CACHE_PATH + binary_label + "_" + pretrained_model_label + "_inputs.tsv"
    #     with open(file_name, "w", encoding="utf-8") as results_file:
    #         results_file.write("class\ttext\n")

    test_annotate_items = select_fields_where(connection, ITEMS_TABLE, TRAIN_DEV_ITEM_FIELDS, test_item_filter)
    x_test = []
    y_test = []
    for item in test_annotate_items:
        item_text = item[0]
        item_class_4 = item[1]
        numeric_class_value = get_numeric_rating(FD_CLASS_VALUES, item_class_4, binary_classifier=binary_classifier)

        x_test.append(item_text)
        y_test.append(numeric_class_value)

        # if EXPORT_CHECK_INPUT_DATA_RESULTS_FILES:
        #     document = item_text.replace(r"\n", r" ")
        #     document = document.replace(r"\t", r" ")
        #     document = re.sub(r"\s\s+", r" ", document)
        #     document = re.sub(r"\.” ([A-Z])", r".”. \1", document)
        #     with open(file_name, "a", encoding="utf-8") as results_file:
                # results_file.write(str(numeric_class_value) + "\t" + document[0:50] + "\n")

    input_ids_test, attention_masks_test = get_transformer_inputs(x_test, pretrained_model, max_lenght)

    y_test = torch.tensor(y_test)

    test_ds = TensorDataset(input_ids_test, attention_masks_test, y_test)

    logging.info("%s dataset size: %s", TRAIN_LABEL, len(train_ds))
    logging.info("dev dataset size: %s", len(dev_ds))
    logging.info("%s (test) dataset size: %s", test_label, len(test_ds))

    number_of_classes = get_number_of_classes(binary_classifier, FD_CLASS_VALUES)

    logging.info("Number of classes: %s", number_of_classes)

    return train_ds, dev_ds, test_ds, number_of_classes, None
