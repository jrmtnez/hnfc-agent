import torch
import logging
import pandas as pd

from torch.utils.data import TensorDataset, random_split
from sklearn.model_selection import train_test_split

from agent.data.entities.config import FD_CLASS_VALUES, ITEMS_TABLE, CTT3_TEST_ITEM_FILTER, CTT3SIM_LABEL, CTT3_LABEL
from agent.classifiers.utils.class_mgmt import get_numeric_rating, get_number_of_classes
from agent.data.sql.sql_mgmt import get_connection, select_fields_where
from agent.classifiers.transformer_classifier_text_input_mgmt import get_transformer_inputs

TRAIN_DEV_ITEM_FIELDS = "text, item_class_4"


def get_datasets(pretrained_model, max_lenght, val_split=0.8, binary_classifier=False,
                 _expand_tokenizer=False, _use_saved_model=True, _tokenizer_type="text"):

    logging.info("Loading %s train dataset...", CTT3SIM_LABEL)

    full_df = pd.read_json("external_datasets/full.json")

    logging.info("%s train dataset size: %s", CTT3SIM_LABEL, len(full_df))

    train_df, _ = train_test_split(full_df, test_size=0.2, random_state=25, shuffle=True)

    x_train = []
    y_train = []
    for _, article  in train_df.iterrows():
        x_train.append(article.text)

        binary_class_value = 0
        class_value = "F"
        if article.article_rating.lower() == "other":
            class_value = "NA"
        if article.article_rating.lower() == "false":
            class_value = "F"
        if article.article_rating.lower() == "partially false":
            class_value = "PF"
        if article.article_rating.lower() == "true":
            class_value = "T"
            binary_class_value = 1

        if binary_classifier:
            y_train.append(binary_class_value)
        else:
            y_train.append(get_numeric_rating(FD_CLASS_VALUES, class_value))

    items_max_lenght = [len(x.split()) for x in x_train]
    items_max_lenght = max(items_max_lenght)

    if items_max_lenght < max_lenght:
        max_lenght = items_max_lenght

    logging.info("Max. sequence lenght: %s", max_lenght)

    input_ids_train, attention_masks_train = get_transformer_inputs(x_train, pretrained_model, max_lenght)
    y_train = torch.tensor(y_train)  # multiclass needs longs instead of floats

    dataset = TensorDataset(input_ids_train, attention_masks_train, y_train)

    additional_val_split = 0.8
    train_size = int(additional_val_split * len(dataset))
    val_size = len(dataset) - train_size
    dataset, _ = random_split(dataset, [train_size, val_size])

    if val_split is None:

        logging.info("Loading %s dataset...", CTT3_LABEL)

        connection = get_connection()
        dev_annotate_items = select_fields_where(connection, ITEMS_TABLE, TRAIN_DEV_ITEM_FIELDS, CTT3_TEST_ITEM_FILTER)
        x_test = []
        y_test = []
        for item in dev_annotate_items:
            item_text = item[0]
            item_class_4 = item[1]
            x_test.append(item_text)
            y_test.append(get_numeric_rating(FD_CLASS_VALUES, item_class_4, binary_classifier=binary_classifier))

        input_ids_test, attention_masks_test = get_transformer_inputs(x_test, pretrained_model, max_lenght)

        y_test = torch.tensor(y_test)

        train_ds = dataset
        dev_ds = TensorDataset(input_ids_test, attention_masks_test, y_test)
    else:
        logging.info("Loading dev item dataset...")

        dev_size = 1 - val_split
        train_ds, dev_ds = train_test_split(dataset, test_size=dev_size, random_state=25, shuffle=True)

    logging.info("%s dataset size: %s", CTT3SIM_LABEL, len(train_ds))
    logging.info("%s dataset size: %s", CTT3_LABEL,  len(dev_ds))

    number_of_classes = get_number_of_classes(binary_classifier, FD_CLASS_VALUES)

    logging.info("Number of classes: %s", number_of_classes)

    return train_ds, dev_ds, None, number_of_classes, None
