import pandas as pd
import logging
import torch
import json
import re

from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from scipy.special import softmax
from os.path import join, exists

from agent.data.entities.config import ITEMS_TABLE, SENTENCES_TABLE, FD_CLASS_VALUES
from agent.data.entities.config import DATA_CACHE_PATH
from agent.data.sql.sql_mgmt import get_connection, select_fields_where, update_non_text_field, update_text_field
from agent.classifiers.utils.class_mgmt import get_number_of_classes, get_numeric_rating, get_textual_rating
from agent.classifiers.ffnn_trainer import test_loop


ITEM_FIELDS = "id, item_class, item_class_4, text"
SENTENCE_FIELDS = "sentence_class_auto, sentence_class_score_auto, check_worthy_score_auto"

EXPORT_CHECK_DATASET_FILES = False

def build_item_fd_features(item_filter, ds_label, review_level, use_cache=True):

    ds_cache_file = f"count_features_{ds_label}.json"
    if exists(join(DATA_CACHE_PATH, ds_cache_file)) and use_cache:
        logging.info("Loading cached %s fd features...", ds_label)

        item_features_df = pd.read_json(join(DATA_CACHE_PATH, ds_cache_file))
    else:
        logging.info("Buiding %s fd features...", ds_label)

        if EXPORT_CHECK_DATASET_FILES:
            test_file = join(DATA_CACHE_PATH, f"dataset_check_{ds_label}.tsv")
            write_check_file = open(test_file , "w", encoding="utf-8")

        item_features_list = []

        connection = get_connection()
        items = select_fields_where(connection, ITEMS_TABLE, ITEM_FIELDS, item_filter)

        for item in items:
            item_id = item[0]
            item_class = item[1]
            item_class_4 = item[2]
            item_text = item[3][0:50]

            if EXPORT_CHECK_DATASET_FILES:
                item_text2 = item_text
                item_text2 = item_text2.replace(r"\n", r" ")
                item_text2 = item_text2.replace(r"\t", r" ")
                item_text2 = re.sub(r"\s\s+", r" ", item_text2)
                write_check_file.write(f"{ds_label}\t{item_id}\t{item_text2}\n")

            f_count = 0
            pf_count = 0
            t_count = 0
            na_count = 0
            score_sum = 0
            t05_count = 0
            t04_count = 0
            t00_count = 0
            cw_sum = 0

            sentence_filter = f"""
                item_id = {item_id} AND
                review_level = {review_level}
                """

            sentences = select_fields_where(connection, SENTENCES_TABLE, SENTENCE_FIELDS, sentence_filter)
            for sentence in sentences:

                predicted_sentence_class4 = sentence[0]        # sentence_class_4_auto
                predicted_sentence_class2_score = sentence[1]  # sentence_class_score_auto
                predicted_cw_score = sentence[2]               # check_worthy_score_auto

                if predicted_sentence_class4 == 'F':
                    f_count += 1
                if predicted_sentence_class4 == 'PF':
                    pf_count += 1
                if predicted_sentence_class4 == 'T':
                    t_count += 1
                if predicted_sentence_class4 == 'NA':
                    na_count += 1

                score_sum = score_sum + predicted_sentence_class2_score
                if predicted_sentence_class2_score > 0.5:
                    t05_count += 1
                if predicted_sentence_class2_score > 0.4:
                    t04_count += 1
                if predicted_sentence_class2_score <= 0.4:
                    t00_count += 1

                cw_sum = cw_sum + predicted_cw_score

            total_count = f_count + pf_count + t_count + na_count

            item_features = {}
            # item_features["item_id"] = item_id
            item_features["f_count"] = f_count
            item_features["pf_count"] = pf_count
            item_features["t_count"] = t_count
            item_features["na_count"] = na_count
            if total_count != 0:
                item_features["f_perc"] = f_count / total_count
                item_features["pf_perc"] = pf_count / total_count
                item_features["t_perc"] = t_count / total_count
                item_features["na_perc"] = na_count / total_count
            else:
                item_features["f_perc"] = 0
                item_features["pf_perc"] = 0
                item_features["t_perc"] = 0
                item_features["na_perc"] = 0
            item_features["t05_count"] = t05_count
            item_features["t04_count"] = t04_count
            item_features["t00_count"] = t00_count
            if total_count != 0:
                item_features["t05_perc"] = t05_count / total_count
                item_features["t04_perc"] = t04_count / total_count
                item_features["t00_perc"] = t00_count / total_count
            else:
                item_features["t05_perc"] = 0
                item_features["t04_perc"] = 0
                item_features["t00_perc"] = 0
            item_features["score_sum"] = score_sum
            if total_count != 0:
                item_features["score_avg"] = score_sum / total_count
            else:
                item_features["score_avg"] = 0
            item_features["cw_sum"] = cw_sum
            if total_count != 0:
                item_features["cw_avg"] = cw_sum / total_count
            else:
                item_features["cw_avg"] = 0

            item_features["item_class"] = item_class
            item_features["item_class_4"] = get_numeric_rating(FD_CLASS_VALUES, item_class_4)
            item_features["text"] = item_text
            item_features_list.append(item_features)

        connection.commit()
        connection.close()

        with open(join(DATA_CACHE_PATH, ds_cache_file) , "w", encoding="utf-8") as write_file:
            json.dump(item_features_list, write_file, indent=4, separators=(",", ": "))

        item_features_df = pd.DataFrame(item_features_list)

    return item_features_df


def get_raw_datasets_from_filter(train_item_filter, test_item_filter, train_label, test_label, review_level, binary_classifier=True):

    train_df = build_item_fd_features(train_item_filter, train_label, review_level )
    test_df = build_item_fd_features(test_item_filter, test_label, review_level)

    logging.info("Binary classifier: %s", binary_classifier)
    if binary_classifier:
        y_train = train_df["item_class"].values
        y_test = test_df["item_class"].values
    else:
        y_train = train_df["item_class_4"].values
        y_test = test_df["item_class_4"].values

    x_train = train_df.iloc[:, :-3]
    x_test = test_df.iloc[:, :-3]

    logging.info("Train dataset size: %s", len(train_df))
    logging.info("Test dataset size: %s", len(test_df))

    return x_train, y_train, x_test, y_test


def get_raw_dataframes_from_filter(train_item_filter, test_item_filter, train_label, test_label, review_level, binary_classifier=True):

    train_df = build_item_fd_features(train_item_filter, train_label, review_level )
    test_df = build_item_fd_features(test_item_filter, test_label, review_level)

    if binary_classifier:
        train_df = train_df.drop(['item_class_4'], axis=1)
        test_df = test_df.drop(['item_class_4'], axis=1)
    else:
        train_df = train_df.drop(['item_class'], axis=1)
        test_df = test_df.drop(['item_class'], axis=1)

    train_df = train_df.drop(['text'], axis=1)
    test_df = test_df.drop(['text'], axis=1)

    logging.info("Train dataset size: %s", len(train_df))
    logging.info("Test dataset size: %s", len(test_df))

    return train_df, test_df


def get_datasets_from_filter(train_item_filter, test_item_filter, train_label, test_label, review_level, binary_classifier=True, val_split=0.8):

    x_train, y_train, x_test, y_test  = get_raw_datasets_from_filter(train_item_filter, test_item_filter,
                                                                     train_label, test_label, review_level,
                                                                     binary_classifier=binary_classifier)

    x_train = torch.tensor(x_train.to_numpy(), dtype=torch.float)
    x_test = torch.tensor(x_test.to_numpy(), dtype=torch.float)
    y_train = torch.tensor(y_train)
    y_test = torch.tensor(y_test)

    dataset = TensorDataset(x_train, y_train)

    dev_size = 1 - val_split

    train_ds, dev_ds = train_test_split(dataset, test_size=dev_size, random_state=25, shuffle=True)

    test_ds = TensorDataset(x_test, y_test)

    number_of_classes = get_number_of_classes(binary_classifier, FD_CLASS_VALUES)
    input_size = x_train[:].shape[1]

    logging.info("Input size: %s", input_size)
    logging.info("Number of classes: %s", number_of_classes)

    return train_ds, dev_ds, test_ds, input_size, number_of_classes


def get_raw_dataset_to_annotate(item_filter, item_label, review_level, binary_classifier=True):

    item_df = build_item_fd_features(item_filter, item_label, review_level, use_cache=False)
    item_ids = item_df["item_id"].values
    x_train = item_df.iloc[:, :-2]

    return x_train, item_ids


def annotate_dataset(model, binary_classifier, batch_size, device, item_filter, item_label, review_level=8):

    if binary_classifier:
        target_field = "item_class_auto"
    else:
        target_field = "item_class_4_auto"

    logging.info("Loading dataset to annotate...")

    connection = get_connection()

    x_features, x_id = get_raw_dataset_to_annotate(item_filter, item_label, review_level,
                                                    binary_classifier=binary_classifier)

    features_ds = TensorDataset(x_features)

    class_predictions, flat_predictions = test_loop(model, batch_size, device, features_ds)

    raw_predictions = softmax(flat_predictions, axis=1)[:, 1]

    for i in range(len(x_features)):
        if binary_classifier:
            update_non_text_field(connection, ITEMS_TABLE, "id = " + str(x_id[i]),
                                  target_field, raw_predictions[i])
        else:
            update_text_field(connection, ITEMS_TABLE, "id = " + str(x_id[i]),
                              target_field, get_textual_rating(FD_CLASS_VALUES, class_predictions[i]))
    connection.commit()

    # # we make the fields where it should be annotated have the value of the
    # # prediction because in external datasets there is no manual annotation
    external_sentences = select_fields_where(connection, SENTENCES_TABLE, CW_EXTERNAL_SENTENCE_FIELDS, cw_external_sentence_filter)
    for sentence in external_sentences:
        sentence_id = sentence[0]
        health_terms_auto = sentence[1]
        check_worthy_auto = sentence[2]

        update_text_field(connection, SENTENCES_TABLE, "id = " + str(sentence_id), "check_worthy", check_worthy_auto)
        health_terms = (health_terms_auto > 0)
        update_non_text_field(connection, SENTENCES_TABLE, "id = " + str(sentence_id), "health_terms", health_terms)
    connection.commit()
