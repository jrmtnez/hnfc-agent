import torch
import logging

from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from scipy.special import softmax

from agent.data.entities.config import CW_CLASS_VALUES, SENTENCES_TABLE
from agent.data.entities.config import CW_TRAIN_SENTENCE_FILTER, CW_NEW_SENTENCE_FILTER, CW_NEW_EXTERNAL_SENTENCE_FILTER
from agent.data.entities.config import CW_EXTERNAL_SENTENCE_FILTER, CW_TEST_SENTENCE_FILTER
from agent.classifiers.utils.class_mgmt import get_numeric_rating, get_textual_rating, get_number_of_classes
from agent.data.sql.sql_mgmt import get_connection, select_fields_where, update_non_text_field, update_text_field, exist_where
from agent.classifiers.transformer_classifier_text_input_mgmt import get_transformer_inputs
from agent.classifiers.transformer_classifier_text_trainer import test_loop

CW_TRAIN_DEV_SENTENCE_FIELDS = "sentence, check_worthy"
CW_TEST_SENTENCE_FIELDS = "id, sentence"
CW_EXTERNAL_SENTENCE_FIELDS = "id, health_terms_auto, check_worthy_auto"


def get_datasets(pretrained_model, max_lenght, val_split=0.8, binary_classifier=False,
                 _expand_tokenizer=None, _use_saved_model=None, _tokenizer_type=None, text_cuis=None):

    logging.info("Loading sentences datasets...")

    connection = get_connection()
    train_annotate_sentences = select_fields_where(connection,
                                                   SENTENCES_TABLE,
                                                   CW_TRAIN_DEV_SENTENCE_FIELDS,
                                                   CW_TRAIN_SENTENCE_FILTER)
    x_train = []
    y_train = []
    for sentence in train_annotate_sentences:
        sentence_text = sentence[0]
        check_worthy = sentence[1]
        x_train.append(sentence_text)
        # in binary classification 1 -> FR, 0 -> NA, NF, FNR, FRC
        y_train.append(get_numeric_rating(CW_CLASS_VALUES, check_worthy, binary_classifier=binary_classifier))

    sentences_max_lenght = [len(x.split()) for x in x_train]
    sentences_max_lenght = max(sentences_max_lenght)

    if sentences_max_lenght < max_lenght:
        max_lenght = sentences_max_lenght

    input_ids_train, attention_masks_train = get_transformer_inputs(x_train, pretrained_model, max_lenght)

    y_train = torch.tensor(y_train)  # multiclass needs longs instead of floats

    dataset = TensorDataset(input_ids_train, attention_masks_train, y_train)

    #
    # To unify criteria, we leave as the partition method the one provided by sklearn
    #
    # train_size = int(val_split * len(dataset))
    # dev_size = len(dataset) - train_size
    # train_ds, dev_ds = random_split(dataset, [train_size, dev_size])

    dev_size = 1 - val_split
    train_ds, dev_ds = train_test_split(dataset, test_size=dev_size, random_state=25, shuffle=True)

    test_annotate_sentences = select_fields_where(connection,
                                                  SENTENCES_TABLE,
                                                  CW_TRAIN_DEV_SENTENCE_FIELDS,
                                                  CW_TEST_SENTENCE_FILTER)
    x_test = []
    y_test = []
    for sentence in test_annotate_sentences:
        sentence_text = sentence[0]
        check_worthy = sentence[1]
        x_test.append(sentence_text)
        y_test.append(get_numeric_rating(CW_CLASS_VALUES, check_worthy, binary_classifier=binary_classifier))

    input_ids_test, attention_masks_test = get_transformer_inputs(x_test, pretrained_model, max_lenght)

    y_test = torch.tensor(y_test)  # multiclass needs longs instead of floats

    test_ds = TensorDataset(input_ids_test, attention_masks_test, y_test)

    logging.info("Train dataset size: %s", len(train_ds))
    logging.info("Dev dataset size: %s", len(dev_ds))
    logging.info("Test dataset size: %s", len(test_ds))

    number_of_classes = get_number_of_classes(binary_classifier, CW_CLASS_VALUES)

    logging.info("Number of classes: %s", number_of_classes)

    return train_ds, dev_ds, test_ds, number_of_classes, None


def annotate_dataset(model, pretrained_model, binary_classifier, max_lenght, batch_size, device,
                     _dataset=None, _expand_tokenizer=False, _tokenizer_type=None, text_cuis=None):

    if binary_classifier:
        target_field = "check_worthy_score_auto"
    else:
        target_field = "check_worthy_auto"

    logging.info("Loading dataset to annotate...")

    if _dataset == "new":
        cw_sentence_filter = CW_NEW_SENTENCE_FILTER
        cw_external_sentence_filter = CW_NEW_EXTERNAL_SENTENCE_FILTER

    if _dataset == "train":
        cw_sentence_filter = CW_TRAIN_SENTENCE_FILTER
        cw_external_sentence_filter = CW_NEW_EXTERNAL_SENTENCE_FILTER

    if _dataset == "test":
        cw_sentence_filter = CW_TEST_SENTENCE_FILTER
        cw_external_sentence_filter = CW_NEW_EXTERNAL_SENTENCE_FILTER

    # launched from hnfc_launch_89_cw_classifiers to annotate again
    if _dataset == "external":
        cw_sentence_filter = CW_EXTERNAL_SENTENCE_FILTER
        cw_external_sentence_filter = CW_EXTERNAL_SENTENCE_FILTER

    connection = get_connection()
    to_annotate_sentences = select_fields_where(connection,
                                                SENTENCES_TABLE,
                                                CW_TEST_SENTENCE_FIELDS,
                                                cw_sentence_filter)

    if len(to_annotate_sentences) > 0:
        x_test = []
        x_id = []
        for sentence in to_annotate_sentences:
            sentence_id = sentence[0]
            sentence_text = sentence[1]

            x_test.append(sentence_text)
            x_id.append(sentence_id)

        input_ids_test, attention_masks_test = get_transformer_inputs(x_test, pretrained_model, max_lenght)

        test_ds = TensorDataset(input_ids_test, attention_masks_test)

        class_predictions, flat_predictions = test_loop(model, batch_size, device, test_ds)
        raw_predictions = softmax(flat_predictions, axis=1)[:, 1]

        for i in range(len(x_test)):
            if binary_classifier:
                update_non_text_field(connection, SENTENCES_TABLE, "id = " + str(x_id[i]),
                                      target_field, raw_predictions[i])
                                    #   target_field, class_predictions[i])
            else:
                update_text_field(connection, SENTENCES_TABLE, "id = " + str(x_id[i]),
                                  target_field, get_textual_rating(CW_CLASS_VALUES, class_predictions[i]))
        connection.commit()

        # we make the fields where it should be annotated have the value of the
        # prediction because in external datasets there is no manual annotation
        external_sentences = select_fields_where(connection, SENTENCES_TABLE, CW_EXTERNAL_SENTENCE_FIELDS, cw_external_sentence_filter)
        for sentence in external_sentences:
            sentence_id = sentence[0]
            health_terms_auto = sentence[1]
            check_worthy_auto = sentence[2]

            update_text_field(connection, SENTENCES_TABLE, "id = " + str(sentence_id), "check_worthy", check_worthy_auto)
            health_terms = (health_terms_auto > 0)
            update_non_text_field(connection, SENTENCES_TABLE, "id = " + str(sentence_id), "health_terms", health_terms)
        connection.commit()


def exist_cw_sentences_to_annotate():

    cw_sentence_filter = CW_NEW_SENTENCE_FILTER
    cw_external_sentence_filter = CW_NEW_EXTERNAL_SENTENCE_FILTER

    connection = get_connection()
    exist_sentences = exist_where(connection, SENTENCES_TABLE, cw_sentence_filter)
    exist_external_sentences = exist_where(connection, SENTENCES_TABLE, cw_external_sentence_filter)

    return exist_sentences or exist_external_sentences
