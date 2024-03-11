import torch
import logging

from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from scipy.special import softmax

from agent.data.sql.sql_mgmt import get_connection, select_fields_where, update_non_text_field, update_text_field
from agent.classifiers.data.lstm_dataset_mgmt import get_lstm_inputs
from agent.classifiers.transformer_classifier_triple_input_mgmt import get_transformer_inputs
from agent.classifiers.data.transformer_text_triple_fc_dataset_mgmt import get_cuis_spo, get_raw_datasets
from agent.classifiers.utils.class_mgmt import get_textual_rating, get_number_of_classes
from agent.data.entities.config import SENTENCES_TABLE, FC_CLASS_VALUES, FC_TRAIN_TEST_SENTENCE_FIELDS
from agent.data.entities.config import FC_TRAIN_SENTENCES_FILTER, FC_TEST_SENTENCES_FILTER, FC_NEW_SENTENCES_FILTER
from agent.classifiers.tr_ensemble_triple_trainer import test_loop


def get_datasets(pretrained_model, max_lenght, val_split=0.8, lowercase=True, binary_classifier=False,
                 expand_tokenizer=False, use_saved_model=False, shrink_seq_lenght=False, tokenizer_type="text",
                 cuis=False):

    logging.info("Loading datasets...")

    x_train, y_train, x_test, y_test, train_annotate_sentences, test_annotate_sentences = get_raw_datasets(binary_classifier=binary_classifier,
                                                                                                           cuis=cuis)
    x_train_seqs, x_test_seqs = get_lstm_inputs(x_train, x_test, lowercase=lowercase)

    x_train_lstm = torch.tensor(x_train_seqs)
    x_test_lstm = torch.tensor(x_test_seqs)
    y_train = torch.tensor(y_train)  # multiclass needs longs instead of floats
    y_test = torch.tensor(y_test)

    lstm_seq_lenght = x_train_lstm[:].shape[1]

    logging.info("LSTM sequence lenght: %s", lstm_seq_lenght)

    x_train_text, _, _, _, _, _ = get_raw_datasets(binary_classifier=binary_classifier, cuis=False)

    if shrink_seq_lenght:
        sentences_max_lenght = [len(x.split()) for x in x_train_text]
        sentences_max_lenght = max(sentences_max_lenght)

        if sentences_max_lenght < max_lenght:
            max_lenght = sentences_max_lenght

    input_ids_train, attention_masks_train, segment_ids_train, new_vocab_size = get_transformer_inputs(train_annotate_sentences,
                                                                                                       pretrained_model, max_lenght,
                                                                                                       expand_tokenizer=expand_tokenizer,
                                                                                                       use_saved_model=use_saved_model,
                                                                                                       tokenizer_type=tokenizer_type)

    dataset = TensorDataset(input_ids_train, attention_masks_train, segment_ids_train, x_train_lstm, y_train)

    test_size = 1 - val_split
    train_ds, dev_ds = train_test_split(dataset, test_size=test_size, random_state=25, shuffle=True)

    input_ids_test, attention_masks_test, segment_ids_test, _ = get_transformer_inputs(test_annotate_sentences,
                                                                                       pretrained_model, max_lenght,
                                                                                       expand_tokenizer=expand_tokenizer,
                                                                                       use_saved_model=use_saved_model,
                                                                                       tokenizer_type=tokenizer_type)

    test_ds = TensorDataset(input_ids_test, attention_masks_test, segment_ids_test, x_test_lstm, y_test)

    logging.info("Train dataset size: %s", len(train_ds))
    logging.info("Dev dataset size: %s", len(dev_ds))
    logging.info("Test dataset size: %s", len(test_ds))

    number_of_classes = get_number_of_classes(binary_classifier, FC_CLASS_VALUES)

    logging.info("Number of classes: %s", number_of_classes)

    return train_ds, dev_ds, test_ds, lstm_seq_lenght, number_of_classes, new_vocab_size


def annotate_dataset(model, pretrained_model, binary_classifier, max_lenght, batch_size, device,
                     dataset="new", tokenizer_type="text", cuis=False, expand_tokenizer=False,
                     use_saved_model=True, lowercase=True, unk_tokens=False):

    if binary_classifier:
        target_field = "sentence_class_score_auto"
    else:
        target_field = "sentence_class_auto"

    if dataset == "train":
        sentece_filter = FC_TRAIN_SENTENCES_FILTER  # for final fake news article classification
    if dataset == "test":
        sentece_filter = FC_TEST_SENTENCES_FILTER
    if dataset == "new":
        sentece_filter = FC_NEW_SENTENCES_FILTER

    logging.info("Loading dataset to annotate...")

    connection = get_connection()
    to_annotate_sentences = select_fields_where(connection, SENTENCES_TABLE, FC_TRAIN_TEST_SENTENCE_FIELDS, sentece_filter)
    if len(to_annotate_sentences) > 0:
        x_test = []
        x_id = []
        for sentence in to_annotate_sentences:
            sentence_id = sentence[0]
            sentence_text = sentence[1]
            subject_cuis = sentence[3]
            predicate_cuis = sentence[4]
            object_cuis = sentence[5]

            if cuis:
                x_test.append(get_cuis_spo(subject_cuis, predicate_cuis, object_cuis, unk_tokens=unk_tokens))
            else:
                x_test.append(sentence_text)
            x_id.append(sentence_id)

        _, x_test_seqs = get_lstm_inputs(x_test, x_test, lowercase=lowercase)

        x_test_lstm = torch.tensor(x_test_seqs)

        input_ids_test, attention_masks_test, segment_ids_test, _ = get_transformer_inputs(to_annotate_sentences,
                                                                                           pretrained_model, max_lenght,
                                                                                           expand_tokenizer=expand_tokenizer,
                                                                                           use_saved_model=use_saved_model,
                                                                                           tokenizer_type=tokenizer_type)

        test_ds = TensorDataset(input_ids_test, attention_masks_test, segment_ids_test, x_test_lstm)

        class_predictions, flat_predictions = test_loop(model, batch_size, device, test_ds)
        raw_predictions = softmax(flat_predictions, axis=1)[:, 1]

        for i in range(len(x_test)):
            if binary_classifier:
                update_non_text_field(connection, SENTENCES_TABLE, "id = " + str(x_id[i]),
                                      target_field, raw_predictions[i])
            else:
                update_text_field(connection, SENTENCES_TABLE, "id = " + str(x_id[i]),
                                  target_field, get_textual_rating(FC_CLASS_VALUES, class_predictions[i]))

        connection.commit()
