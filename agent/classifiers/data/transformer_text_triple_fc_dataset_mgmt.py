import torch
import logging

from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from scipy.special import softmax
from nltk.stem import WordNetLemmatizer

from agent.data.entities.config import FC_CLASS_VALUES, SENTENCES_TABLE, UNK_TOKEN, FC_TRAIN_TEST_SENTENCE_FIELDS, CUIS_SPO_SEP
from agent.data.entities.config import FC_TRAIN_SENTENCES_FILTER, FC_TEST_SENTENCES_FILTER, FC_NEW_SENTENCES_FILTER
from agent.data.entities.config import FC_CTT3FR_SENTENCES_FILTER
from agent.data.sql.sql_mgmt import get_connection, select_fields_where, update_non_text_field, update_text_field, exist_where
from agent.classifiers.utils.class_mgmt import get_numeric_rating, get_textual_rating, get_number_of_classes
from agent.classifiers.transformer_classifier_triple_input_mgmt import get_transformer_inputs
from agent.classifiers.transformer_classifier_triple_trainer import test_loop

from agent.nlp.graph_mgmt import SENTENCE_FIELDS, DEFAULT_SEM_GROUPS
from agent.nlp.graph_mgmt import load_phrasal_verbs, get_valid_groups, get_sem_type_names
from agent.nlp.graph_mgmt import get_norm_p, get_norm_so, match_extraction_with_spo2

def remove_unk_tokens(text):
    from_tokens = text.split()
    tokens = [t for t in from_tokens if t != UNK_TOKEN]
    return " ".join(tokens)


def get_cuis_spo(subject_cuis, predicate_cuis, object_cuis, unk_tokens=False):
    if not unk_tokens:
        subject_cuis = remove_unk_tokens(subject_cuis)
        predicate_cuis = remove_unk_tokens(predicate_cuis)
        object_cuis = remove_unk_tokens(object_cuis)
    return subject_cuis + " " + CUIS_SPO_SEP + " " + predicate_cuis + " " + CUIS_SPO_SEP + " " + object_cuis


def get_text_cuis_spo(big_subject, big_predicate, big_object, predicate,
                      spo_type, phrasal_verbs_set, lemmatizer, sentence_text,
                      sem_type_names_dict, metamap_extraction, node_type):

    norm_p, tokens_pos = get_norm_p(predicate, big_predicate, spo_type, phrasal_verbs_set, lemmatizer)
    s_ext, _, o_ext = match_extraction_with_spo2(metamap_extraction, sentence_text, big_subject, predicate, big_object)

    norm_s_list, norm_s_str = get_norm_so(s_ext, DEFAULT_SEM_GROUPS, sem_type_names_dict, node_type=node_type, add_brackets=False)
    norm_o_list, norm_o_str  = get_norm_so(o_ext, DEFAULT_SEM_GROUPS, sem_type_names_dict, node_type=node_type, add_brackets=False)

    return norm_s_str + " " + CUIS_SPO_SEP + " " + norm_p + " " + CUIS_SPO_SEP + " " + norm_o_str


def get_raw_datasets(binary_classifier=False, cuis=False, unk_tokens=False):
    if cuis:
        lemmatizer = WordNetLemmatizer()
        phrasal_verbs_set = load_phrasal_verbs()
        s_valid_groups_dict = get_valid_groups(which="subject")
        o_valid_groups_dict = get_valid_groups(which="object")
        sem_type_names_dict = get_sem_type_names()
        node_type = "text_cui"

    connection = get_connection()

    train_annotate_sentences = select_fields_where(connection, SENTENCES_TABLE, SENTENCE_FIELDS, FC_TRAIN_SENTENCES_FILTER)
    x_train = []
    y_train = []
    for sentence in train_annotate_sentences:
        sentence_text = sentence[1]
        sentence_class = sentence[2]
        subject_cuis = sentence[3]
        predicate_cuis = sentence[4]
        object_cuis = sentence[5]

        big_subject = sentence[6]
        big_predicate = sentence[7]
        big_object = sentence[8]
        spo_type = sentence[12]
        predicate = sentence[14]
        metamap_extraction = sentence[16]

        if cuis:
            x_train.append(get_cuis_spo(subject_cuis, predicate_cuis, object_cuis, unk_tokens=unk_tokens))
            # x_train.append(get_text_cuis_spo(big_subject, big_predicate, big_object, predicate,
            #                spo_type, phrasal_verbs_set, lemmatizer, sentence_text,
            #                sem_type_names_dict, metamap_extraction, node_type))
        else:
            x_train.append(sentence_text)
        y_train.append(get_numeric_rating(FC_CLASS_VALUES, sentence_class, binary_classifier=binary_classifier))

    test_annotate_sentences = select_fields_where(connection, SENTENCES_TABLE, SENTENCE_FIELDS, FC_TEST_SENTENCES_FILTER)
    x_test = []
    y_test = []
    for sentence in test_annotate_sentences:
        sentence_text = sentence[1]
        sentence_class = sentence[2]
        subject_cuis = sentence[3]
        predicate_cuis = sentence[4]
        object_cuis = sentence[5]

        big_subject = sentence[6]
        big_predicate = sentence[7]
        big_object = sentence[8]
        spo_type = sentence[12]
        predicate = sentence[14]
        metamap_extraction = sentence[16]

        if cuis:
            x_test.append(get_cuis_spo(subject_cuis, predicate_cuis, object_cuis, unk_tokens=unk_tokens))
            # x_test.append(get_text_cuis_spo(big_subject, big_predicate, big_object, predicate,
            #               spo_type, phrasal_verbs_set, lemmatizer, sentence_text,
            #               sem_type_names_dict, metamap_extraction, node_type))
        else:
            x_test.append(sentence_text)
        y_test.append(get_numeric_rating(FC_CLASS_VALUES, sentence_class, binary_classifier=binary_classifier))

    connection.close()

    return x_train, y_train, x_test, y_test, train_annotate_sentences, test_annotate_sentences


def get_datasets(pretrained_model, max_lenght, val_split=0.8, binary_classifier=False,
                 _expand_tokenizer=False, _use_saved_model=False, _tokenizer_type="text", _shrink_seq_lenght=False,
                 text_cuis=False):

    logging.info("Loading sentences datasets...")

    x_train, y_train, _, y_test, train_annotate_sentences, test_annotate_sentences = get_raw_datasets(binary_classifier=binary_classifier)

    if _shrink_seq_lenght:
        sentences_max_lenght = [len(x.split()) for x in x_train]
        sentences_max_lenght = max(sentences_max_lenght)

        if sentences_max_lenght < max_lenght:
            max_lenght = sentences_max_lenght

    input_ids_train, attention_masks_train, segment_ids_train, new_vocab_size = get_transformer_inputs(train_annotate_sentences,
                                                                                                       pretrained_model, max_lenght,
                                                                                                       expand_tokenizer=_expand_tokenizer,
                                                                                                       use_saved_model=_use_saved_model,
                                                                                                       tokenizer_type=_tokenizer_type,
                                                                                                       text_cuis=text_cuis)
    y_train = torch.tensor(y_train)  # multiclass needs longs instead of floats

    dataset = TensorDataset(input_ids_train, attention_masks_train, segment_ids_train, y_train)

    # NOTE: random_split does not work properly with text/tf-idf data so to get the same train/dev
    # NOTE: distribution in ffnn and transformers (needed for ensembles) we use only sklearn train_test_split
    # train_size = int(val_split * len(dataset))
    # dev_size = len(dataset) - train_size
    # train_ds, dev_ds = random_split(dataset, [train_size, dev_size])

    dev_size = 1 - val_split
    train_ds, dev_ds = train_test_split(dataset, test_size=dev_size, random_state=25, shuffle=True)

    input_ids_test, attention_masks_test, segment_ids_test, _ = get_transformer_inputs(test_annotate_sentences,
                                                                                       pretrained_model, max_lenght,
                                                                                       expand_tokenizer=_expand_tokenizer,
                                                                                       use_saved_model=_use_saved_model,
                                                                                       tokenizer_type=_tokenizer_type,
                                                                                       text_cuis=text_cuis)
    y_test = torch.tensor(y_test)

    test_ds = TensorDataset(input_ids_test, attention_masks_test, segment_ids_test, y_test)

    logging.info("Train dataset size: %s", len(train_ds))
    logging.info("Dev dataset size: %s", len(dev_ds))
    logging.info("Test dataset size: %s", len(test_ds))

    number_of_classes = get_number_of_classes(binary_classifier, FC_CLASS_VALUES)

    logging.info("Number of classes: %s", number_of_classes)

    return train_ds, dev_ds, test_ds, number_of_classes, new_vocab_size


def annotate_dataset(model, pretrained_model, binary_classifier, max_lenght, batch_size, device,
                     _dataset=None, _expand_tokenizer=False, _tokenizer_type="text", text_cuis=False):

    if binary_classifier:
        target_field = "sentence_class_score_auto"
    else:
        target_field = "sentence_class_auto"

    if _dataset == "train":
        sentece_filter = FC_TRAIN_SENTENCES_FILTER  # for final fake news article classification
                                                    # subsequently discarded because we would be using instances of previously observed sentences
    if _dataset == "test":
        sentece_filter = FC_TEST_SENTENCES_FILTER
    if _dataset == "new":
        sentece_filter = FC_NEW_SENTENCES_FILTER
    if _dataset == "external":
        sentece_filter = FC_CTT3FR_SENTENCES_FILTER

    logging.info("Loading dataset to annotate...")

    connection = get_connection()
    to_annotate_sentences = select_fields_where(connection, SENTENCES_TABLE, FC_TRAIN_TEST_SENTENCE_FIELDS, sentece_filter)
    if len(to_annotate_sentences) > 0:

        x_test = []
        x_id = []
        for sentence in to_annotate_sentences:
            sentence_id = sentence[0]
            sentence_text = sentence[1]

            x_test.append(sentence_text)
            x_id.append(sentence_id)

        input_ids_test, attention_masks_test, segment_ids_test, _ = get_transformer_inputs(to_annotate_sentences, pretrained_model, max_lenght,
                                                                                           expand_tokenizer=_expand_tokenizer, tokenizer_type=_tokenizer_type,
                                                                                           text_cuis=text_cuis)
        test_ds = TensorDataset(input_ids_test, attention_masks_test, segment_ids_test)

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


def exist_fc_sentences_to_annotate():

    cw_sentence_filter = FC_NEW_SENTENCES_FILTER

    connection = get_connection()
    exist_sentences = exist_where(connection, SENTENCES_TABLE, cw_sentence_filter)

    return exist_sentences
