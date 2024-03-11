import torch
import logging

from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split

from agent.data.entities.config import FC_CLASS_VALUES
from agent.classifiers.utils.class_mgmt import get_number_of_classes
from agent.classifiers.data.transformer_text_triple_fc_dataset_mgmt  import get_raw_datasets
from agent.nlp.text_sequencer import Sequencer

MAX_VOCAB_SIZE = 15000
SEQ_LEN = 40


def get_lstm_inputs(x_train_text, x_test_text, lowercase=True):
    sequencer = Sequencer(x_train_text, MAX_VOCAB_SIZE, tokenizer="nltk", lang="en", lower=lowercase)
    logging.info("Found %s unique tokens.", sequencer.unique_word_count)
    x_train_seqs = sequencer.fit_on_text(x_train_text, SEQ_LEN)
    x_test_seqs = sequencer.fit_on_text(x_test_text, SEQ_LEN)

    return x_train_seqs, x_test_seqs


def get_max_seq_len(x_seqs):
    max_seq_len = 0
    for sentence in x_seqs:
        seq_len = 0
        for token in sentence:
            if token > 0:
                seq_len += 1
        if seq_len > max_seq_len:
            max_seq_len = seq_len
    return max_seq_len


def get_datasets(val_split=0.8, lowercase=True, binary_classifier=False, cuis=False, unk_tokens=False):
    logging.info("Loading datasets...")

    x_train, y_train, x_test, y_test, _, _ = get_raw_datasets(binary_classifier=binary_classifier, cuis=cuis, unk_tokens=unk_tokens)

    x_train_seqs, x_test_seqs = get_lstm_inputs(x_train, x_test, lowercase=lowercase)

    logging.info("Max. sequence lenght: %s", get_max_seq_len(x_train_seqs))

    x_train_seqs = torch.tensor(x_train_seqs)
    x_test_seqs = torch.tensor(x_test_seqs)
    y_train = torch.tensor(y_train)
    y_test = torch.tensor(y_test)

    logging.info("Sequence lenght: %s", len(x_test_seqs[0]))
    logging.info("%s train instances", len(x_train_seqs))
    logging.info("%s test instances", len(x_test_seqs))

    dataset = TensorDataset(x_train_seqs, y_train)
    dev_size = 1 - val_split
    train_ds, dev_ds = train_test_split(dataset, test_size=dev_size, random_state=25, shuffle=True)

    test_ds = TensorDataset(x_test_seqs, y_test)

    logging.info("Train dataset size: %s", len(train_ds))
    logging.info("Dev dataset size: %s", len(dev_ds))
    logging.info("Test dataset size: %s", len(test_ds))

    number_of_classes = get_number_of_classes(binary_classifier, FC_CLASS_VALUES)

    logging.info("Number of classes: %s", number_of_classes)

    return train_ds, dev_ds, test_ds, number_of_classes
