import logging

from datetime import datetime
from transformers import BertForSequenceClassification

from agent.data.entities.config import DEV_LABEL, TEST_LABEL, EXTERNAL_LABEL, CTT3_LABEL, CTT3FR_LABEL, CTT3HEALTH_LABEL
from agent.classifiers.tr_ffnn_ensemble_classifier import tr_ffnn
from agent.classifiers.data.wrappers.tr_ffnn_fd_dataset_dev import get_datasets as get_fd_dev_datasets
from agent.classifiers.data.wrappers.tr_ffnn_fd_dataset_test import get_datasets as get_fd_test_datasets
from agent.classifiers.data.wrappers.tr_ffnn_fd_dataset_external import get_datasets as get_fd_external_datasets
from agent.classifiers.data.wrappers.tr_ffnn_fd_dataset_ct21t3fr import get_datasets as get_fd_ct21t3fr_datasets
from agent.classifiers.data.wrappers.tr_ffnn_fd_dataset_ct21t3 import get_datasets as get_fd_ct21t3_datasets
from agent.classifiers.data.wrappers.tr_ffnn_fd_dataset_ct21t3health import get_datasets as get_fd_ct21t3health_datasets
from agent.classifiers.tr_ensemble_text_trainer import train_loop, eval_loop

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)


if __name__ == "__main__":
    starting_time = datetime.now()

    # F1: 0.803
    logging.info("Evaluating transformer-ffnn ensemble on %s dataset for fake-news detection task...", DEV_LABEL)
    tr_ffnn(for_task_label="fd",
            binary_classifier=True,
            ffnn_cuis=False,
            ffnn_use_class_weight=False,
            ffnn_epochs=10,
            ffnn_activation="relu",
            tr_pretrained_model_label="bert-base-uncased",
            tr_pretrained_model=BertForSequenceClassification,
            tr_tokenizer_type="text",
            tr_expand_tokenizer = False,
            tr_max_lenght=150,
            tr_epochs=5,
            tr_batch_size=16,
            tr_seed_val=0,
            ens_activation="sigmoid",
            ens_hidden_layers=1,
            ens_dropout=0.5,
            ens_epochs=10,
            ens_batch_size=16,
            ens_seed_val=0,
            use_gpu=True,
            label=DEV_LABEL,
            get_data_function=get_fd_dev_datasets,
            annotation_function=None,
            train_function=train_loop,
            eval_function=eval_loop,
            use_saved_model=False)

    logging.info("-" * 80)
    logging.info("Evaluating transformer-ffnn ensemble on %s dataset for fake-news detection task...", TEST_LABEL)
    tr_ffnn(for_task_label="fd",
            binary_classifier=True,
            ffnn_cuis=False,
            ffnn_epochs=100,
            tr_pretrained_model_label="bert-base-uncased",
            tr_pretrained_model=BertForSequenceClassification,
            tr_tokenizer_type="text",
            tr_expand_tokenizer = False,
            tr_max_lenght=150,
            tr_epochs=5,
            tr_batch_size=16,
            tr_seed_val=0,
            ens_activation="sigmoid",
            ens_hidden_layers=1,
            ens_dropout=0.5,
            ens_epochs=10,
            ens_batch_size=16,
            ens_seed_val=0,
            use_gpu=True,
            label=TEST_LABEL,
            get_data_function=get_fd_test_datasets,
            annotation_function=None,
            train_function=train_loop,
            eval_function=eval_loop,
            use_saved_model=False)

    logging.info("-" * 80)
    logging.info("Evaluating transformer-ffnn ensemble on %s dataset for fake-news detection task...", EXTERNAL_LABEL)
    tr_ffnn(for_task_label="fd",
            binary_classifier=True,
            ffnn_cuis=False,
            ffnn_epochs=100,
            tr_pretrained_model_label="bert-base-uncased",
            tr_pretrained_model=BertForSequenceClassification,
            tr_tokenizer_type="text",
            tr_expand_tokenizer = False,
            tr_max_lenght=150,
            tr_epochs=5,
            tr_batch_size=16,
            tr_seed_val=0,
            ens_activation="sigmoid",
            ens_hidden_layers=1,
            ens_dropout=0.5,
            ens_epochs=10,
            ens_batch_size=16,
            ens_seed_val=0,
            use_gpu=True,
            label=EXTERNAL_LABEL,
            get_data_function=get_fd_external_datasets,
            annotation_function=None,
            train_function=train_loop,
            eval_function=eval_loop,
            use_saved_model=False)

    logging.info("-" * 80)
    logging.info("Evaluating transformer-ffnn ensemble on %s dataset for fake-news detection task...", CTT3FR_LABEL)
    tr_ffnn(for_task_label="fd",
            binary_classifier=True,
            ffnn_cuis=False,
            ffnn_epochs=100,
            tr_pretrained_model_label="bert-base-uncased",
            tr_pretrained_model=BertForSequenceClassification,
            tr_tokenizer_type="text",
            tr_expand_tokenizer = False,
            tr_max_lenght=150,
            tr_epochs=5,
            tr_batch_size=16,
            tr_seed_val=0,
            ens_activation="sigmoid",
            ens_hidden_layers=1,
            ens_dropout=0.5,
            ens_epochs=10,
            ens_batch_size=16,
            ens_seed_val=0,
            use_gpu=True,
            label=CTT3FR_LABEL,
            get_data_function=get_fd_ct21t3fr_datasets,
            annotation_function=None,
            train_function=train_loop,
            eval_function=eval_loop,
            use_saved_model=False)

    logging.info("-" * 80)
    logging.info("Evaluating transformer-ffnn ensemble on %s dataset for fake-news detection task...", CTT3_LABEL)
    tr_ffnn(for_task_label="fd",
            binary_classifier=True,
            ffnn_cuis=False,
            ffnn_epochs=100,
            tr_pretrained_model_label="bert-base-uncased",
            tr_pretrained_model=BertForSequenceClassification,
            tr_tokenizer_type="text",
            tr_expand_tokenizer = False,
            tr_max_lenght=150,
            tr_epochs=5,
            tr_batch_size=16,
            tr_seed_val=0,
            ens_activation="sigmoid",
            ens_hidden_layers=1,
            ens_dropout=0.5,
            ens_epochs=10,
            ens_batch_size=16,
            ens_seed_val=0,
            use_gpu=True,
            label=CTT3_LABEL,
            get_data_function=get_fd_ct21t3_datasets,
            annotation_function=None,
            train_function=train_loop,
            eval_function=eval_loop,
            use_saved_model=False)

    logging.info("Evaluating transformer-ffnn ensemble on %s dataset for fake-news detection task...", CTT3HEALTH_LABEL)
    tr_ffnn(for_task_label="fd",
            binary_classifier=True,
            ffnn_cuis=False,
            ffnn_use_class_weight=False,
            ffnn_epochs=10,
            ffnn_activation="relu",
            tr_pretrained_model_label="bert-base-uncased",
            tr_pretrained_model=BertForSequenceClassification,
            tr_tokenizer_type="text",
            tr_expand_tokenizer = False,
            tr_max_lenght=150,
            tr_epochs=5,
            tr_batch_size=16,
            tr_seed_val=0,
            ens_activation="sigmoid",
            ens_hidden_layers=1,
            ens_dropout=0.5,
            ens_epochs=10,
            ens_batch_size=16,
            ens_seed_val=0,
            use_gpu=True,
            label=DEV_LABEL,
            get_data_function=get_fd_ct21t3health_datasets,
            annotation_function=None,
            train_function=train_loop,
            eval_function=eval_loop,
            use_saved_model=False)

    ending_time = datetime.now()
    logging.info("Total time: %s.", ending_time - starting_time)
