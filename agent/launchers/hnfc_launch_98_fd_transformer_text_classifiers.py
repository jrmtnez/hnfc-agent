import logging

from datetime import datetime
from transformers import BertForSequenceClassification

from agent.data.entities.config import DEV_LABEL, TEST_LABEL, EXTERNAL_LABEL, CTT3_LABEL, CTT3FR_LABEL, CTT3SIM_LABEL, CTT3HEALTH_LABEL
from agent.classifiers.transformer_classifier import transformer_classifier
from agent.classifiers.transformer_classifier_text_trainer import train_loop, eval_loop
from agent.classifiers.data.wrappers.transformer_text_fd_dataset_dev import get_datasets as get_fd_datasets
from agent.classifiers.data.wrappers.transformer_text_fd_dataset_test import get_datasets as get_fd_datasets_test
from agent.classifiers.data.wrappers.transformer_text_fd_dataset_external import get_datasets as get_fd_external_datasets
from agent.classifiers.data.wrappers.transformer_text_fd_dataset_ct21t3fr import get_datasets as get_fd_ct21t3fr_datasets
from agent.classifiers.data.wrappers.transformer_text_fd_dataset_ct21t3 import get_datasets as get_fd_ct21t3_datasets
from agent.classifiers.data.wrappers.transformer_text_fd_dataset_ct21t3health import get_datasets as get_fd_ct21t3health_datasets
from agent.classifiers.data.transformer_text_fd_dataset_mgmt_ct21t3sim import get_datasets as get_fd_ct21t3sim_datasets

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)


if __name__ == "__main__":
    starting_time = datetime.now()

    binary_classifier = True

    # F1: 0.855
    logging.info("Evaluating transformers on %s dataset for fake-news detection task...", DEV_LABEL)
    transformer_classifier(for_task_label="fd",
                           binary_classifier=binary_classifier,
                           pretrained_model_label="bert-base-uncased",
                           pretrained_model=BertForSequenceClassification,
                           get_data_function=get_fd_datasets,
                           annotate_dataset_function=None,
                           train_loop_function=train_loop,
                           eval_loop_function=eval_loop,
                           max_lenght=150,
                           epochs=5,
                           batch_size=16,
                           seed_val=0,
                           val_split=None,
                           use_gpu=False,
                           use_saved_model=True,
                           evaluate_model=True,
                           annotate_new_instances=False,
                           label=DEV_LABEL)

    logging.info("-" * 80)
    logging.info("Evaluating transformers on %s dataset for fake-news detection task...", TEST_LABEL)
    transformer_classifier(for_task_label="fd",
                           binary_classifier=binary_classifier,
                           pretrained_model_label="bert-base-uncased",
                           pretrained_model=BertForSequenceClassification,
                           get_data_function=get_fd_datasets_test,
                           annotate_dataset_function=None,
                           train_loop_function=train_loop,
                           eval_loop_function=eval_loop,
                           max_lenght=150,
                           epochs=5,
                           batch_size=16,
                           seed_val=0,
                           val_split=None,
                           use_gpu=False,
                           use_saved_model=True,
                           evaluate_model=True,
                           annotate_new_instances=False,
                           label=TEST_LABEL)

    logging.info("-" * 80)
    logging.info("Evaluating transformers on %s dataset for fake-news detection task...", EXTERNAL_LABEL)
    transformer_classifier(for_task_label="fd",
                           binary_classifier=binary_classifier,
                           pretrained_model_label="bert-base-uncased",
                           pretrained_model=BertForSequenceClassification,
                           get_data_function=get_fd_external_datasets,
                           annotate_dataset_function=None,
                           train_loop_function=train_loop,
                           eval_loop_function=eval_loop,
                           max_lenght=150,
                           epochs=5,
                           batch_size=16,
                           seed_val=0,
                           val_split=None,
                           use_gpu=False,
                           use_saved_model=True,
                           evaluate_model=True,
                           annotate_new_instances=False,
                           label=EXTERNAL_LABEL)

    logging.info("-" * 80)
    logging.info("Evaluating transformers on %s dataset for fake-news detection task...", CTT3FR_LABEL)
    transformer_classifier(for_task_label="fd",
                           binary_classifier=binary_classifier,
                           pretrained_model_label="bert-base-uncased",
                           pretrained_model=BertForSequenceClassification,
                           get_data_function=get_fd_ct21t3fr_datasets,
                           annotate_dataset_function=None,
                           train_loop_function=train_loop,
                           eval_loop_function=eval_loop,
                           max_lenght=150,
                           epochs=5,
                           batch_size=16,
                           seed_val=0,
                           val_split=None,
                           use_gpu=False,
                           use_saved_model=True,
                           evaluate_model=True,
                           annotate_new_instances=False,
                           label=CTT3FR_LABEL)

    logging.info("-" * 80)
    logging.info("Evaluating transformers on %s dataset for fake-news detection task...", CTT3_LABEL)
    transformer_classifier(for_task_label="fd",
                           binary_classifier=binary_classifier,
                           pretrained_model_label="bert-base-uncased",
                           pretrained_model=BertForSequenceClassification,
                           get_data_function=get_fd_ct21t3_datasets,
                           annotate_dataset_function=None,
                           train_loop_function=train_loop,
                           eval_loop_function=eval_loop,
                           max_lenght=150,
                           epochs=5,
                           batch_size=16,
                           seed_val=0,
                           val_split=None,
                           use_gpu=False,
                           use_saved_model=True,
                           evaluate_model=True,
                           annotate_new_instances=False,
                           label=CTT3_LABEL)

    logging.info("-" * 80)
    logging.info("Evaluating transformers on %s dataset for fake-news detection task...", CTT3SIM_LABEL)
    transformer_classifier(for_task_label="fd",
                           binary_classifier=binary_classifier,
                           pretrained_model_label="bert-base-uncased",
                           pretrained_model=BertForSequenceClassification,
                           get_data_function=get_fd_ct21t3sim_datasets,
                           annotate_dataset_function=None,
                           train_loop_function=train_loop,
                           eval_loop_function=eval_loop,
                           max_lenght=150,
                           epochs=5,
                           batch_size=16,
                           seed_val=0,
                           val_split=None,
                           use_gpu=False,
                           use_saved_model=True,
                           evaluate_model=True,
                           annotate_new_instances=False,
                           label=CTT3SIM_LABEL)

    logging.info("-" * 80)
    logging.info("Evaluating transformers on %s dataset for fake-news detection task...", CTT3HEALTH_LABEL)
    transformer_classifier(for_task_label="fd",
                           binary_classifier=binary_classifier,
                           pretrained_model_label="bert-base-uncased",
                           pretrained_model=BertForSequenceClassification,
                           get_data_function=get_fd_ct21t3health_datasets,
                           annotate_dataset_function=None,
                           train_loop_function=train_loop,
                           eval_loop_function=eval_loop,
                           max_lenght=150,
                           epochs=5,
                           batch_size=16,
                           seed_val=0,
                           val_split=None,
                           use_gpu=False,
                           use_saved_model=True,
                           evaluate_model=True,
                           annotate_new_instances=False,
                           label=CTT3HEALTH_LABEL)

    ending_time = datetime.now()
    logging.info("Total time: %s.", ending_time - starting_time)
