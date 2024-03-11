import logging

from datetime import datetime

from agent.classifiers.utils.device_mgmt import get_gradient_accumulation_steps
from agent.classifiers.transformer_classifier import transformer_classifier
from agent.classifiers.data.wrappers.transformer_text_fd_dataset_dev import get_datasets as get_fd_datasets_dev
from agent.classifiers.data.wrappers.transformer_text_fd_dataset_external import get_datasets as get_fd_external_datasets
from agent.classifiers.transformer_classifier_text_trainer import train_loop as fd_transformer_train_loop
from agent.classifiers.transformer_classifier_text_trainer import eval_loop as fd_transformer_eval_loop
from agent.data.entities.config import DEV_LABEL, EXTERNAL_LABEL
from agent.data.entities.config import ROOT_LOGGER_ID


logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(ROOT_LOGGER_ID)
logger.setLevel(logging.INFO)


LABEL = "test_20230712"
LABEL = "test_20240225"  # full pipeline
                         # para full pipeline se ha quitado el filtro sobre productos needs_revision = false
USE_GPU = True
BATCH_SIZE = 16
SEED_VAL = 0
USE_EARLY_STOPPING = True
BINARY_CLASS = [True,
                # False
                ]
USE_SAVED_MODEL = True


def get_grad_acc_steps_and_bs(pretrained_model_label):
    if pretrained_model_label in ["bert-large-cased", "bert-large-uncased", "bionlp/bluebert_pubmed_uncased_L-24_H-1024_A-16"]:
        return get_gradient_accumulation_steps(BATCH_SIZE)
    else:
        return 1, BATCH_SIZE


def run_fd_grid_search():

    # --------------------------------
    # --- item fake news detection ---
    # --------------------------------

    for get_data_functions in [
                               (DEV_LABEL, get_fd_datasets_dev),        # test
                            #    (DEV_LABEL, get_fd_external_datasets),   # recovery
                               ]:
        for binary_classifier in BINARY_CLASS:
            for pretrained_model_label in [
                                           "funnel-transformer/intermediate",
                                           "bert-base-cased",
                                           "bert-base-uncased",
                                           "emilyalsentzer/Bio_ClinicalBERT",
                                           "dmis-lab/biobert-v1.1",
                                        #    "bionlp/bluebert_pubmed_uncased_L-24_H-1024_A-16",
                                           "bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12",
                                           "albert-base-v2"
                                           ]:

                gradient_accumulation_steps, batch_size = get_grad_acc_steps_and_bs(pretrained_model_label)

                for max_lenght in [150, 200, 250]:
                    for epochs in [10]:
                        for seed_val_2 in [0, 12, 33, 42, 54, 63, 70, 79, 85, 96]:
                            transformer_classifier(for_task_label="fd",
                                                   binary_classifier=binary_classifier,
                                                   pretrained_model_label=pretrained_model_label,
                                                   get_data_function=get_data_functions[1],
                                                   annotate_dataset_function=None,
                                                   train_loop_function=fd_transformer_train_loop,
                                                   eval_loop_function=fd_transformer_eval_loop,
                                                   max_lenght=max_lenght,
                                                   epochs=epochs,
                                                   use_early_stopping=USE_EARLY_STOPPING,
                                                   batch_size=batch_size,
                                                   gradient_accumulation_steps=gradient_accumulation_steps,
                                                   val_split=None,
                                                   use_gpu=USE_GPU,
                                                   seed_val=seed_val_2,
                                                   use_saved_model=USE_SAVED_MODEL,
                                                   evaluate_model=True,
                                                   annotate_new_instances=False,
                                                   label=get_data_functions[0])


if __name__ == "__main__":

    logging.info("Launching grid search...")

    starting_time = datetime.now()

    run_fd_grid_search()

    ending_time = datetime.now()
    logging.info("Total time: %s.", ending_time - starting_time)
