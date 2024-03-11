import logging

from datetime import datetime

from agent.classifiers.utils.device_mgmt import get_gradient_accumulation_steps
from agent.classifiers.transformer_classifier import transformer_classifier
from agent.classifiers.data.transformer_text_cw_dataset_mgmt import get_datasets as cw_get_datasets
from agent.classifiers.data.transformer_text_cw_dataset_mgmt import annotate_dataset as cw_annotate_dataset
from agent.classifiers.transformer_classifier_text_trainer import train_loop as fd_transformer_train_loop
from agent.classifiers.transformer_classifier_text_trainer import eval_loop as fd_transformer_eval_loop
from agent.data.entities.config import ROOT_LOGGER_ID


logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(ROOT_LOGGER_ID)
logger.setLevel(logging.INFO)


LABEL = "test_20230712"
# LABEL = "test_20240207"
USE_GPU = True
BATCH_SIZE = 16
SEED_VAL = 0
USE_EARLY_STOPPING = True
BINARY_CLASS = [
                True,
                # False
                ]
USE_SAVED_MODEL = True


def get_grad_acc_steps_and_bs(pretrained_model_label):
    if pretrained_model_label in ["bert-large-cased", "bert-large-uncased"]: # , "bionlp/bluebert_pubmed_uncased_L-24_H-1024_A-16"
        return get_gradient_accumulation_steps(BATCH_SIZE)
    else:
        return 1, BATCH_SIZE


def run_cw_grid_search(cw_transformer=False):

    if cw_transformer:
        for binary_classifier in BINARY_CLASS:
            for pretrained_model_label in [
                                        #    "funnel-transformer/intermediate",
                                        #    "bert-base-cased",
                                        #    "bert-base-uncased",
                                        #    "emilyalsentzer/Bio_ClinicalBERT",
                                        #    "dmis-lab/biobert-v1.1",
                                        #    "bionlp/bluebert_pubmed_uncased_L-24_H-1024_A-16",
                                           "bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12",
                                           "albert-base-v2"
                                           ]:

                gradient_accumulation_steps, batch_size = get_grad_acc_steps_and_bs(pretrained_model_label)

                for epochs in [10]:
                    for seed_val_2 in [0, 12, 33, 42, 54, 63, 70, 79, 85, 96]:
                        transformer_classifier(for_task_label="cw",
                                               binary_classifier=binary_classifier,
                                               pretrained_model_label=pretrained_model_label,
                                               get_data_function=cw_get_datasets,
                                               annotate_dataset_function=cw_annotate_dataset,
                                               train_loop_function=fd_transformer_train_loop,
                                               eval_loop_function=fd_transformer_eval_loop,
                                               epochs=epochs,
                                               use_early_stopping=USE_EARLY_STOPPING,
                                               gradient_accumulation_steps=gradient_accumulation_steps,
                                               batch_size=batch_size,
                                               seed_val=seed_val_2,
                                               max_lenght=128,
                                               use_gpu=USE_GPU,
                                               use_saved_model=USE_SAVED_MODEL,
                                               annotate_new_instances=False,
                                               annotate_test_instances=False,
                                               annotate_train_instances=False,
                                               label=LABEL)


if __name__ == "__main__":

    logging.info("Launching grid search...")

    starting_time = datetime.now()

    run_cw_grid_search(cw_transformer=True)

    ending_time = datetime.now()

    logging.info("Total time: %s.", ending_time - starting_time)
