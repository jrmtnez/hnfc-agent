#
# ATENCIÓN, para cargar los archivos pt de julio 2023 se debe utilizar:
# - transformers==4.16.2
# - protobuf==3.20.3 (no comprobado que sea necesario)
#


import logging

from datetime import datetime

from agent.classifiers.utils.device_mgmt import get_gradient_accumulation_steps
from agent.classifiers.transformer_classifier import transformer_classifier
from agent.classifiers.data.transformer_text_cw_dataset_mgmt import get_datasets as cw_get_datasets
from agent.classifiers.data.transformer_text_cw_dataset_mgmt import annotate_dataset as cw_annotate_dataset
from agent.classifiers.transformer_classifier_text_trainer import train_loop as fd_transformer_train_loop
from agent.classifiers.transformer_classifier_text_trainer import eval_loop as fd_transformer_eval_loop
from agent.data.entities.config import ROOT_LOGGER_ID

from agent.pipeline.review_level_mgmt import update_sentence_review_level, update_item_review_level

from agent.nlp.spo_extractor import extract_spos_from_sentences, match_tokens_cuis_in_sentences
from agent.nlp.spo_extractor import extract_spos_from_sentences_given_predicate, clear_spos
from agent.nlp.spo_extractor import extract_spos_from_sentences_aux_predicate
from agent.nlp.spo_extractor import extract_spos_from_sentences_given_manual_predicate
from agent.nlp.spo_extractor import mark_no_predicate_external_sentences
from agent.classifiers.fc_preliminary_classifier import do_preliminary_fc_classification

from agent.pipeline.review_level_mgmt import update_sentence_review_level_spo_to_check, update_sentence_review_level_spo_completed


logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(ROOT_LOGGER_ID)
logger.setLevel(logging.INFO)


# LABEL = "test_20230712"
LABEL = "test_20240301"
USE_GPU = True
EPOCHS = 10
BATCH_SIZE = 16
SEED_VAL = 42
USE_EARLY_STOPPING = True
BINARY_CLASS = [
                True,
                # False
                ]
USE_SAVED_MODEL = True


def get_grad_acc_steps_and_bs(pretrained_model_label):
    if pretrained_model_label in ["bert-large-cased", "bert-large-uncased", "bionlp/bluebert_pubmed_uncased_L-24_H-1024_A-16"]:
        return get_gradient_accumulation_steps(BATCH_SIZE)
    else:
        return 1, BATCH_SIZE


def run_cw_grid_search(cw_transformer=False):

    if cw_transformer:
        for custom_model in [
                             True,
                            #  False
                             ]:
            for binary_classifier in BINARY_CLASS:
                for pretrained_model_label in [
                                               "funnel-transformer/intermediate",
                                            #    "bert-base-cased",
                                            #    "bert-base-uncased",
                                            #    "emilyalsentzer/Bio_ClinicalBERT",
                                            #    "dmis-lab/biobert-v1.1",
                                            # #    "bionlp/bluebert_pubmed_uncased_L-24_H-1024_A-16",
                                            #    "bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12",
                                            #    "albert-base-v2"
                                               ]:

                    gradient_accumulation_steps, batch_size = get_grad_acc_steps_and_bs(pretrained_model_label)

                    for epochs in [EPOCHS]:
                        for seed_val_2 in [
                                        #    0,
                                        #    12,
                                        #    33,
                                        #    42,
                                        #    54,
                                        #    63,
                                        #    70,
                                           79,
                                        #    85,
                                        #    96
                                           ]:
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
                                                   custom_model=custom_model,
                                                   annotate_new_instances=False,
                                                   annotate_test_instances=True,
                                                #    annotate_test_instances=False,
                                                   annotate_train_instances=False,
                                                   label=LABEL)


def launch_spo_extractor():
    clear_spos(5)

    for external in [False]:
        extract_spos_from_sentences_given_predicate(5, external=external, full_pipeline=True)
        extract_spos_from_sentences(5, external=external, full_pipeline=True)
        extract_spos_from_sentences_aux_predicate(5, external=external, full_pipeline=True)
        match_tokens_cuis_in_sentences(5, external=external, full_pipeline=True)


if __name__ == "__main__":

    logging.info("Launching full pipeline...")

    starting_time = datetime.now()

    # run_cw_grid_search(cw_transformer=True)

    # update_sentence_review_level(3, to_level=4, to_level_without_validations=5)
    # update_item_review_level()

    # launch_spo_extractor()

    # update_sentence_review_level_spo_to_check(5, 6, full_pipeline=True)
    # update_sentence_review_level_spo_completed(6, 7, full_pipeline=True)
    update_item_review_level()

    ending_time = datetime.now()

    logging.info("Total time: %s.", ending_time - starting_time)
