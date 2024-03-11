import logging
import fnmatch
import os
import time

from datetime import datetime

from agent.data.entities.config import ROOT_LOGGER_ID
from agent.classifiers.transformer_classifier import transformer_classifier
from agent.classifiers.transformer_classifier_text_trainer import train_loop as text_train_loop
from agent.classifiers.transformer_classifier_text_trainer import eval_loop as text_eval_loop
from agent.classifiers.data.transformer_text_cw_dataset_mgmt import get_datasets as get_cw_datasets
from agent.classifiers.data.transformer_text_cw_dataset_mgmt import annotate_dataset as annotate_cw_dataset
from agent.classifiers.data.transformer_text_cw_dataset_mgmt import exist_cw_sentences_to_annotate
from agent.pipeline.review_level_mgmt import update_sentence_review_level, update_item_review_level
from agent.nlp.spo_extractor import extract_spos_from_sentences, match_tokens_cuis_in_sentences
from agent.nlp.spo_extractor import extract_spos_from_sentences_given_predicate, clear_spos
from agent.nlp.spo_extractor import extract_spos_from_sentences_aux_predicate
from agent.nlp.spo_extractor import extract_spos_from_sentences_given_manual_predicate
from agent.nlp.spo_extractor import mark_no_predicate_external_sentences
from agent.classifiers.fc_preliminary_classifier import do_preliminary_fc_classification
from agent.pipeline.review_level_mgmt import update_sentence_review_level_spo_to_check
from agent.pipeline.review_level_mgmt import update_sentence_review_level_spo_skip_validations
from agent.pipeline.review_level_mgmt import update_sentence_review_level_spo_completed
from agent.classifiers.ffnn_classifier import ffnn
from agent.classifiers.data.transformer_text_triple_fc_dataset_mgmt import get_datasets
from agent.classifiers.data.transformer_text_triple_fc_dataset_mgmt import annotate_dataset
from agent.classifiers.data.transformer_text_triple_fc_dataset_mgmt import exist_fc_sentences_to_annotate
from agent.classifiers.transformer_classifier_triple_trainer import train_loop, eval_loop
from agent.classifiers.transformer_classifier import transformer_classifier
from agent.classifiers.tr_ffnn_ensemble_classifier import tr_ffnn
from agent.classifiers.data.tr_ffnn_dataset_mgmt import get_datasets as get_ens_datasets
from agent.classifiers.data.tr_ffnn_dataset_mgmt import annotate_dataset as annotate_ens_dataset
from agent.classifiers.tr_ensemble_triple_trainer import train_loop as train_ens_loop
from agent.classifiers.tr_ensemble_triple_trainer import eval_loop as eval_ens_loop
from agent.data.entities.config import DEV_LABEL, TEST_LABEL
from agent.data.entities.config import SKIP_VAL_LABEL
from agent.classifiers.fd_naive_item_classifier import fd_naive_classifier, evaluate_items_for_filter


logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(ROOT_LOGGER_ID)
logger.setLevel(logging.INFO)

USE_GPU = False
RETRAIN_COMPONENTS = True
LABEL = "prod_221119"
USE_SAVED_MODEL = True
EVALUATE_MODEL = False
USE_EARLY_STOPPING = True
BATCH_SIZE = 16
RUN_OPTION_BIN = 3
RUN_OPTION_MC = 3


if __name__ == "__main__":

    starting_time = datetime.now()

    if exist_cw_sentences_to_annotate():

        logging.info("Evaluating sentences check-worthiness...")

        transformer_classifier(for_task_label="cw",
                               binary_classifier=True,
                               pretrained_model_label="bert-base-uncased",
                               get_data_function=get_cw_datasets,
                               annotate_dataset_function=annotate_cw_dataset,
                               train_loop_function=text_train_loop,
                               eval_loop_function=text_eval_loop,
                               epochs=2,
                               batch_size=16,
                               use_gpu=USE_GPU,
                               use_saved_model=True,
                               evaluate_model=False,
                               annotate_new_instances=True)

        transformer_classifier(for_task_label="cw",
                               binary_classifier=False,
                               pretrained_model_label="bert-base-cased",
                               get_data_function=get_cw_datasets,
                               annotate_dataset_function=annotate_cw_dataset,
                               train_loop_function=text_train_loop,
                               eval_loop_function=text_eval_loop,
                               epochs=2,
                               batch_size=16,
                               use_gpu=USE_GPU,
                               use_saved_model=True,
                               evaluate_model=False,
                               annotate_new_instances=True)
    else:
        logging.info("No new sentences pending check-worthiness evaluation!!")

    update_sentence_review_level(3, to_level=4, to_level_without_validations=5)
    update_item_review_level()

    logging.info("Extracting SPOs...")

    clear_spos(5)

    for external in [False, True]:
        extract_spos_from_sentences_given_predicate(5, external=external)
        extract_spos_from_sentences(5, external=external)
        extract_spos_from_sentences_aux_predicate(5, external=external)
        extract_spos_from_sentences_given_manual_predicate(5, external=external)
        match_tokens_cuis_in_sentences(5, external=external)
        do_preliminary_fc_classification(external=external)
        extract_spos_from_sentences_given_manual_predicate(6, external=external)
        match_tokens_cuis_in_sentences(6, external=external)
        if external:
            mark_no_predicate_external_sentences(5)

    update_sentence_review_level_spo_to_check(5, 6)
    update_sentence_review_level_spo_skip_validations(5, 7)
    update_sentence_review_level_spo_completed(6, 7)
    update_item_review_level()

    if exist_fc_sentences_to_annotate():

        logging.info("Evaluating sentences truthfulness...")

        # --- binary classification ---

        binary_classifier = True
        cuis = False
        ffnn_use_class_weight = False
        ffnn_activation = "relu"
        ffnn_seed_val = 70
        ffnn_epochs = 250
        ffnn_hidden_layer = 1000

        tr_tokenizer_type = "text"
        tr_epochs = 10
        tr_seed_val = 96
        pretrained_model_label = "bert-base-cased"
        expand_tokenizer = False

        if RETRAIN_COMPONENTS:

            if RUN_OPTION_BIN == 1 or RUN_OPTION_BIN == 0:
                ffnn(for_task_label="fc",
                     binary_classifier=binary_classifier,
                     cuis=cuis,
                     hidden_layer_size=ffnn_hidden_layer,
                     activation_fn=ffnn_activation,
                     use_class_weight=ffnn_use_class_weight,
                     epochs=ffnn_epochs,
                     batch_size=BATCH_SIZE,
                     seed_val=ffnn_seed_val,
                     use_gpu=USE_GPU,
                     use_saved_model=USE_SAVED_MODEL,
                     label=LABEL)

            if RUN_OPTION_BIN == 2 or RUN_OPTION_BIN == 0:
                transformer_classifier(for_task_label="fc",
                                       binary_classifier=binary_classifier,
                                       pretrained_model_label=pretrained_model_label,
                                       tokenizer_type=tr_tokenizer_type,
                                       get_data_function=get_datasets,
                                       annotate_dataset_function=annotate_dataset,
                                       train_loop_function=train_loop,
                                       eval_loop_function=eval_loop,
                                       epochs=tr_epochs,
                                       batch_size=BATCH_SIZE,
                                       seed_val=tr_seed_val,
                                       max_lenght=128,
                                       expand_tokenizer=expand_tokenizer,
                                       use_gpu=USE_GPU,
                                       evaluate_model=EVALUATE_MODEL,
                                       use_saved_model=USE_SAVED_MODEL,
                                       annotate_new_instances=(
                                       RUN_OPTION_BIN == 2),
                                       annotate_test_instances=False,
                                       annotate_train_instances=False,
                                       label=LABEL)

        if RUN_OPTION_BIN == 3 or RUN_OPTION_BIN == 0:

            tr2_seed_val = 33
            ens_activation = "relu"
            ens_hidden_layers = 2
            dropout = 0
            epochs = 10
            seed_val_2 = 54  # precision: 0.692, recall: 0.707, f1: 0.698

            if pretrained_model_label in ["bert-base-cased", "bert-base-uncased", "albert-base-v2"]:
                tr_last_transformer_layer_index = 12
            if pretrained_model_label in ["funnel-transformer/intermediate"]:
                tr_last_transformer_layer_index = 18
            if pretrained_model_label in ["bert-large-cased", "bert-large-uncased"]:
                tr_last_transformer_layer_index = 24

            if RETRAIN_COMPONENTS:

                ffnn(for_task_label="fc",
                     binary_classifier=binary_classifier,
                     cuis=cuis,
                     hidden_layer_size=ffnn_hidden_layer,
                     activation_fn=ffnn_activation,
                     use_class_weight=ffnn_use_class_weight,
                     epochs=ffnn_epochs,
                     batch_size=BATCH_SIZE,
                     seed_val=ffnn_seed_val,
                     use_gpu=USE_GPU,
                     use_saved_model=USE_SAVED_MODEL,
                     label=LABEL)

                transformer_classifier(for_task_label="fc",
                                       binary_classifier=binary_classifier,
                                       pretrained_model_label=pretrained_model_label,
                                       tokenizer_type=tr_tokenizer_type,
                                       get_data_function=get_datasets,
                                       annotate_dataset_function=annotate_dataset,
                                       train_loop_function=train_loop,
                                       eval_loop_function=eval_loop,
                                       epochs=tr_epochs,
                                       batch_size=BATCH_SIZE,
                                       seed_val=tr_seed_val,
                                       max_lenght=128,
                                       expand_tokenizer=expand_tokenizer,
                                       use_gpu=USE_GPU,
                                       evaluate_model=EVALUATE_MODEL,
                                       use_saved_model=USE_SAVED_MODEL,
                                       annotate_new_instances=(
                                       RUN_OPTION_BIN == 2),
                                       annotate_test_instances=False,
                                       annotate_train_instances=False,
                                       label=LABEL)

                transformer_classifier(for_task_label="fc",
                                       binary_classifier=binary_classifier,
                                       pretrained_model_label="bert-base-uncased",
                                       tokenizer_type="spo_variable",
                                       get_data_function=get_datasets,
                                       annotate_dataset_function=annotate_dataset,
                                       train_loop_function=train_loop,
                                       eval_loop_function=eval_loop,
                                       epochs=10,
                                       batch_size=BATCH_SIZE,
                                       seed_val=tr2_seed_val,
                                       max_lenght=128,
                                       expand_tokenizer=False,
                                       text_cuis=True,
                                       use_gpu=USE_GPU,
                                       use_saved_model=USE_SAVED_MODEL,
                                       annotate_new_instances=False,
                                       annotate_test_instances=False,
                                       annotate_train_instances=False,
                                       label=LABEL)

            tr_ffnn(for_task_label="fc",
                    binary_classifier=binary_classifier,
                    ffnn_cuis=cuis,
                    ffnn_use_class_weight=ffnn_use_class_weight,
                    ffnn_activation=ffnn_activation,
                    ffnn_hidden_layer_size=ffnn_hidden_layer,
                    ffnn_seed_val=ffnn_seed_val,
                    tr_pretrained_model_label=pretrained_model_label,
                    tr_expand_tokenizer=expand_tokenizer,
                    tr_tokenizer_type=tr_tokenizer_type,
                    tr_text_cuis=False,
                    tr_last_transformer_layer_index=tr_last_transformer_layer_index,
                    tr_epochs=tr_epochs,
                    tr_batch_size=BATCH_SIZE,
                    tr_seed_val=tr_seed_val,
                    tr_max_lenght=128,
                    tr2_pretrained_model_label="bert-base-uncased",
                    tr2_expand_tokenizer=False,
                    tr2_tokenizer_type="spo_variable",
                    tr2_text_cuis=True,
                    tr2_last_transformer_layer_index=tr_last_transformer_layer_index,
                    tr2_epochs=10,
                    tr2_batch_size=BATCH_SIZE,
                    tr2_seed_val=tr2_seed_val,
                    tr2_max_lenght=128,
                    ens_mode="all",
                    ens_activation=ens_activation,
                    ens_hidden_layers=ens_hidden_layers,
                    ens_dropout=dropout,
                    ens_epochs=epochs,
                    use_early_stopping=USE_EARLY_STOPPING,
                    patience=2,
                    ens_batch_size=BATCH_SIZE,
                    ens_seed_val=seed_val_2,
                    use_gpu=USE_GPU,
                    evaluate_model=EVALUATE_MODEL,
                    annotate_new_instances=(RUN_OPTION_BIN == 3),
                    annotate_test_instances=False,
                    annotate_train_instances=False,
                    annotate_external_instances=False,
                    label=LABEL,
                    get_data_function=get_ens_datasets,
                    annotation_function=annotate_ens_dataset,
                    train_function=train_ens_loop,
                    eval_function=eval_ens_loop,
                    use_saved_model=USE_SAVED_MODEL)

        # --- multiclass classification ---

        binary_classifier = False
        cuis = True
        ffnn_use_class_weight = True
        ffnn_activation = "relu"
        ffnn_seed_val = 70
        ffnn_epochs = 250
        ffnn_hidden_layer = 1000

        tr_tokenizer_type = "text"
        tr_epochs = 10
        tr_seed_val = 96
        pretrained_model_label = "bert-base-cased"
        expand_tokenizer = False

        if RETRAIN_COMPONENTS:

            if RUN_OPTION_MC == 1 or RUN_OPTION_MC == 0:
                ffnn(for_task_label="fc",
                     binary_classifier=binary_classifier,
                     cuis=cuis,
                     hidden_layer_size=ffnn_hidden_layer,
                     activation_fn=ffnn_activation,
                     use_class_weight=ffnn_use_class_weight,
                     epochs=ffnn_epochs,
                     batch_size=BATCH_SIZE,
                     seed_val=ffnn_seed_val,
                     use_gpu=USE_GPU,
                     use_saved_model=USE_SAVED_MODEL,
                     label=LABEL)

            if RUN_OPTION_MC == 2 or RUN_OPTION_MC == 0:
                transformer_classifier(for_task_label="fc",
                                       binary_classifier=binary_classifier,
                                       pretrained_model_label=pretrained_model_label,
                                       tokenizer_type=tr_tokenizer_type,
                                       get_data_function=get_datasets,
                                       annotate_dataset_function=annotate_dataset,
                                       train_loop_function=train_loop,
                                       eval_loop_function=eval_loop,
                                       epochs=tr_epochs,
                                       batch_size=BATCH_SIZE,
                                       seed_val=tr_seed_val,
                                       max_lenght=128,
                                       expand_tokenizer=expand_tokenizer,
                                       use_gpu=USE_GPU,
                                       use_saved_model=USE_SAVED_MODEL,
                                       evaluate_model=EVALUATE_MODEL,
                                       annotate_new_instances=(RUN_OPTION_MC == 2),
                                       annotate_test_instances=False,
                                       annotate_train_instances=False,
                                       label=LABEL)

        if RUN_OPTION_MC == 3 or RUN_OPTION_MC == 0:

            tr2_seed_val = 85
            ens_activation = "relu"
            ens_hidden_layers = 2
            dropout = 0.2
            epochs = 10
            seed_val_2 = 54
            seed_val_2 = 70  # precision: 0.534, recall: 0.525, f1: 0.511

            if pretrained_model_label in ["bert-base-cased", "bert-base-uncased", "albert-base-v2"]:
                tr_last_transformer_layer_index = 12
            if pretrained_model_label in ["funnel-transformer/intermediate"]:
                tr_last_transformer_layer_index = 18
            if pretrained_model_label in ["bert-large-cased", "bert-large-uncased"]:
                tr_last_transformer_layer_index = 24

            if RETRAIN_COMPONENTS:

                ffnn(for_task_label="fc",
                     binary_classifier=binary_classifier,
                     cuis=cuis,
                     hidden_layer_size=ffnn_hidden_layer,
                     activation_fn=ffnn_activation,
                     use_class_weight=ffnn_use_class_weight,
                     epochs=ffnn_epochs,
                     batch_size=BATCH_SIZE,
                     seed_val=ffnn_seed_val,
                     use_gpu=USE_GPU,
                     use_saved_model=USE_SAVED_MODEL,
                     label=LABEL)

                transformer_classifier(for_task_label="fc",
                                       binary_classifier=binary_classifier,
                                       pretrained_model_label=pretrained_model_label,
                                       tokenizer_type=tr_tokenizer_type,
                                       get_data_function=get_datasets,
                                       annotate_dataset_function=annotate_dataset,
                                       train_loop_function=train_loop,
                                       eval_loop_function=eval_loop,
                                       epochs=tr_epochs,
                                       batch_size=BATCH_SIZE,
                                       seed_val=tr_seed_val,
                                       max_lenght=128,
                                       expand_tokenizer=expand_tokenizer,
                                       use_gpu=USE_GPU,
                                       use_saved_model=USE_SAVED_MODEL,
                                       evaluate_model=EVALUATE_MODEL,
                                       annotate_new_instances=(RUN_OPTION_MC == 2),
                                       annotate_test_instances=False,
                                       annotate_train_instances=False,
                                       label=LABEL)

                transformer_classifier(for_task_label="fc",
                                       binary_classifier=binary_classifier,
                                       pretrained_model_label="bert-base-uncased",
                                       tokenizer_type="spo_variable",
                                       get_data_function=get_datasets,
                                       annotate_dataset_function=annotate_dataset,
                                       train_loop_function=train_loop,
                                       eval_loop_function=eval_loop,
                                       epochs=10,
                                       batch_size=BATCH_SIZE,
                                       seed_val=tr2_seed_val,
                                       max_lenght=128,
                                       expand_tokenizer=False,
                                       text_cuis=True,
                                       use_gpu=USE_GPU,
                                       use_saved_model=USE_SAVED_MODEL,
                                       annotate_new_instances=False,
                                       annotate_test_instances=False,
                                       annotate_train_instances=False,
                                       label=LABEL)

            tr_ffnn(for_task_label="fc",
                    binary_classifier=binary_classifier,
                    ffnn_cuis=cuis,
                    ffnn_use_class_weight=ffnn_use_class_weight,
                    ffnn_activation=ffnn_activation,
                    ffnn_hidden_layer_size=ffnn_hidden_layer,
                    ffnn_seed_val=ffnn_seed_val,
                    tr_pretrained_model_label=pretrained_model_label,
                    tr_expand_tokenizer=expand_tokenizer,
                    tr_tokenizer_type=tr_tokenizer_type,
                    tr_text_cuis=False,
                    tr_last_transformer_layer_index=tr_last_transformer_layer_index,
                    tr_epochs=tr_epochs,
                    tr_batch_size=BATCH_SIZE,
                    tr_seed_val=tr_seed_val,
                    tr_max_lenght=128,
                    tr2_pretrained_model_label="bert-base-uncased",
                    tr2_expand_tokenizer=False,
                    tr2_tokenizer_type="spo_variable",
                    tr2_text_cuis=True,
                    tr2_last_transformer_layer_index=tr_last_transformer_layer_index,
                    tr2_epochs=10,
                    tr2_batch_size=BATCH_SIZE,
                    tr2_seed_val=tr2_seed_val,
                    tr2_max_lenght=128,
                    ens_mode="all",
                    ens_activation=ens_activation,
                    ens_hidden_layers=ens_hidden_layers,
                    ens_dropout=dropout,
                    ens_epochs=epochs,
                    use_early_stopping=USE_EARLY_STOPPING,
                    patience=2,
                    ens_batch_size=BATCH_SIZE,
                    ens_seed_val=seed_val_2,
                    use_gpu=USE_GPU,
                    evaluate_model=EVALUATE_MODEL,
                    annotate_new_instances=(RUN_OPTION_MC == 3),
                    annotate_test_instances=False,
                    annotate_train_instances=False,
                    annotate_external_instances=False,
                    label=LABEL,
                    get_data_function=get_ens_datasets,
                    annotation_function=annotate_ens_dataset,
                    train_function=train_ens_loop,
                    eval_function=eval_ens_loop,
                    use_saved_model=USE_SAVED_MODEL)

    else:
        logging.info("No new sentences pending truthfulness evaluation!!")

    update_sentence_review_level(7, to_level_without_validations=8)
    update_item_review_level()

    logging.info("Classifying fake news classifiers on level 8...")
    fd_naive_classifier(review_level=8, skip_validations=True, instance_type="Other", item_filter_label=SKIP_VAL_LABEL)
    fd_naive_classifier(review_level=8, skip_validations=False, instance_type="Dev", item_filter_label=DEV_LABEL)

    logging.info("Evaluating fake news classifiers on level 8...")
    evaluate_items_for_filter(review_level=8, skip_validations=True, instance_type="Other", item_filter_label=SKIP_VAL_LABEL)
    evaluate_items_for_filter(review_level=8, skip_validations=False, instance_type="Dev", item_filter_label=TEST_LABEL)

    update_sentence_review_level(8, to_level=9, to_level_without_validations=9)
    update_item_review_level()

    logging.info("Classifying fake news classifiers on level 9...")
    fd_naive_classifier(review_level=9, skip_validations=True, instance_type="Other", item_filter_label=SKIP_VAL_LABEL)
    fd_naive_classifier(review_level=9, skip_validations=False, instance_type="Dev", item_filter_label=DEV_LABEL)

    logging.info("Evaluating fake news classifiers on level 9...")
    evaluate_items_for_filter(review_level=9, skip_validations=True, instance_type="Other", item_filter_label=SKIP_VAL_LABEL)
    evaluate_items_for_filter(review_level=9, skip_validations=False, instance_type="Dev", item_filter_label=TEST_LABEL)

    ending_time = datetime.now()
    logging.info("Total time: %s", ending_time - starting_time)
