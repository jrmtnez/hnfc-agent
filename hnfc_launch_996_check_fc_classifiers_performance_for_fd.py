import logging

from datetime import datetime

from agent.classifiers.utils.device_mgmt import get_gradient_accumulation_steps
from agent.classifiers.transformer_classifier import transformer_classifier
from agent.classifiers.ffnn_classifier import ffnn
from agent.classifiers.tr_ffnn_ensemble_classifier import tr_ffnn
from agent.classifiers.data.transformer_text_triple_fc_dataset_mgmt import get_datasets as get_fc_transformer_datasets
from agent.classifiers.data.transformer_text_triple_fc_dataset_mgmt import annotate_dataset as annotate_fc_transformer_dataset
from agent.classifiers.data.tr_ffnn_dataset_mgmt import get_datasets as get_fc_tr_ffnn_datasets
from agent.classifiers.data.tr_ffnn_dataset_mgmt import annotate_dataset as annotate_fc_tr_ffnn_dataset
from agent.classifiers.transformer_classifier_triple_trainer import train_loop as fc_transformer_train_loop
from agent.classifiers.transformer_classifier_triple_trainer import eval_loop as fc_transformer_eval_loop
from agent.classifiers.tr_ensemble_triple_trainer import train_loop as tr_ensemble_triple_train_loop
from agent.classifiers.tr_ensemble_triple_trainer import eval_loop as tr_ensemble_triple_eval_loop
from agent.data.entities.config import ROOT_LOGGER_ID


logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(ROOT_LOGGER_ID)
logger.setLevel(logging.INFO)


LABEL = "test_20230712"
LABEL = "test_20240101"
USE_GPU = True
BATCH_SIZE = 16
SEED_VAL = 0
USE_EARLY_STOPPING = True
BINARY_CLASS = [True,
                # False
                ]
USE_SAVED_MODEL = False


def get_last_transformer_layer_index(pretrained_model_label):
    if pretrained_model_label in ["bert-base-cased", "bert-base-uncased", "albert-base-v2", "emilyalsentzer/Bio_ClinicalBERT",
                                  "dmis-lab/biobert-v1.1", "bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12"]:
        return 12
    if pretrained_model_label in ["funnel-transformer/intermediate"]:
        return 18
    if pretrained_model_label in ["bert-large-cased", "bert-large-uncased", "bionlp/bluebert_pubmed_uncased_L-24_H-1024_A-16"]:
        return 24


def get_grad_acc_steps_and_bs(pretrained_model_label):
    if pretrained_model_label in ["bert-large-cased", "bert-large-uncased", "bionlp/bluebert_pubmed_uncased_L-24_H-1024_A-16"]:
        return get_gradient_accumulation_steps(BATCH_SIZE)
    else:
        return 1, BATCH_SIZE


def run_fc_grid_search(fc_transformer=False,
                       fc_ffnn=False,
                       fc_ensemble_transformer_ffnn=False):

    # ------------------------------
    # --- sentence fact checking ---
    # ------------------------------

    if fc_transformer:
        for binary_classifier in BINARY_CLASS:
            for pretrained_model_label in [
                                        #    "funnel-transformer/intermediate",
                                        #    "bert-base-cased",
                                        #    "bert-base-uncased",
                                        #    "bert-large-cased",
                                        #    "bert-large-uncased",
                                        #    "emilyalsentzer/Bio_ClinicalBERT",
                                        #    "dmis-lab/biobert-v1.1",
                                        #    "bionlp/bluebert_pubmed_uncased_L-24_H-1024_A-16",
                                           "bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12",
                                        #    "albert-base-v2",
                                           ]:

                gradient_accumulation_steps, batch_size = get_grad_acc_steps_and_bs(pretrained_model_label)

                for tokenizer_type in [
                                       "text",
                                    #    "spo_variable",
                                    #    "spo_fixed"
                                       ]:
                    if tokenizer_type == "text":
                        TEXT_CUIS_OPTIONS = [False]
                    else:
                        TEXT_CUIS_OPTIONS = [
                                             True,
                                             False
                                             ]
                    for text_cuis in TEXT_CUIS_OPTIONS:
                        for expand_tokenizer in [
                                                #  True,
                                                 False
                                                 ]:
                            for epochs in [10]:
                                # for seed_val_2 in range(100):
                                # for seed_val_2 in [0, 12, 33, 42, 54, 63, 70, 79, 85, 96]:
                                for seed_val_2 in [7, 12]:
                                    transformer_classifier(for_task_label="fc",
                                                           binary_classifier=binary_classifier,
                                                           pretrained_model_label=pretrained_model_label,
                                                           get_data_function=get_fc_transformer_datasets,
                                                           annotate_dataset_function=annotate_fc_transformer_dataset,
                                                           train_loop_function=fc_transformer_train_loop,
                                                           eval_loop_function=fc_transformer_eval_loop,
                                                           epochs=epochs,
                                                           use_early_stopping=USE_EARLY_STOPPING,
                                                           patience=2,
                                                           gradient_accumulation_steps=gradient_accumulation_steps,
                                                           batch_size=batch_size,
                                                           seed_val=seed_val_2,
                                                           max_lenght=128,
                                                           tokenizer_type=tokenizer_type,
                                                           text_cuis=text_cuis,
                                                           expand_tokenizer=expand_tokenizer,
                                                           use_gpu=USE_GPU,
                                                           use_saved_model=USE_SAVED_MODEL,
                                                           annotate_new_instances=False,
                                                           annotate_test_instances=False,
                                                           annotate_train_instances=False,
                                                           label=LABEL)


    if fc_transformer:
        for binary_classifier in BINARY_CLASS:
            for pretrained_model_label in [
                                        #    "funnel-transformer/intermediate",
                                        #    "bert-base-cased",
                                        #    "bert-base-uncased",
                                        #    "bert-large-cased",
                                        #    "bert-large-uncased",
                                        #    "emilyalsentzer/Bio_ClinicalBERT",
                                        #    "dmis-lab/biobert-v1.1",
                                        #    "bionlp/bluebert_pubmed_uncased_L-24_H-1024_A-16",
                                           "bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12",
                                        #    "albert-base-v2",
                                           ]:

                gradient_accumulation_steps, batch_size = get_grad_acc_steps_and_bs(pretrained_model_label)

                for tokenizer_type in [
                                    #    "text",
                                    #    "spo_variable",
                                       "spo_fixed"
                                       ]:
                    if tokenizer_type == "text":
                        TEXT_CUIS_OPTIONS = [False]
                    else:
                        TEXT_CUIS_OPTIONS = [
                                             True,
                                            #  False
                                             ]
                    for text_cuis in TEXT_CUIS_OPTIONS:
                        for expand_tokenizer in [
                                                #  True,
                                                 False
                                                 ]:
                            for epochs in [10]:
                                # for seed_val_2 in range(100):
                                # for seed_val_2 in [0, 12, 33, 42, 54, 63, 70, 79, 85, 96]:
                                for seed_val_2 in [96]:
                                    transformer_classifier(for_task_label="fc",
                                                           binary_classifier=binary_classifier,
                                                           pretrained_model_label=pretrained_model_label,
                                                           get_data_function=get_fc_transformer_datasets,
                                                           annotate_dataset_function=annotate_fc_transformer_dataset,
                                                           train_loop_function=fc_transformer_train_loop,
                                                           eval_loop_function=fc_transformer_eval_loop,
                                                           epochs=epochs,
                                                           use_early_stopping=USE_EARLY_STOPPING,
                                                           patience=2,
                                                           gradient_accumulation_steps=gradient_accumulation_steps,
                                                           batch_size=batch_size,
                                                           seed_val=seed_val_2,
                                                           max_lenght=128,
                                                           tokenizer_type=tokenizer_type,
                                                           text_cuis=text_cuis,
                                                           expand_tokenizer=expand_tokenizer,
                                                           use_gpu=USE_GPU,
                                                           use_saved_model=USE_SAVED_MODEL,
                                                           annotate_new_instances=False,
                                                           annotate_test_instances=False,
                                                           annotate_train_instances=False,
                                                           label=LABEL)


    if fc_ffnn:
        for binary_classifier in BINARY_CLASS:
            for cuis in [
                         False,
                         True
                         ]:
                for hidden_layer_size in [
                                        #   100,
                                        #   500,
                                        #   1000,
                                          1500
                                          ]:
                    for activation_fn in [
                                        #   "sigmoid",
                                          "tanh",
                                        #   "relu"
                                          ]:
                        if binary_classifier:
                            use_class_weight_options = [
                                                        # True,
                                                        False
                                                        ]
                        else:
                            use_class_weight_options = [True, False]
                        for use_class_weight in use_class_weight_options:
                            for epochs in [250]:
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
                                    ffnn(for_task_label="fc",
                                         binary_classifier=binary_classifier,
                                         cuis=cuis,
                                         hidden_layer_size=hidden_layer_size,
                                         activation_fn=activation_fn,
                                         use_class_weight=use_class_weight,
                                         epochs=epochs,
                                         use_early_stopping=USE_EARLY_STOPPING,
                                         patience=10,
                                         batch_size=BATCH_SIZE,
                                         seed_val=seed_val_2,
                                         use_gpu=USE_GPU,
                                         label=LABEL,
                                         use_saved_model=USE_SAVED_MODEL)

    if fc_ensemble_transformer_ffnn:
        for binary_classifier in BINARY_CLASS:
            for ens_mode in [
                            #  "tr_tr2",
                             "all"
                             ]:
                if ens_mode == "tr_tr2":
                    cuis_options = [False]
                else:
                    cuis_options = [True,
                                    # False
                                    ]
                for cuis in cuis_options:
                    for tr_tokenizer_type in ["text",
                                            #   "spo_variable",
                                            #   "spo_fixed"
                                              ]:
                        for pretrained_model_label in [#"funnel-transformer/intermediate",
                                                    #    "bert-base-cased",
                                                       # "bert-base-uncased",
                                                       # "bert-large-cased",
                                                       # "bert-large-uncased",
                                                       # "emilyalsentzer/Bio_ClinicalBERT",
                                                    #    "dmis-lab/biobert-v1.1",
                                                       # "bionlp/bluebert_pubmed_uncased_L-24_H-1024_A-16",
                                                       "bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12",
                                                    #    "albert-base-v2",
                                                       ]:
                            for ens_activation in ["relu"]:
                                for ens_hidden_layers in [1, 2]:
                                    if binary_classifier:
                                        use_class_weight_options = [False]
                                        ffnn_hidden_layer_size = 100
                                        ffnn_seed_val = 70
                                        tr_seed_val = 63
                                        tr_expand_tokenizer = False
                                    else:
                                        use_class_weight_options = [True]
                                        ffnn_hidden_layer_size = 500
                                        ffnn_seed_val = 42
                                        tr_seed_val = 85
                                        tr_expand_tokenizer = False
                                    for use_class_weight in use_class_weight_options:
                                        for tr2_tokenizer_type in [
                                                                #    "spo_variable",
                                                                   "spo_fixed"
                                                                   ]:
                                            if binary_classifier:
                                                # tr2_pretrained_model_label = "bionlp/bluebert_pubmed_uncased_L-24_H-1024_A-16"
                                                tr2_pretrained_model_label = "bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12"
                                                if tr2_tokenizer_type in ["spo_variable"]:
                                                    tr2_seed_val = 12
                                                    tr2_expand_tokenizer = False
                                                else:
                                                    tr2_seed_val = 96
                                                    tr2_expand_tokenizer = False
                                            else:
                                                tr2_pretrained_model_label = "bert-base-uncased"
                                                tr2_seed_val = 33,
                                                tr2_expand_tokenizer = True
                                            dropout_options = [
                                                            #    0,
                                                               0.2,
                                                            #    0.5
                                                               ]
                                            for dropout in dropout_options:
                                                for epochs in [10]:
                                                    for seed_val_2 in range(100):
                                                    # for seed_val_2 in [0, 12, 33, 42, 54, 63, 70, 79, 85, 96]:
                                                        tr_ffnn(for_task_label="fc",
                                                                binary_classifier=binary_classifier,
                                                                ffnn_cuis=cuis,
                                                                ffnn_use_class_weight=use_class_weight,
                                                                ffnn_activation="tanh",
                                                                ffnn_hidden_layer_size=ffnn_hidden_layer_size,
                                                                ffnn_seed_val=ffnn_seed_val,
                                                                tr_pretrained_model_label=pretrained_model_label,
                                                                tr_expand_tokenizer=tr_expand_tokenizer,
                                                                tr_tokenizer_type=tr_tokenizer_type,
                                                                tr_text_cuis=False,
                                                                tr_last_transformer_layer_index=get_last_transformer_layer_index(pretrained_model_label),
                                                                tr_epochs=10,
                                                                tr_batch_size=16,
                                                                tr_seed_val=tr_seed_val,
                                                                tr_max_lenght=128,
                                                                tr2_pretrained_model_label=tr2_pretrained_model_label,
                                                                tr2_expand_tokenizer=tr2_expand_tokenizer,
                                                                tr2_tokenizer_type=tr2_tokenizer_type,
                                                                tr2_text_cuis=True,
                                                                tr2_last_transformer_layer_index=get_last_transformer_layer_index(tr2_pretrained_model_label),
                                                                tr2_epochs=10,
                                                                tr2_batch_size=16,
                                                                tr2_seed_val=tr2_seed_val,
                                                                tr2_max_lenght=128,
                                                                ens_mode=ens_mode,
                                                                ens_activation=ens_activation,
                                                                ens_hidden_layers=ens_hidden_layers,
                                                                ens_dropout=dropout,
                                                                ens_epochs=epochs,
                                                                use_early_stopping=USE_EARLY_STOPPING,
                                                                patience=2,
                                                                ens_batch_size=BATCH_SIZE,
                                                                ens_seed_val=seed_val_2,
                                                                use_gpu=USE_GPU,
                                                                label=LABEL,
                                                                get_data_function=get_fc_tr_ffnn_datasets,
                                                                annotation_function=annotate_fc_tr_ffnn_dataset,
                                                                train_function=tr_ensemble_triple_train_loop,
                                                                eval_function=tr_ensemble_triple_eval_loop,
                                                                annotate_train_instances=False,
                                                                annotate_test_instances=False,
                                                                annotate_new_instances=False,
                                                                annotate_external_instances=False,
                                                                use_saved_model=USE_SAVED_MODEL)


if __name__ == "__main__":

    logging.info("Launching grid search...")

    starting_time = datetime.now()

    run_fc_grid_search(fc_transformer=True,
                       fc_ffnn=True,
                       fc_ensemble_transformer_ffnn=False)

    ending_time = datetime.now()

    logging.info("Total time: %s.", ending_time - starting_time)
