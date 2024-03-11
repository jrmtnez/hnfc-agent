import logging

from datetime import datetime
from transformers import AlbertForSequenceClassification

from agent.classifiers.tr_ffnn_ensemble_classifier import tr_ffnn
from agent.classifiers.data.tr_ffnn_dataset_mgmt import get_datasets, annotate_dataset
from agent.classifiers.tr_ensemble_triple_trainer import train_loop, eval_loop

from agent.classifiers.data.transformer_text_triple_fc_dataset_mgmt import get_datasets as tr_get_datasets
from agent.classifiers.data.transformer_text_triple_fc_dataset_mgmt import annotate_dataset as tr_annotate_dataset
from agent.classifiers.transformer_classifier_triple_trainer import train_loop as tr_train_loop
from agent.classifiers.transformer_classifier_triple_trainer import eval_loop as tr_eval_loop
from agent.classifiers.transformer_classifier import transformer_classifier

from agent.classifiers.ffnn_classifier import ffnn

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)


if __name__ == "__main__":
    starting_time = datetime.now()


    # -----------------------------------------------------------------------------------------------------------------------
    # paper binary classifier F1: 0.694
    # fc_ens_tr_ffnn_bin_True_cui_False_tt_text_hl_2_ac_sigmoid_do_0_ep_10_bs_16_sv_33_gpu_True_test_20220213
    # -----------------------------------------------------------------------------------------------------------------------

    USE_GPU = True

    BINARY_CLASSIFIER = True

    TR_PRETRAINED_MODEL_LABEL = "albert-base-v2"
    SelectedTransformerModel = AlbertForSequenceClassification
    TR_TOKENIZER_TYPE = "text"
    TR_EPOCHS = 10
    TR_EXPAND_TOKENIZER = False
    TR_SEED_VAL = 54

    FFNN_CUIS = False
    FFNN_USE_CLASS_WEIGHT = True
    FFNN_ACTIVATION = "sigmoid"
    FFNN_HIDDEN_LAYER_SIZE = 100
    FFNN_EPOCHS = 250
    FFNN_SEED_VAL = 70

    ENS_ACTIVATION = "sigmoid"
    ENS_DROPOUT = 0
    ENS_EPOCHS = 10
    ENS_SEED_VAL = 33


    # embedded transformer F1: 0.668
    # fc_tr_bin_True_pm_albert-base-v2_tt_text_ex_False_ml_128_ep_10_bs_16_sv_54_gpu_True_test_20220213
    transformer_classifier(for_task_label="fc",
                           binary_classifier=BINARY_CLASSIFIER,
                           pretrained_model_label=TR_PRETRAINED_MODEL_LABEL,
                           pretrained_model=SelectedTransformerModel,
                           tokenizer_type=TR_TOKENIZER_TYPE,
                           get_data_function=tr_get_datasets,
                           annotate_dataset_function=tr_annotate_dataset,
                           train_loop_function=tr_train_loop,
                           eval_loop_function=tr_eval_loop,
                           epochs=TR_EPOCHS,
                           batch_size=16,
                           seed_val=TR_SEED_VAL,
                           max_lenght=128,
                           expand_tokenizer=TR_EXPAND_TOKENIZER,
                           use_gpu=USE_GPU,
                           use_saved_model=True,
                           evaluate_model=True,
                           annotate_new_instances=False,
                           annotate_test_instances=False,
                           annotate_train_instances=False,
                           label="test_20220213")

    # embedded ffnn F1: 0.412
    # fc_ffnn_bin_True_cui_False_af_sigmoid_hl_100_cw_True_ep_250_bs_16_sv_70_gpu_True_test_20220213
    ffnn(for_task_label="fc",
         binary_classifier=BINARY_CLASSIFIER,
         cuis=FFNN_CUIS,
         hidden_layer_size=FFNN_HIDDEN_LAYER_SIZE,
         activation_fn=FFNN_ACTIVATION,
         use_class_weight=FFNN_USE_CLASS_WEIGHT,
         epochs=FFNN_EPOCHS,
         batch_size=16,
         seed_val=FFNN_SEED_VAL,
         use_gpu=USE_GPU,
         use_saved_model=True,
         label="test_20220213")

    tr_ffnn(for_task_label="fc",
            binary_classifier=BINARY_CLASSIFIER,
            ffnn_cuis=FFNN_CUIS,
            ffnn_use_class_weight=FFNN_USE_CLASS_WEIGHT,
            ffnn_activation=FFNN_ACTIVATION,
            ffnn_epochs=FFNN_EPOCHS,
            ffnn_seed_val=FFNN_SEED_VAL,
            tr_pretrained_model_label=TR_PRETRAINED_MODEL_LABEL,
            tr_pretrained_model=SelectedTransformerModel,
            tr_tokenizer_type=TR_TOKENIZER_TYPE,
            tr_expand_tokenizer=TR_EXPAND_TOKENIZER,
            tr_max_lenght=128,
            tr_epochs=TR_EPOCHS,
            tr_batch_size=16,
            tr_seed_val=TR_SEED_VAL,
            ens_activation=ENS_ACTIVATION,
            ens_hidden_layers=2,
            ens_dropout=ENS_DROPOUT,
            ens_epochs=ENS_EPOCHS,
            ens_batch_size=16,
            ens_seed_val=ENS_SEED_VAL,
            ens_mode="both",
            use_gpu=USE_GPU,
            label="test_20220213",
            get_data_function=get_datasets,
            annotation_function=annotate_dataset,
            train_function=train_loop,
            eval_function=eval_loop,
            annotate_new_instances=False,
            annotate_test_instances=True,
            annotate_train_instances=False,
            annotate_external_instances=False,
            use_saved_model=True)



    # -----------------------------------------------------------------------------------------------------------------------
    # paper multiclass classifier F1: 0.483
    # fc_ens_tr_ffnn_bin_False_cui_True_tt_spo_variable_hl_2_ac_sigmoid_do_0.2_ep_10_bs_16_sv_54_gpu_True_test_20220213
    # -----------------------------------------------------------------------------------------------------------------------

    BINARY_CLASSIFIER = False

    TR_PRETRAINED_MODEL_LABEL = "albert-base-v2"
    SelectedTransformerModel = AlbertForSequenceClassification
    TR_TOKENIZER_TYPE = "spo_variable"
    TR_EPOCHS = 10
    TR_EXPAND_TOKENIZER = False
    TR_SEED_VAL = 54

    FFNN_CUIS = True
    FFNN_USE_CLASS_WEIGHT = False
    FFNN_ACTIVATION = "sigmoid"
    FFNN_HIDDEN_LAYER_SIZE = 100
    FFNN_EPOCHS = 250
    FFNN_SEED_VAL = 70

    ENS_ACTIVATION = "sigmoid"
    ENS_DROPOUT = 0.2
    ENS_EPOCHS = 10
    ENS_SEED_VAL = 54


    # embedded transformer F1: 0.440
    # fc_tr_bin_False_pm_albert-base-v2_tt_spo_variable_ex_False_ml_128_ep_10_bs_16_sv_54_gpu_True_test_20220213
    transformer_classifier(for_task_label="fc",
                           binary_classifier=BINARY_CLASSIFIER,
                           pretrained_model_label=TR_PRETRAINED_MODEL_LABEL,
                           pretrained_model=SelectedTransformerModel,
                           tokenizer_type=TR_TOKENIZER_TYPE,
                           get_data_function=tr_get_datasets,
                           annotate_dataset_function=tr_annotate_dataset,
                           train_loop_function=tr_train_loop,
                           eval_loop_function=tr_eval_loop,
                           epochs=TR_EPOCHS,
                           batch_size=16,
                           seed_val=TR_SEED_VAL,
                           max_lenght=128,
                           expand_tokenizer=TR_EXPAND_TOKENIZER,
                           use_gpu=USE_GPU,
                           use_saved_model=True,
                           evaluate_model=True,
                           annotate_new_instances=False,
                           annotate_test_instances=False,
                           annotate_train_instances=False,
                           label="test_20220213")

    # embedded ffnn F1: 0.154
    # fc_ffnn_bin_False_cui_True_af_sigmoid_hl_100_cw_False_ep_250_bs_16_sv_70_gpu_True_test_20220213
    ffnn(for_task_label="fc",
         binary_classifier=BINARY_CLASSIFIER,
         cuis=FFNN_CUIS,
         hidden_layer_size=FFNN_HIDDEN_LAYER_SIZE,
         activation_fn=FFNN_ACTIVATION,
         use_class_weight=FFNN_USE_CLASS_WEIGHT,
         epochs=FFNN_EPOCHS,
         batch_size=16,
         seed_val=FFNN_SEED_VAL,
         use_gpu=USE_GPU,
         use_saved_model=True,
         label="test_20220213")

    tr_ffnn(for_task_label="fc",
            binary_classifier=BINARY_CLASSIFIER,
            ffnn_cuis=FFNN_CUIS,
            ffnn_use_class_weight=FFNN_USE_CLASS_WEIGHT,
            ffnn_activation=FFNN_ACTIVATION,
            ffnn_epochs=FFNN_EPOCHS,
            ffnn_seed_val=FFNN_SEED_VAL,
            tr_pretrained_model_label=TR_PRETRAINED_MODEL_LABEL,
            tr_pretrained_model=SelectedTransformerModel,
            tr_tokenizer_type=TR_TOKENIZER_TYPE,
            tr_expand_tokenizer=TR_EXPAND_TOKENIZER,
            tr_max_lenght=128,
            tr_epochs=TR_EPOCHS,
            tr_batch_size=16,
            tr_seed_val=TR_SEED_VAL,
            ens_activation=ENS_ACTIVATION,
            ens_hidden_layers=2,
            ens_dropout=ENS_DROPOUT,
            ens_epochs=ENS_EPOCHS,
            ens_batch_size=16,
            ens_seed_val=ENS_SEED_VAL,
            ens_mode="both",
            use_gpu=USE_GPU,
            label="test_20220213",
            get_data_function=get_datasets,
            annotation_function=annotate_dataset,
            train_function=train_loop,
            eval_function=eval_loop,
            annotate_new_instances=False,
            annotate_test_instances=True,
            annotate_train_instances=False,
            annotate_external_instances=False,
            use_saved_model=True)

    ending_time = datetime.now()
    logging.info("Total time: %s.", ending_time - starting_time)
