from agent.classifiers.tr_ffnn_ensemble_classifier import tr_ffnn
from agent.classifiers.fd_naive_item_classifier import fd_naive_classifier
from agent.classifiers.data.tr_ffnn_dataset_mgmt import get_datasets as get_fc_tr_ffnn_datasets
from agent.classifiers.data.tr_ffnn_dataset_mgmt import annotate_dataset as annotate_fc_tr_ffnn_dataset
from agent.classifiers.tr_ensemble_triple_trainer import train_loop as tr_ensemble_triple_train_loop
from agent.classifiers.tr_ensemble_triple_trainer import eval_loop as tr_ensemble_triple_eval_loop
from agent.data.entities.config import DEV_LABEL, EXTERNAL_LABEL


LABEL = "test_20230712"
USE_GPU = True
BATCH_SIZE = 16
USE_EARLY_STOPPING = True
USE_SAVED_MODEL = True


def run_over_fc_binary_annotation():

    binary_classifier = True

    ffnn_epochs = 250
    cuis = True
    ffnn_use_class_weight = False
    ffnn_activation = "tanh"
    ffnn_seed_val = 70
    ffnn_hidden_layer = 100

    ffnn_seed_val = 0  # prueba 1
    ffnn_hidden_layer = 500  # prueba 1
    ffnn_use_class_weight = True  # prueba 1
    ffnn_seed_val = 79  # prueba 2
    ffnn_hidden_layer = 1500  # prueba 2
    ffnn_use_class_weight = False  # prueba 2

    tr_epochs = 10
    tr_tokenizer_type = "text"
    tr_seed_val = 12
    tr_seed_val = 7  # de pruebas significancia estadística
    tr_expand_tokenizer = False
    tr_pretrained_model_label = "bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12"

    tr2_tokenizer_type = "spo_fixed"
    tr2_seed_val = 96
    tr2_expand_tokenizer = False
    tr2_pretrained_model_label = "bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12"

    ens_activation = "relu"
    ens_hidden_layers = 2
    ens_hidden_layers = 1  # de pruebas significancia estadística
    dropout = 0.2
    epochs = 10

    if tr_pretrained_model_label in ["bert-base-cased", "bert-base-uncased", "albert-base-v2", "emilyalsentzer/Bio_ClinicalBERT", "dmis-lab/biobert-v1.1", "bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12"]:
        tr_last_transformer_layer_index = 12
    if tr_pretrained_model_label in ["funnel-transformer/intermediate"]:
        tr_last_transformer_layer_index = 18
    if tr_pretrained_model_label in ["bert-large-cased", "bert-large-uncased", "bionlp/bluebert_pubmed_uncased_L-24_H-1024_A-16"]:
        tr_last_transformer_layer_index = 24

    if tr2_pretrained_model_label in ["bert-base-cased", "bert-base-uncased", "albert-base-v2", "emilyalsentzer/Bio_ClinicalBERT", "dmis-lab/biobert-v1.1", "bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12"]:
        tr2_last_transformer_layer_index = 12
    if tr2_pretrained_model_label in ["funnel-transformer/intermediate"]:
        tr2_last_transformer_layer_index = 18
    if tr2_pretrained_model_label in ["bert-large-cased", "bert-large-uncased", "bionlp/bluebert_pubmed_uncased_L-24_H-1024_A-16"]:
        tr2_last_transformer_layer_index = 24

    for seed_val_2 in [0, 12, 33, 42, 54, 63, 70, 79, 85, 96]:

        tr_ffnn(for_task_label="fc",
                binary_classifier=binary_classifier,
                ffnn_cuis=cuis,
                ffnn_use_class_weight=ffnn_use_class_weight,
                ffnn_activation=ffnn_activation,
                ffnn_hidden_layer_size=ffnn_hidden_layer,
                ffnn_seed_val=ffnn_seed_val,
                ffnn_epochs=ffnn_epochs,
                tr_pretrained_model_label=tr_pretrained_model_label,
                tr_expand_tokenizer=tr_expand_tokenizer,
                tr_tokenizer_type=tr_tokenizer_type,
                tr_text_cuis=False,
                tr_last_transformer_layer_index=tr_last_transformer_layer_index,
                tr_epochs=tr_epochs,
                tr_batch_size=BATCH_SIZE,
                tr_seed_val=tr_seed_val,
                tr_max_lenght=128,
                tr2_pretrained_model_label=tr2_pretrained_model_label,
                tr2_expand_tokenizer=tr2_expand_tokenizer,
                tr2_tokenizer_type=tr2_tokenizer_type,
                tr2_text_cuis=True,
                tr2_last_transformer_layer_index=tr2_last_transformer_layer_index,
                tr2_epochs=tr_epochs,
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
                label=LABEL,
                get_data_function=get_fc_tr_ffnn_datasets,
                annotation_function=annotate_fc_tr_ffnn_dataset,
                train_function=tr_ensemble_triple_train_loop,
                eval_function=tr_ensemble_triple_eval_loop,
                annotate_train_instances=False,
                annotate_test_instances=True,           # para resultados artículo
                annotate_new_instances=False,
                annotate_external_instances=True,      # para comparación con dataset recovery
                save_ensemble_model=False,
                use_saved_model=USE_SAVED_MODEL)

        fd_naive_classifier(review_level=9, skip_validations=False, instance_type="Dev", item_filter_label=DEV_LABEL, evaluate_items=True,
                            binary=True, use_multiclass_predictions_for_binary_classification=False)
        fd_naive_classifier(review_level=9, skip_validations=True, instance_type="Other", item_filter_label=EXTERNAL_LABEL, evaluate_items=True,
                            binary=True, use_multiclass_predictions_for_binary_classification=False)


def run_over_mc_fc_annotation():

    binary_classifier = False

    ffnn_epochs = 250
    cuis = True
    ffnn_use_class_weight = True
    ffnn_activation = "tanh"
    ffnn_seed_val = 42
    ffnn_hidden_layer = 500

    tr_epochs = 10
    tr_tokenizer_type = "text"
    tr_seed_val = 85
    tr_expand_tokenizer = False
    tr_pretrained_model_label = "funnel-transformer/intermediate"

    tr2_tokenizer_type = "spo_variable"
    tr2_seed_val = 33
    tr2_expand_tokenizer = True
    tr2_pretrained_model_label = "bert-base-uncased"

    ens_activation = "relu"
    ens_hidden_layers = 1
    dropout = 0
    epochs = 10

    if tr_pretrained_model_label in ["bert-base-cased", "bert-base-uncased", "albert-base-v2", "emilyalsentzer/Bio_ClinicalBERT", "dmis-lab/biobert-v1.1"]:
        tr_last_transformer_layer_index = 12
    if tr_pretrained_model_label in ["funnel-transformer/intermediate"]:
        tr_last_transformer_layer_index = 18
    if tr_pretrained_model_label in ["bert-large-cased", "bert-large-uncased", "bionlp/bluebert_pubmed_uncased_L-24_H-1024_A-16"]:
        tr_last_transformer_layer_index = 24

    if tr2_pretrained_model_label in ["bert-base-cased", "bert-base-uncased", "albert-base-v2", "emilyalsentzer/Bio_ClinicalBERT", "dmis-lab/biobert-v1.1"]:
        tr2_last_transformer_layer_index = 12
    if tr2_pretrained_model_label in ["funnel-transformer/intermediate"]:
        tr2_last_transformer_layer_index = 18
    if tr2_pretrained_model_label in ["bert-large-cased", "bert-large-uncased", "bionlp/bluebert_pubmed_uncased_L-24_H-1024_A-16"]:
        tr2_last_transformer_layer_index = 24

    for seed_val_2 in [0, 12, 33, 42, 54, 63, 70, 79, 85, 96]:

        tr_ffnn(for_task_label="fc",
                binary_classifier=binary_classifier,
                ffnn_cuis=cuis,
                ffnn_use_class_weight=ffnn_use_class_weight,
                ffnn_activation=ffnn_activation,
                ffnn_hidden_layer_size=ffnn_hidden_layer,
                ffnn_seed_val=ffnn_seed_val,
                ffnn_epochs=ffnn_epochs,
                tr_pretrained_model_label=tr_pretrained_model_label,
                tr_expand_tokenizer=tr_expand_tokenizer,
                tr_tokenizer_type=tr_tokenizer_type,
                tr_text_cuis=False,
                tr_last_transformer_layer_index=tr_last_transformer_layer_index,
                tr_epochs=tr_epochs,
                tr_batch_size=16,
                tr_seed_val=tr_seed_val,
                tr_max_lenght=128,
                tr2_pretrained_model_label=tr2_pretrained_model_label,
                tr2_expand_tokenizer=tr2_expand_tokenizer,
                tr2_tokenizer_type=tr2_tokenizer_type,
                tr2_text_cuis=True,
                tr2_last_transformer_layer_index=tr2_last_transformer_layer_index,
                tr2_epochs=10,
                tr2_batch_size=16,
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
                label=LABEL,
                get_data_function=get_fc_tr_ffnn_datasets,
                annotation_function=annotate_fc_tr_ffnn_dataset,
                train_function=tr_ensemble_triple_train_loop,
                eval_function=tr_ensemble_triple_eval_loop,
                annotate_train_instances=False,
                annotate_test_instances=True,           # para resultados artículo
                annotate_new_instances=False,
                annotate_external_instances=True,      # para comparación con dataset recovery
                save_ensemble_model=True,
                use_saved_model=USE_SAVED_MODEL)

        # fd_naive_classifier(review_level=9, skip_validations=False, instance_type="Dev", item_filter_label=DEV_LABEL, evaluate_items=True,
        #                     binary=True, use_multiclass_predictions_for_binary_classification=True)
        # fd_naive_classifier(review_level=9, skip_validations=True, instance_type="Other", item_filter_label=EXTERNAL_LABEL, evaluate_items=True,
        #                     binary=True, use_multiclass_predictions_for_binary_classification=True)
        fd_naive_classifier(review_level=9, skip_validations=False, instance_type="Dev", item_filter_label=DEV_LABEL, evaluate_items=True,
                            binary=False)
        # fd_naive_classifier(review_level=9, skip_validations=True, instance_type="Other", item_filter_label=EXTERNAL_LABEL, evaluate_items=True,
        #                     binary=False)


if __name__ == "__main__":

    run_over_fc_binary_annotation()
    # run_over_mc_fc_annotation()
