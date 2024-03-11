import logging

from datetime import datetime
from transformers import BertForSequenceClassification

from agent.classifiers.data.tr_lstm_dataset_mgmt import get_datasets, annotate_dataset
from agent.classifiers.tr_lstm_ensemble_classifier import tr_lstm
from agent.classifiers.tr_ensemble_triple_trainer import train_loop, eval_loop

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)


if __name__ == "__main__":
    starting_time = datetime.now()

    tr_lstm(for_task_label="fc",
            binary_classifier=True,
            lstm_cuis=True,
            lstm_classif_vector="hidden",
            lstm_embeddings_dim=250,
            lstm_epochs=25,
            lstm_layers=1,
            lstm_hidden_layer_size=100,
            lstm_bidirectional=True,
            lstm_dropout=0,
            lstm_add_output_layer=True,
            lstm_use_class_weight=False,
            lstm_batch_size=16,
            lstm_seed_val=0,
            tr_pretrained_model_label="bert-base-cased",
            tr_pretrained_model=BertForSequenceClassification,
            tr_tokenizer_type="text",
            tr_max_lenght=250,
            tr_expand_tokenizer=True,
            tr_epochs=10,
            tr_batch_size=16,
            tr_seed_val=0,
            ens_activation="relu",
            ens_hidden_layers=2,
            ens_dropout=0.5,
            ens_epochs=5,
            ens_batch_size=16,
            ens_seed_val=0,
            use_gpu=True,
            label="test_211121",
            get_data_function=get_datasets,
            annotation_function=annotate_dataset,
            train_function=train_loop,
            eval_function=eval_loop,
            use_saved_model=True)


    ending_time = datetime.now()
    logging.info("Total time: %s.", ending_time - starting_time)
