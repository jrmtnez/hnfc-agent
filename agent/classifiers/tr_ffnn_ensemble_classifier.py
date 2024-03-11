import torch
import logging

from datetime import datetime
from torch import nn
from os.path import exists
from transformers import logging as transformers_logging
from transformers import AlbertForSequenceClassification

from agent.data.entities.config import MODELS_CACHE_PATH
from agent.classifiers.ffnn_classifier import FFNNClassifier
from agent.classifiers.utils.random_mgmt import set_random_seed
from agent.classifiers.utils.models_cache_mgmt import get_model
from agent.classifiers.evaluation.evaluation_mgmt import get_evaluation_measures
from agent.classifiers.utils.device_mgmt import get_device


logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)
transformers_logging.set_verbosity_error()


class TransformerFFNNEnsemble(nn.Module):
    def __init__(self, num_labels, transformer_model, transformer2_model, ffnn_model, mode="all",
                 activation="sigmoid", dropout=0.2, hidden_layers=1, last_transformer_layer_index=12,
                 last_transformer2_layer_index=12):
        super(TransformerFFNNEnsemble, self).__init__()

        self.mode = mode
        self.hidden_layers = hidden_layers
        self.last_transformer_layer_index = last_transformer_layer_index    # 12 layers + embedding layer (BERT)
                                                                            # 6 x 3 = 18 layers + embedding layer (funnel)
        self.last_transformer2_layer_index = last_transformer2_layer_index  # 12 layers + embedding layer (BERT)
                                                                            # 6 x 3 = 18 layers + embedding layer (funnel)
        self.transformer_model = transformer_model
        self.transformer2_model = transformer2_model
        self.ffnn_model = ffnn_model

        if self.mode == "transformer":
            self.input_size = self.transformer_model.config.hidden_size
        if self.mode == "transformer2":
            self.input_size = self.transformer2_model.config.hidden_size
        if self.mode == "ffnn":
            self.input_size = ffnn_model.hidden_layer_size
        if self.mode == "tr_ffnn":
            self.input_size = self.transformer_model.config.hidden_size + ffnn_model.hidden_layer_size
        if self.mode == "tr2_ffnn":
            self.input_size = self.transformer2_model.config.hidden_size + ffnn_model.hidden_layer_size
        if self.mode == "tr_tr2":
            self.input_size = self.transformer_model.config.hidden_size + self.transformer2_model.config.hidden_size
        if self.mode == "all":
            self.input_size = self.transformer_model.config.hidden_size + self.transformer2_model.config.hidden_size + ffnn_model.hidden_layer_size


        if self.hidden_layers == 1:
            self.hidden = nn.Linear(self.input_size, num_labels)
        if self.hidden_layers == 2:
            self.hidden1_size = self.input_size // 2
            self.hidden1 = nn.Linear(self.input_size, self.hidden1_size)
            self.hidden2 = nn.Linear(self.hidden1_size, num_labels)
        # if self.hidden_layers == 2: # funciona peor con 3 capas
        #     self.hidden1_size = self.input_size // 2
        #     self.hidden1 = nn.Linear(self.input_size, self.hidden1_size)
        #     self.hidden2_size = self.hidden1_size // 2
        #     self.hidden2 = nn.Linear(self.hidden1_size, self.hidden2_size)
        #     self.hidden3 = nn.Linear(self.hidden2_size, num_labels)

        if activation == "relu":
            self.activation = nn.ReLU()
        if activation == "tanh":
            self.activation = nn.Tanh()
        if activation == "sigmoid":
            self.activation = nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask, segment_ids, input_ids2, attention_mask2, segment_ids2, ffnn_features):
        if self.mode == "transformer":
            with torch.no_grad():
                if segment_ids is None:
                    transformer_output = self.transformer_model(input_ids=input_ids, attention_mask=attention_mask)
                else:
                    transformer_output = self.transformer_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=segment_ids)
            last_transformer_layer = transformer_output.hidden_states[self.last_transformer_layer_index]

            first_token = last_transformer_layer[:, 0, :]
            output = first_token

        if self.mode == "transformer2":
            with torch.no_grad():
                if segment_ids2 is None:
                    transformer2_output = self.transformer2_model(input_ids=input_ids2, attention_mask=attention_mask2)
                else:
                    transformer2_output = self.transformer2_model(input_ids=input_ids2, attention_mask=attention_mask2, token_type_ids=segment_ids2)
            last_transformer2_layer = transformer2_output.hidden_states[self.last_transformer2_layer_index]

            first_token = last_transformer2_layer[:, 0, :]
            output = first_token


        if self.mode == "ffnn":
            with torch.no_grad():
                ffnn_hidden, _ = self.ffnn_model(ffnn_features)
            output = ffnn_hidden

        if self.mode == "tr_ffnn":
            with torch.no_grad():
                if segment_ids is None:
                    transformer_output = self.transformer_model(input_ids=input_ids, attention_mask=attention_mask)
                else:
                    transformer_output = self.transformer_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=segment_ids)
                ffnn_hidden, _ = self.ffnn_model(ffnn_features)

            last_transformer_layer = transformer_output.hidden_states[self.last_transformer_layer_index]
            first_token = last_transformer_layer[:, 0, :]

            output = torch.cat((first_token, ffnn_hidden), dim=1)

        if self.mode == "tr2_ffnn":
            with torch.no_grad():
                if segment_ids2 is None:
                    transformer2_output = self.transformer2_model(input_ids=input_ids2, attention_mask=attention_mask2)
                else:
                    transformer2_output = self.transformer2_model(input_ids=input_ids2, attention_mask=attention_mask2, token_type_ids=segment_ids2)
                ffnn_hidden, _ = self.ffnn_model(ffnn_features)

            last_transformer2_layer = transformer2_output.hidden_states[self.last_transformer2_layer_index]
            first_token2 = last_transformer2_layer[:, 0, :]

            output = torch.cat((first_token2, ffnn_hidden), dim=1)


        if self.mode == "tr_tr2":
            with torch.no_grad():
                if segment_ids is None:
                    transformer_output = self.transformer_model(input_ids=input_ids, attention_mask=attention_mask)
                else:
                    transformer_output = self.transformer_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=segment_ids)

                if segment_ids2 is None:
                    transformer2_output = self.transformer2_model(input_ids=input_ids2, attention_mask=attention_mask2)
                else:
                    transformer2_output = self.transformer2_model(input_ids=input_ids2, attention_mask=attention_mask2, token_type_ids=segment_ids2)

            last_transformer_layer = transformer_output.hidden_states[self.last_transformer_layer_index]
            first_token = last_transformer_layer[:, 0, :]

            last_transformer2_layer = transformer2_output.hidden_states[self.last_transformer2_layer_index]
            first_token2 = last_transformer2_layer[:, 0, :]

            output = torch.cat((first_token, first_token2), dim=1)


        if self.mode == "all":
            with torch.no_grad():
                if segment_ids is None:
                    transformer_output = self.transformer_model(input_ids=input_ids, attention_mask=attention_mask)
                else:
                    transformer_output = self.transformer_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=segment_ids)

                if segment_ids2 is None:
                    transformer2_output = self.transformer2_model(input_ids=input_ids2, attention_mask=attention_mask2)
                else:
                    transformer2_output = self.transformer2_model(input_ids=input_ids2, attention_mask=attention_mask2, token_type_ids=segment_ids2)

                ffnn_hidden, _ = self.ffnn_model(ffnn_features)

            last_transformer_layer = transformer_output.hidden_states[self.last_transformer_layer_index]
            first_token = last_transformer_layer[:, 0, :]

            last_transformer2_layer = transformer2_output.hidden_states[self.last_transformer2_layer_index]
            first_token2 = last_transformer2_layer[:, 0, :]

            output = torch.cat((first_token, first_token2, ffnn_hidden), dim=1)


        output = self.dropout(output)
        if self.hidden_layers == 1:
            output = self.hidden(output)
        if self.hidden_layers == 2:
            output = self.hidden1(output)
            output = self.activation(output)
            output = self.dropout(output)
            output = self.hidden2(output)
        # if self.hidden_layers == 2: # funciona peor con 3 capas
        #     output = self.hidden1(output)
        #     output = self.activation(output)
        #     output = self.hidden2(output)
        #     output = self.activation(output)
        #     output = self.dropout(output)
        #     output = self.hidden3(output)
        output = self.softmax(output)

        return output


def tr_ffnn(for_task_label="",
            binary_classifier=False,
            ffnn_cuis=False,
            ffnn_use_class_weight=True,
            ffnn_activation="sigmoid",
            ffnn_hidden_layer_size=100,
            ffnn_epochs=250,
            ffnn_seed_val=0,
            tr_pretrained_model_label="albert-base-v2",
            tr_tokenizer_type="text",
            tr_text_cuis=False,
            tr_last_transformer_layer_index=12,
            tr_max_lenght=250,
            tr_expand_tokenizer=True,
            tr_epochs=2,
            tr_batch_size=16,
            tr_seed_val=0,
            tr2_pretrained_model_label="albert-base-v2",
            tr2_tokenizer_type="text",
            tr2_text_cuis=False,
            tr2_last_transformer_layer_index=12,
            tr2_max_lenght=250,
            tr2_expand_tokenizer=False,
            tr2_epochs=2,
            tr2_batch_size=16,
            tr2_seed_val=0,
            ens_activation='sigmoid',
            ens_hidden_layers=2,
            ens_dropout=0,
            ens_epochs=250,
            use_early_stopping=True,
            patience=2,
            ens_batch_size=16,
            ens_seed_val=42,
            use_gpu=True,
            label="",
            get_data_function=None,
            annotation_function=None,
            train_function=None,
            eval_function=None,
            use_saved_model=False,
            evaluate_model=True,
            annotate_new_instances=True,
            annotate_test_instances=True,
            annotate_train_instances=True,
            annotate_external_instances=True,
            save_ensemble_model=False,
            learning_rate=5e-4,
            epsilon=1e-8,
            ens_mode="all"):

    use_ffnn = ens_mode in ["ffnn", "tr_ffnn", "tr2_ffnn", "all"]
    use_tr = ens_mode in ["transformer", "tr_ffnn", "tr_tr2", "all"]
    use_tr2 = ens_mode in ["transformer2", "tr2_ffnn", "tr_tr2", "all"]

    tr_mn1 = f"{for_task_label}_tr_bin_{binary_classifier}_pm_{tr_pretrained_model_label}_tt_{tr_tokenizer_type}_"
    tr_mn2 = f"tc_{tr_text_cuis}_ex_{tr_expand_tokenizer}_ml_{tr_max_lenght}_"
    tr_mn3 = f"ep_{tr_epochs}_bs_{tr_batch_size}_sv_{tr_seed_val}_gpu_{use_gpu}_{label}"
    transformer_model_name = "".join([tr_mn1, tr_mn2, tr_mn3]).replace("/", "-")

    tr2_mn1 = f"{for_task_label}_tr_bin_{binary_classifier}_pm_{tr2_pretrained_model_label}_tt_{tr2_tokenizer_type}_"
    tr2_mn2 = f"tc_{tr2_text_cuis}_ex_{tr2_expand_tokenizer}_ml_{tr2_max_lenght}_"
    tr2_mn3 = f"ep_{tr2_epochs}_bs_{tr2_batch_size}_sv_{tr2_seed_val}_gpu_{use_gpu}_{label}"
    transformer2_model_name = "".join([tr2_mn1, tr2_mn2, tr2_mn3]).replace("/", "-")

    ffnn_batch_size = 16

    ffnn_mn1 = f"{for_task_label}_ffnn_bin_{binary_classifier}_cui_{ffnn_cuis}_af_{ffnn_activation}_hl_{ffnn_hidden_layer_size}_"
    ffnn_mn2 = f"cw_{ffnn_use_class_weight}_"
    ffnn_mn3 = f"ep_{ffnn_epochs}_bs_{ffnn_batch_size}_sv_{ffnn_seed_val}_gpu_{use_gpu}_{label}"
    ffnn_model_name = "".join([ffnn_mn1, ffnn_mn2, ffnn_mn3]).replace("/", "-")

    mn2 = ""
    mn3 = ""
    mn4 = ""

    mn1 = f"{for_task_label}_ens_{ens_mode}_bin_{binary_classifier}_"
    if use_ffnn:
        mn2 = f"cui_{ffnn_cuis}_cw_{ffnn_use_class_weight}_"
    if use_tr:
        mn3 = f"tt_{tr_tokenizer_type}_pm_{tr_pretrained_model_label}_tc_{tr_text_cuis}_"
    if use_tr2:
        mn4 = f"tt2_{tr2_tokenizer_type}_pm2_{tr2_pretrained_model_label}_tc2_{tr2_text_cuis}_"
    mn5 = f"hl_{ens_hidden_layers}_ac_{ens_activation}_do_{ens_dropout}_"
    mn6 = f"ep_{ens_epochs}_bs_{ens_batch_size}_sv_{ens_seed_val}_gpu_{use_gpu}_{label}"
    model_name = "".join([mn1, mn2, mn3, mn4, mn5, mn6]).replace("/", "-")

    set_random_seed(ens_seed_val, use_gpu)

    device = get_device(use_gpu)

    train_ds, dev_ds, test_ds, ffnn_input_size, num_labels, new_vocab_size, new_vocab_size2 = get_data_function(tr_pretrained_model_label, tr2_pretrained_model_label,
                                                                                                                tr_max_lenght, tr2_max_lenght,
                                                                                                                binary_classifier=binary_classifier,
                                                                                                                _use_saved_model=use_saved_model,
                                                                                                                _expand_tokenizer=tr_expand_tokenizer,
                                                                                                                _expand_tokenizer2=tr2_expand_tokenizer,
                                                                                                                _tokenizer_type=tr_tokenizer_type,
                                                                                                                _tokenizer_type2=tr2_tokenizer_type,
                                                                                                                _text_cuis=tr_text_cuis,
                                                                                                                _text_cuis2=tr2_text_cuis,
                                                                                                                _cuis=ffnn_cuis)

    # transformer_model = tr_pretrained_model.from_pretrained(tr_pretrained_model_label,
    #                                                         num_labels=num_labels,
    #                                                         output_attentions=False,
    #                                                         output_hidden_states=True)

    transformer_model = None
    if use_tr:
        transformer_model = get_model(tr_pretrained_model_label, binary_classifier, num_labels)

        if tr_expand_tokenizer:
            transformer_model.resize_token_embeddings(new_vocab_size)

        logging.info("Loading tr model: %s", transformer_model_name)
        transformer_model.load_state_dict(torch.load(MODELS_CACHE_PATH + transformer_model_name + ".pt"))

        if use_gpu:
            transformer_model.to(device)

        transformer_model.eval()

    transformer2_model = None
    if use_tr2:
        transformer2_model = get_model(tr2_pretrained_model_label, binary_classifier, num_labels)

        if tr2_expand_tokenizer:
            transformer2_model.resize_token_embeddings(new_vocab_size2)

        logging.info("Loading tr2 model : %s", transformer2_model_name)
        transformer2_model.load_state_dict(torch.load(MODELS_CACHE_PATH + transformer2_model_name + ".pt"))

        if use_gpu:
            transformer2_model.to(device)

        transformer2_model.eval()

    # no parece necesario cuando se utiliza eval()
    # for param in transformer_model.parameters():
    #     param.requires_grad = False

    ffnn_model = None
    if use_ffnn:
        ffnn_model = FFNNClassifier(ffnn_input_size, ffnn_hidden_layer_size, num_labels)
        logging.info("Loading ffnn model: %s", ffnn_model_name)
        ffnn_model.load_state_dict(torch.load(MODELS_CACHE_PATH + ffnn_model_name + ".pt"))
        if use_gpu:
            ffnn_model.to(device)
        ffnn_model.eval()

    model = TransformerFFNNEnsemble(num_labels, transformer_model, transformer2_model, ffnn_model, mode=ens_mode,
                                    activation=ens_activation, dropout=ens_dropout, hidden_layers=ens_hidden_layers,
                                    last_transformer_layer_index=tr_last_transformer_layer_index,
                                    last_transformer2_layer_index=tr2_last_transformer_layer_index)

    if use_gpu:
        model.to(device)

    saved_model_file = MODELS_CACHE_PATH + model_name + ".pt"

    logging.info("Checking saved model: %s", model_name)

    action = ""

    if exists(saved_model_file) and use_saved_model:
        logging.debug("Loading saved model...")
        model.load_state_dict(torch.load(saved_model_file))
        action = "load"
    else:
        logging.debug("Training model...")
        model = train_function(model, learning_rate, epsilon, ens_batch_size, ens_epochs, device, train_ds, dev_ds,
                               for_task_label, binary_classifier, use_early_stopping=use_early_stopping, patience=patience)
        if save_ensemble_model:
            # TODO lo comentamos para la grid search, de lo contradio el disco se llena
            torch.save(model.state_dict(), saved_model_file)
        action = "train"

    LOG_FILE = "temp/run.log"
    log_file_body = f"{datetime.now()}\t{action}\t{model_name}"
    with open(LOG_FILE, "a", encoding="utf-8") as results_file:
        results_file.write(log_file_body + "\n")

    model.eval()
    if test_ds is None:
        test_ds = dev_ds

    if evaluate_model:
        class_predictions, labels = eval_function(model, ens_batch_size, device, test_ds)
        get_evaluation_measures(f"{model_name}", labels, class_predictions,
                                for_task_label, binary_classifier, save_evaluation=True)

    # --- checking binary classification from multiclass predictions ---
    # >>> worse than dedicated binay classifier
    # for i in range(class_predictions.size(dim=0)):
    #     class_predictions[i] = class_predictions[i] == 3
    # for i in range(labels.size(dim=0)):
    #     labels[i] = labels[i] == 3
    # get_evaluation_measures(f"{model_name}", labels, class_predictions, save_evaluation=True)

    if annotation_function is not None:
        if annotate_new_instances:
            annotation_function(model, tr_pretrained_model_label, tr2_pretrained_model_label, binary_classifier, tr_max_lenght, tr2_max_lenght, ens_batch_size, device,
                                dataset="new", tokenizer_type=tr_tokenizer_type, tokenizer_type2=tr2_tokenizer_type, use_saved_model=use_saved_model,
                                expand_tokenizer=tr_expand_tokenizer, expand_tokenizer2=tr2_expand_tokenizer, cuis=ffnn_cuis, text_cuis=tr_text_cuis, text_cuis2=tr2_text_cuis)
            logging.info("New sentences annotated")
        if annotate_test_instances:
            annotation_function(model, tr_pretrained_model_label, tr2_pretrained_model_label, binary_classifier, tr_max_lenght, tr2_max_lenght, ens_batch_size, device,
                                dataset="test", tokenizer_type=tr_tokenizer_type, tokenizer_type2=tr2_tokenizer_type, use_saved_model=use_saved_model,
                                expand_tokenizer=tr_expand_tokenizer, expand_tokenizer2=tr2_expand_tokenizer, cuis=ffnn_cuis, text_cuis=tr_text_cuis, text_cuis2=tr2_text_cuis)
            logging.info("Test dataset annotated")

        if annotate_train_instances:
            annotation_function(model, tr_pretrained_model_label, tr2_pretrained_model_label, binary_classifier, tr_max_lenght, tr2_max_lenght, ens_batch_size, device,
                                dataset="train", tokenizer_type=tr_tokenizer_type, tokenizer_type2=tr2_tokenizer_type, use_saved_model=use_saved_model,
                                expand_tokenizer=tr_expand_tokenizer, expand_tokenizer2=tr2_expand_tokenizer, cuis=ffnn_cuis, text_cuis=tr_text_cuis, text_cuis2=tr2_text_cuis)
            logging.info("Train dataset annotated")
        if annotate_external_instances:
            annotation_function(model, tr_pretrained_model_label, tr2_pretrained_model_label, binary_classifier, tr_max_lenght, tr2_max_lenght, ens_batch_size, device,
                                dataset="external", tokenizer_type=tr_tokenizer_type, tokenizer_type2=tr2_tokenizer_type, use_saved_model=use_saved_model,
                                expand_tokenizer=tr_expand_tokenizer, expand_tokenizer2=tr2_expand_tokenizer, cuis=ffnn_cuis, text_cuis=tr_text_cuis, text_cuis2=tr2_text_cuis)
            logging.info("External dataset annotated")
