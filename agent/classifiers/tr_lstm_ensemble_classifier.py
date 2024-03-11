import torch
import logging

from torch import nn
from os.path import exists
from transformers import logging as transformers_logging

from agent.data.entities.config import MODELS_CACHE_PATH
from agent.classifiers.lstm_classifier import LSTMClassifierOutput, LSTMClassifierHidden
from agent.classifiers.utils.random_mgmt import set_random_seed
from agent.classifiers.utils.models_cache_mgmt import get_model
from agent.classifiers.evaluation.evaluation_mgmt import get_evaluation_measures
from agent.classifiers.utils.device_mgmt import get_device


logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)
transformers_logging.set_verbosity_error()


class TransformerLSTMEnsemble(nn.Module):
    def __init__(self, num_labels, transformer_model, lstm_model, mode="lstm",
                 activation="sigmoid", dropout=0.2, hidden_layers=1):
        super(TransformerLSTMEnsemble, self).__init__()

        self.mode = mode
        self.hidden_layers = hidden_layers

        self.transformer_model = transformer_model
        self.lstm_model = lstm_model

        if self.mode == "transformer":
            self.input_size = self.transformer_model.config.hidden_size
        if self.mode == "lstm":
            self.input_size = lstm_model.hidden_layer_size
        if self.mode == "both":
            self.input_size = self.transformer_model.config.hidden_size + lstm_model.hidden_layer_size

        if self.hidden_layers == 1:
            self.hidden = nn.Linear(self.input_size, num_labels)
        if self.hidden_layers == 2:
            self.hidden1_size = self.input_size // 2
            self.hidden1 = nn.Linear(self.input_size, self.hidden1_size)
            self.hidden2 = nn.Linear(self.hidden1_size, num_labels)
        if activation == "relu":
            self.activation = nn.ReLU()
        if activation == "tanh":
            self.activation = nn.Tanh()
        if activation == "sigmoid":
            self.activation = nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask, segment_ids, lstm_features):
        if self.mode == "transformer":
            with torch.no_grad():
                transformer_output = self.transformer_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=segment_ids)
            last_transformer_layer = transformer_output.hidden_states[12]  # 12 layers + embedding layer
            first_token = last_transformer_layer[:, 0, :]
            output = first_token

        if self.mode == "lstm":
            with torch.no_grad():
                lstm_hidden, _ = self.lstm_model(lstm_features)
            output = lstm_hidden
            # output = [batch_size, hidden_dim]

        if self.mode == "both":
            with torch.no_grad():
                transformer_output = self.transformer_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=segment_ids)
                lstm_hidden, _ = self.lstm_model(lstm_features)

            last_transformer_layer = transformer_output.hidden_states[12]
            first_token = last_transformer_layer[:, 0, :]

            output = torch.cat((first_token, lstm_hidden), dim=1)

        output = self.dropout(output)
        if self.hidden_layers == 1:
            output = self.hidden(output)
        if self.hidden_layers == 2:
            output = self.hidden1(output)
            output = self.activation(output)
            output = self.dropout(output)
            output = self.hidden2(output)
        output = self.softmax(output)

        return output


def tr_lstm(for_task_label="",
            binary_classifier=False,
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
            tr_pretrained_model=None,
            tr_tokenizer_type="text",
            tr_max_lenght=250,
            tr_expand_tokenizer=True,
            tr_epochs=10,
            tr_batch_size=16,
            tr_seed_val=0,
            ens_hidden_layers=2,
            ens_activation='sigmoid',
            ens_dropout=0,
            ens_epochs=250,
            use_early_stopping=True,
            ens_batch_size=16,
            ens_seed_val=42,
            use_gpu=True,
            label="",
            get_data_function=None,
            annotation_function=None,
            train_function=None,
            eval_function=None,
            learning_rate=5e-4,
            epsilon=1e-8,
            ens_mode="both",
            use_saved_model=False,
            annotate_new_sentences=True,
            annotate_test_sentences=True,
            annotate_train_sentences=True):

    tr_mn1 = f"{for_task_label}_tr_bin_{binary_classifier}_pm_{tr_pretrained_model_label}_tt_{tr_tokenizer_type}_"
    tr_mn2 = f"ex_{tr_expand_tokenizer}_ml_{tr_max_lenght}_"
    tr_mn3 = f"ep_{tr_epochs}_bs_{tr_batch_size}_sv_{tr_seed_val}_gpu_{use_gpu}_{label}"
    transformer_model_name = "".join([tr_mn1, tr_mn2, tr_mn3]).replace("/", "-")

    lstm_mn1 = f"{for_task_label}_lstm_bin_{binary_classifier}_ed_{lstm_embeddings_dim}_cv_{lstm_classif_vector}_ll_{lstm_layers}_"
    lstm_mn2 = f"bi_{lstm_bidirectional}_do_{lstm_dropout}_ol_{lstm_add_output_layer}_hd_{lstm_hidden_layer_size}_cui_{lstm_cuis}_"
    lstm_mn3 = f"cw_{lstm_use_class_weight}_ep_{lstm_epochs}_bs_{lstm_batch_size}_sv_{lstm_seed_val}_gpu_{use_gpu}_{label}"
    lstm_model_name = "".join([lstm_mn1, lstm_mn2, lstm_mn3]).replace("/", "-")

    mn1 = f"{for_task_label}_ens_tr_lstm_bin_{binary_classifier}_cui_{lstm_cuis}_cv_{lstm_classif_vector}_tt_{tr_tokenizer_type}_"
    mn2 = f"ex_{tr_expand_tokenizer}_hl_{ens_hidden_layers}_ac_{ens_activation}_do_{ens_dropout}_"
    mn3 = f"ep_{ens_epochs}_bs_{ens_batch_size}_sv_{ens_seed_val}_gpu_{use_gpu}_{label}"
    model_name = "".join([mn1, mn2, mn3]).replace("/", "-")

    set_random_seed(ens_seed_val, use_gpu)
    
    device = get_device(use_gpu)

    train_ds, dev_ds, test_ds, _, num_labels, new_vocab_size = get_data_function(tr_pretrained_model_label,
                                                                                 tr_max_lenght,
                                                                                 binary_classifier=binary_classifier,
                                                                                 expand_tokenizer=tr_expand_tokenizer,
                                                                                 use_saved_model=use_saved_model,
                                                                                 cuis=lstm_cuis)

    # transformer_model = tr_pretrained_model.from_pretrained(tr_pretrained_model_label,
    #                                                         num_labels=num_labels,
    #                                                         output_attentions=False,
    #                                                         output_hidden_states=True)
    transformer_model = get_model(tr_pretrained_model_label, binary_classifier, num_labels)

    if tr_expand_tokenizer:
        transformer_model.resize_token_embeddings(new_vocab_size)

    logging.info("Loading tr model: %s", transformer_model_name)
    transformer_model.load_state_dict(torch.load(MODELS_CACHE_PATH + transformer_model_name + ".pt"))


    if lstm_classif_vector == "output":
        lstm_model = LSTMClassifierOutput(vocab_size=25000, embeddings_dim=lstm_embeddings_dim, hidden_dim=lstm_hidden_layer_size,
                                          num_labels=num_labels, lstm_layers=lstm_layers, bidirectional=lstm_bidirectional,
                                          dropout=lstm_dropout, add_output_layer=lstm_add_output_layer)
    else:
        lstm_model = LSTMClassifierHidden(vocab_size=25000, embeddings_dim=lstm_embeddings_dim, hidden_dim=lstm_hidden_layer_size,
                                          num_labels=num_labels, lstm_layers=lstm_layers, bidirectional=lstm_bidirectional,
                                          dropout=lstm_dropout, add_output_layer=lstm_add_output_layer)


    logging.info("Loading lstm model: %s", lstm_model_name)
    lstm_model.load_state_dict(torch.load(MODELS_CACHE_PATH + lstm_model_name + ".pt"))

    if use_gpu:
        transformer_model.to(device)
        lstm_model.to(device)

    transformer_model.eval()
    lstm_model.eval()

    model = TransformerLSTMEnsemble(num_labels, transformer_model, lstm_model, mode=ens_mode,
                                    activation=ens_activation, dropout=ens_dropout, hidden_layers=ens_hidden_layers)

    if use_gpu:
        model.to(device)

    saved_model_file = MODELS_CACHE_PATH + model_name + ".pt"

    if exists(saved_model_file) and use_saved_model:
        logging.debug("Loading saved model...")
        model.load_state_dict(torch.load(saved_model_file))
    else:
        logging.debug("Training model...")
        model = train_function(model, learning_rate, epsilon, ens_batch_size, ens_epochs, device, train_ds, dev_ds,
                               use_early_stopping=use_early_stopping)
        torch.save(model.state_dict(), saved_model_file)

    model.eval()
    class_predictions, labels = eval_function(model, ens_batch_size, device, test_ds)
    get_evaluation_measures(f"{model_name}", labels, class_predictions, save_evaluation=True)

    if annotate_new_sentences:
        annotation_function(model, tr_pretrained_model_label, binary_classifier, tr_max_lenght, ens_batch_size, device,
                            dataset="new", tokenizer_type=tr_tokenizer_type, use_saved_model=use_saved_model,
                            expand_tokenizer=tr_expand_tokenizer, cuis=lstm_cuis)
        logging.info("New sentences annotated")

    if annotate_test_sentences:
        annotation_function(model, tr_pretrained_model_label, binary_classifier, tr_max_lenght, ens_batch_size, device,
                            dataset="test", tokenizer_type=tr_tokenizer_type, use_saved_model=use_saved_model,
                            expand_tokenizer=tr_expand_tokenizer, cuis=lstm_cuis)
        logging.info("Test dataset annotated")

    if annotate_train_sentences:
        annotation_function(model, tr_pretrained_model_label, binary_classifier, tr_max_lenght, ens_batch_size, device,
                            dataset="train", tokenizer_type=tr_tokenizer_type, use_saved_model=use_saved_model,
                            expand_tokenizer=tr_expand_tokenizer, cuis=lstm_cuis)
        logging.info("Train dataset annotated")
