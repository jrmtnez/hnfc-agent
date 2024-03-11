import torch
import logging

from torch import nn
from os.path import exists

from agent.data.entities.config import MODELS_CACHE_PATH
from agent.classifiers.evaluation.evaluation_mgmt import get_evaluation_measures
from agent.classifiers.utils.random_mgmt import set_random_seed
from agent.classifiers.ffnn_classifier import train_loop, eval_loop
from agent.classifiers.data.lstm_dataset_mgmt import get_datasets as get_lstm_datasets
from agent.classifiers.utils.device_mgmt import get_device


logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)


class LSTMClassifierOutput(nn.Module):
    # con secuencias de entrada grandes (250) no se ve mejora en F1, con secuenciás de 32 tokens sí
    def __init__(self, vocab_size, embeddings_dim, hidden_dim, num_labels, lstm_layers, bidirectional, dropout,
                 add_output_layer=True):
        # bidirectional => num_directions = 2
        super().__init__()

        if bidirectional:
            num_directions = 2
        else:
            num_directions = 1

        self.hidden_layer_size = hidden_dim * num_directions
        self.lstm_layers = lstm_layers
        self.add_output_layer = add_output_layer
        self.embedding = nn.Embedding(vocab_size, embeddings_dim, padding_idx=0)
        self.lstm = nn.LSTM(embeddings_dim, hidden_dim, num_layers=lstm_layers,
                            bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.activation = torch.nn.Sigmoid()
        self.linear = torch.nn.Linear(self.hidden_layer_size, self.hidden_layer_size)
        self.output = torch.nn.Linear(self.hidden_layer_size, num_labels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x = [batch_size, seq_lenght]
        x = self.embedding(x)
        # x = [batch_size, seq_lenght, embeddings_dim]
        output, (_, _) = self.lstm(x)   # contains the output features (h_t) from the last layer of the LSTM
        # output = [batch_size, seq_lenght, hidden_dim * num_directions]
        x = output[:, -1, :]
        # x = [batch_size, hidden_dim * num_directions]

        if self.add_output_layer:
            x = self.linear(x)
            # x = [batch_size, hidden_dim * num_directions]
            x = self.activation(x)
            # x = [batch_size, hidden_dim * num_directions]
            x = self.dropout(x)
            # x = [batch_size, hidden_dim * num_directions]

        y = self.output(x)
        # y = [batch_size, num_labels]
        y = self.softmax(y)
        # y = [batch_size, num_labels]

        return x, y


class LSTMClassifierHidden(nn.Module):
    def __init__(self, vocab_size, embeddings_dim, hidden_dim, num_labels, lstm_layers, bidirectional, dropout,
                 add_output_layer=True):
        # bidirectional => num_directions = 2
        super().__init__()

        self.hidden_layer_size = hidden_dim
        self.lstm_layers = lstm_layers
        self.add_output_layer = add_output_layer

        self.embedding = nn.Embedding(vocab_size, embeddings_dim, padding_idx=0)
        self.lstm = nn.LSTM(embeddings_dim, hidden_dim, num_layers=lstm_layers,
                            bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.activation = torch.nn.Sigmoid()
        self.linear = torch.nn.Linear(hidden_dim, hidden_dim)
        self.output = torch.nn.Linear(hidden_dim, num_labels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x = [batch_size, seq_lenght]
        x = self.embedding(x)
        # x = [batch_size, seq_lenght, embeddings_dim]
        _, (hidden, _) = self.lstm(x)   # contains the final hidden state for each element in the batch
        # hidden = [lstm_layers * num_directions, batch_size, hidden_dim]

        x = hidden[-1]
        # x = [batch_size, hidden_dim]

        y = x
        if self.add_output_layer:
            y = self.linear(y)
            # y = [batch_size, hidden_dim]
            y = self.activation(y)
            # y = [batch_size, hidden_dim]
            y = self.dropout(y)
            # y = [batch_size, hidden_dim]

        y = self.output(y)
        # y = [batch_size, num_labels]
        y = self.softmax(y)
        # y = [batch_size, num_labels]

        return x, y


def lstm(for_task_label="",
         binary_classifier=False,
         cuis=False,
         lstm_classif_vector="output",
         embeddings_dim=150,
         lstm_layers=1,
         hidden_layer_size=100,
         bidirectional=True,
         dropout=0,
         add_output_layer=True,
         use_class_weight=False,
         use_early_stopping=True,
         epochs=250,
         batch_size=16,
         learning_rate=5e-4,
         epsilon=1e-8,
         use_saved_model=False,
         use_gpu=True,
         seed_val=42,
         vocab_size=25000,
         label=""):

    mn1 = f"{for_task_label}_lstm_bin_{binary_classifier}_ed_{embeddings_dim}_cv_{lstm_classif_vector}_ll_{lstm_layers}_"
    mn2 = f"bi_{bidirectional}_do_{dropout}_ol_{add_output_layer}_hd_{hidden_layer_size}_cui_{cuis}_cw_{use_class_weight}_"
    mn3 = f"ep_{epochs}_bs_{batch_size}_sv_{seed_val}_gpu_{use_gpu}_{label}"
    model_name = "".join([mn1, mn2, mn3]).replace("/", "-")

    set_random_seed(seed_val, use_gpu)

    device = get_device(use_gpu)

    train_ds, dev_ds, test_ds, output_size = get_lstm_datasets(binary_classifier=binary_classifier, cuis=cuis)

    if lstm_classif_vector == "output":
        model = LSTMClassifierOutput(vocab_size=vocab_size, embeddings_dim=embeddings_dim, hidden_dim=hidden_layer_size,
                                     num_labels=output_size, lstm_layers=lstm_layers, bidirectional=bidirectional,
                                     dropout=dropout, add_output_layer=add_output_layer)
    if lstm_classif_vector == "hidden":
        model = LSTMClassifierHidden(vocab_size=vocab_size, embeddings_dim=embeddings_dim, hidden_dim=hidden_layer_size,
                                     num_labels=output_size, lstm_layers=lstm_layers, bidirectional=bidirectional,
                                     dropout=dropout, add_output_layer=add_output_layer)
    if use_gpu:
        model.to(device)

    saved_model_file = MODELS_CACHE_PATH + model_name + ".pt"

    if exists(saved_model_file) and use_saved_model:
        logging.info("Loading saved model...")
        model.load_state_dict(torch.load(saved_model_file))
    else:
        logging.info("Training model...")
        model = train_loop(model, learning_rate, epsilon, batch_size, epochs, device, train_ds, dev_ds,
                           use_class_weight=use_class_weight, use_early_stopping=use_early_stopping)
        torch.save(model.state_dict(), saved_model_file)

    model.eval()
    class_predictions, labels = eval_loop(model, batch_size, device, test_ds)
    get_evaluation_measures(f"{model_name}", labels, class_predictions, save_evaluation=True)
