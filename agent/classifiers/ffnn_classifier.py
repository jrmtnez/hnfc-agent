import torch
import logging

from torch import nn
from os.path import exists

from agent.data.entities.config import DEV_LABEL, MODELS_CACHE_PATH, NEW_ITEM_FILTER, NEW_LABEL
from agent.classifiers.utils.random_mgmt import set_random_seed
from agent.classifiers.evaluation.evaluation_mgmt import get_evaluation_measures

from agent.classifiers.data.tfidf_dataset_mgmt import get_datasets as get_tfidf_datasets
from agent.classifiers.ffnn_trainer import train_loop, eval_loop
from agent.classifiers.utils.device_mgmt import get_device

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)


class FFNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_layer_size, num_labels, activation_fn='sigmoid'):
        super().__init__()

        self.hidden_layer_size = hidden_layer_size
        self.num_labels = num_labels
        self.hidden = nn.Linear(input_size, hidden_layer_size)
        if activation_fn == 'sigmoid':
            self.activation = torch.nn.Sigmoid()
        if activation_fn == 'tanh':
            self.activation = torch.nn.Tanh()
        if activation_fn == 'relu':
            self.activation = torch.nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.output = torch.nn.Linear(hidden_layer_size, num_labels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.hidden(x)
        x = self.activation(x)
        y = self.dropout(x)
        y = self.output(y)
        y = self.softmax(y)

        return x, y


def ffnn(for_task_label="",
         binary_classifier=False,
         cuis=False,
         activation_fn='sigmoid',
         hidden_layer_size=100,
         use_class_weight=False,
         use_early_stopping=True,
         patience=2,
         epochs=250,
         batch_size=16,
         seed_val=42,
         use_gpu=True,
         label="",
         learning_rate=5e-4,
         epsilon=1e-8,
         use_features=None,
         annotate_dataset_function=None,
         item_filter_label=DEV_LABEL,
         use_saved_model=False):

    if use_features is None:
        mn1 = f"{for_task_label}_ffnn_bin_{binary_classifier}_cui_{cuis}_af_{activation_fn}_"
        mn3 = f"ep_{epochs}_bs_{batch_size}_sv_{seed_val}_gpu_{use_gpu}_{label}"
    else:
        mn1 = f"{for_task_label}_ffnn_bin_{binary_classifier}_cui_{cuis}_af_{activation_fn}_"
        mn3 = f"ep_{epochs}_bs_{batch_size}_sv_{seed_val}_gpu_{use_gpu}_{item_filter_label}"

    mn2 = f"hl_{hidden_layer_size}_cw_{use_class_weight}_"

    model_name = "".join([mn1, mn2, mn3]).replace("/", "-")

    set_random_seed(seed_val, use_gpu)

    device = get_device(use_gpu)

    items_dataset = False
    if for_task_label == "fd":
        items_dataset = True

    if use_features is None:
        train_ds, dev_ds, test_ds, input_size, output_size = get_tfidf_datasets(binary_classifier=binary_classifier, cuis=cuis, items_dataset=items_dataset)
    else:
        train_ds, dev_ds, test_ds, input_size, output_size = use_features

    model = FFNNClassifier(input_size, hidden_layer_size, output_size, activation_fn=activation_fn)

    if use_gpu:
        model.to(device)

    saved_model_file = MODELS_CACHE_PATH + model_name + ".pt"

    if exists(saved_model_file) and use_saved_model:
        logging.info("Loading saved model...")
        model.load_state_dict(torch.load(saved_model_file))
    else:
        logging.info("Training model...")
        model = train_loop(model, learning_rate, epsilon, batch_size, epochs, device, train_ds, dev_ds, for_task_label, binary_classifier,
                           use_class_weight=use_class_weight, use_early_stopping=use_early_stopping, patience=patience)
        torch.save(model.state_dict(), saved_model_file)

    model.eval()
    class_predictions, labels = eval_loop(model, batch_size, device, test_ds)
    get_evaluation_measures(f"{model_name}", labels, class_predictions, for_task_label, binary_classifier,
                            save_evaluation=True, show_classif_report=True)

    if annotate_dataset_function is not None:
        annotate_dataset_function(model, binary_classifier, device, NEW_ITEM_FILTER, NEW_LABEL, review_level=8)
