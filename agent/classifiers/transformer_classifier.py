import torch
import logging

from pynvml import *

from transformers import logging as transformers_logging
from accelerate import Accelerator

from os.path import exists

from agent.data.entities.config import MODELS_CACHE_PATH
from agent.data.entities.config import DATA_CACHE_PATH, EXPORT_CHECK_INPUT_DATA_RESULTS_FILES
from agent.classifiers.utils.random_mgmt import set_random_seed
from agent.classifiers.utils.models_cache_mgmt import get_model
from agent.classifiers.evaluation.evaluation_mgmt import get_evaluation_measures

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
transformers_logging.set_verbosity_error()


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    logging.info("GPU memory occupied: %s MB, total %s MB.",
                 info.used // 1024 ** 2, info.total // 1024 ** 2)


def transformer_classifier(for_task_label="",
                           binary_classifier=False,
                           pretrained_model_label="",
                           get_data_function=None,
                           train_loop_function=None,
                           eval_loop_function=None,
                           annotate_dataset_function=None,
                           tokenizer_type="text",
                           expand_tokenizer=False,
                           text_cuis=False,
                           max_lenght=128,
                           epochs=3,
                           use_early_stopping=True,
                           patience=2,
                           batch_size=32,
                           seed_val=42,
                           use_gpu=True,
                           label="",
                           learning_rate=2e-5,
                           epsilon=1.5e-8,
                           warmup_steps=0,
                           val_split=0.8,
                           use_gradient_checkpointing=False,
                           gradient_accumulation_steps=1,
                           custom_model=False,
                           use_saved_model=False,
                           evaluate_model=True,
                           annotate_new_instances=False,
                           annotate_test_instances=False,
                           annotate_train_instances=False,
                           annotate_external_instances=False):

    mn1 = f"{for_task_label}_tr_bin_{binary_classifier}_pm_{pretrained_model_label}_"
    if custom_model:
        mn1 = mn1 + "cus_"
    mn2 = f"tt_{tokenizer_type}_tc_{text_cuis}_ex_{expand_tokenizer}_ml_{max_lenght}_"
    mn3 = f"ep_{epochs}_bs_{batch_size}_sv_{seed_val}_gpu_{use_gpu}_{label}"
    model_name = "".join([mn1, mn2, mn3]).replace("/", "-")

    set_random_seed(seed_val, use_gpu)

    cpu = not use_gpu
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps, cpu=cpu)

    device = accelerator.device

    train_ds, dev_ds, test_ds, num_labels, new_vocab_size = get_data_function(pretrained_model_label,
                                                                              max_lenght,
                                                                              val_split=val_split,
                                                                              binary_classifier=binary_classifier,
                                                                              _expand_tokenizer=expand_tokenizer,
                                                                              _use_saved_model=use_saved_model,
                                                                              _tokenizer_type=tokenizer_type,
                                                                              text_cuis=text_cuis)

    model = get_model(pretrained_model_label, binary_classifier, num_labels, custom_model=custom_model)

    if new_vocab_size is not None and expand_tokenizer:
        model.resize_token_embeddings(new_vocab_size)

    if use_gpu:
        model.to(device)

    saved_model_file = MODELS_CACHE_PATH + model_name + ".pt"

    if exists(saved_model_file) and use_saved_model:
        logging.info("Loading saved model...")
        model.load_state_dict(torch.load(saved_model_file))
    else:

        if use_gradient_checkpointing:
            model.gradient_checkpointing_enable()

        logging.info("Training model...")
        model = train_loop_function(accelerator, model, learning_rate, epsilon, warmup_steps, batch_size, epochs, device, train_ds, dev_ds,
                                    for_task_label, binary_classifier, use_class_weight=custom_model, use_early_stopping=use_early_stopping,
                                    patience=patience)
        torch.save(model.state_dict(), saved_model_file)

        print_gpu_utilization()

    model.eval()
    if evaluate_model:
        if test_ds is not None:
            class_predictions, labels = eval_loop_function(model, batch_size, device, test_ds)
        else:
            class_predictions, labels = eval_loop_function(model, batch_size, device, dev_ds)
        get_evaluation_measures(f"{model_name}", labels, class_predictions, for_task_label, binary_classifier, save_evaluation=True, show_classif_report=True)

        if EXPORT_CHECK_INPUT_DATA_RESULTS_FILES:

            logging.info("Exporting check result data file...")
            if binary_classifier:
                binary_label = "bin"
            else:
                binary_label = "mc"

            file_name = DATA_CACHE_PATH + binary_label + "_" + pretrained_model_label.replace("/", "-") + "_predictions.tsv"

            with open(file_name, "w", encoding="utf-8") as results_file:
                results_file.write("y_test\ty_pred\n")

            for class_value, pred_value in zip(labels.tolist(), class_predictions.tolist()):
                with open(file_name, "a", encoding="utf-8") as results_file:
                    results_file.write(str(class_value) + "\t" + str(pred_value) + "\n")

    if annotate_new_instances and annotate_dataset_function is not None:
        annotate_dataset_function(model, pretrained_model_label, binary_classifier, max_lenght, batch_size, device,
                                  _dataset="new", _expand_tokenizer=expand_tokenizer, _tokenizer_type=tokenizer_type,
                                  text_cuis=text_cuis)
        logging.info("New instances annotated")

    if annotate_test_instances and annotate_dataset_function is not None:
        annotate_dataset_function(model, pretrained_model_label, binary_classifier, max_lenght, batch_size, device,
                                  _dataset="test", _expand_tokenizer=expand_tokenizer, _tokenizer_type=tokenizer_type,
                                  text_cuis=text_cuis)
        logging.info("Test instances annotated")

    if annotate_train_instances and annotate_dataset_function is not None:
        annotate_dataset_function(model, pretrained_model_label, binary_classifier, max_lenght, batch_size, device,
                                  _dataset="train", _expand_tokenizer=expand_tokenizer, _tokenizer_type=tokenizer_type,
                                  text_cuis=text_cuis)
        logging.info("Train instances annotated")

    if annotate_external_instances and annotate_dataset_function is not None:
        annotate_dataset_function(model, pretrained_model_label, binary_classifier, max_lenght, batch_size, device,
                                  _dataset="external", _expand_tokenizer=expand_tokenizer, _tokenizer_type=tokenizer_type,
                                  text_cuis=text_cuis)
        logging.info("External instances annotated")
