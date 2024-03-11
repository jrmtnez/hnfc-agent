import logging
import torch

from os.path import exists
from transformers import AlbertForSequenceClassification, BertForSequenceClassification, FunnelForSequenceClassification, AutoTokenizer
from transformers import AlbertModel, BertModel, FunnelModel
from agent.classifiers.custom_transformer import ModelWithCustomLossFunction

from agent.data.entities.config import FC_CLASS_VALUES, CW_CLASS_VALUES
from agent.classifiers.utils.class_mgmt import get_number_of_classes
from agent.data.entities.config import ROOT_LOGGER_ID

HUGGING_FACE_TOKENIZERS_PATH = "./.hf_models/tokenizers/"
HUGGING_FACE_MODELS_PATH = "./.hf_models/models/"
PRETRAINED_MODELS = {"funnel-transformer/intermediate": FunnelForSequenceClassification,
                     "bert-base-cased": BertForSequenceClassification,
                     "bert-base-uncased": BertForSequenceClassification,
                     "bert-large-cased": BertForSequenceClassification,
                     "bert-large-uncased": BertForSequenceClassification,
                     "emilyalsentzer/Bio_ClinicalBERT": BertForSequenceClassification,
                     "dmis-lab/biobert-v1.1": BertForSequenceClassification,
                     "bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12": BertForSequenceClassification,
                     "bionlp/bluebert_pubmed_uncased_L-24_H-1024_A-16": BertForSequenceClassification,
                     "albert-base-v2": AlbertForSequenceClassification}

PRETRAINED_BASE_MODELS = {"funnel-transformer/intermediate": FunnelModel,
                          "bert-base-cased": BertModel,
                          "bert-base-uncased": BertModel,
                          "bert-large-cased": BertModel,
                          "bert-large-uncased": BertModel,
                          "emilyalsentzer/Bio_ClinicalBERT": BertModel,
                          "dmis-lab/biobert-v1.1": BertModel,
                          "bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12": BertModel,
                          "bionlp/bluebert_pubmed_uncased_L-24_H-1024_A-16": BertModel,
                          "albert-base-v2": AlbertModel}

logger = logging.getLogger(ROOT_LOGGER_ID)


def refresh_cache():
    for pretrained_model_label in PRETRAINED_MODELS:
        logger.info("Refreshing model: %s", pretrained_model_label)

        tokenizer = torch.hub.load("huggingface/pytorch-transformers", "tokenizer",
                                   get_tokenizer_path(pretrained_model_label), verbose=False)
        tokenizer.save_pretrained(get_tokenizer_path(pretrained_model_label, local=True))

        for binary_classifier in [True, False]:

            number_of_classes = get_number_of_classes(binary_classifier, FC_CLASS_VALUES)
            pretrained_model_obj = PRETRAINED_MODELS.get(pretrained_model_label)
            model = pretrained_model_obj.from_pretrained(get_model_path(pretrained_model_label, binary_classifier, number_of_classes),
                                                         num_labels=number_of_classes,
                                                         output_attentions=False,
                                                         output_hidden_states=True)
            model.save_pretrained(get_model_path(pretrained_model_label, binary_classifier, number_of_classes, local=True))

        binary_classifier = False
        number_of_classes = get_number_of_classes(binary_classifier, CW_CLASS_VALUES)
        pretrained_model_obj = PRETRAINED_MODELS.get(pretrained_model_label)
        model = pretrained_model_obj.from_pretrained(get_model_path(pretrained_model_label, binary_classifier, number_of_classes),
                                                     num_labels=number_of_classes,
                                                     output_attentions=False,
                                                     output_hidden_states=True)
        model.save_pretrained(get_model_path(pretrained_model_label, binary_classifier, number_of_classes, local=True))

    for pretrained_model_label in PRETRAINED_BASE_MODELS:
        logger.info("Refreshing base model: %s", pretrained_model_label)

        pretrained_model_obj = PRETRAINED_BASE_MODELS.get(pretrained_model_label)
        model = pretrained_model_obj.from_pretrained(get_model_path(pretrained_model_label, None, None, base_model=True))
        model.save_pretrained(get_model_path(pretrained_model_label, None, None, base_model=True, local=True))


def get_tokenizer_path(pretrained_model_label, local=False):
    pretrained_tokenizer_path = HUGGING_FACE_TOKENIZERS_PATH + pretrained_model_label
    if local:
        return pretrained_tokenizer_path
    if not exists(pretrained_tokenizer_path):
        pretrained_tokenizer_path = pretrained_model_label
    return pretrained_tokenizer_path


def get_model_path(pretrained_model_label, binary_classifier, number_of_classes, base_model=False, local=False):
    if base_model:
        pretrained_model_path = HUGGING_FACE_MODELS_PATH + pretrained_model_label.replace("/", "-") + "_base"
    else:
        pretrained_model_path = HUGGING_FACE_MODELS_PATH + pretrained_model_label.replace("/", "-") + "_" + str(binary_classifier) + "_" + str(number_of_classes)
    if local:
        return pretrained_model_path
    if not exists(pretrained_model_path):
        pretrained_model_path = pretrained_model_label
    return pretrained_model_path


# def get_tokenizer(pretrained_model_label):
#     if pretrained_model_label == "fancy":
#         tokenizer = AutoTokenizer.from_pretrained(get_tokenizer_path("bert-base-uncased", local=True))
#     else:
#         tokenizer = AutoTokenizer.from_pretrained(get_tokenizer_path(pretrained_model_label, local=True))
#     return tokenizer


# def get_model(pretrained_model_label, binary_classifier, num_labels):
#     if pretrained_model_label == "fancy":
#         model = FancyBertModelWithCustomLossFunction(num_labels=num_labels,
#                                                      output_attentions=False,
#                                                      output_hidden_states=True)
#     else:
#         pretrained_model = PRETRAINED_MODELS.get(pretrained_model_label)
#         model = pretrained_model.from_pretrained(get_model_path(pretrained_model_label, binary_classifier, num_labels, local=True),
#                                                 num_labels=num_labels,
#                                                 output_attentions=False,
#                                                 output_hidden_states=True)
#     return model


def get_tokenizer(pretrained_model_label):
    tokenizer = AutoTokenizer.from_pretrained(get_tokenizer_path(pretrained_model_label, local=True))
    return tokenizer


def get_model(pretrained_model_label, binary_classifier, num_labels, custom_model=False):
    if custom_model:
        pretrained_model = PRETRAINED_BASE_MODELS.get(pretrained_model_label)
        pretrained_base_model = pretrained_model.from_pretrained(get_model_path(pretrained_model_label, None, None, local=True, base_model=True))
        model = ModelWithCustomLossFunction(pretrained_base_model,
                                            pretrained_model_label=pretrained_model_label,
                                            num_labels=num_labels,
                                            output_attentions=False,
                                            output_hidden_states=True)
    else:
        pretrained_model = PRETRAINED_MODELS.get(pretrained_model_label)
        model = pretrained_model.from_pretrained(get_model_path(pretrained_model_label, binary_classifier, num_labels, local=True),
                                                 num_labels=num_labels,
                                                 output_attentions=False,
                                                 output_hidden_states=True)
    return model