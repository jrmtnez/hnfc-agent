import torch

from agent.classifiers.utils.models_cache_mgmt import get_tokenizer


def get_special_tokens(pretrained_model):
    if pretrained_model in ["bert-base-cased", "bert-base-uncased", "bert-large-cased", "bert-large-uncased",
                            "emilyalsentzer/Bio_ClinicalBERT", "dmis-lab/biobert-v1.1",
                            "bionlp/bluebert_pubmed_uncased_L-24_H-1024_A-16", "bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12",
                            "albert-base-v2", "roberta-base", "roberta-large", "fancy"]:
        return "[CLS]", "[SEP]"
    if pretrained_model in ["funnel-transformer/intermediate"]:
        return "<cls>", "<sep>"


def tokenize_text(text, tokenizer, max_lenght, cls_token, sep_token):

    tokenized_text = tokenizer.tokenize(text)
    if len(tokenized_text) > max_lenght - 2:
        tokenized_text = tokenized_text[:(max_lenght - 2)]

    text_tokens = [cls_token] + tokenized_text + [sep_token]
    text_segment_ids = [0] * len(text_tokens)

    text_input_ids = tokenizer.convert_tokens_to_ids(text_tokens)
    text_input_mask = [1] * len(text_input_ids)

    text_padding = [0] * (max_lenght - len(text_input_ids))

    text_input_ids += text_padding
    text_input_mask += text_padding
    text_segment_ids += text_padding

    assert len(text_input_ids) == max_lenght
    assert len(text_input_mask) == max_lenght
    assert len(text_segment_ids) == max_lenght

    text_input_ids_pt = torch.tensor(text_input_ids).reshape(1, -1)
    text_input_mask_pt = torch.tensor(text_input_mask).reshape(1, -1)
    text_segment_ids_pt = torch.tensor(text_segment_ids).reshape(1, -1)

    return text_input_ids_pt, text_input_mask_pt, text_segment_ids_pt


def get_transformer_inputs(x_text, pretrained_model, max_lenght):
    input_ids = []
    attention_masks = []

    # tokenizer = torch.hub.load("huggingface/pytorch-transformers", "tokenizer", pretrained_model, verbose=False)
    tokenizer = get_tokenizer(pretrained_model)

    cls_token, sep_token = get_special_tokens(pretrained_model)

    for text in x_text:
        text_input_ids, text_input_mask, _ = tokenize_text(text, tokenizer, max_lenght, cls_token, sep_token)
        input_ids.append(text_input_ids)
        attention_masks.append(text_input_mask)

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return input_ids, attention_masks
