import torch
import logging

from agent.nlp.vocabulary import get_text8_vocabulary, get_corpus_vocabulary
from agent.classifiers.utils.models_cache_mgmt import get_tokenizer
from agent.nlp.preprocess import remove_stop_words

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_special_tokens(pretrained_model):
    if pretrained_model in ["bert-base-cased", "bert-base-uncased", "bert-large-cased", "bert-large-uncased",
                            "emilyalsentzer/Bio_ClinicalBERT", "dmis-lab/biobert-v1.1",
                            "bionlp/bluebert_pubmed_uncased_L-24_H-1024_A-16",
                            "bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12",
                            "albert-base-v2", "roberta-base", "roberta-large"]:
        return "[CLS]", "[SEP]"
    if pretrained_model in ["funnel-transformer/intermediate"]:
        return "<cls>", "<sep>"


def tokenize_sentence(sentence, tokenizer, max_lenght, cls_token, sep_token, tokenizer_type="text", text_cuis=False):

    if tokenizer_type == "text":
        return tokenize_sentence_text(sentence, tokenizer, max_lenght, cls_token, sep_token)
    elif tokenizer_type == "spo_variable":
        return tokenize_sentence_spo_variable(sentence, tokenizer, max_lenght, cls_token, sep_token, text_cuis=text_cuis)
    elif tokenizer_type == "spo_fixed":
        return tokenize_sentence_spo_fixed(sentence, tokenizer, max_lenght, cls_token, sep_token, text_cuis=text_cuis)


def expand_tokenizer_vocabulary(model_name, max_common_words=2500, max_words_to_add=1000, use_saved_model=False):
    common_words = get_text8_vocabulary(max_words=max_common_words)
    new_tokens = get_corpus_vocabulary(common_words, max_words=max_words_to_add, use_saved_model=use_saved_model)

    # tokenizer = torch.hub.load("huggingface/pytorch-transformers", "tokenizer", model_name, verbose=False)
    tokenizer = get_tokenizer(model_name)

    num_added_toks = tokenizer.add_tokens(new_tokens)
    new_size = len(tokenizer)

    logger.info("Tokens added: %s, vocabulary size: %s, increment %s %%", num_added_toks, new_size, num_added_toks / new_size * 100)

    return tokenizer


def tokenize_sentence_text(sentence, tokenizer, max_lenght, cls_token, sep_token):

    sentence_text = sentence[1]

    tokenized_sentence = tokenizer.tokenize(sentence_text)
    if len(tokenized_sentence) > max_lenght - 2:
        tokenized_sentence = tokenized_sentence[:(max_lenght - 2)]

    sentence_tokens = [cls_token] + tokenized_sentence + [sep_token]
    sentence_segment_ids = [0] * len(sentence_tokens)

    sentence_input_ids = tokenizer.convert_tokens_to_ids(sentence_tokens)
    sentence_input_mask = [1] * len(sentence_input_ids)

    sentence_padding = [0] * (max_lenght - len(sentence_input_ids))

    sentence_input_ids += sentence_padding
    sentence_input_mask += sentence_padding
    sentence_segment_ids += sentence_padding

    assert len(sentence_input_ids) == max_lenght
    assert len(sentence_input_mask) == max_lenght
    assert len(sentence_segment_ids) == max_lenght

    sentence_input_ids_pt = torch.tensor(sentence_input_ids).reshape(1, -1)
    sentence_input_mask_pt = torch.tensor(sentence_input_mask).reshape(1, -1)
    sentence_segment_ids_pt = torch.tensor(sentence_segment_ids).reshape(1, -1)

    return sentence_input_ids_pt, sentence_input_mask_pt, sentence_segment_ids_pt


def tokenize_sentence_spo_variable(sentence, tokenizer, max_lenght, cls_token, sep_token, text_cuis=False):

    if text_cuis:
        big_subject = sentence[17].replace("[UNK]", "")
        big_predicate = sentence[18].replace("[UNK]", "")
        big_object = sentence[19].replace("[UNK]", "")
    else:
        big_subject = sentence[6]
        big_predicate = sentence[7]
        big_object = sentence[8]

    # big_subject = remove_stop_words(big_subject)
    # big_predicate = remove_stop_words(big_predicate)
    # big_object = remove_stop_words(big_object)

    # print(big_subject)
    # print(big_predicate)
    # print(big_object)

    tokenized_subject = tokenizer.tokenize(big_subject)
    tokenized_predicate = tokenizer.tokenize(big_predicate)
    tokenized_object = tokenizer.tokenize(big_object)


    while len(tokenized_subject) + len(tokenized_predicate) + len(tokenized_object) + 4 > max_lenght:
        if len(tokenized_subject) > len(tokenized_object):
            tokenized_subject.pop()
        else:
            tokenized_object.pop()
    subject_tokens = [cls_token] + tokenized_subject + [sep_token]
    subject_segment_ids = [0] * len(subject_tokens)
    predicate_tokens = tokenized_predicate + [sep_token]
    predicate_segment_ids = [1] * len(predicate_tokens)
    object_tokens = tokenized_object + [sep_token]
    object_segment_ids = [0] * len(object_tokens)

    triple_input_ids = tokenizer.convert_tokens_to_ids(subject_tokens + predicate_tokens + object_tokens)
    triple_input_mask = [1] * len(triple_input_ids)
    triple_segment_ids = subject_segment_ids + predicate_segment_ids + object_segment_ids

    padding = [0] * (max_lenght - len(triple_input_ids))
    triple_input_ids += padding
    triple_input_mask += padding
    triple_segment_ids += padding

    assert len(triple_input_ids) == max_lenght
    assert len(triple_input_mask) == max_lenght
    assert len(triple_segment_ids) == max_lenght

    triple_input_ids_pt = torch.tensor(triple_input_ids).reshape(1, -1)
    triple_input_mask_pt = torch.tensor(triple_input_mask).reshape(1, -1)
    triple_segment_ids_pt = torch.tensor(triple_segment_ids).reshape(1, -1)

    return triple_input_ids_pt, triple_input_mask_pt, triple_segment_ids_pt


def tokenize_sentence_spo_fixed(sentence, tokenizer, max_lenght, cls_token, sep_token, text_cuis=False):
    so_lenght = (max_lenght - 4) // 3
    p_lenght = so_lenght + (max_lenght - 4) % 3

    # big_subject = sentence[6]
    # big_predicate = sentence[7]
    # big_object = sentence[8]

    if text_cuis:
        big_subject = sentence[17].replace("[UNK]", "")
        big_predicate = sentence[18].replace("[UNK]", "")
        big_object = sentence[19].replace("[UNK]", "")
    else:
        big_subject = sentence[6]
        big_predicate = sentence[7]
        big_object = sentence[8]

    tokenized_subject = tokenizer.tokenize(big_subject)
    tokenized_predicate = tokenizer.tokenize(big_predicate)
    tokenized_object = tokenizer.tokenize(big_object)

    while len(tokenized_subject) > so_lenght:
        tokenized_subject.pop()
    while len(tokenized_predicate) > p_lenght:
        tokenized_predicate.pop()
    while len(tokenized_object) > so_lenght:
        tokenized_object.pop()

    padding = [' '] * (so_lenght - len(tokenized_subject))
    tokenized_subject += padding
    padding = [' '] * (p_lenght - len(tokenized_predicate))
    tokenized_predicate += padding
    padding = [' '] * (so_lenght - len(tokenized_object))
    tokenized_object += padding

    subject_tokens = [cls_token] + tokenized_subject + [sep_token]
    subject_segment_ids = [0] * len(subject_tokens)
    predicate_tokens = tokenized_predicate + [sep_token]
    predicate_segment_ids = [1] * len(predicate_tokens)
    object_tokens = tokenized_object + [sep_token]
    object_segment_ids = [0] * len(object_tokens)

    triple_input_ids = tokenizer.convert_tokens_to_ids(subject_tokens + predicate_tokens + object_tokens)
    triple_input_mask = [1] * len(triple_input_ids)
    triple_segment_ids = subject_segment_ids + predicate_segment_ids + object_segment_ids

    assert len(triple_input_ids) == max_lenght
    assert len(triple_input_mask) == max_lenght
    assert len(triple_segment_ids) == max_lenght

    triple_input_ids_pt = torch.tensor(triple_input_ids).reshape(1, -1)
    triple_input_mask_pt = torch.tensor(triple_input_mask).reshape(1, -1)
    triple_segment_ids_pt = torch.tensor(triple_segment_ids).reshape(1, -1)

    return triple_input_ids_pt, triple_input_mask_pt, triple_segment_ids_pt



def get_transformer_inputs(x_text, pretrained_model, max_lenght, expand_tokenizer=False, use_saved_model=False,
                           tokenizer_type="text", text_cuis=False):
    input_ids = []
    attention_masks = []
    segment_ids = []

    if expand_tokenizer:
        tokenizer = expand_tokenizer_vocabulary(pretrained_model, use_saved_model=use_saved_model)
    else:
        # tokenizer = torch.hub.load("huggingface/pytorch-transformers", "tokenizer", pretrained_model, verbose=False)
        tokenizer = get_tokenizer(pretrained_model)

    new_vocab_size = len(tokenizer)

    cls_token, sep_token = get_special_tokens(pretrained_model)

    for sentence in x_text:
        triple_input_ids, triple_input_mask, triple_segment_ids = tokenize_sentence(sentence, tokenizer, max_lenght,
                                                                                    cls_token, sep_token,
                                                                                    tokenizer_type=tokenizer_type,
                                                                                    text_cuis=text_cuis)
        input_ids.append(triple_input_ids)
        attention_masks.append(triple_input_mask)
        segment_ids.append(triple_segment_ids)

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    segment_ids = torch.cat(segment_ids, dim=0)

    return input_ids, attention_masks, segment_ids, new_vocab_size
