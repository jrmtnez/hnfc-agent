import logging
import torch
# from sre_constants import IN_IGNORE

from transformers import FSMTForConditionalGeneration, FSMTTokenizer
from transformers import MarianMTModel, MarianTokenizer
from transformers import MBartForConditionalGeneration, MBartTokenizer

from agent.data.sql.sql_mgmt import select_fields_where, get_connection, execute_query, select_one
from agent.data.entities.config import ROOT_LOGGER_ID
from agent.data.entities.item import insert_item, item_tuple_to_dict
from agent.data.entities.sentence import insert_sentence, sentence_tuple_to_dict
from agent.classifiers.utils.device_mgmt import get_device


logger = logging.getLogger(ROOT_LOGGER_ID)

TRANSFORMER_TYPE = 1  # ["fsmt", "marian", "mbart"]
INSERT_DATA = False
NEW_REVIEW_LEVEL = 7

def get_back_translation_models(forward_model_name, backward_model_name):
    if TRANSFORMER_TYPE == 0:
        forward_tokenizer = FSMTTokenizer.from_pretrained(forward_model_name)
        forward_model = FSMTForConditionalGeneration.from_pretrained(forward_model_name)

        backward_tokenizer = FSMTTokenizer.from_pretrained(backward_model_name)
        backward_model = FSMTForConditionalGeneration.from_pretrained(backward_model_name)

    if TRANSFORMER_TYPE == 1:
        forward_tokenizer = MarianTokenizer.from_pretrained(forward_model_name)
        forward_model = MarianMTModel.from_pretrained(forward_model_name)

        backward_tokenizer = MarianTokenizer.from_pretrained(backward_model_name)
        backward_model = MarianMTModel.from_pretrained(backward_model_name)

    if TRANSFORMER_TYPE == 2:
        forward_tokenizer = MBartTokenizer.from_pretrained(forward_model_name)
        forward_model = MBartForConditionalGeneration.from_pretrained(forward_model_name)

        backward_tokenizer = MBartTokenizer.from_pretrained(backward_model_name)
        backward_model = MBartForConditionalGeneration.from_pretrained(backward_model_name)

    return [forward_tokenizer, forward_model, backward_tokenizer, backward_model]


def do_back_translation(input_sentence, bt_models, use_gpu=True):

    forward_tokenizer = bt_models[0]
    forward_model = bt_models[1]
    backward_tokenizer = bt_models[2]
    backward_model = bt_models[3]
    
    device = get_device(use_gpu)

    forward_model.to(device)
    backward_model.to(device)

    if TRANSFORMER_TYPE == 0:
        # fails with GPU
        # empty back-translated predicates
        input_ids = forward_tokenizer.encode(input_sentence, return_tensors="pt")
        input_ids.to(device)
        translated_tokens = forward_model.generate(input_ids, max_new_tokens=200)
        forward_translation = forward_tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

        input_ids = backward_tokenizer.encode(forward_translation, return_tensors="pt")
        input_ids.to(device)
        translated_tokens = backward_model.generate(input_ids, max_new_tokens=200)
        backward_translation = backward_tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

    if TRANSFORMER_TYPE == 1:
        input_ids = forward_tokenizer(input_sentence, return_tensors="pt", padding=True)
        input_ids.to(device)
        translated_tokens = forward_model.generate(**input_ids, max_new_tokens=200)
        forward_translation = forward_tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

        input_ids = backward_tokenizer(forward_translation, return_tensors="pt", padding=True)
        input_ids.to(device)
        translated_tokens = backward_model.generate(**input_ids, max_new_tokens=200)
        backward_translation = backward_tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

    if TRANSFORMER_TYPE == 2:
        inputs_ids = forward_tokenizer(input_sentence, return_tensors="pt")
        inputs_ids.to(device)
        translated_tokens = forward_model.generate(**inputs_ids, max_new_tokens=200)
        forward_translation = forward_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

        inputs_ids = backward_tokenizer(forward_translation, return_tensors="pt")
        inputs_ids.to(device)
        translated_tokens = backward_model.generate(**inputs_ids, max_new_tokens=200)
        backward_translation = backward_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

    return backward_translation


def item_back_translation(item_tuple, connection, bt_models, bt_label):

    item_id = item_tuple[0]

    logger.info(">>> Item Id: %s", item_id)

    if item_id == 0:
        exit

    new_new_sentence_id = select_one(connection, "annotate_sentence", "new_sentence_id")

    annotate_sentences = select_fields_where(connection, "annotate_sentence", "*",
                                             f"item_id = {item_id} ORDER BY item_id, new_sentence_id")

    new_item_dict = item_tuple_to_dict(item_tuple)

    item_text_bt = ""

    for annotate_sentence in annotate_sentences:
        new_sentence_dict = sentence_tuple_to_dict(annotate_sentence)

        sentence_id = annotate_sentence[0]
        new_sentence_id = annotate_sentence[12]
        review_level = annotate_sentence[14]
        sentence_text = annotate_sentence[5].strip()
        big_subject = annotate_sentence[18].strip()
        predicate = annotate_sentence[20].strip()
        big_object = annotate_sentence[17].strip()
        instance_type  = annotate_sentence[31]

        big_subject_bt = ""
        predicate_bt = ""
        big_object_bt = ""
        sentence_text_bt = ""

        if review_level == 9:

            big_subject_bt = do_back_translation(big_subject, bt_models)
            predicate_bt = do_back_translation(predicate, bt_models)
            big_object_bt = do_back_translation(big_object, bt_models)
            # sentence_text_bt = big_subject_bt +  " " + predicate_bt + " " + big_object_bt
            sentence_text_bt = do_back_translation(sentence_text, bt_models)

            logger.info("Subject:                     %s", big_subject)
            logger.info("Subject (back translated):   %s", big_subject_bt)
            logger.info("Predicate:                   %s", predicate)
            logger.info("Predicate (back translated): %s", predicate_bt)
            logger.info("Object:                      %s", big_object)
            logger.info("Object (back translated):    %s", big_object_bt)

        if review_level == 5:

            sentence_text_bt = do_back_translation(sentence_text, bt_models)

            logger.info("Sentence:                    %s", sentence_text[:128])
            logger.info("Sentence (back translated):  %s", sentence_text_bt[:128])

        if item_text_bt == "":
            item_text_bt = sentence_text_bt
        else:
            item_text_bt = item_text_bt + "\n" + sentence_text_bt

        if review_level != 5:
            new_sentence_dict["review_level"] = NEW_REVIEW_LEVEL

        new_sentence_dict["sentence"] = sentence_text_bt
        new_sentence_dict["subject"] = big_subject_bt
        new_sentence_dict["big_subject"] = big_subject_bt
        new_sentence_dict["predicate"] = predicate_bt
        new_sentence_dict["big_predicate"] = predicate_bt
        new_sentence_dict["manually_identified_predicate"] = predicate_bt
        new_sentence_dict["object"] = big_object_bt
        new_sentence_dict["big_object"] = big_object_bt
        new_sentence_dict["from_sentence_id"] = sentence_id
        new_sentence_dict["from_item_id"] = item_id
        new_sentence_dict["instance_type"] = instance_type + "_" + bt_label
        if new_sentence_id == 0:
            new_sentence_dict["new_sentence_id"] = 0
        else:
            new_new_sentence_id = new_new_sentence_id + 10
            new_sentence_dict["new_sentence_id"] = new_new_sentence_id

        if INSERT_DATA:
            insert_sentence(connection, new_sentence_dict)

    new_item_dict["review_level"] = NEW_REVIEW_LEVEL
    new_item_dict["text"] = item_text_bt
    new_item_dict["item_class_auto"] = 0
    new_item_dict["item_class_4_auto"] = ""
    new_item_dict["from_item_id"] = item_id
    new_item_dict["instance_type"] = instance_type + "_" + bt_label

    if INSERT_DATA:
        insert_item(connection, new_item_dict)

        sql_query = f"""
            UPDATE annotate_sentence
            SET item_id = annotate_item.id
            FROM annotate_item
            WHERE annotate_sentence.from_item_id = {item_id} AND
                  annotate_sentence.from_item_id = public.annotate_item.from_item_id
        """
        execute_query(connection, sql_query)


def process_single_item(item_id, forward_model_name, backward_model_name, bt_label):

    bt_models = get_back_translation_models(forward_model_name, backward_model_name)

    connection = get_connection()

    annotate_items = select_fields_where(connection, "annotate_item", "*", f"id = {item_id}")
    item_back_translation(annotate_items[0], connection, bt_models, bt_label)

    connection.commit()
    connection.close()


def process_train_items(forward_model_name, backward_model_name, bt_label):

    bt_models = get_back_translation_models(forward_model_name, backward_model_name)

    connection = get_connection()

    annotate_items = select_fields_where(connection, "annotate_item", "*", "review_level = 9 AND skip_validations = false AND instance_type = 'Train' ORDER BY id")
    # annotate_items = select_fields_where(connection, "annotate_item", "*", "review_level = 9 AND skip_validations = false AND instance_type = 'Train' AND id > 472 ORDER BY id")
    for annotate_item in annotate_items:
        item_back_translation(annotate_item, connection, bt_models, bt_label)
        connection.commit()
    connection.close()
