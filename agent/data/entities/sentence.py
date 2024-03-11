import logging
import json
import unicodedata
import spacy

from agent.data.entities.config import BACKUPS_PATH, ROOT_LOGGER_ID
from agent.data.sql.sql_mgmt import execute_query, update_non_text_field, get_connection
from agent.data.sql.sql_mgmt import select_fields_where, select_one, exist_where
from agent.nlp.tokenizers.sentence_tokenizer import tokenize_sentences

logger = logging.getLogger(ROOT_LOGGER_ID)


def insert_sentence(connection, sentence_dict):
    insert_sentence_qry = f"""
        INSERT INTO public.annotate_sentence(
            item_id, rating, needs_revision, check_worthy, sentence, title, common_health_terms,
            metamap_extraction_xml, health_terms, health_terms_auto, metamap_extraction,
            new_sentence_id, sentence_class, review_level, check_worthy_auto, check_worthy_score_auto,
            big_object, big_subject, object, predicate, subject, object_cuis, predicate_cuis,
            subject_cuis, sentence_class_auto, big_predicate, spo_type, manually_identified_predicate,
            sentence_class_score_auto, skip_validations, instance_type, subject_text_cuis,
            predicate_text_cuis, object_text_cuis, instance_type_2, from_sentence_id, from_item_id, predicate_not_found)
        VALUES (
             {sentence_dict["item_id"]},
             {sentence_dict["rating"]},
             {sentence_dict["needs_revision"]},
            '{sentence_dict["check_worthy"].replace("'", "''")}',
            '{sentence_dict["sentence"].replace("'", "''")}',
            '{sentence_dict["title"].replace("'", "''")}',
            '{sentence_dict["common_health_terms"].replace("'", "''")}',
            '{sentence_dict["metamap_extraction_xml"].replace("'", "''")}',
             {sentence_dict["health_terms"]},
             {sentence_dict["health_terms_auto"]},
            '{sentence_dict["metamap_extraction"].replace("'", "''")}',
             {sentence_dict["new_sentence_id"]},
            '{sentence_dict["sentence_class"].replace("'", "''")}',
             {sentence_dict["review_level"]},
            '{sentence_dict["check_worthy_auto"]}',
             {sentence_dict["check_worthy_score_auto"]},
            '{sentence_dict["big_object"].replace("'", "''")}',
            '{sentence_dict["big_subject"].replace("'", "''")}',
            '{sentence_dict["object"].replace("'", "''")}',
            '{sentence_dict["predicate"].replace("'", "''")}',
            '{sentence_dict["subject"].replace("'", "''")}',
            '{sentence_dict["object_cuis"]}',
            '{sentence_dict["predicate_cuis"]}',
            '{sentence_dict["subject_cuis"]}',
            '{sentence_dict["sentence_class_auto"]}',
            '{sentence_dict["big_predicate"].replace("'", "''")}',
            '{sentence_dict["spo_type"]}',
            '{sentence_dict["manually_identified_predicate"].replace("'", "''")}',
             {sentence_dict["sentence_class_score_auto"]},
             {sentence_dict["skip_validations"]},
            '{sentence_dict["instance_type"]}',
            '{sentence_dict["subject_text_cuis"].replace("'", "''")}',
            '{sentence_dict["predicate_text_cuis"].replace("'", "''")}',
            '{sentence_dict["object_text_cuis"].replace("'", "''")}',
            '{sentence_dict["instance_type_2"]}',
             {sentence_dict["from_sentence_id"]},
             {sentence_dict["from_item_id"]},
             {sentence_dict["predicate_not_found"]})
        """
    execute_query(connection, insert_sentence_qry)


def sentence_tuple_to_dict(sentence_tuple):
    sentence_dict = {}
    sentence_dict["id"] = sentence_tuple[0]
    sentence_dict["item_id"] = sentence_tuple[1]
    sentence_dict["rating"] = sentence_tuple[2]
    sentence_dict["needs_revision"] = sentence_tuple[3]
    sentence_dict["check_worthy"] = sentence_tuple[4]
    sentence_dict["sentence"] = sentence_tuple[5]
    sentence_dict["title"] = sentence_tuple[6]
    sentence_dict["common_health_terms"] = sentence_tuple[7]
    sentence_dict["metamap_extraction_xml"] = sentence_tuple[8]
    sentence_dict["health_terms"] = sentence_tuple[9]
    sentence_dict["health_terms_auto"] = sentence_tuple[10]
    sentence_dict["metamap_extraction"] = sentence_tuple[11]
    sentence_dict["new_sentence_id"] = sentence_tuple[12]
    sentence_dict["sentence_class"] = sentence_tuple[13]
    sentence_dict["review_level"] = sentence_tuple[14]
    sentence_dict["check_worthy_auto"] = sentence_tuple[15]
    sentence_dict["check_worthy_score_auto"] = sentence_tuple[16]
    sentence_dict["big_object"] = sentence_tuple[17]
    sentence_dict["big_subject"] = sentence_tuple[18]
    sentence_dict["object"] = sentence_tuple[19]
    sentence_dict["predicate"] = sentence_tuple[20]
    sentence_dict["subject"] = sentence_tuple[21]
    sentence_dict["object_cuis"] = sentence_tuple[22]
    sentence_dict["predicate_cuis"] = sentence_tuple[23]
    sentence_dict["subject_cuis"] = sentence_tuple[24]
    sentence_dict["sentence_class_auto"] = sentence_tuple[25]
    sentence_dict["big_predicate"] = sentence_tuple[26]
    sentence_dict["spo_type"] = sentence_tuple[27]
    sentence_dict["manually_identified_predicate"] = sentence_tuple[28]
    sentence_dict["sentence_class_score_auto"] = sentence_tuple[29]
    sentence_dict["skip_validations"] = sentence_tuple[30]
    sentence_dict["instance_type"] = sentence_tuple[31]
    sentence_dict["subject_text_cuis"] = sentence_tuple[32]
    sentence_dict["predicate_text_cuis"] = sentence_tuple[33]
    sentence_dict["object_text_cuis"] = sentence_tuple[34]
    sentence_dict["instance_type_2"] = sentence_tuple[35]
    sentence_dict["from_sentence_id"] = sentence_tuple[36]
    sentence_dict["from_item_id"] = sentence_tuple[37]
    sentence_dict["predicate_not_found"] = sentence_tuple[38]
    return sentence_dict


def generate_sentences_from_items(connection, review_level=1):

    item_fields = "id, title, skip_validations, rating, text"
    items = select_fields_where(connection, "annotate_item", item_fields, f"review_level = {review_level}")

    if exist_where(connection, "annotate_sentence", "true"):
        new_sentence_id = select_one(connection, "annotate_sentence", "new_sentence_id")
    else:
        new_sentence_id = 10

    nlp = spacy.load("en_core_web_sm")

    for item in items:
        item_id = item[0]
        title = item[1]
        skip_validations = item[2]
        rating = item[3]
        text = item[4]

        if not exist_where(connection, "annotate_sentence", "item_id = " + str(item_id)):

            if review_level < 5:
                if skip_validations:
                    review_level = 3    # external article to evaluate
                else:
                    review_level = 2    # article collected for corpus

            tokenized_sentences = tokenize_sentences(text, spacy_model=nlp)

            for tokenized_sentence in tokenized_sentences:
                new_sentence_id = new_sentence_id + 10
                sentence_tuple = (0, item_id, rating, False, "NA", tokenized_sentence, title, "",  "", False, 0, "", new_sentence_id, "NA", review_level, "", 0, "", "", "", "", "", "", "", "", "", "", "", "", 0, skip_validations, "Other", "", "", "", "", 0, 0, False)
                insert_sentence(connection, sentence_tuple_to_dict(sentence_tuple))

            update_non_text_field(connection, "annotate_item", "id = " + str(item_id), "review_level", review_level)
            connection.commit()


def export_annotate_sentences(connection, path, review_level=3, clean_image_chars=False):
    sentences_exported = 0

    sentence_fields = "id, sentence, skip_validations, rating"
    annotate_sentences = select_fields_where(connection, "annotate_sentence", sentence_fields,
                                             f"review_level = {review_level} ORDER BY item_id, new_sentence_id")
    for annotate_sentence in annotate_sentences:
        sentence_id = annotate_sentence[0]
        sentence_text = annotate_sentence[1]
        if clean_image_chars:
            sentence_text = "".join(c for c in sentence_text if "So" not in unicodedata.category(c))

        with open(path + str(sentence_id) + ".txt", "w", encoding="utf-8") as file:
            file.write(sentence_text)
        remove_blank_lines_on_file(path + str(sentence_id) + ".txt")
        sentences_exported = sentences_exported + 1
    logger.debug("%s sentences exported to %s.", sentences_exported, path)


def remove_blank_lines_on_file(file_name):
    text = ""
    with open(file_name, "r", encoding="utf-8") as file:
        for line in file:
            if line != "\n":
                text += line + "\n"
    with open(file_name, "w", encoding="utf-8") as file:
        file.write(text)


def export_to_json():
    connection = get_connection()

    sentences = select_fields_where(connection, "annotate_sentence", '*', 'skip_validations = true', return_dict=True)

    with open(BACKUPS_PATH + "all_sentences_external.json", "w", encoding="utf-8") as write_file:
        json.dump(sentences, write_file, indent=4, separators=(",", ": "), default=str)

    sentences = select_fields_where(connection, "annotate_sentence", '*', 'skip_validations = false', return_dict=True)

    with open(BACKUPS_PATH + "all_sentences_corpus.json", "w", encoding="utf-8") as write_file:
        json.dump(sentences, write_file, indent=4, separators=(",", ": "), default=str)
