import logging
import json
import unicodedata
import spacy

from agent.data.entities.config import BACKUPS_PATH, ROOT_LOGGER_ID
from agent.data.sql.sql_mgmt import execute_query, update_non_text_field, get_connection
from agent.data.sql.sql_mgmt import select_fields_where, select_one, exist_where
from agent.data.sql.sql_mgmt import get_connection

logger = logging.getLogger(ROOT_LOGGER_ID)


if __name__ == "__main__":

    logging.info("Exporting datasets...")

    connection = get_connection()
    item_fields = "id, url, review_url, item_class, item_class_4"
    items = select_fields_where(connection, "annotate_item", item_fields, "review_level > 0 AND skip_validations = false ORDER BY id")

    sentence_fields = "new_sentence_id, check_worthy, sentence_class, instance_type"

    dataset = []

    for item in items:
        item_id = item[0]
        url = item[1]
        review_url = item[2]
        item_class = item[3]
        item_class_4 = item[4]

        instance = {}
        instance["item_id"] = item_id
        instance["item_url"] = url
        instance["review_url"] = review_url
        instance["item_class_mc"] = item_class_4
        if item_class == 1:
            instance["item_class"] = "T"
        else:
            instance["item_class"] = "F"

        sentences = select_fields_where(connection, "annotate_sentence", sentence_fields, f"review_level > 0 AND skip_validations = false AND item_id = {item_id} ORDER BY new_sentence_id")

        sentences_list = []

        for sentence in sentences:
            new_sentence_id = sentence[0]
            check_worthy = sentence[1]
            sentence_class = sentence[2]
            instance_type = sentence[3]

            sentence_instance = {}
            sentence_instance["sentence_id"] = new_sentence_id
            sentence_instance["check_worthy"] = check_worthy

            sentence_instance["sentence_class_mc"] = sentence_class

            if sentence_class == "T":
                sentence_instance["sentence_class"] = 'T'
            else:
                sentence_instance["sentence_class"] = 'F'

            sentence_instance["pipeline_instance_type"] = instance_type

            if instance_type == 'Dev' or instance_type == 'Other':
                instance_type = 'Test'
            sentence_instance["instance_type"] = instance_type

            sentences_list.append(sentence_instance)

        instance["sentences"] = sentences_list

        dataset.append(instance)

    with open("temp/keane_dataset.json", "w", encoding="utf-8") as write_file:
        json.dump(dataset, write_file, indent=4, separators=(",", ": "), default=str)
