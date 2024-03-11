import logging
import json

from agent.data.entities.config import BACKUPS_PATH, ROOT_LOGGER_ID
from agent.data.sql.sql_mgmt import execute_query, get_connection
from agent.data.sql.sql_mgmt import select_fields_where

logger = logging.getLogger(ROOT_LOGGER_ID)


def insert_triple(connection, triple_dict):
    insert_triple_qry = f"""
        INSERT INTO public.annotate_triple(
            item_id, sentence_id, publication_date, subject, predicate, object, sentence_class,
            sentence_check_worthy, item_class_4, instance_type, score, exist_so_in_kb)
        VALUES (
             {triple_dict["item_id"]},
             {triple_dict["sentence_id"]},
            '{triple_dict["publication_date"]}',
            '{triple_dict["subject"].replace("'", "''")}',
            '{triple_dict["predicate"].replace("'", "''")}',
            '{triple_dict["object"].replace("'", "''")}',
            '{triple_dict["sentence_class"].replace("'", "''")}',
            '{triple_dict["sentence_check_worthy"].replace("'", "''")}',
            '{triple_dict["item_class_4"].replace("'", "''")}',
            '{triple_dict["instance_type"].replace("'", "''")}',
             {triple_dict["score"]},
             {triple_dict["exist_so_in_kb"]}
            )
        """
    execute_query(connection, insert_triple_qry)


def triple_tuple_to_dict(triple_tuple):
    triple_dict = {}
    triple_dict["id"] = triple_tuple[0]
    triple_dict["item_id"] = triple_tuple[1]
    triple_dict["sentence_id"] = triple_tuple[2]
    triple_dict["publication_date"] = triple_tuple[3]
    triple_dict["subject"] = triple_tuple[4]
    triple_dict["predicate"] = triple_tuple[5]
    triple_dict["object"] = triple_tuple[6]
    triple_dict["sentence_class"] = triple_tuple[7]
    triple_dict["sentence_check_worthy"] = triple_tuple[8]
    triple_dict["item_class_4"] = triple_tuple[9]
    triple_dict["instance_type"] = triple_tuple[10]
    triple_dict["score"] = triple_tuple[11 ]
    triple_dict["exist_so_in_kb"] = triple_tuple[12]
    return triple_dict


def export_to_json():
    connection = get_connection()

    triples = select_fields_where(connection, "annotate_triple", '*', 'true', return_dict=True)

    with open(BACKUPS_PATH + "all_triples.json", "w", encoding="utf-8") as write_file:
        json.dump(triples, write_file, indent=4, separators=(",", ": "), default=str)

