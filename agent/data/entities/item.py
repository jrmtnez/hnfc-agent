import logging
import json

from agent.data.entities.config import BACKUPS_PATH, ROOT_LOGGER_ID
from agent.data.sql.sql_mgmt import get_connection, execute_query, execute_read_query, select_fields_where

logger = logging.getLogger(ROOT_LOGGER_ID)


def insert_crawled_item_instance(connection, item, crawl_date, review_level):
    publication_date = item["publication_date"]
    if publication_date == "":
        publication_date = "1 January 1753"

    insert_item_qry = f"""
        INSERT INTO
        annotate_item (
            title, type, abstract, publication_date, source_entity,
            url, text, review_entity, review_url, claim,
            original_rating, review_summary,
            rating, item_class, crawl_date, review_level,
            url_ok, no_health_useless, needs_revision, skip_validations,
            item_class_auto, item_class_4_auto, item_class_4,
            instance_type, revision_reason, review_url_2, review_entity_2,
            country, lang, main_claim, main_topic, instance_type_2, from_item_id)
        VALUES
        ('{item["title"].replace("'", "''")}',
         '{item["type"].replace("'", "''")}',
         '{item["abstract"].replace("'", "''")}',
         '{publication_date}',
         '{item["source_entity"].replace("'", "''")}',
         '{item["url"].replace("'", "''")}',
         '{item["text"].replace("'", "''")}',
         '{item["review_entity"].replace("'", "''")}',
         '{item["review_url"].replace("'", "''")}',
         '{item["claim"].replace("'", "''")}',
         '{item["original_rating"].replace("'", "''")}',
         '{item["review_summary"].replace("'", "''")}',
          {item["rating"]},
          {item["class"]},
         '{crawl_date}',
          {review_level},
          false,
          false,
          false,
          {item["skip_validations"]},
          0,
         'NA',
         '{item["item_class_4"]}',
         'Other',
         '',
         '{item["review_url_2"]}',
         '{item["review_entity_2"]}',
         '{item["country"]}',
         '{item["lang"]}',
         '',
         '',
         '',
         0)
        """
    execute_query(connection, insert_item_qry)


def exist_item(connection, title, review_entity):
    # apostrophes not properly escaped in some cases
    exist_item_qry = f"""
        SELECT title
        FROM annotate_item
        WHERE title = '{title.replace("'", "''")}'
        AND review_entity = '{review_entity.replace("'", "''")}'
        """
    return len(execute_read_query(connection, exist_item_qry))


def select_last_items(connection):
    select_last_items_qry = """
        SELECT
            title,
            type,
            abstract,
            publication_date as "d [date]",
            rating,
            item_class
        FROM annotate_item
        ORDER BY publication_date DESC, id DESC LIMIT 15
        """
    return execute_read_query(connection, select_last_items_qry)


def export_to_json():
    connection = get_connection()

    items = select_fields_where(connection, "annotate_item", '*', 'skip_validations = true', return_dict=True)

    with open(BACKUPS_PATH + "all_items_external.json", "w", encoding="utf-8") as write_file:
        json.dump(items, write_file, indent=4, separators=(",", ": "), default=str)

    items = select_fields_where(connection, "annotate_item", '*', 'skip_validations = false', return_dict=True)

    with open(BACKUPS_PATH + "all_items_corpus.json", "w", encoding="utf-8") as write_file:
        json.dump(items, write_file, indent=4, separators=(",", ": "), default=str)


def insert_item(connection, item_dict):
    publication_date = item_dict["publication_date"]
    if publication_date == "":
        publication_date = "1 January 1753"

    insert_item_qry = f"""
        INSERT INTO
        annotate_item (
            title, type, abstract, publication_date, source_entity,
            url, text, review_entity, review_url, claim,
            original_rating, review_summary,
            rating, item_class, crawl_date, review_level,
            url_ok, no_health_useless, needs_revision, skip_validations,
            item_class_auto, item_class_4_auto, item_class_4,
            instance_type, revision_reason, review_url_2, review_entity_2,
            country, lang, main_claim, main_topic, instance_type_2, from_item_id)
        VALUES
        ('{item_dict["title"].replace("'", "''")}',
         '{item_dict["type"].replace("'", "''")}',
         '{item_dict["abstract"].replace("'", "''")}',
         '{item_dict["publication_date"]}',
         '{item_dict["source_entity"].replace("'", "''")}',
         '{item_dict["url"].replace("'", "''")}',
         '{item_dict["text"].replace("'", "''")}',
         '{item_dict["review_entity"].replace("'", "''")}',
         '{item_dict["review_url"].replace("'", "''")}',
         '{item_dict["claim"].replace("'", "''")}',
         '{item_dict["original_rating"].replace("'", "''")}',
         '{item_dict["review_summary"].replace("'", "''")}',
          {item_dict["rating"]},
          {item_dict["class"]},
         '{item_dict["crawl_date"]}',
          {item_dict["review_level"]},
          {item_dict["url_ok"]},
          {item_dict["no_health_useless"]},
          {item_dict["needs_revision"]},
          {item_dict["skip_validations"]},
          {item_dict["item_class_auto"]},
         '{item_dict["item_class_4_auto"]}',
         '{item_dict["item_class_4"]}',
         '{item_dict["instance_type"]}',
         '{item_dict["revision_reason"]}',
         '{item_dict["review_url_2"]}',
         '{item_dict["review_entity_2"]}',
         '{item_dict["country"]}',
         '{item_dict["lang"]}',
         '{item_dict["main_claim"].replace("'", "''")}',
         '{item_dict["main_topic"].replace("'", "''")}',
         '{item_dict["instance_type_2"]}',
          {item_dict["from_item_id"]})
        """
    execute_query(connection, insert_item_qry)


def item_tuple_to_dict(item_tuple):
    item_dict = {}
    item_dict["id"] = item_tuple[0]
    item_dict["title"] = item_tuple[1]
    item_dict["type"] = item_tuple[2]
    item_dict["abstract"] = item_tuple[3]
    item_dict["publication_date"] = item_tuple[4]
    item_dict["source_entity"] = item_tuple[5]
    item_dict["url"] = item_tuple[6]
    item_dict["text"] = item_tuple[7]
    item_dict["review_entity"] = item_tuple[8]
    item_dict["review_url"] = item_tuple[9]
    item_dict["claim"] = item_tuple[10]
    item_dict["original_rating"] = item_tuple[11]
    item_dict["review_summary"] = item_tuple[12]
    item_dict["rating"] = item_tuple[13]
    item_dict["class"] = item_tuple[14]
    item_dict["crawl_date"] = item_tuple[15]
    item_dict["review_level"] = item_tuple[16]
    item_dict["url_ok"] = item_tuple[17]
    item_dict["no_health_useless"] = item_tuple[18]
    item_dict["needs_revision"] = item_tuple[19]
    item_dict["skip_validations"] = item_tuple[20]
    item_dict["item_class_auto"] = item_tuple[21]
    item_dict["item_class_4_auto"] = item_tuple[22]
    item_dict["item_class_4"] = item_tuple[23]
    item_dict["instance_type"] = item_tuple[24]
    item_dict["revision_reason"] = item_tuple[25]
    item_dict["review_url_2"] = item_tuple[26]
    item_dict["review_entity_2"] = item_tuple[27]
    item_dict["country"] = item_tuple[28]
    item_dict["lang"] = item_tuple[29]
    item_dict["main_claim"] = item_tuple[30]
    item_dict["main_topic"] = item_tuple[31]
    item_dict["instance_type_2"] = item_tuple[32]
    item_dict["from_item_id"] = item_tuple[33]
    return item_dict


def insert_item_with_pk(connection, item_dict):
    publication_date = item_dict["publication_date"]
    if publication_date == "":
        publication_date = "1 January 1753"

    insert_item_qry = f"""
        INSERT INTO
        annotate_item (
            id, title, type, abstract, publication_date, source_entity,
            url, text, review_entity, review_url, claim,
            original_rating, review_summary,
            rating, item_class, crawl_date, review_level,
            url_ok, no_health_useless, needs_revision, skip_validations,
            item_class_auto, item_class_4_auto, item_class_4,
            instance_type, revision_reason, review_url_2, review_entity_2,
            country, lang, main_claim, main_topic, instance_type_2, from_item_id)
        VALUES
        ( {item_dict["id"]},
         '{item_dict["title"].replace("'", "''")}',
         '{item_dict["type"].replace("'", "''")}',
         '{item_dict["abstract"].replace("'", "''")}',
         '{item_dict["publication_date"]}',
         '{item_dict["source_entity"].replace("'", "''")}',
         '{item_dict["url"].replace("'", "''")}',
         '{item_dict["text"].replace("'", "''")}',
         '{item_dict["review_entity"].replace("'", "''")}',
         '{item_dict["review_url"].replace("'", "''")}',
         '{item_dict["claim"].replace("'", "''")}',
         '{item_dict["original_rating"].replace("'", "''")}',
         '{item_dict["review_summary"].replace("'", "''")}',
          {item_dict["rating"]},
          {item_dict["class"]},
         '{item_dict["crawl_date"]}',
          {item_dict["review_level"]},
          {item_dict["url_ok"]},
          {item_dict["no_health_useless"]},
          {item_dict["needs_revision"]},
          {item_dict["skip_validations"]},
          {item_dict["item_class_auto"]},
         '{item_dict["item_class_4_auto"]}',
         '{item_dict["item_class_4"]}',
         '{item_dict["instance_type"]}',
         '{item_dict["revision_reason"]}',
         '{item_dict["review_url_2"]}',
         '{item_dict["review_entity_2"]}',
         '{item_dict["country"]}',
         '{item_dict["lang"]}',
         '{item_dict["main_claim"].replace("'", "''")}',
         '{item_dict["main_topic"].replace("'", "''")}',
         '{item_dict["instance_type_2"]}',
          {item_dict["from_item_id"]})
        """
    execute_query(connection, insert_item_qry)

