import logging

from tqdm import tqdm

from agent.data.sql.sql_mgmt import get_connection, update_non_text_field, select_fields_where, select_one

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

ITEMS_TABLE = "annotate_item"
SENTENCES_TABLE = "annotate_sentence"


def update_sentence_review_level(from_level, to_level=None, to_level_without_validations=None):
    logging.info("Updating sentence review level from %s to %s and %s.", from_level, to_level, to_level_without_validations)
    connection = get_connection()

    if to_level is not None:
        sentence_filter = f"review_level = {from_level} AND skip_validations = false"
        update_non_text_field(connection, SENTENCES_TABLE, sentence_filter, "review_level", to_level)

    if to_level_without_validations is not None:
        sentence_filter = f"review_level = {from_level} AND skip_validations = true"
        update_non_text_field(connection, SENTENCES_TABLE, sentence_filter, "review_level", to_level_without_validations)

    connection.commit()
    connection.close()


def update_item_review_level():
    logging.info("Updating item review level...")

    connection = get_connection()

    annotate_items = select_fields_where(connection, ITEMS_TABLE, "id", "review_level > 0 ORDER BY id")

    for annotate_item in tqdm(annotate_items):
        sentence_filter = f"item_id = {annotate_item[0]} AND review_level > 0"
        review_level = select_one(connection, SENTENCES_TABLE, "review_level", where=sentence_filter, ascending=False)
        update_non_text_field(connection, ITEMS_TABLE, f"id = {annotate_item[0]}", "review_level", review_level)
    connection.commit()
    connection.close()


def update_sentence_review_level_spo_to_check(review_level, new_review_level, full_pipeline=False):
    #
    # Move sentences to check spos to next level
    #

    if full_pipeline:
        # ya se han puesto las instancias de training con check_worthy_score_auto = 1 o 0

        # sentence_filter = f"""
        #     review_level = {review_level} AND
        #     check_worthy_score_auto >= 0.5 AND
        #     instance_type = 'Dev' AND
        #     skip_validations = false
        #     """
        sentence_filter = f"""
            review_level = {review_level} AND
            check_worthy_score_auto >= 0.5 AND
            skip_validations = false
            """

    else:
        sentence_filter = f"""
            review_level = {review_level} AND
            check_worthy = 'FR' AND
            skip_validations = false
            """

    connection = get_connection()
    update_non_text_field(connection, SENTENCES_TABLE, sentence_filter, "review_level", new_review_level)
    connection.commit()
    connection.close()


def update_sentence_review_level_spo_completed(review_level, new_review_level, full_pipeline=False):
    #
    # Move completed sentences to new level
    #
    if full_pipeline:
        # ya se han puesto las instancias de training con check_worthy_score_auto = 1 o 0

        # sentence_filter = f"""
        #     review_level = {review_level} AND
        #     check_worthy_score_auto >= 0.5 AND
        #     skip_validations = false AND
        #     instance_type = 'Dev' AND
        #     big_subject <> '' AND
        #     big_predicate <> '' AND
        #     big_object <> ''
        #     """
        sentence_filter = f"""
            review_level = {review_level} AND
            check_worthy_score_auto >= 0.5 AND
            skip_validations = false AND
            big_subject <> '' AND
            big_predicate <> '' AND
            big_object <> ''
            """
    else:
        sentence_filter = f"""
            review_level = {review_level} AND
            check_worthy = 'FR' AND
            skip_validations = false AND
            manually_identified_predicate = big_predicate AND
            manually_identified_predicate <> ''
            """

    connection = get_connection()
    update_non_text_field(connection, SENTENCES_TABLE, sentence_filter, "review_level", new_review_level)
    connection.commit()
    connection.close()


def update_sentence_review_level_spo_skip_validations(review_level, new_review_level):
    #
    # Move completed sentences to check to new level (sencences external to the corpus)
    #

    sentence_filter = f"""
        review_level = {review_level} AND
        check_worthy_auto = 'FR' AND
        skip_validations = true AND
        big_subject <> '' AND
        big_predicate <> '' AND
        big_object <> ''
        """

    connection = get_connection()
    update_non_text_field(connection, SENTENCES_TABLE, sentence_filter, "review_level", new_review_level)
    connection.commit()
    connection.close()
