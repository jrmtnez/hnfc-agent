import logging

from agent.data.sql.sql_mgmt import get_connection, update_text_field

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

TABLE = "annotate_sentence"


def do_preliminary_fc_classification(external=False):

    logger.info("Doing preliminary fc classification (do_preliminary_fc_classification)")

    if external:
        field_name = "check_worthy_auto"
    else:
        field_name = "check_worthy"

    connection = get_connection()
    update_text_field(connection, TABLE,
                      f"review_level = 5 AND {field_name} = 'FR' AND rating < 25 AND skip_validations = {external}",
                      "sentence_class", "F")
    update_text_field(connection, TABLE,
                      f"review_level = 5 AND {field_name} = 'FR' AND rating >= 25 AND rating <= 50 AND skip_validations = {external}",
                      "sentence_class", "PF")
    update_text_field(connection, TABLE,
                      f"review_level = 5 AND {field_name} = 'FR' AND rating > 50 AND skip_validations = {external}",
                      "sentence_class", "T")
    connection.commit()
