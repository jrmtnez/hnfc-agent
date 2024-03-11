import logging

from agent.data.entities.config import ROOT_LOGGER_ID
from agent.data.sql.sql_mgmt import get_connection
from agent.nlp.metamap_mgmt import parse_and_update_annotate_sentences, update_common_terms

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(ROOT_LOGGER_ID)
logger.setLevel(logging.INFO)


if __name__ == "__main__":
    connection = get_connection()
    parse_and_update_annotate_sentences(connection)
    connection.commit()
    update_common_terms(connection)
    connection.commit()
    connection.close()
