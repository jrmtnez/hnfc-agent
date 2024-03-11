import logging

from agent.data.entities.config import ROOT_LOGGER_ID
from agent.data.sql.sql_mgmt import get_connection
from agent.data.entities.sentence import export_annotate_sentences
from agent.data.entities.config import METAMAP_INPUT_PATH

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(ROOT_LOGGER_ID)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    connection = get_connection()
    export_annotate_sentences(connection, METAMAP_INPUT_PATH, clean_image_chars=True)
    connection.commit()
    connection.close()
