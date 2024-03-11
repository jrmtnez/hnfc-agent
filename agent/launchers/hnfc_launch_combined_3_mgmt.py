import logging
import fnmatch
import os
import time

from datetime import datetime

from agent.data.entities.config import ROOT_LOGGER_ID, METAMAP_INPUT_PATH
from agent.data.sql.sql_mgmt import get_connection
from agent.nlp.metamap_mgmt import parse_and_update_annotate_sentences, update_common_terms

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(ROOT_LOGGER_ID)
logger.setLevel(logging.INFO)


if __name__ == "__main__":

    starting_time = datetime.now()

    connection = get_connection()

    number_of_files = len(fnmatch.filter(os.listdir(METAMAP_INPUT_PATH), '*.txt'))

    while len(fnmatch.filter(os.listdir(METAMAP_INPUT_PATH), '*.xml')) < number_of_files:
        logging.info("Waiting for MetaMap to finish...")
        time.sleep(30)

    logging.info("Importing biomedical data from MetaMap...")

    parse_and_update_annotate_sentences(connection)
    connection.commit()
    update_common_terms(connection)
    connection.commit()
    connection.close()

