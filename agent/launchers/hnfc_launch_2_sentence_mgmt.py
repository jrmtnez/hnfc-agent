import logging

from datetime import datetime

from agent.data.entities.config import ROOT_LOGGER_ID
from agent.data.sql.sql_mgmt import get_connection
from agent.data.entities.sentence import generate_sentences_from_items


logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(ROOT_LOGGER_ID)
logger.setLevel(logging.INFO)


if __name__ == "__main__":

    logging.info("Launching sentence management...")

    starting_time = datetime.now()

    connection = get_connection()
    generate_sentences_from_items(connection)
    connection.commit()
    connection.close()

    ending_time = datetime.now()
    logging.info("Total time: %s", ending_time - starting_time)
