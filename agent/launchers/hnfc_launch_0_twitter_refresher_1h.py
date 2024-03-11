import logging
from datetime import datetime

from agent.data.entities.config import ROOT_LOGGER_ID
from agent.data.sql.sql_mgmt import get_connection
from agent.data.entities.item import select_last_items
from agent.crawlers.twitter_crawler import get_tweets

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(ROOT_LOGGER_ID)
logger.setLevel(logging.INFO)


if __name__ == "__main__":

    logging.info("Launching twitter refresher...")
    starting_time = datetime.now()

    connection = get_connection()
    last_items = select_last_items(connection)
    for item in last_items:
        logging.info("Refreshing: %s %s",item[3], item[0])
        get_tweets(connection, item[0], item[2], item[1], item[4], item[5])

    connection.commit()
    ending_time = datetime.now()
    logging.info("Total time: %s", ending_time - starting_time)
