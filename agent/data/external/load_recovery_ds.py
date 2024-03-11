import logging
import pandas as pd


from datetime import date, datetime

from agent.data.entities.config import ROOT_LOGGER_ID
from agent.data.sql.sql_mgmt import get_connection
from agent.data.entities.item import insert_item, item_tuple_to_dict

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(ROOT_LOGGER_ID)
logger.setLevel(logging.INFO)


RECOVERY_DS_NEWS_DATA = "data/external_ds/recovery/recovery-news-data.csv"


def import_items():


    connection = get_connection()

    # columns:
    # Unnamed: 0,news_id,url,publisher,publish_date,author,title,image,body_text,political_bias,country,reliability

    data = pd.read_csv(RECOVERY_DS_NEWS_DATA)

    for _, item in data.iterrows():
    
        logger.info("%s %s %s", item["news_id"], item["publish_date"], item["title"])

        if item["reliability"] == 1:
            item_class_4 = 'T'
        else:
            item_class_4 = 'F'

        title = str(item["title"])
        if str(title) == "nan":
            title = ""

        publication_date = item["publish_date"]
        if str(publication_date) == "nan":
            publication_date = "1 January 1753"


        item_tuple = (
            0,
            title,
            'ReCOVery',
            '',
            publication_date,
            item["publisher"],        
            item["url"],        
            item["body_text"],        
            '',
            '',
            '',
            '',
            "Political bias: " + str(item["political_bias"]),        
            item["reliability"] * 100,        
            item["reliability"],        
            date.today(),    
            1,
            True,
            False,
            False,
            True,
            0,
            'NA',
            item_class_4,
            'Other',
            '',
            '',
            '',
            item["country"],                
            '',
            '',
            '',
            '',
            item["news_id"])
    
        insert_item(connection, item_tuple_to_dict(item_tuple))

    connection.commit()
    connection.close()

if __name__ == "__main__":

    logging.info("Launching sentence management...")

    starting_time = datetime.now()

    import_items()

    ending_time = datetime.now()
    logging.info("Total time: %s", ending_time - starting_time)


