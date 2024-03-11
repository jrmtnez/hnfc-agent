import json
import logging

from datetime import datetime

from agent.data.entities.config import ROOT_LOGGER_ID
from agent.data.sql.sql_mgmt import get_connection
from agent.crawlers.snopes_crawler import crawl as crawl_snopes
from agent.crawlers.politifact_crawler import crawl as crawl_politifact
from agent.crawlers.fullfact_crawler import crawl as crawl_fullfact

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(ROOT_LOGGER_ID)
logger.setLevel(logging.INFO)


if __name__ == "__main__":
    logging.info("Launching agent...")

    starting_time = datetime.now()

    connection = get_connection()

    # --- Snopes ---
    for category in ["health", "health-politics"]:
        url_to_crawl = f"https://www.snopes.com/fact-check/category/{category}/?pagenum="
        crawled_data = crawl_snopes(connection, url_to_crawl, 5, category)

        with open(f"data/snopes_{category}_last.json", "w", encoding="utf-8") as write_file:
            json.dump(crawled_data, write_file, indent=4, separators=(",", ": "))

      # --- Politifact ---
    url_to_crawl = "https://www.politifact.com/factchecks/list/?page="
    for category in ["health-check", "coronavirus"]:
        crawled_data = crawl_politifact(connection, url_to_crawl, 5, category)

        with open("data/politifact_" + category + "_last.json", "w", encoding="utf-8") as write_file:
            json.dump(crawled_data, write_file, indent=4, separators=(",", ": "))

    # --- Full Fact ---
    url_to_crawl = "https://fullfact.org/health/all/?page="
    crawled_data = crawl_fullfact(connection, url_to_crawl, 5)


    ending_time = datetime.now()
    logging.info("Total time: %s", ending_time - starting_time)
