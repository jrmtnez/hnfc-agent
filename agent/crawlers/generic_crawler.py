import logging
import requests

from os.path import exists, join
from bs4 import BeautifulSoup
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from agent.data.entities.config import ROOT_LOGGER_ID, RAW_DATA_PATH
from agent.crawlers.utils.random_header import get_random_header
from agent.crawlers.utils.scraping_utils import get_paragraphs

logger = logging.getLogger(ROOT_LOGGER_ID)


def crawl_selenium(item_id, url, driver, legacy_mode=True):

    logging.info("Checking item %s", item_id)

    file_path = join(RAW_DATA_PATH, str(item_id) + ".txt")

    if not exists(file_path):
        wait = WebDriverWait(driver, 120)
        driver.get(url)
        get_url = driver.current_url
        wait.until(EC.url_to_be(url))
        if get_url == url:
            page_source = driver.page_source

            soup = BeautifulSoup(page_source, "lxml")

            item_text = ""
            if legacy_mode:
                item_content = soup.find("html")
                if item_content is not None:
                    item_text = get_paragraphs(item_content).strip()
            else:
                item_text = soup.get_text()

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(item_text)


def crawl(item_id, url, legacy_mode=True):

    logging.info("Checking item %s", item_id)

    header = get_random_header()

    file_path = join(RAW_DATA_PATH, str(item_id) + ".txt")

    if not exists(file_path):
        logging.info("Scrapping item %s: %s", item_id, url)

        req = None
        try:
            req = requests.get(url, headers=header, timeout=120)
        except Exception:
            req = requests.get(url, headers=header, timeout=120, verify=False)

        soup = BeautifulSoup(req.text, "lxml")

        item_text = ""
        if legacy_mode:
            item_content = soup.find("html")
            if item_content is not None:
                item_text = get_paragraphs(item_content).strip()
        else:
            item_text = soup.get_text()

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(item_text)
