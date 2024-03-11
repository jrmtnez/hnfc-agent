import time
import logging
from datetime import date

import re
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from agent.data.entities.config import ROOT_LOGGER_ID
from agent.crawlers.twitter_crawler import get_tweets
from agent.crawlers.utils.scraping_utils import get_paragraphs
from agent.crawlers.utils.random_header import get_random_header
from agent.crawlers.utils.check_urls import is_url_allowed, is_url_ok
from agent.data.entities.item import insert_crawled_item_instance, exist_item

logger = logging.getLogger(ROOT_LOGGER_ID)


def crawl(connection, url, last_page, category):

    logging.info("Crawling Politifact: %s", category)

    result = []

    header = get_random_header()

    for page in tqdm(range(last_page), total=last_page):

        if (page % 3) <= 0:
            time.sleep(10)

        req = requests.get(url + str(page + 1) + "&category=" + category)
        soup = BeautifulSoup(req.text, "lxml")


        item_list = soup.find_all("li", {"class": "o-listicle__item"})
        for item in item_list:

            item_dict = crawl_item(connection, header, item, category)

            if item_dict is not None:
                result.append(item_dict)
                insert_crawled_item_instance(connection, item_dict, date.today(), 0)
                # get_tweets(connection, item_dict["title"], item_dict["abstract"], item_dict["type"],
                #            item_dict["rating"], item_dict["class"])
                connection.commit()

    return result


def crawl_item(connection, header, item, category):

    item_dict = {}

    item_type_text = "Politifact " + category
    item_title_text = ""
    item_abstract_text = ""
    item_date_text = ""
    item_source_entity_text = ""
    item_url = ""
    item_text = ""

    review_entity_text = "Politifact"
    review_url = ""
    claim_text = ""
    rating_text = ""
    review_summary_text = ""

    item_class_4 = 'F'
    rating = -1

    item_title_block = item.find("div", {"class": "m-statement__quote"})
    if item_title_block is not None:
        item_title = item_title_block.find("a")
        if item_title is not None:

            logging.debug("URL: %s", item_title_block.find("a").get("href"))

            item_title_text = item_title.text.strip()
            claim_text = item_title.text.strip()

            if exist_item(connection, item_title_text, review_entity_text) == 0:

                item_date = item.find("footer", {"class": "m-statement__footer"})
                if item_date is not None:
                    item_date_text = item_date.text.split("•")[-1].strip()

                item_rating = item.find("div", {"class": "m-statement__meter"})
                if item_rating is not None:
                    item_rating_img = item_rating.find("img", alt=True)
                    if item_rating_img["alt"] is not None:
                        rating_text = item_rating_img["alt"].strip()

                        rating_criteria = rating_text
                        if rating_criteria == "true":
                            rating = 100
                            item_class_4 = 'T'
                        if rating_criteria == "mostly-true":
                            rating = 75
                            item_class_4 = 'T'
                        if rating_criteria == "barely-true":
                            rating = 50
                            item_class_4 = 'PF'
                        if rating_criteria == "half-true":
                            rating = 50
                            item_class_4 = 'PF'
                        if rating_criteria == "pants-fire":
                            rating = 0
                            item_class_4 = 'F'
                        if rating_criteria == "false":
                            rating = 0
                            item_class_4 = 'F'

                item_rating = item.find("div", {"class": "m-statement__meter"})
                if item_rating is not None:
                    item_rating_img = item_rating.find("img", alt=True)
                    if item_rating_img["alt"] is not None:
                        rating_text = item_rating_img["alt"].strip()

                item_rating = item.find("div", {"class": "m-statement__meter"})
                if item_rating is not None:
                    item_rating_img = item_rating.find("img", alt=True)
                    if item_rating_img["alt"] is not None:
                        rating_text = item_rating_img["alt"].strip()

                review_url = item_title_block.find("a").get("href")

                if review_url is not None:
                    review_url = "https://www.politifact.com" + review_url
                    req2 = requests.get(review_url)
                    soup2 = BeautifulSoup(req2.text, "lxml")

                    review_content = soup2.find("html")

                    item_abstract = review_content.find("meta", property="og:description")
                    if item_abstract["content"] is not None:
                        item_abstract_text = item_abstract["content"].strip()

                    item_body = review_content.find("article", {"class": "m-textblock"})
                    if item_body is not None:
                        review_summary_text = item_body.text

                        item_urls = item_body.find_all("a")
                        for url in item_urls:
                            item_url = url.get("href")

                            if item_url is not None and is_url_allowed(item_url) and is_url_ok(item_url):
                                try:
                                    req3 = requests.get(item_url, headers=header)
                                except BaseException:
                                    req3 = None

                                if req3 is not None:
                                    soup3 = BeautifulSoup(req3.text, "lxml")
                                    item_content = soup3.find("html")
                                    if item_content is not None:
                                        item_text = get_paragraphs(item_content).strip()
                                        item_source_entity_text = re.findall('(?<=//)([^/]+)', item_url)[0]
                                        break
                                else:
                                    item_url = ""
                            else:
                                item_url = ""
                else:
                    review_url = ""

                item_dict["title"] = item_title_text
                item_dict["type"] = item_type_text
                item_dict["abstract"] = item_abstract_text
                item_dict["publication_date"] = item_date_text
                item_dict["source_entity"] = item_source_entity_text
                item_dict["url"] = item_url
                item_dict["text"] = item_text
                item_dict["review_entity"] = review_entity_text
                item_dict["review_url"] = review_url
                item_dict["claim"] = claim_text
                item_dict["original_rating"] = rating_text
                item_dict["review_summary"] = review_summary_text
                item_dict["skip_validations"] = True

                item_dict["rating"] = rating
                item_dict["item_class_4"] = item_class_4

                if item_dict["rating"] > 50:
                    item_dict["class"] = 1
                else:
                    item_dict["class"] = 0

                item_dict["review_entity_2"] = ""
                item_dict["review_url_2"] = ""
                item_dict["lang"] = ""
                item_dict["country"] = ""

                return item_dict
    return None
