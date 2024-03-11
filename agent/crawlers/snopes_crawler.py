import time
import logging
from datetime import date

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


def crawl(connection, base_url, last_page, category):

    logging.info("Crawling Snopes - %s", category)

    result = []

    header = get_random_header()

    for page in tqdm(range(last_page), total=last_page):

        if (page % 3) <= 0:
            time.sleep(10)

        url = base_url + str(page + 1)
        req = requests.get(url, headers=header)
        soup = BeautifulSoup(req.text, "lxml")

        item_list_container = soup.find("div", {"class": "article_list_cont"})

        if item_list_container is not None:

            item_list = item_list_container.find_all("div", {"class": "article_wrapper"})
            for item in item_list:

                item_dict = crawl_item(connection, header, item)

                if item_dict is not None:
                    result.append(item_dict)
                    insert_crawled_item_instance(connection, item_dict, date.today(), 0)
                    # get_tweets(connection, item_dict["title"], item_dict["abstract"], item_dict["type"],
                    #            item_dict["rating"], item_dict["class"])
                    connection.commit()

    return result


def crawl_item(connection, header, item):

    item_dict = {}

    item_type_text = "Claim Review"
    item_title_text = ""
    item_abstract_text = ""
    item_date_text = ""
    item_source_entity_text = ""
    item_url = ""
    item_text = ""

    review_entity_text = "Snopes"
    review_url = ""
    claim_text = ""
    rating_text = ""
    review_summary_text = ""

    item_title = item.find("h3", {"class": "article_title"})
    if item_title is not None:
        item_title_text = item_title.text.strip()

        if exist_item(connection, item_title_text, review_entity_text) == 0:

            review_url_item = item.find("a", {"class": "outer_article_link_wrapper"})
            if review_url_item is not None:

                review_url = review_url_item.get("href")

                req2 = requests.get(review_url, headers=header)
                soup2 = BeautifulSoup(req2.text, "lxml")

                review_content = soup2.find("main", {"id": "article_main"})

                title_container = review_content.find("section", {"class": "title-container"})

                item_abstract = title_container.find("h2")
                item_abstract_text = item_abstract.text.strip()


                item_date = review_content.find("span", {"class": "updated_date"})
                if item_date is None:
                    item_date = review_content.find("h3", {"class": "publish_date"})
                if item_date is not None:
                    item_date_text = item_date.text
                    item_date_text = item_date_text.strip()
                    item_date_list = item_date_text.split()[1:]
                    item_date_text = " ".join(item_date_list)


                item_claim = review_content.find("div", {"class": "claim_cont"})
                if item_claim is not None:
                    claim_text = item_claim.text.strip()

                rating_img_wrap = review_content.find("div", {"class": "rating_img_wrap"})
                if rating_img_wrap is not None:
                    rating_img = rating_img_wrap.find("img", {"class": "lazy-image"}, alt=True)
                    rating_text = rating_img["alt"]

                item_card = review_content.find("article", {"id": "article-content"})

                if item_card is not None:

                    review_summary_text = get_paragraphs(item_card)

                    item_urls = item_card.find_all("a")
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
            item_dict["skip_validations"] = False

            item_class_4 = 'F'
            rating = 0
            rating_criteria = item_dict["original_rating"].strip()
            if rating_criteria == "True":
                rating = 100
                item_class_4 = 'T'
            if rating_criteria == "Correct Attribution":
                rating = 100
                item_class_4 = 'T'
            if rating_criteria == "Mostly True":
                rating = 75
                item_class_4 = 'T'
            if rating_criteria == "Mixture":
                rating = 50
                item_class_4 = 'PF'
            if rating_criteria == "Outdated":
                rating = 50
                item_class_4 = 'PF'
            if rating_criteria == "Mostly False":
                rating = 25
                item_class_4 = 'PF'
            if rating_criteria == "Unproven":
                rating = 25
                item_class_4 = 'PF'
            if rating_criteria == "Labeled Satire":
                rating = 25
                item_class_4 = 'PF'
            if rating_criteria == "Legend":
                rating = 25
                item_class_4 = 'PF'
            if rating_criteria == "Miscaptioned":
                rating = 25
                item_class_4 = 'PF'
            if rating_criteria == "Misattributed":
                rating = 25
                item_class_4 = 'PF'
            if rating_criteria == "Research In Progress":
                rating = 25
                item_class_4 = 'NA'

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
