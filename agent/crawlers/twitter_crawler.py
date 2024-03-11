import tweepy
import time
import datetime
import json
import logging

from agent.data.entities.config import ROOT_LOGGER_ID
from agent.keys.twitter_keys import CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET
from agent.nlp.keyword_extractor import get_key_terms
from agent.data.entities.tweet import exist_twitter_tweet, insert_tweeter_tweet_instance
from agent.data.entities.twitter_user import exist_twitter_user, insert_tweeter_user_instance


logger = logging.getLogger(ROOT_LOGGER_ID)


def get_tweets(connection, title, abstract, tweet_type, item_rating, item_class):

    query_terms = get_key_terms(title + " " + abstract)
    logging.debug("Query terms: %s.", query_terms)

    if query_terms == "":
        return

    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    api = tweepy.API(auth, wait_on_rate_limit=True)

    tweets_list = []
    text_query = query_terms + " -filter:retweets"
    count = 100
    today_datetime = str(datetime.datetime.today())

    try:
        for tweet in api.search_tweets(q=text_query, count=count, lang="en", tweet_mode="extended"):
            tweets_list.append(tweet._json)

            if not exist_twitter_tweet(connection, tweet.id_str):
                insert_tweeter_tweet_instance(connection, tweet, title, tweet_type, today_datetime, item_rating, item_class)

            if not exist_twitter_user(connection, tweet.user.id_str):
                insert_tweeter_user_instance(connection, tweet.user, today_datetime)
    except BaseException as e:
        print("Failed on_status,", str(e))
        time.sleep(3)

    with open("data/" + "tweets_last.json", "w", encoding="utf-8") as write_file:
        json.dump(tweets_list, write_file, indent=4, separators=(",", ": "))
