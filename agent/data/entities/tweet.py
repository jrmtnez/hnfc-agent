import logging
import json

from agent.data.entities.config import BACKUPS_PATH
from agent.data.sql.sql_mgmt import execute_query, execute_read_query, select_fields_where, get_connection


logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def insert_tweeter_tweet_instance(connection, tweeter_tweet, title, tweet_type, crawl_date, item_rating, item_class):
    in_reply_to_screen_name = ""
    if tweeter_tweet.in_reply_to_screen_name is not None:
        in_reply_to_screen_name = tweeter_tweet.in_reply_to_screen_name.replace("'", "''")
    quoted_status_id_str = ''
    if tweeter_tweet.is_quote_status:
        quoted_status_id_str = tweeter_tweet.quoted_status_id_str

    insert_tweeter_tweet_qry = f"""
        INSERT INTO
        twitter_tweets (
            id_str, created_at, full_text, user_id_str, in_reply_to_status_id_str,
            in_reply_to_user_id_str, in_reply_to_screen_name, is_quote_status,
            quoted_status_id_str, retweet_count, favorite_count, favorited,
            retweeted, lang, title, type, crawl_date, item_rating, item_class)
        VALUES
        ('{tweeter_tweet.id_str}',
         '{tweeter_tweet.created_at}',
         '{tweeter_tweet.full_text.replace("'", "''")}',
         '{tweeter_tweet.user.id_str}',
         '{tweeter_tweet.in_reply_to_status_id_str}',
         '{tweeter_tweet.in_reply_to_user_id_str}',
         '{in_reply_to_screen_name}',
          {tweeter_tweet.is_quote_status},
         '{quoted_status_id_str}',
          {tweeter_tweet.retweet_count},
          {tweeter_tweet.favorite_count},
          {tweeter_tweet.favorited},
          {tweeter_tweet.retweeted},
         '{tweeter_tweet.lang}',
         '{title}',
         '{tweet_type}',
         '{crawl_date}',
          {item_rating},
          {item_class})
        """
    execute_query(connection, insert_tweeter_tweet_qry)

    insert_tweeter_tweet_qry = f"""
        INSERT INTO
        annotate_tweet (
            id_str, created_at, full_text, user_id_str, in_reply_to_status_id_str,
            in_reply_to_user_id_str, in_reply_to_screen_name, is_quote_status,
            quoted_status_id_str, retweet_count, favorite_count, favorited,
            retweeted, lang, title, type, crawl_date, item_rating, item_class,
            check_worthy, tweet_class, needs_revision, review_level)
        VALUES
        ('{tweeter_tweet.id_str}',
         '{tweeter_tweet.created_at}',
         '{tweeter_tweet.full_text.replace("'", "''")}',
         '{tweeter_tweet.user.id_str}',
         '{tweeter_tweet.in_reply_to_status_id_str}',
         '{tweeter_tweet.in_reply_to_user_id_str}',
         '{in_reply_to_screen_name}',
          {tweeter_tweet.is_quote_status},
         '{quoted_status_id_str}',
          {tweeter_tweet.retweet_count},
          {tweeter_tweet.favorite_count},
          {tweeter_tweet.favorited},
          {tweeter_tweet.retweeted},
         '{tweeter_tweet.lang}',
         '{title}',
         '{tweet_type}',
         '{crawl_date}',
          {item_rating},
          {item_class},
         'NF',
          0,
          False,
          0)
        """
    execute_query(connection, insert_tweeter_tweet_qry)


def exist_twitter_tweet(connection, id_str):
    exist_twitter_tweet_qry = f"""
        SELECT id_str
        FROM annotate_tweet
        WHERE id_str = '{id_str}'
        """
    return len(execute_read_query(connection, exist_twitter_tweet_qry))


def export_to_json():
    connection = get_connection()

    tweets = select_fields_where(connection, "annotate_tweet", '*', 'true', return_dict=True)

    with open(BACKUPS_PATH + "all_tweets.json", "w", encoding="utf-8") as write_file:
        json.dump(tweets, write_file, indent=4, separators=(",", ": "), default=str)
