import logging
import json

from agent.data.entities.config import BACKUPS_PATH
from agent.data.sql.sql_mgmt import execute_query, execute_read_query, get_connection, select_fields_where


logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def insert_tweeter_user_instance(connection, tweeter_user, crawl_date):
    insert_tweeter_user_qry = f"""
        INSERT INTO
        twitter_users (
            id_str, name, screen_name, location, description,
            followers_count, friends_count, listed_count, favourites_count,
            statuses_count, created_at, verified, crawl_date)
        VALUES
        ('{tweeter_user.id_str}',
         '{tweeter_user.name.replace("'", "''")}',
         '{tweeter_user.screen_name.replace("'", "''")}',
         '{tweeter_user.location.replace("'", "''")}',
         '{tweeter_user.description.replace("'", "''")}',
          {tweeter_user.followers_count},
          {tweeter_user.friends_count},
          {tweeter_user.listed_count},
          {tweeter_user.favourites_count},
          {tweeter_user.statuses_count},
         '{tweeter_user.created_at}',
          {tweeter_user.verified},
         '{crawl_date}')
        """
    execute_query(connection, insert_tweeter_user_qry)

    insert_tweeter_user_qry = f"""
        INSERT INTO
        annotate_twitteruser (
            id_str, name, screen_name, location, description,
            followers_count, friends_count, listed_count, favourites_count,
            statuses_count, created_at, verified, crawl_date)
        VALUES
        ('{tweeter_user.id_str}',
         '{tweeter_user.name.replace("'", "''")}',
         '{tweeter_user.screen_name.replace("'", "''")}',
         '{tweeter_user.location.replace("'", "''")}',
         '{tweeter_user.description.replace("'", "''")}',
          {tweeter_user.followers_count},
          {tweeter_user.friends_count},
          {tweeter_user.listed_count},
          {tweeter_user.favourites_count},
          {tweeter_user.statuses_count},
         '{tweeter_user.created_at}',
          {tweeter_user.verified},
         '{crawl_date}')
        """
    execute_query(connection, insert_tweeter_user_qry)


def exist_twitter_user(connection, id_str):
    exist_twitter_user_qry = f"""
        SELECT id_str
        FROM annotate_twitteruser
        WHERE id_str = '{id_str}'
        """
    return len(execute_read_query(connection, exist_twitter_user_qry))


def export_to_json():
    connection = get_connection()

    twitter_user = select_fields_where(connection, "annotate_twitteruser", '*', 'true', return_dict=True)

    with open(BACKUPS_PATH + "all_twitter_users.json", "w", encoding="utf-8") as write_file:
        json.dump(twitter_user, write_file, indent=4, separators=(",", ": "), default=str)