import logging
import psycopg2
import yaml

from psycopg2 import OperationalError
from psycopg2.extras import RealDictCursor

from agent.data.entities.config import CONFIG_FILE_PATH

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def read_config(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def get_connection(db_name=None, db_host=None):
    config = read_config(CONFIG_FILE_PATH)

    if db_name is None:
        db_name = config["database"]["name"]
    if db_host is None:
        db_host = config["database"]["host"]

    connection = create_connection(db_name, config["database"]["username"], config["database"]["password"], db_host, config["database"]["port"])
    return connection


def create_connection(db_name, db_user, db_password, db_host, db_port):
    connection = None
    try:
        connection = psycopg2.connect(
            database=db_name,
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port,
        )
        logger.debug("Connection to %s %s established.", db_host, db_name)
    except OperationalError as e:
        logger.error("create_connection: %s", e)
    return connection


def execute_query(connection, query):
    cursor = connection.cursor()
    try:
        cursor.execute(query)
    except OperationalError as e:
        logger.error("execute_query: %s", e)


def execute_read_query(connection, query, return_dict=False):

    if return_dict:
        cursor = connection.cursor(cursor_factory=RealDictCursor)
    else:
        cursor = connection.cursor()

    result = None
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        return result
    except OperationalError as e:
        logger.error("execute_read_query: %s", e)


def select_where(connection, table, where_condition, return_dict=False):
    select_all_qry = f"""
        SELECT * FROM {table}
        WHERE {where_condition}
        """
    return execute_read_query(connection, select_all_qry, return_dict=return_dict)


def select_fields_where(connection, table, fields, where_condition, return_dict=False):
    select_all_qry = f"""
        SELECT {fields} FROM {table}
        WHERE {where_condition}
        """
    return execute_read_query(connection, select_all_qry, return_dict=return_dict)


def select_one(connection, table, field, where="", ascending=False, return_dict=False):
    ascending_clause = "DESC"
    if ascending:
        ascending_clause = "ASC"
    where_clause = ""
    if where != "":
        where_clause = "WHERE " + where
    select_one_qry = f"""
        SELECT {field} FROM {table}
        {where_clause}
        ORDER BY {field} {ascending_clause}
        LIMIT 1
        """
    return execute_read_query(connection, select_one_qry, return_dict=return_dict)[0][0]


def exist_where(connection, table, where_condition):
    exist_qry = f"""
        SELECT *
        FROM {table}
        WHERE {where_condition}
        """
    return len(execute_read_query(connection, exist_qry))


def update_text_field(connection, table, where_condition, field_name, value):
    update_qry = f"""
        UPDATE {table}
        SET {field_name}='{value.replace("'", "''")}'
        WHERE {where_condition};
        """
    execute_query(connection, update_qry)


def update_non_text_field(connection, table, where_condition, field_name, value):
    update_qry = f"""
        UPDATE {table}
        SET {field_name}={value}
        WHERE {where_condition};
        """
    execute_query(connection, update_qry)
