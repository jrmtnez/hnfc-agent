import logging
import requests

from agent.data.entities.config import ROOT_LOGGER_ID
from agent.data.sql.sql_mgmt import get_connection, execute_read_query

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger(ROOT_LOGGER_ID)
logger.setLevel(logging.DEBUG)


KG_DATABASE_NAME = "conceptnet5"
KG_DATABASE_HOST = "192.168.251.32"
SUBJECT_VALID_PREDICATES = ["IsA", "Synonym", "Causes", "HasProperty", "MadeOf", "PartOf"]
OBJECT_VALID_PREDICATES = ["IsA", "Synonym"]


def get_related_objects(subject, verbose=True):

    SQL_QUERY = f"""
        SELECT relations.uri as predicate, end_nodes.uri as object, edges.weight
        FROM edges, nodes as start_nodes, nodes as end_nodes, relations
        WHERE start_nodes.uri = '/c/en/{subject}' AND
              start_nodes.id = edges.start_id AND
              relations.id = edges.relation_id AND
              end_nodes.id = edges.end_id AND
              start_nodes.uri <> end_nodes.uri
        """

    related_objects = []

    connection2 = get_connection(db_name=KG_DATABASE_NAME, db_host=KG_DATABASE_HOST)
    result = execute_read_query(connection2, SQL_QUERY)
    for relation in result:
        if len(relation[1].split("/")) > 2:
            object = relation[1].split("/")[3]
            predicate = relation[0].split("/")[2]
            lang = relation[1].split("/")[2]
            weight = relation[2]

            if predicate in SUBJECT_VALID_PREDICATES and lang == "en":
                if verbose:
                    logger.debug("%s %s %s %s", subject, predicate, object, weight)

                relation_dict = {}
                relation_dict["object"] = object
                relation_dict["predicate"] = predicate
                relation_dict["weight"] = weight

                related_objects.append(relation_dict)
    connection2.close()

    return related_objects


def get_related_subjects(object, verbose=True):

    SQL_QUERY = f"""
        SELECT start_nodes.uri as subject, relations.uri as predicate, edges.weight
        FROM edges, nodes as start_nodes, nodes as end_nodes, relations
        WHERE end_nodes.uri = '/c/en/{object}' AND
              start_nodes.id = edges.start_id AND
              relations.id = edges.relation_id AND
              end_nodes.id = edges.end_id AND
              start_nodes.uri <> end_nodes.uri
        """

    related_objects = []

    connection2 = get_connection(db_name=KG_DATABASE_NAME, db_host=KG_DATABASE_HOST)
    result = execute_read_query(connection2, SQL_QUERY)
    for relation in result:
        if len(relation[0].split("/")) > 2:
            subject = relation[0].split("/")[3]
            predicate = relation[1].split("/")[2]
            lang = relation[0].split("/")[2]
            weight = relation[2]

            if predicate in OBJECT_VALID_PREDICATES and lang == "en":
                if verbose:
                    logger.debug("%s %s %s %s", subject, predicate, object, weight)

                relation_dict = {}
                relation_dict["subject"] = subject
                relation_dict["predicate"] = predicate
                relation_dict["weight"] = weight

                related_objects.append(relation_dict)

    connection2.close()

    return related_objects



def get_triples_from_conceptnet_api(query):

    cn_triples = []
    response_json = requests.get(f"https://api.conceptnet.io/c/en/{query}").json()

    for edge in response_json["edges"]:
        logger.debug("%s %s %s", edge["start"]["label"], edge["rel"]["label"], edge["end"]["label"])
        cn_triples.append([edge["start"]["label"], edge["rel"]["label"], edge["end"]["label"]])

    return cn_triples


def exist_relation_in_conceptnet_api(query):
    response_json = requests.get(f"https://api.conceptnet.io/c/en/{query}").json()
    edges = response_json["edges"]
    return len(edges) > 0


def get_kb_connection():
    return get_connection(db_name=KG_DATABASE_NAME, db_host=KG_DATABASE_HOST)

def exist_relation_in_conceptnet(connection2, subject_object):

    SQL_QUERY = f"""
        SELECT id
        FROM nodes
        WHERE (nodes.uri = '/c/en/{subject_object}' OR nodes.uri = '/c/en/{subject_object}')
        """
    result = execute_read_query(connection2, SQL_QUERY)

    return len(result) > 0
