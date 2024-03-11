import logging

from datetime import datetime

from agent.nlp.graph_mgmt import build_items_graph, build_sentences_graph, get_valid_groups
from agent.nlp.metamap_mgmt import read_sem_groups_to_list
from agent.data.entities.config import ROOT_LOGGER_ID


logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(ROOT_LOGGER_ID)
logger.setLevel(logging.INFO)

VALID_SUBJECT_SEM_GROUPS = ["CHEM"]
VALID_OBJECT_SEM_GROUPS = ["DISO"]
ALL_SEM_GROUPS = read_sem_groups_to_list()

if __name__ == "__main__":
    logging.info("Launching graphs builder...")

    starting_time = datetime.now()

    build_items_graph()
    # build_sentences_graph(valid_subject_sem_groups=VALID_SUBJECT_SEM_GROUPS,
    #                       valid_object_sem_groups=VALID_OBJECT_SEM_GROUPS,
    #                       break_down_by_cuis=False)

    build_sentences_graph(valid_s_sem_groups=None,
                          valid_o_sem_groups=None,
                          break_down_by_token=True,
                          node_type="text_cui")


    ending_time = datetime.now()
    logging.info("Total time: %s.", ending_time - starting_time)
