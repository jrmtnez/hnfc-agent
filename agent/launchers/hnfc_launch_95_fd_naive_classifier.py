import logging

from datetime import datetime

from agent.data.entities.config import TRAIN_ITEM_FILTER, DEV_ITEM_FILTER, TEST_ITEM_FILTER
from agent.data.entities.config import CTT3FR_ITEM_FILTER, CTT3_TEST_ITEM_FILTER, CORPUS_VERIFICATION_FILTER
from agent.data.entities.config import TRAIN_LABEL, DEV_LABEL, TEST_LABEL, CTT3_LABEL, CTT3FR_LABEL, CORPUS_VERIF_LABEL
from agent.classifiers.fd_naive_item_classifier import fd_naive_classifier

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)


if __name__ == "__main__":
    starting_time = datetime.now()

    # TODO actualizar parámetros
    fd_naive_classifier(item_filter=CORPUS_VERIFICATION_FILTER, item_filter_label=CORPUS_VERIF_LABEL, corpus_verification=True, evaluate_items=True)
    fd_naive_classifier(item_filter=TRAIN_ITEM_FILTER, item_filter_label=TRAIN_LABEL, corpus_verification=False, evaluate_items=True)
    fd_naive_classifier(item_filter=DEV_ITEM_FILTER, item_filter_label=DEV_LABEL, corpus_verification=False, evaluate_items=True)
    fd_naive_classifier(item_filter=TEST_ITEM_FILTER, item_filter_label=TEST_LABEL, corpus_verification=False, evaluate_items=True)
    fd_naive_classifier(item_filter=CTT3FR_ITEM_FILTER, item_filter_label=CTT3FR_LABEL, corpus_verification=False, evaluate_items=True)
    fd_naive_classifier(item_filter=CTT3_TEST_ITEM_FILTER, item_filter_label=CTT3_LABEL, corpus_verification=False, evaluate_items=True)

    ending_time = datetime.now()
    logging.info("Total time: %s.", ending_time - starting_time)
