import logging
from datetime import datetime

from agent.data.entities.config import ROOT_LOGGER_ID
from agent.nlp.spo_extractor import extract_spos_from_sentences, match_tokens_cuis_in_sentences
from agent.nlp.spo_extractor import extract_spos_from_sentences_given_predicate, clear_spos
from agent.nlp.spo_extractor import extract_spos_from_sentences_aux_predicate
from agent.nlp.spo_extractor import extract_spos_from_sentences_given_manual_predicate
from agent.nlp.spo_extractor import mark_no_predicate_external_sentences
from agent.classifiers.fc_preliminary_classifier import do_preliminary_fc_classification

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(ROOT_LOGGER_ID)
logger.setLevel(logging.INFO)


DATA_AUGMENTATION = False

if __name__ == "__main__":

    logging.info("Launching SPO extractor...")

    starting_time = datetime.now()

    clear_spos(5)

    for external in [False, True]:
        if DATA_AUGMENTATION:
            extract_spos_from_sentences_given_manual_predicate(5, external=external)
        else:
            extract_spos_from_sentences_given_predicate(5, external=external)
            extract_spos_from_sentences(5, external=external)
            extract_spos_from_sentences_aux_predicate(5, external=external)
            extract_spos_from_sentences_given_manual_predicate(5, external=external)
        match_tokens_cuis_in_sentences(5, external=external)
        do_preliminary_fc_classification(external=external)
        extract_spos_from_sentences_given_manual_predicate(6, external=external)
        match_tokens_cuis_in_sentences(6, external=external)
        if external:
            mark_no_predicate_external_sentences(5)

    ending_time = datetime.now()
    logging.info("Total time: %s.", ending_time - starting_time)
# https://192.168.251.28:8000/annotate/cwsentences/238471/update/?next=/annotate/sentences6/