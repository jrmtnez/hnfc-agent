import logging

from datetime import datetime

from agent.pipeline.review_level_mgmt import update_sentence_review_level_spo_to_check
from agent.pipeline.review_level_mgmt import update_sentence_review_level_spo_skip_validations
from agent.pipeline.review_level_mgmt import update_sentence_review_level_spo_completed
from agent.pipeline.review_level_mgmt import update_item_review_level

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)


if __name__ == "__main__":

    logging.info("Launching review level updater 7...")

    starting_time = datetime.now()

    update_sentence_review_level_spo_to_check(5, 6)
    update_sentence_review_level_spo_skip_validations(5, 7)
    update_sentence_review_level_spo_completed(6, 7)
    update_item_review_level()

    ending_time = datetime.now()

    logging.info("Total time: %s.", ending_time - starting_time)
