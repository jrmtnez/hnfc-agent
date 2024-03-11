import logging

from datetime import datetime

from agent.pipeline.review_level_mgmt import update_sentence_review_level, update_item_review_level

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)


if __name__ == "__main__":
    logging.info("Launching review level updater 5...")

    starting_time = datetime.now()

    update_sentence_review_level(3, to_level=4, to_level_without_validations=5)
    update_item_review_level()

    ending_time = datetime.now()
    logging.info("Total time: %s.", ending_time - starting_time)
