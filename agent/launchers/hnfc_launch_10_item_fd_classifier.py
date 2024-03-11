import logging

from datetime import datetime

from agent.data.entities.config import DEV_ITEM_FILTER, TEST_ITEM_FILTER, DEV_LABEL, TEST_LABEL
from agent.data.entities.config import SKIP_VAL_ITEM_FILTER, SKIP_VAL_LABEL
from agent.classifiers.fd_naive_item_classifier import fd_naive_classifier, evaluate_items_for_filter
from agent.pipeline.review_level_mgmt import update_sentence_review_level, update_item_review_level

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)


if __name__ == "__main__":

    logging.info("Launching fake news classifiers...")

    starting_time = datetime.now()

    logging.info("Classifying fake news classifiers on level 8...")
    fd_naive_classifier(review_level=8, skip_validations=True, instance_type="Other", item_filter_label=SKIP_VAL_LABEL)
    fd_naive_classifier(review_level=8, skip_validations=False, instance_type="Dev", item_filter_label=DEV_LABEL)

    logging.info("Evaluating fake news classifiers on level 8...")
    evaluate_items_for_filter(review_level=8, skip_validations=True, instance_type="Other", item_filter_label=SKIP_VAL_LABEL)
    evaluate_items_for_filter(review_level=8, skip_validations=False, instance_type="Dev", item_filter_label=TEST_LABEL)

    update_sentence_review_level(8, to_level=9, to_level_without_validations=9)
    update_item_review_level()

    logging.info("Classifying fake news classifiers on level 9...")
    fd_naive_classifier(review_level=9, skip_validations=True, instance_type="Other", item_filter_label=SKIP_VAL_LABEL)
    fd_naive_classifier(review_level=9, skip_validations=False, instance_type="Dev", item_filter_label=DEV_LABEL)

    logging.info("Evaluating fake news classifiers on level 9...")
    evaluate_items_for_filter(review_level=9, skip_validations=True, instance_type="Other", item_filter_label=SKIP_VAL_LABEL)
    evaluate_items_for_filter(review_level=9, skip_validations=False, instance_type="Dev", item_filter_label=TEST_LABEL)

    ending_time = datetime.now()
    logging.info("Total time: %s.", ending_time - starting_time)
