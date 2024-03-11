import logging

from datetime import datetime

from agent.classifiers.tree_classifiers_df import tree_classifier
from agent.classifiers.data.wrappers.count_features_fd_dataset_dev import get_raw_dataframes
from agent.data.entities.config import DEV_LABEL

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)


if __name__ == "__main__":
    starting_time = datetime.now()

    logging.info("Evaluating random forest count features on %s dataset for fake-news detection task...", DEV_LABEL)
    for binary_classifier in [True, False]:
        tree_classifier(for_task_label="fd",
                        binary_classifier=binary_classifier,
                        seed_val=42,
                        classifier="rf",
                        n_estimators=4,
                        get_data_function=get_raw_dataframes,
                        item_filter_label=DEV_LABEL,
                        label="")

    ending_time = datetime.now()
    logging.info("Total time: %s.", ending_time - starting_time)
