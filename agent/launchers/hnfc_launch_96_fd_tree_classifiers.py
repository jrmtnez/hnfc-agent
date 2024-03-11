import logging

from datetime import datetime

from agent.classifiers.tree_classifiers import tree_classifier
from agent.classifiers.data.wrappers.count_features_fd_dataset_dev import get_raw_datasets as get_datasets_dev
from agent.classifiers.data.wrappers.count_features_fd_dataset_test import get_raw_datasets as get_datasets_test
from agent.classifiers.data.wrappers.count_features_fd_dataset_external import get_raw_datasets as get_datasets_external
from agent.classifiers.data.wrappers.count_features_fd_dataset_ct21t3fr import get_raw_datasets as get_datasets_ct21t3fr
from agent.classifiers.data.wrappers.count_features_fd_dataset_ct21t3 import get_raw_datasets as get_datasets_ct21t3
from agent.classifiers.data.wrappers.count_features_fd_dataset_ct21t3health import get_raw_datasets as get_datasets_ct21t3health
from agent.data.entities.config import DEV_LABEL, TEST_LABEL, EXTERNAL_LABEL, CTT3_LABEL, CTT3FR_LABEL, CTT3HEALTH_LABEL

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)


if __name__ == "__main__":
    starting_time = datetime.now()

    binary_classifier = True

    logging.info("Evaluating random forest count features on %s dataset for fake-news detection task...", DEV_LABEL)
    tree_classifier(for_task_label="fd",
                    binary_classifier=binary_classifier,
                    seed_val=0,
                    test_seeds=False,
                    classifier="rf",
                    n_estimators=100,
                    get_data_function=get_datasets_dev,
                    item_filter_label=DEV_LABEL,
                    label="")

    logging.info("-" * 80)
    logging.info("Evaluating gradient boosting count features on %s dataset for fake-news detection task...", DEV_LABEL)
    tree_classifier(for_task_label="fd",
                    binary_classifier=binary_classifier,
                    seed_val=0,
                    test_seeds=False,
                    classifier="gb",
                    n_estimators=100,
                    get_data_function=get_datasets_dev,
                    item_filter_label=DEV_LABEL,
                    label="")

    logging.info("-" * 80)
    logging.info("Evaluating random forest count features on %s dataset for fake-news detection task...", TEST_LABEL)
    tree_classifier(for_task_label="fd",
                    binary_classifier=binary_classifier,
                    seed_val=0,
                    test_seeds=False,
                    classifier="rf",
                    n_estimators=100,
                    get_data_function=get_datasets_test,
                    item_filter_label=TEST_LABEL,
                    label="")

    logging.info("-" * 80)
    logging.info("Evaluating gradient boosting count features on %s dataset for fake-news detection task...", TEST_LABEL)
    tree_classifier(for_task_label="fd",
                    binary_classifier=binary_classifier,
                    seed_val=0,
                    test_seeds=False,
                    classifier="gb",
                    n_estimators=100,
                    get_data_function=get_datasets_test,
                    item_filter_label=TEST_LABEL,
                    label="")

    logging.info("-" * 80)
    logging.info("Evaluating random forest count features on %s dataset for fake-news detection task...", EXTERNAL_LABEL)
    tree_classifier(for_task_label="fd",
                    binary_classifier=binary_classifier,
                    seed_val=0,
                    test_seeds=False,
                    classifier="rf",
                    n_estimators=100,
                    get_data_function=get_datasets_external,
                    item_filter_label=EXTERNAL_LABEL,
                    label="")

    logging.info("-" * 80)
    logging.info("Evaluating gradient boosting count features on %s dataset for fake-news detection task...", EXTERNAL_LABEL)
    tree_classifier(for_task_label="fd",
                    binary_classifier=binary_classifier,
                    seed_val=0,
                    test_seeds=False,
                    classifier="gb",
                    n_estimators=100,
                    get_data_function=get_datasets_external,
                    item_filter_label=EXTERNAL_LABEL,
                    label="")

    logging.info("-" * 80)
    logging.info("Evaluating random forest count features on %s dataset for fake-news detection task...", CTT3FR_LABEL)
    tree_classifier(for_task_label="fd",
                    binary_classifier=binary_classifier,
                    seed_val=0,
                    test_seeds=False,
                    classifier="rf",
                    n_estimators=100,
                    get_data_function=get_datasets_ct21t3fr,
                    item_filter_label=CTT3FR_LABEL,
                    label="")

    logging.info("-" * 80)
    logging.info("Evaluating gradient boosting count features on %s dataset for fake-news detection task...", CTT3FR_LABEL)
    tree_classifier(for_task_label="fd",
                    binary_classifier=binary_classifier,
                    seed_val=0,
                    test_seeds=False,
                    classifier="gb",
                    n_estimators=100,
                    get_data_function=get_datasets_ct21t3fr,
                    item_filter_label=CTT3FR_LABEL,
                    label="")

    logging.info("-" * 80)
    logging.info("Evaluating random forest count features on %s dataset for fake-news detection task...", CTT3_LABEL)
    tree_classifier(for_task_label="fd",
                    binary_classifier=binary_classifier,
                    seed_val=0,
                    test_seeds=False,
                    classifier="rf",
                    n_estimators=100,
                    get_data_function=get_datasets_ct21t3,
                    item_filter_label=CTT3_LABEL,
                    label="")

    logging.info("-" * 80)
    logging.info("Evaluating gradient boosting count features on %s dataset for fake-news detection task...", CTT3_LABEL)
    tree_classifier(for_task_label="fd",
                    binary_classifier=binary_classifier,
                    seed_val=0,
                    test_seeds=False,
                    classifier="gb",
                    n_estimators=100,
                    get_data_function=get_datasets_ct21t3,
                    item_filter_label=CTT3_LABEL,
                    label="")

    logging.info("-" * 80)
    logging.info("Evaluating gradient boosting count features on %s dataset for fake-news detection task...", CTT3HEALTH_LABEL)
    tree_classifier(for_task_label="fd",
                    binary_classifier=binary_classifier,
                    seed_val=0,
                    test_seeds=False,
                    classifier="gb",
                    n_estimators=100,
                    get_data_function=get_datasets_ct21t3health,
                    item_filter_label=CTT3HEALTH_LABEL,
                    label="")

    logging.info("-" * 80)
    logging.info("Evaluating gradient boosting count features on %s dataset for fake-news detection task...", CTT3HEALTH_LABEL)
    tree_classifier(for_task_label="fd",
                    binary_classifier=binary_classifier,
                    seed_val=0,
                    test_seeds=False,
                    classifier="gb",
                    n_estimators=100,
                    get_data_function=get_datasets_ct21t3health,
                    item_filter_label=CTT3HEALTH_LABEL,
                    label="")

    ending_time = datetime.now()
    logging.info("Total time: %s.", ending_time - starting_time)
