import logging

from datetime import datetime

from agent.data.entities.config import ROOT_LOGGER_ID
from agent.classifiers.data.liwc_management import export_sentence_data_for_liwc_features, get_sentence_liwc_features
from agent.classifiers.tree_classifiers import tree_classifier

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(ROOT_LOGGER_ID)
logger.setLevel(logging.INFO)


def run_tree_baseline(classifier="rf", binary_classifier=False):

    for n_estimators in [4, 8, 10, 12, 15, 20, 100, 200, 500, 1000]:
        for seed_val in [0, 12, 33, 42, 54, 63, 70, 79, 85, 96]:
            tree_classifier(for_task_label="fd",
                            binary_classifier=binary_classifier,
                            seed_val=seed_val,
                            test_seeds=False,
                            classifier=classifier,
                            n_estimators=n_estimators,
                            get_data_function=get_sentence_liwc_features,
                            dev_dataset=False,
                            item_filter_label="dev+other",
                            label="liwc_baseline")


if __name__ == "__main__":
    logging.info("Launching baselines...")

    starting_time = datetime.now()

    # export_sentence_data_for_liwc_features()

    # ¡IMPORTANTE!
    # es necesario revisar los archivos generados por LIWC y verificar que en cada línea
    # no aparecen comillas " o aparece un número par de ellas

    # get_sentence_liwc_features(binary_classifier=True)

    run_tree_baseline(classifier="rf", binary_classifier=True)
    run_tree_baseline(classifier="gb", binary_classifier=True)
    run_tree_baseline(classifier="rf", binary_classifier=False)
    run_tree_baseline(classifier="gb", binary_classifier=False)

    ending_time = datetime.now()
    logging.info("Total time: %s.", ending_time - starting_time)
