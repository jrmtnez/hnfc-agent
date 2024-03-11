import logging

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from agent.data.entities.config import DEV_LABEL
from agent.classifiers.evaluation.evaluation_mgmt import get_evaluation_measures
from agent.classifiers.utils.random_mgmt import set_random_seed


def tree_classifier(for_task_label="",
                    binary_classifier=True,
                    seed_val=0,
                    test_seeds=False,
                    classifier="rf",
                    n_estimators=1000,
                    get_data_function=None,
                    dev_dataset=True,
                    item_filter_label=DEV_LABEL,
                    label=DEV_LABEL):

    x_train, y_train, x_test, y_test = get_data_function(binary_classifier=binary_classifier, dev_dataset=dev_dataset)

    logging.info("Training model %s...", classifier)

    if test_seeds:
        for seed_val in range(0, 1000):

            set_random_seed(seed_val, False)

            if classifier == "rf":
                clf = RandomForestClassifier(n_estimators=n_estimators)
            if classifier == "gb":
                clf = GradientBoostingClassifier(loss="deviance", n_estimators=n_estimators, verbose=False)
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)

            get_evaluation_measures(f"{for_task_label}_{classifier}_bin_{binary_classifier}_est_{n_estimators}_{item_filter_label}_seed_{seed_val}_{label}",
                                    y_test, y_pred, save_evaluation=True, show_classif_report=False)
    else:
        set_random_seed(seed_val, False)

        if classifier == "rf":
            clf = RandomForestClassifier(n_estimators=n_estimators)
        if classifier == "gb":
            clf = GradientBoostingClassifier(loss="log_loss", n_estimators=n_estimators, verbose=False)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)

        get_evaluation_measures(f"{for_task_label}_{classifier}_bin_{binary_classifier}_est_{n_estimators}_{item_filter_label}_seed_{seed_val}_{label}",
                                y_test, y_pred, save_evaluation=True, show_classif_report=False)
