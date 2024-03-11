import logging
import warnings
import graphviz
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import export_graphviz
from sklearn.feature_selection import SelectFromModel
from treeinterpreter import treeinterpreter as ti

from agent.data.entities.config import DEV_LABEL, FD_CLASS_VALUES
from agent.data.entities.config import DATA_CACHE_PATH, EXPORT_CHECK_INPUT_DATA_RESULTS_FILES
from agent.classifiers.evaluation.evaluation_mgmt import get_evaluation_measures
from agent.classifiers.utils.random_mgmt import set_random_seed
from agent.classifiers.utils.class_mgmt import get_textual_rating


def tree_classifier(for_task_label="",
                    binary_classifier=True,
                    do_feature_selection=False,
                    max_features=None,
                    seed_val=0,
                    classifier="rf",
                    n_estimators=100,
                    get_data_function=None,
                    item_filter_label=DEV_LABEL,
                    label=DEV_LABEL):

    warnings.filterwarnings('ignore')

    train_df, test_df = get_data_function(binary_classifier=binary_classifier)

    if binary_classifier:
        x_train = train_df.loc[:, train_df.columns != 'item_class']
        y_train = train_df.loc[:, 'item_class'].values
        x_test = test_df.loc[:, test_df.columns != 'item_class']
        y_test = test_df.loc[:, 'item_class'].values
    else:
        x_train = train_df.loc[:, train_df.columns != 'item_class_4']
        y_train = train_df.loc[:, 'item_class_4'].values
        x_test = test_df.loc[:, test_df.columns != 'item_class_4']
        y_test = test_df.loc[:, 'item_class_4'].values

    logging.info("")
    logging.info("Training model %s...", classifier)

    set_random_seed(seed_val, False)

    if classifier == "rf":
        clf = RandomForestClassifier(n_estimators=n_estimators, )
    if classifier == "gb":
        clf = GradientBoostingClassifier(loss="deviance", n_estimators=n_estimators, verbose=False)
    clf.fit(x_train, y_train)

    logging.info("")
    logging.info("Initial feature names/importances: %s",
                 ", ".join(str(n) + " " + "{:.3f}".format(i) for [n, i] in zip(clf.feature_names_in_, clf.feature_importances_)))

    y_pred = clf.predict(x_test)

    get_evaluation_measures(f"{for_task_label}_{classifier}_bin_{binary_classifier}_{item_filter_label}_seed_{seed_val}_{label}",
                            y_test, y_pred, save_evaluation=True, show_classif_report=False)

    feature_names = x_train.columns

    if do_feature_selection:

        # The features are considered unimportant and removed if the corresponding importance
        # of the feature values are below the provided threshold parameter.

        model = SelectFromModel(clf, prefit=True, max_features=max_features)
        x_train = model.transform(x_train)
        x_test = model.transform(x_test)

        selected_feature_idx = model.get_support()
        feature_names = feature_names[selected_feature_idx]

        logging.info("")
        logging.info("Selected feature names: %s", ", ".join(str(n) for n in feature_names))
        logging.info("Training model selected features %s...", classifier)

        set_random_seed(seed_val, False)

        if classifier == "rf":
            clf = RandomForestClassifier(n_estimators=n_estimators, )
        if classifier == "gb":
            clf = GradientBoostingClassifier(loss="deviance", n_estimators=n_estimators, verbose=False)
        clf.fit(x_train, y_train)

        y_pred = clf.predict(x_test)

        get_evaluation_measures(f"{for_task_label}_{classifier}_bin_{binary_classifier}_{item_filter_label}_seed_{seed_val}_{label}",
                                y_test, y_pred, save_evaluation=True, show_classif_report=False)

    if EXPORT_CHECK_INPUT_DATA_RESULTS_FILES:

        logging.info("Generating tree graph ...")

        if binary_classifier:
            textual_class_values = ["F", "T"]
        else:
            class_values = []
            for class_value in y_pred:
                if not class_value in class_values:
                    class_values.append(class_value)
            class_values.sort()

            textual_class_values = [get_textual_rating(FD_CLASS_VALUES, class_value) for class_value in class_values]

        dot_data = export_graphviz(clf.estimators_[1],
                                feature_names=feature_names,
                                class_names=textual_class_values,
                                filled=True,
                                rounded=True)
        if binary_classifier:
            binary_label = "bin"
        else:
            binary_label = "mc"

        graph = graphviz.Source(dot_data, format="png")
        graph.render(DATA_CACHE_PATH + binary_label + "_" + classifier)

        logging.info("Generating features file with predictions ...")

        test_df["y_test"] = pd.Series(list(y_test)).values
        test_df["y_pred"] = pd.Series(list(y_pred)).values

        file_name = DATA_CACHE_PATH + binary_label + "_" + classifier + "_predictions.tsv"
        with open(file_name, "w", encoding="utf-8") as results_file:
            for column in test_df.columns:
                results_file.write(str(column) + "\t")
            results_file.write("\n")

            for _, row in test_df.iterrows():
                for cell in row:
                    results_file.write(str(cell) + "\t")
                results_file.write("\n")


        # Random Forest explicability
        # https://blog.datadive.net/random-forest-interpretation-with-scikit-learn/

        logging.info("Calculating feature contributions:")

        file_name = DATA_CACHE_PATH + binary_label + "_" + classifier + "_feature_contribution.tsv"
        with open(file_name, "w", encoding="utf-8") as results_file:
            # results_file.write("Instance\tPrediction\tBias (trainset prior)\tFeature\tContribution\n")
            results_file.write("Instance\ty_test\ty_pred\tPrediction\tBias (trainset prior)")
            for feature in feature_names:
                results_file.write(f"\t{feature}")
            results_file.write("\n")

            prediction, bias, contributions = ti.predict(clf, x_test)
            for i in range(len(x_test)):
                results_file.write(f"{i}\t{y_test[i]}\t{y_pred[i]}\t{prediction[i][1]}\t{bias[i][1]}")
                for contribution, feature in zip(contributions[i], feature_names):
                    # logging.info("Instance: %s Prediction: %-11s Bias (trainset prior): %s Feature: %-10s Contribution: %s",
                    #              i, np.around(prediction[i], 3), np.around(bias[i], 3), feature, np.around(contribution, 3))
                    # results_file.write(f"{i}\t{prediction[i]}\t{bias[i]}\t{feature}\t{contribution}\n")
                    logging.info("Instance: %s Prediction: %-11s Bias (trainset prior): %s Feature: %-10s Contribution: %s",
                                 i, np.around(prediction[i][1], 3), np.around(bias[i][1], 3), feature, np.around(contribution[1], 3))
                    results_file.write(f"\t{contribution[1]}")
                results_file.write("\n")
