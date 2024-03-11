import logging

from os.path import exists
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import classification_report

from agent.data.entities.config import RESULTS_FILE_TSV, RESULTS_FILE_TXT
from agent.data.entities.config import CW_CLASS_VALUES, FC_CLASS_VALUES, FD_CLASS_VALUES


def get_evaluation_measures(model_name, y_test, class_predictions, task, binary, save_evaluation=False, show_classif_report=False,
                            multiclass_numeric_labels=True):

    # define "positive" class depending on task: true in "cw" and false in "fc", "fd"
    if task == "cw":
        if binary:
            labels = [1]
        else:
            labels = [len(CW_CLASS_VALUES) - 1]
    if task == "fc":
        if binary:
            labels = [0]
        else:
            labels = [class_value for class_value in range(0, len(FC_CLASS_VALUES) - 1)]
    if task == "fd":
        if binary:
            labels = [0]
        else:
            if multiclass_numeric_labels:
                labels = [class_value for class_value in range(0, len(FD_CLASS_VALUES) - 1)]
            else:
                labels = ["NA", "F", "PF"]

    acc_score = accuracy_score(y_test, class_predictions)

    prec_score = precision_score(y_test, class_predictions, zero_division=0, average="macro")
    rec_score = recall_score(y_test, class_predictions, zero_division=0, average="macro")
    f1 = f1_score(y_test, class_predictions, zero_division=0, average="macro")

    prec_score_positive = precision_score(y_test, class_predictions, zero_division=0, average="macro", labels=labels)
    rec_score_positive = recall_score(y_test, class_predictions, zero_division=0, average="macro", labels=labels)
    f1_positive = f1_score(y_test, class_predictions, zero_division=0, average="macro", labels=labels)

    results_log = f"{model_name}, accuracy {acc_score:.3f}, precision: {prec_score:.3f}, recall: {rec_score:.3f}, f1: {f1:.3f}, precision+: {prec_score_positive:.3f}, recall+: {rec_score_positive:.3f}, f1+: {f1_positive:.3f}, labels+: {labels}"
    logging.info(results_log)

    if show_classif_report:
        logging.info("\n%s", classification_report(y_test, class_predictions, zero_division=0, digits=4))

    if save_evaluation:
        results_file_tsv_header = "accuracy\tprecision\trecall\tf1\tprecision+\trecall+\tf1+\tmodel"
        results_file_tsv_body = f"{acc_score:.3f}\t{prec_score:.3f}\t{rec_score:.3f}\t{f1:.3f}\t{prec_score_positive:.3f}\t{rec_score_positive:.3f}\t{f1_positive:.3f}\t{model_name}"

        if not exists(RESULTS_FILE_TSV):
            with open(RESULTS_FILE_TSV, "w", encoding="utf-8") as results_file:
                results_file.write(results_file_tsv_header + "\n")
        with open(RESULTS_FILE_TSV, "a", encoding="utf-8") as results_file:
            results_file.write(results_file_tsv_body + "\n")

        results_file_txt_header = "accuracy\tprecision\trecall\tf1\t\tprecision+\trecall+\tf1+\t\tmodel"
        results_file_txt_body = f"{acc_score:.3f}\t\t{prec_score:.3f}\t\t{rec_score:.3f}\t{f1:.3f}\t{prec_score_positive:.3f}\t\t{rec_score_positive:.3f}\t{f1_positive:.3f}\t{model_name}"

        if not exists(RESULTS_FILE_TXT):
            with open(RESULTS_FILE_TXT, "w", encoding="utf-8") as results_file:
                results_file.write(results_file_txt_header + "\n")
        with open(RESULTS_FILE_TXT, "a", encoding="utf-8") as results_file:
            results_file.write(results_file_txt_body + "\n")


def get_f1_measure(y_test, class_predictions):
    return f1_score(y_test, class_predictions, zero_division=0, average="macro")
