import logging

from agent.data.sql.sql_mgmt import get_connection, select_fields_where
from agent.data.sql.sql_mgmt import update_text_field, update_non_text_field
from agent.classifiers.evaluation.evaluation_mgmt import get_evaluation_measures
from agent.data.entities.config import ITEMS_TABLE, SENTENCES_TABLE

ITEM_FIELDS = "id, item_class_4, item_class_4_auto, item_class_auto"
SENTENCE_FIELDS = "sentence_class, sentence_class_auto, sentence_class_score_auto"


def fd_naive_classifier(review_level=8,
                        skip_validations=False,
                        instance_type="",
                        item_filter_label="",
                        corpus_verification=False,
                        evaluate_items=False,
                        binary=False,
                        use_multiclass_predictions_for_binary_classification=False):

    logging.info("Loading item datasets...")

    connection = get_connection()

    # item_filter = f"skip_validations = {skip_validations} AND review_level = {review_level} AND needs_revision = false AND instance_type = '{instance_type}'"
    item_filter = f"skip_validations = {skip_validations} AND review_level = {review_level} AND instance_type = '{instance_type}'"

    items = select_fields_where(connection, ITEMS_TABLE, ITEM_FIELDS, item_filter)

    items_count = len(items)

    logging.info("%s dataset size: %s", item_filter_label, items_count)

    for item in items:
        item_id = item[0]
        item_class_4 = item[1]
        item_class_4_auto = item[2]

        f_count = 0
        pf_count = 0
        t_count = 0

        bin_f_count = 0
        bin_t_count = 0

        sentence_count = 0
        acc_score_auto = 0

        sentences_filter = f"item_id = {item_id} AND review_level = {review_level}"
        sentences = select_fields_where(connection, SENTENCES_TABLE, SENTENCE_FIELDS, sentences_filter)
        for sentence in sentences:
            sentence_class_4 = sentence[0]
            sentence_class_4_auto = sentence[1]
            sentence_class_score_auto = sentence[2]

            if corpus_verification:
                predicted_sentence_class = sentence_class_4  # sentence_class_4 -> ground truth
                if sentence_class_4 == 'T':
                    predicted_sentence_class_score = 1
                else:
                    predicted_sentence_class_score = 0
            else:
                predicted_sentence_class = sentence_class_4_auto
                predicted_sentence_class_score = sentence_class_score_auto

            if predicted_sentence_class == 'F':
                f_count += 1
            if predicted_sentence_class == 'PF':
                pf_count += 1
            if predicted_sentence_class == 'T':
                t_count += 1

            if predicted_sentence_class_score <= 0.5:
                bin_f_count += 1
            else:
                bin_t_count += 1

            acc_score_auto += sentence_class_score_auto
            sentence_count += 1

        update_text_field(connection, ITEMS_TABLE, f"id = {item_id}", "item_class_4_auto",
                          evaluate_item_4(f_count, pf_count, t_count))

        if use_multiclass_predictions_for_binary_classification:
            update_non_text_field(connection, ITEMS_TABLE, f"id = {item_id}", "item_class_auto",
                                  evaluate_item_binary(f_count, pf_count, t_count))
        else:
            update_non_text_field(connection, ITEMS_TABLE, f"id = {item_id}", "item_class_auto",
                                  evaluate_item_binary2(bin_f_count, bin_t_count))
            # update_non_text_field(connection, ITEMS_TABLE, f"id = {item_id}", "item_class_auto",
            #                       evaluate_item_binary3(acc_score_auto, sentence_count))

        logging.debug("item id %s, item class 4: %s, item class 4 auto: %s", item_id, item_class_4, item_class_4_auto)

    connection.commit()
    connection.close()

    if evaluate_items and items_count > 0:
        evaluate_items_for_filter(review_level=review_level, skip_validations=skip_validations,
                                  instance_type=instance_type, item_filter_label=item_filter_label, binary=binary,
                                  use_multiclass_predictions_for_binary_classification=use_multiclass_predictions_for_binary_classification)


def evaluate_item_4(f_count, pf_count, t_count):
    if f_count > 0:
        return 'F'
    if pf_count > 0:
        return 'PF'
    if t_count > 0:
        return 'T'
    return 'NA'


def evaluate_item_binary(f_count, pf_count, t_count):
    if f_count > 0:
        return 0
    if pf_count > 0:
        return 0
    if t_count > 0:
        return 1
    return 0


def evaluate_item_binary2(bin_f_count, bin_t_count):

    if bin_f_count > 0:
        return 0
    if bin_t_count > 0:
        return 1
    return 0


def evaluate_item_binary3(acc_score_auto, sentence_count):
    if sentence_count > 0:
        if acc_score_auto / sentence_count > 0.5:
            return 1
        else:
            return 0
    return 0


def evaluate_items_for_filter(review_level=9,
                              skip_validations=False,
                              instance_type="",
                              item_filter_label="",
                              binary=False,
                              use_multiclass_predictions_for_binary_classification=False):

    connection = get_connection()

    # item_filter = f"skip_validations = {skip_validations} AND review_level = {review_level} AND needs_revision = false AND instance_type = '{instance_type}'"
    item_filter = f"skip_validations = {skip_validations} AND review_level = {review_level} AND instance_type = '{instance_type}'"

    items = select_fields_where(connection, ITEMS_TABLE, ITEM_FIELDS, item_filter)

    items_count = len(items)
    logging.info("Evaluating on %s dataset with size %s...", item_filter_label, items_count)
    if items_count == 0:
        logging.info("Nothing to do!")
        return

    y_test = []
    class_predictions = []
    y_test_4 = []
    class_predictions_4 = []
    for item in items:

        item_class_4 = item[1]
        item_class_4_auto = item[2]
        item_class_auto = item[3]

        y_test_4.append(item_class_4)
        class_predictions_4.append(item_class_4_auto)

        y_test.append(item_class_4 == 'T')
        if use_multiclass_predictions_for_binary_classification:
            class_predictions.append(item_class_4_auto == 'T')
        else:
            class_predictions.append(item_class_auto == 1)

        # print(item[0], item_class_4, item_class_4 == 'T')

    if binary:
        get_evaluation_measures(f"fd_naive_bin_True_{item_filter_label}", y_test, class_predictions,
                                "fd", binary, save_evaluation=True, show_classif_report=False)
    else:
        get_evaluation_measures(f"fd_naive_bin_False_{item_filter_label}", y_test_4, class_predictions_4,
                                "fd", binary, save_evaluation=True, show_classif_report=False,
                                multiclass_numeric_labels=False)
    connection.close()
