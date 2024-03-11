import logging

from datetime import datetime

from agent.data.entities.config import DEV_LABEL, TEST_LABEL, EXTERNAL_LABEL, CTT3_LABEL, CTT3FR_LABEL, CTT3HEALTH_LABEL
from agent.classifiers.ffnn_classifier import ffnn
from agent.classifiers.data.wrappers.count_features_fd_dataset_dev import get_datasets as get_datasets_dev
from agent.classifiers.data.wrappers.count_features_fd_dataset_test import get_datasets as get_datasets_test
from agent.classifiers.data.wrappers.count_features_fd_dataset_external import get_datasets as get_datasets_external
from agent.classifiers.data.wrappers.count_features_fd_dataset_ct21t3fr import get_datasets as get_datasets_ct21t3fr
from agent.classifiers.data.wrappers.count_features_fd_dataset_ct21t3 import get_datasets as get_datasets_ct21t3
from agent.classifiers.data.wrappers.count_features_fd_dataset_ct21t3health import get_datasets as get_datasets_ct21t3health

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

LABEL = "test_220116"

if __name__ == "__main__":
    starting_time = datetime.now()

    binary_classifier = True
    use_class_weights = False

    # F1: 0.765
    logging.info("Evaluating ffnn count features on %s dataset for fake-news detection task...", DEV_LABEL)
    train_ds, dev_ds, test_ds, input_size, output_size = get_datasets_dev(review_level=9, binary_classifier=binary_classifier)
    ffnn(for_task_label="fd",
         binary_classifier=binary_classifier,
         cuis=False,
         hidden_layer_size=100,
         activation_fn="relu",
         use_class_weight=use_class_weights,
         use_features=[train_ds, dev_ds, test_ds, input_size, output_size],
         epochs=10,
         batch_size=16,
         seed_val=0,
         use_gpu=False,
         use_saved_model=True,
         item_filter_label=DEV_LABEL,
         label=LABEL)

    logging.info("-" * 80)
    logging.info("Evaluating ffnn count features on %s dataset for fake-news detection task...", TEST_LABEL)
    train_ds, dev_ds, test_ds, input_size, output_size = get_datasets_test(review_level=9, binary_classifier=binary_classifier)
    ffnn(for_task_label="fd",
         binary_classifier=binary_classifier,
         cuis=False,
         hidden_layer_size=100,
         activation_fn="sigmoid",
         use_class_weight=use_class_weights,
         use_features=[train_ds, dev_ds, test_ds, input_size, output_size],
         epochs=100,
         batch_size=16,
         seed_val=0,
         use_gpu=False,
         use_saved_model=True,
         item_filter_label=TEST_LABEL,
         label="test_211121")

    logging.info("-" * 80)
    logging.info("Evaluating ffnn count features on %s dataset for fake-news detection task...", EXTERNAL_LABEL)
    train_ds, dev_ds, test_ds, input_size, output_size = get_datasets_external(review_level=9, binary_classifier=binary_classifier)
    ffnn(for_task_label="fd",
         binary_classifier=binary_classifier,
         cuis=False,
         hidden_layer_size=100,
         activation_fn="sigmoid",
         use_class_weight=use_class_weights,
         use_features=[train_ds, dev_ds, test_ds, input_size, output_size],
         epochs=100,
         batch_size=16,
         seed_val=0,
         use_gpu=False,
         use_saved_model=True,
         item_filter_label=EXTERNAL_LABEL,
         label="test_211121")

    logging.info("-" * 80)
    logging.info("Evaluating ffnn count features on %s dataset for fake-news detection task...", CTT3FR_LABEL)
    train_ds, dev_ds, test_ds, input_size, output_size = get_datasets_ct21t3fr(review_level=9, binary_classifier=binary_classifier)
    ffnn(for_task_label="fd",
         binary_classifier=binary_classifier,
         cuis=False,
         hidden_layer_size=100,
         activation_fn="sigmoid",
         use_class_weight=use_class_weights,
         use_features=[train_ds, dev_ds, test_ds, input_size, output_size],
         epochs=100,
         batch_size=16,
         seed_val=0,
         use_gpu=False,
         use_saved_model=True,
         item_filter_label=CTT3FR_LABEL,
         label="test_211121")

    logging.info("-" * 80)
    logging.info("Evaluating ffnn count features on %s dataset for fake-news detection task...", CTT3_LABEL)
    train_ds, dev_ds, test_ds, input_size, output_size = get_datasets_ct21t3(review_level=9, binary_classifier=binary_classifier)
    ffnn(for_task_label="fd",
         binary_classifier=binary_classifier,
         cuis=False,
         hidden_layer_size=100,
         activation_fn="sigmoid",
         use_class_weight=use_class_weights,
         use_features=[train_ds, dev_ds, test_ds, input_size, output_size],
         epochs=100,
         batch_size=16,
         seed_val=0,
         use_gpu=False,
         use_saved_model=True,
         item_filter_label=CTT3_LABEL,
         label="test_211121")

    logging.info("-" * 80)
    logging.info("Evaluating ffnn count features on %s dataset for fake-news detection task...", CTT3HEALTH_LABEL)
    train_ds, dev_ds, test_ds, input_size, output_size = get_datasets_ct21t3health(review_level=9, binary_classifier=binary_classifier)
    ffnn(for_task_label="fd",
         binary_classifier=binary_classifier,
         cuis=False,
         hidden_layer_size=100,
         activation_fn="sigmoid",
         use_class_weight=use_class_weights,
         use_features=[train_ds, dev_ds, test_ds, input_size, output_size],
         epochs=100,
         batch_size=16,
         seed_val=0,
         use_gpu=False,
         use_saved_model=True,
         item_filter_label=CTT3HEALTH_LABEL,
         label="test_211121")

    ending_time = datetime.now()
    logging.info("Total time: %s.", ending_time - starting_time)
