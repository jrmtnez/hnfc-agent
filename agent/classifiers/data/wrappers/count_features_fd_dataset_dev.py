from agent.data.entities.config import TRAIN_ITEM_FILTER, DEV_ITEM_FILTER, TRAIN_LABEL, DEV_LABEL
from agent.classifiers.data.count_features_fd_dataset_mgmt import get_datasets_from_filter, get_raw_datasets_from_filter
from agent.classifiers.data.count_features_fd_dataset_mgmt import get_raw_dataframes_from_filter

def get_datasets(review_level=9, binary_classifier=True, val_split=0.8):
    return get_datasets_from_filter(TRAIN_ITEM_FILTER, DEV_ITEM_FILTER, TRAIN_LABEL, DEV_LABEL, review_level,
                                    binary_classifier=binary_classifier, val_split=val_split)

def get_raw_datasets(review_level=9, binary_classifier=True, dev_dataset=None):
    # dev_dataset: for compatibility
    return get_raw_datasets_from_filter(TRAIN_ITEM_FILTER, DEV_ITEM_FILTER, TRAIN_LABEL, DEV_LABEL, review_level,
                                        binary_classifier=binary_classifier)

def get_raw_dataframes(review_level=9, binary_classifier=True):
    return get_raw_dataframes_from_filter(TRAIN_ITEM_FILTER, DEV_ITEM_FILTER, TRAIN_LABEL, DEV_LABEL, review_level,
                                          binary_classifier=binary_classifier)
