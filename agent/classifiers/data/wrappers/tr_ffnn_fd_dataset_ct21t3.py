from agent.data.entities.config import CTT3_TEST_ITEM_FILTER, CTT3_LABEL
from agent.classifiers.data.tr_ffnn_fd_dataset_mgmt import get_datasets_from_filter

def get_datasets(pretrained_model, max_lenght, binary_classifier=False,
                 _expand_tokenizer=False, _use_saved_model=True, _tokenizer_type="text", _cuis=None):

    return get_datasets_from_filter(pretrained_model, CTT3_TEST_ITEM_FILTER, CTT3_LABEL, max_lenght,
                                    binary_classifier=binary_classifier)
