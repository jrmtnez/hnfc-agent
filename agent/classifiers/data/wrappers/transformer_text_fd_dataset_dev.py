from agent.data.entities.config import DEV_ITEM_FILTER, DEV_LABEL
from agent.classifiers.data.transformer_text_fd_dataset_mgmt import get_datasets_from_filter


def get_datasets(pretrained_model, max_lenght, val_split=0.8, binary_classifier=False,
                 _expand_tokenizer=False, _use_saved_model=True, _tokenizer_type="text", text_cuis=None):

    return get_datasets_from_filter(pretrained_model, DEV_ITEM_FILTER, DEV_LABEL, max_lenght,
                                    val_split=val_split, binary_classifier=binary_classifier)
