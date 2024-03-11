from agent.data.entities.config import CTT3HEALTH_ITEM_FILTER, CTT3HEALTH_LABEL
from agent.classifiers.data.transformer_text_fd_dataset_mgmt import get_datasets_from_filter

def get_datasets(pretrained_model, max_lenght, val_split=0.8, binary_classifier=False,
                 _expand_tokenizer=False, _use_saved_model=True, _tokenizer_type="text"):

    return get_datasets_from_filter(pretrained_model, CTT3HEALTH_ITEM_FILTER, CTT3HEALTH_LABEL, max_lenght,
                                    val_split=val_split, binary_classifier=binary_classifier)
