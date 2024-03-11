import logging

from datetime import datetime

from agent.data.entities.config import ROOT_LOGGER_ID
from agent.data.augmentation.data_augmentation_mgmt import process_train_items, TRANSFORMER_TYPE
# from agent.data.augmentation.data_augmentation_mgmt import process_single_item, do_back_translation, get_back_translation_models


logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(ROOT_LOGGER_ID)
logger.setLevel(logging.INFO)


if __name__ == "__main__":

    logging.info("Launching data augmentation management...")

    starting_time = datetime.now()

    if TRANSFORMER_TYPE == 0:
        forward_model_name = "facebook/wmt19-en-de"
        backward_model_name = "facebook/wmt19-de-en"
        bt_label = "FSMT_DE"

    if TRANSFORMER_TYPE == 1:
        # forward_model_name = "Helsinki-NLP/opus-mt-tc-big-en-fr"
        # backward_model_name = "Helsinki-NLP/opus-mt-tc-big-fr-en"
        # bt_label = "Marian_FR"

        forward_model_name = "Helsinki-NLP/opus-mt-en-it"
        backward_model_name = "Helsinki-NLP/opus-mt-it-en"
        bt_label = "Marian_IT"

        # forward_model_name = "Helsinki-NLP/opus-mt-en-de"
        # backward_model_name = "Helsinki-NLP/opus-mt-de-en"
        # bt_label = "Marian_DE"

        # forward_model_name = "Helsinki-NLP/opus-mt-en-es"
        # backward_model_name = "Helsinki-NLP/opus-mt-es-en"
        # bt_label = "Marian_ES"

    if TRANSFORMER_TYPE == 2:
        forward_model_name = "mrm8488/mbart-large-finetuned-opus-en-es-translation"
        backward_model_name = "mrm8488/mbart-large-finetuned-opus-es-en-translation"
        bt_label = "MBart_ES"


    # input_sentence = "Rapeseed oil smoke causes lung cancer, Amal Kumar Maj."
    # print(do_back_translation(input_sentence, get_back_translation_models(forward_model_name, backward_model_name)))

    # process_single_item(1137, forward_model_name, backward_model_name, bt_label)

    process_train_items(forward_model_name, backward_model_name, bt_label)

    ending_time = datetime.now()
    logging.info("Total time: %s", ending_time - starting_time)
