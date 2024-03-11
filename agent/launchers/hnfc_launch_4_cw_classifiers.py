# updated 19/11/2022

import logging

from datetime import datetime

from agent.classifiers.transformer_classifier import transformer_classifier
from agent.classifiers.transformer_classifier_text_trainer import train_loop, eval_loop
from agent.classifiers.data.transformer_text_cw_dataset_mgmt import get_datasets, annotate_dataset
from agent.classifiers.data.transformer_text_cw_dataset_mgmt import exist_cw_sentences_to_annotate



logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)


USE_GPU = False

if __name__ == "__main__":

    if exist_cw_sentences_to_annotate():
        starting_time = datetime.now()

        logging.info("Evaluating sentences check-worthiness...")

        # F1: 0.8994
        transformer_classifier(for_task_label="cw",
                            binary_classifier=True,
                            pretrained_model_label="bert-base-uncased",
                            get_data_function=get_datasets,
                            annotate_dataset_function=annotate_dataset,
                            train_loop_function=train_loop,
                            eval_loop_function=eval_loop,
                            epochs=2,
                            batch_size=16,
                            use_gpu=USE_GPU,
                            use_saved_model=True,
                            evaluate_model=False,
                            annotate_new_instances=True)

        # F1: 0.7087
        transformer_classifier(for_task_label="cw",
                            binary_classifier=False,
                            pretrained_model_label="bert-base-cased",
                            get_data_function=get_datasets,
                            annotate_dataset_function=annotate_dataset,
                            train_loop_function=train_loop,
                            eval_loop_function=eval_loop,
                            epochs=2,
                            batch_size=16,
                            use_gpu=USE_GPU,
                            use_saved_model=True,
                            evaluate_model=False,
                            annotate_new_instances=True)

        ending_time = datetime.now()
        logging.info("Total time: %s.", ending_time - starting_time)
