import logging

from datetime import datetime
from transformers import FunnelForSequenceClassification

from agent.classifiers.transformer_classifier import transformer_classifier
from agent.classifiers.transformer_classifier_text_trainer import train_loop, eval_loop
from agent.classifiers.data.transformer_text_cw_dataset_mgmt import get_datasets, annotate_dataset


logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

if __name__ == "__main__":
    logging.info("Launching CW classifiers...")

    starting_time = datetime.now()

    transformer_classifier(for_task_label="cw",
                           binary_classifier=True,
                           pretrained_model_label="funnel-transformer/intermediate",
                           pretrained_model=FunnelForSequenceClassification,
                           get_data_function=get_datasets,
                           annotate_dataset_function=annotate_dataset,
                           train_loop_function=train_loop,
                           eval_loop_function=eval_loop,
                           epochs=2,
                           batch_size=16,
                           use_gpu=True,
                           use_saved_model=True,
                           evaluate_model=True,
                           annotate_train_instances=True,
                           annotate_new_instances=False,
                           annotate_external_instances=False)

    transformer_classifier(for_task_label="cw",
                           binary_classifier=False,
                           pretrained_model_label="funnel-transformer/intermediate",
                           pretrained_model=FunnelForSequenceClassification,
                           get_data_function=get_datasets,
                           annotate_dataset_function=annotate_dataset,
                           train_loop_function=train_loop,
                           eval_loop_function=eval_loop,
                           epochs=10,
                           batch_size=16,
                           seed_val=0,
                           use_gpu=True,
                           use_saved_model=False,
                           evaluate_model=True,
                           annotate_train_instances=True,
                           annotate_new_instances=False,
                           annotate_external_instances=False)

    ending_time = datetime.now()
    logging.info("Total time: %s.", ending_time - starting_time)
