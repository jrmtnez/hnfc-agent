import logging

from datetime import datetime
from transformers import AlbertForSequenceClassification, BertForSequenceClassification, FunnelForSequenceClassification

from agent.classifiers.data.transformer_text_triple_fc_dataset_mgmt import get_datasets, annotate_dataset
from agent.classifiers.transformer_classifier_triple_trainer import train_loop, eval_loop
from agent.classifiers.transformer_classifier import transformer_classifier


logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)


if __name__ == "__main__":
    starting_time = datetime.now()

    # transformer_classifier(for_task_label="fc",
    #                        binary_classifier=True,
    #                        pretrained_model_label="albert-base-v2",
    #                        pretrained_model=AlbertForSequenceClassification,
    #                        get_data_function=get_datasets,
    #                        annotate_dataset_function=annotate_dataset,
    #                        train_loop_function=train_loop,
    #                        eval_loop_function=eval_loop,
    #                        epochs=2,
    #                        batch_size=16,
    #                        seed_val=0,
    #                        max_lenght=250,
    #                        expand_tokenizer=True,
    #                        use_gpu=False,
    #                        use_saved_model=True,
    #                        annotate_new_instances=False,
    #                        annotate_test_instances=True,
    #                        annotate_train_instances=False,
    #                        label="test_211121")

    # transformer_classifier(for_task_label="fc",
    #                        binary_classifier=False,
    #                        pretrained_model_label="bert-base-cased",
    #                        pretrained_model=BertForSequenceClassification,
    #                        get_data_function=get_datasets,
    #                        annotate_dataset_function=annotate_dataset,
    #                        train_loop_function=train_loop,
    #                        eval_loop_function=eval_loop,
    #                     #    epochs=10,
    #                        epochs=1,
    #                        batch_size=16,
    #                        seed_val=0,
    #                        max_lenght=250,
    #                        expand_tokenizer=True,
    #                        use_gpu=False,
    #                        use_saved_model=True,
    #                        evaluate_model=True,
    #                        annotate_new_instances=False,
    #                        annotate_test_instances=True,
    #                        annotate_train_instances=False,
    #                        label="test_211121")

    # -----------------------------------------------------------------------------------------------------------------------
    # paper binary classifier F1: 0.697
    #
    # fc_tr_bin_True_pm_funnel-transformer-intermediate_tt_spo_variable_ex_True_ml_128_ep_2_bs_16_sv_0_gpu_True_test_20220213
    # -----------------------------------------------------------------------------------------------------------------------

    transformer_classifier(for_task_label="fc",
                           binary_classifier=True,
                           pretrained_model_label="funnel-transformer/intermediate",
                           pretrained_model=FunnelForSequenceClassification,
                           tokenizer_type="spo_variable",
                           get_data_function=get_datasets,
                           annotate_dataset_function=annotate_dataset,
                           train_loop_function=train_loop,
                           eval_loop_function=eval_loop,
                           epochs=2,
                           batch_size=16,
                           seed_val=0,
                           max_lenght=128,
                           expand_tokenizer=True,
                           use_gpu=True,
                           use_saved_model=False,
                           evaluate_model=True,
                           annotate_new_instances=False,
                           annotate_test_instances=True,
                           annotate_train_instances=False,
                           label="test_20220213")


    ending_time = datetime.now()
    logging.info("Total time: %s.", ending_time - starting_time)
