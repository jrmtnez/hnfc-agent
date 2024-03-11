import logging

from datetime import datetime

from agent.classifiers.ffnn_classifier import ffnn

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

USE_GPU = False


if __name__ == "__main__":
    starting_time = datetime.now()

    ffnn(for_task_label="fc",
         binary_classifier=True,
         cuis=False,
         hidden_layer_size=100,
         activation_fn="sigmoid",
         use_class_weight=True,
         epochs=100,
         batch_size=16,
         seed_val=0,
         use_gpu=USE_GPU,
         label="test_211121")


    ffnn(for_task_label="fc",
         binary_classifier=False,
         cuis=False,
         hidden_layer_size=100,
         activation_fn="relu",
         use_class_weight=False,
         epochs=100,
         batch_size=16,
         seed_val=0,
         use_gpu=USE_GPU,
         label="test_211121")

    ending_time = datetime.now()
    logging.info("Total time: %s.", ending_time - starting_time)
