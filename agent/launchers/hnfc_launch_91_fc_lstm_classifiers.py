import logging

from datetime import datetime

from agent.classifiers.lstm_classifier import lstm

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

USE_GPU = True


if __name__ == "__main__":
    starting_time = datetime.now()

    lstm(for_task_label="fc",
         binary_classifier=True,
         cuis=True,
         lstm_classif_vector="output",
         embeddings_dim=250,
         lstm_layers=3,
         hidden_layer_size=1000,
         bidirectional=True,
         dropout=0.5,
         add_output_layer=False,
         use_class_weight=False,
         epochs=25,
         batch_size=16,
         seed_val=0,
         use_gpu=USE_GPU,
         label="test_211121")




    ending_time = datetime.now()
    logging.info("Total time: %s.", ending_time - starting_time)
