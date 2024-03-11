import logging
import pandas as pd

from agent.classifiers.evaluation.statistic_test import normality_test_jarque_bera, normality_test_shapiro_wilk
from agent.classifiers.evaluation.statistic_test import paired_t_test, wilcoxon_test


logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)


INPUT_DATA_FILE = "results/tests/fc_bin_stat_sig_data.tsv"
RESULT_FILE = "results/tests/fc_bin_stat_sig_result.tsv"

if __name__ == "__main__":

    logging.info("Launching statistical significance test...")

    plain_data_df = pd.read_csv(INPUT_DATA_FILE, dtype=object, sep='\t')

    grouped_data_df = plain_data_df.groupby("model")

    g1_count = 0
    g2_count = 0

    with open(RESULT_FILE, "w", encoding="utf-8") as results_file:
        results_file.write("model1\tmodel2\tmodel1_avg_f1\tmodel2_avg_f1\tpvalue\n")

        for group_1 in grouped_data_df:

            g1_count += 1
            g2_count = 0
            model_1_label = group_1[0][0:50]
            g1_f1_per_seed = group_1[1]["f1"].to_numpy(dtype=float)

            logging.info("")
            logging.info("Model: %s %s", g1_count, model_1_label)
            logging.info("Number of samples: %s", len(g1_f1_per_seed))
            logging.info("F1 scores: %s", g1_f1_per_seed)
            logging.info("Mean F1 score: %.5f", g1_f1_per_seed.mean())
            logging.info("Normal distribution: %s", normality_test_shapiro_wilk(g1_f1_per_seed))
            logging.info("")

            for group_2 in grouped_data_df:

                g2_count += 1
                model_2_label = group_2[0][0:50]
                g2_f1_per_seed = group_2[1]["f1"].to_numpy(dtype=float)
                logging.info("F1 scores: %s", g2_f1_per_seed)

                # p_value = paired_t_test(g1_f1_per_seed, g2_f1_per_seed)
                # logging.info("%s %s paired T-test p-value: %.10f", model_1_label, model_2_label, p_value)

                if g1_count == g2_count:
                    p_value = 1
                else:
                    # p_value = wilcoxon_test(g1_f1_per_seed, g2_f1_per_seed, alternative="greater")
                    p_value = wilcoxon_test(g1_f1_per_seed, g2_f1_per_seed)
                logging.info("%s_%s %s_%s Wilcoxon p-value: %.10f", g1_count, model_1_label, g2_count, model_2_label, p_value)

                results_file.write(f"{g1_count}_{model_1_label}\t{g2_count}_{model_2_label}\t{g1_f1_per_seed.mean():.5f}\t{g2_f1_per_seed.mean():.5f}\t{p_value:.15f}\n")
