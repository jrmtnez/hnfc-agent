import logging

from scipy.stats import ttest_rel, jarque_bera, shapiro, wilcoxon

from agent.data.entities.config import ROOT_LOGGER_ID

ALPHA = 0.05
NORMAL = "normal distribution"
NOT_NORMAL = "not normal distribution"
STAT_SIGN = "statistically significant"
NOT_STAT_SIGN = "not statistically significant"

logger = logging.getLogger(ROOT_LOGGER_ID)


def normality_test_jarque_bera(sample):
    logging.debug("Performing Jarque-Bera normality test...")
    result = jarque_bera(sample)
    logging.debug("T-statistic value: %s", result.statistic)
    logging.debug("P-value: %s", result.pvalue)
    if result.pvalue > ALPHA:
        logging.debug("P-value (%s) > alpha (%s) => %s", result.pvalue, ALPHA, NORMAL)
    else:
        logging.debug("P-value (%s) <= alpha (%s) => %s", result.pvalue, ALPHA, NOT_NORMAL)
    return result.pvalue > ALPHA


def normality_test_shapiro_wilk(sample):
    logging.debug("Performing Shapiro-Wilk normality test...")
    result = shapiro(sample)
    logging.debug("T-statistic value: %s", result.statistic)
    logging.debug("P-value: %s", result.pvalue)
    if result.pvalue > ALPHA:
        logging.debug("P-value (%s) > alpha (%s) => %s", result.pvalue, ALPHA, NORMAL)
    else:
        logging.debug("P-value (%s) <= alpha (%s) => %s", result.pvalue, ALPHA, NOT_NORMAL)
    return result.pvalue > ALPHA


def paired_t_test(sample1, sample2):
    logging.debug("Performing Paired T-Test significance test...")
    t_stat, p_value = ttest_rel(sample1, sample2)
    logging.debug("T-statistic value: %s", t_stat)
    logging.debug("P-value: %s", p_value)
    if p_value < ALPHA:
        logging.debug("P-value (%s) < alpha (%s) => %s", p_value, ALPHA, STAT_SIGN)
    else:
        logging.debug("P-value (%s) >= alpha (%s) => %s", p_value, ALPHA, NOT_STAT_SIGN)
    return p_value


def wilcoxon_test(sample1, sample2, alternative="two-sided"):
    logging.debug("Performing Wilcoxon significance test...")
    result = wilcoxon(sample1, sample2, alternative=alternative)
    logging.debug("T-statistic value: %s", result.statistic)
    logging.debug("P-value: %s", result.pvalue)
    if result.pvalue < ALPHA:
        logging.debug("P-value (%s) < alpha (%s) => %s", result.pvalue, ALPHA, STAT_SIGN)
    else:
        logging.debug("P-value (%s) >= alpha (%s) => %s", result.pvalue, ALPHA, NOT_STAT_SIGN)
    return result.pvalue


def significance_test_normal_dist(sample1, sample2):
    if not normality_test_shapiro_wilk(sample1):
        return "sample 1 not normal"
    if not normality_test_shapiro_wilk(sample2):
        return "sample 2 not normal"
    if paired_t_test(sample1, sample2) < ALPHA:
        return "statistically significant"
    else:
        return "not statistically significant"


def significance_test_not_normal_dist(sample1, sample2):
    if wilcoxon_test(sample1, sample2) < ALPHA:
        return "statistically significant"
    else:
        return "not statistically significant"
