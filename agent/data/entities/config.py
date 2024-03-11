CONFIG_FILE_PATH = "config.yaml"

ROOT_LOGGER_ID = "agent_launchers"

BACKUPS_PATH = "data/backups/"

METAMAP_INPUT_PATH = "data/metamap_sent/"

UMLS_SEMANTIC_TYPES = "data/umls/SemanticTypes_2018AB.txt"
UMLS_SEMANTIC_GROUPS = "data/umls/SemGroups.txt"

DATA_CACHE_PATH = ".data_cache/"
MODELS_CACHE_PATH = ".models_cache/"
EXPORT_CHECK_INPUT_DATA_RESULTS_FILES = True

RAW_DATA_PATH = "install/raw_data"
JSON_DATA_PATH = "install/json_data"
ITEM_LIST_FILE = "item_list.json"


UNK_TOKEN = "[UNK]"
CUIS_SPO_SEP = "[SEP]"

RESULTS_FILE_TSV = "results/results.tsv"
RESULTS_FILE_TXT = "results/results.txt"

CW_CLASS_VALUES = ["NA", "NF", "FNR", "FRC", "FR"]
FC_CLASS_VALUES = ["NA", "F", "PF", "T"]
FD_CLASS_VALUES = ["NA", "F", "PF", "T"]

ITEMS_TABLE = "annotate_item"

NEW_LABEL = "new"
TRAIN_LABEL = "train"
DEV_LABEL = "dev"
TEST_LABEL = "test"
EXTERNAL_LABEL = "external"
SKIP_VAL_LABEL = "skip_val"
CTT3_LABEL = "ctt3"
CTT3FR_LABEL = "ctt3fr"
CTT3HEALTH_LABEL = "ctt3health"
CTT3SIM_LABEL = "ctt3trainsim"   # same train dataset used in 2021 competition (bigger than official train dataset)
CORPUS_VERIF_LABEL = "cvf"

NEW_ITEM_FILTER = """
    review_level = 8  AND
    needs_revision = false
    """

# instance_type = 'Train'
# (instance_type = 'Train' OR instance_type = 'Train_MBart_ES' OR instance_type = 'Train_Marian_DE' OR instance_type = 'Train_Marian_IT')
TRAIN_ITEM_FILTER = """
    skip_validations = false AND
    review_level = 9  AND
    needs_revision = false AND
    (instance_type = 'Train' OR instance_type = 'Other')
    """
# (instance_type = 'Dev' OR instance_type = 'Other')
# instance_type = 'Dev'
DEV_ITEM_FILTER = """
    skip_validations = false AND
    review_level = 9  AND
    needs_revision = false AND
    instance_type = 'Dev'
    """
TEST_ITEM_FILTER = """
    skip_validations = false AND
    review_level = 9  AND
    needs_revision = false AND
    instance_type = 'Dev'
    """

# --- full pipeline

TRAIN_ITEM_FILTER = """
    skip_validations = false AND
    review_level = 9  AND
    (instance_type = 'Train' OR instance_type = 'Other')
    """
DEV_ITEM_FILTER = """
    skip_validations = false AND
    review_level = 9  AND
    instance_type = 'Dev'
    """
TEST_ITEM_FILTER = """
    skip_validations = false AND
    review_level = 9  AND
    instance_type = 'Dev'
    """
# +++ full pipeline

SKIP_VAL_ITEM_FILTER = """
    skip_validations = true AND
    review_level = 8  AND
    needs_revision = false AND
    instance_type = 'Other'
    """
EXTERNAL_ITEM_FILTER = """
    skip_validations = true AND
    review_level = 9
    """
CTT3_TEST_ITEM_FILTER = """
    skip_validations = true AND
    type = 'Check That! 2021 Task 3A Test Dataset'
    """
CTT3FR_ITEM_FILTER = """
    skip_validations = true AND
    review_level = 9 AND
    type = 'Check That! 2021 Task 3A Test Dataset'
    """
CTT3HEALTH_ITEM_FILTER = """
    skip_validations = true AND
    review_level = 9 AND
    instance_type = 'Health'
    """
CORPUS_VERIFICATION_FILTER = """
    skip_validations = false AND
    review_level = 9 AND
    needs_revision = false
    """
RECOVERY_ITEM_FILTER = """
    skip_validations = true AND
    review_level = 9 AND
    instance_type = 'ReCOVery'
    """

SENTENCES_TABLE = "annotate_sentence"

CW_TRAIN_SENTENCE_FILTER = """
    skip_validations = false AND
    review_level > 4 AND
    (instance_type = 'Train' OR instance_type = 'Other')
    ORDER BY item_id, new_sentence_id
    """

CW_TEST_SENTENCE_FILTER = """
    skip_validations = false AND
    review_level > 4 AND
    instance_type = 'Dev'
    ORDER BY item_id, new_sentence_id
    """


CW_NEW_SENTENCE_FILTER = """
    review_level = 3
    ORDER BY item_id, new_sentence_id
    """

CW_NEW_EXTERNAL_SENTENCE_FILTER = """
    review_level = 3 AND
    skip_validations = true
    ORDER BY item_id, new_sentence_id
    """

CW_EXTERNAL_SENTENCE_FILTER = """
    review_level > 0 AND
    instance_type = 'Health'
    ORDER BY item_id, new_sentence_id
    """

# added fields for 2 transformer ensemble 11/11/22
# FC_TRAIN_TEST_SENTENCE_FIELDS = "id, sentence, sentence_class, subject_cuis, predicate_cuis, object_cuis, big_subject, big_predicate, big_object"
FC_TRAIN_TEST_SENTENCE_FIELDS = "id, sentence, sentence_class, subject_cuis, predicate_cuis, object_cuis, big_subject, big_predicate, big_object, new_sentence_id, item_id, sentence_class, spo_type, subject, predicate, object, metamap_extraction, subject_text_cuis, predicate_text_cuis, object_text_cuis"

# (instance_type = 'Train' OR instance_type = 'Other')
# (instance_type = 'Train' OR instance_type = 'Train_MBart_ES' OR instance_type = 'Train_Marian_DE' OR instance_type = 'Train_Marian_IT')
# instance_type = 'Train'
FC_TRAIN_SENTENCES_FILTER = """
    skip_validations = false AND
    review_level > 7 AND
    (instance_type = 'Train' OR instance_type = 'Other')
    ORDER BY item_id, new_sentence_id
    """

# (instance_type = 'Dev' OR instance_type = 'Other')
# instance_type = 'Dev'
FC_TEST_SENTENCES_FILTER = """
    skip_validations = false AND
    review_level > 7 AND
    instance_type = 'Dev'
    ORDER BY item_id, new_sentence_id
    """

FC_NEW_SENTENCES_FILTER = """
    review_level = 7 AND
    instance_type = 'Other'
    ORDER BY item_id, new_sentence_id
    """

FC_CTT3FR_SENTENCES_FILTER = """
    review_level > 7 AND
    instance_type = 'Health'
    ORDER BY item_id, new_sentence_id
    """

FC_EXTERNAL_SENTENCES_FILTER = """
    skip_validations = true AND
    review_level = 9 AND
    instance_type = 'Other'
    ORDER BY item_id, new_sentence_id
    """


TWEETS_TABLE = "annotate_tweet"

TEXT8_VOC = "text_data/text8_vocabulary.json"
TWEETS_VOC = "text_data/tweets_vocabulary.json"
CORPUS_VOC = "text_data/corpus_vocabulary.json"
TEXT8_PATH = "text_data/text8.txt"
