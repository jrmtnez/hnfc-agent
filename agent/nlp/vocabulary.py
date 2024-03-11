import nltk
import json
import logging

from os.path import exists
from tqdm import tqdm

from agent.data.entities.config import SENTENCES_TABLE, TWEETS_TABLE, FC_TRAIN_SENTENCES_FILTER
from agent.data.entities.config import TEXT8_PATH, TEXT8_VOC, TWEETS_VOC, CORPUS_VOC
from agent.data.sql.sql_mgmt import get_connection, select_fields_where

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


SENTENCE_FIELDS = "sentence"
TWEET_FIELDS = "full_text"

def get_corpus_vocabulary(common_words, min_word_lenght=2, max_words=500, use_saved_model=False):
    if exists(CORPUS_VOC) and use_saved_model:
        with open(CORPUS_VOC, encoding="utf-8") as json_file:
            sorted_word_count = json.load(json_file)
    else:
        connection = get_connection()
        annotate_sentences = select_fields_where(connection, SENTENCES_TABLE, SENTENCE_FIELDS, FC_TRAIN_SENTENCES_FILTER)

        if logger.getEffectiveLevel() == logging.DEBUG:
            sentences_set = tqdm(annotate_sentences)
        else:
            sentences_set = annotate_sentences

        i = 0
        full_word_count = {}
        for sentence in sentences_set:
            sentence_text = sentence[0]

            sentence_tokens = nltk.word_tokenize(sentence_text)
            for word in sentence_tokens:
                if len(word) >= min_word_lenght:
                    if common_words.get(word) is None:
                        if full_word_count.get(word) is None:
                            i = i + 1
                            full_word_count[word] = 1
                        else:
                            full_word_count[word] = 1 + full_word_count.get(word)

        connection.close()

        sorted_word_count = {k: v for k, v in sorted(full_word_count.items(), reverse=True, key=lambda item: item[1])}

        with open(CORPUS_VOC, "w", encoding="utf-8") as write_file:
            json.dump(sorted_word_count, write_file, indent=4, separators=(",", ": "))

    truncated_word_list = []
    i = 0
    for word in sorted_word_count:
        i = i + 1
        if i > max_words:
            break
        truncated_word_list.append(word)

    return truncated_word_list


def get_tweets_vocabulary(min_word_lenght=2):

    connection = get_connection()
    tweets = select_fields_where(connection, TWEETS_TABLE, TWEET_FIELDS, "true")

    if logger.getEffectiveLevel() == logging.DEBUG:
        tweets_set = tqdm(tweets)
    else:
        tweets_set = tweets

    i = 0
    full_word_count = {}
    for tweet in tweets_set:
        full_text = tweet[0]

        tweet_tokens = nltk.word_tokenize(full_text)
        for word in tweet_tokens:
            if len(word) >= min_word_lenght:
                if full_word_count.get(word) is None:
                    i = i + 1
                    full_word_count[word] = 1
                else:
                    full_word_count[word] = 1 + full_word_count.get(word)

    connection.close()

    sorted_word_count = {k: v for k, v in sorted(full_word_count.items(), reverse=True, key=lambda item: item[1])}

    with open(TWEETS_VOC, "w", encoding="utf-8") as write_file:
        json.dump(sorted_word_count, write_file, indent=4, separators=(",", ": "))


def get_text8_vocabulary(min_word_lenght=2, max_words=1000):
    if exists(TEXT8_VOC):
        with open(TEXT8_VOC, encoding="utf-8") as json_file:
            data = json.load(json_file)

        truncated_word_index = {}
        i = 0
        for word in data:
            i = i + 1
            if i > max_words:
                break
            truncated_word_index[word] = i

        return truncated_word_index

    with open(TEXT8_PATH, encoding="utf-8") as text_file:
        lines = text_file.readlines()

    i = 0
    full_word_count = {}
    for line in tqdm(lines):
        line_tokens = nltk.word_tokenize(line)
        for word in line_tokens:
            if len(word) >= min_word_lenght:
                if full_word_count.get(word) is None:
                    i = i + 1
                    full_word_count[word] = 1
                else:
                    full_word_count[word] = 1 + full_word_count.get(word)

    sorted_word_count = {k: v for k, v in sorted(full_word_count.items(), reverse=True, key=lambda item: item[1])}

    truncated_word_index = {}
    i = 0
    for word in sorted_word_count:
        i = i + 1
        if i > max_words:
            break
        truncated_word_index[word] = i

    with open(TEXT8_VOC, "w", encoding="utf-8") as write_file:
        json.dump(sorted_word_count, write_file, indent=4, separators=(",", ": "))

    return truncated_word_index
