import numpy as np
import re
import stanfordnlp
import nltk

class Sequencer():
    def __init__(self, text, max_words, tokenizer="re", lang="en", oov_token="<UNK>", min_word_lenght=1, lower=True):
        self.lang = lang
        self.lower = lower
        self.min_word_lenght = min_word_lenght
        self.tokenizer = tokenizer
        self.word_index, self.reverse_word_index, self.unique_word_count = self.create_word_index(text, max_words, oov_token)

    def word_tokenize(self, text):
        if self.lower:
            text = text.lower()
        if self.tokenizer == "re":
            return re.findall(r"\w+", text)
        if self.tokenizer == "nltk":
            return nltk.word_tokenize(text)
        if self.tokenizer == "stanfordnlp":
            tokens = []
            if self.lang == "ar":
                nlp = stanfordnlp.Pipeline(processors='tokenize,mwt', lang='ar')
            if self.lang == "en":
                nlp = stanfordnlp.Pipeline(processors='tokenize', lang='en')
            if self.lang == "es":
                nlp = stanfordnlp.Pipeline(processors='tokenize', lang='es')

            doc = nlp(text)
            for _, sentence in enumerate(doc.sentences):
                for token in sentence.tokens:
                    tokens.append(token.text)
            return tokens

    def create_word_index(self, text, max_words, oov_token):
        # build a full word index, full reverse index and word count
        full_word_index = {oov_token: 1}
        full_reverse_word_index = {1: oov_token}
        full_word_count = {}
        i = 1
        for chunk in text:
            words_array = self.word_tokenize(chunk)

            for word in words_array:
                # only words with more than min_word_lenght chars
                if len(word) >= self.min_word_lenght:
                    if full_word_index.get(word) is None:
                        i = i + 1
                        full_word_index[word] = i
                        full_reverse_word_index[i] = word
                        full_word_count[word] = 1
                    else:
                        full_word_count[word] = 1 + full_word_count.get(word)

        unique_word_count = len(full_word_index)

        #  sort word count index descending
        sorted_word_count = {k: v for k, v in sorted(full_word_count.items(), reverse=True, key=lambda item: item[1])}

        # build word indexes truncated to "max_words"
        truncated_word_index = {oov_token: 1}
        truncated_reverse_word_index = {1: oov_token}
        i = 1
        for word in sorted_word_count:
            i = i + 1
            if i > max_words:
                break
            truncated_word_index[word] = i
            truncated_reverse_word_index[i] = word

        return truncated_word_index, truncated_reverse_word_index, unique_word_count

    def get_word_dict_index(self, word):
        if self.word_index.get(word) is None:
            return 1
        else:
            return self.word_index.get(word)

    def fit_on_text(self, text, max_len):
        indexed_text = []
        for chunk in text:
            # truncate result if lenght is bigger than "max_len"
            words_array = self.word_tokenize(chunk)[:max_len]

            # apply truncated index to tokenized text
            word_list = [self.get_word_dict_index(word) for word in words_array]

            # append zeroes if lenght is smaller than "max_len"
            while len(word_list) < max_len:
                word_list.append(0)

            indexed_text.append(np.array(word_list))

        return np.array(indexed_text)

    def get_words(self, text):
        tokenized_text = []
        for chunk in text:
            words_array = self.word_tokenize(chunk)
            tokenized_text.append(words_array)
        # return np.array(tokenized_text)
        return tokenized_text
