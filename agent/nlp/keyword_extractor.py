
import spacy
from collections import OrderedDict


def get_key_terms(text):
    nlp = spacy.load("en_core_web_sm")

    candidate_pos = ["PROPN"]
    candidate_deps = ["nsubj"]

    doc = nlp(text)

    selected_words = []
    for sent in doc.sents:
        for token in sent:
            if (token.pos_ in candidate_pos or token.dep_ in candidate_deps) and token.is_stop is False:
                if token.pos_ == "PROPN":
                    selected_words.append(token.text)
                else:
                    selected_words.append(token.text.lower())

    key_terms = ""
    if len(selected_words) > 0:
        key_terms = " ".join(OrderedDict((w,w) for w in selected_words).keys())

    return key_terms
