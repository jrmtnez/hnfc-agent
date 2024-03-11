import nltk
import re

# def tokenize_sentences(document):
#     document = document.replace(r"\n", r" ")
#     document = document.replace(r"\t", r" ")
#     document = re.sub(r"\s\s+", r" ", document)
#     document = re.sub(r"\.” ([A-Z])", r".”. \1", document)
#     sentences = nltk.sent_tokenize(document)

#     clean_sentences = []
#     for sentence in sentences:
#         sentence = re.sub(r"\.”.", r".”", sentence)
#         clean_sentences.append(sentence)

#     return clean_sentences


def tokenize_sentences(paragraphs_document, spacy_model=None):

    clean_sentences = []

    for document in paragraphs_document.split("<annotate_paragraph>"):
        document = document.replace(r"\n", r" ")
        document = document.replace(r"\t", r" ")
        document = re.sub(r"\s\s+", r" ", document)
        document = re.sub(r"\.” ([A-Z])", r".”. \1", document)

        if spacy_model is None:
            sentences = nltk.sent_tokenize(document)
        else:
            spacy_doc = spacy_model(document)  # for spacy sentences
            sentences = list(spacy_doc.sents)

        for sentence in sentences:
            sentence = str(sentence)
            sentence = re.sub(r"\.”.", r".”", sentence)
            clean_sentences.append(sentence)

    return clean_sentences
