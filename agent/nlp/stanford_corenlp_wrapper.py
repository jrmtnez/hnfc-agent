from collections import defaultdict
from stanza.server import CoreNLPClient

from agent.data.sql.sql_mgmt import get_connection, select_fields_where, execute_read_query

TABLE = "annotate_sentence"
FIELDS = "id, sentence, metamap_extraction, big_subject, predicate, big_object, manually_identified_predicate"


def get_standard_pipeline_client(be_quiet=True, timeout=30000, memory='3G'):
    client = CoreNLPClient(annotators=["tokenize", "ssplit", "pos", "lemma", "ner", "parse", "depparse", "coref"],
                           be_quiet=be_quiet,
                           timeout=timeout,
                           memory=memory)
    return client


def get_ie_pipeline_client(be_quiet=True, timeout=30000, memory='3G'):
    client = CoreNLPClient(annotators=["openie", "coref"],
                           be_quiet=be_quiet,
                           timeout=timeout,
                           memory=memory)
    return client


def get_item_from_sentences(item_id):
    sentences_filter = f"item_id = {item_id} ORDER BY item_id, new_sentence_id"

    connection = get_connection()
    sentences = select_fields_where(connection, TABLE, FIELDS, sentences_filter)

    text = ""
    for sentence in sentences:
        sentence_text = sentence[1]
        if sentence_text[-1].isalnum():
            text += sentence_text + ".\n"
        else:
            text += sentence_text + "\n"

    connection.close()

    return text


def get_sentence(sentence_id):
    sentences_filter = f"id = {sentence_id}"

    connection = get_connection()
    sentences = select_fields_where(connection, TABLE, FIELDS, sentences_filter)

    text = ""
    for sentence in sentences:
        text = sentence[1]

    connection.close()

    return text


def get_triples(text, client, export_analysis=False):

    OPENIE_PROPS = {
        "openie.resolve_coref": "true",             # default value = "false"
        "openie.triple.all_nominals": "false",      # default value = "false"
        "openie.triple.strict": "true"              # default value = "true"
    }

    ann = client.annotate(text, properties=OPENIE_PROPS)

    if export_analysis:
        with open('temp/ann_openie.txt', 'w') as f:
            f.write(str(ann))

    print(text)
    for sentence in ann.sentence:
        for triples in sentence.openieTriple:
            print("S:", triples.subject)
            print("P:", triples.relation)
            print("O:", triples.object)
            print("Confidence:", triples.confidence)
            print()


def get_corefs(text, client, export_analysis=False):

    COREF_PROPS = {
        "coref.algorithm": "neural"                # default value = "statistical", values = "deterministic", "statistical", "neural"
    }

    ann = client.annotate(text, properties=COREF_PROPS)

    if export_analysis:
        with open('temp/ann_coref.txt', 'w') as f:
            f.write(str(ann))

    # print(ann.corefChain)

    # animacy = defaultdict(dict)
    # for x in ann.corefChain:
    #     for y in x.mention:
    #         print(y.number)
    #         print(y.animacy)
    #         print(y.mentionID)
    #         print(y)

    #         for i in range(y.beginIndex, y.endIndex):
    #             animacy[y.sentenceIndex][i] = True
    #             print(y.sentenceIndex, i)

    # for sent_idx, sent in enumerate(ann.sentence):
    #     print("[Sentence {}]".format(sent_idx+1))
    #     for t_idx, token in enumerate(sent.token):
    #         animate = animacy[sent_idx].get(t_idx, False)
    #         print("{:12s}\t{:12s}\t{:6s}\t{:20s}\t{}".format(token.word, token.lemma, token.pos, token.ner, animate))
    #     print("")

    # for sentence in ann.sentence:
        # print(sentence)
    # print(ann.sentence[0])

    sentence = ann.sentence[0]
    print(sentence)
    print(" ".join(token.word for token in sentence.token))