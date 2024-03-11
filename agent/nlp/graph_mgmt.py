import logging
import re
import networkx as nx
import matplotlib.pyplot as plt
# import pandas as pd
import nltk


from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

from agent.data.sql.sql_mgmt import select_fields_where, get_connection
from agent.nlp.metamap_mgmt import match_extraction_with_spo2, DEFAULT_SEM_GROUPS
from agent.data.entities.config import UNK_TOKEN, ROOT_LOGGER_ID, UMLS_SEMANTIC_TYPES

logger = logging.getLogger(ROOT_LOGGER_ID)

ITEM_TABLE = "annotate_item"
ITEM_FIELDS = "id, main_topic, main_claim, item_class, item_class_4"
ITEM_FILTER = "skip_validations = false AND review_level > 0 ORDER BY id"

ITEM_SPO_GRAPH_FILE_NAME = "data/graphs/item_spo_graph.gexf"
ITEM_TOPICS_GRAPH_FILE_NAME = "data/graphs/item_topics_graph.gexf"

SENTENCE_TABLE = "annotate_sentence"
# SENTENCE_FIELDS = "new_sentence_id, item_id, sentence_class, spo_type, subject, predicate, object, big_subject, big_predicate, big_object, subject_cuis, predicate_cuis, object_cuis, metamap_extraction, sentence"
SENTENCE_FIELDS = "id, sentence, sentence_class, subject_cuis, predicate_cuis, object_cuis, big_subject, big_predicate, big_object, new_sentence_id, item_id, sentence_class, spo_type, subject, predicate, object, metamap_extraction, subject_text_cuis, predicate_text_cuis, object_text_cuis"
SENTENCE_FILTER = "skip_validations = false AND review_level = 9 ORDER BY item_id, new_sentence_id"
SENTENCE_SPO_GRAPH_FILE_NAME = "data/graphs/sentence_spo_graph.gexf"
SENTENCE_PREDICATES_FILE_NAME = "data/graphs/sentence_predicates.tsv"
SENTENCE_TRIPLES_FILE_NAME = "data/graphs/sentence_triples.tsv"
PHRASAL_VERBS_FILE_NAME = "data/english/english_phrasal_verbs.txt"

LAST_TOP_PREDICATES_FILE = "data/graphs/top_predicates_last.tsv"
TOP_PREDICATES_FILE = "data/graphs/top_predicates.tsv"

NEGATION_TOKENS = ["n't", "no" ,"not", "t"]


def build_items_graph():
    lemmatizer = WordNetLemmatizer()

    connection = get_connection()

    spo_triples_list = []
    topics_list =[]
    items = select_fields_where(connection, ITEM_TABLE, ITEM_FIELDS, ITEM_FILTER)
    for item in items:
        item_id = item[0]
        main_topics = item[1]
        main_claim = item[2]
        item_class_4 = item[4]

        main_claim_list = get_elements_list(main_claim)
        main_claim_list[1] = lemmatizer.lemmatize(main_claim_list[1], pos='v')
        main_claim_list.append(item_class_4)
        main_claim_list.append(item_id)
        spo_triples_list.append(main_claim_list)

        logger.debug("item id: %s, claims: %s", item_id, main_claim_list)

        item_topics_list = get_elements_list(main_topics)
        for item_topic in item_topics_list:
            topics_list.append([str(item_id), item_class_4, item_topic])

        logger.debug("item id: %s, topics: %s", item_id, item_topics_list)

    # spo_triples_df = pd.DataFrame(spo_triples_list, columns=['subject', 'predicate', 'object', 'item_class_4', 'item_id'])
    # create_spo_networkx_graph(spo_triples_df)
    save_spo_gexf_graph(spo_triples_list, ITEM_SPO_GRAPH_FILE_NAME)
    save_topics_gexf_graph(topics_list, ITEM_TOPICS_GRAPH_FILE_NAME)


def build_sentences_graph(valid_s_sem_groups=None, valid_o_sem_groups=None, break_down_by_token=True,
                          export_top_predicates=True, node_type="cui"):


    get_s_valid_groups_per_p = (valid_s_sem_groups is None)
    get_o_valid_groups_per_p = (valid_o_sem_groups is None)

    connection = get_connection()

    lemmatizer = WordNetLemmatizer()
    phrasal_verbs_set = load_phrasal_verbs()

    p_file = open(SENTENCE_PREDICATES_FILE_NAME, "w", encoding="utf-8")
    p_file.write("item_id\tsentence_id\tspo_type\tpredicate\tbig_predicate\tone_word_predicate\ttokens_pos\r\n")

    triples_file = open(SENTENCE_TRIPLES_FILE_NAME, "w", encoding="utf-8")
    triples_file.write("item_id\tsentence_id\tsubject\tpredicate\tobject\tvalid subject groups\tvalid object groups\r\n")

    spo_triples_list = []
    sentences = select_fields_where(connection, SENTENCE_TABLE, SENTENCE_FIELDS, SENTENCE_FILTER)

    p_dict = {}

    s_valid_groups_dict = get_valid_groups(which="subject")
    o_valid_groups_dict = get_valid_groups(which="object")

    sem_type_names_dict = get_sem_type_names()

    if logger.getEffectiveLevel() == logging.DEBUG:
        sentence_iterator = sentences
    else:
        sentence_iterator = tqdm(sentences)

    for sentence in sentence_iterator:
        sentence_id = sentence[9]
        item_id = sentence[10]
        sentence_class = sentence[11]
        spo_type = sentence[12]
        predicate = sentence[14]
        big_subject = sentence[6]
        big_predicate = sentence[7]
        big_object = sentence[8]
        metamap_extraction = sentence[16]
        sentence_text = sentence[1]

        norm_p, tokens_pos = get_norm_p(predicate, big_predicate, spo_type, phrasal_verbs_set, lemmatizer)

        if p_dict.get(norm_p) is None:
            p_dict[norm_p] = 1
        else:
            p_dict[norm_p] = p_dict[norm_p] + 1

        p_file.write(f"{item_id}\t{sentence_id}\t{spo_type}\t{predicate}\t{big_predicate}\t{norm_p}\t{tokens_pos}\r\n")

        s_ext, _, o_ext = match_extraction_with_spo2(metamap_extraction, sentence_text, big_subject, predicate, big_object)

        if get_s_valid_groups_per_p:
            valid_s_sem_groups = s_valid_groups_dict.get(norm_p)
            if valid_s_sem_groups is None or valid_s_sem_groups == "[]":
                valid_s_sem_groups = DEFAULT_SEM_GROUPS
        if get_o_valid_groups_per_p:
            valid_o_sem_groups = o_valid_groups_dict.get(norm_p)
            if valid_o_sem_groups is None or valid_o_sem_groups == "[]":
                valid_o_sem_groups = DEFAULT_SEM_GROUPS

        norm_s_list, norm_s_str = get_norm_so(s_ext, valid_s_sem_groups, sem_type_names_dict, node_type=node_type)
        norm_o_list, norm_o_str  = get_norm_so(o_ext, valid_o_sem_groups, sem_type_names_dict, node_type=node_type)

        if break_down_by_token:
            for norm_s in norm_s_list:
                for norm_o in norm_o_list:
                    triples_file.write(f"{item_id}\t{sentence_id}\t{norm_s}\t{norm_p}\t{norm_o}\t{valid_s_sem_groups}\t{valid_o_sem_groups}\r\n")

                    if len(norm_s_list) > 0 and len(norm_o_list) > 0:
                        spo_triples_list.append(new_triple(norm_s, norm_p, norm_o, sentence_class, sentence_id))
                        spo_triples_list.append(new_triple(norm_s, "included_in", f"item_{item_id}", sentence_class, sentence_id))
                        spo_triples_list.append(new_triple(norm_p, "included_in", f"item_{item_id}", sentence_class, sentence_id))
                        spo_triples_list.append(new_triple(norm_o, "included_in", f"item_{item_id}", sentence_class, sentence_id))

                    logger.debug("item id:      %s", item_id)
                    logger.debug("sentence id:  %s", sentence_id)
                    logger.debug("subject:      %s", norm_s_str)
                    logger.debug("predicate:    %s", norm_p)
                    logger.debug("object:       %s", norm_o_str)
                    logger.debug("----------")
        else:
            triples_file.write(f"{item_id}\t{sentence_id}\t{big_subject}\t{norm_p}\t{big_object}\t\t\r\n")
            triples_file.write(f"{item_id}\t{sentence_id}\t{norm_s_str}\t{norm_p}\t{norm_o_str}\t\t\r\n")

            if len(norm_s_list) > 0 and len(norm_o_list) > 0:
                spo_triples_list.append(new_triple(norm_s_str, norm_p, norm_o_str, sentence_class, sentence_id))
                spo_triples_list.append(new_triple(norm_s_str, "included_in", f"item_{item_id}", sentence_class, sentence_id))
                spo_triples_list.append(new_triple(norm_p, "included_in", f"item_{item_id}", sentence_class, sentence_id))
                spo_triples_list.append(new_triple(norm_o_str, "included_in", f"item_{item_id}", sentence_class, sentence_id))

            logger.debug("item id:      %s", item_id)
            logger.debug("sentence id:  %s", sentence_id)
            logger.debug("subject:      %s", norm_s_str)
            logger.debug("predicate:    %s", norm_p)
            logger.debug("object:       %s", norm_o_str)
            logger.debug("----------")

    p_file.close()
    triples_file.close()

    top_p_dict = {}
    sorted_values = sorted(p_dict.values(), reverse=True)
    for i in sorted_values:
        for k in p_dict:
            if p_dict[k] == i:
                top_p_dict[k] = p_dict[k]

    if export_top_predicates:
        with open(LAST_TOP_PREDICATES_FILE, "w", encoding="utf-8") as f:
            for item in top_p_dict.items():
                f.write(f"{item[0]}\t{item[1]}\t[]\t[]\r\n")

    # spo_triples_df = pd.DataFrame(spo_triples_list, columns=['subject', 'predicate', 'object', 'sentence_class', 'sentence_id'])
    # create_spo_networkx_graph(spo_triples_df)
    save_spo_gexf_graph(spo_triples_list, SENTENCE_SPO_GRAPH_FILE_NAME)


def new_triple(s, p, o, sentence_class, sentence_id):
    claim_list = []
    claim_list.append(s)
    claim_list.append(p)
    claim_list.append(o)
    claim_list.append(sentence_class)
    claim_list.append(sentence_id)
    return claim_list


def get_norm_so(ext_tokens, valid_sem_groups, sem_type_names_dict, node_type="sem_group", add_brackets=True):
    normalized_list = []
    cuis_list = []
    for ext_token in ext_tokens:
        if ext_token["group"] in valid_sem_groups:
            if ext_token["CandidateCUI"] != UNK_TOKEN:
                if ext_token["CandidateCUI"] not in cuis_list:  # avoid same cui in several tokens
                    cuis_list.append(ext_token["CandidateCUI"])
                    token = ""
                    avoid_duplicates = False
                    if node_type == "sem_group":
                        token = f'{ext_token["group"]}'
                        avoid_duplicates = True
                    if node_type == "sem_type":
                        token = f'{ext_token["SemType"]}'
                        avoid_duplicates = True
                    if node_type == "sem_type_name":
                        token = f'{sem_type_names_dict.get(ext_token["SemType"])}'
                        avoid_duplicates = True
                    if node_type == "cui":
                        token = f'{ext_token["CandidateCUI"]}'
                    if node_type == "text_cui":
                        token = f'{ext_token["CandidatePreferred"]}'
                    if node_type == "sem_type_sem_group":
                        token = f'[{ext_token["SemType"]}][{ext_token["group"]}]'
                    if node_type == "cui_all":
                        token = f'[{ext_token["CandidateCUI"]}][{ext_token["SemType"]}][{ext_token["group"]}]'
                    if node_type == "text_cui_all":
                        token = f'[{ext_token["CandidatePreferred"]}][{ext_token["SemType"]}][{ext_token["group"]}]'

                    token = change_wrong_chars(token)

                    if add_brackets:
                        token = f"[{token}]"

                    insert_token = True
                    if  avoid_duplicates and token in normalized_list:
                        insert_token = False

                    if insert_token:
                        normalized_list.append(token)

    normalized_string = " ".join(normalized_list)

    return normalized_list, normalized_string


def change_wrong_chars(token):
    token = token.replace("<50", "Less Than 50")
    return token;


def get_valid_groups(which=None):
    valid_groups_dict = {}
    with open(TOP_PREDICATES_FILE, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
        for line in lines:
            line_tokens = line.split("\t")
            if len(line_tokens) == 4:
                if which == "subject":
                    valid_groups_dict[line_tokens[0]] = line_tokens[2]
                if which == "object":
                    valid_groups_dict[line_tokens[0]] = line_tokens[3]
    return valid_groups_dict


def get_sem_type_names():
    sem_types_dict = {}
    with open(UMLS_SEMANTIC_TYPES, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
        for line in lines:
            line_tokens = line.split("|")
            sem_types_dict[line_tokens[0]] = line_tokens[2]
    return sem_types_dict


def get_norm_p(p, big_predicate, spo_type, phrasal_verbs_set, lemmatizer):
    negation = False
    norm_p = ""
    temp_p = ""

    temp_p = big_predicate.lower()
    temp_p = preprocess_tokens(temp_p)

    tokens_pos = nltk.pos_tag(nltk.word_tokenize(temp_p))

    if len(temp_p.split()) > 1 or len(tokens_pos) > 1:

        for (token, pos) in tokens_pos:
            if token in NEGATION_TOKENS:
                negation = True

        # last occurrence of each tag
        temp_p2 = ""
        for (token, pos) in tokens_pos:
            if pos == "VBG":
                temp_p2 = token

        if temp_p2 == "":
            for (token, pos) in tokens_pos:
                if pos == "VBN":
                    temp_p2 = token

        if temp_p2 == "":
            for (token, pos) in tokens_pos:
                if pos == "VB":
                    temp_p2 = token

        if temp_p2 == "":
            for (token, pos) in tokens_pos:
                if pos == "VBD":
                    temp_p2 = token

        if temp_p2 == "":
            for (token, pos) in tokens_pos:
                if pos == "VBZ":
                    temp_p2 = token

        if temp_p2 == "":
            for (token, pos) in tokens_pos:
                if pos == "VBP":
                    temp_p2 = token

        if spo_type in ["manually id", "sentence", "clause", "quoted text"]:
            if len(temp_p2) < 2 or temp_p2 != p.lower():
                if len(p.split()) == 1:
                    temp_p2 = p.lower()

        if temp_p2 == "":
            predicate = p.lower()
            if predicate in phrasal_verbs_set:
                temp_p2 = predicate

        if temp_p2 == "":
            for (token, pos) in tokens_pos:
                if pos == "NN":
                    temp_p2 = token

        temp_p = temp_p2

    norm_p = process_norm_p(temp_p, lemmatizer, negation=negation)

    return norm_p, tokens_pos


def load_phrasal_verbs():
    # extracted from https://en.wiktionary.org/wiki/Category:English_phrasal_verbs
    with open(PHRASAL_VERBS_FILE_NAME, "r", encoding="utf-8") as f:
        phrasal_verbs_set = set(f.read().splitlines())
    return phrasal_verbs_set


def preprocess_tokens(predicate):
    tokens = predicate.split()
    new_tokens = []
    for token in tokens:
        if token in ["’re"]:
            token = "are"
        new_tokens.append(token)
    return " ".join(new_tokens)


def process_norm_p(p, lemmatizer, negation=False):
    p = lemmatizer.lemmatize(p, pos='v')
    if p == "be":
        p = "is"
    if negation:
        p = "not " + p
    return p


def create_spo_networkx_graph(spo_triples_df):
    spo_graph = nx.from_pandas_edgelist(spo_triples_df, 'subject', 'object', create_using=nx.MultiDiGraph())
    node_deg = nx.degree(spo_graph)
    layout = nx.spring_layout(spo_graph, k=0.15, iterations=20)
    plt.figure(num=None, figsize=(120, 90), dpi=80)
    nx.draw_networkx(
        spo_graph,
        node_size=[int(deg[1]) * 500 for deg in node_deg],
        arrowsize=20,
        linewidths=1.5,
        pos=layout,
        edge_color='red',
        edgecolors='black',
        node_color='white',
        )
    labels = dict(zip(list(zip(spo_triples_df.subject, spo_triples_df.object)),
                  spo_triples_df['predicate'].tolist()))
    nx.draw_networkx_edge_labels(spo_graph, pos=layout, edge_labels=labels,
                                 font_color='red')
    plt.axis('off')
    plt.show()


def save_spo_gexf_graph(spo_triples, file_name):
    nodes = []
    for spo_triple in spo_triples:
        for i in [0, 2]:
            if not spo_triple[i] in nodes:
                nodes.append(spo_triple[i])

    with open(file_name, "w", encoding="utf-8") as f:
        f.write("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\r\n")
        f.write("<gexf xmlns=\"http://www.gexf.net/1.2draft\" version=\"1.2\" xmlns:viz=\"http://www.gexf.net/1.2draft/viz\" >\r\n")
        f.write("    <graph mode=\"static\" defaultedgetype=\"directed\">\r\n")
        f.write("        <nodes>\r\n")
        for node in nodes:
            f.write(f"            <node id = \"{filter_string(node)}\" label = \"{filter_string(node)}\"> <viz:color r=\"123\" g=\"104\" b=\"238\" a=\"0.5\" /> </node>\r\n")
        f.write("        </nodes>\r\n")
        f.write("        <edges>\r\n")
        edge_id = 0
        for triple in spo_triples:
            edge_id = edge_id + 1
            f.write(f"            <edge id = \"{edge_id}\" source = \"{filter_string(triple[0])}\" target= \"{filter_string(triple[2])}\" label = \"{filter_string(triple[1])}\" />\r\n")
        f.write("        </edges>\r\n")
        f.write("    </graph>\r\n")
        f.write("</gexf>\r\n")


def save_topics_gexf_graph(topics, file_name):
    nodes = []
    for topic in topics:
        for i in [0, 2]:
            if not topic[i] in nodes:
                nodes.append(topic[i])


    with open(file_name, "w", encoding="utf-8") as f:
        f.write("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\r\n")
        f.write("<gexf xmlns=\"http://www.gexf.net/1.2draft\" version=\"1.2\" xmlns:viz=\"http://www.gexf.net/1.2draft/viz\" >\r\n")
        f.write("    <graph mode=\"static\" defaultedgetype=\"directed\">\r\n")
        f.write("        <nodes>\r\n")
        for node in nodes:
            f.write(f"            <node id = \"{filter_string(node)}\" label = \"{filter_string(node)}\"> <viz:color r=\"123\" g=\"104\" b=\"238\" a=\"0.5\" /> </node>\r\n")
        f.write("        </nodes>\r\n")
        f.write("        <edges>\r\n")
        edge_id = 0
        for topic in topics:
            edge_id = edge_id + 1
            f.write(f"            <edge id = \"{edge_id}\" source = \"{filter_string(topic[0])}\" target= \"{filter_string(topic[2])}\" label = \"{filter_string(topic[1])}\" />\r\n")
        f.write("        </edges>\r\n")
        f.write("    </graph>\r\n")
        f.write("</gexf>\r\n")


def get_elements_list(item_topics):
    item_topics_list = re.findall(r'\[([^]]*)\]', item_topics)
    return item_topics_list


def filter_string(string):
    string = string.replace("&", "and")
    return string
