import logging
import stanza
import re
import sys
import json

from tqdm import tqdm
from stanza.pipeline.core import DownloadMethod

from agent.data.entities.config import ROOT_LOGGER_ID
from agent.data.sql.sql_mgmt import get_connection, select_fields_where, update_text_field, update_non_text_field
from agent.nlp.metamap_mgmt import match_extraction_with_spo, match_extraction_with_spo2, get_tokens_str
from agent.nlp.kg_mgmt import get_related_subjects, get_related_objects

logger = logging.getLogger(ROOT_LOGGER_ID)

# STANZA_DOWNLOAD_METHOD = DownloadMethod.NONE
# STANZA_DOWNLOAD_METHOD = DownloadMethod.DOWNLOAD_RESOURCES
STANZA_DOWNLOAD_METHOD = DownloadMethod.REUSE_RESOURCES

TABLE = "annotate_sentence"
FIELDS = "id, sentence, metamap_extraction, big_subject, predicate, big_object, manually_identified_predicate"

DEBUG_NLP_DOC = False

PRESENT_PARTICIPLE_PREFIXES = ["is", "are", "was", "were", "have been", "had been", "will be", "shall be", "will have been", "would be", "would have been",
                               "isn't", "aren't", "wasn't", "weren't", "haven't been", "hadn't been", "won't be", "shalln't be", "won't have been", "wouldn't be", "wouldn't have been",
                               "is not", "are not", "was not", "were not", "have not been", "had not been", "will not be", "shall not be", "will not have been", "would not be", "would not have been"]

PAST_PARTICIPLE_PREFIXES = ["has", "have", "had", "will have", "would have",
                            "hasn't", "haven't", "hadn't", "won't have", "wouldn't have",
                            "has not", "have not", "had not", "will not have", "would not have"]

PAST_PARTICIPLE_PASSIVE_PREFIXES = ["is", "is being", "are", "are being", "was", "was being", "have been", "has been", "has been being", "had been", "will be", "will have been",
                                    "isn't", "isn't being", "aren't", "aren't being", "wasn't", "wasn't being", "haven't been", "hasn't been", "hasn't been being", "hadn't been", "won't be", "won't have been",
                                    "is not", "is not being", "are not", "are not being", "was not", "was not being", "have not been", "has not been", "has not been being", "had not been", "will not be", "will not have been"]

INFINITIVE_PREFIXES = ["choose to", "decide to", "expect to", "forget to", "hate to", "hope to", "intend to", "learn to",
                       "like to", "love to", "mean to", "plan to", "prefer to", "remember to", "want to", "would like to",
                       "would love to", "agree to", "promise to", "refuse to", "threaten to", "arrange to", "attempt to",
                       "fail to", "help to", "manage to", "end to", "try to", "tend to", "need to",
                       "chooses to", "decides to", "expects to", "forgets to", "hates to", "hopes to", "intends to", "learns to",
                       "likes to", "loves to", "means to", "plans to", "prefers to", "remembers to", "wants to", "would likes to",
                       "would loves to", "agrees to", "promises to", "refuses to", "threatens to", "arranges to", "attempts to",
                       "fails to", "helps to", "manages to", "ends to", "tries to", "tends to", "needs to",
                       "is able to", "are able to", "was able to", "were able to",
                       "to", "will", "would",
                       "won't", "wouldn't",
                       "will not", "would not"]

INFINITIVE_MODAL_PREFIXES = ["can", "could", "may", "might", "should", "ought to", "must",
                             "can't", "couldn't", "mayn't", "mightn't", "shouldn't", "oughtn't to", "mustn't",
                             "cannot", "could not", "may not", "might not", "should not", "ought not to", "must not"]

INFINITIVE_AUX_PREFIXES = ["did", "do", "does",
                           "didn't", "don't", "doesn't",
                           "did not", "do not", "does not"]

PRESENT_PARTICIPLES = ["causing", "spreading", "acquiring", "activating", "affecting", "producing", "attacking",
                       "developing", "inducing", "damaging", "leading", "promoting", "transmitting", "triggering",
                       "yielding", "hurting", "harming", "disrupting", "resulting", "contributing", "dying",
                       "experiencing", "infecting", "releasing", "associating", "showing", "providing",
                       "recovering", "contaminating", "predicting", "diagnosing", "linking", "injuring",
                       "suggesting", "banning", "living", "making", "feeding", "working", "acting", "blowing",
                       "reflecting", "noticing",

                       "improving", "helping", "preventing", "killing", "eliminating", "fighting", "protecting",
                       "avoiding", "combating", "blocking", "curing", "benefiting", "destroying", "stopping",
                       "neutralizing", "healing", "truncating", "relying", "reversing", "treating", "remedying",
                       "inhibiting", "hampering",

                       "increasing", "reducing", "slowing", "decreasing", "lowering", "boosting", "amplifying",
                       "fueling", "fuelling", "elevating", "exacerbating", "raising", "rising", "quieting",
                       "tripling", "doubling", "varying", "halving",

                       "containing", "contradicting", "keeping", "gaining", "receiving", "becoming",
                       "prescribing"]

PAST_PARTICIPLES = ["caused", "spread", "acquired", "activated", "affected", "produced", "attacked",
                    "developed", "induced", "damaged", "led", "promoted", "transmitted", "triggered",
                    "yielded", "hurt", "harmed", "disrupted", "resulted", "contributed", "died",
                    "experienced", "infected", "released", "associated", "showed", "provided",
                    "recovered", "contaminated", "predicted", "diagnosed", "linked", "injured",
                    "suggested", "banned", "lived", "made", "fed", "worked", "acted", "blew",
                    "reflected", "noticed",

                    "improved", "helped", "prevented", "killed", "eliminated", "fought", "protected",
                    "avoided", "combated", "blocked", "cured", "benefited", "destroyed", "stopped",
                    "neutralized", "healed", "truncated", "relied", "reversed", "treated", "remedied",
                    "inhibited", "hampered",

                    "increased", "reduced", "slowed", "decreased", "lowered", "boosted", "amplified",
                    "fueled", "fuelled", "elevated", "exacerbated", "raised", "rose", "quieted",
                    "tripled", "doubled", "varied", "halved",

                    "contained", "contradicted", "kept", "gained", "received", "become",
                    "prescribed"]

INFINITIVES = ["cause", "spread", "acquire", "activate", "affect", "produce", "attack",
               "develop", "induce", "damage", "lead", "promote", "transmit", "trigger",
               "yield", "hurt", "harm", "disrupt", "result", "contribute", "die",
               "experience", "infect", "release", "associate", "show", "provide",
               "recover", "contaminate", "predict", "diagnose", "link", "injure",
               "suggest", "ban", "live", "make", "feed", "work", "act", "blow",
               "reflect", "notice",

               "improve", "help", "prevent", "kill", "eliminate", "fight", "protect",
               "avoid", "combat", "block", "cure", "benefit", "destroy", "stop",
               "neutralize", "heal", "truncate", "rely", "reverse", "treat", "remedy",
               "inhibit", "hamper",

               "increase", "reduce", "slow", "decrease", "lower", "boost", "amplify",
               "fuel", "fuel", "elevate", "exacerbate", "raise", "rise", "quiet",
               "triple", "double", "vary", "halve",

               "contain", "contradict", "keep", "gain", "receive", "become",
               "prescribe"]

PRESENT3RDS = ["causes", "spreads", "acquires", "activates", "affects", "produces", "attacks",
               "develops", "induces", "damages", "leads", "promotes", "transmits", "triggers",
               "yields", "hurts", "harms", "disrupts", "results", "contributes", "dies",
               "experiences", "infects", "releases", "associates", "shows", "provides",
               "recovers", "contaminates", "predicts", "diagnoses", "links", "injures",
               "suggests", "bans", "lives", "makes", "feeds", "works", "acts", "blows",
               "reflects", "notices",

               "improves", "helps", "prevents", "kills", "eliminates", "fights", "protects",
               "avoids", "combats", "blocks", "cures", "benefits", "destroys", "stops",
               "neutralizes", "heals", "truncates", "relies", "reverses", "treats", "remedies",
               "inhibits", "hampers",

               "increases", "reduces", "slows", "decreases", "lowers", "boosts", "amplifies",
               "fuels", "fuels", "elevates", "exacerbates", "raises", "rises", "quiets",
               "triples", "doubles", "varies", "halves",

               "contains", "contradicts", "keeps", "gains", "receives", "becomes",
               "prescribes"]

IRREGULAR_PAST_SIMPLES = ["became"]

MODIFIERS = ["also", "successfully", "necessarily", "often", "even", "still", "possibly", "totally", "usually"]

# PRESENT_PARTICIPLES = ["infecting"]
# PAST_PARTICIPLES = ["infected"]
# INFINITIVES = ["infect"]
# PRESENT3RDS = ["infects"]


def clear_spos(review_level):

    logger.info("stage 0 (clear_spos): clearing spo fields")

    connection = get_connection()
    update_text_field(connection, TABLE, f"review_level <= {review_level} AND predicate_not_found = false", "spo_type", "")
    update_text_field(connection, TABLE, f"review_level <= {review_level} AND predicate_not_found = false", "subject", "")
    update_text_field(connection, TABLE, f"review_level <= {review_level} AND predicate_not_found = false", "predicate", "")
    update_text_field(connection, TABLE, f"review_level <= {review_level} AND predicate_not_found = false", "object", "")
    update_text_field(connection, TABLE, f"review_level <= {review_level} AND predicate_not_found = false", "big_subject", "")
    update_text_field(connection, TABLE, f"review_level <= {review_level} AND predicate_not_found = false", "big_predicate", "")
    update_text_field(connection, TABLE, f"review_level <= {review_level} AND predicate_not_found = false", "big_object", "")
    connection.commit()
    connection.close()


def mark_no_predicate_external_sentences(review_level):
    connection = get_connection()
    update_non_text_field(connection, TABLE,
                          f"""
                          review_level <= {review_level} AND
                          skip_validations = true AND
                          check_worthy_auto = 'FR' AND
                          predicate_not_found = false AND
                          (big_subject = '' OR big_predicate = '' OR big_object = '')
                          """,
                          "predicate_not_found", True)
    connection.commit()
    connection.close()


def extract_spos_from_sentences_given_predicate(review_level, external=False, full_pipeline=False):
    #
    # stage 1: looking for SPOs from a predefined set of predicates
    #

    logger.info("Stage 1 - external: %s (extract_spos_from_sentences_given_predicate): looking for SPOs from a predefined set of predicates", external)

    long_predicates, short_predicates = build_predicates()

    # con predicate_not_found evitamos que los items externos sin predicado se calculen una y otra vez
    connection = get_connection()
    if external:
        stage1_sentences_filter = f"""
            review_level = {review_level} AND
            skip_validations = true AND
            check_worthy_auto = 'FR' AND
            predicate = '' AND
            predicate_not_found = false
            ORDER BY item_id, new_sentence_id
            """
    else:
        if full_pipeline:
            stage1_sentences_filter = f"""
                review_level = {review_level} AND
                skip_validations = false AND
                check_worthy_score_auto >= 0.5 AND
                predicate = ''
                ORDER BY item_id, new_sentence_id
                """
        else:
            stage1_sentences_filter = f"""
                review_level = {review_level} AND
                skip_validations = false AND
                check_worthy = 'FR' AND
                predicate = ''
                ORDER BY item_id, new_sentence_id
                """

    sentences = select_fields_where(connection, TABLE, FIELDS, stage1_sentences_filter)

    if logger.getEffectiveLevel() == logging.DEBUG:
        sentences_set = sentences
    else:
        sentences_set = tqdm(sentences)

    for sentence in sentences_set:
        sentence_id = sentence[0]
        sentence_text = sentence[1]

        logger.debug("Id: %s, text: %s", sentence_id, sentence_text[0:100])

        found_predicate = False

        for p in long_predicates:
            found_predicate = split_sentence_given_predicate_with_update(connection, sentence_id, sentence_text, p, "given long")
            if found_predicate:
                break

        if not found_predicate:
            for p in short_predicates:
                found_predicate = split_sentence_given_predicate_with_update(connection, sentence_id, sentence_text, p, "given short")
                if found_predicate:
                    break

        connection.commit()
    connection.close()


def build_predicates():

    for present_participle, past_participle, infinitive, present3rd in zip(PRESENT_PARTICIPLES, PAST_PARTICIPLES, INFINITIVES, PRESENT3RDS):
        logger.debug("present participle: %s, past participle: %s, infinitive: %s, present 3rd: %s",
                     present_participle, past_participle, infinitive, present3rd)

    long_predicates = []
    short_predicates = []

    for present_participle, past_participle, infinitive, present3rd in zip(PRESENT_PARTICIPLES, PAST_PARTICIPLES, INFINITIVES, PRESENT3RDS):
        # is causing
        for present_participle_prefix in PRESENT_PARTICIPLE_PREFIXES:
            logger.debug("%s %s", present_participle_prefix, present_participle)
            long_predicates.append(f"{present_participle_prefix} {present_participle}")

        # is often causing
        for present_participle_prefix in PRESENT_PARTICIPLE_PREFIXES:
            for modifier in MODIFIERS:
                logger.debug("%s %s %s", present_participle_prefix, modifier, present_participle)
                long_predicates.append(f"{present_participle_prefix} {modifier} {present_participle}")

        # can be causing
        for infinitive_modal_prefix in INFINITIVE_MODAL_PREFIXES:
            logger.debug("%s be %s", infinitive_modal_prefix, present_participle)
            long_predicates.append(f"{infinitive_modal_prefix} be {present_participle}")

        # has caused
        for past_participle_prefix in PAST_PARTICIPLE_PREFIXES:
            logger.debug("%s %s", past_participle_prefix, past_participle)
            long_predicates.append(f"{past_participle_prefix} {past_participle}")

        # has often caused
        for past_participle_prefix in PAST_PARTICIPLE_PREFIXES:
            for modifier in MODIFIERS:
                logger.debug("%s %s %s", past_participle_prefix, modifier, past_participle)
                long_predicates.append(f"{past_participle_prefix} {modifier} {past_participle}")

        # are caused
        for past_participle_passive_prefix in PAST_PARTICIPLE_PASSIVE_PREFIXES:
            logger.debug("%s %s", past_participle_passive_prefix, past_participle)
            long_predicates.append(f"{past_participle_passive_prefix} {past_participle}")

        # are often caused
        for past_participle_passive_prefix in PAST_PARTICIPLE_PASSIVE_PREFIXES:
            for modifier in MODIFIERS:
                logger.debug("%s %s %s", past_participle_passive_prefix, modifier, past_participle)
                long_predicates.append(f"{past_participle_passive_prefix} {modifier} {past_participle}")

        # to cause, will cause, expects to cause
        for infinitive_prefix in INFINITIVE_PREFIXES:
            logger.debug("%s %s", infinitive_prefix, infinitive)
            long_predicates.append(f"{infinitive_prefix} {infinitive}")

        # can cause
        for infinitive_modal_prefix in INFINITIVE_MODAL_PREFIXES:
            logger.debug("%s %s", infinitive_modal_prefix, infinitive)
            long_predicates.append(f"{infinitive_modal_prefix} {infinitive}")

        # can even cause
        for infinitive_modal_prefix in INFINITIVE_MODAL_PREFIXES:
            for modifier in MODIFIERS:
                logger.debug("%s %s %s", infinitive_modal_prefix, modifier, infinitive)
                long_predicates.append(f"{infinitive_modal_prefix} {modifier} {infinitive}")

        # can be caused
        for infinitive_modal_prefix in INFINITIVE_MODAL_PREFIXES:
            logger.debug("%s be %s", infinitive_modal_prefix,past_participle)
            long_predicates.append(f"{infinitive_modal_prefix} be {past_participle}")

        # can also caused
        for infinitive_modal_prefix in INFINITIVE_MODAL_PREFIXES:
            for modifier in MODIFIERS:
                logger.debug("%s %s %s", infinitive_modal_prefix, modifier, past_participle)
                long_predicates.append(f"{infinitive_modal_prefix} {modifier} {past_participle}")

        # can also be caused
        for infinitive_modal_prefix in INFINITIVE_MODAL_PREFIXES:
            for modifier in MODIFIERS:
                logger.debug("%s %s be %s", infinitive_modal_prefix, modifier, past_participle)
                long_predicates.append(f"{infinitive_modal_prefix} {modifier} be {past_participle}")

        # did not cause
        for infinitive_aux_prefix in INFINITIVE_AUX_PREFIXES:
            logger.debug("%s %s", infinitive_aux_prefix, infinitive)
            long_predicates.append(f"{infinitive_aux_prefix} {infinitive}")

        # did not necessarily cause
        for infinitive_aux_prefix in INFINITIVE_AUX_PREFIXES:
            for modifier in MODIFIERS:
                logger.debug("%s %s %s", infinitive_aux_prefix, modifier, infinitive)
                long_predicates.append(f"{infinitive_aux_prefix} {modifier} {infinitive}")

        logger.debug(infinitive)
        logger.debug(present3rd)
        logger.debug(present_participle)
        logger.debug(past_participle)
        short_predicates.append(infinitive)
        short_predicates.append(present3rd)
        short_predicates.append(present_participle)
        short_predicates.append(past_participle)

    for irregular_past_simple in IRREGULAR_PAST_SIMPLES:
        short_predicates.append(irregular_past_simple)

    return long_predicates, short_predicates


def split_sentence_given_predicate_with_update(connection, sentence_id, sentence_text, p, spo_type):

    s, o = split_sentence_given_predicate(sentence_text, p)

    found_predicate = (s != "" and o != "")
    if found_predicate:
        update_text_field(connection, TABLE, "id = " + str(sentence_id), "spo_type", spo_type)
        update_text_field(connection, TABLE, "id = " + str(sentence_id), "subject", s)
        update_text_field(connection, TABLE, "id = " + str(sentence_id), "predicate", p)
        update_text_field(connection, TABLE, "id = " + str(sentence_id), "object", o)
        update_text_field(connection, TABLE, "id = " + str(sentence_id), "big_subject", s)
        update_text_field(connection, TABLE, "id = " + str(sentence_id), "big_predicate", p)
        update_text_field(connection, TABLE, "id = " + str(sentence_id), "big_object", o)

    return found_predicate


def split_sentence_given_predicate(sentence_text, p):

    s = ""
    o = ""
    pos = -1

    sentence_text_lower = sentence_text.lower().replace("’", "'")

    compiled_re = re.compile(f"(\\b{p.lower()}\\b)")
    addresses = compiled_re.finditer(sentence_text_lower)
    for address in addresses:
        pos = address.span()[0]
        break

    if pos != -1:
        s = sentence_text[:pos]

        o = ''
        prev_char = ''
        next_char = ''
        if len(sentence_text) > pos + len(p):
            o = sentence_text[pos + len(p):]
            prev_char = sentence_text[pos - 1]
            next_char = sentence_text[pos + len(p)]

        logger.debug(sentence_text)
        logger.debug("S: %s", s)
        logger.debug("P: %s", p)
        logger.debug("O: %s", o)
        logger.debug("Prev char %s", prev_char)
        logger.debug("Next char %s", next_char)
        logger.debug("")

    return s, o


def extract_spos_from_sentences(review_level, external=False, full_pipeline=False):
    #
    # stage 2: looking for SPOs with stanza NLP tool
    #

    logger.info("Stage 2 - external: %s (extract_spos_from_sentences): looking for SPOs with stanza NLP tool", external)

    if external:
        sentences_filter = f"""
            review_level = {review_level} AND
            skip_validations = true AND
            check_worthy_auto = 'FR' AND
            predicate = '' AND
            predicate_not_found = false
            ORDER BY item_id, new_sentence_id
            """
    else:
        if full_pipeline:
            sentences_filter = f"""
                review_level = {review_level} AND
                skip_validations = false AND
                check_worthy_score_auto >= 0.5 AND
                predicate = ''
                ORDER BY item_id, new_sentence_id
                """
        else:
            sentences_filter = f"""
                review_level = {review_level} AND
                skip_validations = false AND
                check_worthy = 'FR' AND
                predicate = ''
                ORDER BY item_id, new_sentence_id
                """

    connection = get_connection()

    sentences = select_fields_where(connection, TABLE, FIELDS, sentences_filter)

    if logger.getEffectiveLevel() == logging.DEBUG:
        sentences_set = sentences
    else:
        sentences_set = tqdm(sentences)

    for sentence in sentences_set:
        sentence_id = sentence[0]
        sentence_text = sentence[1]

        quoted_sentence = get_quoted_sentence(sentence_text)
        if quoted_sentence != "":
            spos = get_spos_from_document_dependency_tree(quoted_sentence)
            if len(spos) > 0:
                update_text_field(connection, TABLE, "id = " + str(sentence_id), "spo_type", "quoted text")
        else:
            spos = get_spos_from_document_dependency_tree(sentence_text)
            if len(spos) > 0:
                update_text_field(connection, TABLE, "id = " + str(sentence_id), "spo_type", "sentence")
        if len(spos) == 0:
            clauses = sentence_text.split(",")
            for clause in clauses:
                logger.debug(clause)
                spos = get_spos_from_clause_dependency_tree(clause)
                if len(spos) > 0:
                    update_text_field(connection, TABLE, "id = " + str(sentence_id), "spo_type", "clause")
                    break

        # TODO to decide if improves the system
        # if len(spos) == 0:
        #     spos = get_spo_from_sentence_constituency_tree(sentence_text)
        #     if len(spos) > 0:
        #         update_text_field(connection, TABLE, "id = " + str(sentence_id), "spo_type", "sentence_const")


        if len(spos) > 0:
            update_text_field(connection, TABLE, "id = " + str(sentence_id), "subject", spos[0][0])
            update_text_field(connection, TABLE, "id = " + str(sentence_id), "predicate", spos[0][1])
            update_text_field(connection, TABLE, "id = " + str(sentence_id), "object", spos[0][2])
            # in 'bigs' we manintain extracted subject and object because it can be a quoted text
            update_text_field(connection, TABLE, "id = " + str(sentence_id), "big_subject", spos[0][3])
            update_text_field(connection, TABLE, "id = " + str(sentence_id), "big_predicate", spos[0][4])
            update_text_field(connection, TABLE, "id = " + str(sentence_id), "big_object", spos[0][5])

    connection.commit()
    connection.close()


def get_quoted_sentence(document):
    result = re.findall(r'"(.*?)"', document)
    if len(result) > 0:
        return result[0]

    result = re.findall(r'“(.*?)”', document)
    if len(result) > 0:
        return result[0]

    return ""


def get_clause(w, w_list, w_plain_list):
    if w is None:
        return ""
    for left_w in w_list:
        if left_w.id < w.id and left_w.head == w.id:
            get_clause(left_w, w_list, w_plain_list)
    w_plain_list.append(w)
    for right_w in w_list:
        if right_w.id > w.id and right_w.head == w.id:
            get_clause(right_w, w_list, w_plain_list)


def get_predicate(w, w_list, ext_predicate):
    for left_w in w_list:
        if left_w.id < w.id and left_w.head == w.id and left_w.upos in ["AUX", "COP", "PART"]:
            ext_predicate.append(left_w)
    ext_predicate.append(w)
    for right_w in w_list:
        if right_w.id > w.id and right_w.head == w.id and right_w.upos in ["AUX", "COP", "PART"]:
            ext_predicate.append(right_w)


def print_word_list(word_list):
    w_text = []
    w_upos = []
    w_deprel = []
    for w in word_list:
        w_text.append(w.text)
        w_upos.append(w.upos)
        w_deprel.append(w.deprel)

    print(" ".join(w_text))
    print(" ".join(w_upos))
    print(" ".join(w_deprel))


def get_spos_from_document_dependency_tree(document, clean_doc=False):

    nlp = stanza.Pipeline(tokenize_no_ssplit=True, verbose=False, download_method=STANZA_DOWNLOAD_METHOD)

    if clean_doc:
        document = clean_document(document)

    doc = nlp(document)
    if DEBUG_NLP_DOC:
        with open("get_spos_from_document.json", "w", encoding="utf-8") as f:
            f.write(str(doc))

    result = []
    for sent in doc.sentences:

        s = None
        p = None
        o = None

        words = []
        for w in sent.words:
            if w.deprel == "root":
                p = w
            else:
                words.append(w)

        for w in words:
            if w.head == p.id and w.upos in ["NOUN", "PROPN"] and w.deprel in ["nsubj", "nsubj:pass"]:
                s = w

        if s is None:
            for w in words:
                if w.head == p.id and w.upos == "PRON" and w.xpos == "PRP" and w.deprel in ["nsubj", "nsubj:pass"]:
                    s = w

        for w in words:
            if w.head == p.id and w.upos in ["NOUN", "PROPN"] and w.deprel in ["obj", "obl"]:
                o = w

        if o is None:
            for w in words:
                if w.head == p.id and w.upos == "PRON" and w.xpos == "PRP" and w.deprel in ["obj", "obl"]:
                    o = w

        if o is None:
            for w in words:
                if w.head == p.id and w.upos == "ADJ" and w.deprel in ["xcomp", "ccomp"]:
                    o = w

        big_s = ""
        big_o = ""
        if s is not None and p is not None and o is not None:
            logger.debug(f"<S> {s.text} <P> {p.text} <O> {o.text}")

            ext_predicate = []
            get_predicate(p, words, ext_predicate)

            big_p = ""
            p_min_id = sys.maxsize
            p_max_id = 0
            last_position = -1
            for w in ext_predicate:
                if w.id < p_min_id:
                    p_min_id = w.id
                if w.id > p_max_id:
                    p_max_id = w.id
                if big_p == "":
                    big_p = w.text
                else:
                    if last_position == w.start_char:
                        big_p += w.text     # ca n't -> can't
                    else:
                        big_p += " " + w.text
                last_position = w.end_char

            if s.id < p_min_id and p_max_id < o.id:
                for w in words:
                    if w.id < p_min_id:
                        if big_s == "":
                            big_s = w.text
                        else:
                            big_s += " " + w.text
                    if w.id > p_max_id:
                        if big_o == "":
                            big_o = w.text
                        else:
                            big_o += " " + w.text
                logger.debug(f"<Big S> {big_s}  <Big P> {big_p} <Big O> {big_o}")

            result.append([s.text, p.text, o.text, big_s, big_p, big_o])

    return result


def get_spos_from_clause_dependency_tree(clause, clean_doc=False):

    nlp = stanza.Pipeline(tokenize_no_ssplit=True, verbose=False, download_method=STANZA_DOWNLOAD_METHOD)

    if clean_doc:
        clause = clean_document(clause)

    cl = nlp(clause)
    if DEBUG_NLP_DOC:
        with open("get_spos_from_document.json", "w") as f:
            f.write(str(cl))

    result = []
    for sent in cl.sentences:

        words = []
        s = None
        p = None
        o = None

        for w in sent.words:
            if w.deprel == "root" and w.upos in ["VERB", "AUX", "COP"]:
                p = w
            else:
                words.append(w)

        if p is None:
            words = []
            for w in sent.words:
                if w.deprel == "cop":
                    p = w
                else:
                    words.append(w)

        if p is None:
            words = []
            for w in sent.words:
                if w.upos in ["VERB"]:
                    p = w
                else:
                    words.append(w)

        if p is not None:
            for w in words:
                if w.head == p.id and w.upos in ["NOUN", "PROPN"] and w.deprel == "nsubj":
                    s = w
                if w.head == p.id and w.upos == "PRON" and w.xpos == "PRP" and w.deprel == "nsubj":
                    s = w
                if w.head == p.id and w.upos == "PRON" and w.feats == "PronType=Rel" and w.deprel == "nsubj":
                    s = w

            for w in words:
                if w.head == p.id and w.upos in ["NOUN", "PROPN"] and w.deprel == "obj":
                    o = w
                if w.head == p.id and w.upos == "PRON" and w.xpos == "PRP" and w.deprel == "obj":
                    o = w
                if w.head == p.id and w.upos == "PRON" and w.feats == "PronType=Rel" and w.deprel == "obj":
                    o = w

            if o is None:
                for w in words:
                    if w.head == p.id and w.upos in ["NOUN", "PROPN"] and w.deprel == "obl":
                        o = w

        big_s = ""
        big_o = ""
        if s is not None and p is not None and o is not None:
            logger.debug(f"CLAUSE>> <S> {s.text} <P> {p.text} <O> {o.text}")

            ext_predicate = []
            get_predicate(p, words, ext_predicate)

            big_p = ""
            p_min_id = sys.maxsize
            p_max_id = 0
            last_position = -1
            for w in ext_predicate:
                if w.id < p_min_id:
                    p_min_id = w.id
                if w.id > p_max_id:
                    p_max_id = w.id
                if big_p == "":
                    big_p = w.text
                else:
                    if last_position == w.start_char:
                        big_p += w.text     # ca n't -> can't
                    else:
                        big_p += " " + w.text
                last_position = w.end_char


            if s.id < p_min_id and p_max_id < o.id:
                for w in words:
                    if w.id < p_min_id:
                        if big_s == "":
                            big_s = w.text
                        else:
                            big_s += " " + w.text
                    if w.id > p_max_id:
                        if big_o == "":
                            big_o = w.text
                        else:
                            big_o += " " + w.text
                logger.debug(f"CLAUSE>> <Big S> {big_s}  <P> {p.text} <Big O> {big_o}")

            result.append([s.text, p.text, o.text, big_s, big_p, big_o])

    return result


def decontracted(document):
    document = re.sub(r"’", "'", document)
    document = re.sub(r"won\'t", "will not", document)
    document = re.sub(r"can\'t", "can not", document)
    document = re.sub(r"n\'t", " not", document)
    document = re.sub(r"\'re", " are", document)
    document = re.sub(r"\'s", " is", document)
    document = re.sub(r"\'d", " would", document)
    document = re.sub(r"\'ll", " will", document)
    document = re.sub(r"\'t", " not", document)
    document = re.sub(r"\'ve", " have", document)
    document = re.sub(r"\'m", " am", document)
    return document


def clean_document(document):
    document = decontracted(document)
    document = re.sub(r"\n+", ". ", document)
    document = re.sub("[^A-Za-z0-9?' .,-]+", " ", document)
    document = " ".join(document.split())
    logger.debug("Document after clean:")
    logger.debug(document)
    return document


def extract_spos_from_sentences_aux_predicate(review_level, external=False, full_pipeline=False):
    #
    # stage 3: looking for SPOs from be-have predicates
    #

    logger.info("Stage 3 - external: %s (extract_spos_from_sentences_aux_predicate): looking for SPOs from be-have predicates", external)

    if external:
        sentences_filter = f"""
            review_level = {review_level} AND
            skip_validations = true AND
            check_worthy_auto = 'FR' AND
            predicate = '' AND
            predicate_not_found = false
            ORDER BY item_id, new_sentence_id
            """
    else:
        if full_pipeline:
            sentences_filter = f"""
                review_level = {review_level} AND
                skip_validations = false AND
                check_worthy_score_auto >= 0.5 AND
                predicate = ''
                ORDER BY item_id, new_sentence_id
                """
        else:
            sentences_filter = f"""
                review_level = {review_level} AND
                skip_validations = false AND
                check_worthy = 'FR' AND
                predicate = ''
                ORDER BY item_id, new_sentence_id
                """

    predicates = build_aux_predicates()

    connection = get_connection()

    sentences = select_fields_where(connection, TABLE, FIELDS, sentences_filter)

    if logger.getEffectiveLevel() == logging.DEBUG:
        sentences_set = sentences
    else:
        sentences_set = tqdm(sentences)

    for sentence in sentences_set:
        sentence_id =  sentence[0]
        sentence_text = sentence[1]
        if len(sentence_text.split()) <= 40:
            for p in predicates:
                found_predicate = split_sentence_given_predicate_with_update(connection, sentence_id, sentence_text, p, "be-have")
                if found_predicate:
                    break

    connection.commit()
    connection.close()


def build_aux_predicates():

    predicates = []

    # can be
    for infinitive_modal_prefix in INFINITIVE_MODAL_PREFIXES:
        logger.debug(f"{infinitive_modal_prefix} be")
        predicates.append(f"{infinitive_modal_prefix} be")

    # is, are
    for present_participle_prefix in PRESENT_PARTICIPLE_PREFIXES:
        logger.debug(f"{present_participle_prefix}")
        predicates.append(f"{present_participle_prefix}")

    # have, has
    for past_participle_prefix in PAST_PARTICIPLE_PREFIXES:
        logger.debug(f"{past_participle_prefix}")
        predicates.append(f"{past_participle_prefix}")

    return predicates


def match_tokens_cuis_in_sentences(review_level, sentence_id=None, external=False, full_pipeline = False,
                                   compare_with_old_cuis_extraction=False):
    #
    # last stage: asign CUIs to SPOs
    #

    logger.info(f"Last stage  - external: {external} (match_tokens_cuis_in_sentences): asign CUIs to SPOs on review level {review_level}")

    if sentence_id is None:
        clear_where_clause = f"review_level = {review_level} AND skip_validations = {external}"
        if external:
            update_where_clause = f"""review_level = {review_level} AND
                                      skip_validations = true AND
                                      check_worthy_auto = 'FR' AND
                                      predicate_not_found = false
                                      ORDER BY item_id, new_sentence_id
                                      """
        else:
            if full_pipeline:
                update_where_clause = f"""review_level = {review_level} AND
                                            skip_validations = false AND
                                            check_worthy_score_auto >= 0.5
                                            ORDER BY item_id, new_sentence_id
                                            """
            else:
                update_where_clause = f"""review_level = {review_level} AND
                                            skip_validations = false AND
                                            check_worthy = 'FR'
                                            ORDER BY item_id, new_sentence_id
                                            """
    else:
        clear_where_clause = f"id = {sentence_id}"
        update_where_clause = f"id = {sentence_id} ORDER BY item_id, new_sentence_id"

    connection = get_connection()

    update_text_field(connection, TABLE, clear_where_clause, "subject_cuis", "")
    update_text_field(connection, TABLE, clear_where_clause, "predicate_cuis", "")
    update_text_field(connection, TABLE, clear_where_clause, "object_cuis", "")

    update_text_field(connection, TABLE, clear_where_clause, "subject_text_cuis", "")
    update_text_field(connection, TABLE, clear_where_clause, "predicate_text_cuis", "")
    update_text_field(connection, TABLE, clear_where_clause, "object_text_cuis", "")

    sentences = select_fields_where(connection, TABLE, FIELDS, update_where_clause)

    for sentence in tqdm(sentences):
        sentence_id2 = sentence[0]
        sentence_text = sentence[1]
        metamap_extraction = sentence[2]
        big_subject = sentence[3]
        predicate = sentence[4]
        big_object = sentence[5]

        if external and (big_subject == "" or predicate == "" or big_object == ""):
            continue

        s_ext, p_ext, o_ext = match_extraction_with_spo2(metamap_extraction, sentence_text,
                                                         big_subject, predicate, big_object)

        s_cuis2 = ""
        s_text_cuis2 = ""
        if s_ext is not None:
            s_cuis2 = get_tokens_str("CandidateCUI", s_ext)
            s_text_cuis2 = get_tokens_str("CandidatePreferred", s_ext)

        p_cuis2 = ""
        p_text_cuis2 = ""
        if p_ext is not None:
            p_cuis2 = get_tokens_str("CandidateCUI", p_ext)
            p_text_cuis2 = get_tokens_str("CandidatePreferred", p_ext)

        o_cuis2 = ""
        o_text_cuis2 = ""
        if o_ext is not None:
            o_cuis2 = get_tokens_str("CandidateCUI", o_ext)
            o_text_cuis2 = get_tokens_str("CandidatePreferred", o_ext)

        if compare_with_old_cuis_extraction:

            _, _, _, s_cuis, p_cuis, o_cuis = match_extraction_with_spo(metamap_extraction, big_subject, predicate, big_object)

            assert len(s_cuis.split()) == len(s_cuis2.split()), f"different subject cuis:\n  {s_cuis}\n  {s_cuis2}"
            assert len(p_cuis.split()) == len(p_cuis2.split()), f"different predicate cuis:\n  {p_cuis}\n  {p_cuis2}"
            assert len(o_cuis.split()) == len(o_cuis2.split()), f"different object cuis:\n  {o_cuis}\n  {o_cuis2}"

        # update_text_field(connection, TABLE, "id = " + str(sentence_id2), "subject_cuis", s_cuis)
        # update_text_field(connection, TABLE, "id = " + str(sentence_id2), "predicate_cuis", p_cuis)
        # update_text_field(connection, TABLE, "id = " + str(sentence_id2), "object_cuis", o_cuis)

        update_text_field(connection, TABLE, "id = " + str(sentence_id2), "subject_cuis", s_cuis2)
        update_text_field(connection, TABLE, "id = " + str(sentence_id2), "predicate_cuis", p_cuis2)
        update_text_field(connection, TABLE, "id = " + str(sentence_id2), "object_cuis", o_cuis2)

        update_text_field(connection, TABLE, "id = " + str(sentence_id2), "subject_text_cuis", s_text_cuis2)
        update_text_field(connection, TABLE, "id = " + str(sentence_id2), "predicate_text_cuis", p_text_cuis2)
        update_text_field(connection, TABLE, "id = " + str(sentence_id2), "object_text_cuis", o_text_cuis2)

    connection.commit()
    connection.close()


def extract_spos_from_sentences_given_manual_predicate(review_level, external=False, full_pipeline=False):
    #
    # stage 4: looking for SPOs from a manually entered predicate
    #

    logger.info(f"stage 4 - External: {external} (extract_spos_from_sentences_given_manual_predicate): looking for SPOs from a manually entered predicate")

    if external:
        sentences_filter = f"""
            review_level = {review_level} AND
            skip_validations = true AND
            check_worthy_auto = 'FR' AND
            predicate_not_found = false AND
            manually_identified_predicate <> big_predicate
            ORDER BY item_id, new_sentence_id
            """
    else:
        if full_pipeline:
            sentences_filter = f"""
                review_level = {review_level} AND
                skip_validations = false AND
                check_worthy_score_auto >= 0.5 AND
                predicate = ''
                ORDER BY item_id, new_sentence_id
                """
        else:
            sentences_filter = f"""
                review_level = {review_level} AND
                skip_validations = false AND
                check_worthy = 'FR' AND
                manually_identified_predicate <> big_predicate
                ORDER BY item_id, new_sentence_id
                """

    connection = get_connection()

    sentences = select_fields_where(connection, TABLE, FIELDS, sentences_filter)

    if logger.getEffectiveLevel() == logging.DEBUG:
        sentences_set = sentences
    else:
        sentences_set = tqdm(sentences)

    for sentence in sentences_set:
        sentence_id = sentence[0]
        sentence_text = sentence[1]
        manually_identified_predicate = sentence[6]
        split_sentence_given_predicate_with_update(connection, sentence_id, sentence_text, manually_identified_predicate, "manually id")

    connection.commit()
    connection.close()


def get_spo_from_sentence_constituency_tree(document, clean_doc=False, show_tags=False, nlp_object=None):

    if nlp_object is None:
        nlp_object = stanza.Pipeline(tokenize_no_ssplit=False, verbose=False, download_method=STANZA_DOWNLOAD_METHOD)

    if clean_doc:
        document = clean_document(document)

    nlp_doc = nlp_object(document)
    if DEBUG_NLP_DOC:
        with open("get_spos_from_document.json", "w", encoding="utf-8") as f:
            f.write(str(nlp_doc))

    result = []
    for nlp_sent in nlp_doc.sentences:

        logger.debug("")

        s, p, o, s_names_list, o_names_list = walk_constituency_tree(nlp_sent.constituency, 0, "", "", [], [], [], [], [], show_tags=show_tags)

        # flattening nested lists
        s_text = " ".join(sum(s, []))
        p_text = " ".join(sum(p, []))
        o_text = " ".join(sum(o, []))
        s_names = " ".join(sum(s_names_list, [])).lower()
        o_names = " ".join(sum(o_names_list, [])).lower()

        result.append([s_text, p_text, o_text, s_text, p_text, o_text, s_names_list, o_names_list])

        logger.info("")
        logger.info("Sentence: %s", document)

        if len(s) > 0 and len(p) > 0 and len(o) > 0 and len(s_names_list) > 0 and len(o_names_list) > 0:
            logger.info("")
            logger.info("Sentence OK! +++++")
        else:
            logger.info("")
            logger.info("Sentence KO! -----")

        logger.info("")
        logger.info("S: %s", s_text)
        logger.info("P: %s", p_text)
        logger.info("O: %s", o_text)
        logger.info("")
        logger.info("S names: %s", s_names)
        for name in s_names_list:
            name = name[0].lower()
            related_objects = get_related_objects(name, verbose=False)
            if len(related_objects) > 0:
                logger.debug("  %s:", name)
            for related_object in related_objects:
                logger.debug("    %s %s %s", related_object["predicate"], related_object["object"], related_object["weight"])
        logger.info("O names: %s", o_names)
        for name in o_names_list:
            name = name[0].lower()
            related_subjects = get_related_subjects(name, verbose=False)
            if len(related_subjects) > 0:
                logger.debug("  %s:", name)
            for related_subject in related_subjects:
                logger.debug("    %s %s %s", related_subject["subject"], related_subject["predicate"], related_subject["weight"])
        logger.info("")

    return result


def walk_constituency_tree(tree, level, parent_label, grandparent_label, s, p, o, s_names, o_names, show_tags=False):

    if level > 4:
        return

    if tree.label == "NP" and parent_label == "S":
        if show_tags:
            logger.debug("Subject N%s: %s",level, tree)
        else:
            logger.debug("Subject N%s: %s",level, tree.leaf_labels())
        s.append(tree.leaf_labels())
        s_names = walk_np_constituency_tree(tree, level, s_names)

    if parent_label == "VP" and grandparent_label == "S":
        if tree.label in ["VBP", "VBZ", "VB", "MD"]:
            if show_tags:
                logger.debug("Predicate N%s: %s",level, tree)
            else:
                logger.debug("Predicate N%s: %s",level, tree.leaf_labels())
            p.append(tree.leaf_labels())
        else:
            if tree.label not in ["VP"]:
                if show_tags:
                    logger.debug("Object N%s: %s",level, tree)
                else:
                    logger.debug("Object N%s: %s",level, tree.leaf_labels())
                o.append(tree.leaf_labels())
                o_names = walk_np_constituency_tree(tree, level, o_names)

    if parent_label == "VP" and grandparent_label == "VP":
        if tree.label in ["VBP", "VBZ", "VB"]:
            if show_tags:
                logger.debug("Predicate N%s: %s",level, tree)
            else:
                logger.debug("Predicate N%s: %s",level, tree.leaf_labels())
            p.append(tree.leaf_labels())
        else:
            if tree.label not in ["VP"]:
                if show_tags:
                    logger.debug("Object N%s: %s",level, tree)
                else:
                    logger.debug("Object N%s: %s",level, tree.leaf_labels())
                o.append(tree.leaf_labels())
                o_names = walk_np_constituency_tree(tree, level, o_names)

    for child in tree.children:
        walk_constituency_tree(child, level + 1, tree.label, parent_label, s, p, o, s_names, o_names, show_tags=show_tags)

    return s, p, o, s_names, o_names


def walk_np_constituency_tree(np_tree, level, names_list):

    if np_tree.label in ["NN", "NNS", "NNP", "JJ", "PRP", "EX"]:
        name = np_tree.leaf_labels()
        if np_tree.label in ["PRP", "EX"]: # tag "Personal pronoun" and "Existential" in subjects
            name[0] = name[0] + "#" + np_tree.label
        logger.debug("%s N%s: %s",np_tree.label, level, name)
        names_list.append(name)

    multiword_name_list = []
    multiword_names_list = []
    if np_tree.label == "NP":
        for child in np_tree.children:
            if child.label in ["NN", "NNS", "NNP", "JJ"]:
              multiword_name_list.append(child.leaf_labels())
            else:
                if len(multiword_name_list) > 1:
                    multiword_names_list.append(multiword_name_list)
                multiword_name_list = []

            if len(multiword_name_list) > 1 and multiword_name_list not in multiword_names_list:
                multiword_names_list.append(multiword_name_list)

        if len(multiword_names_list) > 0:
            for multiword_name_list in multiword_names_list:
                #  longest multiword
                logger.debug("%s N%s:%s", np_tree.label, level, multiword_name_list)
                multiword_name = "_".join(sum(multiword_name_list, []))
                names_list.append([f"{multiword_name}"])

                #  all consecutive multiword combinations
                for pos in range(len(multiword_name_list) - 1):
                    for width in range(2, len(multiword_name_list)):
                        if width + pos <=  len(multiword_name_list):
                            multiword_name_list2 = multiword_name_list[pos: width + pos]
                            logger.debug("%s N%s:%s", np_tree.label, level, multiword_name_list2)
                            multiword_name = "_".join(sum(multiword_name_list2, []))
                            names_list.append([f"{multiword_name}"])

    for child in np_tree.children:
        walk_np_constituency_tree(child, level + 1, names_list)

    return names_list
