import xml.etree.ElementTree as ET
import json
import csv
import copy
import logging

from difflib import SequenceMatcher

from agent.data.entities.config import ROOT_LOGGER_ID
from agent.data.sql.sql_mgmt import select_fields_where, update_text_field, update_non_text_field
from agent.data.entities.config import METAMAP_INPUT_PATH, UMLS_SEMANTIC_TYPES, UMLS_SEMANTIC_GROUPS, UNK_TOKEN

logger = logging.getLogger(ROOT_LOGGER_ID)


DEFAULT_SEM_GROUPS = ["DISO", "CHEM", "LIVB", "GENE", "ANAT", "DEVI", "PROC", "ORGA", "GEOG", "ACTI", "PHEN"]

def read_sem_types_to_dict():
    sem_types_dict = {}
    with open(UMLS_SEMANTIC_TYPES, encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='|')
        for row in csv_reader:
            sem_types_dict[row[0]] = row[1]
    return sem_types_dict


def read_sem_groups_to_dict():
    sem_groups_dict = {}
    with open(UMLS_SEMANTIC_GROUPS, encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='|')
        for row in csv_reader:
            sem_groups_dict[row[2]] = row[0]
    return sem_groups_dict


def read_sem_groups_to_list():
    sem_groups_list = []
    with open(UMLS_SEMANTIC_GROUPS, encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='|')
        for row in csv_reader:
            if row[0] not in sem_groups_list:
                sem_groups_list.append(row[0])
    return sem_groups_list

def parse_xml_file_first_elements(file_name, connection, table_name, sentence_id_field):
    sentence_id = file_name.split(".")[0]
    if connection is None:
        print(sentence_id)

    generate_json = False
    extracted_data = []
    extracted_xml_text = ""

    sem_types_dict = read_sem_types_to_dict()
    sem_groups_dict = read_sem_groups_to_dict()

    try:
        tree = ET.parse(METAMAP_INPUT_PATH + file_name)

        with open(METAMAP_INPUT_PATH + file_name, "r", encoding="utf-8") as f:
            lines_list = f.readlines()
            extracted_xml_text = " ".join(lines_list)

    except BaseException:
        logger.info('XML file not parsed: %s', file_name)
        tree = None

    if tree is not None:
        root = tree.getroot()
        for mmo in root:
            mmo_dict = {}
            utterances_list = []
            utterances = mmo.find("Utterances")
            for utterance in utterances:
                utterance_dict = {}
                phrases_list = []
                phrases = utterance.find("Phrases")
                for phrase in phrases:
                    phrase_text = phrase.find("PhraseText")
                    mappings = phrase.find("Mappings")
                    if mappings.attrib["Count"] != "0":
                        phrase_dict = {}
                        phrase_dict["PhraseText"] = phrase_text.text

                        syntax_units_list = []
                        syntax_units = phrase.find("SyntaxUnits")
                        if syntax_units.attrib["Count"] != "0":
                            for syntax_unit in syntax_units:
                                syntax_unit_element_dict = {}
                                for syntax_unit_element in syntax_unit:
                                    if syntax_unit_element.tag in ["SyntaxType", "LexMatch", "InputMatch", "LexCat"]:
                                        syntax_unit_element_dict[syntax_unit_element.tag] = syntax_unit_element.text
                                    tokens = ""
                                    if syntax_unit_element.tag == "Tokens":
                                        for token in syntax_unit_element:
                                            if tokens == "":
                                                tokens = token.text
                                            else:
                                                tokens = tokens + " " + token.text
                                        syntax_unit_element_dict[syntax_unit_element.tag] = tokens
                                syntax_units_list.append(syntax_unit_element_dict)

                        mapping = mappings.find("Mapping")
                        mapping_candidates = mapping.find("MappingCandidates")
                        candidates_list = []
                        for candidate in mapping_candidates:
                            candidate_dict = {}
                            candidate_matched = candidate.find("CandidateMatched")
                            candidate_dict["CandidateMatched"] = candidate_matched.text
                            candidate_preferred = candidate.find("CandidatePreferred")
                            candidate_dict["CandidatePreferred"] = candidate_preferred.text
                            candidate_cui = candidate.find("CandidateCUI")
                            candidate_dict["CandidateCUI"] = candidate_cui.text
                            candidate_negated = candidate.find("Negated")
                            candidate_dict["Negated"] = candidate_negated.text
                            sem_types = candidate.find("SemTypes")
                            if sem_types.attrib["Count"] != "0":
                                sem_type = sem_types.find("SemType")
                                candidate_dict["SemType"] = sem_type.text
                                candidate_dict["tui"] = sem_types_dict[candidate_dict["SemType"]]
                                candidate_dict["group"] = sem_groups_dict[candidate_dict["tui"]]
                                generate_json = True

                            tokens = ""
                            matched_words = candidate.find("MatchedWords")
                            for token in matched_words:
                                if tokens == "":
                                    tokens = token.text
                                else:
                                    tokens = tokens + " " + token.text
                            candidate_dict["MatchedWords"] = tokens

                            concept_pis = candidate.find("ConceptPIs")
                            concept_pis_list = []
                            if concept_pis.attrib["Count"] != "0":
                                for concept_pi in concept_pis:
                                    concept_pi_dict = {}
                                    concept_pi_start_pos = concept_pi.find("StartPos")
                                    concept_pi_dict["StartPos"] = concept_pi_start_pos.text
                                    concept_pi_lenght = concept_pi.find("Length")
                                    concept_pi_dict["Length"] = concept_pi_lenght.text
                                    concept_pis_list.append(concept_pi_dict)
                            candidate_dict["ConceptPIs"] = concept_pis_list

                            candidates_list.append(candidate_dict)
                        phrase_dict["SyntaxUnits"] = syntax_units_list
                        phrase_dict["MappingCandidates"] = candidates_list
                        phrases_list.append(phrase_dict)

                        extracted_data.append(phrase_dict)
                utterance_dict["Utterance"] = phrases_list
                utterances_list.append(utterance_dict)
            mmo_dict["Utterances"] = utterances_list
            # extracted_data.append(mmo_dict)

    if connection:
        update_text_field(connection, table_name, sentence_id_field + " = " + sentence_id,
                        "metamap_extraction_xml", extracted_xml_text)

    if generate_json:
        if connection is not None:
            update_text_field(connection, table_name, sentence_id_field + " = " + sentence_id,
                              "metamap_extraction", json.dumps(extracted_data))
        with open(METAMAP_INPUT_PATH + sentence_id + ".json", "w", encoding="utf-8") as write_file:
            json.dump(extracted_data, write_file, indent=4, separators=(",", ": "))
    else:
        if connection is not None:
            update_text_field(connection, table_name, sentence_id_field + " = " + sentence_id, "metamap_extraction", "")


def parse_and_update_annotate_sentences(connection, review_level=3, sentence_id=None):
    if sentence_id is None:
        sentence_filter = f"review_level = {review_level} ORDER BY item_id, new_sentence_id"
    else:
        sentence_filter = f"id = {sentence_id} ORDER BY item_id, new_sentence_id"

    annotate_sentences = select_fields_where(connection, "annotate_sentence", "id", sentence_filter)

    for annotate_sentence in annotate_sentences:
        sentence_id = annotate_sentence[0]
        file_name = str(sentence_id) + ".xml"
        parse_xml_file_first_elements(file_name, connection, "annotate_sentence", "id")


def sem_type_group_ok(sem_type, group):
    restricted_sem_types = ["T195", "T123", "T200", "T131", "T121", "T101", "T007", "T005", "T061", "T034"]
    restricted_sem_groups = ["DISO"]
    return sem_type in restricted_sem_types or group in restricted_sem_groups


def update_common_terms(connection, review_level=3):

    update_text_field(connection, "annotate_sentence", f"review_level = {review_level}", "common_health_terms", "")
    update_non_text_field(connection, "annotate_sentence", f"review_level = {review_level}", "health_terms_auto", "0")

    annotate_sentences = select_fields_where(connection, "annotate_sentence", "id, metamap_extraction",
                                             f"review_level = {review_level} ORDER BY item_id, new_sentence_id")
    for annotate_sentence in annotate_sentences:
        sentence_id = annotate_sentence[0]
        metamap_extraction = annotate_sentence[1]
        metamap_json = ""
        if metamap_extraction != "":
            metamap_json = json.loads(metamap_extraction)

        common_health_terms = ""
        common_health_terms_list = []
        mark_metamap_sem_type = 0

        for root in metamap_json:
            for mapping_candidate in root["MappingCandidates"]:
                if sem_type_group_ok(mapping_candidate["tui"], mapping_candidate["group"]):

                    new_term = mapping_candidate["group"] + " " + mapping_candidate["tui"]

                    if new_term not in common_health_terms_list:
                        common_health_terms_list.append(new_term)
                        if common_health_terms == "":
                            common_health_terms = new_term
                        else:
                            common_health_terms = common_health_terms + "\n" + new_term
                        mark_metamap_sem_type += 1

        update_text_field(connection, "annotate_sentence", "id = " + str(sentence_id),
                          "common_health_terms", common_health_terms)
        update_non_text_field(connection, "annotate_sentence", "id = " + str(sentence_id),
                              "health_terms_auto", mark_metamap_sem_type)


def match_extraction_with_spo(extraction, s, p, o):
    """
    Old method with approximate matches, some multiword cuis are ignored
    OBSOLETE
    """

    if extraction == "" or s == "" or p == "" or o == "":
        return None, None, None, "", "", ""

    s_tokens = s.split()
    p_tokens = p.split()
    o_tokens = o.split()

    metamap_json = json.loads(extraction)

    candidates = []
    for item in metamap_json:
        for candidate in item["MappingCandidates"]:
            candidates.append(candidate)
            logger.debug("Added mapping candidate: %s", candidate["MatchedWords"])

    ext_s, s_cuis = match_clause(s_tokens, candidates)
    ext_p, p_cuis = match_clause(p_tokens, candidates)
    ext_o, o_cuis = match_clause(o_tokens, candidates)

    return ext_s, ext_p, ext_o, " ".join(s_cuis), " ".join(p_cuis), " ".join(o_cuis)


def match_clause(tokens, candidates):
    """
    It tries to token-token-pair the textual sentence/clause with the CUIs extracted by MetaMap.
    When a CUI is multitoken, it is assigned to the first matching token of the sentence/clause.
    OBSOLETE
    """
    extended_tokens = []
    cuis = []

    token_id = 0
    pending_tokens = copy.deepcopy(tokens)
    for token in tokens:
        token_id += 1
        found_candidate = None
        for candidate in candidates:
            multi_token_cantidate = candidate["MatchedWords"].split()

            candidates_string = candidate["MatchedWords"].lower()
            tokens_string = " ".join(pending_tokens[: len(multi_token_cantidate)]).lower()
            logger.debug("Tokens: %s =? Candidates: %s", tokens_string, candidates_string)
            for candidate_token in multi_token_cantidate:
                if SequenceMatcher(None, candidates_string, tokens_string).ratio() > 0.8 and \
                   SequenceMatcher(None, token.lower(), candidate_token.lower()).ratio() > 0.8:
                    logger.debug("Token: %s =? Candidate: %s", token.lower(), candidate_token.lower())
                    if found_candidate is None:
                        found_candidate = candidates.pop(0)
                        for concept_pi in found_candidate["ConceptPIs"]:
                            logger.debug("%s %s %s", found_candidate["MatchedWords"].lower(), concept_pi["StartPos"], concept_pi["Length"])

                        logger.debug("Removed candidate found: %s", found_candidate["MatchedWords"])
                    while found_candidate["MatchedWords"] != candidate["MatchedWords"]:
                        found_candidate = candidates.pop(0)
                        logger.debug("Removed additional candidate: %s", found_candidate["MatchedWords"])
                    break
            if found_candidate is not None:
                break
        extended_token = {}
        extended_token["Token"] = token
        if found_candidate is not None:
            logger.debug("Token %s, insert CUI: %s %s", token, found_candidate["MatchedWords"], found_candidate["CandidateCUI"])
            extended_token["MatchedWords"] = found_candidate["MatchedWords"]
            extended_token["CandidateCUI"] = found_candidate["CandidateCUI"]
            extended_token["SemType"] = found_candidate["SemType"]
            extended_token["tui"] = found_candidate["tui"]
            extended_token["group"] = found_candidate["group"]
            extended_token["token_id"] = token_id
            extended_tokens.append(extended_token)
            cuis.append(found_candidate["CandidateCUI"])
        else:
            logger.debug("Token %s, insert CUI: %s", token, UNK_TOKEN)
            extended_token["MatchedWords"] = ""
            extended_token["CandidateCUI"] = UNK_TOKEN
            extended_token["SemType"] = UNK_TOKEN
            extended_token["tui"] = UNK_TOKEN
            extended_token["group"] = UNK_TOKEN
            extended_token["token_id"] = token_id
            extended_tokens.append(extended_token)
            cuis.append(UNK_TOKEN)

        pending_tokens.pop(0)

    return extended_tokens, cuis


def match_extraction_with_spo2(extraction, sentence, s, p, o):
    """
    New method, exact cuis match after retrieve additional
    data from Metamap xml file
    """

    if extraction == "" or s == "" or p == "" or o == "":
        return None, None, None

    metamap_json = json.loads(extraction)

    candidates_dict = {}
    candidates = []

    for item in metamap_json:
        for candidate in item["MappingCandidates"]:
            candidates.append(candidate)
            logger.debug("Added mapping candidate: %s", candidate["MatchedWords"])

            concept_pis = candidate["ConceptPIs"]
            for concept_pi in concept_pis:
                logger.debug("%s %s", concept_pi["StartPos"], concept_pi["Length"])
                start_pos = int(concept_pi["StartPos"])
                lenght = int(concept_pi["Length"])
                logger.debug(sentence[start_pos: start_pos + lenght])
                candidates_dict[start_pos] = candidate

    sentence_tokens = (s + " " + p + " " + o).split()
    predicate_token_start_pos = len(s.split())  # 0 = first token
    predicate_token_end_pos = predicate_token_start_pos + len(p.split()) - 1
    sentence_lenght = len(sentence)

    # logger.debug("%s",candidates_dict)

    logger.debug("sentence_tokens: %s", sentence_tokens)
    logger.debug("----------")
    logger.debug("predicate_token_start_pos: %s", predicate_token_start_pos)
    logger.debug("predicate_token_end_pos: %s", predicate_token_end_pos)
    logger.debug("sentence_lenght: %s", sentence_lenght)
    logger.debug("----------")

    logger.debug("Aligning Metamap and Stanza sentences...")

    token_pos = 0
    token_char_pos = 0
    sentence_char_position = 0
    extended_tokens = []

    for token in sentence_tokens:
        label = ""
        if token_pos < predicate_token_start_pos:
            label = "subject"
        if token_pos >= predicate_token_start_pos and token_pos <= predicate_token_end_pos:
            label = "predicate"
        if token_pos > predicate_token_end_pos:
            label = "object"

        logger.debug("%s %s %s %s", token_pos, token, ">>>", label)

        found_candidate = None
        for token_char_pos in range(len(token)):
            cui = ""
            candidate = candidates_dict.get(sentence_char_position)
            if candidate is not None:
                found_candidate = candidate
                cui = candidate["CandidateCUI"]

            logger.debug("  chars >>> %s %s %s %s", token[token_char_pos], sentence[sentence_char_position], sentence_char_position, cui)

            if sentence[sentence_char_position] == token[token_char_pos]:
                if sentence_char_position < sentence_lenght -1:
                    sentence_char_position += 1

            if sentence[sentence_char_position] == " ":
                if sentence_char_position < sentence_lenght -1:
                    sentence_char_position += 1

        extended_token = {}
        extended_token["token_id"] = token_pos
        extended_token["token_text"] = token
        if found_candidate is None:
            extended_token["CandidateMatched"] = UNK_TOKEN,
            extended_token["CandidatePreferred"] = UNK_TOKEN
            extended_token["CandidateCUI"] = UNK_TOKEN
            extended_token["Negated"] = UNK_TOKEN
            extended_token["SemType"] = UNK_TOKEN
            extended_token["tui"] = UNK_TOKEN
            extended_token["group"] = UNK_TOKEN
            extended_token["MatchedWords"] = UNK_TOKEN,
        else:
            extended_token["CandidateMatched"] = found_candidate["CandidateMatched"]
            extended_token["CandidatePreferred"] = found_candidate["CandidatePreferred"]
            extended_token["CandidateCUI"] = found_candidate["CandidateCUI"]
            extended_token["Negated"] = found_candidate["Negated"]
            extended_token["SemType"] = found_candidate["SemType"]
            extended_token["tui"] = found_candidate["tui"]
            extended_token["group"] = found_candidate["group"]
            extended_token["MatchedWords"] = found_candidate["MatchedWords"]

        extended_tokens.append(extended_token)

        token_pos += 1

    subject_ext_tokens = extended_tokens[:predicate_token_start_pos]
    predicate_ext_tokens = extended_tokens[predicate_token_start_pos: predicate_token_end_pos + 1]
    object_ext_tokens = extended_tokens[predicate_token_end_pos + 1:]

    logger.debug("----------")
    logger.debug("subject:")
    for token in subject_ext_tokens:
        print_token(token)
    logger.debug("predicate:")
    for token in predicate_ext_tokens:
        print_token(token)

    logger.debug("object:")
    for token in object_ext_tokens:
        print_token(token)
    logger.debug("----------")



    # logger.debug(get_tokens_str("CandidateCUI", subject_ext_tokens))
    # logger.debug(get_tokens_str("CandidateCUI", predicate_ext_tokens))
    # logger.debug(get_tokens_str("CandidateCUI", object_ext_tokens))

    # logger.debug(get_tokens_str("CandidatePreferred", subject_ext_tokens))
    # logger.debug(get_tokens_str("CandidatePreferred", predicate_ext_tokens))
    # logger.debug(get_tokens_str("CandidatePreferred", object_ext_tokens))

    return subject_ext_tokens, predicate_ext_tokens, object_ext_tokens


def get_tokens_str(key, token_list):
    return " ".join(token[key] for token in token_list)


def print_token(token):
    if token['CandidateCUI'] == UNK_TOKEN:
        logger.debug("    %s %s", token["token_id"], token["token_text"])
    else:
        logger.debug("    %s %s >>> %s %s %s %s %s %s", token["token_id"], token["token_text"],
            token["SemType"], token['group'], token['tui'],
            token['CandidateCUI'], token['MatchedWords'], f"[{token['CandidatePreferred']}]")
