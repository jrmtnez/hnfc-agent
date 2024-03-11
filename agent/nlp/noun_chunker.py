import logging
import nltk
import copy

from nltk.tokenize import word_tokenize
from nltk import word_tokenize, pos_tag, ParentedTree

from agent.data.entities.config import ROOT_LOGGER_ID

logger = logging.getLogger(ROOT_LOGGER_ID)

NN_NN_RULE = r"NP: {<NN.?>+<NN.?>}"
JJ_NN_RULE = r"NP: {<JJ>+<NN>}"
DT_JJ_NN_RULE = r"NP: {<DT>?<JJ.*>*<NN.*>+}"
DT_PP_JJ_NN_RULE = r"NP: {<DT|PP\$>?<JJ.*>*<NN.*>+}"
DT_PP_JJ_NN_RULE = r"""
    NP: {<DT|PP\$>?<JJ.*>*<NN.*>+<IN><DT|PP\$>?<JJ.*>*<NN.*>+<IN><DT|PP\$>?<JJ.*>*<NN.*>+}
        {<DT|PP\$>?<JJ.*>*<NN.*>+<IN><DT|PP\$>?<JJ.*>*<NN.*>+}
        {<DT|PP\$>?<JJ.*>*<NN.*>+}
"""

# the brain ’ s chemistry
DT_PP_JJ_NN_RULE = r"""
    NP: {<DT|PP\$>?<JJ.*>*<NN.*>*<JJ.*>*<NN.*>+<IN><DT|PP\$>?<JJ.*>*<NN.*>*<JJ.*>*<NN.*>+<IN><DT|PP\$>?<JJ.*>*<NN.*>*<JJ.*>*<NN.*>+}
        {<DT|PP\$>?<JJ.*>*<NN.*>*<JJ.*>*<NN.*>+<IN><DT|PP\$>?<JJ.*>*<NN.*>*<JJ.*>*<NN.*>+}
        {<DT|PP\$>?<JJ.*>*<NN.*>*<JJ.*>*<NN.*>+}
"""



def match_tokens_in_noun_chunks(tagged_sentence, tagged_chunks):
    # build new sentence with tokens replaced by noun chunks when it is possible

    pending_tokens = copy.deepcopy(tagged_sentence)

    chunked_sentence = []

    index = 0
    last_position = 0
    for chunk in tagged_chunks:
        for chunk_token in chunk[0]:
            while len(pending_tokens) > 0 and pending_tokens[0] != chunk_token:
                if index >= last_position:
                    chunked_sentence.append([[pending_tokens[0]], False, index, index])
                index += 1
                pending_tokens.pop(0)

            chunk_size = len(chunk[0])
            chunk.append(True)
            chunk.append(index)
            chunk.append(index + chunk_size - 1)

            last_position = index + chunk_size
            chunked_sentence.append(chunk)
            break

    # tokens after last noun chunk
    while len(pending_tokens) > 0:
        if index >= last_position:
            chunked_sentence.append([[pending_tokens[0]], False, index, index])
        index += 1
        pending_tokens.pop(0)

    logger.debug("")
    logger.debug("Chunked sentence  [[('chunk token 1', 'chunk token 1 POS'), ('chunk token 2', 'chunk token 2 POS'), ...], is a chunk name, start id, end id]:")
    for index, token in enumerate(chunked_sentence):
        logger.debug("%3s, %s", index, token)

    return tagged_chunks, chunked_sentence


def regex_chunker(sentence, expression):
    tagged_sentence = pos_tag(word_tokenize(sentence))
    chunk_parser = nltk.RegexpParser(expression)
    initial_chunked_sentence = chunk_parser.parse(tagged_sentence)

    tagged_chunks = []
    plain_chunks = []
    for chunk in initial_chunked_sentence.subtrees(filter=lambda t: t.label() == 'NP'):
        tagged_chunks.append([chunk.leaves()])
        plain_chunks.append(" ".join([w for w, t in chunk.leaves()]))

    tagged_chunks, chunked_sentence = match_tokens_in_noun_chunks(tagged_sentence, tagged_chunks)

    # logger.debug("")
    # logger.debug("Noun chunks:")
    # for tagged_chunk in tagged_chunks:
    #     logger.debug(tagged_chunk)

    noun_chunk_ids = []
    for index, token in enumerate(chunked_sentence):
        if token[1]:
            noun_chunk_ids.append(index)

    return chunked_sentence, tagged_chunks, noun_chunk_ids
