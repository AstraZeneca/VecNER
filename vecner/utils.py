"""
A set of tools used throughout this work
"""
from typing import List as _List
from typing import Dict as _Dict
from typing import Any as _Any
from typing import Tuple as _Tuple
from string import ascii_letters as _ascii_letters
import numpy as _np


def flatten_lexicon(lexicon: _Dict[str, _List]) -> _List[_Tuple[str, _Any]]:
    """

    Args:
        lexicon (Dict[AnyStr, List]): the custom made lexicon in the form of -> {
                                                                                    'name'  :[
                                                                                        'term1',
                                                                                        'term2',
                                                                                    ]
                                                                                    ,...
                                                                                }
    Returns:
        List[Tuple[str, str]]: the output in the form of -> [
                                                                ('name', 'term1'),
                                                                ('name', 'term2')
                                                            ]
    """

    adjusted_lexicon = []

    for name, entities in lexicon.items():

        adjusted_lexicon.extend([(name, x) for x in entities])

    return adjusted_lexicon


def phrase_representer(phrase: str, model: object) -> _np.ndarray:

    """
    Computes the average word representations from a series of words
    Args:
        phrase               : str -> the sequence of words in string format i.e. 'the cat ate'
        model           : pre-trained w2vec gensim model
    Returns:
        seq_rep         : np.array -> a numpy array averaged representation of all words in the sequence
                                        (those found actually)
    """

    seq_rep = _np.zeros(model.vector_size)

    for j, word in enumerate(phrase.split()):

        ## if the word is not found in the model add a zero vector
        try:

            seq_rep += model[word]

        except KeyError:

            seq_rep += _np.zeros(model.vector_size) + 1e-20

        if j > 0:

            seq_rep /= j

    return seq_rep


def lexicon_extender(
    lexicon: _Dict[str, _List], model: object, sensitivity: int = 10
) -> _Dict[str, _Dict[str, float]]:
    """
    Expands the lexicon from : Group : List[terms] to ->
                                Group : Dict[str, float]
    such that:
        * the Group represents a driver group (i.e. safety)
        * the Dict contains extended terms (str) and their cosine similarity to primary terms
          in the original lexicon
    example --> 'time' : {'time' : 1.00, 'month' : 0.95, 'year': 0.96, ..}
    Args:
        lexicon (Dict[str, List]): the lexicon in the original format
        model (Callable): gensim pretrained embedding model

    Returns:
        Dict[str, Dict]: the exteneded lexicon
    """

    ## placeholder
    new_lexicon: _Dict[_Any, _Dict[_Any, float]] = {}

    for group in lexicon:

        terms = lexicon[group]

        new_lexicon[group] = {}

        ## loop over each term under each name
        for term in terms:

            ## add the parent term as perfect match
            new_lexicon[group][term] = 1.0

            ## compute the vector representation of the phrase
            combi = phrase_representer(term, model)

            ## find topn similar (sensitivity) words
            topn = model.similar_by_vector(combi, sensitivity)

            ## now for each of those topn words
            for word in topn:

                ## if the word is in the lexicon already
                ## and its similarity is lower continue
                if (word[0] in new_lexicon[group]) and (
                    new_lexicon[group][word[0]] > word[1]
                ):
                    continue

                ## also if its below a similarity threshold (manually checked)
                elif (word[0].isdecimal()) or (word[1] < 0.3):

                    continue

                ## if the word is simply 1 character then we don't need it
                elif (len(word[0]) == 1) and (set(word[0]).difference(_ascii_letters)):

                    continue

                ## if all these pass then add it to the new_lexicon
                else:

                    new_lexicon[group][word[0]] = word[1]

    return new_lexicon


def merge_on_overlap(doc: object, drivers: _List[_Dict]) -> _List[_Dict]:

    """
    Merges and sorts entities (i.e. drivers for this usecase).
    Merge is done as several entities are overlapping (i.e. the same entities were collected
    multiple times) and they are not in order. Also some overlap within others. This is a naive
    implementation which can be improved.
    input:
        drivers                 : list[dict] -> list of collected drivers
    Returns:
        new_ents                : list[dict] -> list of sorted and merged drivers
    """

    ## sort by start position the drivers
    drivers = sorted(drivers, key=lambda x: x["start"])

    new_ents: _List[_Dict[_Any, _Any]] = []

    ## for each driver
    for jj, ent in enumerate(drivers):

        if ent["text"].isdigit():

            continue

        ## if its the first entity in the list add it else
        if (jj > 0) and (len(new_ents) > 0):

            ## check if there is an overlap with the previous one and simply change that
            ## rather than adding a new one
            cur_ent_range = set(_np.arange(ent["start"], ent["end"]))
            prev_ent_range = set(_np.arange(new_ents[-1]["start"], new_ents[-1]["end"]))

            ## check if there is an overlap with the previous one and simply change that
            ## rather than adding a new one
            if len(cur_ent_range.intersection(prev_ent_range)) > 0:

                if ent["text"] not in new_ents[-1]["text"]:

                    new_ents[-1]["text"] += " " + ent["text"]

                local_text = doc.text[new_ents[-1]["start"] : ent["end"]]

                new_ents[-1]["text"] = local_text
                new_ents[-1]["end"] = ent["end"]

            else:

                new_ents.append(ent)

        else:

            new_ents.append(ent)

    return new_ents

def cosine_similarity(a : _np.array, b: _np.array) -> float:
    """
    Computes the cosine similarity between two numpy vectors
    Args:
        a (_np.array): vector of size N
        b (_np.array): vector of size N

    Returns:
        float: the cosine similarity
    """

    similarity_score =  _np.dot(a,b) / (_np.linalg.norm(a) * _np.linalg.norm(b))

    return similarity_score