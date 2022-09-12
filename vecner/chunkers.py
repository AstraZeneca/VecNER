from typing import List as _List
from typing import Dict as _Dict
from typing import Any as _Any
from typing import Set as _Set
import numpy as _np
from .utils import merge_on_overlap as _merge_on_overlap


def noun_chunker(
    doc: object, ents: _List[_Dict[str, _Any]], ids: _Dict[_Any, _Any] = None
) -> _List[_Dict]:
    """
    Entity chunking based on spacy's noun_chunks. This function does not depend only
    on the noun-chunks extracted by spacy, but has a condition that the noun chunks
    MUST have a part or be a driver previously identified.
    Args:
        text                    : spacy.doc object -> the text passed through spacy's nlp
        drivers                 : list[dict] -> list of collected drivers
    Returns:
        new_ents                : list[dict] -> list of sorted and merged drivers
    """

    del ids

    ## default if no drivers
    if len(ents) == 0:
        return []

    # ## set some boundaries on the first driver found
    # min_, max_ = ents[0]['start'], ents[-1]['end']

    new_ents = {}

    for entity in ents:

        ent_s = entity["start"]
        ent_e = entity["end"]

        ent_r = set(_np.arange(ent_s, ent_e))

        found = 0

        ## for all chunks in the document
        for chunk in doc.noun_chunks:

            ## find the chunk start or end
            chunk_s = chunk.start_char
            chunk_e = chunk.end_char

            chunk_r = set(_np.arange(chunk_s, chunk_e))

            # if (chunk_s < min_) or (chunk_s > max_):

            #     continue

            ## now if there is a union (i.e. our driver and noun chunk) overlap
            ## then it means that the driver is part of the chunk or vice versa
            if len(chunk_r.intersection(ent_r)) > 0:

                found = 1

                new_ents[chunk_s] = {
                    "start": chunk_s,
                    "end": chunk_e,
                    "text": chunk.text,
                    "label": entity["label"],
                    "idx": entity["idx"],
                }

        if found == 0:

            new_ents[ent_s] = entity

    return list(new_ents.values())


def edge_chunker(
    doc: object, ids: _Dict[int, str], ents: _List[_Dict[str, _Any]] = None
) -> _List[_Dict]:
    """
    Entity chunking based on identifying critical tokens that form part of an entity
    through a dependency graph
    Args:
        text                    : spacy.doc object -> the text passed through spacy's nlp
        drivers                 : list[dict] -> list of collected drivers
    Returns:
        new_ents                : list[dict] -> list of sorted and merged drivers
    """

    del ents

    new_ents = []
    extended_ids: _Set[int] = set()

    for tok in doc:

        if tok.i in extended_ids:

            continue

        if tok.i in ids:

            if tok.left_edge == tok.right_edge:

                local_ids = [tok.head.i, tok.i]
                local_text = doc[min(local_ids) : max(local_ids) + 1]

                new_ents.append(
                    {
                        "text": local_text.text,
                        "idx": min(local_ids),
                        "label": ids[tok.i],
                        "start": local_text.start_char,
                        "end": local_text.end_char,
                    }
                )

            else:

                local_ids = [tok.left_edge.i, tok.i, tok.right_edge.i]
                local_text = doc[min(local_ids) : max(local_ids) + 1]

                new_ents.append(
                    {
                        "text": local_text.text,
                        "idx": min(local_ids),
                        "label": ids[tok.i],
                        "start": local_text.start_char,
                        "end": local_text.end_char,
                    }
                )

    return _merge_on_overlap(doc, new_ents)
