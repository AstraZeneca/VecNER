"""
** EXAMPLE **
An example of a custom rule chunker. To try out, rename to custom_file.py
"""
from typing import List, Dict, Any
from vecner.utils import merge_on_overlap

def rule_chunker(
                    doc     : object,
                    ents    : List[Dict[str, any]],
                    ids     : Any = None
                ) -> List[Dict[str, any]]:
    """
    Identifies adjacent entities based on rules described below
    and merges them based on their overlap

    Args:
        sentence (str): the text sequence
        found_entities (List[dict]): previously matched entities with the lexicon at word-level
        excluded_terms (List[str]): list of terms that you would like to exclude
        additional_terms (int): additional terms to add to the excluded terms
        model (Callable): gensim based model

    Returns:
        list (List[Dict[str, any]]): same as found_entities
    """

    ## restructuring for easier managind and id'ing
    restruct = {x['idx']:x for x in ents}
    ids = {x['idx'] : x['label'] for x in ents}

    for tok in doc:

        ## if token is in the previously found entities
        if tok.i in ids:

            ## this part is for linking children entities to the entities to form one entity
            for child in tok.children:

                ## IF both the child and tokens are entities.
                ## IF the child is a compound, det or a mod AND when the token is in front of the child then add
                ## i.e. stage : tok -> II : child
                if (child.i in ids)and((child.dep_ in ('compound', 'det'))or('mod' in child.dep_)):

                    if (tok.i > child.i):

                        restruct[child.i] = {
                            'start' : child.idx,
                            'end'   : tok.idx + len(tok.text),
                            'text'  : doc[child.i: tok.i + 1].text,
                            'idx'   : child.i,
                            'label' : ids[tok.i]
                        }

                        try:

                            del restruct[tok.i]

                        except KeyError:

                            pass

                    ## else if the child is in front
                    elif (tok.i < child.i)and(child.dep_ in ('nummod')):

                        restruct[tok.i] = {
                            'start' : tok.idx,
                            'end'   : child.idx + len(child.text),
                            'text'  : doc[tok.i: child.i + 1].text,
                            'idx'   : tok.i,
                            'label' : ids[tok.i]
                        }

                        try:

                            del restruct[child.i]

                        except KeyError:

                            pass

    return merge_on_overlap(
        doc,
        list(restruct.values())
    )


