"""
Contains all the matchers used to perform lexical based NER
"""
from typing import List as _List  # all _* are hidden from module, no need to be visible
from typing import Dict as _Dict
from typing import Any as _Any
from typing import Union as _Union
import importlib.util as _libutil
import sys as _sys
import os as _os
import logging as _logging
import spacy
from nltk import word_tokenize
from nltk.corpus import stopwords
from spacy.matcher import PhraseMatcher
from scipy import spatial
from .utils import phrase_representer, lexicon_extender
from .chunkers import noun_chunker, edge_chunker

class ExactMatcher:
    """
    Maps the lexicon to a spacy PhraseMatcher
    for exact matching.
    Args:
        lexicon         : Dict[str, List[str]] -> e.g. {
                                                    'time' : [
                                                        'year',
                                                        'month'
                                                    ],
                                                    ..
                                                }
        spacy_model     : str -> defaults to 'en_core_web_sm'
    """

    def __init__(
        self, lexicon: _Dict[str, _List[str]], spacy_model: str = "en_core_web_sm"
    ):

        self.lexicon = lexicon
        self.spacy_model = spacy_model

        ## load spacy model and matcher
        self.nlp = spacy.load(spacy_model)
        self.matcher = PhraseMatcher(self.nlp.vocab)

        ## assert lexicon is not empty
        ## empty dicts by default return False
        assert bool(
            lexicon
        ), """
            lexicon arg is empty
            """

        ## add items to matcher
        for group, items in lexicon.items():

            self.matcher.add(group, [self.nlp(x) for x in items])

        _logging.info("** mapped lexicon to matcher **")

    def map(
        self, text: str, tokenize: bool = True, lowecase: bool = True
    ) -> _Dict[str, _Any]:
        """
        maps the matcher recorded lexicon items to the string
        Args:
            text (str): the input string
            tokenize (bool, optional): whether to apply nltk tokenization. Defaults to True.
            lowecase (bool, optional): whether to lowercase the text for better matching. Defaults to True.

        Returns:
            List[Dict[str, any]]: a dictionary of entity information
        """

        ## to maintain upper case and lowercase
        inpt = text
        ## tokenize the text
        if tokenize:
            inpt = " ".join(word_tokenize(inpt))
        ## lower case the text
        if lowecase:
            inpt = inpt.lower()

        ## perform matching
        doc = self.nlp(inpt)
        matches = self.matcher(doc)

        ents: _List[_Dict[_Any, _Any]] = []
        words_in_ents: _Dict[int, str] = {}

        ## if no matches then return empty list
        if len(matches) == 0:

            return {
                "doc": self.nlp(" ".join(word_tokenize(text))),
                "ents": ents,
                "ids": words_in_ents,
            }

        ## if not then record all matches
        for match in matches:

            match_id, start_i, end_i = match

            matched_phrase = doc[start_i:end_i]
            label = self.matcher.vocab.strings[match_id]

            for entid in range(start_i, end_i):

                words_in_ents[entid] = label

            ents.append(
                {
                    "text": matched_phrase.text,
                    "start": matched_phrase.start_char,
                    "end": matched_phrase.end_char,
                    "label": label,
                    "idx": start_i,
                }
            )

        ## sort them
        ents = list(sorted(ents, key=lambda x: x["idx"]))

        return {
            "doc": self.nlp(" ".join(word_tokenize(text))),
            "ents": ents,
            "ids": words_in_ents,
        }


class ThresholdMatcher:
    """
    Maps lexicon terms and names to document terms
    by using a w2vec model to find those which are similar
    in the lexicon and sequence. Similarity is measured by cosine similarity
    where a threshold determines what is added and what not
    Args:
        lexicon         : Dict[str, List[str]] -> e.g. {
                                                    'time' : [
                                                        'year',
                                                        'month'
                                                    ],
                                                    ..
                                                }
        w2vec_model     : object -> the w2vec model from gensim KeyedVectors
        threshold       : float -> the similarity threshold value (cosine)
        exclude_pos     : list -> whether to exclude any words with POS tags
        exclude_dep     : list -> whether to exclude particular dependency words
        in_pipeline     : bool -> whether this follows the ExactMatcher output
        remove_stopwords: bool -> whether to remove nltk.corpus.stopwords from the analysis
        spacy_model     : str -> defaults to 'en_core_web_sm'
        chunking_method : str -> as entities identified are tokens, this class offers also
                                 chunking methods
    """

    def __init__(
        self,
        lexicon: _Dict[str, _List],
        w2vec_model: object,
        threshold: float = 0.6,
        exclude_pos: _List[str] = ["PUNCT"],
        exclude_dep: _List[str] = None,
        in_pipeline: bool = True,
        remove_stopwords: bool = True,
        spacy_model: str = "en_core_web_sm",
        chunking_method: str = "noun_chunking",
    ):

        assert (threshold > 0.0) and (
            threshold < 1.0
        ),  """
            Threshold must be between 0 and 1
            """

        if exclude_dep is None:
            exclude_dep = []

        self.lexicon = lexicon
        self.w2vec_model = w2vec_model
        self.threshold = threshold
        self.exclude_pos = exclude_pos
        self.exclude_dep = exclude_dep
        self.in_pipeline = in_pipeline
        self.remove_stopwords = remove_stopwords
        self.stoplist = stopwords.words("english")
        self.chunking_method = chunking_method
        self.nlp = spacy.load(spacy_model)

        ## assert lexicon is not empty
        ## empty dicts by default return False
        assert bool(
            lexicon
        ), """
                                lexicon arg is empty
                                """

        assert self.chunking_method in (
            "noun_chunking",
            "custom_chunking",
            "edge_chunking",
        ),  """
            Chunker must be one of
                * noun_chunking
                * custom_chunking
                * edge_chunking
            For rule-based also define
            rules (see variable rules).
            """

        if self.chunking_method == "noun_chunking":

            self.chunker = noun_chunker

        elif self.chunking_method == "custom_chunking":

            cwd = _os.getcwd()
            file_path = f"{cwd}/custom_file.py"

            spec = _libutil.spec_from_file_location("custom_rules", file_path)
            custom_module = _libutil.module_from_spec(spec)
            _sys.modules["custom_rules"] = custom_module
            spec.loader.exec_module(custom_module)

            self.chunker = custom_module.rule_chunker

        else:

            self.chunker = edge_chunker

    def map(
        self,
        document: _Union[str, object],
        ents: _List[_Dict[str, _Any]] = None,
        ids: _Dict[int, str] = None,
    ) -> _Dict[str, _Any]:
        """
        maps the matcher recorded lexicon items to the string using a fuzzy matching method
        by computing the cosine similarity of a word in a lexicon with the word in the sentence
        then chunks matched terms using the user-specified chunking method
        Args:
            document    : Union
                str         -> the input document as string
                object      -> the input document as a spacy.doc
            ents    : List[Dict[str, Any]]  -> if part of a pipeline the pre-computed entity list
            ids     : Dict[str, Any]        -> if part of a pipeline the pre-computed word indexes (in sequence)
                                               and associated labels
        Returns:
            _List[Dict[str, any]]
        """

        if ents is None:
            ents = []
        if ids is None:
            ids = {}

        if self.in_pipeline:

            assert (
                isinstance(document, str) is False
            ), """
                                                        Must be in spacy.tokens.doc.Doc format
                                                        """

            ## copying to preserve case-sensitive aspect of the text
            ## however the matching is always done on lower-case characters
            output = document.text
            document = self.nlp(document.text.lower())

        else:

            assert isinstance(
                document, str
            ), """
                                                Document must be in str format, except if running in a pipeline.
                                                Then use in_pipeline = True
                                                """
            ## if first pass then tokenize and then pass in spacy nlp
            document = " ".join(word_tokenize(document))
            output = document
            document = self.nlp(document.lower())

        single_terms = self.__find_single_terms(document=document, ents=ents, ids=ids)

        chunked_entities = self.chunker(
            doc=document, ents=single_terms["ents"], ids=single_terms["ids"]
        )

        return {
            "doc": self.nlp(output),
            "ents": chunked_entities,
            "ids": {x["idx"]: x["label"] for x in chunked_entities},
        }

    def __find_single_terms(
        self,
        document: _Union[str, object],
        ents: _List[_Dict[str, _Any]] = None,
        ids: _Dict[int, str] = None,
    ) -> _Dict[str, _Any]:
        """
        matches single terms with the pre-specified lexicon by using a user-specified w2vec model
        and then computing the similarity between the term in the lexicon AND each term in the sequence
        Args:
            document (_Union[str, object]):
                str         -> the input document as string
                object      -> the input document as a spacy.doc
            ents (_List[_Dict[str, _Any]], optional):   if part of a pipeline the pre-computed entity list.
                                                        Defaults to None.
            ids (_Dict[int, str], optional):    if part of a pipeline the pre-computed word indexes (in sequence)
                                                and associated labels.
                                                Defaults to None.

        Returns:
            _Dict[str, _Any]
        """

        if ents is None:
            ents = []
        if ids is None:
            ids = {}

        ## loop over the document
        for entid, word in enumerate(document):

            if (self.remove_stopwords) and (word.text in self.stoplist):

                continue

            if (word.pos_ in self.exclude_pos) or (word.dep_ in self.exclude_dep):

                continue

            if entid in ids:

                continue

            ## might match with one or more groups
            ## choose the one with the lowest distance
            distance_checks = {}

            for name, pre_specified_entities in self.lexicon.items():

                for entity in pre_specified_entities:

                    term_representation = phrase_representer(
                        entity, model=self.w2vec_model
                    )

                    if word.text not in self.w2vec_model:

                        continue

                    ## compute cosine similarity (1 - distance)
                    similarity = 1 - spatial.distance.cosine(
                        term_representation, self.w2vec_model[word.text]
                    )

                    ## if our cosine similarity is under the threshold
                    ## and the collective phrase vector is not full of zeros
                    ## add to our collection
                    if (similarity > self.threshold) and (
                        term_representation.sum() > 0.0
                    ):

                        distance_checks[name] = similarity

                        if entid in ids:

                            continue

                        else:

                            ids[entid] = name

                            ents.append(
                                {
                                    "start": word.idx,
                                    "end": word.idx + len(word.text),
                                    "text": word.text,
                                    "label": name,
                                    "idx": word.i,
                                }
                            )

            ## as one word can match across different categories
            ## its best if we select the category (name) with the highest
            ## cosine similarity
            if len(distance_checks) > 1:

                ents[-1]["label"] = max(distance_checks, key=distance_checks.get)

        ## sort them
        ents = list(sorted(ents, key=lambda x: x["idx"]))

        return {
            "doc": document,
            "ents": ents,
            "ids": ids,
        }


class ExtendedMatcher:
    """
    maps lexicon terms and names to document terms
    by using a w2vec model first expand the lexicon with simialr terms
    and then find those in the sequence
    Args:
        lexicon         : Dict[str, List[str]] -> e.g. {
                                                    'time' : [
                                                        'year',
                                                        'month'
                                                    ],
                                                    ..
                                                }
        w2vec_model     : object -> the w2vec model from gensim KeyedVectors
        sensitivity     : int -> how many tokens to map as similar to the lexicon
        exclude_pos     : list -> whether to exclude any words with POS tags
        exclude_dep     : list -> whether to exclude particular dependency words
        in_pipeline     : bool -> whether this follows the ExactMatcher output
        remove_stopwords: bool -> whether to remove nltk.corpus.stopwords from the analysis
        spacy_model     : str -> defaults to 'en_core_web_sm'
        chunking_method : str -> as entities identified are tokens, this class offers also
                                 chunking methods
    """

    def __init__(
        self,
        lexicon: _Dict[str, _List],
        w2vec_model: object,
        sensitivity: int = 15,
        exclude_pos: _List[str] = ["PUNCT"],
        exclude_dep: _List[str] = None,
        in_pipeline: bool = True,
        remove_stopwords: bool = True,
        spacy_model: str = "en_core_web_sm",
        chunking_method: str = "noun_chunking",
    ):

        if exclude_dep is None:
            exclude_dep = []

        self.exclude_pos = exclude_pos
        self.exclude_dep = exclude_dep
        self.in_pipeline = in_pipeline
        self.remove_stopwords = remove_stopwords
        self.stoplist = stopwords.words("english")
        self.stoplist = stopwords.words("english")
        self.chunking_method = chunking_method
        self.nlp = spacy.load(spacy_model)

        ## assert lexicon is not empty
        ## empty dicts by default return False
        assert bool(
            lexicon
        ), """
            lexicon arg is empty
            """

        assert self.chunking_method in (
            "noun_chunking",
            "custom_chunking",
            "edge_chunking",
        ),  """
            Chunker must be one of
                * noun_chunking
                * custom_chunking
                * edge_chunking
            For rule-based also define
            rules (see variable rules).
            """

        if self.chunking_method == "noun_chunking":

            self.chunker = noun_chunker

        elif self.chunking_method == "custom_chunking":

            cwd = _os.getcwd()
            file_path = f"{cwd}/custom_file.py"

            spec = _libutil.spec_from_file_location("custom_rules", file_path)
            custom_module = _libutil.module_from_spec(spec)
            _sys.modules["custom_rules"] = custom_module
            spec.loader.exec_module(custom_module)

            self.chunker = custom_module.rule_chunker

        else:

            self.chunker = edge_chunker

        self.extended_lexicon = lexicon_extender(
            model=w2vec_model,
            sensitivity=sensitivity,
            lexicon=lexicon,
        )

    def map(
        self,
        document: _Union[str, object],
        ents: _List[_Dict[str, _Any]] = None,
        ids: _Dict[int, str] = None,
    ) -> _Dict[str, _Any]:
        """
        maps the matcher generated EXTENDED lexicon items to the string using term matching
        then chunks matched terms using the user-specified chunking method
        Args:
            document    : Union
                str         -> the input document as string
                object      -> the input document as a spacy.doc
            ents    : List[Dict[str, Any]]  -> if part of a pipeline the pre-computed entity list
            ids     : Dict[str, Any]        -> if part of a pipeline the pre-computed word indexes (in sequence)
                                               and associated labels
        Returns:
            _List[Dict[str, any]]
        """

        if ents is None:
            ents = []
        if ids is None:
            ids = {}

        if self.in_pipeline:

            assert (
                isinstance(document, str) is False
            ), """
                                                        Must be in spacy.tokens.doc.Doc format
                                                        """

            ## copying to preserve case-sensitive aspect of the text
            ## however the matching is always done on lower-case characters
            output = document.text
            document = self.nlp(document.text.lower())

        else:

            assert isinstance(
                document, str
            ), """
                                                Document must be in str format, except if running in a pipeline.
                                                Then use in_pipeline = True
                                                """
            ## if first pass then tokenize and then pass in spacy nlp
            document = " ".join(word_tokenize(document))
            output = document
            document = self.nlp(document.lower())

        single_terms = self.__find_single_terms(document=document, ents=ents, ids=ids)

        chunked_entities = self.chunker(
            doc=document, ents=single_terms["ents"], ids=single_terms["ids"]
        )

        return {
            "doc": self.nlp(output),
            "ents": chunked_entities,
            "ids": {x["idx"]: x["label"] for x in chunked_entities},
        }

    def __find_single_terms(
        self,
        document: _Union[str, object],
        ents: _List[_Dict[str, _Any]] = None,
        ids: _Dict[int, str] = None,
    ) -> _Dict[str, _Any]:
        """
        matches single terms with the EXTENDED lexicon (built on the pre-specified lexicon)
        Args:
            document (_Union[str, object]):
                str         -> the input document as string
                object      -> the input document as a spacy.doc
            ents (_List[_Dict[str, _Any]], optional):   if part of a pipeline the pre-computed entity list.
                                                        Defaults to None.
            ids (_Dict[int, str], optional):    if part of a pipeline the pre-computed word indexes (in sequence)
                                                and associated labels.
                                                Defaults to None.

        Returns:
            _Dict[str, _Any]
        """

        if ents is None:

            ents = []

        if ids is None:

            ids = None

        ## loop over the document
        for entid, word in enumerate(document):

            if (self.remove_stopwords) and (word.text in self.stoplist):

                continue

            if (word.pos_ in self.exclude_pos) or (word.dep_ in self.exclude_dep):

                continue

            if entid in ids:

                continue

            ## might match with one or more groups
            ## choose the one with the lowest distance
            label_checks = {}
            ## fuzzy matching
            for name, extended_entities in self.extended_lexicon.items():

                ## abstract distance threshold (will tune as a hyperparameter)
                if word.text in extended_entities:

                    label_checks[name] = extended_entities[word.text]

                    if entid in ids:

                        continue

                    else:

                        ids[entid] = name

                        ents.append(
                            {
                                "start": word.idx,
                                "end": word.idx + len(word.text),
                                "text": word.text,
                                "label": name,
                                "idx": word.i,
                            }
                        )

            ## if a word matches across multiple categories (names)
            ## then find the one with the highest similarity
            if len(label_checks) > 1:

                ents[-1]["label"] = max(label_checks, key=label_checks.get)
                ids[ents[-1]["idx"]] = ents[-1]["label"]

        ## sort them
        ents = list(sorted(ents, key=lambda x: x["idx"]))

        if self.remove_stopwords:

            ents = [x for x in ents if x["text"] not in self.stoplist]

        return {
            "doc": document,
            "ents": ents,
            "ids": ids,
        }
