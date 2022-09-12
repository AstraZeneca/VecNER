import unittest
import gensim.downloader as api
from vecner import ExactMatcher, ExtendedMatcher, ThresholdMatcher

api.info().keys()

model = api.load("glove-wiki-gigaword-100")

# custom defined lexicon
test_lexicon = {
    "service": ["rude", "friendly"],
    "general": ["clean", "dirty", "decoration", "atmosphere"],
    "food": [
        "tasty",
        "disgusting",
        "delicious burger",
    ],
}


class TestVecnerMatchers(unittest.TestCase):
    """
    class for testing the vecners.__init__
    Args:
        unittest (object): unittest.TestCase
    """

    def test_exact_matcher(self) -> None:
        """
        testing the exact matcher with a known example
        """

        test_text = "what a delicious burger !"
        known_match = "delicious burger"
        known_label = "food"

        matcher = ExactMatcher(test_lexicon)

        ## check outputs
        output = matcher.map(test_text)

        assert "doc" in output
        assert "ents" in output
        assert "ids" in output

        doc = output["doc"]

        assert isinstance(doc, str) is False

        ents = output["ents"]

        assert len(ents) == 1

        ents = ents[0]

        assert "text" in ents
        assert "start" in ents
        assert "end" in ents
        assert "label" in ents
        assert "idx" in ents

        entity = ents["text"]
        start = ents["start"]
        end = ents["end"]
        idx = ents["idx"]
        label = ents["label"]

        assert entity == known_match
        assert doc.text[start:end] == known_match
        assert doc[idx].text == known_match.split()[0]
        assert label == known_label

        ids = output["ids"]

        assert len(ids) == len(known_match.split())
        assert len([x for x in ids.values() if x == known_label]) == len(ids)

    def test_threshold_matcher(self) -> None:
        """
        testing the threshold matcher with an observed example
        """

        test_text = "the place was nasty !"
        known_match = "nasty"
        known_label = "general"

        matcher = ThresholdMatcher(
            lexicon=test_lexicon, w2vec_model=model, in_pipeline=False, threshold=0.6
        )

        output = matcher.map(test_text)

        assert "doc" in output
        assert "ents" in output
        assert "ids" in output

        doc = output["doc"]

        assert isinstance(doc, str) is False

        ents = output["ents"]

        assert len(ents) == 1

        ents = ents[0]

        assert "text" in ents
        assert "start" in ents
        assert "end" in ents
        assert "label" in ents
        assert "idx" in ents

        entity = ents["text"]
        start = ents["start"]
        end = ents["end"]
        idx = ents["idx"]
        label = ents["label"]

        assert entity == known_match
        assert doc.text[start:end] == known_match
        assert doc[idx].text == known_match.split()[0]
        assert label == known_label

        ids = output["ids"]

        assert len(ids) == len(known_match.split())
        assert len([x for x in ids.values() if x == known_label]) == len(ids)

    def test_extended_matcher(self) -> None:
        """
        testing the extended matcher with an observed example
        """

        test_text = "the place was nasty !"
        known_match = "nasty"
        known_label = "general"

        matcher = ExtendedMatcher(
            lexicon=test_lexicon, w2vec_model=model, in_pipeline=False, sensitivity=10
        )

        output = matcher.map(test_text)

        assert "doc" in output
        assert "ents" in output
        assert "ids" in output

        doc = output["doc"]

        assert isinstance(doc, str) is False

        ents = output["ents"]

        assert len(ents) == 1

        ents = ents[0]

        assert "text" in ents
        assert "start" in ents
        assert "end" in ents
        assert "label" in ents
        assert "idx" in ents

        entity = ents["text"]
        start = ents["start"]
        end = ents["end"]
        idx = ents["idx"]
        label = ents["label"]

        assert entity == known_match
        assert doc.text[start:end] == known_match
        assert doc[idx].text == known_match.split()[0]
        assert label == known_label

        ids = output["ids"]

        assert len(ids) == len(known_match.split())
        assert len([x for x in ids.values() if x == known_label]) == len(ids)


if __name__ == "__main__":

    unittest.main()
