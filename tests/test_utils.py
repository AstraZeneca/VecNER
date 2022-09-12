import unittest
import numpy as np
import gensim.downloader as api
import vecner

api.info().keys()

model = api.load("glove-wiki-gigaword-100")

lexicon = {"time": ["one", "two"], "money": ["dollars", "euros"]}

print(vecner.__version__)


class TestVecnerUtils(unittest.TestCase):
    """
    class for testing the vecners.utils
    Args:
        unittest (object): unittest.TestCase
    """

    def test_flatten_lexicon(self) -> None:
        """
        testing flattening lexicon
        from dict to flat list of tuples
        """

        flattened = vecner.utils.flatten_lexicon(lexicon=lexicon)

        assert flattened == [
            ("time", "one"),
            ("time", "two"),
            ("money", "dollars"),
            ("money", "euros"),
        ]

        flattened = vecner.utils.flatten_lexicon(lexicon={})

        assert not flattened

    def test_phrase_representer(self) -> None:
        """
        testsing the phrase to vec function
        """

        phrase = "hours months does_not_exist"

        manual_check = (
            model["hours"] + model["months"] + np.zeros(model.vector_size)
        ) / 2

        auto_check = vecner.utils.phrase_representer(phrase=phrase, model=model)

        ## rounding due to errors in floating points
        assert (
            np.around(auto_check, decimals=3) == np.around(manual_check, decimals=3)
        ).all()

    def test_lexicon_extender(self) -> None:
        """
        test lexicon extensions
        """
        raise NotImplementedError
        return

    def test_merge_on_overlap(self) -> None:
        """
        testin merge on overlap fun that chunks entities
        """
        raise NotImplementedError
        return


if __name__ == "__main__":

    unittest.main()
