import os
import unittest

from transf_functions import extract_all


def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return True


class TestUnitTests(unittest.TestCase):

    def test_given_zip_path_when_call_extract_all_then_return_none(self):
        assert extract_all(os.path.realpath("data.zip"), os.path.realpath("data/")) is None
        assert find("artists_norm.csv", os.path.realpath("data"))
        assert find("albums_norm.csv", os.path.realpath("data"))
        assert find("tracks_norm.csv", os.path.realpath("data"))
