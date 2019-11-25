
import pytest
import pandas as pd

from quantipy.core.link import Link
from quantipy.core.stack import Stack
from quantipy.core.helpers.functions import load_json
from quantipy.core.view_generators.view_maps import QuantipyViews

XKS_MIN = ["q2b", "Wave", "q2", "q3", "q5_1"]
YKS_MIN = ["@", "q2b", "Wave", "q2", "q3", "q5_1"]

CBASE = "x|f|x:|||cbase"
RBASE = "x|f|:y|||rbase"
COUNTS = "x|f|:|||counts"
DEFAULT = "x|default|:|||default"

PATH = "./tests/"
FILENAME = "Example Data (A)"

DK = FILENAME
FK = 'no_filter'


@pytest.fixture(scope="module")
def example_meta():
    path_meta = "{}{}.json".format(PATH, FILENAME)
    return load_json(path_meta)


@pytest.fixture(scope="module")
def example_data():
    path_data = "{}{}.csv".format(PATH, FILENAME)
    return pd.DataFrame.from_csv(path_data)


@pytest.fixture(scope="module")
def stack(example_meta, example_data):
    stack = Stack(name=FILENAME)
    stack.add_data(
        data_key=stack.name,
        meta=example_meta,
        data=example_data)
    stack.add_link(
        data_keys=stack.name,
        filters="no_filter",
        x=XKS_MIN,
        y=YKS_MIN,
        views=QuantipyViews(["default"]),
        weights=[None])
    return stack


class TestLinkObject:

    def test_link_is_a_subclassed_dict(self, stack):
        for x in XKS_MIN:
            for y in YKS_MIN:
                link = stack[DK][FK][x][y]
                assert isinstance(link, dict)
                assert isinstance(link, Link)

    def test_link_behaves_like_a_dict(self, stack):
        key = "some_key_name"
        value = "some_value"

        for x in XKS_MIN:
            for y in YKS_MIN:
                link = stack[DK][FK][x][y]
                link[key] = value
                assert key in link.keys()

    def test_get_meta(self, stack):
        for x in XKS_MIN:
            for y in YKS_MIN:
                link = stack[DK][FK][x][y]
                assert link.get_meta() == stack[DK].meta

    def test_get_data(self, stack):
        for x in XKS_MIN:
            for y in YKS_MIN:
                link = stack[DK][FK][x][y]
                assert link.get_data().equals(stack[DK][FK].data)
