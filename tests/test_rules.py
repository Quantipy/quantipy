
import pytest
# from collections import OrderedDict
from quantipy.__imports__ import *  # noqa
from quantipy import (
    Rules,
    DataSet,
    Stack,
)
from .parameters.params_rules import *  # noqa
from .expectations.exp_view_mapper import * # noqa

NAME = "Example Data (A)"
DATA = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")


@pytest.fixture(scope="class")
def dataset():
    name = NAME
    path = DATA
    ds = DataSet.from_quantipy(name, path)
    ds._meta._clean_custom_sets_and_libs(True)
    return ds


@pytest.fixture(scope="class")
def stack(dataset):
    stack = Stack(NAME, add_data={NAME: {"dataset": dataset}})
    return stack


class TestRules:

    @staticmethod
    def _assert_index(df, expected):
        assert df.index.values.tolist() == expected

    @staticmethod
    def _assert_columns(df, expected):
        assert df.columns.values.tolist() == expected

    @pytest.mark.parametrize("params, expect", list(zip(SLICEX_PARAMS, SLICEX_EXP)))
    def test_rules(self, stack, params, expect):
        views = ["counts"]
        for slicer in params.get("slice", []):
            stack[NAME].meta.set_slicing(*slicer)

        for sort in params.get("sort", []):
            stack[NAME].meta.set_sorting(*sort)
            if not sort[1] == "@":
                views.append(sort[1])

        stack.add_link(
            NAME, None, params["xk"], params["yk"], views, params["weight"])
        link = stack[NAME][None][params["xk"]][params["yk"]]

        ct = link.crosstab(params["weight"], rules=False)
        natural_x = ct.index.values.tolist()
        natural_y = ct.columns.values.tolist()

        ct_rx = link.crosstab(params["weight"], rules=["x"])
        self._assert_index(ct_rx, expect["x"])
        self._assert_columns(ct_rx, natural_y)

        ctr_ry = link.crosstab(params["weight"], rules=["y"])
        self._assert_index(ctr_ry, natural_x)
        self._assert_columns(ctr_ry, expect["y"])

        ct_rxy = link.crosstab(params["weight"], rules=["x", "y"])
        self._assert_index(ct_rxy, expect["x"])
        self._assert_columns(ct_rxy, expect["y"])


