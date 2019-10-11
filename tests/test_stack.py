
import os
import pytest
import numpy as np
import pandas as pd

from collections import OrderedDict
from .test_batch import _get_batch

from quantipy.core.link import Link
from quantipy.core.view import View
from quantipy.core.stack import Stack
from quantipy.core.cache import Cache
from quantipy.core.helpers.functions import load_json
from quantipy.core.tools.qp_decorators import modify
from quantipy.core.view_generators.view_maps import QuantipyViews
from quantipy.core.view_generators.view_specs import calc

XKS_MIN = ["q2b", "Wave", "q2", "q3", "q5_1"]
YKS_MIN = ["@", "q2b", "Wave", "q2", "q3", "q5_1"]

CBASE = "x|f|x:|||cbase"
RBASE = "x|f|:y|||rbase"
COUNTS = "x|f|:|||counts"
DEFAULT = "x|default|:|||default"

PATH = "./tests/"
FILENAME = "Example Data (A)"


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


@pytest.fixture(scope="function")
def stack_empty(example_meta, example_data):
    stack_empty = Stack(name="cache")
    stack_empty.add_data(
        data_key=stack_empty.name,
        meta=example_meta,
        data=example_data)
    return stack_empty


class TestStackObject:

    def test_stack_is_a_subclassed_dict(self, stack):
        assert isinstance(stack, dict)
        assert isinstance(stack, Stack)

    def test_stack_behaves_like_a_dict(self, stack):
        key = "some_key_name"
        value = "some_value"
        stack[key] = value
        assert key in stack.keys()
        del stack[key]

    def test_cache_is_created(self, stack_empty):

        # Assert that it has a Cache that is empty
        assert "cache" in stack_empty["cache"].__dict__.keys()
        assert isinstance(stack_empty["cache"].cache, Cache)
        assert Cache() == stack_empty["cache"].cache

        # Run the Aggregations
        self.add_link(stack_empty)

        # Assert that it has a Cache that is NOT empty
        assert "cache" in stack_empty["cache"].__dict__.keys()
        assert isinstance(stack_empty["cache"].cache, Cache)
        assert not [] == stack_empty["cache"].cache.keys()

        # Manually remove the cache
        del stack_empty["cache"].cache
        stack_empty["cache"].cache = Cache()
        assert Cache() == stack_empty["cache"].cache

        # Rerun the Aggregations
        self.add_link(stack_empty)

        # Assert that it has a Cache has been recreated
        assert "cache" in stack_empty["cache"].__dict__.keys()
        assert isinstance(stack_empty["cache"].cache, Cache)
        assert not [] == stack_empty["cache"].cache.keys()

    def test_add_data(self, caplog, example_meta, example_data):

        temp_stack = Stack()
        # Test data_key errors
        for data_key in [1, 1.1, [1], {}]:
            caplog.clear()
            with pytest.raises(TypeError):
                temp_stack.add_data(data_key=data_key)
            expect = "All data keys must be one of the following types: "
            assert expect == caplog.records[0].message[:50]

        # Test data errors
        for data in [1, pd.Series([1, 2, 3, 4, 5]), "data"]:
            caplog.clear()
            with pytest.raises(TypeError):
                temp_stack.add_data(data_key=temp_stack.name, data=data)
            expect = (
                "The 'data' given to Stack.add_data() must be one of the "
                "following types: ")
            assert expect == caplog.records[0].message[:73]

        # Test meta errors
        for meta in [1, [1], "meta"]:
            caplog.clear()
            with pytest.raises(TypeError):
                temp_stack.add_data(data_key=temp_stack.name, meta=meta)
            expect = (
                "The 'meta' given to Stack.add_data() must be one of the "
                "following types: ")
            assert expect == caplog.records[0].message[:73]

        # Test proxy data key
        temp_stack.add_data(data_key="test")
        assert "test" in temp_stack
        del temp_stack["test"]

        # Test meta-only data key
        temp_stack.add_data(data_key="test", meta=example_meta)
        assert "test" in temp_stack
        del temp_stack["test"]

        # Test data-only data key
        temp_stack.add_data(data_key="test", data=example_data)
        assert "test" in temp_stack
        del temp_stack["test"]

        # Test meta+data data key
        temp_stack.add_data(
            data_key="test", meta=example_meta, data=example_data)
        assert "test" in temp_stack

        # The data and meta keys exist
        assert "meta" in temp_stack["test"].__dict__
        assert "data" in temp_stack["test"].__dict__

        # The data and meta attributes exist (after using stack.add_data())
        assert hasattr(temp_stack["test"], "meta")
        assert hasattr(temp_stack["test"], "data")

        # The data and meta attributes should be the correct instance type
        assert isinstance(temp_stack["test"].meta, (dict, OrderedDict))
        assert isinstance(temp_stack["test"].data, pd.DataFrame)

    def test_add_link_generates_links_and_views(self, stack):
        # Test that a Link objects sits behind all y-keys
        self.verify_links_and_views_exist_in_nest(stack)

    def test_reduce(self, stack):
        self.add_link(
            stack,
            fk=["no_filter", "Wave == 1"],
            views=["default", "counts"])

        k = "Wave == 1"
        others = ["no_filter"]
        stack.reduce(filters=k)
        content = stack.describe("filter").index.tolist()
        assert k not in content
        assert all(k in content for k in others)

        k = "q5_1"
        others = ["q2b", "Wave", "q2", "q3"]
        stack.reduce(x=k)
        content = stack.describe("x").index.tolist()
        assert k not in content
        assert all(k in content for k in others)

        k = "q5_1"
        others = ["@", "q2b", "Wave", "q2", "q3"]
        stack.reduce(y=k)
        content = stack.describe("y").index.tolist()
        assert k not in content
        assert all(k in content for k in others)

        k = COUNTS
        others = [DEFAULT]
        stack.reduce(views=k)
        content = stack.describe("view").index.tolist()
        assert k not in content
        assert all(k in content for k in others)

        self.add_link(stack)

        # Test error handling for non-existant keys
        keys = ["filters", "views", "x", "y"]
        non_key = "this key is unvalid"
        for k in keys:
            with pytest.raises(ValueError):
                stack.reduce(**{k: non_key})

    def test_getting_1D_views(self, stack):

        # Test that x="@" links produce Views
        self.add_link(stack, xk=["@"], yk=["Wave"])
        view = stack[stack.name]["no_filter"]["@"]["Wave"][DEFAULT]
        assert isinstance(view, View)

        # Test that y="@" links produce Views
        self.add_link(stack, xk=["Wave"], yk=["@"])
        view = stack[stack.name]["no_filter"]["Wave"]["@"][DEFAULT]
        assert isinstance(view, View)

    def test_filters(self, stack, example_data):
        filters = [
            "no_filter",
            "Wave == 1",
            "Wave == 2 and age > 30",
            "age > 30",
        ]
        x_keys = ["Wave", "age"]
        y_keys = ["@"] + x_keys
        self.add_link(stack, fk=filters, xk=x_keys, yk=y_keys)
        # Test all the populated filter keys exist in the stack
        for f in filters:
            for x in x_keys:
                for y in y_keys:
                    df = stack[stack.name][f][x][y][DEFAULT].dataframe
                    assert isinstance(df, pd.DataFrame)

        # Test the filters have calculated correctly
        # no_filter
        df = stack[stack.name]["no_filter"]["Wave"]["@"][DEFAULT].dataframe
        assert df.iloc[(0, 0)] == example_data.shape[0]
        # Other filters
        for f in filters[1:]:
            df = stack[stack.name][f]["Wave"]["@"][DEFAULT].dataframe
            assert df.iloc[(0, 0)] == example_data.query(f).shape[0]

        # set back stack instance
        stack.reduce(filters=filters)
        self.add_link(stack)

    def test_add_link_x_y_equal(self, stack):
        dk = stack.name
        fk = "no_filter"

        # Test that x==y links have populated correctly
        for xy in XKS_MIN:
            # Test x==y requests produce Link objects
            link = stack[dk][fk][xy][xy]
            assert isinstance(link, Link)
            # Test x==y requests produce View objects
            view = link[DEFAULT]
            assert isinstance(view, View)
            # Test x==y requests produce dataframes where index and columns are
            # the same (with the execption of the "All"-margin)
            df = view.dataframe
            index = df.index.get_level_values(1).tolist()
            columns = df.columns.get_level_values(1).tolist()
            assert index == columns

    def test_add_link_lazy(self, stack):

        views = stack.describe('view').index.tolist()
        assert views == [DEFAULT]

        self.add_link(stack, views=["counts"])
        views = stack.describe('view').index.tolist()
        assert views == [DEFAULT, COUNTS]

        # Test lazy y-keys when 1 x key is given
        self.add_link(stack, xk=XKS_MIN[0], views=["cbase"])
        lazy_y = stack.describe(
            index=["y"],
            query="x=='{}' and view=='{}'".format(XKS_MIN[0], CBASE)
        ).index.tolist()
        assert all(y in YKS_MIN for y in lazy_y)

        # Test lazy x-keys when 1 y key is given
        self.add_link(stack, yk=YKS_MIN[0], views=["rbase"])
        lazy_x = stack.describe(
            index=["x"],
            query="y=='{}' and view=='{}'".format(YKS_MIN[0], RBASE)
        ).index.tolist()
        assert all(x in XKS_MIN for x in lazy_x)

        # set back stack instance
        stack.reduce(filters="no_filter")
        self.add_link(stack)

    def test_describe(self, stack_empty):
        dk = [stack_empty.name]
        fk = ["no_filter", "Wave == 1"]
        xk = ['gender', 'locality', 'ethnicity', 'religion', 'q1']
        yk = ['q2', 'q3', 'q8', 'q9']
        vk = ["default", "cbase", "rbase", "counts"]
        vk_notation = [DEFAULT, RBASE, COUNTS, CBASE]
        self.add_link(stack_empty, fk=fk, xk=xk, yk=yk, views=vk)

        # Test describe returns a pandas DataFrame
        contents = stack_empty.describe()
        assert isinstance(contents, pd.DataFrame)

        # Test describe returns df with required columns
        expected_columns = ["data", "filter", "x", "y", "view", "#"]
        actual_columns = contents.columns.tolist()
        assert expected_columns == actual_columns

        # Test desribe returns df with expected number of rows
        expected_rows = len(dk) * len(fk) * len(xk) * len(yk) * len(vk)
        actual_rows = contents.shape[0]
        assert expected_rows == actual_rows

        # Test the returned df contains everything expected and nothing
        # unexpected
        self.verify_contains_expected_not_unexpected(
            contents, dk, fk, xk, yk, vk_notation)

        # Test index & column parameters
        column_names = ["data", "filter", "x", "y", "view"]

        # check index column parameters
        for column in column_names:
            described_index = stack_empty.describe(index=[column])
            described_column = stack_empty.describe(columns=[column])
            assert column in described_index.index.names
            assert column in described_column.index.names

            for index in column_names:
                if column == index:
                    continue
                content = stack_empty.describe(index=[index], columns=[column])
                assert index in content.index.names
                assert column in content.columns.names

        # Test query parameter
        xk = XKS_MIN[:2]
        yk = XKS_MIN[-2:]
        query = "x in {} and y in {} and view=='{}'".format(xk, yk, DEFAULT)
        contents = stack_empty.describe(query=query)

        # Test the returned df contains everything expected and nothing
        # unexpected
        self.verify_contains_expected_not_unexpected(contents, xk=xk, yk=yk)

        # Test that query can be used in conjunction with index
        contents = stack_empty.describe(index=["view"], query="x=='gender'")
        assert all(vk in contents.index.tolist() for vk in [
            DEFAULT, CBASE, COUNTS, RBASE])

        # Test that query can be used in conjunction with columns
        contents = stack_empty.describe(columns=["view"], query="x=='gender'")
        assert all(vk in contents.index.tolist() for vk in [
            DEFAULT, CBASE, COUNTS, RBASE])

    def test_refresh(self, stack_empty, example_data):
        dk = stack_empty.name
        all_filters = ["Wave==1", "no_filter"]
        all_x = ["q1", "q2", "q2b", "q3", "q4"]
        all_y = ["@", "gender", "locality", "ethnicity"]
        weights = [None, "weight_a"]

        stack_empty.add_link(
            x=all_x,
            y=all_y,
            weights=weights,
            filters=all_filters,
            views=["counts"])
        stack_empty.add_link(
            x=["q2"],
            y=["gender"],
            weights=None,
            views=["c%"])
        stack_empty.add_link(
            x=["q1", "q3"],
            y=["@", "locality"],
            weights="weight_a",
            filters=["Wave==1"],
            views=["cbase"])

        content = stack_empty.describe(columns="data", index="view")

        stack_empty.refresh(
            data_key=dk,
            new_data_key="new_key",
            new_weight="weight_b")

        content2 = stack_empty.describe(columns="data", index="view")
        assert content.values.sum() == 85.0
        assert content2[dk].sum() == 85.0
        assert content2["new_key"].sum() == 130.0
        stack_empty.reduce(data_keys="new_key")

        mod_data = example_data.copy().head(500)
        stack_empty.refresh(
            data_key=dk, new_data_key="new_key", new_data=mod_data)

        content3 = stack_empty.describe(columns="data", index="view")
        assert content.values.sum() == 85.0
        assert content3[dk].sum() == 85.0
        assert content3["new_key"].sum() == 85.0

    def test_refresh_remove_weight(self, stack_empty):
        fks = ["Wave==1", "no_filter"]
        xks = ["q1", "q2", "q2b", "q3", "q4"]
        yks = ["@", "gender", "locality", "ethnicity"]
        weight = ["weight_a"]

        stack_empty.add_link(
            x=xks, y=yks, weights=weight, filters=fks, views=["counts"])
        stack_empty.add_link(
            x=["q2"], y=["gender"], weights=None, views=["c%"])
        stack_empty.add_link(
            x=["q1", "q3"], y=["@", "locality"], weights="weight_a",
            filters=["Wave==1"], views=["cbase"])

        content = stack_empty.describe(columns="data", index="view")

        stack_empty.refresh(
            data_key=stack_empty.name, new_data_key="new_key", new_weight="")
        content2 = stack_empty.describe(columns="data", index="view")

        assert content.values.sum() == 45.0
        assert content2[stack_empty.name].sum() == 45.0
        assert content2["new_key"].sum() == 89.0

    def test_save_dataset(self, stack_empty):
        """
        This tests save/load methods using the dataset parameter.
        """
        path_stack = "{}{}.stack".format(PATH, FILENAME)
        stack_empty.save(path_stack=path_stack, dataset=True)

        for key in stack_empty.keys():
            path_json = path_stack.replace(
                '.stack',
                ' [{}].json'.format(key))
            path_csv = path_stack.replace(
                '.stack',
                ' [{}].csv'.format(key))
            assert os.path.exists(path_json)
            assert os.path.exists(path_csv)
            os.remove(path_json)
            os.remove(path_csv)
        os.remove(path_stack)

    def test_save_describe(self, stack_empty):
        """
        This tests save/load methods using the describe parameter.
        """
        path_stack = "{}{}.stack".format(PATH, FILENAME)
        stack_empty.save(path_stack=path_stack, describe=True)
        path_describe = path_stack.replace(".stack", ".xlsx")
        assert os.path.exists(path_describe)
        os.remove(path_describe)
        os.remove(path_stack)

    def test_save_and_load_with_and_without_cache(self, stack):
        """
        This tests that the cache is stored and loaded with and without cache
        """
        path_stack = "{}{}.stack".format(PATH, FILENAME)
        path_cache = "{}{}.cache".format(PATH, FILENAME)
        compressiontype = [None, "gzip"]

        if os.path.exists(path_stack):
            os.remove(path_stack)
        if os.path.exists(path_cache):
            os.remove(path_cache)

        caches = {}
        for key in stack.keys():
            assert "cache" in stack[key].__dict__.keys()
            assert isinstance(stack[key].cache, Cache)
            assert not stack[key].cache == Cache()
            caches[key] = stack[key].cache

        for compression in compressiontype:
            # Save the stack WITHOUT the cache
            stack.save(path_stack, compression, store_cache=False)
            assert os.path.exists(path_stack)
            assert not os.path.exists(path_cache)

            new_stack = Stack.load(path_stack, compression=compression)
            for k in stack.keys():
                assert k in new_stack

            # Ensure that there is NO cache
            for key in new_stack.keys():
                assert Cache() == new_stack[key].cache

            # Save the stack WITH the cache
            stack.save(path_stack, compression, store_cache=True)
            assert os.path.exists(path_stack)
            assert os.path.exists(path_cache)

            new_stack = Stack.load(path_stack, compression, load_cache=True)
            for k in stack.keys():
                assert k in new_stack

            # Ensure that there IS a cache
            for key, value in caches.items():
                assert "matrices" in value.keys()
                assert "weight_vectors" in value.keys()
                for sect_def in value["matrices"]:
                    mat1, codes1 = value["matrices"][sect_def]
                    mat2, codes2 = new_stack[key].cache["matrices"][sect_def]
                    assert isinstance(mat1, np.ndarray)
                    assert isinstance(mat2, np.ndarray)
                    assert np.array_equal(mat1, mat2)
                    assert isinstance(codes1, list)
                    assert isinstance(codes2, list)
                    assert np.array_equal(codes1, codes2)

                assert not id(caches[key]) == id(new_stack[key].cache)
            os.remove(path_stack)
            os.remove(path_cache)

    def test_save_load_stack_improved(self, stack):
        """
        This tests save/load methods using dataframes and
        verifies that the source data is still intact after load
        and that the structure, meta and data attributes and views are intact.
        """
        path_stack = "{}{}.stack".format(PATH, FILENAME)
        compressiontype = [None, "gzip"]

        # Ensure that the stack has the correct structure, attributes
        # and views.
        for dk in stack.keys():
            # Does the loaded stack actually have the data and meta attributes
            assert hasattr(stack[dk], "data")
            assert hasattr(stack[dk], "meta")
            assert isinstance(stack[dk].data, pd.DataFrame)

        # Save and load the stack and test that everything is still available,
        # the loaded stack has the same views, attributes as the generated one
        for compression in compressiontype:
            if os.path.exists(path_stack):
                os.remove(path_stack)
            assert not os.path.exists(path_stack)
            stack.save(path_stack=path_stack, compression=compression)
            assert os.path.exists(path_stack)
            loaded_stack = Stack.load(path_stack, compression=compression)
            # Ensure that it is not the same stack (in memory)
            assert not id(loaded_stack) == id(stack)
            # Test all of the keys in the loaded stack
            for dk in loaded_stack:
                # Does the loaded stack actually have the data and meta attr
                assert hasattr(loaded_stack[dk], "data")
                assert hasattr(loaded_stack[dk], "meta")
                assert isinstance(loaded_stack[dk].data, pd.DataFrame)

                # Verify that the metadata is also loaded
                assert loaded_stack[dk].meta == stack[dk].meta
                assert not id(loaded_stack[dk].meta) == (stack[dk].meta)

                for fk in loaded_stack[dk]:
                    for x in loaded_stack[dk][fk]:
                        for y in loaded_stack[dk][fk][x]:
                            for view in loaded_stack[dk][fk][x][y]:
                                v1 = stack[dk][fk][x][y][view]
                                v2 = loaded_stack[dk][fk][x][y][view]
                                assert type(v1) == type(v2)
            if os.path.exists(path_stack):
                os.remove(path_stack)

    def test_stack_aggregate(self):
        b1, ds = _get_batch("test1", full=True)
        b2, ds = _get_batch("test2", ds, False)
        b3, ds = _get_batch("test3", ds, False)
        b1.add_downbreak(["q1", "q6", "age"])
        b1.add_crossbreak(["gender", "q2"])
        b1.extend_filter({"q1": {"age": [20, 21, 22]}})
        b1.set_weights("weight_a")
        b2.add_downbreak(["q1", "q6"])
        b2.add_crossbreak(["gender", "q2"])
        b2.set_weights("weight_b")
        b2.transpose("q6")
        b3.add_downbreak(["q1", "q7"])
        b3.add_crossbreak(["q2b"])
        b3.add_y_on_y("y_on_y")
        b3.set_weights(["weight_a", "weight_b"])
        stack = ds.populate(verbose=False)
        stack.aggregate(["cbase", "counts", "c%"], True,
                        "age", ["test1", "test2"], verbose=False)
        stack.aggregate(["cbase", "counts", "c%", "counts_sum", "c%_sum"],
                        False, None, ["test3"], verbose=False)
        index = [
            "x|f.c:f|x:|y|weight_a|c%_sum",
            "x|f.c:f|x:|y|weight_b|c%_sum",
            "x|f.c:f|x:||weight_a|counts_sum",
            "x|f.c:f|x:||weight_b|counts_sum",
            "x|f|:|y|weight_a|c%",
            "x|f|:|y|weight_b|c%",
            "x|f|:||weight_a|counts",
            "x|f|:||weight_b|counts",
            "x|f|x:||weight_a|cbase",
            "x|f|x:||weight_b|cbase",
            "x|f|x:|||cbase"]
        cols = [
            "@", "age", "q1", "q2b", "q6", u"q6_1", u"q6_2", u"q6_3", u"q7",
            u"q7_1", u"q7_2", u"q7_3", u"q7_4", u"q7_5", u"q7_6"]
        values = [
            ["NONE", "NONE", 2.0, 2.0, "NONE", "NONE", "NONE", "NONE", 1.0,
                2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            ["NONE", "NONE", 2.0, 2.0, "NONE", "NONE", "NONE", "NONE", 1.0,
                2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            ["NONE", "NONE", 2.0, 2.0, "NONE", "NONE", "NONE", "NONE", 1.0,
                2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            ["NONE", "NONE", 2.0, 2.0, "NONE", "NONE", "NONE", "NONE", 1.0,
                2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            ["NONE", 3.0, 5.0, 2.0, 1.0, 3.0, 3.0, 3.0, 1.0, 2.0, 2.0, 2.0,
                2.0, 2.0, 2.0],
            [1.0, "NONE", 4.0, 2.0, 1.0, 3.0, 3.0, 3.0, 1.0, 2.0, 2.0, 2.0,
                2.0, 2.0, 2.0],
            ["NONE", 3.0, 5.0, 2.0, 1.0, 3.0, 3.0, 3.0, 1.0, 2.0, 2.0, 2.0,
                2.0, 2.0, 2.0],
            [1.0, "NONE", 4.0, 2.0, 1.0, 3.0, 3.0, 3.0, 1.0, 2.0, 2.0, 2.0,
                2.0, 2.0, 2.0],
            ["NONE", 3.0, 5.0, 2.0, 1.0, 3.0, 3.0, 3.0, 1.0, 2.0, 2.0, 2.0,
                2.0, 2.0, 2.0],
            [1.0, "NONE", 4.0, 2.0, 1.0, 3.0, 3.0, 3.0, 1.0, 2.0, 2.0, 2.0,
                2.0, 2.0, 2.0],
            [1.0, 3.0, 6.0, "NONE", 2.0, 6.0, 6.0, 6.0, "NONE", "NONE", "NONE",
                "NONE", "NONE", "NONE", "NONE"]]

        describe = stack.describe("view", "x").replace(np.NaN, "NONE")
        assert describe.index.tolist() == index
        assert describe.columns.tolist() == cols
        assert describe.values.tolist() == values

    def test_cumulative_sum(self):
        b, ds = _get_batch("test1", full=True)
        stack = ds.populate(verbose=False)
        stack.aggregate(
            ["cbase", "counts", "c%"], batches="all", verbose=False)
        stack.cumulative_sum(["q1", "q6"], "all", verbose=False)
        describe = stack.describe("view", "x").replace(np.NaN, "NONE")
        index = [
            "x|f.c:f|x++:|y|weight_a|c%_cumsum",
            "x|f.c:f|x++:||weight_a|counts_cumsum",
            "x|f|:|y|weight_a|c%",
            "x|f|:||weight_a|counts",
            "x|f|x:||weight_a|cbase",
            "x|f|x:|||cbase"]
        cols = ["age", "q1", "q2", "q6", u"q6_1", u"q6_2", u"q6_3"]
        values = [
            ["NONE", 3.0, "NONE", 1.0, 3.0, 3.0, 3.0],
            ["NONE", 3.0, "NONE", 1.0, 3.0, 3.0, 3.0],
            ["NONE", "NONE", 3.0, "NONE", "NONE", "NONE", "NONE"],
            ["NONE", "NONE", 3.0, "NONE", "NONE", "NONE", "NONE"],
            [3.0, 3.0, 3.0, 1.0, 3.0, 3.0, 3.0],
            [3.0, 3.0, 3.0, 1.0, 3.0, 3.0, 3.0]]
        assert describe.index.tolist() == index
        assert describe.columns.tolist() == cols
        assert describe.values.tolist() == values

    def test_add_nets(self):
        b, ds = _get_batch("test1", full=True)
        stack = ds.populate(verbose=False)
        stack.aggregate(
            ["cbase", "counts", "c%"], batches="all", verbose=False)
        calcu = calc((2, "-", 1), "difference", "en-GB")
        stack.add_nets(
            ["q1", "q6"], [{"Net1": [1, 2]}, {"Net2": [3, 4]}], "after",
            calcu, _batches="all", recode=False, verbose=False)
        index = [
            "x|f.c:f|x[{1,2}+],x[{3,4}+],x[{3,4}-{1,2}]*:|y|weight_a|net",
            "x|f.c:f|x[{1,2}+],x[{3,4}+],x[{3,4}-{1,2}]*:||weight_a|net",
            "x|f|:|y|weight_a|c%",
            "x|f|:||weight_a|counts",
            "x|f|x:||weight_a|cbase",
            "x|f|x:|||cbase"]
        cols = ["age", "q1", "q2", "q6", u"q6_1", u"q6_2", u"q6_3"]
        values = [
            ["NONE", 3.0, "NONE", 1.0, 3.0, 3.0, 3.0],
            ["NONE", 3.0, "NONE", 1.0, 3.0, 3.0, 3.0],
            ["NONE", "NONE", 3.0, "NONE", "NONE", "NONE", "NONE"],
            ["NONE", "NONE", 3.0, "NONE", "NONE", "NONE", "NONE"],
            [3.0, 3.0, 3.0, 1.0, 3.0, 3.0, 3.0],
            [3.0, 3.0, 3.0, 1.0, 3.0, 3.0, 3.0]]
        describe = stack.describe("view", "x").replace(np.NaN, "NONE")
        assert describe.index.tolist() == index
        assert describe.columns.tolist() == cols
        assert describe.values.tolist() == values

    def test_recode_from_net_def(self):
        b, ds = _get_batch("test1", full=True)
        stack = ds.populate()
        stack.add_nets(["q1"], [{"Net1": [1, 2]}, {"Net2": [3, 4]}], "after",
                       recode="collect_codes", _batches="all", verbose=False)
        values = ds["q1_rc"].value_counts().values.tolist()
        expect = [5297, 2264, 694]
        assert expect == values
        stack.add_nets(["q1"], [{"Net1": [1, 2]}, {"Net2": [3, 4]}], "after",
                       recode="drop_codes", _batches="all", verbose=False)
        values = ds["q1_rc"].value_counts().values.tolist()
        expect = [5297, 694]
        assert expect == values
        stack.add_nets(["q1"], [{"Net1": [1, 2]}, {"Net2": [3, 4]}], "after",
                       recode="extend_codes", _batches="all", verbose=False)
        values = ds["q1_rc"].value_counts().values.tolist()
        expect = [2999, 2298, 894, 477, 397, 369, 297, 194, 131, 104, 91, 4]
        assert expect == values
        assert "delimited set" == ds._get_type("q1_rc")

    def test_add_stats(self):
        b, ds = _get_batch("test1", full=True)
        stack = ds.populate(verbose=False)
        stack.aggregate(
            ["cbase", "counts", "c%"], batches="all", verbose=False)
        stack.add_stats(
            "q6", ["mean"], rescale={1: 3, 2: 2, 3: 1}, factor_labels=False,
            _batches="all", verbose=False)
        stack.add_stats(
            "q1", ["mean"], "age", factor_labels=False, verbose=False,
            _batches="all")
        index = [
            "x|d.mean|age:||weight_a|stat",
            "x|d.mean|x[{3,2,1}]:||weight_a|stat",
            "x|f|:|y|weight_a|c%",
            "x|f|:||weight_a|counts",
            "x|f|x:||weight_a|cbase",
            "x|f|x:|||cbase"]
        cols = ["age", "q1", "q2", "q6", u"q6_1", u"q6_2", u"q6_3"]
        values = [
            ["NONE", 3.0, "NONE", "NONE", "NONE", "NONE", "NONE"],
            ["NONE", "NONE", "NONE", 1.0, 3.0, 3.0, 3.0],
            ["NONE", 3.0, 3.0, 1.0, 3.0, 3.0, 3.0],
            ["NONE", 3.0, 3.0, 1.0, 3.0, 3.0, 3.0],
            [3.0, 3.0, 3.0, 1.0, 3.0, 3.0, 3.0],
            [3.0, 3.0, 3.0, 1.0, 3.0, 3.0, 3.0]]
        describe = stack.describe("view", "x").replace(np.NaN, "NONE")
        assert describe.index.tolist() == index
        assert describe.columns.tolist() == cols
        assert describe.values.tolist() == values

    def test_recode_from_stat_def(self):
        b, ds = _get_batch("test1", full=True)
        stack = ds.populate()
        stack.add_stats(
            "q6", ["mean"], rescale={1: 0, 2: 50, 3: 100}, factor_labels=False,
            _batches="all", verbose=False, recode=True)
        expect_ind = [0.0, 50.0, 100.0]
        index = ds["q6_1_rc"].value_counts().index.tolist()
        expect_val = [3074, 2620, 875]
        values = ds["q6_1_rc"].value_counts().values.tolist()
        assert expect_val == values
        assert expect_ind == index
        with pytest.raises(ValueError):
            stack.add_stats("q6", other_source="q1")

    def test_factor_labels(self):

        def _factor_on_values(values, axis="x"):
            return all(
                v["text"]["{} edits".format(axis)]["en-GB"].endswith(
                    "[{}]".format(v["value"]))
                for v in values)

        b1, ds = _get_batch("test1", full=True)
        b1.add_downbreak(["q1", "q2b", "q6"])
        b1.set_variable_text("q1", "some new text1")
        b1.set_variable_text("q6", "some new text1")
        b2, ds = _get_batch("test2", ds, True)
        b2.add_downbreak(["q1", "q2b", "q6"])
        b2.set_variable_text("q1", "some new text2")
        stack = ds.populate()
        stack.aggregate(["cbase", "counts", "c%"], batches="all")
        stack.add_stats(["q1", "q2b", "q6"], ["mean"], _batches="all")
        for dk in stack.keys():
            meta = stack[dk].meta
            batches = meta["sets"]["batches"]
            # q1, both batches have meta_edits
            values = batches["test1"]["meta_edits"]["q1"]["values"]
            assert _factor_on_values(values)
            values = batches["test2"]["meta_edits"]["q1"]["values"]
            assert _factor_on_values(values)
            values = meta["columns"]["q1"]["values"]
            assert all("x_edits" not in v["text"] for v in values)
            # q2b, no batch has meta_edits
            values = meta["columns"]["q2b"]["values"]
            assert _factor_on_values(values)
            assert all("q2b" not in b["meta_edits"]
                       for n, b in batches.items())
            # q6, one batch with meta_edits and one without
            values = batches["test1"]["meta_edits"]["lib"]["q6"]
            assert _factor_on_values(values)
            assert _factor_on_values(values, "y")
            values = meta["lib"]["values"]["q6"]
            assert _factor_on_values(values)
            assert _factor_on_values(values, "y")
            assert "q6" not in batches["test2"]["meta_edits"]

    # =========================================================================
    # helpers
    # =========================================================================

    def add_link(self, stack_inst, fk=None, xk=None, yk=None, views=None,
                 weights=None):
        if not fk:
            fk = "no_filter"
        if not xk:
            xk = XKS_MIN
        if not yk:
            yk = YKS_MIN
        if not views:
            views = ["default"]
        if not isinstance(weights, list):
            weights = [weights]
        for weight in weights:
            stack_inst.add_link(
                data_keys=stack_inst.name,
                filters=fk,
                x=xk,
                y=yk,
                views=QuantipyViews(views),
                weights=weight)

    def yield_links(self, nest):
        """ Yields all the Links in nest, which could be either a stack
        or a chain.
        """
        for dk in nest.keys():
            filters = nest[dk]
            for fk in filters.keys():
                xks = nest[dk][fk]
                for xk in xks.keys():
                    yks = nest[dk][fk][xk]
                    for yk in yks.keys():
                        link = nest[dk][fk][xk][yk]
                        yield link

    def verify_links_and_views_exist_in_nest(self, stack_inst):
        """
        Verifies that Links and Views sit in the appropriate levels of the
        given stack.
        """
        for link in self.yield_links(stack_inst):
            assert isinstance(link, Link)
            for vk in link.keys():
                view = link[vk]
                assert isinstance(view, View)
                assert isinstance(view.dataframe, pd.DataFrame)

    @modify(to_list=["dk", "fk", "xk", "yk", "vk"])
    def verify_contains_expected_not_unexpected(self, contents, dk=None,
                                                fk=None, xk=None, yk=None,
                                                vk=None):
        """
        Verifies that contents (a stack/chain.describe() result) contains all
        the keys passed and no keys that weren"t passed.
        """
        keys1 = [dk, fk, xk, yk, vk]
        keys2 = ["data", "filter", "x", "y", 'view']
        for k1, k2 in zip(keys1, keys2):
            if k1:
                has_k = contents[k2].unique().tolist()
                assert all(k in k1 for k in has_k)
