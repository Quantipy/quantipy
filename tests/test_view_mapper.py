
import pytest
from collections import OrderedDict
from quantipy.__imports__ import *  # noqa
from quantipy import (
    ViewMapper,
    DataSet,
    Stack,
    Link,
    # Meta,
)
from .expectations.exp_view_mapper import * ## noqa

NAME1 = "engine_B"
NAME2 = "Example Data (A)"
DATA = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")


@pytest.fixture(scope="class")
def dataset1():
    name = NAME1
    path = DATA
    ds = DataSet.from_quantipy(name, path)
    ds._meta._clean_custom_sets_and_libs(True)
    return ds


@pytest.fixture(scope="class")
def stack1(dataset1):
    stack = Stack(NAME2, add_data={NAME1: {"dataset": dataset1}})
    return stack


@pytest.fixture(scope="class")
def dataset2():
    name = NAME2
    path = DATA
    ds = DataSet.from_quantipy(name, path)
    ds._meta._clean_custom_sets_and_libs(True)
    return ds


@pytest.fixture(scope="function")
def stack2(dataset2):
    stack = Stack(NAME2, add_data={NAME2: {"dataset": dataset2}})
    return stack


class TestViewMapper:

    def test_add_method(self):
        vm = ViewMapper()
        assert "plus" not in vm

        vm.add_method("plus", lambda x: 2)
        assert vm["plus"]["method"] is not None
        assert vm["plus"]["kwargs"] is not None

        vm.add_method("plus", "+")
        assert vm["plus"]["method"] == "+"

        vm.add_method("minus", "-", {"key": "value"})
        assert vm["minus"]["method"] == "-"
        assert vm["minus"]["kwargs"] == {"key": "value"}

    def test__apply_to(self, stack1):
        views = ["cbase", "counts", "mean"]
        vkeys = ["x|f|x:|||cbase", "x|f|:|||counts", "x|d.mean|x:|||mean"]
        vm = ViewMapper(views)

        xks = yks = ["profile_gender", "age_group", "q4"]
        stack1.add_link(NAME1, x=xks, y=yks)

        for xk in xks:
            for yk in yks:
                link = stack1[NAME1][None][xk][yk]
                vm._apply_to(link)

                for view, vk in zip(views, vkeys):
                    if vm[view]["method"] == "descriptives" and xk == "q4":
                        assert vk not in link
                    else:
                        assert vk in link

    def test_iterations_object(self, stack2):
        stack2.add_link(
            data_key=NAME2,
            filters=[None],
            x=["q2b", "Wave", "q2", "q3", "q5_1"],
            y=["@"],
            views=ViewMapper(["default", "cbase", "counts", "c%"]),
            weights=[None, "weight_a", "weight_b"]
        )
        # weighted an unweighted versions of all basic views are created
        view_keys = stack2.describe("view").index.tolist()
        assert all([vk in view_keys for vk in ITERATIONS_BASIC])

        # Create a ViewMapper with template
        xnets = ViewMapper()
        iterators = {
            "rel_to": [None, "y"],
            "axis": "x",
            "weights": [None, "weight_a"]}
        xnets.make_template(method="frequency", iterators=iterators)

        # add method and specify weight for link
        xnets.add_method(
            name="ever",
            kwargs={"axis": "x", "text": "Ever", "logic": [1, 2]})
        stack2.add_link(NAME2, x="q2b", y="@", views=xnets.subset("ever"),
                        weights="weight_b")
        view_keys = stack2.describe("view").index.tolist()
        assert all([vk in view_keys for vk in ITERATIONS_EVER_WGT])

        # add views to link using template iterations
        stack2.add_link(NAME2, x="q2b", y="@", views=xnets.subset("ever"))
        view_keys = stack2.describe("view").index.tolist()
        assert all([vk in view_keys for vk in ITERATIONS_EVER])

        # add multiple methods at the same time using the same iterators
        xnets.add_method(
            name="ever (multi test)",
            kwargs={"text": "Ever", "logic": [1, 2]})
        xnets.add_method(
            name="never (multi test)",
            kwargs={"text": "Never", "logic": [2, 3]})
        stack2.add_link(
            NAME2, x="q2b", y=["@"],
            views=xnets.subset(["ever (multi test)", "never (multi test)"]))
        view_keys = stack2.describe("view").index.tolist()
        assert all([vk in view_keys for vk in ITERATIONS_EVER_NEVER])

    # -------------------------------------------------------------------------
    # Quantipy views
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize("xk, weight, value, vk, meta", [
        ("age", None, *DEFAULT_INT_UNWGT),
        ("age", "weight_a", *DEFAULT_INT_WGT),
        ("weight_b", None, *DEFAULT_FLOAT_UNWGT),
        ("weight_b", "weight_a", *DEFAULT_FLOAT_WGT),
        ("gender", None, *DEFAULT_SINGLE_UNWGT),
        ("gender", "weight_a", *DEFAULT_SINGLE_WGT),
        ("q9", None, *DEFAULT_DELIMITED_SET_UNWGT),
        ("q9", "weight_a", *DEFAULT_DELIMITED_SET_WGT),
        ])
    def test_default_view(self, stack2, xk, weight, value, vk, meta):
        vm = ViewMapper(["default"])
        stack2.add_link(NAME2, x=xk, y="@", views=vm, weights=weight)

        link = stack2[NAME2][None][xk]["@"]
        view = link[vk]
        assert list(link.keys()) == [vk]
        assert round(np.nansum(view.dataframe.values), 6) == value
        assert view.meta() == meta

    def test_default_int_on_int(self, stack2):
        vm = ViewMapper(["default"])
        stack2.add_link(NAME2, x="age", y="age", views=vm)

        link = stack2[NAME2][None]["age"]["age"]
        view = link["x|default|:|||default"]
        df = view.dataframe
        assert view._xk == view._yk
        assert np.array_equal(
            df.xs("mean", axis=0, level=1).xs(26, axis=1, level=1).values,
            [[26]])

    def test_default_delimited_set_on_delimited_set(self, stack2):
        vm = ViewMapper(["default"])
        stack2.add_link(NAME2, x="q9", y="q9", views=vm, weights="weight_a")

        link = stack2[NAME2][None]["q9"]["q9"]
        view = link["x|default|:||weight_a|default"]
        df = view.dataframe
        assert np.allclose(
            df.xs("All", level=1, axis=0).values,
            DEFAULT_SET_ON_SET["x_all"])
        assert np.allclose(
            df.xs("All", level=1, axis=1).values,
            DEFAULT_SET_ON_SET["y_all"])
        index = df.index.get_level_values(1).tolist()
        assert index == DEFAULT_SET_ON_SET["x_axis"]

    @pytest.mark.parametrize("views, xk, yk, weight, expectations", [
        (["cbase", "rbase"], "weight_b", "gender", "weight_a", BASES_FLOAT_ON_SINGLE),  # noqa
        (["counts", "c%", "r%"], "q1", "q3", "weight_a", FREQ_SINGLE_ON_SET),
        (["counts", "c%", "r%"], "religion", "q1", None, FREQ_SINGLE_ON_SINGLE),  # noqa
        ])
    def test_frequency_view(self, stack2, views, xk, yk, weight, expectations):
        vm = ViewMapper(views)
        stack2.add_link(NAME2, x=xk, y=yk, views=vm, weights=weight)

        link = stack2[NAME2][None][xk][yk]

        for vk, exp in expectations.items():

            view = link[vk]
            df = view.dataframe
            assert np.allclose(
                df.ix[slice(*exp["slice"][0]), slice(*exp["slice"][1])].values,
                exp["values"])
            assert view.meta() == exp["meta"]

    def test_frequency_net(self, stack2):
        vm = ViewMapper(["counts"])
        vm.add_method(
            "net", "frequency", kwargs={"axis": "x", "logic": [1, 2, 3, 8761]})
        stack2.add_link(NAME2, x="q9", y="gender", views=vm, weights="weight_a")

        link = stack2[NAME2][None]["q9"]["gender"]
        vk = "x|f|x[{1,2,3,8761}]:||weight_a|net"
        df = link[vk].dataframe
        assert np.allclose(df.values, FREQUENCY_NET)

    def test_frequency_nps(self, stack2):
        vm = ViewMapper(["counts", "cbase"])
        vm.add_method(
            "nps", "frequency",
            kwargs={
                "rel_to": "y",
                "logic": [{"A": [1, 2, 3]}, {"B": [4]}, {"C": [5, 6]}],
                "axis": "x",
                "calc": {"score": ("A", sub, "C")}})
        stack2.add_link(NAME2, x="q1", y="q9", views=vm)

        link = stack2[NAME2][None]["q1"]["q9"]
        vk = "x|f.c:f|x[{1,2,3}],x[{4}],x[{5,6}],x[{1,2,3}-{5,6}]:|y||nps"
        df = link[vk].dataframe
        assert np.allclose(df.values, FREQUENCY_NPS)

    def test_desc_means_basic(self, stack2):
        xks = ["religion", "age", "weight_b"]
        yks = ["religion", "q9"]
        vm = ViewMapper(["mean"])
        stack2.add_link(NAME2, x=xks, y=yks, views=vm)
        vk = "x|d.mean|x:|||mean"
        for xk in xks:
            for yk in yks:
                link = stack2[NAME2][None][xk][yk]
                df = link[vk].dataframe
                assert np.allclose(df.ix[:, :15].values,
                                   DESC_MEANS_BASIC[(xk, yk)])

    def test_desc_means_complex(self, stack2):
        vm = ViewMapper(["mean"])
        kwargs = vm["mean"]["kwargs"]
        kwargs.update({
            "exclude": [2, 3],
            "rescale": {4: 300, 7: 900, 187: 555, 99: 900}})
        xks = ["religion"]
        yks = ["religion", "q9"]
        stack2.add_link(NAME2, x=xks, y=yks, views=vm, weights="weight_a")
        for xk in xks:
            for yk in yks:
                exp = DESC_MEANS_COMPLEX[(xk, yk)]
                link = stack2[NAME2][None][xk][yk]
                df = link[exp["vk"]].dataframe
                assert np.allclose(np.nan_to_num(df.ix[0, :15].values),
                                   exp["values"])

    @pytest.mark.parametrize("views, xk, yk, expectations", [
        (["counts_cumsum"], "q8", "gender", CUM_SUM_COUNTS),
        (["c%_cumsum"], "q5", "@", CUM_SUM_CPCTS)])
    def test_cumulative_sum_view(self, stack2, views, xk, yk, expectations):
        vm = ViewMapper(views)
        stack2.add_link(NAME2, x=xk, y=yk, views=vm)

        link = stack2[NAME2][None][xk][yk]

        for vk, values in expectations.items():
            df = link[vk].dataframe
            assert np.allclose(df.values, values)

    def test_base_effective(self, stack2):
        vm = ViewMapper(["ebase"])
        xks = ["gender"]
        yks = ["@", "q8"]
        stack2.add_link(NAME2, x=xks, y=yks, views=vm, weights="weight_a")

        vk = "x|f|x:||weight_a|ebase"
        for xk in xks:
            for yk in yks:
                link = stack2[NAME2][None][xk][yk]
                values = EFFECTIVE_BASE[(xk, yk)]
                df = link[vk].dataframe
                assert np.allclose(df.round(1).values.tolist(), values)

    def test_desc_other_source(self, stack2):
        vm1 = ViewMapper(
            template={
                "method": "descriptives",
                "kwargs": {"iterators": {"stats": ["mean", "stddev"]}}})
        vm1.add_method(
            "foreign_stats", kwargs={"source": "weight_a", "axis": "x"})
        vm2 = ViewMapper(
            template={
                "method": "coltests",
                "kwargs": {"metric": "means", "iterators": {"level": [0.10]}}})
        vm2.add_method("source_sig_test", kwargs={"text": "sigtest source"})
        stack2.add_link(
            NAME2, x="q2b", y="q2b", views=[vm1, vm2], weights="weight_b")

        link = stack2[NAME2][None]["q2b"]["q2b"]
        for vk, values in DESC_OTHER_SOURCE.items():
            df = link[vk].dataframe.replace(np.NaN, "NONE")
            assert df.values.tolist() == values

    @pytest.mark.parametrize((
        "xk, yk, weight, mean_name, mean_kwargs, sig_name, sig_kwargs, "
        "expectations"), COLTEST_MEAN_INPUT_EXPECT)
    def test_coltests_means(self, stack2, xk, yk, weight, mean_name,
                            mean_kwargs, sig_name, sig_kwargs, expectations):
        vm_mean = ViewMapper()
        vm_mean.add_method(mean_name, "descriptives", mean_kwargs)
        vm_sig = ViewMapper()
        vm_sig.add_method(sig_name, "coltests", sig_kwargs)
        stack2.add_link(
            NAME2, x=xk, y=yk, weights=weight, views=[vm_mean, vm_sig])

        link = stack2[NAME2][None][xk][yk]
        for vk, exp in expectations.items():
            view = link[vk]
            df = view.dataframe.replace(np.NaN, 'NONE')
            assert df.values.tolist() == exp["values"]
            assert view.is_meanstest == exp["level"]

    @pytest.mark.parametrize((
        "xk, yk, weight, sig_name, sig_kwargs, expectations"),
        COLTEST_PROPS_INPUT_EXPECT)
    def test_coltests_props(self, stack2, xk, yk, weight, sig_name, sig_kwargs,
                            expectations):
        vm = ViewMapper(['counts', 'cbase'])
        vm_sig = ViewMapper()
        vm_sig.add_method(sig_name, "coltests", sig_kwargs)
        stack2.add_link(NAME2, x=xk, y=yk, weights=weight, views=[vm, vm_sig])

        link = stack2[NAME2][None][xk][yk]
        for vk, exp in expectations.items():
            view = link[vk]
            df = view.dataframe.replace(np.NaN, 'NONE')
            assert df.values.tolist() == exp["values"]
            assert view.is_propstest == exp["level"]
            assert view.text == exp["text"]

    def test_coltests_means_props(self, stack2):
        vm = ViewMapper(['counts', 'mean'])
        print(vm)
        kwargs = {
            'test_total': True,
            "iterators": {'metric': ['props', 'means']}}
        vm.add_method("total_tests", "coltests", kwargs)
        stack2.add_link(
            NAME2, x="q7_1", y="q8", weights="weight_a", views=vm)

        link = stack2[NAME2][None]["q7_1"]["q8"]
        print (link.keys())
        for vk, values in COLTEST_MEANS_PROPS.items():
            view = link[vk]
            df = view.dataframe.replace(np.NaN, 'NONE')
            assert df.values.tolist() == values

    def test_coltests_props_on_net(self, stack2):
        vm = ViewMapper()
        kwargs = {  # noqa
            "logic": [
                {'Z': [1, 2, 3], 'expand': 'after', 'text': {'en-GB': 'some text1'}},
                {'A': [4, 5], 'text': {'en-GB': 'some text2'}},
                {'F': [6, 7, 8], 'expand': 'before', 'text': {'en-GB': 'some text3'}}],
            "axis": "x",
            "complete": True,
            "calc": {'my_calc': ('Z', sub, 'F')}
        }
        vm.add_method("blocknet", "frequency", kwargs)
        kwargs = {
            'test_total': True,
            "iterators": {'metric': ['props']}}
        vm.add_method("total_tests_blocks", "coltests", kwargs)
        stack2.add_link(NAME2, x="q7_1", y="q8", weights="weight_a", views=vm)
        link = stack2[NAME2][None]["q7_1"]["q8"]
        for vk, values in COLTEST_PROP_NET1.items():
            view = link[vk]
            df = view.dataframe.replace(np.NaN, 'NONE')
            assert df.values.tolist() == values

    def test_coltests_props_on_net2(self, stack2):
        stack2[NAME2].meta.reorder_values("q5", [98, 1, 2, 5, 4, 3, 97])
        stack2[NAME2].meta.reorder_values("q8", [98, 96, 5, 1, 2, 3, 4])
        vm = ViewMapper(["counts"])
        kwargs = {  # noqa
            "logic": [
                {'Z': [98, 97], 'expand': 'after', 'text': {'en-GB': 'some text1'}},
                {'A': [4, 1], 'expand': 'before', 'text': {'en-GB': 'some text2'}}],
            "axis": "x",
            "complete": True
        }
        vm.add_method("blocknet", "frequency", kwargs)
        kwargs = {
            'test_total': True,
            "level": 0.1}
        vm.add_method("tests", "coltests", kwargs)
        stack2.add_link(NAME2, x="q5_1", y="q8", views=vm)
        link = stack2[NAME2][None]["q5_1"]["q8"]
        for vk, exp in COLTEST_PROP_NET2.items():
            view = link[vk]
            df = view.dataframe.replace(np.NaN, 'NONE')
            if "values" in exp:
                assert df.values.tolist() == exp["values"]
            assert df.index.get_level_values(1).tolist() == exp["index"]
            assert df.columns.get_level_values(1).tolist() == exp["columns"]
