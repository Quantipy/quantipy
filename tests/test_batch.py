
import pytest
from collections import OrderedDict
from quantipy.__imports__ import *  # noqa
from quantipy import (
    Batch,
    DataSet,
    Meta,
)

NAME = "Example Data (A)"
DATA = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
BNAME = "test"

@pytest.fixture(scope="class")
def dataset():
    return DataSet.from_quantipy(NAME, DATA)

@pytest.fixture(scope="class")
def batch(dataset):
    return Batch(dataset, BNAME, "c", "weight", 0.05)

@pytest.fixture(scope="class")
def b_meta(batch):
    return batch.dataset._meta["sets"]["batches"][BNAME]

def _assert_raise(caplog, err, msg, func, *args, **kwargs):
    caplog.clear()
    with pytest.raises(err):
        func(*args, **kwargs)
    assert caplog.records[-1].message == msg

def _assert_warn(caplog, msg, func, *args, **kwargs):
    caplog.clear()
    func(*args, **kwargs)
    assert caplog.records[-1].message == msg


class TestBatch:

    def test_init(self, batch, b_meta):
        assert batch.name == BNAME
        assert batch.cell_items == ["c"]
        assert batch.weights == ["weight"]
        for key in b_meta.keys():
            trans = {
                "xks": "downbreaks",
                "yks": "crossbreaks",
            }
            assert b_meta[key] == getattr(batch, trans.get(key, key))

    def test_add_get_batch(self, batch, dataset, caplog):
        msg = "Batch 'name' must not contain '-'!"
        _assert_raise(caplog, ValueError, msg, dataset.add_batch, "new-batch")
        dataset.add_batch("new_batch")
        batches = ["old_add_batch", "old_batch", BNAME, "new_batch"]
        assert dataset.batches == batches
        b = dataset.get_batch("new_batch")
        assert isinstance(b, Batch)
        b.as_addition(BNAME)
        assert BNAME in b.mains
        b.remove()
        assert b.name not in batch.additions

    def test_old_batches(self, dataset, caplog):
        old = dataset.get_batch("old_batch")
        add = dataset.get_batch("old_add_batch")
        assert isinstance(old, Batch)
        assert isinstance(add, Batch)
        assert old.additions == ["old_add_batch"]
        assert add.mains == ["old_batch"]
        assert isinstance(old.meta, Meta)
        assert isinstance(add.meta, Meta)
        assert old.meta.get_text("gender") == "Gender"
        assert add.meta.get_text("q6") == "fake label"
        msg = "'{}' is already included!".format(BNAME)
        _assert_raise(
            caplog, ValueError, msg, old.__setattr__, "name", BNAME)
        old.name = "old_b_update"
        add.name = "old_add_b_update"
        assert old.additions == ["old_add_b_update"]
        assert add.mains == ["old_b_update"]
        old.remove()
        assert add.mains == []

    def test_combine_batches(self, batch):
        b2 = batch.clone("clone", ("male only", {"gender": 1}), True)
        assert isinstance(b2, Batch)
        assert "maleonly" in batch.dataset
        assert "maleonly" in batch.meta
        assert "maleonly" in b2.meta
        assert b2.mains == [BNAME]
        assert batch.additions == ["clone"]
        b2.as_main()
        assert b2.mains == []
        assert batch.additions == []
        b2.remove()

    def test_downbreaks(self, batch, b_meta, caplog):
        msg = "Not found in 'Meta': 'Q1'"
        _assert_raise(caplog, KeyError, msg, batch.add_downbreak, "Q1")
        batch.downbreaks = ["q1", "q2b"]
        batch.extend_downbreaks({"q2b": "q2"})
        batch.extend_downbreaks(["q3", "q4", "q5"])
        xks = [
            "q1", "q2", "q2b", "q3", "q4", "q5", "q5_1", "q5_2", "q5_3",
            "q5_4", "q5_5", "q5_6"]
        assert batch.downbreaks == b_meta["xks"] == xks
        x_y_map = [(x, ["@"]) for x in xks]
        assert batch.x_y_map == b_meta["x_y_map"] == x_y_map

    def test_hide_empty(self, batch, b_meta, dataset):
        dataset.copy("gender", "gender_empty", False)
        dataset.copy("q5", "q5_empty", False)
        dataset.copy("q6", "q6_empty", False)
        dataset["q6_empty_1"] = 1
        xks = ["gender", "gender_empty", "q5", "q5_empty", "q6", "q6_empty"]
        batch.downbreaks = xks
        assert batch.downbreaks == dataset.unroll(xks, both="all")
        batch.hide_empty()
        assert batch.downbreaks == dataset.unroll([
            "gender", "q5", "q6", "q6_empty", "q6_empty_1"], both="all")

    def test_crossbreaks(self, batch, b_meta, caplog):
        batch.downbreaks = ["q1", "q2b", "q3", "q4", "q5"]
        msg = "Not found in 'Meta': 'Q1'"
        _assert_raise(
            caplog, KeyError, msg, batch.__setattr__, "crossbreaks", "Q1")
        batch.crossbreaks = ["gender", "q2b"]
        for total, yks in zip([False, True],
                              [["gender", "q2b"], ["@", "gender", "q2b"]]):
            batch.total = total
            assert batch.crossbreaks == b_meta["yks"] == ["gender", "q2b"]
            x_y_map = [
                (x, ["@"]) if batch.dataset.is_array(x) else (x, yks)
                for x in batch.downbreaks]
            assert batch.x_y_map == b_meta["x_y_map"] == x_y_map
        batch.extend_crossbreaks("age", "q1")
        assert batch.x_y_map[0][1][-1] == "age"
        batch.extend_crossbreaks("age")
        for x, y in batch.x_y_map:
            if not (batch.dataset.is_array(x) or x == "@"):
                assert y[-1] == "age"
        batch.replace_crossbreaks(["gender"], "q3")
        assert batch.x_y_map[2][1] == ["@", "gender"]

    def test_build_info(self, batch, b_meta, caplog):
        msg = "'build_info' must be of type 'dict'!"
        _assert_raise(
            caplog, TypeError, msg, batch.__setattr__, "build_info", "info")
        b_info = {"key": "value"}
        batch.build_info = b_info
        assert batch.build_info == b_meta["build_info"] == b_info

    def test_cell_items(self, batch, b_meta, caplog):
        msg = "'ci' cell items must be either 'c', 'p' or 'cp'."
        _assert_raise(
            caplog, ValueError, msg, batch.__setattr__, "cell_items", "counts")
        ci = ["c", "cp"]
        batch.cell_items = ci
        assert batch.cell_items == b_meta["cell_items"] == ci

    def test_text_key(self, batch, b_meta, caplog):
        msg = "'tk' is not a valid textkey!"
        _assert_raise(
            caplog, ValueError, msg, batch.__setattr__, "text_key", "tk")
        tk = "en-GB"
        batch.text_key = tk
        assert batch.text_key == b_meta["meta"].text_key == tk

    def test_unweighted_counts(self, batch, b_meta, caplog):
        msg = "'unweighted_counts' must be of type 'bool'!"
        _assert_raise(
            caplog, TypeError, msg, batch.__setattr__,
            "unweighted_counts", "yes")
        unwgt = True
        batch.unweighted_counts = unwgt
        assert batch.unweighted_counts == b_meta["unweighted_counts"] == unwgt

    def test_variables(self, batch, b_meta, caplog):
        msg = "Not found in 'Meta': 'bla'"
        _assert_raise(
            caplog, KeyError, msg, batch.__setattr__, "variables", "bla")
        variables = ["q2", "q5"]
        batch.variables = variables
        var = ["q2", "q5", "q5_1", "q5_2", "q5_3", "q5_4", "q5_5", "q5_6"]
        assert batch.variables == b_meta["variables"] == var

    def test_weights(self, batch, b_meta, caplog):
        msg = "['weight_a', 'new_w'] contains invalid variable name."
        _assert_raise(
            caplog, ValueError, msg, batch.__setattr__,
            "weights", ['weight_a', 'new_w'])
        batch.weights = None
        assert batch.weights == b_meta["weights"] == [None]
        weights = ['weight_a', 'weight_b']
        batch.weights = weights
        assert batch.weights == b_meta["weights"] == weights

    def test_filter(self, batch, b_meta, dataset, caplog):
        msg = "'gender' is not a valid filter variable."
        _assert_raise(
            caplog, ValueError, msg, batch.__setattr__, "filter", "gender")
        msg = (
            "'filter_def' must either be name of a filter variable or"
            "tuple in form  of (filter_name, filter_logic).")
        _assert_raise(
            caplog, TypeError, msg, batch.__setattr__, "filter", {"age": 18})
        batch.filter = None
        assert batch.get_codes_in_data("gender") == [1, 2]
        assert batch.filter == b_meta["filter"] == None
        assert batch.sample_size == dataset._data.shape[0]

        batch.filter = ("femaleonly", {"gender": 2})
        assert batch.filter == b_meta["filter"] == "femaleonly"
        assert batch.sample_size == 4303
        assert batch.get_codes_in_data("gender") == [2]
        assert batch.get_codes_in_data("q2b") == [1, 2, 3]

        batch.filter = "maleonly"
        assert batch.filter == b_meta["filter"] == "maleonly"
        assert batch.sample_size == 3952
        batch.extend_filter({"age": is_lt(30)}, "q1")
        assert batch.extended_filters_per_x["q1"] == "maleonly_q1"
        assert len(dataset.manifest_filter("maleonly_q1")) == 1402

    def test_clone_filtered_batch_with_oe(self, batch, b_meta, caplog):
        msg = "'{}' is already included!".format(BNAME)
        _assert_raise(caplog, ValueError, msg, batch.clone, BNAME)
        batch.set_verbatims("age")
        oe = {
            "title": "open ends",
            "filter": batch.filter,
            'columns': ["age"],
            'break_by': [],
            'incl_nan': False,
            'drop_empty': True,
            'replace': {}}
        assert batch.verbatims == b_meta["verbatims"] == [oe]
        new = "NEW"
        nb = batch.clone(new, "femaleonly", False)
        oe["filter"] = "femaleonly"
        assert nb.verbatims == [oe]
        assert nb.filter == "femaleonly"
        nb.remove()

    def test_sigproperties(self, batch, b_meta, caplog):
        batch.total = True
        batch.set_sigproperties(None)
        key = "siglevels"
        assert batch.sigproperties[key] == b_meta["sigproperties"][key] == []
        batch.total = False
        caplog.clear()
        batch.set_sigproperties("0.05")
        msg = "Cannot add sigtests without total column."
        assert caplog.records[-1].message == msg

    def test_sections(self, batch, b_meta):
        batch.downbreaks = ["q1", "q2", "q3", "q4", "q5"]
        batch.set_section("q2", "all except q1")
        assert batch.sections["all except q1"] == batch.downbreaks[1:]
        assert b_meta["sections"]["q2"] == "all except q1"
        for section in batch.sections.keys():
            batch.del_section(section)
        assert not batch.sections and not b_meta["sections"]

    def test_total(self, batch, b_meta, caplog):
        msg = "'total' must be of type 'bool'."
        _assert_raise(caplog, TypeError, msg, batch.__setattr__, "total", "no")
        batch.total = True
        for x, y in batch.x_y_map:
            if not x == "@":
                assert y[0] == "@"

    def test_leveled(self, batch, dataset, b_meta, caplog):
        msg = "Can only level arrays!"
        _assert_raise(
            caplog, TypeError, msg, batch.__setattr__, "leveled", ["q1", "q6"])
        batch.total = True
        batch.downbreaks = ["q1", "q6"]
        batch.crossbreaks = ["gender"]
        batch.leveled = ["q6"]
        assert batch.leveled == b_meta["leveled"] == ["q6"]
        assert batch.x_y_map[2] == ("q6_level", ["@", "gender"])
        assert "q6_level" in dataset
        assert "q6_level" in batch.meta

    def test_transposed(self, batch, b_meta, caplog):
        msg = "Can only transpose arrays!"
        _assert_raise(
            caplog, TypeError, msg, batch.__setattr__, "transposed",
            ["q1", "q6"])
        batch.total = True
        batch.downbreaks = ["q1", "q5"]
        batch.crossbreaks = ["gender"]
        batch.transposed = ["q5"]
        assert batch.transposed == b_meta["transposed"] == ["q5"]
        assert batch.x_y_map[2] == ("@", ["q5"])

    def test_hiding(self, batch, dataset):
        batch.total = True
        batch.downbreaks = ["q5"]
        batch.crossbreaks = ["gender"]
        batch.set_hiding("q5", [1], axis="x", hide_values=False)
        assert batch.meta.get_rules("q5")["dropx"] == {'values': ['q5_1']}
        assert dataset.get_rules("q5").get("dropx") is None
        for x, y in batch.x_y_map:
            assert x not in ["q5_1"]

    def test_skip_items(self, batch, b_meta, caplog):
        msg = "Can only skip array items!"
        _assert_raise(
            caplog, TypeError, msg, batch.__setattr__, "skip_items",
            ["q1", "q6"])
        batch.downbreaks = ["q1", "q5"]
        batch.crossbreaks = ["gender"]
        batch.skip_items = ["q5"]
        assert batch.skip_items == b_meta["skip_items"] == ["q5"]
        for x, y in batch.x_y_map:
            assert x not in batch.dataset.get_sources("q5")

    def test_set_verbatims(self, batch, b_meta, caplog):
        batch.del_verbatims()
        add = batch.clone("add", as_addition=True)
        msg = "Cannot add verbatims to additional batches."
        _assert_raise(
            caplog, NotImplementedError, msg, add.set_verbatims, [])
        msg = "duplicates '['q1']' included in oe and break_by."
        _assert_raise(
            caplog, ValueError, msg, batch.set_verbatims, ['q1', 'age'],
            break_by="q1")
        msg = "'title' must be of type string."
        _assert_raise(
            caplog, TypeError, msg, batch.set_verbatims, 'age',
            title=["age"])
        msg = "'replacements' must be of type dict."
        _assert_raise(
            caplog, TypeError, msg, batch.set_verbatims, 'age',
            replacements=("__NA__", ""))
        msg = "'q1' is not a valid filter variable."
        _assert_raise(
            caplog, ValueError, msg, batch.set_verbatims, 'age',
            filter_by="q1")

        batch.set_verbatims("age", title="oe1", replacements={"NA": ""},
                            filter_by=("oe_f", {"gender": 1}))
        batch.set_verbatims("q1", title="oe2", replacements={"NA": ""},
                            filter_by="oe_f")
        expect = [
            {
                'title': "oe1",
                'filter': "oe_f",
                'columns': ["age"],
                'break_by': [],
                'incl_nan': False,
                'drop_empty': True,
                'replace': {"en-GB": {"NA": ""}}
            },
            {
                'title': "oe2",
                'filter': "oe_f",
                'columns': ["q1"],
                'break_by': [],
                'incl_nan': False,
                'drop_empty': True,
                'replace': {"en-GB": {"NA": ""}}
            }
        ]
        assert batch.verbatims == b_meta["verbatims"] == expect
        batch.set_verbatims("age", title="oe1", break_by="q1")
        assert len(batch.verbatims) == 2
        assert batch.verbatims[0]["break_by"] == ["q1"]

    def test_y_on_y(self, batch, b_meta, caplog):
        batch.del_y_on_y()
        msg = "'name' attribute for add_y_on_y must be a str!"
        _assert_raise(caplog, TypeError, msg, batch.set_y_on_y, ["BACK"])
        msg = "'q1' is not a valid filter variable."
        _assert_raise(caplog, ValueError, msg, batch.set_y_on_y, "BACK", "q1")

        batch.set_y_on_y("BACK1")
        batch.set_y_on_y("BACK2", ("yony_f", {"gender": 1}))
        batch.set_y_on_y("BACK3", "yony_f")
        expect = ["BACK1", "BACK2", "BACK3"]
        assert batch.y_on_y == b_meta["y_on_y"] == expect
        expect = {
            "BACK1": batch.filter,
            "BACK2": "yony_f",
            "BACK3": "yony_f",
        }
        assert batch.y_filter_map == b_meta["y_filter_map"] == expect
        batch.del_y_on_y()
        assert batch.y_on_y == []
        assert batch.y_filter_map == {}

    def test_to_dataset(self, dataset):
        b = dataset.add_batch("new_ds")
        b.downbreaks = ["q1", "q5"]
        b.crossbreaks = ["q2b"]
        b.filter = ("filter_var", {"age": is_lt(30)})
        b.set_hiding("q5", 1, axis="x", hide_values=False)
        b.set_hiding("q1", 1, axis="x")
        b.set_verbatims("q8a", replacements={"__NA__": ""})
        b.weight = "weight_a"
        b2 = dataset.add_batch("new_ds2")
        b2.downbreaks = ["gender", "q6"]
        b2.as_addition("new_ds")
        ds = b.to_dataset(misc="RecordNo", additions=False, apply_edits=True)
        expect = ['RecordNo', 'q1', 'q2b', 'q5', 'q8a', 'filter_var']
        assert ds.variables() == expect
        assert ds._data.shape == (2965, 11)
        assert ds.get_sources("q5") == ["q5_2", "q5_3", "q5_4", "q5_5", "q5_6"]
        ds = b.to_dataset(misc="RecordNo", additions="sort_within",
                          mode=['x', 'y', 'v', 'oe', 'w'])
        expect = [
            'RecordNo', 'q1', 'q2b', 'q5', 'q8a', 'gender', 'q6']
        assert ds.variables() == expect
        assert ds._data.shape == (8255, 15)
        b2.filter = ("filter_var2", {"gender": 1})
        ds = b.to_dataset(misc="RecordNo", additions="sort_between")
        expect = [
            'RecordNo', 'gender', 'q1', 'q2b', 'q5', 'q6', 'q8a', 'filter_var',
            'filter_var2']
        assert ds.variables() == expect
        assert ds._data.shape == (5515, 17)

    def test_deprecation(self, dataset, caplog):
        b = dataset.add_batch("deprecation")
        WARN = (
            "This method will be deprecated soon.\n"
            "Please use property setter of '{}' instead.").format
        _assert_warn(caplog, WARN("batch.name"), b.rename, "deprecated")
        _assert_warn(caplog, WARN("batch.text_key"), b.set_language, "de-DE")
        _assert_warn(caplog, WARN("batch.cell_items"), b.set_cell_items, "cp")
        _assert_warn(caplog, WARN("batch.unweighted_counts"),
                     b.set_unweighted_counts, True)
        _assert_warn(caplog, WARN("batch.weights"), b.set_weights, None)
        _assert_warn(caplog, WARN("batch.variables"), b.add_variables, ["q1"])
        _assert_warn(caplog, WARN("batch.downbreaks"), b.add_downbreak, ["q1"])
        _assert_warn(caplog, WARN("batch.leveled"), b.level, ["q6"])
        _assert_warn(caplog, WARN("batch.crossbreaks"), b.add_crossbreak,
                     ["q1"])
        _assert_warn(caplog, WARN("batch.transposed"), b.transpose, ["q6"])
        _assert_warn(caplog, WARN("batch.total"), b.add_total, True)
        _assert_warn(caplog, WARN("batch.filter"), b.set_filter, None)
        b.remove()
