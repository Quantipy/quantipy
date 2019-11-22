
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

def _assert_exists(name, suffix=[]):
    if not isinstance(suffix, list):
        suffix = [suffix]
    for suf in suffix:
        path = os.path.join(DATA, "{}.{}".format(name, suf))
        assert os.path.exists(path)

def _assert_raise(caplog, err, msg, func, *args, **kwargs):
    caplog.clear()
    with pytest.raises(err):
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

    def test_add_get_batch(self, dataset, caplog):
        msg = "Batch 'name' must not contain '-'!"
        _assert_raise(caplog, ValueError, msg, dataset.add_batch, "new-batch")
        dataset.add_batch("new_batch")
        assert dataset.batches == [BNAME, "new_batch"]
        b = dataset.get_batch("new_batch")
        assert isinstance(b, Batch)
        b.remove()

    def test_combine_batches(self, batch):
        b2 = batch.clone("clone", ("male only", {"gender": 1}), True)
        assert isinstance(b2, Batch)
        assert "maleonly" in batch.dataset
        assert "maleonly" in batch.meta
        assert "maleonly" in b2.meta
        assert b2.mains == [BNAME]
        assert batch.additions == ["clone"]
        b2.remove()
        assert batch.additions == []

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

    def test_crossbreaks(self, batch, b_meta, caplog):
        msg = "Not found in 'Meta': 'Q1'"
        _assert_raise(caplog, KeyError, msg, batch.add_crossbreak, "Q1")
        batch.crossbreaks = ["gender", "q2b"]
        for total, yks in zip([False, True],
                              [["gender", "q2b"], ["@", "gender", "q2b"]]):
            batch.total = total
            assert batch.crossbreaks == b_meta["yks"] == ["gender", "q2b"]
            x_y_map = [
                (x, ["@"]) if batch.dataset.is_array(x) else (x, yks)
                for x in batch.downbreaks]
            assert batch.x_y_map == b_meta["x_y_map"] == x_y_map