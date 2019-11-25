import unittest
import os.path
import quantipy as qp
import copy
from quantipy.core.tools.dp.dimensions.dimlabels import (
    qp_dim_languages,
    DimLabels)


def _get_dataset():
    path = os.path.dirname(os.path.abspath(__file__)) + '/'
    name = 'Example Data (A)'
    casedata = '{}.csv'.format(name)
    metadata = '{}.json'.format(name)
    dataset = qp.DataSet(name, False)
    dataset.set_verbose_infomsg(False)
    dataset.set_verbose_errmsg(False)
    dataset.read_quantipy(path+metadata, path+casedata)
    return dataset

def loop_for_text_objs(obj, func, name=None):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == 'text' and isinstance(v, dict):
                func(copy.deepcopy(v), name)
            else:
                loop_for_text_objs(v, func, k)
    elif isinstance(obj, list):
        for o in obj:
            loop_for_text_objs(o, func, name)

class TestDimLabels(unittest.TestCase):

    def _test_dimlabels(self, x, name):
        d_labels = DimLabels(name, 'de-DE')
        d_labels.add_text(x)
        languages = {v: k for k, v in qp_dim_languages.items()}
        for lab in d_labels.labels:
            t1 = lab.default_lan == 'DEU'
            if lab.labeltype:
                t2 = lab.text == x[lab.labeltype].pop(languages[lab.language])
                if not x[lab.labeltype]: del x[lab.labeltype]
            else:
                t2 = lab.text == x.pop(languages[lab.language])
            self.assertTrue(t1 and t2)
        self.assertEqual({}, x)

    def test_dimlabels(self):
        dataset = _get_dataset()
        dataset.set_variable_text('q1', 'test label', 'en-GB', 'x')
        dataset.set_value_texts('q2b', dict(dataset.values('q2b')), 'sv-SE')
        loop_for_text_objs(dataset._meta, self._test_dimlabels)

