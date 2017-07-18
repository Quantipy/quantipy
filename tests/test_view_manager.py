import unittest
import os.path
import numpy as np
import pandas as pd
import quantipy as qp

from quantipy.core.tools.view.logic import (
    has_any, has_all, has_count,
    not_any, not_all, not_count,
    is_lt, is_ne, is_gt,
    is_le, is_eq, is_ge,
    union, intersection)

from quantipy.core.tools.dp.prep import frange
freq = qp.core.tools.dp.prep.frequency
cross = qp.core.tools.dp.prep.crosstab

class TestViewManager(unittest.TestCase):


    def _get_stack(self, wgt=False, stats=True, nets=True, tests=True):
        dataset = self._get_dataset(500)
        batch = dataset.add_batch('viewmanager')
        x = ['q5', 'q8', 'q9']
        y = ['@', 'gender',  'q4', 'q8']
        batch.add_x(x)
        batch.add_y(y)
        if wgt:
            batch.set_weights('weight_a')
        stack = dataset.populate()
        basic_views = ['cbase', 'counts', 'c%', 'counts_sum', 'c%_sum']
        stack.aggregate(views=basic_views, verbose=False)
        return stack

    def _get_dataset(self, cases=None):
        path = os.path.dirname(os.path.abspath(__file__)) + '/'
        name = 'Example Data (A)'
        casedata = '{}.csv'.format(name)
        metadata = '{}.json'.format(name)
        dataset = qp.DataSet(name, False)
        dataset.set_verbose_infomsg(False)
        dataset.read_quantipy(path+metadata, path+casedata)
        if cases:
            dataset._data = dataset._data.head(cases)
        return dataset



    def test_c_basics_no_test(self):
        stack = self._get_stack(self, True)

