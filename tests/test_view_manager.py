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

from quantipy.core.view_generators.view_specs import ViewManager

class TestViewManager(unittest.TestCase):

    def _get_stack(self, unwgt=True, wgt=False, stats=True, nets=True, tests=True):
        dataset = self._get_dataset(500)
        batch = dataset.add_batch('viewmanager')
        x = ['q5', 'q8', 'q9']
        y = ['@', 'gender',  'q4', 'q8']
        batch.add_x(x)
        batch.add_y(y)
        if wgt:
            if unwgt:
                batch.set_weights([None, 'weight_a'])
            else:
                batch.set_weights('weight_a')
        if tests:
            batch.set_sigtests(levels=[0.05])
        stack = dataset.populate()
        basic_views = ['cbase', 'counts', 'c%', 'counts_sum', 'c%_sum']
        stack.aggregate(views=basic_views, verbose=False)
        stack.add_tests(verbose=False)
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

    def test_basics_no_sigtest(self):
        stack = self._get_stack(self, wgt=True, tests=False)
        vm = ViewManager(stack)

        # --------------------------------------------------------------------
        # cell items: counts, unweighted
        vm_views = vm.get_views(cell_items='c').group().views
        expected = ['x|f|x:|||cbase',
                    'x|f|:|||counts',
                    'x|f.c:f|x++:|||counts_cumsum',
                    'x|f.c:f|x:|||counts_sum']
        self.assertEqual(vm_views, expected)

        # --------------------------------------------------------------------
        # cell items: percentages, unweighted
        vm_views = vm.get_views(cell_items='p').group().views
        expected = ['x|f|x:|||cbase',
                    'x|f|:|y||c%',
                    'x|f.c:f|x++:|y||c%_cumsum',
                    'x|f.c:f|x:|y||c%_sum']
        self.assertEqual(vm_views, expected)

        # --------------------------------------------------------------------
        # cell items: counts + percentages, unweighted
        vm_views = vm.get_views(cell_items='cp').group().views
        expected = ['x|f|x:|||cbase',
                    ('x|f|:|||counts', 'x|f|:|y||c%'),
                    ('x|f.c:f|x++:|||counts_cumsum', 'x|f.c:f|x++:|y||c%_cumsum'),
                    ('x|f.c:f|x:|||counts_sum', 'x|f.c:f|x:|y||c%_sum')]
        self.assertEqual(vm_views, expected)

        # --------------------------------------------------------------------
        # cell items: counts, weighted, base 'auto'
        vm_views = vm.get_views(cell_items='c', weights='weight_a').group().views
        expected = ['x|f|x:||weight_a|cbase',
                    'x|f|:||weight_a|counts',
                    'x|f.c:f|x++:||weight_a|counts_cumsum',
                    'x|f.c:f|x:||weight_a|counts_sum']
        self.assertEqual(vm_views, expected)

        # --------------------------------------------------------------------
        # cell items: percentages, weighted, base 'auto'
        vm_views = vm.get_views(cell_items='p', weights='weight_a').group().views
        expected = ['x|f|x:||weight_a|cbase',
                    'x|f|:|y|weight_a|c%',
                    'x|f.c:f|x++:|y|weight_a|c%_cumsum',
                    'x|f.c:f|x:|y|weight_a|c%_sum']
        self.assertEqual(vm_views, expected)

        # --------------------------------------------------------------------
        # cell items: counts + percentages, weighted, base 'auto'
        vm_views = vm.get_views(cell_items='cp', weights='weight_a').group().views
        expected = ['x|f|x:||weight_a|cbase',
                    ('x|f|:||weight_a|counts', 'x|f|:|y|weight_a|c%'),
                    ('x|f.c:f|x++:||weight_a|counts_cumsum', 'x|f.c:f|x++:|y|weight_a|c%_cumsum'),
                    ('x|f.c:f|x:||weight_a|counts_sum', 'x|f.c:f|x:|y|weight_a|c%_sum')]
        self.assertEqual(vm_views, expected)

        # --------------------------------------------------------------------
        # cell items: counts, weighted, base 'both'
        vm_views = vm.get_views(cell_items='c', weights='weight_a',
                                bases='both').group().views
        expected = ['x|f|x:|||cbase', 'x|f|x:||weight_a|cbase',
                    'x|f|:||weight_a|counts',
                    'x|f.c:f|x++:||weight_a|counts_cumsum',
                    'x|f.c:f|x:||weight_a|counts_sum']
        self.assertEqual(vm_views, expected)

        # --------------------------------------------------------------------
        # cell items: percentages, weighted, base 'both'
        vm_views = vm.get_views(cell_items='p', weights='weight_a',
                                bases='both').group().views
        expected = ['x|f|x:|||cbase', 'x|f|x:||weight_a|cbase',
                    'x|f|:|y|weight_a|c%',
                    'x|f.c:f|x++:|y|weight_a|c%_cumsum',
                    'x|f.c:f|x:|y|weight_a|c%_sum']
        self.assertEqual(vm_views, expected)

        # --------------------------------------------------------------------
        # cell items: counts + percentages, weighted, base 'both'
        vm_views = vm.get_views(cell_items='cp', weights='weight_a',
                                bases='both').group().views
        expected = ['x|f|x:|||cbase', 'x|f|x:||weight_a|cbase',
                    ('x|f|:||weight_a|counts', 'x|f|:|y|weight_a|c%'),
                    ('x|f.c:f|x++:||weight_a|counts_cumsum', 'x|f.c:f|x++:|y|weight_a|c%_cumsum'),
                    ('x|f.c:f|x:||weight_a|counts_sum', 'x|f.c:f|x:|y|weight_a|c%_sum')]
        self.assertEqual(vm_views, expected)


    def test_basics_with_sigtest(self):
        stack = self._get_stack(self, wgt=True, tests=True)
        vm = ViewManager(stack, tests=['0.05'])

        # --------------------------------------------------------------------
        # cell items: counts, unweighted
        vm_views = vm.get_views(cell_items='c').group().views
        expected = ['x|f|x:|||cbase',
                    ('x|f|:|||counts', 'x|t.props.Dim.05|:|||significance'),
                    'x|f.c:f|x++:|||counts_cumsum',
                    'x|f.c:f|x:|||counts_sum']
        self.assertEqual(vm_views, expected)

        # --------------------------------------------------------------------
        # cell items: cp, weighted, 'both' bases
        vm_views = vm.get_views(cell_items='cp', weights='weight_a',
                                bases='both').group().views
        expected = ['x|f|x:|||cbase', 'x|f|x:||weight_a|cbase',
                    ('x|f|:||weight_a|counts',
                     'x|f|:|y|weight_a|c%',
                     'x|t.props.Dim.05|:||weight_a|significance'),
                    ('x|f.c:f|x++:||weight_a|counts_cumsum',
                     'x|f.c:f|x++:|y|weight_a|c%_cumsum'),
                   ('x|f.c:f|x:||weight_a|counts_sum',
                    'x|f.c:f|x:|y|weight_a|c%_sum')]
        self.assertEqual(vm_views, expected)
