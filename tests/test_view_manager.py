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

from ViewManager_expectations import EXPECT as E

class TestViewManager(unittest.TestCase):

    def _get_stack(self, unwgt=True, wgt=True, stats=True, nets=True, tests=True):
        dataset = self._get_dataset(500)

        x = ['q5', 'q8', 'q9']
        y = ['@', 'gender',  'q4', 'q8']
        if wgt and unwgt:
            w = [None, 'weight_a']
        elif wgt:
            w = ['weight_a']
        else:
            w = None
        batch = dataset.add_batch('viewmanager', weights=w, tests=[0.05] if tests else None)
        batch.add_x(x)
        batch.add_y(y)
        stack = dataset.populate()
        basic_views = ['cbase', 'counts', 'c%', 'counts_sum', 'c%_sum']
        stack.aggregate(views=basic_views, verbose=False)
        if nets:
            stack.add_nets(['q5', 'q9'], [{'Top3': [1, 2, 3]}], verbose=False)
            stack.add_nets(['q8'], [{'Top2': [1, 2]}], expand='after', verbose=False)
        if stats:
            stack.add_stats(x, ['mean'], rescale={1:100, 2:50, 3:0}, verbose=False)
            stack.add_stats('q5', ['mean', 'stddev'], custom_text='stat2', verbose=False)
        if tests:
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

    # cell_items: c/p/cp
    # basics:     b
    # nets:       n
    # stats:      s
    # tests:      t
    # weights:    w
    # bases:      auto/both/wgt/unwgt

    ############################# test_basics ############################
    # --------------------------------------------------------------------
    # cell items: counts, unweighted
    def test_vm_c_b(self):
        stack = self._get_stack(self)
        path_stack = './tests/stack_vm.stack'
        stack.save(path_stack)
        vm = ViewManager(stack, basics=True, nets=False, stats=None, tests=None)
        vm_views = vm.get_views(cell_items='c', weights=None, bases='auto').group().views
        self.assertEqual(vm_views, E['c_b'])


    # --------------------------------------------------------------------
    # cell items: percentages, unweighted
    def test_vm_p_b(self):
        path_stack = './tests/stack_vm.stack'
        stack = qp.Stack('viewmanager').load(path_stack, load_cache=True)
        vm = ViewManager(stack, basics=True, nets=False, stats=None, tests=None)
        vm_views = vm.get_views(cell_items='p', weights=None, bases='auto').group().views
        self.assertEqual(vm_views, E['p_b'])

    # --------------------------------------------------------------------
    # cell items: counts + percentages, unweighted
    def test_vm_cp_b(self):
        path_stack = './tests/stack_vm.stack'
        stack = qp.Stack('viewmanager').load(path_stack, load_cache=True)
        vm = ViewManager(stack, basics=True, nets=False, stats=None, tests=None)
        vm_views = vm.get_views(cell_items='cp', weights=None, bases='auto').group().views
        self.assertEqual(vm_views, E['cp_b'])
    # --------------------------------------------------------------------
    # cell items: counts, weighted, base 'auto'
    def test_vm_c_b_w_auto(self):
        path_stack = './tests/stack_vm.stack'
        stack = qp.Stack('viewmanager').load(path_stack, load_cache=True)
        vm = ViewManager(stack, basics=True, nets=False, stats=None, tests=None)
        vm_views = vm.get_views(cell_items='c', weights='weight_a', bases='auto').group().views
        self.assertEqual(vm_views, E['c_b_w_auto'])

    # --------------------------------------------------------------------
    # cell items: percentages, weighted, base 'auto'
    def test_vm_p_b_w_auto(self):
        path_stack = './tests/stack_vm.stack'
        stack = qp.Stack('viewmanager').load(path_stack, load_cache=True)
        vm = ViewManager(stack, basics=True, nets=False, stats=None, tests=None)
        vm_views = vm.get_views(cell_items='p', weights='weight_a', bases='auto').group().views
        self.assertEqual(vm_views, E['p_b_w_auto'])

    # --------------------------------------------------------------------
    # cell items: counts + percentages, weighted, base 'auto'
    def test_vm_cp_b_w_auto(self):
        path_stack = './tests/stack_vm.stack'
        stack = qp.Stack('viewmanager').load(path_stack, load_cache=True)
        vm = ViewManager(stack, basics=True, nets=False, stats=None, tests=None)
        vm_views = vm.get_views(cell_items='cp', weights='weight_a', bases='auto').group().views
        self.assertEqual(vm_views, E['cp_b_w_auto'])
    # --------------------------------------------------------------------
    # cell items: counts, unweighted, base 'both'
    def test_vm_c_b_w_both(self):
        path_stack = './tests/stack_vm.stack'
        stack = qp.Stack('viewmanager').load(path_stack, load_cache=True)
        vm = ViewManager(stack, basics=True, nets=False, stats=None, tests=None)
        vm_views = vm.get_views(cell_items='c', weights='weight_a', bases='both').group().views
        self.assertEqual(vm_views, E['c_b_w_both'])

    # --------------------------------------------------------------------
    # cell items: percentages, unweighted, base 'both'
    def test_vm_p_b_w_both(self):
        path_stack = './tests/stack_vm.stack'
        stack = qp.Stack('viewmanager').load(path_stack, load_cache=True)
        vm = ViewManager(stack, basics=True, nets=False, stats=None, tests=None)
        vm_views = vm.get_views(cell_items='p', weights='weight_a', bases='both').group().views
        self.assertEqual(vm_views, E['p_b_w_both'])
        path_cache = './tests/stack_vm.cache'
        os.remove(path_stack)
        os.remove(path_cache)

    # --------------------------------------------------------------------
    # cell items: counts + percentages, unweighted, base 'both'
    def test_vm_cp_b_w_both(self):
        path_stack = './tests/stack_vm.stack'
        stack = qp.Stack('viewmanager').load(path_stack, load_cache=True)
        vm = ViewManager(stack, basics=True, nets=False, stats=None, tests=None)
        vm_views = vm.get_views(cell_items='cp', weights='weight_a', bases='both').group().views
        self.assertEqual(vm_views, E['cp_b_w_both'])


    ############################ test_complex ############################
    # --------------------------------------------------------------------
    # counts + percentage, basics, nets, weighted, base both
    def test_vm_cp_b_n_w_both(self):
        path_stack = './tests/stack_vm.stack'
        stack = qp.Stack('viewmanager').load(path_stack, load_cache=True)
        vm = ViewManager(stack, basics=True, nets=True, stats=None, tests=None)
        vm_views = vm.get_views(cell_items='cp', weights='weight_a', bases='both').group().views
        self.assertEqual(vm_views, E['cp_b_n_w_both'])

    # --------------------------------------------------------------------
    # percentage, basics, nets, weighted, base both
    def test_vm_p_b_n_w_both(self):
        path_stack = './tests/stack_vm.stack'
        stack = qp.Stack('viewmanager').load(path_stack, load_cache=True)
        vm = ViewManager(stack, basics=True, nets=True, stats=None, tests=None)
        vm_views = vm.get_views(cell_items='p', weights='weight_a', bases='both').group().views
        self.assertEqual(vm_views, E['p_b_n_w_both'])

    # # --------------------------------------------------------------------
    # # counts + percentage, basics, stats, weighted, base both
    # def test_vm_cp_b_s_w_both(self):
    #     path_stack = './tests/stack_vm.stack'
    #     stack = qp.Stack('viewmanager').load(path_stack, load_cache=True)
    #     vm = ViewManager(stack, basics=True, nets=False, stats=['mean', 'stddev'], tests=None)
    #     vm_views = vm.get_views(cell_items='cp', weights='weight_a', bases='both').group().views
    #     self.assertEqual(vm_views, E['cp_b_s_w_both'])

    # # --------------------------------------------------------------------
    # # percentage, basics, stats, weighted, base both
    # def test_vm_p_b_s_w_both(self):
    #     path_stack = './tests/stack_vm.stack'
    #     stack = qp.Stack('viewmanager').load(path_stack, load_cache=True)
    #     vm = ViewManager(stack, basics=True, nets=False, stats=['mean', 'stddev'], tests=None)
    #     vm_views = vm.get_views(cell_items='p', weights='weight_a', bases='both').group().views
    #     self.assertEqual(vm_views, E['p_b_s_w_both'])

    # --------------------------------------------------------------------
    # counts + percentage, basics, tests, weighted, base both
    def test_vm_cp_b_t_w_both(self):
        path_stack = './tests/stack_vm.stack'
        stack = qp.Stack('viewmanager').load(path_stack, load_cache=True)
        vm = ViewManager(stack, basics=True, nets=False, stats=None, tests=['0.05'])
        vm_views = vm.get_views(cell_items='cp', weights='weight_a', bases='both').group().views
        self.assertEqual(vm_views, E['cp_b_t_w_both'])

    # --------------------------------------------------------------------
    # percentage, basics, tests, weighted, base both
    def test_vm_p_b_t_w_both(self):
        path_stack = './tests/stack_vm.stack'
        stack = qp.Stack('viewmanager').load(path_stack, load_cache=True)
        vm = ViewManager(stack, basics=True, nets=False, stats=None, tests=['0.05'])
        vm_views = vm.get_views(cell_items='p', weights='weight_a', bases='both').group().views
        self.assertEqual(vm_views, E['p_b_t_w_both'])

    # # --------------------------------------------------------------------
    # # counts + percentage, basics, nets, stats, weighted, base both
    # def test_vm_cp_b_n_s_w_both(self):
    #     path_stack = './tests/stack_vm.stack'
    #     stack = qp.Stack('viewmanager').load(path_stack, load_cache=True)
    #     vm = ViewManager(stack, basics=True, nets=True, stats=['mean', 'stddev'], tests=None)
    #     vm_views = vm.get_views(cell_items='cp', weights='weight_a', bases='both').group().views
    #     self.assertEqual(vm_views, E['cp_b_n_s_w_both'])

    # # --------------------------------------------------------------------
    # # percentage, basics, nets, stats, weighted, base both
    # def test_vm_p_b_n_s_w_both(self):
    #     path_stack = './tests/stack_vm.stack'
    #     stack = qp.Stack('viewmanager').load(path_stack, load_cache=True)
    #     vm = ViewManager(stack, basics=True, nets=True, stats=['mean', 'stddev'], tests=None)
    #     vm_views = vm.get_views(cell_items='p', weights='weight_a', bases='both').group().views
    #     self.assertEqual(vm_views, E['p_b_n_s_w_both'])

    # --------------------------------------------------------------------
    # counts + percentage, basics, nets, tests, weighted, base both
    def test_vm_cp_b_n_t_w_both(self):
        path_stack = './tests/stack_vm.stack'
        stack = qp.Stack('viewmanager').load(path_stack, load_cache=True)
        vm = ViewManager(stack, basics=True, nets=True, stats=None, tests=['0.05'])
        vm_views = vm.get_views(cell_items='cp', weights='weight_a', bases='both').group().views
        self.assertEqual(vm_views, E['cp_b_n_t_w_both'])

    # --------------------------------------------------------------------
    # percentage, basics, nets, tests, weighted, base both
    def test_vm_p_b_n_t_w_both(self):
        path_stack = './tests/stack_vm.stack'
        stack = qp.Stack('viewmanager').load(path_stack, load_cache=True)
        vm = ViewManager(stack, basics=True, nets=True, stats=None, tests=['0.05'])
        vm_views = vm.get_views(cell_items='p', weights='weight_a', bases='both').group().views
        self.assertEqual(vm_views, E['p_b_n_t_w_both'])

    # # --------------------------------------------------------------------
    # # counts + percentage, basics, stats, tests, weighted, base both
    # def test_vm_cp_s_t_w_both(self):
    #     path_stack = './tests/stack_vm.stack'
    #     stack = qp.Stack('viewmanager').load(path_stack, load_cache=True)
    #     vm = ViewManager(stack, basics=True, nets=False, stats=['mean', 'stddev'], tests=['0.05'])
    #     vm_views = vm.get_views(cell_items='cp', weights='weight_a', bases='both').group().views
    #     self.assertEqual(vm_views, E['cp_b_s_t_w_both'])

    # # --------------------------------------------------------------------
    # # percentage, basics, stats, tests, weighted, base both
    # def test_vm_p_b_s_t_w_both(self):
    #     path_stack = './tests/stack_vm.stack'
    #     stack = qp.Stack('viewmanager').load(path_stack, load_cache=True)
    #     vm = ViewManager(stack, basics=True, nets=False, stats=['mean', 'stddev'], tests=['0.05'])
    #     vm_views = vm.get_views(cell_items='p', weights='weight_a', bases='both').group().views
    #     self.assertEqual(vm_views, E['p_b_s_t_w_both'])

    # # --------------------------------------------------------------------
    # # counts + percentage, basics, nets, stats, tests, weighted, base both
    # def test_vm_cp_n_s_t_w_both(self):
    #     path_stack = './tests/stack_vm.stack'
    #     stack = qp.Stack('viewmanager').load(path_stack, load_cache=True)
    #     vm = ViewManager(stack, basics=True, nets=True, stats=['mean', 'stddev'], tests=['0.05'])
    #     vm_views = vm.get_views(cell_items='cp', weights='weight_a', bases='both').group().views
    #     self.assertEqual(vm_views, E['cp_b_n_s_t_w_both'])

    # # --------------------------------------------------------------------
    # # percentage, basics, nets, stats, tests, weighted, base both
    # def test_vm_p_b_s_t_w_both(self):
    #     path_stack = './tests/stack_vm.stack'
    #     stack = qp.Stack('viewmanager').load(path_stack, load_cache=True)
    #     vm = ViewManager(stack, basics=True, nets=True, stats=['mean', 'stddev'], tests=['0.05'])
    #     vm_views = vm.get_views(cell_items='p', weights='weight_a', bases='both').group().views
    #     self.assertEqual(vm_views, E['p_b_n_s_t_w_both'])

        # path_cache = './tests/stack_vm.cache'
        # os.remove(path_stack)
        # os.remove(path_cache)

