import unittest
import os.path
import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal
import test_helper
import copy

from operator import lt, le, eq, ne, ge, gt

from pandas.core.index import Index
__index_symbol__ = {
    Index.union: ',',
    Index.intersection: '&',
    Index.difference: '~',
    Index.sym_diff: '^'
}

from collections import defaultdict, OrderedDict
from quantipy.core.stack import Stack
from quantipy.core.chain import Chain
from quantipy.core.link import Link
from quantipy.core.view_generators.view_mapper import ViewMapper
from quantipy.core.view_generators.view_maps import QuantipyViews
from quantipy.core.view import View
from quantipy.core.helpers import functions
from quantipy.core.helpers.functions import load_json
from quantipy.core.tools.dp.prep import (
    frange,
    frequency,
    crosstab
)
from quantipy.core.tools.view.query import get_dataframe
from quantipy.core.dataset import DataSet

EXTENDED_TESTS = False
COUNTER = 0

class TestRules(unittest.TestCase):

    def setUp(self):
        self.path = './tests/'
        # self.path = ''
        self.path = 'C:/Users/alt/AppData/Local/Continuum/Anaconda/Lib/site-packages/quantipy/tests/'
        project_name = 'Example Data (A)'

        # Load Example Data (A) data and meta into self
        name_data = '%s.csv' % (project_name)
        path_data = '%s%s' % (self.path, name_data)
        self.example_data_A_data = pd.DataFrame.from_csv(path_data)
        name_meta = '%s.json' % (project_name)
        path_meta = '%s%s' % (self.path, name_meta)
        self.example_data_A_meta = load_json(path_meta)

        # Variables by type for Example Data A
        self.dk = 'Example Data (A)'
        self.fk = 'no_filter'
        self.single = ['gender', 'locality', 'ethnicity', 'religion', 'q1']
        self.delimited_set = ['q2', 'q3', 'q8', 'q9']
        self.q5 = ['q5_1', 'q5_2', 'q5_3']

    def test_slicex(self):

        meta = self.example_data_A_meta
        data = self.example_data_A_data

        col_x = 'religion'
        col_y = 'ethnicity'

        ################## values
        meta['columns'][col_x]['rules'] = {
            'x': {'slicex': {'values': [1, 3, 5, 7, 9, 11, 13, 15]}}}

        meta['columns'][col_y]['rules'] = {
            'y': {'slicex': {'values': [2, 4, 6, 8, 10, 12, 14, 16]}}}

        rules_values_x = {
            'unwtd': index_items(col_x, all=True,
                values=[1, 3, 5, 7, 9, 11, 13, 15]),
            'iswtd': index_items(col_x, all=True,
                values=[1, 3, 5, 7, 9, 11, 13, 15])}

        rules_values_y = {
            'unwtd': index_items(col_y, all=True,
                values=[2, 4, 6, 8, 10, 12, 14, 16]),
            'iswtd': index_items(col_y, all=True,
                values=[2, 4, 6, 8, 10, 12, 14, 16])}

        confirm_crosstabs(
            self,
            meta, data,
            [None, 'weight_a'],
            col_x, col_y,
            rules_values_x,
            rules_values_y)

    def _get_dataset(self):
        meta = self.example_data_A_meta
        data = self.example_data_A_data
        dataset = DataSet('rules_test')
        dataset.set_verbose_infomsg(False)
        dataset.from_components(data, meta)
        return dataset

    def _get_stack_with_links(self, dataset, x=None, y=None, w=None):
        stack = Stack()
        stack.add_data(dataset.name, dataset._data, dataset._meta)
        if not x: x = '@'
        if not y: y = '@'
        stack.add_link(x=x, y=y, weights=w)
        return stack

    def test_sortx_summaries_mean(self):
        dataset = self._get_dataset()
        x = 'q5'
        y = '@'
        dataset.sorting(x, on='mean')
        stack = self._get_stack_with_links(dataset, x)
        stack.add_link(x=x, y=y, views=['cbase', 'counts', 'c%', 'mean'])

        vks = ['x|f|x:|||cbase', 'x|f|:|||counts', 'x|f|:|y||c%',
               'x|d.mean|x:|||mean']

        chains = stack.get_chain(data_keys=dataset.name,
                                 filters='no_filter',
                                 x=[x], y=[y], rules=True,
                                views=vks,
                                orient_on='x')
        chain = chains[0]
        for vk in vks:
            v = chain['rules_test']['no_filter'][x][y][vk]
            l = stack['rules_test']['no_filter'][x][y][vk]
            check_chain_view_dataframe = v.dataframe.reindex_like(l.dataframe)
            self.assertTrue(check_chain_view_dataframe.equals(l.dataframe))
            actual_order = v.dataframe.index.get_level_values(1).tolist()
            expected_order = ['q5_4', 'q5_6', 'q5_1', 'q5_3', 'q5_5', 'q5_2']
            self.assertEqual(actual_order, expected_order)

    def test_sortx_summaries_value(self):
        dataset = self._get_dataset()
        x = 'q5'
        y  = '@'
        dataset.sorting(x, on=3, ascending=True)
        stack = self._get_stack_with_links(dataset, x)
        stack.add_link(x=x, y=y, views=['cbase', 'counts', 'c%', 'mean'])

        vks = ['x|f|x:|||cbase', 'x|f|:|||counts', 'x|f|:|y||c%',
               'x|d.mean|x:|||mean']

        chains = stack.get_chain(data_keys=dataset.name,
                                 filters='no_filter',
                                 x=[x], y=[y], rules=True,
                                views=vks,
                                orient_on='x')
        chain = chains[0]
        for vk in vks:
            v = chain['rules_test']['no_filter'][x][y][vk]
            l = stack['rules_test']['no_filter'][x][y][vk]

            check_chain_view_dataframe = v.dataframe.reindex_like(l.dataframe)
            self.assertTrue(check_chain_view_dataframe.equals(l.dataframe))
            actual_order = v.dataframe.index.get_level_values(1).tolist()
            expected_order = ['q5_4', 'q5_5', 'q5_6', 'q5_1', 'q5_3', 'q5_2']
            self.assertEqual(actual_order, expected_order)

    def test_sortx_summaries_items(self):
        dataset = self._get_dataset()
        x  = '@'
        y = 'q5'
        dataset.sorting(y, on='q5_2', ascending=False)
        stack = self._get_stack_with_links(dataset, y=y)
        stack.add_link(x=x, y=y, views=['cbase', 'counts', 'c%', 'mean'])

        vks = ['x|f|x:|||cbase', 'x|f|:|||counts', 'x|f|:|y||c%',
               'x|d.mean|x:|||mean']

        chains = stack.get_chain(data_keys=dataset.name,
                                 filters='no_filter',
                                 x=[x], y=[y], rules=True,
                                views=vks,
                                orient_on='x')
        chain = chains[0]
        for vk in vks:
            v = chain['rules_test']['no_filter'][x][y][vk]
            l = stack['rules_test']['no_filter'][x][y][vk]

            if not 'd.mean' in vk and not 'cbase' in vk:
                check_chain_view_dataframe = v.dataframe.reindex_like(l.dataframe)
                self.assertTrue(check_chain_view_dataframe.equals(l.dataframe))
                actual_order = v.dataframe.index.get_level_values(1).tolist()
                expected_order = [3, 5, 98, 2, 1, 97, 4]
                self.assertEqual(actual_order, expected_order)

    def test_sortx_expand_net_within(self):
        dataset = self._get_dataset()
        x = 'q2'
        y = ['@', 'gender']
        dataset.sorting(x, on='@', within=True, between=False, fix=98)
        stack = self._get_stack_with_links(dataset, x=x, y=y)

        net = [{'test A': [1, 2, 3], 'text': {'en-GB': 'Lab1'}},
               {'test B': [5, 6, 97], 'text': {'en-GB': 'Lab2'}}]
        net_view = ViewMapper().make_template('frequency')
        view_name = 'expandnet'
        options = {'logic': net,
                   'expand': 'after',
                   'complete': True,
                   'axis': 'x',
                   'iterators': {'rel_to': [None, 'y']}}
        net_view.add_method(view_name, kwargs=options)
        stack.add_link(x=x, y=y, views=net_view)

        vks = ['x|f|x[{1,2,3}+],x[{5,6,97}+]*:|||expandnet',
               'x|f|x[{1,2,3}+],x[{5,6,97}+]*:|y||expandnet']
        chains = stack.get_chain(data_keys=dataset.name,
                                 filters='no_filter',
                                 x=[x], y=y, rules=True,
                                 views=vks,
                                 orient_on='x')
        chain = chains[0]
        for yk in y:
            for vk in vks:
                v = chain['rules_test']['no_filter'][x][yk][vk]
                l = stack['rules_test']['no_filter'][x][yk][vk]
                check_chain_view_dataframe = v.dataframe.reindex_like(l.dataframe)
                self.assertTrue(check_chain_view_dataframe.equals(l.dataframe))
                actual_order = v.dataframe.index.get_level_values(1).tolist()
                expected_order = ['test A', 3, 2, 1, 4, 'test B', 97, 5, 6, 98]
                self.assertEqual(actual_order, expected_order)

    def test_sortx_expand_net_between(self):
        dataset = self._get_dataset()
        x = 'q2'
        y = ['@', 'gender']
        dataset.sorting(x, on='@', within=False, between=True, ascending=True,
                        fix=98)
        stack = self._get_stack_with_links(dataset, x=x, y=y)

        net = [{'test A': [1, 2, 3], 'text': {'en-GB': 'Lab1'}},
               {'test B': [5, 6, 97], 'text': {'en-GB': 'Lab2'}}]
        net_view = ViewMapper().make_template('frequency')
        view_name = 'expandnet'
        options = {'logic': net,
                   'expand': 'after',
                   'complete': True,
                   'axis': 'x',
                   'iterators': {'rel_to': [None, 'y']}}
        net_view.add_method(view_name, kwargs=options)
        stack.add_link(x=x, y=y, views=net_view)

        vks = ['x|f|x[{1,2,3}+],x[{5,6,97}+]*:|||expandnet',
               'x|f|x[{1,2,3}+],x[{5,6,97}+]*:|y||expandnet']
        chains = stack.get_chain(data_keys=dataset.name,
                                 filters='no_filter',
                                 x=[x], y=y, rules=True,
                                 views=vks,
                                 orient_on='x')
        chain = chains[0]
        for yk in y:
            for vk in vks:
                v = chain['rules_test']['no_filter'][x][yk][vk]
                l = stack['rules_test']['no_filter'][x][yk][vk]
                check_chain_view_dataframe = v.dataframe.reindex_like(l.dataframe)
                self.assertTrue(check_chain_view_dataframe.equals(l.dataframe))
                actual_order = v.dataframe.index.get_level_values(1).tolist()
                expected_order = [4, 'test B', 5, 6, 97, 'test A', 1, 2, 3, 98]
                self.assertEqual(actual_order, expected_order)

    def test_sortx_expand_net_within_between(self):
        dataset = self._get_dataset()
        x = 'q2'
        y = ['@', 'gender']
        dataset.sorting(x, on='@', within=True, between=True, ascending=False,
                        fix=98)
        stack = self._get_stack_with_links(dataset, x=x, y=y)

        net = [{'test A': [1, 2, 3], 'text': {'en-GB': 'Lab1'}},
               {'test B': [5, 6, 97], 'text': {'en-GB': 'Lab2'}}]
        net_view = ViewMapper().make_template('frequency')
        view_name = 'expandnet'
        options = {'logic': net,
                   'expand': 'after',
                   'complete': True,
                   'axis': 'x',
                   'iterators': {'rel_to': [None, 'y']}}
        net_view.add_method(view_name, kwargs=options)
        stack.add_link(x=x, y=y, views=net_view)

        test_view = ViewMapper().make_template('coltests')
        view_name = 'test'
        options = {'level': 0.2}
        test_view.add_method(view_name, kwargs=options)
        stack.add_link(x=x, y=y, views=test_view)

        vks = ['x|f|x[{1,2,3}+],x[{5,6,97}+]*:|||expandnet',
               'x|f|x[{1,2,3}+],x[{5,6,97}+]*:|y||expandnet',
               'x|t.props.Dim.20|x[{1,2,3}+],x[{5,6,97}+]*:|||test']
        chains = stack.get_chain(data_keys=dataset.name,
                                 filters='no_filter',
                                 x=[x], y=y, rules=True,
                                 views=vks,
                                 orient_on='x')
        chain = chains[0]
        for yk in y:
            for vk in vks:
                v = chain['rules_test']['no_filter'][x][yk][vk]
                l = stack['rules_test']['no_filter'][x][yk][vk]
                check_chain_view_dataframe = v.dataframe.reindex_like(l.dataframe)
                self.assertTrue(check_chain_view_dataframe.equals(l.dataframe))
                actual_order = v.dataframe.index.get_level_values(1).tolist()
                expected_order = ['test A', 3, 2, 1, 'test B', 97, 5, 6, 4, 98]
                self.assertEqual(actual_order, expected_order)


    def test_sortx(self):

        meta = self.example_data_A_meta
        data = self.example_data_A_data

        col_x = 'religion'
        col_y = 'ethnicity'

        ################## sort_on - default
        meta['columns'][col_x]['rules'] = {'x': {'sortx': {}}}
        meta['columns'][col_y]['rules'] = {'y': {'sortx': {}}}

        rules_values_x = {
            'unwtd': index_items(col_x, all=True,
                values=[2, 1, 3, 15, 4, 5, 16, 6, 10, 12, 14, 11, 7, 13, 8, 9]),
            'iswtd': index_items(col_x, all=True,
                values=[2, 1, 3, 15, 4, 5, 16, 6, 12, 10, 14, 11, 7, 13, 9, 8])}

        rules_values_y = {
            'unwtd': index_items(col_y, all=True,
                values=[1, 2, 16, 7, 15, 12, 3, 11, 14, 6, 8, 10, 9, 5, 4, 13]),
            'iswtd': index_items(col_y, all=True,
                values=[1, 2, 16, 7, 12, 11, 3, 15, 8, 9, 10, 14, 5, 6, 4, 13])}

        confirm_crosstabs(
            self,
            meta, data,
            [None, 'weight_a'],
            col_x, col_y,
            rules_values_x,
            rules_values_y)

        ################## sort_on - '@'
        meta['columns'][col_x]['rules'] = {
            'x': {'sortx': {'sort_on': '@'}}}
        meta['columns'][col_y]['rules'] = {
            'y': {'sortx': {'sort_on': '@'}}}

        rules_values_x = {
            'unwtd': index_items(col_x, all=True,
                values=[2, 1, 3, 15, 4, 5, 16, 6, 10, 12, 14, 11, 7, 13, 8, 9]),
            'iswtd': index_items(col_x, all=True,
                values=[2, 1, 3, 15, 4, 5, 16, 6, 12, 10, 14, 11, 7, 13, 9, 8])}

        rules_values_y = {
            'unwtd': index_items(col_y, all=True,
                values=[1, 2, 16, 7, 15, 12, 3, 11, 14, 6, 8, 10, 9, 5, 4, 13]),
            'iswtd': index_items(col_y, all=True,
                values=[1, 2, 16, 7, 12, 11, 3, 15, 8, 9, 10, 14, 5, 6, 4, 13])}

        confirm_crosstabs(
            self,
            meta, data,
            [None, 'weight_a'],
            col_x, col_y,
            rules_values_x,
            rules_values_y)

        ################## fixed
        meta['columns'][col_x]['rules'] = {
            'x': {'sortx': {'fixed': [5, 1, 3]}}}

        meta['columns'][col_y]['rules'] = {
            'y': {'sortx': {'fixed': [6, 2, 4]}}}

        rules_values_x = {
            'unwtd': index_items(col_x, all=True,
                values=[2, 15, 4, 16, 6, 10, 12, 14, 11, 7, 13, 8, 9, 5, 1, 3]),
            'iswtd': index_items(col_x, all=True,
                values=[2, 15, 4, 16, 6, 12, 10, 14, 11, 7, 13, 9, 8, 5, 1, 3])}

        rules_values_y = {
            'unwtd': index_items(col_y, all=True,
                values=[1, 16, 7, 15, 12, 3, 11, 14, 8, 10, 9, 5, 13, 6, 2, 4]),
            'iswtd': index_items(col_y, all=True,
                values=[1, 16, 7, 12, 11, 3, 15, 8, 9, 10, 14, 5, 13, 6, 2, 4])}

        confirm_crosstabs(
            self,
            meta, data,
            [None, 'weight_a'],
            col_x, col_y,
            rules_values_x,
            rules_values_y)

        ################## with_weight
        meta['columns'][col_x]['rules'] = {
            'x': {'sortx': {'with_weight': 'weight_b'}}}

        meta['columns'][col_y]['rules'] = {
            'y': {'sortx': {'with_weight': 'weight_b'}}}

        rules_values_x = {
            'unwtd': index_items(col_x, all=True,
                values=[2, 1, 3, 15, 4, 5, 16, 12, 6, 10, 14, 11, 7, 13, 9, 8]),
            'iswtd': index_items(col_x, all=True,
                values=[2, 1, 3, 15, 4, 5, 16, 12, 6, 10, 14, 11, 7, 13, 9, 8])}

        rules_values_y = {
            'unwtd': index_items(col_y, all=True,
                values=[1, 2, 16, 7, 11, 3, 12, 15, 8, 9, 10, 5, 14, 6, 4, 13]),
            'iswtd': index_items(col_y, all=True,
                values=[1, 2, 16, 7, 11, 3, 12, 15, 8, 9, 10, 5, 14, 6, 4, 13])}

        confirm_crosstabs(
            self,
            meta, data,
            [None, 'weight_a'],
            col_x, col_y,
            rules_values_x,
            rules_values_y)

    def test_dropx(self):

        meta = self.example_data_A_meta
        data = self.example_data_A_data

        col_x = 'religion'
        col_y = 'ethnicity'

        ################## values
        meta['columns'][col_x]['rules'] = {
            'x': {'dropx': {'values': [1, 3, 5, 7, 9, 11, 13, 15]}}}

        meta['columns'][col_y]['rules'] = {
            'y': {'dropx': {'values': [2, 4, 6, 8, 10, 12, 14, 16]}}}

        rules_values_x = {
            'unwtd': index_items(col_x, all=True,
                values=[2, 4, 6, 8, 10, 12, 14, 16]),
            'iswtd': index_items(col_x, all=True,
                values=[2, 4, 6, 8, 10, 12, 14, 16])}

        rules_values_y = {
            'unwtd': index_items(col_y, all=True,
                values=[1, 3, 5, 7, 9, 11, 13, 15]),
            'iswtd': index_items(col_y, all=True,
                values=[1, 3, 5, 7, 9, 11, 13, 15])}

        confirm_crosstabs(
            self,
            meta, data,
            [None, 'weight_a'],
            col_x, col_y,
            rules_values_x,
            rules_values_y)

    def test_rules_frequency(self):

        meta = self.example_data_A_meta
        data = self.example_data_A_data

        col = 'religion'

        ################## slicex
        meta['columns'][col]['rules'] = {
            'x': {'slicex': {'values': [1, 3, 5, 7, 9, 10, 11, 13, 15]}},
            'y': {'slicex': {'values': [2, 4, 6, 8, 10, 12, 14, 16]}}}

        rules_values_x = {
            'unwtd': index_items(col, all=True,
                values=[1, 3, 5, 7, 9, 10, 11, 13, 15]),
            'iswtd': index_items(col, all=True,
                values=[1, 3, 5, 7, 9, 10, 11, 13, 15])}

        rules_values_y = {
            'unwtd': index_items(col, all=True,
                values=[2, 4, 6, 8, 10, 12, 14, 16]),
            'iswtd': index_items(col, all=True,
                values=[2, 4, 6, 8, 10, 12, 14, 16])}

        confirm_frequencies(
            self,
            meta, data,
            [None, 'weight_a'],
            col,
            rules_values_x,
            rules_values_y)

        ################## sortx
        meta['columns'][col]['rules'] = {
            'x': {'sortx': {'fixed': [5, 1, 3]}},
            'y': {'sortx': {'fixed': [6, 2, 4]}}}

        rules_values_x = {
            'unwtd': index_items(col, all=True,
                values=[2, 15, 4, 16, 6, 10, 12, 14, 11, 7, 13, 8, 9, 5, 1, 3]),
            'iswtd': index_items(col, all=True,
                values=[2, 15, 4, 16, 6, 12, 10, 14, 11, 7, 13, 9, 8, 5, 1, 3])}

        rules_values_y = {
            'unwtd': index_items(col, all=True,
                values=[1, 3, 15, 5, 16, 10, 12, 14, 11, 7, 13, 8, 9, 6, 2, 4]),
            'iswtd': index_items(col, all=True,
                values=[1, 3, 15, 5, 16, 12, 10, 14, 11, 7, 13, 9, 8, 6, 2, 4])}

        confirm_frequencies(
            self,
            meta, data,
            [None, 'weight_a'],
            col,
            rules_values_x,
            rules_values_y)

        ################## dropx
        meta['columns'][col]['rules'] = {
            'x': {'dropx': {'values': [1, 3, 5, 7, 9, 11, 13, 15]}},
            'y': {'dropx': {'values': [2, 4, 6, 8, 10, 12, 14, 16]}}}

        rules_values_x = {
            'unwtd': index_items(col, all=True,
                values=[2, 4, 6, 8, 10, 12, 14, 16]),
            'iswtd': index_items(col, all=True,
                values=[2, 4, 6, 8, 10, 12, 14, 16])}

        rules_values_y = {
            'unwtd': index_items(col, all=True,
                values=[1, 3, 5, 7, 9, 11, 13, 15]),
            'iswtd': index_items(col, all=True,
                values=[1, 3, 5, 7, 9, 11, 13, 15])}

        confirm_frequencies(
            self,
            meta, data,
            [None, 'weight_a'],
            col,
            rules_values_x,
            rules_values_y)

        ################## slicex + sortx
        meta['columns'][col]['rules'] = {
            'x': {
                'slicex': {'values': frange('4-13')},
                'sortx': {'fixed': [1, 2]}},
            'y': {
                'slicex': {'values': frange('7-16')},
                'sortx': {'fixed': [15, 16]}}}

        rules_values_x = {
            'unwtd': index_items(col, all=True,
                values=[4, 5, 6, 10, 12, 11, 7, 13, 8, 9, 1, 2]),
            'iswtd': index_items(col, all=True,
                values=[4, 5, 6, 12, 10, 11, 7, 13, 9, 8, 1, 2])}

        rules_values_y = {
            'unwtd': index_items(col, all=True,
                values=[10, 12, 14, 11, 7, 13, 8, 9, 15, 16]),
            'iswtd': index_items(col, all=True,
                values=[12, 10, 14, 11, 7, 13, 9, 8, 15, 16])}

        confirm_frequencies(
            self,
            meta, data,
            [None, 'weight_a'],
            col,
            rules_values_x,
            rules_values_y)

        ################## slicex + dropx
        meta['columns'][col]['rules'] = {
            'x': {
                'slicex': {'values': [1, 3, 5, 7, 9, 11, 13, 15]},
                'dropx': {'values': [3, 7, 11, 15]}},
            'y': {
                'slicex': {'values': [2, 4, 6, 8, 10, 12, 14, 16]},
                'dropx': {'values': [2, 6, 10, 14]}}}

        rules_values_x = {
            'unwtd': index_items(col, all=True,
                values=[1, 5, 9, 13]),
            'iswtd': index_items(col, all=True,
                values=[1, 5, 9, 13])}

        rules_values_y = {
            'unwtd': index_items(col, all=True,
                values=[4, 8, 12, 16]),
            'iswtd': index_items(col, all=True,
                values=[4, 8, 12, 16])}

        confirm_frequencies(
            self,
            meta, data,
            [None, 'weight_a'],
            col,
            rules_values_x,
            rules_values_y)

        ################## sortx + dropx
        meta['columns'][col]['rules'] = {
            'x': {
                'sortx': {'fixed': [1, 2]},
                'dropx': {'values': [5, 11, 13]}},
            'y': {
                'sortx': {'fixed': [15, 16]},
                'dropx': {'values': [7, 13, 14]}}}

        rules_values_x = {
            'unwtd': index_items(col, all=True,
                values=[3, 15, 4, 16, 6, 10, 12, 14, 7, 8, 9, 1, 2]),
            'iswtd': index_items(col, all=True,
                values=[3, 15, 4, 16, 6, 12, 10, 14, 7, 9, 8, 1, 2])}

        rules_values_y = {
            'unwtd': index_items(col, all=True,
                values=[2, 1, 3, 4, 5, 6, 10, 12, 11, 8, 9, 15, 16]),
            'iswtd': index_items(col, all=True,
                values=[2, 1, 3, 4, 5, 6, 12, 10, 11, 9, 8, 15, 16])}

        confirm_frequencies(
            self,
            meta, data,
            [None, 'weight_a'],
            col,
            rules_values_x,
            rules_values_y)

        ################## slicex + sortx + dropx
        meta['columns'][col]['rules'] = {
            'x': {
                'slicex': {'values': frange('4-13')},
                'sortx': {'fixed': [11, 13]},
                'dropx': {'values': [7]}},
            'y': {
                'slicex': {'values': frange('7-16')},
                'sortx': {'fixed': [15, 16]},
                'dropx': {'values': [7, 13]}}}

        rules_values_x = {
            'unwtd': index_items(col, all=True,
                values=[4, 5, 6, 10, 12, 8, 9, 11, 13]),
            'iswtd': index_items(col, all=True,
                values=[4, 5, 6, 12, 10, 9, 8, 11, 13])}

        rules_values_y = {
            'unwtd': index_items(col, all=True,
                values=[10, 12, 14, 11, 8, 9, 15, 16]),
            'iswtd': index_items(col, all=True,
                values=[12, 10, 14, 11, 9, 8, 15, 16])}

        confirm_frequencies(
            self,
            meta, data,
            [None, 'weight_a'],
            col,
            rules_values_x,
            rules_values_y)

    def test_rules_crosstab(self):

        meta = self.example_data_A_meta
        data = self.example_data_A_data

        col_x = 'religion'
        col_y = 'ethnicity'

        ################## slicex + sortx + dropx
        meta['columns'][col_x]['rules'] = {
            'x': {
                'slicex': {'values': frange('4-13')},
                'sortx': {'fixed': [4, 7, 3]},
                'dropx': {'values': [6, 11]}}}

        meta['columns'][col_y]['rules'] = {
            'y': {
                'slicex': {'values': frange('7-16')},
                'sortx': {'fixed': [7, 11, 13]},
                'dropx': {'values': [11, 16]}}}

        rules_values_x = {
            'unwtd': index_items(col_x, all=True,
                values=[5, 10, 12, 13, 8, 9, 4, 7, 3]),
            'iswtd': index_items(col_x, all=True,
                values=[5, 12, 10, 13, 9, 8, 4, 7, 3])}

        rules_values_y = {
            'unwtd': index_items(col_y, all=True,
                values=[15, 12, 14, 8, 10, 9, 7, 13]),
            'iswtd': index_items(col_y, all=True,
                values=[12, 15, 8, 9, 10, 14, 7, 13])}

        confirm_crosstabs(
            self,
            meta, data,
            [None, 'weight_a'],
            col_x, col_y,
            rules_values_x,
            rules_values_y)

        if EXTENDED_TESTS:
            ################## slicex
            meta['columns'][col_x]['rules'] = {
                'x': {'slicex': {'values': [1, 3, 5, 7, 9, 10, 11, 13, 15]}}}

            meta['columns'][col_y]['rules'] = {
                'y': {'slicex': {'values': [2, 4, 6, 8, 10, 12, 14, 16]}}}

            rules_values_x = {
                'unwtd': index_items(col_x, all=True,
                    values=[1, 3, 5, 7, 9, 10, 11, 13, 15]),
                'iswtd': index_items(col_x, all=True,
                    values=[1, 3, 5, 7, 9, 10, 11, 13, 15])}

            rules_values_y = {
                'unwtd': index_items(col_y, all=True,
                    values=[2, 4, 6, 8, 10, 12, 14, 16]),
                'iswtd': index_items(col_y, all=True,
                    values=[2, 4, 6, 8, 10, 12, 14, 16])}

            confirm_crosstabs(
                self,
                meta, data,
                [None, 'weight_a'],
                col_x, col_y,
                rules_values_x,
                rules_values_y)

            ################## sortx
            meta['columns'][col_x]['rules'] = {
                'x': {'sortx': {'fixed': [5, 1, 3]}}}

            meta['columns'][col_y]['rules'] = {
                'y': {'sortx': {'fixed': [6, 2, 4]}}}

            rules_values_x = {
                'unwtd': index_items(col_x, all=True,
                    values=[2, 15, 4, 16, 6, 10, 12, 14, 11, 7, 13, 8, 9, 5, 1, 3]),
                'iswtd': index_items(col_x, all=True,
                    values=[2, 15, 4, 16, 6, 12, 10, 14, 11, 7, 13, 9, 8, 5, 1, 3])}

            rules_values_y = {
                'unwtd': index_items(col_y, all=True,
                    values=[1, 16, 7, 15, 12, 3, 11, 14, 8, 10, 9, 5, 13, 6, 2, 4]),
                'iswtd': index_items(col_y, all=True,
                    values=[1, 16, 7, 12, 11, 3, 15, 8, 9, 10, 14, 5, 13, 6, 2, 4])}

            confirm_crosstabs(
                self,
                meta, data,
                [None, 'weight_a'],
                col_x, col_y,
                rules_values_x,
                rules_values_y)

            ################## dropx
            meta['columns'][col_x]['rules'] = {
                'x': {'dropx': {'values': [1, 3, 5, 7, 9, 11, 13, 15]}}}

            meta['columns'][col_y]['rules'] = {
                'y': {'dropx': {'values': [2, 4, 6, 8, 10, 12, 14, 16]}}}

            rules_values_x = {
                'unwtd': index_items(col_x, all=True,
                    values=[2, 4, 6, 8, 10, 12, 14, 16]),
                'iswtd': index_items(col_x, all=True,
                    values=[2, 4, 6, 8, 10, 12, 14, 16])}

            rules_values_y = {
                'unwtd': index_items(col_y, all=True,
                    values=[1, 3, 5, 7, 9, 11, 13, 15]),
                'iswtd': index_items(col_y, all=True,
                    values=[1, 3, 5, 7, 9, 11, 13, 15])}

            confirm_crosstabs(
                self,
                meta, data,
                [None, 'weight_a'],
                col_x, col_y,
                rules_values_x,
                rules_values_y)

            ################## slicex + sortx
            meta['columns'][col_x]['rules'] = {
                'x': {
                    'slicex': {'values': frange('4-13')},
                    'sortx': {'fixed': [4, 7, 3]}}}

            meta['columns'][col_y]['rules'] = {
                'y': {
                    'slicex': {'values': frange('7-16')},
                    'sortx': {'fixed': [7, 11, 13]}}}

            rules_values_x = {
                'unwtd': index_items(col_x, all=True,
                    values=[5, 6, 10, 12, 11, 13, 8, 9, 4, 7, 3]),
                'iswtd': index_items(col_x, all=True,
                    values=[5, 6, 12, 10, 11, 13, 9, 8, 4, 7, 3])}

            rules_values_y = {
                'unwtd': index_items(col_y, all=True,
                    values=[16, 15, 12, 14, 8, 10, 9, 7, 11, 13]),
                'iswtd': index_items(col_y, all=True,
                    values=[16, 12, 15, 8, 9, 10, 14, 7, 11, 13])}

            confirm_crosstabs(
                self,
                meta, data,
                [None, 'weight_a'],
                col_x, col_y,
                rules_values_x,
                rules_values_y)

            ################## slicex + dropx
            meta['columns'][col_x]['rules'] = {
                'x': {
                    'slicex': {'values': [1, 3, 5, 7, 9, 11, 13, 15]},
                    'dropx': {'values': [3, 7, 11, 15]}}}

            meta['columns'][col_y]['rules'] = {
                'y': {
                    'slicex': {'values': [2, 4, 6, 8, 10, 12, 14, 16]},
                    'dropx': {'values': [2, 6, 10, 14]}}}

            rules_values_x = {
                'unwtd': index_items(col_x, all=True,
                    values=[1, 5, 9, 13]),
                'iswtd': index_items(col_x, all=True,
                    values=[1, 5, 9, 13])}

            rules_values_y = {
                'unwtd': index_items(col_y, all=True,
                    values=[4, 8, 12, 16]),
                'iswtd': index_items(col_y, all=True,
                    values=[4, 8, 12, 16])}

            confirm_crosstabs(
                self,
                meta, data,
                [None, 'weight_a'],
                col_x, col_y,
                rules_values_x,
                rules_values_y)

            ################## sortx + dropx
            meta['columns'][col_x]['rules'] = {
                'x': {
                    'sortx': {'fixed': [4, 7, 3]},
                    'dropx': {'values': [5, 10]}}}

            meta['columns'][col_y]['rules'] = {
                'y': {
                    'sortx': {'fixed': [7, 11, 13]},
                    'dropx': {'values': [4, 12]}}}

            rules_values_x = {
                'unwtd': index_items(col_x, all=True,
                    values=[2, 1, 15, 16, 6, 12, 14, 11, 13, 8, 9, 4, 7, 3]),
                'iswtd': index_items(col_x, all=True,
                    values=[2, 1, 15, 16, 6, 12, 14, 11, 13, 9, 8, 4, 7, 3])}

            rules_values_y = {
                'unwtd': index_items(col_y, all=True,
                    values=[1, 2, 16, 15, 3, 14, 6, 8, 10, 9, 5, 7, 11, 13]),
                'iswtd': index_items(col_y, all=True,
                    values=[1, 2, 16, 3, 15, 8, 9, 10, 14, 5, 6, 7, 11, 13])}

            confirm_crosstabs(
                self,
                meta, data,
                [None, 'weight_a'],
                col_x, col_y,
                rules_values_x,
                rules_values_y)

    def test_rules_get_dataframe(self):

        meta = self.example_data_A_meta
        data = self.example_data_A_data

        col_x = 'religion'
        col_y = 'ethnicity'

        xks = [col_x]
        yks = ['@', col_y]

        test_views = [
            'cbase', 'rbase', 'ebase',
            'counts', 'c%', 'r%',
            'mean']

        weights = [None, 'weight_a']

        ################## slicex + sortx + dropx
        meta['columns'][col_x]['rules'] = {
            'x': {
                'slicex': {'values': frange('4-13')},
                'sortx': {'fixed': [4, 7, 3]},
                'dropx': {'values': [6, 11]}}}

        meta['columns'][col_y]['rules'] = {
            'y': {
                'slicex': {'values': frange('7-16')},
                'sortx': {'fixed': [7, 11, 13]},
                'dropx': {'values': [11, 16]}}}

        rules_values_x = {
            'unwtd': index_items(col_x, all=False,
                values=[5, 10, 12, 13, 8, 9, 4, 7, 3]),
            'iswtd': index_items(col_x, all=False,
                values=[5, 12, 10, 13, 9, 8, 4, 7, 3])}

        rules_values_y = {
            'unwtd': index_items(col_y, all=False,
                values=[15, 12, 14, 8, 10, 9, 7, 13]),
            'iswtd': index_items(col_y, all=False,
                values=[12, 15, 8, 9, 10, 14, 7, 13])}

        stack = get_stack(self, meta, data, xks, yks, test_views, weights,
                          extras=True)

        confirm_get_dataframe(
            self, stack, col_x, col_y,
            rules_values_x, rules_values_y)

        if EXTENDED_TESTS:
            ################## slicex
            meta['columns'][col_x]['rules'] = {
                'x': {'slicex': {'values': [1, 3, 5, 7, 9, 10, 11, 13, 15]}}}

            meta['columns'][col_y]['rules'] = {
                'y': {'slicex': {'values': [2, 4, 6, 8, 10, 12, 14, 16]}}}

            rules_values_x = {
                'unwtd': index_items(col_x, all=False,
                    values=[1, 3, 5, 7, 9, 10, 11, 13, 15]),
                'iswtd': index_items(col_x, all=False,
                    values=[1, 3, 5, 7, 9, 10, 11, 13, 15])}

            rules_values_y = {
                'unwtd': index_items(col_y, all=False,
                    values=[2, 4, 6, 8, 10, 12, 14, 16]),
                'iswtd': index_items(col_y, all=False,
                    values=[2, 4, 6, 8, 10, 12, 14, 16])}

            stack = get_stack(self, meta, data, xks, yks, test_views, weights,
                              extras=True)

            confirm_get_dataframe(
                self, stack, col_x, col_y,
                rules_values_x, rules_values_y)

            ################## sortx
            meta['columns'][col_x]['rules'] = {
                'x': {'sortx': {'fixed': [5, 1, 3]}}}

            meta['columns'][col_y]['rules'] = {
                'y': {'sortx': {'fixed': [6, 2, 4]}}}

            rules_values_x = {
                'unwtd': index_items(col_x, all=False,
                    values=[2, 15, 4, 16, 6, 10, 12, 14, 11, 7, 13, 8, 9, 5, 1, 3]),
                'iswtd': index_items(col_x, all=False,
                    values=[2, 15, 4, 16, 6, 12, 10, 14, 11, 7, 13, 9, 8, 5, 1, 3])}

            rules_values_y = {
                'unwtd': index_items(col_y, all=False,
                    values=[1, 16, 7, 15, 12, 3, 11, 14, 8, 10, 9, 5, 13, 6, 2, 4]),
                'iswtd': index_items(col_y, all=False,
                    values=[1, 16, 7, 12, 11, 3, 15, 8, 9, 10, 14, 5, 13, 6, 2, 4])}

            stack = get_stack(self, meta, data, xks, yks, test_views, weights,
                              extras=True)

            confirm_get_dataframe(
                self, stack, col_x, col_y,
                rules_values_x, rules_values_y)

            ################## dropx
            meta['columns'][col_x]['rules'] = {
                'x': {'dropx': {'values': [1, 3, 5, 7, 9, 11, 13, 15]}}}

            meta['columns'][col_y]['rules'] = {
                'y': {'dropx': {'values': [2, 4, 6, 8, 10, 12, 14, 16]}}}

            rules_values_x = {
                'unwtd': index_items(col_x, all=False,
                    values=[2, 4, 6, 8, 10, 12, 14, 16]),
                'iswtd': index_items(col_x, all=False,
                    values=[2, 4, 6, 8, 10, 12, 14, 16])}

            rules_values_y = {
                'unwtd': index_items(col_y, all=False,
                    values=[1, 3, 5, 7, 9, 11, 13, 15]),
                'iswtd': index_items(col_y, all=False,
                    values=[1, 3, 5, 7, 9, 11, 13, 15])}

            stack = get_stack(self, meta, data, xks, yks, test_views, weights,
                              extras=True)

            confirm_get_dataframe(
                self, stack, col_x, col_y,
                rules_values_x, rules_values_y)

            ################## slicex + sortx
            meta['columns'][col_x]['rules'] = {
                'x': {
                    'slicex': {'values': frange('4-13')},
                    'sortx': {'fixed': [4, 7, 3]}}}

            meta['columns'][col_y]['rules'] = {
                'y': {
                    'slicex': {'values': frange('7-16')},
                    'sortx': {'fixed': [7, 11, 13]}}}

            rules_values_x = {
                'unwtd': index_items(col_x, all=False,
                    values=[5, 6, 10, 12, 11, 13, 8, 9, 4, 7, 3]),
                'iswtd': index_items(col_x, all=False,
                    values=[5, 6, 12, 10, 11, 13, 9, 8, 4, 7, 3])}

            rules_values_y = {
                'unwtd': index_items(col_y, all=False,
                    values=[16, 15, 12, 14, 8, 10, 9, 7, 11, 13]),
                'iswtd': index_items(col_y, all=False,
                    values=[16, 12, 15, 8, 9, 10, 14, 7, 11, 13])}

            stack = get_stack(self, meta, data, xks, yks, test_views, weights,
                              extras=True)

            confirm_get_dataframe(
                self, stack, col_x, col_y,
                rules_values_x, rules_values_y)

            ################## slicex + dropx
            meta['columns'][col_x]['rules'] = {
                'x': {
                    'slicex': {'values': [1, 3, 5, 7, 9, 11, 13, 15]},
                    'dropx': {'values': [3, 7, 11, 15]}}}

            meta['columns'][col_y]['rules'] = {
                'y': {
                    'slicex': {'values': [2, 4, 6, 8, 10, 12, 14, 16]},
                    'dropx': {'values': [2, 6, 10, 14]}}}

            rules_values_x = {
                'unwtd': index_items(col_x, all=False,
                    values=[1, 5, 9, 13]),
                'iswtd': index_items(col_x, all=False,
                    values=[1, 5, 9, 13])}

            rules_values_y = {
                'unwtd': index_items(col_y, all=False,
                    values=[4, 8, 12, 16]),
                'iswtd': index_items(col_y, all=False,
                    values=[4, 8, 12, 16])}

            stack = get_stack(self, meta, data, xks, yks, test_views, weights,
                              extras=True)

            confirm_get_dataframe(
                self, stack, col_x, col_y,
                rules_values_x, rules_values_y)

            ################## sortx + dropx
            meta['columns'][col_x]['rules'] = {
                'x': {
                    'sortx': {'fixed': [4, 7, 3]},
                    'dropx': {'values': [5, 10]}}}

            meta['columns'][col_y]['rules'] = {
                'y': {
                    'sortx': {'fixed': [7, 11, 13]},
                    'dropx': {'values': [4, 12]}}}

            rules_values_x = {
                'unwtd': index_items(col_x, all=False,
                    values=[2, 1, 15, 16, 6, 12, 14, 11, 13, 8, 9, 4, 7, 3]),
                'iswtd': index_items(col_x, all=False,
                    values=[2, 1, 15, 16, 6, 12, 14, 11, 13, 9, 8, 4, 7, 3])}

            rules_values_y = {
                'unwtd': index_items(col_y, all=False,
                    values=[1, 2, 16, 15, 3, 14, 6, 8, 10, 9, 5, 7, 11, 13]),
                'iswtd': index_items(col_y, all=False,
                    values=[1, 2, 16, 3, 15, 8, 9, 10, 14, 5, 6, 7, 11, 13])}

            stack = get_stack(self, meta, data, xks, yks, test_views, weights,
                              extras=True)

            confirm_get_dataframe(
                self, stack, col_x, col_y,
                rules_values_x, rules_values_y)

    def test_rules_get_chain(self):

        meta = self.example_data_A_meta
        data = self.example_data_A_data

        col_x = 'religion'
        col_y = 'ethnicity'

        others = ['q5_1']

        xks = [col_x]
        yks = ['@', col_y] + others

        test_views = [
            'cbase', 'rbase', 'ebase',
            'counts', 'c%', 'r%',
            'mean']

        weights = [None, 'weight_a']

        ################## slicex + sortx + dropx
        meta['columns'][col_x]['rules'] = {
            'x': {
                'slicex': {'values': frange('4-13')},
                'sortx': {'fixed': [4, 7, 3]},
                'dropx': {'values': [6, 11]}}}

        meta['columns'][col_y]['rules'] = {
            'y': {
                'slicex': {'values': frange('7-16')},
                'sortx': {'fixed': [7, 11, 13]},
                'dropx': {'values': [11, 16]}}}

        rules_values_x = {
            'unwtd': index_items(col_x, all=False,
                values=[5, 10, 12, 13, 8, 9, 4, 7, 3]),
            'iswtd': index_items(col_x, all=False,
                values=[5, 12, 10, 13, 9, 8, 4, 7, 3])}

        rules_values_y = {
            'unwtd': index_items(col_y, all=False,
                values=[15, 12, 14, 8, 10, 9, 7, 13]),
            'iswtd': index_items(col_y, all=False,
                values=[12, 15, 8, 9, 10, 14, 7, 13])}

        confirm_xy_chains(
            self, meta, data,
            col_x, col_y, others,
            test_views, weights,
            rules_values_x, rules_values_y)

        if EXTENDED_TESTS:
            ################## slicex
            meta['columns'][col_x]['rules'] = {
                'x': {'slicex': {'values': [1, 3, 5, 7, 9, 10, 11, 13, 15]}}}

            meta['columns'][col_y]['rules'] = {
                'y': {'slicex': {'values': [2, 4, 6, 8, 10, 12, 14, 16]}}}

            rules_values_x = {
                'unwtd': index_items(col_x, all=False,
                    values=[1, 3, 5, 7, 9, 10, 11, 13, 15]),
                'iswtd': index_items(col_x, all=False,
                    values=[1, 3, 5, 7, 9, 10, 11, 13, 15])}

            rules_values_y = {
                'unwtd': index_items(col_y, all=False,
                    values=[2, 4, 6, 8, 10, 12, 14, 16]),
                'iswtd': index_items(col_y, all=False,
                    values=[2, 4, 6, 8, 10, 12, 14, 16])}

            confirm_xy_chains(
                self, meta, data,
                col_x, col_y, others,
                test_views, weights,
                rules_values_x, rules_values_y)

            ################## sortx
            meta['columns'][col_x]['rules'] = {
                'x': {'sortx': {'fixed': [5, 1, 3]}}}

            meta['columns'][col_y]['rules'] = {
                'y': {'sortx': {'fixed': [6, 2, 4]}}}

            rules_values_x = {
                'unwtd': index_items(col_x, all=False,
                    values=[2, 15, 4, 16, 6, 10, 12, 14, 11, 7, 13, 8, 9, 5, 1, 3]),
                'iswtd': index_items(col_x, all=False,
                    values=[2, 15, 4, 16, 6, 12, 10, 14, 11, 7, 13, 9, 8, 5, 1, 3])}

            rules_values_y = {
                'unwtd': index_items(col_y, all=False,
                    values=[1, 16, 7, 15, 12, 3, 11, 14, 8, 10, 9, 5, 13, 6, 2, 4]),
                'iswtd': index_items(col_y, all=False,
                    values=[1, 16, 7, 12, 11, 3, 15, 8, 9, 10, 14, 5, 13, 6, 2, 4])}

            confirm_xy_chains(
                self, meta, data,
                col_x, col_y, others,
                test_views, weights,
                rules_values_x, rules_values_y)

            ################## dropx
            meta['columns'][col_x]['rules'] = {
                'x': {'dropx': {'values': [1, 3, 5, 7, 9, 11, 13, 15]}}}

            meta['columns'][col_y]['rules'] = {
                'y': {'dropx': {'values': [2, 4, 6, 8, 10, 12, 14, 16]}}}

            rules_values_x = {
                'unwtd': index_items(col_x, all=False,
                    values=[2, 4, 6, 8, 10, 12, 14, 16]),
                'iswtd': index_items(col_x, all=False,
                    values=[2, 4, 6, 8, 10, 12, 14, 16])}

            rules_values_y = {
                'unwtd': index_items(col_y, all=False,
                    values=[1, 3, 5, 7, 9, 11, 13, 15]),
                'iswtd': index_items(col_y, all=False,
                    values=[1, 3, 5, 7, 9, 11, 13, 15])}

            confirm_xy_chains(
                self, meta, data,
                col_x, col_y, others,
                test_views, weights,
                rules_values_x, rules_values_y)

            ################## slicex + sortx
            meta['columns'][col_x]['rules'] = {
                'x': {
                    'slicex': {'values': frange('4-13')},
                    'sortx': {'fixed': [4, 7, 3]}}}

            meta['columns'][col_y]['rules'] = {
                'y': {
                    'slicex': {'values': frange('7-16')},
                    'sortx': {'fixed': [7, 11, 13]}}}

            rules_values_x = {
                'unwtd': index_items(col_x, all=False,
                    values=[5, 6, 10, 12, 11, 13, 8, 9, 4, 7, 3]),
                'iswtd': index_items(col_x, all=False,
                    values=[5, 6, 12, 10, 11, 13, 9, 8, 4, 7, 3])}

            rules_values_y = {
                'unwtd': index_items(col_y, all=False,
                    values=[16, 15, 12, 14, 8, 10, 9, 7, 11, 13]),
                'iswtd': index_items(col_y, all=False,
                    values=[16, 12, 15, 8, 9, 10, 14, 7, 11, 13])}

            stack = get_stack(self, meta, data, xks, yks, test_views, weights,
                              extras=True)

            confirm_xy_chains(
                self, meta, data,
                col_x, col_y, others,
                test_views, weights,
                rules_values_x, rules_values_y)

            ################## slicex + dropx
            meta['columns'][col_x]['rules'] = {
                'x': {
                    'slicex': {'values': [1, 3, 5, 7, 9, 11, 13, 15]},
                    'dropx': {'values': [3, 7, 11, 15]}}}

            meta['columns'][col_y]['rules'] = {
                'y': {
                    'slicex': {'values': [2, 4, 6, 8, 10, 12, 14, 16]},
                    'dropx': {'values': [2, 6, 10, 14]}}}

            rules_values_x = {
                'unwtd': index_items(col_x, all=False,
                    values=[1, 5, 9, 13]),
                'iswtd': index_items(col_x, all=False,
                    values=[1, 5, 9, 13])}

            rules_values_y = {
                'unwtd': index_items(col_y, all=False,
                    values=[4, 8, 12, 16]),
                'iswtd': index_items(col_y, all=False,
                    values=[4, 8, 12, 16])}

            confirm_xy_chains(
                self, meta, data,
                col_x, col_y, others,
                test_views, weights,
                rules_values_x, rules_values_y)

            ################## sortx + dropx
            meta['columns'][col_x]['rules'] = {
                'x': {
                    'sortx': {'fixed': [4, 7, 3]},
                    'dropx': {'values': [5, 10]}}}

            meta['columns'][col_y]['rules'] = {
                'y': {
                    'sortx': {'fixed': [7, 11, 13]},
                    'dropx': {'values': [4, 12]}}}

            rules_values_x = {
                'unwtd': index_items(col_x, all=False,
                    values=[2, 1, 15, 16, 6, 12, 14, 11, 13, 8, 9, 4, 7, 3]),
                'iswtd': index_items(col_x, all=False,
                    values=[2, 1, 15, 16, 6, 12, 14, 11, 13, 9, 8, 4, 7, 3])}

            rules_values_y = {
                'unwtd': index_items(col_y, all=False,
                    values=[1, 2, 16, 15, 3, 14, 6, 8, 10, 9, 5, 7, 11, 13]),
                'iswtd': index_items(col_y, all=False,
                    values=[1, 2, 16, 3, 15, 8, 9, 10, 14, 5, 6, 7, 11, 13])}

            confirm_xy_chains(
                self, meta, data,
                col_x, col_y, others,
                test_views, weights,
                rules_values_x, rules_values_y)

    def test_rules_coltests(self):

        meta = self.example_data_A_meta
        data = self.example_data_A_data

        col_x = 'q5_1'
        col_y = 'locality'

        xks = [col_x]
        yks = ['@', col_y]

        test_views = [
            'cbase', 'counts', 'mean']

        weights = [None]

        dk = 'test'
        fk = 'no_filter'
        xk = col_x
        yk = col_y

        stack = get_stack(
            self, meta, data, xks, yks, test_views, weights,
            extras=True, coltests=True)

        ################## slicex
        ######### counts
        meta['columns'][col_y]['rules'] = {
            'y': {'slicex': {'values': [5, 2, 3]}}}

        vk = 'x|t.props.askia.01|:|||askia tests'

        rules_values_df = pd.DataFrame([
            [np.NaN, np.NaN, np.NaN],
            [np.NaN, np.NaN, np.NaN],
            [np.NaN, np.NaN, np.NaN],
            [np.NaN, np.NaN, np.NaN],
            [np.NaN, np.NaN, np.NaN],
            ['[2]', np.NaN, np.NaN],
            [np.NaN, np.NaN, np.NaN]])

        keys = [dk, fk, xk, yk, vk]
        df = get_dataframe(stack, keys=keys, rules=True)

        actual = df.fillna(0).values.tolist()
        expected = rules_values_df.fillna(0).values.tolist()
        self.assertSequenceEqual(actual, expected)

        ######### net
        meta['columns'][col_y]['rules'] = {
            'y': {'slicex': {'values': [3, 1, 5]}}}

        vk = 'x|t.props.askia.10|x[{1,2,3}]:|||askia tests'

        rules_values_df = pd.DataFrame([
            [np.NaN, '[5]', np.NaN]])

        keys = [dk, fk, xk, yk, vk]
        df = get_dataframe(stack, keys=keys, rules=True)

        actual = df.fillna(0).values.tolist()
        expected = rules_values_df.fillna(0).values.tolist()
        self.assertSequenceEqual(actual, expected)

        ######### block net
        meta['columns'][col_y]['rules'] = {
            'y': {'slicex': {'values': [4, 1, 3]}}}

        vk = 'x|t.props.askia.10|x[{1,2}],x[{2,3}],x[{1,3}]:|||askia tests'

        rules_values_df = pd.DataFrame([
            [np.NaN, np.NaN, np.NaN],
            [np.NaN, '[3, 4]', np.NaN],
            [np.NaN, '[4]', np.NaN]])

        keys = [dk, fk, xk, yk, vk]
        df = get_dataframe(stack, keys=keys, rules=True)

        actual = df.fillna(0).values.tolist()
        expected = rules_values_df.fillna(0).values.tolist()
        self.assertSequenceEqual(actual, expected)

        ######### mean
        meta['columns'][col_y]['rules'] = {
            'y': {'slicex': {'values': [5, 2, 4]}}}

        vk = 'x|t.means.askia.10|x:|||askia tests'

        rules_values_df = pd.DataFrame([
            ['[2, 4]', np.NaN, '[2]']])

        keys = [dk, fk, xk, yk, vk]
        df = get_dataframe(stack, keys=keys, rules=True)

        actual = df.fillna(0).values.tolist()
        expected = rules_values_df.fillna(0).values.tolist()
        self.assertSequenceEqual(actual, expected)

        ################## sortx
        ######### counts
        meta['columns'][col_y]['rules'] = {
            'y': {'sortx': {'fixed': [1, 2]}}}

        vk = 'x|t.props.askia.01|:|||askia tests'

        rules_values_df = pd.DataFrame([
            [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
            [np.NaN, np.NaN, np.NaN, '[5]', np.NaN],
            [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
            ['[1]', np.NaN, np.NaN, np.NaN, np.NaN],
            [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
            [np.NaN, '[1, 2]', np.NaN, np.NaN, np.NaN],
            [np.NaN, '[1]', np.NaN, np.NaN, np.NaN]])

        keys = [dk, fk, xk, yk, vk]
        df = get_dataframe(stack, keys=keys, rules=True)

        actual = df.fillna(0).values.tolist()
        expected = rules_values_df.fillna(0).values.tolist()
        self.assertSequenceEqual(actual, expected)

        ######### net
        meta['columns'][col_y]['rules'] = {
            'y': {'sortx': {'fixed': [1, 2]}}}

        vk = 'x|t.props.askia.10|x[{1,2,3}]:|||askia tests'

        rules_values_df = pd.DataFrame([
            [np.NaN, np.NaN, np.NaN, '[4, 5]', '[4]']])

        keys = [dk, fk, xk, yk, vk]
        df = get_dataframe(stack, keys=keys, rules=True)

        actual = df.fillna(0).values.tolist()
        expected = rules_values_df.fillna(0).values.tolist()
        self.assertSequenceEqual(actual, expected)

        ######### block net
        meta['columns'][col_y]['rules'] = {
            'y': {'sortx': {'fixed': [1, 2]}}}

        vk = 'x|t.props.askia.10|x[{1,2}],x[{2,3}],x[{1,3}]:|||askia tests'

        rules_values_df = pd.DataFrame([
            ['[5]', np.NaN, np.NaN, '[2, 5]', np.NaN],
            [np.NaN, np.NaN, np.NaN, '[3, 4, 5]', '[4, 5]'],
            [np.NaN, np.NaN, np.NaN, '[4]', np.NaN]])

        keys = [dk, fk, xk, yk, vk]
        df = get_dataframe(stack, keys=keys, rules=True)

        actual = df.fillna(0).values.tolist()
        expected = rules_values_df.fillna(0).values.tolist()
        self.assertSequenceEqual(actual, expected)

        ######### mean
        meta['columns'][col_y]['rules'] = {
            'y': {'sortx': {'fixed': [1, 2]}}}

        vk = 'x|t.means.askia.10|x:|||askia tests'

        rules_values_df = pd.DataFrame([
            ['[1]', '[1, 2, 3, 4]', '[1, 2, 3]', np.NaN, '[1]']])

        keys = [dk, fk, xk, yk, vk]
        df = get_dataframe(stack, keys=keys, rules=True)

        actual = df.fillna(0).values.tolist()
        expected = rules_values_df.fillna(0).values.tolist()
        self.assertSequenceEqual(actual, expected)

        ################## dropx
        ######### counts
        meta['columns'][col_y]['rules'] = {
            'y': {'dropx': {'values': [1, 4]}}}

        vk = 'x|t.props.askia.01|:|||askia tests'

        rules_values_df = pd.DataFrame([
            [np.NaN, np.NaN, np.NaN],
            [np.NaN, np.NaN, np.NaN],
            [np.NaN, np.NaN, np.NaN],
            [np.NaN, np.NaN, np.NaN],
            [np.NaN, np.NaN, np.NaN],
            [np.NaN, np.NaN, '[2]'],
            [np.NaN, np.NaN, np.NaN]])

        keys = [dk, fk, xk, yk, vk]
        df = get_dataframe(stack, keys=keys, rules=True)

        actual = df.fillna(0).values.tolist()
        expected = rules_values_df.fillna(0).values.tolist()
        self.assertSequenceEqual(actual, expected)

        ######### net
        meta['columns'][col_y]['rules'] = {
            'y': {'dropx': {'values': [1, 3]}}}

        vk = 'x|t.props.askia.10|x[{1,2,3}]:|||askia tests'

        rules_values_df = pd.DataFrame([
            ['[4]', np.NaN, np.NaN]])

        keys = [dk, fk, xk, yk, vk]
        df = get_dataframe(stack, keys=keys, rules=True)

        actual = df.fillna(0).values.tolist()
        expected = rules_values_df.fillna(0).values.tolist()
        self.assertSequenceEqual(actual, expected)

        ######### block net
        meta['columns'][col_y]['rules'] = {
            'y': {'dropx': {'values': [2, 4]}}}

        vk = 'x|t.props.askia.10|x[{1,2}],x[{2,3}],x[{1,3}]:|||askia tests'

        rules_values_df = pd.DataFrame([
            ['[5]', '[5]', np.NaN],
            ['[3, 5]', np.NaN, np.NaN],
            [np.NaN, np.NaN, np.NaN]])

        keys = [dk, fk, xk, yk, vk]
        df = get_dataframe(stack, keys=keys, rules=True)

        actual = df.fillna(0).values.tolist()
        expected = rules_values_df.fillna(0).values.tolist()
        self.assertSequenceEqual(actual, expected)

        ######### mean
        meta['columns'][col_y]['rules'] = {
            'y': {'dropx': {'values': [1, 3]}}}

        vk = 'x|t.means.askia.10|x:|||askia tests'

        rules_values_df = pd.DataFrame([
            [np.NaN, '[2]', '[2, 4]']])

        keys = [dk, fk, xk, yk, vk]
        df = get_dataframe(stack, keys=keys, rules=True)

        actual = df.fillna(0).values.tolist()
        expected = rules_values_df.fillna(0).values.tolist()
        self.assertSequenceEqual(actual, expected)

    def test_rules_coltests_flag_bases(self):

        meta = self.example_data_A_meta
        data = self.example_data_A_data

        col_x = 'q5_1'
        col_y = 'locality'

        xks = [col_x]
        yks = ['@', col_y]

        test_views = [
            'cbase', 'counts', 'mean']

        weights = [None]

        dk = 'test'
        fk = 'no_filter'
        xk = col_x
        yk = col_y

        minimum = 1000
        small = 2000

        stack = get_stack(
            self, meta, data, xks, yks, test_views, weights,
            extras=True, coltests=True, flag_bases=[minimum, small])

        ################## slicex
        ######### counts
        meta['columns'][col_y]['rules'] = {
            'y': {'slicex': {'values': [5, 2, 3]}}}

        vk = 'x|t.props.Dim.05|:|||askia tests'

        rules_values_df = pd.DataFrame([
            ['**', np.NaN, '[2]*'],
            ['**', np.NaN, '*'],
            ['**', np.NaN, '*'],
            ['**', np.NaN, '*'],
            ['**', np.NaN, '*'],
            ['**', np.NaN, '*'],
            ['**', np.NaN, '*']])

        keys = [dk, fk, xk, yk, vk]
        df = get_dataframe(stack, keys=keys, rules=True)

        cbase = 'x|f|x:|||cbase'
        keys_cbase = [dk, fk, xk, yk, cbase]
        df_cbase = get_dataframe(stack, keys=keys_cbase, rules=True)

        is_minimum = [c<=minimum for c in df_cbase.values[0]]
        is_small = [c>minimum and c<=small for c in df_cbase.values[0]]

        actual = is_minimum
        expected = [True, False, False]
        self.assertSequenceEqual(actual, expected)

        actual = is_small
        expected = [False, False, True]
        self.assertSequenceEqual(actual, expected)

        actual = df.fillna(0).values.tolist()
        expected = rules_values_df.fillna(0).values.tolist()
        self.assertSequenceEqual(actual, expected)

        ################## sortx
        ######### counts
        meta['columns'][col_y]['rules'] = {
            'y': {'sortx': {'fixed': [1, 2]}}}

        vk = 'x|t.props.Dim.05|:|||askia tests'

        rules_values_df = pd.DataFrame([
            ['[1, 2]*', '**', '**', np.NaN, np.NaN],
            ['*', '**', '**', '[2, 3]', np.NaN],
            ['*', '**', '**', np.NaN, np.NaN],
            ['[1]*', '**', '**', np.NaN, '[1]'],
            ['*', '**', '**', np.NaN, np.NaN],
            ['*', '**', '**', np.NaN, np.NaN],
            ['*', '**', '**', np.NaN, np.NaN]])

        keys = [dk, fk, xk, yk, vk]
        df = get_dataframe(stack, keys=keys, rules=True)

        cbase = 'x|f|x:|||cbase'
        keys_cbase = [dk, fk, xk, yk, cbase]
        df_cbase = get_dataframe(stack, keys=keys_cbase, rules=True)

        is_minimum = [c<=minimum for c in df_cbase.values[0]]
        is_small = [c>minimum and c<=small for c in df_cbase.values[0]]

        actual = is_minimum
        expected = [False, True, True, False, False]
        self.assertSequenceEqual(actual, expected)

        actual = is_small
        expected = [True, False, False, False, False]
        self.assertSequenceEqual(actual, expected)

        actual = df.fillna(0).values.tolist()
        expected = rules_values_df.fillna(0).values.tolist()
        self.assertSequenceEqual(actual, expected)

        ################## dropx
        ######### counts
        meta['columns'][col_y]['rules'] = {
            'y': {'dropx': {'values': [1, 4]}}}

        vk = 'x|t.props.Dim.05|:|||askia tests'

        rules_values_df = pd.DataFrame([
            [np.NaN, '[2]*', '**'],
            [np.NaN, '*', '**'],
            [np.NaN, '*', '**'],
            [np.NaN, '*', '**'],
            [np.NaN, '*', '**'],
            [np.NaN, '*', '**'],
            [np.NaN, '*', '**']])

        keys = [dk, fk, xk, yk, vk]
        df = get_dataframe(stack, keys=keys, rules=True)

        cbase = 'x|f|x:|||cbase'
        keys_cbase = [dk, fk, xk, yk, cbase]
        df_cbase = get_dataframe(stack, keys=keys_cbase, rules=True)

        is_minimum = [c<=minimum for c in df_cbase.values[0]]
        is_small = [c>minimum and c<=small for c in df_cbase.values[0]]

        actual = is_minimum
        expected = [False, False, True]
        self.assertSequenceEqual(actual, expected)

        actual = is_small
        expected = [False, True, False]
        self.assertSequenceEqual(actual, expected)

        actual = df.fillna(0).values.tolist()
        expected = rules_values_df.fillna(0).values.tolist()
        self.assertSequenceEqual(actual, expected)

# ##################### Helper functions #####################

def index_items(col, values, all=False):
    """
    Return a correctly formed list of tuples to matching an index.
    """

    items = [
        (col, i)
        for i in values
    ]

    if all: items = [(col, 'All')] + items

    return items

def confirm_frequencies(self, meta, data,
                        weights,
                        col,
                        rules_values_x,
                        rules_values_y):
    """
    Confirms all variations of rules applied with frequency.
    """

    df = frequency(meta, data, x=col)
    natural_x = df.index.values.tolist()
    natural_y = natural_x

    frequ_x = [(col, '@')]
    frequ_y = frequ_x

    for weight in weights:

        if weight is None:
            rules_x = rules_values_x['unwtd']
            rules_y = rules_values_y['unwtd']
        else:
            rules_x = rules_values_x['iswtd']
            rules_y = rules_values_y['iswtd']

        # rules=True
        fx = frequency(meta, data, x=col, weight=weight, rules=True)
        fy = frequency(meta, data, y=col, weight=weight, rules=True)

#         print fx
#         print zip(*rules_x)[1]
#         print zip(*rules_y)[1]
        confirm_index_columns(self, fx, rules_x, frequ_x)
        confirm_index_columns(self, fy, frequ_x, rules_y)

        # rules=False
        fx = frequency(meta, data, x=col, weight=weight, rules=False)
        fy = frequency(meta, data, y=col, weight=weight, rules=False)
        confirm_index_columns(self, fx, natural_x, frequ_x)
        confirm_index_columns(self, fy, frequ_x, natural_y)

        # rules=x
        fx = frequency(meta, data, x=col, weight=weight, rules=['x'])
        fy = frequency(meta, data, y=col, weight=weight, rules=['x'])
        confirm_index_columns(self, fx, rules_x, frequ_x)
        confirm_index_columns(self, fy, frequ_x, natural_y)

        # rules=y
        fx = frequency(meta, data, x=col, weight=weight, rules=['y'])
        fy = frequency(meta, data, y=col, weight=weight, rules=['y'])
        confirm_index_columns(self, fx, natural_x, frequ_x)
        confirm_index_columns(self, fy, frequ_x, rules_y)

        # rules=xy
        fx = frequency(meta, data, x=col, weight=weight, rules=['x', 'y'])
        fy = frequency(meta, data, y=col, weight=weight, rules=['x', 'y'])
        confirm_index_columns(self, fx, rules_x, frequ_x)
        confirm_index_columns(self, fy, frequ_x, rules_y)

def confirm_crosstabs(self, meta, data,
                      weights,
                      col_x, col_y,
                      rules_values_x,
                      rules_values_y):
    """
    Confirms all variations of rules applied with frequency.
    """

    fx = frequency(meta, data, x=col_x)
    natural_x = fx.index.values.tolist()

    fy = frequency(meta, data, y=col_y)
    natural_y = fy.columns.values.tolist()

    for weight in weights:

        if weight is None:
            rules_x = rules_values_x['unwtd']
            rules_y = rules_values_y['unwtd']
        else:
            rules_x = rules_values_x['iswtd']
            rules_y = rules_values_y['iswtd']

        for xtotal in [False, True]:
            # rules=True
            df = crosstab(meta, data, col_x, col_y, weight=weight, rules=True, xtotal=xtotal)
            confirm_index_columns(self, df, rules_x, rules_y)

            # print df
#             print df.index
#             print df.columns
#             print zip(*rules_x)[1]
#             print zip(*rules_y)[1]

            # rules=False
            df = crosstab(meta, data, col_x, col_y, weight=weight, rules=False, xtotal=xtotal)
            confirm_index_columns(self, df, natural_x, natural_y)

            # rules=x
            df = crosstab(meta, data, col_x, col_y, weight=weight, rules=['x'], xtotal=xtotal)
            confirm_index_columns(self, df, rules_x, natural_y)

            # rules=y
            df = crosstab(meta, data, col_x, col_y, weight=weight, rules=['y'], xtotal=xtotal)
            confirm_index_columns(self, df, natural_x, rules_y)

            # rules=xy
            df = crosstab(meta, data, col_x, col_y, weight=weight, rules=['x', 'y'], xtotal=xtotal)
            confirm_index_columns(self, df, rules_x, rules_y)

def confirm_get_dataframe(self, stack, col_x, col_y,
                          rules_values_x, rules_values_y):
    """
    Confirms all variations of rules applied with frequency.
    """

    keys = ['dk', 'fk', 'xk', 'yk', 'vk']
    keys[0] = dk = 'test'
    keys[1] = fk = 'no_filter'
    keys[2] = xk = col_x
    keys[3] = yk = col_y

    meta = stack[dk].meta
    data = stack[dk].data

    vks = stack.describe()['view'].values.tolist()

    for xk in [col_x]:
        keys[2] = xk

        for yk in ['@', col_y]:
            if xk=='@' and yk=='@':
                continue
            keys[3] = yk

            for vk in vks:
                keys[4] = vk

#                 if 'mean' in vk:
#                     print vk

                rules_x, natural_x, rules_y, natural_y = get_xy_values(
                    meta, data,
                    col_x, col_y,
                    xk, yk, vk,
                    rules_values_x, rules_values_y
                )

                # rules=True
                df = get_dataframe(stack, keys=keys, rules=True)
#                 print df
#                 print df.index
#                 print df.columns
#                 print zip(*rules_x)[1]
#                 print zip(*rules_y)[1]
                confirm_index_columns(self, df, rules_x, rules_y)

                # rules=False
                df = get_dataframe(stack, keys=keys, rules=False)
                confirm_index_columns(self, df, natural_x, natural_y)

                # rules=x
                df = get_dataframe(stack, keys=keys, rules=['x'])
                confirm_index_columns(self, df, rules_x, natural_y)

                # rules=y
                df = get_dataframe(stack, keys=keys, rules=['y'])
                confirm_index_columns(self, df, natural_x, rules_y)

                # rules=xy
                df = get_dataframe(stack, keys=keys, rules=['x', 'y'])
                confirm_index_columns(self, df, rules_x, rules_y)

def confirm_xy_chains(self, meta, data, col_x, col_y, others, views, weights,
                      rules_values_x, rules_values_y):

    stack = get_stack(
        self, meta, data,
        [col_x],
        ['@', col_y] + others,
        views,
        weights,
        extras=True)

    confirm_get_xchain(
        self, stack, col_x, col_y, others,
        rules_values_x, rules_values_y)

    stack = get_stack(
        self, meta, data,
        [col_x] + others,
        [col_y],
        views,
        weights,
        extras=True)

    confirm_get_ychain(
        self, stack, col_x, col_y, others,
        rules_values_x, rules_values_y)

def confirm_get_xchain(self, stack, col_x, col_y, others,
                       rules_values_x, rules_values_y):
    """
    Confirms all variations of rules applied with frequency.
    """

    keys = ['dk', 'fk', 'xk', 'yk', 'vk']
    keys[0] = dk = 'test'
    keys[1] = fk = 'no_filter'
    keys[2] = xk = col_x
    keys[3] = yk = col_y

    meta = stack[dk].meta
    data = stack[dk].data

    xks = [col_x]
    yks = ['@', col_y] + others

    confirm_get_chain(
        self,
        meta, data,
        stack, keys,
        col_x, col_y,
        xks, yks,
        rules_values_x, rules_values_y,
        others)

def confirm_get_ychain(self, stack, col_x, col_y, others,
                       rules_values_x, rules_values_y):
    """
    Confirms all variations of rules applied with frequency.
    """

    keys = ['dk', 'fk', 'xk', 'yk', 'vk']
    keys[0] = dk = 'test'
    keys[1] = fk = 'no_filter'
    keys[2] = xk = col_x
    keys[3] = yk = col_y

    meta = stack[dk].meta
    data = stack[dk].data

    xks = [col_x] + others
    yks = [col_y]

    confirm_get_chain(
        self,
        meta, data,
        stack, keys,
        col_x, col_y,
        xks, yks,
        rules_values_x, rules_values_y,
        others)

def confirm_get_chain(self,
                      meta, data,
                      stack, keys,
                      col_x, col_y,
                      xks, yks,
                      rules_values_x, rules_values_y,
                      others=[]):

    vks = stack.describe()['view'].values.tolist()

    weight = None
    chain_true_unwtd = stack.get_chain(x=xks, y=yks, views=vks, rules=True, rules_weight=weight)
    chain_false_unwtd = stack.get_chain(x=xks, y=yks, views=vks, rules=False, rules_weight=weight)
    chain_x_unwtd = stack.get_chain(x=xks, y=yks, views=vks, rules=['x'], rules_weight=weight)
    chain_y_unwtd = stack.get_chain(x=xks, y=yks, views=vks, rules=['y'], rules_weight=weight)
    chain_xy_unwtd = stack.get_chain(x=xks, y=yks, views=vks, rules=['x', 'y'], rules_weight=weight)

    weight = 'weight_a'
    chain_true_wtd = stack.get_chain(x=xks, y=yks, views=vks, rules=True, rules_weight=weight)
    chain_false_wtd = stack.get_chain(x=xks, y=yks, views=vks, rules=False, rules_weight=weight)
    chain_x_wtd = stack.get_chain(x=xks, y=yks, views=vks, rules=['x'], rules_weight=weight)
    chain_y_wtd = stack.get_chain(x=xks, y=yks, views=vks, rules=['y'], rules_weight=weight)
    chain_xy_wtd = stack.get_chain(x=xks, y=yks, views=vks, rules=['x', 'y'], rules_weight=weight)

    for xk in xks:
        keys[2] = xk

        for yk in yks:
            if xk=='@' and yk=='@':
                continue
            keys[3] = yk

            for vk in vks:
                keys[4] = vk

                for weight in [None, 'weight_a']:

#                     if xk=='q5_1' and yk=='ethnicity' and vk=='x|f|x:|||ebase':
#                         print xk, yk, vk

#                     if vk=='x|f|:y|||rbase' and yk=='q5_1':
#                         print vk

                    rules_x, natural_x, rules_y, natural_y = get_xy_values(
                        meta, data,
                        col_x, col_y,
                        xk, yk, vk,
                        rules_values_x, rules_values_y,
                        others,
                        rules_weight=weight
                    )

                    # rules=True
                    if weight is None:

                        df = get_dataframe(chain_true_unwtd, keys=keys, rules=False)
#                         print df
#                         print df.index
#                         print df.columns
#                         print zip(*rules_x)[1]
#                         print zip(*rules_y)[1]
                        confirm_index_columns(self, df, rules_x, rules_y)

                        # rules=False
                        df = get_dataframe(chain_false_unwtd, keys=keys, rules=False)
                        confirm_index_columns(self, df, natural_x, natural_y)

                        # rules=x
                        df = get_dataframe(chain_x_unwtd, keys=keys, rules=False)
                        confirm_index_columns(self, df, rules_x, natural_y)

                        # rules=y
                        df = get_dataframe(chain_y_unwtd, keys=keys, rules=False)
                        confirm_index_columns(self, df, natural_x, rules_y)

                        # rules=xy
                        df = get_dataframe(chain_xy_unwtd, keys=keys, rules=False)
                        confirm_index_columns(self, df, rules_x, rules_y)

                    else:
                        df = get_dataframe(chain_true_wtd, keys=keys, rules=False)
#                         print df
#                         print df.index
#                         print df.columns
#                         print zip(*rules_x)[1]
#                         print zip(*rules_y)[1]
                        confirm_index_columns(self, df, rules_x, rules_y)

                        # rules=False
                        df = get_dataframe(chain_false_wtd, keys=keys, rules=False)
                        confirm_index_columns(self, df, natural_x, natural_y)

                        # rules=x
                        df = get_dataframe(chain_x_wtd, keys=keys, rules=False)
                        confirm_index_columns(self, df, rules_x, natural_y)

                        # rules=y
                        df = get_dataframe(chain_y_wtd, keys=keys, rules=False)
                        confirm_index_columns(self, df, natural_x, rules_y)

                        # rules=xy
                        df = get_dataframe(chain_xy_wtd, keys=keys, rules=False)
                        confirm_index_columns(self, df, rules_x, rules_y)



def get_xy_values(meta, data,
                  col_x, col_y,
                  xk, yk, vk,
                  rules_values_x, rules_values_y,
                  others=[], rules_weight='auto'):

    v_method = vk.split('|')[1]
    relation = vk.split('|')[2]
    relative = vk.split('|')[3]
    weight = vk.split('|')[4]
    shortnam = vk.split('|')[5]

    condensed_x = relation.split(":")[0].startswith('x') or v_method.startswith('d.')
    condensed_y = relation.split(":")[1].startswith('y')

    if rules_weight=='auto':
        rules_weight = None if weight=='' else weight

    if rules_weight is None:
        rules_x = rules_values_x['unwtd']
        rules_y = rules_values_y['unwtd']
    else:
        rules_x = rules_values_x['iswtd']
        rules_y = rules_values_y['iswtd']

    if xk in others:
        fx = frequency(meta, data, x=xk)
        natural_x = fx.index.values.tolist()
        natural_x.remove((xk, 'All'))
        rules_x = natural_x
        if condensed_x:
            if shortnam=='Block net':
                rules_x = natural_x = [
                    (xk, 'bn1'),
                    (xk, 'bn2'),
                    (xk, 'bn3')]
            elif shortnam in ['cbase', 'ebase']:
                rules_x = natural_x = [(xk, 'All')]
            else:
                rules_x = natural_x = [(xk, shortnam)]
    elif xk=='@':
        if condensed_x:
            if shortnam=='Block net':
                rules_x = natural_x = [
                    (col_x, 'bn1'),
                    (col_x, 'bn2'),
                    (col_x, 'bn3')]
            elif shortnam in ['cbase', 'ebase']:
                rules_x = natural_x = [(col_y, 'All')]
            else:
                rules_x = natural_x = [(col_y, shortnam)]
        else:
            rules_x = natural_x = [(col_y, '@')]
    elif condensed_x:
        if shortnam=='Block net':
            rules_x = natural_x = [
                (col_x, 'bn1'),
                (col_x, 'bn2'),
                (col_x, 'bn3')]
        elif shortnam in ['cbase', 'ebase']:
            rules_x = natural_x = [(xk, 'All')]
        else:
            rules_x = natural_x = [(xk, shortnam)]
    else:
        fx = frequency(meta, data, x=col_x)
        natural_x = fx.index.values.tolist()
        natural_x.remove((col_x, 'All'))

    if yk in others:
        fy = frequency(meta, data, y=yk)
        natural_y = fy.columns.values.tolist()
        natural_y.remove((yk, 'All'))
        rules_y = natural_y
        if condensed_y:
            if shortnam=='Block net':
                rules_y = natural_y = [
                    (yk, 'bn1'),
                    (yk, 'bn2'),
                    (yk, 'bn3')]
            elif shortnam in ['rbase']:
                rules_y = natural_y = [(yk, 'All')]
            else:
                rules_y = natural_y = [(yk, shortnam)]
    elif yk=='@':
        if condensed_y:
            if shortnam=='Block net':
                rules_y = natural_y = [
                    (col_y, 'bn1'),
                    (col_y, 'bn2'),
                    (col_y, 'bn3')]
            elif shortnam in ['rbase']:
                rules_y = natural_y = [(col_x, 'All')]
            else:
                rules_y = natural_y = [(col_x, shortnam)]
        else:
            rules_y = natural_y = [(col_x, '@')]
    elif condensed_y:
        if shortnam=='Block net':
            rules_y = natural_y = [
                (col_y, 'bn1'),
                (col_y, 'bn2'),
                (col_y, 'bn3')]
        elif shortnam in ['rbase']:
            rules_y = natural_y = [(col_y, 'All')]
        else:
            rules_y = natural_y = [(col_y, shortnam)]
    else:
        fy = frequency(meta, data, y=col_y)
        natural_y = fy.columns.values.tolist()
        natural_y.remove((col_y, 'All'))

    return rules_x, natural_x, rules_y, natural_y

def str_index_values(index):
    """
    Make sure level 1 of the multiindex are all strings
    """
    values = index.values.tolist()
    values = zip(*[zip(*values)[0], [str(i) for i in zip(*values)[1]]])
    return values

def confirm_index_columns(self, df, expected_x, expected_y):
    """
    Confirms index and columns are as expected.
    """
#     global COUNTER

#     actual_x = str_index_values(df.index)
#     actual_y = str_index_values(df.columns)
    actual_x = df.index.values.tolist()
    actual_y = df.columns.values.tolist()

    # print
    # print actual_x
    # print expected_x
    # print actual_y
    # print expected_y

    # Remove xtotal from columns if present
    if len(df.columns.levels[0])>1:
        actual_y = actual_y[1:]
    self.assertEqual(actual_x, expected_x)
    self.assertEqual(actual_y, expected_y)

#     COUNTER = COUNTER + 2
#     print COUNTER

def get_stack(self, meta, data, xks, yks, views, weights,
              extras=False, coltests=False, flag_bases=None):

    stack = Stack('test')
    stack.add_data('test', data, meta)
    stack.add_link(x=xks, y=yks, views=views, weights=weights)

    if extras or coltests:

        # Add a basic net
        net_views = ViewMapper(
            template={
                'method': QuantipyViews().frequency,
                'kwargs': {'iterators': {'rel_to': [None, 'y']}}})
        net_views.add_method(
            name='Net 1-3',
            kwargs={'logic': [1, 2, 3], 'axis': 'x',
                    'text': {'en-GB': '1-3'}})
        stack.add_link(x=xks, y=yks, views=net_views, weights=weights)

        # Add block net
        net_views.add_method(
            name='Block net',
            kwargs={
                'logic': [
                    {'bn1': [1, 2]},
                    {'bn2': [2, 3]},
                    {'bn3': [1, 3]}], 'axis': 'x'})
        stack.add_link(x=xks, y=yks, views=net_views.subset(['Block net']), weights=weights)

        # Add NPS
        ## TO DO

        # Add standard deviation
        stddev_views = ViewMapper(
            template = {
                'method': QuantipyViews().descriptives,
                'kwargs': {'stats': 'stddev'}})
        stddev_views.add_method(name='stddev')
        stack.add_link(x=xks, y=yks, views=stddev_views, weights=weights)

    if coltests:

        if flag_bases is None:
            test_views = ViewMapper(
                template={
                    'method': QuantipyViews().coltests,
                    'kwargs': {
                        'mimic': 'askia',
                        'iterators': {
                            'metric': ['props', 'means'],
                            'level': ['low', 'mid', 'high']}}})
        else:
            test_views = ViewMapper(
                template={
                    'method': QuantipyViews().coltests,
                    'kwargs': {
                        'mimic': 'Dim',
                        'flag_bases': flag_bases,
                        'iterators': {
                            'metric': ['props', 'means'],
                            'level': ['low', 'mid', 'high']}}})

        test_views.add_method('askia tests')
        stack.add_link(x=xks, y=yks, views=test_views)

    return stack
