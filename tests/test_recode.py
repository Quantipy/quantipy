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
    recode,
    frequency,
    crosstab
)
from quantipy.core.tools.view.query import get_dataframe

class TestRules(unittest.TestCase):

    def setUp(self):
        self.path = './tests/'
#         self.path = ''
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
                  
    def test_recode_parameters(self):
                     
        meta = self.example_data_A_meta
        data = self.example_data_A_data
         
        col_a = 'q1'
        col_b = 'gender'
        col_c = 'gender_r'
                   
        ################## target
        mapper = {
            1: {col_a: [1, 3, 5, 7, 9, 98]},
            2: {col_a: [2, 4, 6, 8, 96, 99]}}
        s = recode(
            meta, data,
            target=col_b,
            mapper=mapper)
        self.assertTrue(s.name, col_b)
        
        ################## explicit mapper
        mapper = {
            1: {col_a: [1, 3, 5, 7, 9, 98]},
            2: {col_a: [2, 4, 6, 8, 96, 99]}}
        data[col_b] = recode(
            meta, data,
            target=col_b,
            mapper=mapper)
        df = check(meta, data, col_a, col_b)
        
        actual = df.values.tolist()
        expected = [
            [8255.0, 8255.0, 3791.0, 4464.0], 
            [ 297.0,  297.0,  297.0,    0.0], 
            [ 397.0,  397.0,    0.0,  397.0], 
            [2298.0, 2298.0, 2298.0,    0.0], 
            [2999.0, 2999.0,    0.0, 2999.0], 
            [ 194.0,  194.0,  194.0,    0.0], 
            [ 477.0,  477.0,    0.0,  477.0], 
            [ 894.0,  894.0,  894.0,    0.0], 
            [ 131.0,  131.0,    0.0,  131.0], 
            [   4.0,    4.0,    4.0,    0.0], 
            [  91.0,   91.0,    0.0,   91.0], 
            [ 104.0,  104.0,  104.0,    0.0], 
            [ 369.0,  369.0,    0.0,  369.0]]
        self.assertSequenceEqual(actual, expected)
        
        ################## default mapper
        mapper = {
            1: [1, 3, 5, 7, 9, 98],
            2: [2, 4, 6, 8, 96, 99]}
        data[col_b] = recode(
            meta, data,
            target=col_b,
            mapper=mapper,
            default=col_a)
        df = check(meta, data, col_a, col_b)
        self.assertSequenceEqual(actual, expected)
        
        ################## initialize np.NaN
        meta['columns'][col_c] = copy.deepcopy(meta['columns'][col_b])
        data[col_c] = data[col_b].copy()
        
        actual = data[col_c].values.tolist()
        expected = data[col_b].values.tolist()
        self.assertSequenceEqual(actual, expected)
        
        mapper = {
            1: {col_a: [1, 3, 5, 7, 9, 98]}}
        data[col_c] = recode(
            meta, data,
            target=col_c,
            mapper=mapper,
            initialize=np.NaN)
        df = check(meta, data, col_a, col_c)
        
        actual = df.columns.tolist()
        expected = [
            ('q1', '@'), 
            ('gender_r', 'All'), 
            ('gender_r', 1)]
        self.assertSequenceEqual(actual, expected)
        
        actual = df.values.tolist()
        expected = [
            [8255.0, 3791.0, 3791.0], 
            [ 297.0,  297.0,  297.0], 
            [ 397.0,    0.0,    0.0], 
            [2298.0, 2298.0, 2298.0], 
            [2999.0,    0.0,    0.0], 
            [ 194.0,  194.0,  194.0], 
            [ 477.0,    0.0,    0.0], 
            [ 894.0,  894.0,  894.0], 
            [ 131.0,    0.0,    0.0], 
            [   4.0,    4.0,    4.0], 
            [  91.0,    0.0,    0.0], 
            [ 104.0,  104.0,  104.0], 
            [ 369.0,    0.0,    0.0]]
        self.assertSequenceEqual(actual, expected)
        
        ################## initialize with copy
        meta['columns'][col_c] = copy.deepcopy(meta['columns'][col_b])
        data[col_c] = np.NaN
        
        actual = data[col_c].values.tolist()
        expected = [np.NaN] * len(actual)
        self.assertTrue(all([np.isnan(value) for value in actual]))
        self.assertTrue(all([np.isnan(value) for value in expected]))
        self.assertEqual(len(actual), len(expected))
        
        mapper = {
            1: {col_a: [1, 3, 5, 7, 9, 98]}}
        data[col_c] = recode(
            meta, data,
            target=col_c,
            mapper=mapper,
            initialize=col_b)
        df = check(meta, data, col_a, col_c)
        
        actual = df.columns.tolist()
        expected = [
            ('q1', '@'), 
            ('gender_r', 'All'), 
            ('gender_r', 1), 
            ('gender_r', 2)]
        self.assertSequenceEqual(actual, expected)
        
        actual = df.values.tolist()
        expected = [
            [8255.0, 8255.0, 3791.0, 4464.0], 
            [ 297.0,  297.0,  297.0,    0.0], 
            [ 397.0,  397.0,    0.0,  397.0], 
            [2298.0, 2298.0, 2298.0,    0.0], 
            [2999.0, 2999.0,    0.0, 2999.0], 
            [ 194.0,  194.0,  194.0,    0.0], 
            [ 477.0,  477.0,    0.0,  477.0], 
            [ 894.0,  894.0,  894.0,    0.0], 
            [ 131.0,  131.0,    0.0,  131.0], 
            [   4.0,    4.0,    4.0,    0.0], 
            [  91.0,   91.0,    0.0,   91.0], 
            [ 104.0,  104.0,  104.0,    0.0], 
            [ 369.0,  369.0,    0.0,  369.0]]
        self.assertSequenceEqual(actual, expected)
        
        ################## append
        meta['columns'][col_c] = copy.deepcopy(meta['columns'][col_b])
        meta['columns'][col_c]['type'] = "delimited set"
        meta['columns'][col_c]['values'] = [
            {"text": {"en-GB": "Male"}, "value": 1}, 
            {"text": {"en-GB": "Female"}, "value": 2},
            {"text": {"en-GB": "Rabbit"}, "value": 3}, 
            {"text": {"en-GB": "Hare"}, "value": 4}]
        data[col_c] = np.NaN
        
        actual = data[col_c].values.tolist()
        expected = [np.NaN] * len(actual)
        self.assertTrue(all([np.isnan(value) for value in actual]))
        self.assertTrue(all([np.isnan(value) for value in expected]))
        self.assertEqual(len(actual), len(expected))
        
        mapper = {
            3: [1],
            4: [2]}
        data[col_c] = recode(
            meta, data,
            target=col_c,
            mapper=mapper,
            default=col_b,
            initialize=col_b,
            append=True)
        df = check(meta, data, col_b, col_c)
        
        actual = df.columns.tolist()
        expected = [
            ('gender', '@'), 
            ('gender_r', 'All'), 
            ('gender_r', 1), 
            ('gender_r', 2), 
            ('gender_r', 3), 
            ('gender_r', 4)]
        self.assertSequenceEqual(actual, expected)
        
        actual = df.values.tolist()
        expected = [
            [8255.0, 8255.0, 3791.0, 4464.0, 3791.0, 4464.0], 
            [3791.0, 3791.0, 3791.0,    0.0, 3791.0,    0.0], 
            [4464.0, 4464.0,    0.0, 4464.0,    0.0, 4464.0]]
        self.assertSequenceEqual(actual, expected)
        
        ################## fillna
        meta['columns'][col_c] = copy.deepcopy(meta['columns'][col_b])
        meta['columns'][col_c]['values'] = [
            {"text": {"en-GB": "Male"}, "value": 1}, 
            {"text": {"en-GB": "Female"}, "value": 2},
            {"text": {"en-GB": "Unknown"}, "value": 99}]
        data[col_c] = np.NaN
        
        actual = data[col_c].values.tolist()
        expected = [np.NaN] * len(actual)
        self.assertTrue(all([np.isnan(value) for value in actual]))
        self.assertTrue(all([np.isnan(value) for value in expected]))
        self.assertEqual(len(actual), len(expected))
        
        mapper = {
            1: {col_a: [1, 3, 5, 7, 9, 98]}}
        data[col_c] = recode(
            meta, data,
            target=col_c,
            mapper=mapper,
            fillna=99)
        df = check(meta, data, col_a, col_c)
        
        actual = df.columns.tolist()
        expected = [
            ('q1', '@'), 
            ('gender_r', 'All'), 
            ('gender_r', 1), 
            ('gender_r', 99)]
        self.assertSequenceEqual(actual, expected)
        
        actual = df.values.tolist()
        expected = [
            [8255.0, 8255.0, 3791.0, 4464.0], 
            [ 297.0,  297.0,  297.0,    0.0], 
            [ 397.0,  397.0,    0.0,  397.0], 
            [2298.0, 2298.0, 2298.0,    0.0], 
            [2999.0, 2999.0,    0.0, 2999.0], 
            [ 194.0,  194.0,  194.0,    0.0], 
            [ 477.0,  477.0,    0.0,  477.0], 
            [ 894.0,  894.0,  894.0,    0.0], 
            [ 131.0,  131.0,    0.0,  131.0], 
            [   4.0,    4.0,    4.0,    0.0], 
            [  91.0,   91.0,    0.0,   91.0], 
            [ 104.0,  104.0,  104.0,    0.0], 
            [ 369.0,  369.0,    0.0,  369.0]]
        self.assertSequenceEqual(actual, expected)
        
        
# ##################### Helper functions #####################

def check(meta, data, original, recode):
    """
    Concatenate a frequency and crosstab to verify a recode
    """
    
    f = frequency(meta, data, original)
    ct = crosstab(meta, data, original, recode)
    df = pd.concat([f, ct], axis=1)
    
    return df
