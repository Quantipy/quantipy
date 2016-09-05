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

class TestDataSet(unittest.TestCase):

    def check_freq(self, dataset, var, show='values'):
        return freq(dataset._meta, dataset._data, var, show=show)

    def _get_dataset(self):
        path = os.path.dirname(os.path.abspath(__file__)) + '/'
        name = 'Example Data (A)'
        casedata = '{}.csv'.format(name)
        metadata = '{}.json'.format(name)
        dataset = qp.DataSet(name)
        dataset.read_quantipy(path+metadata, path+casedata)
        return dataset

    def test_read_quantipy(self):
        dataset = self._get_dataset()
        self.assertTrue(isinstance(dataset._data, pd.DataFrame))
        self.assertTrue(isinstance(dataset._meta, dict))

    def test_fileinfo(self):
        dataset = self._get_dataset()
        meta_def_key =  dataset._meta['lib']['default text']
        self.assertTrue(dataset.path is not None)
        self.assertTrue(dataset.name == 'Example Data (A)')
        self.assertTrue(dataset.filtered == 'no_filter')
        self.assertTrue(dataset.text_key == meta_def_key)
        self.assertTrue(dataset.text_key == 'en-GB')

    def test_filter(self):
        dataset = self._get_dataset()
        f = intersection([{'gender': [2]},
                          {'age': frange('35-45')}])
        alias = 'men: 35 to 45 years old'
        dataset.filter(alias, f, inplace=True)
        # alias copied correctly?
        self.assertEqual(dataset.filtered, alias)
        # correctly sliced?
        expected_index_len = 1509
        self.assertEqual(len(dataset._data.index), expected_index_len)
        self.assertEqual(dataset['age'].value_counts().sum(), expected_index_len)
        expected_gender_codes = [2]
        expected_age_codes = [35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45]
        self.assertTrue(dataset['gender'].value_counts().index.tolist() ==
                        expected_gender_codes)
        self.assertTrue(sorted(dataset['age'].value_counts().index.tolist()) ==
                        expected_age_codes)

    def test_reorder_values(self):
        dataset = self._get_dataset()
        dataset.reorder_values('q8', [96, 1, 98, 4, 3, 2, 5])
        df_vals = self.check_freq(dataset, 'q8')
        df_texts = self.check_freq(dataset, 'q8', 'text')
        meta = dataset.meta('q8')
        df_vals_index = df_vals.index.get_level_values(1).tolist()
        df_vals_index.remove('All')
        df_texts_index = df_texts.index.get_level_values(1).tolist()
        df_texts_index.remove('All')
        # correctly indexed?
        self.assertTrue(df_vals_index == meta['codes'].tolist())
        self.assertTrue(df_texts_index == meta['texts'].tolist())
        # correct values?
        expected = [[2367.0],
                    [283.0],
                    [949.0],
                    [49.0],
                    [970.0],
                    [595.0],
                    [216.0],
                    [1235.0]]
        self.assertEqual(df_vals.values.tolist(), expected)

    def test_set_missings_flagging(self):
        dataset = self._get_dataset()
        dataset.set_missings('q8', {'exclude': [1, 2, 98, 96]})
        meta = dataset.meta('q8')[['codes', 'missing']]
        meta.index.name = None
        meta.columns.name = None
        missings = [[1, 'exclude'],
                    [2, 'exclude'],
                    [3, None],
                    [4, None],
                    [5, None],
                    [96, 'exclude'],
                    [98, 'exclude']]
        expected_meta = pd.DataFrame(missings, columns=['codes', 'missing'])
        self.assertTrue(all(meta == expected_meta))

    def test_set_missings_results(self):
        dataset = self._get_dataset()
        dataset.set_missings('q8', {'exclude': [1, 2, 98, 96]})
        df = self.check_freq(dataset, 'q8')
        # check the base
        base_size = df.iloc[0, 0]
        expected_base_size = 1058
        self.assertEqual(base_size, expected_base_size)
        # check the index
        index = df.index.get_level_values(1).tolist()
        index.remove('All')
        expected_index = [3, 4, 5]
        self.assertEqual(index, expected_index)
        # check categories
        cat_vals = df.iloc[1:, 0].values.tolist()
        expected_cat_vals = [595, 970, 1235]
        self.assertEqual(cat_vals, expected_cat_vals)

    def test_remove_values(self):
        dataset = self._get_dataset()
        dataset.remove_values('q5_1', [1, 2, 97, 98])
        # removed from meta data?
        expected_cat_meta = [[3, "Probably wouldn't"],
                             [4, 'Probably would if asked'],
                             [5, 'Very likely']]
        self.assertEqual(dataset.meta('q5_1')[['codes', 'texts']].values.tolist(),
                         expected_cat_meta)
        # removed from case data?
        expected_cat_vals = [1, 2, 3]
        self.assertTrue(sorted(dataset._data['q5_1'].value_counts().index.tolist()),
                        expected_cat_vals)
        # does the engine correctly handle it?
        df = self.check_freq(dataset, 'q5_1', show='text')
        expected_index = [cat[1] for cat in expected_cat_meta]
        df_index = df.index.get_level_values(1).tolist()
        df_index.remove('All')
        self.assertTrue(df_index == expected_index)
        expected_results =  [[5194.0],
                             [2598.0],
                             [124.0],
                             [2472.0]]
        self.assertEqual(df.values.tolist(), expected_results)

