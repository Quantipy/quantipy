
# import pdb;
import unittest
import os.path
import numpy as np
import pandas as pd
import quantipy as qp

freq = qp.core.tools.dp.prep.frequency

# from collections import OrderedDict
# from quantipy.core.view_generators.view_maps import QuantipyViews
# from quantipy.core.view import View
# from quantipy.core.stack import Stack
# from quantipy.core.view_generators.view_mapper import ViewMapper
# from quantipy.core.tools.dp import io
# from quantipy.core.tools.view.query import get_dataframe
# from quantipy.core.helpers.functions import emulate_meta

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
        self.assertTrue(dataset.path is not None)
        self.assertTrue(dataset.name == 'Example Data (A)')
        self.assertTrue(dataset.filtered == 'no_filter')
        self.assertTrue(dataset.text_key == dataset._meta['lib']['default text'])
        self.assertTrue(dataset.text_key == 'en-GB')

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
