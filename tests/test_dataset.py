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

class TestDataSet(unittest.TestCase):

    def check_freq(self, dataset, var, show='values'):
        return freq(dataset._meta, dataset._data, var, show=show)

    def check_cross(self, dataset, x, y, show='values', rules=False):
        return cross(dataset._meta, dataset._data, x=x, y=y,
                     show=show, rules=rules)

    def _get_dataset(self):
        path = os.path.dirname(os.path.abspath(__file__)) + '/'
        name = 'Example Data (A)'
        casedata = '{}.csv'.format(name)
        metadata = '{}.json'.format(name)
        dataset = qp.DataSet(name)
        dataset.set_verbose_infomsg(False)
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
        self.assertTrue(dataset._verbose_errors is True)
        self.assertTrue(dataset._verbose_infos is False)

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

    def test_reorder_values_raises_on_incomplete_list(self):
        dataset = self._get_dataset()
        dataset.set_verbose_errmsg(False)
        new_order = [3, 2, 1]
        self.assertRaises(ValueError, dataset.reorder_values, 'q8', new_order)


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
        expected_meta = pd.DataFrame(missings,
                                     index=xrange(1, len(missings)+1),
                                     columns=['codes', 'missing'])
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
        expected_cat_vals = [cat[0] for cat in expected_cat_meta]
        self.assertEqual(sorted(dataset._data['q5_1'].value_counts().index.tolist()),
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

    def test_extend_values_autocodes(self):
        dataset = self._get_dataset()
        meta_before = dataset.meta('q8')[['codes', 'texts']]
        add_values = ['CAT A', 'CAT B']
        dataset.extend_values('q8', add_values)
        meta_after = dataset.meta('q8')[['codes', 'texts']]
        # codes are correctly selected?
        expected_codes_diff = [99, 100]
        codes_diff = sorted(list(set(meta_after['codes'].values)-
                                 set(meta_before['codes'].values)))
        self.assertEqual(codes_diff, expected_codes_diff)
        # texts match?
        expected_values_at_end = ['CAT A', 'CAT B']
        self.assertEqual(meta_after['texts'].tail(2).values.tolist(),
                         expected_values_at_end)

    def test_extend_values_usercodes(self):
        dataset = self._get_dataset()
        meta_before = dataset.meta('q8')[['codes', 'texts']]
        add_values = [(210, 'CAT A'), (102, 'CAT B')]
        dataset.extend_values('q8', add_values)
        meta_after = dataset.meta('q8')[['codes', 'texts']]
        # codes are correct?
        expected_codes_at_end = [210, 102]
        self.assertEqual(meta_after['codes'].tail(2).values.tolist(),
                         expected_codes_at_end)
        # texts match?
        expected_values_at_end = ['CAT A', 'CAT B']
        self.assertEqual(meta_after['texts'].tail(2).values.tolist(),
                         expected_values_at_end)

    def test_extend_values_raises_on_dupes(self):
        dataset = self._get_dataset()
        add_values = [(1, 'CAT A'), (2, 'CAT B')]
        self.assertRaises(ValueError, dataset.extend_values, 'q8', add_values)

    def test_clean_texts_replacements_non_array(self):
        dataset = self._get_dataset()
        replace = {'following': 'TEST IN LABEL',
                   'Breakfast': 'TEST IN VALUES'}
        dataset.clean_texts(replace=replace)
        expected_value = 'TEST IN VALUES'
        expected_label = 'Which of the TEST IN LABEL do you regularly skip?'
        value_text = dataset._get_valuemap('q8', non_mapped='texts')[0]
        column_text = dataset._get_label('q8')
        self.assertEqual(column_text, expected_label)
        self.assertEqual(value_text, expected_value)

    def test_clean_texts_replacements_array(self):
        pass

    def test_sorting_rules_meta(self):
        dataset = self._get_dataset()
        dataset.set_sorting('q8', fix=[3, 98, 100])
        expected_rules = {'x': {'sortx': {'fixed': [3, 98],
                                          'ascending': False}},
                          'y': {}}
        # rule correctly set?: i.e. code 100 removed from fix list since it
        # does not appear in the values meta?
        self.assertEqual(dataset._meta['columns']['q8']['rules'],
                         expected_rules)

    def test_sorting_result(self):
        dataset = self._get_dataset()
        pass

    def test_force_texts(self):
        dataset = self._get_dataset()
        dataset.set_value_texts(name='q4',
                                renamed_vals={1: 'kyllae'},
                                text_key='fi-FI')
        dataset.force_texts(name=None, copy_to='de-DE',
                            copy_from=['fi-FI','en-GB'],
                            update_existing=False)
        q4_de_val0 = dataset._meta['columns']['q4']['values'][0]['text']['de-DE']
        q4_de_val1 = dataset._meta['columns']['q4']['values'][1]['text']['de-DE']
        self.assertEqual(q4_de_val0, 'kyllae')
        self.assertEqual(q4_de_val1, 'No')

        q5_de_val0 = dataset._meta['lib']['values']['q5'][0]['text']['de-DE']
        self.assertEqual(q5_de_val0, 'I would refuse if asked')

        self.assertRaises(ValueError, dataset.force_texts,
                          name='q4', copy_from=['sv-SE'])
