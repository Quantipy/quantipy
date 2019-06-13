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
        self.assertTrue(dataset._dimensions_comp is False)

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

    def test_subset_from_varlist(self):
        dataset = self._get_dataset()
        keep = ['gender', 'q1', 'q5', 'q6']
        sub_ds = dataset.subset(variables=keep)
        # only variables from "keep" are left?
        sub_ds_vars = sub_ds.columns() + sub_ds.masks()
        expected_vars = sub_ds.unroll(keep, both='all')
        self.assertEqual(sorted(expected_vars), sorted(sub_ds_vars))
        # data file set only list the "keep" variables?
        set_vars = sub_ds._variables_from_set('data file')
        self.assertEqual(sorted(keep), sorted(set_vars))
        # 'sets' & 'lib' list only reduced array masks ref.?
        lib_ref = sub_ds._meta['lib']['values']
        expected_lib_ref = ['q5', 'q6']
        self.assertEqual(expected_lib_ref, sorted(lib_ref))
        set_keys = sub_ds._meta['sets'].keys()
        expected_set_keys = ['data file', 'q5', 'q6']
        self.assertEqual(expected_set_keys, sorted(set_keys))
        # DataFrame columns match "keep" list?
        df_cols = sub_ds._data.columns[1:]
        expected_df_cols = sub_ds.unroll(keep)
        self.assertEqual(sorted(expected_df_cols), sorted(df_cols))

    def test_order_full_change(self):
        dataset = self._get_dataset()
        variables = dataset._variables_from_set('data file')
        new_order = list(sorted(variables, key=lambda v: v.lower()))
        dataset.order(new_order)
        new_set_order = dataset._variables_to_set_format(new_order)
        data_file_items = dataset._meta['sets']['data file']['items']
        df_columns = dataset._data.columns.tolist()
        self.assertEqual(new_set_order, data_file_items)
        self.assertEqual(dataset.unroll(new_order), df_columns)

    def test_order_repos_change(self):
        dataset = self._get_dataset()
        repos = [{'age': ['q8', 'q5']},
                 {'q6': 'q7'},
                 {'q5': 'weight_a'}]
        dataset.order(reposition=repos)
        data_file_items = dataset._meta['sets']['data file']['items']
        df_columns = dataset._data.columns.tolist()
        expected_items = ['record_number', 'unique_id', 'q8', 'weight_a', 'q5',
                          'age', 'birth_day', 'birth_month', 'birth_year',
                          'gender', 'locality', 'ethnicity', 'religion', 'q1',
                          'q2', 'q2b', 'q3', 'q4', 'q7', 'q6', 'q8a', 'q9',
                          'q9a', 'Wave', 'weight_b', 'start_time', 'end_time',
                          'duration', 'q14_1', 'q14_2', 'q14_3', 'RecordNo']
        expected_columns = dataset.unroll(expected_items)
        self.assertEqual(dataset._variables_to_set_format(expected_items),
                         data_file_items)
        self.assertEqual(expected_columns, df_columns)

    def test_categorical_metadata_additions(self):
        dataset = self._get_dataset()
        name, qtype, label = 'test', 'single', 'TEST VAR'
        cats1 = [(4, 'Cat1'), (5, 'Cat2')]
        cats2 = ['Cat1', 'Cat2']
        cats3 = [1, 2]
        for check, cat in enumerate([cats1, cats2, cats3], start=1):
            dataset.add_meta(name, qtype, label, cat)
            values = dataset.values(name)
            if check == 1:
                self.assertTrue(values, cats1)
            elif check == 2:
                expected_vals = [(1, 'Cat1'), (2, 'Cat2')]
                self.assertTrue(values, expected_vals)
            elif check == 3:
                expected_vals = [(1, ''), (2, '')]
                self.assertTrue(values, expected_vals)

    def test_array_metadata(self):
        dataset = self._get_dataset()
        meta, data = dataset.split()
        name, qtype, label = 'array_test', 'delimited set', 'TEST LABEL TEXT'
        cats = ['Cat 1', 'Cat 2', 'Cat 3', 'Cat 4', 'Cat 5']
        items1 = [(1, 'ITEM A'), (3, 'ITEM B'), (6, 'ITEM C')]
        items2 = ['ITEM A', 'ITEM B', 'ITEM C']
        items3 = [4, 5, 6]
        for check, items in enumerate([items1, items2, items3], start=1):
            dataset.add_meta(name, qtype, label, cats, items)
            sources = dataset.sources(name)
            # catgeories correct?
            expected_vals = list(enumerate(cats, start=1))
            self.assertEqual(dataset.values(name), expected_vals)
            # items correct?
            items = dataset.items(name)
            if check == 1:
                expected_items = [('array_test_1', 'ITEM A'),
                                  ('array_test_3', 'ITEM B'),
                                  ('array_test_6', 'ITEM C')]
                self.assertEqual(items, expected_items)
            elif check == 2:
                expected_items = [('array_test_1', 'ITEM A'),
                                  ('array_test_2', 'ITEM B'),
                                  ('array_test_3', 'ITEM C')]
                self.assertEqual(items, expected_items)
            elif check == 3:
                expected_items = [('array_test_4', ''),
                                  ('array_test_5', ''),
                                  ('array_test_6', '')]
                self.assertEqual(items, expected_items)
            # value object location correct?
            item_val_ref = dataset._get_value_loc(sources[0])
            mask_val_ref = dataset._get_value_loc(name)
            self.assertEqual(item_val_ref, mask_val_ref)
            lib_ref = 'lib@values@array_test'
            self.assertTrue(meta['columns'][sources[0]]['values'] == lib_ref)
            self.assertTrue(meta['masks'][name]['values'] == lib_ref)
            # sets entry correct?
            self.assertTrue('masks@array_test' in meta['sets']['data file']['items'])
            # parent entry correct?
            for source in dataset.sources(name):
                parent_meta = meta['columns'][source]['parent']
                expected_parent_meta = {'masks@array_test': {'type': 'array'}}
                parent_maskref = dataset.parents(source)
                expected_parent_maskref = ['masks@array_test']
                self.assertEqual(parent_meta, expected_parent_meta)
                self.assertEqual(parent_maskref, expected_parent_maskref)

    def test_rename_via_masks(self):
        dataset = self._get_dataset()
        meta, data = dataset.split()
        new_name = 'q5_new'
        dataset.rename('q5', new_name)
        # name properly changend?
        self.assertTrue('q5' not in dataset.masks())
        self.assertTrue(new_name in dataset.masks())
        # item names updated?
        items = meta['sets'][new_name]['items']
        expected_items = ['columns@q5_new_1',
                          'columns@q5_new_2',
                          'columns@q5_new_3',
                          'columns@q5_new_4',
                          'columns@q5_new_5',
                          'columns@q5_new_6']
        self.assertEqual(items, expected_items)
        sources = dataset.sources(new_name)
        expected_sources = [i.split('@')[-1] for i in expected_items]
        self.assertEqual(sources, expected_sources)
        # lib reference properly updated?
        lib_ref_mask = meta['masks'][new_name]['values']
        lib_ref_items = meta['columns'][dataset.sources(new_name)[0]]['values']
        expected_lib_ref = 'lib@values@q5_new'
        self.assertEqual(lib_ref_mask, lib_ref_items)
        self.assertEqual(lib_ref_items, expected_lib_ref)
        # new parent entry correct?
        parent_spec = meta['columns'][dataset.sources(new_name)[0]]['parent']
        expected_parent_spec = {'masks@{}'.format(new_name): {'type': 'array'}}
        self.assertEqual(parent_spec, expected_parent_spec)
        # sets entries replaced?
        self.assertTrue('masks@q5' not in meta['sets']['data file']['items'])
        self.assertTrue('masks@q5_new' in meta['sets']['data file']['items'])
        self.assertTrue('q5' not in meta['sets'])
        self.assertTrue('q5_new' in meta['sets'])

    def test_copy_via_masks_full(self):
        dataset = self._get_dataset()
        meta, data = dataset.split()
        suffix = 'test'
        new_name = 'q5_test'
        dataset.copy('q5', suffix)
        # name properly changend?
        self.assertTrue('q5' in dataset.masks())
        self.assertTrue(new_name in dataset.masks())
        # item names updated?
        items = meta['sets'][new_name]['items']
        expected_items = ['columns@q5_test_1',
                          'columns@q5_test_2',
                          'columns@q5_test_3',
                          'columns@q5_test_4',
                          'columns@q5_test_5',
                          'columns@q5_test_6']
        self.assertEqual(items, expected_items)
        sources = dataset.sources(new_name)
        old_items_split = [s.split('_') for s in dataset.sources('q5')]
        expected_sources = ['{}_{}_{}'.format('_'.join(ois[:-1]), suffix, ois[-1])
                            for ois in old_items_split]
        self.assertEqual(sources, expected_sources)
        # lib reference properly updated?
        lib_ref_mask = meta['masks'][new_name]['values']
        lib_ref_items = meta['columns'][dataset.sources(new_name)[0]]['values']
        expected_lib_ref = 'lib@values@q5_test'
        self.assertEqual(lib_ref_mask, lib_ref_items)
        self.assertEqual(lib_ref_items, expected_lib_ref)
        # new parent entry correct?
        parent_spec = meta['columns'][dataset.sources(new_name)[0]]['parent']
        expected_parent_spec = {'masks@{}'.format(new_name): {'type': 'array'}}
        self.assertEqual(parent_spec, expected_parent_spec)
        # sets entries replaced?
        self.assertTrue('masks@q5' in meta['sets']['data file']['items'])
        self.assertTrue('masks@q5_test' in meta['sets']['data file']['items'])
        self.assertTrue('q5' in meta['sets'])
        self.assertTrue('q5_test' in meta['sets'])

    def test_copy_via_masks_sliced_and_reduced(self):
        dataset = self._get_dataset()
        meta, data = dataset.split()
        suffix = 'test'
        new_name = 'q5_test'
        slicer = {'gender': [1]}
        copy_only = [1, 2, 3]
        dataset.copy('q5', suffix, slicer=slicer, copy_only=copy_only)
        # name properly changend?
        self.assertTrue('q5' in dataset.masks())
        self.assertTrue(new_name in dataset.masks())
        # item names updated?
        items = meta['sets'][new_name]['items']
        expected_items = ['columns@q5_test_1',
                          'columns@q5_test_2',
                          'columns@q5_test_3',
                          'columns@q5_test_4',
                          'columns@q5_test_5',
                          'columns@q5_test_6']
        self.assertEqual(items, expected_items)
        sources = dataset.sources(new_name)
        old_items_split = [s.split('_') for s in dataset.sources('q5')]
        expected_sources = ['{}_{}_{}'.format('_'.join(ois[:-1]), suffix, ois[-1])
                            for ois in old_items_split]
        self.assertEqual(sources, expected_sources)
        # lib reference properly updated?
        lib_ref_mask = meta['masks'][new_name]['values']
        lib_ref_items = meta['columns'][dataset.sources(new_name)[0]]['values']
        expected_lib_ref = 'lib@values@q5_test'
        self.assertEqual(lib_ref_mask, lib_ref_items)
        self.assertEqual(lib_ref_items, expected_lib_ref)
        # new parent entry correct?
        parent_spec = meta['columns'][dataset.sources(new_name)[0]]['parent']
        expected_parent_spec = {'masks@{}'.format(new_name): {'type': 'array'}}
        self.assertEqual(parent_spec, expected_parent_spec)
        # sets entries replaced?
        self.assertTrue('masks@q5' in meta['sets']['data file']['items'])
        self.assertTrue('masks@q5_test' in meta['sets']['data file']['items'])
        self.assertTrue('q5' in meta['sets'])
        self.assertTrue('q5_test' in meta['sets'])
        # metadata reduced (only codes 1, 2, 3)?
        self.assertTrue(dataset.codes(new_name) == copy_only)
        # data sliced and reduced properly?
        for s in dataset.sources('q5_test'):
            self.assertTrue(set(dataset[s].dropna().unique()) == set(copy_only))
            self.assertTrue(dataset[[s, 'gender']].dropna()['gender'].unique() == 1)

    def test_transpose(self):
        dataset = self._get_dataset(cases=500)
        meta, data = dataset.split()
        dataset.transpose('q5')
        # new items are old values?
        new_items = dataset.items('q5_trans')
        old_values = dataset.values('q5')
        check_old_values = [('q5_trans_{}'.format(element), text)
                            for element, text in old_values]
        self.assertEqual(check_old_values, new_items)
        # new values are former items?
        new_values = dataset.value_texts('q5_trans')
        old_items = dataset.item_texts('q5')
        self.assertEqual(new_values, old_items)
        # parent meta correctly updated?
        trans_parent = meta['columns'][dataset.sources('q5_trans')[0]]['parent']
        expected_parent = {'masks@q5_trans': {'type': 'array'}}
        self.assertEqual(trans_parent, expected_parent)
        # recoded data is correct?
        original_ct =  dataset.crosstab('q5', text=False)
        transposed_ct = dataset.crosstab('q5_trans', text=False)
        self.assertTrue(np.array_equal(original_ct.drop('All', axis=1, level=1).T.values,
                        transposed_ct.drop('All', axis=1, level=1).values))

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
                                     index=range(1, len(missings)+1),
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

    def test_extend_values_no_texts(self):
        dataset = self._get_dataset()
        dataset.set_verbose_infomsg(False)
        meta_before = dataset.meta('q8')[['codes', 'texts']]
        add_values = [3001, 30002, 3003]
        dataset.extend_values('q8', add_values)
        meta_after = dataset.meta('q8')[['codes', 'texts']]
        # codes are correct?
        self.assertEqual(meta_after['codes'].tail(3).values.tolist(),
                         add_values)
        # texts are empty?
        expected_values_at_end = ['', '', '']
        self.assertEqual(meta_after['texts'].tail(3).values.tolist(),
                         expected_values_at_end)

    def test_extend_values_raises_on_dupes(self):
        dataset = self._get_dataset()
        add_values = [(1, 'CAT A'), (2, 'CAT B')]
        self.assertRaises(ValueError, dataset.extend_values, 'q8', add_values)

    def test_text_replacements_non_array(self):
        dataset = self._get_dataset()
        replace = {'following': 'TEST IN LABEL',
                   'Breakfast': 'TEST IN VALUES'}
        dataset.replace_texts(replace=replace)
        expected_value = 'TEST IN VALUES'
        expected_label = 'Which of the TEST IN LABEL do you regularly skip?'
        value_text = dataset._get_valuemap('q8', non_mapped='texts')[0]
        column_text = dataset.text('q8')
        self.assertEqual(column_text, expected_label)
        self.assertEqual(value_text, expected_value)

    def test_sorting_rules_meta(self):
        dataset = self._get_dataset()
        dataset.sorting('q8', fix=[3, 98, 100])
        expected_rules = {'x': {'sortx': {'fixed': [3, 98],
                                          'within': False,
                                          'between': False,
                                          'ascending': False,
                                          'sort_on': '@',
                                          'with_weight': 'auto'}},
                          'y': {}}
        # rule correctly set?: i.e. code 100 removed from fix list since it
        # does not appear in the values meta?
        self.assertEqual(dataset._meta['columns']['q8']['rules'],
                         expected_rules)

    def test_force_texts(self):
        dataset = self._get_dataset()
        dataset.set_value_texts(name='q4',
                                renamed_vals={1: 'kyllae'},
                                text_key='fi-FI')
        dataset.force_texts(copy_to='de-DE',
                            copy_from=['fi-FI','en-GB'],
                            update_existing=False)
        q4_de_val0 = dataset._meta['columns']['q4']['values'][0]['text']['de-DE']
        q4_de_val1 = dataset._meta['columns']['q4']['values'][1]['text']['de-DE']
        self.assertEqual(q4_de_val0, 'kyllae')
        self.assertEqual(q4_de_val1, 'No')

        q5_de_val0 = dataset._meta['lib']['values']['q5'][0]['text']['de-DE']
        self.assertEqual(q5_de_val0, 'I would refuse if asked')

    def test_validate(self):
        dataset = self._get_dataset()
        meta = dataset._meta
        meta['columns']['q1']['values'][0]['text']['x edits'] = 'test'
        meta['columns']['q1']['name'] = 'Q1'
        meta['columns'].pop('q2')
        meta['masks']['q5']['text'] = {'en-GB': ''}
        meta['masks']['q6']['text'].pop('en-GB')
        meta['columns'].pop('q6_3')
        meta['columns']['q8']['text'] = ''
        meta['columns']['q8']['values'][3]['text'] = ''
        meta['columns']['q8']['values'] = meta['columns']['q8']['values'][0:5]
        index = ['q1', 'q2', 'q5', 'q6', 'q6_1', 'q6_2', 'q6_3', 'q8']
        data = {'name':     ['x', '',  '',  '',  '',  '',  '',  '' ],
                'q_label':  ['',  '',  'x', '',  '',  '',  '',  'x'],
                'values':   ['x', '',  '',  '',  '',  '',  '',  'x'],
                'text keys': ['',  '',  '',  'x', 'x', 'x', '',  'x'],
                'source':   ['',  '',  '',  'x', '',  '',  '',  '' ],
                'codes':    ['',  'x', '',  '',  '',  '',  'x', 'x']}
        df = pd.DataFrame(data, index=index)
        df = df[['name', 'q_label', 'values', 'text keys', 'source', 'codes']]
        df_validate = dataset.validate(False, verbose=False)
        self.assertTrue(df.equals(df_validate))

    def test_compare(self):
        dataset = self._get_dataset()
        ds = dataset.clone()
        dataset.set_value_texts('q1', {2: 'test'})
        dataset.set_variable_text('q8', 'test', ['en-GB', 'sv-SE'])
        dataset.remove_values('q6', [1, 2])
        dataset.convert('q6_3', 'delimited set')
        index = ['q1', 'q6', 'q6_1', 'q6_2', 'q6_3', 'q8']
        data = {'type':         ['', '', '', '', 'x', ''],
                'q_label':      ['', '', '', '', '', 'en-GB, sv-SE, '],
                'codes':        ['', 'x', 'x', 'x', 'x', ''],
                'value texts': ['2: en-GB, ', '', '', '', '', '']}
        df = pd.DataFrame(data, index=index)
        df = df[['type', 'q_label', 'codes', 'value texts']]
        df_comp = dataset.compare(ds)
        self.assertTrue(df.equals(df_comp))

    def test_convert_str_to_delimited_set(self):
        dataset = self._get_dataset()
        ds = dataset.clone()
        ds.add_meta('del_set','string','Delimited set as a string')
        ds._data['del_set'] = ds._data['q9']
        ds.convert('del_set',to="delimited set")
        self.assertEqual(ds.crosstab('q9').iloc[:,0].tolist(),
                         ds.crosstab('del_set').iloc[:,0].tolist())

    def test_uncode(self):
        dataset = self._get_dataset()
        dataset.uncode('q8',{1: 1, 2:2, 5:5}, 'q8', intersect={'gender':1})
        dataset.uncode('q8',{3: 3, 4:4, 98:98}, 'q8', intersect={'gender':2})
        df = dataset.crosstab('q8', 'gender')
        result = [[ 1797.,   810.,   987.],
                  [  476.,     0.,   476.],
                  [  104.,     0.,   104.],
                  [  293.,   293.,     0.],
                  [  507.,   507.,     0.],
                  [  599.,     0.,   599.],
                  [  283.,   165.,   118.],
                  [   26.,    26.,     0.]]
        self.assertEqual(df.values.tolist(), result)

    def test_derotate_df(self):
        dataset = self._get_dataset()
        levels = {'visit': ['visit_1', 'visit_2', 'visit_3']}
        mapper = [{'q14r{:02}'.format(r): ['q14r{0:02}c{1:02}'.format(r, c)
                  for c in range(1, 4)]} for r in frange('1-5')]
        ds = dataset.derotate(levels, mapper, 'gender', 'record_number')
        df_h = ds._data.head(10)
        df_val = [[x if not np.isnan(x) else 'nan' for x in line]
                  for line in df_h.values.tolist()]
        result_df = [[1.0, 2.0, 1.0, 4.0, 4.0, 4.0, 8.0, 1.0, 2.0, 4.0, 2.0, 3.0, 1.0],
                     [1.0, 2.0, 2.0, 4.0, 4.0, 4.0, 8.0, 3.0, 3.0, 2.0, 4.0, 3.0, 1.0],
                     [1.0, 3.0, 1.0, 1.0, 1.0, 8.0, 'nan', 4.0, 3.0, 1.0, 3.0, 1.0, 2.0],
                     [1.0, 4.0, 1.0, 5.0, 5.0, 4.0, 8.0, 2.0, 3.0, 2.0, 3.0, 1.0, 1.0],
                     [1.0, 4.0, 2.0, 4.0, 5.0, 4.0, 8.0, 2.0, 1.0, 3.0, 2.0, 1.0, 1.0],
                     [1.0, 5.0, 1.0, 3.0, 3.0, 5.0, 8.0, 4.0, 2.0, 2.0, 1.0, 3.0, 1.0],
                     [1.0, 5.0, 2.0, 5.0, 3.0, 5.0, 8.0, 3.0, 3.0, 3.0, 1.0, 2.0, 1.0],
                     [1.0, 6.0, 1.0, 2.0, 2.0, 8.0, 'nan', 4.0, 2.0, 3.0, 4.0, 2.0, 1.0],
                     [1.0, 7.0, 1.0, 3.0, 3.0, 3.0, 8.0, 2.0, 1.0, 3.0, 2.0, 4.0, 1.0],
                     [1.0, 7.0, 2.0, 3.0, 3.0, 3.0, 8.0, 3.0, 2.0, 1.0, 2.0, 3.0, 1.0]]
        result_columns = ['@1', 'record_number', 'visit', 'visit_levelled',
                          'visit_1', 'visit_2', 'visit_3', 'q14r01', 'q14r02',
                          'q14r03', 'q14r04', 'q14r05', 'gender']
        df_len = 18520
        self.assertEqual(df_val, result_df)
        self.assertEqual(df_h.columns.tolist(), result_columns)
        self.assertEqual(len(ds._data.index), df_len)
        path_json = '{}/{}.json'.format(ds.path, ds.name)
        path_csv = '{}/{}.csv'.format(ds.path, ds.name)
        os.remove(path_json)
        os.remove(path_csv)

    def test_derotate_freq(self):
        dataset = self._get_dataset()
        levels = {'visit': ['visit_1', 'visit_2', 'visit_3']}
        mapper = [{'q14r{:02}'.format(r): ['q14r{0:02}c{1:02}'.format(r, c)
                  for c in range(1, 4)]} for r in frange('1-5')]
        ds = dataset.derotate(levels, mapper, 'gender', 'record_number')
        val_c = {'visit': {'val': {1: 8255, 2: 6174, 3: 4091},
                   'index': [1, 2, 3]},
                 'visit_levelled': {'val': {4: 3164, 1: 3105, 5: 3094, 6: 3093, 3: 3082, 2: 2982},
                                   'index': [4, 1, 5, 6, 3,2]},
                 'visit_1': {'val': {4: 3225, 6: 3136, 3: 3081, 2: 3069, 1: 3029, 5: 2980},
                             'index': [4, 6, 3, 2, 1, 5]},
                 'visit_2': {'val': {1: 2789, 6: 2775, 5: 2765, 3: 2736, 4: 2709, 2: 2665, 8: 2081},
                             'index': [1, 6, 5, 3, 4, 2, 8]},
                 'visit_3': {'val': {8: 4166, 5: 2181, 4: 2112, 3: 2067, 1: 2040, 6: 2001, 2: 1872},
                             'index': [8, 5, 4, 3, 1, 6, 2]},
                 'q14r01': {'val': {3: 4683, 1: 4653, 4: 4638, 2: 4546},
                            'index': [3, 1, 4, 2]},
                 'q14r02': {'val': {4: 4749, 2: 4622, 1: 4598, 3: 4551},
                            'index': [4, 2, 1, 3]},
                 'q14r03': {'val': {1: 4778, 4: 4643, 3: 4571, 2: 4528},
                            'index': [1, 4, 3, 2]},
                 'q14r04': {'val': {1: 4665, 2: 4658, 4: 4635, 3: 4562},
                            'index': [1, 2, 4, 3]},
                 'q14r05': {'val': {2: 4670, 4: 4642, 1: 4607, 3: 4601},
                           'index': [2, 4, 1, 3]},
                 'gender': {'val': {2: 9637, 1: 8883},
                            'index': [2, 1]}}
        for var in val_c.keys():
            series = pd.Series(val_c[var]['val'], index = val_c[var]['index'])
            compare = all(series == ds._data[var].value_counts())
            self.assertTrue(compare)
        path_json = '{}/{}.json'.format(ds.path, ds.name)
        path_csv = '{}/{}.csv'.format(ds.path, ds.name)
        os.remove(path_json)
        os.remove(path_csv)

    def test_derotate_meta(self):
        dataset = self._get_dataset()
        levels = {'visit': ['visit_1', 'visit_2', 'visit_3']}
        mapper = [{'q14r{:02}'.format(r): ['q14r{0:02}c{1:02}'.format(r, c)
                  for c in range(1, 4)]} for r in frange('1-5')]
        ds = dataset.derotate(levels, mapper, 'gender', 'record_number')
        err = ds.validate(False, False)
        err_s = None
        self.assertEqual(err_s, err)
        path_json = '{}/{}.json'.format(ds.path, ds.name)
        path_csv = '{}/{}.csv'.format(ds.path, ds.name)
        os.remove(path_json)
        os.remove(path_csv)

    def test_interlock(self):
        dataset = self._get_dataset()
        data = dataset._data
        name, lab = 'q4AgeGen', 'q4 Age Gender'
        variables = ['q4',
                     {'age': [(1, '18-35', {'age': frange('18-35')}),
                              (2, '30-49', {'age': frange('30-49')}),
                              (3, '50+', {'age': is_ge(50)})]},
                     'gender']
        dataset.interlock(name, lab, variables)
        val = [1367,1109,1036,831,736,579,571,550,454,438,340,244]
        ind = ['10;','8;','9;','7;','3;','8;10;','1;','4;','2;','7;9;','1;3;','2;4;']
        s = pd.Series(val, index=ind, name='q4AgeGen')
        self.assertTrue(all(s==data['q4AgeGen'].value_counts()))
        values = [(1, u'Yes/18-35/Male'),
                  (2, u'Yes/18-35/Female'),
                  (3, u'Yes/30-49/Male'),
                  (4, u'Yes/30-49/Female'),
                  (5, u'Yes/50+/Male'),
                  (6, u'Yes/50+/Female'),
                  (7, u'No/18-35/Male'),
                  (8, u'No/18-35/Female'),
                  (9, u'No/30-49/Male'),
                  (10, u'No/30-49/Female'),
                  (11, u'No/50+/Male'),
                  (12, u'No/50+/Female')]
        text = 'q4 Age Gender'
        self.assertEqual(values, dataset.values('q4AgeGen'))
        self.assertEqual(text, dataset.text('q4AgeGen'))
        self.assertTrue(dataset.is_delimited_set('q4AgeGen'))

    def test_dichotomous_to_delimited_set(self):
        dataset = self._get_dataset()
        dataset.dichotomize('q8', None, False)
        dataset.to_delimited_set('q8_new', dataset.text('q8'),
                                 ['q8_1', 'q8_2', 'q8_3', 'q8_4', 'q8_5', 'q8_96', 'q8_98'],
                                 from_dichotomous=True, codes_from_name=True)
        self.assertEqual(dataset.values('q8'), dataset.values('q8_new'))
        self.assertEqual(dataset['q8'].value_counts().values.tolist(),
                         dataset['q8_new'].value_counts().values.tolist())
        self.assertRaises(ValueError, dataset.to_delimited_set, 'q8_new', '', ['age', 'gender'])

    def test_categorical_to_delimited_set(self):
        dataset = self._get_dataset()
        self.assertRaises(ValueError, dataset.to_delimited_set, 'q1_1', '', ['q1', 'q2'])
        dataset.to_delimited_set('q5_new',
                         dataset.text('q5'),
                         dataset.sources('q5'),
                         False)
        self.assertEqual(dataset.crosstab('q5_new').values.tolist(),
                         [[8255.0], [3185.0], [2546.0], [4907.0],
                          [287.0], [3907.0], [1005.0], [3640.0]])
        for v in dataset.sources('q5'):
            self.assertEqual(dataset.values('q5_new'), dataset.values(v))

    def test_get_value_texts(self):
        dataset = self._get_dataset()
        values = [(1, u'Regularly'), (2, u'Irregularly'), (3, u'Never')]
        self.assertEqual(values, dataset.values('q2b', 'en-GB'))
        dataset._meta['columns']['q2b']['values'][0]['text']['x edits'] = {'en-GB': 'test'}
        value_texts = ['test', None, None]
        self.assertEqual(value_texts, dataset.value_texts('q2b', 'en-GB', 'x'))

    def test_get_item_texts(self):
        dataset = self._get_dataset()
        items = [(u'q6_1', u'Exercise alone'),
                 (u'q6_2', u'Join an exercise class'),
                 (u'q6_3', u'Play any kind of team sport')]
        self.assertEqual(items, dataset.items('q6', 'en-GB'))
        dataset._meta['masks']['q6']['items'][2]['text']['x edits'] = {'en-GB': 'test'}
        item_texts = [None, None, 'test']
        self.assertEqual(item_texts, dataset.item_texts('q6', 'en-GB', 'x'))

    def test_get_variable_text(self):
        dataset = self._get_dataset()
        text = 'How often do you take part in any of the following? - Exercise alone'
        self.assertEqual(text, dataset.text('q6_1', False, 'en-GB'))
        text = 'Exercise alone'
        self.assertEqual(text, dataset.text('q6_1', True, 'en-GB'))
        text = ''
        self.assertEqual(text, dataset.text('q6_1', True, 'en-GB', 'x'))

    def test_set_value_texts(self):
        dataset = self._get_dataset()
        values = [{u'text': {u'en-GB': u'Strongly disagree'}, u'value': 1},
                  {u'text': {u'en-GB': 'test1'}, u'value': 2},
                  {u'text': {u'en-GB': u'Neither agree nor disagree'}, u'value': 3},
                  {u'text': {u'en-GB': u'Agree', 'y edits': {'en-GB': 'test2'}}, u'value': 4},
                  {u'text': {u'en-GB': u'Strongly agree'}, u'value': 5}]
        dataset.set_value_texts('q14_1', {2: 'test1'}, 'en-GB')
        dataset.set_value_texts('q14_1', {4: 'test2'}, 'en-GB', 'y')
        value_obj = dataset._meta['lib']['values']['q14_1']
        self.assertEqual(value_obj, values)
        values = [{u'text': {u'en-GB': u'test1'}, u'value': 1},
                  {u'text': {u'en-GB': u'Irregularly'}, u'value': 2},
                  {u'text': {u'en-GB': u'Never',
                             u'y edits': {'en-GB': 'test2'},
                             u'x edits': {'en-GB': 'test2'}}, u'value': 3}]
        dataset.set_value_texts('q2b', {1: 'test1'}, 'en-GB')
        dataset.set_value_texts('q2b', {3: 'test2'}, 'en-GB', ['x', 'y'])
        value_obj = dataset._meta['columns']['q2b']['values']
        self.assertEqual(value_obj, values)

    def test_set_item_texts(self):
        dataset = self._get_dataset()
        items = [{u'en-GB': u'Exercise alone'},
                 {u'en-GB': u'Join an exercise class',
                  'sv-SE': 'test1',
                  'x edits': {'sv-SE': 'test', 'en-GB': 'test'}},
                 {u'en-GB': u'Play any kind of team sport',
                  'sv-SE': 'test2'}]
        dataset.set_item_texts('q6', {2: 'test1', 3: 'test2'}, 'sv-SE')
        dataset.set_item_texts('q6', {2: 'test'}, ['en-GB', 'sv-SE'], 'x')
        item_obj = [i['text'] for i in dataset._meta['masks']['q6']['items']]
        self.assertEqual(item_obj, items)

    def test_set_variable_text(self):
        dataset = self._get_dataset()
        text = {'en-GB': 'new text', 'sv-SE': 'new text'}
        dataset.set_variable_text('q6', 'new text', ['en-GB', 'sv-SE'])
        dataset.set_variable_text('q6', 'new', ['da-DK'], 'x')
        text_obj = dataset._meta['masks']['q6']['text']
        self.assertEqual(text_obj, text)
        text = {'en-GB': 'What is your main fitness activity?',
                'x edits': {'en-GB': 'edit'}, 'y edits':{'en-GB': 'edit'}}
        dataset.set_variable_text('q1', 'edit', 'en-GB', ['x', 'y'])

    def test_crosstab(self):
        x = 'q14r01c01'
        dataset = self._get_dataset()
        dataset.crosstab(x)
        self.assertEqual(dataset._meta['columns'][x]['values'],
                         'lib@values@q14_1')
