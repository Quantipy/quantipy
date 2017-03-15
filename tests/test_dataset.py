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
        dataset = qp.DataSet(name)
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
        items = ['ITEM A', 'ITEM B', 'ITEM C']
        dataset.add_meta(name, qtype, label, cats, items)
        sources = dataset.sources(name)
        # catgeories correct?
        expected_vals = list(enumerate(cats, start=1))
        self.assertEqual(dataset.values(name), expected_vals)
        # items correct?
        item_names = ['{}_{}'.format(name, i) for i in range(1, 4)]
        expected_items = zip(item_names, items)
        self.assertEqual(dataset.items(name), expected_items)
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
        dataset.rename('q5', new_name, array_items={1: 'array_element_1',
                                                    2: 'array_element_2'})
        # name properly changend?
        self.assertTrue('q5' not in dataset.masks())
        self.assertTrue(new_name in dataset.masks())
        # item names updated?
        items = meta['sets'][new_name]['items']
        expected_items = ['columns@array_element_1',
                          'columns@array_element_2',
                          'columns@q5_3',
                          'columns@q5_4',
                          'columns@q5_5',
                          'columns@q5_6']
        self.assertEqual(items, expected_items)
        sources = dataset.sources(new_name)
        expected_sources = ['array_element_1', 'array_element_2'] + sources[2:]
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
        dataset.copy('q5', suffix, slicer=None, copy_only=copy_only)
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
        print dataset.codes(new_name)
        self.assertTrue(dataset.codes(new_name) == [1, 2, 3])

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
        self.assertTrue(np.array_equal(original_ct.drop('All', 1, 1).T.values,
                        transposed_ct.drop('All', 1, 1).values))

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

    def test_clean_texts_replacements_non_array(self):
        dataset = self._get_dataset()
        replace = {'following': 'TEST IN LABEL',
                   'Breakfast': 'TEST IN VALUES'}
        dataset.clean_texts(replace=replace)
        expected_value = 'TEST IN VALUES'
        expected_label = 'Which of the TEST IN LABEL do you regularly skip?'
        value_text = dataset._get_valuemap('q8', non_mapped='texts')[0]
        column_text = dataset.text('q8')
        self.assertEqual(column_text, expected_label)
        self.assertEqual(value_text, expected_value)

    def test_clean_texts_replacements_array(self):
        pass

    def test_sorting_rules_meta(self):
        dataset = self._get_dataset()
        dataset.sorting('q8', fix=[3, 98, 100])
        expected_rules = {'x': {'sortx': {'fixed': [3, 98],
                                          'within': False,
                                          'between': False,
                                          'ascending': False,
                                          'sort_on': '@'}},
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

    def test_validate(self):
        dataset = self._get_dataset()
        meta = dataset._meta
        meta['columns']['q1'].pop('values')
        meta['columns'].pop('q2')
        meta['masks']['q5']['items'][1]['source'] = ''
        for mask in ['q5', 'q6', 'q7']:
            meta['masks'][mask]['text'] = {'en-GB': ''}
            for item in  meta['masks'][mask]['items']:
                del item['text']
        meta['masks']['q6']['text'].pop('en-GB')
        meta['lib']['values'].pop('q6')
        meta['columns']['q8']['text'] = ''
        meta['columns']['q8']['values'][3]['text'] = ''
        meta['columns']['q8']['values'] = meta['columns']['q8']['values'][0:5]
        index = ['q1', 'q2', 'q5', 'q6', 'q6_1', 'q6_2', 'q6_3', 'q7', 'q8']
        data = {'Err1': ['', 'x', '', '', '', '', '', '', 'x, value 3'],
                'Err2': ['', 'x', '', 'x', '', '', '', '', ''],
                'Err3': ['', 'x', 'x', '', '', '', '', 'x', ''],
                'Err4': ['x', 'x', '', '', '', '', '', '', ''],
                'Err5': ['', 'x', '', 'x', 'x', 'x', 'x', '', ''],
                'Err6': ['', 'x', 'item  1', '', '', '', '', '', ''],
                'Err7': ['', 'x', '', '', '', '', '', '', 'x']}
        df = pd.DataFrame(data, index=index)
        df_validate = dataset.validate(verbose=False)
        self.assertTrue(df.equals(df_validate))

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
        err = ds.validate(False)
        err_s = (0, 7)
        self.assertEqual(err_s, err.shape)
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
        self.assertTrue(dataset._is_delimited_set('q4AgeGen'))
