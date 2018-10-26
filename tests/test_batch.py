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
from collections import OrderedDict

def _get_dataset():
	path = os.path.dirname(os.path.abspath(__file__)) + '/'
	name = 'Example Data (A)'
	casedata = '{}.csv'.format(name)
	metadata = '{}.json'.format(name)
	dataset = qp.DataSet(name, False)
	dataset.set_verbose_infomsg(False)
	dataset.set_verbose_errmsg(False)
	dataset.read_quantipy(path+metadata, path+casedata)
	return dataset

def _get_batch(name, dataset=None, full=False):
	if not dataset: dataset = _get_dataset()
	batch = qp.Batch(dataset, name)
	if full:
		batch.add_downbreak(['q1', 'q2', 'q6', 'age'])
		batch.add_crossbreak(['gender', 'q2'])
		batch.add_open_ends(['q8a', 'q9a'], 'RecordNo')
		batch.add_filter('men only', {'gender': 1})
		batch.set_weights('weight_a')
	return batch, dataset

def _get_meta(batch):
	name = batch.name
	return batch._meta['sets']['batches'][name]

class TestBatch(unittest.TestCase):

	def test_dataset_add_batch(self):
		dataset = _get_dataset()
		batch1 = dataset.add_batch('batch1')
		batch2 = dataset.add_batch('batch2', 'c', 'weight', .05)
		self.assertTrue(isinstance(batch1, qp.Batch))
		self.assertEqual(len(_get_meta(batch1).keys()), 31)
		b_meta = _get_meta(batch2)
		self.assertEqual(b_meta['name'], 'batch2')
		self.assertEqual(b_meta['cell_items'], ['c'])
		self.assertEqual(b_meta['weights'], ['weight'])
		self.assertEqual(b_meta['sigproperties']['siglevels'], [0.05])

	def test_dataset_get_batch(self):
		batch, ds = _get_batch('test', full=True)
		self.assertRaises(KeyError, ds.get_batch, 'name')
		b = ds.get_batch('test')
		attr = ['xks', 'yks', 'filter', 'filter_names',
				'x_y_map', 'x_filter_map', 'y_on_y',
				'forced_names', 'summaries', 'transposed_arrays', 'verbatims',
				'extended_yks_global', 'extended_yks_per_x',
				'exclusive_yks_per_x', 'extended_filters_per_x', 'meta_edits',
				'cell_items', 'weights', 'sigproperties', 'additional',
				'sample_size', 'language', 'name', 'total', '_filter_slice']
		for a in attr:
			self.assertEqual(batch.__dict__[a], b.__dict__[a])

	def test_from_batch(self):
		ds = _get_dataset()
		ds.force_texts('de-DE', 'en-GB')
		batch1, ds = _get_batch('test1', ds, full=True)
		batch1.set_language('de-DE')
		batch1.hiding('q1', frange('8,9,96-99'))
		batch1.slicing('q1', frange('9-4'))
		batch2, ds = _get_batch('test2', ds)
		batch2.add_downbreak('q1')
		batch2.add_crossbreak('Wave')
		batch2.as_addition('test1')
		n_ds = ds.from_batch('test1', 'RecordNo', 'de-DE', True, 'variables')
		self.assertEqual(n_ds.codes('q1'), [7, 6, 5, 4])
		self.assertEqual(n_ds.variables(), [u'age', u'gender', u'q1', u'q2',
		                  					u'q6', u'q8a', u'q9a', u'Wave',
		                  					u'weight_a', u'RecordNo'])
		self.assertEqual(n_ds['gender'].value_counts().values.tolist(), [3952])
		self.assertEqual(n_ds.value_texts('gender', 'en-GB'), [None, None])
		self.assertEqual(n_ds.value_texts('gender', 'de-DE'), [u'Male', u'Female'])
		self.assertRaises(ValueError, ds.from_batch, 'test1', 'RecordNo', 'fr-FR')

	########################## methods used in _get_batch ####################

	def test_add_downbreak(self):
		batch, ds = _get_batch('test')
		batch.add_downbreak(['q1', 'q2', 'q2b', {'q3': 'q3_label'}, 'q4', {'q5': 'q5_label'}, 'q14_1'])
		b_meta = _get_meta(batch)
		self.assertEqual(b_meta['xks'], ['q1', 'q2', 'q2b', 'q3', 'q4', 'q5',
		                 				 u'q5_1', u'q5_2', u'q5_3', u'q5_4', u'q5_5',
		                 				 u'q5_6', 'q14_1', u'q14r01c01', u'q14r02c01',
		                 				 u'q14r03c01', u'q14r04c01', u'q14r05c01', u'q14r06c01',
		                 				 u'q14r07c01', u'q14r08c01', u'q14r09c01', u'q14r10c01'])
		self.assertEqual(b_meta['forced_names'], {'q3': 'q3_label', 'q5': 'q5_label'})
		self.assertEqual(b_meta['summaries'], ['q5', 'q14_1'])
		x_y_map = [('q1', ['@']), ('q2', ['@']), ('q2b', ['@']),
							   ('q3', ['@']), ('q4', ['@']), ('q5', ['@']),
							   (u'q5_1', ['@']), (u'q5_2', ['@']),
							   (u'q5_3', ['@']), (u'q5_4', ['@']),
							   (u'q5_5', ['@']), (u'q5_6', ['@']),
							   ('q14_1', ['@']), (u'q14r01c01', ['@']),
							   (u'q14r02c01', ['@']), (u'q14r03c01', ['@']),
							   (u'q14r04c01', ['@']), (u'q14r05c01', ['@']),
							   (u'q14r06c01', ['@']), (u'q14r07c01', ['@']),
							   (u'q14r08c01', ['@']), (u'q14r09c01', ['@']),
							   (u'q14r10c01', ['@'])]
		self.assertEqual(b_meta['x_y_map'], x_y_map)

	def test_add_crossbreak(self):
		batch, ds = _get_batch('test')
		batch.add_crossbreak(['gender', 'q2b'])
		b_meta = _get_meta(batch)
		self.assertEqual(b_meta['yks'], ['@', 'gender', 'q2b'])
		self.assertRaises(KeyError, batch.add_crossbreak, ['@', 'GENDER'])
		batch.add_downbreak('q1')
		x_y_map = [('q1', ['@', 'gender', 'q2b'])]
		self.assertEqual(b_meta['x_y_map'], x_y_map)

	def test_add_open_ends(self):
		batch, ds = _get_batch('test')
		self.assertRaises(ValueError, batch.add_open_ends, ['q8a', 'q9a'], None,
						  True, False, True, 'open ends', None)
		batch.add_filter('men only', {'gender': 1})
		batch.add_open_ends(['q8a', 'q9a'], 'RecordNo', filter_by={'age': is_ge(49)})
		verbatims = _get_meta(batch)['verbatims'][0]
		self.assertEqual(len(verbatims['idx']), 118)
		self.assertEqual(verbatims['columns'], ['q8a', 'q9a'])
		self.assertEqual(verbatims['break_by'], ['RecordNo'])
		self.assertEqual(verbatims['title'], 'open ends')
		batch.add_open_ends(['q8a', 'q9a'], 'RecordNo', split=True,
							title=['open ends', 'open ends2'], overwrite=True)
		verbatims = _get_meta(batch)['verbatims']
		self.assertEqual(len(verbatims), 2)

	def test_add_filter(self):
		batch, ds = _get_batch('test', full=True)
		batch.add_downbreak(['q1', 'q2b'])
		batch.add_crossbreak('gender')
		batch.add_filter('men only', {'gender': 1})
		b_meta = _get_meta(batch)
		self.assertEqual(b_meta['filter'], {'men only': {'gender': 1}})
		x_filter_map = OrderedDict([('q1', {'men only': {'gender': 1}}),
									('q2b', {'men only': {'gender': 1}})])
		self.assertEqual(b_meta['x_filter_map'], x_filter_map)
		self.assertEqual(b_meta['filter_names'], ['men only'])

	def test_set_weight(self):
		batch, ds = _get_batch('test')
		self.assertRaises(ValueError, batch.set_weights, 'Weight')
		batch.set_weights('weight_a')
		self.assertEqual(_get_meta(batch)['weights'], ['weight_a'])
		self.assertEqual(batch.weights, ['weight_a'])

	##########################################################################

	def test_copy(self):
		batch1, ds = _get_batch('test', full=True)
		batch2 = batch1.copy('test_copy')
		batch3 = batch1.copy('test_copy2', as_addition=True)
		attributes = ['xks', 'yks', 'filter', 'filter_names', 'x_y_map',
				      'x_filter_map', 'y_on_y', 'forced_names', 'summaries',
					  'transposed_arrays', 'extended_yks_global', 'extended_yks_per_x',
	                  'exclusive_yks_per_x', 'extended_filters_per_x', 'meta_edits',
	                  'cell_items', 'weights', 'sigproperties', 'additional',
	                  'sample_size', 'language']
		for a in attributes:
			value = batch1.__dict__[a]
			value2 = batch2.__dict__[a]
			self.assertEqual(value, value2)
		self.assertEqual(batch3.verbatims, [])
		self.assertEqual(batch3.additional, True)
		self.assertEqual(_get_meta(batch2)['name'], 'test_copy')

	def test_as_addition(self):
		batch1, ds = _get_batch('test1', full=True)
		batch2, ds = _get_batch('test2', ds, True)
		batch2.as_addition('test1')
		self.assertEqual(_get_meta(batch1)['additions'], ['test2'])
		b_meta = _get_meta(batch2)
		self.assertEqual(b_meta['additional'], True)
		self.assertEqual(b_meta['verbatims'], [])
		self.assertEqual(b_meta['y_on_y'], [])

	def test_set_cell_items(self):
		batch, ds = _get_batch('test', full=True)
		self.assertRaises(ValueError, batch.set_cell_items, ['c', 'pc'])
		batch.set_cell_items('c')
		self.assertEqual(_get_meta(batch)['cell_items'], ['c'])
		self.assertEqual(batch.cell_items, ['c'])

	def test_set_language(self):
		batch, ds = _get_batch('test', full=True)
		self.assertRaises(ValueError, batch.set_language, 'en-gb')
		batch.set_language('sv-SE')
		self.assertEqual(_get_meta(batch)['language'], 'sv-SE')
		self.assertEqual(batch.language, 'sv-SE')

	def test_set_sigtest(self):
		batch, ds = _get_batch('test', full=True)
		self.assertRaises(TypeError, batch.set_sigtests, [0.05, '0.01'])
		batch.set_sigtests(.05)
		self.assertEqual(_get_meta(batch)['sigproperties']['siglevels'],  [0.05])

	def test_make_summaries_transpose_arrays(self):
		batch, ds = _get_batch('test')
		b_meta = _get_meta(batch)
		batch.add_downbreak(['q5', 'q6', 'q14_2', 'q14_3', 'q14_1'])
		batch.make_summaries(None)
		self.assertEqual(b_meta['summaries'], [])
		batch.transpose_arrays(['q5', 'q6'], False)
		batch.transpose_arrays(['q14_2', 'q14_3'], True)
		self.assertEqual(b_meta['summaries'], ['q5', 'q6', 'q14_2', 'q14_3'])
		t_a = {'q14_2': True, 'q14_3': True, 'q5': False, 'q6': False}
		self.assertEqual(b_meta['transposed_arrays'], t_a)
		batch.make_summaries('q5')
		self.assertEqual(b_meta['transposed_arrays'], {'q5': False})
		self.assertRaises(ValueError, batch.make_summaries, 'q7')

	def test_extend_y(self):
		batch1, ds = _get_batch('test1', full=True)
		batch2, ds = _get_batch('test2', ds, True)
		b_meta1 = _get_meta(batch1)
		b_meta2 = _get_meta(batch2)
		self.assertRaises(ValueError, batch1.extend_y, 'q2b', 'q5')
		batch1.extend_y('q2b')
		x_y_map = [('q1', ['@', 'gender', 'q2', 'q2b']),
							   ('q2', ['@', 'gender', 'q2', 'q2b']),
							   ('q6', ['@']),
							   (u'q6_1', ['@', 'gender', 'q2', 'q2b']),
							   (u'q6_2', ['@', 'gender', 'q2', 'q2b']),
							   (u'q6_3', ['@', 'gender', 'q2', 'q2b']),
							   ('age', ['@', 'gender', 'q2', 'q2b'])]
		self.assertEqual(b_meta1['x_y_map'], x_y_map)
		self.assertEqual(b_meta1['extended_yks_global'], ['q2b'])
		batch2.extend_y('q2b', 'q2')
		batch2.extend_y('q3', 'q6')
		extended_yks_per_x = {u'q6_3': ['q3'], 'q2': ['q2b'], u'q6_1': ['q3'],
							  u'q6_2': ['q3'], 'q6': ['q3']}
		self.assertEqual(b_meta2['extended_yks_per_x'], extended_yks_per_x)
		x_y_map = [('q1', ['@', 'gender', 'q2']),
							   ('q2', ['@', 'gender', 'q2', 'q2b']),
							   ('q6', ['@']),
							   (u'q6_1', ['@', 'gender', 'q2', 'q3']),
							   (u'q6_2', ['@', 'gender', 'q2', 'q3']),
							   (u'q6_3', ['@', 'gender', 'q2', 'q3']),
							   ('age', ['@', 'gender', 'q2'])]
		self.assertEqual(b_meta2['x_y_map'], x_y_map)

	def test_replace_y(self):
		batch, ds = _get_batch('test', full=True)
		b_meta = _get_meta(batch)
		self.assertRaises(ValueError, batch.replace_y, 'q2b', 'q5')
		batch.replace_y('q2b', 'q6')
		exclusive_yks_per_x = {u'q6_3': ['@', 'q2b'],
							   u'q6_1': ['@', 'q2b'],
							   u'q6_2': ['@', 'q2b'],
							   'q6': ['@', 'q2b']}
		self.assertEqual(b_meta['exclusive_yks_per_x'], exclusive_yks_per_x)
		x_y_map = [('q1', ['@', 'gender', 'q2']),
							   ('q2', ['@', 'gender', 'q2']),
							   ('q6', ['@']),
							   (u'q6_1', ['@', 'q2b']),
							   (u'q6_2', ['@', 'q2b']),
							   (u'q6_3', ['@', 'q2b']),
							   ('age', ['@', 'gender', 'q2'])]
		self.assertEqual(b_meta['x_y_map'], x_y_map)

	def test_extend_filter(self):
		batch, ds = _get_batch('test', full=True)
		b_meta = _get_meta(batch)
		ext_filters = {'q1': {'age': frange('20-25')}, ('q2', 'q6'): {'age': frange('30-35')}}
		batch.extend_filter(ext_filters)
		filter_names = ['men only', '(men only)+(q1)', '(men only)+(q2)',
						'(men only)+(q6)', '(men only)+(q6_1)',
						'(men only)+(q6_2)', '(men only)+(q6_3)']
		self.assertEqual(b_meta['filter_names'], filter_names)
		x_filter_map = OrderedDict(
			[('q1', {'(men only)+(q1)': intersection([{'gender': 1}, {'age': [20, 21, 22, 23, 24, 25]}])}),
			 ('q2', {'(men only)+(q2)': intersection([{'gender': 1}, {'age': [30, 31, 32, 33, 34, 35]}])}),
			 ('q6', {'(men only)+(q6)': intersection([{'gender': 1}, {'age': [30, 31, 32, 33, 34, 35]}])}),
			 (u'q6_1', {'(men only)+(q6_1)': intersection([{'gender': 1}, {'age': [30, 31, 32, 33, 34, 35]}])}),
			 (u'q6_2', {'(men only)+(q6_2)': intersection([{'gender': 1}, {'age': [30, 31, 32, 33, 34, 35]}])}),
			 (u'q6_3', {'(men only)+(q6_3)': intersection([{'gender': 1}, {'age': [30, 31, 32, 33, 34, 35]}])}),
			 ('age', {'men only': {'gender': 1}})])
		self.assertEqual(b_meta['x_filter_map'], x_filter_map)

	def test_add_y_on_y(self):
		batch, ds = _get_batch('test', full=True)
		b_meta = _get_meta(batch)
		batch.add_y_on_y('cross', {'age': frange('20-30')}, 'extend')
		batch.add_y_on_y('back', 'no_filter', 'replace')
		self.assertEqual(b_meta['y_filter_map']['back'], 'no_filter')
		self.assertEqual(b_meta['y_on_y'], ['cross', 'back'])


	######################### meta edit methods ##############################

	def test_hiding(self):
		batch, ds = _get_batch('test', full=True)
		b_meta = _get_meta(batch)
		batch.hiding(['q1', 'q6'], [1, 2], ['x', 'y'])
		for v in ['q1', u'q6_1', u'q6_2', u'q6_3']:
			self.assertTrue(not b_meta['meta_edits'][v]['rules']['x'] == {})
		for v in ['q1', 'q6']:
			self.assertTrue(not b_meta['meta_edits'][v]['rules']['y'] == {})

	def test_sorting(self):
		batch, ds = _get_batch('test', full=True)
		b_meta = _get_meta(batch)
		batch.sorting(['q1', 'q6'])
		for v in ['q1', u'q6_1', u'q6_2', u'q6_3']:
			self.assertTrue(not b_meta['meta_edits'][v]['rules']['x'] == {})
		self.assertTrue(b_meta['meta_edits']['q6'].get('rules') is None)

	def test_slicing(self):
		batch, ds = _get_batch('test', full=True)
		b_meta = _get_meta(batch)
		self.assertRaises(KeyError, batch.slicing, 'q6', [1, 2])
		batch.slicing(['q1', 'q2'], [3, 2, 1], ['x', 'y'])
		for v in ['q1', 'q2']:
			for ax in ['x', 'y']:
				self.assertTrue(not b_meta['meta_edits'][v]['rules'][ax] == {})

