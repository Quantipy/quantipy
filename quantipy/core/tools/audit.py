#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import quantipy as qp
import numpy  as np
from quantipy.core.tools.qp_decorators import *

from difflib import SequenceMatcher
from collections import OrderedDict
import json
import copy


VALID_CONVERT = {'float':
					['single', 'int', 'float'],
				 'int':
				 	['single', 'int'],
				 'delimited set':
				 	['single', 'delimited set'],
				 'single':
				 	['int', 'date', 'string', 'single'],
				 'string':
				 	['single', 'delimited set', 'int', 'float', 'date', 'string']}

class Audit(object):
	"""
	Container for qp.DataSet instances, which get compared.
	"""
	# ------------------------------------------------------------------------
	# Conventions
	# ------------------------------------------------------------------------

	def __init__(self, datasets, path=None, dimensions_comp=True):
		self.datasets = []
		self.path = path
		self._dimensions_comp = dimensions_comp
		self._raise_error = False
		self.ds_names = []
		self.ds_alias = OrderedDict()
		self.all_incl_vars = []
		self.unpaired_vars = None
		self.add_datasets(datasets)

	@modify(to_list='ds')
	def __getitem__(self, ds):
		if not len(set([type(d) for d in ds])) == 1:
			raise TypeError('All slicer must either be str or int.')
		if isinstance(ds[0], (str, unicode)):
			not_incl = [d for d in ds if not d in self.ds_names + self.ds_alias.values()]
			if not_incl:
				raise ValueError('{} is not included.'.format(not_incl))
			datasets = []
			for d in ds:
				for x, name in enumerate(self.ds_alias):
					if d == name or d == self.ds_alias[name]:
						datasets.append(x)
			ds = datasets
		datasets = [d for x, d in enumerate(self.datasets) if x in ds]

		return datasets[0] if len(datasets) == 1 else datasets

	@modify(to_list='ds')
	def __contains__(self, ds):
		for d in ds:
			if not d in self.ds_alias.keys() or d in self.ds_alias.values():
				if self._raise_error:
					raise KeyError('{} is not included.'.format(d))
				else:
					return False
		return True

	# ------------------------------------------------------------------------
	# file i/o
	# ------------------------------------------------------------------------

	def _load_ds(self, name):
		path_json = '{}/{}.json'.format(self.path, name)
		path_csv = '{}/{}.csv'.format(self.path, name)
		dataset = qp.DataSet(name, self._dimensions_comp)
		dataset.set_verbose_infomsg(False)
		dataset.read_quantipy(path_json, path_csv)
		return dataset

	@modify(to_list='datasets')
	def add_datasets(self, datasets):
		"""
		Add DataSet instances to the Audit container.

		Parameters
		----------
		datasets: (list of) qp.DataSet/ str/ dict

		Returns
		-------
		None
		"""
		for ds in datasets:
			if isinstance(ds, dict):
				alias = ds.keys()[0]
				ds = ds.values()[0]
			else:
				alias = None
			if isinstance(ds, (str, unicode)):
				if not self.path:
					msg = 'If elements in datasets are str, a path must be provided: {}'
					raise ValueError(msg.format(ds))
				ds = self._load_ds(ds)
			if not alias:
				alias = ds.name
			if not any(n in self for n in [ds.name, alias]):
				self.ds_names.append(ds.name)
				self.ds_alias[ds.name] = alias
				self.datasets.append(ds)
			else:
				raise ValueError('{} is already in Audit.'.format(ds.name))
		self.all_incl_vars = self._all_incl_vars()
		return None

	def add_path(self, path):
		"""
		Define the path attribute.
		"""
		self.path = path
		return None

	@modify(to_list='names')
	def save(self, names=None, suffix='_audit'):
		"""
		Save all included DataSet instances.
		"""
		if not names:
			names = self.ds_names
		else:
			self._raise_error = True
			names in self
			self._raise_error = False

		path = self.path
		for n in names:
			ds = self[n]
			if not path: path = ds.path
			path = '../' if path == '/' else path
			path_json = '{}/{}{}.json'.format(path, n, suffix)
			path_csv = '{}/{}{}.csv'.format(path, n, suffix)
			ds.write_quantipy(path_json, path_csv)
			print "Created '{}':\n\t{}\n\t{}".format(self._get_alias(n),
			                                      path_json, path_csv)
		return None

	# ------------------------------------------------------------------------
	# helper
	# ------------------------------------------------------------------------

	def _update(self):
		self.all_incl_vars = self._all_incl_vars()
		self.mismatches(verbose=False)
		return None

	@modify(to_list='names')
	def _get_alias(self, names):
		return [self.ds_alias.get(n, n) for n in names if self[n]]
		# return alias[0] if len(alias) == 1 else alias

	# ------------------------------------------------------------------------
	# validate
	# ------------------------------------------------------------------------

	def validate_all(self, spss_limits=False):
		"""
		Runs validate for included DataSets and reports broken instance names.

		Parameters
		----------
		spss_limits: bool, default False
			Define if spss_limits should be tested or not.

		Returns
		-------
		inconsistent: list of str
			Names of inconsistent DataSet instances.
		"""
		inconsistent = []
		for ds in self.datasets:
			if not ds.validate(spss_limits, False) is None:
				inconsistent.append(self.ds_alias[ds.name])
		if not inconsistent:
			print 'No issues found in the datasets!'
		return inconsistent

	# ------------------------------------------------------------------------
	# mismatches
	# ------------------------------------------------------------------------

	def mismatches(self, verbose=True):
		"""
		Reports variables that are not included in all DataSets.

		Returns
		-------
		unpaired: pd.DataFrame
		"""
		var_map = self._misspelling_map()
		unpaired = []

		for var in self.all_incl_vars:
			header = OrderedDict()
			for name in self.ds_names:
				n = self.ds_alias[name]
				if var in var_map[var.lower()].get(n, []):
					header[n] = ''
				else:
					header[n] = []
					for v in self.all_incl_vars:
						if v == var:
							continue
						elif var.lower() == v.lower():
							header[n] = var_map[v.lower()][n]
							break
						elif var.lower() in v.lower() and n in var_map[v.lower()]:
							header[n].extend(var_map[v.lower()][n])
					if not header[n]:
						header[n] = 'x'
			df = pd.DataFrame([header], index=[var])
			if not all(v == '' for v in df.values.tolist()[0]):
				unpaired.append(df)
		if unpaired:
			unpaired = pd.concat(unpaired, axis=0)
			self.unpaired_vars = unpaired
			return unpaired
		else:
			self.unpaired_vars = None
			if verbose:
				print 'No mismatches detected in included DataSets.'
			return None

	def _misspelling_map(self):
		name_map = {}
		for name in self.ds_names:
			n = self.ds_alias[name]
			for v in self[n].variables():
				low = v.lower()
				if name_map.get(low, {}).get(n):
					name_map[low][n].append(v)
				elif name_map.get(low):
					name_map[low].update({n: [v]})
				else:
					name_map[low] = {n: [v]}
		return name_map

	def _all_incl_vars(self):
		all_included = []
		for ds in self.datasets:
			for v in ds.variables():
				if not v in all_included:
					all_included.append(v)
		return all_included

	@modify(to_list=['datasets', 'ignore'])
	@verify(is_str=['name', 'datasets', 'ignore'])
	def rename_by(self, name, datasets=None, ignore=[]):
		"""
		Take over variable names of a defined DataSet.

		Loops over ``self.unpaired_vars`` and if only one alternative variable
		is included, it is renamed by name of the variable in the master
		DataSet.

		Parameters
		----------
		name: str
			Name of the master DataSet from which the variables names are taken.
		datasets: str/ list of str
			Name(s) of the DataSet(s) for which the variables should be renamed.
			If None, all included DataSets are taken, except of the master
			DataSet.
		ignore: str/ list of str
			Name(s) of variables that will not be renamed.

		Returns
		-------
		None
		"""
		if self.unpaired_vars is None:
			self.mismatches(verbose=False)
		if self.unpaired_vars is None:
			print 'No mismatches detected in included DataSets.'
			return None
		name = self._get_alias(name)
		m_ds = self[name]
		if not datasets:
			datasets = [alias for alias in self.ds_alias.values() if not alias == name]
		else:
			datasets = [ds for ds in  self._get_alias(datasets) if not ds == name]
		for ds in datasets:
			for var, incl in self.unpaired_vars[[ds]].iterrows():
				v = incl.values.tolist()[0]
				if isinstance(v, list) and len(v) == 1 and m_ds.var_exists(var):
					self[ds].rename(v[0], var)
		self._update()
		return None

	@modify(to_list='datasets')
	@verify(is_str='datasets')
	def rename_from_mapper(self, datasets=None, mapper={}):
		"""
		Renames variables from mapper for all defined datasets.

		Parameters
		----------
		datasets: str/ list of str
			Name(s) of the DataSet(s) for which the variables should be renamed.
			If None, all included DataSets are taken.
		mapper: dict in form if {str: str}
			The key is renamed into the value.

		Returns
		-------
		None
		"""
		if not isinstance(mapper, dict):
			raise ValueError("'mapper' must be a dict: {str: str}")
		elif not all(isinstance(k, (str, unicode)) and isinstance(v, (str, unicode))
		             for k, v in mapper.items()):
			raise ValueError("'mapper' must be a dict: {str: str}")
		if not datasets: datasets = self.ds_names
		for ds in datasets:
			for k, v in mapper.items():
				if self[ds].var_exists(k):
					self[ds].rename(k, v)
		self._update()
		return None

	@modify(to_list=['datasets', 'ignore'])
	@verify(is_str=['datasets', 'ignore'])
	def remove_mismatches(self, datasets=None, ignore=[]):
		"""
		Remove variables that are not included in all DataSets.

		Parameters
		----------
		datasets: str/ list of str
			Name(s) of the DataSet(s) for which the variables should be removed.
			If None, all included DataSets are taken.
		ignore: str/ list of str
			Name(s) of variables that will not be removed.

		Returns
		-------
		None
		"""
		if self.unpaired_vars is None:
			self.mismatches(verbose=False)
		if self.unpaired_vars is None:
			print 'No mismatches detected in included DataSets.'
			return None
		if not datasets: datasets = self.ds_names
		for ds in datasets:
			for v in self.unpaired_vars.index.tolist():
				if self[ds].var_exists(v) and not v in ignore:
					self[ds].drop(v)
		print 'Removed unpaired variables from {}.'.format(datasets)
		self._update()
		return None

	@modify(to_list=['name', 'datasets', 'ignore'])
	@verify(is_str=['name', 'datasets', 'ignore'])
	def fill_mismatches_by(self, name, datasets=None, ignore=[]):
		"""
		Fill mismatches in DataSets by the metadata if a defined DataSet.

		Parameters
		----------
		name: str
			Name of the master DataSet from which the variable meta is taken.
		datasets: str/ list of str
			Name(s) of the DataSet(s) for which the variable meta should be
			included. If None, all included DataSets are taken.
		ignore: str/ list of str
			Name(s) of variables for which meta is not filled.

		Returns
		-------
		None
		"""
		def _get_missing(ds):
			missing = self.unpaired_vars[ds]
			missing = missing[missing != ''].index.tolist()
			return missing

		if self.unpaired_vars is None:
			self.mismatches(verbose=False)
		if self.unpaired_vars is None:
			print 'No mismatches detected in included DataSets.'
			return None
		for n in self._get_alias(names):
			m_ds = self[n]
			if not datasets:
				use_ds = [ds for ds in self.ds_alias.values() if not ds == n]
			else:
				use_ds = [ds for ds in self._get_alias(datasets) if not ds == n]

			use_ig = ignore + _get_missing(n)
			for ds in use_ds:
				variables = [v for v in _get_missing(ds) if not v in use_ig]
				if variables:
					subset = m_ds.subset(variables)
					self[ds].merge_texts(subset)
			self._update()
		if self.unpaired_vars is None:
			print 'All mismatches are filled.'
		return None

	# ------------------------------------------------------------------------
	# types
	# ------------------------------------------------------------------------

	def report_type_diffs(self):
		"""
		Check if included variables have the same types.
		"""
		all_df = []
		for v in self.all_incl_vars:
			header = OrderedDict()
			for name in self.ds_alias.values():
				if self[name].var_exists(v):
					header[name] = self[name]._get_type(v)
			types = header.values()
			if not len(set(types)) == 1:
				header['valid'] = []
				for to, of in VALID_CONVERT.items():
					if all(ty in of for ty in types):
						header['valid'].append(to)
				if not header['valid']: header['valid'] = ''
			 	all_df.append(pd.DataFrame([header], index=[v]))
		if all_df:
			all_df = pd.concat(all_df, axis=0)
			return all_df.replace(np.NaN, '')
		else:
			print 'No varied types detected in included DataSets.'

	# ------------------------------------------------------------------------
	# labels
	# ------------------------------------------------------------------------

	def _get_tks_for_checking(self, var, obj):
		tks = []
		for name in self.ds_names:
			ds = self[name]
			if ds.var_exists(var):
				if ds.is_array(var):
					if obj == 'label':
						t_obj = ds._meta['masks'][var]['text']
					elif obj == 'values':
						t_obj = ds._meta['lib']['values'][var][0]['text']
				elif ds._is_array_item(var):
					if obj == 'label':
						t_obj = ds._meta['columns'][var]['text']
					elif obj == 'values':
						return []
				else:
					if obj == 'label':
						t_obj = ds._meta['columns'][var]['text']
					elif obj == 'values':
						t_obj = ds._meta['columns'][var]['values'][0]['text']
				for tk, val in t_obj.items():
					if not tk in ['x edits', 'y edits']:
						tks.append(tk)
					else:
						for etk in val.keys():
							tks.append('{}~{}'.format(etk, tk))
		check_tk = set([tk for tk in tks if tks.count(tk) > 1])
		return list(check_tk)

	@staticmethod
	def _compare_texts(tks, tobj1, tobj2, strict):
		diff_tks = []
		for tk in tks:
			tkey = tk.split('~')
			edit = tkey[1] if len(tkey) > 1 else None
			tkey = tkey[0]
			t1 = tobj1.get(edit, tobj1).get(tkey, None)
			t2 = tobj2.get(edit, tobj2).get(tkey, None)
			if t1 and t2:
				s = SequenceMatcher(None, t1, t2)
				if s.ratio() < strict:
					diff_tks.append(tk)
		return diff_tks

	def report_label_diffs(self, strict=0.95):
		"""
		Reports variables that have different labels for the same text_key.

		Parameters
		----------
		strict: float, default 0.9
			Requested similarity of the labels.

		Returns
		-------
		label_diff: pd.DataFrame
			The values of the DataFrame include the text_keys whose texts
			differ.
		"""
		all_df = []
		for v in self.all_incl_vars:
			tks = self._get_tks_for_checking(v, 'label')
			header = OrderedDict()
			for x, n1 in enumerate(self.ds_alias.values(), 1):
				for n2 in self.ds_alias.values()[x:]:
					if all(self[n].var_exists(v) for n in [n1, n2]):
						collection = 'masks' if self[n1].is_array(v) else 'columns'
						tobj1 = self[n1]._meta[collection][v]['text']
						tobj2 = self[n2]._meta[collection][v]['text']
						diff_tks = self._compare_texts(tks, tobj1, tobj2, strict)
						header['{},\n{}'.format(n1, n2)] = diff_tks if diff_tks else ''
					else:
						header['{},\n{}'.format(n1, n2)] = ''
			df = pd.DataFrame([header], index=[v])
			if not all(tk == '' for tk in df.values.tolist()[0]):
				all_df.append(df)
		if all_df:
			return pd.concat(all_df, axis=0)
		else:
			print 'No varied labels detected in included DataSets.'

	@modify(to_list='var')
	@verify(is_str=['var', 'datasets', 'text_key'])
	def show_labels(self, var, datasets, text_key):
		"""
		Display labels of variables in different DataSets.

		Parameters
		----------
		var: str/ list of str
			Displays label texts for these variable(s).
		datasets: list of str
			Names of the  DataSets from which the label texts are taken.
		text_key: str
			Text key for text-based label information. Can be provided as
			``'x edits~tk'`` or ``'y edits~tk'``, then the edited text is taken.
		"""
		ds = self[datasets]
		text_key = text_key.split('~')
		etk = text_key[1].split()[0] if len(text_key) > 1 else None
		text_key = text_key[0]
		for v in var:
			if not all(v in d for d in ds): continue
			texts = [d.text(v, False, text_key, etk) for d in ds]
			print '{}:\n'.format(v)
			for t in texts:
				print '\t{}'.format(t)
			print '*'*60
		return None

	@modify(to_list=['datasets', 'var'])
	@verify(is_str=['name', 'datasets', 'var', 'text_key'])
	def relabel_by(self, var, text_key, name, datasets=None):
		"""
		Take over variable labels of a defined DataSet.

		Parameters
		----------
		var: str/ list of str
			Variables that are relabeled
		text_key: str
			Text key for text-based label information. Can be provided as
			``'x edits~tk'`` or ``'y edits~tk'``, then the edited text is taken.
		name: str
			Name of the master DataSet from which the variables labels are taken.
		datasets: str/ list of str
			Name(s) of the DataSet(s) for which the variables should be relabeled.
			If None, all included DataSets are taken, except of the master
			DataSet.

		Returns
		-------
		None
		"""
		name = self._get_alias(name)
		m_ds = self[name]
		if not datasets:
			datasets = [alias for alias in self.ds_alias.values() if not alias == name]
		else:
			datasets = [ds for ds in  self._get_alias(datasets) if not ds == name]
		text_key = text_key.split('~')
		etk = text_key[1].split()[0] if len(text_key) > 1 else None
		text_key = text_key[0]
		for v in var:
			if not m_ds.var_exists(v): continue
			label = m_ds.text(v, False, text_key, etk)
			for n in datasets:
				ds = self[n]
				if ds.var_exists(v):
					ds.set_variable_text(v, label, text_key, etk)
		return None

	# ------------------------------------------------------------------------
	# categoricals
	# ------------------------------------------------------------------------

	def report_cat_diffs(self):
		"""
		Reports variables that have different categories in the DataSets.

		Returns
		-------
		cat_diff: pd.DataFrame
			The values of the DataFrame include various cats and the text_keys
			whose texts differ.
		"""
		all_df = []
		for v in self.all_incl_vars:
			if any(v in ds and not ds._has_categorical_data(v) for ds in self.datasets):
				continue
			header = OrderedDict()
			cats = []
			for name in self.ds_alias.values():
				if self[name].var_exists(v):
					codes = self[name].codes(v)
					header[name] = codes
					cats.append(codes)
				else:
					header[name] = ''
			if not all(cat == cats[0] for cat in cats):
			 	all_df.append(pd.DataFrame([header], index=[v]))
		if all_df:
			all_df = pd.concat(all_df, axis=0)
			return all_df.replace(np.NaN, '')
		else:
			print 'No varied categories detected in included DataSets.'

	def report_cat_text_diffs(self):
		"""
		Reports variables that have different categories in the DataSets

		Parameters
		----------
		strict: float, default 0.9
			Requested similarity of the labels.

		Returns
		-------
		cat_diff: pd.DataFrame
			The values of the DataFrame include various cats and the text_keys
			whose texts differ.
		"""
		# all_df = []
		# for v in self.all_incl_vars:
		# 	tks = self._get_tks_for_checking(v, 'label')
		# 	header = OrderedDict()
		# 	for x, n1 in enumerate(self.ds_names, 1):
		# 		for n2 in self.ds_names[x:]:
		# 			if all(self[n].var_exists(v) for n in [n1, n2]):
		# 				collection = 'masks' if self[n1].is_array(v) else 'columns'
		# 				tobj1 = self[n1]._meta[collection][v]['text']
		# 				tobj2 = self[n2]._meta[collection][v]['text']
		# 				diff_tks = self._compare_texts(tks, tobj1, tobj2, strict)
		# 				header['{},\n{}'.format(n1, n2)] = diff_tks if diff_tks else ''
		# 			else:
		# 				header['{},\n{}'.format(n1, n2)] = ''
		# 	df = pd.DataFrame([header], index=[v])
		# 	if not all(tk == '' for tk in df.values.tolist()[0]):
		# 		all_df.append(df)
		# if all_df:
		# 	return pd.concat(all_df, axis=0)
		# else:
		# 	print 'No varied labels detected in included DataSets.'

	@verify(is_str=['var', 'text_key'])
	def show_cats(self, var, text_key):
		"""
		Display labels of variables in different DataSets.

		Parameters
		----------
		var: str
			Displays value texts for this variable.
		datasets: tuple of str
			Names of the two DataSets from which the label texts are taken.
		text_key: str
			Text key for text-based label information. Can be provided as
			``'x edits~tk'`` or ``'y edits~tk'``, then the edited text is taken.
		"""
		text_key = text_key.split('~')
		etk = text_key[1].split()[0] if len(text_key) > 1 else None
		text_key = text_key[0]
		all_df = []
		for name in self.ds_alias.values():
			if var in self[name]:
				val = self[name].value_texts(var, text_key, etk)
				codes = self[name].codes(var)
				df = pd.DataFrame(val, index=codes, columns=[name])
				all_df.append(df)
		final_df = reduce(lambda x, y: x.join(y), all_df)
		final_df.index.name = var
		return final_df

	@modify(to_list=['datasets', 'var'])
	@verify(is_str=['name', 'datasets', 'var', 'text_key'])
	def extend_reorder_cats_by(self, var, text_key, name, datasets=None):
		"""
		Take over missing categories for a variable of a defined DataSet.

		Parameters
		----------
		var: str/ list of str
			Variables that are relabeled
		text_key: str
			Text key for text-based label information. Can be provided as
			``'x edits~tk'`` or ``'y edits~tk'``, then the edited text is taken.
		name: str
			Name of the master DataSet from which the variable categories are taken.
		datasets: str/ list of str
			Name(s) of the DataSet(s) for which the variables should be relabeled.
			If None, all included DataSets are taken, except of the master
			DataSet.

		Returns
		-------
		None
		"""
		name = self._get_alias(name)
		m_ds = self[name]
		if not datasets:
			datasets = [alias for alias in self.ds_alias.values() if not alias == name]
		else:
			datasets = [ds for ds in self._get_alias(datasets) if not ds == name]
		for v in var:
			if not v in m_ds: continue
			codes = m_ds.codes(v)
			values = m_ds.value_texts(v, text_key)
			for n in datasets:
				ds = self[n]
				if ds.var_exists(v):
					n_values = [(c, val) for c, val in zip(codes, values)
								if not c in ds.codes(v)]
					if n_values:
						ds.extend_values(v, n_values, text_key)
					ds.reorder_values(v, codes)
		return None

	# ------------------------------------------------------------------------
	# missing array items
	# ------------------------------------------------------------------------

	def unpaired_array_items(self):
		"""
		Check if included arrays have the same items.
		"""
		arrays = [a for a in self.all_incl_vars
				  if any(d.var_exists(a) and d.is_array(a)
				         for d in self.datasets)]
		arrays = []
		total_ais = OrderedDict()
		for v in self.all_incl_vars:
			for d in self.datasets:
				if d.var_exists(v) and d.is_array(v):
					if not v in arrays: arrays.append(v)
					total_ais[v] = []
					for source in d.sources(v):
						if not source in total_ais[v]:
							total_ais[v].append(source)

		all_df = []
		for a in arrays:
			a_header = OrderedDict()
			items_df = []
			for s in total_ais[a]:
				i_header = OrderedDict()
				for name in self.ds_names:
					if not name in a_header:
						if not self[name].var_exists(a):
							a_header[name] = 'x'
						elif not self[name]._get_type(a) == 'array':
							a_header[name] = self[name]._get_type(a)
						else:
							a_header[name] = ''
					if not any(self[name].var_exists(v) for v in [a, s]):
						i_header[name] = 'x'
					elif not s in self[name].sources(a):
						i_header[name] = 'x'
					else:
						i_header[name] = ''
				i_df = pd.DataFrame([i_header], index=[s])
				if not all(v == '' for v in i_df.values.tolist()[0]):
					items_df.append(i_df)
			a_df = pd.DataFrame([a_header], index=[a])
			if not all(v == '' for v in a_df.values.tolist()[0]):
				all_df.append(a_df)
			if items_df:
				items = pd.concat(items_df, axis=0)
				all_df.append(items)
		return pd.concat(all_df, axis=0)

