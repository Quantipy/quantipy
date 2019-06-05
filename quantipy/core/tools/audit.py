#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import quantipy as qp
import numpy  as np
from quantipy.core.tools.qp_decorators import *
from quantipy.core.tools.dp.prep import frange

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
		if isinstance(ds[0], str):
			not_incl = [d for d in ds if not d in self.ds_names + list(self.ds_alias.values())]
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
			if not d in list(self.ds_alias.keys()) or d in list(self.ds_alias.values()):
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
				alias = list(ds.keys())[0]
				ds = list(ds.values())[0]
			else:
				alias = None
			if isinstance(ds, str):
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
		return Nontye

	@modify(to_list='names')
	def save(self, names='all', suffix='_audit'):
		"""
		Save all included DataSet instances.
		"""
		if not names:
			names = []
		elif names == ['all']:
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
			print("Created '{}':\n\t{}\n\t{}".format(self._get_alias(n)[0],
			                                      path_json, path_csv))
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

	@modify(to_list='datasets')
	def _str_to_instance(self, datasets=None):
		if not datasets:
			return self.datasets
		else:
			return self[datasets] if len(datasets) > 1 else [self[datasets]]

	def _is_categorical(self, var, datasets=None):
		use_ds = [ds for ds in self._str_to_instance(datasets) if var in ds]
		return all(ds._has_categorical_data(var) for ds in use_ds)

	def _is_array(self, var, datasets=None):
		use_ds = [ds for ds in self._str_to_instance(datasets) if var in ds]
		return all(ds.is_array(var) for ds in use_ds)

	def _is_array_item(self, var, array, datasets=None):
		use_ds = [ds for ds in self._str_to_instance(datasets) if var in ds]
		return all(var in ds.sources(array) for ds in use_ds)

	@modify(to_list='var')
	def _get_sources(self, var=None, datasets=None):
		if not datasets: datasets = list(self.ds_alias.values())
		if not var:
			var = [v for v in self.all_incl_vars if self._is_array(v, datasets)]
		else:
			var = [v for v in var if self._is_array(v, datasets)]
		total_ais = OrderedDict()
		for v in var:
			for d in datasets:
				if v in self[d] and not v in total_ais: total_ais[v] = []
				for source in self[d].sources(v):
					if not source in total_ais[v]:
						total_ais[v].append(source)
		return total_ais

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
			print('No issues found in the datasets!')
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
				print('No mismatches detected in included DataSets.')
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
	def rename_by(self, name, datasets='all', ignore=[]):
		"""
		Take over variable names of a defined DataSet.

		Loops over ``self.unpaired_vars`` and if only one alternative variable
		is included, it is renamed by name of the variable in the master
		DataSet.

		Parameters
		----------
		name: str
			Name of the master DataSet from which the variables names are taken.
		datasets: str/ list of str, default 'all'
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
			print('No mismatches detected in included DataSets.')
			return None
		name = self._get_alias(name)
		m_ds = self[name]
		if not datasets:
			datasets = []
		elif datasets == ['all']:
			datasets = [alias for alias in list(self.ds_alias.values()) if not alias == name]
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
	def rename_from_mapper(self, datasets='all', mapper={}):
		"""
		Renames variables from mapper for all defined datasets.

		Parameters
		----------
		datasets: str/ list of str, default 'all'
			Name(s) of the DataSet(s) for which the variables should be renamed.
			If 'all', all included DataSets are taken.
		mapper: dict in form if {str: str}
			The key is renamed into the value.

		Returns
		-------
		None
		"""
		if not isinstance(mapper, dict):
			raise ValueError("'mapper' must be a dict: {str: str}")
		elif not all(isinstance(k, str) and isinstance(v, str)
		             for k, v in list(mapper.items())):
			raise ValueError("'mapper' must be a dict: {str: str}")
		if not datasets:
			datasets = []
		elif datasets == ['all']:
			datasets = self.ds_names
		for ds in datasets:
			for k, v in list(mapper.items()):
				if self[ds].var_exists(k):
					self[ds].rename(k, v)
		self._update()
		return None

	@modify(to_list=['datasets', 'ignore'])
	@verify(is_str=['datasets', 'ignore'])
	def remove_mismatches(self, datasets='all', ignore=[]):
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
			print('No mismatches detected in included DataSets.')
			return None
		if not datasets:
			datasets = []
		elif datasets == ['all']:
			datasets = list(self.ds_alias.values())
		for ds in datasets:
			for v in self.unpaired_vars.index.tolist():
				if self[ds].var_exists(v) and not v in ignore:
					self[ds].drop(v)
		print('Removed unpaired variables from {}.'.format(datasets))
		self._update()
		return None

	@modify(to_list=['name', 'datasets', 'ignore'])
	@verify(is_str=['name', 'datasets', 'ignore'])
	def fill_mismatches_by(self, name='all', datasets='all', ignore=[]):
		"""
		Fill mismatches in DataSets by the metadata if a defined DataSet.

		Parameters
		----------
		name: str/ list of str
			Name(s) of the master DataSet(s) from which the variable meta is
			taken.
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
			if self.unpaired_vars is None: return []
			missing = self.unpaired_vars[ds]
			missing = missing[missing != ''].index.tolist()
			return missing

		if self.unpaired_vars is None:
			self.mismatches(verbose=False)
		if self.unpaired_vars is None:
			print('No mismatches detected in included DataSets.')
			return None
		if not name:
			name = []
		elif name == ['all']:
			name = list(self.ds_alias.values())
		for n in self._get_alias(name):
			m_ds = self[n]
			if not datasets:
				use_ds = []
			elif datasets == ['all']:
				use_ds = [ds for ds in list(self.ds_alias.values()) if not ds == n]
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
			print('All mismatches are filled.')
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
			for name in list(self.ds_alias.values()):
				if self[name].var_exists(v):
					header[name] = self[name]._get_type(v)
			types = list(header.values())
			if not len(set(types)) == 1:
				header['valid'] = []
				for to, of in list(VALID_CONVERT.items()):
					if all(ty in of for ty in types):
						header['valid'].append(to)
				if not header['valid']: header['valid'] = ''
			 	all_df.append(pd.DataFrame([header], index=[v]))
		if all_df:
			all_df = pd.concat(all_df, axis=0)
			return all_df.replace(np.NaN, '')
		else:
			print('No varied types detected in included DataSets.')

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
				for tk, val in list(t_obj.items()):
					if not tk in ['x edits', 'y edits']:
						tks.append(tk)
					else:
						for etk in list(val.keys()):
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
			for x, n1 in enumerate(list(self.ds_alias.values()), 1):
				for n2 in list(self.ds_alias.values())[x:]:
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
			all_df = pd.concat(all_df, axis=0)
			return all_df, all_df.index.tolist()
		else:
			print('No varied labels detected in included DataSets.')
			return None, []

	@modify(to_list=['var', 'datasets'])
	@verify(is_str=['var', 'text_key'])
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
		if datasets == ['all']: datasets = self.ds_names
		ds = [self[datasets]] if len(datasets)==1 else self[datasets]
		alias = self._get_alias(datasets)
		tk = text_key.split('~')
		etk = tk[1].split()[0] if len(tk) > 1 else None
		tk = tk[0]
		all_df = []
		for v in var:
			if not all(v in d for d in ds): continue
			texts = [d.text(v, False, tk, etk) if v in d else None
					 for d in ds]
			index = pd.MultiIndex.from_tuples([(v, a) for a in alias])
			df = pd.DataFrame({text_key: texts}, index=index)
			all_df.append(df)
		if not all_df:
			print('No variables to show.')
		else:
			return pd.concat(all_df, axis=0)

	@modify(to_list=['datasets', 'var'])
	@verify(is_str=['name', 'datasets', 'var', 'text_key'])
	def relabel_by(self, var, text_key, name, datasets='all'):
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
			datasets = []
		elif datasets == ['all']:
			datasets = [alias for alias in list(self.ds_alias.values()) if not alias == name]
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
		Reports variables that have different categorie codes in the DataSets.

		Returns
		-------
		cat_diff: pd.DataFrame
			The values of the DataFrame include various cats and the text_keys
			whose texts differ.
		"""
		all_df = []
		for v in self.all_incl_vars:
			if not self._is_categorical(v): continue
			header = OrderedDict()
			cats = []
			for name in list(self.ds_alias.values()):
				if self[name].var_exists(v):
					codes = self[name].codes(v)
					header[name] = codes
					cats.append(codes)
				else:
					header[name] = ''
			if not all(cat == cats[0] for cat in cats):
			 	all_df.append(pd.DataFrame([header], index=[v]))
		if all_df:
			all_df = pd.concat(all_df, axis=0).replace(np.NaN, '')
			return all_df, all_df.index.tolist()
		else:
			print('No varied categories detected in included DataSets.')
			return None, []

	def report_cat_text_diffs(self, strict=0.9):
		"""
		Reports variables that have different categorie texts in the DataSets.

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
		all_df = []
		for var in self.all_incl_vars:
			if not self._is_categorical(var): continue
			tks = self._get_tks_for_checking(var, 'values')
		 	df_var = []
		 	for x, n1 in enumerate(list(self.ds_alias.values()), 1):
				for n2 in list(self.ds_alias.values())[x:]:
		 			if all(self[n].var_exists(var) for n in [n1, n2]):
		 				if self[n1].is_array(var):
		 					vobj1 = self[n1]._meta['lib']['values'][var]
		 					vobj2 = self[n2]._meta['lib']['values'][var]
		 				else:
			 				vobj1 = self[n1]._meta['columns'][var]['values']
			 				vobj2 = self[n2]._meta['columns'][var]['values']
			 			vobj1_dict = {v['value']: v['text'] for v in vobj1}
			 			vobj2_dict = {v['value']: v['text'] for v in vobj2}
			 			diff = []
			 			for c, text in list(vobj1_dict.items()):
			 				diff.append(self._compare_texts(tks, text,
			 				            					vobj2_dict[c],
			 				            					strict) or np.NaN)
			 			index = pd.MultiIndex.from_tuples([(var, c) for c in list(vobj1_dict.keys())])
						df = pd.DataFrame({'{},\n{}'.format(n1, n2): diff}, index=index)
						df_var.append(df)
			if len(df_var) == 0:
				continue
			elif len(df_var) == 1:
				all_df.append(df_var[0])
			else:
				df_var = pd.concat(df_var, axis=1)
				all_df.append(df_var)
		if all_df:
			all_df = pd.concat(all_df, axis=0).dropna(how='all').replace(np.NaN, '')
			if len(all_df) == 0:
				print('No varied value labels detected in included DataSets.')
				return None, []
			variables = []
			for v in all_df.index.get_level_values(0).tolist():
			 	if not v in variables: variables.append(v)
			return all_df, variables
		else:
			print('No varied value labels detected in included DataSets.')
			return None, []

	@modify(to_list=['var'])
	@verify(is_str=['var', 'text_key'])
	def show_cats(self, var, text_key):
		"""
		Display labels of variables in different DataSets.

		Parameters
		----------
		var: str
			Displays value texts for this variable.
		text_key: str
			Text key for text-based label information. Can be provided as
			``'x edits~tk'`` or ``'y edits~tk'``, then the edited text is taken.
		"""
		text_key = text_key.split('~')
		etk = text_key[1].split()[0] if len(text_key) > 1 else None
		text_key = text_key[0]
		df_all_v = []
		for v in var:
			all_df = []
			for name in list(self.ds_alias.values()):
				if v in self[name]:
					val = self[name].value_texts(v, text_key, etk)
					codes = self[name].codes(v)
					index = pd.MultiIndex.from_tuples([(v, c) for c in codes])
					df = pd.DataFrame(val, index=index, columns=[name])
					all_df.append(df)
			all_df = pd.concat(all_df, axis=1)
			df_all_v.append(all_df)
		if not df_all_v:
			print('No variables to show.')
		else:
			return pd.concat(df_all_v, axis=0)

	@modify(to_list=['datasets', 'var'])
	@verify(is_str=['name', 'datasets', 'var', 'text_key'])
	def align_cats_by(self, var, text_key, name, datasets='all',
	                  overwrite=False, extend=False, reorder=False):
		"""
		Take over categories for variable(s) of a defined DataSet.

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
		overwrite: bool, default False
			Overwrite value texts.
		extend: bool, default False
			Add missing values.
		reorder: bool, default False
			Order values by the master DataSet.

		Returns
		-------
		None
		"""
		name = self._get_alias(name)
		m_ds = self[name]
		if not datasets:
			datasets = []
		elif datasets == ['all']:
			datasets = [alias for alias in list(self.ds_alias.values()) if not alias == name]
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
					if extend and n_values:
						ds.extend_values(v, n_values, text_key)
					if overwrite:
						n_texts = {c: val for c, val in zip(codes, values)
								   if c in ds.codes(v)}
						ds.set_value_texts(v, n_texts, text_key)
					if reorder:
						ds.reorder_values(v, codes)
		return None

	# ------------------------------------------------------------------------
	# missing array items
	# ------------------------------------------------------------------------

	@modify(to_list=['array'])
	@verify(is_str=['array'])
	def show_items(self, array, text_key=None):
		"""
		Display items of arrays in different DataSets.

		Parameters
		----------
		array: str/ list of str
			Displays items for these variables.
		text_key: str
			Text key for text-based label information. Can be provided as
			``'x edits~tk'`` or ``'y edits~tk'``, then the edited text is taken.
			If None is provided, the item name will be diplayed instead of the
			the item label.
		"""
		if not text_key:
			label = False
			etk = None
		else:
			label = True
			text_key = text_key.split('~')
			etk = text_key[1].split()[0] if len(text_key) > 1 else None
			text_key = text_key[0]
		df_all_v = []
		for a in array:
			if not self._is_array(a):
				raise ValueError('{} is not an array.'.format(a))
			all_df = []
		 	for name in list(self.ds_alias.values()):
		 		ds = self[name]
		 		if a in ds:
		 			if label:
		 				val = [ds.text(s, True, text_key, etk) for s in ds.sources(a)]
		 				ind = ds.sources(a)
		 			else:
		 				val = ds.sources(a)
		 				ind = frange('1-{}'.format(len(val)))
		 			index = pd.MultiIndex.from_tuples([(a, n) for n in ind])
		 			df = pd.DataFrame(val, index=index, columns=[name])
		 			all_df.append(df)
		 	all_df = pd.concat(all_df, axis=1)
		 	df_all_v.append(all_df)
		if not df_all_v:
			print('No variables to show.')
		else:
			return pd.concat(df_all_v, axis=0)

	def report_item_diffs(self):
		"""
		Reports arrays that have different items in the DataSets.
		"""
		total_ais = self._get_sources()
		all_df = []
		for a in list(total_ais.keys()):
			v_df = []
			for name in list(self.ds_alias.values()):
				if a in self[name]:
					sources = [np.NaN if s in self[name].sources(a) else 'x'
							   for s in total_ais[a]]
					index = pd.MultiIndex.from_tuples([(a, i) for i in total_ais[a]])
					df = pd.DataFrame(sources, index=index, columns=[name])
					v_df.append(df)
			v_df = pd.concat(v_df, axis=1)
			order = self.show_items(a)
			same_order = all(len([val for val in order.loc[a].T[x].unique().tolist()
			                 if isinstance(val, str)]) < 2
						  	 for x in order.loc[a].T.columns)
			if not same_order:
				index = pd.MultiIndex.from_tuples([(a, 'check order')])
				df = pd.DataFrame([['x']*len(self.ds_names)], index=index,
				                  columns = v_df.columns)
				v_df = df.append(v_df)
		 	all_df.append(v_df)
		if all_df:
			all_df = pd.concat(all_df, axis=0).dropna(how='all').replace(np.NaN, '')
			if len(all_df) == 0:
				print('No varied items detected in included DataSets.')
				return None, []
			variables = []
			for v in all_df.index.get_level_values(0).tolist():
			 	if not v in variables: variables.append(v)
			return all_df, variables
		else:
			print('No varied items detected in included DataSets.')
			return None, []

	def report_item_text_diffs(self, strict=0.9):
		"""
		Reports variables that have different item texts in the DataSets.

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
		arrays = self._get_sources()

		all_df = []
		for a in arrays:
			for s in arrays[a]:
				s_df = []
			 	tks = self._get_tks_for_checking(s, 'label')
			  	for x, n1 in enumerate(list(self.ds_alias.values()), 1):
			 		for n2 in list(self.ds_alias.values())[x:]:
			  			if all(self[n].var_exists(s) for n in [n1, n2]):
		 	 				tobj1 = self[n1]._meta['columns'][s]['text']
		 	 				tobj2 = self[n2]._meta['columns'][s]['text']
		 	 				diff = self._compare_texts(tks, tobj1, tobj2, strict) or np.NaN
		 	 			else:
		 	 				diff = np.NaN
			 	 		index = pd.MultiIndex.from_tuples([(a, s)])
			 	 		df = pd.DataFrame({'{},\n{}'.format(n1, n2): diff}, index=index)
			 			s_df.append(df)
			 	if len(s_df) == 0:
					continue
				elif len(s_df) == 1:
					all_df.append(s_df[0])
				else:
					s_df = pd.concat(s_df, axis=1)
					all_df.append(s_df)

		if all_df:
		 	all_df = pd.concat(all_df, axis=0).dropna(how='all').replace(np.NaN, '')
		 	if len(all_df) == 0:
		 		print('No varied item labels detected in included DataSets.')
		 		return None, []
		 	variables = []
		 	for v in all_df.index.get_level_values(0).tolist():
		 	 	if not v in variables: variables.append(v)
		 	return all_df, variables
		else:
		 	print('No varied item labels detected in included DataSets.')
		 	return None, []

	# def _create_df(self, var, datasets, data_func, index_func):
	# 	v_df = []
	# 	for name in datasets:
	# 		ds = self[name]
	# 		if var in ds:
	# 			data = func(var, ds)
	# 			index = index_func(var)
	# 			index = pd.MultiIndex.from_tuples(index)
	# 			df = pd.DataFrame(data, index=index, columns=[name])
	# 			v_df.append(df)
	# 	return pd.concat(v_df, axis=1)