#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import quantipy as qp
from quantipy.core.tools.qp_decorators import *

from collections import OrderedDict
import json

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
		self.ds_names = []
		self.add_datasets(datasets)


	@verify(is_str='ds')
	@modify(to_list='ds')
	def __getitem__(self, ds):
		not_incl = [d for d in ds if not
						any(dataset.name == d for dataset in self.datasets)]
		if not_incl:
			raise ValueError('{} is not included.'.format(not_incl))
		datasets = [d for d in self.datasets if d.name in ds]
		return datasets[0] if len(datasets) == 1 else datasets

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
		datasets: qp.DataSet/ str, list of qp.DataSet/ str

		Returns
		-------
		None
		"""
		for ds in datasets:
			if isinstance(ds, qp.DataSet):
				if not ds.name in self.ds_names:
					self.datasets.append(ds)
			elif not self.path:
				msg = 'If elements in datasets are str, a path must be provided: {}'
				raise ValueError(msg.format(ds))
			else:
				dataset = self._load_ds(ds)
				if not ds in self.ds_names:
					self.datasets.append(dataset)
				else:
					raise ValueError('{} is already in Audit.'.format(ds.name))
			self._get_ds_names()
		return None

	def _get_ds_names(self):
		for ds in self.datasets:
			if not ds.name in self.ds_names:
				self.ds_names.append(ds.name)
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
		elif not all(n in self.ds_names for n in names):
			not_incl = [n not in self.ds_names for n in names]
			raise ValueError('{} is not included.'.format(not_incl))
		path = self.path
		for n in names:
			ds = self[n]
			if not path: path = ds.path
			path = '../' if path == '/' else path
			path_json = '{}/{}{}.json'.format(path, n, suffix)
			path_csv = '{}/{}{}.csv'.format(path, n, suffix)
			ds.write_quantipy(path_json, path_csv)
			print 'Created:\n\t{}\n\t{}'.format(path_json, path_csv)
		return None

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
				inconsistent.append(ds.name)
		if not inconsistent:
			print 'No issues found in the datasets!'
		return inconsistent

	# ------------------------------------------------------------------------
	# mismatches
	# ------------------------------------------------------------------------

	def mismatches(self, misspelling=True):
		"""
		Reports variables that are not included in all DataSets.

		Parameters
		----------
		misspelling: bool, default True
			If True, similar (different lower and upper cases or inclusions)
			variable names are shown.
		Returns
		-------
		unpaired: pd.DataFrame
		"""
		all_included = self._all_incl_vars()
		var_map = self._misspelling_map()
		unpaired = []

		for var in all_included:
			header = OrderedDict()
			for name in self.ds_names:
				if var in var_map[var.lower()].get(name, []):
					header[name] = ''
				elif misspelling:
					header[name] = []
					for v in all_included:
						if v == var:
							continue
						elif var.lower() == v.lower():
							header[name] = var_map[v.lower()][name]
							break
						elif var.lower() in v.lower() and name in var_map[v.lower()]:
							header[name].extend(var_map[v.lower()][name])
					if not header[name]:
						header[name] = 'x'
				else:
					header[name] = 'x'
			df = pd.DataFrame([header], index=[var])
			if not all(v == '' for v in df.values.tolist()[0]):
				unpaired.append(df)
		if unpaired:
			unpaired = pd.concat(unpaired, axis=0)
			return unpaired
		else:
			print 'No unpaired variables found in the datasets!'
			return None

	def _misspelling_map(self):
		name_map = {}
		for name in self.ds_names:
			for v in self[name].variables():
				low = v.lower()
				if name_map.get(low, {}).get(name):
					name_map[low][name].append(v)
				elif name_map.get(low):
					name_map[low].update({name: [v]})
				else:
					name_map[low] = {name: [v]}
		return name_map

	def _all_incl_vars(self):
		all_included = []
		for ds in self.datasets:
			for v in ds.variables():
				if not v in all_included:
					all_included.append(v)
		return all_included