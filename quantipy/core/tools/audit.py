#!/usr/bin/python
# -*- coding: utf-8 -*-

import quantipy as qp
from quantipy.core.tools.qp_decorators import *

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
		not_included = [d for d in ds if not
						any(dataset.name == d for dataset in self.datasets)]
		if not_included:
			raise ValueError('{} is not included.'.format(not_included))
		datasets = [d for d in self.datasets if d.name in ds]
		return datasets[0] if len(datasets) == 1 else datasets

    # ------------------------------------------------------------------------
    # file i/o
    # ------------------------------------------------------------------------

	def _load_ds(self, name):
		path_json = '{}/{}.json'.format(self.path, name)
		path_csv = '{}/{}.csv'.format(self.path, name)
		dataset = qp.DataSet(name, self._dimensions_comp)
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
			self._get_ds_names()
		return None

	def _get_ds_names(self):
		for ds in self.datasets:
			if not ds.name in self.ds_names:
				self.ds_names.append(ds.name)
			else:
				raise ValueError('{} is already in Audit.'.format(ds.name))
		return None

	def add_path(self, path):
		"""
		Define the path attribute.
		"""
		self.path = path
		return None

	def save(self, name, suffix='_audit'):
		"""
		Save all included DataSet instances.
		"""
		for ds in self.datasets:
			pass



