import pandas as pd
import numpy as np
from scipy.stats.stats import _ttest_finish as get_pval 
from itertools import combinations
from collections import defaultdict
import quantipy as qp
import pandas as pd
import numpy as np
from operator import add, sub
from quantipy.core.view import View
from quantipy.core.cache import Cache
import copy

import time
from itertools import chain
# -------------------------------------------
# Class architecture idea: 
# Uses pandas for easy indexing but follows
# the Quantity() "matrix" rationale 
# -------------------------------------------
class Quantity(object):
	"""
	The Quantity object is the main Quantipy aggregation engine.

	Consists of a link's data matrix representation and sectional defintion
	of weight vector (wv), x-codes section (xsect) and y-codes section
	(ysect). The instance methods handle creation, retrieval and manipulation
	of the data input matrices and section definitions as well as the majority
	of statistical calculations.
	"""
	# -------------------------------------------------
	# Instance initialization
	# -------------------------------------------------
	def __init__(self, link, weight=None, use_meta=False, xsect_filter=None):
		super(Quantity, self).__init__()
		# Collect information on wv, xsect, ysect
		# and a possible list of rowfilter indicies
		self._uses_meta = use_meta
		self.d = link.stack[link.data_key].data
		self._dataidx = link.get_data().index
		if self._uses_meta:
			self.meta = link.get_meta()
		else:
			self.meta = None
		self._cache = link.get_cache()
		self._filter = link.filter
		self.x = link.x
		self.y = link.y
		self.w = weight if weight is not None else '@1'
		self.type = self._get_type() 
		self._squeezed = False
		self.xdef = self.ydef = None
		self.matrix = self._get_matrix()
		self.is_empty = False
		self.transposed = False
		self.factorized = None
		self.comb_x = None
		self.comb_y = None      
		self.miss_x = None
		self.miss_y = None
		self._has_x_margin = False
		self._has_y_margin = False

		# self.xsect_filter = xsect_filter
		# Define the data portion associated with the Q instance and
		# attach it via the link's get_data() method
		#if self.y == '@' or self.x == '@':
		#     var = self.x if not self.x == '@' else self.y
		#     portion = [weight, var]
		# else:
		#     portion = [weight, self.x, self.y]
		# self.d = link.stack[link.data_key].data[portion]
		# if not self.d.columns.is_unique:
		#     self.d.columns = [weight, self.x, self.y + '_']
		# # Set the instance object attributes, e.g. the sectional code
		# # definitions
		# self.xdef = None
		# self.ydef = None
		# if not self.x == '@':
		#     self.x_is_mc = True if self.d[self.x].dtype == 'object' else False
		# else:
		#     self.x_is_mc = False
		# if not self.y == '@':
		#     self.y_is_mc = True if self.d[self.y].dtype == 'object' else False
		# else:
		#     self.y_is_mc = False
		# self.result = None
		# self.aggname = None
		# self.is_empty = False
		# self._idx = link.get_data().index
		# self.matrix = self._get_matrix()
		# self.cbase = None
		# self.rbase = None

	def __repr__(self):
		if self.result is not None:
			return '%s' % (self.result)
		else:
			return ('%s, x: %s, xdef: %s y: %s, ydef: %s, w:%s'
					% (Quantity, self.x, self.xdef, self.y, self.ydef, self.w))

	# -------------------------------------------------
	# Matrix creation and retrievel
	# -------------------------------------------------

	def _get_type(self):
		# print '_check_array: MUST CHECK FOR META FIRST'
		if self.x in self.meta['masks'].keys():
			if self.meta['masks'][self.x]['type'] == 'array':
				return 'array'
		else:
			return 'simple'

	def _get_wv(self):
		"""
		Returns the weight vector of the matrix.
		"""
		return self.d[[self.w]].values

	def weight(self):
		"""
		"""
		w = np.repeat(self.wv, self.matrix.shape[1], axis=1)
		self.matrix = self.matrix * w[:, :, None]
		return None

	def unweight(self):
		"""
		"""
		self.matrix /= self.matrix
		return None

	def _get_total(self):
		"""
		Returns a vector of 1s for the matrix. 
		"""
		return self.d[['@1']].values

	def _copy(self):
		m_copy = np.empty_like(self.matrix)
		m_copy[:] = self.matrix
		c = copy.copy(self)
		c.matrix = m_copy
		return c

	def _switch_axes(self):
		if self.transposed:
			self.transposed = False
			self.matrix = self.matrix.swapaxes(1, 2)
		else:
			self.transposed = True
			self.matrix = self.matrix.swapaxes(2, 1)
		self.xdef, self.ydef = self.ydef, self.xdef
		self._x_indexers, self._y_indexers = self._y_indexers, self._x_indexers
		self.comb_x, self.comb_y = self.comb_y, self.comb_x
		self.miss_x, self.miss_y = self.miss_y, self.miss_x
		return self

	def exclude(self, codes, axis='x'):
		"""
		_missingfy - keep_codes=False, keep_base=False
		"""
		self._missingfy(codes, axis=axis, keep_base=False,
					   inplace=True)
	
	def restrict(self, codes, axis='x'):
		"""
		_missingfy - keep_codes=True, keep_base=True
		"""
		self._missingfy(codes, axis=axis, keep_codes=True, keep_base=True,
					   inplace=True)

	def _net_vec(self, codes, axis='x'):
		netted, idx = self._missingfy(codes=codes, axis=axis,
							  keep_codes=True, keep_base=True, indices=True,
							  inplace=False)
		if axis == 'y':
			netted._switch_axes()
		if not self.type == 'array':
			net_vec = np.nansum(netted.matrix[:, netted._x_indexers],
								axis=1, keepdims=True)
		else:
			net_vec = np.sum(netted.matrix[:, netted._x_indexers],
							 axis=1, keepdims=True)
		net_vec/=net_vec
		return net_vec, idx

	def _organize_combine_def(sel, combine_def, method_expand):
		"""
		Sanitize a combine instruction dictionary: names, codes, expands
		"""
		organized_def = []
		if not isinstance(combine_def[0], dict):
			combine_def = [{'net': combine_def, 'expand': method_expand}]
		for cb in combine_def:
			if 'expand' in cb.keys():
				expand = cb['expand']
				del cb['expand']
			else:
				expand = method_expand
			organized_def.append([cb.keys(), cb.values()[0], expand])
		return organized_def 

	def combine(self, codes, axis='x', expand=None):
		"""
		Build simple or logical net vectors, optionally keeping orginating codes. 
		"""
		# check validity and clean combine instructions
		if axis == 'y' and self.type == 'array':
			ni_err = 'Array mask element sections cannot be combined.'
			raise NotImplementedError(ni_err)
		elif axis == 'y' and self.y == '@':
			val_err = 'Total link: no y-codes to combine.'
			raise ValueError(val_err)
		comb_def = self._organize_combine_def(codes, expand)
		combines = []
		names = []
		# generate the net vectors (+ possible expanded originating codes)
		for no, comb in enumerate(comb_def):
			name, group, exp = comb[0], comb[1], comb[2]
			vec, idx = self._net_vec(group, axis=axis)
			if axis == 'y':
				self._switch_axes()
			m_idx = list(set(self._x_indexers) - set(idx))
			if exp is not None:
				if exp == 'after':
					names.extend(name)
					names.extend([c for c in group])
					combines.append(
						np.concatenate([vec, self.matrix[:, m_idx]], axis=1))
				else:
					names.extend([c for c in group])
					names.extend(name)
					combines.append(
						np.concatenate([self.matrix[:, m_idx], vec], axis=1))
			else:
				names.extend([name])
				combines.append(vec)
			if axis == 'y':
				self._switch_axes()
		# re-construct the combined data matrix and
		combines = np.concatenate(combines, axis=1)
		if axis == 'y':
			self._switch_axes()
		combined_matrix = np.concatenate([self.matrix[:, [0]],
										  combines], axis=1)
		if axis == 'y':
			combined_matrix = combined_matrix.swapaxes(1,2)
			self._switch_axes()
		# update the sectional information
		new_sect_def = range(0, len(codes))
		if axis == 'x':
			self.xdef = new_sect_def
			self._x_indexers = self._get_x_indexers()
			self.comb_x = names		
		else:
			self.ydef = new_sect_def
			self._y_indexers = self._get_y_indexers()
			self.comb_y = names
		self.matrix = combined_matrix

	def _missingfy(self, codes, axis='x', keep_codes=False, keep_base=True,
				  indices=False, inplace=True):
		"""
		Clean matrix from entries preserving or modifying the weight vector.

		Parameters
		----------
		codes : list
			A list of codes to be considered in cleaning.
		axis : {'x', 'y'}, default 'x'
			The axis to clean codes on. Refers to the Link object's x- and y-
			axes. 
		keep_codes : bool, default False
			Controls whether the passed codes are kept or erased from the
			Quantity matrix data entries.
		keep_base: bool, default=True
			Controls whether the weight vector is set to np.NaN alongside
			the x-section rows or remains unmodified.
		inplace : bool, default True
			Will overwrite self.matrix with the missingfied matrix by default.
			If ``False``, the method will return a new np.array with the
			modified entries.

		Returns
		-------
		self or numpy.array
			Either a new matrix is returned as numpy.array or the ``matrix``
			property is modified inplace.
		"""
		if inplace:
			missingfied = self
		else:
			missingfied = self._copy()
		if axis == 'y' and self.y == '@' and not self.type == 'array':
			return self
		elif axis == 'y' and self.type == 'array':
			ni_err = 'Cannot missingfy array mask element sections!'
			raise NotImplementedError(ni_err)
		else:
			if axis == 'y':
				missingfied._switch_axes()
			mis_ix = missingfied._get_drop_idx(codes, keep_codes) 
			mis_ix = [code + 1 for code in mis_ix]
			if mis_ix is not None:
				for ix in mis_ix:
					np.place(missingfied.matrix[:, ix],
							 missingfied.matrix[:, ix] > 0, np.NaN)
				if not keep_base:
					if axis == 'x':
						self.miss_x = True
					else:
						self.miss_y = True
					if self.type == 'array':
						mask = np.nansum(np.sum(missingfied.matrix,
												axis=1, keepdims=True),
										 axis=1, keepdims=True) > 0
					else:
						mask = np.nansum(np.sum(missingfied.matrix,
												axis=1, keepdims=False),
										 axis=1, keepdims=True) > 0
					missingfied.matrix[~mask] = np.NaN
				if axis == 'y':
					missingfied._switch_axes()
			if inplace:
				self.matrix = missingfied.matrix
			else:
				if indices:
					return missingfied, mis_ix
				else:
					return missingfied

	def margin(q, axis=None, effective=False):
		"""
		Calculate the matrix' (effective) row, column or total margins ("bases"). 
		
		Parameters
		----------
		axis : {None, 'x', 'y'}, default None
			The axis on that the margin is computed on. If None, will compute
			total.
		effective: bool, default False
			If True, will compute effective instead of the weighted/unweighted
			margins.

		Returns
		-------
		margin : np.array with same shape and orientation as originating axis. 
		"""
		if axis == 'x':
			if not effective:
				return np.nansum(q.matrix[:, :len(q.xdef)]*q.matrix[:,[6]],
								 axis=0, keepdims=True).T
			else:
				return (np.nansum(q.matrix[:, :len(q.xdef)]*q.matrix[:,[6]],
						axis=0, keepdims=True) ** 2 /
						np.nansum(q.matrix[:, :len(q.xdef)]*q.matrix[:,[6]]**2,
						axis=0, keepdims=True)).T

		elif axis == 'y':
			if not effective:
				return np.nansum(q.matrix[:, len(q.xdef)+1:-1]*q.matrix[:,[0]],
					   axis=0, keepdims=True)
			else:
				return (np.nansum(q.matrix[:, len(q.xdef)+1:-1]*q.matrix[:,[0]],
						axis=0, keepdims=True) ** 2 /
						np.nansum(q.matrix[:, len(q.xdef)+1:-1]**2*q.matrix[:,[0]],
						axis=0, keepdims=True))

		elif axis is None:
			if not effective:
				return np.nansum(q.matrix[:, [-1]], keepdims=True)
			else:
				return (np.nansum(q.matrix[:, [-1]], keepdims=True) **2/
						np.nansum(q.matrix[:, [-1]]**2, keepdims=True))
		else:
			raise ValueError('axis must be either "x", "y" or None, not "{}"'.format(axis))
	
	def _get_drop_idx(self, codes, keep):
		"""
		Produces a list of indices referring to the given input matrix's axes
		sections in order to erase data entries.

		Parameters
		----------
		codes : list
			Data codes that should be dropped from or kept in the matrix.
		keep : boolean
			Controls if the the passed code defintion is interpreted as
			"codes to keep" or "codes to drop".
		axis : {'x', 'y'}, default 'x'
			The axis to clean codes on. Refers to the Link object's x- and y-
			axes. 

		Returns
		-------
		drop_idx : list
			List of x section matrix indices.
		"""
		if codes is None:
			return None
		else:
			if keep:
				return [self.xdef.index(code) for code in self.xdef
						if code not in codes]
			else:
				return [self.xdef.index(code) for code in codes
						if code in self.xdef]

	def _factorize(self, axis='x', inplace=True):
		self.factorized = axis
		if inplace:
			factorized = self
		else:
			factorized = self._copy()
		if axis == 'y':
			self._switch_axes()
		f = np.atleast_3d(self.xdef)
		factorized.matrix[:, 1:, :] *= f
		if not inplace:
			return factorized
	
	def means(self, axis='x', margin=True, as_df=True):
		"""
		Calculates the mean of the incoming distribution across the given axis.
		"""
		self.current_agg = 'mean'
		self.result = self._means(axis)
		self._organize_margins(margin)
		if axis == 'y':
			self.result = self.result.T
		if as_df:
			return self.to_df()
		else:
			return self

	def _means(self, axis):
		self._factorize(axis=axis, inplace=True)
		fact_prod = np.nansum(self.matrix, axis=0)
		fact_prod_sum = np.nansum(fact_prod[1:, :], axis=0, keepdims=True)
		bases = fact_prod[[0], :]
		means = fact_prod_sum/bases
		if axis == 'y':
			self._switch_axes()
		self.matrix /= self.matrix
		return means

	def _dispersion(self, axis, measure='sd', return_mean=False):
		"""
		Extracts measures of dispersion from the incoming distribution of
		X vs. Y. Can return the arithm. mean by request as well. Dispersion
		measure supoorted are standard deviation, variance or coeffiecient of
		variation.
		"""
		means = self._means(axis)
		unbiased_n = self.margin(axis) - 1
		factorized = self._factorize(axis, inplace=False)
		diff_sqrt = np.nansum((factorized.matrix[:, 1:, :] - means)**2, axis=1)
		var = np.nansum(diff_sqrt/unbiased_n, axis=0, keepdims=True)
		#CHECK FOR INF/NAN VAR! GITHUB HAS FIX I THINK
		var[var <= 0] = np.NaN
		return var


		# var =  np.nansum(np.nansum(var, axis=0), axis=0)
		# print pd.DataFrame(var)


			# mat = self._unweight()
			# mat = self._factorize(mat, self.xdef)
			# mat = self._rdc_x(mat, self.xdef)

			# np.place(mat[:, 0],
			#          mat[:, 0] == 0, 1e-30)
			# ysects = self._by_ysect(mat, self.ydef)
			# var = np.array([(np.nansum(ymat[:, -1] *
			#                            (ymat[:, 0] - means[:, idx]) ** 2)) /
			#                 unbiased_n[:, idx]
			#                 for idx, ymat in enumerate(ysects)])
			# var[var <= 0] = np.NaN
			# if measure == 'sd':
			#     if return_mean:
			#         return means, np.sqrt(var).T
			#     else:
			#         return np.sqrt(var).T
			# elif measure == 'varc':
			#     if return_mean:
			#         return means, np.sqrt(var).T/means
			#     else:
			#         return np.sqrt(var).T/means
			# else:
			#     if return_mean:
			#         return means, var.T
			#     else:
			#         return var.T


	def _organize_margins(self, margin):
		if self._is_stats_result():
			if self.type == 'array' or self.y == '@':
				self._has_x_margins = False
				self._has_y_margin = False
			else:	
				if not margin:
					self.result = self.result[:, 1:]
					self._has_x_margin = False
					self._has_y_margin = False
				else:
					if self.factorized == 'x':
						self._has_x_margin = False
						self._has_y_margin = True
					else:
						self._has_x_margin = True
						self._has_y_margin = False
		elif self.current_agg == 'freq':
			if self.type == 'array' or self.y == '@':
				if not margin:
					self.result = self.result[1:, :]
					self._has_x_margin = False
					self._has_y_margin = False
				else:
					self._has_x_margin = True
					self._has_y_margin = False
			else:
				if not margin:
					self.result = self.result[1:, 1:]
					self._has_x_margin = False
					self._has_y_margin = False
				else:
					self._has_x_margin = True
					self._has_y_margin = True

		else:
			pass

			

		# bases = self._margin('y') if axis == 'x' else self._margin('x').T
		# factorized = self._factorize(axis)
		# projected = factorized._project_to_other_axis(axis)
		# factor_product_sum = np.nansum(projected.matrix, axis=0, keepdims=True)
		# return factor_product_sum/bases


	
	def _get_y_indexers(self):
		if self._squeezed or self.type == 'simple':
			if self.ydef is not None:
				return range(1, len(self.ydef)+1)
			else:
				return [1]
		else:
			y_indexers = []
			xdef_len = len(self.xdef)
			zero_based_ys = [idx for idx in xrange(0, xdef_len)]
			for y_no in xrange(0, len(self.ydef)):
				if y_no == 0:
					y_indexers.append(zero_based_ys)
				else:
					y_indexers.append([idx + y_no * xdef_len
									   for idx in zero_based_ys])
		return y_indexers

	def _get_x_indexers(self):
		if self._squeezed or self.type == 'simple':
			return range(1, len(self.xdef)+1)
		else:
			x_indexers = []
			upper_x_idx = len(self.ydef)
			start_x_idx = [len(self.xdef) * offset
						   for offset in range(0, upper_x_idx)]
			for x_no in range(0, len(self.xdef)):
				x_indexers.append([idx + x_no for idx in start_x_idx])
			return x_indexers

	def _squeeze_dummies(self):
		"""
		Reshape and replace initial 2D dummy matrix into its 3D equivalent.
		"""
		self.wv = self.matrix[:, [-1]]
		sects = []
		if self.type == 'array':
			x_sections = self._get_x_indexers()
			y_sections = self._get_y_indexers()
			y_total = np.nansum(self.matrix[:, x_sections], axis=1)
			y_total /= y_total
			y_total = y_total[:, None, :]
			for sect in y_sections:
				sect = self.matrix[:, sect]
				sects.append(sect)
			sects = np.dstack(sects)
			self._squeezed = True
			sects = np.concatenate([y_total, sects], axis=1)
			self.matrix = sects
			self._x_indexers = self._get_x_indexers()
			self._y_indexers = []
		elif self.type == 'simple':
			x = self.matrix[:, :len(self.xdef)+1]
			y = self.matrix[:, len(self.xdef)+1:-1]
			for i in range(0, y.shape[1]):
				sects.append(x * y[:, [i]])
			sects = np.dstack(sects)
			self._squeezed = True
			self.matrix = sects
			self._x_indexers = self._get_x_indexers()
			self._y_indexers = self._get_y_indexers()

	def _get_matrix(self):
		wv = self._cache.get_obj('weight_vectors', self.w)
		if wv is None:
			wv = self._get_wv()
			self._cache.set_obj('weight_vectors', self.w, wv)
		total = self._cache.get_obj('weight_vectors', '@1')
		if total is None:
			total = self._get_total()
			self._cache.set_obj('weight_vectors', '@1', total)
		if self.type == 'array':
			xm, self.xdef, self.ydef = self._dummyfy()
			self.matrix = np.concatenate((xm, wv), 1)
		else:
			if self.y == '@':
				xm, self.xdef = self._cache.get_obj('matrices', self.x)
				if xm is None:
					xm, self.xdef = self._dummyfy(self.x)
					self._cache.set_obj('matrices', self.x, (xm, self.xdef))
				self.ydef = None
				self.matrix = np.concatenate((total, xm, total, wv), 1)
			elif self.x == '@':
				xm, self.xdef = self._dummyfy(self.y)
				self.ydef = None
				self.matrix = np.concatenate((xm, wv), axis=1)
			else:
				xm, self.xdef = self._cache.get_obj('matrices', self.x)
				if xm is None:
					xm, self.xdef = self._dummyfy(self.x)
					self._cache.set_obj('matrices', self.x, (xm, self.xdef))
				ym, self.ydef = self._cache.get_obj('matrices', self.y)
				if ym is None:
					ym, self.ydef = self._dummyfy(self.y)
					self._cache.set_obj('matrices', self.y, (ym, self.ydef))
				self.matrix = np.concatenate((total, xm, total, ym, wv), 1)   
		self.matrix = self.matrix[self._dataidx]
		self.matrix = self._clean()
		self._squeeze_dummies()
		return self.matrix

		# if self.xsect_filter is not None:
		#     self.xsect_filter = self.xsect_filter
		#     self.matrix = self._outfilter_xsect()
		# if self.xsect_filter is None:
		#     self.matrix = self.matrix[self._idx]
		#     self.matrix = self._clean()
		# self.matrix = self.weight()
		# self.holds_data = True
		# if np.size(self.matrix) == 0:
		#     self.is_empty = True
		# return self.matrix
	
	def _clean(self):
		"""
		Drop empty sectional rows from the matrix.
		"""
		mat = self.matrix.copy()
		mat_indexer = np.expand_dims(self._dataidx, 1)
		if not self.type == 'array':
			xmask = (np.nansum(mat[:, 0:len(self.xdef)], axis=1) > 0)
			if self.ydef is not None:
				ymask = (np.nansum(mat[:, len(self.xdef)+2:-1], axis=1) > 0)
				self.idx_map =  np.concatenate(
					[np.expand_dims(xmask&ymask, 1), mat_indexer], axis=1)
				return mat[xmask & ymask]
			else:
				self.idx_map =  np.concatenate(
					[np.expand_dims(xmask, 1), mat_indexer], axis=1)
				return mat[xmask]
		else:
			mask = (np.nansum(mat[:, :-1], axis=1) > 0)
			return mat[mask]

	def _dummyfy(self, section=None):
		if section is not None:
			# i.e. Quantipy multicode data
			if self.d[section].dtype == 'object':
				section_data = self.d[section].str.get_dummies(';')
				if self._uses_meta:
					res_codes = self.get_response_codes(section)
					section_data = section_data.reindex(columns=res_codes)
					section_data.replace(np.NaN, 0, inplace=True)
				section_data.columns = [int(col) for col in section_data.columns]
				section_data.sort_index(axis=1, inplace=True)
			# i.e. Quantipy single-coded/numerical data
			else:
				section_data = pd.get_dummies(self.d[section])
				if self._uses_meta and not self._is_raw_numeric(section):
					res_codes = self.get_response_codes(section)
					section_data = section_data.reindex(columns=res_codes)
					section_data.replace(np.NaN, 0, inplace=True)
				section_data.rename(
					columns={
						col: int(col)
						if float(col).is_integer()
						else col
						for col in section_data.columns
					},
					inplace=True)
			return section_data.values, section_data.columns.tolist()   
		elif section is None and self.type == 'array':
			a_i = [i['source'].split('@')[-1] for i in
				   self.meta['masks'][self.x]['items']]
			a_res = self.get_response_codes(self.x)
			dummies = []
			for i in a_i:
				dummies.append(pd.get_dummies(self.d[i]).reindex(columns=a_res))
			a_data = pd.concat(dummies, axis=1) 
			return a_data.values, a_res, a_i

	def get_response_codes(self, var):
		if self.type == 'array':
			res = [c['value'] for c in self.meta['lib']['values'][var]]
		else:
			values = self.meta['columns'][var].get('values', None)
			if 'lib@values' in values:
				vals = values.split('@')[-1]
				values = self.meta['lib']['values'][vals]
			res = [c['value'] for c in values]
		return res
	
	def _is_raw_numeric(self, var):
		return self.meta['columns'][var]['type'] in ['int', 'float']

	def _set_bases(self, axis=None):
		"""
		Updates the the base size parameters of the Link instance. 
		"""
		if not axis: 
			self.rbase = self._margin(axis='x')
			self.cbase = self._margin(axis='y')
			self.tbase = self._margin(axis=None)
		elif axis == 'x':
			self.rbase = self._margin(axis='x')
		elif axis == 'y':
			self.cbase = self._margin(axis='y')
		elif axis == 'joint':
			self.tbase = self._margin(axis=None)
	
	def _has_total_first(self, axis):
		if axis == 'x':
			if not self.type == 'array' and not self._is_stats_result(): 
				return self.rbase.shape[0] == self.result.shape[0]
			elif self._is_stats_result and not self.type == 'array':
				return self.result.shape[0] > 1
			elif self._is_stats_result() and self.type == 'array':
				return False
			else:
				return self.result.shape[0] > len(self.xdef) 
		elif axis == 'y':
			if ((self.y == '@' or self.type == 'array') 
				and not self._is_stats_result()):
				return False
			elif self._is_stats_result and not self.type == 'array':
				return self.result.shape[1] > 1
			elif self._is_stats_result() and self.type == 'array':
				return False
			else:
				return self.cbase.shape[1] == self.result.shape[1]

	def _is_margin(self):
		return self.current_agg in ['tbase', 'cbase', 'rbase']

	def _is_stats_result(self):
		return self.current_agg in ['mean']

			
	def to_df(self):
		if self.current_agg == 'freq':
			if not self.comb_x:
				self.x_agg_vals = self.xdef
			else:
				self.x_agg_vals = self.comb_x
			if not self.comb_y:
				self.y_agg_vals = self.ydef
			else:
				self.y_agg_vals = self.comb_y
		elif self.current_agg == 'cbase':
			self.x_agg_vals = 'All'
			self.y_agg_vals = self.ydef
		elif self.current_agg == 'rbase':
			self.x_agg_vals = self.xdef
			self.y_agg_vals = 'All'
		elif self.current_agg in ['mean']:
			if self.factorized == 'x':
				self.x_agg_vals = self.current_agg
				self.y_agg_vals = self.ydef if not self.comb_y else self.comb_y
			else:
				self.x_agg_vals = self.xdef if not self.comb_x else self.comb_x
				self.y_agg_vals = self.current_agg 
		# can this made smarter WITHOUT 1000000 IF-ELSEs above?:
		if (self.current_agg in ['freq', 'cbase', 'rbase', 'mean'] \
			and not self.type == 'array'):
			if self.x == '@':
				self.x_agg_vals = '@'
			if self.y == '@':
				self.y_agg_vals = '@'
		df = pd.DataFrame(self.result)  
		idx, cols = self._make_multiindex()
		df.index = idx
		df.columns = cols
		self.result = df if self.type == 'simple' else df.T
		return self
			
	def _make_multiindex(self):
		x_grps = self.x_agg_vals
		y_grps = self.y_agg_vals
		if not isinstance(x_grps, list):
			x_grps = [x_grps]
		if not isinstance(y_grps, list):
			y_grps = [y_grps]
		if self._has_x_margin:
			x_grps = ['All'] + x_grps
		if self._has_y_margin:
			y_grps = ['All'] + y_grps
		if self.type == 'array':
			x_unit = y_unit = self.x
			x_names = ['Question', 'Values']
			y_names = ['Array', 'Questions']
		else:
			x_unit = self.x if not self.x == '@' else self.y
			y_unit = self.y if not self.y == '@' else self.x
			x_names = y_names = ['Question', 'Values']
		x = [x_unit, x_grps]
		y = [y_unit, y_grps]
		index = pd.MultiIndex.from_product(x, names=x_names)
		columns = pd.MultiIndex.from_product(y, names=y_names)
		return index, columns

	def count(self, axis=None, margin=True, as_df=True):
		self.current_agg = 'freq'
		self.result =  np.nansum(self.matrix, axis=0)
		self.cbase = self.result[[0], :]
		if self.type == 'simple':
			self.rbase = self.result[:, [0]]
		else:
			self.rbase = None
		self._organize_margins(margin)
		if as_df:
			self.to_df()
		return self

	def margin(self, axis='x', margin=True, effective=False, as_df=True):
		self.current_agg = 'All'
		if axis == 'x':
			return np.nansum(self.matrix, axis=0)[[0], :]
		else:
			return np.nansum(self.matrix, axis=0)[:, [0]]


	# def count(self, show='freq', margin=True, as_df=True):
	# 	"""
	# 	Method to produce a contigency tabulation of the distribution given
	# 	in the Quantity instance. Result is a multiindexed pandas.DataFrame
	# 	that mimics the output generated by the Pandas .pivot_table() method
	# 	extended to match a classical MR cross-tabulation.

	# 	Parameters
	# 	----------
	# 	show : {'freq', 'cbase', 'ebase', 'rbase'} default 'freq'
	# 		The counts aggregate to calculate. Defaults to a contigency table.
	# 	margin : bool, default True
	# 		Controls whether statistic(s) of the marginal distribution are
	# 		shown.
	# 	as_df : bool, default True
	# 		Determines if a pandas.DataFrame or a numpy.array is passed back
	# 		into the  object's ``result`` property.

	# 	Returns
	# 	-------
	# 	self
	# 		Passes a pandas.DataFrame or numpy.array of cell or base counts
	# 		to the ``result`` property.
	# 	"""
	# 	self.current_agg = show
	# 	if self.is_empty:
	# 		self.result = self._empty_calc()
	# 	else:
	# 		#self._set_bases()
	# 		self.cbase = self._margin('y')
	# 		self.rbase = self._margin('x')
	# 		if show == 'freq':
	# 			if self.type == 'array':
	# 				tab = np.vstack([self.cbase[:, 1:], self._cell_n()])
	# 				tab = np.hstack([self.rbase, tab])
	# 			else:
	# 				tab = np.vstack([self.cbase, self._cell_n()])
	# 			# if not self.is_array_mask:
	# 			#   if not self.y == '@':
	# 			#       tab = np.concatenate([self.cbase, self._cell_n()], axis=0)
	# 			#       tab[:,[0]] = self.rbase
	# 			#   else:
	# 			#       tab = np.concatenate([self.tbase, self._cell_n()], axis=0)
	# 			#   self.result = tab
	# 			# else:
	# 			#   tab = np.concatenate([self.cbase[:, 1:], self._cell_n()], axis=0)
	# 			#   tab = np.concatenate([self.rbase, tab], axis=1)
	# 			self.result = tab
	# 		elif show == 'cbase':
	# 			self.result = self.cbase
	# 		elif show == 'rbase':
	# 			self.result = self.rbase
	# 		elif show == 'ebase':
	# 			self.result = self._margin(axis='y', effective=True)
	# 		if self.y == '@':
	# 			pass
	# 			#if not show in ['ebase', 'rbase'] :
	# 			#    self.result = self.result[:, :-1]
	# 			#self.cbase = self.cbase[:, :-1]
	# 	if not margin:
	# 		self._drop_margin()
	# 	if as_df:
	# 		self.to_df()
	# 	return self



	def normalize(self, on='y'):
		"""
		Convert a raw cell count result to its percentage representation.

		Parameters
		----------
		on : {'y', 'x'}, default 'y'
			Defines the base to normalize the result on. ``'y'`` will
			produce column percentages, ``'x'`` will produce row
			percentages.

		Returns
		-------
		self
			Updates an count-based aggregation in the ``result`` property.
		"""
		if on == 'y':
			if self._has_total_first('y') or self.y == '@':
				base = self.cbase
			else:
				base = self.cbase[:, 1:]
		else:
			if self._has_total_first('x'):
				base = self.rbase
			else:
				base = self.rbase[1:, :]
		if isinstance(self.result, pd.DataFrame):
			if on == 'y':
				base = np.repeat(base, self.result.shape[0], axis=0)
			else:
				base = np.repeat(base, self.result.shape[1], axis=1)
		self.result = self.result / base * 100
		return self


# class Quantity(object):
#     """
#     The Quantity object is the main Quantipy aggregation engine.

#     Consists of a link's data matrix representation and sectional defintion
#     of weight vector (wv), x-codes section (xsect) and y-codes section
#     (ysect). The instance methods handle creation, retrieval and manipulation
#     of the data input matrices and section definitions as well as the majority
#     of statistical calculations.
#     """
#     # -------------------------------------------------
#     # Instance initialization
#     # -------------------------------------------------
#     def __init__(self, link, weight=None, use_meta=False, xsect_filter=None):
#         super(Quantity, self).__init__()
#         # Collect information on wv, xsect, ysect
#         # and a possible list of rowfilter indicies
#         self._uses_meta = use_meta
#         if self._uses_meta:
#             self.meta = link.get_meta()
#         else:
#             self.meta = None
#         self._cache = link.get_cache()
#         self._filter = link.filter
#         self.x = link.x
#         self.y = link.y
#         if weight is None:
#             weight = '@1'
#         self.w = weight
#         self.xsect_filter = xsect_filter
#         # Define the data portion associated with the Q instance and
#         # attach it via the link's get_data() method
#         if self.y == '@' or self.x == '@':
#             var = self.x if not self.x == '@' else self.y
#             portion = [weight, var]
#         else:
#             portion = [weight, self.x, self.y]
#         self.d = link.stack[link.data_key].data[portion]
#         if not self.d.columns.is_unique:
#             self.d.columns = [weight, self.x, self.y + '_']
#         # Set the instance object attributes, e.g. the sectional code
#         # definitions
#         self.xdef = None
#         self.ydef = None
#         if not self.x == '@':
#             self.x_is_mc = True if self.d[self.x].dtype == 'object' else False
#         else:
#             self.x_is_mc = False
#         if not self.y == '@':
#             self.y_is_mc = True if self.d[self.y].dtype == 'object' else False
#         else:
#             self.y_is_mc = False
#         self.result = None
#         self.aggname = None
#         self.is_empty = False
#         self._idx = link.get_data().index
#         self.matrix = self._get_matrix()
#         self.cbase = None
#         self.rbase = None

#     def __repr__(self):
#         if self.result is not None:
#             return '%s' % (self.result)
#         else:
#             return ('%s, x: %s, xdef: %s y: %s, ydef: %s, w:%s'
#                     % (Quantity, self.x, self.xdef, self.y, self.ydef, self.w))

#     # -------------------------------------------------
#     # Matrix creation and retrievel
#     # -------------------------------------------------
#     def _get_matrix(self):
#         wv = self._cache.get_obj('weight_vectors', self.w)
#         if wv is None:
#             wv = self.__get_wv()
#             self._cache.set_obj('weight_vectors', self.w, wv)
#         if self.y == '@':
#             xm, self.xdef = self._cache.get_obj('matrices', self.x)
#             if xm is None:
#                 xm, self.xdef = self._get_section(self.x)
#                 self._cache.set_obj('matrices', self.x, (xm, self.xdef))
#             self.ydef = None
#             self.matrix = np.concatenate((xm, wv), 1)
#         elif self.x == '@':
#             xm, self.xdef = self._get_section(self.y)
#             self.ydef = None
#             self.matrix = np.concatenate((xm, wv), axis=1)
#         else:
#             xm, self.xdef = self._cache.get_obj('matrices', self.x)
#             if xm is None:
#                 xm, self.xdef = self._get_section(self.x)
#                 self._cache.set_obj('matrices', self.x, (xm, self.xdef))
#             ym, self.ydef = self._cache.get_obj('matrices', self.y)
#             if ym is None:
#                 ym, self.ydef = self._get_section(self.y)
#                 self._cache.set_obj('matrices', self.y, (ym, self.ydef))
#             self.matrix = np.concatenate((xm, ym, wv), 1) 
#         if self.xsect_filter is not None:
#             self.xsect_filter = self.xsect_filter
#             self.matrix = self._outfilter_xsect()
#         if self.xsect_filter is None:
#             self.matrix = self.matrix[self._idx]
#             self.matrix = self._clean()
#         self.matrix = self.weight()
#         self.holds_data = True
#         if np.size(self.matrix) == 0:
#             self.is_empty = True
#         return self.matrix

#     def __get_wv(self):
#         """
#         Returns the weight vector of the matrix.
#         """
#         return self.d[[self.w]].values

#     def _get_section(self, section):
#         # i.e. Quantipy multicode data
#         if self.d[section].dtype == 'object':
#             section_data = self.d[section].str.get_dummies(';')
#             if self._uses_meta:
#                 res_codes = self.get_response_codes(section)
#                 section_data = section_data.reindex(columns=res_codes)
#                 section_data.replace(np.NaN, 0, inplace=True)
#             section_data.columns = [int(col) for col in section_data.columns]
#             section_data.sort_index(axis=1, inplace=True)
#         # i.e. Quantipy single-coded/numerical data
#         else:
#             section_data = pd.get_dummies(self.d[section])
#             if self._uses_meta and not self._is_raw_numeric(section):
#                 res_codes = self.get_response_codes(section)
#                 section_data = section_data.reindex(columns=res_codes)
#                 section_data.replace(np.NaN, 0, inplace=True)
#             section_data.rename(
#                 columns={
#                     col: int(col)
#                     if float(col).is_integer()
#                     else col
#                     for col in section_data.columns
#                 },
#                 inplace=True
#             )
#         return section_data.values, section_data.columns.tolist()

#     def _clean(self):
#         mat = self.matrix.copy()
#         xmask = (np.nansum(mat[:, :len(self.xdef)], axis=1) > 0)
#         if self.ydef is not None:
#             ymask = (np.nansum(mat[:, len(self.xdef):-1], axis=1) > 0)
#             return mat[xmask & ymask]
#         else:
#             return mat[xmask]

#     def _outfilter_xsect(self):
#         mat = self.matrix.copy()
#         mask = np.ones(mat.shape[0], dtype=bool)
#         mask[self.xsect_filter] = False
#         mat[mask == True, :len(self.xdef)] = np.NaN
#         return mat

#     # -------------------------------------------------
#     # Matrix manipulation and preparation
#     # -------------------------------------------------
#     def rescale(self, scaling):
#         """
#         Modify the object's ``xdef`` property reflecting new value defintions.

#         Parameters
#         ----------
#         scaling : dict
#             Mapping of old_code: new_code, given as of type int or float.

#         Returns
#         -------
#         self
#         """
#         clean_scaling = {old_code: new_code for old_code, new_code
#                          in scaling.items()
#                          if old_code in self.xdef}
#         xdef_ref = [clean_scaling[code] if code in clean_scaling.keys()
#                     else code for code in self.xdef]
#         self.xdef = xdef_ref
#         return self

#     def _missingfy(self, codes, keep_codes=False, keep_base=True,
#                   inplace=True):
#         """
#         Clean matrix from entries preserving or modifying the weight vector.

#         Parameters
#         ----------
#         codes : list
#             A list of codes to be considered in cleaning.
#         keep_codes : bool, default False
#             Controls whether the passed codes are kept or erased from the
#             Q matrix data entries.
#         keep_base: bool, default=True
#             Controls whether the weight vector is set to np.NaN alongside
#             the x-section rows or remains unmodified.
#         inplace : bool, default True
#             Will overwrite self.matrix with the missingfied matrix by default.
#             If ``False``, the method will return a new np.array with the
#             modified entries.

#         Returns
#         -------
#         self or numpy.array
#             Either a new matrix is returned as numpy.array or the ``matrix``
#             property is modified inplace.
#         """
#         mis_ix = self._get_drop_idx(codes, keep_codes)
#         if mis_ix is not None:
#             matrix = self.matrix.copy()
#             for ix in mis_ix:
#                 np.place(matrix[:, ix], matrix[:, ix] > 0, np.NaN)
#             if not keep_base:
#                 wv_mask = (np.nansum(matrix[:, :len(self.xdef)], axis=1) > 0)
#                 matrix[(~wv_mask), [-1]] = np.NaN
#         if inplace:
#             self.matrix = matrix
#             return self
#         else:
#             return matrix

#     def weight(self):
#         """
#         Multiplies 1-entries of the Q matrix with the weight vector.
#         """
#         matrix = self.matrix.copy()
#         self.matrix[:, :len(self.xdef)] = (
#             self.matrix[:, :len(self.xdef)] * self.matrix[:, [-1]])
#         return self.matrix

#     def _unweight(self):
#         """
#         Returns a copy of the (weighted) input matrix without a
#         weight vector multiplied x section.

#         Parameters
#         ----------
#         mat : np.array
#             1/0 representation of a Link data defintiton.
#             Produced by tools.view.agg.get_matrix().
#         xdef : list
#             x section defintion of the input matrix.
#             Produced by tools.view.agg.get_matrix().

#         Returns
#         -------
#         mat : np.array
#             Unweighted copy of the input matrix with regard to the x section.
#         """
#         matrix = self.matrix.copy()
#         matrix[:, :len(self.xdef)] = (matrix[:, :len(self.xdef)] /
#                                       matrix[:, [-1]])
#         return matrix

#     def _get_drop_idx(self, codes, keep):
#         """
#         Produces a list of indices refering to the input matrix's x section in
#         order to erase data entries.

#         Parameters
#         ----------
#         codes : list
#             Data codes that should be dropped from or kept in the matrix.
#         keep : boolean
#             Controls if the the passed code defintion is interpreted as
#             "codes to keep" or "codes to drop".

#         Returns
#         -------
#         drop_idx : list
#             List of x section matrix indices.
#         """
#         if codes is None:
#             return None
#         else:
#             if keep:
#                 return [self.xdef.index(code) for code in self.xdef
#                         if code not in codes]
#             else:
#                 return [self.xdef.index(code) for code in codes
#                         if code in self.xdef]

#     def _reset(self):
#         """
#         Restore the Quantity data matrix without any rows filtered from it.
#         """
#         empty = self.is_empty
#         self.xsect_filter = None
#         self.matrix = self._get_matrix()
#         self.is_empty = empty
#         return None

#     @staticmethod
#     def _factorize(mat, xdef):
#         matrix = mat.copy()
#         matrix[:, :len(xdef)] = (matrix[:, :len(xdef)] * xdef)
#         return matrix

#     @staticmethod
#     def _rdc_x(mat, xdef):
#         matrix = mat.copy()
#         redx = np.expand_dims(
#             np.nansum(matrix[:, :len(xdef)], axis=1), 1)
#         matrix = np.concatenate(
#             (redx, matrix[:, len(xdef):]), axis=1)
#         return matrix

#     @staticmethod
#     def _by_ysect(mat, ydef):
#         if ydef is None:
#             ydef = [0]
#         len_sects = mat.shape[1]
#         start = len_sects - len(ydef)
#         if ydef is not None and not ydef == [0]:
#             ysects = xrange(start-1, len_sects)
#             return [mat[mat[:, y] > 0] for y in ysects]
#         else:
#             return [mat]

#     # -------------------------------------------------
#     # Extraction of statistical measures
#     # -------------------------------------------------
#     def describe(self, show='summary', margin=True, as_df=True):
#         """
#         Method to produce a numerical summary of the distribution given
#         in the Quantity instance. Result is a multiindexed pandas.DataFrame
#         that mimics the output generated by the Pandas .describe() method
#         extended to match a classical MR cross-tabulation.

#         Parameters
#         ----------
#         show : {'summary', 'mean', 'median', 'var', 'stddev', 'varcoeff',
#                 'sem', 'max', 'min'}, default 'summary'
#             The measure to calculate. Default to a summary output of the most
#             important sample statistics.
#         margin : bool, default True
#             Controls whether statistic(s) of the marginal distribution are
#             shown.
#         as_df : bool, default True
#             Determines if a pandas.DataFrame or a numpy.array is passed back
#             into the  object's ``result`` property.

#         Returns
#         -------
#         self
#             Passes a pandas.DataFrame or numpy.array of numerical summary
#             statistic(s) to the ``result`` property.
#         """
#         self.aggname = show
#         if self.is_empty and not self._uses_meta:
#             self.result = self._empty_calc()
#         else:
#             if show == 'summary':
#                 self.result = np.concatenate((
#                     self._col_n(),
#                     self._mean(),
#                     self._dispersion(),
#                     self._min(),
#                     self._percentile(0.25),
#                     self._percentile(0.5),
#                     self._percentile(0.75),
#                     self._max()),
#                     axis=0)
#             elif show == 'mean':
#                 self.result = self._mean()
#             elif show == 'median':
#                 self.result = self._percentile(0.5)
#             elif show == 'var':
#                 self.result = self._dispersion(measure='var')
#             elif show == 'stddev':
#                 self.result = self._dispersion()
#             elif show == 'varcoeff':
#                 self.result = self._dispersion(measure='varc')
#             elif show == 'sem':
#                 self.result = self._dispersion()/np.sqrt(self._col_n())
#             elif show == 'max':
#                 self.result = self._max()
#             elif show == 'min':
#                 self.result = self._min()
#         if show == 'summary':
#             self.aggname = ['All', 'mean', 'stddev', 'min',
#                             '25%', 'median', '75%', 'max']
#         else:
#             self.aggname = show
#         self._set_bases()
#         if not margin and not self.y == '@':
#             self.result = self.result[:, :-1]
#         if as_df:
#             self.to_df()
#         return self

#     def count(self, show='freq', margin=True, as_df=True):
#         """
#         Method to produce a contigency tabulation of the distribution given
#         in the Quantity instance. Result is a multiindexed pandas.DataFrame
#         that mimics the output generated by the Pandas .pivot_table() method
#         extended to match a classical MR cross-tabulation.

#         Parameters
#         ----------
#         show : {'freq', 'cbase', 'ebase', 'rbase'} default 'freq'
#             The counts aggregate to calculate. Defaults to a contigency table.
#         margin : bool, default True
#             Controls whether statistic(s) of the marginal distribution are
#             shown.
#         as_df : bool, default True
#             Determines if a pandas.DataFrame or a numpy.array is passed back
#             into the  object's ``result`` property.

#         Returns
#         -------
#         self
#             Passes a pandas.DataFrame or numpy.array of cell or base counts
#             to the ``result`` property.
#         """
#         self.aggname = show
#         self._set_bases()
#         if self.is_empty and not self._uses_meta:
#             self.result = self._empty_calc()
#         else:
#             if show == 'freq':
#                 self.result = np.concatenate(
#                     (self.cbase, self._cell_n()), axis=0)
#             elif show == 'cbase':
#                 self.result = self.cbase
#             elif show == 'rbase':
#                 self.result = self.rbase
#             elif show == 'ebase':
#                 self.result = self._effective_n()
#         if not margin:
#             self._drop_margin()
#         if as_df:
#             self.to_df()
#         return self

#     def combine(self, group=None, op=None, op_only=False, margin=True,
#                 as_df=True):
#         """
#         Group codes into new categories.

#         Can produce multiple combinations at once, optionally calculating sums
#         of or differences between them.

#         Parameters
#         ----------
#         group : list of int or list of dict mapping str: list of int
#             A list of int implies a single group to be computed from the
#             passed codes. A list of dict will produce multiple groups named
#             after the keys, combining the list of codes defined as the
#             values.
#         op : dict of str: (operator, [str, str])
#             Defines an optional calculation on the combined code groups. The
#             result is named after the key, operator can either be ``sum`` or
#             ``add`` from the ``operator`` library and must involve two of
#             group names defined by ``group``.
#         op_only : bool, default False
#             If ``op`` is passed, this controls whether the calculation
#             result is shown exclusively or if it will be appended to the
#             group results.
#         margin : bool, default True
#             Controls whether results for the marginal distribution are
#             shown.
#         as_df : bool, default True
#             Determines if a pandas.DataFrame or a numpy.array is passed back
#             into the  object's ``result`` property.

#         Returns
#         -------
#         self
#             Passes a pandas.DataFrame or numpy.array of grouped counts
#             to the ``result`` property.

#         .. note::
#             Code combinations are factoring in multi-coded data, i.e.
#             a group built from a multiple choice question does *not* count
#             the associated codes multiple times but one time when *qualified*
#             for the group definiton.
#         """
#         if group is not None and isinstance(group[0], dict):
#             names = [c.keys()[0] for c in group]
#             combs = [c.values()[0] for c in group]
#             self.result = np.concatenate([self._net(codes=comb)
#                                           for comb in combs], axis=0)
#             if op is not None:
#                 gidx = [names.index(name)
#                         for name in names if name in op.values()[0][1]]
#                 opsres = op.values()[0][0](self.result[gidx[0], :],
#                                            self.result[gidx[1], :])
#                 self.result = np.concatenate((self.result,
#                                               np.expand_dims(opsres, 0)),
#                                              axis=0)
#         else:
#             op = None
#             op_only = False
#             self.result = self._net(codes=group)
#             names = 'net'
#         if op is not None:
#             if op_only:
#                 self.result = np.expand_dims(self.result[-1, :], 0)
#                 self.aggname = op.keys()
#             else:
#                 self.aggname = names + op.keys()
#         else:
#             self.aggname = names
#         self._set_bases(combine=True)
#         if not margin and not self.y == '@':
#             self.result = self.result[:, :-1]
#         if as_df:
#             self.to_df()
#         return self

#     def _set_bases(self, combine=False):
#         if not combine:
#             self.rbase = np.concatenate(
#                 (self._total_n(), self._row_n()), axis=0)
#             self.cbase = self._col_n()
#         else:
#             self.rbase = np.expand_dims(self.result[:, -1], 1)
#             if self.xsect_filter is not None:
#                 self._reset()
#             self.cbase = self._col_n()

#     def _drop_margin(self):
#         if self.result.shape == (1, 1):
#             return self.result
#         else:
#             if self.result.shape[0] == 1:
#                 self.result = self.result[:, :-1]
#             elif self.result.shape[1] == 1:
#                 self.result = self.result[1:, :]
#             else:
#                 self.result = self.result[1:, :-1]
#             return self

#     def _net(self, codes, raw=False):
#         if self.is_empty and not self._uses_meta:
#             net = self._empty_calc()
#         else:
#             orgm = self.matrix
#             if codes is not None:
#                 self._missingfy(codes=codes, keep_codes=True, keep_base=True)
#             self.matrix = self._rdc_x(self.matrix, self.xdef)
#             net = self._group_n()
#             self.matrix = orgm
#         return net

#     def _group_n(self):
#         ysects = self._by_ysect(self.matrix, self.ydef)
#         return np.expand_dims(
#             [np.nansum(mat[:, 0] / mat[:, 0] * mat[:, -1], axis=0)
#              for mat in ysects], 1).T

#     def _empty_calc(self):
#         """
#         """
#         if self.aggname == 'freq':
#             return np.zeros((2, 2))
#         elif self.aggname == 'summary':
#             return np.zeros((8, 2))
#         elif self.aggname == 'mean_stddev':
#             return np.zeros((1, 1)), np.zeros((1, 1))
#         else:
#             return np.zeros((1, 2))

#     def _cases(self):
#         """
#         Extracts the unweighted no. of cases for the the columns of X vs. Y.

#         Returns
#         -------
#         cases : np.array
#             Numpy array storing the number of cases per column.
#         """
#         ysects = self._by_ysect(self.matrix, self.ydef)
#         return np.expand_dims(
#             [ymat.shape[0] for ymat in ysects], 1).T

#     def _cell_n(self):
#         """
#         Extracts raw cell frequencies for the cross-tabulation of X vs. Y.

#         Returns
#         -------
#         cellns : np.array
#             Numpy array storing the absolute cell values per category.
#         """
#         ysects = self._by_ysect(self.matrix, self.ydef)
#         return np.array([np.nansum(ymat[:, :len(self.xdef)], axis=0)
#                          for ymat in ysects]).T

#     def _col_n(self):
#         """
#         Extracts the sample size for the columns of X vs. Y.

#         Returns
#         -------
#         colns : np.array
#             Numpy array storing column n values.
#         """
#         ysects = self._by_ysect(self.matrix, self.ydef)
#         # return np.expand_dims(
#         #                     [np.nansum(ymat[:,-1]) for ymat in ysects], 1).T
#         return np.expand_dims(
#                 [np.nansum((np.sum(ymat[:, :len(self.xdef)], axis=1) /
#                             np.sum(ymat[:, :len(self.xdef)], axis=1) *
#                             ymat[:, -1]))
#                  for ymat in ysects], 1).T

#     def _row_n(self):
#         """
#         Extracts the sample size for the rows of X vs. Y.

#         Returns
#         -------
#         rowns : np.array
#             Numpy array storing row n values.
#         """
#         return np.expand_dims(
#             np.nansum(self.matrix[:, :len(self.xdef)], axis=0), 1)

#     def _total_n(self):
#         """
#         Extracts the total sample size for the X/Y distribution.

#         Returns
#         -------
#         totn : np.array
#             Numpy array storing the total n value.
#         """
#         return np.expand_dims(np.nansum(self.matrix[:, [-1]], axis=0), 1)

#     def _effective_n(self):
#         """
#         Extracts the effective sample size for the columns of X vs. Y.
#         The effective sample size is a measure that corrects for balancing
#         bias introduced by the weighting process.

#         Returns
#         -------
#         effns : np.array
#             Numpy array storing effective n values.
#         """
#         ysects = self._by_ysect(self.matrix, self.ydef)
#         return np.expand_dims(
#             [np.nansum(ymat[:, [-1]])**2 /
#              np.nansum((ymat[:, [-1]])**2)
#              for ymat in ysects], 1).T

#     def _mean(self):
#         """
#         Extracts the arithm. mean from the incoming distribution of X vs. Y.

#         Returns
#         -------
#         means : np.array
#             Numpy array storing mean values.
#         """
#         mat = self._factorize(self.matrix, self.xdef)
#         mat = self._rdc_x(mat, self.xdef)
#         ysects = self._by_ysect(mat, self.ydef)
#         return np.expand_dims([np.nansum(ymat[:, 0] /
#                                np.nansum(ymat[:, -1]))
#                                for ymat in ysects], 1).T

#     def _max(self):
#         """
#         Extracts the maximum from the incoming distribution of X vs. Y.

#         Returns
#         -------
#         maxs : np.array
#             Numpy array storing maximum values.
#         """
#         mat = self._unweight()
#         mat = self._factorize(mat, self.xdef)
#         mat = self._rdc_x(mat, self.xdef)
#         if 0 not in self.xdef:
#             np.place(mat[:, 0], mat[:, 0] == 0, np.NaN)
#         ysects = self._by_ysect(mat, self.ydef)
#         return np.expand_dims([np.nanmax(mat[:, 0])
#                               if mat.shape[0] > 0 else 0
#                               for mat in ysects], 1).T

#     def _min(self):
#         """
#         Extracts the minimum from the incoming distribution of X vs. Y.

#         Returns
#         -------
#         mins : np.array
#             Numpy array storing minimum values.
#         """
#         mat = self._unweight()
#         mat = self._factorize(mat, self.xdef)
#         mat = self._rdc_x(mat, self.xdef)
#         if 0 not in self.xdef:
#             np.place(mat[:, 0], mat[:, 0] == 0, np.NaN)
#         ysects = self._by_ysect(mat, self.ydef)
#         return np.expand_dims([np.nanmin(mat[:, 0])
#                               if mat.shape[0] > 0 else 0
#                               for mat in ysects], 1).T

#     def _percentile(self, perc=0.5):
#         """
#         Computes percentiles from the incoming distribution of X vs.Y and the
#         requested percentile value. The implementation mirrors the algorithm
#         used in SPSS Dimensions and the EXAMINE procedure in SPSS Statistics.
#         It based on the percentile defintion #6 (adjusted for survey weights)
#         in:
#         Hyndman, Rob J. and Fan, Yanan (1996) -
#         "Sample Quantiles in Statistical Packages",
#         The American Statistician, 50, No. 4, 361-365.

#         Parameters
#         ----------
#         perc : float, default=0.5
#             Defines the percentile to be computed. Defaults to 0.5,
#             the sample median.

#         Returns
#         -------
#         percs : np.array
#             Numpy array storing percentile values.
#         """
#         percs = []
#         mat = self._unweight()
#         mat = self._factorize(mat, self.xdef)
#         mat = self._rdc_x(mat, self.xdef)
#         mat[:, -1] = np.nan_to_num(mat[:, -1])
#         ysects = self._by_ysect(mat, self.ydef)
#         for mat in ysects:
#             if mat.shape[0] == 1:
#                 percs.append(mat[0, 0])
#             elif mat.shape[0] == 0:
#                 percs.append(0)
#             else:
#                 sortidx = np.argsort(mat[:, 0])
#                 mat = np.take(mat, sortidx, axis=0)
#                 wsum = np.sum(mat[:, -1], axis=0)
#                 wcsum = np.cumsum(mat[:, -1], axis=0)
#                 k = (wsum+1)*perc
#                 if wcsum[0] > k:
#                     wcsum_k = wcsum[0]
#                     percs.append(mat[0, 0])
#                 elif wcsum[-1] <= k:
#                     percs.append(mat[-1, 0])
#                 else:
#                     wcsum_k = wcsum[wcsum <= k][-1]
#                     p_k_idx = np.searchsorted(np.ndarray.flatten(wcsum), wcsum_k)
#                     p_k = mat[p_k_idx, 0]
#                     p_k1 = mat[p_k_idx+1, 0]
#                     w_k1 = mat[p_k_idx+1, -1]
#                     excess = k - wcsum_k
#                     if excess >= 1.0:
#                         percs.append(p_k1)
#                     else:
#                         if w_k1 >= 1.0:
#                             percs.append((1.0-excess)*p_k + excess*p_k1)
#                         else:
#                             percs.append((1.0-excess/w_k1)*p_k +
#                                          (excess/w_k1)*p_k1)

#         return np.expand_dims(percs, 1).T

	# def _dispersion(self, measure='sd', return_mean=False):
	#     """
	#     Extracts measures of dispersion from the incoming distribution of
	#     X vs. Y. Can return the arithm. mean by request as well. Dispersion
	#     measure supoorted are standard deviation, variance or coeffiecient of
	#     variation.
	#     """
	#     means = self._mean()
	#     unbiased_n = self._col_n() - 1
	#     mat = self._unweight()
	#     mat = self._factorize(mat, self.xdef)
	#     mat = self._rdc_x(mat, self.xdef)
	#     np.place(mat[:, 0],
	#              mat[:, 0] == 0, 1e-30)
	#     ysects = self._by_ysect(mat, self.ydef)
	#     var = np.array([(np.nansum(ymat[:, -1] *
	#                                (ymat[:, 0] - means[:, idx]) ** 2)) /
	#                     unbiased_n[:, idx]
	#                     for idx, ymat in enumerate(ysects)])
	#     var[var <= 0] = np.NaN
	#     if measure == 'sd':
	#         if return_mean:
	#             return means, np.sqrt(var).T
	#         else:
	#             return np.sqrt(var).T
	#     elif measure == 'varc':
	#         if return_mean:
	#             return means, np.sqrt(var).T/means
	#         else:
	#             return np.sqrt(var).T/means
	#     else:
	#         if return_mean:
	#             return means, var.T
	#         else:
	#             return var.T

#     # -------------------------------------------------
#     # Post-processing of calculation results
#     # -------------------------------------------------
#     def normalize(self, on='col'):
#         """
#         Convert a raw cell count result to its percentage representation.

#         Parameters
#         ----------
#         on : {'col', 'row'}, default 'col'
#             Defines the base to normalize the result on. ``'col'`` will
#             produce column percentages, ``'row'`` will produce row
#             percentages.

#         Returns
#         -------
#         self
#             Updates an count-based aggregation in the ``result`` property.
#         """
#         if 'base' in self.aggname or self.is_empty:
#             pass
#         else:
#             if on == 'col':
#                 if isinstance(self.result, pd.DataFrame):
#                     base = np.repeat(self.cbase, self.result.values.shape[0],
#                                      axis=0)
#                 else:
#                     base = self.cbase
#                 if self.result.shape[1] == self.cbase.shape[1]:
#                     self.result = self.result / base*100
#                 else:
#                     self.result = self.result / base[:, :-1] * 100
#             elif on == 'row':
#                 if isinstance(self.result, pd.DataFrame):
#                     base = np.repeat(self.rbase, self.result.values.shape[1],
#                                      axis=1)
#                 else:
#                     base = self.rbase
#                 if self.result.shape[0] == self.rbase.shape[0]:
#                     self.result = self.result / base * 100
#                 else:
#                     self.result = self.result / base[1:, :] * 100
#         self.cbase = None
#         self.rbase = None
#         return self

#     def to_df(self, row_val=None, col_val=None):
#         """
#         Transform a numpy.array of an aggregation into its DataFrame version.

#         Will use the current numpy.array aggregation result found in the
#         ``result`` property of the object instance and convert it to a
#         Quantipy-styled pandas.DataFrame. The DataFrame representation of an
#         aggregation is multiindexed following a Question-Values convention
#         on both the index and column axis.

#         Parameters
#         ----------
#         row_val, col_val : str or list of str, optional
#             If provided, the "Value" level of the multiindex will
#             use it instead of the name of the default name of the aggregation.
#             The length of the passed elements must match the the length of the
#             inner-most "Values" index.

#         Returns
#         -------
#         self
#             Updates the aggregation output stored in the ``result`` property.
#         """
#         x_mi, y_mi = self._make_mi(row_val, col_val)
#         if self.x == '@':
#             self.result = pd.DataFrame(self.result, index=x_mi,
#                                        columns=y_mi).T
#         else:
#             self.result = pd.DataFrame(self.result, index=x_mi, columns=y_mi)
#         return self

#     def _make_mi(self, row_val, col_val):
#         names = ['Question', 'Values']
#         aggname = self.aggname if row_val is None else row_val
#         if not isinstance(aggname, list):
#             aggname = [aggname]
#         xn = self.x if not self.x == '@' else self.y
#         yn = self.y if not self.y == '@' else self.x
#         if row_val is not None and col_val is not None:
#             xv = row_val
#             yv = col_val
#         else:
#             if self.is_empty and not self._uses_meta:
#                 xv = yv = ['None']
#             else:
#                 xv = self.xdef if self.xdef is not None else ['@']
#                 yv = self.ydef if self.ydef is not None else ['@']
#             fully_coded = ['freq']
#             transpose = ['rbase']
#             if self.aggname not in fully_coded:
#                 if self.aggname not in transpose:
#                     xv = aggname
#                 else:
#                     yv = aggname
#             if len(xv) < self.result.shape[0]:
#                 xv = ['All'] + xv
#             if len(yv) < self.result.shape[1]:
#                 yv = yv + ['All']
#         x = [xn, xv]
#         y = [yn, yv]
#         return (pd.MultiIndex.from_product(x, names=names),
#                 pd.MultiIndex.from_product(y, names=names))

#     # -------------------------------------------------
#     # meta data helpers and handlers
#     # (these should be moved to the dataset class when it's)
#     # implemented
#     # -------------------------------------------------    
#     def get_response_codes(self, var):
#         values = self.meta['columns'][var].get('values', None)
#         if 'lib@values' in values:
#             vals = values.split('@')[-1]
#             values = self.meta['lib']['values'][vals]
#         res = [c['value'] for c in values]
#         return res

#     def _is_array_mask(self, var):
#       if var in meta['masks'].keys():
#           if self.meta['masks'][var]['type'] == 'array':
#               return True
#       else:
#           return False
		  
#     def _is_raw_numeric(self, var):
#         return self.meta['columns'][var]['type'] in ['int', 'float']



class Test(object):
	"""
	The Quantipy Test object is a defined by a Link and the view name notation
	string of a counts or means view. All auxiliary figures needed to arrive
	at the test results are computed inside the instance of the object.
	"""    
	def __init__(self, link, view_name_notation):
		super(Test, self).__init__()
		# Infer whether a mean or proportion test is being performed
		view = link[view_name_notation]
		if view.meta()['agg']['method'] == 'descriptives':
			self.metric = 'means'
		else:
			self.metric = 'proportions'
		self.invalid = None
		self.parameters = None
		self.mimic = None
		self.level = None
		# Calculate the required baseline measures for the test using the
		# Q instance
		self.Quantity = qp.Quantity(link, view.weights())
		if view.missing():
			self.Quantity = self.Quantity._missingfy(view.missing(),
													keep_base=False)
		self.cbases = view.cbases[:,:-1]
		self.rbases = view.rbases
		self.tbase = view.cbases[:,-1]

		if self.metric == 'means':
			self.values, self.sd = self.Quantity._dispersion(return_mean=True)
			self.values, self.sd = self.values[:,:-1], self.sd[:,:-1]
		else:
			self.values = view.dataframe.values.copy()
		# Set information about the incoming aggregation
		# to be able to route correctly through the algorithms
		# and re-construct a Quantipy-indexed pd.DataFrame
		self.is_weighted = view.meta()['agg']['is_weighted']
		self.x = view.meta()['x']['name']
		self.xdef = view.dataframe.index.get_level_values(1).tolist()
		self.y = view.meta()['y']['name']
		self.ydef = view.dataframe.columns.get_level_values(1).tolist()
		self.ypairs = list(combinations(self.ydef, 2))
		self.y_is_multi = view.meta()['y']['is_multi']
		self.multiindex = (view.dataframe.index, view.dataframe.columns)

	def __repr__(self):
		return ('%s, test metric: %s, parameters: %s, '
				'mimicked: %s, level: %s ')\
				% (Test, self.metric, self.parameters, self.mimic, self.level)

	def set_params(self, level='mid', mimic='Dim', testtype='pooled',
				   use_ebase=True, ovlp_correc=True, cwi_filter=False):
		"""
		Sets the test algorithm parameters and defines the type of test.

		This method sets the test's global parameters and derives the
		necessary measures for the computation of the test statistic.
		The default values correspond to the SPSS Dimensions Column Tests
		algorithms that control for bias introduced by weighting and
		overlapping samples in the column pairs of multi-coded questions.
		
		.. note:: The Dimensions implementation uses variance pooling.

		Parameters
		----------
		level : str or float, default='mid'
			The level of significance given either as per 'low' = 0.1,
			'mid' = 0.05, 'high' = 0.01 or as specific float, e.g. 0.15.
		mimic : str, default='Dim'
			Will instruct the mimicking of a software specific test.
		testtype : str, default='pooled'
			Global definition of the tests.
		use_ebase : bool, default=True
			If True, will use the effective sample sizes instead of the
			the simple weighted ones when testing a weighted aggregation.
		ovlp_correc : bool, default=True
			If True, will consider and correct for respondent overlap when
			testing between multi-coded column pairs.
		cwi_filter : bool, default=False
			If True, will check an incoming count aggregation for cells that
			fall below a treshhold comparison aggregation that assumes counts
			to be independent.

		Returns
		-------
		self
		"""
		# Check if the aggregation is non-empty
		# and that there are >1 populated columns
		if np.nansum(self.values) == 0 or len(self.ydef) == 1:
			self.invalid = True            
			self.mimic = mimic
			self.comparevalue, self.level = self._convert_level(level)
		else:
			# Set global test algorithm parameters
			self.invalid = False
			# Deactived for now, access to user-defined test setup will be
			# made availabe at later stage!
			# valid_types = ['pooled', 'unpooled']
			# if testtype not in valid_types:
			#     raise ValueError('Test type unknown: "%s". Select from: %s\n'
			#                      % (testtype, valid_types))
			valid_mimics = ['Dim', 'askia']
			if mimic not in valid_mimics:
				raise ValueError('Failed to mimic: "%s". Select from: %s\n'
								 % (mimic, valid_mimics))
			else:
				self.mimic = mimic

			#
			if self.mimic == 'askia':
				self.parameters = {
								   'testtype': 'unpooled',
								   'use_ebase': False,
								   'ovlp_correc': False,
								   'cwi_filter': True
								  }
			elif self.mimic == 'Dim':
				self.parameters = {
								   'testtype': 'pooled',
								   'use_ebase': True,
								   'ovlp_correc': True,
								   'cwi_filter': False
								  }
			self.level = level
			self.comparevalue, self.level = self._convert_level(level)
			# Get value differnces between column pairings
			if self.metric == 'means':
				self.valdiffs = np.array(
					[m1 - m2 for m1, m2 in combinations(self.values[0], 2)])
			if self.metric == 'proportions':
				props = (self.values / self.cbases).T
				self.valdiffs = np.array([p1 - p2 for p1, p2
										  in combinations(props, 2)]).T
			# Set test specific measures as properties of the instance
			if self.metric == 'proportions' and cwi_filter:
				self.values = self._cwi()
			if use_ebase and self.is_weighted:
				self.ebases = self.Quantity.count(show='ebase', margin=False,
												  as_df=False).result
			else:
				self.ebases = self.cbases
			if self.y_is_multi and self.parameters['ovlp_correc']:
				self.overlap = self._overlap()
			else:
				self.overlap = np.zeros(self.valdiffs.shape)
		return self

	# -------------------------------------------------
	# Main algorithm methods to compute test statistics
	# -------------------------------------------------
	def run(self):
		"""
		Performs the testing algorithm and creates an output pd.DataFrame.

		The output is indexed according to Quantipy's Questions->Values
		convention. Significant results between columns are presented as
		lists of integer y-axis codes where the column with the higher value
		is holding the codes of the columns with the lower values. NaN is
		indicating that a cell is not holding any sig. higher values
		compared to the others.
		"""
		if not self.invalid:
			sigs = self.get_sig()
			return self._output(sigs)
		else:
			return self._empty_output()

	def get_sig(self):
		"""
		TODO: implement returning tstats only.
		"""
		stat = self.get_statistic()
		stat = self._convert_statistic(stat)
		if self.metric == 'means':
			stat = pd.DataFrame(stat, index=self.ypairs).T
			diffs = pd.DataFrame(self.valdiffs, index=self.ypairs).T
		elif self.metric == 'proportions':
			stat = pd.DataFrame(stat, columns=self.ypairs)
			diffs = pd.DataFrame(self.valdiffs, columns=self.ypairs)
		if self.mimic == 'Dim':
			return diffs[(diffs != 0) & (stat < self.comparevalue)]
		elif self.mimic == 'askia':
			return diffs[(diffs != 0) & (stat > self.comparevalue)]

	def get_statistic(self):
		"""
		Returns the test statistic of the algorithm.
		"""
		return self.valdiffs / self.get_se()

	def get_se(self):
		"""
		Compute the standard error (se) estimate of the tested metric.

		The calculation of the se is defined by the parameters of the setup.
		The main difference is the handling of variances. **unpooled**
		implicitly assumes variance inhomogenity between the column pairing's
		samples. **pooled** treats variances effectively as equal.
		"""
		if self.metric == 'means':
			if self.parameters['testtype'] == 'unpooled':
				return self._se_mean_unpooled()
			elif self.parameters['testtype'] == 'pooled':
				return self._se_mean_pooled()
		elif self.metric == 'proportions':
			if self.parameters['testtype'] == 'unpooled':
				return self._se_prop_unpooled()
			if self.parameters['testtype'] == 'pooled':
				return self._se_prop_pooled()

	# -------------------------------------------------
	# Conversion methods for levels and statistics
	# -------------------------------------------------
	def _convert_statistic(self, teststat):
		"""
		Convert test statistics to match the decision rule of the test logic.

		Either transforms to p-values or returns the absolute value of the
		statistic, depending on the decision rule of the test.
		This is used to mimic other software packages as some tests'
		decision rules check test-statistic against pre-defined treshholds
		while others check sig. level against p-value.
		"""
		if self.mimic == 'Dim':
			ebases_pairs = [eb1 + eb2 for eb1, eb2
							in combinations(self.ebases[0], 2)]
			dof = ebases_pairs - self.overlap - 2
			dof[dof <= 1] = np.NaN
			return get_pval(dof, teststat)[1]
		elif self.mimic == 'askia':
			return abs(teststat)

	def _convert_level(self, level):
		"""
		Determines the comparison value for the test's decision rule.

		Checks whether the level of test is a string that defines low, medium,
		or high significance or an "actual" level of significance and
		converts it to a comparison level/significance level tuple.
		This is used to mimic other software packages as some test's
		decision rules check test-statistic against pre-defined treshholds
		while others check sig. level against p-value.
		"""
		if isinstance(level, (str, unicode)):
			if level == 'low':
				if self.mimic == 'Dim':
					comparevalue = siglevel = 0.10
				elif self.mimic == 'askia':
					comparevalue = 1.65
					siglevel = 0.10
			elif level == 'mid':
				if self.mimic == 'Dim':
					comparevalue = siglevel = 0.05
				elif self.mimic == 'askia':
					comparevalue = 1.96
					siglevel = 0.05
			elif level == 'high':
				if self.mimic == 'Dim':
					comparevalue = siglevel = 0.01
				elif self.mimic == 'askia':
					comparevalue = 2.576
					siglevel = 0.01
		else:
			if self.mimic == 'Dim':
				comparevalue = siglevel = level
			elif self.mimic == 'askia':
				comparevalue = 1.65
				siglevel = 0.10

		return comparevalue, siglevel

	# -------------------------------------------------
	# Standard error estimates calculation methods
	# -------------------------------------------------
	def _se_prop_unpooled(self):
		"""
		Estimated standard errors of prop. diff. (unpool. var.) per col. pair.
		"""
		props = self.values/self.cbases
		unp_sd = ((props*(1-props))/self.cbases).T
		return np.array([np.sqrt(cat1 + cat2)
						 for cat1, cat2 in combinations(unp_sd, 2)]).T

	def _se_mean_unpooled(self):
		"""
		Estimated standard errors of mean diff. (unpool. var.) per col. pair.
		"""
		sd_base_ratio = self.sd / self.cbases
		return np.array([np.sqrt(sd_b_r1 + sd_b_r2)
						 for sd_b_r1, sd_b_r2
						 in combinations(sd_base_ratio[0], 2)])

	def _se_prop_pooled(self):
		"""
		Estimated standard errors of prop. diff. (pooled var.) per col. pair.

		Controlling for effective base sizes and overlap responses is
		supported and applied as defined by the test's parameters setup.
		"""
		ebases_correc_pairs = np.array([1 / x + 1 / y
										for x, y
										in combinations(self.ebases[0], 2)])

		if self.y_is_multi and self.parameters['ovlp_correc']:
			ovlp_correc_pairs = ((2 * self.overlap) /
								 [x * y for x, y
								  in combinations(self.ebases[0], 2)])
		else:
			ovlp_correc_pairs = self.overlap

		counts_sum_pairs = np.array(
			[c1 + c2 for c1, c2 in combinations(self.values.T, 2)])
		bases_sum_pairs = np.expand_dims(
			[b1 + b2 for b1, b2 in combinations(self.cbases[0], 2)], 1)
		pooled_props = (counts_sum_pairs/bases_sum_pairs).T
		return (np.sqrt(pooled_props * (1 - pooled_props) *
				(np.array(ebases_correc_pairs - ovlp_correc_pairs))))

	def _se_mean_pooled(self):
		"""
		Estimated standard errors of mean diff. (pooled var.) per col. pair.

		Controlling for effective base sizes and overlap responses is
		supported and applied as defined by the test's parameters setup.
		"""
		ssw_base_ratios = self._sum_sq_w(base_ratio=True)
		enum = np.nan_to_num((self.sd ** 2) * (self.cbases-1))
		denom = self.cbases-ssw_base_ratios

		enum_pairs = np.array([enum1 + enum2
							   for enum1, enum2
							   in combinations(enum[0], 2)])
		denom_pairs = np.array([denom1 + denom2
								for denom1, denom2
								in combinations(denom[0], 2)])

		ebases_correc_pairs = np.array([1/x + 1/y
										for x, y
										in combinations(self.ebases[0], 2)])

		if self.y_is_multi and self.parameters['ovlp_correc']:
			ovlp_correc_pairs = ((2*self.overlap) /
								 [x * y for x, y
								  in combinations(self.ebases[0], 2)])
		else:
			ovlp_correc_pairs = self.overlap

		return (np.sqrt((enum_pairs/denom_pairs) *
						(ebases_correc_pairs - ovlp_correc_pairs)))

	# -------------------------------------------------
	# Specific algorithm values & test option measures 
	# -------------------------------------------------
	def _sum_sq_w(self, base_ratio=True):
		"""
		"""
		ysects = self.Quantity._by_ysect(self.Quantity.matrix,
										 self.Quantity.ydef)
		ssw = np.array([np.nansum((ymat[:, [-1]])**2)
						for ymat in ysects])
		if base_ratio:
			return ssw[:-1]/self.cbases
		else:
			return np.array(ssw[:-1])

	def _cwi(self, threshold=5, as_df=False):
		"""
		Derives the count distribution assuming independence between columns.
		"""
		c_col_n = self.cbases
		c_cell_n = self.values
		t_col_n = self.tbase
		if self.rbases.shape[1] > 1:
			t_cell_n = self.rbases[1:,:]
		else:
			t_cell_n = self.rbases[0]
		np.place(t_col_n, t_col_n == 0, np.NaN)
		np.place(t_cell_n, t_cell_n == 0, np.NaN)
		np.place(c_col_n, c_col_n == 0, np.NaN)
		np.place(c_cell_n, c_cell_n == 0, np.NaN)
		cwi = (t_cell_n * c_col_n) / t_col_n
		cwi[cwi < threshold] = np.NaN
		if as_df:
			return pd.DataFrame(c_cell_n + cwi - cwi,
								index=self.xdef, columns=self.ydef)
		else:
			return c_cell_n + cwi - cwi

	def _overlap(self):
		mat = self.Quantity.matrix.copy()
		mat = mat[:, [-1]] * mat[:, len(self.Quantity.xdef):-1]
		mat[mat == 0] = np.NaN

		if self.parameters['use_ebase']:
			# Overlap computation when effective base is being used
			w_sum_sq_paired = np.hstack(
				[np.nansum(mat[:, [col1]] + mat[:, [col2]], axis=0)**2
				 for col1, col2 in combinations(xrange(0, mat.shape[1]), 2)])
			w_sq_sum_paired = np.hstack(
				[np.nansum(mat[:, [col1]]**2 + mat[:, [col2]]**2)
				 for col1, col2 in combinations(xrange(0, mat.shape[1]), 2)])
			return np.nan_to_num((w_sum_sq_paired/w_sq_sum_paired)/2)
		else:
			# Overlap with simple weighted/unweighted base size
			ovlp = np.array(
				[np.nansum(mat[:, [col1]] + mat[:, [col2]])
				 for col1, col2
				 in combinations(xrange(0, mat.shape[1]), 2)])
			return np.nan_to_num(ovlp/2)

	# -------------------------------------------------
	# Output creation 
	# -------------------------------------------------
	def _output(self, sigs):
		res = {col: {row: [] for row in xrange(0, len(self.xdef))}
			   for col in self.ydef}
		for col, val in sigs.iteritems():
			for ix, v in enumerate(val.values):
				if v > 0:
					res[col[0]][ix].append(col[1])
				if v < 0:
					res[col[1]][ix].append(col[0])
		# The str casting in the following two lines should be abandoned at a
		# later stage to increase performance. ExcelPainter will require an
		# update for this.
		sigtest = pd.DataFrame(res).applymap(lambda x: str(x))
		sigtest.replace('[]', np.NaN, inplace=True)
		sigtest.index = self.multiindex[0]
		sigtest.columns = self.multiindex[1]

		return sigtest

	def _empty_output(self):
		"""
		"""
		values = self.values
		values[:] = np.NaN
		if values.shape == (1, 1) or values.shape == (1, 0):
			values = [np.NaN]
		return  pd.DataFrame(values,
							 index=self.multiindex[0],
							 columns=self.multiindex[1])