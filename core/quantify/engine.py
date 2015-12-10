import pandas as pd
import numpy as np
from scipy.stats.stats import _ttest_finish as get_pval
from itertools import combinations, chain
from collections import defaultdict
import quantipy as qp
import pandas as pd
import numpy as np
from operator import add, sub, mul, div
from quantipy.core.view import View
from quantipy.core.cache import Cache
from quantipy.core.tools.view.logic import (
    has_any, has_all, has_count,
    not_any, not_all, not_count,
    is_lt, is_ne, is_gt,
    is_le, is_eq, is_ge,
    union, intersection, get_logic_index)

import copy
import time


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
    def __init__(self, link, weight=None, use_meta=False):
        super(Quantity, self).__init__()
        # Collect information on wv, x- and y-section
        # and a possible list of rowfilter indicies
        self._uses_meta = use_meta
        self.d = link.stack[link.data_key].data
        self._dataidx = link.get_data().index
        if self._uses_meta:
            self.meta = link.get_meta()
        else:
            self.meta = None
        self._cache = link.get_cache()
        self.f = link.filter
        self.x = link.x
        self.y = link.y
        self.w = weight if weight is not None else '@1'
        self.type = self._get_type()
        self._squeezed = False
        self.idx_map = None
        self.xdef = self.ydef = None
        self.matrix = self._get_matrix()
        self.is_empty = False
        self.switched = False
        self.factorized = None
        self.result = None
        self.logical_conditions = []
        self.cbase = self.rbase = None
        self.comb_x = self.comb_y = None
        self.miss_x = self.miss_y = None
        self._has_x_margin = self._has_y_margin = False

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
        Multiply the dummy indicator entries with the weight vector.
        """
        self.matrix *=  np.atleast_3d(self.wv)
        return None

    def unweight(self):
        """
        Remove any weighting by dividing the matrix by itself.
        """
        self.matrix /= self.matrix
        return None

    def _get_total(self):
        """
        Return a vector of 1s for the matrix. 
        """
        return self.d[['@1']].values

    def _copy(self):
        """
        Copy the Quantity instance, i.e. its data matrix, into a new object.
        """
        m_copy = np.empty_like(self.matrix)
        m_copy[:] = self.matrix
        c = copy.copy(self)
        c.matrix = m_copy
        return c

    def _get_response_codes(self, var):
        """
        """
        if self.type == 'array':
            res = [c['value'] for c in self.meta['lib']['values'][var]]
        else:
            values = self.meta['columns'][var].get('values', None)
            if 'lib@values' in values:
                vals = values.split('@')[-1]
                values = self.meta['lib']['values'][vals]
            res = [c['value'] for c in values]
        return res

    def _switch_axes(self):
        """
        """
        if self.switched:
            self.switched = False
            self.matrix = self.matrix.swapaxes(1, 2)
        else:
            self.switched = True
            self.matrix = self.matrix.swapaxes(2, 1)
        self.xdef, self.ydef = self.ydef, self.xdef
        self._x_indexers, self._y_indexers = self._y_indexers, self._x_indexers
        self.comb_x, self.comb_y = self.comb_y, self.comb_x
        self.miss_x, self.miss_y = self.miss_y, self.miss_x
        return self

    def rescale(self, scaling):
        """
        Modify the object's ``xdef`` property reflecting new value defintions.

        Parameters
        ----------
        scaling : dict
            Mapping of old_code: new_code, given as of type int or float.

        Returns
        -------
        self
        """
        proper_scaling = {old_code: new_code for old_code, new_code
                         in scaling.items() if old_code in self.xdef}
        xdef_ref = [proper_scaling[code] if code in proper_scaling.keys()
                    else code for code in self.xdef]
        self.xdef = xdef_ref
        return self

    def exclude(self, codes, axis='x'):
        """
        Wrapper for _missingfy(...keep_codes=False, ..., keep_base=False, ...)
        Excludes specified codes from aggregation.
        """
        self._missingfy(codes, axis=axis, keep_base=False, inplace=True)
    
    def restrict(self, codes, axis='x'):
        """
        Wrapper for _missingfy(...keep_codes=True, ..., keep_base=True, ...)
        Restrict the data matrix entires to contain the specified codes only.
        """
        self._missingfy(codes, axis=axis, keep_codes=True, keep_base=True,
                        inplace=True)

    def filter(self, condition, keep_base=True, inplace=False):
        """
        Use a Quantipy conditional expression to filter the data matrix entires.
        """
        if inplace:
            filtered = self
        else:
            filtered = self._copy()
        qualified_rows = self._get_logic_qualifiers(condition)
        valid_rows = self.idx_map[self.idx_map[:, 0] == 1][:, 1]
        filter_idx = np.in1d(valid_rows, qualified_rows)
        if keep_base:
            filtered.matrix[~filter_idx, 1:, :] = np.NaN
        else:
            filtered.matrix[~filter_idx, :, :] = np.NaN
        if not inplace:
            return filtered

    def _get_logic_qualifiers(self, condition):
        if not isinstance(condition, dict):
            column = self.x
            logic = condition
        else:
            column = condition.keys()[0]
            logic = condition.values()[0]
        idx, logical_expression = get_logic_index(self.d[column], logic)
        logical_expression = logical_expression.split(':')[0]
        if not column == self.x:
            logical_expression = logical_expression.replace('x[', column+'[')
        self.logical_conditions.append(logical_expression)
        return idx

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

    def group(self, groups, axis='x', expand=None):
        """
        Build simple or logical net vectors, optionally keeping orginating codes. 
        """
        # check validity and clean combine instructions
        if axis == 'y' and self.type == 'array':
            ni_err = 'Array mask element sections cannot be combined.'
            raise NotImplementedError(ni_err)
        elif axis == 'y' and self.y == '@':
            val_err = 'Total link has no y-axis codes to combine.'
            raise ValueError(val_err)
        grp_def = self._organize_grp_def(groups, expand)
        combines = []
        names = []
        # generate the net vectors (+ possible expanded originating codes)
        for grp in grp_def:
            name, group, exp, logical = grp[0], grp[1], grp[2], grp[3]
            if not logical:
                vec, idx = self._grp_vec(group, axis=axis)
            else:
                vec = self._logical_grp_vec(group)
            if axis == 'y':
                self._switch_axes()
            if exp is not None:
                m_idx = list(set(self._x_indexers) - set(idx))
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
                names.extend(name)
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
            combined_matrix = combined_matrix.swapaxes(1, 2)
            self._switch_axes()
        # update the sectional information
        new_sect_def = range(0, len(groups))
        if axis == 'x':
            self.xdef = new_sect_def
            self._x_indexers = self._get_x_indexers()
            self.comb_x = names     
        else:
            self.ydef = new_sect_def
            self._y_indexers = self._get_y_indexers()
            self.comb_y = names
        self.matrix = combined_matrix

    def _grp_vec(self, codes, axis='x'):
        netted, idx = self._missingfy(codes=codes, axis=axis,
                                      keep_codes=True, keep_base=True,
                                      indices=True, inplace=False)
        if axis == 'y':
            netted._switch_axes()
        if not self.type == 'array':
            net_vec = np.nansum(netted.matrix[:, netted._x_indexers],
                                axis=1, keepdims=True)
        else:
            net_vec = np.sum(netted.matrix[:, netted._x_indexers],
                             axis=1, keepdims=True)
        net_vec /= net_vec
        return net_vec, idx

    def _logical_grp_vec(self, condition):
        """
        Create net vector of qualified rows based on passed condition.
        """
        self.filter(condition=condition, inplace=True) 
        net_vec = np.nansum(self.matrix[:, self._x_indexers], axis=1,
                            keepdims=True)
        net_vec /= net_vec
        return net_vec
    
    def _grp_type(self, grp_def):
        if isinstance(grp_def, list):
            if not isinstance(grp_def[0], (int, float)):
                return 'block'
            else:
                return 'list'
        elif isinstance(grp_def, tuple):
            return 'logical'
        elif isinstance(grp_def, dict):
            return 'wildcard'

    def _organize_grp_def(self, grp_def, method_expand):
        """
        Sanitize a combine instruction list (of dicts): names, codes, expands.
        """
        organized_def = []
        if not self._grp_type(grp_def) == 'block': 
            grp_def = [{'net': grp_def, 'expand': method_expand}]
        for grp in grp_def:
            if self._grp_type(grp.values()[0]) in ['logical', 'wildcard']:
                if 'expand' in grp.keys():
                    del grp['expand']
                expand = None
                logical = True
            else:
                if 'expand' in grp.keys():
                    grp = copy.deepcopy(grp)
                    expand = grp['expand']
                    del grp['expand']
                else:
                    expand = method_expand
                logical = False
            organized_def.append([grp.keys(), grp.values()[0], expand, logical])
        return organized_def 


    def calc(self, expression, result_only=False):
        """
        Compute (simple) aggregation level arithmetics.
        """
        if not self.result:
            raise ValueError('No aggregation to base calculation on.')
        else:


    def count(self, axis=None, margin=True, as_df=True):
        """
        Count entries over all cells or per axis margin.
        """
        if axis is None:
            self.current_agg = 'freq'
        elif axis == 'x':
            self.current_agg = 'cbase'
        elif axis == 'y':
            self.current_agg = 'rbase'
        if not self.w == '@1':
            self.weight()
        counts = np.nansum(self.matrix, axis=0)
        self.cbase = counts[[0], :]
        if self.type == 'simple':
            self.rbase = counts[:, [0]]
        else:
            self.rbase = None
        if axis is None:
            self.result = counts
        elif axis == 'x':
            self.result = counts[[0], :]
        elif axis == 'y':
            self.result = counts[:, [0]]
        self._organize_margins(margin)
        if as_df:
            self.to_df()
        self.unweight()
        return self
    
    def _effective_n(self, axis=None, margin=True):
        self.weight()
        effective = (np.nansum(self.matrix, axis=0)**2 /
                     np.nansum(self.matrix**2, axis=0))
        self.unweight()
        start_on = 0 if margin else 1
        if axis is None:
            return effective[start_on:, start_on:]
        elif axis == 'x':
            return effective[[0], start_on:]
        else:
            return effective[start_on:, [0]]

    def summarize(self, stat='summary', axis='x', margin=True, as_df=True):
        """
        Calculate distribution statistics across the given axis.
        """
        self.current_agg = stat
        if stat == 'summary':
            stddev,  mean, base = self._dispersion(axis, measure='sd',
                                                   _return_mean=True,
                                                   _return_base=True)
            self.result = np.concatenate([
                base, mean, stddev,
                self._min(axis),
                self._percentile(perc=0.25),
                self._percentile(perc=0.50),
                self._percentile(perc=0.75),
                self._max(axis)
                ], axis=0)
        elif stat == 'mean':
            self.result = self._means(axis)
        elif stat == 'var':
            self.result = self._dispersion(axis, measure='var')
        elif stat == 'stddev':
            self.result = self._dispersion(axis, measure='sd')
        elif stat == 'sem':
            self.result = self._dispersion(axis, measure='sem')
        elif stat == 'varcoeff':
            self.result = self._dispersion(axis, measure='varcoeff')
        elif stat == 'min':
            self.result = self._min(axis)
        elif stat == 'lower_q':
            self.result = self._percentile(perc=0.25)
        elif stat == 'median':
            self.result = self._percentile(perc=0.5)
        elif stat == 'upper_q':
            self.result = self._percentile(perc=0.75)
        elif stat == 'max':
            self.result = self._max(axis)
        self._organize_margins(margin)
        if as_df:
            self.to_df()
        return self

    def means(self, axis='x', margin=True, as_df=True):
        """
        Calculates the mean of the incoming distribution across the given axis.
        """
        self.current_agg = 'mean'
        self.result = self._means(axis)
        self._organize_margins(margin)
        if as_df:
            return self.to_df()
        else:
            return self

    def _factorize(self, axis='x', inplace=True):
        self.factorized = axis
        if inplace:
            factorized = self
        else:
            factorized = self._copy()
        if axis == 'y':
            factorized._switch_axes()
        np.copyto(factorized.matrix[:, 1:, :],
                  np.atleast_3d(factorized.xdef),
                  where=factorized.matrix[:, 1:, :]>0)
        if not inplace:
            return factorized

    def _means(self, axis, _return_base=False):
        fact = self._factorize(axis=axis, inplace=False)
        if not self.w == '@1':
            fact.weight()
        fact_prod = np.nansum(fact.matrix, axis=0)
        fact_prod_sum = np.nansum(fact_prod[1:, :], axis=0, keepdims=True)
        bases = fact_prod[[0], :]
        means = fact_prod_sum/bases
        if axis == 'y':
            self._switch_axes()
            means = means.T
            bases = bases.T
        if _return_base:
            return means, bases
        else:
            return means

    def _dispersion(self, axis='x', measure='sd', _return_mean=False,
                    _return_base=False):
        """
        Extracts measures of dispersion from the incoming distribution of
        X vs. Y. Can return the arithm. mean by request as well. Dispersion
        measure supoorted are standard deviation, variance or coeffiecient of
        variation.
        """
        means, bases = self._means(axis, _return_base=True)
        unbiased_n = bases - 1
        self.unweight()
        factorized = self._factorize(axis, inplace=False)
        # factorized.matrix[:, 1:, :] = (factorized.matrix[:, 1:, :] - means)**2
        factorized.matrix[:, 1:] -= means
        factorized.matrix[:, 1:] *= factorized.matrix[:, 1:, :]        
        # IS THIS NEEDED ANY LONGER? TEST WITH RESCALE {value: 0, ...}
        # np.place(mat[:, 0],
        #  mat[:, 0] == 0, 1e-30)
        if not self.w == '@1':
            factorized.weight()
        diff_sqrt = np.nansum(factorized.matrix[:, 1:], axis=1)    
        disp = np.nansum(diff_sqrt/unbiased_n, axis=0, keepdims=True)
        disp[disp <= 0] = np.NaN
        disp[np.isinf(disp)] = np.NaN
        if measure == 'sd':
            disp = np.sqrt(disp)
        elif measure == 'sem':
            disp = np.sqrt(disp) / np.sqrt((unbiased_n + 1))
        elif measure == 'varcoeff':
            disp = np.sqrt(disp) / means
        self.unweight()
        if _return_mean and _return_base:
            return disp, means, bases
        elif _return_mean:
            return disp, means
        elif _return_base:
            return disp, bases
        else:
            return disp

    def _max(self, axis='x'):
        factorized = self._factorize(axis, inplace=False)
        vals = np.nansum(factorized.matrix[:, 1:, :], axis=1)
        return np.nanmax(vals, axis=0, keepdims=True)

    def _min(self, axis='x'):
        factorized = self._factorize(axis, inplace=False)
        vals = np.nansum(factorized.matrix[:, 1:, :], axis=1)
        np.place(vals, vals == 0, np.inf)
        return np.nanmin(vals, axis=0, keepdims=True)


    def _percentile(self, axis='x', perc=0.5):
        """
        Computes percentiles from the incoming distribution of X vs.Y and the
        requested percentile value. The implementation mirrors the algorithm
        used in SPSS Dimensions and the EXAMINE procedure in SPSS Statistics.
        It based on the percentile defintion #6 (adjusted for survey weights)
        in:
        Hyndman, Rob J. and Fan, Yanan (1996) -
        "Sample Quantiles in Statistical Packages",
        The American Statistician, 50, No. 4, 361-365.

        Parameters
        ----------
        perc : float, default=0.5
            Defines the percentile to be computed. Defaults to 0.5,
            the sample median.

        Returns
        -------
        percs : np.array
            Numpy array storing percentile values.
        """
        percs = []
        factorized = self._factorize(axis, inplace=False)
        vals = np.nansum(np.nansum(factorized.matrix[:, 1:, :], axis=1,
                                   keepdims=True), axis=1)
        weights = (vals/vals)*self.wv
        for shape_i in range(0, vals.shape[1]):
            iter_weights = weights[:, shape_i]
            iter_vals = vals[:, shape_i]
            mask = ~np.isnan(iter_weights)
            iter_weights = iter_weights[mask]
            iter_vals = iter_vals[mask]
            sorter = np.argsort(iter_vals)
            iter_vals = np.take(iter_vals, sorter)
            iter_weights = np.take(iter_weights, sorter)
            iter_wsum = np.nansum(iter_weights, axis=0)
            iter_wcsum = np.cumsum(iter_weights, axis=0)
            k = (iter_wsum + 1.0) * perc
            if iter_wcsum[0] > k:
                wcsum_k = iter_wcsum[0]
                percs.append(iter_vals[0])
            elif iter_wcsum[-1] <= k:
                percs.append(iter_vals[-1])
            else:
                wcsum_k = iter_wcsum[iter_wcsum <= k][-1]
                p_k_idx = np.searchsorted(np.ndarray.flatten(iter_wcsum), wcsum_k)
                p_k = iter_vals[p_k_idx]
                p_k1 = iter_vals[p_k_idx+1]
                w_k1 = iter_weights[p_k_idx+1]
                excess = k - wcsum_k
                if excess >= 1.0:
                    percs.append(p_k1)
                else:
                    if w_k1 >= 1.0:
                        percs.append((1.0-excess)*p_k + excess*p_k1)
                    else:
                        percs.append((1.0-(excess/w_k1))*p_k +
                                     (excess/w_k1)*p_k1)
        return np.array(percs)[None, :]

    def _organize_margins(self, margin):
        if self._is_stats_result() or self._is_margin():
            if self.type == 'array' or self.y == '@':
                self._has_x_margins = False
                self._has_y_margin = False
            else:   
                if not margin:
                    if self.current_agg == 'rbase':
                        self.result = self.result[1:, :]
                    else:
                        self.result = self.result[:, 1:]
                        self._has_x_margin = False
                        self._has_y_margin = False
                else:
                    if self.factorized == 'x' or self.current_agg == 'cbase':
                        self._has_x_margin = False
                        self._has_y_margin = True
                    else:
                        self._has_x_margin = True
                        self._has_y_margin = False
        elif self.current_agg == 'freq' or self.current_agg == 'summary':
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
        # THIS CAN SPEED UP PERFOMANCE BY A GOOD AMOUNT BUT STACK-SAVING
        # TIME & SIZE WILL SUFFER. WE CAN DEL THE "SQUEEZED" COLLECTION AT
        # SAVE STAGE.
        # self._cache.set_obj(collection='squeezed',
        #                     key=self.f+self.w+self.x+self.y,
        #                     obj=(self.xdef, self.ydef,
        #                          self._x_indexers, self._y_indexers,
        #                          self.wv, self.matrix, self.idx_map))

    def _get_matrix(self):
        self.xdef, self.ydef, self._x_indexers, self._y_indexers, self.wv, self.matrix, self.idx_map = self._cache.get_obj('squeezed', self.f+self.w+self.x+self.y)
        if self.xdef is None:
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
            xmask = (np.nansum(mat[:, 1:len(self.xdef)+1], axis=1) > 0)
            if self.ydef is not None:
                ymask = (np.nansum(mat[:, len(self.xdef)+2:-1], axis=1) > 0)
                self.idx_map = np.concatenate(
                    [np.expand_dims(xmask & ymask, 1), mat_indexer], axis=1)
                return mat[xmask & ymask]
            else:
                self.idx_map = np.concatenate(
                    [np.expand_dims(xmask, 1), mat_indexer], axis=1)
                return mat[xmask]
        else:
            mask = (np.nansum(mat[:, :-1], axis=1) > 0)
            self.idx_map = np.concatenate(
                [np.expand_dims(mask, 1), mat_indexer], axis=1)
            return mat[mask]

    def _dummyfy(self, section=None):
        if section is not None:
            # i.e. Quantipy multicode data
            if self.d[section].dtype == 'object':
                section_data = self.d[section].str.get_dummies(';')
                if self._uses_meta:
                    res_codes = self._get_response_codes(section)
                    section_data.columns = [int(col) for col in section_data.columns]
                    section_data = section_data.reindex(columns=res_codes)
                    section_data.replace(np.NaN, 0, inplace=True)
                section_data.sort_index(axis=1, inplace=True)
            # i.e. Quantipy single-coded/numerical data
            else:
                section_data = pd.get_dummies(self.d[section])
                if self._uses_meta and not self._is_raw_numeric(section):
                    res_codes = self._get_response_codes(section)
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
            a_res = self._get_response_codes(self.x)
            dummies = []
            for i in a_i:
                dummies.append(pd.get_dummies(self.d[i]).reindex(columns=a_res))
            a_data = pd.concat(dummies, axis=1)
            return a_data.values, a_res, a_i

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

    def _is_margin(self):
        return self.current_agg in ['tbase', 'cbase', 'rbase']

    def _is_stats_result(self):
        return self.current_agg in ['mean', 'min', 'max', 'varcoeff', 'sem',
                                    'stddev', 'var', 'median', 'upper_q',
                                    'lower_q']

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
        elif self.current_agg == 'summary':
            summary_vals = ['mean', 'stddev', 'min', '25%',
                            'median', '75%', 'max']
            self.x_agg_vals = summary_vals
            self.y_agg_vals = self.ydef
        elif self.current_agg == 'cbase':
            self.x_agg_vals = 'All'
            self.y_agg_vals = self.ydef
        elif self.current_agg == 'rbase':
            self.x_agg_vals = self.xdef
            self.y_agg_vals = 'All'
        elif self._is_stats_result():
            if self.factorized == 'x':
                self.x_agg_vals = self.current_agg
                self.y_agg_vals = self.ydef if not self.comb_y else self.comb_y
            else:
                self.x_agg_vals = self.xdef if not self.comb_x else self.comb_x
                self.y_agg_vals = self.current_agg
        # can this made smarter WITHOUT 1000000 IF-ELSEs above?:
        if ((self.current_agg in ['freq', 'cbase', 'rbase', 'summary'] or
                self._is_stats_result()) and not self.type == 'array'):
            if self.x == '@':
                self.x_agg_vals = '@'
            if self.y == '@':
                self.y_agg_vals = '@'
        df = pd.DataFrame(self.result)
        idx, cols = self._make_multiindex()
        df.index = idx
        df.columns = cols
        self.result = df
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
            if self._has_y_margin or self.y == '@':
                base = self.cbase
            else:
                base = self.cbase[:, 1:]
        else:
            if self._has_x_margin:
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
        self.no_pairs = None
        self.no_diffs = None
        self.parameters = None
        self.mimic = None
        self.level = None
        # Calculate the required baseline measures for the test using the
        # Q instance
        self.Quantity = qp.Quantity(link, view.weights(), use_meta=True)
        if view.missing():
            self.Quantity.exclude(view.missing())
        if self.metric == 'means':
            self.sd, self.values, self.cbases = self.Quantity._dispersion(
                _return_mean=True, _return_base=True)
            self.sd = self.sd[:, 1:]
            self.values = self.values[:, 1:]
            self.cbases = self.cbases[:, 1:]
        else:
            self.values = view.dataframe.values.copy()
            self.cbases = view.cbases[:, 1:]
            self.rbases = view.rbases[1:, :]
            self.tbase = view.cbases[0, 0]
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
            if np.nansum(self.values) == 0:
                self.no_diffs = True
            if len(self.ydef) == 1:
                self.no_pairs = True       
            self.mimic = mimic
            self.comparevalue, self.level = self._convert_level(level)
        else:
            # Set global test algorithm parameters
            self.invalid = False
            self.no_diffs = False
            self.no_pairs = False
            valid_mimics = ['Dim', 'askia']
            if mimic not in valid_mimics:
                raise ValueError('Failed to mimic: "%s". Select from: %s\n'
                                 % (mimic, valid_mimics))
            else:
                self.mimic = mimic

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
                self.ebases = self.Quantity._effective_n(axis='x', margin=False)
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
            stat = pd.DataFrame(stat, columns=self.ypairs)
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
                         in combinations(sd_base_ratio[0], 2)])[None, :]

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
            ovlp_correc_pairs = self.overlap[None, :]

        return (np.sqrt((enum_pairs/denom_pairs) *
                        (ebases_correc_pairs - ovlp_correc_pairs)))

    # -------------------------------------------------
    # Specific algorithm values & test option measures
    # -------------------------------------------------
    def _sum_sq_w(self, base_ratio=True):
        """
        """
        if not self.Quantity.w == '@1':
            self.Quantity.weight()
        ssw = np.nansum(self.Quantity.matrix ** 2, axis=0)[[0], 1:]
        if base_ratio:
            return ssw/self.cbases
        else:
            return ssw

    def _cwi(self, threshold=5, as_df=False):
        """
        Derives the count distribution assuming independence between columns.
        """
        c_col_n = self.cbases
        c_cell_n = self.values
        t_col_n = self.tbase
        if self.rbases.shape[1] > 1:
            t_cell_n = self.rbases[1:, :]
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
        if self.is_weighted:
            self.Quantity.weight()
        m = self.Quantity.matrix.copy()
        m = np.nansum(m[:, 1:, 1:], axis=1)
        if not self.is_weighted:
            m /= m 
        m[m == 0] = np.NaN
        col_pairs = list(combinations(range(0, m.shape[1]), 2)) 
        if self.parameters['use_ebase'] and self.is_weighted:
            # Overlap computation when effective base is being used
            w_sum_sq = np.array([np.nansum(m[:, [c1]] + m[:, [c2]], axis=0)**2
                                 for c1, c2 in col_pairs])
            w_sq_sum = np.array([np.nansum(m[:, [c1]]**2 + m[:, [c2]]**2, axis=0)
                        for c1, c2 in col_pairs])
            return np.nan_to_num((w_sum_sq/w_sq_sum)/2).T
        else:
            # Overlap with simple weighted/unweighted base size
            ovlp = np.array([np.nansum(m[:, [c1]] + m[:, [c2]], axis=0)
                             for c1, c2 in col_pairs])
            return (np.nan_to_num(ovlp)/2).T

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
        if self.metric == 'proportions':
            if self.no_pairs or self.no_diffs:
                values[:] = np.NaN
            if values.shape == (1, 1) or values.shape == (1, 0):
                values = [np.NaN]
        if self.metric == 'means':
            if self.no_pairs:
                values = [np.NaN]
            if self.no_diffs and not self.no_pairs:
                values[:] = np.NaN
        return  pd.DataFrame(values,
                             index=self.multiindex[0],
                             columns=self.multiindex[1])
