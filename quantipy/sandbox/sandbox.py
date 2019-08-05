#  -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import quantipy as qp


# from matplotlib import pyplot as plt
# import matplotlib.image as mpimg

import string
import cPickle

try:
    import seaborn as sns
    from PIL import Image
except:
    pass

from quantipy.core.chain import Chain as ChainCore
from quantipy.core.chainmanager import ChainManager as ChainManagerCore


from quantipy.core.cache import Cache
from quantipy.core.view import View
from quantipy.core.view_generators.view_mapper import ViewMapper
from quantipy.core.view_generators.view_maps import QuantipyViews
from quantipy.core.chain import _TransformedChainDF
from quantipy.core.chainannotations import ChainAnnotations
from quantipy.core.helpers.functions import emulate_meta
from quantipy.core.tools.view.logic import (has_any, has_all, has_count,
                                            not_any, not_all, not_count,
                                            is_lt, is_ne, is_gt,
                                            is_le, is_eq, is_ge,
                                            union, intersection, get_logic_index)
from quantipy.core.helpers.functions import (paint_dataframe,
                                             emulate_meta,
                                             get_text,
                                             finish_text_key)
from quantipy.core.tools.dp.prep import recode
from quantipy.core.tools.qp_decorators import lazy_property

from operator import add, sub, mul, div
from scipy.stats.stats import _ttest_finish as get_pval
from scipy.stats import chi2 as chi2dist
from scipy.stats import f as fdist
from itertools import combinations, chain, product
from collections import defaultdict, OrderedDict, Counter
import gzip

try:
    import dill
except:
    pass

import json
import copy
import time
import sys
import re


from quantipy.core.rules import Rules

import warnings
warnings.simplefilter('always')

_TOTAL = '@'
_AXES = ['x', 'y']


class ChainManager(ChainManagerCore):

    def __init__(self, stack):
        msg = "Please use 'quantipy.core.chainmanager.ChainManager' instead!"
        warnings.warn(msg, DeprecationWarning)
        super(ChainManager, self).__init__(stack)

class Chain(ChainCore):

    def __init__(self, stack, name, structure=None):
        msg = "Please use 'quantipy.core.chain.Chain' instead!"
        warnings.warn(msg, DeprecationWarning)
        super(Chain, self).__init__(stack, name, structure=None)

##############################################################################

class Multivariate(object):
    def __init__(self):
        pass

    def _select_variables(self, x, y=None, w=None, drop_listwise=False):
        x_vars, y_vars = [], []
        if not isinstance(x, list): x = [x]
        if not isinstance(y, list) and not y=='@': y = [y]
        if w is None: w = '@1'
        wrong_var_sel_1_on_1 = 'Can only analyze 1-to-1 relationships.'
        if self.analysis == 'Reduction' and (not (len(x) == 1 and len(y) == 1) or y=='@'):
            raise AttributeError(wrong_var_sel_1_on_1)
        for var in x:
            if self.ds._is_array(var):
                if self.analysis == 'Reduction': raise AttributeError(wrong_var_sel_1_on_1)
                x_a_items = self.ds._get_itemmap(var, non_mapped='items')
                x_vars += x_a_items
            else:
                x_vars.append(var)
        if y and not y == '@':
            for var in y:
                if self.ds._is_array(var):
                    if self.analysis == 'Reduction': raise AttributeError(wrong_var_sel_1_on_1)
                    y_a_items = self.ds._get_itemmap(var, non_mapped='items')
                    y_vars += y_a_items
                else:
                    y_vars.append(var)
        elif y == '@':
            y_vars = x_vars
        if x_vars == y_vars or y is None:
            data_slice = x_vars + [w]
        else:
            data_slice = x_vars + y_vars + [w]
        if self.analysis == 'Relations' and y != '@':
            self.x = self.y = x_vars + y_vars
            self._org_x, self._org_y = x_vars, y_vars
        else:
            self.x = self._org_x = x_vars
            self.y = self._org_y = y_vars
        self.w = w
        self._analysisdata = self.ds[data_slice]
        self._drop_missings()
        if drop_listwise:
            self._analysisdata.dropna(inplace=True)
            valid = self._analysisdata.index
            self.ds._data = self.ds._data.ix[valid, :]
        return None

    def _drop_missings(self):
        data = self._analysisdata.copy()
        for var in data.columns:
            if self.ds._has_missings(var):
                drop = self.ds._get_missing_list(var, globally=False)
                data[var].replace(drop, np.NaN, inplace=True)
        self._analysisdata = data
        return None

    def _has_analysis_data(self):
        if not hasattr(self, '_analysisdata'):
            raise AttributeError('No analysis variables assigned!')

    def _has_yvar(self):
        if self.y is None:
            raise AttributeError('Must select at least one y-variable or '
                                 '"@"-matrix indicator!')

    def _get_quantities(self, create='all'):
        crossed_quantities = []
        single_quantities = []
        helper_stack = qp.Stack()
        helper_stack.add_data(self.ds.name, self.ds._data, self.ds._meta)
        w = self.w if self.w != '@1' else None

        for x, y in product(self.x, self.y):
            helper_stack.add_link(x=x, y=y)
            l = helper_stack[self.ds.name]['no_filter'][x][y]
            crossed_quantities.append(qp.Quantity(l, weight=w))

        for x in self._org_x+self._org_y:
            helper_stack.add_link(x=x, y='@')
            l = helper_stack[self.ds.name]['no_filter'][x]['@']
            single_quantities.append(qp.Quantity(l, weight=w))

        self.single_quantities = single_quantities
        self.crossed_quantities = crossed_quantities
        return None

class Reductions(Multivariate):
    def __init__(self, dataset):
        self.ds = dataset
        self.single_quantities = None
        self.crossed_quantities = None
        self.analysis = 'Reduction'

    def plot(self, type, point_coords):
        plt.set_autoscale_on = False
        plt.figure(figsize=(5, 5))
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        #plt.axvline(x=0.0, c='grey', ls='solid', linewidth=0.9)
        #plt.axhline(y=0.0, c='grey', ls='solid', linewidth=0.9)
        x = plt.scatter(point_coords['x'][0], point_coords['x'][1],
                        edgecolor='w', marker='o', c='red', s=20)
        y = plt.scatter(point_coords['y'][0], point_coords['y'][1],
                        edgecolor='k', marker='^', c='lightgrey', s=20)

        fig = x.get_figure()
        # print fig.get_axes()[0].grid()
        fig.get_axes()[0].tick_params(labelsize=6)
        fig.get_axes()[0].patch.set_facecolor('w')

        fig.get_axes()[0].grid(which='major', linestyle='solid', color='grey',
                               linewidth=0.6)

        fig.get_axes()[0].xaxis.get_major_ticks()[0].label1.set_visible(False)
        x0 = fig.get_axes()[0].get_position().x0
        y0 = fig.get_axes()[0].get_position().y0
        x1 = fig.get_axes()[0].get_position().x1
        y1 = fig.get_axes()[0].get_position().y1

        text = 'Correspondence map'
        plt.figtext(x0+0.015, 1.09-y0, text, fontsize=12, color='w',
                    fontweight='bold', verticalalignment='top',
                    bbox={'facecolor':'red', 'alpha': 0.8, 'edgecolor': 'w',
                          'pad': 10})



        label_map = self._get_point_label_map('CA', point_coords)
        for axis in label_map.keys():
            for lab, coord in label_map[axis].items():
                plt.annotate(lab, coord, ha='left', va='bottom',
                    fontsize=6)
            plt.legend((x, y), (self.x[0], self.y[0]),
                       loc='best', bbox_to_anchor=(1.325, 1.07),
                       ncol=2, fontsize=6, title='                         ')

        x_codes, x_texts = self.ds._get_valuemap(self.x[0], non_mapped='lists')
        y_codes, y_texts = self.ds._get_valuemap(self.y[0], non_mapped='lists')

        text = ' '*80
        for var in zip(x_codes, x_texts):
            text += '\n{}: {}\n'.format(var[0], var[1])
        fig.text(1.06-x0, 0.85, text, fontsize=5, verticalalignment='top',
                          bbox={'facecolor':'red',
                          'edgecolor': 'w', 'pad': 10})
        x_len = len(x_codes)
        text = ' '*80
        for var in zip(y_codes, y_texts):
            text += '\n{}: {}\n'.format(var[0], var[1])
        test = fig.text(1.06-x0, 0.85-((x_len)*0.0155)-((x_len)*0.0155)-0.05, text, fontsize=5, verticalalignment='top',
                          bbox={'facecolor': 'lightgrey', 'alpha': 0.65,
                          'edgecolor': 'w', 'pad': 10})

        logo = Image.open('C:/Users/alt/Documents/IPython Notebooks/Designs/Multivariate class/__resources__/YG_logo.png')
        newax = fig.add_axes([x0+0.005, y0-0.25, 0.1, 0.1], anchor='NE', zorder=-1)
        newax.imshow(logo)
        newax.axis('off')

        fig.savefig(self.ds.path + 'correspond.png', bbox_inches='tight', dpi=300)

    def correspondence(self, x, y, w=None, norm='sym', diags=True, plot=True):
        """
        Perform a (multiple) correspondence analysis.

        Parameters
        ----------
        norm : {'sym', 'princ'}, default 'sym'
            <DESCP>
        summary : bool, default True
            If True, the output will contain a dataframe that summarizes core
            information about the Inertia decomposition.
        plot : bool, default False
            If set to True, a correspondence map plot will be saved in the
            Stack's data path location.
        Returns
        -------
        results: pd.DataFrame
            Summary of analysis results.
        """
        self._select_variables(x, y, w)
        self._get_quantities()
        # 1. Chi^2 analysis
        obs, exp = self.expected_counts(x=x, y=y, return_observed=True)
        chisq, sig = self.chi_sq(x=x, y=y, sig=True)
        inertia = chisq / np.nansum(obs)
        # 2. svd on standardized residuals
        std_residuals = ((obs - exp) / np.sqrt(exp)) / np.sqrt(np.nansum(obs))
        sv, row_eigen_mat, col_eigen_mat, ev = self._svd(std_residuals)
        # 3. row and column coordinates
        a = 0.5 if norm == 'sym' else 1.0
        row_mass = self.mass(x=x, y=y, margin='x')
        col_mass = self.mass(x=x, y=y, margin='y')
        dim = min(row_mass.shape[0]-1, col_mass.shape[0]-1)
        row_sc = (row_eigen_mat * sv[:, 0] ** a) / np.sqrt(row_mass)
        col_sc = (col_eigen_mat.T * sv[:, 0] ** a) / np.sqrt(col_mass)

        if plot:
            # prep coordinates for plot
            item_sep = len(self.single_quantities[0].xdef)
            dim1_c = [r_s[0] for r_s in row_sc] + [c_s[0] for c_s in col_sc]
            # dim2_c = [r_s[1]*(-1) for r_s in row_sc] + [c_s[1]*(-1) for c_s in col_sc]
            dim2_c = [r_s[1] for r_s in row_sc] + [c_s[1] for c_s in col_sc]
            dim1_xitem, dim2_xitem = dim1_c[:item_sep], dim2_c[:item_sep]
            dim1_yitem, dim2_yitem = dim1_c[item_sep:], dim2_c[item_sep:]
            coords = {'x': [dim1_xitem, dim2_xitem],
                      'y': [dim1_yitem, dim2_yitem]}
            self.plot('CA', coords)
            plt.show()

        if diags:
            _dim = xrange(1, dim+1)
            chisq_stats = [chisq, 'sig: {}'.format(sig),
                           'dof: {}'.format((obs.shape[0] - 1)*(obs.shape[1] - 1))]
            _chisq = ([np.NaN] * (dim-3)) + chisq_stats
            _sig = ([np.NaN] * (dim-2)) + [chisq]
            _sv, _ev = sv[:dim, 0], ev[:dim, 0]
            _expl_inertia = 100 * (ev[:dim, 0] / inertia)
            _cumul_expl_inertia = np.cumsum(_expl_inertia)
            _perc_chisq = _expl_inertia / 100 * chisq
            labels = ['Dimension', 'Singular values', 'Eigen values',
                     'explained % of Inertia', 'cumulative % explained',
                     'explained Chi^2', 'Total Chi^2']
            results = pd.DataFrame([_dim, _sv, _ev, _expl_inertia,
                                    _cumul_expl_inertia,_perc_chisq, _chisq]).T
            results.columns = labels
            results.set_index('Dimension', inplace=True)
            return results

    def _get_point_label_map(self, type, point_coords):
        if type == 'CA':
            xcoords = zip(point_coords['x'][0],point_coords['x'][1])
            xlabels = self.crossed_quantities[0].xdef
            x_point_map = {lab: coord for lab, coord in zip(xlabels, xcoords)}
            ycoords = zip(point_coords['y'][0], point_coords['y'][1])
            ylabels = self.crossed_quantities[0].ydef
            y_point_map = {lab: coord for lab, coord in zip(ylabels, ycoords)}
            return {'x': x_point_map, 'y': y_point_map}

    def mass(self, x, y, w=None, margin=None):
        """
        Compute rel. margins or total cell frequencies of a contigency table.
        """
        counts = self.crossed_quantities[0].count(margin=False)
        total = counts.cbase[0, 0]
        if margin is None:
            return counts.result.values / total
        elif margin == 'x':
            return  counts.rbase[1:, :] / total
        elif margin == 'y':
            return  (counts.cbase[:, 1:] / total).T

    def expected_counts(self, x, y, w=None, return_observed=False):
        """
        Compute expected cell distribution given observed absolute frequencies.
        """
        #self.single_quantities, self.crossed_quantities = self._get_quantities()
        counts = self.crossed_quantities[0].count(margin=False)
        total = counts.cbase[0, 0]
        row_m = counts.rbase[1:, :]
        col_m = counts.cbase[:, 1:]
        if not return_observed:
            return (row_m * col_m) / total
        else:
            return counts.result.values, (row_m * col_m) / total

    def chi_sq(self, x, y, w=None, sig=False, as_inertia=False):
        """
        Compute global Chi^2 statistic, optionally transformed into Inertia.
        """
        obs, exp = self.expected_counts(x=x, y=y, return_observed=True)
        diff_matrix = ((obs - exp)**2) / exp
        total_chi_sq = np.nansum(diff_matrix)
        if sig:
            dof = (obs.shape[0] - 1) * (obs.shape[1] - 1)
            sig_result = np.round(1 - chi2dist.cdf(total_chi_sq, dof), 3)
        if as_inertia: total_chi_sq /= np.nansum(obs)
        if sig:
            return total_chi_sq, sig_result
        else:
            return total_chi_sq

    def _svd(self, matrix, return_eigen_matrices=True, return_eigen=True):
        """
        Singular value decomposition wrapping np.linalg.svd().
        """
        u, s, v = np.linalg.svd(matrix, full_matrices=False)
        s = s[:, None]
        if not return_eigen:
            if return_eigen_matrices:
                return s, u, v
            else:
                return s
        else:
            if return_eigen_matrices:
                return s, u, v, (s ** 2)
            else:
                return s, (s ** 2)

class LinearModels(Multivariate):
    """
    OLS REGRESSION, ...
    """
    def __init__(self, dataset):
        self.ds = dataset.copy()
        self.single_quantities = None
        self.crossed_quantities = None
        self.analysis = 'LinearModels'

    def set_model(self, y, x, w=None, intercept=True):
        """
        """
        self._select_variables(x=x, y=y, w=w, drop_listwise=True)
        self._get_quantities()
        self._matrix = self.ds[self.y + self.x + [self.w]].dropna().values
        ymean = self.single_quantities[-1].summarize('mean', as_df=False)
        self._ymean = ymean.result[0, 0]
        self._use_intercept = intercept
        self.dofs = self._dofs()
        predictors = ' + '.join(self.x)
        if self._use_intercept: predictors = 'c + ' + predictors
        self.formula = '{} ~ {}'.format(y, predictors)
        return self

    def _dofs(self):
        """
        """
        correction = 1 if self._use_intercept else 0
        obs = self._matrix[:, -1].sum()
        tdof = obs - correction
        mdof = len(self.x)
        rdof = obs - mdof - correction
        return [tdof, mdof, rdof]

    def _vectors(self):
        """
        """
        w = self._matrix[:, [-1]]
        y = self._matrix[:, [0]]
        x = self._matrix[:, 1:-1]
        x = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
        return w, y, x

    def get_coefs(self, standardize=False):
        coefs = self._coefs() if not standardize else self._betas()
        coef_df = pd.DataFrame(coefs,
                               index = ['-c-'] + self.x
                               if self._use_intercept else self.x,
                               columns = ['b']
                               if not standardize else ['beta'])
        coef_df.replace(np.NaN, '', inplace=True)
        return coef_df

    def _betas(self):
        """
        """
        corr_mat = Relations(self.ds).corr(self.x, self.y, self.w, n=False, sig=None,
                            drop_listwise=True, matrixed=True)
        corr_mat = corr_mat.values
        predictors = corr_mat[:-1, :-1]
        y = corr_mat[:-1, [-1]]
        inv_predictors = np.linalg.inv(predictors)
        betas = inv_predictors.dot(y)
        if self._use_intercept:
            betas = np.vstack([[np.NaN], betas])
        return betas

    def _coefs(self):
        """
        """
        w, y, x = self._vectors()
        coefs = np.dot(np.linalg.inv(np.dot(x.T, x*w)), np.dot(x.T, y*w))
        return coefs

    def get_modelfit(self, r_sq=True):
        anova, fit_stats = self._sum_of_squares()
        dofs = np.round(np.array(self.dofs)[:, None], 0)
        anova_stats = np.hstack([anova, dofs, fit_stats])
        anova_df = pd.DataFrame(anova_stats,
                                index=['total', 'model', 'residual'],
                                columns=['sum of squares', 'dof', 'R', 'R^2'])
        anova_df.replace(np.NaN, '', inplace=True)
        return anova_df


    def _sum_of_squares(self):
        """
        """
        w, y, x = self._vectors()
        x_w = x*w
        hat = x_w.dot(np.dot(np.linalg.inv(np.dot(x.T, x_w)), x.T))
        tss  = (w*(y - self._ymean)**2).sum()[None]
        rss = y.T.dot(np.dot(np.eye(hat.shape[0])-hat, y*w))[0]
        ess = tss-rss
        all_ss = np.vstack([tss, ess, rss])
        rsq = np.vstack([[np.NaN], ess/tss, [np.NaN]])
        r = np.sqrt(rsq)
        all_rs = np.hstack([r, rsq])
        return all_ss, all_rs

    def estimate(self, estimator='ols', diags=True):
        """
        """
        # Wrap up the modularized computation methods
        coefs, betas = self.get_coefs(), self.get_coefs(True)
        modelfit = self.get_modelfit()
        # Compute diagnostics, i.e. standard errors and sig. of estimates/fit
        # prerequisites
        w, _, x = self._vectors()
        rss = modelfit.loc['residual', 'sum of squares']
        ess = modelfit.loc['model', 'sum of squares']
        # coefficients: std. errors, t-stats, sigs
        c_se = np.diagonal(np.sqrt(np.linalg.inv(np.dot(x.T,x*w)) *
                                   (rss/self.dofs[-1])))[None].T
        c_sigs = np.hstack(get_pval(self.dofs[-1], coefs/c_se))
        c_diags = np.round(np.hstack([c_se, c_sigs]), 6)
        c_diags_df = pd.DataFrame(c_diags, index=coefs.index,
                                  columns=['se', 't-stat', 'p'])
        # modelfit: se, F-stat, ...
        m_se = np.vstack([[np.NaN], np.sqrt(rss/self.dofs[-1]), [np.NaN]])
        m_fstat = np.vstack([[np.NaN],
                             (ess/self.dofs[1]) / (rss/self.dofs[-1]),
                             [np.NaN]])
        m_sigs = 1-fdist.cdf(m_fstat, self.dofs[1], self.dofs[-1])
        m_diags = np.round(np.hstack([m_se, m_fstat, m_sigs]), 6)
        m_diags_df = pd.DataFrame(m_diags, index=modelfit.index,
                                  columns=['se', 'F-stat', 'p'])
        # Put everything together
        parameter_results = pd.concat([coefs, betas, c_diags_df], axis=1)
        fit_summary = pd.concat([modelfit, m_diags_df], axis=1).replace(np.NaN, '')
        return parameter_results, fit_summary

    def _lmg_models_per_var(self):
        all_models = self._lmg_combs()
        models_by_var = {x: [] for x in self.x}
        for var in self.x:
            qualified_models = []
            for model in all_models:
                if var in model: qualified_models.append(model)
            for qualified_model in qualified_models:
                q_m = list(qualified_model)
                q_m.remove(var)
                models_by_var[var].append([qualified_model, q_m])
        return models_by_var

    def _lmg_combs(self):
        full = self.x
        lmg_combs = []
        for combine_no in xrange(1, len(full)):
            lmg_combs.extend([list(comb) for comb in
                              list(combinations(full, combine_no))])
        lmg_combs.append(full)
        return lmg_combs

    def _rsq_lmg_subset(self, subset):
        self.set_model(self.y, subset, self.w)
        anova = self.get_modelfit()
        return anova['R^2'].replace('', np.NaN).dropna().values[0]

    def lmg(self, norm=True, plot=False):
        known_rsq = {}
        x_results = {}
        full_len = len(self.x)
        cols = self.y + self.x
        self._analysisdata = self._analysisdata.copy().dropna(subset=cols)
        total_rsq = self._rsq_lmg_subset(self.x)
        all_models = self._lmg_models_per_var()
        model_max_no = len(all_models.keys()) * len(all_models.values()[0])
        # print 'LMG analysis on {} models started...'.format(model_max_no)
        for x, diff_models in all_models.items():
            group_results = {size: [] for size in xrange(1, full_len + 1)}
            for diff_model in diff_models:
                # sys.stdout.write('|')
                # sys.stdout.flush()
                if not diff_model[1]:
                    if tuple(diff_model[0]) in known_rsq.keys():
                        r1 = known_rsq[tuple(diff_model[0])]
                    else:
                        r1 = self._rsq_lmg_subset(diff_model[0])
                        known_rsq[tuple(diff_model[0])] = r1
                    group_results[len(diff_model[0])].append((r1))
                else:
                    if tuple(diff_model[0]) in known_rsq.keys():
                        r1 = known_rsq[tuple(diff_model[0])]
                    else:
                        r1 = self._rsq_lmg_subset(diff_model[0])
                        known_rsq[tuple(diff_model[0])] = r1
                    if tuple(diff_model[1]) in known_rsq.keys():
                        r2 = known_rsq[tuple(diff_model[1])]
                    else:
                        r2 = self._rsq_lmg_subset(diff_model[1])
                        known_rsq[tuple(diff_model[1])] = r2
                    group_results[len(diff_model[0])].append((r1-r2))
            x_results[x] = group_results
        lmgs = []
        for var, results in x_results.items():
            res = np.mean([np.mean(val) for val in results.values()])
            lmgs.append((var, res))
            labs = ['Variable',
                    'Importance {}'.format('(normalized)' if norm else '')]
            result = pd.DataFrame(lmgs, columns=labs)
            result.set_index('Variable', inplace=True)
            result.index.name = 'LMG analysis'
            result.sort(columns=labs[1], ascending=False, inplace=True)
            if norm:
                result = result / total_rsq * 100
        return result

class Relations(Multivariate):
    """
    COV, CORR, SCATTER
    """
    def __init__(self, dataset):
        self.ds = dataset
        self.single_quantities = None
        self.crossed_quantities = None
        self.analysis = 'Relations'

    def _has_matrix_structure(self):
        return self.x == self.y

    def _make_index_pairs(self):
        if self._has_matrix_structure():
            full_range = len(self.x) - 1
        else:
            full_range = len(self.x + self.y) - 1
        x_range = range(0, len(self.x))
        y_range = range(x_range[-1] + 1, full_range + 1)
        if self._has_matrix_structure():
            return list(product(range(0, full_range+1), repeat=2))
        else:
            return list(product(x_range, y_range))

    def _sort_as_paired_stats(self, stat_list, pair_list):
        pairs = {pair: stat for pair, stat in zip(pair_list, stat_list)}
        if self._has_matrix_structure():
            return [(pairs[p[0], p[1]], pairs[p[1], p[0]]) for p in pair_list]
        else:
            return [(pairs[p[0], p[1]], pairs[p[0], p[1]]) for p in pair_list]

    def action_matrix(self, perf, imp, w=None, measures={
            'method': 'simple', 'perf': 'mean', 'imp': 'mean'}):
        """
        ... DESP ...

        Parameters
        ----------
        perf : str or list of str
            DESCP
        imp : list of str
            DESCP
        measures : dict {'method': ..., 'perf_stat': ..., 'imp_stat': ...}
            DECP

        Returns
        -------

        """
        method = measures['method']
        perf_stat, imp_stat = measures['perf'], measures['imp']
        if method in ['corr', 'reg']:
            raise NotImplementedError("{}-method unsupported.".format(method))
        if perf_stat != 'mean' and not isinstance(perf_stat, list):
            raise ValueError("'perf' stat must be list of codes or 'mean'.")
        if imp_stat != 'mean' and not isinstance(imp_stat, list):
            raise ValueError("'imp' stat must be list of codes or 'mean'.")
        # Two simple item batteries of identical length for performance and
        # importance dimensions
        if method == 'simple':
            self._select_variables(perf, imp, w)
            self._get_quantities()
            il = len(self._org_x)
            if perf_stat == 'mean':
                perfs = [q.summarize('mean', as_df=False, margin=False).result[0, 0]
                         for q in self.single_quantities[:il]]
                perfs = np.array(perfs)
            else:
                perfs = [q.group(perf_stat) for q in self.single_quantities[:il]]
                perfs = [p.count(as_df=False, margin=False).normalize().result[0, 0]
                        for p in perfs]
                perfs = np.array(perfs)
            if imp_stat == 'mean':
                imps = [q.summarize('mean', as_df=False, margin=False).result[0, 0]
                        for q in self.single_quantities[il:]]
                imps = np.array(imps)
            else:
                imps = [q.group(imp_stat) for q in self.single_quantities[il:]]
                imps = [i.count(as_df=False, margin=False).normalize().result[0, 0]
                        for i in imps]
                imps = np.array(imps)
        # Centering of data - currently only valid for the 'simple' approach!
        perf_mean, perf_sd = perfs.mean(), perfs.std(ddof=1)
        imps_mean, imps_sd = imps.mean(), imps.std(ddof=1)
        perf_c = (perfs -  perfs.mean()) / perfs.std(ddof=1)
        imps_c = (imps - imps.mean()) / imps.std(ddof=1)

        plt.set_autoscale_on = False
        plt.figure(figsize=(5, 5))
        plt.axvline(x=0.0, c='grey', ls='solid', linewidth=0.9)
        plt.axhline(y=0.0, c='grey', ls='solid', linewidth=0.9)
        if method == 'simple':
            xlim = max(abs(perf_c.min()), perf_c.max()) + 1.0
            ylim = max(abs(imps_c.min()), imps_c.max()) + 1.0
            plt.xlim([-xlim, xlim])
            plt.ylim([-ylim, ylim])
            vals = np.vstack([imps_c, perf_c]).T
        else:
            plt.xlim([0, 6])
            plt.ylim([-1, 1])
            vals = result.values
        x = plt.scatter(vals[:, 1], vals[:, 0],  edgecolor='w', marker='o',
                        c='red', s=80)
        fig = x.get_figure()
        xlab = 'Performance\n({})'
        xlab = xlab.format('mean' if perf_stat == 'mean' else 'top box')
        fig.get_axes()[0].set_xlabel(xlab)
        ylab = 'Importance\n({})'
        ylab = ylab.format('mean' if imp_stat == 'mean' else 'top box')
        fig.get_axes()[0].set_ylabel(ylab)
        plt.tick_params(axis='both', labelbottom='off', labelleft='off')

        x0 = fig.get_axes()[0].get_position().x0
        y0 = fig.get_axes()[0].get_position().y0
        x1 = fig.get_axes()[0].get_position().x1
        y1 = fig.get_axes()[0].get_position().y1

        ax = fig.get_axes()[0]
        fig.text(0.3, 0.87, 'critical improvement', ha='center', va='center',
                 fontsize=7, transform=ax.transAxes)
        fig.text(0.3, 0.16, 'action not required', ha='center', va='center',
                 fontsize=7, transform=ax.transAxes)
        fig.text(0.73, 0.87, 'leverage strengths', ha='center', va='center',
                 fontsize=7, transform=ax.transAxes)
        fig.text(0.73, 0.16, 'resource transfer\nopportunity', ha='center',
                 va='center', fontsize=7, transform=ax.transAxes)

        text = 'Priority matrix'
        plt.figtext(x0+0.015, 1.09-y0, text, fontsize=12, color='w',
                    fontweight='bold', verticalalignment='top',
                    bbox={'facecolor':'red', 'alpha': 0.8, 'edgecolor': 'w',
                          'pad': 10})
        label_vars = self._org_x
        text = ''
        for no, var in enumerate(label_vars, start=1):
            text += '\n{}: {}\n'.format(no, self.ds._get_label(var))
        fig.text(1.06-x0, 1.011-y0, text, fontsize=6, verticalalignment='top',
                 bbox={'facecolor':'lightgrey', 'alpha': 0.65,
                       'edgecolor': 'w', 'pad': 10})

        for no, coord in enumerate(vals, start=1):
            coord = [coord[1], coord[0]]
            plt.annotate(no, coord, ha='center', va='center',
                fontsize=7)

        logo = Image.open('C:/Users/alt/Documents/IPython Notebooks/Designs/Multivariate class/__resources__/YG_logo.png')
        newax = fig.add_axes([x0+0.005, y0-0.15, 0.1, 0.1], anchor='NE', zorder=-1)
        newax.imshow(logo)
        newax.axis('off')
        fig.savefig(self.ds.path + 'action_matrix.png', bbox_inches='tight', dpi=300)

    def cov(self, x, y, w=None, n=False, drop_listwise=False):
        self._select_variables(x, y, w, drop_listwise)
        if self.single_quantities is None: self._get_quantities()
        pairs = self._make_index_pairs()
        means = [q._drop_pairwise().summarize('mean', as_df=False).result[0, 0]
                 for q in self.crossed_quantities]
        means_paired = self._sort_as_paired_stats(means, pairs)
        xprods, unbiased_ns = [], []
        for pair, means_pair in zip(pairs, means_paired):
            data = self._analysisdata.copy()
            data = data.ix[:, [pair[0], pair[1], -1]].dropna().values
            m_diff = data[:, :-1] - means_pair
            xprods.append(np.nansum(m_diff[:, 0] * m_diff[:, 1] * data[:, -1]))
            unbiased_ns.append(np.nansum(data[:, -1]) - 1)
        cov = np.array(xprods) / np.array(unbiased_ns)
        cov = pd.DataFrame(cov.reshape(len(self.x), len(self.y)),
                           index=self.x, columns=self.y)
        cov.index.name = 'Covariance'
        if n:
            paired_ns = [n + 1 for n in unbiased_ns]
            return cov, paired_ns
        else:
            return cov
        # if n:

        #     return cov, paired_ns
        # else:
        #     return cov

    def corr(self, x, y, w=None, n=False, sig='full', drop_listwise=False, matrixed=False, plot=False):
        self._select_variables(x, y, w, drop_listwise)
        self._has_analysis_data()
        self._has_yvar()
        cov, ns = self.cov(x, y, w, True, drop_listwise)
        pairs = self._make_index_pairs()
        stddev = [q.summarize('stddev', as_df=False).result[0, 0]
                  for q in self.crossed_quantities]
        stddev_paired = self._sort_as_paired_stats(stddev, pairs)
        normalizer = [stddev1 * stddev2 for stddev1, stddev2 in stddev_paired]
        corr = cov / np.array(normalizer).reshape(cov.shape)
        ns = pd.DataFrame(np.array(ns).reshape(corr.shape),
                         index=corr.index, columns=corr.columns)
        corr.index.name = None

        if not matrixed:
            ns = ns.loc[self._org_x, self._org_y]
            corr = corr.loc[self._org_x, self._org_y]
        if not n and not sig and not plot:
            corr.index.name = 'Correlations'
            return corr

        if sig and sig not in ['flag', 'full']:
                raise ValueError('"sig" must be one of None, "flag" or "full".')
        sigtest = np.sqrt((ns-2)/(1-corr**2))*corr
        sigtest = pd.DataFrame(get_pval(ns-2, sigtest)[1], index=corr.index, columns=corr.columns)
        sigtest.replace(np.NaN, 0, inplace=True)
        sigtest_flags = sigtest.copy()[sigtest<0.05]
        sigtest_flags[sigtest_flags < 0.01] = 1
        sigtest_flags[(sigtest_flags != 1) & (~np.isnan(sigtest_flags))] = 2
        sigtest_flags.replace(1, '**', inplace=True)
        sigtest_flags.replace(2, '*', inplace=True)
        sigtest_flags.replace(np.NaN, '', inplace=True)

        collect = []
        corr_iter = np.round(corr.copy(), 4)
        sigtest = np.round(sigtest, 3)
        ns = np.round(ns, 0)
        for r, flag, p, n_ in zip(corr_iter.iterrows(), sigtest_flags.iterrows(), sigtest.iterrows(), ns.iterrows()):
            row1 = pd.DataFrame(r[1]).T
            row2 = pd.DataFrame(flag[1]).T
            row3 = pd.DataFrame(p[1]).T
            row4 = pd.DataFrame(n_[1]).T
            collect.append(pd.concat([row1, row2, row3, row4], axis=0))
        final = pd.concat(collect, axis=0)

        var = self._org_x if not matrixed else self._org_x + self._org_y
        stats = ['r', 'sig.', 'p', 'n']
        mi = pd.MultiIndex.from_product([var, stats], names=['', 'Correlations'])
        final.index = mi
        if not n or sig != 'full':
            if n:
                if sig == 'flag':
                    select = ['r', 'sig.', 'n']
                elif sig == 'p':
                    select = ['r', 'p', 'n']
            else:
                if sig == 'full':
                    select = ['r', 'sig.', 'p']
                elif sig == 'flag':
                    select = ['r', 'sig.']
                elif sig == 'p':
                    select = ['r', 'p']
            final = final[[group2 in select for group1, group2 in final.index]]

        if plot:
            if plot not in ['sig', 'full']:
                raise ValueError('"plot" must be one of None, "sig" or "full".')
            if plot == 'sig':
                corr = corr[sigtest < 0.05]
                center = np.mean(corr.replace(np.NaN, 0.0).values)
            else:
                center = np.mean(corr.values)

            colors = sns.blend_palette(['lightgrey', 'red'], as_cmap=True,
                                       n_colors=1000)

            corr_res = sns.heatmap(corr, annot=True, cbar=None, fmt='.2f',
                                   square=True, robust=True, cmap=colors,
                                   center=center, linewidth=1.0,
                                   annot_kws={'size': 8})

            fig = corr_res.get_figure()
            x0 = fig.get_axes()[0].get_position().x0
            y0 = fig.get_axes()[0].get_position().y0
            x1 = fig.get_axes()[0].get_position().x1
            y1 = fig.get_axes()[0].get_position().y1

            text = 'Correlation matrix (Pearson)'
            plt.figtext(x0+0.017, 1.115-y0, text, fontsize=12, color='w',
                        fontweight='bold', verticalalignment='top',
                        bbox={'facecolor':'red', 'alpha': 0.8, 'edgecolor': 'w',
                              'pad': 10})
            if self._has_matrix_structure():
                label_vars = self.x
            else:
                label_vars = self.x + self.y
            text = ''
            for var in label_vars:
                text += '\n{}: {}\n'.format(var, self.ds._get_label(var))
            fig.text(1.06-x0, 1.0-y0, text, fontsize=6, verticalalignment='top',
                     bbox={'facecolor':'lightgrey', 'alpha': 0.65,
                           'edgecolor': 'w', 'pad': 10})
            logo = Image.open('C:/Users/alt/Documents/IPython Notebooks/Designs/Multivariate class/__resources__/YG_logo.png')
            newax = fig.add_axes([x0+0.005, y0-0.25, 0.1, 0.1], anchor='NE', zorder=-1)
            newax.imshow(logo)
            newax.axis('off')
            fig.savefig(self.ds.path + 'corr.png', bbox_inches='tight', dpi=300)

        final.index.name = 'Correlation'
        return final


##############################################################################

class Stack(defaultdict):
    def __init__(self, name=''):
        super(Stack, self).__init__(Stack)
        self.name = name
        self.ds = None

    # ====================================================================
    # THESE NEED TO GET A REVIEW!
    # ====================================================================
    # def __reduce__(self):
    #     arguments = (self.name, )
    #     states = self.__dict__.copy()
    #     if states['ds'] is not None:
    #         states['ds'].__dict__['_cache'] = Cache()
    #     return self.__class__, arguments, states, None, self.iteritems()

    # ====================================================================
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # ====================================================================

    # ------------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------------
    def add_dataset(self, dataset):
        self.ds = dataset

    def save(self, path_stack, compressed=False):
        if compressed:
            f = gzip.open(path_stack, 'wb')
        else:
            f = open(path_stack, 'wb')
        dill.dump(self, f)
        f.close()
        return None

    def load(self, path_stack, compressed=False):
        if compressed:
            f = gzip.open(path_stack, 'rb')
        else:
            f = open(path_stack, 'rb')
        loaded_stack = dill.load(f)
        f.close()
        return loaded_stack

    # ------------------------------------------------------------------------
    # DATA/LINK POPULATION
    # ------------------------------------------------------------------------
    def refresh(self):
        pass

    def populate(self, filters=None, x=None, y=None, weights=None, views=None):
        """
        Populate the Stack instance with Links that (optionally) hold Views.
        """
        if filters is None: filters = ['no_filter']
        for _filter in filters:
            for _x in x:
                for _y in y:
                    if not isinstance(self[self.ds.name][_filter][_x][_y], Link):
                        l = self.ds.link(_filter, _x, _y)
                        l.stack_connection = True
                        self[self.ds.name][_filter][_x][_y] = l
                    else:
                        l = self.get(self.ds.name, _filter, _x, _y)
                        l.stack_connection = True
                    if views is not None:
                        if not isinstance(views, ViewMapper):
                            # Use DefaultViews if no view were given
                            if views is None:
                                pass
                            elif isinstance(views, (list, tuple)):
                                views = QuantipyViews(views=views)
                            else:
                                print 'ERROR - VIEWS CRASHED!'
                        views._apply_to(l, weights)
                        l._clear()

    # ------------------------------------------------------------------------
    # INSPECTION & QUERY
    # ------------------------------------------------------------------------
    def get(self, ds_key=None, filters=None, x=None, y=None):
        """
        Return Link from Stack.
        """
        if ds_key is None and len(self.keys()) > 1:
            key_err = 'Cannot select from multiple datasets when no key is provided.'
            raise KeyError(key_err)
        elif ds_key is None and len(self.keys()) == 1:
            ds_key = self.keys()[0]
        if filters is None: filters = 'no_filter'
        if not isinstance(self[ds_key][filters][x][y], Link):
            l = Link(self.ds, filters, x, y)
        else:
            l = self[ds_key][filters][x][y]
            l._quantify(self.ds)
        return l

    def describe(self, index=None, columns=None, query=None, split_view_names=False):
        """
        Generates a structured overview of all Link defining Stack elements.

        Parameters
        ----------
        index, columns : str of or list of {'data', 'filter', 'x', 'y', 'view'},
                         optional
            Controls the output representation by structuring a pivot-style
            table according to the index and column values.
        query : str
            A query string that is valid for the pandas.DataFrame.query() method.
        split_view_names : bool, default False
            If True, will create an output of unique view name notations split
            up into their components.

        Returns
        -------
        description : pandas.DataFrame
            DataFrame summing the Stack's structure in terms of Links and Views.
        """
        stack_tree = []
        for dk in self.keys():
            path_dk = [dk]
            filters = self[dk]

#             for fk in filters.keys():
#                 path_fk = path_dk + [fk]
#                 xs = self[dk][fk]

            for fk in filters.keys():
                path_fk = path_dk + [fk]
                xs = self[dk][fk]

                for sk in xs.keys():
                    path_sk = path_fk + [sk]
                    ys = self[dk][fk][sk]

                    for tk in ys.keys():
                        path_tk = path_sk + [tk]
                        views = self[dk][fk][sk][tk]

                        if views.keys():
                            for vk in views.keys():
                                path_vk = path_tk + [vk, 1]
                                stack_tree.append(tuple(path_vk))
                        else:
                            path_vk = path_tk + ['|||||', 1]
                            stack_tree.append(tuple(path_vk))

        column_names = ['data', 'filter', 'x', 'y', 'view', '#']
        description = pd.DataFrame.from_records(stack_tree, columns=column_names)
        if split_view_names:
            views_as_series = pd.DataFrame(
                description.pivot_table(values='#', columns='view', aggfunc='count')
                ).reset_index()['view']
            parts = ['xpos', 'agg', 'condition', 'rel_to', 'weights',
                     'shortname']
            description = pd.concat(
                (views_as_series,
                 pd.DataFrame(views_as_series.str.split('|').tolist(),
                              columns=parts)), axis=1)

        description.replace('|||||', np.NaN, inplace=True)
        if query is not None:
            description = description.query(query)
        if not index is None or not columns is None:
            description = description.pivot_table(values='#', index=index, columns=columns,
                                aggfunc='count')
        return description
