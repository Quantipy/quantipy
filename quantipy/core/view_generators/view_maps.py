#-*- coding: utf-8 -*-
from .view_mapper import ViewMapper
from quantipy.core.view import View

import pandas as pd
import numpy as np
import copy
import traceback
import warnings
from collections import defaultdict
from itertools import combinations
from operator import add, sub, mul
from operator import truediv as div

from quantipy.core.tools.view.logic import (
    has_any, has_all, has_count,
    not_any, not_all, not_count,
    is_lt, is_ne, is_gt,
    is_le, is_eq, is_ge,
    union, intersection, get_logic_index)

from quantipy.core.helpers import functions as helpers
import quantipy.core.tools as tools
import quantipy as qp

from quantipy.core.cache import Cache
from quantipy.core.quantify.engine import Level

import time
class QuantipyViews(ViewMapper):
    """
    A collection of extendable MR aggregation and statistic methods.

    View methods are used to generate various numerical or categorical data
    aggregations. Their behaviour is controlled via ``kwargs``.
    """
    def __init_known_methods__(self):
        super(QuantipyViews, self).__init_known_methods__()
        self.known_methods['default']= {
            'method': 'default',
            'kwargs': {
                'text': ''
            }
        }
        self.known_methods['cbase'] = {
            'method': 'frequency',
            'kwargs': {
                'text': 'Base',
                'axis': 'x',
                'condition': 'x'
            }
        }
        self.known_methods['cbase_gross'] = {
            'method': 'frequency',
            'kwargs': {
                'text': 'Gross base',
                'axis': 'x',
                'condition': 'x'
            }
        }
        self.known_methods['rbase'] = {
            'method': 'frequency',
            'kwargs': {
                'text': 'Base',
                'axis': 'y',
                'condition': 'y'            }
        }
        self.known_methods['ebase'] = {
            'method': 'frequency',
            'kwargs': {
                'text': 'Effective Base',
                'axis': 'x',
                'condition': 'x'
            }
        }
        self.known_methods['counts'] = {
            'method': 'frequency',
            'kwargs': {
                'text': '',
                'axis': None
            }
        }
        self.known_methods['c%'] = {
            'method': 'frequency',
            'kwargs': {
                'text': '',
                'axis': None,
                'rel_to': 'y'
            }
        }
        self.known_methods['r%'] = {
            'method': 'frequency',
            'kwargs': {
                'text': '',
                'axis': None,
                'rel_to': 'x'
            }
        }
        self.known_methods['res_c%'] = {
            'method': 'frequency',
            'kwargs': {
                'text': '',
                'axis': None,
                'rel_to': 'counts_sum'
            }
        }
        self.known_methods['counts_sum'] = {
            'method': 'frequency',
            'kwargs': {
                'text': 'Total Sum',
                'condition': 'x',
                'axis': 'x'
            }
        }
        self.known_methods['c%_sum'] = {
            'method': 'frequency',
            'kwargs': {
                'text': 'Total Sum',
                'axis': 'x',
                'condition': 'x',
                'rel_to': 'y'
            }
        }
        self.known_methods['counts_cumsum'] = {
            'method': 'frequency',
            'kwargs': {
                'text': '',
                'axis': 'x',
                'condition': 'x++'
            }
        }
        self.known_methods['c%_cumsum'] = {
            'method': 'frequency',
            'kwargs': {
                'text': '',
                'axis': 'x',
                'condition': 'x++',
                'rel_to': 'y'
            }
        }
        self.known_methods['mean'] = {
            'method': 'descriptives',
            'kwargs': {
                'axis': 'x',
                'text': ''
            }
        }
        self.known_methods['stddev'] = {
            'method': 'descriptives',
            'kwargs': {
                'stats': 'stddev',
                'axis': 'x',
                'text': ''
            }
        }
        self.known_methods['min'] = {
            'method': 'descriptives',
            'kwargs': {
                'stats': 'min',
                'axis': 'x',
                'text': ''
            }
        }
        self.known_methods['max'] = {
            'method': 'descriptives',
            'kwargs': {
                'stats': 'max',
                'axis': 'x',
                'text': ''
            }
        }
        self.known_methods['median'] = {
            'method': 'descriptives',
            'kwargs': {
                'stats': 'median',
                'axis': 'x',
                'text': ''
            }
        }

    def default(self, link, name, kwargs):
        """
        Adds a file meta dependent aggregation to a Stack.

        Checks the Link definition against the file meta and produces
        either a numerical or categorical summary tabulation including
        marginal the results.

        Parameters
        ----------
        link : Quantipy Link object.
        name : str
            The shortname applied to the view.
        kwargs : dict

        Returns
        -------
        None
            Adds requested View to the Stack, storing it under the full
            view name notation key.
        """
        view = View(link, name, kwargs)
        pos, relation, rel_to, weights, text = view.get_std_params()
        meta = link.get_meta()
        categorical = ['single', 'delimited set']
        numeric = ['int', 'float']
        string = ['string']
        categorizable = categorical + numeric
        x_type, y_type, transpose = self._get_method_types(link)
        q = qp.Quantity(link, weight=weights)
        if q.type == 'array' and not q.y == '@':
            pass
        else:
            if link.y == '@':
                if x_type in categorical or x_type == 'array':
                    view_df = q.count().result
                elif x_type in numeric:
                    view_df = q.summarize().result
                    view_df.drop((link.x, 'All'), axis=0, inplace=True)
                elif x_type in string:
                    view_df = tools.view.agg.make_default_str_view(data, x=link.x)
            elif link.x == '@':
                if y_type in categorical:
                    view_df = q.count().result
                elif y_type in numeric:
                    view_df = q.summarize().result
                    view_df.drop((link.y, 'All'), axis=1, inplace=True)
            else:
                if x_type in categorical and y_type in categorizable:
                    view_df = q.count().result
                elif x_type in numeric and y_type in categorizable:
                    view_df =  q.summarize().result
                    view_df.drop((link.x, 'All'), axis=0, inplace=True)
                    view_df.drop((link.y, 'All'), axis=1, inplace=True)
            notation = view.notation('default', ':')
            view.dataframe = view_df
            view._notation = notation
            link[notation] = view

    def frequency(self, link, name, kwargs):
        """
        Adds count-based views on a Link defintion to the Stack object.

        ``frequency`` is able to compute several aggregates that are based on
        the count of code values in uni- or bivariate Links. This includes
        bases / samples sizes, raw or normalized cell frequencies and code
        summaries like simple and complex nets.

        Parameters
        ----------
        link : Quantipy Link object.
        name : str
            The shortname applied to the view.
        kwargs : dict
        Keyword arguments (specific)
        text : str, optional, default None
            Sets an optional label in the meta component of the view that is
            used when the view is passed into a Quantipy build (e.g. Excel,
            Powerpoint).
        logic : list of int, list of dicts or core.tools.view.logic operation
            If a list is passed this instructs a simple net of the codes given
            as int. Multiple nets can be generated via a list of dicts that
            map names to lists of ints. For complex logical statements,
            expression are parsed to identify the qualifying rows in the data.
            For example::

                # simple net
                'logic': [1, 2, 3]

                # multiple nets/code groups
                'logic': [{'A': [1, 2]}, {'B': [3, 4]}, {'C', [5, 6]}]

                # code logic
                'logic': has_all([1, 2, 3])

        calc : TODO

        calc_only : TODO

        Returns
        -------
        None
            Adds requested View to the Stack, storing it under the full
            view name notation key.

        .. note:: Net codes take into account if a variable is
                  multi-coded. The net will therefore consider qualifying
                  cases and not the raw sum of the frequencies
                  per category, i.e. no multiple counting of cases.
        """
        view = View(link, name, kwargs=kwargs)
        axis, condition, rel_to, weights, text = view.get_std_params()
        logic, expand, complete, calc, exclude, rescale = view.get_edit_params()
        # ====================================================================
        # This block of kwargs should be removed
        # parameter overwriting should be done using the template
        # NOT QP core code!
        if kwargs.get('combine', False):
            view._kwargs['expand'], expand = None, None
            view._kwargs['complete'], complete = False, False
            if logic is not None:
                for no, logic_def in enumerate(logic):
                    if 'expand' in list(logic_def.keys()):
                        logic_def['expand'] = None
                        logic[no] = logic_def
                view._kwargs['logic'] = logic
        # --------------------------------------------------------------------
        # This block of code resolves the rel_to arg. in order to be able to use
        # rebased % computations. We are also adjusting for the regular notation
        # string here...
        # We need to avoid the forced overwriting of the kwarg and use the actual
        # rel_to != 'x', 'y', 'counts_sum' string...
        per_cell = False
        if not rel_to in ['', None, 'x', 'y', 'counts_sum']:
            view._kwargs['rel_to'] = 'y'
            rel_to_kind = rel_to.split('.')
            if len(rel_to_kind) == 2:
                rel_to = rel_to_kind[0]
                if rel_to_kind[1] == 'cells':
                    per_cell = True
                elif rel_to_kind[1] == 'y':
                    per_cell = False
            try:
                link['x|f|:||{}|counts'.format(weights)]._kwargs['rebased'] = True
            except:
                pass
        # ====================================================================
        w = weights if weights is not None else None
        ignore = True if name == 'cbase_gross' else False
        q = qp.Quantity(link, w, ignore_flags=ignore)
        if q.type == 'array' and not q.y == '@':
            pass
        else:
            if q.leveled:
                leveled = Level(q)
                if rel_to is not None:
                    leveled.percent()
                elif axis == 'x':
                    leveled.base()
                else:
                    leveled.count()
                view.dataframe = leveled.lvldf
            elif logic is not None:
                try:
                    q.group(groups=logic, axis=axis, expand=expand, complete=complete)
                except NotImplementedError as e:
                    warnings.warn('NotImplementedError: {}'.format(e))
                    return None
                q.count(axis=None, as_df=False, margin=False)
                condition = view.spec_condition(link, q.logical_conditions, expand)
            else:
                eff = True if name == 'ebase' else False
                raw = True if name in ['counts_sum', 'c%_sum'] else False
                cum_sum = True if name in ['counts_cumsum', 'c%_cumsum'] else False
                if cum_sum: axis = None
                if eff: axis = 'x'
                q.count(axis=axis, raw_sum=raw, effective=eff, cum_sum=cum_sum,
                        margin=False, as_df=False)
            if rel_to is not None:
                if q.type == 'array':
                    rel_to = 'y'
                q.normalize(rel_to, per_cell=per_cell)
            q.to_df()
            view.cbases = q.cbase
            view.rbases = q.rbase
            if calc is not None:
                calc_only = kwargs.get('calc_only', False)
                q.calc(calc, axis, result_only=calc_only)
            if calc is not None or name in ['counts_sum', 'c%_sum', 'counts_cumsum', 'c%_cumsum']:
                method_nota = 'f.c:f'
            else:
                method_nota = 'f'
            notation = view.notation(method_nota, condition)
            view._notation = notation
            if not q.leveled:
                if q.type == 'array':
                    view.dataframe = q.result.T if link.y == '@' else q.result
                else:
                    view.dataframe = q.result
            view._kwargs['exclude'] = q.miss_x

            link[notation] = view

    def descriptives(self, link, name, kwargs):
        """
        Adds num. distribution statistics of a Link defintion to the Stack.

        ``descriptives`` views can apply a range of summary statistics.
        Measures include statistics of centrality, dispersion and mass.

        Parameters
        ----------
        link : Quantipy Link object.
        name : str
            The shortname applied to the view.
        kwargs : dict
        Keyword arguments (specific)
        text : str, optional, default None
            Sets an optional label suffix for the meta component of the view
            which will be appended to the statistic name and used when the
            view is passed into a Quantipy build (e.g. Excel, Powerpoint).
        stats : str, default 'mean'
            The measure to compute.
        exclude : list of int
             Codes that will not be considered calculating the result.
        rescale : dict
            A mapping of {old code: new code}, e.g.::

                {
                 1: 0,
                 2: 25,
                 3: 50,
                 4: 75,
                 5: 100
                }
        drop : bool
            If ``rescale`` provides a new scale defintion, ``drop`` will remove
            all codes that are not transformed. Acts as a shorthand for manually
            passing any remaining codes in ``exclude``.

        Returns
        -------
        None
            Adds requested View to the Stack, storing it under the full
            view name notation key.
        """
        view = View(link, name, kwargs=kwargs)
        if not view._x['is_multi'] or kwargs.get('source'):
            view = View(link, name, kwargs=kwargs)
            axis, condition, rel_to, weights, text = view.get_std_params()
            logic, expand, complete, calc, exclude, rescale = view.get_edit_params()
            stat = kwargs.get('stats', 'mean')
            view._kwargs['calc_only'] = True
            w = weights if weights is not None else None
            q = qp.Quantity(link, w)

            if kwargs.get('source', None):
                q = self._swap_and_rebase(q, kwargs['source'])
            if q.type == 'array' and not q.y == '@':
                pass
            else:
                if exclude is not None:
                    q.exclude(exclude, axis=axis)
                if rescale is not None:
                    drop = kwargs.get('drop', False)
                    q.rescale(rescale, drop)
                    if drop:
                        view._kwargs['exclude'] = q.miss_x
                condition = view.spec_condition(link)
                q.summarize(stat=stat, margin=False, as_df=True)
                if calc:
                    q.calc(calc, result_only=True)
                    method_nota = 'd.' + stat + '.c:f'
                else:
                    method_nota = 'd.' + stat
                notation = view.notation(method_nota, condition)
                view.cbases = q.cbase
                view.rbases = q.rbase
                if q.type == 'array':
                    view.dataframe = q.result.T if link.y == '@' else q.result
                else:
                    view.dataframe = q.result
                view._notation = notation
                view.translate_metric(set_value='meta')
                view._kwargs['exclude'] = q.miss_x
                link[notation] = view

    def coltests(self, link, name, kwargs):
        """
        Will test appropriate views from a Stack for stat. sig. differences.

        Tests can be performed on frequency aggregations (generated by
        ``frequency``) and means (from ``summarize``) and will compare all
        unique column pair combinations.

        Parameters
        ----------
        link : Quantipy Link object.
        name : str
            The shortname applied to the view.
        kwargs : dict
        Keyword arguments (specific):
        text : str, optional, default None
            Sets an optional label in the meta component of the view that is
            used when the view is passed into a Quantipy build (e.g. Excel,
            Powerpoint).
        metric : {'props', 'means'}, default 'props'
            Determines whether a proportion or means test algorithm is
            performed.
        test_total : bool, deafult False
            If True, the each View's y-axis column will be tested against the
            uncoditional total of its x-axis.
        mimic : {'Dim', 'askia'}, default 'Dim'
            It is possible to mimic the test logics used in other statistical
            software packages by passing them as instructions. The method will
            then choose the appropriate test parameters.
        level: {'high', 'mid', 'low'} or float
            Sets the level of significance to which the test is carried out.
            Given as str the levels correspond to ``'high'`` = 0.01, ``'mid'``
            = 0.05 and ``'low'`` = 0.1. If a float is passed the specified
            level will be used.
        flags : list of two int, default None
            Base thresholds for Dimensions-like tests, e.g. [30, 100]. First
            int is minimum base for reported results, second int controls small
            base indication.

        Returns
        -------
        None
            Adds requested View to the Stack, storing it under the full
            view name notation key.

        .. note::

            Mimicking the askia software (``mimic`` = ``'askia'``)
            restricts the values to be one of ``'high'``, ``'low'``,
            ``'mid'``. Any other value passed will make the algorithm fall
            back to ``'low'``. Mimicking Dimensions (``mimic`` =
            ``'Dim'``) can use either the str or float version.
        """
        view = View(link, name, kwargs=kwargs)
        axis, condition, rel_to, weights, text = view.get_std_params()
        cache = self._cache = link.get_cache()
        metric = kwargs.get('metric', 'props')
        mimic = kwargs.get('mimic', 'Dim')
        level = kwargs.get('level', 'low')
        flags = kwargs.get('flag_bases', None)
        test_total = kwargs.get('test_total', False)
        stack = link.stack
        get = 'count' if metric == 'props' else 'mean'
        views = self._get_view_names(cache, stack, weights, get=get)
        for in_view in views:
            try:
                view = View(link, name, kwargs=kwargs)
                condition = in_view.split('|')[2]
                test = qp.Test(link, in_view, test_total)
                if mimic == 'Dim':
                    test.set_params(level=level, flag_bases=flags)
                elif mimic == 'askia':
                    test.set_params(testtype='unpooled',
                                    level=level, mimic=mimic,
                                    use_ebase=False,
                                    ovlp_correc=False,
                                    cwi_filter=True)
                view_df = test.run()
                notation = view.notation('t.{}.{}.{}{}'.format(metric, mimic,
                                     '{:.2f}'.format(test.level)[2:],
                                     '+@' if test_total else ''),
                                     condition)
                view.dataframe = view_df
                view._notation = notation
                link[notation] = view
            except:
                pass

    @staticmethod
    def _swap_and_rebase(quantity, variable, axis='x'):
        rebase_on = {quantity.x: not_count(0)}
        org_x = quantity.x
        quantity.swap(var=variable, axis=axis, update_axis_def=False)
        try:
            quantity.filter(rebase_on, keep_base=False, inplace=True)
        except:
            warn = "Couldn't rebase 'source'-swapped array-type Quantity: {} "
            warn += "on {}\nPlease check descriptive stats results for correct "
            warn += "base sizes!"
            warnings.warn(warn.format(org_x, variable))
        return quantity

    @staticmethod
    def _get_view_names(cache, stack, weights, get='count'):
        """
        Filter the views contained in a Stack by specific names.

        Parameters
        ----------
        cache : quantipy.core.Cache
        stack : quantipy.core.Stack
        weights : str, default None
        get : {'count', 'mean'}, default 'count'
            text

        Returns
        -------
        view_name_list : list of str
            text
        """
        w = weights if weights is not None else ''
        view_name_list = cache.get_obj(get+'_view_names', w+'_names')
        if view_name_list is None:
            # this change works, but it looks worrying, this line used to be
            #allviews = stack.describe(columns='view').index.tolist()
            allviews = stack.describe(index='view').index.tolist()
            if get == 'count':
                ignorenames = ['cbase', 'rbase', 'ebase', 'counts_sum',
                               'c%_sum', 'cbase_gross']
                view_name_list = [v for v in allviews
                                  if v.split('|')[1].startswith('f')
                                  and not v.split('|')[3]=='y'
                                  and not v.split('|')[-1] in ignorenames
                                  and v.split('|')[-2] == w]
            else:
                view_name_list = [v for v in allviews
                                  if v.split('|')[1] == 'd.mean'
                                  and v.split('|')[-2] == w]
            cache.set_obj(get+'_view_names', w+'_names', view_name_list)

        return view_name_list
