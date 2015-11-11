 #-*- coding: utf-8 -*-
from view_mapper import ViewMapper
from quantipy.core.view import View

import pandas as pd
import numpy as np
import copy
import traceback
from collections import defaultdict
from itertools import combinations
from operator import add, sub, mul, div

from quantipy.core.helpers import functions as helpers
import quantipy.core.tools as tools
import quantipy as qp

from quantipy.core.cache import Cache

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
                'pos': 'x',
                'relation': 'x:y'
            }
        }
        self.known_methods['rbase'] = {
            'method': 'frequency',
            'kwargs': {
            	'text': 'Base',
                'pos': 'x',
                'relation': 'y:x'            }
        }
        self.known_methods['ebase'] = {
            'method': 'frequency',
            'kwargs': {
                'text': 'Effective Base',
                'pos': 'x',
                'relation': 'x:y'
            }
        }
        self.known_methods['counts'] = {
            'method': 'frequency',
            'kwargs': {
            	'text': ''
            }
        }        
        self.known_methods['c%'] = {
            'method': 'frequency',
            'kwargs': {
            	'text': '',
                'rel_to': 'y'
            }
        }
        self.known_methods['r%'] = {
            'method': 'frequency',
            'kwargs': {
            	'text': '',
                'rel_to': 'x'
            }
        }
        self.known_methods['mean'] = {
            'method': 'descriptives',
            'kwargs': {
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
        view = View(link, kwargs)
        pos, relation, rel_to, weights, text = view.std_params()
        meta = link.get_meta()
        categorical = ['single', 'delimited set']
        numeric = ['int', 'float']
        string = ['string']
        categorizable = categorical + numeric
        x_type, y_type, transpose = self._get_method_types(link)
        q = qp.Quantity(link, weight=weights)
        if link.y == '@':
            if x_type in categorical:
                view_df = q.count().result
            elif x_type in numeric:
                view_df = q.describe().result
                view_df.drop((link.x, 'All'), axis=0, inplace=True)
            elif x_type in string:
                view_df = tools.view.agg.make_default_str_view(data, x=link.x)
        elif link.x == '@':
            if y_type in categorical:
                view_df = q.count().result
            elif y_type in numeric:
                view_df = q.describe().result
                view_df.drop((link.y, 'All'), axis=1, inplace=True)
        else:
            if x_type in categorical and y_type in categorizable:
                view_df = q.count().result
            elif x_type in numeric and y_type in categorizable:
                view_df =  q.describe().result
                view_df.drop((link.x, 'All'), axis=0, inplace=True)
                view_df.drop((link.y, 'All'), axis=1, inplace=True)
        
        relation = view.spec_relation()
        notation = view.notation('default', name, relation)
        view.dataframe = view_df
        view.name = notation
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
        # link : Quantipy Link object.
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
        func_name = 'frequency'
        func_type = 'countbased'
        view = View(link, kwargs=kwargs)
        pos, relation, rel_to, weights, text = view.std_params()
        q = qp.Quantity(link, weights, use_meta=True)        
        logic = kwargs.get('logic', None)
        calc = kwargs.get('calc', None)
        val_name = None

        if name in ['ebase', 'cbase', 'rbase']:
            freq = q.count(name, margin=False, as_df=False)
        elif name in ['counts', 'c%', 'r%']:
            freq = q.count('freq', margin=False, as_df=False)
        elif logic:
            if isinstance(logic, list):
                if not isinstance(logic[0], dict):
                    val_name = name
                if calc:
                    calc_only = kwargs.get('calc_only', False)
                else:
                    calc_only = False
                freq = q.combine(logic, op=calc, op_only=calc_only,
                                 margin=False, as_df=False)
                relation = view.spec_relation()
            else:
                val_name = name
                casedata = link.get_data().copy()
                idx, relation = tools.view.logic.get_logic_index(
                    casedata[link.x], logic, casedata)
                filtered_q = qp.Quantity(link, weights, idx)
                freq = filtered_q.combine(margin=False, as_df=False)
        view.cbases = freq.cbase
        view.rbases = freq.rbase
        if rel_to is not None:
            base = 'col' if rel_to == 'y' else 'row'
            freq = freq.normalize(base)
        view_df = freq.to_df(val_name).result
        notation = view.notation(func_name, name, relation)
        view.name = notation        
        view.dataframe = view_df
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

        stats : str, default 'mean'
            The measure to compute.

        Returns
        -------
        None
            Adds requested View to the Stack, storing it under the full
            view name notation key.
        """
        view = View(link, kwargs=kwargs)
        if not view._x['is_multi']:
            func_name = 'descriptives'
            func_type = 'distribution statistics'
            pos, relation, rel_to, weights, text = view.std_params()

            stat = kwargs.get('stats', 'mean')
            exclude = view.missing()
            rescale = view.rescaling()
            q = qp.Quantity(link, weights, use_meta=True)         
            
            if exclude is not None:
                q = q.missingfy(exclude, keep_base=False)
            if rescale is not None:
                q = q.rescale(rescale)
            view.fulltext_for_stat(stat)
            relation = view.spec_relation(link)
            view_df = q.describe(show=stat, margin=False, as_df=True)
            notation = view.notation(stat, name, relation)
            view.cbases = view_df.cbase
            view.rbases = view_df.rbase
            view.dataframe = view_df.result
            view.name = notation
            link[notation] = view
        
    def coltests(self, link, name, kwargs):
        """
        Will test appropriate views from a Stack for stat. sig. differences.

        Tests can be performed on frequency aggregations (generated by
        ``frequency``) and means (from ``descriptives``) and will compare all
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
        mimic : {'Dim', 'askia'}, default 'Dim'
            It is possible to mimic the test logics used in other statistical
            software packages by passing them as instructions. The method will
            then choose the appropriate test parameters.
        level: {'high', 'mid', 'low'} or float
            Sets the level of significance to which the test is carried out.
            Given as str the levels correspond to ``'high'`` = 0.01, ``'mid'``
            = 0.05 and ``'low'`` = 0.1. If a float is passed the specified
            level will be used.

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
        func_name = 'coltests'
        func_type = 'column differences tests'
        view = View(link, kwargs=kwargs)
        pos, relation, rel_to, weights, text = view.std_params()

        cache = self._cache = link.get_cache()

        metric = kwargs.get('metric', 'props')
        mimic = kwargs.get('mimic', 'Dim')
        level = kwargs.get('level', 'low')
        stack = link.stack

        get = 'count' if metric == 'props' else 'mean'
        views = self._get_view_names(cache, stack, weights, get=get)
        for in_view in views:             
            try:
                view = View(link, kwargs=kwargs)
                relation = in_view.split('|')[2]                
                test = qp.Test(link, in_view)
                if mimic == 'Dim':
                    test.set_params(level=level)
                elif mimic == 'askia':
                    test.set_params(testtype='unpooled',
                                    level=level, mimic=mimic,
                                    use_ebase=False,
                                    ovlp_correc=False,
                                    cwi_filter=True)
                view_df = test.run()
                siglevel = test.level
                notation = tools.view.query.set_fullname(
                    pos,
                    '%s.%s.%s.%s' % ('tests',
                                     metric,
                                     mimic,
                                     "{:.2f}".format(siglevel)[2:]),
                    relation, rel_to, weights, name)
                view.dataframe = view_df
                view.name = notation
                link[notation] = view
            except:
                pass


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
            allviews = stack.describe(columns='view').index.tolist()
            if get == 'count':
                ignorenames = ['cbase', 'rbase', 'ebase']
                view_name_list = [v for v in allviews
                                  if v.split('|')[1] == 'frequency'
                                  and not v.split('|')[3]=='y'
                                  and not v.split('|')[-1] in ignorenames
                                  and v.split('|')[-2] == w]
            else:
                view_name_list = [v for v in allviews
                                  if v.split('|')[1] == 'mean'
                                  and v.split('|')[-2] == w]
            cache.set_obj(get+'_view_names', w+'_names', view_name_list)

        return view_name_list