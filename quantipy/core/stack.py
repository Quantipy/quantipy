
#-*- coding: utf-8 -*-
import io
import itertools
import json
import pandas as pd
import numpy as np
import quantipy as qp
import copy
import time
import sys
import warnings

from .link import Link
from .chain import Chain
from .view import View
from .helpers import functions
from .view_generators.view_mapper import ViewMapper
from .view_generators.view_maps import QuantipyViews
from quantipy.core.tools.qp_decorators import modify
from quantipy.core.tools.dp.spss.reader import parse_sav_file
from quantipy.core.tools.dp.io import unicoder, write_quantipy
from quantipy.core.tools.dp.prep import frequency, verify_test_results, frange
from quantipy.core.tools.view.logic import (
    has_any, has_all, has_count,
    not_any, not_all, not_count,
    is_lt, is_ne, is_gt,
    is_le, is_eq, is_ge,
    union, intersection, get_logic_index)
from .cache import Cache

import itertools
from collections import defaultdict, OrderedDict

# Pickle modules
import pickle

# Compression methods
import gzip

from quantipy.sandbox.sandbox import Chain as NewChain
from quantipy.sandbox.sandbox import ChainManager


class Stack(defaultdict):
    """
    Container of quantipy.Link objects holding View objects.

    A Stack is nested dictionary that structures the data and variable
    relationships storing all View aggregations performed.
    """

    def __init__(self,
                 name="",
                 add_data=None):

        super(Stack, self).__init__(Stack)

        self.name = name
        self.key = None
        self.parent = None

        # This is the root of the stack
        # It is used by the get/set methods to determine
        # WHERE in the stack those methods are.
        self.stack_pos = "stack_root"

        self.x_variables = None
        self.y_variables = None

        self.__view_keys = []

        if add_data:
            for key in add_data:
                if isinstance(add_data[key], dict):
                    self.add_data(
                        data_key=key,
                        data=add_data[key].get('data', None),
                        meta=add_data[key].get('meta', None)
                    )
                elif isinstance(add_data[key], tuple):
                    self.add_data(
                        data_key=key,
                        data=add_data[key][0],
                        meta=add_data[key][1]
                    )
                else:
                    raise TypeError(
                        "All data_key values must be one of the following types: "
                        "<dict> or <tuple>. "
                        "Given: %s" % (type(add_data[key]))
                    )

    def __setstate__(self, attr_dict):
        self.__dict__.update(attr_dict)

    def __reduce__(self):
        arguments = (self.name, )
        state = self.__dict__.copy()
        if 'cache' in state:
            state.pop('cache')
            state['cache'] = Cache() # Empty the cache for storage
        return self.__class__, arguments, state, None, iter(self.items())

    def __setitem__(self, key, val):
        """ The 'set' method for the Stack(dict)

            It 'sets' the value in it's correct place in the Stack
            AND applies a 'stack_pos' value depending on WHERE in
            the stack the value is being placed.
        """
        super(Stack, self).__setitem__(key, val)

        # The 'meta' portion of the stack is a standar dict (not Stack)
        try:
            if isinstance(val, Stack) and val.stack_pos is "stack_root":
                val.parent = self
                val.key = key

                # This needs to be compacted and simplified.
                if self.stack_pos is "stack_root":
                    val.stack_pos = "data_root"
                elif self.stack_pos is "data_root":
                    val.stack_pos = "filter"
                elif self.stack_pos is "filter":
                    val.stack_pos = "x"

        except AttributeError:
            pass

    def __getitem__(self, key):
        """ The 'get' method for the Stack(dict)

            The method 'gets' a value from the stack. If 'stack_pos' is 'y'
            AND the value isn't a Link instance THEN it tries to query the
            stack again with the x/y variables swapped and IF that yelds
            a result that is a Link object THEN it sets a 'transpose' variable
            as True in the result and the result is transposed.
        """
        val = defaultdict.__getitem__(self, key)
        return val

    def add_data(self, data_key, data=None, meta=None, ):
        """
        Sets the data_key into the stack, optionally mapping data sources it.

        It is possible to handle the mapping of data sources in different ways:

        * no meta or data (for proxy links not connected to source data)
        * meta only (for proxy links with supporintg meta)
        * data only (meta will be inferred if possible)
        * data and meta

        Parameters
        ----------
        data_key : str
            The reference name for a data source connected to the Stack.
        data : pandas.DataFrame
            The input (case) data source.
        meta : dict or OrderedDict
            A quantipy compatible metadata source that describes the case data.

        Returns
        -------
        None
        """
        self._verify_key_types(name='data', keys=data_key)

        if data_key in list(self.keys()):
            warning_msg = "You have overwritten data/meta for key: ['%s']."
            print(warning_msg % (data_key))

        if data is not None:
            if isinstance(data, pd.DataFrame):
                if meta is None:
                    # To do: infer meta from DataFrame
                    meta = {'info': None, 'lib': None, 'sets': None,
                            'columns': None, 'masks': None}
                # Add a special column of 1s
                data['@1'] = np.ones(len(data.index))
                data.index = list(range(0, len(data.index)))
            else:
                raise TypeError(
                    "The 'data' given to Stack.add_data() must be one of the following types: "
                    "<pandas.DataFrame>"
                )

        if not meta is None:
            if isinstance(meta, (dict, OrderedDict)):
                # To do: verify incoming meta
                pass
            else:
                raise TypeError(
                    "The 'meta' given to Stack.add_data() must be one of the following types: "
                    "<dict>, <collections.OrderedDict>."
                )

        # Add the data key to the stack
        # self[data_key] = {}

        # Add the meta and data to the data_key position in the stack
        self[data_key].meta = meta
        self[data_key].data = data
        self[data_key].cache = Cache()
        self[data_key]['no_filter'].data = self[data_key].data

    def remove_data(self, data_keys):
        """
        Deletes the data_key(s) and associated data specified in the Stack.

        Parameters
        ----------
        data_keys : str or list of str
            The data keys to remove.

        Returns
        -------
        None
        """
        self._verify_key_types(name='data', keys=data_keys)
        if isinstance(data_keys, str):
            data_keys = [data_keys]
        for data_key in data_keys:
            del self[data_key]

    def variable_types(self, data_key, only_type=None, verbose=True):
        """
        Group variables by data types found in the meta.

        Parameters
        ----------
        data_key : str
            The reference name of a case data source hold by the Stack instance.
        only_type : {'int', 'float', 'single', 'delimited set', 'string',
                     'date', time', 'array'}, optional
            Will restrict the output to the given data type.

        Returns
        -------
        types : dict or list of str
            A summary of variable names mapped to their data types, in form of
            {type_name: [variable names]} or a list of variable names
            confirming only_type.
        """
        if self[data_key].meta['columns'] is None:
            return 'No meta attached to data_key: %s' %(data_key)
        else:
            types = {
                'int': [],
                'float': [],
                'single': [],
                'delimited set': [],
                'string': [],
                'date': [],
                'time': [],
                'array': []
            }
            not_found = []
            for col in self[data_key].data.columns:
                if not col in ['@1', 'id_L1', 'id_L1.1']:
                    try:
                        types[
                              self[data_key].meta['columns'][col]['type']
                             ].append(col)
                    except:
                        not_found.append(col)
            for mask in list(self[data_key].meta['masks'].keys()):
                types[self[data_key].meta['masks'][mask]['type']].append(mask)
            if not_found and verbose:
                print('%s not found in meta file. Ignored.' %(not_found))
            if only_type:
                return types[only_type]
            else:
                return types

    def apply_meta_edits(self, batch_name, data_key, filter_key=None,
                         freeze=False):
        """
        Take over meta_edits from Batch definitions.

        Parameters
        ----------
        batch_name: str
            Name of the Batch whose meta_edits are taken.
        data_key: str
            Accessing this metadata: ``self[data_key].meta``
            Batch definitions are takes from here and this metadata is modified.
        filter_key: str, default None
            Currently not implemented!
            Accessing this metadata: ``self[data_key][filter_key].meta``
            Batch definitions are takes from here and this metadata is modified.
        """
        if filter_key:
            raise NotImplementedError("'filter_key' is not implemented.")
        if freeze:
            self.freeze_master_meta(data_key)
        meta = self[data_key].meta
        batch = meta['sets']['batches'][batch_name]
        for name, e_meta in list(batch['meta_edits'].items()):
            if name == 'lib':
                continue
            elif name in meta['masks']:
                meta['masks'][name] = e_meta
                try:
                    lib = batch['meta_edits']['lib'][name]
                    meta['lib']['values'][name] = lib
                except:
                    pass
            else:
                meta['columns'][name] = e_meta
        meta['lib']['default text'] = batch['language']
        return None

    def freeze_master_meta(self, data_key, filter_key=None):
        """
        Save ``.meta`` in ``.master_meta`` for a defined data_key.

        Parameters
        ----------
        data_key: str
            Using: ``self[data_key]``
        filter_key: str, default None
            Currently not implemented!
            Using: ``self[data_key][filter_key]``
        """
        if filter_key:
            raise NotImplementedError("'filter_key' is not implemented.")
        self[data_key].master_meta = copy.deepcopy(self[data_key].meta)
        self[data_key].meta = copy.deepcopy(self[data_key].meta)
        return None

    def restore_meta(self, data_key, filter_key=None):
        """
        Restore the ``.master_meta`` for a defined data_key if it exists.

        Undo self.apply_meta_edits()

        Parameters
        ----------
        data_key: str
            Accessing this metadata: ``self[data_key].meta``
        filter_key: str, default None
            Currently not implemented!
            Accessing this metadata: ``self[data_key][filter_key].meta``
        """
        if filter_key:
            raise NotImplementedError("'filter_key' is not implemented.")
        try:
            self[data_key].meta = copy.deepcopy(self[data_key].master_meta)
        except:
            pass
        return None

    def get_chain(self, *args, **kwargs):

        if qp.OPTIONS['new_chains']:
                chain = ChainManager(self)
                chain = chain.get(*args, **kwargs)
                return chain
        else:
            def _get_chain(name=None, data_keys=None, filters=None, x=None, y=None,
                           views=None, orient_on=None, select=None,
                           rules=False, rules_weight=None, described=None):
                """
                Construct a "chain" shaped subset of Links and their Views from the Stack.

                A chain is a one-to-one or one-to-many relation with an orientation that
                defines from which axis (x or y) it is build.

                Parameters
                ----------
                name : str, optional
                    If not provided the name of the chain is generated automatically.
                data_keys, filters, x, y, views : str or list of str
                    Views will be added reflecting the order in ``views`` parameter. If
                    both ``x`` and ``y`` have multiple items, you must specify the
                    ``orient_on`` parameter.
                orient_on : {'x', 'y'}, optional
                     Must be specified if both ``x`` and ``y`` are lists of multiple
                     items.
                select : tbc.
                    :TODO: document this!

                Returns
                -------
                chain : Chain object instance
                """

                #Make sure all the given keys are in lists
                data_keys = self._force_key_as_list(data_keys)
                # filters = self._force_key_as_list(filters)
                views = self._force_key_as_list(views)

                #Make sure all the given keys are in lists
                x = self._force_key_as_list(x)
                y = self._force_key_as_list(y)

                if orient_on is None:
                    if len(x)==1:
                        orientation = 'x'
                    elif len(y)==1:
                        orientation = 'y'
                    else:
                        orientation = 'x'
                else:
                    orientation = orient_on
                if described is None:
                    described = self.describe()

                if isinstance(rules, bool):
                    if rules:
                        rules = ['x', 'y']
                    else:
                        rules = []

                if orient_on:
                    if x is None:
                        x = described['x'].drop_duplicates().values.tolist()
                    if y is None:
                        y = described['y'].drop_duplicates().values.tolist()
                    if views is None:
                        views = self._Stack__view_keys
                        views = [v for v in views if '|default|' not in v]
                    chains = self.__get_chains(
                        name=name,
                        data_keys=data_keys,
                        filters=filters,
                        x=x,
                        y=y,
                        views=views,
                        orientation=orient_on,
                        select=select,
                        rules=rules,
                        rules_weight=rules_weight)
                    return chains
                else:
                    chain = Chain(name)
                    found_views = []

                    #Make sure all the given keys are in lists
                    x = self._force_key_as_list(x)
                    y = self._force_key_as_list(y)

                    if data_keys is None:
                        # Apply lazy data_keys if none given
                        data_keys = list(self.keys())

                    the_filter = "no_filter" if filters is None else filters

                    if self.__has_list(data_keys):
                        for key in data_keys:
                            # Use describe method to get x keys if not supplied.
                            if x is None:
                                x_keys = described['x'].drop_duplicates().values.tolist()
                            else:
                                x_keys = x

                            # Use describe method to get y keys if not supplied.
                            if y is None:
                                y_keys = described['y'].drop_duplicates().values.tolist()
                            else:
                                y_keys = y

                             # Use describe method to get view keys if not supplied.
                            if views is None:
                                v_keys = described['view'].drop_duplicates().values.tolist()
                                v_keys = [v_key for v_key in v_keys if '|default|'
                                          not in v_key]
                            else:
                                v_keys = views

                            chain._derive_attributes(
                                key, the_filter, x_keys, y_keys, views, orientation=orientation)

                            # Apply lazy name if none given
                            if name is None:
                                chain._lazy_name()

                            for x_key in x_keys:
                                self._verify_key_exists(
                                    x_key,
                                    stack_path=[key, the_filter]
                                )

                                for y_key in y_keys:
                                    self._verify_key_exists(
                                        y_key,
                                        stack_path=[key, the_filter, x_key])


                                    try:
                                        base_text = self[key].meta['columns'][x_key]['properties']['base_text']
                                        if isinstance(base_text, str):
                                            if base_text.startswith(('Base:', 'Bas:')):
                                                base_text = base_text.split(':')[-1].lstrip()
                                        elif isinstance(base_text, dict):
                                            for text_key in list(base_text.keys()):
                                                if base_text[text_key].startswith(('Base:', 'Bas:')):
                                                    base_text[text_key] = base_text[text_key].split(':')[-1].lstrip()
                                        chain.base_text = base_text
                                    except:
                                        pass
                                    if views is None:
                                        chain[key][the_filter][x_key][y_key] = self[key][the_filter][x_key][y_key]
                                    else:
                                        stack_link = self[key][the_filter][x_key][y_key]
                                        link_keys = list(stack_link.keys())
                                        chain_link = {}
                                        chain_view_keys = [k for k in views if k in link_keys]
                                        for vk in chain_view_keys:
                                            stack_view = stack_link[vk]
                                            # Get view dataframe
                                            rules_x_slicer = self.axis_slicer_from_vartype(
                                                rules, 'x', key, the_filter, x_key, y_key, rules_weight)

                                            rules_y_slicer = self.axis_slicer_from_vartype(
                                                rules, 'y', key, the_filter, x_key, y_key, rules_weight)
                                            if rules_x_slicer is None and rules_y_slicer is None:
                                                # No rules to apply
                                                view_df = stack_view.dataframe
                                            else:
                                                # Apply rules
                                                viable_axes = functions.rule_viable_axes(self[key].meta, vk, x_key, y_key)
                                                transposed_array_sum = x_key == '@' and y_key in self[key].meta['masks']
                                                if not viable_axes:
                                                    # Axes are not viable for rules application
                                                    view_df = stack_view.dataframe
                                                else:
                                                    view_df = stack_view.dataframe.copy()
                                                    if 'x' in viable_axes and not rules_x_slicer is None:
                                                        # Apply x-rules
                                                        rule_codes = set(rules_x_slicer)
                                                        view_codes = set(view_df.index.tolist())
                                                        if not rule_codes - view_codes:
                                                            view_df = view_df.loc[rules_x_slicer]
                                                    if 'x' in viable_axes and transposed_array_sum and rules_y_slicer:
                                                        view_df = view_df.loc[rules_y_slicer]
                                                    if 'y' in viable_axes and not rules_y_slicer is None:
                                                        # Apply y-rules
                                                        view_df = view_df[rules_y_slicer]
                                                        if vk.split('|')[1].startswith('t.'):
                                                            view_df = verify_test_results(view_df)
                                            chain_view = View(
                                                link=stack_link,
                                                name = stack_view.name,
                                                kwargs=stack_view._kwargs)
                                            chain_view._notation = vk
                                            chain_view.grp_text_map = stack_view.grp_text_map
                                            chain_view.dataframe = view_df
                                            chain_view._custom_txt = stack_view._custom_txt
                                            chain_view.add_base_text = stack_view.add_base_text
                                            chain_link[vk] = chain_view
                                            if vk not in found_views:
                                                found_views.append(vk)

                                        chain[key][the_filter][x_key][y_key] = chain_link
                    else:
                        raise ValueError(
                            "One or more of your data_keys ({data_keys}) is not"
                            " in the stack ({stack_keys})".format(
                                data_keys=data_keys,
                                stack_keys=list(self.keys())
                            )
                        )

                    # Make sure chain.views only contains views that actually exist
                    # in the chain
                    if found_views:
                        chain.views = [
                            view
                            for view in chain.views
                            if view in found_views]
                    return chain

            return _get_chain(*args, **kwargs)

    def reduce(self, data_keys=None, filters=None, x=None, y=None, variables=None, views=None):
        '''
        Remove keys from the matching levels, erasing discrete Stack portions.

        Parameters
        ----------
        data_keys, filters, x, y, views : str or list of str

        Returns
        -------
        None
        '''

        # Ensure given keys are all valid types
        self._verify_multiple_key_types(
            data_keys=data_keys,
            filters=filters,
            x=x,
            y=y,
            variables=variables,
            views=views
        )

        # Make sure all the given keys are in lists
        data_keys = self._force_key_as_list(data_keys)
        filters = self._force_key_as_list(filters)
        views = self._force_key_as_list(views)
        if not variables is None:
            variables = self._force_key_as_list(variables)
            x = variables
            y = variables
        else:
            x = self._force_key_as_list(x)
            y = self._force_key_as_list(y)

        # Make sure no keys that don't exist anywhere were passed
        key_check = {
            'data': data_keys,
            'filter': filters,
            'x': x,
            'y': y,
            'view': views
        }

        contents = self.describe()
        for key_type, keys in key_check.items():
            if not keys is None:
                uk = contents[key_type].unique()
                if not any([tk in uk for tk in keys]):
                    raise ValueError(
                        "Some of the %s keys passed to stack.reduce() "
                        "weren't found. Found: %s. "
                        "Given: %s" % (key_type, uk, keys)
                    )

        if not data_keys is None:
            for dk in data_keys:
                try:
                    del self[dk]
                except:
                    pass

        for dk in list(self.keys()):
            if not filters is None:
                for fk in filters:
                    try:
                        del self[dk][fk]
                    except:
                        pass

            for fk in list(self[dk].keys()):
                if not x is None:
                    for xk in x:
                        try:
                            del self[dk][fk][xk]
                        except:
                            pass

                for xk in list(self[dk][fk].keys()):
                    if not y is None:
                        for yk in y:
                            try:
                                del self[dk][fk][xk][yk]
                            except:
                                pass

                    for yk in list(self[dk][fk][xk].keys()):
                        if not views is None:
                            for vk in views:
                                try:
                                    del self[dk][fk][xk][yk][vk]
                                except:
                                    pass

    def add_link(self, data_keys=None, filters=['no_filter'], x=None, y=None,
                 views=None, weights=None, variables=None):
        """
        Add Link and View defintions to the Stack.

        The method can be used flexibly: It is possible to pass only Link
        defintions that might be composed of filter, x and y specifications,
        only views incl. weight variable selections or arbitrary combinations of
        the former.

        :TODO: Remove ``variables`` from parameter list and method calls.

        Parameters
        ----------
        data_keys : str, optional
            The data_key to be added to. If none is given, the method will try
            to add to all data_keys found in the Stack.
        filters : list of str describing filter defintions, default ['no_filter']
            The string must be a valid input for the
            pandas.DataFrame.query() method.
        x, y : str or list of str
            The x and y variables to constrcut Links from.
        views : list of view method names.
            Can be any of Quantipy's preset Views or the names of created
            view method specifications.
        weights : list, optional
            The names of weight variables to consider in the data aggregation
            process. Weight variables must be of type ``float``.

        Returns
        -------
        None
        """
        if data_keys is None:
            data_keys = list(self.keys())
        else:
            self._verify_key_types(name='data', keys=data_keys)
            data_keys = self._force_key_as_list(data_keys)

        if not isinstance(views, ViewMapper):
            # Use DefaultViews if no view were given
            if views is None:
                pass
            #    views = DefaultViews()
            elif isinstance(views, (list, tuple)):
                views = QuantipyViews(views=views)
            else:
                raise TypeError(
                    "The views past to stack.add_link() must be type <quantipy.view_generators.ViewMapper>, "
                    "or they must be a list of method names known to <quantipy.view_generators.QuantipyViews>."
                )

        qplogic_filter = False
        if not isinstance(filters, dict):
            self._verify_key_types(name='filter', keys=filters)
            filters = self._force_key_as_list(filters)
            filters = {f: f for f in filters}
            # if filters.keys()[0] != 'no_filter':
            #     msg = ("Warning: pandas-based filtering will be deprecated in the "
            #           "future!\nPlease switch to quantipy-logic expressions.")
            #     print UserWarning(msg)
        else:
            qplogic_filter = True

        if not variables is None:
            if not x is None or not y is None:
                raise ValueError(
                    "You cannot pass both 'variables' and 'x' and/or 'y' to stack.add_link() "
                    "at the same time."
                )
        x = self._force_key_as_list(x)
        y = self._force_key_as_list(y)

        # Get the lazy y keys none were given and there is only 1 x key
        if not x is None:
            if len(x)==1 and y is None:
                y = self.describe(
                    index=['y'],
                    query="x=='%s'" % (x[0])
                ).index.tolist()

        # Get the lazy x keys none were given and there is only 1 y key
        if not y is None:
            if len(y)==1 and x is None:
                x = self.describe(
                    index=['x'],
                    query="y=='%s'" % (y[0])
                ).index.tolist()

        for dk in data_keys:
            self._verify_key_exists(dk)
            for filter_def, logic in list(filters.items()):
                # qp.OPTIONS-based hack to allow faster stack filters
                # ------------------------------------------------------
                if qp.OPTIONS['fast_stack_filters']:
                    if not filter_def in list(self[dk].keys()):
                        if filter_def=='no_filter':
                            self[dk][filter_def].data = self[dk].data
                            self[dk][filter_def].meta = self[dk].meta
                        else:
                            if not qplogic_filter:
                                try:
                                    self[dk][filter_def].data = self[dk].data.query(logic)
                                    self[dk][filter_def].meta = self[dk].meta
                                except Exception as ex:
                                    raise UserWarning('A filter definition is invalid and will be skipped: {filter_def}'.format(filter_def=filter_def))
                                    continue
                            else:
                                dataset = qp.DataSet('stack')
                                dataset.from_components(self[dk].data, self[dk].meta,
                                                        reset=False)
                                f_dataset = dataset.filter(filter_def, logic, inplace=False)
                                self[dk][filter_def].data = f_dataset._data
                                self[dk][filter_def].meta = f_dataset._meta
                else:
                    if filter_def=='no_filter':
                        self[dk][filter_def].data = self[dk].data
                        self[dk][filter_def].meta = self[dk].meta
                    else:
                        if not qplogic_filter:
                            try:
                                self[dk][filter_def].data = self[dk].data.query(logic)
                                self[dk][filter_def].meta = self[dk].meta
                            except Exception as ex:
                                raise UserWarning('A filter definition is invalid and will be skipped: {filter_def}'.format(filter_def=filter_def))
                                continue
                        else:
                            dataset = qp.DataSet('stack')
                            dataset.from_components(self[dk].data, self[dk].meta)
                            f_dataset = dataset.filter(filter_def, logic, inplace=False)
                            self[dk][filter_def].data = f_dataset._data
                            self[dk][filter_def].meta = f_dataset._meta
                fdata = self[dk][filter_def].data

                if len(fdata) == 0:
                    raise UserWarning('A filter definition resulted in no cases and will be skipped: {filter_def}'.format(filter_def=filter_def))
                    continue
                self.__create_links(data=fdata, data_key=dk, the_filter=filter_def, x=x, y=y, views=views, weights=weights, variables=variables)

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
        for dk in list(self.keys()):
            path_dk = [dk]
            filters = self[dk]

#             for fk in filters.keys():
#                 path_fk = path_dk + [fk]
#                 xs = self[dk][fk]

            for fk in list(filters.keys()):
                path_fk = path_dk + [fk]
                xs = self[dk][fk]

                for sk in list(xs.keys()):
                    path_sk = path_fk + [sk]
                    ys = self[dk][fk][sk]

                    for tk in list(ys.keys()):
                        path_tk = path_sk + [tk]
                        views = self[dk][fk][sk][tk]

                        if list(views.keys()):
                            for vk in list(views.keys()):
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

    def refresh(self, data_key, new_data_key='', new_weight=None,
                new_data=None, new_meta=None):
        """
        Re-run all or a portion of Stack's aggregations for a given data key.

        refresh() can be used to re-weight the data using a new case data
        weight variable or to re-run all aggregations based on a changed source
        data version (e.g. after cleaning the file/ dropping cases) or a
        combination of the both.

        .. note::
            Currently this is only supported for the preset QuantipyViews(),
            namely: ``'cbase'``, ``'rbase'``, ``'counts'``, ``'c%'``,
            ``'r%'``, ``'mean'``, ``'ebase'``.

        Parameters
        ----------
        data_key : str
            The Links' data key to be modified.
        new_data_key : str, default ''
            Controls if the existing data key's files and aggregations will be
            overwritten or stored via a new data key.
        new_weight : str
            The name of a new weight variable used to re-aggregate the Links.
        new_data : pandas.DataFrame
            The case data source. If None is given, the
            original case data found for the data key will be used.
        new_meta : quantipy meta document
            A meta data source associated with the case data. If None is given,
            the original meta definition found for the data key will be used.

        Returns
        -------
        None
        """
        content = self.describe()[['data', 'filter', 'x', 'y', 'view']]
        content = content[content['data'] == data_key]
        put_meta = self[data_key].meta if new_meta is None else new_meta
        put_data = self[data_key].data if new_data is None else new_data
        dk = new_data_key if new_data_key else data_key
        self.add_data(data_key=dk, data=put_data, meta=put_meta)
        skipped_views = []
        for _, f, x, y, view in content.values:
            shortname = view.split('|')[-1]
            if shortname not in ['default', 'cbase', 'cbase_gross',
                                 'rbase', 'counts', 'c%',
                                 'r%', 'ebase', 'mean',
                                 'c%_sum', 'counts_sum']:
                if view not in skipped_views:
                    skipped_views.append(view)
                    warning_msg = ('\nOnly preset QuantipyViews are supported.'
                                   'Skipping: {}').format(view)
                    print(warning_msg)
            else:
                view_weight = view.split('|')[-2]
                if not x in [view_weight, new_weight]:
                    if new_data is None and new_weight is not None:
                        if not view_weight == '':
                            if new_weight == '':
                                weight = [None, view_weight]
                            else:
                                weight = [view_weight, new_weight]
                        else:
                            if new_weight == '':
                                weight = None
                            else:
                                weight = [None, new_weight]
                        self.add_link(data_keys=dk, filters=f, x=x, y=y,
                                      weights=weight, views=[shortname])
                    else:
                        if view_weight == '':
                            weight = None
                        elif new_weight is not None:
                            if not (view_weight == new_weight):
                                if new_weight == '':
                                    weight = [None, view_weight]
                                else:
                                    weight = [view_weight, new_weight]
                            else:
                                weight = view_weight
                        else:
                            weight = view_weight
                        try:
                            self.add_link(data_keys=dk, filters=f, x=x, y=y,
                                          weights=weight, views=[shortname])
                        except ValueError as e:
                            print('\n', e)
        return None

    def save(self, path_stack, compression="gzip", store_cache=True,
             decode_str=False, dataset=False, describe=False):
        """
        Save Stack instance to .stack file.

        Parameters
        ----------
        path_stack : str
            The full path to the .stack file that should be created, including
            the extension.
        compression : {'gzip'}, default 'gzip'
            The intended compression type.
        store_cache : bool, default True
            Stores the MatrixCache in a file in the same location.
        decode_str : bool, default=True
            If True the unicoder function will be used to decode all str
            objects found anywhere in the meta document/s.
        dataset : bool, default=False
            If True a json/csv will be saved parallel to the saved stack
            for each data key in the stack.
        describe : bool, default=False
            If True the result of stack.describe().to_excel() will be
            saved parallel to the saved stack.

        Returns
        -------
        None
        """
        protocol = pickle.HIGHEST_PROTOCOL
        if not path_stack.endswith('.stack'):
            raise ValueError(
                "To avoid ambiguity, when using Stack.save() you must provide the full path to "
                "the stack file you want to create, including the file extension. For example: "
                "stack.save(path_stack='./output/MyStack.stack'). Your call looks like this: "
                "stack.save(path_stack='%s', ...)" % (path_stack)
            )

        # Make sure there are no str objects in any meta documents. If
        # there are any non-ASCII characters will be encoded
        # incorrectly and lead to UnicodeDecodeErrors in Jupyter.
        if decode_str:
            for dk in list(self.keys()):
                self[dk].meta = unicoder(self[dk].meta)

        if compression is None:
            f = open(path_stack, 'wb')
            pickle.dump(self, f, protocol)
        else:
            f = gzip.open(path_stack, 'wb')
            pickle.dump(self, f, protocol)

        if store_cache:
            caches = {}
            for key in list(self.keys()):
                caches[key] = self[key].cache

            path_cache = path_stack.replace('.stack', '.cache')
            if compression is None:
                f1 = open(path_cache, 'wb')
                pickle.dump(caches, f1, protocol)
            else:
                f1 = gzip.open(path_cache, 'wb')
                pickle.dump(caches, f1, protocol)

            f1.close()

        f.close()

        if dataset:
            for key in list(self.keys()):
                path_json = path_stack.replace(
                    '.stack',
                    ' [{}].json'.format(key))
                path_csv = path_stack.replace(
                    '.stack',
                    ' [{}].csv'.format(key))
                write_quantipy(
                    meta=self[key].meta,
                    data=self[key].data,
                    path_json=path_json,
                    path_csv=path_csv)

        if describe:
            path_describe = path_stack.replace('.stack', '.xlsx')
            self.describe().to_excel(path_describe)

    # def get_slice(data_key=None, x=None, y=None, filters=None, views=None):
    #     """  """
    #     pass

    # STATIC METHODS

    @staticmethod
    def from_sav(data_key, filename, name=None, path=None, ioLocale="en_US.UTF-8", ioUtf8=True):
        """
        Creates a new stack instance from a .sav file.

        Parameters
        ----------
        data_key : str
            The data_key for the data and meta in the sav file.
        filename : str
            The name to the sav file.
        name : str
            A name for the sav (stored in the meta).
        path : str
            The path to the sav file.
        ioLocale : str
            The locale used in during the sav processing.
        ioUtf8 : bool
            Boolean that indicates the mode in which text communicated to or
            from the I/O module will be.

        Returns
        -------
        stack : stack object instance
            A stack instance that has a data_key with data and metadata
            to run aggregations.
        """
        if name is None:
            name = data_key

        meta, data = parse_sav_file(filename=filename, path=path, name=name, ioLocale=ioLocale, ioUtf8=ioUtf8)
        return Stack(add_data={name: {'meta': meta, 'data':data}})

    @staticmethod
    def load(path_stack, compression="gzip", load_cache=False):
        """
        Load Stack instance from .stack file.

        Parameters
        ----------
        path_stack : str
            The full path to the .stack file that should be created, including
            the extension.
        compression : {'gzip'}, default 'gzip'
            The compression type that has been used saving the file.
        load_cache : bool, default False
            Loads MatrixCache into the Stack a .cache file is found.

        Returns
        -------
        None
        """


        if not path_stack.endswith('.stack'):
            raise ValueError(
                "To avoid ambiguity, when using Stack.load() you must provide the full path to "
                "the stack file you want to create, including the file extension. For example: "
                "stack.load(path_stack='./output/MyStack.stack'). Your call looks like this: "
                "stack.load(path_stack='%s', ...)" % (path_stack)
            )

        if compression is None:
            f = open(path_stack, 'rb')
        else:
            f = gzip.open(path_stack, 'rb')
        new_stack = pickle.load(f)
        f.close()

        if load_cache:
            path_cache = path_stack.replace('.stack', '.cache')
            if compression is None:
                f = open(path_cache, 'rb')
            else:
                f = gzip.open(path_cache, 'rb')
            caches = pickle.load(f)
            for key in list(caches.keys()):
                if key in list(new_stack.keys()):
                    new_stack[key].cache = caches[key]
                else:
                    raise ValueError(
                        "Tried to insert a loaded MatrixCache in to a data_key in the stack that"
                        "is not in the stack. The data_key is '{}', available keys are {}"
                        .format(key, list(caches.keys()))
                    )
            f.close()

        return new_stack


    # PRIVATE METHODS

    def __get_all_y_keys(self, data_key, the_filter="no_filter"):
        if(self.stack_pos == 'stack_root'):
            return self[data_key].y_variables
        else:
            raise KeyError("get_all_y_keys can only be called from a stack at root level. Current level is '{0}'".format(self.stack_pos))

    def __get_all_x_keys(self, data_key, the_filter="no_filter"):
        if(self.stack_pos == 'stack_root'):
            return self[data_key].x_variables
        else:
            raise KeyError("get_all_x_keys can only be called from a stack at root level. Current level is '{0}'".format(self.stack_pos))

    def __get_all_x_keys_except(self, data_key, exception):
        keys = self.__get_all_x_keys(data_key)
        return [i for i in keys if i != exception[0]]

    def __get_all_y_keys_except(self, data_key, exception):
        keys = self.__get_all_y_keys(data_key)
        return [i for i in keys if i != exception[0]]

    def __set_x_key(self, key):
        if self.x_variables is None:
            self.x_variables = set(key)
        else:
            self.x_variables.update(key)

    def __set_y_key(self, key):
        if self.y_variables is None:
            self.y_variables = set(key)
        else:
            self.y_variables.update(key)

    def _set_x_and_y_keys(self, data_key, x, y):
        """
        Sets the x_variables and y_variables in the data part of the stack for this data_key, e.g. stack['Jan'].
        This method can also be used to add to the current lists and it makes sure the list stays unique.
        """
        if self.stack_pos == 'stack_root':
            self[data_key].__set_x_key(x)
            self[data_key].__set_y_key(y)
        else:
            raise KeyError("set_x_keys can only be called from a stack at root level. Current level is '{0}'".format(self.stack_pos))

    def __create_combinations(self, data, data_key, x=None, y=None, weight=None, variables=None):
        if isinstance(y, str):
            y = [y]
        if isinstance(x, str):
            x = [x]

        has_metadata = self[data_key].meta is not None and not isinstance(self[data_key].meta, Stack)

        # any(...) returns true if ANY of the vars are not None
        if any([x, y]) and variables is not None:
            # Raise an error if variables AND x/y are BOTH supplied
            raise ValueError("Either use the 'variables' OR 'x', 'y' NOT both.")

        if not any([x, y]):
            if variables is None:
                if not has_metadata:
                    # "fully-lazy" method. (variables, x and y are all None)
                    variables = data.columns.tolist()

        if variables is not None:
            x = variables
            y = variables
            variables = None

        # Ensure that we actually have metadata
        if has_metadata:
            # THEN we try to create the combinations with metadata
            combinations = self.__create_combinations_with_meta(data=data, data_key=data_key, x=x, y=y, weight=weight)
        else:
            # Either variables or both x AND y are supplied. Then create the combinations from that.
            combinations = self.__create_combinations_no_meta(data=data, data_key=data_key, x=x, y=y, weight=weight)

        unique_list = set([item for comb in combinations for item in comb])

        return combinations, unique_list

    def __create_combinations_with_meta(self, data, data_key, x=None, y=None, weight=None):
        # TODO: These meta functions should possibly be in the helpers functions
        metadata_columns = list(self[data_key].meta['columns'].keys())
        for mask, mask_data in self[data_key].meta['masks'].items():
            # TODO :: Get the static list from somewhere. not hardcoded.
            if mask_data['type'].lower() in ['array', 'dichotomous set',
                                             "categorical set"]:
                metadata_columns.append(mask)
                for item in mask_data['items']:
                    if "source" in item:
                        column = item["source"].split('@')[1]
                        metadata_columns.remove(column)
            elif mask_data['type'].lower() in ["overlay"]:
                pass
        # Use all from the metadata, if nothing is specified (fully-lazy)
        if x is None and y is None:
            x = metadata_columns
            y = metadata_columns
        if all([x, y]):
            metadata_columns = list(set(metadata_columns + x + y))
        elif x is not None:
            metadata_columns = list(set(metadata_columns + x))
        elif y is not None:
            metadata_columns = list(set(metadata_columns + y))
        combinations = functions.create_combinations_from_array(sorted(metadata_columns))

        for var in [x, y]:
            if var is not None:
                if weight in var:
                    var.remove(weight)
        if all([x, y]):
            combinations = [(x_item, y_item) for x_item, y_item in combinations
                            if x_item in x and y_item in y]
        elif x is not None:
            combinations = [(x_item, y_item) for x_item, y_item in combinations
                            if x_item in x]
        elif y is not None:
            combinations = [(x_item, y_item) for x_item, y_item in combinations
                            if y_item in y]

        return combinations

    def __create_combinations_no_meta(self, data, data_key, x=None, y=None, weight=None):
        if x is None:
            x = data.columns.tolist()
        if y is None:
            y = data.columns.tolist()
        for var in [x, y]:
            if weight in var:
                var.remove(weight)
        combinations = [(x_item, y_item) for x_item in x for y_item
                        in y if x_item != y_item]
        self._set_x_and_y_keys(data_key, x, y)

        return combinations

    def __create_links(self, data, data_key, views, variables=None, x=None, y=None,
                       the_filter=None, store_view_in_link=False, weights=None):
        if views is not None:
            has_links = True if list(self[data_key][the_filter].keys()) else False
            if has_links:
                xs = list(self[data_key][the_filter].keys())
                if x is not None:
                    valid_x = [xk for xk in xs if xk in x]
                    valid_x.extend(x)
                    x = set(valid_x)
                else:
                    x = xs
                ys = list(set(itertools.chain.from_iterable(
                    [list(self[data_key][the_filter][xk].keys())
                     for xk in xs])))
                if y is not None:
                    valid_y = [yk for yk in ys if yk in y]
                    valid_y.extend(y)
                    y = set(valid_y)
                else:
                    y = ys
        if self._x_and_y_keys_in_file(data_key, data, x, y):
            for x_key, y_key in itertools.product(x, y):
                if x_key==y_key and x_key=='@':
                    continue
                if y_key == '@':
                    if not isinstance(self[data_key][the_filter][x_key][y_key], Link):
                        link = Link(
                                    the_filter=the_filter,
                                    x=x_key,
                                    y='@',
                                    data_key=data_key,
                                    stack=self,
                                    store_view=store_view_in_link,
                                    create_views=False
                                    )
                        self[data_key][the_filter][x_key]['@'] = link
                    else:
                        link = self[data_key][the_filter][x_key]['@']
                elif x_key == '@':
                    if not isinstance(self[data_key][the_filter][x_key][y_key], Link):
                        link = Link(
                                    the_filter=the_filter,
                                    x='@',
                                    y=y_key,
                                    data_key=data_key,
                                    stack=self,
                                    store_view=store_view_in_link,
                                    create_views=False
                                    )
                        self[data_key][the_filter]['@'][y_key] = link
                    else:
                        link = self[data_key][the_filter]['@'][y_key]
                else:
                    if not isinstance(self[data_key][the_filter][x_key][y_key], Link):
                        link = Link(
                                    the_filter=the_filter,
                                    x=x_key,
                                    y=y_key,
                                    data_key=data_key,
                                    stack=self,
                                    store_view=store_view_in_link,
                                    create_views=False
                                    )
                        self[data_key][the_filter][x_key][y_key] = link
                    else:
                        link = self[data_key][the_filter][x_key][y_key]
                if views is not None:
                    views._apply_to(link, weights)

    def _x_and_y_keys_in_file(self, data_key, data, x, y):
        data_columns = data.columns.tolist()
        if '>' in ','.join(y): y = self._clean_from_nests(y)
        if '>' in ','.join(x):
            raise NotImplementedError('x-axis Nesting not supported.')
        x_not_found = [var for var in x if not var in data_columns
                       and not var == '@']
        y_not_found = [var for var in y if not var in data_columns
                       and not var == '@']
        if x_not_found is not None:
            masks_meta_lookup_x = [var for var in x_not_found
                                   if var in list(self[data_key].meta['masks'].keys())]
            for found_in_meta in masks_meta_lookup_x:
                x_not_found.remove(found_in_meta)
        if y_not_found is not None:
            masks_meta_lookup_y = [var for var in y_not_found
                                   if var in list(self[data_key].meta['masks'].keys())]
            for found_in_meta in masks_meta_lookup_y:
                y_not_found.remove(found_in_meta)
        if not x_not_found and not y_not_found:
            return True
        elif x_not_found and y_not_found:
            raise ValueError(
                'data key {}: x: {} and y: {} not found.'.format(
                    data_key, x_not_found, y_not_found))
        elif x_not_found:
            raise ValueError(
                'data key {}: x: {} not found.'.format(
                    data_key, x_not_found))
        elif y_not_found:
            raise ValueError(
                'data key {}: y: {} not found.'.format(
                    data_key, y_not_found))

    def _clean_from_nests(self, variables):
        cleaned = []
        nests = [var for var in variables if '>' in var]
        non_nests = [var for var in variables if not '>' in var]
        for nest in nests:
            cleaned.extend([var.strip() for var in nest.split('>')])
        non_nests += cleaned
        non_nests = list(set(non_nests))
        return non_nests

    def __clean_column_names(self, columns):
        """
        Remove extra doublequotes if there are any
        """
        cols = []
        for column in columns:
            cols.append(column.replace('"', ''))
        return cols

    def __generate_key_from_list_of(self, list_of_keys):
        """
        Generate keys from a list (or tuple).
        """
        list_of_keys = list(list_of_keys)
        list_of_keys.sort()
        return ",".join(list_of_keys)

    def __has_list(self, small):
        """
        Check if object contains a list of strings.
        """
        keys = list(self.keys())
        for i in range(len(keys)-len(small)+1):
            for j in range(len(small)):
                if keys[i+j] != small[j]:
                    break
            else:
                return i, i+len(small)
        return False

    def __get_all_combinations(self, list_of_items):
        """Generates all combinations of items from a list """
        return [itertools.combinations(list_of_items, index+1)
                for index in range(len(list_of_items))]

    def __get_stack_pointer(self, stack_pos):
        """Takes a stack_pos and returns the stack with that location
            raises an exception IF the stack pointer is not found
        """
        if self.parent.stack_pos == stack_pos:
            return self.parent
        else:
            return self.parent.__get_stack_pointer(stack_pos)

    def __get_chains(self, name, data_keys, filters, x, y, views,
                     orientation, select, rules,
                     rules_weight):
        """
        List comprehension wrapper around .get_chain().
        """
        if orientation == 'y':
            return [
                self.get_chain(
                    name=name,
                    data_keys=data_keys,
                    filters=filters,
                    x=x,
                    y=y_var,
                    views=views,
                    select=select,
                    rules=rules,
                    rules_weight=rules_weight
                )
                for y_var in y
            ]
        elif orientation == 'x':
            return [
                self.get_chain(
                    name=name,
                    data_keys=data_keys,
                    filters=filters,
                    x=x_var,
                    y=y,
                    views=views,
                    select=select,
                    rules=rules,
                    rules_weight=rules_weight
                )
                for x_var in x
            ]
        else:
            raise ValueError(
                "Unknown orientation type. Please use 'x' or 'y'."
            )

    def _verify_multiple_key_types(self, data_keys=None, filters=None, x=None,
                                   y=None, variables=None, views=None):
        """
        Verify that the given keys str or unicode or a list or tuple of those.
        """
        if data_keys is not None:
            self._verify_key_types(name='data', keys=data_keys)

        if filters is not None:
            self._verify_key_types(name='filter', keys=filters)

        if x is not None:
            self._verify_key_types(name='x', keys=x)

        if y is not None:
            self._verify_key_types(name='y', keys=y)

        if variables is not None:
            self._verify_key_types(name='variables', keys=variables)

        if views is not None:
            self._verify_key_types(name='view', keys=views)

    def _verify_key_exists(self, key, stack_path=[]):
        """
        Verify that the given key exists in the stack at the path targeted.
        """
        error_msg = (
            "Could not find the {key_type} key '{key}' in: {stack_path}. "
            "Found {keys_found} instead."
        )
        try:
            dk = stack_path[0]
            fk = stack_path[1]
            xk = stack_path[2]
            yk = stack_path[3]
            vk = stack_path[4]
        except:
            pass
        try:
            if len(stack_path) == 0:
                if key not in self:
                    key_type, keys_found = 'data', list(self.keys())
                    stack_path = 'stack'
                    raise ValueError
            elif len(stack_path) == 1:
                if key not in self[dk]:
                    key_type, keys_found = 'filter', list(self[dk].keys())
                    stack_path = "stack['{dk}']".format(
                        dk=dk)
                    raise ValueError
            elif len(stack_path) == 2:
                if key not in self[dk][fk]:
                    key_type, keys_found = 'x', list(self[dk][fk].keys())
                    stack_path = "stack['{dk}']['{fk}']".format(
                        dk=dk, fk=fk)
                    raise ValueError
            elif len(stack_path) == 3:
                meta = self[dk].meta
                if self._is_array_summary(meta, xk, None) and not key == '@':
                    pass
                elif key not in self[dk][fk][xk]:
                    key_type, keys_found = 'y', list(self[dk][fk][xk].keys())
                    stack_path = "stack['{dk}']['{fk}']['{xk}']".format(
                        dk=dk, fk=fk, xk=xk)
                    raise ValueError
            elif len(stack_path) == 4:
                if key not in self[dk][fk][xk][yk]:
                    key_type, keys_found = 'view', list(self[dk][fk][xk][yk].keys())
                    stack_path = "stack['{dk}']['{fk}']['{xk}']['{yk}']".format(
                        dk=dk, fk=fk, xk=xk, yk=yk)
                    raise ValueError
        except ValueError:
            print(error_msg.format(
                key_type=key_type,
                key=key,
                stack_path=stack_path,
                keys_found=keys_found
            ))

    def _force_key_as_list(self, key):
        """Returns key as [key] if it is str or unicode"""
        return [key] if isinstance(key, str) else key

    def _verify_key_types(self, name, keys):
        """
        Verify that the given keys str or unicode or a list or tuple of those.
        """
        if isinstance(keys, (list, tuple)):
            for key in keys:
                self._verify_key_types(name, key)
        elif isinstance(keys, str):
            pass
        else:
            raise TypeError(
                "All %s keys must be one of the following types: "
                "<str> or <unicode>, "
                "<list> of <str> or <unicode>, "
                "<tuple> of <str> or <unicode>. "
                "Given: %s" % (name, keys)
            )

    def _find_groups(self, view):
        groups = OrderedDict()
        logic = view._kwargs.get('logic')
        description = view.describe_block()
        groups['codes'] = [c for c, d in list(description.items()) if d == 'normal']
        net_names = [v for v, d in list(description.items()) if d == 'net']
        for l in logic:
            new_l = copy.deepcopy(l)
            for k in l:
                if k not in net_names:
                    del new_l[k]
            groups[list(new_l.keys())[0]] = list(new_l.values())[0]
        groups['codes'] = [c for c, d in list(description.items()) if d == 'normal']
        return groups

    def sort_expanded_nets(self, view, within=True, between=True, ascending=False,
                           fix=None):
        if not within and not between:
            return view.dataframe
        df = view.dataframe
        name = df.index.levels[0][0]
        if not fix:
            fix_codes = []
        else:
            if not isinstance(fix, list):
                fix_codes = [fix]
            else:
                fix_codes = fix
            fix_codes = [c for c in fix_codes if c in
                         df.index.get_level_values(1).tolist()]
        net_groups = self._find_groups(view)
        sort_col = (df.columns.levels[0][0], '@')
        sort = [(name, v) for v in df.index.get_level_values(1)
                if (v in net_groups['codes'] or
                v in list(net_groups.keys())) and not v in fix_codes]
        if between:
            if pd.__version__ == '0.19.2':
                temp_df = df.loc[sort].sort_values(sort_col, 0, ascending=ascending)
            else:
                temp_df = df.loc[sort].sort_index(0, sort_col, ascending=ascending)
        else:
            temp_df = df.loc[sort]
        between_order = temp_df.index.get_level_values(1).tolist()
        code_group_list = []
        for g in between_order:
            if g in net_groups:
                code_group_list.append([g] + net_groups[g])
            elif g in net_groups['codes']:
                code_group_list.append([g])
        final_index = []
        for g in code_group_list:
            is_code = len(g) == 1
            if not is_code:
                fixed_net_name = g[0]
                sort = [(name, v) for v in g[1:]]
                if within:
                    if pd.__version__ == '0.19.2':
                        temp_df = df.loc[sort].sort_values(sort_col, 0, ascending=ascending)
                    else:
                        temp_df = df.loc[sort].sort_index(0, sort_col, ascending=ascending)
                else:
                    temp_df = df.loc[sort]
                new_idx = [fixed_net_name] + temp_df.index.get_level_values(1).tolist()
                final_index.extend(new_idx)
            else:
                final_index.extend(g)
        final_index = [(name, i) for i in final_index]
        if fix_codes:
            fix_codes = [(name, f) for f in fix_codes]
            final_index.extend(fix_codes)
        df = df.reindex(final_index)
        return df

    def get_frequency_via_stack(self, data_key, the_filter, col, weight=None):
        weight_notation = '' if weight is None else weight
        vk = 'x|f|:||{}|counts'.format(weight_notation)
        try:
            f = self[data_key][the_filter][col]['@'][vk].dataframe
        except (KeyError, AttributeError) as e:
            try:
                f = self[data_key][the_filter]['@'][col][vk].dataframe.T
            except (KeyError, AttributeError) as e:
                f = frequency(self[data_key].meta, self[data_key].data, x=col, weight=weight)
        return f

    def get_descriptive_via_stack(self, data_key, the_filter, col, weight=None):
        l = self[data_key][the_filter][col]['@']
        w = '' if weight is None else weight
        mean_key = [k for k in list(l.keys()) if 'd.mean' in k.split('|')[1] and
                    k.split('|')[-2] == w]
        if not mean_key:
            msg = "No mean view to sort '{}' on found!"
            raise RuntimeError(msg.format(col))
        elif len(mean_key) > 1:
            msg = "Multiple mean views found for '{}'. Unable to sort!"
            raise RuntimeError(msg.format(col))
        else:
            mean_key = mean_key[0]
        vk = mean_key
        d = l[mean_key].dataframe
        return d

    def _is_array_summary(self, meta, x, y):
        return x in meta['masks']

    def _is_transposed_summary(self, meta, x, y):
        return x == '@' and y in meta['masks']

    def axis_slicer_from_vartype(self, all_rules_axes, rules_axis, dk, the_filter, x, y, rules_weight):
        if rules_axis == 'x' and 'x' not in all_rules_axes:
            return None
        elif rules_axis == 'y' and 'y' not in all_rules_axes:
            return None
        meta = self[dk].meta

        array_summary = self._is_array_summary(meta, x, y)
        transposed_summary = self._is_transposed_summary(meta, x, y)

        axis_slicer = None

        if rules_axis == 'x':
            if not array_summary and not transposed_summary:
                axis_slicer = self.get_rules_slicer_via_stack(
                    dk, the_filter, x=x, weight=rules_weight)
            elif array_summary:
                axis_slicer = self.get_rules_slicer_via_stack(
                    dk, the_filter, x=x, y='@', weight=rules_weight,
                    slice_array_items=True)
            elif transposed_summary:
                axis_slicer = self.get_rules_slicer_via_stack(
                    dk, the_filter, x='@', y=y, weight=rules_weight)
        elif rules_axis == 'y':
            if not array_summary and not transposed_summary:
                axis_slicer = self.get_rules_slicer_via_stack(
                    dk, the_filter, y=y, weight=rules_weight)
            elif array_summary:
                axis_slicer = self.get_rules_slicer_via_stack(
                    dk, the_filter, x=x, y='@', weight=rules_weight,
                    slice_array_items=False)
            elif transposed_summary:
                axis_slicer = self.get_rules_slicer_via_stack(
                    dk, the_filter, x='@', y=y, weight=rules_weight)

        return axis_slicer

    def get_rules_slicer_via_stack(self, data_key, the_filter,
                                   x=None, y=None, weight=None,
                                   slice_array_items=False):
        m = self[data_key].meta
        array_summary = self._is_array_summary(m, x, y)
        transposed_summary = self._is_transposed_summary(m, x, y)

        rules = None

        if not array_summary and not transposed_summary:
            if not x is None:
                try:
                    rules = self[data_key].meta['columns'][x]['rules']['x']
                    col = x
                except:
                    pass
            elif not y is None:
                try:
                    rules = self[data_key].meta['columns'][y]['rules']['y']
                    col = y
                except:
                    pass

        elif array_summary:
            if slice_array_items:
                try:
                    rules = self[data_key].meta['masks'][x]['rules']['x']
                    col = x
                except:
                    pass
            else:
                try:
                    rules = self[data_key].meta['masks'][x]['rules']['y']
                    col = x
                except:
                    pass

        elif transposed_summary:
                try:
                    rules = self[data_key].meta['masks'][y]['rules']['x']
                    col = y
                except:
                    pass

        if not rules: return None
        views = list(self[data_key][the_filter][col]['@'].keys())
        w = '' if weight is None else weight
        expanded_net = [v for v in views if '}+]' in v
                        and v.split('|')[-2] == w
                        and v.split('|')[1] == 'f' and
                        not v.split('|')[3] == 'x']
        if expanded_net:
            if len(expanded_net) > 1:
                if len(expanded_net) == 2:
                    if expanded_net[0].split('|')[2] == expanded_net[1].split('|')[2]:
                        expanded_net = expanded_net[0]
                else:
                    msg = "Multiple 'expand' using views found for '{}'. Unable to sort!"
                    raise RuntimeError(msg.format(col))
            else:
                expanded_net = expanded_net[0]
        if 'sortx' in rules:
            on_mean = rules['sortx'].get('sort_on', '@') == 'mean'
        else:
            on_mean = False
        if 'sortx' in rules and on_mean:
            f = self.get_descriptive_via_stack(
                data_key, the_filter, col, weight=weight)
        elif 'sortx' in rules and expanded_net:
            within = rules['sortx'].get('within', False)
            between = rules['sortx'].get('between', False)
            fix = rules['sortx'].get('fixed', False)
            ascending = rules['sortx'].get('ascending', False)
            view = self[data_key][the_filter][col]['@'][expanded_net]
            f = self.sort_expanded_nets(view, between=between, within=within,
                                        ascending=ascending, fix=fix)
        else:
            f = self.get_frequency_via_stack(
                data_key, the_filter, col, weight=weight)

        if transposed_summary or (not slice_array_items and array_summary):
            rules_slicer = functions.get_rules_slicer(f.T, rules)
        else:
            if not expanded_net or ('sortx' in rules and on_mean):
                rules_slicer = functions.get_rules_slicer(f, rules)
            else:
                rules_slicer = f.index.values.tolist()
        try:
            rules_slicer.remove((col, 'All'))
        except:
            pass
        return rules_slicer

    @modify(to_list='batches')
    def _check_batches(self, dk, batches='all'):
        """
        Returns a list of valid ``qp.Batch`` names.

        Parameters
        ----------
        batches: str/ list of str, default 'all'
            Included names are checked against valid ``qp.Batch`` names. If
            batches='all', all valid ``Batch`` names are returned.

        Returns
        -------
        list of str
        """
        if not batches:
            return []
        elif batches[0] == 'all':
            return list(self[dk].meta['sets']['batches'].keys())
        else:
            valid = list(self[dk].meta['sets']['batches'].keys())
            not_valid = [b for b in batches if not b in valid]
            if not_valid:
                msg = '``Batch`` name not found in ``Stack``: {}'
                raise KeyError(msg.format(not_valid))
            return batches

    def _x_y_f_w_map(self, dk, batches='all'):
        """
        """
        def _append_loop(mapping, x, fi, w, ys):
            if fi: fi = fi.encode('utf8')
            fn = 'no_filter' if fi is None else fi
            f = 'no_filter' if fi is None else {fi: {fi: 0}}
            if not x in mapping:
                mapping[x] = {fn: {'f': f, tuple(w): ys}}
            elif not fn in mapping[x]:
                mapping[x][fn] = {'f': f, tuple(w): ys}
            elif not tuple(w) in mapping[x][fn]:
                mapping[x][fn][tuple(w)] = ys
            elif not all(y in mapping[x][fn][tuple(w)] for y in ys):
                yks = set(mapping[x][fn][tuple(w)]).union(set(ys))
                mapping[x][fn][tuple(w)] = list(yks)
            return None

        arrays = self.variable_types(dk, verbose=False)['array']
        mapping = {}
        y_on_y = {}
        batches = self._check_batches(dk, batches)
        for batch in batches:
            b = self[dk].meta['sets']['batches'][batch]
            xy = b['x_y_map']
            f  = b['x_filter_map']
            fy = b['y_filter_map']
            w  = b['weights']
            for x, y in xy:
                if x == '@':
                    y = y[0]
                    _append_loop(mapping, x, f[y], w, [y])
                else:
                    _append_loop(mapping, x, f[x], w, y)
            for yy in b['y_on_y']:
                for x in b['yks'][1:]:
                    _append_loop(mapping, x, fy[yy], w, b['yks'])
                    _append_loop(y_on_y, x, fy[yy], w, b['yks'])
        return mapping, y_on_y

    @modify(to_list=['views', 'categorize', 'xs', 'batches'])
    def aggregate(self, views, unweighted_base=True, categorize=[],
                  batches='all', xs=None, bases={}, verbose=True):

        """
        Add views to all defined ``qp.Link`` in ``qp.Stack``.

        Parameters
        ----------
        views: str or list of str or qp.ViewMapper
            ``views`` that are added.
        unweighted_base: bool, default True
            If True, unweighted 'cbase' is added to all non-arrays.
            This parameter will be deprecated in future, please use bases
            instead.
        categorize: str or list of str
            Determines how numerical data is handled: If provided, the
            variables will get counts and percentage aggregations
            (``'counts'``, ``'c%'``) alongside the ``'cbase'`` view. If False,
            only ``'cbase'`` views are generated for non-categorical types.
        batches: str/ list of str, default 'all'
            Name(s) of ``qp.Batch`` instance(s) that are used to aggregate the
            ``qp.Stack``.
        xs: list of str
            Names of variable, for which views are added.
        bases: dict
            Defines which bases should be aggregated, weighted or unweighted.

        Returns
        -------
            None, modify ``qp.Stack`` inplace
        """
        # Preparing bases if older version with unweighed_base is used
        valid_bases = ['cbase', 'cbase_gross', 'ebase']
        if not bases and any(v in valid_bases for v in views):
            new_bases = {}
            for ba in valid_bases:
                if ba in views:
                    new_bases[ba] = {'unwgt': False if ba=='ebase' else unweighted_base,
                                     'wgt': True}
            views = [v for v in views if not v in valid_bases]
        else:
            new_bases = bases

        # Check if views are complete
        if views and isinstance(views[0], ViewMapper):
            views = views[0]
            complete = views[list(views.keys())[0]]['kwargs'].get('complete', False)
        elif any('cumsum' in v for v in views):
            complete = True
        else:
            complete = False

        # get counts + net views
        count_net_views = ['counts', 'counts_sum', 'counts_cumsum']
        if isinstance(views, ViewMapper) and list(views.keys()) == ['net']:
            counts_nets = qp.ViewMapper()
            counts_nets.make_template('frequency', {'rel_to': [None, 'y']})
            options = {'logic': views['net']['kwargs']['logic'],
                       'axis': 'x',
                       'expand': views['net']['kwargs']['expand'],
                       'complete': views['net']['kwargs']['complete'],
                       'calc': views['net']['kwargs']['calc']}
            counts_nets.add_method('net', kwargs=options)
        else:
            counts_nets = [v for v in views if v in count_net_views]

        x_in_stack = self.describe('x').index.tolist()
        for dk in list(self.keys()):
            batches = self._check_batches(dk, batches)
            if not batches: return None
            # check for unweighted_counts
            batch = self[dk].meta['sets']['batches']
            unwgt_c = any(batch[b].get('unwgt_counts') for b in batches)
            # get map and conditions for aggregation
            x_y_f_w_map, y_on_y = self._x_y_f_w_map(dk, batches)
            if not xs:
                xs = [x for x in list(x_y_f_w_map.keys()) if x in x_in_stack]
            else:
                xs = [x for x in xs if x in x_in_stack or isinstance(x, tuple)]

            v_typ = self.variable_types(dk, verbose=False)
            numerics = v_typ['int'] + v_typ['float']
            masks = self[dk].meta['masks']
            num_arrays = [m for m in masks if masks[m]['subtype'] in ['int', 'float']]
            if num_arrays: numerics = numerics + num_arrays
            skipped = [x for x in xs if (x in numerics and not x in categorize)
                       and not isinstance(x, tuple)]
            total_len = len(xs)
            # loop over map and aggregate views
            if total_len == 0:
                msg = "Cannot aggregate, 'xs' contains no valid variables."
                raise ValueError(msg)
            for idx, x in enumerate(xs, start=1):
                y_trans = None
                if isinstance(x, tuple):
                    y_trans = x[1]
                    x = x[0]
                if not x in list(x_y_f_w_map.keys()):
                    msg = "Cannot find {} in qp.Stack for ``qp.Batch`` '{}'"
                    raise KeyError(msg.format(x, batches))
                v = [] if x in skipped else views
                for f_dict in list(x_y_f_w_map[x].values()):
                    f = f_dict['f']
                    f_key = list(f.keys())[0] if isinstance(f, dict) else f
                    for weight, y in list(f_dict.items()):
                        if weight == 'f': continue
                        if y_trans: y = y_trans
                        w = list(weight) if weight else None
                        # add bases
                        for ba, weights in list(new_bases.items()):
                            ba_w = [b_w for b_w in w if not b_w is None]
                            if weights.get('wgt') and ba_w:
                                self.add_link(dk, f, x=x, y=y, views=[ba], weights=ba_w)
                            if (weights.get('wgt') and not ba_w) or weights.get('unwgt'):
                                self.add_link(dk, f, x=x, y=y, views=[ba], weights=None)
                        # remove existing nets for link if new view is a net
                        if isinstance(v, ViewMapper) and v.get('net') and not y_trans:
                            for ys in y:
                                link = self[dk][f_key][x][ys]
                                for view in list(link.keys()):
                                    is_net = view.split('|')[-1] == 'net'
                                    has_w = view.split('|')[-2]
                                    if not has_w: has_w = None
                                    if is_net and has_w in list(f_dict.keys()):
                                        del link[view]
                        # add unweighted views for counts/ nets
                        if unwgt_c and counts_nets and not None in w:
                            self.add_link(dk, f, x=x, y=y, views=counts_nets, weights=None)
                        # add common views
                        self.add_link(dk, f, x=x, y=y, views=v, weights=w)
                        # remove views if complete (cumsum/ nets)
                        if complete:
                            for ys in y:
                                y_on_ys = y_on_y.get(x, {}).get(f_key, {}).get(tuple(w), [])
                                if ys in y_on_ys: continue
                                link = self[dk][f_key][x][ys]
                                for ws in w:
                                    pct = 'x|f|:|y|{}|c%'.format('' if not ws else ws)
                                    counts = 'x|f|:||{}|counts'.format('' if not ws else ws)
                                    for view in [pct, counts]:
                                        if view in link:
                                            del link[view]
                if verbose:
                    done = float(idx) / float(total_len) *100
                    print('\r', end=' ')
                    time.sleep(0.01)
                    print('Stack [{}]: {} %'.format(dk, round(done, 1)), end=' ')
                    sys.stdout.flush()
            print('\n')

            if skipped and verbose:
                msg = ("\n\nWarning: Found {} non-categorized numeric variable(s): {}.\n"
                       "Descriptive statistics must be added!")
                print(msg.format(len(skipped), skipped))
        return None

    @modify(to_list=['on_vars', '_batches'])
    def cumulative_sum(self, on_vars, _batches='all', verbose=True):
        """
        Add cumulative sum view to a specified collection of xks of the stack.

        Parameters
        ----------
        on_vars : list
            The list of x variables to add the view to.
        _batches: str or list of str
            Only for ``qp.Links`` that are defined in this ``qp.Batch``
            instances views are added.

        Returns
        -------
        None
            The stack instance is modified inplace.
        """
        for dk in list(self.keys()):
            _batches = self._check_batches(dk, _batches)
            if not _batches or not on_vars: return None
            meta = self[dk].meta
            data = self[dk].data
            for v in on_vars:
                if v in meta['sets']:
                    items = [i.split('@')[-1] for i in meta['sets'][v]['items']]
                    on_vars = list(set(on_vars + items))

            self.aggregate(['counts_cumsum', 'c%_cumsum'], False, [], _batches, on_vars, verbose=verbose)
        return None

    def _add_checking_chain(self, dk, cluster, name, x, y, views):
        key, view, c_view = views
        if isinstance(cluster, ChainManager):
            c_stack = cluster.stack
        else:
            c_stack = qp.Stack('checks')
            c_stack.add_data('checks', data=self[dk].data, meta=self[dk].meta)
        c_stack.add_link(x=x, y=y, views=view, weights=None)
        c_stack.add_link(x=x, y=y, views=c_view, weights=None)
        c_views = c_stack.describe('view').index.tolist()
        len_v_keys = len(view)
        view_keys = ['x|f|x:|||cbase', 'x|f|:|||counts'][0:len_v_keys]
        c_views = view_keys + [v for v in c_views
                   if v.endswith('{}_check'.format(key))]
        if isinstance(cluster, ChainManager):
            cluster.get('checks', 'no_filter', x, y, c_views, folder=name, rules=False)
        else:
            if name == 'stat_check':
                chain = c_stack.get_chain(x=x, y=y, views=c_views, orient_on='x')
                name = [v for v in c_views if v.endswith('{}_check'.format(key))][0]
                cluster[name] = chain
            else:
                chain = c_stack.get_chain(name=name, x=x, y=y, views=c_views)
                cluster.add_chain(chain)
        return cluster

    @staticmethod
    def recode_from_net_def(dataset, on_vars, net_map, expand, recode='auto',
                            text_prefix='Net:', mis_in_rec=False,
                            verbose=True):
        """
        Create variables from net definitions.
        """
        def _is_simple_net(net_map):
            return all(isinstance(list(net.values())[0], list) for net in net_map)

        def _dissect_defs(ds, var, net_map, recode, text_prefix):
            mapper = []
            if recode == 'extend_codes':
                mapper += [(x, y, {var: x}) for (x,y) in ds.values(var)]
                max_code = max(ds.codes(var))
            elif recode == 'drop_codes':
                max_code = 0
            elif 'collect_codes' in recode:
                max_code = 0

            appends = []
            labels = {}
            s_net = True
            simple_nets = []
            for x, net in enumerate(net_map, 1):
                n = copy.deepcopy(net)
                if net.get('text'):
                    labs = n.pop('text')
                else:
                    labs = {ds.text_key: list(n.keys())[0]}
                code = max_code + x
                for tk, lab in list(labs.items()):
                    if not tk in labels: labels[tk] = {}
                    labels[tk].update({code: '{} {}'.format(text_prefix, lab)})
                appends.append((code, str(code), {var: list(n.values())[0]}))
                if not isinstance(list(n.values())[0], list):
                    s_net = False
                    simple_nets = []
                if s_net:
                    simple_nets.append(
                        ('{} {}'.format(text_prefix, labs[ds.text_key]), list(n.values())[0]))
            mapper += appends
            q_type = 'delimited set' if ds._is_delimited_set_mapper(mapper) else 'single'
            return mapper, q_type, labels, simple_nets

        forced_recode = False
        valid = ['extend_codes', 'drop_codes', 'collect_codes']
        if recode == 'auto':
            recode = 'collect_codes'
            forced_recode = True
        if not any(rec in recode for rec in valid):
            raise ValueError("'recode' must be one of {}".format(valid))

        dataset._meta['sets']['to_array'] = {}
        for var in on_vars[:]:
            if dataset.is_array(var): continue
            # get name for new variable
            suffix = '_rc'
            for s in [str(x) if not x == 1 else '' for x in frange('1-5')]:
                suf = suffix + s
                name = '{}{}'.format(dataset._dims_free_arr_item_name(var), suf)
                if dataset.var_exists(name):
                    if dataset._meta['columns'][name]['properties'].get('recoded_net'):
                        break
                else:
                    break

            # collect array items
            if dataset._is_array_item(var):
                to_array_set = dataset._meta['sets']['to_array']
                parent = dataset._maskname_from_item(var)
                arr_name = dataset._dims_free_arr_name(parent) + suf
                if arr_name in dataset:
                    msg = "Cannot create array {}. Variable already exists!"
                    if not dataset.get_property(arr_name, 'recoded_net'):
                        raise ValueError(msg.format(arr_name))
                no = dataset.item_no(var)
                if not arr_name in to_array_set:
                    to_array_set[arr_name] = [parent, [name], [no]]
                else:
                    to_array_set[arr_name][1].append(name)
                    to_array_set[arr_name][2].append(no)

            # create mapper to derive new variable
            mapper, q_type, labels, simple_nets = _dissect_defs(
                dataset, var, net_map, recode, text_prefix)
            dataset.derive(name, q_type, dataset.text(var), mapper)

            # meta edits for new variable
            for tk, labs in list(labels.items()):
                dataset.set_value_texts(name, labs, tk)
                text = dataset.text(var, tk) or dataset.text(var, None)
                dataset.set_variable_text(name, text, tk)

            # properties
            props = dataset._meta['columns'][name]['properties']
            props.update({'recoded_net': var})
            if 'properties' in dataset._meta['columns'][var]:
                for pname, prop in list(dataset._meta['columns'][var]['properties'].items()):
                    if pname == 'survey': continue
                    props[pname] = prop
            if simple_nets:
                props['simple_org_expr'] = simple_nets

            if verbose:
                print('Created: {}'. format(name))
            if forced_recode:
                warnings.warn("'{}' was a forced recode.".format(name))

            # order, remove codes
            if 'collect_codes' in recode:
                if not mis_in_rec and dataset._get_missing_list(var):
                    other_logic = intersection([
                        {var: not_count(0)},
                        {name: has_count(0)},
                        {var: not_any(dataset._get_missing_list(var))}])
                else:
                    other_logic = intersection(
                        [{var: not_count(0)}, {name: has_count(0)}])
                if dataset._is_array_item(var) or dataset.take(other_logic).tolist():
                    cat_name = recode.split('@')[-1] if '@' in recode else 'Other'
                    code = len(mapper)+1
                    dataset.extend_values(name, [(code, str(code))])
                    for tk in list(labels.keys()):
                        dataset.set_value_texts(name, {code: cat_name}, tk)
                    dataset.recode(name, {code: other_logic})
            if recode == 'extend_codes' and expand:
                codes = dataset.codes(var)
                new = [c for c in dataset.codes(name) if not c in codes]
                order = []
                remove = []
                for x, y, z in mapper[:]:
                    if not x in new:
                        order.append(x)
                    else:
                        vals = list(z.values())[0]
                        if not isinstance(vals, list):
                            remove.append(vals)
                            vals = [vals]
                        if expand == 'after':
                            idx = order.index(codes[min([codes.index(v) for v in vals])])
                        elif expand == 'before':
                            idx = order.index(codes[max([codes.index(v) for v in vals])]) + 1
                        order.insert(idx, x)

                dataset.reorder_values(name, order)
                dataset.remove_values(name, remove)

        for arr_name, arr_items in list(dataset._meta['sets']['to_array'].items()):
            org_mask = arr_items[0]
            m_items = arr_items[1]
            m_order = arr_items[2]
            m_items = [item[1] for item in sorted(zip(m_order, m_items))]
            dataset.to_array(arr_name, m_items, '', False)
            dims_name = dataset._dims_compat_arr_name(arr_name)
            prop = dataset._meta['masks'][dims_name]['properties']
            prop['recoded_net'] = org_mask
            if 'properties' in dataset._meta['masks'][org_mask]:
                for p, v in list(dataset._meta['masks'][org_mask]['properties'].items()):
                    if p == 'survey': continue
                    prop[p] = v
            n_i0 = dataset.sources(dims_name)[0]
            simple_net = dataset._meta['columns'][n_i0]['properties'].get('simple_org_expr')
            if simple_net:
                dataset._meta['masks'][dims_name]['properties'].update(
                    {'simple_org_expr': simple_net})
            if verbose:
                msg = "Array {} built from recoded view variables!"
                print(msg.format(dims_name))
        del dataset._meta['sets']['to_array']

        return None


    @modify(to_list=['on_vars', '_batches'])
    def add_nets(self, on_vars, net_map, expand=None, calc=None, rebase=None,
                 text_prefix='Net:', checking_cluster=None, _batches='all',
                 recode='auto', mis_in_rec=False, verbose=True):
        """
        Add a net-like view to a specified collection of x keys of the stack.

        Parameters
        ----------
        on_vars : list
            The list of x variables to add the view to.
        net_map : list of dicts
            The listed dicts must map the net/band text label to lists of
            categorical answer codes to group together, e.g.:

            >>> [{'Top3': [1, 2, 3]},
            ...  {'Bottom3': [4, 5, 6]}]
            It is also possible to provide enumerated net definition dictionaries
            that are explicitly setting ``text`` metadata per ``text_key`` entries:

            >>> [{1: [1, 2], 'text': {'en-GB': 'UK NET TEXT',
            ...                       'da-DK': 'DK NET TEXT',
            ...                       'de-DE': 'DE NET TEXT'}}]
        expand : {'before', 'after'}, default None
            If provided, the view will list the net-defining codes after or before
            the computed net groups (i.e. "overcode" nets).
        calc : dict, default None
            A dictionary that is attaching a text label to a calculation expression
            using the the net definitions. The nets are referenced as per
            'net_1', 'net_2', 'net_3', ... .
            Supported calculation expressions are add, sub, div, mul. Example:

            >>> {'calc': ('net_1', add, 'net_2'), 'text': {'en-GB': 'UK CALC LAB',
            ...                                            'da-DK': 'DA CALC LAB',
            ...                                            'de-DE': 'DE CALC LAB'}}
        rebase : str, default None
            Use another variables margin's value vector for column percentage
            computation.
        text_prefix : str, default 'Net:'
            By default each code grouping/net will have its ``text`` label prefixed
            with 'Net: '. Toggle by passing None (or an empty str, '').
        checking_cluster : quantipy.Cluster, default None
            When provided, an automated checking aggregation will be added to the
            ``Cluster`` instance.
        _batches: str or list of str
            Only for ``qp.Links`` that are defined in this ``qp.Batch``
            instances views are added.
        recode: {'extend_codes', 'drop_codes', 'collect_codes', 'collect_codes@cat_name'},
                 default 'auto'
            Adds variable with nets as codes to DataSet/Stack. If 'extend_codes',
            codes are extended with nets. If 'drop_codes', new variable only
            contains nets as codes. If 'collect_codes' or 'collect_codes@cat_name'
            the variable contains nets and another category that summarises all
            codes which are not included in any net. If no cat_name is provided,
            'Other' is taken as default
        mis_in_rec: bool, default False
            Skip or include codes that are defined as missing when recoding
            from net definition.
        Returns
        -------
        None
            The stack instance is modified inplace.
        """

        def _netdef_from_map(net_map, expand, prefix, text_key):
            netdef = []
            for no, net in enumerate(net_map, start=1):
                if 'text' in net:
                    logic = net[no]
                    text = net['text']
                else:
                    logic = list(net.values())[0]
                    text = {t: list(net.keys())[0] for t in text_key}
                if not isinstance(logic, list) and isinstance(logic, int):
                    logic = [logic]
                if prefix and not expand:
                    text = {k: '{} {}'.format(prefix, v) for k, v in list(text.items())}
                if expand:
                    text = {k: '{} (NET)'.format(v) for k, v in list(text.items())}
                netdef.append({'net_{}'.format(no): logic, 'text': text})
            return netdef

        def _check_and_update_calc(calc_expression, text_key):
            if not isinstance(calc_expression, dict):
                err_msg = ("'calc' must be a dict in form of\n"
                           "{'calculation label': (net # 1, operator, net # 2)}")
                raise TypeError(err_msg)
            for k, v in list(calc_expression.items()):
                if not k in ['text', 'calc_only']: exp = v
                if not k == 'calc_only': text = v
            if not 'text' in calc_expression:
                text = {tk: text for tk in text_key}
                calc_expression['text'] = text
            if not isinstance(exp, (tuple, list)) or len(exp) != 3:
                err_msg = ("Not properly formed expression found in 'calc':\n"
                           "{}\nMust be provided as (net # 1, operator, net # 2)")
                raise TypeError(err_msg.format(exp))
            return calc_expression

        for dk in list(self.keys()):
            _batches = self._check_batches(dk, _batches)
            only_recode = not _batches and recode
            if not _batches and not recode: return None
            meta = self[dk].meta
            data = self[dk].data
            check_on = []
            for v in on_vars[:]:
                if v in meta['sets']:
                    items = [i.split('@')[-1] for i in meta['sets'][v]['items']]
                    on_vars = list(set(on_vars)) + items
                    check_on.append(items[0])
                elif meta['columns'][v].get('parent'):
                    msg = 'Nets can not be added to a single array item: {}'
                    raise ValueError(msg.format(v))
                else:
                    check_on.append(v)
                for b in _batches:
                    batch = meta['sets']['batches'][b]
                    transposed = batch.get('transposed_arrays', batch['transposed'])
                    if v in transposed:
                        on_vars += [('@', v)]
                        break
            if not only_recode:
                all_batches = copy.deepcopy(meta['sets']['batches'])
                for n, b in list(all_batches.items()):
                    if not n in _batches: all_batches.pop(n)
                languages = list(set(b['language'] for n, b in list(all_batches.items())))
                netdef = _netdef_from_map(net_map, expand, text_prefix, languages)
                if calc:
                    calc = _check_and_update_calc(calc, languages)
                    calc_only = calc.get('calc_only', False)
                else:
                    calc_only = False
                view = qp.ViewMapper()
                if not rebase:
                    view.make_template('frequency', {'rel_to': [None, 'y']})
                else:
                    rebase = '{}.base'.format(rebase)
                    view.make_template('frequency', {'rel_to': [None, rebase]})
                options = {'logic': netdef,
                           'axis': 'x',
                           'expand': expand if expand in ['after', 'before'] else None,
                           'complete': True if expand else False,
                           'calc': calc,
                           'calc_only': calc_only}
                view.add_method('net', kwargs=options)
                self.aggregate(view, False, [], _batches, on_vars, verbose=verbose)

            if recode:
                ds = ds = qp.DataSet(dk, dimensions_comp=meta['info'].get('dimensions_comp'))
                ds.from_stack(self, dk)
                on_vars = [x for x in on_vars if x in self.describe('x').index.tolist()]
                self.recode_from_net_def(ds, on_vars, net_map, expand, recode,
                                         text_prefix, mis_in_rec,
                                         verbose)

            if checking_cluster in [None, False] or only_recode: continue
            if isinstance(checking_cluster, ChainManager):
                cc_keys = checking_cluster.folder_names
            else:
                cc_keys = list(checking_cluster.keys())
            view['net_check'] = view.pop('net')
            view['net_check']['kwargs']['iterators'].pop('rel_to')
            for v in check_on:
                v_net = '{}_net'.format(v)
                v_net = v_net.split('.')[-1]
                if not v_net in cc_keys:
                    checking_cluster = self._add_checking_chain(dk, checking_cluster,
                                            v_net, v, ['@', v], ('net', ['cbase'], view))

        return None

    @staticmethod
    def _factor_labs(values, axis, rescale, drop, exclude, factor_labels,
                     has_factors):
        if not rescale: rescale = {}
        ignore = [v['value'] for v in values if v['value'] in exclude or
                  (not v['value'] in list(rescale.keys()) and drop)]
        if factor_labels == '()':
            new_lab = '{} ({})'
            split = ('(', ')')
        else:
            new_lab = '{} [{}]'
            split = ('[', ']')
        factors_mapped = {}
        for v in values:
            if v['value'] in ignore: continue
            has_xedits  = v['text'].get('x edits', {})
            has_yedits  = v['text'].get('y edits', {})
            if not has_xedits:  v['text']['x edits'] = {}
            if not has_yedits:  v['text']['y edits'] = {}

            factor = rescale[v['value']] if rescale else v['value']
            for tk, text in list(v['text'].items()):
                if tk in ['x edits', 'y edits']: continue
                for ax in axis:
                    try:
                        t = v['text']['{} edits'.format(ax)][tk]
                    except:
                        t = text
                    if has_factors:
                        fac = t.split(split[0])[-1].replace(split[1], '')
                        if fac == str(factor): continue
                    v['text']['{} edits'.format(ax)][tk] = new_lab.format(t, factor)
        return values

    @staticmethod
    def _add_factor_meta(dataset, var, options):
        if not dataset._has_categorical_data(var):
            return None
        rescale = options[0]
        drop = options[1]
        exclude = options[2]
        dataset.clear_factors(var)
        all_codes = dataset.codes(var)
        if rescale:
            fm = rescale
        else:
            fm = {c: c for c in all_codes}
        if not drop and rescale:
            for c in all_codes:
                if not c in fm:
                    fm[c] = c
        if exclude:
            for e in exclude:
                if e in fm:
                    del fm[e]
        dataset.set_factors(var, fm)
        return None

    @modify(to_list=['on_vars', 'stats', 'exclude', '_batches'])
    def add_stats(self, on_vars, stats=['mean'], other_source=None, rescale=None,
                  drop=True, exclude=None, factor_labels=True, custom_text=None,
                  checking_cluster=None, _batches='all', recode=False, verbose=True):
        """
        Add a descriptives view to a specified collection of xks of the stack.

        Valid descriptives views: {'mean', 'stddev', 'min', 'max', 'median', 'sem'}

        Parameters
        ----------
        on_vars : list
            The list of x variables to add the view to.
        stats : list of str, default ``['mean']``
            The metrics to compute and add as a view.
        other_source : str
            If provided the Link's x-axis variable will be swapped with the
            (numerical) variable provided. This can be used to attach statistics
            of a different variable to a Link definition.
        rescale : dict
            A dict that maps old to new codes, e.g. {1: 5, 2: 4, 3: 3, 4: 2, 5: 1}
        drop : bool, default True
            If ``rescale`` is provided all codes that are not mapped will be
            ignored in the computation.
        exclude : list
            Codes/values to ignore in the computation.
        factor_labels : bool / str, default True
            Writes the (rescaled) factor values next to the category text label.
            If True, square-brackets are used. If '()', normal brackets are used.
        custom_text : str, default None
            A custom string affix to put at the end of the requested statistics'
            names.
        checking_cluster : quantipy.Cluster, default None
            When provided, an automated checking aggregation will be added to the
            ``Cluster`` instance.
        _batches: str or list of str
            Only for ``qp.Links`` that are defined in this ``qp.Batch``
            instances views are added.
        recode: bool, default False
            Create a new variable that contains only the values
            which are needed for the stat computation. The values and the included
            data will be rescaled.

        Returns
        -------
        None
            The stack instance is modified inplace.
        """

        def _recode_from_stat_def(dataset, on_vars, rescale, drop, exclude, verbose):
            for var in on_vars:
                if dataset.is_array(var): continue
                suffix = '_rc'
                for s in [str(x) if not x == 1 else '' for x in frange('1-5')]:
                    suf = suffix + s
                    name = '{}{}'.format(var, suf)
                    if dataset.var_exists(name):
                        if dataset._meta['columns'][name]['properties'].get('recoded_stat'):
                            break
                    else:
                        break
                if not rescale:
                    rescale = {x: x for x in dataset.codes(var)}
                else:
                    rescale = copy.deepcopy(rescale)
                if drop or exclude:
                    for x in list(rescale.keys()):
                        if not x in dataset.codes(var) or x in exclude:
                            rescale.pop(x)
                dataset.add_meta(name, 'float', dataset.text(var))
                for x, y in list(rescale.items()):
                    sl = dataset.take({var: x})
                    dataset[sl, name] = y
                if verbose:
                    print('Created: {}'. format(name))
                dataset._meta['columns'][name]['properties'].update({'recoded_stat': var})
            return None

        def _add_factors(v, meta, values, args):
            if isinstance(values, str):
                p = values.split('@')[-1]
                p_meta = meta.get('masks', meta)[p]
                p_lib = meta['lib'].get('values', meta['lib'])
                has_factors = p_meta.get('properties', {}).get('factor_labels', False)
                v_args = args + [has_factors]
                values = p_lib[p]
                p_lib[p] = self._factor_labs(values, ['x', 'y'], *v_args)
                if not p_meta.get('properties'): p_meta['properties'] = {}
                p_meta['properties'].update({'factor_labels': True})
            else:
                v_meta = meta.get('columns', meta)[v]
                has_factors = v_meta.get('properties', {}).get('factor_labels')
                v_args = args + [has_factors]
                v_meta['values'] = self._factor_labs(values, ['x'], *v_args)
                if not v_meta.get('properties'): v_meta['properties'] = {}
                v_meta['properties'].update({'factor_labels': True})
            return None

        if other_source and not isinstance(other_source, str):
            raise ValueError("'other_source' must be a str!")
        if not rescale: drop = False

        options = {'stats': '',
                   'source': other_source,
                   'rescale': rescale,
                   'drop': drop, 'exclude': exclude,
                   'axis': 'x',
                   'text': '' if not custom_text else custom_text}
        warn = "\nCannot add stats on '{}'.\n"
        for dk in list(self.keys()):
            _batches = self._check_batches(dk, _batches)
            if not _batches: return None
            meta = self[dk].meta
            data = self[dk].data
            check_on = []
            no_os = not other_source
            for v in on_vars:
                if v in meta['sets']:
                    if meta['masks'][v]['subtype'] == 'delimited set' and no_os:
                        w = warn + 'Stats are not valid on delimited sets!\n'
                        print(w.format(v))
                        continue
                    items = [i.split('@')[-1] for i in meta['sets'][v]['items']]
                    on_vars = list(set(on_vars + items))
                    check_on = list(set(check_on + [items[0]]))
                elif not meta['columns'][v].get('values'):
                    w = warn + 'No values found!\n'
                    print(w.format(v))
                    continue
                elif meta['columns'][v]['type'] == 'delimited set' and no_os:
                    w = warn + 'Stats are not valid on delimited sets!\n'
                    print(w.format(v))
                    continue
                elif not isinstance(meta['columns'][v]['values'], list):
                    parent = list(meta['columns'][v]['parent'].keys())[0].split('@')[-1]
                    items = [i.split('@')[-1] for i in meta['sets'][parent]['items']]
                    check_on = list(set(check_on + [items[0]]))
                else:
                    check_on = list(set(check_on + [v]))
                for b in _batches:
                    batch = meta['sets']['batches'][b]
                    transposed = batch.get('transposed_arrays', batch['transposed'])
                    if v in transposed:
                        on_vars += [('@', v)]
                        break

                ds = qp.DataSet(dk, dimensions_comp=meta['info'].get('dimensions_comp'))
                ds.from_stack(self, dk)
                if not other_source:
                    self._add_factor_meta(ds, v, (rescale, drop, exclude))


            view = qp.ViewMapper()
            view.make_template('descriptives')
            for stat in stats:
                options['stats'] = stat
                view.add_method('stat', kwargs=options)
                self.aggregate(view, False, on_vars, _batches, on_vars, verbose=verbose)

            if recode:
                if other_source:
                    raise ValueError('Cannot recode if other_source is provided.')
                ds = qp.DataSet(dk, dimensions_comp=meta['info'].get('dimensions_comp'))
                ds.from_stack(self, dk)
                on_vars = [x for x in on_vars if x in self.describe('x').index.tolist()]
                _recode_from_stat_def(ds, on_vars, rescale, drop, exclude, verbose)

            if factor_labels:
                args = [rescale, drop, exclude, factor_labels]
                all_batches = list(meta['sets']['batches'].keys())
                if not _batches: _batches = all_batches
                batches = [b for b in all_batches if b in _batches]
                for v in check_on:
                    globally = False
                    for b in batches:
                        batch_me = meta['sets']['batches'][b]['meta_edits']
                        values = batch_me.get(v, {}).get('values', [])
                        if not values:
                            globally = True
                        else:
                            _add_factors(v, batch_me, values, args)
                    if globally:
                        values = meta['columns'][v]['values']
                        _add_factors(v, meta, values, args)
                    if isinstance(checking_cluster, ChainManager):
                        cm_meta = checking_cluster.stack['checks'].meta
                        values = cm_meta['columns'][v]['values']
                        _add_factors(v, cm_meta, values, args)
            if checking_cluster and 'mean' in stats and check_on:

                options['stats'] = 'mean'
                c_view = qp.ViewMapper().make_template('descriptives')
                c_view.add_method('stat_check', kwargs=options)

                views = ('stat', ['cbase', 'counts'], c_view)
                checking_cluster = self._add_checking_chain(dk, checking_cluster,
                                            'stat_check', check_on, ['@'], views)

        return None

    @modify(to_list=['_batches'])
    def add_tests(self, _batches='all', verbose=True):
        """
        Apply coltests for selected batches.

        Sig. Levels are taken from ``qp.Batch`` definitions.

        Parameters
        ----------
        _batches: str or list of str
            Only for ``qp.Links`` that are defined in this ``qp.Batch``
            instances views are added.

        Returns
        -------
        None
        """
        self._remove_coltests()

        if verbose:
            start = time.time()

        for dk in list(self.keys()):
            _batches = self._check_batches(dk, _batches)
            if not _batches: return None
            for batch_name in _batches:
                batch = self[dk].meta['sets']['batches'][batch_name]
                sigpro = batch.get('sigproperties', {})
                levels = batch.get('sigproperties', batch).get('siglevels', [])
                weight = batch['weights']
                x_y    = batch['x_y_map']
                x_f    = batch['x_filter_map']
                y_f    = batch['y_filter_map']
                yks    = batch['yks']

                if levels:
                    vm_tests = qp.ViewMapper().make_template(
                    method='coltests',
                    iterators={'metric': ['props', 'means'],
                               'mimic': sigpro.get('mimic', ['Dim']),
                               'level': levels})
                    vm_tests.add_method('significance',
                                        kwargs = {'flag_bases': sigpro.get('flag_bases', [30, 100]),
                                                  'test_total': sigpro.get('test_total', None),
                                                  'groups': 'Tests'})
                    for yy in batch['y_on_y']:
                        if y_f[yy]:
                            fy = y_f[yy].encode('utf8')
                            f = {fy: {fy: 0}}
                        else:
                            f = ['no_filter']
                        self.add_link(filters=f, x=yks[1:], y=yks,
                                      views=vm_tests, weights=weight)
                    total_len = len(x_y)
                    for idx, xy in enumerate(x_y, 1):
                        x, y = xy
                        if x == '@': continue
                        if x_f[x]:
                            fx = x_f[x].encode('utf8')
                            f = {fx: {fx: 0}}
                        else:
                            f = ['no_filter']
                        self.add_link(filters=f, x=x, y=y,
                                       views=vm_tests, weights=weight)
                        if verbose:
                            done = float(idx) / float(total_len) *100
                            print('\r', end=' ')
                            time.sleep(0.01)
                            print('Batch [{}]: {} %'.format(batch_name, round(done, 1)), end=' ')
                            sys.stdout.flush()
                if verbose and levels: print('\n')
        if verbose: print('Sig-Tests:', time.time()-start)
        return None

    def _remove_coltests(self, props=True, means=True):
        """
        Remove coltests from stack.

        Parameters
        ----------
        props : bool, default=True
            If True, column proportion test view will be removed from stack.
        means : bool, default=True
            If True, column mean test view will be removed from stack.
        """
        for dk in list(self.keys()):
            for fk in list(self[dk].keys()):
                for xk in list(self[dk][fk].keys()):
                    for yk in list(self[dk][fk][xk].keys()):
                        for vk in list(self[dk][fk][xk][yk].keys()):
                            del_prop = props and 't.props' in vk
                            del_mean = means and 't.means' in vk
                            if del_prop or del_mean:
                                del self[dk][fk][xk][yk][vk]
                                del self[dk][fk][xk][yk][vk]
        return None
