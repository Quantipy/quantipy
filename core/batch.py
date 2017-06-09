#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import quantipy as qp

from collections import OrderedDict

from quantipy.core.tools.qp_decorators import *

import copy as org_copy
import warnings
import re

def meta_editor(self, dataset_func):
    """
    Decorator for inherited DataSet methods.
    """
    def edit(*args, **kwargs):
        # get name and type of the variable dor correct dict refernces
        name = args[0] if args else kwargs['name']
        is_array = self._is_array(name)
        is_array_item = self._is_array_item(name)
        has_edits = name in self.meta_edits
        parent = self._maskname_from_item(name) if is_array_item else None
        parent_edits = parent in self.meta_edits
        source = self.sources(name) if is_array else []
        source_edits = [s in self.meta_edits for s in source]
        # create DataSet clone to leave global meta data untouched
        ds_clone = self.clone()
        # are we adding to aleady existing batch meta edits? (use copy then!)
        var_edits = [(name, has_edits), (parent, parent_edits)]
        var_edits += [(s, s_edit) for s, s_edit in zip(source, source_edits)]
        for var, edits in var_edits:
            if edits:
                copied_meta = org_copy.deepcopy(self.meta_edits[var])
                if not self._is_array(var):
                    ds_clone._meta['columns'][var] = copied_meta
                else:
                    ds_clone._meta['masks'][var] = copied_meta
                if self.meta_edits['lib'].get(var):
                    lib = self.meta_edits['lib'][var]
                    ds_clone._meta['lib']['values'][var] = lib
        # use qp.DataSet method to apply the edit
        dataset_func(ds_clone, *args, **kwargs)
        # grab edited meta data and collect via Batch.meta_edits attribute
        if not self._is_array(name):
            meta = ds_clone._meta['columns'][name]
            text_edits = ['set_col_text_edit', 'set_val_text_edit']
            if dataset_func.func_name in text_edits and is_array_item:
                self.meta_edits[parent] = ds_clone._meta['masks'][parent]
                lib = ds_clone._meta['lib']['values'][parent]
                self.meta_edits['lib'][parent] = lib
        else:
            meta = ds_clone._meta['masks'][name]
            self.meta_edits['lib'][name] = ds_clone._meta['lib']['values'][name]
        self.meta_edits[name] = meta
    return edit

def not_implemented(dataset_func):
    """
    Decorator for UNALLOWED DataSet methods.
    """
    def _unallowed_inherited_method(*args, **kwargs):
        err_msg = 'DataSet method not allowed for Batch editing!'
        raise NotImplementedError(err_msg)
    return _unallowed_inherited_method


class Batch(qp.DataSet):
    """
    A Batch is a container for structuring a Link collection's
    specifications aimed at Excel and/or PPTX build Clusters.
    """
    def __init__(self, dataset, name, ci=['c', 'p'], weights=None, tests=None):
        if '-' in name: raise ValueError("Batch 'name' must not contain '-'!")
        self.name = name
        if not 'batches' in dataset._meta['sets']:
           dataset._meta['sets']['batches'] = OrderedDict()
        dataset._meta['sets']['batches'][name] = {'name': name,
                                                  'additions': []}
        meta, data = dataset.split()
        self._meta = meta
        self._data = data.copy()
        self.xks = []
        self.yks = ['@']
        self.extended_yks_global = None
        self.extended_yks_per_x = {}
        self.exclusive_yks_per_x = {}
        self.extended_filters_per_x = {}
        self.filter = 'no_filter'
        self.filter_names = ['no_filter']
        self.x_y_map = None
        self.x_filter_map = None
        self.y_on_y = None
        self.forced_names = {}
        self.summaries = []
        self.transposed_arrays = {}
        self.verbatims = OrderedDict()
        self.verbatim_names = []
        self.cell_items = ci
        self.weights = weights
        self.siglevels = tests
        self.additional = False
        self.meta_edits = {'lib': {}}
        self.sample_size = None
        self.text_key = dataset.text_key
        self.language = dataset.text_key
        self.valid_tks = dataset.valid_tks
        # self._update()
        # DECORATED / OVERWRITTEN DataSet methods
        self.hiding = meta_editor(self, qp.DataSet.hiding.__func__)
        self.sorting = meta_editor(self, qp.DataSet.sorting.__func__)
        self.slicing = meta_editor(self, qp.DataSet.slicing.__func__)
        self.set_variable_text = meta_editor(self, qp.DataSet.set_variable_text.__func__)
        self.set_value_texts = meta_editor(self, qp.DataSet.set_value_texts.__func__)
        self.set_property = meta_editor(self, qp.DataSet.set_property.__func__)
        # UNALLOWED DataSet methods
        self.add_meta = not_implemented(qp.DataSet.add_meta.__func__)
        self.derive = not_implemented(qp.DataSet.derive.__func__)

    def _update(self):
        """
        Update Batch metadata with Batch attributes.
        """
        self._map_x_to_y()
        self._map_x_to_filter()
        for attr in ['xks', 'yks', 'filter', 'filter_names',
                     'x_y_map', 'x_filter_map', 'y_on_y',
                     'forced_names', 'summaries', 'transposed_arrays', 'verbatims',
                     'verbatim_names', 'extended_yks_global', 'extended_yks_per_x',
                     'exclusive_yks_per_x', 'extended_filters_per_x', 'meta_edits',
                     'cell_items', 'weights', 'siglevels', 'additional',
                     'sample_size', 'language']:
            attr_update = {attr: self.__dict__[attr]}
            self._meta['sets']['batches'][self.name].update(attr_update)

    def copy(self, name):
        """
        Create a copy of Batch instance.
        """
        org_name = self.name
        org_meta = org_copy.deepcopy(self._meta['sets']['batches'][org_name])
        batch_copy = org_copy.deepcopy(self)
        self._meta['sets']['batches'][name] = org_meta
        batch_copy._meta['sets']['batches'][name] = org_meta
        batch_copy.name = name
        if batch_copy.verbatims:
            batch_copy.verbatims = {}
            warning = ("Copied Batch '{}' contains open end data summaries...\n"
                       "Any filters added to the copy will not persist "
                       "on verbatims so they have been removed! "
                       "Please add them again!")
            warnings.warn(warning.format(name))
            return batch_copy

    @modify(to_list='ci')
    def set_cell_items(self, ci):
        """
        Assign cell items ('c', 'p', 'cp').
        """
        if ci not in [['c'], ['p'], ['c', 'p'], ['p', 'c'], ['cp']]:
            raise ValueError("'ci' cell items must be either 'c', 'p' or 'cp'.")
        self.cell_items = ci
        self._update()
        return None

    @modify(to_list='w')
    @verify(variables={'w': 'columns'})
    def set_weights(self, w):
        """
        Assign a weight variable setup.
        """
        self.weights = w
        self._update()
        return None

    @modify(to_list='levels')
    def set_sigtests(self, levels=None, mimic=None, flags=None, test_total=None):
        """
        Specify a significance test setup.
        """
        if levels:
            if not all(isinstance(l, float) for l in levels):
                raise TypeError('All significance levels must be provided as floats!')
            levels = sorted(levels)
            self.siglevels = levels
        if mimic or flags or test_total:
            err = ("Changes to 'mimic', 'flags', 'test_total' currently not allowed!")
            raise NotImplementedError(err)
        self._update()
        return None

    @verify(text_keys='text_key')
    def set_language(self, text_key):
        """
        Set ``Batch.language`` indicated via the ``text_key`` for Build exports.
        """
        self.language = text_key
        self._update()
        return None

    def as_addition(self, batch_name):
        """
        Treat the Batch as additional aggregations, independent from the
        global Batch & Build setup.
        """
        self._meta['sets']['batches'][batch_name]['additions'].append(self.name)
        self.additional = True
        self.verbatims = {}
        self.y_on_y = None
        msg = ("Batch '{}' specified as addition to Batch '{}'. Any open end "
               "summaries and 'y_on_y' agg. have been removed!")
        print msg.format(self.name, batch_name)
        self._update()
        return None

    @modify(to_list='xks')
    def add_x(self, xks):
        """
        Set the x (downbreak) variables of the Batch.
        """
        clean_xks = self._check_forced_names(xks)
        self.xks = clean_xks
        self._update()
        masks = [x for x in self.xks if x in self.masks()]
        self.make_summaries(masks)
        print 'Array summaries are created for {}.'.format(masks)
        return None

    @modify(to_list='arrays')
    @verify(variables={'arrays': 'masks'})
    def make_summaries(self, arrays):
        """
        """
        self.summaries = arrays
        for t_array in self.transposed_arrays.keys():
            if not t_array in arrays:
                self.transposed_arrays.pop(t_array)
        self._update()
        return None

    @modify(to_list='arrays')
    @verify(variables={'arrays': 'masks'})
    def transpose_arrays(self, arrays, replace=True):
        """
        """
        for array in arrays:
            if not array in self.summaries:
                self.summaries.append(array)
            self.transposed_arrays[array] = replace
        self._update()
        return None

    @modify(to_list='yks')
    @verify(variables={'yks': 'both'})
    def add_y(self, yks):
        """
        Set the y (crossbreak/banner) variables of the Batch.
        """
        non_cat = [y for y in yks if not self._has_categorical_data(y)]
        if non_cat:
            msg = 'Cannot add yks with non-categorical data: {}'.format(non_cat)
            raise ValueError(msg)
        yks = ['@'] + yks
        self.yks = yks
        self._update()
        return None

    def add_x_per_y(self, x_on_y_map):
        """
        Add individual combinations of x and y variables to the Batch.
        """
        raise NotImplemetedError('NOT YET SUPPPORTED')
        if not isinstance(x_on_y_map, list): x_on_y_maps = [x_on_y_map]
        if not isinstance(x_on_y_maps[0], dict):
            raise TypeError('Must pass a (list of) dicts!')
        for x_on_y_map in x_on_y_maps:
            for x, y in x_on_y_map.items():
                if not isinstance(y, list): y = [y]
                if isinstance(x, tuple): x = {x[0]: x[1]}
        return None

    def add_filter(self, filter_name, filter_logic):
        """
        Apply a (global) filter to all the variables found in the Batch.
        """
        self.filter = {filter_name: filter_logic}
        if filter_name not in self.filter_names:
            self.filter_names = [filter_name]
        self._update()
        return None

    @modify(to_list=['oe', 'break_by', 'title'])
    @verify(variables={'oe': 'columns', 'break_by': 'columns'})
    def add_open_ends(self, oe, break_by=None, drop_empty=True, incl_nan=False,
                      split=False, title='open ends', filter_by=None):
        """
        Create respondent level based listings of open-ended text data.

        Parameters
        ----------
        oe : str or list of str
            The open-ended questions / verbatims to be added to the stack.
        break_by : str or list of str, default None
            If provided, these variables will be presented alongside the ``oe``
            data.
        drop_empty : bool, default True
            Case data that is missing valid entries will be dropped from the
            output.
        incl_nan: bool, default False
            Show __NaN__ in the output.
        split: bool, default False
            If True len of oe must be same size as len of title. Each oe is 
            saved with its own title.
        title : str, default 'open ends'
            Specifies the the ``Cluster`` / Excel sheet name for the output.
        filter_by : A Quantipy logical expression, default None
            An additional logical filter that should be applied to the case data.
            Any ``filter`` provided by a ``batch`` will be respected
            automatically.

        Returns
        -------
        None
        """
        def _add_oe(oe, break_by, title, drop_empty, incl_nan, filter_by):
            columns = break_by + oe
            oe_data = self._data.copy()
            if self.filter != 'no_filter':
                ds = qp.DataSet('open_ends')
                ds.from_components(oe_data, self._meta)
                slicer = ds.take(self.filter.values()[0])
                oe_data = oe_data.loc[slicer, :]
            if filter_by:
                ds = qp.DataSet('open_ends')
                ds.from_components(oe_data, self._meta)
                slicer = ds.take(filter_by)
                oe_data = oe_data.loc[slicer, :]
            oe_data = oe_data[columns]
            oe_data.replace('__NA__', np.NaN, inplace=True)
            if drop_empty:
                oe_data.dropna(subset=oe, how='all', inplace=True)
            if not incl_nan:
                for col in oe:
                    oe_data[col].replace(np.NaN, '', inplace=True)
            self.verbatims[title] = oe_data
            self.verbatim_names.extend(oe)

        if split: 
            if not len(oe) == len(title):
                msg = "Cannot derive verbatim DataFrame 'title' with more than 1 'oe'"
                raise ValueError(msg)
            for t, open_end in zip(title, oe):
                open_end = [open_end]
                _add_oe(open_end, break_by, t, drop_empty, incl_nan, filter_by)
        else:
            _add_oe(oe, break_by, title[0], drop_empty, incl_nan, filter_by)
        self._update()
        return None

    @modify(to_list=['ext_yks', 'on'])
    @verify(variables={'ext_yks': 'both', 'on': 'both'})
    def extend_y(self, ext_yks, on=None):
        """
        Add y (crossbreak/banner) variables to specific x (downbreak) variables.
        """
        if not on:
            self.yks.extend(ext_yks)
            if not self.extended_yks_global:
                self.extended_yks_global = ext_yks
            else:
                self.extended_yks_global.extend(ext_yks)
        else:
            on = self.unroll(on, both='all')
            for x in on:
                self.extended_yks_per_x.update({x: ext_yks})
        self._update()
        return None

    @modify(to_list=['new_yks', 'on'])
    @verify(variables={'new_yks': 'both', 'on': 'both'})
    def replace_y(self, new_yks, on):
        """
        Replace y (crossbreak/banner) variables on specific x (downbreak) variables.
        """
        on = self.unroll(on, both='all')
        for x in on:
            self.exclusive_yks_per_x.update({x: new_yks})
        self._update()
        return None

    def extend_filter(self, ext_filters):
        """
        Apply additonal filtering to specific x (downbreak) variables.
        """
        for variables, logic in ext_filters.items():
            if not isinstance(variables, tuple):
                variables = [variables]
            elif len(variables) == 1:
                variables = [variables]
            for v in variables:
                new_filter = self._combine_filters({v: logic})
                if not new_filter.keys()[0] in self.filter_names:
                    self.filter_names.append(new_filter.keys()[0])
                self.extended_filters_per_x.update({v: new_filter})
        self._update()
        return None

    def add_y_on_y(self, name):
        """
        Produce aggregations crossing the (main) y variables with each other.
        """
        self.y_on_y = name
        self._update()
        return None

    def _map_x_to_y(self):
        """
        """
        mapping = OrderedDict()
        for x in self.xks:
            mapping[x] = org_copy.deepcopy(self.yks)
            if x in self.extended_yks_per_x:
                mapping[x].extend(self.extended_yks_per_x[x])
            if x in self.exclusive_yks_per_x:
                mapping[x] = self.exclusive_yks_per_x[x]
            if x in self._meta['masks']:
                xks = self.sources(x)
                for x2 in xks:
                    mapping[x2] = org_copy.deepcopy(self.yks)
                    if x2 in self.extended_yks_per_x:
                        mapping[x2].extend(self.extended_yks_per_x[x2])
                    if x2 in self.exclusive_yks_per_x:
                        mapping[x2] = self.exclusive_yks_per_x[x2]
        self.x_y_map = mapping
        return None

    def _map_x_to_filter(self):
        """
        """
        mapping = OrderedDict()
        for x in self.xks:
            mapping[x] = org_copy.deepcopy(self.filter)
            if x in self.extended_filters_per_x:
                mapping[x] = self.extended_filters_per_x[x]
            if x in self._meta['masks']:
                xks = self.sources(x)
                for x2 in xks:
                    mapping[x2] = org_copy.deepcopy(self.filter)
                    if x2 in self.extended_filters_per_x:
                        mapping[x2] = self.extended_filters_per_x[x2]
        self.x_filter_map = mapping
        return None

    def _check_forced_names(self, variables):
        """
        """
        xks = []
        renames = {}
        for x in variables:
            if isinstance(x, dict):
                xks.append(x.keys()[0])
                renames[x.keys()[0]] = x.values()[0]
            elif isinstance(x, tuple):
                xks.append(x[0])
                renames[x[0]] = x[1]
            else:
                xks.append(x)
        self.forced_names = renames
        return xks

    def _combine_filters(self, ext_filters):
        """
        """
        old_filter = self.filter
        no_global_filter = old_filter == 'no_filter'
        if no_global_filter:
            combined_name = '(no_filter)+({})'.format(ext_filters.keys()[0])
            new_filter = {combined_name: ext_filters.values()[0]}
        else:
            old_filter_name = old_filter.keys()[0]
            old_filter_logic = old_filter.values()[0]
            new_filter_name = ext_filters.keys()[0]
            new_filter_logic = ext_filters.values()[0]
            combined_name = '({})+({})'.format(old_filter_name, new_filter_name)
            combined_logic = intersection([old_filter_logic, new_filter_logic])
            new_filter = {combined_name: combined_logic}
        return new_filter

