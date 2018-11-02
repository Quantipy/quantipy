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

from quantipy.core.tools.view.logic import (
    has_any, has_all, has_count,
    not_any, not_all, not_count,
    is_lt, is_ne, is_gt,
    is_le, is_eq, is_ge,
    union, intersection, get_logic_index)

def meta_editor(self, dataset_func):
    """
    Decorator for inherited DataSet methods.
    """
    def edit(*args, **kwargs):
        # get name and type of the variable dor correct dict refernces
        name = args[0] if args else kwargs['name']
        if not isinstance(name, list): name = [name]
        # create DataSet clone to leave global meta data untouched
        if self.edits_ds is None:
            self.edits_ds = qp.DataSet.clone(self)
        ds_clone = self.edits_ds
        var_edits = []
        # args/ kwargs for min_value_count
        if dataset_func.func_name == 'min_value_count':
            if len(args) < 3 and not 'weight' in kwargs:
                kwargs['weight'] = self.weights[0]

            if len(args) < 4 and not 'condition' in kwargs:
                if not self.filter == 'no_filter':
                    kwargs['condition'] = self.filter.values()[0]
        # args/ kwargs for sorting
        elif dataset_func.func_name == 'sorting':
            if len(args) < 7 and not 'sort_by_weight' in kwargs:
                kwargs['sort_by_weight'] = self.weights[0]

        for n in name:
            is_array = self.is_array(n)
            is_array_item = self._is_array_item(n)
            has_edits = n in self.meta_edits
            parent = self._maskname_from_item(n) if is_array_item else None
            parent_edits = parent in self.meta_edits
            source = self.sources(n) if is_array else []
            source_edits = [s in self.meta_edits for s in source]
            # are we adding to aleady existing batch meta edits? (use copy then!)
            var_edits += [(n, has_edits), (parent, parent_edits)]
            var_edits += [(s, s_edit) for s, s_edit in zip(source, source_edits)]
        for var, edits in var_edits:
            if edits:
                copied_meta = org_copy.deepcopy(self.meta_edits[var])
                if not self.is_array(var):
                    ds_clone._meta['columns'][var] = copied_meta
                else:
                    ds_clone._meta['masks'][var] = copied_meta
                if self.meta_edits['lib'].get(var):
                    lib = self.meta_edits['lib'][var]
                    ds_clone._meta['lib']['values'][var] = lib
        # use qp.DataSet method to apply the edit
        dataset_func(ds_clone, *args, **kwargs)
        # grab edited meta data and collect via Batch.meta_edits attribute
        for n in self.unroll(name, both='all'):
            if not self.is_array(n):
                meta = ds_clone._meta['columns'][n]
                text_edits = ['set_col_text_edit', 'set_val_text_edit']
                if dataset_func.func_name in text_edits and is_array_item:
                    self.meta_edits[parent] = ds_clone._meta['masks'][parent]
                    lib = ds_clone._meta['lib']['values'][parent]
                    self.meta_edits['lib'][parent] = lib
            else:
                meta = ds_clone._meta['masks'][n]
                if ds_clone._has_categorical_data(n):
                    self.meta_edits['lib'][n] = ds_clone._meta['lib']['values'][n]
            self.meta_edits[n] = meta
        if dataset_func.func_name in ['hiding', 'slicing', 'min_value_count', 'sorting']:
            self._update()
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
        sets = dataset._meta['sets']
        if not 'batches' in sets: sets['batches'] = OrderedDict()
        self.name = name
        meta, data = dataset.split()
        self._meta = meta
        self._data = data
        self.edits_ds = None
        self.valid_tks = dataset.valid_tks
        self.text_key = dataset.text_key
        self.sample_size = None
        self._verbose_errors = dataset._verbose_errors
        self._verbose_infos = dataset._verbose_infos
        self._dimensions_comp = dataset._dimensions_comp

        # RENAMED DataSet methods
        self._dsfilter = qp.DataSet.filter.__func__

        if sets['batches'].get(name):
            if self._verbose_infos:
                print "Load Batch '{}'.".format(name)
            self._load_batch()
        else:
            sets['batches'][name] = {'name': name, 'additions': []}
            self.xks = []
            self.yks = ['@']
            self._variables = []
            self.total = True
            self.extended_yks_global = None
            self.extended_yks_per_x = {}
            self.exclusive_yks_per_x = {}
            self.extended_filters_per_x = {}
            self.filter = None
            self.filter_names = []
            self.x_y_map = None
            self.x_filter_map = None
            self.y_on_y = []
            self.y_on_y_filter = {}
            self.y_filter_map = {}
            self.forced_names = {}
            self.summaries = []
            self.transposed_arrays = {}
            self.skip_items = []
            self.verbatims = []
            # self.verbatim_names = []
            self.set_cell_items(ci)   # self.cell_items
            self.unwgt_counts = False
            self.set_weights(weights) # self.weights
            self.set_sigtests(tests)  # self.sigproperties
            self.additional = False
            self.meta_edits = {'lib': {}}
            self.build_info = {}
            self.set_language(dataset.text_key) # self.language
            self._update()

        # DECORATED / OVERWRITTEN DataSet methods
        # self.hide_empty_items = meta_editor(self, qp.DataSet.hide_empty_items.__func__)
        self.hiding = meta_editor(self, qp.DataSet.hiding.__func__)
        self.min_value_count = meta_editor(self, qp.DataSet.min_value_count.__func__)
        self.sorting = meta_editor(self, qp.DataSet.sorting.__func__)
        self.slicing = meta_editor(self, qp.DataSet.slicing.__func__)
        self.set_variable_text = meta_editor(self, qp.DataSet.set_variable_text.__func__)
        self.set_value_texts = meta_editor(self, qp.DataSet.set_value_texts.__func__)
        self.set_property = meta_editor(self, qp.DataSet.set_property.__func__)
        # UNALLOWED DataSet methods
        # self.add_meta = not_implemented(qp.DataSet.add_meta.__func__)
        self.derive = not_implemented(qp.DataSet.derive.__func__)
        self.remove_items = not_implemented(qp.DataSet.remove_items.__func__)
        self.set_missings = not_implemented(qp.DataSet.set_missings.__func__)

    def _update(self):
        """
        Update Batch metadata with Batch attributes.
        """
        self._map_x_to_y()
        self._map_x_to_filter()
        self._map_y_on_y_filter()
        self._samplesize_from_batch_filter()
        attrs = self.__dict__
        for attr in ['xks', 'yks', 'variables', 'filter', 'filter_names',
                     'x_y_map', 'x_filter_map', 'y_on_y', 'y_on_y_filter',
                     'forced_names', 'summaries', 'transposed_arrays', 'verbatims',
                     'extended_yks_global', 'extended_yks_per_x',
                     'exclusive_yks_per_x', 'extended_filters_per_x', 'meta_edits',
                     'cell_items', 'weights', 'sigproperties', 'additional',
                     'sample_size', 'language', 'name', 'skip_items', 'total',
                     'unwgt_counts', 'y_filter_map', 'build_info'
                     ]:
            attr_update = {attr: attrs.get(attr, attrs.get('_{}'.format(attr)))}
            self._meta['sets']['batches'][self.name].update(attr_update)

    def _load_batch(self):
        """
        Fill batch attributes with information from meta.
        """
        bdefs = self._meta['sets']['batches'][self.name]
        for attr in ['xks', 'yks', 'variables', 'filter', 'filter_names',
                     'x_y_map', 'x_filter_map', 'y_on_y', 'y_on_y_filter',
                     'forced_names', 'summaries', 'transposed_arrays', 'verbatims',
                     'extended_yks_global', 'extended_yks_per_x',
                     'exclusive_yks_per_x', 'extended_filters_per_x', 'meta_edits',
                     'cell_items', 'weights', 'sigproperties', 'additional',
                     'sample_size', 'language', 'skip_items', 'total', 'unwgt_counts',
                     'y_filter_map', 'build_info',
                     ]:
            attr_load = {attr: bdefs.get(attr, bdefs.get('_{}'.format(attr)))}
            self.__dict__.update(attr_load)

    def clone(self, name, b_filter=None, as_addition=False):
        """
        Create a copy of Batch instance.

        Parameters
        ----------
        name: str
            Name of the Batch instance that is copied.
        b_filter: tuple (str, dict/ complex logic)
            Filter logic which is applied on the new batch.
            (filtername, filterlogic)
        as_addition: bool, default False
            If True, the new batch is added as addition to the master batch.

        Returns
        -------
        New/ copied Batch instance.
        """
        org_name = self.name
        org_meta = org_copy.deepcopy(self._meta['sets']['batches'][org_name])
        self._meta['sets']['batches'][name] = org_meta
        verbose = self._verbose_infos
        self.set_verbose_infomsg(False)
        batch_copy = self.get_batch(name)
        self.set_verbose_infomsg(verbose)
        batch_copy.set_verbose_infomsg(verbose)
        if b_filter:
            batch_copy.add_filter(b_filter[0], b_filter[1])
        if batch_copy.verbatims and b_filter and not as_addition:
            for oe in batch_copy.verbatims:
                data = self._data.copy()
                series_data = data['@1'].copy()[pd.Index(oe['idx'])]
                slicer, _ = get_logic_index(series_data, b_filter[1], data)
                oe['idx'] = slicer.tolist()
        if as_addition:
            batch_copy.as_addition(self.name)
        batch_copy._update()
        return batch_copy

    def remove(self):
        """
        Remove instance from meta object.
        """
        name = self.name
        adds = self._meta['sets']['batches'][name]['additions']
        if adds:
            for bname, bdef in self._meta['sets']['batches'].items():
                if bname == name: continue
                for add in adds[:]:
                    if add in bdef['additions']:
                        adds.remove(add)
        for add in adds:
            self._meta['sets']['batches'][add]['additional'] = False

        del(self._meta['sets']['batches'][name])
        if self._verbose_infos:
            print "Batch '%s' is removed from meta-object." % name
        self = None
        return None

    def _rename_in_additions(self, find_bname, new_name):
        for bname, bdef in self._meta['sets']['batches'].items():
            if find_bname in bdef['additions']:
                adds = bdef['additions']
                adds[adds.index(find_bname)] = new_name
                bdef['additions'] = adds
        return None

    def rename(self, new_name):
        """
        Rename instance, updating ``DataSet`` references to the definiton, too.
        """
        if new_name in self._meta['sets']['batches']:
            raise KeyError("'%s' is already included!" % new_name)
        batches = self._meta['sets']['batches']
        org_name = self.name
        batches[new_name] = batches.pop(org_name)
        self._rename_in_additions(org_name, new_name)
        self.name = new_name
        self._update()
        return None

    @modify(to_list='ci')
    def set_cell_items(self, ci):
        """
        Assign cell items ('c', 'p', 'cp').

        Parameters
        ----------
        ci: str/ list of str, {'c', 'p', 'cp'}
            Cell items used for this Batch instance.

        Returns
        -------
        None
        """
        if any(c not in ['c', 'p', 'cp'] for c in ci):
            raise ValueError("'ci' cell items must be either 'c', 'p' or 'cp'.")
        self.cell_items = ci
        self._update()
        return None

    def set_unwgt_counts(self, unwgt=False):
        """
        Assign if counts (incl. nets) should be aggregated unweighted.
        """
        self.unwgt_counts = unwgt
        self._update()
        return None

    @modify(to_list='w')
    def set_weights(self, w):
        """
        Assign a weight variable setup.

        Parameters
        ----------
        w: str/ list of str
            Name(s) of the weight variable(s).

        Returns
        -------
        None
        """
        if not w:
            w = [None]
        elif any(we is None for we in w):
            w = [None] + [we for we in w if not we is None]
        self.weights = w
        if any(weight not in self.columns() for weight in w if not weight is None):
            raise ValueError('{} is not in DataSet.'.format(w))
        self._update()
        return None

    @modify(to_list='levels')
    def set_sigtests(self, levels=None, flags=[30, 100], test_total=False, mimic=None):
        """
        Specify a significance test setup.

        Parameters
        ----------
        levels: float/ list of float
            Level(s) for significance calculation(s).
        mimic/ flags/ test_total:
            Currently not implemented.

        Returns
        -------
        None
        """
        if levels and self.total:
            if not all(isinstance(l, float) for l in levels):
                raise TypeError('All significance levels must be provided as floats!')
            levels = sorted(levels)
        else:
            levels = []

        self.sigproperties = {'siglevels': levels,
                              'test_total': test_total,
                              'flag_bases': flags,
                              'mimic': ['Dim']}
        if mimic :
            err = ("Changes to 'mimic' are currently not allowed!")
            raise NotImplementedError(err)
        self._update()
        return None

    @verify(text_keys='text_key')
    def set_language(self, text_key):
        """
        Set ``Batch.language`` indicated via the ``text_key`` for Build exports.

        Parameters
        ----------
        text_key: str
            The text_key used as language for the Batch instance

        Returns
        -------
        None
        """
        self.language = text_key
        self._update()
        return None

    def as_addition(self, batch_name):
        """
        Treat the Batch as additional aggregations, independent from the
        global Batch & Build setup.

        Parameters
        ----------
        batch_name: str
            Name of the Batch instance where the current instance is added to.

        Returns
        -------
        None
        """
        self._meta['sets']['batches'][batch_name]['additions'].append(self.name)
        self.additional = True
        self.verbatims = []
        self.y_on_y = []
        self.y_on_y_filter = {}
        if self._verbose_infos:
            msg = ("Batch '{}' specified as addition to Batch '{}'. Any open end "
                   "summaries and 'y_on_y' agg. have been removed!")
            print msg.format(self.name, batch_name)
        self._update()
        return None


    @modify(to_list='varlist')
    def add_variables(self, varlist):
        """
        Text

        Parameters
        ----------
        varlist : list
            A list of variable names.

        Returns
        -------
        None
        """
        self._variables = []
        if '@' in varlist: varlist.remove('@')
        if '@1' in varlist: varlist.remove('@1')
        for v in varlist:
            if not v in self._variables:
                self._variables.append(v)
        self._update()
        return None


    @modify(to_list='dbrk')
    def add_downbreak(self, dbrk):
        """
        Set the downbreak (x) variables of the Batch.

        Parameters
        ----------
        dbrk: str, list of str, dict, list of dict
            Names of variables that are used as downbreaks. Forced names for
            Excel outputs can be given in a dict, for example:
            xks = ['q1', {'q2': 'forced name for q2'}, 'q3', ....]

        Returns
        -------
        None
        """
        clean_xks = self._check_forced_names(dbrk)
        self.xks = self.unroll(clean_xks, both='all')
        self._update()
        masks = [x for x in self.xks if x in self.masks()]
        self.make_summaries(masks, [], _verbose=False)
        return None


    @modify(to_list='xks')
    def add_x(self, xks):
        """
        Set the x (downbreak) variables of the Batch.

        Parameters
        ----------
        xks: str, list of str, dict, list of dict
            Names of variables that are used as downbreaks. Forced names for
            Excel outputs can be given in a dict, for example:
            xks = ['q1', {'q2': 'forced name for q2'}, 'q3', ....]

        Returns
        -------
        None
        """
        w = "'add_x()' will be deprecated in a future version. Please use 'add_downbreak()' instead!"
        warnings.warn(w)
        self.add_downbreak(xks)

    @modify(to_list=['ext_xks'])
    def extend_x(self, ext_xks):
        """
        Extend downbreak variables with additional variables.

        Parameters
        ----------
        ext_xks: str/ dict, list of str/dict
            Name(s) of variable(s) that are added as downbreak. If a dict is
            provided, the variable is added in front of the belonging key.
            Example::
            >>> ext_xks = ['var1', {'existing_x': ['var2', 'var3']}]

            var1 is added at the end of the downbreaks, var2 and var3 are
            added in front of the variable existing_x.

        Returns
        -------
            None
        """
        for x in ext_xks:
            if isinstance(x, dict):
                for pos, var in x.items():
                    if not isinstance(var, list): var = [var]
                    var = self.unroll(var, both='all')
                    for v in var:
                        if not self.var_exists(pos):
                            raise KeyError('{} is not included.'.format(pos))
                        elif not v in self.xks:
                            self.xks.insert(self.xks.index(pos), v)
                        if self.is_array(v) and not v in self.summaries:
                            self.summaries.append(v)
            elif not self.var_exists(x):
                raise KeyError('{} is not included.'.format(x))
            elif x not in self.xks:
                self.xks.extend(self.unroll(x, both='all'))
                if self.is_array(x) and not x in self.summaries:
                    self.summaries.append(x)
        self._update()
        return None

    def hide_empty(self, xks=True, summaries=True):
        """
        Drop empty variables and hide array items from summaries.

        Parametes
        ---------
        xks : bool, default True
            Controls dropping "regular" variables and array items due to being
            empty.
        summaries : bool, default True
            Controls whether or not empty array items are hidden (by applying
            rules) in summary aggregations. Summaries that would end up with
            no valid items are automatically dropped altogether.

        Returns
        -------
        None
        """
        if self.filter == 'no_filter':
            cond = None
        else:
            cond = self.filter.values()[0]
        removed_sum = []
        for x in self.xks[:]:
            if self.is_array(x):
                e_items = self.empty_items(x, cond, False)
                if not e_items: continue
                sources = self.sources(x)
                if summaries:
                    self.hiding(x, e_items, axis='x', hide_values=False)
                    if len(e_items) == len(sources):
                        if x in self.xks: self.xks.remove(x)
                        if x in self.summaries: self.summaries.remove(x)
                        removed_sum.append(x)
                if xks:
                    for i in e_items:
                        if sources[i-1] in self.xks:
                            self.xks.remove(sources[i-1])
            elif not self._is_array_item(x):
                if self[self.take(cond), x].count() == 0:
                    self.xks.remove(x)
        if removed_sum:
            msg = "Dropping summaries for {} - all items hidden!"
            warnings.warn(msg.format(removed_sum))
        self._update()
        return None

    @modify(to_list=['arrays'])
    @verify(variables={'arrays': 'masks'})
    def make_summaries(self, arrays, exclusive=False, _verbose=None):
        """
        Summary tables are created for defined arrays.

        Parameters
        ----------
        arrays: str/ list of str
            List of arrays for which summary tables are created. Summary tables
            can only be created for arrays that are included in ``self.xks``.
        exclusive: bool/ list, default False
            If True only summaries are created and items skipped. ``exclusive``
            parameter can be provided for a selection of arrays. Example::
            >>> b.make_summaries(['array1', 'array2'], exclusive = ['array2'])
        Returns
        -------
        None
        """
        if _verbose is None: _verbose = self._verbose_infos
        if any(a not in self.xks for a in arrays):
            msg = '{} not defined as xks.'.format([a for a in arrays if not a in self.xks])
            raise ValueError(msg)
        self.summaries = arrays
        if exclusive:
            if isinstance(exclusive, bool):
                self.skip_items = arrays
            else:
                self.skip_items = [a for a in exclusive if a in arrays]
        else:
            self.skip_items = []
        if arrays:
            msg = 'Array summaries setup: Creating {}.'.format(arrays)
        else:
            msg = 'Array summaries setup: Creating no summaries!'
        if _verbose:
            print msg
        for t_array in self.transposed_arrays.keys():
            if not t_array in arrays:
                self.transposed_arrays.pop(t_array)
        self._update()
        return None

    @modify(to_list='arrays')
    @verify(variables={'arrays': 'masks'})
    def transpose_arrays(self, arrays, replace=False):
        """
        Transposed summary tables are created for defined arrays.

        Parameters
        ----------
        arrays: str/ list of str
            List of arrays for which transposed summary tables are created.
            Transposed summary tables can only be created for arrays that are
            included in ``self.xks``.
        replace: bool, default True
            If True only the transposed table is created, if False transposed
            and normal summary tables are created.

        Returns
        -------
        None
        """
        if any(a not in self.xks for a in arrays):
            msg = '{} not defined as xks.'.format([a for a in arrays if not a in self.xks])
            raise ValueError(msg)
        if any(a not in self.summaries for a in arrays):
            ar = list(set(self.summaries + arrays))
            a = [v for v in self.xks if v in ar]
            self.make_summaries(a, [])
        for array in arrays:
            self.transposed_arrays[array] = replace
            self._update()
        return None

    def add_total(self, total=True):
        """
        Define if '@' is added to y_keys.
        """
        if not total:
            self.set_sigtests(None)
            if self._verbose_infos:
                print 'sigtests are removed from batch.'
        self.total = total
        self.add_crossbreak(self.yks)
        return None

    @modify(to_list='xbrk')
    @verify(variables={'xbrk': 'both'}, categorical='xbrk')
    def add_crossbreak(self, xbrk):
        """
        Set the y (crossbreak/banner) variables of the Batch.

        Parameters
        ----------
        xbrk: str, list of str
            Variables that are added as crossbreaks. '@'/ total is added
            automatically.

        Returns
        -------
        None
        """
        yks = [y for y in xbrk if not y=='@']
        yks = self.unroll(yks)
        if self.total:
            yks = ['@'] + yks
        self.yks = yks
        self._update()
        return None

    @modify(to_list='yks')
    @verify(variables={'yks': 'both'}, categorical='yks')
    def add_y(self, yks):
        """
        Set the y (crossbreak/banner) variables of the Batch.

        Parameters
        ----------
        yks: str, list of str
            Variables that are added as crossbreaks. '@'/ total is added
            automatically.

        Returns
        -------
        None
        """
        w = "'add_y()' will be deprecated in a future version. Please use 'add_crossbreak()' instead!"
        warnings.warn(w)
        self.add_crossbreak(yks)

    def add_x_per_y(self, x_on_y_map):
        """
        Add individual combinations of x and y variables to the Batch.

        !!! Currently not implemented !!!
        """
        raise NotImplementedError('NOT YET SUPPPORTED')
        if not isinstance(x_on_y_map, list): x_on_y_maps = [x_on_y_map]
        if not isinstance(x_on_y_maps[0], dict):
            raise TypeError('Must pass a (list of) dicts!')
        for x_on_y_map in x_on_y_maps:
            for x, y in x_on_y_map.items():
                if not isinstance(y, list): y = [y]
                if isinstance(x, tuple): x = {x[0]: x[1]}
        return None

    def add_filter(self, filter_name, filter_logic=None, overwrite=False):
        """
        Apply a (global) filter to all the variables found in the Batch.

        Parameters
        ----------
        filter_name: str
            Name for the added filter.
        filter_logic: complex logic
            Logic for the added filter.

        Returns
        -------
        None
        """
        name = filter_name.encode('utf8').replace(' ', '_').replace('~', '_')
        if self.is_filter(name):
            if not (filter_logic is None or overwrite):
                raise ValueError("'{}' is already a filter-variable. Cannot "
                                 "apply a new logic.".format(name))
            elif overwrite:
                self.drop(name)
                print 'Overwrite filter var: {}'.format(name)
                self.add_filter_var(name, filter_logic, overwrite)

        else:
            self.add_filter_var(name, filter_logic, overwrite)
        self.filter = name
        if not name in self.filter_names:
            self.filter_names.append(name)
        self._update()
        return None

    def remove_filter(self):
        """
        Remove all defined (global + extended) filters from Batch.
        """
        self.filter = None
        self.filter_names = []
        self.extended_filters_per_x = {}
        self.y_on_y_filter = {}
        self._update()
        return None

    @modify(to_list=['oe', 'break_by', 'title'])
    @verify(variables={'oe': 'columns', 'break_by': 'columns'})
    def add_open_ends(self, oe, break_by=None, drop_empty=True, incl_nan=False,
                      replacements=None, split=False, title='open ends',
                      filter_by=None, overwrite=True):
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
        replacements: dict, default None
            Replace strings in data.
        split: bool, default False
            If True len of oe must be same size as len of title. Each oe is
            saved with its own title.
        title : str, default 'open ends'
            Specifies the the ``Cluster`` / Excel sheet name for the output.
        filter_by : A Quantipy logical expression, default None
            An additional logical filter that should be applied to the case data.
            Any ``filter`` provided by a ``batch`` will be respected
            automatically.
        overwrite : bool, default False
            If True and used title is already existing in self.verbatims, then
            it gets overwritten

        Returns
        -------
        None
        """
        if self.additional:
            err_msg = "Cannot add open end DataFrames to as_addition()-Batches!"
            raise NotImplementedError(err_msg)
        dupes = [v for v in oe if v in break_by]
        if dupes:
            raise ValueError("'{}' included in oe and break_by.".format("', '".join(dupes)))
        def _add_oe(oe, break_by, title, drop_empty, incl_nan, filter_by, overwrite):
            if filter_by:
                f_name = title if not self.filter else '%s_%s' % (self.filter, title)
                f_name = self._verify_filter_name(f_name, number=True)
                logic = {'label': title, 'logic': filter_by}
                if self.filter:
                    suffix = f_name[len(self.filter)+1:]
                    self.extend_filter_var(self.filter, logic, suffix)
                else:
                    self.add_filter_var(f_name, logic)
                slicer = f_name
            else:
                slicer = self.filter
            if any(oe['title'] == title for oe in self.verbatims) and not overwrite:
                return None
            oe = {
                'title': title,
                'filter': slicer,
                'columns': oe,
                'break_by': break_by,
                'incl_nan': incl_nan,
                'drop_empty': drop_empty,
                'replace': replacements}
            if any(o['title'] == title for o in self.verbatims):
                for x, o in enumerate(self.verbatims):
                    if o['title'] == title:
                        self.verbatims[x] = oe
            else:
                self.verbatims.append(oe)

        if len(oe) + len(break_by) == 0:
            raise ValueError("Please add any variables as 'oe' or 'break_by'.")
        if split:
            if not len(oe) == len(title):
                msg = "Cannot derive verbatim DataFrame 'title' with more than 1 'oe'"
                raise ValueError(msg)
            for t, open_end in zip(title, oe):
                open_end = [open_end]
                _add_oe(open_end, break_by, t, drop_empty, incl_nan, filter_by, overwrite)
        else:
            _add_oe(oe, break_by, title[0], drop_empty, incl_nan, filter_by, overwrite)
        self._update()
        return None

    @modify(to_list=['ext_yks', 'on'])
    @verify(variables={'ext_yks': 'columns'})
    def extend_y(self, ext_yks, on=None):
        """
        Add y (crossbreak/banner) variables to specific x (downbreak) variables.

        Parameters
        ----------
        ext_yks: str/ dict, list of str/ dict
            Name(s) of variable(s) that are added as crossbreak. If a dict is
            provided, the variable is added in front of the beloning key.
            Example::
            >>> ext_yks = ['var1', {'existing_y': ['var2', 'var3']}]

            var1 is added at the end of the crossbreaks, var2 and var3 are
            added in front of the variable existing_y.
        on: str/ list of str
            Name(s) of variable(s) in the xks (downbreaks) for which the
            crossbreak should be extended.

        Returns
        -------
        None
        """
        ext_yks = [e for e in ext_yks if not e in self.yks]
        if not on:
            self.yks.extend(ext_yks)
            if not self.extended_yks_global:
                self.extended_yks_global = ext_yks
            else:
                self.extended_yks_global.extend(ext_yks)
        else:
            if any(o not in self.xks for o in on):
                msg = '{} not defined as xks.'.format([o for o in on if not o in self.xks])
                raise ValueError(msg)
            on = self.unroll(on, both='all')
            for x in on:
                x_ext_yks = [e for e in ext_yks
                             if not e in self.extended_yks_per_x.get(x, [])]
                self.extended_yks_per_x.update({x: x_ext_yks})
        self._update()
        return None

    @modify(to_list=['new_yks', 'on'])
    @verify(variables={'new_yks': 'both', 'on': 'both'})
    def replace_y(self, new_yks, on):
        """
        Replace y (crossbreak/banner) variables on specific x (downbreak) variables.

        Parameters
        ----------
        ext_yks: str/ list of str
            Name(s) of variable(s) that are used as crossbreak.
        on: str/ list of str
            Name(s) of variable(s) in the xks (downbreaks) for which the
            crossbreak should be replaced.

        Returns
        -------
        None
        """
        if any(o not in self.xks for o in on):
            msg = '{} not defined as xks.'.format([o for o in on if not o in self.xks])
            raise ValueError(msg)
        on = self.unroll(on, both='all')
        if not '@' in new_yks: new_yks = ['@'] + new_yks
        for x in on:
            self.exclusive_yks_per_x.update({x: new_yks})
        self._update()
        return None

    def extend_filter(self, ext_filters):
        """
        Apply additonal filtering to specific x (downbreak) variables.

        Parameters
        ----------
        ext_filters: dict
            dict with variable name(s) as key, str or tupel of str, and logic
            as value. For example:
            ext_filters = {'q1': {'gender': 1}, ('q2', 'q3'): {'gender': 2}}

        Returns
        -------
        None
        """
        for variables, logic in ext_filters.items():
            if not isinstance(variables, (list, tuple)):
                variables = [variables]
            for v in variables:
                if self.filter:
                    log = {'label': v, 'logic': logic}
                    f_name = '{}_{}'.format(self.filter, v)
                    self.extend_filter_var(self.filter, log, v)
                else:
                    f_name = '{}_f'.format(v)
                    self.add_filter_var(f_name, log)
                self.extended_filters_per_x.update({v: f_name})
        self._update()
        return None

    def add_y_on_y(self, name, y_filter=None, main_filter='extend'):
        """
        Produce aggregations crossing the (main) y variables with each other.

        Parameters
        ----------
        name: str
            key name for the y on y aggregation.
        y_filter: str (filter var name) or dict (complex logic), default None
            Add a filter for the y on y aggregation. If None is provided
            the main batch filter is taken ('extend') or no filter logic is
            applied ('replace').
        main_filter: {'extend', 'replace'}, default 'extend'
            Defines if the main batch filter is extended or
            replaced by the y_on_y filter.

        Note:
            If the y_filter is provided as a str (filter var name),
            main_filter is automatically set to 'replace'.

        Returns
        -------
        None
        """
        if not isinstance(name, basestring):
            raise TypeError("'name' attribute for add_y_on_y must be a str!")
        elif not main_filter in ['extend', 'replace'] or main_filter is None:
            raise ValueError("'main_filter' must be either 'extend' or 'replace'.")
        if not name in self.y_on_y:
            self.y_on_y.append(name)
        if isinstance(y_filter, basestring):
            if not self.is_filter(y_filter):
                raise ValueError('{} is not a valid filter var.'.format(y_filter))
            else:
                main_filter = 'replace'
        self.y_on_y_filter[name] = (main_filter, y_filter)
        self._update()
        return None

    def _map_x_to_y(self):
        """
        Combine all defined cross and downbreaks in a map.

        Returns
        -------
        None
        """
        def _order_yks(yks):
            y_keys = []
            for y in yks:
                if isinstance(y, dict):
                    for pos, var in y.items():
                        if not isinstance(var, list): var = [var]
                        for v in var:
                            if not v in y_keys:
                                y_keys.insert(y_keys.index(pos), v)
                elif not y in y_keys:
                    y_keys.append(y)
            return y_keys

        def _get_yks(x):
            if x in self.exclusive_yks_per_x:
                yks = self.exclusive_yks_per_x[x]
            else:
                yks = org_copy.deepcopy(self.yks)
                yks.extend(self.extended_yks_per_x.get(x, []))
                yks = _order_yks(yks)
            return yks

        mapping = []
        for x in self.xks:
            if self.is_array(x):
                if x in self.summaries and not self.transposed_arrays.get(x):
                    mapping.append((x, ['@']))
                if not x in self.skip_items:
                    try:
                        hiding = self.meta_edits[x]['rules']['x']['dropx']['values']
                    except:
                        hiding = self._get_rules(x).get('dropx', {}).get('values', [])
                    for x2 in self.sources(x):
                        if x2 in hiding:
                            continue
                        elif x2 in self.xks:
                            mapping.append((x2, _get_yks(x2)))
                if x in self.transposed_arrays:
                    mapping.append(('@', [x]))
            elif self._is_array_item(x) and self._maskname_from_item(x) in self.xks:
                continue
            else:
                mapping.append((x, _get_yks(x)))
        self.x_y_map = mapping
        return None

    def _map_x_to_filter(self):
        """
        Combine all defined downbreaks with its beloning filter in a map.

        Returns
        -------
        None
        """
        mapping = {}
        for x in self.xks:
            if self._is_array_item(x):
                continue
            name = self.extended_filters_per_x.get(x, self.filter)
            mapping[x] = name
            if self.is_array(x):
                for x2 in self.sources(x):
                    if x2 in self.xks:
                        mapping[x2] = name
            if name and not name in self.filter_names:
                self.filter_names.append(name)
        self.x_filter_map = mapping
        return None

    def _map_y_on_y_filter(self):
        """
        Get all y_on_y filters and map them with the main filter.
        Returns
        -------
        None
        """
        self.y_filter_map = {}
        for y_on_y in self.y_on_y:
            ext_rep, y_f = self.y_on_y_filter[y_on_y]
            logic = {'label': y_on_y, 'logic': y_f}
            if ext_rep == 'replace':
                if not y_f:
                    f = None
                elif isinstance(y_f, basestring):
                    f = y_f
                else:
                    f = self._verify_filter_name(y_on_y, number=True)
                    self.add_filter_var(f, logic)
            elif ext_rep == 'extend':
                if not y_f:
                    f = self.filter
                elif not self.filter:
                    f = self._verify_filter_name(y_on_y, number=True)
                    self.add_filter_var(f, logic)
                else:
                    f = '{}_{}'.format(self.filter, y_on_y)
                    f = self._verify_filter_name(f, number=True)
                    suf = f[len(self.filter)+1:]
                    self.extend_filter_var(self.filter, logic, suf)
            self.y_filter_map[y_on_y] = f
        return None

    def _check_forced_names(self, variables):
        """
        Store forced names for xks and return adjusted list of downbreaks.

        Parameters
        ----------
        variables: list of str/dict/tuple
            Variables that are checked. If a dict or tupel is provided, the
            key/ first item is used as variable name and the value/ second
            item as forced name.

        Returns
        -------
        xks: list of str
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
            if not self.var_exists(xks[-1]):
                raise ValueError('{} is not in DataSet.'.format(xks[-1]))
        self.forced_names = renames
        return xks

    def _samplesize_from_batch_filter(self):
        """
        Calculates sample_size from existing filter.
        """
        if self.filter:
            idx = self.manifest_filter(self.filter)
        else:
            idx = self._data.index
        self.sample_size = len(idx)
        return None
