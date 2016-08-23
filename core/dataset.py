 #!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

import quantipy as qp
from quantipy.core.tools.dp.io import (
    read_quantipy as r_quantipy,
    read_dimensions as r_dimensions,
    read_decipher as r_decipher,
    read_spss as r_spss,
    read_ascribe as r_ascribe,
    write_spss as w_spss,
    write_quantipy as w_quantipy)
from quantipy.core.helpers.functions import emulate_meta
from quantipy.core.tools.view.logic import (
    has_any, has_all, has_count,
    not_any, not_all, not_count,
    is_lt, is_ne, is_gt,
    is_le, is_eq, is_ge,
    union, intersection, get_logic_index)
from quantipy.core.tools.dp.prep import (
    hmerge as _hmerge,
    vmerge as _vmerge,
    recode as _recode)

from cache import Cache

import copy as org_copy
import json
import warnings

class DataSet(object):
    """
    A set of casedata (required) and meta data (optional).

    DESC.
    """
    def __init__(self, name):
        self.path = None
        self.name = name
        self.filtered = 'no_filter'
        self._data = None
        self._meta = None
        self._tk = None
        self._cache = Cache()

    # ------------------------------------------------------------------------
    # ITEM ACCESS / OVERRIDING
    # ------------------------------------------------------------------------
    def __getitem__(self, var):
        var = self._prep_varlist(var)
        if len(var) == 1: var = var[0]
        return self._data[var]

    def __setitem__(self, name, val):
        self._data[name] = val

    # ------------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------------
    def read_quantipy(self, path_meta, path_data):
        """
        Load Quantipy .csv/.json files, connecting as data and meta components.

        Parameters
        ----------
        path_meta : str
            The full path (optionally with extension ``'.json'``, otherwise
            assumed as such) to the meta data defining ``'.json'`` file.
        path_data : str
            The full path (optionally with extension ``'.csv'``, otherwise
            assumed as such) to the case data defining ``'.csv'`` file.

        Returns
        -------
        None
            The ``DataSet`` is modified inplace, connected to Quantipy native
            data and meta components.
        """
        if path_meta.endswith('.json'): path_meta = path_meta.replace('.json', '')
        if path_data.endswith('.csv'): path_data = path_data.replace('.csv', '')
        self._meta, self._data = r_quantipy(path_meta+'.json', path_data+'.csv')
        self._set_file_info(path_data, path_meta)
        return None

    def read_dimensions(self, path_meta, path_data):
        if path_meta.endswith('.mdd'): path_meta = path_meta.replace('.mdd', '')
        if path_data.endswith('.ddf'): path_data = path_data.replace('.ddf', '')
        self._meta, self._data = r_dimensions(path_meta+'.mdd', path_data+'.ddf')
        self._set_file_info(path_data, path_meta)

    def read_spss(self, path_sav, **kwargs):
        if path_sav.endswith('.sav'): path_sav = path_sav.replace('.sav', '')
        self._meta, self._data = r_spss(path_sav+'.sav', ioLocale=None)
        self._set_file_info(path_sav)

    def write_quantipy(self, path_meta=None, path_data=None):
        meta, data = self._meta, self._data
        if path_data is None and path_meta is None:
            path = self.path
            name = self.name
            path_meta = '{}/{}.json'.format(path, name)
            path_data = '{}/{}.csv'.format(path, name)
        w_quantipy(meta, data, path_meta, path_data)

    def _set_file_info(self, path_data, path_meta=None):
        self.path = '/'.join(path_data.split('/')[:-1]) + '/'
        if path_meta:
            self._tk = self._meta['lib']['default text']
        else:
            self._tk = None
        self._data['@1'] = np.ones(len(self._data))
        self._meta['columns']['@1'] = {'type': 'int'}
        self._data.index = list(xrange(0, len(self._data.index)))
        return None

    def split(self, save=False):
        meta, data = self._meta, self._data
        if save:
            path = self.path
            name = self.name
            w_quantipy(meta, data, path+name+'.json', path+name+'.csv')
        return meta, data

    def copy(self):
        copied = org_copy.deepcopy(self)
        return copied

    def data(self):
        return self._data

    # NEW !!!!
    def meta(self, name=None):
        if not name:
            return self._meta
        else:
            self.show_meta(self._meta['columns'][name])
            return None

    def cache(self):
        return self._cache

    # NEW !!!!
    def show_meta(self, obj, indent=True):
        def represent(obj):
            if isinstance(obj, np.generic):
                return np.asscalar(obj)
            else:
                return repr(obj)
        print json.dumps(
            obj,
            sort_keys=True,
            indent=4 if indent else None,
            default=represent)

    # ------------------------------------------------------------------------
    # Extending DataSets
    # ------------------------------------------------------------------------
    def hmerge(self, dataset, on=None, left_on=None, right_on=None,
               overwrite_text=False, from_set=None, inplace=True, verbose=True):

        """
        Merge Quantipy datasets together using an index-wise identifer.

        This function merges two Quantipy datasets together, updating variables
        that exist in the left dataset and appending others. New variables
        will be appended in the order indicated by the 'data file' set if
        found, otherwise they will be appended in alphanumeric order.
        This merge happend horizontally (column-wise). Packed kwargs will be
        passed on to the pandas.DataFrame.merge() method call, but that merge
        will always happen using how='left'.

        Parameters
        ----------
        dataset : ``quantipy.DataSet``
            The dataset to merge into the current ``DataSet``.
        on : str, default=None
            The column to use as a join key for both datasets.
        left_on : str, default=None
            The column to use as a join key for the left dataset.
        right_on : str, default=None
            The column to use as a join key for the right dataset.
        overwrite_text : bool, default=False
            If True, text_keys in the left meta that also exist in right
            meta will be overwritten instead of ignored.
        from_set : str, default=None
            Use a set defined in the right meta to control which columns are
            merged from the right dataset.
        inplace : bool, default True
            If True, the ``DataSet`` will be modified inplace with new/updated
            columns. Will return a new ``DataSet`` instance if False.
        verbose : bool, default=True
            Echo progress feedback to the output pane.

        Returns
        -------
        None or new_dataset : ``quantipy.DataSet``
            If the merge is not applied ``inplace``, a ``DataSet`` instance
            is returned.
        """
        ds_left = (self._meta, self._data)
        ds_right = (dataset._meta, dataset._data)
        merged_meta, merged_data = _hmerge(
            ds_left, ds_right, on=on, left_on=left_on, right_on=right_on,
            overwrite_text=overwrite_text, from_set=from_set, verbose=verbose)
        if inplace:
            self._data = merged_data
            self._meta = merged_meta
            return None
        else:
            new_dataset = self.copy()
            new_dataset._data = merged_data
            new_dataset._meta = merged_meta
            return new_dataset

    def update(self, data, on='identity'):
        """
        """
        ds_left = (self._meta, self._data)
        update_meta = self._meta.copy()
        update_items = ['columns@{}'.format(name) for name
                        in data.columns.tolist()]
        update_meta['sets']['update'] = {'items': update_items}
        ds_right = (update_meta, data)
        merged_meta, merged_data = _hmerge(
            ds_left, ds_right, on=on, from_set='update', verbose=False)
        self._meta, self._data = merged_meta, merged_data
        del self._meta['sets']['update']
        return None

    def vmerge(self, dataset, on=None, left_on=None, right_on=None,
               row_id_name=None, left_id=None, right_id=None, row_ids=None,
               overwrite_text=False, from_set=None, reset_index=True,
               inplace=True, verbose=True):
        """
        Merge Quantipy datasets together by appending rows.

        This function merges two Quantipy datasets together, updating variables
        that exist in the left dataset and appending others. New variables
        will be appended in the order indicated by the 'data file' set if
        found, otherwise they will be appended in alphanumeric order. This
        merge happens vertically (row-wise).

        Parameters
        ----------
        dataset : (A list of multiple) ``quantipy.DataSet``
            One or multiple datasets to merge into the current ``DataSet``.
        on : str, default=None
            The column to use to identify unique rows in both datasets.
        left_on : str, default=None
            The column to use to identify unique in the left dataset.
        right_on : str, default=None
            The column to use to identify unique in the right dataset.
        row_id_name : str, default=None
            The named column will be filled with the ids indicated for each
            dataset, as per left_id/right_id/row_ids. If meta for the named
            column doesn't already exist a new column definition will be
            added and assigned a reductive-appropriate type.
        left_id : str/int/float, default=None
            Where the row_id_name column is not already populated for the
            dataset_left, this value will be populated.
        right_id : str/int/float, default=None
            Where the row_id_name column is not already populated for the
            dataset_right, this value will be populated.
        row_ids : list of str/int/float, default=None
            When datasets has been used, this list provides the row ids
            that will be populated in the row_id_name column for each of
            those datasets, respectively.
        overwrite_text : bool, default=False
            If True, text_keys in the left meta that also exist in right
            meta will be overwritten instead of ignored.
        from_set : str, default=None
            Use a set defined in the right meta to control which columns are
            merged from the right dataset.
        reset_index : bool, default=True
            If True pandas.DataFrame.reindex() will be applied to the merged
            dataframe.
        inplace : bool, default True
            If True, the ``DataSet`` will be modified inplace with new/updated
            rows. Will return a new ``DataSet`` instance if False.
        verbose : bool, default=True
            Echo progress feedback to the output pane.

        Returns
        -------
        None or new_dataset : ``quantipy.DataSet``
            If the merge is not applied ``inplace``, a ``DataSet`` instance
            is returned.
        """
        if isinstance(dataset, list):
            dataset_left = None
            dataset_right = None
            datasets = [(self._meta, self._data)]
            merge_ds = [(ds._meta, ds._data) for ds in dataset]
            datasets.extend(merge_ds)
        else:
            dataset_left = (self._meta, self._data)
            dataset_right = (dataset._meta, dataset._data)
            datasets = None
        merged_meta, merged_data = _vmerge(
            dataset_left, dataset_right, datasets, on=on, left_on=left_on,
            right_on=right_on, row_id_name=row_id_name, left_id=left_id,
            right_id=right_id, row_ids=row_ids, overwrite_text=overwrite_text,
            from_set=from_set, reset_index=reset_index, verbose=verbose)
        if inplace:
            self._data = merged_data
            self._meta = merged_meta
            return None
        else:
            new_dataset = self.copy()
            new_dataset._data = merged_data
            new_dataset._meta = merged_meta
            return new_dataset


    # ------------------------------------------------------------------------
    # META INSPECTION/MANIPULATION/EDITING/HANDLING
    # ------------------------------------------------------------------------
    def copy_var(self, name, suffix='rec'):
        """
        Copy meta and case data of the variable defintion given per ``name``.

        Parameters
        ----------
        name : str
            The originating column variable name keyed in ``meta['columns']``.
        suffix : str, default 'rec'
            The new variable name will be constructed by suffixing the original
            ``name`` with ``_suffix``, e.g. ``'age_rec``.

        Returns
        -------
        None
            DataSet is modified inplace, adding a copy to both the data and meta
            component.
        """
        if self._is_array(name):
            raise NotImplementedError('Cannot copy array masks!')
        copy_name = '{}_{}'.format(name, suffix)
        self._data[copy_name] = self._data[name].copy()
        meta_copy = copy.deepcopy(self._meta['columns'][name])
        self._meta['columns'][copy_name] = meta_copy
        self._meta['sets']['data file']['items'].append('columns@' + copy_name)

    def rename(self, name, new_name):
        """
        Change meta and case name references of the variable defintion.

        Parameters
        ----------
        name : str
            The originating column variable name keyed in ``meta['columns']``.
        new_name : str
            The new variable name.

        Returns
        -------
        None
            DataSet is modified inplace. The new name reference is placed into
            both the data and meta component.
        """
        if self._is_array(name):
            raise NotImplementedError('Cannot rename array masks!')
        if new_name in self._data.columns:
            msg = "Cannot rename '{}' into '{}'. Column name already exists!"
            raise ValueError(msg.format(name, new_name))
        self._data.rename(columns={name: new_name}, inplace=True)
        self._meta['columns'][new_name] = self._meta['columns'][name].copy()
        del self._meta['columns'][name]
        old_set_entry = 'columns@{}'.format(name)
        new_set_entry = 'columns@{}'.format(new_name)
        new_datafile_items = [i if i != old_set_entry else new_set_entry for i
                              in self._meta['sets']['data file']['items']]
        self._meta['sets']['data file']['items'] = new_datafile_items
        return None

    def add_meta(self, name, qtype, label, categories=None, text_key=None):
        """
        Create and insert a well-formed meta object into the existing meta document.

        Parameters
        ----------
        name : str
            The column variable name keyed in ``meta['columns']``.
        qtype : [``int``, ``float``, ``single``, ``delimited set``]
            The structural type of the data the meta describes.
        label : str
            The ``text`` label information.
        categories : list of str or tuples in form of (int, str), default None
            When a list of str is given, the categorical values will simply be
            enumerated and maped to the category labels. Alternatively codes can
            mapped to categorical labels, e.g.:
            [(1, 'Elephant'), (2, 'Mouse'), (999, 'No animal')]
        text_key : str, default project.LANGUAGE
            Text key for text-based label information. Uses the ``project.py``
            information by default.

        Returns
        -------
        None
        """
        if name in self._meta['columns']:
            msg = "Cannot create meta for '{}', it already exists!"
            print msg.format(name)
            return None
        if not text_key: text_key = self._tk
        categorical = ['delimited set', 'single']
        numerical = ['int', 'float']
        if not qtype in ['delimited set', 'single', 'float', 'int']:
            raise NotImplementedError('Type {} data unsupported'.format(qtype))
        if qtype in categorical and not categories:
            val_err = "Must provide 'categories' when requesting data of type {}."
            raise ValueError(val_err.format(qtype))
        elif qtype in numerical and categories:
            val_err = "Numerical data of type {} does not accept 'categories'."
            raise ValueError(val_err.format(qtype))
        else:
            if not isinstance(categories, list) and qtype in categorical:
                raise TypeError("'Categories' must be a list of labels "
                                "('str') or  a list of tuples of codes ('int') "
                                "and lables ('str').")
        new_meta = {'text': {text_key: label}, 'type': qtype}
        if categories:
            if isinstance(categories[0], dict):
                new_meta['values'] = categories
            else:
                new_meta['values'] = self._make_value_list(categories, text_key)
        self._meta['columns'][name] = new_meta
        self._meta['sets']['data file']['items'].append('columns@{}'.format(name))
        return None

    def recode(self, target, mapper, default=None, append=False,
               intersect=None, initialize=None, fillna=None, inplace=True):
        """
        """
        meta = self._meta
        data = self._data
        if not target in meta['columns']:
            raise ValueError(("{} not found in meta['columns'].",
                              "Please create meta data first!").format(target))
        recode_series = _recode(meta, data, target, mapper,
                                default, append, intersect, initialize, fillna)
        if inplace:
            self._data[target] = recode_series
            self._verify_data_vs_meta_codes(target)
            return None
        else:
            return recode_series

    def derive_categorical(self, name, label, qtype, cond_map, text_key=None):
        """
        """
        if not text_key: text_key = self._tk
        append = qtype == 'delimited set'
        categories = [(cond[0], cond[1]) for cond in cond_map]
        idx_mapper = {cond[0]: cond[2] for cond in cond_map}
        self.add_meta(name, qtype, label, categories, text_key)
        self.recode(name, idx_mapper, append=append)
        return None

    def _make_value_list(self, categories, text_key, start_at=None):
        if not start_at:
            start_at = 1
        if not all([isinstance(cat, tuple) for cat in categories]):
            vals = [self._value(no, text_key, lab) for no, lab in
                    enumerate(categories, start_at)]
        else:
            vals = [self._value(cat[0], text_key, cat[1]) for cat in categories]
        return vals

    @staticmethod
    def _value(value, text_key, text):
        """
        Return a well-formed Quantipy value object from the given arguments.

        Parameters
        ----------
        value : int
            The numeric value to be given to the returned value object.
        text_key : str
            The text key to be used when genereating the returned value
            object's text object.
        text : str
            The label to be given to the returned value object.
        """

        return {'value': value, 'text': {text_key: text}}

    def _clean_missing_map(self, var, missing_map):
        """
        Generate a map of missings that only contains valid flag names
        and existing meta value texts.
        """
        valid_flags = ['d.exclude', 'exclude']
        valid_codes = self._get_valuemap(var, non_mapped='codes')
        valid_map = {}
        for mtype, mcodes in missing_map.items():
            if mtype in valid_flags:
                codes = [c for c in mcodes if c in valid_codes]
                if codes: valid_map[mtype] = codes
        return valid_map


    def set_missings(self, var=None, missing_map='default', ignore=None):
        """
        Flag category defintions for exclusion in aggregations.

        Parameters
        ----------
        var : str or list of str
            Variable(s) to apply the meta flags to.
        missing_map: 'default' or dict of {code(s): 'flag'}, default 'default'
            A mapping of codes to flags that can either be 'exclude' (globally
            ignored) or 'd.exclude' (only ignored in descriptive statistics).
            Passing 'default' is using a preset list of (TODO: specify) value
            for exclusion.
        ignore : str or list of str, default None
            A list of variables that should be ignored when applying missing
            flags via the 'default' list method.

        Returns
        -------
        None
        """
        if not missing_map == 'default':
            missing_map = self._clean_missing_map(var, missing_map)
        var = self._prep_varlist(var)
        ignore = self._prep_varlist(ignore, keep_unexploded=True)
        if missing_map == 'default':
            self._set_default_missings(ignore)
        else:
            for v in var:
                if self._has_missings(v):
                    self.meta()['columns'][v].update({'missings': missing_map})
                else:
                    self.meta()['columns'][v]['missings'] = missing_map
        return None

    def reorder_codes(self, name, new_order):
        """
        Apply a new order to the value codes defined by the meta data component.

        Parameters
        ----------
        name : str
            The originating column variable name keyed in ``meta['columns']``.
        new_order : list of int
            The new code order of the DataSet variable.

        Returns
        -------
        None
            DataSet is modified inplace.
        """
        if self._is_array(name):
            raise NotImplementedError('Cannot reorder codes on array masks!')
        self._verify_old_vs_new_codes(name, new_order)
        values = self._get_value_loc(name)
        new_values = [value for i in new_order for value in values
                      if value['value'] == i]
        # LEFT IN FOR LATER - WILL CURRENTLY RAISE WHEN INPUT IS ARRAY
        if self._get_type(name) == 'array':
            self._meta['lib']['values'][name] = new_values
        else:
            self._meta['columns'][name]['values'] = new_values
        return None

    def remove_codes(self, name, remove):
        """
        Erase value codes safely from both meta and case data components.

        Parameters
        ----------
        name : str
            The originating column variable name keyed in ``meta['columns']``.
        remove : int or list of int
            The codes to be removed from the DataSet variable.

        Returns
        -------
        None
            DataSet is modified inplace.
        """
        if self._is_array(name):
            raise NotImplementedError('Cannot remove codes from array masks!')
        if not isinstance(remove, list): remove = [remove]
        values = self._get_value_loc(name)
        new_values = [value for i in remove for value in values
                      if value['value'] not in remove]
        # LEFT IN FOR LATER - WILL CURRENTLY RAISE WHEN INPUT IS ARRAY
        if self._get_type(name) == 'array':
            self._meta['lib']['values'][name] = new_values
        else:
            self._meta['columns'][name]['values'] = new_values
        if self._is_delimited_set(name):
            self._remove_from_delimited_set_data(name, remove)
        else:
            self._data.replace(r, np.NaN, inplace=True)
        self._verify_data_vs_meta_codes(name)
        return None

    def _remove_from_delimited_set_data(self, name, remove):
        """
        """
        data = self._data[name].copy()
        data.replace(np.NaN, '-NAN-', inplace=True)
        data = data.apply(lambda x: x.split(';'))
        data = data.apply(lambda x: x[0] if (x == ['-NAN-'] or x == [''])
                          else x)
        data = data.apply(lambda x: [c for c in x if c != ''
                                     and int(c) not in remove]
                                     if isinstance(x, list) else x)
        data = data.apply(lambda x: ';'.join(x) + ';' if x != '-NAN-'
                          else np.NaN)
        self._data[name] = data
        return None


    def describe(self, var=None, type=None, text_key=None):
        """
        Inspect the DataSet's global or variable level structure.
        """
        if text_key is None: text_key = self._tk
        if var is not None:
            return self._get_meta(var, type, text_key)
        if self._meta['columns'] is None:
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
                'array': [],
                'N/A': []
            }
            not_found = []
            for col in self._data.columns:
                if not col in ['@1', 'id_L1', 'id_L1.1']:
                    try:
                        types[
                              self._meta['columns'][col]['type']
                             ].append(col)
                    except:
                        types['N/A'].append(col)
            for mask in self._meta['masks'].keys():
                types[self._meta['masks'][mask]['type']].append(mask)
            idx_len = max([len(t) for t in types.values()])
            for t in types.keys():
                typ_padded = types[t] + [''] * (idx_len - len(types[t]))
                types[t] = typ_padded
            types = pd.DataFrame(types)
            types.columns.name = 'size: {}'.format(len(self._data))
            if type:
                types = pd.DataFrame(types[type]).replace('', np.NaN)
                types = types.dropna()
                types.columns.name = 'count: {}'.format(len(types))
            return types

    def unmask(self, var):
        if not self._is_array(var):
            raise KeyError('{} is not a mask.'.format(var))
        else:
            return self._get_itemmap(var=var, non_mapped='items')

    def _set_default_missings(self, ignore=None):
        excludes = ['weißnicht', 'keineangabe', 'weißnicht/keineangabe',
                    'keineangabe/weißnicht', 'kannmichnichterinnern',
                    'weißichnicht', 'nichtindeutschland']
        d = self.describe()
        cats = []
        valids = ['array', 'single', 'delimited set']
        for valid in valids:
            cats.extend(d[valid].replace('', np.NaN).dropna().values.tolist())
        for cat in cats:
            if cat not in ignore:
                flags_code = []
                vmap = self._get_valuemap(cat)
                for exclude in excludes:
                    code = self._code_from_text(vmap, exclude)
                    if code:
                        flags_code.append(code)
                if flags_code:
                    self.set_missings(cat, {'exclude': tuple(flags_code)})

    def _get_missing_map(self, var):
        if self._is_array(var):
            var = self._get_itemmap(var, non_mapped='items')
        else:
            if not isinstance(var, list): var = [var]
        for v in var:
            if self._has_missings(v):
                return self.meta()['columns'][v]['missings']
            else:
                return None

    def _get_missing_list(self, var, globally=True):
        if self._has_missings(var):
            miss = self._get_missing_map(var)
            if globally:
                return miss['exclude']
            else:
                miss_list = []
                for miss_type in miss.keys():
                    miss_list.extend(miss[miss_type])
                return miss_list
        else:
            return None

    def _prep_varlist(self, varlist, keep_unexploded=False):
        if varlist:
            if not isinstance(varlist, list): varlist = [varlist]
            clean_varlist = []
            for v in varlist:
                if self._is_array(v):
                    clean_varlist.extend(self._get_itemmap(v, non_mapped='items'))
                    if keep_unexploded: clean_varlist.append(v)
                else:
                    clean_varlist.append(v)
            return clean_varlist
        else:
            return [varlist]

    def _code_from_text(self, valuemap, text):
        check = dict(valuemap)
        for c, t in check.items():
            t = t.replace(' ', '').lower()
            if t == text: return c

    def _get_type(self, var):
        if var in self._meta['masks'].keys():
            return self._meta['masks'][var]['type']
        else:
             return self._meta['columns'][var]['type']

    def _has_missings(self, var):
        if self._get_type(var) == 'array':
            var = self._get_itemmap(var, non_mapped='items')[0]
        return 'missings' in self.meta()['columns'][var].keys()

    def _is_numeric(self, var):
        return self._get_type(var) in ['float', 'int']

    def _is_array(self, var):
        return self._get_type(var) == 'array'

    def _is_multicode_array(self, mask_element):
        return self[mask_element].dtype == 'object'

    def _is_delimited_set(self, name):
        return self._meta['columns'][name]['type'] == 'delimited set'

    def _verify_data_vs_meta_codes(self, name):
        """
        """
        if self._is_delimited_set(name):
            data_codes = self._data[name].str.get_dummies(';').columns.tolist()
            data_codes = [int(c) for c in data_codes]
        else:
            data_codes = pd.get_dummies(self._data[name]).columns.tolist()
        meta_codes = self._get_valuemap(name, non_mapped='codes')
        wild_codes = [code for code in data_codes if code not in meta_codes]
        if wild_codes:
            msg = "Warning: Meta not consistent with case data for '{}'!"
            print '*' * 60
            print msg.format(name)
            print '*' * 60
            print 'Found in data: {}'.format(data_codes)
            print 'Defined as per meta: {}'.format(meta_codes)
            raise ValueError('Please review your data processing!')
        return None

    def _verify_old_vs_new_codes(self, name, new_codes):
        """
        """
        org_codes = [value['value'] for value in self._get_value_loc(name)]
        equal = set(org_codes) == set(new_codes)
        if not equal:
            missing_codes = [c for c in org_codes if c not in new_codes]
            wild_codes = [c for c in new_codes if c not in org_codes]
            print '*' * 60
            if missing_codes:
                msg = "Warning: Code order is incomplete for '{}'!"
                print msg.format(name)
            if wild_codes:
                msg = "Warning: Order contains unknown codes for '{}'!"
                print msg.format(name)
            print '*' * 60
            if missing_codes: print 'Missing: {}'.format(missing_codes)
            if wild_codes: print 'Unknown: {}'.format(wild_codes)
            raise ValueError('Please review your data processing!')
        return None

    def _verify_column_in_meta(self, name):
        if not isinstance(name, list): name = [name]
        for n in name:
            if n not in self._meta['columns']:
                raise KeyError("'{}' not found in meta data!".format(n))
        return None
    def _get_label(self, var, text_key=None):
        if text_key is None: text_key = self._tk
        if self._get_type(var) == 'array':
            return self._meta['masks'][var]['text'][text_key]
        else:
            return self._meta['columns'][var]['text'][text_key]

    def _get_meta_loc(self, var):
        if self._get_type(var) == 'array':
            return self._meta['lib']['values']
        else:
            return self._meta['columns']

    def _get_value_loc(self, var):
        if self._is_numeric(var):
            raise KeyError("Numerical columns do not have 'values' meta.")
        loc = self._get_meta_loc(var)
        if not self._is_array(var):
            return emulate_meta(self._meta, loc[var].get('values', None))
        else:
            return emulate_meta(self._meta, loc[var])

    def _get_valuemap(self, var, text_key=None, non_mapped=None):
        if text_key is None: text_key = self._tk
        vals = self._get_value_loc(var)
        if non_mapped in ['codes', 'lists', None]:
            codes = [int(v['value']) for v in vals]
            if non_mapped == 'codes':
                return codes
        if non_mapped in ['texts', 'lists', None]:
            texts = [v['text'][text_key] for v in vals]
            if non_mapped == 'texts':
                return texts
        if non_mapped == 'lists':
            return codes, texts
        else:
            return zip(codes, texts)

    def _get_itemmap(self, var, text_key=None, non_mapped=None):
        if text_key is None: text_key = self._tk
        if non_mapped in ['items', 'lists', None]:
            items = [i['source'].split('@')[-1]
                     for i in self._meta['masks'][var]['items']]
            if non_mapped == 'items':
                return items
        if non_mapped in ['texts', 'lists', None]:
            items_texts = [self._meta['columns'][i]['text'][text_key]
                           for i in items]
            if non_mapped == 'texts':
                return items_texts
        if non_mapped == 'lists':
            return items, items_texts
        else:
            return zip(items, items_texts)

    def _get_meta(self, var, type=None,  text_key=None):
        if text_key is None: text_key = self._tk
        var_type = self._get_type(var)
        label = self._get_label(var, text_key)
        missings = self._get_missing_map(var)
        if not self._is_numeric(var):
            codes, texts = self._get_valuemap(var, non_mapped='lists')
            if missings:
                codes_copy = codes[:]
                for miss_types, miss_codes in missings.items():
                    for code in miss_codes:
                        codes_copy[codes_copy.index(code)] = miss_types
                missings = [c  if isinstance(c, (str, unicode)) else None
                            for c in codes_copy]
            else:
                missings = [None] * len(codes)
            if var_type == 'array':
                items, items_texts = self._get_itemmap(var, non_mapped='lists')
                idx_len = max((len(codes), len(items)))
                if len(codes) > len(items):
                    pad = (len(codes) - len(items))
                    items = self._pad_meta_list(items, pad)
                    items_texts = self._pad_meta_list(items_texts, pad)
                elif len(codes) < len(items):
                    pad = (len(items) - len(codes))
                    codes = self._pad_meta_list(codes, pad)
                    texts = self._pad_meta_list(texts, pad)
                    missings = self._pad_meta_list(missings, pad)
                elements = [items, items_texts, codes, texts, missings]
                columns = ['items', 'item texts', 'codes', 'texts', 'missing']
            else:
                idx_len = len(codes)
                elements = [codes, texts, missings]
                columns = ['codes', 'texts', 'missing']
            meta_s = [pd.Series(element, index=range(0, idx_len))
                      for element in elements]
            meta_df = pd.concat(meta_s, axis=1)
            meta_df.columns = columns
            meta_df.columns.name = var_type
            meta_df.index.name = '{}: {}'.format(var, label)
        else:
            meta_df = pd.DataFrame(['N/A'])
            meta_df.columns = [var_type]
            meta_df.index = ['{}: {}'.format(var, label)]
        return meta_df

    @staticmethod
    def _pad_meta_list(meta_list, pad_to_len):
        return meta_list + ([''] * pad_to_len)

    # ------------------------------------------------------------------------
    # DATA MANIPULATION/HANDLING
    # ------------------------------------------------------------------------
    def make_dummy(self, var, partitioned=False):
        if not self._is_array(var):
            if self[var].dtype == 'object': # delimited set-type data
                dummy_data = self[var].str.get_dummies(';')
                if self.meta is not None:
                    var_codes = self._get_valuemap(var, non_mapped='codes')
                    dummy_data.columns = [int(col) for col in dummy_data.columns]
                    dummy_data = dummy_data.reindex(columns=var_codes)
                    dummy_data.replace(np.NaN, 0, inplace=True)
                if not self.meta:
                    dummy_data.sort_index(axis=1, inplace=True)
            else: # single, int, float data
                dummy_data = pd.get_dummies(self[var])
                if self.meta and not self._is_numeric(var):
                    var_codes = self._get_valuemap(var, non_mapped='codes')
                    dummy_data = dummy_data.reindex(columns=var_codes)
                    dummy_data.replace(np.NaN, 0, inplace=True)
                dummy_data.rename(
                    columns={
                        col: int(col)
                        if float(col).is_integer()
                        else col
                        for col in dummy_data.columns
                    },
                    inplace=True)
            if not partitioned:
                cols = ['{}_{}'.format(var, c) for c in var_codes]
                dummy_data.columns = cols
                return dummy_data
            else:
                return dummy_data.values, dummy_data.columns.tolist()
        else: # array-type data
            items = self._get_itemmap(var, non_mapped='items')
            codes = self._get_valuemap(var, non_mapped='codes')
            dummy_data = []
            if self._is_multicode_array(items[0]):
                for i in items:
                    i_dummy = self[i].str.get_dummies(';')
                    i_dummy.columns = [int(col) for col in i_dummy.columns]
                    dummy_data.append(i_dummy.reindex(columns=codes))
            else:
                for i in items:
                    dummy_data.append(
                        pd.get_dummies(self[i]).reindex(columns=codes))
            dummy_data = pd.concat(dummy_data, axis=1)
            if not partitioned:
                cols = ['{}_{}'.format(i, c) for i in items for c in codes]
                dummy_data.columns = cols
                return dummy_data
            else:
                return dummy_data.values, codes, items

    def slicer(self, condition):
        """
        Create an index slicer to select rows from the DataFrame component.

        Parameters
        ----------
        condition : Quantipy logic expression
            A logical condition expressed as Quantipy logic that determines
            which subset of the case data rows to be kept.

        Returns
        -------
        slicer : pandas.Index
            The indices fulfilling the passed logical condition.
        """
        full_data = self._data.copy()
        series_data = full_data[full_data.columns[0]].copy()
        slicer, _ = get_logic_index(series_data, condition, full_data)
        return slicer

    def fill_conditional(self, name, selection, update):
        """
        """
        if self._is_delimited_set(name): update = '{};'.format(update)
        self._data.loc[selection, name] = update
        return None


    def code_count(self, var, ignore=None, total=None):
        data = self.make_dummy(var)
        is_array = self._is_array(var)
        if ignore:
            if ignore == 'meta': ignore = self._get_missing_map(var).keys()
            if is_array:
                ignore = [col for col in data.columns for i in ignore
                          if col.endswith(str(i))]
            slicer = [code for code in data.columns if code not in ignore]
            data = data[slicer]
        if total:
            return data.sum().sum()
        else:
            if is_array:
                items = self._get_itemmap(var, non_mapped='items')
                data = pd.concat([data[[col for col in data.columns
                                        if col.startswith(item)]].sum(axis=1)
                                  for item in items], axis=1)
                data.columns = items
            else:
                data = pd.DataFrame(data.sum(axis=0))
                data.columns = [var]
            return data

    def filter(self, alias, condition, inplace=False):
        """
        Filter the DataSet using a Quantipy logical expression.
        """
        if not inplace:
            data = self._data.copy()
        else:
            data = self._data
        filter_idx = get_logic_index(pd.Series(data.index), condition, data)
        filtered_data = data.iloc[filter_idx[0], :]
        if inplace:
            self.filtered = alias
            self._data = filtered_data
        else:
            new_ds = DataSet(self.name)
            new_ds._data = filtered_data
            new_ds._meta = self._meta
            new_ds.filtered = alias
            return new_ds

    # ------------------------------------------------------------------------
    # LINK OBJECT CONVERSION & HANDLERS
    # ------------------------------------------------------------------------
    def link(self, filters=None, x=None, y=None, views=None):
        """
        Create a Link instance from the DataSet.
        """
        #raise NotImplementedError('Links from DataSet currently not supported!')
        if filters is None: filters = 'no_filter'
        l = qp.sandbox.Link(self, filters, x, y)
        return l