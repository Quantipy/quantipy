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
    recode as _recode,
    frequency as _frequency,
    crosstab as _crosstab,
    frange)

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
        self.text_key = None
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
        """
        Load Dimensions .ddf/.mdd files, connecting as data and meta components.

        Parameters
        ----------
        path_meta : str
            The full path (optionally with extension ``'.mdd'``, otherwise
            assumed as such) to the meta data defining ``'.mdd'`` file.
        path_data : str
            The full path (optionally with extension ``'.ddf'``, otherwise
            assumed as such) to the case data defining ``'.ddf'`` file.

        Returns
        -------
        None
            The ``DataSet`` is modified inplace, connected to Quantipy data
            and meta components that have been converted from their Dimensions
            source files.
        """
        if path_meta.endswith('.mdd'): path_meta = path_meta.replace('.mdd', '')
        if path_data.endswith('.ddf'): path_data = path_data.replace('.ddf', '')
        self._meta, self._data = r_dimensions(path_meta+'.mdd', path_data+'.ddf')
        self._set_file_info(path_data, path_meta)
        return None

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
            self.text_key = self._meta['lib']['default text']
        else:
            self.text_key = None
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

    def meta(self, name=None):
        if not name:
            return self._meta
        else:
            return self.describe(name)

    def cache(self):
        return self._cache

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
        Update the ``DataSet`` with the case data entries found in ``data``.

        Parameters
        ----------
        data : ``pandas.DataFrame``
            A dataframe that contains a subset of columns from the ``DataSet``
            case data component.
        on : str, default 'identity'
            The column to use as a join key.

        Returns
        -------
        None
            DataSet is modified inplace.
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

    def merge_texts(self, dataset):
        """
        TO DO

        Parameters
        ----------

        Returns
        -------
        """
        empty_data = dataset._data.copy()
        dataset._data = dataset._data[dataset._data.index < 0]
        self.vmerge(dataset, verbose=False)
        return None

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

    def _verify_variable_meta_not_exist(self, name, is_array):
        """
        """
        msg = ''
        if not is_array:
            if name in self._meta['columns']:
                msg = "Overwriting meta for '{}', column already exists!"
        else:
            if name in self._meta['masks']:
                msg = "Overwriting meta for '{}', mask already exists!"
        if msg:
            print msg.format(name)
        else:
            return None

    @staticmethod
    def _item(item_name, text_key, text):
        """
        """
        return {'source': 'columns@{}'.format(item_name),
                'text': {text_key: text}}

    def _make_items_object(self, item_definition, text_key):
        pass


    def _add_array(self, name, qtype, label, items, categories, text_key, dims_like):
        """
        """
        if dims_like:
            array_name = self._dims_array_name(name)
        else:
            array_name = name
        item_objects = []
        if isinstance(items[0], (str, unicode)):
            items = [(no, label) for no, label in enumerate(items, start=1)]
        value_ref = 'lib@values@{}'.format(array_name)
        values = None
        for i in items:
            item_no = i[0]
            item_lab = i[1]
            item_name = self._array_item_name(i[0], name, dims_like)
            item_objects.append(self._item(item_name, text_key, item_lab))
            column_lab = '{} - {}'.format(label, item_lab)
            self.add_meta(name=item_name, qtype=qtype, label=column_lab,
                          categories=categories, items=None, text_key=text_key)
            if not values:
                values = self._meta['columns'][item_name]['values']
            self._meta['columns'][item_name]['values'] = value_ref
            self._meta['sets']['data file']['items'].remove('columns@{}'.format(item_name))
        mask_meta = {'items': item_objects, 'type': 'array',
                     'values': value_ref, 'text': {text_key: label}}
        self._meta['lib']['values'][array_name] = values
        self._meta['masks'][array_name] = mask_meta
        self._meta['sets']['data file']['items'].append('masks@{}'.format(array_name))
        self._meta['sets'][array_name] = {'items': [i['source'] for i in item_objects]}
        return None

    def unify_values(self, name, code_map):
        """
        TO DO

        Parameters
        ----------

        Returns
        -------
        None
        """
        for old_code, new_code in code_map.items():
            self.fill_conditional(name, {name: [old_code]}, new_code)
            self.remove_values(name, old_code)
        return None


    def add_meta(self, name, qtype, label, categories=None, items=None, text_key=None,
                 dimensions_like_grids=False):
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
            enumerated and mapped to the category labels. Alternatively codes
            can be mapped to categorical labels, e.g.:
            [(1, 'Elephant'), (2, 'Mouse'), (999, 'No animal')]
        items : list of str or tuples in form of (int, str), default None
            If provided will automatically create an array type mask.
            When a list of str is given, the item number will simply be
            enumerated and mapped to the category labels. Alternatively
            numerical values can be mapped explicitly to items labels, e.g.:
            [(1 'The first item'), (2, 'The second item'), (99, 'Last item')]
        text_key : str, default project.LANGUAGE
            Text key for text-based label information. Uses the ``project.py``
            information by default.

        Returns
        -------
        None
            ``DataSet`` is modified inplace, meta data and ``_data`` columns
            will be added
        """
        make_array_mask = True if items else False
        if make_array_mask and dimensions_like_grids:
            test_name = self._dims_array_name(name)
        else:
            test_name = name
        self._verify_variable_meta_not_exist(test_name, make_array_mask)
        if not text_key: text_key = self.text_key
        if make_array_mask:
            self._add_array(name, qtype, label, items, categories, text_key,
                            dimensions_like_grids)
            return None
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
        new_meta = {'text': {text_key: label}, 'type': qtype, 'name': name}
        if categories:
            if isinstance(categories[0], dict):
                new_meta['values'] = categories
            else:
                new_meta['values'] = self._make_values_list(categories, text_key)
        self._meta['columns'][name] = new_meta
        self._meta['sets']['data file']['items'].append('columns@{}'.format(name))
        self._data[name] = '' if qtype == 'delimited set' else np.NaN
        return None

    @staticmethod
    def _dims_array_name(name):
        return '{}.{}_grid'.format(name, name)

    @staticmethod
    def _array_item_name(item_no, var_name, dims_like):
        item_name = '{}_{}'.format(var_name, item_no)
        if dims_like:
            item_name = var_name + '[{' + item_name + '}].' + var_name + '_grid'
        return item_name

    def _make_items_list(self, name, text_key):
        """
        Is this equivalent to make_values_list() needed?
        """
        pass

    def _verify_same_value_codes_meta(self, name_a, name_b):
        value_codes_a = self._get_valuemap(name_a, non_mapped='codes')
        value_codes_b = self._get_valuemap(name_b, non_mapped='codes')
        if not set(value_codes_a) == set(value_codes_b):
            msg = "'{}' and '{}' do not share the same code values!"
            raise ValueError(msg.format(name_a, name_b))
        return None

    def copy_array_data(self, source, target, source_items=None,
                        target_items=None):
        """
        """
        self._verify_same_value_codes_meta(source, target)
        all_source_items = self._get_itemmap(source, non_mapped='items')
        all_target_items = self._get_itemmap(target, non_mapped='items')
        if source_items:
            source_items = [all_source_items[i-1] for i in source_items]
        else:
            source_items = all_source_items
        if target_items:
            target_items = [all_target_items[i-1] for i in target_items]
        else:
            target_items = all_target_items
        for s, t in zip(source_items, target_items):
                self[t] = self[s]
        return None

    def transpose_array(self, name, new_name=None, ignore_items=None,
                        ignore_values=None, text_key=None):
        """
        Create a new array mask with transposed items / values structure.

        This method will automatically create meta and case data additions in
        the ``DataSet`` instance.

        Parameters
        ----------
        name : str
            The originating mask variable name keyed in ``meta['masks']``.
        new_name : str, default None
            The name of the new mask. If not provided explicitly, the new_name
            will be constructed constructed by suffixing the original
            ``name`` with ``_suffix``, e.g. ``'Q2Array_trans``.
        ignore_items : int or list of int, default None
            If provided, the items listed by their order number in the
            ``_meta['masks'][name]['items']`` object will not be part of the
            transposed array. This means they will be ignored while creating
            the new value codes meta.
        ignore_codes : int or list of int, default None
            If provided, the listed code values will not be part of the
            transposed array. This means they will not be part of the new
            item meta.
        text_key : str
            The text key to be used when generating text objects, i.e.
            item and value labels.

        Returns
        -------
        None
            DataSet is modified inplace.
        """
        if not self._get_type(name) == 'array':
            raise TypeError("'{}' is not an array mask!".format(name))
        org_name = name
        # Get array item and value structure
        reg_items_object = self._get_itemmap(name)
        if ignore_items:
            if not isinstance(ignore_items, list):
                ignore_items = [ignore_items]
            reg_items_object = [i for idx, i in
                                enumerate(reg_items_object, start=1)
                                if idx not in ignore_items]
        reg_item_names = [item[0] for item in reg_items_object]
        reg_item_texts = [item[1] for item in reg_items_object]

        reg_value_object = self._get_valuemap(name)
        if ignore_values:
            if not isinstance(ignore_values, list):
                ignore_values = [ignore_values]
            reg_value_object = [v for v in reg_value_object if v[0]
                                not in ignore_values]
        reg_val_codes = [v[0] for v in reg_value_object]
        reg_val_texts = [v[1] for v in reg_value_object]

        # Transpose the array structure: values --> items, items --> values
        trans_items = [(code, value) for code, value in
                       zip(reg_val_codes, reg_val_texts)]
        trans_values = [(idx, text) for idx, text in
                        enumerate(reg_item_texts, start=1)]
        label = self._get_label(name, text_key=text_key)

        # Figure out if a Dimensions grid is the input
        if '.' in name:
            name = name.split('.')[0]
            dimensions_like = True
        else:
            dimensions_like = False
        if not new_name:
            new_name = '{}_{}'.format(name, suffix)

        # Create the new meta data entry for the transposed array structure
        qtype = 'delimited set'
        self.add_meta(new_name, qtype, label, trans_values, trans_items,
                      text_key, dimensions_like_grids=dimensions_like)
        if dimensions_like:
            new_name = '{}.{}_grid'.format(new_name, new_name)

        # Do the case data transformation by looping through items and
        # convertig value code entries...
        trans_items = self._get_itemmap(new_name, 'items')
        trans_values = self._get_valuemap(new_name, 'codes')
        for reg_item_name, new_val_code in zip(reg_item_names, trans_values):
            for reg_val_code, trans_item in zip(reg_val_codes, trans_items):
                if trans_item not in self._data.columns:
                    if qtype == 'delimited set':
                        self[trans_item] = ''
                    else:
                        self[trans_item] = np.NaN
                slicer = {reg_item_name: [reg_val_code]}
                update_with = new_val_code
                self.fill_conditional(trans_item, slicer, update_with)
        print 'Transposed array: {} into {}'.format(org_name, new_name)

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

    def fill_conditional(self, name, selection, update, append=True):
        """
        Use a quantipy logical condition to select and update case data codes.

        Parameters
        ----------
        name : str
            The originating column variable name keyed in ``_meta['columns']``.
        selection : Quantipy logic expression
            A logical condition expressed as Quantipy logic that determines
            which subset of the case data rows to be kept.
        update : int
            The code to insert into the selected column data.
        append : bool, default True
            Defines if the ``update`` code is appended when a ``delimited set``
            type column is found or existing data entries will be overwritten.

        Returns
        -------
        None
            The ``DataSet._data`` component is modified inplace.
        """
        selection  = self.slicer(selection)
        if self._is_delimited_set(name):
            update = '{};'.format(update)
        else:
            append = False
        if append:
            data = self._data.loc[selection, name]
            data.replace(np.NaN, '', inplace=True)
            self._data.loc[selection, name] = data.astype(str) + update
        else:
            self._data.loc[selection, name] = update
        return None

    def recode(self, target, mapper, default=None, append=False,
               intersect=None, initialize=None, fillna=None, inplace=True):
        """
        """
        meta = self._meta
        data = self._data
        if not target in meta['columns']:
            raise ValueError(("{} not found in meta['columns'].".format(target),
                              "Please create meta data first!"))
        recode_series = _recode(meta, data, target, mapper,
                                default, append, intersect, initialize, fillna)
        if inplace:
            self._data[target] = recode_series
            self._verify_data_vs_meta_codes(target)
            return None
        else:
            return recode_series

    def derive_categorical(self, name, qtype, label, cond_map, text_key=None):
        """
        Create meta and recode case data by specifying derived category logics.

        Parameters
        ----------
        name : str
            The column variable name keyed in ``meta['columns']``.
        qtype : [``int``, ``float``, ``single``, ``delimited set``]
            The structural type of the data the meta describes.
        label : str
            The ``text`` label information.
        cond_map : list of tuples
            Tuples of three elements with following structure:
            (code, 'Label goes here', <qp logic expression here>), e.g.:
            (1, 'Men between 30 and 40',
             intersection([{'gender': [1]}, {'age': frange('30-40')}]))
        text_key : str, default None
            Text key for text-based label information. Will automatically fall
            back to the instance's text_key property information if not provided.

        Returns
        -------
        None
            ``DataSet`` is modified inplace.
        """
        if not text_key: text_key = self.text_key
        append = qtype == 'delimited set'
        categories = [(cond[0], cond[1]) for cond in cond_map]
        idx_mapper = {cond[0]: cond[2] for cond in cond_map}
        self.add_meta(name, qtype, label, categories, items=None, text_key=text_key)
        self.recode(name, idx_mapper, append=append)
        return None

    def band_numerical(self, name, label, num_name, bands, text_key=None):
        """
        """
        if not self._meta['columns'][num_name]['type'] == 'int':
            msg = "Can only band int type data! {} is {}."
            msg = msg.format(num_name, self._meta['columns'][num_name]['type'])
            raise TypeError(msg)
        if not text_key: text_key = self.text_key
        bands = [str(band).replace(' ', '') for band in bands]
        bands = [(idx, band, {num_name: frange(band)}) for idx, band
                 in enumerate(bands, start=1)]
        self.derive_categorical(name, 'single', label, bands, text_key)
        return None

    def _make_values_list(self, categories, text_key, start_at=None):
        if not start_at:
            start_at = 1
        if not all([isinstance(cat, tuple) for cat in categories]):
            vals = [self._value(no, text_key, lab) for no, lab in
                    enumerate(categories, start_at)]
        else:
            vals = [self._value(cat[0], text_key, cat[1]) for cat in categories]
        return vals

    def weight(self, weight_scheme, unique_key='identity', report=True,
               inplace=True):
        """
        """
        meta, data = self.split()
        engine = qp.WeightEngine(data, meta)
        engine.add_scheme(weight_scheme, key=unique_key)
        engine.run()
        if report:
            print engine.get_report()
        if inplace:
            scheme_name = weight_scheme.name
            weight_name = 'weights_{}'.format(scheme_name)
            weight_description = '{} weights'.format(scheme_name)
            data_wgt = engine.dataframe(scheme_name)[[unique_key, weight_name]]
            data_wgt.rename(columns={weight_name: 'weight'}, inplace=True)
            if 'weight' not in self._meta['columns']:
                self.add_meta('weight', 'float', weight_description)
            self.update(data_wgt, on=unique_key)
        else:
            return data_wgt

    @staticmethod
    def _value(value, text_key, text):
        """
        Return a well-formed Quantipy value object from the given arguments.

        Parameters
        ----------
        value : int
            The numeric value to be given to the returned value object.
        text_key : str
            The text key to be used when generating the returned value
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
            if not isinstance(mcodes, list): mcodes = [mcodes]
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
        unflag = False
        if not missing_map: unflag = True
        if unflag:
            var = self._prep_varlist(var)
            for v in var:
                if 'missings' in self.meta()['columns'][v]:
                    del self.meta()['columns'][v]['missings']
        else:
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

    def reorder_values(self, name, new_order):
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

    def remove_values(self, name, remove):
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

    def extend_values(self, name, ext_values, text_key=None):
        """
        Add to the 'values' object of existing column or mask meta data.

        Parameters
        ----------
        meta : dict
            A Quantipy metadata document.
        name : str
            The column variable name keyed in ``_meta['columns']`` or
            ``_meta['masks']``.
        ext_values : list of str or tuples in form of (int, str), default None
            When a list of str is given, the categorical values will simply be
            enumerated and maped to the category labels. Alternatively codes can
            mapped to categorical labels, e.g.:
            [(1, 'Elephant'), (2, 'Mouse'), (999, 'No animal')]
        text_key : str, default None
            Text key for text-based label information. Will automatically fall
            back to the instance's text_key property information if not provided.

        Returns
        -------
        None
            The ``DataSet`` is modified inplace.
        """
        is_array = self._is_array(name)
        if not self._has_categorical_data(name):
            raise TypeError('{} does not contain categorical values meta!')
        if not text_key: text_key = self.text_key
        if not isinstance(ext_values, list): ext_values = [ext_values]
        value_obj = self._get_valuemap(name, text_key=text_key)
        codes = [code for code, text in value_obj]
        if isinstance(ext_values[0], tuple):
            dupes = []
            for ext_value in ext_values:
                if ext_value[0] in codes:
                    dupes.append(ext_value)
            if dupes:
                msg = 'Cannot add codes since they already exists: \n {}'
                raise ValueError(msg.format(dupes))
        else:
            start_here = self._next_consecutive_code(codes)
            ext_values = self._make_values_list(ext_values, text_key, start_here)
        if is_array:
            self._meta['lib']['values'][name].extend(ext_values)
        else:
            self._meta['columns'][name]['values'].extend(ext_values)
        return None


    def set_text_key(self, text_key):
        """
        """
        self.text_key = text_key
        self._meta['lib']['default text'] = text_key
        return None

    def set_value_texts(self, name, renamed_vals, text_key=None):
        """
        Rename or add value texts in the 'values' object.

        This method works for array masks and column meta data.

        Parameters
        ----------
        name : str
            The column variable name keyed in ``_meta['columns']`` or
            ``_meta['masks']``.
        renamed_vals : dict with codes and new value texts
            {1: 'new label for code=1', 5: 'new label for code=5'}
            key/code will be ignored if it doesn't exist in the 'values' object
        text_key : str, default None
            Text key for text-based label information. Will automatically fall
            back to the instance's _tk property information if not provided.

        Returns
        -------
        None
            The ``DataSet`` is modified inplace.
        """
        if not self._has_categorical_data(name):
            raise TypeError('{} does not contain categorical values meta!')
        if not text_key: text_key = self.text_key

        if not self._is_array(name):
            obj_values = self._meta['columns'][name]['values']
            new_obj_values = []
        else:
            obj_values = self._meta['lib']['values'][name]
            new_obj_values = []

        for item in obj_values:
            val = item['value']
            if val in renamed_vals.keys():
                value_texts = item['text']
                if text_key in value_texts.keys():
                    item['text'][text_key] = renamed_vals[val]
                else:
                    item['text'].update({text_key: renamed_vals[val]})
            new_obj_values.append(item)

        if not self._is_array(name):
            self._meta['columns'][name]['values'] = new_obj_values
        else:
            self._meta['lib']['values'][name] = new_obj_values
        return None

    def set_column_text(self, name, new_text, text_key=None):
        """
        TO DO

        Parameters
        ----------

        Returns
        -------
        """
        self._verify_column_in_meta(name)
        if not text_key: text_key = self.text_key
        if text_key in self._meta['columns'][name]['text'].keys():
            self._meta['columns'][name]['text'][text_key] = new_text
        else:
            self._meta['columns'][name]['text'].update({text_key: new_text})
        return None

    @classmethod
    def _consecutive_codes(cls, codes):
        return sorted(codes) == range(min(codes), max(codes)+1)

    @classmethod
    def _highest_code(cls, codes):
        return max(codes)

    @classmethod
    def _lowest_code(cls, codes):
        return min(codes)

    @classmethod
    def _next_consecutive_code(cls, codes):
        if cls._consecutive_codes(codes):
            return len(codes) + 1
        else:
            return cls._highest_code(codes) + 1

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
        if text_key is None: text_key = self.text_key
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
                    flags_code = set(flags_code)
                    mis_map = {'exclude': list(flags_code)}
                    self.set_missings(cat, mis_map)
        return None

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
        has_missings = False
        if self._get_type(var) == 'array':
            var = self._get_itemmap(var, non_mapped='items')[0]
        if 'missings' in self.meta()['columns'][var].keys():
            if len(self.meta()['columns'][var]['missings'].keys()) > 0:
                has_missings = True
        return has_missings

    def _is_numeric(self, var):
        return self._get_type(var) in ['float', 'int']

    def _is_array(self, var):
        return self._get_type(var) == 'array'

    def _is_multicode_array(self, mask_element):
        return self[mask_element].dtype == 'object'

    def _is_delimited_set(self, name):
        return self._meta['columns'][name]['type'] == 'delimited set'

    def _has_categorical_data(self, name):
        if self._is_array(name):
            name = self._get_itemmap(name, non_mapped='items')[0]
        if self._meta['columns'][name]['type'] in ['single', 'delimited set']:
            return True
        else:
            return False

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
        if text_key is None: text_key = self.text_key
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

    def _get_valuemap(self, var, non_mapped=None,  text_key=None):
        if text_key is None: text_key = self.text_key
        vals = self._get_value_loc(var)
        if non_mapped in ['codes', 'lists', None]:
            codes = [int(v['value']) for v in vals]
            if non_mapped == 'codes':
                return codes
        if non_mapped in ['texts', 'lists', None]:
            texts = [v['text'][text_key] if text_key in v['text'] else None
                     for v in vals]
            if non_mapped == 'texts':
                return texts
        if non_mapped == 'lists':
            return codes, texts
        else:
            return zip(codes, texts)

    def _get_itemmap(self, var, non_mapped=None, text_key=None):
        if text_key is None: text_key = self.text_key
        if non_mapped in ['items', 'lists', None]:
            items = [i['source'].split('@')[-1]
                     for i in self._meta['masks'][var]['items']]
            if non_mapped == 'items':
                return items
        if non_mapped in ['texts', 'lists', None]:
            items_texts = [i['text'][text_key] for i in
                           self._meta['masks'][var]['items']]
            if non_mapped == 'texts':
                return items_texts
        if non_mapped == 'lists':
            return items, items_texts
        else:
            return zip(items, items_texts)

    def _get_meta(self, var, type=None,  text_key=None):
        if text_key is None: text_key = self.text_key
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