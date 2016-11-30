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

from quantipy.core.helpers.functions import (
    filtered_set,
    emulate_meta)

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
    frequency as fre,
    crosstab as ct,
    frange)

from cache import Cache

import copy as org_copy
import json
import warnings

from itertools import product

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
        self._verbose_errors = True
        self._verbose_infos = True
        self._cache = Cache()
        self.columns = None
        self.masks = None
        self.sets = None
        self.singles = None
        self.delimited_sets = None
        self.ints = None
        self.floats = None
        self.dates = None
        self.strings = None

    # ------------------------------------------------------------------------
    # item access / instance handlers
    # ------------------------------------------------------------------------
    def __getitem__(self, var):
        if isinstance(var, tuple):
            sliced_access = True
            slicer = var[0]
            var = var[1]
        else:
            sliced_access = False
        var = self._prep_varlist(var)
        if len(var) == 1: var = var[0]
        if sliced_access:
            return self._data.ix[slicer, var]
        else:
            return self._data[var]

    def __setitem__(self, name, val):
        if isinstance(name, tuple):
            sliced_insert = True
            slicer = name[0]
            name = name[1]
        else:
            sliced_insert = False
        scalar_insert = isinstance(val, (int, float, str, unicode))
        if scalar_insert and self._has_categorical_data(name):
            if not val in self.codes(name) and not np.isnan(val):
                msg = "{} is undefined for '{}'! Valid: {}"
                raise ValueError(msg.format(val, name, self.codes(name)))
        if sliced_insert:
            self._data.loc[slicer, name] = val
        else:
            self._data[name] = val

    @staticmethod
    def start_meta(text_key='main'):
        """
        Starts a new/empty Quantipy meta document.

        Parameters
        ----------
        text_key : str, default None
            The default text key to be set into the new meta document.

        Returns
        -------
        meta : dict
            Quantipy meta object
        """
        meta = {
            'info': {
                'text': ''
            },
            'lib': {
                'default text': text_key,
                'values': {}
            },
            'columns': {},
            'masks': {},
            'sets': {
                'data file': {
                    'text': {text_key: 'Variable order in source file'},
                    'items': []
                }
            },
            'type': 'pandas.DataFrame'
        }
        return meta

    def _get_columns(self, vtype=None):
        meta = self._meta['columns']
        if vtype:
            return [c for c in meta.keys() if self._get_type(c) == vtype]
        else:
            return meta.keys()

    def _get_masks(self):
        return self._meta['masks'].keys()

    def _get_sets(self):
        return self._meta['sets'].keys()

    def set_verbose_errmsg(self, verbose=True):
        """
        """
        if not isinstance(verbose, bool):
            msg = 'Can only assign boolean values, found {}'
            raise ValueError(msg.format(type(verbose)))
        self._verbose_errors = verbose
        return None

    def set_verbose_infomsg(self, verbose=True):
        """
        """
        if not isinstance(verbose, bool):
            msg = 'Can only assign boolean values, found {}'
            raise ValueError(msg.format(type(verbose)))
        self._verbose_infos = verbose
        return None

    @classmethod
    def set_encoding(cls, encoding):
        """
        Hack sys.setdefaultencoding() to escape ASCII hell.

        Parameters
        ----------
        encoding : str
            The name of the encoding to default to.
        """
        import sys
        default_stdout = sys.stdout
        default_stderr = sys.stderr
        reload(sys)
        sys.setdefaultencoding(encoding)
        sys.stdout = default_stdout
        sys.stderr = default_stderr

    def clone(self):
        """
        Get a deep copy of the ``DataSet`` instance.
        """
        cloned = org_copy.deepcopy(self)
        return cloned

    def split(self, save=False):
        """
        Return the ``meta`` and ``data`` components of the DataSet instance.

        Parameters
        ----------
        save : bool, default False
            If True, the ``meta`` and ``data`` objects will be saved to disk,
            using the instance's ``name`` and ``path`` attributes to determine
            the file location.

        Returns
        -------
        meta, data : dict, pandas.DataFrame
            The meta dict and the case data DataFrame as separate objects.
        """
        meta, data = self._meta, self._data
        if save:
            path = self.path
            name = self.name
            w_quantipy(meta, data, path+name+'.json', path+name+'.csv')
        return meta, data

    def meta(self, name=None, text_key=None):
        """
        Provide a *pretty* summary for variable meta given as per ``name``.

        Parameters
        ----------
        name : str, default None
            The column variable name keyed in ``_meta['columns']`` or
            ``_meta['masks']``. If None, the entire ``meta`` component of the
            ``DataSet`` instance will be returned.
        text_key : str, default None
            The text_key that should be used when taking labels from the
            source meta. If the given text_key is not found for any
            particular text object, the ``DataSet.text_key`` will be used
            instead.

        Returns
        ------
        meta : dict or pandas.DataFrame
            Either a DataFrame that sums up the meta information on a ``mask``
            or ``column`` or the meta dict as a whole is
        """
        if not name:
            return self._meta
        else:
            return self.describe(name, text_key=text_key)

    def variables(self, only_type=None):
        """
        Get an overview of all the variables ordered by their type.

        Parameters
        ----------
        only_type : str or list of str, default None
            Restrict the overview to these data types.

        Returns
        -------
        overview : pandas.DataFrame
            The variables per data type inside the ``DataSet``.
        """
        return self.describe(only_type=only_type)

    def values(self, name, text_key=None):
        """
        Get categorical data's paired code and texts information from the meta.

        Parameters
        ----------
        name : str
            The column variable name keyed in ``_meta['columns']`` or
            ``_meta['masks']``.
        text_key : str, default None
            The text_key that should be used when taking labels from the
            source meta. If the given text_key is not found for any
            particular text object, the ``DataSet.text_key`` will be used
            instead.

        Returns
        -------
        values : list of tuples
            The list of the numerical category codes and their ``texts``
            packed as tuples.
        """
        if not self._has_categorical_data(name):
            err_msg = '{} does not contain categorical values meta!'
            raise TypeError(err_msg.format(name))
        if not text_key: text_key = self.text_key
        return self._get_valuemap(name, text_key=text_key)

    def codes(self, name):
        """
        Get categorical data's numerical code values.

        Parameters
        ----------
        name : str
            The column variable name keyed in ``_meta['columns']``.

        Returns
        -------
        codes : list
            The list of category codes.
        """
        return self._get_valuemap(name, non_mapped='codes')

    def value_texts(self, name, text_key=None):
        """
        Get categorical data's text information.

        Parameters
        ----------
        name : str
            The column variable name keyed in ``_meta['columns']``.

        Returns
        -------
        texts : list
            The list of category texts.
        """
        return self._get_valuemap(name, non_mapped='texts', text_key=text_key)

    def items(self, name, text_key=None):
        """
        Get the array's paired item names and texts information from the meta.

        Parameters
        ----------
        name : str
            The column variable name keyed in ``_meta['masks']``.
        text_key : str, default None
            The text_key that should be used when taking labels from the
            source meta. If the given text_key is not found for any
            particular text object, the ``DataSet.text_key`` will be used
            instead.

        Returns
        -------
        items : list of tuples
            The list of source item names (from ``_meta['columns']``) and their
            ``text`` information packed as tuples.
        """
        if not self._is_array(name):
            err_msg = '{} is not an array mask!'
            raise TypeError(err_msg.format(name))
        if not text_key: text_key = self.text_key
        return self._get_itemmap(name, text_key=text_key)

    def sources(self, name):
        """
        Get the ``_meta['columns']`` elements for the passed array mask name.

        Parameters
        ----------
        name : str
            The mask variable name keyed in ``_meta['masks']``.

        Returns
        -------
        sources : list
            The list of source elements from the array definition.
        """
        return self._get_itemmap(name, non_mapped='items')

    def item_texts(self, name, text_key=None):
        """
        Get the ``text`` meta data for the items of the passed array mask name.

        Parameters
        ----------
        name : str
            The mask variable name keyed in ``_meta['masks']``.
        text_key : str, default None
            The text_key that should be used when taking labels from the
            source meta. If the given text_key is not found for any
            particular text object, the ``DataSet.text_key`` will be used
            instead.

        Returns
        -------
        texts : list
            The list of item texts for the array elements.
        """
        return self._get_itemmap(name, non_mapped='texts', text_key=text_key)


    def data(self):
        """
        Return the ``data`` component of the ``DataSet`` instance.
        """
        return self._data

    def _cache(self):
        return self._cache

    # ------------------------------------------------------------------------
    # file i/o / conversions
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
        """
        Load SPSS Statistics .sav files, converting and connecting data/meta.

        Parameters
        ----------
        path_sav : str
            The full path (optionally with extension ``'.sav'``, otherwise
            assumed as such) to the ``'.sav'`` file.

        Returns
        -------
        None
            The ``DataSet`` is modified inplace, connected to Quantipy data
            and meta components that have been converted from the SPSS
            source file.
        """
        if path_sav.endswith('.sav'): path_sav = path_sav.replace('.sav', '')
        self._meta, self._data = r_spss(path_sav+'.sav', **kwargs)
        self._set_file_info(path_sav)
        return None

    def write_quantipy(self, path_meta=None, path_data=None):
        """
        Write the data and meta components to .csv/.json files.

        The resulting files are well-defined native Quantipy source files.

        Parameters
        ----------
        path_meta : str, default None
            The full path (optionally with extension ``'.json'``, otherwise
            assumed as such) for the saved the DataSet._meta component.
            If not provided, the instance's ``name`` and ```path`` attributes
            will be used to determine the file location.
        path_data : str, default None
            The full path (optionally with extension ``'.ddf'``, otherwise
            assumed as such) for the saved DataSet._data component.
            If not provided, the instance's ``name`` and ```path`` attributes
            will be used to determine the file location.

        Returns
        -------
        None
        """
        meta, data = self._meta, self._data
        if path_data is None and path_meta is None:
            path = self.path
            name = self.name
            path_meta = '{}/{}.json'.format(path, name)
            path_data = '{}/{}.csv'.format(path, name)
        elif path_data is not None and path_meta is not None:
            if not path_meta.endswith('.json'):
                path_meta = '{}.json'.format(path_meta)
            if not path_data.endswith('.csv'):
                path_data = '{}.csv'.format(path_data)
        else:
            msg = 'Must either specify or omit both `path_meta` and `path_data`!'
            raise ValueError(msg)
        w_quantipy(meta, data, path_meta, path_data)
        return None

    def write_spss(self, path_sav=None, index=True, text_key=None,
                   mrset_tag_style='__', drop_delimited=True, from_set=None,
                   verbose=True):
        """
        Parameters
        ----------
        path_sav : str, default None
            The full path (optionally with extension ``'.json'``, otherwise
            assumed as such) for the saved the DataSet._meta component.
            If not provided, the instance's ``name`` and ```path`` attributes
            will be used to determine the file location.
        index : bool, default False
            Should the index be inserted into the dataframe before the
            conversion happens?
        text_key : str, default None
            The text_key that should be used when taking labels from the
            source meta. If the given text_key is not found for any
            particular text object, the ``DataSet.text_key`` will be used
            instead.
        mrset_tag_style : str, default '__'
            The delimiting character/string to use when naming dichotomous
            set variables. The mrset_tag_style will appear between the
            name of the variable and the dichotomous variable's value name,
            as taken from the delimited set value that dichotomous
            variable represents.
        drop_delimited : bool, default True
            Should Quantipy's delimited set variables be dropped from
            the export after being converted to dichotomous sets/mrsets?
        from_set : str
            The set name from which the export should be drawn.
        Returns
        -------
        None
        """
        self.set_encoding('cp1252')
        meta, data = self._meta, self._data
        if not text_key: text_key = self.text_key
        if not path_sav:
            path_sav = '{}/{}.sav'.format(self.path, self.name)
        else:
            if not path_sav.endswith('.sav'):
                path_sav = '{}.sav'.format(path_sav)
        w_spss(path_sav, meta, data, index=index, text_key=text_key,
               mrset_tag_style=mrset_tag_style, drop_delimited=drop_delimited,
               from_set=from_set, verbose=verbose)
        self.set_encoding('utf-8')
        return None

    def from_components(self, data_df, meta_dict=None, text_key=None):
        """
        Attach a data and meta directly to the ``DataSet`` instance.

        .. note:: Except testing for appropriate object types, this method
            offers no additional safeguards or consistency/compability checks
            with regard to the passed data and meta documents!

        Parameters
        ----------
        data_df : pandas.DataFrame
            A DataFrame that contains case data entries for the ``DataSet``.
        meta_dict: dict, default None
            A dict that stores meta data describing the columns of the data_df.
            It is assumed to be well-formed following the Quantipy meta data
            structure.
        text_key : str, default None
            The text_key to be used. If not provided, it will be attempted to
            use the 'default text' from the ``meta['lib']`` definition.

        Returns
        -------
        None
        """
        if not isinstance(data_df, pd.DataFrame):
            msg = 'data_df must be a pandas.DataFrame, passed {}.'
            raise TypeError(msg.format(type(data_df)))
        if not isinstance(meta_dict, dict):
            msg = 'meta_dict must be of type dict, passed {}.'
            raise TypeError(msg.format(type(meta_dict)))
        self._data = data_df
        if meta_dict:
            self._meta = meta_dict
        if not text_key:
            try:
                self.text_key = self._meta['lib']['default text']
            except KeyError:
                warning = "No 'text_key' provided and unable to derive"
                warning = warning + " 'text_key' information from passed meta!"
                warning = warning + " 'DataSet._meta might be corrupt!"
                warnings.warn(warning)
                self.text_key = None
        return None

    def _set_file_info(self, path_data, path_meta=None):
        self.path = '/'.join(path_data.split('/')[:-1]) + '/'
        try:
            self.text_key = self._meta['lib']['default text']
        except:
            self.text_key = None
        self._data['@1'] = np.ones(len(self._data))
        self._meta['columns']['@1'] = {'type': 'int'}
        self._data.index = list(xrange(0, len(self._data.index)))
        self.columns = self._get_columns()
        self.masks = self._get_masks()
        self.sets = self._get_sets()
        self.singles = self._get_columns('single')
        self.delimited_sets = self._get_columns('delimited set')
        self.ints = self._get_columns('int')
        self.floats = self._get_columns('float')
        self.dates = self._get_columns('date')
        self.strings = self._get_columns('string')
        if self._verbose_infos: self._show_file_info()
        return None

    def _show_file_info(self):
        file_spec = 'DataSet: {}\nrows: {} - columns: {}'
        if not self.path: self.path = '/'
        file_name = '{}{}'.format(self.path, self.name)
        print file_spec.format(file_name, len(self._data.index),
                               len(self._data.columns)-1)
        return None

    def list_variables(self, numeric=False, text=False, blacklist=None):
        """
        Get list with all variable names except date,boolean,(string,numeric)

        Parameters
        ----------
        numeric : bool, default False
            If True, int/float variables are included in list.
        text : bool, default False
            If True, string variables are included in list.
        blacklist: list of str,
            Variables that should be excluded

        Returns
        -------
        list of str
        """
        meta = self._meta
        items_list = meta['sets']['data file']['items']

        except_list = ['date','boolean']
        if not text: except_list.append('string')
        if not numeric: except_list.extend(['int','float'])

        var_list =[]
        if not isinstance(blacklist, list):
            blacklist = [blacklist]
        if not blacklist: blacklist=[]
        for item in items_list:
            key, var_name = item.split('@')
            if key == 'masks':
                for element in meta[key][var_name]['items']:
                    blacklist.append(element['source'].split('@')[-1])
            if var_name in blacklist: continue
            if meta[key][var_name]['type'] in except_list: continue
            var_list.append(var_name)
        return var_list


    def create_set(self, setname='new_set', based_on='data file', included=None,
                   excluded=None, strings='keep', arrays='both', replace=None,
                   overwrite=False):
        """
        Create a new set in ``dataset._meta['sets']``.

        Parameters
        ----------
        setname : str
            Name of the new set.
        based_on : str
            Name of set that can be reduced or expanded.
        included : str or list/set/tuple of str
            Names of the variables to be included in the new set. If None all
            variables in ``based_on`` are taken.
        excluded : str or list/set/tuple of str
            Names of the variables to be excluded in the new set.
        strings : {'keep', 'drop', 'only'}
            Keep, drop or only include string variables.
        arrays : {'both', 'masks', 'columns'}
            Add for arrays ``masks@varname`` or ``columns@varname`` or both.
        replace : dict
            Replace a variable in the set with an other.
            Example: {'q1': 'q1_rec'}, 'q1' and 'q1_rec' must be included in
                     ``based_on``. 'q1' will be removed and 'q1_rec' will be
                     moved to this position.
        overwrite: boolean
            Overwrite if ``meta['sets'][name] already exist.
        Returns
        -------
        None
            The ``DataSet`` is modified inplace.
        """
        meta = self._meta
        # proove setname
        if not isinstance(setname, str):
            raise TypeError("'setname' must be a str.")
        if setname in meta['sets'] and not overwrite:
            raise KeyError("{} is already in `meta['sets'].`".format(setname))
        # proove based_on
        if not based_on in meta['sets']:
            raise KeyError("'based_on' is not in `meta['sets'].`")

        # proove included
        if not included:
            included = [var.split('@')[-1]
                        for var in meta['sets'][based_on]['items']]
        elif not isinstance(included, list): included = [included]

        # proove replace
        if not replace: replace = {}
        elif not isinstance(replace, dict):
            raise TypeError("'replace' must be a dict.")
        else:
            for var in replace.keys() + replace.values():
                if var not in included:
                    raise KeyError("{} is not in 'included'".format(var))

        # proove arrays
        if not arrays in ['both', 'masks', 'columns']:
            raise ValueError (
                "'arrays' must be either 'both', 'masks' or 'columns'.")

        # filter set and create new set
        fset = filtered_set(meta=meta,
                     based_on=based_on,
                     masks=meta['masks'] if arrays=='columns' else None,
                     included=included,
                     excluded=excluded,
                     strings=strings)

        if arrays=='both':
            new_items = []
            items = fset['items']
            for item in items:
                new_items.append(item)
                if item.split('@')[0]=='masks':
                    for i in meta['masks'][item.split('@')[-1]]['items']:
                        new_items.append(i['source'])
            fset['items'] = new_items

        if replace:
            new_items = fset['items']
            for k, v in replace.items():
                for x, item in enumerate(new_items):
                    if v == item.split('@')[-1]: posv, move = x, item
                    if k == item.split('@')[-1]: posk = x
                new_items[posk] = move
                new_items.pop(posv)
            fset['items'] = new_items

        add = {setname: fset}
        meta['sets'].update(add)

        return None

    # ------------------------------------------------------------------------
    # extending / merging
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
            new_dataset = self.clone()
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
               overwrite_text=False, from_set=None, uniquify_key=None,
               reset_index=True, inplace=True, verbose=True):
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
        uniquify_key : str, default None
            A int-like column name found in all the passed ``DataSet`` objects
            that will be protected from having duplicates. The original version
            of the column will be kept under its name prefixed with 'original_'.
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
            if uniquify_key:
                self._make_unique_key(uniquify_key, row_id_name)
            return None
        else:
            new_dataset = self.clone()
            new_dataset._data = merged_data
            new_dataset._meta = merged_meta
            if uniquify_key:
                new_dataset._make_unique_key(uniquify_key, row_id_name)
            return new_dataset

    def check_dupe(self, name='identity'):
        return self.duplicates(name=name)

    def duplicates(self, name='identity'):
        """
        Returns a list with duplicated values for the provided name.

        Parameters
        ----------
        name : str, default 'identity'
            The column variable name keyed in ``meta['columns']``.

        Returns
        -------
        vals : list
            A list of duplicated values found in the named variable.
        """
        qtype = self._get_type(name)
        if qtype in ['array', 'delimited set', 'float']:
            raise TypeError("Can not check duplicates for type '{}'.".format(qtype))
        vals = self._data[name].value_counts()
        vals = vals.copy().dropna()
        if qtype == 'string':
            vals = vals.drop('__NA__')
        vals = vals[vals >= 2].index.tolist()
        if not qtype == 'string':
            vals = [int(i) for i in vals]
        return vals


    def _make_unique_key(self, id_key_name, multiplier):
        """
        """
        columns = self._meta['columns']
        if not id_key_name in columns:
            raise KeyError("'id_key_name' is not in 'meta['columns']'!")
        elif columns[id_key_name]['type'] not in ['int', 'float']:
            raise TypeError("'id_key_name' must be of type int, float, single!")
        elif not multiplier in columns:
            raise KeyError("'multiplier' is not in 'meta['columns']'!")
        elif columns[multiplier]['type'] not in ['single', 'int', 'float']:
            raise TypeError("'multiplier' must be of type int, float, single!")
        org_key_col = self._data.copy()[id_key_name]
        new_name = 'original_{}'.format(id_key_name)
        name, qtype, lab = new_name, 'int', 'Original ID'
        self.add_meta(name, qtype, lab)
        self[new_name] = org_key_col
        self[id_key_name] += self[multiplier].astype(int) * 100000000
        return None

    def merge_texts(self, dataset):
        """
        Add additional ``text`` versions from other ``text_key`` meta.

        Case data will be ignored during the merging process.

        Parameters
        ----------
        dataset : (A list of multiple) ``quantipy.DataSet``
            One or multiple datasets that provide new ``text_key`` meta.

        Returns
        -------
        None
        """
        if not isinstance(dataset, list):
            dataset = [dataset]
        for ds in dataset:
            empty_data = ds._data.copy()
            ds._data = ds._data[ds._data.index < 0]
        self.vmerge(dataset, verbose=False)
        return None

    # ------------------------------------------------------------------------
    # meta data editing
    # ------------------------------------------------------------------------
    def add_meta(self, name, qtype, label, categories=None, items=None, text_key=None,
                 dimensions_like_grids=False):
        """
        Create and insert a well-formed meta object into the existing meta document.

        Parameters
        ----------
        name : str
            The column variable name keyed in ``meta['columns']``.
        qtype : {'int', 'float', 'single', 'delimited set', 'date', 'string'}
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
        text_key : str, default None
            Text key for text-based label information. Uses the
            ``DataSet.text_key`` information if not provided.

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
        if not qtype in ['delimited set', 'single', 'float', 'int',
                         'date', 'string']:
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
        datafile_setname = 'columns@{}'.format(name)
        if datafile_setname not in self._meta['sets']['data file']['items']:
            self._meta['sets']['data file']['items'].append(datafile_setname)
        self._data[name] = '' if qtype == 'delimited set' else np.NaN
        return None

    def categorize(self, name, categorized_name=None):
        """
        Categorize an ``int``/``string``/``text`` variable to ``single``.

        The ``values`` object of the categorized variable is populated with the
        unique values found in the originating variable (ignoring np.NaN /
        empty row entries).

        Parameters
        ----------
        name : str
            The column variable name keyed in ``meta['columns']`` that will
            be categorized.
        categorized_name : str
            If provided, the categorized variable's new name will be drawn
            from here, otherwise a default name in form of ``'name#'`` will be
            used.

        Returns
        -------
        None
            DataSet is modified inplace, adding the categorized variable to it.
        """
        org_type = self._get_type(name)
        valid_types = ['int', 'string', 'date']
        if org_type not in valid_types:
            raise TypeError('Can only categorize {}!'.format(valid_types))
        new_var_name = categorized_name or '{}#'.format(name)
        self.copy(name)
        self.convert('{}_rec'.format(name), 'single')
        self.rename('{}_rec'.format(name), new_var_name)
        return None

    def convert(self, name, to):
        """
        Convert meta and case data between compatible variable types.

        Wrapper around the separate ``as_TYPE()`` conversion methods.

        Parameters
        ----------
        name : str
            The column variable name keyed in ``meta['columns']`` that will
            be converted.
        to : {'int', 'float', 'single', 'delimited set', 'string'}
            The variable type to convert to.

        Returns
        -------
        None
            The DataSet variable is modified inplace.
        """
        valid_types = ['int', 'float', 'single', 'delimited set', 'string']
        if not to in valid_types:
            raise TypeError("Cannot convert to type {}!".format(to))
        if to == 'int':
            self.as_int(name, False)
        elif to == 'float':
            self.as_float(name, False)
        elif to == 'single':
            self.as_single(name, False)
        elif to == 'delimited set':
            self.as_delimited_set(name, False)
        elif to == 'string':
            self.as_string(name, False)
        return None

    def as_float(self, name, show_warning=True):
        """
        Change type from ``single`` or ``int`` to ``float``.

        Parameters
        ----------
        name : str
            The column variable name keyed in ``meta['columns']``.

        Returns
        -------
        None
        """
        warning = "'as_float()' will be removed alongside other individual"
        warning = warning + " conversion methods soon! Use 'convert()' instead!"
        if show_warning: warnings.warn(warning)
        org_type = self._get_type(name)
        if org_type == 'float': return None
        valid = ['single', 'int']
        if not org_type in valid:
            msg = 'Cannot convert variable {} of type {} to float!'
            raise TypeError(msg.format(name, org_type))
        if org_type == 'single':
            self.as_int(name)
        if org_type == 'int':
            self._meta['columns'][name]['type'] = 'float'
            self._data[name] = self._data[name].apply(
                    lambda x: float(x) if not np.isnan(x) else np.NaN)
        return None

    def as_int(self, name, show_warning=True):
        """
        Change type from ``single`` to ``int``.

        Parameters
        ----------
        name : str
            The column variable name keyed in ``meta['columns']``.

        Returns
        -------
        None
        """
        warning = "'as_int()' will be removed alongside other individual"
        warning = warning + " conversion methods soon! Use 'convert()' instead!"
        if show_warning: warnings.warn(warning)
        org_type = self._get_type(name)
        if org_type == 'int': return None
        valid = ['single']
        if not org_type in valid:
            msg = 'Cannot convert variable {} of type {} to int!'
            raise TypeError(msg.format(name, org_type))
        self._meta['columns'][name]['type'] = 'int'
        self._meta['columns'][name].pop('values')
        return None

    def as_delimited_set(self, name, show_warning=True):
        """
        Change type from ``single`` to ``delimited set``.

        Parameters
        ----------
        name : str
            The column variable name keyed in ``meta['columns']``.

        Returns
        -------
        None
        """
        warning = "'as_delimited_set()' will be removed alongside other individual"
        warning = warning + " conversion methods soon! Use 'convert()' instead!"
        if show_warning: warnings.warn(warning)
        org_type = self._get_type(name)
        if org_type == 'delimited set': return None
        valid = ['single']
        if not org_type in valid:
            msg = 'Cannot convert variable {} of type {} to delimited set!'
            raise TypeError(msg.format(name, org_type))
        self._meta['columns'][name]['type'] = 'delimited set'
        self._data[name] = self._data[name].apply(
            lambda x: str(int(x)) + ';' if not np.isnan(x) else np.NaN)
        return None

    def as_single(self, name, show_warning=True):
        """
        Change type from ``int``/``date``/``string`` to ``single``.

        Parameters
        ----------
        name : str
            The column variable name keyed in ``meta['columns']``.

        Returns
        -------
        None
        """
        warning = "'as_single()' will be removed alongside other individual"
        warning = warning + " conversion methods soon! Use 'convert()' instead!"
        if show_warning: warnings.warn(warning)
        org_type = self._get_type(name)
        if org_type == 'single': return None
        valid = ['int', 'date', 'string']
        if not org_type in valid:
            msg = 'Cannot convert variable {} of type {} to single!'
            raise TypeError(msg.format(name, org_type))
        text_key = self.text_key
        if org_type == 'int':
            num_vals = sorted(self._data[name].dropna().astype(int).unique())
            values_obj = [self._value(num_val, text_key, str(num_val))
                          for num_val in num_vals]
        elif org_type == 'date':
            vals = self._data[name].order().astype(str).unique()
            values_obj = [self._value(i, text_key, v) for i,  v
                          in enumerate(vals, start=1)]
            self._data[name] = self._data[name].astype(str)
            replace_map = {v: i for i, v in enumerate(vals, start=1)}
            self._data[name].replace(replace_map, inplace=True)
        elif org_type == 'string':
            self[name] = self[name].replace('__NA__', np.NaN)
            vals = sorted(self[name].dropna().unique().tolist())
            values_obj = [self._value(i, text_key, v) for i, v
                          in enumerate(vals, start=1)]
            replace_map = {v: i for i, v in enumerate(vals, start=1)}
            if replace_map:
                self._data[name].replace(replace_map, inplace=True)
        self._meta['columns'][name]['type'] = 'single'
        self._meta['columns'][name]['values'] = values_obj
        return None

    def as_string(self, name, show_warning=True):
        """
        Change type from ``int``/``float``/``date``/``single`` to ``string``.

        Parameters
        ----------
        name : str
            The column variable name keyed in ``meta['columns']``.

        Returns
        -------
        None
        """
        warning = "'as_string()' will be removed alongside other individual"
        warning = warning + " conversion methods soon! Use 'convert()' instead!"
        if show_warning: warnings.warn(warning)
        org_type = self._get_type(name)
        if org_type == 'string': return None
        valid = ['single', 'int', 'float', 'date']
        if not org_type in valid:
            msg = 'Cannot convert variable {} of type {} to text!'
            raise TypeError(msg.format(name, org_type))
        self._meta['columns'][name]['type'] = 'string'
        if self._get_type == 'single':
            self._meta['columns'][name].pop('values')
        self._data[name] = self._data[name].astype(str)
        return None

    def rename(self, name, new_name=None, array_items=None):
        """
        Change meta and data column name references of the variable defintion.

        Parameters
        ----------
        name : str
            The originating column variable name keyed in ``meta['columns']``
            or ``meta['masks']``.
        new_name : str
            The new variable name.
        array_items: dict
            Item position and new name for item. Example: {4: 'q5_4'} then
            q5[{q5_4}].q5_grid is renamed to q5_4.


        Returns
        -------
        None
            DataSet is modified inplace. The new name reference is placed into
            both the data and meta component.
        """
        data = self._data

        lib = self._meta['lib']
        sets = self._meta['sets']
        masks = self._meta['masks']
        columns = self._meta['columns']

        if self._is_array(name):
            if new_name:
                # update meta['sets']
                sets[new_name] = sets[name].copy()
                del sets[name]
                o_set_entry = 'masks@{}'.format(name)
                n_set_entry = 'masks@{}'.format(new_name)
                n_datafile_items = [i if i != o_set_entry else n_set_entry
                                    for i in sets['data file']['items']]
                sets['data file']['items'] = n_datafile_items

                # update meta['lib']
                val = lib['values']
                val[new_name] = val[name]
                del val[name]
                val['ddf'][new_name] = val['ddf'][name].copy()
                del val['ddf'][name]

                # update meta['masks']
                masks[new_name] = masks[name].copy()
                del masks[name]
                values = 'lib@values@{}'.format(new_name)
                masks[new_name]['values'] = values
                items = [i.split('@')[-1] for i in sets[new_name]['items']]
                for item in items:
                    columns[item]['values'] = values

            if array_items:
                # update meta['sets']
                if not new_name: new_name = name
                variables = {}

                new_items = []
                for x, item in enumerate(sets[new_name]['items'], 1):
                    if x in array_items:
                        new_items.append('columns@{}'.format(array_items[x]))
                        variables[item] = array_items[x]
                    else:
                        new_items.append(item)
                sets[new_name]['items'] = new_items

                # update meta['masks']
                new_items = []
                for item in masks[new_name]['items']:
                    if item['source'] in variables:
                        item['source'] = 'columns@{}'.format(
                                                    variables[item['source']])
                    new_items.append(item)
                masks[new_name]['items'] = new_items

                for var, nvar in variables.items():
                    self.rename(var.split('@')[-1], nvar)

        else:
            if not new_name:
                raise ValueError("'new_name' is needed to rename column variables")
            if new_name in data.columns:
                msg = "Cannot rename '{}' into '{}'. Column name already exists!"
                raise ValueError(msg.format(name, new_name))
            data.rename(columns={name: new_name}, inplace=True)
            columns[new_name] = columns[name].copy()
            del columns[name]
            columns[new_name]['name'] = new_name
            o_set_entry = 'columns@{}'.format(name)
            n_set_entry = 'columns@{}'.format(new_name)
            n_datafile_items = [i if i != o_set_entry else n_set_entry
                                  for i in sets['data file']['items']]
            sets['data file']['items'] = n_datafile_items
        return None

    def reorder_values(self, name, new_order=None):
        """
        Apply a new order to the value codes defined by the meta data component.

        Parameters
        ----------
        name : str
            The column variable name keyed in ``_meta['columns']`` or
            ``_meta['masks']``.
        new_order : list of int, default None
            The new code order of the DataSet variable. If no order is given,
            the ``values`` object is sorted ascending.

        Returns
        -------
        None
            DataSet is modified inplace.
        """
        self._verify_var_in_dataset(name)
        values = self._get_value_loc(name)
        if not new_order:
            new_order = list(sorted(self._get_valuemap(name, 'codes')))
        else:
            self._verify_old_vs_new_codes(name, new_order)
        new_values = [value for i in new_order for value in values
                      if value['value'] == i]
        if self._get_type(name) == 'array':
            self._meta['lib']['values'][name] = new_values
        else:
            self._meta['columns'][name]['values'] = new_values
        return None

    def drop(self, name, ignore_items=False):
        """
        Drops variables from meta and data componenets of the ``DataSet``.

        Parameters
        ----------
        name : str or list of str
            The column variable name keyed in ``_meta['columns']`` or
            ``_meta['masks']``.
        ignore_items: bool
            If False source variables for arrays in ``_meta['columns']``
            are dropped, otherwise kept.
        Returns
        -------
        None
            DataSet is modified inplace.
        """
        def remove_loop(obj, var):
            if isinstance(obj, dict):
                try:
                    obj.pop(var)
                except:
                    pass
                for key in obj:
                    remove_loop(obj[key],var)
        meta = self._meta
        data = self._data
        if not isinstance(name, list): name = [name]
        if not ignore_items:
            for var in name:
                if self._is_array(var):
                    items = [i['source'].split('@')[-1]
                            for i in meta['masks'][var]['items']]
                    name += items
        data_drop = []
        for var in name:
            if not self._is_array(var): data_drop.append(var)
            remove_loop(meta, var)
        data.drop(data_drop, 1, inplace=True)
        return None

    def remove_values(self, name, remove):
        """
        Erase value codes safely from both meta and case data components.

        Attempting to remove all value codes from the variable's value object
        will raise a ``ValueError``!

        Parameters
        ----------
        name : str
            The originating column variable name keyed in ``meta['columns']``
            or ``meta['masks']``.
        remove : int or list of int
            The codes to be removed from the ``DataSet`` variable.
        Returns
        -------
        None
            DataSet is modified inplace.
        """
        self._verify_var_in_dataset(name)
        if not isinstance(remove, list): remove = [remove]
        values = self._get_value_loc(name)
        codes = self.codes(name)
        ignore_codes = [r for r in remove if r not in codes]
        if ignore_codes:
            print 'Warning: Cannot remove values...'
            print '*' * 60
            msg = "Codes {} not found in values object of '{}'!"
            print msg.format(ignore_codes, name)
            print '*' * 60
        new_values = [value for value in values
                      if value['value'] not in remove]
        if not new_values:
            msg = "Cannot remove all codes from the value object of '{}'!"
            raise ValueError(msg.format(name))
        if self._get_type(name) == 'array':
            self._meta['lib']['values'][name] = new_values
        else:
            self._meta['columns'][name]['values'] = new_values
        if self._is_array(name):
            items = self._get_itemmap(name, 'items')
            for i in items:
                self.remove_values(i, remove)
        else:
            if self._is_delimited_set(name):
                self._remove_from_delimited_set_data(name, remove)
            else:
                self._data[name].replace(remove, np.NaN, inplace=True)
            self._verify_data_vs_meta_codes(name)
        return None

    def remove_items(self, name, remove):
        """
        Erase array mask items safely from both meta and case data components.

        Parameters
        ----------
        name : str
            The originating column variable name keyed in ``meta['masks']``.
        remove : int or list of int
            The items listed by their order number in the
            ``_meta['masks'][name]['items']`` object will be droped from the
            ``mask`` definition.

        Returns
        -------
        None
            DataSet is modified inplace.
        """
        if not self._is_array(name):
            raise NotImplementedError('Cannot remove items from non-arrays!')
        if not isinstance(remove, list): remove = [remove]
        items = self._get_itemmap(name, 'items')
        drop_item_names = [item for idx, item in enumerate(items, start=1)
                        if idx in remove]
        keep_item_idxs = [idx for idx, item in enumerate(items, start=1)
                          if idx not in remove]
        new_items = self._meta['masks'][name]['items']
        new_items = [item for idx, item in enumerate(new_items, start=1)
                     if idx in keep_item_idxs]
        self._meta['masks'][name]['items'] = new_items
        for drop_item_name in drop_item_names:
            self._data.drop(drop_item_name, axis=1, inplace=True)
            del self._meta['columns'][drop_item_name]
            col_ref = 'columns@{}'.format(drop_item_name)
            self._meta['sets']['data file']['items'].remove(col_ref)
            self._meta['sets'][name]['items'].remove(col_ref)
        return None

    def extend_values(self, name, ext_values, text_key=None, safe=True):
        """
        Add to the 'values' object of existing column or mask meta data.

        Attempting to add already existing value codes or providing already
        present value texts will both raise a ``ValueError``!

        Parameters
        ----------
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
        safe : bool, default True
            If set to False, duplicate value texts are allowed when extending
            the ``values`` object.

        Returns
        -------
        None
            The ``DataSet`` is modified inplace.
        """
        self._verify_var_in_dataset(name)
        is_array = self._is_array(name)
        if not self._has_categorical_data(name):
            err_msg = '{} does not contain categorical values meta!'
            raise TypeError(err_msg.format(name))
        if not text_key: text_key = self.text_key
        if not isinstance(ext_values, list): ext_values = [ext_values]
        value_obj = self._get_valuemap(name, text_key=text_key)
        codes = self.codes(name)
        texts = self.value_texts(name)
        if not isinstance(ext_values[0], tuple):
            start_here = self._next_consecutive_code(codes)
        else:
            start_here = None
        ext_values = self._make_values_list(ext_values, text_key, start_here)
        dupes = []
        for ext_value in ext_values:
            code, text = ext_value['value'], ext_value['text'][text_key]
            if code in codes or (text in texts and safe):
                dupes.append((code, text))
        if dupes:
            msg = 'Cannot add values since code and/or text already exists: {}'
            raise ValueError(msg.format(dupes))
        if is_array:
            self._meta['lib']['values'][name].extend(ext_values)
            ext_lib_ref = self._meta['lib']['values'][name]
            for item in self._get_itemmap(name, 'items'):
                self._meta['columns'][item]['values'] = ext_lib_ref
        else:
            self._meta['columns'][name]['values'].extend(ext_values)
        return None

    def set_text_key(self, text_key):
        """
        Set the default text_key of the ``DataSet``.

        .. note:: A lot of the instance methods will fall back to the default
            text key in ``_meta['lib']['default text']``. It is therefore
            important to use this method with caution, i.e. ensure that the
            meta contains ``text`` entries for the ``text_key`` set.

        Parameters
        ----------
        text_key : {'en-GB', 'da-DK', 'fi-FI', 'nb-NO', 'sv-SE', 'de-DE'}
            The text key that will be set in ``_meta['lib']['default text']``.

        Returns
        -------
        None
        """
        self._is_valid_text_key(text_key)
        self.text_key = text_key
        self._meta['lib']['default text'] = text_key
        return None

    def find_duplicate_texts(self, name, text_key=None):
        """
        Collect values that share the same text information to find duplicates.

        Parameters
        ----------
        name : str
            The column variable name keyed in ``_meta['columns']`` or
            ``_meta['masks']``.
        text_key : str, default None
            Text key for text-based label information. Will automatically fall
            back to the instance's ``text_key`` property information if not
            provided.
        """
        if not text_key: text_key = self.text_key
        values = self._get_valuemap(name, text_key=text_key)
        dupes_check = []
        text_dupes = []
        for value in values:
            if value[1] in dupes_check:
                text_dupes.append(value[1])
            dupes_check.append(value[1])
        text_dupes = list(set(text_dupes))
        dupes = []
        for value in values:
            if value[1] in text_dupes:
                dupes.append(value)
        dupes = list(sorted(dupes, key=lambda x: x[1]))
        return dupes

    def force_texts(self, name=None, copy_to=None, copy_from=None,
                    update_existing=False, excepts=None):
        """
        Copy info from existing text_key to a new one or update the existing

        Parameters
        ----------
        name : str / list of str / None
            Variable names for that the text info are forced
            None -> all meta objects in masks and columns
        copy_to : str
            {'en-GB', 'da-DK', 'fi-FI', 'nb-NO', 'sv-SE', 'de-DE'}
            None -> _meta['lib']['default text']
            The text key that will be filled.
        copy from : str / list
            {'en-GB', 'da-DK', 'fi-FI', 'nb-NO', 'sv-SE', 'de-DE'}
            You can also enter a list with text_keys, if the first text_key
            doesn't exist, it takes the next one
        update_existing : bool
            True : copy_to will be filled in any case
            False: copy_to will be filled if it's empty/not existing
        excepts : str or list of str
            If provided, the variables passed are ignored while transferring
            ``text`` information.

        Returns
        -------
        None
        """
        def _force_texts(tk_dict, copy_to, copy_from, update_existing):
            if isinstance(tk_dict, dict):
                new_text_key = None
                for new_tk in reversed(copy_from):
                    if new_tk in tk_dict.keys():
                        new_text_key = new_tk
                if not new_text_key:
                    raise ValueError('{} is no existing text_key'.format(copy_from))
                if update_existing:
                    tk_dict.update({copy_to: tk_dict[new_text_key]})
                else:
                    if not copy_to in tk_dict.keys():
                        tk_dict.update({copy_to: tk_dict[new_text_key]})
            return tk_dict

        meta = self._meta
        if not isinstance(name, list) and name != None: name = [name]
        if not isinstance(excepts, list): excepts = [excepts]
        excepts.append('@1')
        if copy_to == None: copy_to = meta['lib']['default text']
        if copy_from == None:
            raise ValueError('parameter copy_from needs an input')
        elif not isinstance(copy_from, list): copy_from = [copy_from]

        #grids / masks
        for mask_name, mask_def in meta['masks'].items():
            if mask_name in excepts or not (name == None or mask_name in name):
                continue
            mask_def['text'] = _force_texts(tk_dict= mask_def['text'],
                                        copy_to=copy_to,
                                        copy_from=copy_from,
                                        update_existing=update_existing)
            for no, item in enumerate(mask_def['items']):
                if 'text' in item.keys():
                    item['text'] = _force_texts(tk_dict= item['text'],
                                            copy_to=copy_to,
                                            copy_from=copy_from,
                                            update_existing=update_existing)
                    mask_def['items'][no]['text'] = item['text']

            #lib
            for no, value in enumerate(meta['lib']['values'][mask_name]):
                value['text'] == _force_texts(tk_dict= value['text'],
                                        copy_to=copy_to,
                                        copy_from=copy_from,
                                        update_existing=update_existing)
                meta['lib']['values'][mask_name][no]['text'] = value['text']

        #columns
        for column_name, column_def in meta['columns'].items():
            if not (name == None or column_name in name) or column_name in excepts:
                continue
            column_def['text'] = _force_texts(tk_dict= column_def['text'],
                                        copy_to=copy_to,
                                        copy_from=copy_from,
                                        update_existing=update_existing)
            if ('values' in column_def.keys() and
                isinstance(column_def['values'], list)):
                for no, value in enumerate(column_def['values']):
                    value['text'] = _force_texts(tk_dict= value['text'],
                                        copy_to=copy_to,
                                        copy_from=copy_from,
                                        update_existing=update_existing)
                    column_def['values'][no]['text'] = value['text']


    @classmethod
    def _is_valid_text_key(cls, tk):
        """
        """
        valid_tks = ['en-GB', 'da-DK', 'fi-FI', 'nb-NO', 'sv-SE', 'de-DE']
        if tk not in valid_tks:
            msg = "{} is not a valid text_key! Supported are: \n {}"
            msg = msg.format(tk, valid_tks)
            raise ValueError(msg)
        else:
            return True

    def clean_texts(self, clean_html=True, replace=None):
        """
        Cycle through all meta ``text`` objects replacing unwanted tags/terms.

        Parameters
        ----------
        clean_html : bool, default True
            If True, all ``text``s will be stripped from any html tags.
            Currently uses the regular expression: '<.*?>'
        replace : dict, default None
            A dictionary mapping {unwanted string: replacement string}.

        Returns
        -------
        None
        """
        def remove_html(text):
            """
            """
            import re
            remove = re.compile('<.*?>')
            text = re.sub(remove, '', text)
            remove = '(<|\$)(.|\n)+?(>|.raw |.raw)'
            return re.sub(remove, '', text)

        def replace_from_dict(obj, tk, replace_map):
            """
            """
            for k, v, in replace_map.items():
                text = obj['text'][tk]
                obj['text'][tk] = text.replace(k, v)

        meta = self._meta
        for mask_name, mask_def in meta['masks'].items():
            try:
                for tk in mask_def['text']:
                    text = mask_def['text'][tk]
                    if clean_html:
                        mask_def['text'][tk] = remove_html(text)
                    if replace:
                        replace_from_dict(mask_def, tk, replace)
            except:
                pass
            try:
                for no, item in enumerate(mask_def['items']):
                    for tk in item['text']:
                        text = item['text'][tk]
                        if clean_html:
                            mask_def['items'][no]['text'][tk] = remove_html(text)
                        if replace:
                            replace_from_dict(item, tk, replace)
            except:
                pass
            mask_vals = meta['lib']['values'][mask_name]
            try:
                for no, val in enumerate(mask_vals):
                    for tk in val['text']:
                        text = val['text'][tk]
                        if clean_html:
                            mask_vals[no]['text'][tk] = remove_html(text)
                        if replace:
                            replace_from_dict(val, tk, replace)
            except:
                pass
        for column_name, column_def in meta['columns'].items():
            try:
                for tk in column_def['text']:
                    text = column_def['text'][tk]
                    if clean_html:
                        column_def['text'][tk] = remove_html(text)
                    if replace:
                        replace_from_dict(column_def, tk, replace)
                if 'values' in column_def:
                    for no, value in enumerate(column_def['values']):
                        for tk in value['text']:
                            text = value['text'][tk]
                            if clean_html:
                                column_def['values'][no]['text'][tk] = remove_html(text)
                            if replace:
                                replace_from_dict(value, tk, replace)
            except:
                pass

    def set_value_texts(self, name, renamed_vals, text_key=None):
        """
        Rename or add value texts in the 'values' object.

        This method works for array masks and column meta data.

        Parameters
        ----------
        name : str
            The column variable name keyed in ``_meta['columns']`` or
            ``_meta['masks']``.
        renamed_vals : dict
            A dict mapping with following structure:
            ``{1: 'new label for code=1', 5: 'new label for code=5'}``
            Codes will be ignored if they do not exist in the 'values' object.
        text_key : str, default None
            Text key for text-based label information. Will automatically fall
            back to the instance's ``text_key`` property information if not
            provided.

        Returns
        -------
        None
            The ``DataSet`` is modified inplace.
        """
        self._verify_var_in_dataset(name)
        if not self._has_categorical_data(name):
            err_msg = '{} does not contain categorical values meta!'
            raise TypeError(err_msg.format(name))
        if not text_key: text_key = self.text_key
        if not self._is_array(name):
            obj_values = self._meta['columns'][name]['values']
            new_obj_values = []
        else:
            obj_values = self._meta['lib']['values'][name]
            new_obj_values = []
        ignore = [k for k in renamed_vals.keys() if k not in self.codes(name)]
        if ignore:
            print 'Warning: Cannot set new value texts...'
            print '*' * 60
            msg = "Codes {} not found in values object of '{}'!"
            print msg.format(ignore, name)
            print '*' * 60
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

    def set_item_texts(self, name, renamed_items, text_key=None):
        """
        Rename or add item texts in the 'items' object of a ``mask``.

        Parameters
        ----------
        name : str
            The column variable name keyed in ``_meta['masks']``.
        renamed_items : dict
            A dict mapping with following structure (array mask items are
            assumed to be passed by their order number):
            ``{1: 'new label for item #1',
               5: 'new label for item #5'}``
        text_key : str, default None
            Text key for text-based label information. Will automatically fall
            back to the instance's ``text_key`` property information if not
            provided.

        Returns
        -------
        None
            The ``DataSet`` is modified inplace.
        """
        if not self._is_array(name):
            raise KeyError('{} is not a mask.'.format(name))
        if not text_key: text_key = self.text_key
        items = self.sources(name)
        item_obj = self._meta['masks'][name]['items']
        for item_no, item_text in renamed_items.items():
            text_update = {text_key: item_text}
            i = items[item_no - 1]
            self._meta['columns'][i]['text'].update(text_update)
            for i_obj in item_obj:
                if i_obj['source'].split('@')[-1] == i:
                    i_obj['text'].update(text_update)
        return None

    def set_col_text_edit(self, name, edited_text, axis='x'):
        """
        Inject a question label edit that will take effect at build stage.

        Parameters
        ----------
        name : str
            The column variable name keyed in ``_meta['columns']``.
        edited_text : str
            The desired question label text.
        axis: {'x', 'y', ['x', 'y']}, default 'x'
            The axis the edited text should appear on.

        Returns
        -------
        None
        """
        if not isinstance(axis, list): axis = [axis]
        if axis not in [['x'], ['y'], ['x', 'y'], ['y', 'x']]:
            raise ValueError('No valid axis provided!')
        for ax in axis:
            tk = 'x edits' if ax == 'x' else 'y edits'
            self.set_column_text(name, edited_text, tk)

    def set_val_text_edit(self, name, edited_vals, axis='x'):
        """
        Inject cat. value label edits that will take effect in at build stage.

        Parameters
        ----------
        name : str
            The column variable name keyed in ``_meta['columns']``.
        edited_vals : dict
            Mapping of value code to ``'text'`` label.
        axis: {'x', 'y', ['x', 'y']}, default 'x'
            The axis the edited text should appear on.

        Returns
        -------
        None
        """
        if not isinstance(axis, list): axis = [axis]
        if axis not in [['x'], ['y'], ['x', 'y'], ['y', 'x']]:
            raise ValueError('No valid axis provided!')
        for ax in axis:
            tk = 'x edits' if ax == 'x' else 'y edits'
            self.set_value_texts(name, edited_vals, tk)

    def set_property(self, name, prop_name, prop_value, ignore_items=False):
        """
        Access and set the value of a meta object's ``properties`` collection.

        Parameters
        ----------
        name : str
            The originating column variable name keyed in ``meta['columns']``
            or ``meta['masks']``.
        prop_name : str
            The property key name.
        prop_value : any
            The value to be set for the property. Must be of valid type and
            have allowed values(s) with regard to the property.
        ignore_items : bool, default False
            When ``name`` refers to a variable from the ``'masks'`` collection,
            setting to True will ignore any ``items`` and only apply the
            property to the ``mask`` itself.
        Returns
        -------
        None
        """
        valid_props = ['base_text']
        if prop_name not in valid_props:
            raise ValueError("'prop_name' must be one of {}").format(valid_props)
        prop_update = {prop_name: prop_value}
        if  self._is_array(name):
            if not 'properties' in self._meta['masks'][name]:
                self._meta['masks'][name]['properties'] = {}
            self._meta['masks'][name]['properties'].update(prop_update)
            if not ignore_items:
                items = self.sources(name)
                for i in items:
                    self.set_property(i, prop_name, prop_value)
        else:
            if not 'properties' in self._meta['columns'][name]:
                self._meta['columns'][name]['properties'] = {}
            self._meta['columns'][name]['properties'].update(prop_update)
        return None

    def set_sliced(self, name, slicer, axis='y'):
        """
        Set or update ``rules[axis]['slicex']`` meta for the named column.

        Quantipy builds will respect the kept codes and *show them exclusively*
        in results.

        .. note:: This is not a replacement for ``DataSet.set_missings()`` as
            missing values are respected also in computations.

        Parameters
        ----------
        name : str
            The column variable name keyed in ``_meta['columns']``.
        slice : int or list of int
            Values indicated by their ``int`` codes will be shown in
            ``Quantipy.View.dataframe``s, respecting the provided order.
        axis : {'x', 'y'}, default 'y'
            The axis to slice the values on.

        Returns
        -------
        None
        """
        if self._is_array(name):
            raise NotImplementedError('Cannot slice codes from arrays!')
        if 'rules' not in self._meta['columns'][name]:
            self._meta['columns'][name]['rules'] = {'x': {}, 'y': {}}
        if not isinstance(slicer, list): slicer = [slicer]
        slicer = self._clean_codes_against_meta(name, slicer)
        rule_update = {'slicex': {'values': slicer}}
        self._meta['columns'][name]['rules'][axis].update(rule_update)
        return None

    def set_hidden(self, name, hide, axis='y'):
        """
        Set or update ``rules[axis]['dropx']`` meta for the named column.

        Quantipy builds will respect the hidden codes and *cut* them from
        results.

        .. note:: This is not equivalent to ``DataSet.set_missings()`` as
            missing values are respected also in computations.

        Parameters
        ----------
        name : str
            The column variable name keyed in ``_meta['columns']``.
        hide : int or list of int
            Values indicated by their ``int`` codes will be dropped from
            ``Quantipy.View.dataframe``s.
        axis : {'x', 'y'}, default 'y'
            The axis to drop the values from.

        Returns
        -------
        None
        """
        if self._is_array(name):
            raise NotImplementedError('Cannot hide codes on arrays!')
        if 'rules' not in self._meta['columns'][name]:
            self._meta['columns'][name]['rules'] = {'x': {}, 'y': {}}
        if not isinstance(hide, list): hide = [hide]
        hide = self._clean_codes_against_meta(name, hide)
        if set(hide) == set(self._get_valuemap(name, 'codes')):
            msg = "Cannot hide all values of '{}'' on '{}'-axis"
            raise ValueError(msg.format(name, axis))
        rule_update = {'dropx': {'values': hide}}
        self._meta['columns'][name]['rules'][axis].update(rule_update)
        return None

    def set_sorting(self, name, fix=None, ascending=False):
        """
        Set or update ``rules['x']['sortx']`` meta for the named column.

        Parameters
        ----------
        name : str
            The column variable name keyed in ``_meta['columns']``.
        fix : int or list of int, default None
            Values indicated by their ``int`` codes will be ignored in
            the sorting operation.
        ascending : bool, default False
            By default frequencies are sorted in descending order. Specify
            ``True`` to sort ascending.

        Returns
        -------
        None
        """
        if self._is_array(name):
            raise NotImplementedError('Cannot sort arrays / array items!')
        if 'rules' not in self._meta['columns'][name]:
            self._meta['columns'][name]['rules'] = {'x': {}, 'y': {}}
        if fix:
            if not isinstance(fix, list): fix = [fix]
        else:
            fix = []
        fix = self._clean_codes_against_meta(name, fix)
        rule_update = {'sortx': {'ascending': ascending, 'fixed': fix}}
        self._meta['columns'][name]['rules']['x'].update(rule_update)
        return None

    def set_variable_text(self, name, new_text, text_key=None):
        """
        Apply a new or update a column's/masks' meta text object.

        Parameters
        ----------
        name : str
            The originating column variable name keyed in ``meta['columns']``
            or ``meta['masks']``.
        new_text : str
            The ``text`` (label) to be set.
        text_key : str, default None
            Text key for text-based label information. Will automatically fall
            back to the instance's text_key property information if not provided.

        Returns
        -------
        None
            The ``DataSet`` is modified inplace.
        """
        self._verify_var_in_dataset(name)
        if not text_key: text_key = self.text_key
        collection = 'masks' if self._is_array(name) else 'columns'
        if text_key in self._meta[collection][name]['text'].keys():
            self._meta[collection][name]['text'][text_key] = new_text
        else:
            text_update = {text_key: new_text}
            self._meta[collection][name]['text'].update(text_update)
        return None

    # will be removed soon!
    def set_column_text(self, name, new_text, text_key=None):
        """
        Apply a new or update a column's meta text object.

        Parameters
        ----------

        Returns
        -------
        """
        warning = "'set_column_text()' will be removed soon!"
        warning = warning + " Use 'set_variable_text()' instead!"
        warnings.warn(warning)
        self._verify_column_in_meta(name)
        if not text_key: text_key = self.text_key
        if text_key in self._meta['columns'][name]['text'].keys():
            self._meta['columns'][name]['text'][text_key] = new_text
        else:
            self._meta['columns'][name]['text'].update({text_key: new_text})
        return None

    def set_mask_text(self, name, new_text, text_key=None):
        """
        Apply a new or update a masks' meta text object.

        Parameters
        ----------

        Returns
        -------
        """
        warning = "'set_mask_text()' will be removed soon!"
        warning = warning + " Use 'set_variable_text()' instead!"
        warnings.warn(warning)
        if not text_key: text_key = self.text_key
        if text_key in self._meta['masks'][name]['text'].keys():
            self._meta['masks'][name]['text'][text_key] = new_text
        else:
            self._meta['masks'][name]['text'].update({text_key: new_text})
        return None

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
        datafile_setname = 'masks@{}'.format(array_name)
        if datafile_setname not in self._meta['sets']['data file']['items']:
            self._meta['sets']['data file']['items'].append(datafile_setname)
        self._meta['sets'][array_name] = {'items': [i['source'] for i in item_objects]}
        return None

    def copy_var(self, name, suffix='rec', copy_data=True):
        # WILL BE REMOVED SOON
        self.copy(name, suffix, copy_data)

    def copy(self, name, suffix='rec', copy_data=True):
        """
        Copy meta and case data of the variable defintion given per ``name``.

        Parameters
        ----------
        name : str
            The originating column variable name keyed in ``meta['columns']``
            or ``meta['masks']``.
        suffix : str, default 'rec'
            The new variable name will be constructed by suffixing the original
            ``name`` with ``_suffix``, e.g. ``'age_rec``.
        Returns
        -------
        None
            DataSet is modified inplace, adding a copy to both the data and meta
            component.
        """
        self._verify_var_in_dataset(name)
        copy_name = '{}_{}'.format(name, suffix)
        if self._is_array(name):
            items = self._get_itemmap(name, 'items')
            mask_meta_copy = org_copy.deepcopy(self._meta['masks'][name])
            if not 'masks@' + copy_name in self._meta['sets']['data file']['items']:
                self._meta['sets']['data file']['items'].append('masks@' + copy_name)
            mask_set = []
            for i, i_meta in zip(items, mask_meta_copy['items']):
                self.copy(i, suffix, copy_data)
                i_name = '{}_{}'.format(i, suffix)
                i_meta['source'] = 'columns@{}'.format(i_name)
                mask_set.append('columns@{}'.format(i_name))
            lib_ref = 'lib@values@{}'.format(copy_name)
            lib_copy = org_copy.deepcopy(self._meta['lib']['values'][name])
            mask_meta_copy['values'] = lib_ref
            self._meta['masks'][copy_name] = mask_meta_copy
            self._meta['lib']['values'][copy_name] = lib_copy
            self._meta['sets'][copy_name] = {'items': mask_set}
        else:
            if copy_data:
                self._data[copy_name] = self._data[name].copy()
            else:
                self._data[copy_name] = np.NaN
            meta_copy = org_copy.deepcopy(self._meta['columns'][name])
            self._meta['columns'][copy_name] = meta_copy
            self._meta['columns'][copy_name]['name'] = copy_name
            if not 'columns@' + copy_name in self._meta['sets']['data file']['items']:
                self._meta['sets']['data file']['items'].append('columns@' + copy_name)
        return None

    def code_count(self, name, count_only=None):
        """
        Get the total number of codes/entries found per row.

        .. note:: Will be 0/1 for type ``single`` and range between 0 and the
            number of possible values for type ``delimited set``.

        Parameters
        ----------
        name : str
            The column variable name keyed in ``meta['columns']``.
        count_only : int or list of int, default None
            Pass a list of codes that should no be counted.

        Returns
        -------
        count : pandas.Series
            A series with the results as ints.
        """
        if self._is_array(name) or self._is_numeric(name):
            raise TypeError('Can only count codes on categorical data columns!')
        dummy = self.make_dummy(name, partitioned=False)
        if count_only:
            if not isinstance(count_only, list): count_only = [count_only]
            dummy = dummy[count_only]
        count = dummy.sum(axis=1)
        return count

    def is_nan(self, name):
        """
        Detect empty entries in the ``_data`` rows.

        Parameters
        ----------
        name : str
            The column variable name keyed in ``meta['columns']``.

        Returns
        -------
        count : pandas.Series
            A series with the results as bool.
        """
        if self._is_array(name):
            raise TypeError("Can only check 'np.NaN' on non-mask variables!")
        return self._data[name].isnull()

    def any(self, name, codes):
        """
        Return a logical has_any() slicer for the passed codes.

        .. note:: When applied to an array mask, the has_any() logic is ex-
            tended to the item sources, i.e. the it must itself be true for
            *at least one of* the items.

        Parameters
        ----------
        name : str, default None
            The column variable name keyed in ``_meta['columns']`` or
            ``_meta['masks']``.
        codes : int or list of int
            The codes to build the logical slicer from.

        Returns
        -------
        slicer : pandas.Index
            The indices fulfilling has_any([codes]).
        """
        if not isinstance(codes, list): codes = [codes]
        if self._is_array(name):
            logics = []
            for s in self.sources(name):
                logics.append({s: has_any(codes)})
            slicer = self.slicer(union(logics))
        else:
            slicer = self.slicer({name: has_any(codes)})
        return slicer

    def all(self, name, codes):
        """
        Return a logical has_all() slicer for the passed codes.

        .. note:: When applied to an array mask, the has_all() logic is ex-
            tended to the item sources, i.e. the it must itself be true for
            *all* the items.

        Parameters
        ----------
        name : str, default None
            The column variable name keyed in ``_meta['columns']`` or
            ``_meta['masks']``.
        codes : int or list of int
            The codes to build the logical slicer from.

        Returns
        -------
        slicer : pandas.Index
            The indices fulfilling has_all([codes]).
        """
        if not isinstance(codes, list): codes = [codes]
        if self._is_array(name):
            logics = []
            for s in self.sources(name):
                logics.append({s: has_all(codes)})
            slicer = self.slicer(intersection(logics))
        else:
            slicer = self.slicer({name: has_all(codes)})
        return slicer

    def crosstab(self, x, y=None, w=None, pct=False, decimals=1, text=True,
                 rules=False, xtotal=False):
        """
        """
        meta, data = self.split()
        y = '@' if not y else y
        get = 'count' if not pct else 'normalize'
        show = 'values' if not text else 'text'
        return ct(meta, data, x=x, y=y, get=get, weight=w, show=show,
                  rules=rules, xtotal=xtotal, decimals=decimals)

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

    def _clean_codes_against_meta(self, name, codes):
        return [c for c in codes if c in self._get_valuemap(name, 'codes')]

    @staticmethod
    def _item(item_name, text_key, text):
        """
        """
        return {'source': 'columns@{}'.format(item_name),
                'text': {text_key: text}}

    def copy_array_data(self, source, target, source_items=None,
                        target_items=None, slicer=None):
        """
        """
        self._verify_same_value_codes_meta(source, target)
        all_source_items = self._get_itemmap(source, non_mapped='items')
        all_target_items = self._get_itemmap(target, non_mapped='items')
        if slicer: mask = self.slicer(slicer)
        if source_items:
            source_items = [all_source_items[i-1] for i in source_items]
        else:
            source_items = all_source_items
        if target_items:
            target_items = [all_target_items[i-1] for i in target_items]
        else:
            target_items = all_target_items
        for s, t in zip(source_items, target_items):
                if slicer:
                    self._data.loc[mask, t] = self._data.loc[mask, s]
                else:
                    self[t] = self[s]
        return None

    def unify_values(self, name, code_map, slicer=None, exclusive=False):
        """
        Use a mapping of old to new codes to replace code values in ```_data``.

        .. note:: Experimental! Check results carefully!

        Parameters
        ----------
        name : str
            The column variable name keyed in ``meta['columns']``.
        code_map : dict
            A mapping of ``{old: new}``; ``old`` and ``new`` must be the
            int-type code values from the column meta data.
        slicer : Quantipy logic statement, default None
            If provided, the values will only be unified for cases where the
            condition holds.
        exclusive : bool, default False
            If True, the recoded unified value will replace whatever is already
            found in the ``_data`` column, ignoring ``delimited set`` typed data
            to which normally would get appended to.

        Returns
        -------
        None
        """
        append = self._is_delimited_set(name)
        if exclusive: append = False
        for old_code, new_code in code_map.items():
            self.recode(name, {new_code: {name: [old_code]}},
                        append=append, intersect=slicer)
            if not slicer:
                self.remove_values(name, old_code)
            else:
                msg = "Unified {} >> {} on data slice. Remove values meta if needed!"
                print msg.format(old_code, new_code)
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

    def transpose_array(self, name, new_name=None, ignore_items=None,
                        ignore_values=None, copy_data=True, text_key=None):
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
            ``name`` with '_trans', e.g. ``'Q2Array_trans``.
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
            new_name = '{}_trans'.format(name)

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
                if copy_data:
                    slicer = {reg_item_name: [reg_val_code]}
                    update_with = new_val_code
                    self.recode(trans_item, {update_with: slicer},
                                append=True)
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

    def recode(self, target, mapper, default=None, append=False,
               intersect=None, initialize=None, fillna=None, inplace=True):
        """
        Create a new or copied series from data, recoded using a mapper.

        This function takes a mapper of {key: logic} entries and injects the
        key into the target column where its paired logic is True. The logic
        may be arbitrarily complex and may refer to any other variable or
        variables in data. Where a pre-existing column has been used to
        start the recode, the injected values can replace or be appended to
        any data found there to begin with. Note that this function does
        not edit the target column, it returns a recoded copy of the target
        column. The recoded data will always comply with the column type
        indicated for the target column according to the meta.

        Parameters
        ----------
        target : str
            The column variable name keyed in ``_meta['columns']`` that is the
            target of the recode. If not found in ``_meta`` this will fail
            with an error. If ``target`` is not found in data.columns the
            recode will start from an empty series with the same index as
            ``_data``. If ``target`` is found in data.columns the recode will
            start from a copy of that column.
        mapper : dict
            A mapper of {key: logic} entries.
        default : str, default None
            The column name to default to in cases where unattended lists
            are given in your logic, where an auto-transformation of
            {key: list} to {key: {default: list}} is provided. Note that
            lists in logical statements are themselves a form of shorthand
            and this will ultimately be interpreted as:
            {key: {default: has_any(list)}}.
        append : bool, default False
            Should the new recodd data be appended to values already found
            in the series? If False, data from series (where found) will
            overwrite whatever was found for that item instead.
        intersect : logical statement, default None
            If a logical statement is given here then it will be used as an
            implied intersection of all logical conditions given in the
            mapper.
        initialize : str or np.NaN, default None
            If not None, a copy of the data named column will be used to
            populate the target column before the recode is performed.
            Alternatively, initialize can be used to populate the target
            column with np.NaNs (overwriting whatever may be there) prior
            to the recode.
        fillna : int, default=None
            If not None, the value passed to fillna will be used on the
            recoded series as per pandas.Series.fillna().
        inplace : bool, default True
            If True, the ``DataSet`` will be modified inplace with new/updated
            columns. Will return a new recoded ``pandas.Series`` instance if
            False.

        Returns
        -------
        None or recode_series
            Either the ``DataSet._data`` is modfied inplace or a new
            ``pandas.Series`` is returned.
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
            if not self._is_numeric(target):
                self._verify_data_vs_meta_codes(target)
            return None
        else:
            return recode_series

    def interlock(self, name, label, variables, val_text_sep = '/'):
        """
        Build a new category-intersected variable from >=2 incoming variables.

        Parameters
        ----------
        name : str
            The new column variable name keyed in ``_meta['columns']``.
        label : str
            The new text label for the created variable.
        variables : list of >= 2 str
            The column names of the variables that are feeding into the
            intersecting recode operation.
        val_text_sep : str, default '/'
            The passed character (or any other str value) wil be used to
            separate the incoming individual value texts to make up the inter-
            sected category value texts, e.g.: 'Female/18-30/London'.

        Returns
        -------
        None
        """
        if not isinstance(variables, list) or len(variables) < 2:
            raise ValueError("'variables' must be a list of at least two items!")
        if any(self._is_array(v) for v in variables):
            raise TypeError('Cannot interlock within array-typed variables!')
        if any(self._is_delimited_set(v) for v in variables):
            qtype = 'delimited set'
        else:
            qtype = 'single'
        codes = [self._get_valuemap(v, 'codes') for v in variables]
        texts = [self._get_valuemap(v, 'texts') for v in variables]
        zipped = zip(list(product(*codes)), list(product(*texts)))
        categories = []
        cat_id = 0
        for codes, texts in zipped:
            cat_id += 1
            label = val_text_sep.join(texts)
            rec = [{v: [c]} for v, c in zip(variables, codes)]
            rec = intersection(rec)
            categories.append((cat_id, label, rec))
        self.derive(name, qtype, label, categories)
        return None

    def derive_categorical(self, name, qtype, label, cond_map, text_key=None):
        warning = "'derive_categorical()' will be removed soon!"
        warning = warning + " Use 'derive()' instead!"
        warnings.warn(warning)
        return self.derive(name, qtype, label, cond_map, text_key)

    def derive(self, name, qtype, label, cond_map, text_key=None):
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

    def band_numerical(self, name, bands, new_name=None, label=None, text_key=None):
        warning = "'band_numerical()' will be removed soon!"
        warning = warning + " Use 'band()' instead!"
        warnings.warn(warning)
        return self.band(name, bands, new_name, label, text_key)

    def band(self, name, bands, new_name=None, label=None, text_key=None):
        """
        Group numeric data with band defintions treated as group text labels.

        Wrapper around ``derive()`` for quick banding of numeric
        data.

        Parameters
        ----------
        name : str
            The column variable name keyed in ``_meta['columns']`` that will
            be banded into summarized categories.
        bands : list of int/tuple *or* dict mapping the former to value texts
            The categorical bands to be used. Bands can be single numeric
            values or ranges, e.g.: [0, (1, 10), 11, 12, (13, 20)].
            Be default, each band will also make up the value text of the
            category created in the ``_meta`` component. To specify custom
            texts, map each band to a category name e.g.:
             [{'A': 0},
              {'B': (1, 10)},
              {'C': 11},
              {'D': 12},
              {'E': (13, 20)}]
        new_name : str, default None
            The created variable will be named ``'<name>_banded'``, unless a
            desired name is provided explicitly here.
        label : str, default None
            The created variable's text label will be identical to the origi-
            nating one's passed in ``name``, unless a desired label is provided
            explicitly here.
        text_key : str, default None
            Text key for text-based label information. Uses the
            ``DataSet.text_key`` information if not provided.

        Returns
        -------
        None
            ``DataSet`` is modified inplace.
        """
        if self._is_array(name):
            raise TypeError('Cannot band array mask!')
        if not self._is_numeric(name):
            msg = "Can only band numeric typed data! {} is {}."
            msg = msg.format(name, self._get_type(name))
            raise TypeError(msg)
        if not text_key: text_key = self.text_key
        if not new_name: new_name = '{}_banded'.format(new_name)
        if not label: label = self._get_label(name, text_key)
        franges = []
        for idx, band in enumerate(bands, start=1):
            lab = None
            if isinstance(band, dict):
                lab = band.keys()[0]
                band = band.values()[0]
            if isinstance(band, tuple):
                r = '{}-{}'.format(band[0], band[1])
            else:
                r = str(band)
            franges.append([idx, lab or r, {name: frange(r)}])
        self.derive(new_name, 'single', label, franges,
                                text_key=text_key)

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

    def weight(self, weight_scheme, weight_name='weight', unique_key='identity',
               report=True, inplace=True):
        """
        Weight the ``DataSet`` according to a well-defined weight scheme.

        Parameters
        ----------
        weight_scheme : quantipy.Rim instance
            A rim weights setup with defined targets. Can include multiple
            weight groups and/or filters.
        weight_name : str, default 'weight'
            A name for the float variable that is added to pick up the weight
            factors.
        unique_key : str, default 'identity'.
            A variable inside the ``DataSet`` instance that will be used to
            the map individual case weights to their matching rows.
        report : bool, default True
            If True, will report a summary of the weight algorithm run
            and factor outcomes.
        inplace : bool, default True
            If True, the weight factors are merged back into the ``DataSet``
            instance. Will otherwise return the ``pandas.DataFrame`` that
            contains the weight factors, the ``unique_key`` and all variables
            that have been used to compute the weights (filters, target
            variables, etc.).

        Returns
        -------
        None or ``pandas.DataFrame``
            Will either create a new column called ``'weight'`` in the
            ``DataSet`` instance or return a ``DataFrame`` that contains
            the weight factors.
        """
        meta, data = self.split()
        engine = qp.WeightEngine(data, meta)
        engine.add_scheme(weight_scheme, key=unique_key)
        engine.run()
        org_wname = weight_name
        if report:
            print engine.get_report()
        if inplace:
            scheme_name = weight_scheme.name
            weight_name = 'weights_{}'.format(scheme_name)
            weight_description = '{} weights'.format(scheme_name)
            data_wgt = engine.dataframe(scheme_name)[[unique_key, weight_name]]
            data_wgt.rename(columns={weight_name: org_wname}, inplace=True)
            if org_wname not in self._meta['columns']:
                self.add_meta(org_wname, 'float', weight_description)
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
            Passing 'default' is using a preset list of (TODO: specify) values
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



    def describe(self, var=None, only_type=None, text_key=None):
        """
        Inspect the DataSet's global or variable level structure.
        """
        if text_key is None: text_key = self.text_key
        if var is not None:
            return self._get_meta(var, only_type, text_key)
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
            if only_type:
                if not isinstance(only_type, list): only_type = [only_type]
                types = types[only_type]
                types = types.replace('', np.NaN).dropna(how='all')
            else:
                types =  types[['single', 'delimited set', 'array', 'int',
                                'float', 'string', 'date', 'time', 'N/A']]
            types.columns.name = 'size: {}'.format(len(self._data))
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

    def _verify_data_vs_meta_codes(self, name, raiseError=True):
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
            if self._verbose_errors:
                msg = "Warning: Meta not consistent with case data for '{}'!"
                print '*' * 60
                print msg.format(name)
                if raiseError: print '*' * 60
                print 'Found in data: {}'.format(data_codes)
                print 'Defined as per meta: {}'.format(meta_codes)
            if raiseError:
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
            if self._verbose_errors:
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
            raise TypeError("Numerical columns do not have 'values' meta.")
        if not self._has_categorical_data(var):
            raise TypeError("Variable '{}' is not categorical!".format(var))
        loc = self._get_meta_loc(var)
        if not self._is_array(var):
            return emulate_meta(self._meta, loc[var].get('values', None))
        else:
            return emulate_meta(self._meta, loc[var])

    def _get_valuemap(self, var, non_mapped=None, text_key=None):
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

    def _verify_var_in_dataset(self, name):
        if not name in self._meta['masks'] and not name in self._meta['columns']:
            raise KeyError("'{}' not found in DataSet!".format(name))

    def _get_meta(self, var, type=None, text_key=None):
        self._verify_var_in_dataset(var)
        if text_key is None: text_key = self.text_key
        var_type = self._get_type(var)
        label = self._get_label(var, text_key)
        missings = self._get_missing_map(var)
        if self._has_categorical_data(var):
            codes, texts = self._get_valuemap(var, non_mapped='lists',
                                              text_key=text_key)
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
                items, items_texts = self._get_itemmap(var, non_mapped='lists',
                                                       text_key=text_key)
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
            meta_df.index = xrange(1, len(meta_df.index) + 1)
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
            vartype = self._get_type(var)
            if vartype == 'delimited set':
                try:
                    dummy_data = self[var].str.get_dummies(';')
                except:
                    dummy_data = self._data[[var]]
                    dummy_data.columns = [0]
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
                return dummy_data
            else:
                return dummy_data.values, codes, items

    def filter(self, alias, condition, inplace=False):
        """
        Filter the DataSet using a Quantipy logical expression.
        """
        data = self._data.copy()
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
            new_ds.text_key = self.text_key
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


    # ------------------------------------------------------------------------
    # validate the dataset
    # ------------------------------------------------------------------------

    def validate(self, verbose=True):
        """
        Identify and report inconsistencies in the ``DataSet`` instance.
        """

        def err_appender(text, err_var, app, count, text_key):
            if not isinstance(text, dict):
                if err_var[0] == '': err_var[0] += app + count
                elif err_var[0] == 'x': err_var[0] += ', ' +app + count
                else: err_var[0] += ', '  + count
            elif self.text_key not in text:
                if err_var[1] == '': err_var[1] += app + count
                elif err_var[1] == 'x': err_var[1] += ', ' +app + count
                else: err_var[1] += ', '  + count
            elif text[text_key] in [None, '', ' ']:
                if err_var[2] == '': err_var[2] += app + count
                elif err_var[2] == 'x': err_var[2] += ', ' +app + count
                else: err_var[2] += ', '  + count
            return err_var


        def append_loop(err_var_item, app, count):
            if app in err_var_item:
                err_var_item += ', ' + str(count)
            else:
                err_var_item += app + ' ' + str(count)
            return err_var_item

        def data_vs_meta_codes(name):
            if not name in self._data: return False
            if self._is_delimited_set(name):
                data_codes = self._data[name].str.get_dummies(';').columns.tolist()
                data_codes = [int(c) for c in data_codes]
            else:
                data_codes = pd.get_dummies(self._data[name]).columns.tolist()
            meta_codes = self._get_valuemap(name, non_mapped='codes')
            wild_codes = [code for code in data_codes if code not in meta_codes]
            return wild_codes

        meta = self._meta
        data = self._data

        text_key = self.text_key

        msg = ("Error explanations:\n"
               "\tErr1: Text object is not a dict.\n"
               "\tErr2: Text object does not contain dataset text_key '{}'.\n"
               "\tErr3: Text object has empty text mapping.\n"
               "\tErr4: Categorical object does not contain any 'Values'.\n"
               "\tErr5: Categorical object has badly formatted 'Values'.\n"
               "\t\t (not a list or reference does not exist)\n"
               "\tErr6: 'Source' reference does not exist.\n"
               "\tErr7: '._data' contains codes that are not in ._meta.\n")
        msg = msg.format(text_key)

        err_columns = ['Err{}'.format(x) for x in range(1,8)]
        err_df = pd.DataFrame(columns=err_columns)

        for ma in meta['masks']:
            if ma.startswith('qualityControl_'): continue

            err_var = ['' for x in range(7)]

            mask = meta['masks'][ma]
            if 'text' in mask:
                text = mask['text']
                err_var = err_appender(text, err_var, '', 'x', text_key)
            for x, item in enumerate(mask['items']):
                if 'text' in item:
                    text = item['text']
                    err_var = err_appender(text, err_var, 'item ',
                                                str(x), text_key)
                if 'source' in item:
                    if (isinstance(item['source'], basestring)
                        and '@' in item['source']):
                        try:
                            ref = item['source'].split('@')
                            if not ref[-1] in meta[ref[0]]:
                                err_var[5] = append_loop(err_var[5],
                                                         'item ', x)
                        except:
                            err_var[5] = append_loop(err_var[5], 'item ', x)
                    else:
                        err_var[5] = append_loop(err_var[5], 'item ', x)
            if not 'values' in mask:
                err_var[3] = 'x'
            elif not (isinstance(mask['values'], list) or
                      isinstance(mask['values'], basestring) and
                      mask['values'].split('@')[-1] in meta['lib']['values']):
                err_var[4] = 'x'

            if not ('').join(err_var) == '':
                new_err = pd.DataFrame([err_var], index=[ma],
                                       columns=err_columns)
                err_df = err_df.append(new_err)

        excepts = [col for col in data if col not in meta['columns']]

        for col in meta['columns']:
            if col.startswith('qualityControl_'): continue

            err_var = ['' for x in range(7)]

            column = meta['columns'][col]
            if 'text' in column:
                text = column['text']
                err_var = err_appender(text, err_var, '', 'x', text_key)
            if 'values' in column:
                if not (isinstance(column['values'], list) or
                        isinstance(column['values'], basestring) and
                        column['values'].split('@')[-1] in meta['lib']['values']):
                    err_var[4] = 'x'
                for x, val in enumerate(column['values']):
                    if 'text' in val:
                        text = val['text']
                        err_var = err_appender(text, err_var, 'value ',
                                                    str(x), text_key)
            elif ('values' not in column and
                 column['type'] in ['delimited set', 'single']):
                err_var[3] = 'x'


            if (self._has_categorical_data(col) and err_var[3] == '' and
                err_var[4] == ''):
                if data_vs_meta_codes(col):
                    err_var[6] = 'x'



            if not ('').join(err_var) == '':
                new_err = pd.DataFrame([err_var], index=[col],
                                       columns=err_columns)
                err_df = err_df.append(new_err)

        for col in excepts:
            if col.startswith('id_'): continue
            else:
                new_err = pd.DataFrame([['x' for x in range(7)]], index=[col],
                                       columns=err_columns)
                err_df = err_df.append(new_err)

        if verbose:
            if not len(err_df) == 0:
                print msg
                return err_df.sort()
            else:
                print 'no issues found in dataset'
        else:
            return err_df.sort()


    def validate_backup(self, text=True, categorical=True, codes=True):
        """
        Validates variables/ text objects/ ect in the dataset
        """

        meta = self._meta
        data = self._data

        # validate text-objects
        if text:
            self.validate_text_objects(test_object=None, name=None)

        # validate categorical objects (single, delimited set, array)
        if codes: categorical = True
        if categorical:
            error_list = self.validate_categorical_objects()

        # validate data vs meta codes
        if codes:
            for key in data.keys():
                if key.startswith('id_') or key in error_list: continue
                elif self._has_categorical_data(key):
                    self._verify_data_vs_meta_codes(key, raiseError=False)



    def _proof_values(self, variable, name, error_list):
        msg = "Warning: Meta is not consistent for '{}'!"

        if not 'values' in variable.keys():
            error_list.append(name)
            print '*' * 60
            print msg.format(name)
            print "Meta doesn't contain any codes"
            return error_list

        values = variable['values']
        if isinstance(values, list):
            return error_list
        elif (isinstance(values, basestring) and
            values.split('@')[-1] in self._meta['lib']['values']):
            return error_list
        else:
            error_list.append(name)
            print '*' * 60
            print msg.format(name)
            print "Codes are not a list or reference doesn't exist"
            return error_list


    def validate_categorical_objects(self):

        meta = self._meta
        data = self._data

        error_list = []

        # validate delimited set, single
        for col in meta['columns']:
            var = meta['columns'][col]
            if var['type'] in ['delimited set', 'single']:
                error_list = self._proof_values(variable=var, name=col,
                                                error_list=error_list)

        # validate array
        for mask in meta['masks']:
            arr = meta['masks'][mask]
            error_list = self._proof_values(variable=arr, name=mask,
                                            error_list=error_list)
            for item in arr['items']:
                ref = item['source'].split('@')
                if ref[-1] in meta[ref[0]]: continue
                else:
                    print '*' * 60
                    print "Warning: Meta is not consistent for '{}'!".format(mask)
                    print "Source reference {} doesn't exist".format(ref[-1])

        return error_list


    @classmethod
    def _validate_text_objects(cls, test_object, text_key, name):

        msg = "Warning: Text object is not consistent: '{}'!"
        if not isinstance(test_object, dict):
            print '*' * 60
            print msg.format(name)
            print 'Text object is not a dict'
        elif text_key not in test_object.keys():
            print '*' * 60
            print msg.format(name)
            print 'Text object does not contain dataset-text_key {}'.format(
                text_key)
        elif test_object[text_key] in [None, '', ' ']:
            print '*' * 60
            print msg.format(name)
            print 'Text object has empty text mapping'


    def validate_text_objects(self, test_object, name=None):
        """
        Prove all text objects in the dataset.
        """

        meta = self._meta
        text_key = self.text_key

        if test_object == None : test_object= self._meta
        if name == None:
            name = 'meta'

        if isinstance(test_object, dict):
            for key in test_object.keys():
                new_name = name + '[' + key + ']'
                if 'text' == key:
                    self._validate_text_objects(test_object['text'],
                                                text_key, new_name)
                elif (key in ['properties', 'data file'] or
                    'qualityControl' in key):
                    continue
                else:
                    self.validate_text_objects(test_object[key], new_name)
        elif isinstance(test_object, list):
            for i, item in enumerate(test_object):
                new_name = name + '[' + str(i) + ']'
                self.validate_text_objects(item,new_name)


    # ------------------------------------------------------------------------
    # checking equality of variables and datasets
    # ------------------------------------------------------------------------

    def _compare(self, var1, var2, check_ds=None, text_key=None):
        """
        Compares types, codes, values, question labels of two variables.

        Parameters
        ----------
        var1: str
            Variablename that gets checked.
        var2: str
            Variablename that gets checked.
        check_ds: DataSet instance
            var2 is in this DataSet instance.
        """
        if not check_ds: check_ds = self
        if not text_key: text_key = self.text_key
        msg = '*' * 60 + "\n'{}' and '{}' are not identical:"
        if not self._get_label(var1, text_key) == check_ds._get_label(var2, text_key):
            msg = msg + '\n  - not the same label.'
        if not self._get_type(var1) == check_ds._get_type(var2):
            msg = msg + '\n  - not the same type.'
        if self._has_categorical_data(var1) and self._has_categorical_data(var2):
            if not (self._get_valuemap(var1, None, text_key) ==
                    check_ds._get_valuemap(var2, None, text_key)):
                msg = msg + '\n  - not the same values object.'
        if (self._is_array(var1) and
            not (self._get_itemmap(var1, None, text_key) ==
            check_ds._get_itemmap(var2, None, text_key))):
            msg = msg + '\n  - not the same items object.'
        if not msg[-1] == ':': print msg.format(var1, var2)
        return None

    def compare(self, dataset=None, variables=None, text_key=None):
        """
        Compares types, codes, values, question labels of two datasets.

        Parameters
        ----------
        dataset : quantipy.DataSet instance
            Test if all variables in the provided ``dataset`` are also in
            ``self`` and compare their metadata definititons.
        variables : tuple of str, e.g. ('var1', 'var2')
            If no other ``dataset`` is provided, both variables are taken from
            ``self``, otherwise 'var1' is from ``self``, 'var2' is from
            ``dataset``.

        Returns
        -------
        None
        """

        if not dataset: dataset = self
        if not text_key: text_key = self.text_key
        meta = self._meta
        check_meta = dataset._meta
        if isinstance(variables, tuple):
            var1, var2 = variables
            if not var1 in meta['columns'].keys() + meta['masks'].keys():
                raise ValueError('{} is not in dataset.'.format(var1))
            if not var2 in check_meta['columns'].keys() + check_meta['masks'].keys():
                raise ValueError('{} is not in dataset.'.format(var2))
            self._compare(var1, var2, dataset, text_key)
        elif not variables:
            vars1 = {item.split('@')[1] : item.split('@')[0]
                     for item in meta['sets']['data file']['items']}
            vars2 = {item.split('@')[1] : item.split('@')[0]
                     for item in check_meta['sets']['data file']['items']}
            prove = [key for key in vars2 if key in vars1]
            for key in prove:
                self._compare(key, key, dataset, text_key)
        else:
            raise ValueError("'variables' must be a tuple of two str or None.")
        return None

# ============================================================================

    def parrot(self):
        from IPython.display import Image
        from IPython.display import display
        try:
            return display(Image(url="https://m.popkey.co/3a9f4b/jZZ83.gif"))
        except:
            print ':sad_parrot: Looks like the parrot url is not longer there!'
