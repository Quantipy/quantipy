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

from quantipy.core.tools.qp_decorators import *

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
    frange,
    index_mapper)

from cache import Cache

import copy as org_copy
import json
import warnings
import re

from itertools import product, chain
from collections import OrderedDict

VALID_TKS = ['en-GB', 'da-DK', 'fi-FI', 'nb-NO', 'sv-SE', 'de-DE', 'fr-FR']

class DataSet(object):
    """
    A set of casedata (required) and meta data (optional).

    DESC.
    """
    def __init__(self, name, dimensions_comp=True):
        self.path = None
        self.name = name
        self.filtered = 'no_filter'
        self._data = None
        self._meta = None
        self.text_key = None
        self.valid_tks = VALID_TKS
        self._verbose_errors = True
        self._verbose_infos = True
        self._cache = Cache()
        self._dimensions_comp = dimensions_comp
        return None

    # ------------------------------------------------------------------------
    # item access / instance handlers
    # ------------------------------------------------------------------------
    def _get_columns(self, vtype=None):
        if self._meta:
            meta = self._meta['columns']
            if vtype:
                return [c for c in meta.keys() if self._get_type(c) == vtype]
            else:
                return meta.keys()
        else:
            return None

    def _get_masks(self):
        if self._meta:
            return self._meta['masks'].keys()
        else:
            return None

    def _get_sets(self):
        if self._meta:
            return self._meta['sets'].keys()
        else:
            return None

    def columns(self):
        return self._get_columns()

    def sets(self):
        return self._get_sets()

    def masks(self):
        return self._get_masks()

    def singles(self):
        return self._get_columns('single')

    def delimited_sets(self):
        return self._get_columns('delimited set')

    def ints(self):
        return self._get_columns('int')

    def floats(self):
        return self._get_columns('float')

    def dates(self):
        return self._get_columns('date')

    def strings(self):
        return self._get_columns('string')

    def __getitem__(self, var):
        if isinstance(var, tuple):
            sliced_access = True
            slicer = var[0]
            var = var[1]
        else:
            sliced_access = False
        var = self.unroll(var)
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
        if self._get_type(name) == 'delimited set' and scalar_insert:
            val = '{};'.format(val)
        if sliced_insert:
            self._data.loc[slicer, name] = val
        else:
            self._data[name] = val

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

    def _fix_array_meta(self):
        """
        Update array meta. Workaround for badly converted meta.
        """
        # insert 'parent' entry
        for c in self.columns():
            if not 'parent' in self._meta['columns'][c]:
                self._meta['columns'][c]['parent'] = {}
        for mask in self.masks():
            # fix 'subtype' property
            if not 'subtype' in self._meta['masks'][mask]:
                subtype = self._get_type(self.sources(mask)[0])
                self._meta['masks'][mask]['subtype'] = subtype
            # fix 'parent' meta on arrays
            parent_def = {'masks@{}'.format(mask): {'type': 'array'}}
            for s in self.sources(mask):
                if self._meta['columns'][s]['parent'] == {}:
                    self._meta['columns'][s]['parent'] = parent_def

    def _fix_array_item_vals(self):
        """
        Update value meta for array items. Workaround for badly converted meta.
        """
        for m in self.masks():
            lib_vals = 'lib@values@{}'.format(m)
            for s in self.sources(m):
                self._meta['columns'][s]['values'] = lib_vals

    def _clean_datafile_set(self):
        """
        Drop references from ['sets']['data file']['items'] if they do not exist
        in the ``DataSet`` columns or masks definitions.
        """
        items = self._meta['sets']['data file']['items']
        n_items = [i for i in items if self.var_exists( i.split('@')[-1])]
        self._meta['sets']['data file']['items'] = n_items
        return None

    def repair(self):
        """
        Try to fix legacy meta data inconsistencies and badly shaped array /
        datafile items ``'sets'`` meta definitions.
        """
        self._fix_array_meta()
        self._fix_array_item_vals()
        self.repair_text_edits()
        self._clean_datafile_set()
        return None


    @verify(variables={'name': 'both'}, text_keys='text_key', axis='axis_edit')
    def meta(self, name=None, text_key=None, axis_edit=None):
        """
        Provide a *pretty* summary for variable meta given as per ``name``.

        Parameters
        ----------
        name : str, default None
            The variable name keyed in ``_meta['columns']`` or ``_meta['masks']``.
            If None, the entire ``meta`` component of the ``DataSet`` instance
            will be returned.
        text_key : str, default None
            The text_key that should be used when taking labels from the
            source meta.
        axis_edit : {'x', 'y'}, default None
            If provided the text_key is taken from the x/y edits dict.

        Returns
        ------
        meta : dict or pandas.DataFrame
            Either a DataFrame that sums up the meta information on a ``mask``
            or ``column`` or the meta dict as a whole is
        """
        if not name:
            return self._meta
        else:
            return self.describe(name, text_key=text_key, axis_edit=axis_edit)

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

    @verify(variables={'name': 'both'}, text_keys='text_key', axis='axis_edit')
    def text(self, name, shorten=True, text_key=None, axis_edit=None):
        """
        Return the variables text label information.

        Parameters
        ----------
        name : str, default None
            The variable name keyed in ``_meta['columns']`` or ``_meta['masks']``.
        shorten : bool, default True
            If True, ``text`` label meta from array items will not report
            the parent mask's ``text``. Setting it to False will show the
            "full" label.
        text_key : str, default None
            The default text key to be set into the new meta document.
        axis_edit : {'x', 'y'}, default None
            If provided the text_key is taken from the x/y edits dict.

        Returns
        -------
        text : str
            The text metadata.
        """
        def _text_from_textobj(textobj, text_key, axis_edit):
            if axis_edit:
                a_edit = '{} edits'.format(axis_edit)
                return textobj.get(a_edit, {}).get(text_key, None)
            else:
                return textobj.get(text_key, None)

        if text_key is None: text_key = self.text_key
        shorten = False if not self._is_array_item(name) else shorten
        collection = 'masks' if self._is_array(name) else 'columns'
        if not shorten:
            return _text_from_textobj(self._meta[collection][name]['text'],
                                      text_key, axis_edit)
        else:
            parent = self._maskname_from_item(name)
            item_no = self.item_no(name)
            item_texts = self._meta['masks'][parent]['items'][item_no-1]['text']
            return _text_from_textobj(item_texts, text_key, axis_edit)

    @verify(variables={'name': 'both'}, categorical='name',
            text_keys='text_key', axis='axis_edit')
    def values(self, name, text_key=None, axis_edit=None):
        """
        Get categorical data's paired code and texts information from the meta.

        Parameters
        ----------
        name : str
            The column variable name keyed in ``_meta['columns']`` or
            ``_meta['masks']``.
        text_key : str, default None
            The text_key that should be used when taking labels from the
            source meta.
        axis_edit : {'x', 'y'}, default None
            If provided the text_key is taken from the x/y edits dict.

        Returns
        -------
        values : list of tuples
            The list of the numerical category codes and their ``texts``
            packed as tuples.
        """
        return self._get_valuemap(name, text_key=text_key, axis_edit=axis_edit)

    @verify(variables={'name': 'both'})
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

    @verify(variables={'name': 'both'}, text_keys='text_key', axis='axis_edit')
    def value_texts(self, name, text_key=None, axis_edit=None):
        """
        Get categorical data's text information.

        Parameters
        ----------
        name : str
            The column variable name keyed in ``_meta['columns']``.
        text_key : str, default None
            The text_key that should be used when taking labels from the
            source meta.
        axis_edit : {'x', 'y'}, default None
            If provided the text_key is taken from the x/y edits dict.

        Returns
        -------
        texts : list
            The list of category texts.
        """
        return self._get_valuemap(name, 'texts', text_key, axis_edit)

    @verify(variables={'name': 'columns'}, categorical='name')
    def codes_in_data(self, name):
        """
        Get a list of codes that exist in data.
        """
        if self._is_delimited_set(name):
            if not self._data[name].dropna().empty:
                data_codes = self._data[name].str.get_dummies(';').columns.tolist()
                data_codes = [int(c) for c in data_codes]
            else:
                data_codes = []
        else:
            data_codes = pd.get_dummies(self._data[name]).columns.tolist()
        return data_codes

    @verify(variables={'name': 'masks'}, text_keys='text_key', axis='axis_edit')
    def items(self, name, text_key=None, axis_edit=None):
        """
        Get the array's paired item names and texts information from the meta.

        Parameters
        ----------
        name : str
            The column variable name keyed in ``_meta['masks']``.
        text_key : str, default None
            The text_key that should be used when taking labels from the
            source meta.
        axis_edit : {'x', 'y'}, default None
            If provided the text_key is taken from the x/y edits dict.

        Returns
        -------
        items : list of tuples
            The list of source item names (from ``_meta['columns']``) and their
            ``text`` information packed as tuples.
        """
        return self._get_itemmap(name, text_key=text_key, axis_edit=axis_edit)

    @verify(variables={'name': 'both'})
    def parents(self, name):
        """
        Get the ``parent`` meta information for masks-structured column elements.

        Parameters
        ----------
        name : str
            The mask variable name keyed in ``_meta['columns']``.

        Returns
        -------
        parents : list
            The list of parents the ``_meta['columns']`` variable is attached to.
        """
        if not self._is_array_item(name):
            return []
        else:
            return [parent for parent in self._meta['columns'][name]['parent']]

    @verify(variables={'name': 'both'})
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
        if not self._is_array(name):
            return []
        else:
            return self._get_itemmap(name, non_mapped='items')

    @verify(variables={'name': 'masks'}, text_keys='text_key', axis='axis_edit')
    def item_texts(self, name, text_key=None, axis_edit=None):
        """
        Get the ``text`` meta data for the items of the passed array mask name.

        Parameters
        ----------
        name : str
            The mask variable name keyed in ``_meta['masks']``.
        text_key : str, default None
            The text_key that should be used when taking labels from the
            source meta.
        axis_edit : {'x', 'y'}, default None
            If provided the text_key is taken from the x/y edits dict.

        Returns
        -------
        texts : list
            The list of item texts for the array elements.
        """
        return self._get_itemmap(name, 'texts', text_key, axis_edit)

    @verify(variables={'name': 'columns'})
    def item_no(self, name):
        """
        Return the order/position number of passed array item variable name.

        Parameters
        ----------
        name : str
            The column variable name keyed in ``_meta['columns']``.

        Returns
        -------
        no : int
            The positional index of the item (starting from 1).
        """
        sources = self.sources(self._maskname_from_item(name))
        return sources.index(name) + 1

    def data(self):
        """
        Return the ``data`` component of the ``DataSet`` instance.
        """
        return self._data

    def _cache(self):
        return self._cache

    def _add_inferred_meta(self, tk):
        msg = "Inferring meta data from pd.DataFrame.columns ({})..."
        msg = msg.format(len(self._data.columns))
        print msg
        self._data.reset_index(inplace=True)
        self._data.drop('index', axis=1, inplace=True)
        self._meta = self.start_meta(tk)
        self.text_key = tk
        for col in self._data.columns:
            name = col
            pdtype = str(self._data[col].dtype)
            if 'int' in pdtype:
                qptype = 'int'
            elif 'float' in pdtype:
                qptype = 'float'
            elif pdtype == 'object':
                qptype = 'string'
            else:
                qptype = None
            if not qptype:
                msg = "Could not infer type for {} (dtype: {})!"
                print msg.format(name, pdtype)
            else:
                msg = "{}: dtype: {} - converted: {}"
                print msg.format(name, pdtype, qptype)
                self.add_meta(name, qptype, '', replace=False)
        return None


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

    # ------------------------------------------------------------------------
    # verify
    # ------------------------------------------------------------------------

    @modify(to_list='name')
    def var_exists(self, name):
        variables = self._get_masks() + self._get_columns()
        return all(var in variables for var in name)

    # ------------------------------------------------------------------------
    # file i/o / conversions
    # ------------------------------------------------------------------------
    def read_quantipy(self, path_meta, path_data, reset=True):
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
        reset : bool, default True
            Clean the `'lib'` and ``'sets'`` metadata collections from non-native
            entries, e.g. user-defined information or helper metadata.

        Returns
        -------
        None
            The ``DataSet`` is modified inplace, connected to Quantipy native
            data and meta components.
        """
        if path_meta.endswith('.json'): path_meta = path_meta.replace('.json', '')
        if path_data.endswith('.csv'): path_data = path_data.replace('.csv', '')
        self._meta, self._data = r_quantipy(path_meta+'.json', path_data+'.csv')
        self._set_file_info(path_data, path_meta, reset=reset)
        for col in self.columns():
            if self._dims_compat_arr_name(col) in self.masks():
                renamed = '{}_{}'.format(col, self._get_type(col).replace(' ', '_'))
                msg = ("*** WARNING ***: Found {}-type variable name also in "
                       "'masks'. Renaming to '{}'")
                print msg.format(self._get_type(col), renamed)
                self.rename(col, renamed)
        self.undimensionize()
        if self._dimensions_comp:
            self.dimensionize()
            self._meta['info']['dimensions_comp'] = True
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
        self.undimensionize()
        if self._dimensions_comp:
            self.dimensionize()
            self._meta['info']['dimensions_comp'] = True
        return None

    @verify(text_keys='text_key')
    def read_ascribe(self, path_meta, path_data, text_key):
        """
        Load Dimensions .xml/.txt files, connecting as data and meta components.

        Parameters
        ----------
        path_meta : str
            The full path (optionally with extension ``'.xml'``, otherwise
            assumed as such) to the meta data defining ``'.xml'`` file.
        path_data : str
            The full path (optionally with extension ``'.txt'``, otherwise
            assumed as such) to the case data defining ``'.txt'`` file.

        Returns
        -------
        None
            The ``DataSet`` is modified inplace, connected to Quantipy data
            and meta components that have been converted from their Ascribe
            source files.
        """
        if path_meta.endswith('.xml'): path_meta = path_meta.replace('.xml', '')
        if path_data.endswith('.txt'): path_data = path_data.replace('.txt', '')
        self._meta, self._data = r_ascribe(path_meta+'.xml', path_data+'.txt', text_key)
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

    @verify(text_keys='text_key')
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

    @verify(text_keys='text_key')
    def from_components(self, data_df, meta_dict=None, reset=True, text_key=None):
        """
        Attach data and meta directly to the ``DataSet`` instance.

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
        reset : bool, default True
            Clean the `'lib'` and ``'sets'`` metadata collections from non-native
            entries, e.g. user-defined information or helper metadata.
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
        if meta_dict and not isinstance(meta_dict, dict):
            msg = 'meta_dict must be of type dict, passed {}.'
            raise TypeError(msg.format(type(meta_dict)))
        self._data = data_df
        if meta_dict:
            self._meta = meta_dict
        else:
            if not text_key: text_key = 'en-GB'
            self._add_inferred_meta(text_key)
        if not text_key:
            try:
                self.text_key = self._meta['lib']['default text']
            except (KeyError, TypeError):
                warning = "No 'text_key' provided and unable to derive"
                warning = warning + " 'text_key' information from passed meta!"
                warning = warning + " 'DataSet._meta might be corrupt!"
                warnings.warn(warning)
                self.text_key = None
        self.set_verbose_infomsg(False)
        self._set_file_info('', reset=reset)
        if self._meta['info'].get('dimensions_comp'):
            self._dimensions_comp = True
        else:
            self._dimensions_comp = False
        return None

    def from_stack(self, stack, datakey=None, dk_filter=None, reset=True):
        """
        Use ``quantipy.Stack`` data and meta to create a ``DataSet`` instance.

        Parameters
        ----------
        stack : quantipy.Stack
            The Stack instance to convert.
        datakey : str
            The reference name where meta and data information are stored.
        dk_filter: string, default None
            Filter name if the stack contains more than one filters. If None
            'no_filter' will be used.
        reset : bool, default True
            Clean the `'lib'` and ``'sets'`` metadata collections from non-native
            entries, e.g. user-defined information or helper metadata.

        Returns
        -------
        None
        """
        if datakey is None and len(stack.keys()) > 1:
            msg = 'Please specify the datakey, stack has more than one.'
            raise ValueError(msg)
        elif datakey is None:
            datakey = stack.keys()[0]
        elif not datakey in stack.keys():
            msg = 'datakey does not exist.'
            raise KeyError(msg)

        if not dk_filter:
            dk_f = 'no_filter'
        elif dk_filter in stack[datakey].keys():
            msg = 'Please insert an existing filter of the stack:\n{}'.format(
                stack[datakey].keys())
            raise KeyError(msg)

        meta = stack[datakey].meta
        data = stack[datakey][dk_f].data
        self.name = datakey
        self.filtered = dk_f
        self.from_components(data, meta, reset=reset)

        return None

    @verify(variables={'unique_key': 'columns'})
    def from_excel(self, path_xlsx, merge=True, unique_key='identity'):
        """
        Converts excel files to a dataset or/and merges variables.

        Parameters
        ----------
        path_xlsx : str
            Path where the excel file is stored. The file must have exactly
            one sheet with data.
        merge : bool
            If True the new data from the excel file will be merged on the
            dataset.
        unique_key : str
            If ``merge=True`` an hmerge is done on this variable.

        Returns
        -------
        new_dataset : ``quantipy.DataSet``
            Contains only the data from excel.
            If ``merge=True`` dataset is modified inplace.
        """

        xlsx = pd.read_excel(path_xlsx, sheetname=None)

        if not len(xlsx.keys()) == 1:
            raise KeyError("The XLSX must have exactly 1 sheet.")
        key = xlsx.keys()[0]
        sheet = xlsx[key]
        if merge and not unique_key in sheet.columns:
            raise KeyError(
            "The coding sheet must a column named '{}'.".format(unique_key))

        new_ds = qp.DataSet('excel_data')
        new_ds._data = pd.DataFrame()
        new_ds._meta = new_ds.start_meta()
        for col in sheet.columns.tolist():
            new_ds.add_meta(col, 'int', col)
        new_ds._data = sheet

        if merge:
            self.hmerge(new_ds, on=unique_key, verbose=False)

        return new_ds

    def _set_file_info(self, path_data, path_meta=None, reset=True):
        self.path = '/'.join(path_data.split('/')[:-1]) + '/'
        self.text_key = self._meta['lib'].get('default text')
        self.valid_tks = self._meta['lib'].get('valid text', VALID_TKS)
        self._data['@1'] = np.ones(len(self._data))
        self._meta['columns']['@1'] = {'type': 'int'}
        self._data.index = list(xrange(0, len(self._data.index)))
        if self._verbose_infos: self._show_file_info()
        if reset:
            # drop user-defined / unknown 'sets' & 'lib' entries:
            valid_sets = self.masks() + ['data file', 'batches']
            found_sets = self._meta['sets'].keys()
            valid_libs = ['default text', 'valid text', 'values']
            found_libs = self._meta['lib'].keys()
            for set_def in found_sets:
                if set_def not in valid_sets:
                    del self._meta['sets'][set_def]
            for lib_def in found_libs:
                if lib_def not in valid_libs:
                    del self._meta['lib'][lib_def]
        return None

    def _show_file_info(self):
        file_spec = ('DataSet: {}\nrows: {} - columns: {}\n'
                     'Dimensions compatibility mode: {}')
        if not self.path: self.path = '/'
        file_name = '{}{}'.format(self.path, self.name)
        print file_spec.format(file_name, len(self._data.index),
                               len(self._data.columns)-1, self._dimensions_comp)
        return None

    # ------------------------------------------------------------------------
    # lists/sets of variables
    # ------------------------------------------------------------------------

    @modify(to_list=['varlist', 'keep', 'both'])
    @verify(variables={'varlist': 'both', 'keep': 'masks'})
    def unroll(self, varlist, keep=None, both=None):
        """
        Replace mask with their items, optionally excluding/keeping certain ones.

        Parameters
        ----------
        varlist : list
           A list of meta ``'columns'`` and/or ``'masks'`` names.
        keep : str or list, default None
            The names of masks that will not be replaced with their items.
        both : 'all', str or list of str, default None
            The names of masks that will be included both as themselves and as
            collections of their items.

        Returns
        -------
        unrolled : list
            The modified ``varlist``.
        """
        if both and both[0] == 'all':
            both = [mask for mask in varlist if mask in self._meta['masks']]
        unrolled = []
        for var in varlist:
            if not self._is_array(var):
                unrolled.append(var)
            else:
                if not var in keep:
                    if var in both:
                        unrolled.append(var)
                    unrolled.extend(self.sources(var))
                else:
                    unrolled.append(var)
        return unrolled

    @modify(to_list='blacklist')
    def list_variables(self, numeric=False, text=False, blacklist=None):
        """
        Get list with all variable names except date, boolean, (string, numeric)

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
        for item in items_list:
            key, var_name = item.split('@')
            if key == 'masks':
                for element in meta[key][var_name]['items']:
                    blacklist.append(element['source'].split('@')[-1])
            if var_name in blacklist: continue
            if meta[key][var_name]['type'] in except_list: continue
            var_list.append(var_name)
        return var_list

    @modify(to_list=['included', 'excluded'])
    @verify(variables={'included': 'both', 'excluded': 'both'})
    def create_set(self, setname='new_set', based_on='data file', included=None,
                   excluded=None, strings='keep', arrays='masks', replace=None,
                   overwrite=False):
        """
        Create a new set in ``dataset._meta['sets']``.

        Parameters
        ----------
        setname : str, default 'new_set'
            Name of the new set.
        based_on : str, default 'data file'
            Name of set that can be reduced or expanded.
        included : str or list/set/tuple of str
            Names of the variables to be included in the new set. If None all
            variables in ``based_on`` are taken.
        excluded : str or list/set/tuple of str
            Names of the variables to be excluded in the new set.
        strings : {'keep', 'drop', 'only'}, default 'keep'
            Keep, drop or only include string variables.
        arrays : {'both', 'masks', 'columns'}, default both
            Add for arrays ``masks@varname`` or ``columns@varname`` or both.
        replace : dict
            Replace a variable in the set with an other.
            Example: {'q1': 'q1_rec'}, 'q1' and 'q1_rec' must be included in
                     ``based_on``. 'q1' will be removed and 'q1_rec' will be
                     moved to this position.
        overwrite : bool, default False
            Overwrite if ``meta['sets'][name] already exist.
        Returns
        -------
        None
            The ``DataSet`` is modified inplace.
        """
        meta = self._meta
        sets = meta['sets']
        # prove setname
        if not isinstance(setname, str):
            raise TypeError("'setname' must be a str.")
        if setname in sets and not overwrite:
            raise KeyError("{} is already in `meta['sets'].`".format(setname))
        # prove based_on
        if not based_on in sets:
            raise KeyError("'based_on' is not in `meta['sets'].`")
        # prove included
        if not included: included = [var.split('@')[-1] for var in sets[based_on]['items']]

        # prove replace
        if not replace: replace = {}
        elif not isinstance(replace, dict):
            raise TypeError("'replace' must be a dict.")
        else:
            for var in replace.keys() + replace.values():
                if var not in included:
                    raise KeyError("{} is not in 'included'".format(var))

        # prove arrays
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
        sets.update(add)

        return None

    # ------------------------------------------------------------------------
    # extending / merging
    # ------------------------------------------------------------------------

    @modify(to_list=['dataset'])
    @verify(variables={'on': 'columns', 'left_on': 'columns'})
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
        ds_right = [(ds._meta, ds._data) for ds in dataset]
        if on is None and right_on in self.columns():
            id_backup = self._data[right_on].copy()
        else:
            id_backup = None
        merged_meta, merged_data = _hmerge(
            ds_left, ds_right, on=on, left_on=left_on, right_on=right_on,
            overwrite_text=overwrite_text, from_set=from_set, verbose=verbose)
        if id_backup is not None:
            merged_data[right_on] = id_backup
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

    @modify(to_list=['dataset'])
    @verify(variables={'on': 'columns', 'left_on': 'columns'})
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
        datasets = [(self._meta, self._data)]
        merge_ds = [(ds._meta, ds._data) for ds in dataset]
        datasets.extend(merge_ds)
        merged_meta, merged_data = _vmerge(
            None, None, datasets, on=on, left_on=left_on,
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

    @verify(variables={'id_key_name': 'columns', 'multiplier': 'columns'})
    def _make_unique_key(self, id_key_name, multiplier):
        """
        """
        columns = self._meta['columns']
        if columns[id_key_name]['type'] not in ['int', 'float']:
            raise TypeError("'id_key_name' must be of type int, float, single!")
        elif columns[multiplier]['type'] not in ['single', 'int', 'float']:
            raise TypeError("'multiplier' must be of type int, float, single!")
        org_key_col = self._data.copy()[id_key_name]
        new_name = 'original_{}'.format(id_key_name)
        name, qtype, lab = new_name, 'int', 'Original ID'
        self.add_meta(name, qtype, lab)
        self[new_name] = org_key_col
        self[id_key_name] += self[multiplier].astype(int) * 100000000
        return None

    @modify(to_list='dataset')
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
        for ds in dataset:
            empty_data = ds._data.copy()
            ds._data = ds._data[ds._data.index < 0]
        self.vmerge(dataset, verbose=False)
        return None

    # ------------------------------------------------------------------------
    # meta data editing
    # ------------------------------------------------------------------------
    @verify(text_keys='text_key')
    def add_meta(self, name, qtype, label, categories=None, items=None,
        text_key=None, replace=True):
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
        categories : list of str, int, or tuples in form of (int, str), default None
            When a list of str is given, the categorical values will simply be
            enumerated and mapped to the category labels. If only int are
            provided, text labels are assumed to be an empty str ('') and a
            warning is triggered. Alternatively, codes can be mapped to categorical
            labels, e.g.: ``[(1, 'Elephant'), (2, 'Mouse'), (999, 'No animal')]``
        items : list of str, int, or tuples in form of (int, str), default None
            If provided will automatically create an array type mask.
            When a list of str is given, the item number will simply be
            enumerated and mapped to the category labels. If only int are
            provided, item text labels are assumed to be an empty str ('') and
            a warning is triggered. Alternatively, numerical values can be
            mapped explicitly to items labels, e.g.:
            ``[(1 'The first item'), (2, 'The second item'), (99, 'Last item')]``
        text_key : str, default None
            Text key for text-based label information. Uses the
            ``DataSet.text_key`` information if not provided.
        replace : bool, default True
            If True, an already existing corresponding ``pd.DataFrame``
            column in the case data component will be overwritten with a
            new (empty) one.

        Returns
        -------
        None
            ``DataSet`` is modified inplace, meta data and ``_data`` columns
            will be added
        """
        make_array_mask = True if items else False
        test_name = name
        self._verify_variable_meta_not_exist(test_name, make_array_mask)
        if not text_key: text_key = self.text_key
        if make_array_mask:
            self._add_array(name, qtype, label, items, categories, text_key)
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
        new_meta = {'text': {text_key: label},
                    'type': qtype,
                    'name': name,
                    'parent': {},
                    'properties': {'created': True}}
        if categories:
            new_meta['values'] = self._make_values_list(categories, text_key)
        self._meta['columns'][name] = new_meta
        datafile_setname = 'columns@{}'.format(name)
        if datafile_setname not in self._meta['sets']['data file']['items']:
            self._meta['sets']['data file']['items'].append(datafile_setname)
        if replace:
            self._data[name] = '' if qtype == 'delimited set' else np.NaN
        return None

    def _check_and_update_element_def(self, element_def):
        all_int = all(isinstance(v, int) for v in element_def)
        all_str = all(isinstance(v, (str, unicode)) for v in element_def)
        all_tuple = all(isinstance(v, tuple) for v in element_def)
        if not (all_int or all_str or all_tuple):
            err = ("The provided value or item element defintion is invalid:\n{}\n"
                   "Please provide either a list of int, a list of str or a "
                   "list of tuple!")
            raise TypeError(err.format(element_def))
        if all_int:
            if self._verbose_infos:
                warn_msg = ("'text' label information missing, only numerical "
                            "codes created for the element object. Remember to "
                            "add value 'text' metadata manually!")
                warnings.warn(warn_msg)
            element_def = [(c, '') for c in element_def]
        return element_def

    def _make_values_list(self, categories, text_key, start_at=None):
        categories = self._check_and_update_element_def(categories)
        if not start_at:
            start_at = 1
        if not all([isinstance(cat, tuple) for cat in categories]):
            vals = [self._value(no, text_key, lab) for no, lab in
                    enumerate(categories, start_at)]
        else:
            vals = [self._value(cat[0], text_key, cat[1]) for cat in categories]
        return vals

    @verify(variables={'name': 'columns'})
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

    @modify(to_list='codes')
    @verify(variables={'name': 'masks'}, text_keys='text_key')
    def flatten(self, name, codes, new_name=None, text_key=None):
        """
        Create a variable that groups array mask item answers to categories.

        Parameters
        ----------
        name : str
            The array variable name keyed in ``meta['masks']`` that will
            be converted.
        codes : int, list of int
            The answers codes that determine the categorical grouping.
            Item labels will become the category labels.
        new_name : str, default None
            The name of the new delimited set variable. If None, ``name`` is
            suffixed with '_rec'.
        text_key : str, default None
            Text key for text-based label information. Uses the
            ``DataSet.text_key`` information if not provided.
        Returns
        -------
        None
            The DataSet is modified inplace, delimited set variable is added.
        """
        if not new_name:
            if '.' in name:
                new_name = '{}_rec'.format(name.split('.')[0])
            else:
                new_name = '{}_rec'.format(name)
        if not text_key: text_key = self.text_key
        label = self._meta['masks'][name]['text'][text_key]
        cats = self.item_texts(name)
        self.add_meta(new_name, 'delimited set', label, cats)
        for x, source in enumerate(self.sources(name), 1):
            self.recode(new_name, {x: {source: codes}}, append=True)
        return None

    @verify(variables={'name': 'columns'})
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
            self._as_int(name)
        elif to == 'float':
            self._as_float(name)
        elif to == 'single':
            self._as_single(name)
        elif to == 'delimited set':
            self._as_delimited_set(name)
        elif to == 'string':
            self._as_string(name)
        if self._is_array_item(name):
            self._meta['masks'][self._maskname_from_item(name)]['subtype'] = to
        return None

    def _as_float(self, name):
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
        org_type = self._get_type(name)
        if org_type == 'float': return None
        valid = ['single', 'int']
        if not org_type in valid:
            msg = 'Cannot convert variable {} of type {} to float!'
            raise TypeError(msg.format(name, org_type))
        if org_type == 'single':
            self._as_int(name)
        if org_type == 'int':
            self._meta['columns'][name]['type'] = 'float'
            self._data[name] = self._data[name].apply(
                    lambda x: float(x) if not np.isnan(x) else np.NaN)
        return None

    def _as_int(self, name):
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
        org_type = self._get_type(name)
        if org_type == 'int': return None
        valid = ['single']
        if not org_type in valid:
            msg = 'Cannot convert variable {} of type {} to int!'
            raise TypeError(msg.format(name, org_type))
        self._meta['columns'][name]['type'] = 'int'
        self._meta['columns'][name].pop('values')
        return None

    def _as_delimited_set(self, name):
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

    def _as_single(self, name):
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

    def _as_string(self, name):
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

    @verify(variables={'name': 'both'})
    def rename(self, name, new_name):
        """
        Change meta and data column name references of the variable defintion.

        Parameters
        ----------
        name : str
            The originating column variable name keyed in ``meta['columns']``
            or ``meta['masks']``.
        new_name : str
            The new variable name.

        Returns
        -------
        None
            DataSet is modified inplace. The new name reference is placed into
            both the data and meta component.
        """
        renames = {}
        if new_name in self._data.columns:
            msg = "Cannot rename '{}' into '{}'. Column name already exists!"
            raise ValueError(msg.format(name, new_name))

        self.undimensionize([name] + self.sources(name))

        if self._dimensions_comp:
            name = name.split('.')[0]
        for s in self.sources(name):
            new_s_name = '{}_{}'.format(new_name, s.split('_')[-1])
            self._add_all_renames_to_mapper(renames, s, new_s_name)

        self._add_all_renames_to_mapper(renames, name, new_name)
        self.rename_from_mapper(renames)
        if self._dimensions_comp: self.dimensionize(new_name)
        return None

    def rename_from_mapper(self, mapper, keep_original=False):
        """
        Rename meta objects and data columns using mapper.

        Parameters
        ----------
        mapper : dict
            A renaming mapper in the form of a dict of {old: new} that
            will be used to rename columns throughout the meta and data.


        Returns
        -------
        None
            DataSet is modified inplace.
        """

        def rename_meta(meta, mapper):
            """
            Rename lib@values, masks, set items and columns using mapper.
            """
            rename_lib_values(meta['lib']['values'], mapper)
            rename_masks(meta['masks'], mapper, keep_original)
            rename_columns(meta['columns'], mapper, keep_original)
            rename_sets(meta['sets'], mapper, keep_original)
            if not keep_original:
                rename_set_items(meta['sets'], mapper)

        def rename_lib_values(lib_values, mapper):
            """
            Rename lib@values objects using mapper.
            """
            for name, rename in mapper.iteritems():
                if name in lib_values:
                    lib_values[rename] = org_copy.deepcopy(lib_values[name])
                    if not keep_original: del lib_values[name]

        def rename_masks(masks, mapper, keep_original):
            """
            Rename mask objects using mapper.
            """
            for name, rename in mapper.iteritems():
                if name in masks:
                    masks[rename] = org_copy.deepcopy(masks[name])
                    if not keep_original: del masks[name]
                    masks[rename]['name'] = rename

                    if masks[rename].get('values'):
                        values = masks[rename]['values']
                        if isinstance(values, (str, unicode)):
                            if values in mapper:
                                masks[rename]['values'] = mapper[values]

                    items = masks[rename]['items']
                    for i, item in enumerate(items):
                        for key in ['source', 'values']:
                            if item.get(key):
                                if item[key] in mapper:
                                    items[i][key] = mapper[item[key]]

        def rename_columns(columns, mapper, keep_original):
            """
            Rename column objects using mapper.
            """
            for name, rename in mapper.iteritems():
                if name in columns:
                    columns[rename] = org_copy.deepcopy(columns[name])
                    if 'parent' in columns[name]:
                        parents = columns[name]['parent']
                    else:
                        parents = {}
                    if not keep_original: del columns[name]
                    columns[rename]['name'] = rename
                    for parent_name, parent_spec in parents.items():
                        new_parent_map = {}
                        if parent_name in mapper:
                            new_name = mapper[parent_name]
                            new_parent_map[new_name] = parent_spec
                            columns[rename]['parent'] = new_parent_map
                    if columns[rename].get('values'):
                        values = columns[rename]['values']
                        if isinstance(values, (str, unicode)):
                            if values in mapper:
                                columns[rename]['values'] = mapper[values]

        def rename_sets(sets, mapper, keep_original):
            """
            Rename set object items using mapper.
            """
            for name, rename in mapper.iteritems():
                if name in sets:
                    sets[rename] = org_copy.deepcopy(sets[name])
                    if not keep_original: del sets[name]
                    sets[rename]['name'] = rename
                    # copied from 'rename_set_items'
                    items = sets[rename].get('items', False)
                    if items:
                        for i, item in enumerate(items):
                            if item in mapper:
                                items[i] = mapper[item]

        def rename_set_items(sets, mapper):
            """
            Rename standard set object items using mapper.
            """
            for set_name in sets.keys():
                try:
                    items = sets[set_name].get('items', False)
                    if items:
                        for i, item in enumerate(items):
                            if item in mapper:
                                items[i] = mapper[item]
                except (AttributeError, KeyError, TypeError, ValueError):
                    pass

        rename_meta(self._meta, mapper)
        if not keep_original: self._data.rename(columns=mapper, inplace=True)

    def dimensionizing_mapper(self, names=None):
        """
        Return a renaming dataset mapper for dimensionizing names.

        Parameters
        ----------
        None


        Returns
        -------
        mapper : dict
            A renaming mapper in the form of a dict of {old: new} that
            maps non-Dimensions naming conventions to Dimensions naming
            conventions.
        """
        masks = self._meta['masks']
        columns = self._meta['columns']

        mapper = {}
        if not names:
            names = masks.keys()
        for mask_name, mask in masks.iteritems():
            if mask_name in names:
                new_mask_name = '{mn}.{mn}_grid'.format(mn=mask_name)
                mapper[mask_name] = new_mask_name

                mask_mapper = 'masks@{mn}'.format(mn=mask_name)
                new_mask_mapper = 'masks@{nmn}'.format(nmn=new_mask_name)
                mapper[mask_mapper] = new_mask_mapper

                values_mapper = 'lib@values@{mn}'.format(mn=mask_name)
                new_values_mapper = 'lib@values@{nmn}'.format(nmn=new_mask_name)
                mapper[values_mapper] = new_values_mapper

                items = masks[mask_name]['items']
                for i, item in enumerate(items):
                    col_name = item['source'].split('@')[-1]
                    new_col_name = '{mn}[{{{cn}}}].{mn}_grid'.format(
                        mn=mask_name, cn=col_name
                    )
                    mapper[col_name] = new_col_name

                    col_mapper = 'columns@{cn}'.format(cn=col_name)
                    new_col_mapper = 'columns@{ncn}'.format(ncn=new_col_name)
                    mapper[col_mapper] = new_col_mapper

        return mapper

    def undimensionizing_mapper(self, names=None):
        """
        Return a renaming dataset mapper for un-dimensionizing names.

        Parameters
        ----------
        None


        Returns
        -------
        mapper : dict
            A renaming mapper in the form of a dict of {old: new} that
            maps Dimensions naming conventions to non-Dimensions naming
            conventions.
        """

        masks = self._meta['masks']
        columns = self._meta['columns']

        mask_pattern = '(^.+)\..+$'
        column_pattern = '(?<=\[{)(.*?)(?=}\])'

        mapper = {}
        if not names:
            names = masks.keys() + columns.keys()
        for mask_name in masks.keys():
            if mask_name in names:
                matches = re.findall(mask_pattern, mask_name)
                if matches:
                    new_mask_name = matches[0]
                    mapper[mask_name] = new_mask_name

                    mask_mapper = 'masks@{mn}'.format(mn=mask_name)
                    new_mask_mapper = 'masks@{nmn}'.format(nmn=new_mask_name)
                    mapper[mask_mapper] = new_mask_mapper

                    values_mapper = 'lib@values@{mn}'.format(mn=mask_name)
                    new_values_mapper = 'lib@values@{nmn}'.format(nmn=new_mask_name)
                    mapper[values_mapper] = new_values_mapper

        for col_name in columns.keys():
            if col_name in names:
                matches = re.findall(column_pattern, col_name)
                if matches:
                    new_col_name = matches[0]
                    mapper[col_name] = new_col_name
                    col_mapper = 'columns@{mn}'.format(mn=col_name)
                    new_col_mapper = 'columns@{nmn}'.format(nmn=new_col_name)
                    mapper[col_mapper] = new_col_mapper
        return mapper

    @modify(to_list='names')
    @verify(variables={'names': 'both'})
    def dimensionize(self, names=None):
        """
        Rename the dataset columns for Dimensions compatibility.
        """
        mapper = self.dimensionizing_mapper(names)
        self.rename_from_mapper(mapper)

    @modify(to_list='names')
    @verify(variables={'names': 'both'})
    def undimensionize(self, names=None, mapper_to_meta=False):
        """
        Rename the dataset columns to remove Dimensions compatibility.
        """
        mapper = self.undimensionizing_mapper(names)
        self.rename_from_mapper(mapper)
        if mapper_to_meta: self._meta['sets']['rename_mapper'] = mapper

    @verify(variables={'name': 'both'})
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

    @modify(to_list='name')
    @verify(variables={'name': 'both'})
    def drop(self, name, ignore_items=False):
        """
        Drops variables from meta and data components of the ``DataSet``.

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
        for var in name:
            if self._is_array(var):
                if not ignore_items:
                    name += self.sources(var)
                else:
                    df_items = meta['sets']['data file']['items']
                    ind = df_items.index('masks@{}'.format(var))
                    n_items = df_items[:ind] + self._get_source_ref(var) + df_items[ind+1:]
                    meta['sets']['data file']['items'] = n_items
                    values = meta['lib']['values'][var]
                    for source in self.sources(var):
                        meta['columns'][source]['values'] = values
                        meta['columns'][source]['parent'] = {}

        df_items = meta['sets']['data file']['items']
        n_items = [i for i in df_items if not i.split('@')[-1] in name]
        meta['sets']['data file']['items'] = n_items
        data_drop = []
        for var in name:
            if not self._is_array(var): data_drop.append(var)
            remove_loop(meta, var)
        data.drop(data_drop, 1, inplace=True)
        return None

    @modify(to_list='remove')
    @verify(variables={'name': 'both'})
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
        # Do we need to modify a mask's lib def.?
        if not self._is_array(name) and self._is_array_item(name):
            name = self._maskname_from_item(name)
        # Are any meta undefined codes provided? - Warn user!
        values = self._get_value_loc(name)
        codes = self.codes(name)
        ignore_codes = [r for r in remove if r not in codes]
        if ignore_codes:
            print 'Warning: Cannot remove values...'
            print '*' * 60
            msg = "Codes {} not found in values object of '{}'!"
            print msg.format(ignore_codes, name)
            print '*' * 60
            remove = [x for x in remove if x not in ignore_codes]
        # Would be remove all defined values? - Prevent user from doing this!
        new_values = [value for value in values
                      if value['value'] not in remove]
        if not new_values:
            msg = "Cannot remove all codes from the value object of '{}'!"
            raise ValueError(msg.format(name))
        # Apply new ``values`` definition
        if self._is_array(name):
            self._meta['lib']['values'][name] = new_values
        else:
            self._meta['columns'][name]['values'] = new_values
        # Remove values in ``data``
        if self._is_array(name):
            items = self._get_itemmap(name, 'items')
            for i in items:
                self.uncode(i, {x: {i: x} for x in remove})
                self._verify_data_vs_meta_codes(i)
        else:
            self.uncode(name, {x: {name: x} for x in remove})
            self._verify_data_vs_meta_codes(name)
        return None


    # @modify(to_list='unite')
    # @verify(variables={'name': 'both'}, categorical='name', text_keys='text_key')
    # def unite_values(self, name, unite, reindex_codes=False, text_key=None):
    #     """
    #     Collapse codes into a new unifying category.

    #     Parameters
    #     ----------
    #     name : str
    #         The column variable name keyed in ``_meta['columns']`` or
    #         ``_meta['masks']``.
    #     unite : (list of) tuple (label, list of codes) or list of codes
    #         If only list(s) of codes are provided, the new value ``text`` label
    #         will be a '//'-delimited concatenation of the originating values'
    #         texts.
    #     reindex_code : bool, default False
    #         If True, the ``values`` object's codes will be re-enumerated from
    #         1. By default, the new value will take the ``unite`` list(s)
    #         starting code.
    #     text_key : str, default None
    #         Text key for text-based label information. Will automatically fall
    #         back to the instance's text_key property information if not provided.

    #     Returns
    #     -------
    #     None
    #         DataSet is modified inplace.
    #     """
    #     if not text_key: text_key = self.text_key
    #     if isinstance(unite[0], int): unite = [unite]
    #     prep_unite = []
    #     for udef in unite:
    #         if not isinstance(udef, (list, tuple)):
    #             type_err = ("Items in 'unite' must either be lists of codes "
    #                         "or tuples of (text label, list of codes)!")
    #             raise TypeError(type_err)
    #         if isinstance(udef, tuple):
    #             if not isinstance(udef[0], (str, unicode)):
    #                 type_err = ("First tuple element must be value meta 'text' "
    #                             "label, not {}!".format(type(udef[0])))
    #             if not isinstance(udef[1], (list)):
    #                 type_err = ("Second tuple element must be list of value "
    #                             "codes, not {}!".format(type(udef[1])))
    #             prep_unite.append(udef)
    #         else:
    #             values = self.values(name, text_key=text_key)
    #             unite_texts = [label for code, label in values if code in udef]
    #             new_text = '//'.join(unite_texts)
    #             prep_unite.append((new_text, udef))
    #     all_codes = chain.from_iterable(udef[1] for udef in prep_unite)
    #     if len(set(all_codes)) != len(all_codes):
    #         val_err = ("Codes must be mutually exclusive in the 'unite' list. "
    #                    "Cannot unify with duplicate codes.")
    #         raise TypeError(val_err)






    @modify(to_list='remove')
    @verify(variables={'name': 'masks'})
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
            if col_ref in self._meta['sets']['data file']['items']:
                self._meta['sets']['data file']['items'].remove(col_ref)
            self._meta['sets'][name]['items'].remove(col_ref)
        return None

    @modify(to_list='ext_values')
    @verify(variables={'name': 'both'}, categorical='name', text_keys='text_key')
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
        # Do we need to modify a mask's lib def.?
        if not self._is_array(name) and self._is_array_item(name):
            name = self._maskname_from_item(name)
        use_array = self._is_array(name)
        if not text_key: text_key = self.text_key
        value_obj = self._get_valuemap(name, text_key=text_key)
        codes = self.codes(name)
        texts = self.value_texts(name)
        if not isinstance(ext_values[0], tuple):
            start_here = self._highest_code(codes) + 1
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
        if use_array:
            self._meta['lib']['values'][name].extend(ext_values)
        else:
            self._meta['columns'][name]['values'].extend(ext_values)
        return None

    @verify(text_keys='text_key')
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
        self.text_key = text_key
        self._meta['lib']['default text'] = text_key
        return None

    @modify(to_list='new_tk')
    def extend_valid_tks(self, new_tk):
        for tk in new_tk:
            if not tk in self.valid_tks:
                self.valid_tks.append(tk)
        self._meta['lib']['valid text'] = self.valid_tks
        return None

    @verify(variables={'name': 'both'}, text_keys='text_key')
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

    @staticmethod
    def _force_texts(text_dict, copy_to, copy_from, update_existing):
        new_text_key = None
        for new_tk in reversed(copy_from):
            if new_tk in text_dict.keys():
                new_text_key = new_tk
        if not new_text_key:
            raise ValueError('{} is no existing text_key'.format(copy_from))
        if not copy_to in text_dict.keys() or update_existing:
            text_dict.update({copy_to: text_dict[new_text_key]})

    @modify(to_list='copy_from')
    @verify(text_keys=['copy_to', 'copy_from'])
    def force_texts(self, copy_to=None, copy_from=None, update_existing=False):
        """
        Copy info from existing text_key to a new one or update the existing one.

        Parameters
        ----------
        copy_to : str
            {'en-GB', 'da-DK', 'fi-FI', 'nb-NO', 'sv-SE', 'de-DE'}
            None -> _meta['lib']['default text']
            The text key that will be filled.
        copy_from : str / list
            {'en-GB', 'da-DK', 'fi-FI', 'nb-NO', 'sv-SE', 'de-DE'}
            You can also enter a list with text_keys, if the first text_key
            doesn't exist, it takes the next one
        update_existing : bool
            True : copy_to will be filled in any case
            False: copy_to will be filled if it's empty/not existing

        Returns
        -------
        None
        """
        if copy_to is None:
            copy_to = self.text_key
        elif not isinstance(copy_to, str):
            raise ValueError('`copy_to` must be a str.')
        copy_from.append(copy_to)

        text_func = self._force_texts
        args = ()
        kwargs = {'copy_to': copy_to,
                  'copy_from': copy_from,
                  'update_existing': update_existing}
        DataSet._apply_to_texts(text_func, self._meta, args, kwargs)
        return None

    @staticmethod
    def _remove_html(text_dict):
        htmls = ['_', '**', '*']
        for tk, text in text_dict.items():
            if not tk in ['x edits', 'y edits']:
                for html in htmls:
                    text = text.replace(html, '')
                remove = re.compile('<.*?>')
                text = re.sub(remove, '', text)
                remove = '(<|\$)(.|\n)+?(>|.raw |.raw)'
                text_dict[tk] = re.sub(remove, '', text)
            else:
                for etk, etext in text_dict[tk].items():
                    for html in htmls:
                        etext = etext.replace(html, '')
                    remove = re.compile('<.*?>')
                    etext = re.sub(remove, '', etext)
                    remove = '(<|\$)(.|\n)+?(>|.raw |.raw)'
                    text_dict[tk][etk] = re.sub(remove, '', etext)

    def remove_html(self):
        """
        Cycle through all meta ``text`` objects removing html tags.

        Currently uses the regular expression '<.*?>' in _remove_html()
        classmethod.

        Returns
        -------
        None
        """
        text_func = self._remove_html
        args = ()
        kwargs = {}
        DataSet._apply_to_texts(text_func, self._meta, args, kwargs)
        return None

    @staticmethod
    def _replace_from_dict(text_dict, replace_map, text_key):
        for tk, text in text_dict.items():
            if tk in text_key:
                for k, v in replace_map.items():
                    text_dict[tk] = text_dict[tk].replace(k, v)
            elif tk in ['x edits', 'y edits']:
                for etk, etext in text_dict[tk].items():
                    if etk in text_key:
                        for k, v in replace_map.items():
                            text_dict[tk][etk] = text_dict[tk][etk].replace(k, v)

    @modify(to_list='text_key')
    @verify(text_keys='text_key')
    def replace_texts(self, replace, text_key=None):
        """
        Cycle through all meta ``text`` objects replacing unwanted strings.

        Parameters
        ----------
        replace : dict, default Nonea
            A dictionary mapping {unwanted string: replacement string}.
        text_key : str / list of str, default None
            {None, 'en-GB', 'da-DK', 'fi-FI', 'nb-NO', 'sv-SE', 'de-DE'}
            The text_keys for which unwanted strings are replaced.
        Returns
        -------
        None
        """
        if not text_key: text_key = self.valid_tks
        text_func = self._replace_from_dict
        args = ()
        kwargs = {'replace_map': replace,
                  'text_key': text_key}
        DataSet._apply_to_texts(text_func, self._meta, args, kwargs)
        return None

    @staticmethod
    def _convert_edits(text_dict, text_key):
        edits = ['x edits', 'y edits']
        for edit in edits:
            if text_dict.get(edit, {}).get(text_key):
                text_dict[edit] = text_dict[edit][text_key]
            elif edit in text_dict:
                text_dict.pop(edit)

    @staticmethod
    def _convert_text_edits(meta_dict, text_key):
        """
        Take a defined text_key text as edits text for all text objects.

        Parameters
        ----------
        text_key : str
            The text_key that is set to the edits.

        Returns
        -------
        None
        """
        text_func = DataSet._convert_edits
        args = ()
        kwargs = {'text_key': text_key}
        DataSet._apply_to_texts(text_func, meta_dict, args, kwargs)
        return None

    @staticmethod
    def _repair_text_edits(text_dict, text_key):
        for ax in ['x edits', 'y edits']:
            if not isinstance(text_dict.get(ax, {}), dict):
                text_dict[ax] = {tk: text_dict[ax]
                                 for tk in text_dict.keys() if tk in text_key}

    @verify(text_keys='text_key')
    def repair_text_edits(self, text_key=None):
        """
        Cycle through all meta ``text`` objects repairing axis edits.

        Parameters
        ----------
        text_key : str / list of str, default None
            {None, 'en-GB', 'da-DK', 'fi-FI', 'nb-NO', 'sv-SE', 'de-DE'}
            The text_keys for which text edits should be included.
        Returns
        -------
        None
        """
        if text_key is None: text_key = self.valid_tks
        text_func = self._repair_text_edits
        args = ()
        kwargs = {'text_key': text_key}
        DataSet._apply_to_texts(text_func, self._meta, args, kwargs)
        return None


    @staticmethod
    def _apply_to_texts(text_func, meta_dict, args, kwargs):
        """
        Cycle through all ``text`` objects editing them via the passed function.
        """
        if isinstance(meta_dict, dict):
            for key in meta_dict.keys():
                if key in ['sets', 'ddf']:
                    pass
                elif key == 'text' and isinstance(meta_dict[key], dict):
                    text_func(meta_dict[key], *args, **kwargs)
                else:
                    DataSet._apply_to_texts(text_func, meta_dict[key], args, kwargs)

        elif isinstance(meta_dict, list):
            for item in meta_dict:
                DataSet._apply_to_texts(text_func, item, args, kwargs)

    @modify(to_list='arrays')
    @verify(variables={'arrays': 'masks'})
    def cut_item_texts(self, arrays=None):
        """
        Remove array text from array item texts.

        Parameters
        ----------
        arrays : str, list of str, default None
            Cut texts for items of these arrays. If None, all keys in
            ``._meta['masks']`` are taken.
        """
        if not arrays: arrays = self.masks()
        for a in arrays:
            for item in self.sources(a):
                i = self._meta['columns'][item]
                for tk in self.valid_tks:
                    text = self.text(item, True, tk)
                    if text: i['text'][tk] = text
                for ed in ['x', 'y']:
                    if i['text'].get('{} edits'.format(ed)):
                        for tk in self.valid_tks:
                            text = self.text(item, True, tk, ed)
                            if text: i['text']['{} edits'.format(ed)][tk] = text
        return None

    @modify(to_list=['text_key', 'axis_edit'])
    @verify(variables={'name': 'both'}, text_keys='text_key', axis='axis_edit')
    def set_variable_text(self, name, new_text, text_key=None, axis_edit=None):
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
        axis_edit: {'x', 'y', ['x', 'y']}, default None
            If the ``new_text`` of the variable should only be considered temp.
            for build exports, the axes on that the edited text should appear
            can be provided.

        Returns
        -------
        None
            The ``DataSet`` is modified inplace.
        """
        def _get_text(name, shorten, text_key, axis_edit):
            text = self.text(name, shorten, text_key, axis_edit)
            if text is None:
                text = self.text(name, shorten, text_key, None)
            if text is None:
                text = self.text(name, shorten, self.text_key, None)
            return text

        collection = 'masks' if self._is_array(name) else 'columns'
        textobj = self._meta[collection][name]['text']
        if not text_key and not axis_edit:
            text_key = [self.text_key]
        elif not text_key and axis_edit:
            text_key = [tk for tk in textobj.keys()
                        if tk not in ['x edits', 'y edits']]

        if self._is_array_item(name):
            parent = self._maskname_from_item(name)
            p_obj = self._meta['masks'][parent]['text']

            for tk in text_key:
                if axis_edit:
                    if not tk in p_obj: continue
                    for ax in axis_edit:
                        p_text = _get_text(parent, False, tk, ax)
                        a_edit = '{} edits'.format(ax)
                        if not a_edit in p_obj: p_obj[a_edit] = {}
                        p_obj[a_edit].update({tk: p_text})
                else:
                    if not tk in p_obj:
                        p_text = _get_text(parent, False, tk, None)
                        p_obj.update({tk: p_text})
            n_items = []
            for item in self._meta['masks'][parent]['items']:
                if name in item['source']:
                    i_textobj = item['text']
                    for tk in text_key:
                        if axis_edit:
                            for ax in axis_edit:
                                if not tk in i_textobj: continue
                                a_edit = '{} edits'.format(ax)
                                if not a_edit in i_textobj: i_textobj[a_edit] = {}
                                i_textobj[a_edit].update({tk: new_text})
                        else:
                            i_textobj.update({tk: new_text})
                n_items.append(item)
            self._meta['masks'][parent]['items'] = n_items

        for tk in text_key:
            if axis_edit:
                if not tk in textobj: continue
                for ax in axis_edit:
                    a_edit = '{} edits'.format(ax)
                    if not a_edit in textobj: textobj[a_edit] = {}
                    if self._is_array_item(name):
                        p_text = self.text(parent, False, tk, ax)
                        n_text = '{} - {}'.format(p_text, new_text)
                    else:
                        n_text = new_text
                    textobj[a_edit].update({tk: n_text})
            else:
                if self._is_array_item(name):
                    p_text = self.text(parent, False, tk, None)
                    n_text = '{} - {}'.format(p_text, new_text)
                else:
                    n_text = new_text
                textobj.update({tk: n_text})

        if collection == 'masks':
            for s in self.sources(name):
                for tk in text_key:
                    if axis_edit:
                        for ax in axis_edit:
                            item_text = _get_text(s, True, tk, ax)
                            self.set_variable_text(s, item_text, tk, ax)
                    else:
                        item_text = _get_text(s, True, tk, None)
                        self.set_variable_text(s, item_text, tk)
        return None

    @modify(to_list=['text_key', 'axis_edit'])
    @verify(variables={'name': 'both'}, categorical='name', text_keys='text_key', axis='axis_edit')
    def set_value_texts(self, name, renamed_vals, text_key=None, axis_edit=None):
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
        axis_edit: {'x', 'y', ['x', 'y']}, default None
            If ``renamed_vals`` should only be considered temp. for build
            exports, the axes on that the edited text should appear can be
            provided.

        Returns
        -------
        None
            The ``DataSet`` is modified inplace.
        """
        # Do we need to modify a mask's lib def.?
        if not self._is_array(name) and self._is_array_item(name):
            name = self._maskname_from_item(name)
        use_array = self._is_array(name)

        valuesobj = self._get_value_loc(name)
        new_valuesobj = []

        if not text_key:
            if not axis_edit:
                text_key = [self.text_key]
            else:
                text_key = valuesobj[0]['text'].keys()

        ignore = [k for k in renamed_vals.keys() if k not in self.codes(name)]

        if ignore:
            print 'Warning: Cannot set new value texts...'
            print '*' * 60
            msg = "Codes {} not found in values object of '{}'!"
            print msg.format(ignore, name)
            print '*' * 60

        text_key = [tk for tk in text_key if tk not in ['x edits', 'y edits']]
        for value in valuesobj:
            val = value['value']
            if val in renamed_vals.keys():
                value_texts = value['text']
                for tk in text_key:
                    if axis_edit:
                        for ax in axis_edit:
                            edit_key = 'x edits' if ax == 'x' else 'y edits'
                            if not edit_key in value_texts: value_texts[edit_key] = {}
                            if tk in value_texts:
                                value_texts[edit_key][tk] = renamed_vals[val]
                    else:
                        if tk in value_texts.keys():
                            value['text'][tk] = renamed_vals[val]
                        else:
                            value['text'].update({tk: renamed_vals[val]})
            new_valuesobj.append(value)
        if not use_array:
            self._meta['columns'][name]['values'] = new_valuesobj
        else:
            self._meta['lib']['values'][name] = new_valuesobj
        return None

    @modify(to_list=['text_key', 'axis_edit'])
    @verify(variables={'name': 'masks'}, text_keys='text_key', axis='axis_edit')
    def set_item_texts(self, name, renamed_items, text_key=None, axis_edit=None):
        """
        Rename or add item texts in the ``items`` objects of ``masks``.

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
        axis_edit: {'x', 'y', ['x', 'y']}, default None
            If the ``new_text`` of the variable should only be considered temp.
            for build exports, the axes on that the edited text should appear
            can be provided.

        Returns
        -------
        None
            The ``DataSet`` is modified inplace.
        """
        for item_no, item_text in renamed_items.items():
            source = self.sources(name)[item_no - 1]
            self.set_variable_text(source, item_text, text_key, axis_edit)
        return None

    @verify(variables={'name': 'both'})
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
            raise ValueError("'prop_name' must be one of {}".format(valid_props))
        prop_update = {prop_name: prop_value}
        if self._is_array(name):
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

    @modify(to_list='name')
    @verify(variables={'name': 'columns'}, axis='axis')
    def slicing(self, name, slicer, axis='y'):
        """
        Set or update ``rules[axis]['slicex']`` meta for the named column.

        Quantipy builds will respect the kept codes and *show them exclusively*
        in results.

        .. note:: This is not a replacement for ``DataSet.set_missings()`` as
            missing values are respected also in computations.

        Parameters
        ----------
        name : str or list of str
            The column variable(s) name keyed in ``_meta['columns']``.
        slice : int or list of int
            Values indicated by their ``int`` codes will be shown in
            ``Quantipy.View.dataframe``s, respecting the provided order.
        axis : {'x', 'y'}, default 'y'
            The axis to slice the values on.

        Returns
        -------
        None
        """
        for n in name:
            if self._is_array_item(n):
                raise ValueError('Cannot slice on array items.')
            if 'rules' not in self._meta['columns'][n]:
                self._meta['columns'][n]['rules'] = {'x': {}, 'y': {}}
            if not isinstance(slicer, list): slicer = [slicer]
            slicer = self._clean_codes_against_meta(n, slicer)
            rule_update = {'slicex': {'values': slicer}}
            for ax in axis:
                self._meta['columns'][n]['rules'][ax].update(rule_update)
        return None

    @modify(to_list='name')
    @verify(variables={'name': 'both'}, axis='axis')
    def hiding(self, name, hide, axis='y', hide_values=True):
        """
        Set or update ``rules[axis]['dropx']`` meta for the named column.

        Quantipy builds will respect the hidden codes and *cut* them from
        results.

        .. note:: This is not equivalent to ``DataSet.set_missings()`` as
            missing values are respected also in computations.

        Parameters
        ----------
        name : str or list of str
            The column variable(s) name keyed in ``_meta['columns']``.
        hide : int or list of int
            Values indicated by their ``int`` codes will be dropped from
            ``Quantipy.View.dataframe``s.
        axis : {'x', 'y'}, default 'y'
            The axis to drop the values from.
        hide_values : bool, default True
            Only considered if ``name`` refers to a mask. If True, values are
            hidden on all mask items. If False, mask items are hidden by position
            (only for array summaries).

        Returns
        -------
        None
        """
        for n in name:
            collection = 'columns' if not self._is_array(n) else 'masks'
            if 'rules' not in self._meta[collection][n]:
                self._meta[collection][n]['rules'] = {'x': {}, 'y': {}}
            if not isinstance(hide, list): hide = [hide]

            if collection == 'masks' and 'y' in axis and not hide_values:
                raise ValueError('Cannot hide mask items on y axis!')
            for ax in axis:
                if collection == 'masks' and ax == 'x' and not hide_values:
                    sources = self.sources(n)
                    hide = [sources[idx-1]
                            for idx, s in enumerate(sources, start=1) if idx in hide]
                else:
                    hide = self._clean_codes_against_meta(n, hide)
                    if set(hide) == set(self._get_valuemap(n, 'codes')):
                        msg = "Cannot hide all values of '{}'' on '{}'-axis"
                        raise ValueError(msg.format(n, ax))
                if collection == 'masks' and ax == 'x' and hide_values:
                    for s in self.sources(n):
                        self.hiding(s, hide, 'x')
                else:
                    rule_update = {'dropx': {'values': hide}}
                    self._meta[collection][n]['rules'][ax].update(rule_update)
        return None

    @modify(to_list='name')
    @verify(variables={'name': 'both'})
    def sorting(self, name, on='@', within=False, between=False, fix=None,
                ascending=False, sort_by_weight=None):
        """
        Set or update ``rules['x']['sortx']`` meta for the named column.

        Parameters
        ----------
        name : str or list of str
            The column variable(s) name keyed in ``_meta['columns']``.
        within : bool, default True
            Applies only to variables that have been aggregated by creating a
            an ``expand`` grouping / overcode-style ``View``:
            If True, will sort frequencies inside each group.
        between : bool, default True
            Applies only to variables that have been aggregated by creating a
            an ``expand`` grouping / overcode-style ``View``:
            If True, will sort group and regular code frequencies with regard
            to each other.
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
        for n in name:
            is_array = self._is_array(n)
            collection = 'masks' if is_array else 'columns'
            if on != '@' and not is_array:
                msg = "Column to sort on can only be changed for array summaries!"
                raise NotImplementedError(msg)
            if on == '@' and is_array:
                for source in self.sources(n):
                    self.sorting(source, fix=fix, within=within,
                                 between=between, ascending=ascending)
            else:
                if 'rules' not in self._meta[collection][n]:
                    self._meta[collection][n]['rules'] = {'x': {}, 'y': {}}
                if fix:
                    if not isinstance(fix, list): fix = [fix]
                else:
                    fix = []
                if not is_array:
                    fix = self._clean_codes_against_meta(n, fix)
                else:
                    fix = self._clean_items_against_meta(n, fix)
                rule_update = {'sortx': {'ascending': ascending,
                                         'within': within,
                                         'between': between,
                                         'fixed': fix,
                                         'sort_on': on,
                                         'with_weight': sort_by_weight}}
                self._meta[collection][n]['rules']['x'].update(rule_update)
        return None

    def _add_array(self, name, qtype, label, items, categories, text_key):
        """
        """
        dims_comp = self._dimensions_comp
        item_objects = []
        array_name = name
        items = self._check_and_update_element_def(items)
        if isinstance(items[0], (str, unicode)):
            items = [(idx, txt) for idx, txt in enumerate(items, start=1)]
        value_ref = 'lib@values@{}'.format(array_name)
        values = None
        for i in items:
            item_no = i[0]
            item_lab = i[1]
            item_name = self._array_item_name(item_no, self._dims_free_arr_name(name))
            item_objects.append(self._item(item_name, text_key, item_lab))
            column_lab = '{} - {}'.format(label, item_lab)
            # add array items to 'columns' meta
            self.add_meta(name=item_name, qtype=qtype, label=column_lab,
                          categories=categories, items=None, text_key=text_key)
            # update the items' values objects
            if not values:
                values = self._meta['columns'][item_name]['values']
            self._meta['columns'][item_name]['values'] = value_ref
             # apply the 'parent' spec meta to the items
            parent_spec = {'masks@{}'.format(name) : {'type': 'array'}}
            self._meta['columns'][item_name]['parent'] = parent_spec
            # remove 'columns'-referencing 'sets' meta
            self._meta['sets']['data file']['items'].remove('columns@{}'.format(item_name))
        # generate the 'masks' meta
        mask_meta = {'items': item_objects, 'type': 'array', 'subtype': qtype,
                     'values': value_ref, 'text': {text_key: label},
                     'name': array_name}
        self._meta['lib']['values'][array_name] = values
        self._meta['masks'][array_name] = mask_meta
        datafile_setname = 'masks@{}'.format(array_name)
        if datafile_setname not in self._meta['sets']['data file']['items']:
            self._meta['sets']['data file']['items'].append(datafile_setname)
        self._meta['sets'][array_name] = {'items': [i['source'] for i in item_objects]}
        if self._dimensions_comp: self.dimensionize(name)
        return None

    def _get_subtype(self, name):
        if not self._is_array(name):
            return None
        else:
            return self._meta['masks'][name]['subtype']

    def _add_to_datafile_items_set(self, name):
        datafile_items = self._meta['sets']['data file']['items']
        if self._is_array(name):
            append_name = 'masks@{}'.format(name)
        else:
            append_name = 'columns@{}'.format(name)
        if not append_name in datafile_items and not self._is_array_item(name):
            datafile_items.append(append_name)
        return None

    def _add_all_renames_to_mapper(self, mapper, old, new):
        mapper['masks@{}'.format(old)] = 'masks@{}'.format(new)
        mapper['columns@{}'.format(old)] = 'columns@{}'.format(new)
        mapper['lib@values@{}'.format(old)] = 'lib@values@{}'.format(new)
        mapper[old] = new
        return mapper

    @classmethod
    def _dims_free_arr_name(cls, arr_name):
        return arr_name.split('.')[0]

    def _dims_compat_arr_name(self, arr_name):
        arr_name = self._dims_free_arr_name(arr_name)
        if self._dimensions_comp:
            return '{}.{}_grid'.format(arr_name, arr_name)
        else:
            return arr_name

    @modify(to_list=['copy_only', 'copy_not'])
    def copy(self, name, suffix='rec', copy_data=True, slicer=None, copy_only=None,
             copy_not=None):
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
        copy_data : bool, default True
            The new variable assumes the ``data`` of the original variable.
        slicer : dict
            If the data is copied it is possible to filter the data with a
            complex logic. Example: slicer = {'q1': not_any([99])}
        copy_only: int or list of int, default None
            If provided, the copied version of the variable will only contain
            (data and) meta for the specified codes.
        copy_not: int or list of int, default None
            If provided, the copied version of the variable will contain
            (data and) meta for the all codes, except of the indicated.

        Returns
        -------
        None
            DataSet is modified inplace, adding a copy to both the data and meta
            component.
        """
        if copy_only and copy_not:
            raise ValueError("Must pass either 'copy_only' or 'copy_not', not both!")
        verify_name = name[0] if isinstance(name, tuple) else name
        is_array = self._is_array(verify_name)
        array_item_copied = isinstance(name, tuple)
        dims_comp = self._dimensions_comp
        meta = self._meta
        if not 'renames' in meta['sets']: meta['sets']['renames'] = {}
        renames = meta['sets']['renames']
        # are we dealing with an recursive array item copy?
        if array_item_copied:
            copy_name = '{}_{}'.format(name[1], suffix)
            name = name[0]
        else:
            copy_name = '{}_{}'.format(self._dims_free_arr_name(name), suffix)
        # force stripped names...
        if not renames:
            self.undimensionize([name] + self.sources(name))
            name = self._dims_free_arr_name(name)
        if is_array:
            # copy meta and create rename mapper for array items
            renames = self._add_all_renames_to_mapper(renames, name, copy_name)
            meta['masks'][copy_name] = org_copy.deepcopy(meta['masks'][name])
            meta['sets'][copy_name] = org_copy.deepcopy(meta['sets'][name])
            self._add_to_datafile_items_set(copy_name)
            for item in self.sources(name):
                item_name_split = item.split('_')
                element_name = '_'.join(item_name_split[:-1])
                element_no = item_name_split[-1]
                new_item_name = '{}_{}_{}'.format(element_name, suffix, element_no)
                self.copy((item, element_name), '{}_{}'.format(suffix, element_no),
                          copy_data, slicer=slicer, copy_only=copy_only)
                renames[item] = new_item_name
        else:
            # copy regular 'columns' meta data
            renames = self._add_all_renames_to_mapper(renames, name, copy_name)
            meta['columns'][copy_name] = org_copy.deepcopy(meta['columns'][name])
            meta['columns'][copy_name]['name'] = copy_name
            self._add_to_datafile_items_set(copy_name)
            # handle the case data copy for columns (incl.slicing)
            if copy_data:
                if slicer:
                    self._data[copy_name] = np.NaN
                    take = self.take(slicer)
                    self[take, copy_name] = self._data[name].copy()
                else:
                    self._data[copy_name] = self._data[name].copy()
            else:
                self._data[copy_name] = np.NaN

        # run the renaming for the copied variable
        self.rename_from_mapper(renames, keep_original=True)
        # set type 'created'
        if is_array:
            for s in self.sources(copy_name):
                if meta['columns'][s].get('properties'):
                    for q_type in ['survey', 'open', 'system', 'merged']:
                        meta['columns'][s]['properties'][q_type] = False
                    meta['columns'][s]['properties']['created'] = True
        elif not self._is_array_item(copy_name):
            if meta['columns'][copy_name].get('properties'):
                for q_type in ['survey', 'open', 'system', 'merged']:
                    meta['columns'][copy_name]['properties'][q_type] = False
                meta['columns'][copy_name]['properties']['created'] = True
        # finished, i.e. not any longer inside a recursive array item copy?
        if is_array:
            finalized = len(self.sources(name)) == len(self.sources(copy_name))
        elif self._is_array_item(name):
            finalized = False
        else:
            finalized = True
        if finalized:
            #reduce the meta/data?
            if copy_not:
                remove = [c for c in self.codes(copy_name) if c in copy_not]
                self.remove_values(copy_name, remove)
            if copy_only:
                remove = [c for c in self.codes(copy_name) if not c in copy_only]
                self.remove_values(copy_name, remove)
            del meta['sets']['renames']
            # restore Dimensions-like names if in compatibility mode
            if self._dimensions_comp:
                self.dimensionize(copy_name)
                self.dimensionize(name)
        return None

    @modify(to_list=['count_only', 'count_not'])
    @verify(variables={'name': 'both'}, categorical='name')
    def code_count(self, name, count_only=None, count_not=None):
        """
        Get the total number of codes/entries found per row.

        .. note:: Will be 0/1 for type ``single`` and range between 0 and the
            number of possible values for type ``delimited set``.

        Parameters
        ----------
        name : str
            The column variable name keyed in ``meta['columns']`` or
            ``meta['masks']``.
        count_only : int or list of int, default None
            Pass a list of codes to restrict counting to.
        count_not : int or list of int, default None
            Pass a list of codes that should no be counted.

        Returns
        -------
        count : pandas.Series
            A series with the results as ints.
        """
        if count_only and count_not:
            raise ValueError("Must pass either 'count_only' or 'count_not', not both!")
        dummy = self.make_dummy(name, partitioned=False)
        if count_not:
            count_only = list(set([c for c in dummy.columns if c not in count_not]))
        if count_only:
            dummy = dummy[count_only]
        count = dummy.sum(axis=1)
        return count


    @verify(variables={'name': 'columns'})
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
        return self._data[name].isnull()

    @modify(to_list='codes')
    @verify(variables={'name': 'both'})
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
        if self._is_array(name):
            logics = []
            for s in self.sources(name):
                logics.append({s: has_any(codes)})
            slicer = self.take(union(logics))
        else:
            slicer = self.take({name: has_any(codes)})
        return slicer

    @modify(to_list='codes')
    @verify(variables={'name': 'both'})
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
        if self._is_array(name):
            logics = []
            for s in self.sources(name):
                logics.append({s: has_all(codes)})
            slicer = self.take(intersection(logics))
        else:
            slicer = self.take({name: has_all(codes)})
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
        if msg and self._verbose_infos:
            print msg.format(name)
        else:
            return None

    def _clean_codes_against_meta(self, name, codes):
        valid = [c for c in codes if c in self._get_valuemap(name, 'codes')]
        deduped_valid = []
        for v in valid:
            if v not in deduped_valid: deduped_valid.append(v)
        return deduped_valid

    def _clean_items_against_meta(self, name, items):
        return [i for i in items if i in self.sources(name)]

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
        if slicer: mask = self.take(slicer)
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
    def _array_item_name(item_no, var_name):
        item_name = '{}_{}'.format(var_name, item_no)
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

    @modify(to_list=['ignore_items', 'ignore_values'])
    @verify(variables={'name': 'masks'}, text_keys='text_key')
    def transpose(self, name, new_name=None, ignore_items=None,
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
        org_name = name
        # Get array item and value structure
        reg_items_object = self._get_itemmap(name)
        if ignore_items:
            reg_items_object = [i for idx, i in
                                enumerate(reg_items_object, start=1)
                                if idx not in ignore_items]
        reg_item_names = [item[0] for item in reg_items_object]
        reg_item_texts = [item[1] for item in reg_items_object]

        reg_value_object = self._get_valuemap(name)
        if ignore_values:
            reg_value_object = [v for v in reg_value_object if v[0]
                                not in ignore_values]
        reg_val_codes = [v[0] for v in reg_value_object]
        reg_val_texts = [v[1] for v in reg_value_object]

        # Transpose the array structure: values --> items, items --> values
        trans_items = [(code, value) for code, value in
                       zip(reg_val_codes, reg_val_texts)]
        trans_values = [(idx, text) for idx, text in
                        enumerate(reg_item_texts, start=1)]
        label = self.text(name, False, text_key)
        # Create the new meta data entry for the transposed array structure
        if not new_name:
            new_name = '{}_trans'.format(self._dims_free_arr_name(name))
            dims_compat_name = self._dims_compat_arr_name(new_name)
        qtype = 'delimited set'
        self.add_meta(new_name, qtype, label, trans_values, trans_items, text_key)
        # Do the case data transformation by looping through items and
        # convertig value code entries...
        if self._dimensions_comp: new_name = self._dims_compat_arr_name(new_name)
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
                    self.recode(trans_item, {new_val_code: slicer},
                                append=True)
        # if self._dimensions_comp: self.dimensionize(new_name)
        if self._verbose_infos:
            print 'Transposed array: {} into {}'.format(org_name, dims_compat_name)

    def take(self, condition):
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
        series_data = full_data['@1'].copy()
        slicer, _ = get_logic_index(series_data, condition, full_data)
        return slicer

    @verify(variables={'target': 'columns'})
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
            Should the new recoded data be appended to values already found
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
            Either the ``DataSet._data`` is modified inplace or a new
            ``pandas.Series`` is returned.
        """
        meta = self._meta
        data = self._data
        recode_series = _recode(meta, data, target, mapper,
                                default, append, intersect, initialize, fillna)
        if inplace:
            self._data[target] = recode_series
            if not self._is_numeric(target):
                self._verify_data_vs_meta_codes(target)
            return None
        else:
            return recode_series

    @verify(variables={'target': 'both'})
    def uncode(self, target, mapper, default=None, intersect=None, inplace=True):
        """
        Create a new or copied series from data, recoded using a mapper.

        Parameters
        ----------
        target : str
            The variable name that is the target of the uncode. If it is keyed
            in ``_meta['masks']`` the uncode is done for all mask items.
            If not found in ``_meta`` this will fail with an error.
        mapper : dict
            A mapper of {key: logic} entries.
        default : str, default None
            The column name to default to in cases where unattended lists
            are given in your logic, where an auto-transformation of
            {key: list} to {key: {default: list}} is provided. Note that
            lists in logical statements are themselves a form of shorthand
            and this will ultimately be interpreted as:
            {key: {default: has_any(list)}}.
        intersect : logical statement, default None
            If a logical statement is given here then it will be used as an
            implied intersection of all logical conditions given in the
            mapper.
        inplace : bool, default True
            If True, the ``DataSet`` will be modified inplace with new/updated
            columns. Will return a new recoded ``pandas.Series`` instance if
            False.

        Returns
        -------
        None or uncode_series
            Either the ``DataSet._data`` is modfied inplace or a new
            ``pandas.Series`` is returned.
        """
        meta = self._meta
        data = self._data
        if self._is_array(target):
            targets = self.sources(target)
            if inplace:
                for t in targets:
                    self.uncode(t, mapper, default, intersect, inplace)
                return None
            else:
                uncode_series = []
                for t in targets:
                    uncode_series.append(self.uncode(t, mapper, default,
                                                     intersect, inplace))
                return uncode_series
        else:
            if not target in meta['columns']:
                raise ValueError("{} not found in meta['columns'].".format(target))

            if not isinstance(mapper, dict):
                raise ValueError("'mapper' must be a dictionary.")

            if not (default is None or default in meta['columns']):
                raise ValueError("'%s' not found in meta['columns']." % (default))

            index_map = index_mapper(meta, data, mapper, default, intersect)

            uncode_series = self[target].copy()
            for code, index in index_map.items():
                uncode_series[index] = uncode_series[index].apply(lambda x:
                                                    self._remove_code(x, code))

            if inplace:
                self._data[target] = uncode_series
                if not self._is_numeric(target):
                    self._verify_data_vs_meta_codes(target)
                return None
            else:
                return uncode_series

    @classmethod
    def _remove_code(cls, x, code):
        if x is np.NaN:
            return np.NaN
        elif ';' in str(x):
            x = str(x).split(';')
            x = [y for y in x if not (y == str(code))]
            x = ';'.join(x)
            if x =='':
                x = np.NaN
        elif x == code:
            x = np.NaN
        return x

    def interlock(self, name, label, variables, val_text_sep = '/'):
        """
        Build a new category-intersected variable from >=2 incoming variables.

        Parameters
        ----------
        name : str
            The new column variable name keyed in ``_meta['columns']``.
        label : str
            The new text label for the created variable.
        variables : list of >= 2 str or dict (mapper)
            The column names of the variables that are feeding into the
            intersecting recode operation. Or dicts/mapper to create temporary
            variables for interlock. Can also be a mix of str and dict. Example:
            ['gender',
             {'agegrp': [(1, '18-34', {'age': frange('18-34')}),
                         (2, '35-54', {'age': frange('35-54')}),
                         (3, '55+', {'age': is_ge(55)})]},
             'region']
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

        i_variables = []
        new_variables = []
        for var in variables:
            if isinstance(var, dict):
                v = var.keys()[0]
                mapper = var.values()[0]
                if self._is_delimited_set_mapper(mapper):
                    qtype = 'delimited set'
                else:
                    qtype = 'single'
                self.derive('{}_temp'.format(v), qtype, v, mapper)
                i_variables.append('{}_temp'.format(v))
                new_variables.append('{}_temp'.format(v))
            else:
                i_variables.append(var)

        if any(self._is_array(v) for v in i_variables):
            raise TypeError('Cannot interlock within array-typed variables!')
        if any(self._is_delimited_set(v) for v in i_variables):
            qtype = 'delimited set'
        else:
            qtype = 'single'

        codes = [self._get_valuemap(v, 'codes') for v in i_variables]
        texts = [self._get_valuemap(v, 'texts') for v in i_variables]
        zipped = zip(list(product(*codes)), list(product(*texts)))
        categories = []
        cat_id = 0
        for codes, texts in zipped:
            cat_id += 1
            cat_label = val_text_sep.join(texts)
            rec = [{v: [c]} for v, c in zip(i_variables, codes)]
            rec = intersection(rec)
            categories.append((cat_id, cat_label, rec))
        self.derive(name, qtype, label, categories)
        for var in new_variables:
            self.drop(var)
        return None

    @verify(text_keys='text_key')
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
            Tuples of either two or three elements of following structures:

            2 elements, no labels provided:
            (code, <qp logic expression here>), e.g.:
            ``(1, intersection([{'gender': [1]}, {'age': frange('30-40')}]))``

            2 elements, no codes provided:
            ('text label', <qp logic expression here>), e.g.:
            ``('Cat 1', intersection([{'gender': [1]}, {'age': frange('30-40')}]))``

            3 elements, with codes + labels:
            (code, 'Label goes here', <qp logic expression here>), e.g.:
            ``(1, 'Men, 30 to 40', intersection([{'gender': [1]}, {'age': frange('30-40')}]))``

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
        err_msg = ("'cond_map' structure not understood. Must pass a list "
                   "of 2 (code, logic) / (text, logic) or 3 (code, text label, "
                   "logic) element tuples!")
        if all(len(cond) == 3 for cond in cond_map):
            categories = [(cond[0], cond[1]) for cond in cond_map]
            idx_mapper = {cond[0]: cond[-1] for cond in cond_map}
        elif all(len(cond) == 2 for cond in cond_map):
            all_int = all(isinstance(cond[0], int) for cond in cond_map)
            all_str = all(isinstance(cond[0], (str, unicode)) for cond in cond_map)
            if not (all_str or all_int):
                raise TypeError(err_msg)
            categories = [cond[0] for cond in cond_map]
            if all_int:
                idx_mapper = {cond[0]: cond[-1] for cond in cond_map}
            if all_str:
                idx_mapper = {c: cond[-1] for c, cond in enumerate(cond_map, start=1)}
        else:
            raise TypeError(err_msg)
        self.add_meta(name, qtype, label, categories, items=None, text_key=text_key)
        self.recode(name, idx_mapper, append=append)
        return None

    @verify(variables={'name': 'columns'}, text_keys='text_key')
    def band(self, name, bands, new_name=None, label=None, text_key=None):
        """
        Group numeric data with band definitions treated as group text labels.

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
        if not self._is_numeric(name):
            msg = "Can only band numeric typed data! {} is {}."
            msg = msg.format(name, self._get_type(name))
            raise TypeError(msg)
        if not text_key: text_key = self.text_key
        if not new_name: new_name = '{}_banded'.format(name)
        if not label: label = self.text(name, False, text_key)
        franges = []
        for idx, band in enumerate(bands, start=1):
            lab = None
            if isinstance(band, dict):
                lab = band.keys()[0]
                band = band.values()[0]
            if isinstance(band, tuple):
                if band[0] < 0:
                    raise ValueError('Cannot band with lower bound < 0.')
                elif band[1] < 0:
                    raise ValueError('Cannot band with upper bound < 0.')
                r = '{}-{}'.format(band[0], band[1])
                franges.append([idx, lab or r, {name: frange(r)}])
            else:
                r = str(band)
                franges.append([idx, lab or r, {name: [band]}])

        self.derive(new_name, 'single', label, franges,
                                text_key=text_key)

        return None

    # ------------------------------------------------------------------------
    # derotate the dataset
    # ------------------------------------------------------------------------

    def _derotate_df(self, mapper, levels, other=None, dropna=True):
        """
        Returns derotated ``dataframe``.
        """
        data = self._data
        dfs = []
        level = levels.keys()[0]
        for question_group in mapper:
            new_var = question_group.keys()[0]
            q_group = question_group.values()[0]

            df = data[q_group]
            df = df.stack().reset_index([1])
            df.columns = [level, new_var]
            df[level] = df[level].map({el: ind for ind, el in enumerate(
                                           q_group, 1)})
            df.set_index([level], append=True, drop=True, inplace=True)
            dfs.append(df)

        new_df = pd.concat(dfs, axis=1)
        new_df = new_df.reset_index(1)

        new_df = new_df.join(data[other])

        new_df.index = list(xrange(0, len(new_df.index)))

        return new_df

    def _derotate_meta(self, mapper, other):
        """
        Returns derotated ``meta``.
        """
        meta = self._meta
        new_meta = self.start_meta(self.text_key)
        new_meta['info']['name'] = '{}_derotate'.format(meta['info']['name'])

        for var in other:
            new_meta = self._assume_meta(new_meta, var, var)

        for question_group in mapper:
            new_var = question_group.keys()[0]
            old_var = question_group.values()[0][0]
            new_meta = self._assume_meta(new_meta, new_var, old_var)
        return new_meta

    def _assume_meta(self, new_meta, new_var, old_var):
        """
        Assumes meta information for variables to other meta object.
        """
        meta = self._meta
        n_masks = new_meta['masks']
        n_cols = new_meta['columns']
        n_sets = new_meta['sets']
        n_lib_v = new_meta['lib']['values']

        if self._is_array(old_var):
            n_masks[new_var] = org_copy.deepcopy(meta['masks'][old_var])
            n_masks[new_var]['name'] = new_var
            if self._has_categorical_data(old_var):
                n_lib_v[new_var] = meta['lib']['values'][old_var]
            n_sets[new_var] = org_copy.deepcopy(meta['sets'][old_var])
            n_sets['data file']['items'].append('masks@{}'.format(new_var))
            for var in self.sources(old_var):
                new_meta = self._assume_meta(new_meta, var, var)
        else:
            n_cols[new_var] = org_copy.deepcopy(meta['columns'][old_var])
            n_cols[new_var]['name'] = new_var
            if self._is_array_item(old_var):
                if not self._maskname_from_item(old_var) in new_meta['masks']:
                    n_cols[new_var]['parent'] = {}
                    n_cols[new_var]['values'] = self._get_value_loc(old_var)
                    n_sets['data file']['items'].append('columns@{}'.format(new_var))
            else:
                n_sets['data file']['items'].append('columns@{}'.format(new_var))

        return new_meta

    @modify(to_list='other')
    def derotate(self, levels, mapper, other=None, unique_key='identity',
                 dropna=True):
        """
        Derotate data and meta using the given mapper, and appending others.

        This function derotates data using the specification defined in
        mapper, which is a list of dicts of lists, describing how
        columns from data can be read as a heirarchical structure.

        Returns derotated DataSet instance and saves data and meta as json
        and csv.

        Parameters
        ----------
        levels : dict
            The name and values of a new column variable to identify cases.

        mapper : list of dicts of lists
            A list of dicts matching where the new column names are keys to
            to lists of source columns. Example:
            mapper = [{'q14_1': ['q14_1_1', 'q14_1_2', 'q14_1_3']},
                      {'q14_2': ['q14_2_1', 'q14_2_2', 'q14_2_3']},
                      {'q14_3': ['q14_3_1', 'q14_3_2', 'q14_3_3']}]

        unique_key: str
            Name of column variable that will be copied to new dataset.

        other: list (optional; default=None)
            A list of additional columns from the source data to be appended
            to the end of the resulting stacked dataframe.

        dropna: boolean (optional; default=True)
            Passed through to the pandas.DataFrame.stack() operation.

        Returns
        -------
        new ``qp.DataSet`` instance
        """
        data = self._data
        meta = self._meta

        if not (isinstance(levels.values()[0], list) and isinstance(levels, dict)):
            raise ValueError('``levels`` must be a ``dict`` of ``lists``.')
        if not all(isinstance(e, dict) and isinstance(e.values()[0], list) and
                   isinstance(mapper, list) for e in mapper):
            msg = '``mapper`` must be ``list`` of ``dicts`` of ``lists``.'
            raise ValueError(msg)
        for q_group in mapper:
            if not len(levels.values()[0]) == len(q_group.values()[0]):
                raise ValueError('``lists`` of source ``columns`` and level '
                                 'variables must have same length.')
        level = levels.keys()[0]
        if other:
            exist_vars = [unique_key] + other + levels[level]
        else:
            exist_vars = [unique_key] + levels[level]
        for var in exist_vars:
            if not (var in meta['columns'] or var in meta['masks']):
                msg = "{} not found in dataset.".format(var)
                raise KeyError(msg)

        # derotated data
        add_cols = self.unroll(exist_vars)
        new_df = self._derotate_df(mapper, levels, add_cols, dropna)

        # new meta
        new_meta = self._derotate_meta(mapper, exist_vars)

        ds = DataSet('{}_derotated'.format(self.name))
        ds.from_components(new_df, new_meta)
        ds.path = self.path

        # some recodes/edits
        lev = ds._data[level]
        ds.add_meta(level, 'single', level, levels[level])
        ds._data[level] = lev

        ds.add_meta('{}_levelled'.format(level), 'single', level,
                    self.values(levels[level][0]))

        for x, lev in enumerate(levels[level], 1):
            rec = {y: {lev: y} for y in ds.codes('{}_levelled'.format(level))}
            ds.recode('{}_levelled'.format(level), rec, intersect={level: x})

        cols = (['@1', unique_key, level, '{}_levelled'.format(level)] +
                levels[level] + [new_var.keys()[0] for new_var in mapper] +
                self.unroll(other))
        ds._data = ds._data[cols]

        # save ``DataSet`` instance as json and csv
        path_json = '{}/{}.json'.format(ds.path, ds.name)
        path_csv = '{}/{}.csv'.format(ds.path, ds.name)
        ds.write_quantipy(path_json, path_csv)

        return ds

    @verify(variables={'variables': 'columns'})
    def to_array(self, name, variables, label):
        """
        Combines column variables with same ``values`` meta into an array.

        Parameters
        ----------
        name: str
            Name of new grid.
        variables: list of str or list of dicts
            Variable names that become items of the array. New item labels can
            be added as dict. Example:
            variables = ['q1_1', {'q1_2': 'shop 2'}, {'q1_3': 'shop 3'}]
        label: str
            Text label for the mask itself.

        Returns
        -------
        None
        """
        meta = self._meta

        newname = self._dims_compat_arr_name(name)
        if self.var_exists(newname):
            raise ValueError('{} does already exist.'.format(name))
        var_list = [v.keys()[0] if isinstance(v, dict)
                     else v for v in variables]
        to_comb = {v.keys()[0]: v.values()[0] for v in variables if isinstance(v, dict)}
        for var in var_list:
            to_comb[var] = self.text(var) if var in variables else to_comb[var]

        first = var_list[0]
        subtype = self._get_type(variables[0])
        if self._has_categorical_data(variables[0]):
            categorical = True
            if not all(self.codes(var) == self.codes(first) for var in var_list):
                raise ValueError("Variables must have same 'codes' in meta.")
            elif not all(self.values(var) == self.values(first) for var in var_list):
                msg = 'Not all variables have the same value texts. Assume valuemap'
                msg += ' of {} for the mask'.format(first)
                warnings.warn(msg)
            val_map = self._get_value_loc(first)
        else:
            categorical = False
        items = []
        name_set = []
        for v in var_list:
            item = {'properties': {},
                    'source': 'columns@{}'.format(v),
                    'text': {self.text_key: to_comb[v]}}
            if categorical:
                meta['columns'][v]['values'] = 'lib@values@{}'.format(name)
            meta['columns'][v]['parent'] = {'masks@{}'.format(name): {'type': 'array'}}
            name_set.append('columns@{}'.format(v))
            items.append(item)
        meta['masks'][name] = {'name': name,
                               'items': items,
                               'properties': {},
                               'text': {self.text_key: label},
                               'type': 'array',
                               'subtype': subtype}
        if categorical:
            meta['masks'][name]['values'] = 'lib@values@{}'.format(name)
            meta['lib']['values'][name] = val_map
        meta['sets'][name] = {'items': name_set}
        meta['sets']['data file']['items'].append('masks@{}'.format(name))
        meta['sets']['data file']['items'] = [v for v in meta['sets']['data file']['items']
                                                if not v in name_set]

        if self._dimensions_comp:
            self.dimensionize(name)
        return None

    def weight(self, weight_scheme, weight_name='weight', unique_key='identity',
               subset=None, report=True, path_report=None, inplace=True):
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
        subset : Quantipy complex logic expression
            A logic to filter the DataSet, weighting only the remaining subset.
        report : bool, default True
            If True, will report a summary of the weight algorithm run
            and factor outcomes.
        path_report : str, default None
            A file path to save an .xlsx version of the weight report to.
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
        if subset:
            ds = self.filter('subset', subset, False)
            meta, data = ds.split()
        else:
            meta, data = self.split()
        engine = qp.WeightEngine(data, meta)
        engine.add_scheme(weight_scheme, key=unique_key)
        engine.run()
        org_wname = weight_name
        if report:
            print engine.get_report()
            print
        if path_report:
            df = engine.get_report()
            full_file_path = '{} ({}).xlsx'.format(path_report, weight_name)
            df.to_excel(full_file_path)
            print 'Weight report saved to:\n{}'.format(full_file_path)
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

    @verify(variables={'var': 'both', 'ignore': 'both'})
    def set_missings(self, var, missing_map='default', ignore=None):
        """
        Flag category definitions for exclusion in aggregations.

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
        if not missing_map:
            var = self.unroll(var)
            for v in var:
                if 'missings' in self._meta['columns'][v]:
                    del self._meta['columns'][v]['missings']
        elif missing_map == 'default':
            ignore = self.unroll(ignore, both='all')
            self._set_default_missings(ignore)
        else:
            ignore = self.unroll(ignore, both='all')
            var = self.unroll(var)
            for v in var:
                missing_map = self._clean_missing_map(v, missing_map)
                if self._has_missings(v):
                    self._meta['columns'][v].update({'missings': missing_map})
                else:
                    self._meta['columns'][v]['missings'] = missing_map
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

    def describe(self, var=None, only_type=None, text_key=None, axis_edit=None):
        """
        Inspect the DataSet's global or variable level structure.
        """
        if text_key is None: text_key = self.text_key
        if var is not None:
            return self._get_meta(var, only_type, text_key, axis_edit)
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
                return self._meta['columns'][v]['missings']
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

    def _is_delimited_set_mapper(self, mapper):
        if isinstance(mapper, list):
            logics = [val[-1] for val in mapper]
        elif isinstance(mapper, dict):
            logics = mapper.values()
        else:
            msg = ("mapper must have the form: {1: logic, 2: logic,...} or ",
                   "[(1, label, logic), (2, label, logic),...]")
            raise ValueError(msg)

        logic_series = []
        for log in logics:
            index = self.take(log)
            s = pd.Series(index=index, data=True)
            logic_series.append(s)
        df = pd.concat(logic_series, axis=1)
        df = df.sum(1)
        if len(df.value_counts()) > 1:
            return True
        else:
            return False

    def _has_missings(self, var):
        if self._is_array(var): var = self.sources(var)[0]
        return self._meta['columns'][var].get('missings', False)

    def _is_numeric(self, var):
        num = ['float', 'int']
        if self._is_array(var):
            return self._get_subtype(var) in num
        else:
            return self._get_type(var) in num

    def _is_array(self, var):
        return self._get_type(var) == 'array'

    def _is_array_item(self, name):
        return self._meta['columns'].get(name, {}).get('parent', False)

    def _maskname_from_item(self, item_name):
        return self.parents(item_name)[0].split('@')[-1]

    def _is_multicode_array(self, mask_element):
        return self[mask_element].dtype == 'object'

    def _is_delimited_set(self, name):
        if self._is_array(name):
            return self._meta['masks'][name]['subtype'] == 'delimited set'
        else:
            return self._meta['columns'][name]['type'] == 'delimited set'

    def _has_categorical_data(self, name):
        if self._is_array(name): name = self.sources(name)[0]
        return self._meta['columns'][name]['type'] in ['single', 'delimited set']

    def _verify_data_vs_meta_codes(self, name, raiseError=True):
        data_codes = self.codes_in_data(name)
        meta_codes = self.codes(name)
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
        org_codes = self.codes(name)
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

    def _get_meta_loc(self, var):
        if self._is_array(var):
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

    def _get_valuemap(self, var, non_mapped=None, text_key=None, axis_edit=None):
        if text_key is None: text_key = self.text_key
        vals = self._get_value_loc(var)
        if non_mapped in ['codes', 'lists', None]:
            codes = [int(v['value']) for v in vals]
            if non_mapped == 'codes':
                return codes
        if non_mapped in ['texts', 'lists', None]:
            if axis_edit:
                a_edit = '{} edits'.format(axis_edit)
                texts = [v['text'][a_edit][text_key]
                         if text_key in v['text'].get(a_edit, []) else None
                         for v in vals]
            else:
                texts = [v['text'][text_key] if text_key in v['text'] else None
                         for v in vals]
            if non_mapped == 'texts':
                return texts
        if non_mapped == 'lists':
            return codes, texts
        else:
            return zip(codes, texts)

    def _get_itemmap(self, var, non_mapped=None, text_key=None, axis_edit=None):
        if text_key is None: text_key = self.text_key
        if non_mapped in ['items', 'lists', None]:
            items = [i['source'].split('@')[-1]
                     for i in self._meta['masks'][var]['items']]
            if non_mapped == 'items':
                return items
        if non_mapped in ['texts', 'lists', None]:
            if axis_edit:
                a_edit = '{} edits'.format(axis_edit)
                items_texts = [i['text'][a_edit][text_key]
                               if text_key in i['text'].get(a_edit, []) else None
                               for i in self._meta['masks'][var]['items']]
            else:
                items_texts = [i['text'][text_key]
                               if text_key in i['text'] else None
                               for i in self._meta['masks'][var]['items']]
            if non_mapped == 'texts':
                return items_texts
        if non_mapped == 'lists':
            return items, items_texts
        else:
            return zip(items, items_texts)

    def _get_source_ref(self, var):
        if self._is_array(var):
            return [i['source'] for i in self._meta['masks'][var]['items']]
        else:
            return []

    def _get_meta(self, var, type=None, text_key=None, axis_edit=None):
        if text_key is None: text_key = self.text_key
        is_array = self._is_array(var)
        if is_array:
            var_type = self._meta['masks'][var]['subtype']
        else:
            var_type = self._get_type(var)
        label = self.text(var, False, text_key, axis_edit)
        missings = self._get_missing_map(var)
        make_fame = self._has_categorical_data(var) or self._is_array(var)
        if make_fame:
            if self._has_categorical_data(var):
                codes, texts = self._get_valuemap(var, 'lists', text_key, axis_edit)
                if missings:
                    codes_copy = codes[:]
                    for miss_types, miss_codes in missings.items():
                        for code in miss_codes:
                            codes_copy[codes_copy.index(code)] = miss_types
                    missings = [c  if isinstance(c, (str, unicode)) else None
                                for c in codes_copy]
                else:
                    missings = [None] * len(codes)
            else:
                codes = texts = []
                missings = []
            if is_array:
                items, items_texts = self._get_itemmap(var, 'lists',
                                                       text_key, axis_edit)
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
                    dummy_data.sort_values(axis=1, inplace=True)
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
            items = self.sources(var)
            # items = self._get_itemmap(var, non_mapped='items')
            if self._has_categorical_data(var):
                codes = self._get_valuemap(var, non_mapped='codes')
            else:
                codes = []
                for i in items:
                    codes.extend(self._data[i].dropna().unique().tolist())
                codes = sorted(list(set(codes)))
            dummy_data = []
            if self._is_multicode_array(items[0]):
                for i in items:
                    try:
                        i_dummy = self[i].str.get_dummies(';')
                        i_dummy.columns = [int(col) for col in i_dummy.columns]
                        # dummy_data.append(i_dummy.reindex(columns=codes))
                    except:
                        i_dummy = self._data[[i]]
                        i_dummy.columns = [0]
                    dummy_data.append(i_dummy.reindex(columns=codes))
            else:
                for i in items:
                    if codes:
                        dummy_data.append(
                            pd.get_dummies(self[i]).reindex(columns=codes))
                    else:
                        dummy_data.append(pd.get_dummies(self[i]))
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
        filter_idx, _ = get_logic_index(pd.Series(data.index), condition, data)
        filtered_data = data.iloc[filter_idx, :]
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
    # add Batch to dataset/ get Batch from dataset
    # ------------------------------------------------------------------------
    @modify(to_list=['ci', 'weights', 'tests'])
    def add_batch(self, name, ci=['c', 'p'], weights=None, tests=None):
        return qp.Batch(self, name, ci, weights, tests)

    def get_batch(self, name):
        """
        Get existing Batch instance from DataSet meta information.

        Parameters
        ----------
        name: str
            Name of existing Batch instance.
        """
        batches = self._meta['sets'].get('batches', {})
        if not batches.get(name):
            raise KeyError('No Batch found named {}.'.format(name))
        return qp.Batch(self, name)

    @modify(to_list='batches')
    def populate(self, batches=None):
        """
        Create a ``qp.Stack`` based on all available ``qp.Batch`` definitions.

        Parameters
        ----------
        batches: str/ list of str
            Name(s) of ``qp.Batch`` instances that are used to populate the
            ``qp.Stack``.

        Returns
        -------
        qp.Stack
        """
        if not self._meta['sets'].get('batches'):
            raise KeyError('No ``Batch`` defined! Cannot populate ``Stack``!')
        if batches:
            non_valid = [b for b in batches
                         if not b in self._meta['sets']['batches'].keys()]
            if non_valid:
                raise KeyError('No ``Batch`` named {} defined!'.format(non_valid))
        else:
            batches = self._meta['sets']['batches'].keys()

        dk = self.name
        meta = self._meta
        data = self._data
        stack = qp.Stack(name='aggregations', add_data={dk: (data, meta)})

        for name in batches:
            batch = meta['sets']['batches'][name]
            xs = self.unroll(batch['xks'], both='all')
            fs = batch['x_filter_map']
            f  = batch['filter']
            ys = batch['x_y_map']
            y  = batch['yks']
            s  = batch['summaries']
            ta = batch['transposed_arrays']
            total_len = len(xs)
            for idx, x in enumerate(xs, start=1):
                if self._is_array(x) and not x in s: continue
                if x in ta: stack.add_link(dk, fs[x], x='@', y=x)
                if not ta.get(x):
                    if not x in s:
                        stack.add_link(dk, fs[x], x=x, y=ys[x])
                    else:
                        stack.add_link(dk, fs[x], x=x, y='@')
            if batch['y_on_y']:
                stack.add_link(dk, f, x=y[1:], y=y)
        return stack

    # ------------------------------------------------------------------------
    # validate the dataset
    # ------------------------------------------------------------------------

    def validate(self, verbose=True):
        """
        Identify and report inconsistencies in the ``DataSet`` instance.

        ----------------------------------------------------------------------

        name: column/mask name and meta[collection][var]['name'] are not identical

        q_label: text object is badly formatted or has empty text mapping

        values: categorical variable does not contain values, value text is badly
        formatted or has empty text mapping

        text_keys: dataset.text_key is not included or existing text keys are not
        consistent (also for parents)

        source: parents or items do not exist

        codes: codes in data component are not included in meta component
        """
        def validate_text_obj(text_obj):
            edits = ['x edits', 'y edits']
            if not isinstance(text_obj, dict):
                return False
            else:
                for tk, text in text_obj.items():
                    if ((tk in edits and not validate_text_obj(text_obj[tk]))
                        or text in [None, '', ' ']):
                        return False
            return True

        def validate_value_obj(value_obj):
            if not value_obj:
                return False
            else:
                for val in value_obj:
                    if not 'value' in val or not validate_text_obj(val.get('text')):
                        return False
            return True

        def collect_and_validate_tks(all_text_obj):
            edits = ['x edits', 'y edits']
            tks = []
            for obj in all_text_obj:
                if not isinstance(obj, dict): continue
                for tk in obj.keys():
                    if tk in ['x edits', 'y edits']: continue
                    if not tk in tks: tks.append(tk)
            if not self.text_key in tks: return False
            for obj in all_text_obj:
                if not isinstance(obj, dict): continue
                if not all(tk in obj for tk in tks): return False
            return True

        msg = 'Please check the following variables, metadata is inconsistent.'
        err_columns = ['name', 'q_label', 'values', 'text keys', 'source', 'codes']
        err_df = pd.DataFrame(columns=err_columns)

        skip = [v for v in self.masks() + self.columns() if v.startswith('qualityControl_')]
        skip += ['@1', 'id_L1.1', 'id_L1']

        for v in self.columns() + self.masks():
            if v in skip: continue
            collection = 'masks' if self._is_array(v) else 'columns'
            var = self._meta[collection][v]
            err_var = ['' for x in range(6)]
            # check name
            if not var.get('name') == v: err_var[0] = 'x'
            # check q_label
            if not validate_text_obj(var.get('text')): err_var[1] = 'x'
            # check values
            if self._has_categorical_data(v):
                values = self._get_value_loc(v)
                if not validate_value_obj(values):
                    err_var[2] = 'x'
                    values = []
            else:
                values = []
            # check sources
            if self._is_array_item(v):
                source = self._maskname_from_item(v)
                s = self._meta['masks'][source]
                s_tks = [s.get('text')]
                if not self.var_exists(source): err_var[4] = 'x'
            elif self._is_array(v):
                source = self.sources(v)
                s_tks = []
                if not all(self.var_exists(i) for i in source): err_var[4] = 'x'
            else:
                s_tks = []
            # check text_keys
            all_text_obj = [var.get('text', {})] + [val.get('text', {}) for val in values] + s_tks
            if not collect_and_validate_tks(all_text_obj): err_var[3] = 'x'
            # check codes
            if not self._is_array(v) and self._has_categorical_data(v):
                data_c = self.codes_in_data(v)
                meta_c = self.codes(v)
                if [c for c in data_c if not c in meta_c]: err_var[5] = 'x'
            if any(x=='x' for x in err_var):
                new_err = pd.DataFrame([err_var], index=[v], columns=err_columns)
                err_df = err_df.append(new_err)

        for c in [c for c in self._data.columns if not c in self._meta['columns']
                  and not c in skip]:
            err_var = ['' for x in range(6)]
            err_var[5] = 'x'
            new_err = pd.DataFrame([err_var], index=[c], columns=err_columns)
            err_df = err_df.append(new_err)

        if not all(self.var_exists(v.split('@')[-1])
                   for v in self._meta['sets']['data file']['items']) and verbose:
            print "'dataset._meta['sets']['data file']['items']' is not consistent!"
        if not len(err_df) == 0:
            if verbose:
                print msg
                print self.validate.__doc__
            return err_df.sort_index()
        else:
            if verbose: print 'No issues found in the dataset!'
            return None

    # ------------------------------------------------------------------------
    # checking equality of datasets
    # ------------------------------------------------------------------------

    @modify(to_list=['variables', 'text_key'])
    @verify(text_keys='text_key')
    def compare(self, dataset, variables=None, strict=False, text_key=None):
        """
        Compares types, codes, values, question labels of two datasets.

        Parameters
        ----------
        dataset : quantipy.DataSet instance
            Test if all variables in the provided ``dataset`` are also in
            ``self`` and compare their metadata definitions.
        variables : str, list of str
            Check only these variables
        strict : bool, default False
            If True lower/ upper cases and spaces are taken into account.
        text_key : str, list of str
            The textkeys for which texts are compared.

        Returns
        -------
        None
        """
        def _comp_texts(text1, text2, strict):
            equal = True
            if strict:
                if not text1 == text2: equal = False
            else:
                if not text1:
                    text1 = ' '
                else:
                    text1 = text1.encode('cp1252').decode('ascii', errors='ignore').replace(' ', '').lower()
                if not text2:
                    text2 = ' '
                else:
                    text2 = text2.encode('cp1252').decode('ascii', errors='ignore').replace(' ', '').lower()
                if not (text1 in text2 or text2 in text1): equal = False
            return equal

        columns = ['type', 'q_label', 'codes', 'value texts']
        df = pd.DataFrame(columns=columns)

        if not text_key: text_key = self.valid_tks
        vars1 = self.masks() + self.columns()
        vars2 = dataset.masks() + dataset.columns()
        if not variables: variables = vars2
        comp = [key for key in vars2 if key in vars1 and key in variables]
        no_comp = [key for key in vars2 if not key in vars1 and key in variables]
        if no_comp:
            print '{} are not included in main DataSet.\n'.format(no_comp)
        for var in comp:
            if var == '@1': continue
            row = ['' for x in range(4)]
            if not self._get_type(var) == dataset._get_type(var):
                row[0] = 'x'
            if self._has_categorical_data(var):
                codes1 = self.codes(var)
                codes2 = dataset.codes(var)
                if not codes1 == codes2:
                    row[2] = 'x'
                else:
                    val_texts = {c: '' for c in codes1}
                    for tk in text_key:
                        for values, text2 in zip(self.values(var, tk),
                                                 dataset.value_texts(var, tk)):
                            c, text1 = values
                            if not _comp_texts(text1, text2, strict):
                                val_texts[c] += '{}, '.format(tk)
                    if not all(text=='' for text in val_texts.values()):
                        for c, tk in val_texts.items():
                            if not tk == '':
                                row[3] += '{}: {}'.format(c, tk)
            for tk in text_key:
                text1 = self.text(var, True, tk)
                text2 = dataset.text(var, True, tk)
                if not _comp_texts(text1, text2, strict):
                    row[1] += '{}, '.format(tk)
            if not all(x=='' for x in row):
                new_row = pd.DataFrame([row], index=[var], columns=columns)
                df = df.append(new_row)
        if not len(df) == 0: return df.sort_index()

# ============================================================================

    def parrot(self):
        from IPython.display import Image
        from IPython.display import display
        try:
            return display(Image(url="https://m.popkey.co/3a9f4b/jZZ83.gif"))
        except:
            print ':sad_parrot: Looks like the parrot url is not longer there!'

    def _verify_column_in_meta(self, name):
        warning = "'_verify_column_in_meta' will be removed soon.\n"
        warning += "Please use 'var_exists' instead!"
        warnings.warn(warning)
        if not isinstance(name, list): name = [name]
        for n in name:
            if n not in self._meta['columns']:
                raise KeyError("'{}' not found in meta data!".format(n))
        return None

    @verify(variables={'var': 'masks'})
    def unmask(self, var):
        warning = "'unmask' will be removed soon.\n"
        warning += "Please use 'sources' instead!"
        warnings.warn(warning)
        return self._get_itemmap(var=var, non_mapped='items')

    def _prep_varlist(self, varlist, keep_unexploded=False):
        warning = "'_prep_varlist' will be removed soon.\n"
        warning += "Please use 'unroll' instead!"
        warnings.warn(warning)
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