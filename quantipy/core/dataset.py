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
    write_quantipy as w_quantipy,
    write_dimensions as w_dimensions)

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
import time
import sys
import os
from itertools import product, chain
from collections import OrderedDict, Counter

VALID_TKS = ['en-GB', 'da-DK', 'fi-FI', 'nb-NO', 'sv-SE', 'de-DE', 'fr-FR',
             'ar-AR', 'es-ES', 'it-IT']
VAR_SUFFIXES = ['_rc', '_net', ' (categories', ' (NET', '_rec']

BLACKLIST_VARIABLES = ['batches', 'columns', 'info', 'items', 'lib', 'masks',
                       'name', 'parent', 'properties', 'text', 'type', 'sets',
                       'subtype', 'values']

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
        self._dimensions_suffix = '_grid'
        return None

    def __contains__(self, name):
        return self.var_exists(name)

    def __delitem__(self, name):
        self.drop(name)
        return None

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
        if (self._get_type(name) == 'delimited set' and scalar_insert
            and not np.isnan(val)):
            val = '{};'.format(val)
        if sliced_insert:
            self._data.loc[slicer, name] = val
        else:
            self._data[name] = val

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

    def singles(self, array_items=True):
        singles = self._get_columns('single')
        if array_items:
            return singles
        else:
            return [v for v in singles if not self._is_array_item(v)]

    def delimited_sets(self, array_items=True):
        delimited_sets = self._get_columns('delimited set')
        if array_items:
            return delimited_sets
        else:
            return [v for v in delimited_sets if not self._is_array_item(v)]

    def ints(self, array_items=True):
        ints = self._get_columns('int')
        if array_items:
            return ints
        else:
            return [v for v in ints if not self._is_array_item(v)]

    def floats(self, array_items=True):
        floats = self._get_columns('float')
        if array_items:
            return floats
        else:
            return [v for v in floats if not self._is_array_item(v)]

    def dates(self):
        return self._get_columns('date')

    def strings(self):
        return self._get_columns('string')

    def created(self):
        return [v for v in self.variables() if self.get_property(v, 'created')]

    def _stat_view_recodes(self):
        return [v for v in self.variables() if
                self.get_property(v, 'recoded_stat')]

    def _net_view_recodes(self):
        return [v for v in self.variables() if
                self.get_property(v, 'recoded_net')]


    def batches(self):
        if 'batches' in self._meta['sets']:
            return self._meta['sets']['batches'].keys()
        else:
            return []

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

    def set_dim_comp(self, dimensions_comp):
        self._dimensions_comp = dimensions_comp
        self._meta['info']['dimensions_comp'] = dimensions_comp
        return None

    def set_dim_suffix(self, suffix=None):
        if not suffix:
            suffix = self._meta['info'].get('dimensions_suffix', self._dimensions_suffix)
        if not suffix == self._dimensions_suffix:
            self._dimensions_suffix = suffix
        self._meta['info']['dimensions_suffix'] = suffix
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

    def _get_cache(self):
        return self._cache

    def _clear_cache(self):
        self._cache = Cache()

    def _get_type(self, var):
        if var in self._meta['masks'].keys():
            return self._meta['masks'][var]['type']
        else:
            return self._meta['columns'][var]['type']

    def _get_subtype(self, name):
        if not self.is_array(name):
            return None
        else:
            return self._meta['masks'][name]['subtype']

    # ------------------------------------------------------------------------
    # is .../ has ...
    # ------------------------------------------------------------------------

    def is_single(self, name):
        return self._get_type(name) == 'single'

    def is_delimited_set(self, name):
        if self.is_array(name):
            return self._get_subtype(name) == 'delimited set'
        else:
            return self._get_type(name) == 'delimited set'

    def _is_delimited_set(self, name):
        warnings.warn('will be removed soon, please use ``.is_delimited_set()`` instead.')
        return self.is_delimited_set(name)

    def is_int(self, name):
        return self._get_type(name) == 'int'

    def is_float(self, name):
        return self._get_type(name) == 'float'

    def is_string(self, name):
        return self._get_type(name) == 'string'

    def is_date(self, name):
        return self._get_type(name) == 'date'

    def is_array(self, name):
        return self._get_type(name) == 'array'

    def _is_array(self, var):
        warnings.warn('will be removed soon, please use ``.is_array()`` instead.')
        return self.is_array(var)

    def _is_array_item(self, name):
        return self._meta['columns'].get(name, {}).get('parent', False)

    def _is_multicode_array(self, mask_element):
        return self[mask_element].dtype == 'object'

    @verify(variables={'name': 'columns'})
    def is_like_numeric(self, name):
        """
        Test if a ``string``-typed variable can be expressed numerically.

        Parameters
        ----------
        name : str
            The column variable name keyed in ``_meta['columns']``.

        Returns
        -------
        bool
        """
        if self.is_array(name):
            raise TypeError("Cannot test array masks for numeric likeness!")
        if not self._meta['columns'][name]['type'] == 'string':
            err = "Column '{}' is not of type string (but {})."
            raise TypeError(err.format(name, self._meta['columns'][name]['type']))
        s = self._data[name]
        try:
            s.apply(lambda x: int(x))
            return True
        except:
            try:
                s.apply(lambda x: float(x))
                return True
            except:
                return False

    def _is_numeric(self, var):
        num = ['float', 'int']
        if self.is_array(var):
            return self._get_subtype(var) in num
        else:
            return self._get_type(var) in num

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
        if self.is_array(var): var = self.sources(var)[0]
        return self._meta['columns'][var].get('missings', False)

    def _has_categorical_data(self, name):
        if self.is_array(name): name = self.sources(name)[0]
        return self._meta['columns'][name]['type'] in ['single', 'delimited set']

    # ------------------------------------------------------------------------
    # file i/o / conversions
    # ------------------------------------------------------------------------

    def save(self):
        """
        Save the current state of the DataSet's data and meta.

        The saved file will be temporarily stored inside the cache. Use this
        to take a snapshot of the DataSet state to easily revert back to at a
        later stage.

        .. note:: This method is designed primarily for use in interactive
            Python environments like iPython/Jupyter notebook applications.
        """
        if self._data is None and self._meta is None:
            w = "No data/meta components found in the DataSet."
            warnings.warn(w)
            return None
        ds_clone = self.clone()
        self._cache['savepoint'] = ds_clone.split()
        return None

    def revert(self):
        """
        Return to a previously saved state of the DataSet.

        .. note:: This method is designed primarily for use in interactive
            Python environments like iPython/Jupyter and their notebook
            applications.
        """
        if not 'savepoint' in self._cache:
            w = "No saved session DataSet file found!"
            warnings.warn(w)
            return None
        self._meta, self._data = self._cache['savepoint']
        print 'Reverted to last savepoint of {}'.format(self.name)
        return None

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
        if not self._dimensions_comp == 'ignore':
            d_comp = self._dimensions_comp
            self._meta['info']['dimensions_comp'] = d_comp
            self.set_dim_suffix()
            self.undimensionize()
        if d_comp is True: self.dimensionize()
        self._rename_blacklist_vars()
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
        if not self._dimensions_comp == 'ignore':
            d_comp = self._dimensions_comp
            self._meta['info']['dimensions_comp'] = d_comp
            self.set_dim_suffix()
            self.undimensionize()
        if d_comp is True: self.dimensionize()
        self._rename_blacklist_vars()
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
        self._rename_blacklist_vars()
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
        self._rename_blacklist_vars()
        return None

    @verify(text_keys='text_key')
    def write_dimensions(self, path_mdd=None, path_ddf=None, text_key=None,
                         run=True, clean_up=True):
        """
        Build Dimensions/SPSS Base Professional .ddf/.mdd data pairs.

        .. note:: SPSS Data Collection Base Professional must be installed on
            the machine. The method is creating .mrs and .dms scripts which are
            executed through the software's API.

        Parameters
        ----------
        path_mdd : str, default None
            The full path (optionally with extension ``'.mdd'``, otherwise
            assumed as such) for the saved the DataSet._meta component.
            If not provided, the instance's ``name`` and ```path`` attributes
            will be used to determine the file location.
        path_ddf : str, default None
            The full path (optionally with extension ``'.ddf'``, otherwise
            assumed as such) for the saved DataSet._data component.
            If not provided, the instance's ``name`` and ```path`` attributes
            will be used to determine the file location.
        text_key : str, default None
            The desired ``text_key`` for all ``text`` label information. Uses
            the ``DataSet.text_key`` information if not provided.
        run : bool, default True
            If True, the method will try to run the metadata creating .mrs
            script and execute a DMSRun for the case data transformation in
            the .dms file.
        clean_up : bool, default True
            By default, all helper files from the conversion (.dms, .mrs,
            paired .csv files, etc.) will be deleted after the process has
            finished.

        Returns
        -------
        A .ddf/.mdd pair is saved at the provided path location.
        """
        ds_clone = self.clone()
        if not text_key: text_key = ds_clone.text_key
        if ds_clone._dimensions_comp:
            ds_clone.undimensionize()
        # naming rules for Dimensions are applied
        ds_clone.dimensionize()
        meta, data = ds_clone._meta, ds_clone._data
        if path_ddf is None and path_mdd is None:
            path = ds_clone.path
            name = ds_clone.name
            path_mdd = os.path.join(path, ''.join([name, '.mdd']))
            path_ddf = os.path.join(path, ''.join([name, '.ddf']))
        elif path_ddf is not None and path_mdd is not None:
            if not path_mdd.endswith('.mdd'):
                path_mdd = ''.join([path_mdd, '.mdd'])
            if not path_ddf.endswith('.ddf'):
                path_ddf = ''.join([path_ddf, '.ddf'])
        else:
            msg = "Must either specify or omit both 'path_mdd' and 'path_ddf'!"
            raise ValueError(msg)
        path_mdd = path_mdd.replace('//', '/')
        path_ddf = path_ddf.replace('//', '/')
        w_dimensions(meta, data, path_mdd, path_ddf, text_key=text_key,
                     run=run, clean_up=clean_up)
        file_msg = u"\nSaved files to:\n{} and\n{}".format(path_mdd, path_ddf)
        print file_msg
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
            The full path (optionally with extension ``'.csv'``, otherwise
            assumed as such) for the saved DataSet._data component.
            If not provided, the instance's ``name`` and ```path`` attributes
            will be used to determine the file location.

        Returns
        -------
        A .csv/.json pair is saved at the provided path location.
        """
        meta, data = self._meta, self._data
        if path_data is None and path_meta is None:
            path = self.path
            name = self.name
            path_meta = os.path.join(path, ''.join([name, '.json']))
            path_data = os.path.join(path, ''.join([name, '.csv']))
        elif path_data is not None and path_meta is not None:
            if not path_meta.endswith('.json'):
                path_meta = ''.join([path_meta, '.json'])
            if not path_data.endswith('.csv'):
                path_data = ''.join([path_data, '.csv'])
        else:
            msg = "Must either specify or omit both 'path_meta' and 'path_data'!"
            raise ValueError(msg)
        w_quantipy(meta, data, path_meta, path_data)
        return None

    @verify(text_keys='text_key')
    def write_spss(self, path_sav=None, index=True, text_key=None,
                   mrset_tag_style='__', drop_delimited=True, from_set=None,
                   verbose=True):
        """
        Convert the Quantipy DataSet into a SPSS .sav data file.

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
        A SPSS .sav file is saved at the provided path location.
        """
        self.set_encoding('cp1252')
        meta, data = self._meta, self._data
        if not text_key: text_key = self.text_key
        if not path_sav:
            path_sav = os.path.join(self.path, ''.join([self.name, '.sav']))
        else:
            if not path_sav.endswith('.sav'):
                path_sav = ''.join([path_sav, '.sav'])
        w_spss(path_sav, meta, data, index=index, text_key=text_key,
               mrset_tag_style=mrset_tag_style, drop_delimited=drop_delimited,
               from_set=from_set, verbose=verbose)
        self.set_encoding('utf-8')
        return None

    @verify(variables={'key': 'both'})
    def meta_to_json(self, key=None, collection=None):
        """
        Save a meta object as json file.

        Parameters
        ----------
        key: str, default None
            Name of the variable whose metadata is saved, if key is not
            provided included collection or the whole meta is saved.
        collection: str {'columns', 'masks', 'sets', 'lib'}, default None
            The meta object is taken from this collection.

        Returns
        -------
        None
        """
        meta = self._meta
        if key: k = '@{}'.format(key)
        col = {'columns': 'columns{}'.format(k if key else ''),
               'masks':   'masks{}'.format(k if key else ''),
               'sets':    'sets{}'.format(k if key else ''),
               'lib':     'lib@values{}'.format(k if key else '')}
        if collection and not collection in col.keys():
            raise ValueError('collection must be one of {}'.format(col.keys()))
        if key and not collection:
            collection = 'masks' if key in self.masks() else 'columns'
        if not (key or collection):
            obj = meta
            name = 'meta'
        else:
            obj_p = col[collection].split('@')
            obj = meta[obj_p.pop(0)]
            while obj_p:
                obj = obj[obj_p.pop(0)]
            name = '{}{}'.format(collection, '_{}'.format(key.split('.')[0])
                                 if key else '')
        ds_path = '../' if self.path == '/' else self.path
        path = os.path.join(ds_path, ''.json([self.name, '_', name, '.json']))
        with open(path, 'w') as file:
            json.dump(obj, file)
        print u'create: {}'.format(path)
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
        if 'id_L1' in self._data.columns:
            self._data.drop('id_L1', axis=1, inplace=True)
        if 'id_L1.1' in self._data.columns:
            self._data.drop('id_L1.1', axis=1, inplace=True)
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
        self.set_dim_suffix()
        self._rename_blacklist_vars()
        return None

    def from_stack(self, stack, data_key=None, dk_filter=None, reset=True):
        """
        Use ``quantipy.Stack`` data and meta to create a ``DataSet`` instance.

        Parameters
        ----------
        stack : quantipy.Stack
            The Stack instance to convert.
        data_key : str
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
        if data_key is None and len(stack.keys()) > 1:
            msg = 'Please specify a data_key, the Stack contains more than one.'
            raise ValueError(msg)
        elif data_key is None:
            data_key = stack.keys()[0]
        elif not data_key in stack.keys():
            msg = "data_key '{}' does not exist.".format(data_key)
            raise KeyError(msg)

        if not dk_filter:
            dk_f = 'no_filter'
        elif dk_filter in stack[data_key].keys():
            msg = 'Please pass an existing filter of the Stack:\n{}'.format(
                stack[data_key].keys())
            raise KeyError(msg)

        meta = stack[data_key].meta
        data = stack[data_key][dk_f].data
        self.name = data_key
        self.filtered = dk_f
        self.from_components(data, meta, reset=reset)

        return None

    def _vars_from_batch(self, batchdef, mode='batch-full'):
        """
        """
        xs = self.roll_up(batchdef['xks'])
        ys = batchdef['yks']
        for add_y_coll in ['extended_yks_per_x', 'exclusive_yks_per_x']:
            if batchdef[add_y_coll]:
                for y in batchdef[add_y_coll].values:
                    if not y in ys: ys.append(y)
        if '@' in ys: ys.remove('@')
        oe = []
        for verbatim in batchdef['verbatims']:
            oe += verbatim['columns']
        oe = list(set(oe))
        w = batchdef['weights']
        if None in w:
            w.remove(None)
        if mode in ['batch-full', 'batch-x']:
            batch_vars = xs
            if mode == 'batch-full':
                for y in ys:
                    if not y in batch_vars: batch_vars.append(y)
                for verbatim in oe:
                    if not oe in batch_vars:
                        batch_vars.append(verbatim)
        if w: batch_vars.extend(w)
        return batch_vars


    @modify(to_list=['text_key', 'include'])
    @verify(text_keys='text_key', variables={'include': 'both'})
    def _from_batch(self, batch_name, include='identity', text_key=[],
                    apply_edits=True, additions='variables'):
        """
        """
        # get the main batch definition to construct a dataset from...
        batch_def = self._meta['sets']['batches'][batch_name]
        # filter it if needed:
        if batch_def['filter'] == 'no_filter':
            b_ds = self.clone()
        else:
            b_ds = self.filter(batch_name, batch_def['filter'].values()[0])
        # build the variable collection based in Batch setup & requirements:
        main_variables = b_ds._vars_from_batch(batch_def, 'batch-full')
        if additions in ['full', 'filters']:
            print 'manifest_filters() needed, add filters to list...'
            pass
        if additions in ['full', 'variables']:
            if batch_def['additions']:
                for add_batch in batch_def['additions']:
                    add_batch_def = b_ds._meta['sets']['batches'][add_batch]
                    add_vars = b_ds._vars_from_batch(add_batch_def)
                    add_vars = [v for v in add_vars if not v in main_variables]
                    main_variables.extend(add_vars)
        # add any custom variables...
        if include:
            include = [v for v in include if not v in main_variables]
            main_variables = include + main_variables
        # subset the dataset variables...
        b_ds.subset(main_variables, inplace=True)
        b_ds.order(main_variables)
        self._rename_blacklist_vars()
        return b_ds


    @modify(to_list=['text_key', 'include'])
    @verify(text_keys='text_key', variables={'include': 'both'})
    def from_batch(self, batch_name, include='identity', text_key=[],
                   apply_edits=True, additions='variables'):
        """
        Get a filtered subset of the DataSet using qp.Batch definitions.

        Parameters
        ----------
        batch_name: str
            Name of a Batch included in the DataSet.
        include: str/ list of str
            Name of variables that get included even if they are not in Batch.
        text_key: str/ list of str, default None
            Take over all texts of the included text_key(s), if None is provided
            all included text_keys are taken.
        apply_edits: bool, default True
            meta_edits and rules are used as/ applied on global meta of the
            new DataSet instance.
        additions: {'variables', 'filters', 'full', None}
            Extend included variables by the xks, yks and weights of the
            additional batches if set to 'variables', 'filters' will create
            new 1/0-coded variables that reflect any filters defined. Selecting
            'full' will do both, ``None`` will ignore additional Batches completely.

        Returns
        -------
        b_ds : ``quantipy.DataSet``
        """
        def _apply_edits_rules(ds, name, b_meta):
            if ds.is_array(name) and b_meta.get(name):
                ds._meta['masks'][name] = b_meta[name]
                try:
                    ds._meta['lib']['values'][name] = b_meta['lib'][name]
                except:
                    pass
            elif b_meta.get(name):
                ds._meta['columns'][name] = b_meta[name]
            if not ds._is_array_item(name):
                for axis in ['x', 'y']:
                    if all(rule in ds._get_rules(name, axis) for rule in ['dropx', 'slicex']):
                        drops = ds._get_rules(name, axis)['dropx']['values']
                        slicer = ds._get_rules(name, axis)['slicex']['values']
                    elif 'dropx' in ds._get_rules(name, axis):
                        drops = ds._get_rules(name, axis)['dropx']['values']
                        slicer = ds.codes(name)
                    elif 'slicex' in ds._get_rules(name, axis):
                        drops = []
                        slicer = ds._get_rules(name, axis)['slicex']['values']
                    else:
                        drops = slicer = []
                    if drops or slicer:
                        if not all(isinstance(c, int) for c in drops):
                            item_no = [ds.item_no(v) for v in drops]
                            ds.remove_items(name, item_no)
                        else:
                            codes = ds.codes(name)
                            n_codes = [c for c in slicer if not c in drops]
                            if not len(n_codes) == len(codes):
                                remove = [c for c in codes if not c in n_codes]
                                ds.remove_values(name, remove)
                            ds.reorder_values(name, n_codes)
                            if ds.is_array(name):
                                ds._meta['masks'][name].pop('rules')
                            else:
                                ds._meta['columns'][name].pop('rules')
                        return None

        def _manifest_filters(ds, batch_name):
            all_batches = ds._meta['sets']['batches']
            add_batches = ds._meta['sets']['batches'][batch_name]['additions']
            if not adds: return None
            filters = []
            mtextkey = ds._meta['sets']['batches'][batch_name]['language']
            for add_batch in add_batches:
                if all_batches[add_batch]['filter'] != 'no_filter':
                    filters.append(
                        (add_batch,
                         all_batches[add_batch]['language'],
                         all_batches[add_batch]['filter'])
                        )
            if not filters: return None
            cats = [(1, 'active')]
            fnames = []
            for no, f in enumerate(filters, start=1):
                fname = 'filter_{}'.format(no)
                fnames.append(fname)
                source = f[0]
                ftextkey = f[1]
                flogic = f[2].values()[0]
                flabel = f[2].keys()[0]
                ds.add_meta(fname, 'single', flabel, cats, text_key=ftextkey)
                if not mtextkey == ftextkey:
                    ds.set_variable_text(fname, flabel, text_key=mtextkey)
                    ds.set_value_texts(fname, {1: 'active'}, text_key=mtextkey)
                ds._meta['columns'][fname]['properties']['recoded_filter'] = source
                ds[ds.take(flogic), fname] = 1
            return fnames

        batches = self._meta['sets'].get('batches', {})
        if not batch_name in batches:
            msg = 'No Batch named "{}" is included in DataSet.'
            raise KeyError(msg.format(batch_name))
        else:
            batch = batches[batch_name]
        if not text_key: text_key = self.valid_tks
        if not batch['language'] in text_key:
            msg = 'Batch-textkey {} is not included in {}.'
            raise ValueError(msg.format(batch['language'], text_key))
        # Create a new instance by filtering or cloning
        if batch['filter'] == 'no_filter':
            b_ds = self.clone()
        else:
            b_ds = self.filter(batch_name, batch['filter'].values()[0])

        # Get a subset of variables (xks, yks, oe, weights)
        if additions in ['full', 'variables']:
            adds = batch['additions']
        else:
            adds = []
        if additions in ['full', 'filters']:
            filter_vars = _manifest_filters(b_ds, batch_name)
        else:
            filter_vars = []

        variables = include

        for b_name, ba in batches.items():
            if not b_name in [batch_name] + adds: continue
            variables += ba['xks'] + ba['yks']
            for oe in ba['verbatims']:
                variables += oe['columns']
            variables += ba['weights']
            for yks in ba['extended_yks_per_x'].values() + ba['exclusive_yks_per_x'].values():
                variables += yks
        if filter_vars: variables += filter_vars
        variables = list(set([v for v in variables if not v in ['@', None]]))
        variables = b_ds.roll_up(variables)
        b_ds.subset(variables, inplace=True)
        # Modify meta of new instance
        b_ds.name = b_ds._meta['info']['name'] = batch_name
        b_ds.set_text_key(batch['language'])
        for b in b_ds._meta['sets']['batches'].keys():
            if not b in [batch_name] + adds: b_ds._meta['sets']['batches'].pop(b)
        b_ds._meta['sets']['batches'][batch_name]['filter'] = 'no_filter'
        b_ds._meta['sets']['batches'][batch_name]['filter_names'] = ['no_filter']
        # apply edits
        if apply_edits:
            b_edits = b_ds._meta['sets']['batches'][batch_name]['meta_edits']
            for var in b_ds.variables():
                if b_ds.var_exists(var):
                    _apply_edits_rules(b_ds, var, b_edits)
        # select text_keys
        if text_key:
            b_ds.select_text_keys(text_key)
        self._rename_blacklist_vars()
        return b_ds

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
        self._rename_blacklist_vars()
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
        file_spec = (u'DataSet: {}\nrows: {} - columns: {}\n'
                     u'Dimensions compatibility mode: {}')
        if not self.path: self.path = '/'
        file_name = os.path.join(self.path, self.name)
        print file_spec.format(
            file_name,
            len(self._data.index),
            len(self._data.columns)-1,
            self._dimensions_comp
        ).encode('utf-8')
        return None

    def _rename_blacklist_vars(self):
        blacklist_txt = (u'Variables identified as part of a blacklist: {}. \n'
                         u'They have been renamed by adding "_" as prefix')
        blacklist_var = []
        for var in BLACKLIST_VARIABLES:
            n_var = '_%s' % var
            if var in self and not n_var in self:
                self.rename(var, u'_{}'.format(var))
                blacklist_var.append(var)
            elif var in self:
                w = "{} cannot be renamed because {} is already used".format(var, n_var)
                warnings.warn(w)
        if blacklist_var:
            print blacklist_txt.format(blacklist_var).encode('utf-8')
        return None

    # ------------------------------------------------------------------------
    # Inspecting
    # ------------------------------------------------------------------------

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

    @modify(to_list='blacklist')
    def variables(self, setname='data file', numeric=True, string=True,
                   date=True, boolean=True, blacklist=None):
        """
        View all DataSet variables listed in their global order.

        Parameters
        ----------
        setname : str, default 'data file'
            The name of the variable set to query. Defaults to the main
            variable collection stored via 'data file'.
        numeric : bool, default True
            Include ``int`` and ``float`` type variables?
        string : bool, default True
            Include ``string`` type variables?
        date : bool, default True
            Include ``date`` type variables?
        boolean : bool, default True
            Include ``boolean`` type variables?
        blacklist : list, default None
            A list of variables names to exclude from the variable listing.

        Returns
        -------
        varlist : list
            The list of variables registered in the queried ``set``.
        """
        varlist = []
        except_list = []
        dsvars = self._variables_from_set(setname)
        if not numeric: except_list.extend(['int', 'float'])
        if not string: except_list.append('string')
        if not date: except_list.append('date')
        if not boolean: except_list.append('boolean')
        for dsvar in dsvars:
            if self._get_type(dsvar) in except_list: continue
            if dsvar in blacklist: continue
            varlist.append(dsvar)
        return varlist

    def _variables_from_set(self, setname):
        """
        Return the variables registered under the provided ``meta['sets']`` key.

        Parameters
        ----------
        setname : str
            The name of the set to query.

        Returns
        -------
        set_vars : list of str
            The list of variable names belonging to the set.
        """
        sets = self._meta['sets']
        if not setname in sets:
            err = "'{}' is no valid set name.".format(setname)
            raise KeyError(err)
        else:
            set_items = sets[setname]['items']
        set_vars = [v.split('@')[-1] for v in set_items]
        return set_vars

    def by_type(self, types=None):
        """
        Get an overview of all the variables ordered by their type.

        Parameters
        ----------
        types : str or list of str, default None
            Restrict the overview to these data types.

        Returns
        -------
        overview : pandas.DataFrame
            The variables per data type inside the ``DataSet``.
        """
        return self.describe(only_type=types)

    def find(self, str_tags=None, suffixed=False):
        """
        Find variables by searching their names for substrings.

        Parameters
        ----------
        str_tags : (list of) str
            The strings tags to look for in the variable names. If not provided,
            the modules' default global list of substrings from VAR_SUFFIXES
            will be used.
        suffixed : bool, default False
            If set to True, only variable names that end with a given string
            sequence will qualify.

        Returns
        -------
        found : list
            The list of matching variable names.
        """
        if not str_tags:
            str_tags = VAR_SUFFIXES
        else:
            if not isinstance(str_tags, list): str_tags = [str_tags]
        found = []
        variables = self.variables()
        for v in variables:
            for str_tag in str_tags:
                if suffixed:
                    if v.endswith(str_tag): found.append(v)
                else:
                    if str_tag in v: found.append(v)
        return found

    def names(self, ignore_items=True):
        """
        Find all semi-duplicate variable names that are different only by case.

        .. note:: Will return self.variables() if no semi-duplicates are found.

        Returns
        -------
        semi_dupes : pd.DataFrame
            An overview of case-sensitive spelling differences in otherwise
            equal variable names.
        """
        all_names = self.variables()
        if not ignore_items:
            all_names = self.unroll(all_names, both='all')
        lower_names = [n.lower() for n in all_names]
        multiple_names = [k for k, v in Counter(lower_names).items() if v > 1]
        if not multiple_names: return self.variables()
        semi_dupes = OrderedDict()
        for name in all_names:
            if name.lower() in multiple_names:
                if not name.lower() in semi_dupes:
                    semi_dupes[name.lower()] = [name]
                elif not name in semi_dupes[name.lower()]:
                    semi_dupes[name.lower()].append(name)
        return pd.DataFrame(semi_dupes)

    def resolve_name(self, name):
        """
        """
        multiples = isinstance(self.names(), pd.DataFrame)
        in_multiples = multiples and name in self.names().keys()
        if not name in self or in_multiples:
            all_names = self.variables()
            lowered = [v.lower() for v in all_names]
            resolved = []
            if name.lower() in lowered:
                offset = 0
                while name.lower() in lowered:
                    pos = lowered.index(name.lower(), offset)
                    lowered.pop(pos)
                    resolved.append(all_names[pos + offset])
                    offset += 1
                return resolved if len(resolved) > 1 else resolved[0]
            else:
                return None
        else:
            return name

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

    @modify(to_list='name')
    def var_exists(self, name):
        variables = self._get_masks() + self._get_columns()
        return all(var in variables for var in name)

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
                return textobj.get(a_edit, {}).get(text_key, '')
            else:
                return textobj.get(text_key, '')

        if text_key is None: text_key = self.text_key
        shorten = False if not self._is_array_item(name) else shorten
        collection = 'masks' if self.is_array(name) else 'columns'
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

    @verify(variables={'name': 'both'})
    def factors(self, name):
        """
        Get categorical data's stat. factor values.

        Parameters
        ----------
        name : str
            The column variable name keyed in ``_meta['columns']`` or
            ``_meta['masks']``.

        Returns
        -------
        factors : OrderedDict
            A ``{value: factor}`` mapping.
        """
        val_loc = self._get_value_loc(name)
        factors = OrderedDict()
        for val in val_loc:
            f = val.get('factor', None)
            if f: factors[val['value']] = f
        return factors

    @verify(variables={'name': 'columns'}, categorical='name')
    def codes_in_data(self, name):
        """
        Get a list of codes that exist in data.
        """
        if self.is_delimited_set(name):
            if not self._data[name].dropna().empty:
                data_codes = self._data[name].str.get_dummies(';').columns.tolist()
                data_codes = [int(c) for c in data_codes]
            else:
                data_codes = []
        else:
            data_codes = pd.get_dummies(self._data[name]).columns.tolist()
        return data_codes

    @modify(to_list='text_label')
    @verify(variables={'name': 'both'}, text_keys='text_key')
    def code_from_label(self, name, text_label, text_key=None, exact=True,
                        flat=True):
        """
        Return the code belonging to the passed ``text`` label (if present).

        Parameters
        ----------
        name : str
            The originating variable name keyed in ``meta['columns']``
            or ``meta['masks']``.
        text_label : str or list of str
            The value text(s) to search for.
        text_key : str, default None
            The desired ``text_key`` to search through. Uses the
            ``DataSet.text_key`` information if not provided.
        exact : bool, default True
            ``text_label`` must exactly match a categorical value's ``text``.
            If False, it is enough that the category *contains* the ``text_label``.
        flat : If a list is passed for ``text_label``, return all found codes
            as a regular list. If False, return a list of lists matching the order
            of the ``text_label`` list.

        Returns
        -------
        codes : list
            The list of value codes found for the passed label ``text``.
        """
        vals= self.values(name, text_key=text_key)
        codes = []
        for text in text_label:
            sub_codes = []
            for c, l in vals:
                if l and text in l and not exact:
                    sub_codes.append(c)
                elif l == text:
                    sub_codes.append(c)
            codes.extend(sub_codes) if flat else codes.append(sub_codes)
        if not codes:
            return None
        else:
            if isinstance(codes[0], list) and len(codes) == 1: codes = codes[0]
            return codes

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
        if not self.is_array(name):
            return []
        else:
            return self._get_itemmap(name, non_mapped='items')

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

    def crosstab(self, x, y=None, w=None, pct=False, decimals=1, text=True,
                 rules=False, xtotal=False):
        """
        """
        meta, data = self.split()
        y = '@' if not y else y
        get = 'count' if not pct else 'normalize'
        show = 'values' if not text else 'text'
        return ct(org_copy.deepcopy(meta), data, x=x, y=y, get=get, weight=w,
                  show=show, rules=rules, xtotal=xtotal, decimals=decimals)

    def data(self):
        """
        Return the ``data`` component of the ``DataSet`` instance.
        """
        return self._data

    # ------------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------------

    @staticmethod
    def _is_all_ints(s):
        try:
            return all(s.dropna().astype(int) == s.dropna())
        except:
            return False

    def _all_str_are_int(self, s):
        temp_s = s.apply(lambda x: float(x)).dropna()
        return self._is_all_ints(temp_s)

    def _get_pd_dtype(self, s):
        if self._is_all_ints(s):
            return 'int'
        else:
            return str(s.dtype)

    def _add_inferred_meta(self, tk):
        self._data.reset_index(inplace=True)
        if 'index' in self._data.columns:
            self._data.drop('index', axis=1, inplace=True)
        msg = "Inferring meta data from pd.DataFrame.columns ({})..."
        msg = msg.format(len(self._data.columns))
        print msg
        self._meta = self.start_meta(tk)
        self.text_key = tk
        for col in self._data.columns:
            name = col
            pdtype = self._get_pd_dtype(self._data[col])
            if 'int' in pdtype:
                qptype = 'int'
            elif 'float' in pdtype:
                qptype = 'float'
            elif pdtype == 'object':
                qptype = 'string'
            else:
                qptype = None
            if not qptype:
                if self._verbose_infos:
                    msg = "Could not infer type for {} (dtype: {})!"
                    print msg.format(name, pdtype)
                    self._data.drop(col, axis=1, inplace=True)
            else:
                if self._verbose_infos:
                    msg = "{}: dtype: {} - converted: {}"
                    print msg.format(name, pdtype, qptype)
                self.add_meta(name, qptype, '', replace=False)
        msg = "Converted {} columns!"
        msg = msg.format(len(self._data.columns))
        print msg
        return None

    def _variables_to_set_format(self, variables):
        """
        """
        set_formatted = ['masks@{}'.format(v) if self.is_array(v)
                         else 'columns@{}'.format(v) for v in variables]
        return set_formatted

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
        codes = [v['value'] for v in vals]
        dupes = [c for c, count in Counter(codes).items() if count > 1]
        if dupes:
            err = "Cannot resolve category definition due to code duplicates: {}"
            raise ValueError(err.format(dupes))
        return vals

    def _add_to_datafile_items_set(self, name):
        datafile_items = self._meta['sets']['data file']['items']
        if self.is_array(name):
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
    def _dims_free_arr_item_name(cls, item_name):
        if '[' in item_name:
            return item_name.split('[{')[1].split('}]')[0]
        else:
            return item_name

    @classmethod
    def _dims_free_arr_name(cls, arr_name):
        return arr_name.split('.')[0]

    def _dims_compat_arr_name(self, arr_name):
        arr_name = self._dims_free_arr_name(arr_name)
        if self._dimensions_comp and not self._dimensions_comp == 'ignore':
            return '{}.{}{}'.format(arr_name, arr_name, self._dimensions_suffix)
        else:
            return arr_name

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

    def _verify_same_value_codes_meta(self, name_a, name_b):
        value_codes_a = self._get_valuemap(name_a, non_mapped='codes')
        value_codes_b = self._get_valuemap(name_b, non_mapped='codes')
        if not set(value_codes_a) == set(value_codes_b):
            msg = "'{}' and '{}' do not share the same code values!"
            raise ValueError(msg.format(name_a, name_b))
        return None

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

    @classmethod
    def _consecutive_codes(cls, codes):
        return sorted(codes) == range(min(codes), max(codes)+1)

    @classmethod
    def _highest_code(cls, codes):
        return max(codes)

    @classmethod
    def _lowest_code(cls, codes):
        return min(codes)

    def _code_from_text(self, valuemap, text):
        check = dict(valuemap)
        for c, t in check.items():
            t = t.replace(' ', '').lower()
            if t == text: return c

    def _get_missing_map(self, var):
        if self.is_array(var):
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

    def _maskname_from_item(self, item_name):
        return self.parents(item_name)[0].split('@')[-1]

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

    def _get_rules(self, var, axis='x'):
        if self.is_array(var):
            rules = self._meta['masks'][var].get('rules', {}).get(axis, {})
        else:
            rules = self._meta['columns'][var].get('rules', {}).get(axis, {})
        return rules

    def _get_meta_loc(self, var):
        if self.is_array(var):
            return self._meta['lib']['values']
        else:
            return self._meta['columns']

    def _get_value_loc(self, var):
        if self._is_numeric(var):
            raise TypeError("Numerical columns do not have 'values' meta.")
        if not self._has_categorical_data(var):
            raise TypeError("Variable '{}' is not categorical!".format(var))
        loc = self._get_meta_loc(var)
        if not self.is_array(var):
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
        if self.is_array(var):
            return [i['source'] for i in self._meta['masks'][var]['items']]
        else:
            return []

    def _get_meta(self, var, type=None, text_key=None, axis_edit=None):
        if text_key is None: text_key = self.text_key
        is_array = self.is_array(var)
        if is_array:
            var_type = self._meta['masks'][var]['subtype']
        else:
            var_type = self._get_type(var)
        label = self.text(var, False, text_key, axis_edit)
        missings = self._get_missing_map(var)
        make_fame = self._has_categorical_data(var) or self.is_array(var)
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
    # fix/ repair meta data
    # ------------------------------------------------------------------------

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
            if self._has_categorical_data(m):
                lib_vals = 'lib@values@{}'.format(m)
                self._meta['masks'][m]['values'] = lib_vals
                for s in self.sources(m):
                    self._meta['columns'][s]['values'] = lib_vals
        return None

    def _clean_datafile_set(self):
        """
        Drop references from ['sets']['data file']['items'] if they do not exist
        in the ``DataSet`` columns or masks definitions.
        """
        items = self._meta['sets']['data file']['items']
        n_items = [i for i in items if self.var_exists(i.split('@')[-1])]
        self._meta['sets']['data file']['items'] = n_items
        return None

    def _fix_varnames(self):
        """
        """
        masks = self._meta['masks']
        cols = self._meta['columns']
        for name, meta in masks.items():
            meta['name'] = name
        for name, meta in cols.items():
            meta['name'] = name
        return None

    @modify(to_list='arrays')
    @verify(variables={'arrays': 'masks'})
    def restore_item_texts(self, arrays=None):
        """
        Restore array item texts.

        Parameters
        ----------
        arrays : str, list of str, default None
            Restore texts for items of these arrays. If None, all keys in
            ``._meta['masks']`` are taken.
        """
        if not arrays: arrays = self.masks()
        for a in arrays:
            sources = self.sources(a)
            for tk, ed in product(self.valid_tks, [None, 'x', 'y']):
                if (any(self.text(i, True, tk, ed)==self.text(i, False, tk, ed)
                    for i in sources) and self.text(a, text_key=tk, axis_edit=ed)):
                    rename_items = {self.item_no(i): self.text(i, True, tk, ed)
                                    for i in sources if self.text(i, True, tk, ed)}
                    self.set_item_texts(a, rename_items, tk, ed)
                elif not any(self.text(i, True, tk, ed) in self.text(i, False, tk, ed)
                    for i in sources if self.text(i, False, tk, ed)) and self.text(a, text_key=tk, axis_edit=ed):
                    rename_items = {self.item_no(i): self.text(i, True, tk, ed)
                                    for i in sources if self.text(i, True, tk, ed)}
                    self.set_item_texts(a, rename_items, tk, ed)
        return None

    def repair(self):
        """
        Try to fix legacy meta data inconsistencies and badly shaped array /
        datafile items ``'sets'`` meta definitions.
        """
        self._fix_varnames()
        self._fix_array_meta()
        self._fix_array_item_vals()
        self.repair_text_edits()
        self.restore_item_texts()
        self._clean_datafile_set()
        return None

    # ------------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------------

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
        if self.is_array(name):
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
        if self.is_array(name):
            logics = []
            for s in self.sources(name):
                logics.append({s: has_all(codes)})
            slicer = self.take(intersection(logics))
        else:
            slicer = self.take({name: has_all(codes)})
        return slicer

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

    def validate(self, spss_limits=False, verbose=True):
        """
        Identify and report inconsistencies in the ``DataSet`` instance.

        name:
            column/mask name and ``meta[collection][var]['name']`` are not identical
        q_label:
            text object is badly formatted or has empty text mapping
        values:
            categorical variable does not contain values, value text is badly
            formatted or has empty text mapping
        text_keys:
            dataset.text_key is not included or existing text keys are not
            consistent (also for parents)
        source:
            parents or items do not exist
        codes:
            codes in data component are not included in meta component
        spss limit name:
            length of name is greater than spss limit (64 characters)
            (only shown if spss_limits=True)
        spss limit q_label:
            length of q_label is greater than spss limit (256 characters)
            (only shown if spss_limits=True)
        spss limit values:
            length of any value text is greater than spss limit (120 characters)
            (only shown if spss_limits=True)
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

        def validate_limits(text_obj, limit):
            if isinstance(text_obj, dict):
                for text in text_obj.values():
                    if isinstance(text, (str, unicode)):
                        if len(text) > limit:
                            return False
                    elif not validate_limits(text.values(), limit):
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
        err_columns = ['name', 'q_label', 'values', 'text keys', 'source', 'codes',
                       'spss limit name', 'spss limit q_label', 'spss limit values']
        if not spss_limits: err_columns = err_columns[:6]
        err_df = pd.DataFrame(columns=err_columns)

        skip = [v for v in self.masks() + self.columns() if v.startswith('qualityControl_')]
        skip += ['@1', 'id_L1.1', 'id_L1']

        for v in self.columns() + self.masks():
            if v in skip: continue
            collection = 'masks' if self.is_array(v) else 'columns'
            var = self._meta[collection][v]
            err_var = ['' for x in range(9)]
            # check name
            if not var.get('name') == v: err_var[0] = 'x'
            if len(var.get('name', '')) > 64: err_var[6] = 'x'
            # check q_label
            if not validate_text_obj(var.get('text')):
                err_var[1] = 'x'
            elif not validate_limits(var.get('text', {}), 256):
                err_var[7] = 'x'
            # check values
            if self._has_categorical_data(v):
                values = self._get_value_loc(v)
                if not validate_value_obj(values):
                    err_var[2] = 'x'
                    values = []
                elif not all(validate_limits(c.get('text', {}), 120) for c in values):
                    err_var[8] = 'x'
            else:
                values = []
            # check sources
            if self._is_array_item(v):
                source = self._maskname_from_item(v)
                s = self._meta['masks'][source]
                s_tks = [s.get('text')]
                if not self.var_exists(source): err_var[4] = 'x'
            elif self.is_array(v):
                source = self.sources(v)
                s_tks = []
                if not all(self.var_exists(i) for i in source): err_var[4] = 'x'
            else:
                s_tks = []
            # check text_keys
            all_text_obj = [var.get('text', {})] + [val.get('text', {}) for val in values] + s_tks
            if not collect_and_validate_tks(all_text_obj): err_var[3] = 'x'
            # check codes
            if not self.is_array(v) and self._has_categorical_data(v):
                data_c = self.codes_in_data(v)
                meta_c = self.codes(v)
                if [c for c in data_c if not c in meta_c]: err_var[5] = 'x'
            if not spss_limits:
                err_var = err_var[:6]
                err_columns = err_columns[:6]
            if any(x=='x' for x in err_var):
                new_err = pd.DataFrame([err_var], index=[v], columns=err_columns)
                err_df = err_df.append(new_err)

        for c in [c for c in self._data.columns if not c in self._meta['columns']
                  and not c in skip]:
            err_var = ['' for x in range(9)]
            err_var[5] = 'x'
            if not spss_limits:
                err_var = err_var[:6]
                err_columns = err_columns[:6]
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

    def weight(self, weight_scheme, weight_name='weight', unique_key='identity',
               subset=None, report=True, path_report=None, inplace=True, verbose=True):
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
        engine.add_scheme(weight_scheme, key=unique_key, verbose=verbose)
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

    # ------------------------------------------------------------------------
    # lists/ sets of variables/ data file items
    # ------------------------------------------------------------------------
    @modify(to_list=['varlist'])
    @verify(variables={'varlist': 'both'})
    def roll_up(self, varlist, ignore_arrays=None):
        """
        Replace any array items with their parent mask variable definition name.

        Parameters
        ----------
        varlist : list
           A list of meta ``'columns'`` and/or ``'masks'`` names.
        ignore_arrays : (list of) str
            A list of array mask names that should not be rolled up if their
            items are found inside ``varlist``.
        Returns
        -------
        rolled_up : list
            The modified ``varlist``.
        """
        if ignore_arrays:
            if not isinstance(ignore_arrays, list):
                ignore_arrays = [ignore_arrays]
        else:
            ignore_arrays = []
        arrays_defs = {arr: self.sources(arr) for arr in self.masks()
                       if not arr in ignore_arrays}
        item_map = {}
        for k, v in arrays_defs.items():
            for item in v:
                item_map[item] = k
        rolled_up = []
        for v in varlist:
            if not self.is_array(v):
                if v in item_map:
                    if not item_map[v] in rolled_up:
                        rolled_up.append(item_map[v])
                else:
                    rolled_up.append(v)
            else:
                rolled_up.append(v)
        return rolled_up

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
            if not self.is_array(var):
                unrolled.append(var)
            else:
                if not var in keep:
                    if var in both:
                        unrolled.append(var)
                    unrolled.extend(self.sources(var))
                else:
                    unrolled.append(var)
        return unrolled

    def _apply_order(self, variables):
        # set order of 'data file' items listing
        datafile_items = self._variables_to_set_format(variables)
        self._meta['sets']['data file']['items'] = datafile_items
        # set pd.DataFrame column order
        column_order = self.unroll(variables)
        self._data = self._data[column_order]
        return None

    def _mapped_by_substring(self):
        suffixed = {}
        suffixed_variables = self.find()
        if suffixed_variables:
            for sv in suffixed_variables:
                for suffix in VAR_SUFFIXES:
                    if suffix in sv:
                        origin = sv.split(suffix)[0]

                        # test name...
                        origin_res = self.resolve_name(origin)
                        if not origin_res:
                            origin_res = origin
                        if isinstance(origin_res, list):
                            if len(origin_res) > 1:
                                msg = "Unable to regroup to {}, ".format(origin)
                                msg += "found semi-duplicate derived names:\n"
                                msg += "{}".format(origin_res)
                                warnings.warn(msg)
                                origin_res = origin
                            else:
                                origin_res = origin_res[0]

                        if not origin_res in suffixed:
                            suffixed[origin_res] = [sv]
                        else:
                            suffixed[origin_res].append(sv)
        return suffixed

    def _mapped_by_meta(self):
        rec_views = {}
        for v in self.variables():
            origin = self.get_property(v, 'recoded_net')
            if origin:
                if not origin in rec_views:
                    rec_views[origin] = [v]
                else:
                    rec_views[origin].append(v)
        return rec_views

    def _map_to_origins(self):
        by_origins = self._mapped_by_substring()
        recoded_views = self._mapped_by_meta()
        varlist = self.variables()
        for var in varlist:
            if var in recoded_views:
                if not var in by_origins:
                    by_origins[var] = recoded_views[var]
                else:
                    for recoded_view in recoded_views[var]:
                        if recoded_view not in by_origins[var]:
                            by_origins[var].append(recoded_view)
        for k, v in by_origins.items():
            if not k in varlist:
                del by_origins[k]
                if not v[0] in varlist:
                    by_origins[v[0]] = v[1:]
        sort_them = []
        for k, v in by_origins.items():
            sort_them.append(k)
            sort_them.extend(v)
        grouped = []
        for v in varlist:
            if v in by_origins:
                grouped.append(v)
                grouped.extend(by_origins[v])
            else:
                if not v in sort_them: grouped.append(v)
        return grouped

    @modify(to_list='reposition')
    def order(self, new_order=None, reposition=None, regroup=False):
        """
        Set the global order of the DataSet variables collection.

        The global order of the DataSet is reflected in the data component's
        pd.DataFrame.columns order and the variable references in the meta
        component's 'data file' items.

        Parameters
        ----------
        new_order : list
            A list of all DataSet variables in the desired order.
        reposition : (List of) dict
            Each dict maps one or a list of variables to a reference variable
            name key. The mapped variables are moved before the reference key.
        regroup : bool, default False
            Attempt to regroup non-native variables (i.e. created either
            manually with ``add_meta()``, ``recode()``, ``derive()``, etc.
            or automatically by manifesting ``qp.View`` objects) with their
            originating variables.

        Returns
        -------
        None
        """
        if (bool(new_order) + bool(reposition) + regroup) > 1:
            err = "Can only either apply ``new_order``, ``reposition`` or "
            err += "``regroup`` variables, not perform multiple operations at once."
            raise ValueError(err)
        if new_order:
            if not sorted(self._variables_from_set('data file')) == sorted(new_order):
                err = "'new_order' must contain all DataSet variables."
                raise ValueError(err)
            check = new_order
        elif reposition:
            check = []
            for r in reposition:
                check.extend(list(r.keys() + r.values()))
        elif regroup:
            new_order = self._map_to_origins()
            check = new_order
        else:
            err = "No ``order`` operation provided, select one of "
            err += "``new_order``, ``regroup``, ``reposition``."
            raise ValueError(err)
        if not all(self.var_exists(v) for v in check):
            err = "At least one variable named in ordering does not exist."
            raise ValueError(err)
        if reposition:
            new_order = self._variables_from_set('data file')
            for repos in reposition:
                before_var = repos.keys()[0]
                repos_vars = repos.values()[0]
                if not isinstance(repos_vars, list): repos_vars = [repos_vars]
                repos_vars = list(reversed(repos_vars))
                idx = new_order.index(before_var)
                for repos_var in repos_vars:
                    new_order.remove(repos_var)
                    new_order.insert(idx, repos_var)
        self._apply_order(new_order)
        return None

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
        arrays : {'masks', 'columns'}, default masks
            For arrays add ``masks@varname`` or ``columns@varname``.
        replace : dict
            Replace a variable in the set with an other.
            Example: {'q1': 'q1_rec'}, 'q1' and 'q1_rec' must be included in
            ``based_on``. 'q1' will be removed and 'q1_rec' will be
            moved to this position.
        overwrite : bool, default False
            Overwrite if ``meta['sets'][name]`` already exist.
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
            raise KeyError("based_on set '{}' is not in meta['sets'].".format(based_on))
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
        if not arrays in ['masks', 'columns']:
            raise ValueError (
                "'arrays' must be either 'masks' or 'columns'.")
        # filter set and create new set
        fset = filtered_set(meta=meta,
                     based_on=based_on,
                     masks=True if arrays == 'masks' else False,
                     included=included,
                     excluded=excluded,
                     strings=strings)

        # if arrays=='both':
        #     new_items = []
        #     items = fset['items']
        #     for item in items:
        #         new_items.append(item)
        #         if item.split('@')[0]=='masks':
        #             for i in meta['masks'][item.split('@')[-1]]['items']:
        #                 new_items.append(i['source'])
        #     fset['items'] = new_items

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
            of the column will be kept under its name prefixed with 'original'.
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

    def drop_duplicates(self, unique_id='identity', keep='first'):
        """
        Drop duplicated cases from self._data.

        Parameters
        ----------
        unique_id : str
            Variable name that gets scanned for duplicates.
        keep : str, {'first', 'last'}
            Keep first or last of the duplicates.
        """
        if self.duplicates(unique_id):
            cases_before = self._data.shape[0]
            self._data.drop_duplicates(subset=unique_id, keep=keep, inplace=True)
            if self._verbose_infos:
                cases_after = self._data.shape[0]
                droped_cases = cases_before - cases_after
                msg = '%s duplicated case(s) dropped, %s cases remaining'
                print msg % (droped_cases, cases_after)
        return None

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
        self.vmerge(dataset, verbose=False, overwrite_text=True)
        return None

    # ------------------------------------------------------------------------
    # Recoding
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
        if replace or not name in self._data.columns:
            self._data[name] = '' if qtype == 'delimited set' else np.NaN
        return None

    def _add_array(self, name, qtype, label, items, categories, text_key):
        """
        """
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
            item_name = '{}_{}'.format(self._dims_free_arr_name(name), item_no)
            item_objects.append(self._item(item_name, text_key, item_lab))
            column_lab = '{} - {}'.format(label, item_lab)
            # add array items to 'columns' meta
            self.add_meta(name=item_name, qtype=qtype, label=column_lab,
                          categories=categories, items=None, text_key=text_key)
            # update the items' values objects
            if not values and categories:
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
        if self._dimensions_comp and not self._dimensions_comp == 'ignore':
            self.dimensionize(name)
        return None

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

    @modify(to_list='ignore')
    @verify(variables={'name': 'columns'}, text_keys='text_key')
    def dichotomize(self, name, value_texts=None, keep_variable_text=True,
                    ignore=None, replace=False, text_key=None):
        """
        """
        if not text_key: text_key = self.text_key
        if not value_texts: value_texts = ('Yes', 'No')
        if not isinstance(value_texts, (list, tuple)):
            err = "'value_texts' must be list-like."
            raise TypeError(err)
        elif len(value_texts) != 2:
            err = "'value_texts' must contain exactly two elements."
            raise ValueError(err)
        elif value_texts[0] == value_texts[1]:
            err = "'value_texts' must contain two different elements."
            raise ValueError(err)
        values = self.values(name, text_key)
        if ignore: values = [v for v in values if v[0] not in ignore]
        new_vars = []
        for value in values:
            code, text = value[0], value[1]
            dicho_name = '{}_{}'.format(name, code)
            new_vars.append(dicho_name)
            if keep_variable_text:
                dicho_label = '{}: {}'.format(self.text(name, text_key), text)
            else:
                dicho_label = text
            cond = [(1, value_texts[0],  {name: [code]})]
            self.derive(dicho_name, 'single', dicho_label, cond)
            self.extend_values(dicho_name, (0, value_texts[1]), text_key=text_key)
            self[self.is_nan(dicho_name), dicho_name] = 0
        if self._verbose_infos:
            print 'created: {}'.format(new_vars)
        if replace:
            new_order = {name: new_vars}
            self.order(reposition=new_order)
            self.drop(name)
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
            if self.is_array(var):
                if not ignore_items:
                    name += self.sources(var)
                else:
                    df_items = meta['sets']['data file']['items']
                    ind = df_items.index('masks@{}'.format(var))
                    n_items = df_items[:ind] + self._get_source_ref(var) + df_items[ind+1:]
                    meta['sets']['data file']['items'] = n_items
                    if self._has_categorical_data(var):
                        values = meta['lib']['values'][var]
                        for source in self.sources(var):
                            meta['columns'][source]['values'] = values
                            meta['columns'][source]['parent'] = {}

        df_items = meta['sets']['data file']['items']
        n_items = [i for i in df_items if not i.split('@')[-1] in name]
        meta['sets']['data file']['items'] = n_items
        data_drop = []
        for var in name:
            if not self.is_array(var): data_drop.append(var)
            remove_loop(meta, var)
        data.drop(data_drop, 1, inplace=True)
        return None

    @modify(to_list=['name'])
    @verify(variables={'name': 'both'})
    def unbind(self, name):
        """
        Remove mask-structure for arrays
        """
        remove = []
        for n in name:
            if not self.is_array(n): continue
            self.drop(n, ignore_items=True)
            remove.append(n)
        if remove and self._verbose_infos:
            print "Remove mask structure for: '{}'".format("', '".join(remove))
        return None

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
        is_array = self.is_array(verify_name)

        array_item_copied = isinstance(name, tuple)
        if not array_item_copied and self._is_array_item(verify_name):
            err = ("Cannot make isolated copy of array item '{}'. "
                   "Please copy array variable '{}' instead!")
            err = err.format(verify_name, self.parents(verify_name)[0].split('@')[-1])
            raise NotImplementedError(err)

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

        check_name = self._dims_compat_arr_name(copy_name)

        if self.var_exists(check_name): self.drop(check_name)

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
            # reduce the meta/data?
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
        Use a mapping of old to new codes to replace code values in ``_data``.

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
        append = self.is_delimited_set(name)
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
        qtype = 'delimited set'
        self.add_meta(new_name, qtype, label, trans_values, trans_items, text_key)
        # Do the case data transformation by looping through items and
        # convertig value code entries...
        new_name = self._dims_compat_arr_name(new_name)
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
        if self._verbose_infos:
            print 'Transposed array: {} into {}'.format(org_name, new_name)

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
            Either the ``DataSet._data`` is modified inplace or a new
            ``pandas.Series`` is returned.
        """
        meta = self._meta
        data = self._data
        if self.is_array(target):
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

            >>> ['gender',
            ...  {'agegrp': [(1, '18-34', {'age': frange('18-34')}),
            ...              (2, '35-54', {'age': frange('35-54')}),
            ...              (3, '55+', {'age': is_ge(55)})]},
            ...  'region']
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

        if any(self.is_array(v) for v in i_variables):
            raise TypeError('Cannot interlock within array-typed variables!')
        if any(self.is_delimited_set(v) for v in i_variables):
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

    @modify(to_list='variables')
    def to_delimited_set(self, name, label, variables, from_dichotomous=True,
                         codes_from_name=True):
        """
        Combines multiple single variables to new delimited set variable.

        Parameters
        ----------
        name: str
            Name of new delimited set
        label: str
            Label text for the new delimited set.
        variables: list of str or list of tuples
            variables that get combined into the new delimited set. If they are
            dichotomous (from_dichotomous=True), the labels of the variables
            are used as category texts or if tuples are included, the second
            items will be used for the category texts.
            If the variables are categorical (from_dichotomous=False) the values
            of the variables need to be eqaul and are taken for the delimited set.
        from_dichotomous: bool, default True
            Define if the input variables are dichotomous or categorical.
        codes_from_name: bool, default True
            If from_dichotomous=True, the codes can be taken from the Variable
            names, if they are in form of 'q01_1', 'q01_3', ...
            In this case the codes will be 1, 3, ....

        Returns
        -------
        None
        """
        if self.var_exists(name):
            raise ValueError('{} does already exist.'.format(name))
        elif not all(isinstance(c, (str, unicode, tuple)) for c in variables):
            raise ValueError('Input of variables must be string or tuple.')
        cols = [c if isinstance(c, (str, unicode)) else c[0] for c in variables]
        if not all(self.var_exists(c) for c in cols):
            not_in_ds = [c for c in cols if not self.var_exists(c)]
            raise KeyError('{} not found in dataset!'.format(not_in_ds))
        elif not all(self._has_categorical_data(c) for c in cols):
            not_cat = [c for c in cols if not self._has_categorical_data(c)]
            raise ValueError('Variables must have categorical data: {}'.format(not_cat))
        if from_dichotomous:
            if not all(x in [0, 1] for c in cols for x in self.codes_in_data(c)):
                non_d = [c for c in cols
                         if not all(x in [0, 1] for x in self.codes_in_data(c))]
                raise ValueError('Variables are not dichotomous: {}'.format(non_d))
            mapper = []
            for x, col in enumerate(variables, 1):
                if codes_from_name:
                    x = int(col.split('_')[-1])
                if isinstance(col, tuple):
                    text = col[1]
                else:
                    text = self.text(col)
                mapper.append((x, text, {col: [1]}))
        else:
            values = self.values(cols[0])
            if not all(self.values(c) == values for c in cols):
                not_eq = [c for c in cols if not self.values(c) == values]
                raise ValueError('Variables must have eqaul values: {}'.format(not_eq))
            mapper = []
            for v in values:
                mapper.append((v[0], v[1], union([{c: v[0]} for c in cols])))

        self.derive(name, 'delimited set', label, mapper)

        return None

    def to_array(self, name, variables, label, safe=True):
        """
        Combines column variables with same ``values`` meta into an array.

        Parameters
        ----------
        name : str
            Name of new grid.
        variables : list of str or list of dicts
            Variable names that become items of the array. New item labels can
            be added as dict. Example:
            variables = ['q1_1', {'q1_2': 'shop 2'}, {'q1_3': 'shop 3'}]
        label : str
            Text label for the mask itself.
        safe : bool, default True
            If True, the method will raise a ``ValueError`` if the provided
            variable name is already present in self. Select ``False`` to
            forcefully overwrite an existing variable with the same name
            (independent of its type).

        Returns
        -------
        None
        """
        meta = self._meta
        newname = self._dims_compat_arr_name(name)
        if self.var_exists(newname):
            if safe:
                raise ValueError('{} does already exist.'.format(name))
            self.drop(newname, ignore_items=True)
        var_list = [v.keys()[0] if isinstance(v, dict)
                     else v for v in variables]
        if not all(self.var_exists(v) for v in var_list):
            raise KeyError("'variables' must be included in DataSet.")
        to_comb = {v.keys()[0]: v.values()[0] for v in variables if isinstance(v, dict)}
        for var in var_list:
            to_comb[var] = self.text(var) if var in variables else to_comb[var]
        first = var_list[0]
        subtype = self._get_type(var_list[0])
        if self._has_categorical_data(var_list[0]):
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

        if self._dimensions_comp and not self._dimensions_comp == 'ignore':
            self.dimensionize(name)
        return None

    # ------------------------------------------------------------------------
    # Converting
    # ------------------------------------------------------------------------

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
        is_num_str = self.is_like_numeric(name) if org_type == 'string' else False
        if not (org_type in valid or is_num_str):
            msg = 'Cannot convert variable {} of type {} to float!'
            raise TypeError(msg.format(name, org_type))
        if org_type == 'single':
            self._as_int(name)
        if org_type in ['int', 'string']:
            self._meta['columns'][name]['type'] = 'float'
            if org_type == 'int':
                self._data[name] = self._data[name].apply(
                        lambda x: float(x) if not np.isnan(x) else np.NaN)
            elif org_type == 'string':
                self._data[name] = self._data[name].apply(lambda x: float(x))
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
        is_num_str = self.is_like_numeric(name) if org_type == 'string' else False
        is_all_ints = self._all_str_are_int(self._data[name])
        is_convertable = is_num_str and is_all_ints
        if not (org_type in valid or is_convertable):
            msg = 'Cannot convert variable {} of type {} to int!'
            raise TypeError(msg.format(name, org_type))
        if self._has_categorical_data(name):
            self._meta['columns'][name].pop('values')
        self._meta['columns'][name]['type'] = 'int'
        if org_type == 'string':
            if is_all_ints:
                self._data[name] = self._data[name].apply(lambda x: int(x))
            else:
                self._data[name] = self._data[name].apply(lambda x: float(x))
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
            values_obj = [self._value(num_val, text_key, unicode(num_val))
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
            values_obj = [self._value(i, text_key, unicode(v)) for i, v
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
        valid = ['single', 'delimited set', 'int', 'float', 'date']
        if not org_type in valid:
            msg = 'Cannot convert variable {} of type {} to text!'
            raise TypeError(msg.format(name, org_type))
        self._meta['columns'][name]['type'] = 'string'
        if self._get_type in ['single', 'delimited set']:
            self._meta['columns'][name].pop('values')
        self._data[name] = self._data[name].astype(str)
        return None

    # renaming
    # ------------------------------------------------------------------------

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
            DataSet is modified inplace. The new name reference replaces the
            original one.
        """
        renames = {}
        if new_name in self._data.columns:
            msg = "Cannot rename '{}' into '{}'. Column name already exists!"
            raise ValueError(msg.format(name, new_name))

        if not self._dimensions_comp == 'ignore':
            self.undimensionize([name] + self.sources(name))
            name = self._dims_free_arr_name(name)

        for no, s in enumerate(self.sources(name), start=1):
            if '_' in s and s.split('_')[-1].isdigit():
                new_s_name = '{}_{}'.format(new_name, s.split('_')[-1])
            else:
                new_s_name = '{}_{}'.format(new_name, no)
            self._add_all_renames_to_mapper(renames, s, new_s_name)

        self._add_all_renames_to_mapper(renames, name, new_name)

        self.rename_from_mapper(renames)

        if self._dimensions_comp and not self._dimensions_comp == 'ignore':
            self.dimensionize(new_name)

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

        def rename_properties(mapper):
            """
            Rename variable properties that reference other variables, i.e.
            'recoded_net', 'recoded_stat' meta objects.
            """
            net_recs = self._net_view_recodes()
            stat_recs = self._stat_view_recodes()
            all_recs = set([r for r in net_recs + stat_recs if r in mapper])
            for rec in all_recs:
                is_array = rec in self.masks()
                if is_array:
                    props = self._meta['masks'][rec]['properties']
                else:
                    props = self._meta['columns'][rec]['properties']
                rn = props.get('recoded_net', None)
                if rn:
                    org_ref = props['recoded_net']
                    props['recoded_net'] = mapper[org_ref]
                rs = props.get('recoded_stat', None)
                if rs:
                    org_ref = props['recoded_stat']
                    props['recoded_stat'] = mapper[org_ref]
            return None

        def rename_meta(meta, mapper):
            """
            Rename lib@values, masks, set items and columns using mapper.
            """
            rename_properties(mapper)
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

        def fix(string):
            tags = [
                "'", '"', ' ', '&', '.', '/', '-',
                '(', ')', '[', ']', '{', '}'
            ]
            for tag in tags:
                string = string.replace(tag, '_')
            return string

        masks = self._meta['masks']
        columns = self._meta['columns']
        suffix = self._dimensions_suffix

        if not names: names = self.variables()
        mapper = {}
        for org_mn, mask in masks.iteritems():
            if org_mn in names:
                mask_name = fix(org_mn)
                new_mask_name = '{mn}.{mn}{s}'.format(mn=mask_name, s=suffix)
                mapper[org_mn] = new_mask_name

                mask_mapper = 'masks@{mn}'.format(mn=org_mn)
                new_mask_mapper = 'masks@{nmn}'.format(nmn=new_mask_name)
                mapper[mask_mapper] = new_mask_mapper

                values_mapper = 'lib@values@{mn}'.format(mn=org_mn)
                new_values_mapper = 'lib@values@{nmn}'.format(nmn=new_mask_name)
                mapper[values_mapper] = new_values_mapper

                items = masks[org_mn]['items']
                for i, item in enumerate(items):
                    org_cn = item['source'].split('@')[-1]
                    col_name = fix(org_cn)
                    new_col_name = '{mn}[{{{cn}}}].{mn}{s}'.format(
                        mn=mask_name, cn=col_name, s=suffix
                    )
                    mapper[org_cn] = new_col_name

                    col_mapper = 'columns@{cn}'.format(cn=org_cn)
                    new_col_mapper = 'columns@{ncn}'.format(ncn=new_col_name)
                    mapper[col_mapper] = new_col_mapper

        for col_name, col in columns.iteritems():
            if col_name in names and not self._is_array_item(col_name):
                new_col_name = fix(col_name)
                if new_col_name == col_name: continue
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
        if not names and self._dimensions_comp:
            raise ValueError('File is already dimensionized.')
        mapper = self.dimensionizing_mapper(names)
        self.rename_from_mapper(mapper)
        if not names:
            self.set_dim_comp(True)
            if 'type' in self:
                self.rename('type', '_type')
        return None

    @modify(to_list='names')
    @verify(variables={'names': 'both'})
    def undimensionize(self, names=None, mapper_to_meta=False):
        """
        Rename the dataset columns to remove Dimensions compatibility.
        """
        mapper = self.undimensionizing_mapper(names)
        self.rename_from_mapper(mapper)
        if mapper_to_meta: self._meta['sets']['rename_mapper'] = mapper
        if not names: self.set_dim_comp(False)

    # value manipulation
    # ------------------------------------------------------------------------

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
        if not self.is_array(name) and self._is_array_item(name):
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
        if self.is_array(name):
            self._meta['lib']['values'][name] = new_values
        else:
            self._meta['columns'][name]['values'] = new_values
        # Remove values in ``data``
        if self.is_array(name):
            items = self._get_itemmap(name, 'items')
            for i in items:
                self.uncode(i, {x: {i: x} for x in remove})
                self._verify_data_vs_meta_codes(i)
        else:
            self.uncode(name, {x: {name: x} for x in remove})
            self._verify_data_vs_meta_codes(name)
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
            enumerated and mapped to the category labels. Alternatively codes can
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
        if not self.is_array(name) and self._is_array_item(name):
            name = self._maskname_from_item(name)
        use_array = self.is_array(name)
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

    # array item manipulation
    # ------------------------------------------------------------------------

    @verify(variables={'name': 'masks'})
    def reorder_items(self, name, new_order):
        """
        Apply a new order to mask items.

        Parameters
        ----------
        name : str
            The variable name keyed in ``_meta['masks']``.
        new_order : list of int, default None
            The new order of the mask items. The included ints match up to
            the number of the items (``DataSet.item_no('item_name')``).

        Returns
        -------
        None
            DataSet is modified inplace.
        """
        sources = self.sources(name)
        s_ref = self._get_source_ref(name)
        org_i = OrderedDict([(self.item_no(s), ref)
                             for s, ref in zip(sources, s_ref)])
        if not set(org_i.keys()) == set(new_order):
            msg = "Only these item numbers are valid for 'new_order': {}"
            raise ValueError(msg.format(org_i.keys()))
        n_set = []
        n_items = []
        for i in new_order:
            ref = org_i[i]
            n_set.append(ref)
            for item in self._meta['masks'][name]['items']:
                if item['source'] == ref:
                    n_items.append(item)
        self._meta['masks'][name]['items'] = n_items
        self._meta['sets'][name]['items'] = n_set
        return None

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

    @modify(to_list=['ext_items'])
    @verify(variables={'name': 'masks'}, text_keys='text_key')
    def extend_items(self, name, ext_items, text_key=None):
        """
        Extend mask items of an existing array.

        Parameters
        ----------
        name: str
            The originating column variable name keyed in ``meta['masks']``.
        ext_items: list of str/ list of dict
            The label of the new item. It can be provided as str, then the new
            column is named by the grid and the item_no, or as dict
            {'new_column': 'label'}.
        text_key: str/ list of str, default None
            Text key for text-based label information. Will automatically fall
            back to the instance's text_key property information if not provided.
        """
        if not text_key: text_key = self.text_key
        self.undimensionize()
        name = self._dims_free_arr_name(name)
        cat = self._has_categorical_data(name)
        source0 = self._meta['columns'][self.sources(name)[0]]
        for n_item in ext_items:
            if isinstance(n_item, dict):
                col = n_item.keys()[0]
                label = n_item.values()[0]
            else:
                col = '{}_{}'.format(name, len(self.sources(name))+1)
                label = n_item
            if self.var_exists(col):
                raise ValueError("Cannot add '{}', as it already exists.".format(col))
            # add column meta
            column = {'name':   col,
                      'text':   {text_key: ''},
                      'type':   source0['type'],
                      'parent': source0['parent'],
                      'properties': {'created': True}}
            if cat:
                column['values'] = source0['values']
            self._meta['columns'][col] = column
            # modify mask meta
            self._meta['masks'][name]['items'].append(
                {'properties': {'created': True},
                 'source':     'columns@{}'.format(col),
                 'text':       {text_key: ''}})
            self._meta['sets'][name]['items'].append('columns@{}'.format(col))
            self.set_variable_text(col, label, text_key)
            self._data[col] = '' if source0['type'] == 'delimited set' else np.NaN
        if self._dimensions_comp and not self._dimensions_comp == 'ignore':
            self.dimensionize()
        return None

    # text_keys and texts manipulation
    # ------------------------------------------------------------------------

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

    @staticmethod
    def _force_texts(text_dict, copy_to, copy_from, update_existing):
        new_text_key = None
        for new_tk in reversed(copy_from):
            if new_tk in text_dict.keys():
                if new_tk in ['x edits', 'y edits']:
                    if text_dict[new_tk].get(copy_to):
                        new_text_key = new_tk
                else:
                    new_text_key = new_tk
        if not new_text_key:
            raise ValueError('{} is no existing text_key'.format(copy_from))
        if not copy_to in text_dict.keys() or update_existing:
            if new_text_key in ['x edits', 'y edits']:
                text = text_dict[new_text_key][copy_to]
            else:
                text = text_dict[new_text_key]
            text_dict.update({copy_to: text})

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
                text = text.replace('<br/>', '\n')
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
    def _select_text_keys(text_dict, text_key):
        if not any(tk in text_dict for tk in text_key):
            msg = 'Cannot select {}. A variable does not contain any of it.'
            raise ValueError(msg.format(text_key))
        for tk in text_dict.keys():
            if not tk in ['x edits', 'y edits']:
                if not tk in text_key:
                    text_dict.pop(tk)
            else:
                for etk in text_dict[tk].keys():
                    if not etk in text_key:
                        text_dict[tk].pop(etk)

    @modify(to_list='text_key')
    @verify(text_keys='text_key')
    def select_text_keys(self, text_key=None):
        """
        Cycle through all meta ``text`` objects keep only selected text_key.

        Parameters
        ----------
        text_key : str / list of str, default None
            {None, 'en-GB', 'da-DK', 'fi-FI', 'nb-NO', 'sv-SE', 'de-DE'}
            The text_keys which should be kept.
        Returns
        -------
        None
        """
        if not text_key: text_key = self.valid_tks
        text_func = self._select_text_keys
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

        collection = 'masks' if self.is_array(name) else 'columns'
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
                if name == item['source'].split('@')[-1]:
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
        if not self.is_array(name) and self._is_array_item(name):
            name = self._maskname_from_item(name)
        use_array = self.is_array(name)

        valuesobj = self._get_value_loc(name)
        new_valuesobj = []

        if not text_key:
            if not axis_edit:
                text_key = [self.text_key]
            else:
                text_key = valuesobj[0]['text'].keys()

        text_key = [tk for tk in text_key if tk not in ['x edits', 'y edits']]

        if self.codes(name):
            ignore = [k for k in renamed_vals.keys() if k not in self.codes(name)]
            if ignore:
                msg = 'Warning: Cannot set new value texts... '
                msg = msg + "Codes {} not found in values object of '{}'!"
                warnings.warn(msg)
        else:
            msg = '{} has empty values object, allowing arbitrary values meta!'
            msg = msg + ' ...falling back to extend_values() now!'
            warnings.warn(msg.format(name))
            for tk in text_key:
                for k, v in renamed_vals.items():
                    self.extend_values(name, (k, v), tk)
            return None


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

            >>> {1: 'new label for item #1',
            ...  5: 'new label for item #5'}
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
    def clear_factors(self, name):
        """
        Remove all factors set in the variable's ``'values'`` object.

        Parameters
        ----------
        name : str
            The column variable name keyed in ``_meta['columns']`` or
            ``_meta['masks']``.

        Returns
        -------
        None
        """
        val_loc = self._get_value_loc(name)
        for value in val_loc:
            value['factor'] = None
        return None

    @verify(variables={'name': 'both'})
    def set_factors(self, name, factormap, safe=False):
        """
        Apply numerical factors to (``single``-type categorical) variables.

        Factors can be read while aggregating descrp. stat. ``qp.Views``.

        Parameters
        ----------
        name : str
            The column variable name keyed in ``_meta['columns']`` or
            ``_meta['masks']``.
        factormap : dict
            A mapping of ``{value: factor}`` (``int`` to ``int``).
        safe : bool, default False
            Set to ``True`` to prevent setting factors to the ``values`` meta
            data of non-``single`` type variables.

        Returns
        -------
        None
        """
        e = False
        if name in self.masks():
            if self._get_subtype(name) != 'single':
                e = True
        else:
            if self._get_type(name) != 'single':
                e = True
        if e:
            if safe:
                err = "Can only set factors to 'single' type categorical variables!"
                raise TypeError(err)
            else:
                return None
        vals = self.codes(name)
        facts = factormap.keys()
        val_loc = self._get_value_loc(name)
        if not all(f in vals for f in facts):
            err = 'At least one factor is mapped to a code that does not exist '
            err += 'in the values object of "{}"!'
            raise ValueError(err.format(name))
        for value in val_loc:
            if value['value'] in factormap:
                value['factor'] = factormap[value['value']]
            else:
                value['factor'] = None
        return None

    # rules and properties
    # ------------------------------------------------------------------------
    @verify(variables={'name': 'both'})
    def get_property(self, name, prop_name, text_key=None):
        """
        """
        mask_ref = self._meta['masks']
        col_ref = self._meta['columns']
        if not text_key: text_key = self.text_key
        valid_props = ['base_text', 'created', 'recoded_net', 'recoded_stat',
                       'recoded_filter', '_no_valid_items', '_no_valid_values',
                       'simple_org_expr']
        if prop_name not in valid_props:
            raise ValueError("'prop_name' must be one of {}".format(valid_props))
        has_props = False
        if self.is_array(name):
            if 'properties' in mask_ref[name]:
                has_props = True
                meta_ref = mask_ref[name]
        else:
            if 'properties' in col_ref[name]:
                has_props = True
                meta_ref = col_ref[name]
        if has_props:
            p = meta_ref['properties'].get(prop_name, None)
            if p:
                if prop_name == 'base_text' and isinstance(p, dict):
                    try:
                        p = p[text_key]
                    except:
                        p = p[self.text_key]
            return p
        else:
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
        valid_props = ['base_text', '_no_valid_items', '_no_valid_values']
        if prop_name not in valid_props:
            raise ValueError("'prop_name' must be one of {}".format(valid_props))
        prop_update = {prop_name: prop_value}
        if self.is_array(name):
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
            ``Quantipy.View.dataframe``\s, respecting the provided order.
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

    def empty(self, name, condition=None):
        """
        Check variables for emptiness (opt. restricted by a condition).

        Parameters
        ----------
        name : (list of) str
            The mask variable name keyed in ``_meta['columns']``.
        condition : Quantipy logic expression, default None
            A logical condition expressed as Quantipy logic that determines
            which subset of the case data rows to be considered.

        Returns
        -------
        empty : bool
        """
        empty = []
        if not isinstance(name, list): name = [name]
        return_bool = len(name) == 1
        if condition:
            df = pd.DataFrame(self[self.take(condition), name])
        else:
            df = self._data
        for n in name:
            if df[n].count() == 0:
                empty.append(n)
        if return_bool:
            return bool(empty)
        else:
            return empty

    @modify(to_list='name')
    @verify(variables={'name': 'masks'})
    def empty_items(self, name, condition=None, by_name=True):
        """
        Test arrays for item emptiness (opt. restricted by a condition).

        Parameters
        ----------
        name : (list of) str
            The mask variable name keyed in ``_meta['masks']``.
        condition : Quantipy logic expression, default None
            A logical condition expressed as Quantipy logic that determines
            which subset of the case data rows to be considered.
        by_name : bool, default True
            Return array items by their name or their index.

        Returns
        -------
        empty : list
            The list of empty items by their source names or positional index
            (starting from 1!, mapped to their parent mask name if more than
            one).
        """
        empty = {}
        if condition:
            df = self[self.take(condition), name].copy()
        else:
            df = self._data.copy()
        for n in name:
            test_df = df[self.unroll(n)].sum()
            slicer = test_df == 0
            empty_items = test_df.loc[slicer].index.values.tolist()
            if not by_name: empty_items = [self.item_no(i) for i in empty_items]
            if empty_items: empty[n] = empty_items
        if empty:
            return empty[name[0]] if len(name) == 1 else empty
        else:
            return None

    @verify(variables={'arrays': 'masks'})
    def hide_empty_items(self, condition=None, arrays=None):
        """
        Apply ``rules`` meta to automatically hide empty array items.

        Parameters
        ----------
        name : (list of) str, default None
            The array mask variable names keyed in ``_meta['masks']``. If not
            explicitly provided will test all array mask definitions.
        condition : Quantipy logic expression
            A logical condition expressed as Quantipy logic that determines
            which subset of the case data rows to be considered.

        Returns
        -------
        None
        """
        if not arrays: arrays = self.masks()
        if arrays and not isinstance(arrays, list): arrays = [arrays]
        empty_items = self.empty_items(arrays, condition, False)
        if empty_items:
            if isinstance(empty_items, list):
                empty_items = {arrays[0]: empty_items}
            for arr, items in empty_items.items():
                if len(items) == len(self.sources(arr)):
                    self.set_property(arr, '_no_valid_items', True, True)
                self.hiding(arr, items, axis='x', hide_values=False)
        return None

    def fully_hidden_arrays(self):
        """
        Get all array definitions that contain only hidden items.

        Returns
        -------
        hidden : list
            The list of array mask names.
        """
        hidden = []
        for m in self.masks():
            invalid = self.get_property(m, '_no_valid_items')
            if invalid: hidden.append(m)
        return hidden

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
            ``Quantipy.View.dataframe``\s.
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
            collection = 'columns' if not self.is_array(n) else 'masks'
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

    @modify(to_list=['name', 'fix'])
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
            is_array = self.is_array(n)
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
                if not is_array:
                    n_fix = self._clean_codes_against_meta(n, fix)
                else:
                    n_fix = self._clean_items_against_meta(n, fix)
                rule_update = {'sortx': {'ascending': ascending,
                                         'within': within,
                                         'between': between,
                                         'fixed': n_fix,
                                         'sort_on': on,
                                         'with_weight': sort_by_weight}}
                self._meta[collection][n]['rules']['x'].update(rule_update)
        return None

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

    @verify(variables={'var': 'both', 'ignore': 'both'})
    def set_missings(self, var, missing_map='default', ignore=None):
        """
        Flag category definitions for exclusion in aggregations.

        Parameters
        ----------
        var : str or list of str
            Variable(s) to apply the meta flags to.
        missing_map: 'default' or list of codes or dict of {'flag': code(s)}, default 'default'
            A mapping of codes to flags that can either be 'exclude' (globally
            ignored) or 'd.exclude' (only ignored in descriptive statistics).
            Codes provided in a list are flagged as 'exclude'.
            Passing 'default' is using a preset list of (TODO: specify) values
            for exclusion.
        ignore : str or list of str, default None
            A list of variables that should be ignored when applying missing
            flags via the 'default' list method.

        Returns
        -------
        None
        """
        var = self.unroll(var)
        ignore = self.unroll(ignore, both='all')
        if not missing_map:
            for v in var:
                if 'missings' in self._meta['columns'][v]:
                    del self._meta['columns'][v]['missings']
        elif missing_map == 'default':
            self._set_default_missings(ignore)
        else:
            if isinstance(missing_map, list):
                missing_map = {'exclude': missing_map}
            for v in var:
                if v in ignore: continue
                missing_map = self._clean_missing_map(v, missing_map)
                if self._has_missings(v):
                    self._meta['columns'][v].update({'missings': missing_map})
                else:
                    self._meta['columns'][v]['missings'] = missing_map
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
        new_meta['info']['dataset'] = {'name': ''}
        dname = '{}_derotate'.format(meta['info']['dataset']['name'])
        new_meta['info']['dataset']['name'] = dname
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

        if self.is_array(old_var):
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

            >>> mapper = [{'q14_1': ['q14_1_1', 'q14_1_2', 'q14_1_3']},
            ...           {'q14_2': ['q14_2_1', 'q14_2_2', 'q14_2_3']},
            ...           {'q14_3': ['q14_3_1', 'q14_3_2', 'q14_3_3']}]

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
        path_json = os.path.join(ds.path, ''.join([ds.name, '.json']))
        path_csv = os.path.join(ds.path, ''.join([ds.name,  '.csv']))
        ds.write_quantipy(path_json, path_csv)

        return ds

    # ------------------------------------------------------------------------
    # DATA MANIPULATION/HANDLING
    # ------------------------------------------------------------------------

    def make_dummy(self, var, partitioned=False):
        if not self.is_array(var):
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
            new_ds = self.clone()
            new_ds._data = filtered_data
            new_ds.filtered = alias
            return new_ds

    @modify(to_list=['variables'])
    def subset(self, variables=None, from_set=None, inplace=False):
        """
        Create a cloned version of self with a reduced collection of variables.

        Parameters
        ----------
        variables : str or list of str, default None
            A list of variable names to include in the new DataSet instance.
        from_set : str
            The name of an already existing set to base the new DataSet on.

        Returns
        -------
        subset_ds : qp.DataSet
            The new reduced version of the DataSet.
        """
        if not (variables or from_set) or (variables and from_set):
            err = "Must pass either 'variables' or 'from_set'!"
            raise ValueError(err)
        subset_ds = self.clone() if not inplace else self
        sets = subset_ds._meta['sets']
        if variables:
            from_set = 'subset'
            subset_ds.create_set(setname='subset', included=variables)
        else:
            if not from_set in sets:
                err = "'{}' not found in meta 'sets' collection!"
                raise KeyError(err.format(from_set))
            variables = [v.split('@')[-1] for v in sets[from_set]['items']]
        all_vars = subset_ds.columns() + subset_ds.masks()
        for var in all_vars:
            if not var in variables:
                if not self._is_array_item(var): subset_ds.drop(var)
        sets['data file']['items'] = sets[from_set]['items']
        del subset_ds._meta['sets'][from_set]

        if not inplace:
            return subset_ds
        else:
            return None

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
    # BATCH HANDLERS
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
        check = batches.get(name.decode('utf8'))
        if not check:
            check = batches.get(name)
        if not check:
            raise KeyError('No Batch found named {}.'.format(name))
        return qp.Batch(self, name)

    @modify(to_list='batches')
    def populate(self, batches='all', verbose=True):
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
        dk = self.name
        meta = self._meta
        data = self._data
        stack = qp.Stack(name='aggregations', add_data={dk: (data, meta)})
        batches = stack._check_batches(dk, batches)

        for name in batches:
            batch = self._meta['sets']['batches'][name]
            xys = batch['x_y_map']
            fs = batch['x_filter_map']
            fy = batch['y_filter_map']
            f  = batch['filter']
            my  = batch['yks']

            total_len = len(xys) + len(batch['y_on_y'])
            for idx, xy in enumerate(xys, start=1):
                x, y = xy
                if x == '@':
                    stack.add_link(dk, fs[y[0]], x='@', y=y)
                else:
                    stack.add_link(dk, fs[x], x=x, y=y)
                if verbose:
                    done = float(idx) / float(total_len) *100
                    print '\r',
                    time.sleep(0.01)
                    print  'Batch [{}]: {} %'.format(name, round(done, 1)),
                    sys.stdout.flush()
            for idx, y_on_y in enumerate(batch['y_on_y'], len(xys)+1):
                stack.add_link(dk, fy[y_on_y], x=my[1:], y=my)
                if verbose:
                    done = float(idx) / float(total_len) *100
                    print '\r',
                    time.sleep(0.01)
                    print  'Batch [{}]: {} %'.format(name, round(done, 1)),
                    sys.stdout.flush()
            if verbose:
                print '\n'
        return stack

# ============================================================================

    def parrot(self):
        from IPython.display import Image
        from IPython.display import display
        try:
            return display(Image(url="https://m.popkey.co/3a9f4b/jZZ83.gif"))
        except:
            print ':sad_parrot: Looks like the parrot url is not longer there!'
