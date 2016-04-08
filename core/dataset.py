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
from cache import Cache

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
        if isinstance(var, (unicode, str)):
            if not self._is_array(var):
                return self._data[var]
            else:
                items = self._get_itemmap(var, non_mapped='items')
                return self._data[items]
        else:
            return self._data[var]

    # ------------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------------
    def read_quantipy(self, path_meta, path_data):
        self._meta, self._data = r_quantipy(path_meta+'.json', path_data+'.csv')
        self._set_file_info(path_data, path_meta)

    def read_dimensions(self, path_meta, path_data):
        self._meta, self._data = r_dimensions(path_meta+'.mdd', path_data+'.ddf')
        self._set_file_info(path_data, path_meta)

    def read_spss(self, path_sav, **kwargs):
        self._meta, self._data = r_spss(path_sav+'.sav', **kwargs)
        self._set_file_info(path_data)

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

    def data(self):
        return self._data

    def meta(self):
        return self._meta

    def cache(self):
        return self._cache

    def split(self, save=False):
        meta, data = self._meta, self._data
        if save:
            path = self.path
            name = self.name
            w_quantipy(meta, data, path+name+'.json', path+name+'.csv')
        return meta, data

    # ------------------------------------------------------------------------
    # META INSPECTION/MANIPULATION/HANDLING
    # ------------------------------------------------------------------------
    def set_missings(self, var, missing_map=None):
        """
        Flag category defintions in the meta for exclusion in aggregations.

        Parameters
        ----------
        var : str or list of str
            Variable(s) to apply the meta flags to.
        missing_map: dict of {code: 'flag'} or {(tuple of codes): 'flag'}
            A mapping of codes to flags that can either be 'exclude' (globally
            ignored) or 'd.exclude' (only ignored in descriptive statistics).

        Returns
        -------
        None
        """
        if missing_map is None:
            missing_map = {}
        if any(isinstance(k, tuple) for k in missing_map.keys()):
            flat_missing_map = {}
            for miss_code, miss_type in missing_map.items():
                if isinstance(miss_code, tuple):
                    for code in miss_code:
                        flat_missing_map[code] = miss_type
                else:
                    flat_missing_map[miss_code] = miss_type
            missing_map = flat_missing_map
        if self._is_array(var):
            var = self._get_itemmap(var, non_mapped='items')
        else:
            if not isinstance(var, list): var = [var]
        for v in var:
            if self._has_missings(v):
                self.meta()['columns'][v].update({'missings': missing_map})
            else:
                self.meta()['columns'][v]['missings'] = missing_map
        return None

    def slice(self, var, slicer):
        values = self._get_value_loc(var)
        new_values = [value for i in slicer for value in values
                      if value['value']==i]
        if self._get_type(var) == 'array':
            self._meta['lib']['values'][var] = new_values
        else:
            self._meta['columns'][var]['values'] = new_values
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
        missings = self._get_missing_map(var)
        if globally:
            return [c for c in missings.keys() if missings[c] == 'exclude']
        else:
            return [c for c in missings.keys()
                    if missings[c] in ['d.exclude', 'exclude']]

    def describe(self, var=None, restrict_to=None, text_key=None):
        """
        Inspect the DataSet's global or variable level structure.
        """
        if text_key is None: text_key = self._tk
        if var is not None:
            return self._get_meta(var, restrict_to, text_key)
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
            if restrict_to:
                types = pd.DataFrame(types[restrict_to]).replace('', np.NaN)
                types = types.dropna()
                types.columns.name = 'count: {}'.format(len(types))
            return types

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
            raise KeyError('Numerical columns do not have "values" meta.')
        loc = self._get_meta_loc(var)
        if not self._is_array(var):
            return emulate_meta(self._meta, loc[var].get('values', None))
        else:
            return emulate_meta(self._meta, loc[var])

    def _get_valuemap(self, var, text_key=None, non_mapped=None):
        if text_key is None: text_key = self._tk
        vals = self._get_value_loc(var)
        if non_mapped in ['codes', 'lists', None]:
            codes = [v['value'] for v in vals]
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

    def _get_meta(self, var, restrict_to=None,  text_key=None):
        if text_key is None: text_key = self._tk
        var_type = self._get_type(var)
        label = self._get_label(var, text_key)
        missings = self._get_missing_map(var)
        if not self._is_numeric(var):
            codes, texts = self._get_valuemap(var, non_mapped='lists')
            if missings:
                missings = [None if code not in missings else missings[code]
                            for code in codes]
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
                columns = ['items', 'item texts', 'codes', 'texts', 'missing type']
            else:
                idx_len = len(codes)
                elements = [codes, texts, missings]
                columns = ['codes', 'texts', 'missing type']
            meta_s = [pd.Series(element, index=range(0, idx_len))
                      for element in elements]
            meta_df = pd.concat(meta_s, axis=1)
            meta_df.columns = columns
            meta_df.index.name = var_type
            meta_df.columns.name = '{}: {}'.format(var, label)
        else:
            meta_df = pd.DataFrame(['N/A'])
            meta_df.index = [var_type]
            meta_df.columns = ['{}: {}'.format(var, label)]
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