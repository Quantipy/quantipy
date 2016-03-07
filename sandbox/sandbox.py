import numpy as np
import pandas as pd
import quantipy as qp

from quantipy.core.helpers.functions import emulate_meta

from quantipy.core.tools.view.logic import (
    has_any, has_all, has_count,
    not_any, not_all, not_count,
    is_lt, is_ne, is_gt,
    is_le, is_eq, is_ge,
    union, intersection, get_logic_index)

class DataSet(object):
    def __init__(self, name):
        self.name = name
        self._data = None
        self._meta = None
        self._tk = None
        self.path = None

    def read(self, path_data, path_meta):
        self._data = qp.dp.io.load_csv(path_data+'.csv')
        self._meta = qp.dp.io.load_json(path_meta+'.json')
        self.path = '/'.join(path_data.split('/')[:-1])
        self._tk = self._meta['lib']['default text']
        self._data['@1'] = np.ones(len(self._data))

    def data(self):
        return self._data

    def meta(self):
        return self._meta

    def _get_type(self, var):
        if var in self._meta['masks'].keys():
            return self._meta['masks'][var]['type']
        else:
             return self._meta['columns'][var]['type']

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

    def _is_numeric(self, var):
        return self._get_type(var) in ['float', 'int']

    def _get_label(self, var, text_key=None):
        if text_key is None: text_key = self._tk
        if self._get_type(var) == 'array':
            return self._meta['masks'][var]['text'][text_key]
        else:
            return self._meta['columns'][var]['text'][text_key]

    def _get_valuemap(self, var, text_key=None, non_mapped=None):
        if text_key is None: text_key = self._tk
        if self._get_type(var) == 'array':
            vals = self._meta['lib']['values'][var]
        else:
            vals = emulate_meta(self._meta,
                                self._meta['columns'][var].get('values', None))
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
        if non_mapped in ['texts', 'lists', None]:
            items_texts = [self._meta['columns'][i]['text'][text_key]
                           for i in items]
        if non_mapped == 'lists':
            return items, items_texts
        else:
            return zip(items, items_texts)

    @staticmethod
    def _pad_meta_list(meta_list, pad_to_len):
        return meta_list + ([''] * pad_to_len)

    def _get_meta(self, var, restrict_to=None,  text_key=None):

        if text_key is None: text_key = self._tk
        var_type = self._get_type(var)
        label = self._get_label(var, text_key)
        if not self._is_numeric(var):
            codes, texts = self._get_valuemap(var, non_mapped='lists')
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
                elements = [items, items_texts, codes, texts]
                columns = ['items', 'item texts', 'codes', 'texts']
            else:
                idx_len = len(codes)
                elements = [codes, texts]
                columns = ['codes', 'texts']
            meta_s = [pd.Series(element, index=range(0, idx_len))
                      for element in elements]
            meta_df = pd.concat(meta_s, axis=1)
            meta_df.columns = columns
        else:
            meta_df = pd.DataFrame()
        meta_df.index.name = var_type
        meta_df.columns.name = '{}: {}'.format(var, label)
        return meta_df

    def filter(self, condition, inplace=False):
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
            self._data = filtered_data
        else:
            new_ds = DataSet(self.name)
            new_ds._data = filtered_data
            new_ds._meta = self.meta
            return new_ds



    def link(self, f='no_filter', x=None, y=None):
        l = Link(f, x, y)
        l.data = self.data
        l.meta = self.meta

        # l.stack = stack
        # l.quantify()

        return l

class Link(dict):
    def __init__(self, f, x, y):
        self.f = f
        self.x = x
        self.y = y
