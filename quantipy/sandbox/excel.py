# -* coding: utf-8 -*-

import os # for testing only
import json # for testing only
import numpy as np
import pandas as pd
import quantipy as qp
import weakref
import re

from quantipy.core.tools.qp_decorators import lazy_property
from sandbox import ChainManager
from xlsxwriter import Workbook
from xlsxwriter.worksheet import Worksheet
from xlsxwriter.utility import xl_rowcol_to_cell
from itertools import izip, dropwhile, groupby
from operator import itemgetter
from functools import wraps
from excel_formats import ExcelFormats
from excel_formats_constants import DEFAULT_ATTRIBUTES

import warnings; warnings.simplefilter('ignore')

try:
    from functools import lru_cache
except ImportError:
    from functools32 import lru_cache


TEST_SUFFIX = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split()
TEST_PREFIX = ' ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split()

CD_TRANSMAP = {'en-GB': {'cc':    'Cell Contents',
                         'N':     'Counts',
                         'c%':    'Column Percentages',
                         'r%':    'Row Percentages',
                         'str':   'Statistical Test Results',
                         'cp':    'Column Proportions',
                         'cm':    'Means',
                         'stats': 'Statistics',
                         'mb':    'Minimum Base',
                         'sb':    'Small Base'},
               'fr-FR': {'cc':    'Contenu cellule',
                         'N':     'Total',
                         'c%':    'Pourcentage de colonne',
                         'r%':    'Pourcentage de ligne',
                         'str':   u'Résultats test statistique',
                         'cp':    'Proportions de colonne',
                         'cm':    'Moyennes de colonne',
                         'stats': 'Statistiques',
                         'mb':    'Base minimum',
                         'sb':    'Petite base'}}

TOT_REP = [("'@H'", u'\u25BC'), ("'@L'", u'\u25B2')]

ARROW_STYLE = {"'@H'": 'DOWN', "'@L'": 'UP'}

# Initialization data to pass to the worksheet.
SHEET_ATTR = ('str_table',
              'worksheet_meta',
              'optimization',
              'tmpdir',
              'date_1904',
              'strings_to_numbers', 
              'strings_to_formulas',
              'strings_to_urls', 
              'nan_inf_to_errors',
              'default_date_format',
              'default_url_format',
              'excel2003_style',
              'remove_timezone',
              'constant_memory'
              )

#~ create_toc=False,        --> toc         (Excel)
#~ annotations={},          --> annotations (Sheet)

# TODO: formatting
# --------------------
# TODO: add only_first option for y-keys + differing y-keys (!!!????)
# y_key_show = 'first', 'all', 'on_diff'

# TODO: rounding --> find universal qp approach so pptx shows same result

# TODO: grouped_views=None,

# TODO: table_properties=None,

# TODO: italicise_level=None,

# TODO: decimals=None,

# TODO: mask_label_format=None,

# TODO: extract_mask_label=False,

# TODO: show_cell_details=False   --> details -- need discussion # what happens with multi y-axes?


class Excel(Workbook):
    # TODO: docstring

    def __init__(self, filename, toc=False, details=False, dummy_rows=True,
                 **kwargs):
        super(Excel, self).__init__()
        self.filename = filename
        self.toc = toc
        self.details = details
        self.dummy_rows = dummy_rows

        self.properties = dict()
        for attr, default in DEFAULT_ATTRIBUTES.iteritems():
            self.properties[attr] = kwargs.get(attr, default) 

        self._formats = ExcelFormats(**self.properties)

    def __repr__(self):
        return 'Excel(%r)' % self.filename

    def __str__(self):
        return '%s' % self.filename

    def __del__(self):
        del self

    def add_chains(self, chains, sheet_name, annotations=None):
        self._write_chains(chains, sheet_name, annotations=annotations)

    def _write_chains(self, chains, sheet_name, annotations=None):

        worksheet = Sheet(self, chains, sheet_name, self.details,
                          annotations=annotations)

        init_data = {attr: getattr(self, attr, None) for attr in SHEET_ATTR}
        init_data.update({'name': sheet_name,
                          'index': len(self.worksheets_objs)})
        worksheet._initialize(init_data)

        self.worksheets_objs.append(worksheet)

        worksheet.write_chains()

        del worksheet

    def _write_toc(self):
        # sheet_names = (sheet.name for sheet in self.worksheets_objs)
        # for sheet in self.worksheets_objs:
        #     print sheet.name
        raise NotImplementedError('_write_toc')

    @lru_cache()
    def _add_format(self, format_):
        return self.add_format(format_)

    def close(self):
        if self.toc:
            self._write_toc()
        super(Excel, self).close()


class Sheet(Worksheet):
    # TODO: docstring

    def __init__(self, excel, chains, sheet_name, details, annotations=None):
        super(Sheet, self).__init__()
        self.excel = excel
        self.chains = chains
        self.sheet_name = sheet_name
        self.details = details
        self.annotations = annotations
        self.row = 4
        self.column = 0
        self._freeze_loc = None
        self._columns = None
        self._test_letters = None
        self._column_edges = None
        self._view_keys = None
        self._group_order = None

    @lazy_property
    def test_letters(self):
        return self.chains[0].sig_test_letters

    @property
    def column_edges(self):
        if self._column_edges is None:
            self._column_edges = []
        return self._column_edges

    def process_args(self, *args):
        if isinstance(args[-1], dict):
            return args[:-1] + (self.excel._add_format(args[-1]), )
        return args

    def write(self, *args):
        super(Sheet, self).write(*self.process_args(*args))

    def write_row(self, *args):
        super(Sheet, self).write_row(*self.process_args(*args))

    def merge_range(self, *args):
        super(Sheet, self).merge_range(*self.process_args(*args))

    def write_chains(self):
        # TODO: docstring
        if self.annotations:
            for idx, ann in enumerate(self.annotations):
                self.write(idx, 0, ann)

        for i, chain in enumerate(self.chains):

            columns = chain.dataframe.columns
            # make y-axis writing availbale to all chains
            if i == 0:
                self._set_freeze_loc(columns)
                self._set_columns(columns)

            # write frame
            box = Box(self, chain, self.row, self.column)
            box.to_sheet(columns=(i==0))

            del box

        self.freeze_panes(*self._freeze_loc)

        self.hide_gridlines(2)

    def _set_columns(self, columns):
        # TODO: make column width optional --> Properties().
        self.set_column(self.column, self.column, 40)
        self.set_column(self.column + 1, self.column + columns.size, 10)

    def _set_freeze_loc(self, columns):
        l_0 = columns.get_level_values(columns.nlevels - 1).values
        first_column_size = len(np.extract(np.argmin(l_0), l_0))
        self._freeze_loc = ((self.row + columns.nlevels),
                            (self.column + first_column_size + 1))


class Box(object):
    # TODO: docstring

    # _properties_cache = WeakValueDictionarel_y()

    __slots__ = ('sheet', 'chain', '_single_columns','_column_edges',
                 '_lazy_index', '_lazy_columns',
                 '_lazy_values', '_lazy_contents', '_lazy_row_contents',
                 '_lazy_is_weighted', '_lazy_shape', '_lazy_has_tests')

    def __init__(self, sheet, chain, row, column):
        self.sheet = sheet
        self.chain = chain
        self._single_columns = None
        self._column_edges = None

    @property
    def single_columns(self):
        if self._single_columns is None:
            self._single_columns = []
        return self._single_columns

    @property
    def column_edges(self):
        return self.sheet.column_edges

    @lazy_property
    def index(self):
        return self.chain.dataframe.index

    @lazy_property
    def columns(self):
        return self.chain.dataframe.columns

    @lazy_property
    def values(self):
        return self.chain.dataframe.values

    @lazy_property
    def contents(self):
        return self.chain.contents

    @lazy_property
    def row_contents(self):
        return self.contents['rows']

    @lazy_property
    def is_weighted(self):
        return any(x['is_weighted'] for x in self.row_contents.itervalues())

    @lazy_property
    def shape(self):
        return self.chain.dataframe.shape

    @lazy_property
    def has_tests(self):
        return self.chain.dataframe.columns.nlevels % 2

    def to_sheet(self, columns):
        # TODO: Doc string
        if columns:
            self._write_columns()
        self._write_rows()

    def _write_columns(self):
        format_ = self.sheet.excel._formats.y
        column = self.sheet.column + 1
        nlevels = self.columns.nlevels
        for level_id in xrange(nlevels):
            row = self.sheet.row + level_id
            is_tests =  self.has_tests and (level_id == (nlevels - 1))
            is_values = (level_id % 2) or is_tests
            if (level_id == 0) or is_values:
                group_sizes = []
            flat = self.columns.get_level_values(level_id).values.flat
            left = flat.coords[0]
            data = flat.next()
            while True:
                next_ = data
                while data == next_:
                    try:
                        next_ = flat.next()
                    except StopIteration:
                        next_ = None
                right = flat.coords[0] - 2
                if next_ is None:
                    right += 1
                data = self._cell(data)
                if level_id == 0:
                    if left == right:
                        self.single_columns.append(left)
                    self.column_edges.append(right + 1)
                if left not in self.single_columns:
                    if group_sizes and not is_values:
                        limit = right
                        
                        while right != limit:
                            self.sheet.merge_range(row, column + left,
                                                   row, column + right,
                                                   data, format_)
                            left, right = group_sizes.pop(0)
                    if left == right:
                        self.sheet.write(row, column + left, data, format_)
                    else:
                        self.sheet.merge_range(row, column + left,
                                               row, column + right,
                                               data, format_)
                    if is_values:
                        group_sizes.append((left, right))
                data = next_
                left = right + 1
                if next_ is None:
                    break
        for cindex in self.single_columns:
            level = - (1 + self.has_tests)
            data = self._cell(self.columns.get_level_values(level)[cindex])
            self.sheet.merge_range(row - nlevels + 1, column + cindex,
                                   row, column + cindex,
                                   data, format_)
        self.sheet.row = row + 1

    def _write_rows(self):
        column = self.sheet.column

        levels = self.index.get_level_values
        
        self.sheet.write(self.sheet.row, column,
                         levels(0).unique().values[0], 
                         self.sheet.excel._formats.x_left_bold)
        self.sheet.row += 1

        if self.has_tests:
            level_1, values, contents = self._get_dummies(levels(1).values,
                                                          self.values)
        else:
            level_1, values, contents = levels(1).values, self.row_contents

        row_max = max(contents.keys())

        flat = np.c_[level_1.T, values].flat

        bg = True
        bg_required = True
        offset_x = 0
        rel_x, rel_y = flat.coords
        for data in flat:
            row_cont = contents[rel_x]
            if rel_y == 0:
                if data == '':
                    bg = not bg
                    top_required = False
                else:
                    top_required = True
                    bg_required = self._bg(**row_cont)
                formats = []
            name = self._row_format_name(**row_cont)
            format_ = self._format_x_right(name, rel_x, rel_y,
                                           row_max, bg * bg_required,
                                           top_required)
            cell_data = self._cell(data, normalize=self._is_pct(**row_cont))
            self.sheet.write(self.sheet.row + rel_x + offset_x,
                             self.sheet.column + rel_y,
                             cell_data, format_)
            nxt_x, nxt_y = flat.coords
            if rel_x != nxt_x:
                bg = not bg
            rel_x, rel_y = nxt_x, nxt_y
        self.sheet.row += rel_x + offset_x

    def _get_dummies(self, index, values):
        it = iter(zip(xrange(len(index)), index))
        idx, data = next(it)
        group = ''
        dummy = True
        dummy_idx = []
        while True:
            try: 
                ndx, next_ = next(it)
                if next_ == '':
                    if not group:
                        group = data
                    elif self.row_contents[idx]['is_test']:
                        dummy = False
                else:
                    if group and self.row_contents[idx]['is_test']:
                        dummy = False
                    if group and dummy:
                        dummy_idx.append(ndx + len(dummy_idx))
                    if not self.row_contents[idx]['is_c_base']:
                        group = next_
                    dummy = True
                idx, data = ndx, next_
            except StopIteration:
                if group and dummy:
                    dummy_idx.append(ndx + len(dummy_idx) + 1)
                break

        dummy_arr = np.array([[u'' for _ in xrange(len(values[0]))]], dtype=str)
        for idx in dummy_idx:
            try:
                index = np.insert(index, idx, u'')
                values = np.vstack((values[:idx, :], dummy_arr, values[idx:, :]))
            except IndexError:
                index = np.append(index, u'')
                values = np.vstack((values, dummy_arr))
        
        num_dummies = 0
        row_contents = {}
        for key in sorted(self.row_contents):
            if (key + num_dummies) in dummy_idx:
                num_dummies += 1
            row_contents[key+num_dummies] = self.row_contents[key]
        for key in dummy_idx:
            row_contents[key] = {k: v for k, v in row_contents[key-1].iteritems()}
            row_contents[key].update({'is_dummy': True})
        
        return index, values, row_contents

    @lru_cache()
    def _is_pct(self, **contents):
        return contents['is_c_pct'] or contents['is_r_pct']

    @lru_cache()
    def _bg(self, **contents):
        if contents['is_c_base'] or contents['is_net'] or contents['is_sum']:
            return False
        view_types = ('is_counts', 'is_c_pct', 'is_r_pct', 'is_test')
        return any(contents[_] for _ in view_types)

    @lru_cache()
    def _row_format_name(self, **contents):
        result = 'dummy_' if contents.get('is_dummy') else ''
        if contents['is_sum']:
            result += 'sum_'
        if contents['is_c_base']:
            if contents['is_weighted']:
                return result + 'base'
            elif self.is_weighted:
                return result + 'ubase'
            return result + 'base'
        elif contents['is_counts']:
            # net?
            return result + 'count'
        elif contents['is_c_pct'] or contents['is_r_pct']:
            # net?
            return result + 'pct'
        elif contents['is_stat']:
            # type? - mean, meadian, etc.
            return result + 'stat'
        elif contents['is_test']:
            return result + 'test'
        # elif['is_r_base']:
        #     return ?

    def _format_x_right(self, name, rel_x, rel_y, row_max, bg, top):
        if rel_y == 0:
            return self.sheet.excel._formats['x_right_' + name]
        name = self._format_position(rel_x, rel_y, row_max) + name
        if bg:
            name += '_background'
        if not top:
            name += '_no_top'
        return self.sheet.excel._formats[name]

    def _format_position(self, rel_x, rel_y, row_max):
	position = ''
        if rel_y == 1:
	    position = 'left_'
	if rel_y in self.column_edges:
	    position += 'right_'
	if position == '':
	    position = 'interior_'
	if rel_x == 0:
	    position += 'top_'
	if rel_x == row_max:
	    position += 'bottom_'
	return position
    
    @lru_cache()
    # def _cell(self, value, row_index=None):
    def _cell(self, value, normalize=False):
        # if row_index:
        #     normalize = any(self.row_contents[row_index][_]
        #                     for _ in ('is_c_pct', 'is_r_pct'))
        #     return Cell(value, normalize).__repr__()
        return Cell(value, normalize).__repr__()


class Cell(object):

    def __init__(self, data, normalize):
        self.data = data
        self.normalize = normalize

    def __repr__(self):
        try:
            if np.isnan(self.data) or np.isinf(self.data) or self.data == 0:
                return DEFAULT_ATTRIBUTES['frequency_0_rep'] 
        except TypeError:
            pass
        if isinstance(self.data, (str, unicode)):
            return re.sub(r'#pad-\d+', str(), self.data)
        if self.normalize:
            return self.data / 100.
        return self.data


##############################################################################
if __name__ == '__main__':

    PATH_DATA = '../../tests/'
    NAME_PROJ = 'Example Data (A)'
    NAME_META = 'Example Data (A).json'
    NAME_DATA = 'Example Data (A).csv'
    PATH_META = os.path.join(PATH_DATA, NAME_META)
    PATH_DATA = os.path.join(PATH_DATA, NAME_DATA)

    DATA_KEY = ORIENT = 'x'
    FILTER_KEY = 'no_filter'
    # X_KEYS = ['q5_1']
    X_KEYS = ['q5_1', 'q4', 'gender', 'Wave']
    # Y_KEYS = ['@', 'q4']                                        # 1.
    # Y_KEYS = ['@', 'q4', 'q5_2', 'gender', 'Wave']              # 2.
    # Y_KEYS = ['@', 'q4 > gender']                               # 3.
    # Y_KEYS = ['@', 'q4 > gender > Wave']                        # 4.
    Y_KEYS = ['@', 'q4 > gender > Wave', 'q5_1', 'q4 > gender'] # 5.
    TESTS = True

    # WEIGHT = None
    WEIGHT = 'weight_a'

    VIEWS = ('cbase',
             'counts',
             'c%',
             'mean',
             'stddev',  
             'median',
             'counts_sum',
             'c%_sum',
             )

    VIEW_KEYS = ('x|f|x:||%s|cbase' % WEIGHT,
                 'x|f|:||%s|counts' % WEIGHT,
                 'x|f|:|y|%s|c%%' % WEIGHT,
                 'x|t.props.Dim.80|:||%s|test' % WEIGHT,
                 'x|f|x[{1,2,3}]:||%s|No' % WEIGHT,
                 'x|f|x[{1,2,3}]:|y|%s|No' % WEIGHT,
                 'x|t.props.Dim.80|x[{1,2,3}]:||%s|test' % WEIGHT,
                 'x|f|x[{4,5,97}]:||%s|Yes' % WEIGHT,
                 'x|f|x[{4,5,97}]:|y|%s|Yes' % WEIGHT,
                 'x|t.props.Dim.80|x[{4,5,97}]:||%s|test' % WEIGHT,
                 'x|d.mean|x:||%s|mean' % WEIGHT,
                 'x|t.means.Dim.80|x:||%s|test' % WEIGHT,
                 'x|d.stddev|x:||%s|stddev' % WEIGHT,
                 'x|d.median|x:||%s|median' % WEIGHT,
                 'x|f.c:f|x:||%s|counts_sum' % WEIGHT,
                 'x|f.c:f|x:|y|%s|c%%_sum' % WEIGHT,
                )

    weights = [None]
    if WEIGHT is not None:
        VIEW_KEYS = ('x|f|x:|||cbase', ) + VIEW_KEYS
        weights.append(WEIGHT)

    dataset = qp.DataSet(NAME_PROJ, dimensions_comp=False)
    dataset.read_quantipy(PATH_META, PATH_DATA)
    meta, data = dataset.split()
    data = data.head(250)
    stack = qp.Stack(NAME_PROJ, add_data={DATA_KEY: {'meta': meta, 'data': data}})
    stack.add_link(x=X_KEYS, y=Y_KEYS, views=VIEWS, weights=weights)


    rel_to = []
    if 'counts' in VIEWS:
        rel_to.append(None)
    if 'c%' in VIEWS:
        rel_to.append('y')
   
    nets_mapper = qp.ViewMapper(template={'method': qp.QuantipyViews().frequency, 
                                          'kwargs': {'iterators': {'rel_to': rel_to}, 
                                                     'groups': 'Nets'}})
    nets_mapper.add_method(name='No', kwargs={'axis':    'x', 
                                              'logic':   [{'No': [1, 2, 3]}], 
                                              'text':    'Net: No', 
                                              'combine': False})
    stack.add_link(x=X_KEYS[0], y=Y_KEYS, views=nets_mapper, weights=weights)

    nets_mapper = qp.ViewMapper(template={'method': qp.QuantipyViews().frequency, 
                                          'kwargs': {'iterators': {'rel_to': rel_to},
                                                     'groups': 'Nets'}})
    nets_mapper.add_method(name='Yes', kwargs={'axis':    'x', 
                                               'logic':   [{'Yes': [4, 5, 97]}], 
                                               'text':    'Net: Yes',
                                               'combine': False})
    stack.add_link(x=X_KEYS[0], y=Y_KEYS, views=nets_mapper, weights=weights)

    if TESTS:
        test_view = qp.ViewMapper().make_template('coltests')
        view_name = 'test'
        options = {'level': 0.8,
                'metric': 'props',
                # 'test_total': True,
                # 'flag_bases': [30, 100]
                }
        test_view.add_method(view_name, kwargs=options)
        stack.add_link(x=X_KEYS, y=Y_KEYS, views=test_view, weights=weights)


        test_view = qp.ViewMapper().make_template('coltests')
        view_name = 'test'
        options = {'level': 0.8, 'metric': 'means'}
        test_view.add_method(view_name, kwargs=options)
        stack.add_link(x=X_KEYS, y=Y_KEYS, views=test_view, weights=weights)

    # stack.describe().to_csv('d.csv'); stop()

    VIEW_KEYS = ('x|f|x:|||cbase',
                 'x|f|x:||%s|cbase' % WEIGHT,
                 ('x|f|:||%s|counts' % WEIGHT,
                  'x|f|:|y|%s|c%%' % WEIGHT,
                  'x|t.props.Dim.80|:||%s|test' % WEIGHT),
                 ('x|f|x[{1,2,3}]:||%s|No' % WEIGHT,
                  'x|f|x[{1,2,3}]:|y|%s|No' % WEIGHT,
                  'x|t.props.Dim.80|x[{1,2,3}]:||%s|test' % WEIGHT),
                 ('x|f|x[{4,5,97}]:||%s|Yes' % WEIGHT,
                  'x|f|x[{4,5,97}]:|y|%s|Yes' % WEIGHT,
                  'x|t.props.Dim.80|x[{4,5,97}]:||%s|test' % WEIGHT),
                 ('x|d.mean|x:||%s|mean' % WEIGHT,
                  'x|t.means.Dim.80|x:||%s|test' % WEIGHT),
                 'x|d.stddev|x:||%s|stddev' % WEIGHT,
                 'x|d.median|x:||%s|median' % WEIGHT,
                 ('x|f.c:f|x:||%s|counts_sum' % WEIGHT,
                  'x|f.c:f|x:|y|%s|c%%_sum' % WEIGHT),
                )

    chains = ChainManager(stack)

    chains = chains.get(data_key=DATA_KEY, filter_key=FILTER_KEY,
                        x_keys=X_KEYS, y_keys=Y_KEYS,
                        views=VIEW_KEYS, orient=ORIENT,
                        )

    chains.paint_all(transform_tests='full')

    # table props - check editability
    table_properties = dict(
                            # bg_color_default='#F5D04C',
                            # bg_color_label='#FF69B4',
                            bg_color_net='#B2DFEE',
                            # bg_color_test='#98FB98'
                           )
    #

    # -------------
    x = Excel('basic_excel.xlsx',
            details='en-GB',
            **table_properties
            # toc=True # not implemented
            )

    x.add_chains(chains,
            'S H E E T',
            annotations=['Ann. 1', 'Ann. 2', 'Ann. 3', 'Ann. 4']
            )

    x.close()
    # -------------
