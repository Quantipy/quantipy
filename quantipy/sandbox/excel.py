# -* coding: utf-8 -*-

import os # for testing only
import json # for testing only
import numpy as np
import pandas as pd
import quantipy as qp
import weakref
import re
from sandbox import ChainManager
from xlsxwriter import Workbook
from xlsxwriter.worksheet import Worksheet
from xlsxwriter.utility import xl_rowcol_to_cell
from itertools import izip, dropwhile, groupby
from operator import itemgetter
from functools import wraps
from excel_formats import ExcelFormats

import warnings; warnings.simplefilter('once')

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

def lazy_property(func):
    """Decorator that makes a property lazy-evaluated.
    """
    attr_name = '_lazy_' + func.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))
        return getattr(self, attr_name)
    return _lazy_property

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

formats = ExcelFormats()


class Excel(Workbook):
    # TODO: docstring

    def __init__(self, filename, toc=False, details=False):
        super(Excel, self).__init__()
        self.filename = filename
        self.toc = toc
        self.details = details

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

        # Initialization data to pass to the worksheet.
        sheet_attr = ('str_table', 'worksheet_meta', 'optimization', 'tmpdir',
                      'date_1904', 'strings_to_numbers', 'strings_to_formulas',
                      'strings_to_urls', 'nan_inf_to_errors',
                      'default_date_format', 'default_url_format',
                      'excel2003_style', 'remove_timezone', 'constant_memory')

        init_data = {attr: getattr(self, attr, None) for attr in sheet_attr}
        init_data['name'] = sheet_name
        init_data['index'] = len(self.worksheets_objs)
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
                 '_lazy_row_max', '_lazy_index', '_lazy_columns',
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
    def row_max(self):
        return max(self.row_contents.keys())

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
        format_ = formats.y
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
                        left, right = group_sizes.pop(0)
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

        # label
        self.sheet.write(self.sheet.row, column,
                         levels(0).unique().values[0], formats.x_left_bold)
        self.sheet.row += 1

        # categorel_y
        flat = np.c_[levels(1).values.T, self.values].flat

        rel_x, rel_y = flat.coords
        name = self._row_format_name(rel_x)
        for data in flat:
            format = self._format_x_right(name, rel_x, rel_y)
            if format:
                self.sheet.write(self.sheet.row + rel_x,
                                 self.sheet.column + rel_y,
                                 self._cell(data), format)
            nxt_x, nxt_y = flat.coords
            if nxt_x != rel_x:
                name = self._row_format_name(rel_x)
            rel_x, rel_y = nxt_x, nxt_y

        self.sheet.row += rel_x

    def _row_format_name(self, rel_x):
        contents = self.row_contents[rel_x]
        if contents['is_c_base']:
            if contents['is_weighted']:
                return 'base'
            elif self.is_weighted:
                return 'ubase'
            return 'base'
        elif contents['is_counts']:
            # net?
            return 'count'
        elif contents['is_c_pct'] or contents['is_r_pct']:
            # net?
            return 'pct'
        elif contents['is_stat']:
            # type? - mean, meadian, etc.
            return 'stat'
        elif contents['is_test']:
            return 'test'
        # elif['is_r_base']:
        #     return ?

    def _format_x_right(self, name, rel_x, rel_y):
        if rel_y == 0:
            return formats.get('x_right_' + name)
        return formats.get(self._format_position(rel_x, rel_y) + name)

    def _format_position(self, rel_x, rel_y):
	position = ''
        if rel_y == 1:
	    position = 'left_'
	if rel_y in self.column_edges:
	    position += 'right_'
	if position == '':
	    position = 'interior_'
	if rel_x == 0:
	    position += 'top_'
	if rel_x == self.row_max:
	    position += 'bottom_'
	return position

    @staticmethod
    def _cell(value):
        return Cell(value).__repr__()


class Cell(object):

    def __init__(self, data):
        self.data = data

    def __repr__(self):
        try:
            if np.isnan(self.data) or np.isinf(self.data):
                return '-'
        except TypeError:
            pass
        if isinstance(self.data, (str, unicode)):
            return re.sub(r'#pad-\d+', str(), self.data)
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

    # WEIGHT = None
    WEIGHT = 'weight_a'


    VIEWS = ('cbase', 'counts', 'c%', 'mean', 'median')
    VIEW_KEYS = ('x|f|x:||%s|cbase' % WEIGHT,
            'x|f|:||%s|counts' % WEIGHT, 'x|d.mean|x:||%s|mean' % WEIGHT,
            'x|d.median|x:||%s|median' % WEIGHT, 'x|f.c:f|x:||%s|counts_sum' % WEIGHT,
            'x|t.props.Dim.80|:||%s|test' % WEIGHT, 'x|t.means.Dim.80|x:||%s|test' % WEIGHT
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

    chains = ChainManager(stack)

    chains = chains.get(data_key=DATA_KEY, filter_key=FILTER_KEY,
            x_keys=X_KEYS, y_keys=Y_KEYS,
            views=VIEW_KEYS, orient=ORIENT)

    chains.paint_all(transform_tests='full')

    # -------------
    x = Excel('basic_excel.xlsx',
            details='en-GB',
            # toc=True # not implemented
            )

    x.add_chains(chains,
            'S H E E T',
            annotations=['Ann. 1', 'Ann. 2', 'Ann. 3', 'Ann. 4']
            )

    x.close()
    # -------------
