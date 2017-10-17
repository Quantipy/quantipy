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

try:
    from functools import lru_cache
except ImportError:
    from backports.functools_lru_cache import lru_cache



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
        self.start_row = 4
        self.start_column = 1
        self.row = 4
        self.column = 1
        self._freeze_loc = None
        self._columns = None
        self._test_letters = None
        self._view_keys = None
        self._group_order = None

    @lazy_property
    def test_letters(self):
        return self.chains[0].sig_test_letters

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
            if self.row == self.start_row:
                self._set_freeze_loc(columns)
                self._set_columns(columns)

            # write frame
            box = Box(self, chain, self.row, self.column)
            box.to_sheet(columns=(i==0))
            self.row = box.row

            del box

        # freeze panes
        self.freeze_panes(*self._freeze_loc)

    def _set_columns(self, columns):
        # TODO: make column width optional --> Properties().
        self.set_column(self.start_column - 1, self.start_column - 1, 40)
        self.set_column(self.start_column,
                        self.start_column + columns.size - 1,
                        10)

    def _set_freeze_loc(self, columns):
        l_0 = columns.get_level_values(columns.nlevels - 1).values
        first_column_size = len(np.extract(np.argmin(l_0), l_0))
        has_tests = bool(self.test_letters)
        self._freeze_loc = ((self.start_row + columns.nlevels + has_tests),
                            (self.start_column + first_column_size))


class Box(object):
    # TODO: docstring

    # _properties_cache = WeakValueDictionary()

    __slots__ = ('sheet', 'chain', '_frame', '_contents', '_row', '_column',
                 '_single_columns', '_test_letters', '_lazy_test_letters',
                 '_lazy_row_contents', '_lazy_is_weighted')

    def __init__(self, sheet, chain, row, column):
        self.sheet = sheet
        self.chain = chain
        self._frame = None
        self._contents = None
        self._row = None
        self._column = None
        self._single_columns = None
        self._test_letters = None

    @property
    def frame(self):
        if self._frame is None:
            self._frame = self.chain.dataframe
        return self._frame

    @property
    def contents(self):
        if self._contents is None:
            self._contents = self.chain.contents
        return self._contents

    @property
    def row(self):
        if self._row is None:
            self._row = self.sheet.row
        return self._row

    @property
    def column(self):
        if self._column is None:
            self._column = self.sheet.column
        return self._column

    @property
    def single_columns(self):
        if self._single_columns is None:
            self._single_columns = []
        return self._single_columns

    @lazy_property
    def test_letters(self):
        return self.sheet.test_letters

    @lazy_property
    def row_contents(self):
        return self.chain.contents['rows']

    @lazy_property
    def is_weighted(self):
        return any(x['is_weighted'] for x in self.row_contents.itervalues())

    def to_sheet(self, columns):
        # TODO: Doc string
        if columns:
            self._write_columns()
        self._write_rows()

    def _write_columns(self):
        row = self.row
        for level_id in xrange(self.frame.columns.nlevels):
            values = self.frame.columns.get_level_values(level_id).values
            if level_id % 2:
                self._write_column_headers(values, row)
                row += 1
            else:
                self._write_column_titles(level_id, values, row)
                row += 1
        self._write_single_columns(row, values)
        if self.test_letters:
            self.sheet.write_row(self.row, self.column + 1,
                                 self.test_letters, formats.y)
            self._row += 1

    def _write_column_headers(self, values, row):
        for left, right, value in self._values_iter(values):
            if left not in self.single_columns:
                if right == left:
                    self.sheet.write(row, self.column + left,
                                     self._cell(value), formats.y)
                else:
                    self.sheet.merge_range(row, self.column + left, row,
                                           self.column + right,
                                           self._cell(value), formats.y)

    def _write_column_titles(self, index, values, row):
        state = next_ = None
        for idx, value in enumerate(values):
            if state is None:
                state = idx
            if idx < (values.size - 1):
                next_ = values[idx + 1]
                if next_ == value:
                    continue
            if state == idx:
                if idx not in self.single_columns:
                    self.single_columns.append(self.column + idx -1)
                state = None
                continue
            if index > 0:
                i = self.frame.columns.nlevels - 1
                x = (self.frame
                        .columns
                        .get_level_values(index+1)
                        .values[state:idx+1])
                size = x.size
                unique = np.unique(x)
                try:
                    chunk = list(self._lindex(x))[unique.size]
                except IndexError:
                    chunk = size
                repeat = size / chunk
                for i in xrange(repeat):
                    offset = chunk * i
                    l = self.column + state + offset
                    r = self.column + state + offset + chunk - 1
                    self.sheet.merge_range(row, l, row, r, self._cell(value),
                                           formats.y)
            else:
                self.sheet.merge_range(row, self.column + state,
                                       row, self.column + idx,
                                       self._cell(value), formats.y)
            state = None

    def _write_single_columns(self, row, values):
        for idx in self.single_columns:
            offset = (bool(self.test_letters) + 1) % 2
            self.sheet.merge_range(self.row, self.column + idx,
                                   row - offset, self.column + idx,
                                   self._cell(values[idx]), formats.y)
            self._row += row - self.sheet.start_row

    def _write_rows(self):
        levels = self.frame.index.get_level_values

        # label
        self._write_x_label(levels(0).unique().values[0])
       
        # category
        fl = np.c_[levels(1).values.T, self.frame.values].flat

        for x in fl:
            print x, fl.coords

        raise Exception(" ... ")


        # for i, values in enumerate(self.frame.values):
        #     self._write_row(i, values, levels)

    def _write_x_label(self, value):
        self.sheet.write(self.row, self.column - 1,
                         self._cell(value), formats.x_left_bold)
        self._row += 1

    def _write_row(self, idx, values, levels):
        self.sheet.write(self.row, self.column - 1,
                         self._cell(levels(1)[idx]),
                         self._format_x_right(self.row_contents[idx]))
        # value_formats = self._format_x_values(idx, values.size)
        # for e, value in enumerate(values):
        #     self.sheet.write(self.row, self.column + e, self._cell(value))
       	# self._row += 1

    def _values_iter(self, values):
        unique = map(lambda x: x[0], groupby(values))
        return izip(self._lindex(values), self._rindex(values), unique)

    def _format_x_right(self, contents):
        if contents['is_c_base']:
            if contents['is_weighted']:
                return formats.x_right_base
            elif self.is_weighted:
                return formats.x_right_ubase
            return formats.x_right_base
        return formats.x_right

    # def _format_x_values(self, index, size):
    #     print self.row_contents[index]
    #     print index, size

    @classmethod
    def _lindex(cls, lst):
        for _, group in groupby(enumerate(lst), key=itemgetter(1)):
            yield list(next(group))[0]

    @staticmethod
    def _rindex(lst):
        func = lambda x: (len(lst) - x - 1)
        return reversed(map(func, Box._lindex(reversed(lst))))

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
    Y_KEYS = ['@', 'q4 > gender']                               # 3.
    # Y_KEYS = ['@', 'q4 > gender > Wave']                        # 4.
    Y_KEYS = ['@', 'q4 > gender > Wave', 'q5_1', 'q4 > gender'] # 5.

    # WEIGHT = None
    WEIGHT = 'weight_a'


    VIEWS = ('cbase', 'counts', 'c%', 'mean', 'median')
    VIEW_KEYS = ('x|f|x:||%s|cbase' % WEIGHT,
            'x|f|:||%s|counts' % WEIGHT, 'x|d.mean|x:||%s|mean' % WEIGHT,
            'x|d.median|x:||%s|median' % WEIGHT, 'x|f.c:f|x:||%s|counts_sum' % WEIGHT,
            'x|t.props.Dim.80|:||%s|test' % WEIGHT, 'x|t.means.Dim.80|x:||%s|test' % WEIGHT)

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

    print chains

    chains = chains.get(data_key=DATA_KEY, filter_key=FILTER_KEY,
            x_keys=X_KEYS, y_keys=Y_KEYS,
            views=VIEW_KEYS, orient=ORIENT)

    print chains

    chains.paint_all(transform_tests='full')

    for chain in chains:

    	print chain.dataframe.T.head(5).T



    # -------------
    x = Excel('basic excel.xlsx',
            details='en-GB',
            # toc=True # not implemented
            )

    x.add_chains(chains,
            'S H E E T',
            annotations=['Ann. 1', 'Ann. 2', 'Ann. 3', 'Ann. 4']
            )

    x.close()
    # -------------
