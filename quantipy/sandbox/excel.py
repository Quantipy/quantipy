# -*- coding: utf-8 -*-

import os # for testing only
import json # for testing only
import numpy as np
import pandas as pd
import quantipy as qp
import weakref
import re
from sandbox import Chain
from xlsxwriter import Workbook
from xlsxwriter.worksheet import Worksheet
from xlsxwriter.utility import xl_rowcol_to_cell

from itertools import izip, dropwhile, groupby
from operator import itemgetter

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
    '''Decorator that makes a property lazy-evaluated.
    '''
    attr_name = '_lazy_' + func.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))
        return getattr(self, attr_name)
    return _lazy_property

class Excel(Workbook):
    """ TODO: docstring
    """

    def __init__(self, filename):
        super(Excel, self).__init__()
        self.filename = filename

    def __repr__(self):
        return 'Excel(%r)' % self.filename

    def __str__(self):
        return '%s' % self.filename

    def __del__(self):
        del self

    def add_chains(self, chains, sheet_name):
        self._write_chains(chains, sheet_name)

    def _write_chains(self, chains, sheet_name):

        worksheet = Sheet(chains, sheet_name)

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


class Sheet(Worksheet):
    """ TODO: docstring
    """

    def __init__(self, chains, sheet_name):
        super(Sheet, self).__init__()
        self.chains = chains
        self.sheet_name = sheet_name
        self.start_row = 1
        self.start_column = 1
        self.row = 1
        self.column = 1
        # self._has_tests = None
        self._freeze_loc = None
        self._columns = None

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return self.chains.pop(0)
        except IndexError:
            del self.chains
            raise StopIteration
    next = __next__

    def write_chains(self):
        """ TODO: docstring
        """
        if isinstance(self.chains, Chain):
            self.chains = [self.chains]

        for i, chain in enumerate(self):
            frame, contents = chain.dataframe, chain.contents

            if self.row == self.start_row:
                self._set_freeze_loc(frame.columns)
                self._set_columns(frame.columns)

            # write frame
            box = Box(self, frame, contents, self.row, self.column)
            box._write(columns=(i==0))
            self.row = box.row

            del box

        # freeze panes
        self.freeze_panes(*self._freeze_loc)

    def _set_columns(self, columns):
        # TODO: make column width optional --> Properties().
        self.set_column(self.start_column,
                        self.start_column + columns.size -1,
                        10)

    def _set_freeze_loc(self, columns):
        l_0 = columns.get_level_values(0).values
        first_column_size = len(np.extract(np.argmin(l_0), l_0))
        self._freeze_loc = ((self.start_row + columns.nlevels),
                            (self.start_column + first_column_size))

class Box(object):
    """ TODO: docstring
    """

    __slots__ = ('sheet', 'frame', 'contents', 'row', 'column',
                 '_single_columns')

    def __init__(self, sheet, frame, contents, row, column):
        self.sheet = sheet
        self.frame = frame
        self.contents = contents
        self.row = row
        self.column = column
        self._single_columns = None

    @property
    def single_columns(self):
        if self._single_columns is None:
            self._single_columns = []
        return self._single_columns

    def _write(self, columns):
        # write columns
        if columns:
            self._write_columns()
        self._write_rows()

    def _write_columns(self):
        row = self.row
        for lid in xrange(self.frame.columns.nlevels):
            values = self.frame.columns.get_level_values(lid).values
            if lid % 2:
                for left, right, value in self._values_iter(values):
                    if left not in self.single_columns:
                        value = self._clean_pad(value)
                        if right == left:
                            self.sheet.write(row, self.column + left, value)
                        else:
                            self.sheet.merge_range(row, self.column + left,
                                                   row, self.column + right,
                                                   value)
                row += 1
            else:
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

                    value = self._clean_pad(value)
                    if lid > 0:
                        i = self.frame.columns.nlevels - 1
                        x = (self.frame
                                .columns
                                .get_level_values(lid+1)
                                .values[state:idx+1])
                        size = x.size
                        unique = np.unique(x)
                        try:
                            chunk = list(self._lindex(x))[unique.size]
                        except IndexError:
                            chunk = size
                        repeat = size / chunk
                        for r in xrange(repeat):
                            offset = chunk * r
                            left = self.column + state + offset
                            right = self.column + state + offset + chunk - 1
                            self.sheet.merge_range(row, left, row, right,
                                                   value)
                    else:
                        self.sheet.merge_range(row, self.column + state,
                                               row, self.column + idx,
                                               value)
                    state = None
                row += 1
        for idx in self.single_columns:
            self.sheet.merge_range(self.row, self.column + idx,
                                   row - self.row, self.column + idx,
                                   values[idx])
        self.row += row - 1

    def _write_rows(self):
        for value in self.frame.index.get_level_values(0).unique():
            self.sheet.write(self.row, self.column - 1, value)
            self.row += 1
        for i, values in enumerate(self.frame.values):
            self.sheet.write(self.row, self.column - 1,
                             self.frame.index.get_level_values(1)[i])
            for idx, value in enumerate(values):
                if value != value:
                    continue # np.NAN
                self.sheet.write(self.row, self.column + idx, value)
            self.row += 1

    def _values_iter(self, values):
        unique = map(lambda x: x[0], groupby(values))
        return izip(self._lindex(values), self._rindex(values), unique)

    @classmethod
    def _lindex(cls, lst):
        for _, group in groupby(enumerate(lst), key=itemgetter(1)):
            yield list(next(group))[0]

    @staticmethod
    def _rindex(lst):
        return reversed(map(lambda x: len(lst) - x - 1, Box._lindex(reversed(lst))))

    @staticmethod
    def _clean_pad(value):
        pattern = r'#pad-\d+'
        if re.search(pattern, value):
            return ''
        return value

##############################################################################

# PATH_DATA = '../../tests/'
# NAME_PROJ = 'Example Data (A)'
# NAME_META = 'Example Data (A).json'
# NAME_DATA = 'Example Data (A).csv'
# PATH_META = os.path.join(PATH_DATA, NAME_META)
# PATH_DATA = os.path.join(PATH_DATA, NAME_DATA)

# DATA_KEY = ORIENT = 'x'
# FILTER_KEY = 'no_filter'
# # X_KEYS = ['q5_1']
# X_KEYS = ['q5_1', 'q4', 'gender', 'Wave']
# # Y_KEYS = ['@', 'q4']                                        # 1.
# # Y_KEYS = ['@', 'q4', 'q5_2', 'gender', 'Wave']              # 2.
# # Y_KEYS = ['@', 'q4 > gender']                               # 3.
# # Y_KEYS = ['@', 'q4 > gender > Wave']                        # 4.
# Y_KEYS = ['@', 'q4 > gender > Wave', 'q5_1', 'q4 > gender'] # 5.

# VIEWS = ('cbase', 'counts', 'c%', 'mean', 'median')
# VIEW_KEYS = ('x|f|x:|||cbase', 'x|f|:|||counts', 'x|d.mean|x:|||mean',
#              'x|d.median|x:|||median', 'x|f.c:f|x:|||counts_sum')

# dataset = qp.DataSet(NAME_PROJ, dimensions_comp=False)
# dataset.read_quantipy(PATH_META, PATH_DATA)
# meta, data = dataset.split()
# data = data.head(250)
# stack = qp.Stack(NAME_PROJ, add_data={DATA_KEY: {'meta': meta, 'data': data}})
# stack.add_link(x=X_KEYS, y=Y_KEYS, views=VIEWS)
# chain = Chain(stack, name='chain')
# chains = chain.get(data_key=DATA_KEY, filter_key=FILTER_KEY,
#                    x_keys=X_KEYS, y_keys=Y_KEYS,
#                    views=VIEW_KEYS, orient=ORIENT)
# try:
#     chain.paint()
# except AttributeError:
#     chains = [c.paint() for c in chains]

# # -------------

# x = Excel('basic excel.xlsx')

# print x.__class__.__bases__

# print repr(x)
# print str(x)
# print x

# x.add_chains(chains, 'S H E E T')

# print '>', x.x_window
# print '>', x.doc_properties
# print '>', x
# print '>', x
# print '>', x.filename
# print dir(x)

# x.close()
# # # -------------
