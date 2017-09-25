# -* codint: utf-8 -*-

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
# show_cell_details=False   --> details     (Excel)

# TODO: grouped_views=None,
# TODO: table_properties=None,
# TODO: italicise_level=None,
# TODO: decimals=None,
# TODO: mask_label_format=None,
# TODO: extract_mask_label=False,

class Excel(Workbook):
    """ TODO: docstring
    """

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

        worksheet = Sheet(chains, sheet_name, self.details,
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

    def close(self):
        if self.toc:
            self._write_toc()
        super(Excel, self).close()

class Sheet(Worksheet):
    """ TODO: docstring
    """

    def __init__(self, chains, sheet_name, details, annotations=None):
        super(Sheet, self).__init__()
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

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return self.chains.pop(0)
        except IndexError:
            raise StopIteration
    next = __next__

    @lazy_property
    def test_letters(self):
        return self.chains[0].sig_test_letters

    @lazy_property
    def view_keys(self):
        print self.chains[0]
        import json
        print json.dumps(self.chains[0].contents, indent=4)
        raise
        view_keys = self.chains[0].views
        return self.chains[0].sig_test_letters

    @lazy_property
    def group_order(self):
        raise NotImplementedError('_lazy_group_order')

    def write_chains(self):
        """ TODO: docstring
        """

        view_keys = self.view_keys
        test_letters = self.test_letters

        if self.annotations:
            for idx, ann in enumerate(self.annotations):
                self.write(idx, 0, ann)
        if isinstance(self.chains, Chain):
            self.chains = [self.chains]

        for i, chain in enumerate(self):

            columns = chain.dataframe.columns
            if self.row == self.start_row:
                self._set_freeze_loc(columns)
                self._set_columns(columns)

            # write frame
            box = Box(self, chain, self.row, self.column)
            box.write(columns=(i==0))
            self.row = box.row

            del box

        # cell details
        if self.details:
            cell_details, total_levels = self._cell_details(view_keys,
                    self.details,
                    test_letters,
                    group_order=self.group_order)
            # freeze panes
        self.freeze_panes(*self._freeze_loc)

    def _set_columns(self, columns):
        # TODO: make column width optional --> Properties().
        self.set_column(self.start_column,
                self.start_column + columns.size - 1,
                10)

    def _set_freeze_loc(self, columns):
        l_0 = columns.get_level_values(columns.nlevels - 1).values
        first_column_size = len(np.extract(np.argmin(l_0), l_0))
        has_tests = bool(self.test_letters)
        self._freeze_loc = ((self.start_row + columns.nlevels + has_tests),
                (self.start_column + first_column_size))

    def _cell_details(views, default_text=None, testcol_maps={}, group_order=None):
        if default_text in ['en-GB', 'fr-FR']:
            trans_text = default_text
        else:
            trans_text = 'en-GB'

        transmap = CD_TRANSMAP[trans_text]

        has_tests_total = False
        cell_details = ''
        counts = False
        col_pct = False
        for vk in views:
            n = vk.split('|')
            if n[1][0]=='f' and not 'cbase' in n[5]:
                if n[3]=='':
                    counts = True
                elif n[3]=='y':
                    col_pct = True
        proptests = False
        meantests = False
        if testcol_maps.keys():
            test_levels, test_total_levels = [], []
            for vk in views:
                if vk.startswith('x|t.props.'):
                    proptests = True
                    sig = int(vk.split('|')[1].split('.')[-1].split('+')[0])
                    level = (100 - sig)
                    if not level in test_levels:
                        test_levels.append(level)
                    if '+@' in vk:
                        has_tests_total = True
                        if not level in test_total_levels:
                            test_total_levels.append(level)
                elif vk.startswith('x|t.means.'):
                    meantests = True
                    sig = int(vk.split('|')[1].split('.')[-1].split('+')[0])
                    level = (100 - sig)
                    if not level in test_levels:
                        test_levels.append(level)
                    if '+@' in vk:
                        has_tests_total = True
                        if not level in test_total_levels:
                            test_total_levels.append(level)
            test_levels = '/'.join(
                    ['{}%'.format(100-l) for l in sorted(test_levels)])
            test_total_levels = '/'.join(
                    ['{}%'.format(100-l) for l in sorted(test_total_levels)])

            # Find column test pairings to include in details at end of sheet
            test_groups = [testcol_maps[xb] for xb in group_order if not xb=='@']
            test_groups = ', '.join(
                    [
                        '/'.join(
                            [
                                group[str(k)]
                                for k in [
                                    int(k) for k in group.keys()
                                    if '@' not in k]])
                                for group in test_groups])

                    # Finalize details to put at the end of the sheet
        cell_contents = []
        if counts: cell_contents.append(transmap['N'])
        if col_pct: cell_contents.append(transmap['c%'])
        if proptests or meantests:
            cell_contents.append(transmap['str'])
            tests = []
            if proptests: tests.append(transmap['cp'])
            if meantests: tests.append(transmap['cm'])
            tests = ', {} ({}, ({}): {}, {}: 30 (**), {}: 100 (*))'.format(
                    transmap['stats'],
                    ', '.join(tests),
                    test_levels,
                    test_groups,
                    transmap['mb'],
                    transmap['sb'])
        else:
            tests = ''
        cell_contents = ', '.join(cell_contents)
        if cell_contents:
            cell_details = '{} ({}){}'.format(transmap['cc'], cell_contents, tests)
        else:
            cell_details = ''

        if has_tests_total:
            return (cell_details, test_total_levels)
        return (cell_details, False)

class Box(object):
    """ TODO: docstring
    """

    # _properties_cache = WeakValueDictionary()

    __slots__ = ('sheet', 'chain', '_frame', '_contents', '_row', '_column',
            '_single_columns', '_test_letters', '_lazy_test_letters')

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
        return self.chain.sig_test_letters

    def write(self, columns):
        """TODO: Doc string
        """
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
        for idx in self.single_columns:
            offset = (bool(self.test_letters) + 1) % 2
            self.sheet.merge_range(self.row, self.column + idx,
                    row - offset, self.column + idx,
                    self._cell(values[idx]))
            self._row += row - self.sheet.start_row
        if self.test_letters:
            self.sheet.write_row(self.row, self.column + 1, self.test_letters)
            self._row += 1

    def _write_column_headers(self, values, row):
        for left, right, value in self._values_iter(values):
            if left not in self.single_columns:
                if right == left:
                    self.sheet.write(row, self.column + left, self._cell(value))
                else:
                    self.sheet.merge_range(row, self.column + left,
                            row, self.column + right,
                            self._cell(value))

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
                    self.sheet.merge_range(row, l, row, r, self._cell(value))
            else:
                self.sheet.merge_range(row, self.column + state,
                        row, self.column + idx,
                        self._cell(value))
                state = None

    def _values_iter(self, values):
        unique = map(lambda x: x[0], groupby(values))
        return izip(self._lindex(values), self._rindex(values), unique)

    def _write_rows(self):
        levels = self.frame.index.get_level_values
        for value in levels(0).unique():
            self.sheet.write(self.row, self.column - 1, self._cell(value))
            self._row += 1
        for i, values in enumerate(self.frame.values):
            self.sheet.write(self.row, self.column - 1,
                    self._cell(levels(1)[i]))
            for idx, value in enumerate(values):
                self.sheet.write(self.row, self.column + idx, self._cell(value))
            self._row += 1

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

    VIEWS = ('cbase', 'counts', 'c%', 'mean', 'median')
    VIEW_KEYS = ('x|f|x:|||cbase', 'x|f|:|||counts', 'x|d.mean|x:|||mean',
            'x|d.median|x:|||median', 'x|f.c:f|x:|||counts_sum',
            'x|t.props.Dim.80|:|||test', 'x|t.means.Dim.80|x:|||test')

    dataset = qp.DataSet(NAME_PROJ, dimensions_comp=False)
    dataset.read_quantipy(PATH_META, PATH_DATA)
    meta, data = dataset.split()
    data = data.head(250)
    stack = qp.Stack(NAME_PROJ, add_data={DATA_KEY: {'meta': meta, 'data': data}})
    stack.add_link(x=X_KEYS, y=Y_KEYS, views=VIEWS)

    weights = None
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

    chain = Chain(stack, name='chain')
    chains = chain.get(data_key=DATA_KEY, filter_key=FILTER_KEY,
            x_keys=X_KEYS, y_keys=Y_KEYS,
            views=VIEW_KEYS, orient=ORIENT)
    try:
        chain.paint()
    except AttributeError:
        chains = [c.paint() for c in chains]

    # -------------

    x = Excel('basic excel.xlsx',
            details='en-GB',
            # toc=True # not implemented
            )

    # print repr(x)
    # print str(x)
    # print x

    x.add_chains(chains,
            'S H E E T',
            annotations=['Ann. 1', 'Ann. 2', 'Ann. 3', 'Ann. 4']
            )

    # print '>', x.x_window
    # print '>', x.doc_properties
    # print '>', x
    # print '>', x.filename

    x.close()
    # -------------
