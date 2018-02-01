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
from PIL import Image
from difflib import SequenceMatcher 

from excel_formats import ExcelFormats, _Format
from excel_formats_constants import _DEFAULT_ATTRIBUTES, _VIEWS_GROUPS

import cPickle
import warnings; warnings.simplefilter('ignore')

try:
    from functools import lru_cache
except ImportError:
    from functools32 import lru_cache


# Initialization data to pass to the worksheet._
_SHEET_ATTR = ('str_table',
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

# Defaults for Sheet.
_SHEET_DEFAULTS = dict(alternate_bg=True,
                       dummy_tests=False,
                       #column_width_str=10,
                       #df_nan_rep='__NA__',
                       #display_test_level=True,
                       #format_label_row=False,
                       # -------------------------------------------
                       #TODO: add edtiable to Cell class
                       freq_0_rep='-',
                       stat_0_rep='-',
                       arrow_color_high='#2EB08C',
                       arrow_rep_high=u'\u25B2',
                       arrow_color_low='#FC8EAC',
                       arrow_rep_low=u'\u25BC',
                       # -------------------------------------------
                       #img_insert_x=0,
                       #img_insert_y=0,
                       #img_name='qplogo_invert_lg.png',
                       #img_size=[130, 130],
                       #img_url='logo/qplogo_invert_lg.png',
                       #img_x_offset=0,
                       #img_y_offset=0,
                       #no_logo=False,
                       #row_height=12.75,
                       #row_wrap_trigger=44,
                       start_column=0,
                       start_row=0,
                       #stat_0_rep=0.00,
                       #test_seperator='.',
                       y_header_height=33.75,
                       y_row_height=50
                       )


class Excel(Workbook):
    # TODO: docstring

    def __init__(self, filename, toc=False, views_groups=None,
                 italicise_level=None, details=False, decimals=None,
                 image=None, **kwargs):
        super(Excel, self).__init__()
        self.filename = filename
        self.toc = toc
        self.views_groups = views_groups
        self.italicise_level = italicise_level
        self.details = details
        self._decimals = decimals
        self._image = image

        if views_groups:
            views_groups = dict([(k, views_groups[k] if k in views_groups else v)
                                 for k, v in _VIEWS_GROUPS.iteritems()])

        self._formats = ExcelFormats(views_groups, **kwargs)

    def __repr__(self):
        return 'Excel(%r)' % self.filename

    def __str__(self):
        return '%s' % self.filename

    def __del__(self):
        del self

    @lazy_property
    def decimals(self):
        if self._decimals is None:
            return {}
        elif isinstance(self._decimals, int):
            return {_: self._decimals for _ in ('N', 'P', 'D')}
        return self._decimals

    @lazy_property
    def image(self):
        if self._image:
            image = Image.open(self._image['img_url'])
            image.thumbnail(self._image['img_size'], Image.ANTIALIAS)
            image.save(os.path.basename(self._image['img_url']))
        return self._image

    def add_chains(self, chains, sheet_name, annotations=None, **kwargs):
        # TODO: docstring
        self._write_chains(chains, sheet_name, annotations, **kwargs)

    def _write_chains(self, chains, sheet_name, annotations, **kwargs):
        worksheet = Sheet(self, chains, sheet_name, annotations, **kwargs)

        init_data = {attr: getattr(self, attr, None) for attr in _SHEET_ATTR}
        init_data.update({'name': sheet_name,
                          'index': len(self.worksheets_objs)})
        worksheet._initialize(init_data)

        self.worksheets_objs.append(worksheet)

        worksheet.write_chains()

        del worksheet

    def _write_toc(self):
        raise NotImplementedError('_write_toc')

    @lru_cache()
    def _add_format(self, format_):
        return self.add_format(format_)

    # def close(self):
    #    print '...........'
    #    if self.toc:
    #        self._write_toc()
    #    self.close()



class Sheet(Worksheet):
    # TODO: docstring

    def __init__(self, excel, chains, sheet_name, annotations, **kwargs):
        super(Sheet, self).__init__()
        self.excel = excel
        self.chains = chains
        self.sheet_name = sheet_name
        self.annotations = annotations

        for name in _SHEET_DEFAULTS:
            value_or_default = kwargs.get(name, _SHEET_DEFAULTS[name])
            setattr(self, name, value_or_default)

        self._row = self.start_row
        self._column = self.start_column

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

    def write(self, *args):
        if isinstance(args[-1], dict):
            args = args[:-1] + (self.excel._add_format(args[-1]), )
        super(Sheet, self).write(*args)

    def write_rich_string(self, *args):
        args, rich_text = ((args[0], args[1]), ), args[2:]
        for arg in rich_text:
            if isinstance(arg, dict):
                args = args + (self.excel._add_format(arg), )
            else:
                args = args + (arg, )
        args = (xl_rowcol_to_cell(*args[0]), ) + args[1:]
        super(Sheet, self).write_rich_string(*args)

    def merge_range(self, *args):
        args = args[:-1] + (self.excel._add_format(args[-1]), )
        super(Sheet, self).merge_range(*args)

    def write_chains(self):
        # TODO: docstring
        if self.annotations:
            for ann in self.annotations:
                self.write(self._row, self._column, ann)
                self._row += 1

        for i, chain in enumerate(self.chains):

            try:
                columns = chain.dataframe.columns

                # make y-axis writing availbale to all chains
                if i == 0:
                    self._set_freeze_loc(columns)
                    self._set_columns(columns)

            except AttributeError:
                columns = chain.structure.columns

            # write frame
            box = Box(self, chain, self._row, self._column)
            box.to_sheet(columns=(i==0))

            del box

        if self._freeze_loc:
            self.freeze_panes(*self._freeze_loc)

        self.hide_gridlines(2)

        if self.excel.details and all(c.structure is None for c in self.chains):
            format_ = self.excel._formats._cell_details
            cd = None
            arrow_descriptions = None
            for chain in self.chains:
                cds = chain.cell_details
                if len(cds) == 3 and not arrow_descriptions:
                    arrow_descriptions = cds[1:] 
                if cd is None:
                    cd = cds[0]
                else:
                    if cd <> cds[0]:
                        long = max((cd, cds[0]), key=len)
                        short = min((cd, cds[0]), key=len)
                        sm = SequenceMatcher(None, long, short)
                        for tag, i1, i2, j1, j2 in sm.get_opcodes():
                            if tag == 'insert':
                                long = long[:i1] + short[j1:j2] + long[i2:]
                        cd = long

            self.write(self._row + 1, self._column + 1, cd, format_)
            if arrow_descriptions:
                arrow_format = _Format(**{'font_color': self.arrow_color_high})
                arrow_format = self.excel._add_format(arrow_format)
                self.write_rich_string(self._row + 2, self._column + 1,
                                       arrow_format, self.arrow_rep_high,
                                       format_, cds[1], format_)
                arrow_format = _Format(**{'font_color': self.arrow_color_low})
                arrow_format = self.excel._add_format(arrow_format)
                self.write_rich_string(self._row + 3, self._column + 1,
                                       arrow_format, self.arrow_rep_low,
                                       format_, cds[2], format_)

        if self.excel.image:

            self.insert_image(self.excel.image['img_insert_x'],
                              self.excel.image['img_insert_y'],
                              self.excel.image['img_url'],
                              dict(x_offset=self.excel.image['img_x_offset'],
                                   y_offset=self.excel.image['img_y_offset']))

    def _set_columns(self, columns):
        # TODO: make column width optional --> Properties().
        self.set_column(self._column, self._column, 40)
        self.set_column(self._column + 1, self._column + columns.size, 10)

    def _set_freeze_loc(self, columns):
        if list(columns.labels[0]).count(0) == 1:
            offset = 1
        else:
            offset = 0
        self._freeze_loc = ((self._row + columns.nlevels),
                            (self._column + offset + 1))


class Box(object):
    # TODO: docstring

    __slots__ = ('sheet', 'chain', '_single_columns','_column_edges',
                 '_columns', '_italic', '_lazy_excel', '_lazy_index',
                 '_lazy_columns', '_lazy_values', '_lazy_contents',
                 '_lazy_is_weighted', '_lazy_shape', '_lazy_has_tests',
                 '_lazy_arrow_rep', '_lazy_arrow_color', '_lazy_header_left',
                 '_lazy_header_center', '_lazy_header_title', '_lazy_notes')

    def __init__(self, sheet, chain, row, column):
        self.sheet = sheet
        self.chain = chain
        self._single_columns = None
        self._column_edges = None

        self._columns = []
        self._italic = []

    @lazy_property
    def excel(self):
        return self.sheet.excel

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
        descr = self.chain.describe()
        protocol = cPickle.HIGHEST_PROTOCOL
        contents = cPickle.loads(cPickle.dumps(self.chain.contents, protocol))

        def _contents(c, d):
            if 0 in c.values()[0]:
                for i, value in enumerate(c.values()):
                    c[i] = _contents(value, d[i])
            else:
                for idx, value in enumerate(c.values()):
                    if value['is_block']:
                        calc_part, calc_or_block = d[idx][-2:]
                        if calc_or_block == 'has_calc':
                            if calc_part == 'calc':
                                value['block_type'] = 'calc'
                            else:
                                value['block_type'] = 'calc_' + calc_part
                        else:
                            value['block_type'] = calc_or_block
                    elif value['is_calc_only']:
                        value['block_type'] = 'calc'
                    c[idx] = value
            return c

        return _contents(contents, descr)

    @lazy_property
    def is_weighted(self):
        if self.chain.array_style == 0:
            return any(y['is_weighted']
                       for x in self.contents.itervalues()
                       for y in x.itervalues())
        return any(x['is_weighted'] for x in self.contents.itervalues())

    @lazy_property
    def shape(self):
        return self.chain.dataframe.shape

    @lazy_property
    def has_tests(self):
        return self.chain.dataframe.columns.nlevels % 2

    @lazy_property
    def arrow_rep(self):
        return {"'@L'": self.sheet.arrow_rep_high,
                "'@H'": self.sheet.arrow_rep_low}

    @lazy_property
    def arrow_color(self):
        return {"'@L'": self.sheet.arrow_color_high,
                "'@H'": self.sheet.arrow_color_low}

    @lazy_property
    def header_left(self):
        return self.chain.annotations.header_left

    @lazy_property
    def header_center(self):
        return self.chain.annotations.header_center

    @lazy_property
    def header_title(self):
        return self.chain.annotations.header_title

    @lazy_property
    def notes(self):
        return self.chain.annotations.notes

    def to_sheet(self, columns):
        # TODO: Doc string
        if self.chain.structure is not None:
            self._write_data()
        else:
            if columns:
                self._write_columns()
            self._write_rows()

    def _write_data(self):
        format_ = self.excel._formats._data_header

        for rel_y, label in enumerate(self.chain.structure.columns):
            column = self.sheet._column + rel_y
            self.sheet.merge_range(self.sheet._row, column,
                                   self.sheet._row + 1, column,
                                   label, format_)
            self.sheet.set_row(self.sheet._row, self.sheet.y_header_height)
            self.sheet.set_row(self.sheet._row + 1, self.sheet.y_row_height)

        self.sheet.freeze_panes(self.sheet._row + 2, self.sheet._column)

        self.sheet._row += 3

        row_max = self.chain.structure.shape[0] - 1

        flat = self.chain.structure.values.flat
        rel_x, rel_y = flat.coords
        for data in flat:
            name =  'left^right^'
            if rel_x == 0:
                name += 'top^'
            elif rel_x == row_max:
                name += 'bottom^'
            name += 'data'
            format_ = self.excel._formats[name]
            self.sheet.write(self.sheet._row + rel_x,
                             self.sheet._column + rel_y,
                             data, format_)
            rel_x, rel_y = flat.coords

    def _write_columns(self):
        contents = dict()
        format_ = self.excel._formats._y
        column = self.sheet._column + 1
        nlevels = self.columns.nlevels
        for level_id in xrange(nlevels):
            row = self.sheet._row + level_id
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
                data = self._cell(data, **contents)
                if level_id == 0:
                    if left == right:
                        self.single_columns.append(left)
                    self.column_edges.append(right + 1)
                if left not in self.single_columns:
                    if group_sizes and not is_values:
                        r = 0
                        while r != right:
                            
                            self.sheet.merge_range(row, column + group_sizes[0][0],
                                                   row, column + group_sizes[0][1],
                                                   data, format_)
                            _, r = group_sizes.pop(0)
                    elif left == right:
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

            if not self.has_tests or ((level_id + 1) != nlevels) and self.has_tests:
                if (row % 2) == 0:
                    self.sheet.set_row(row, self.sheet.y_header_height)
                else:
                    self.sheet.set_row(row, self.sheet.y_row_height)

        for cindex in self.single_columns:
            level = -(1 + self.has_tests)
            data = self._cell(self.columns.get_level_values(level)[cindex], **contents)
            self.sheet.merge_range(row - nlevels + 1, column + cindex,
                                   row, column + cindex,
                                   data, format_)

        self.sheet._row = row + 1

    def _write_rows(self):
        if self.chain.annotations:
            self._write_annotations(['header_left', 'header_center', 'header_title'])

        column = self.sheet._column

        levels = self.index.get_level_values

        if self.chain._is_mask_item:
            self.sheet.write(self.sheet._row, column,
                                   levels(0).unique().values[0],
                                   self.excel._formats['mask_label'])
            self._format_row(self.excel._formats['mask_label'])
        else:
            self.sheet.write(self.sheet._row, column,
                             levels(0).unique().values[0],
                             self.excel._formats['label'])
            self._format_row(self.excel._formats['label'])
        self.sheet._row += 1

        if self.notes:
            self._write_annotations(['notes'])

        if self.sheet.dummy_tests and self.has_tests:
            level_1, values, contents = self._get_dummies(levels(1).values,
                                                          self.values)
        else:
            level_1, values, contents = levels(1).values, self.values, self.contents

        row_max = max(contents.keys())
        flat = np.c_[level_1.T, values].flat
        rel_x, rel_y = flat.coords

        bg = use_bg = True
        bg_from = bg_x_contents = None
        border_from = False
        arrow_high_format = arrow_low_format = _None = object()

        for data in flat:

            if self.chain.array_style == 0:
                if rel_y == 0:
                    for idx in sorted(contents[rel_x]):
                        if not self._is('base', **contents[rel_x][idx]):
                            bg_x_contents = x_contents = contents[rel_x][idx]
                            bg_from = self._row_format_name(**x_contents)
                            break
                else:
                    x_contents = contents[rel_x][rel_y - 1]

            else:
                x_contents = contents[rel_x]

            name = self._row_format_name(**x_contents)

            if rel_y == 0:
                if data == '':
                    view_border = False
                else:
                    view_border = True
                    if self.sheet.alternate_bg:
                        bg, use_bg = self._alternate_bg(name, bg)
            format_ = self._format_x(name, rel_x, rel_y, row_max,
                                     x_contents.get('dummy'), use_bg,
                                     view_border, border_from)

            if self.excel.italicise_level:
                if rel_y and self.chain.array_style == 0:
                    if self._is('base', **x_contents) and not x_contents['is_weighted']:
                        if data == data:
                            if data <= self.excel.italicise_level:
                                arr_summ_ital = True
                            else:
                                arr_summ_ital = False
                    if arr_summ_ital:
                        if rel_y not in self._italic:
                            self._italic.append(rel_y)
                    elif rel_y in self._italic:
                        self._italic.remove(rel_y)
                else:
                    if rel_y not in self._columns:
                        if self._is('base', **x_contents) and not x_contents['is_weighted']:
                            if data <= self.excel.italicise_level:
                                self._italic.append(rel_y)
                            self._columns.append(rel_y)
                if rel_y in self._italic:
                    format_['italic'] = True

            if rel_y and bg_from:
                bg_format = self._format_x(bg_from, rel_x, rel_y, row_max,
                                          bg_x_contents.get('dummy'), use_bg,
                                          view_border, border_from)
                format_['bg_color'] = bg_format.get('bg_color', '#FFFFFF')

            cell_data = self._cell(data, **x_contents)
            if any(_ in str(cell_data) for _ in ("'@L'", "'@H'")):
                low_base = cell_data.endswith('*')
                if low_base:
                    cell_data = cell_data[:-1]
                parts = cell_data.split('.')
                arrow, cell_data = parts[0], '.'.join(parts[1:])
                if low_base:
                    cell_data = cell_data + '*'
                arrow_rep = self.arrow_rep.get(arrow)
                arrow_color = self.arrow_color.get(arrow)
                arrow_format = {"'@L'": arrow_high_format, "'@H'": arrow_low_format}.get(arrow)

                if arrow_format is _None:
                    arrow_format = self._format_x(name, rel_x, rel_y,
                                                  row_max, x_contents.get('dummy'),
                                                  use_bg, view_border,
                                                  border_from,
                                                  **{'font_color': arrow_color})
                self.sheet.write_rich_string(self.sheet._row + rel_x,
                                             self.sheet._column + rel_y,
                                             arrow_format, arrow_rep,
                                             format_, ' ' + cell_data, format_)
            else:
                self.sheet.write(self.sheet._row + rel_x,
                                 self.sheet._column + rel_y,
                                 cell_data,
                                 format_)
            nxt_x, nxt_y = flat.coords
            rel_x, rel_y = nxt_x, nxt_y
            if rel_y == 0:
                border_from = name
        self.sheet._row += rel_x

    def _write_annotations(self, names):
        for name in names:
            anno = getattr(self, name)
            if anno:
                self.sheet.write(self.sheet._row, self.sheet._column,
                                 anno[0], self.excel._formats[name])
                self._format_row(self.excel._formats[name])
                self.sheet._row += 1

    def _format_row(self, format_):
        for rel_y in xrange(1, self.values.shape[1] + 1):
            self.sheet.write(self.sheet._row, self.sheet._column + rel_y, '', format_)

    @lru_cache()
    def _alternate_bg(self, name, bg):
        freq_view_group = self.excel.views_groups.get(name, '') == 'freq'
        is_freq_test = any(_ in name for _ in ('counts', 'pct', 'propstest'))
        is_mean = 'mean' in name
        not_net_sum = all(_ not in name for _ in ('net', 'sum'))
        if ((is_freq_test and not_net_sum) or freq_view_group) or \
                (not is_mean and self.chain.array_style == 0):
            return not bg, bg
        return self.sheet.alternate_bg, True

    @lru_cache()
    def _row_format_name(self, **contents):
        if contents.get('block_type'):
            if contents['is_propstest']:
                return 'block_' + contents['block_type'] + '_propstest'
            elif contents['is_counts']:
                return 'block_' + contents['block_type'] + '_counts'
            elif contents['is_c_pct']:
                return 'block_' + contents['block_type'] + '_c_pct'
            elif contents['is_r_pct']:
                return 'block_' + contents['block_type'] + '_r_pct'
        if contents['is_meanstest']:
            return 'meanstest'
        elif contents['is_propstest']:
            if contents['is_net']:
                return 'net_propstest'
            elif contents['is_counts_sum']:
                return 'counts_sum'
            elif contents['is_c_pct_sum']:
                return 'c_pct_sum'
            return 'propstest'
        elif contents['is_c_base']:
            if contents['is_weighted']:
                return 'c_base'
            elif self.is_weighted:
                return 'u_c_base'
            return 'c_base'
        elif contents['is_c_base_gross']:
            if contents['is_weighted']:
                return 'c_base_gross'
            elif self.is_weighted:
                return 'u_c_base_gross'
            return 'c_base_gross'
        elif contents['is_r_base']:
            if contents['is_weighted']:
                return 'r_base'
            elif self.is_weighted:
                return 'u_r_base'
            return 'r_base'
        elif contents['is_e_base']:
            if contents['is_weighted']:
                return 'e_base'
            elif self.is_weighted:
                return 'u_e_base'
            return 'e_base'
        elif contents['is_counts']:
            if contents['is_net']:
                return 'net_counts'
            if contents['is_counts_sum']:
                return 'counts_sum'
            return 'counts'
        elif contents['is_c_pct']:
            if contents['is_net']:
                return 'net_c_pct'
            if contents['is_c_pct_sum']:
                return 'c_pct_sum'
            return 'c_pct'
        elif contents['is_r_pct']:
            if contents['is_net']:
                return 'net_r_pct'
            return 'r_pct'
        elif contents['is_mean']:
            return 'mean'
        elif contents['is_stddev']:
            return 'stddev'
        elif contents['is_min']:
            return 'min'
        elif contents['is_max']:
            return 'max'
        elif contents['is_median']:
            return 'median'
        elif contents['is_variance']:
            return 'var'
        elif contents['is_varcoeff']:
            return 'varcoeff'
        elif contents['is_sem']:
            return 'sem'
        elif contents['is_percentile']:
            return contents['stat']

    @lru_cache()
    def _format_x(self, name, rel_x, rel_y, row_max, dummy, bg, view_border,
                  border_from, **kwargs):
        if rel_y == 0:
            format_name = name + '_text'
        else:
            format_name = self._format_position(rel_x, rel_y, row_max)
            if view_border and 'top' not in format_name:
                format_name += 'view_border.%s^' % border_from
            format_name += name
        if not bg:
            format_name += '_no_bg_color'
        format_ = self.excel._formats[format_name]
        if kwargs:
            for key, value in kwargs.iteritems():
                format_[key] = value
        return format_

    def _format_position(self, rel_x, rel_y, row_max):
        position = ''
        if rel_y == 1:
            position = 'left^'
        if rel_y in self.column_edges:
            position += 'right^'
        if position == '':
            position = 'interior^'
        if rel_x == 0:
            position += 'top^'
        if rel_x == row_max:
            position += 'bottom^'
        return position

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
                    elif self._is('test', **self.contents[idx]):
                        dummy = False
                else:
                    if group and self._is('test', **self.contents[idx]):
                        dummy = False
                    if group and dummy:
                        dummy_idx.append(ndx + len(dummy_idx))
                    if not self._is('base', **self.contents[idx]):
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
        contents = {}
        for key in sorted(self.contents):
            if (key + num_dummies) in dummy_idx:
                num_dummies += 1
            contents[key+num_dummies] = self.contents[key]
        for key in dummy_idx:
            contents[key] = {k: v for k, v in contents[key-1].iteritems()}
            contents[key].update({'is_dummy': True,
                                  'is_propstest': True,
                                  'is_meanstest': contents[key-1]['is_stat']})

        return index, values, contents

    @lru_cache()
    def _cell(self, value, **contents):
        normalize, vtype, nan_rep = self._cell_args(**contents)
        return Cell(value, normalize, self.excel.decimals.get(vtype), nan_rep).__repr__()

    def _cell_args(self, **contents):
        pct = self._is('pct', **contents)
        counts = self._is('counts', **contents)
        base = self._is('base', **contents)
        test = self._is('test', **contents)
        freq = (pct or counts) and not base
        stat = contents.get('is_stat')
        if counts or base:
            vtype, nan_rep = 'N', self.sheet.freq_0_rep
        elif pct:
            vtype, nan_rep = 'P', self.sheet.freq_0_rep
        elif stat:
            vtype, nan_rep = 'D', self.sheet.stat_0_rep
        else:
            return False, ' ', None
        if test or (self.chain.array_style == 0 and not freq):
            return pct, vtype, ' '
        return pct, vtype, nan_rep

    @staticmethod
    @lru_cache()
    def _is(name, **contents):
        return any(name in _ for _ in list(filter(contents.get, contents)))

class Cell(object):

    def __init__(self, data, normalize, decimals, nan_rep):
        self.data = data
        self.normalize = normalize
        self.decimals = decimals
        self.nan_rep = nan_rep

    def __repr__(self):
        try:
            if np.isnan(self.data) or np.isinf(self.data) or self.data == 0:
                return self.nan_rep
        except TypeError:
            pass
        if isinstance(self.data, (str, unicode)):
            return re.sub(r'#pad-\d+', str(), self.data)
        if self.normalize:
            if self.decimals is not None:
                if isinstance(self.data, (float, np.float64)):
                    return round(self.data / 100., self.decimals + 2)
            return self.data / 100.
        if self.decimals is not None:
            if isinstance(self.data, (float, np.float64)):
                return round(self.data, self.decimals)
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
    Y_KEYS = ['@', 'q4 > gender', 'q4 > gender > Wave', 'q5_1'] # 5.
    TESTS = True

    # WEIGHT = None
    WEIGHT = 'weight_a'

    VIEWS = ('cbase',
             'cbase_gross',
             #'rbase',
             'ebase',
             'counts',
             'c%',
             'r%',
             #'mean',
             #'stddev',
             #'median',
             #'variance',
             #'varcoeff',
             #'sem',
             #'lower_q',
             #'upper_q',
             'counts_sum',
             'c%_sum',
             #'counts_cumsum',
             #'c%_cumsum',
             )

    VIEW_KEYS = ('x|f|x:|||cbase',
                 'x|f|x:||%s|cbase' % WEIGHT,
                 'x|f|x:|||cbase_gross',
                 'x|f|x:||%s|cbase_gross' % WEIGHT,
                 'x|f|x:|||ebase',
                 'x|f|x:||%s|ebase' % WEIGHT,
                 'x|f|:||%s|counts' % WEIGHT,
                 'x|f|:|y|%s|c%%' % WEIGHT,
                 'x|f|:|x|%s|r%%' % WEIGHT,
                 'x|t.props.Dim.80+@|:||%s|test' % WEIGHT,
                 'x|f|x[{1,2,3}]:||%s|No' % WEIGHT,
                 'x|f|x[{1,2,3}]:|y|%s|No' % WEIGHT,
                 'x|t.props.Dim.80+@|x[{1,2,3}]:||%s|test' % WEIGHT,
                 'x|f|x[{4,5,97}]:||%s|Yes' % WEIGHT,
                 'x|f|x[{4,5,97}]:|y|%s|Yes' % WEIGHT,
                 'x|t.props.Dim.80+@|x[{4,5,97}]:||%s|test' % WEIGHT,
                 'x|d.mean|x:||%s|mean' % WEIGHT,
                 'x|t.means.Dim.80+@|x:||%s|test' % WEIGHT,
                 'x|d.stddev|x:||%s|stddev' % WEIGHT,
                 'x|d.median|x:||%s|median' % WEIGHT,
                 'x|f.c:f|x:||%s|counts_sum' % WEIGHT,
                 'x|f.c:f|x:|y|%s|c%%_sum' % WEIGHT,
                 #'x|f.c:f|x++:||%s|counts_cumsum' % WEIGHT,
                 #'x|f.c:f|x++:|y|%s|c%%_cumsum' % WEIGHT,
                )

    weights = [None]
    if WEIGHT is not None:
        VIEW_KEYS = ('x|f|x:|||cbase', ) + VIEW_KEYS
        weights.append(WEIGHT)


    # RUN RUN RUN RUN RUN RUN RUN RUN RUN RUN RUN RUN RUN RUN RUN RUN
    CA1 = True
    AC1 = False
    ACB1 = False
    ACM1 = False 
    AC0 = False
    ACB0 = False
    ACM0 = False 
    OEC = False

    dataset = qp.DataSet(NAME_PROJ, dimensions_comp=False)
    dataset.read_quantipy(PATH_META, PATH_DATA)
    meta, data = dataset.split()

    # text key modifications ------------------------------------------
    meta['columns']['Wave']['text']['fail'] = 'from text > fake'
    meta['columns']['Wave']['text']['x edits'] = {'fail': 'from text > x edits > fail'}

    meta['columns']['gender']['text']['y edits'] = {'fake': 'from text > y edits > fake'}
    meta['columns']['gender']['text']['x edits'] = {'en-GB': 'from text > x edits > en-GB'}

    meta['columns']['q5_1']['text']['x edits'] = {'fake': 'from text > x edits > fake'}
    meta['columns']['q5_1']['properties'] = {'base_text': {'fake': 'Base: fake',
                                                           'en-GB': 'Base: en-GB'}}
    
    meta['columns']['q4']['text']['fake'] = 'from text > fake'
    meta['columns']['q4']['text']['y edits'] = {'fake': 'from text > y edits > fake'}
    meta['columns']['q4']['properties'] = {'base_text': 'Base: Text'}
    # -----------------------------------------------------------------

    #data = data.head(250)
    data.loc[30:,'q5_2'] = np.NaN
    data.loc[30:,'q5_4'] = np.NaN

    stack = qp.Stack(NAME_PROJ, add_data={DATA_KEY: {'meta': meta, 'data': data}})
    stack.add_link(x=X_KEYS, y=Y_KEYS, views=VIEWS, weights=weights)

    rel_to = []
    if 'counts' in VIEWS:
        rel_to.append(None)
    if 'c%' in VIEWS:
        rel_to.append('y')
    if 'r%' in VIEWS:
        rel_to.append('x')

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

    nets_mapper = qp.ViewMapper(template={'method': qp.QuantipyViews().frequency,
                                          'kwargs': {'iterators': {'rel_to': rel_to},
                                                     'groups': 'Nets'}})
    nets = [{'N1': [1, 2], 'text': {'en-GB': 'Waves 1 & 2 (NET)'}, 'expand': 'after'},
            {'N2': [4, 5], 'text': {'en-GB': 'Waves 4 & 5 (NET)'}, 'expand': 'after'}]
    nets_mapper.add_method(name='BLOCK', kwargs={'axis':      'x',
                                                 'logic':     nets,
                                                 'text':      'Net: ',
                                                 'combine':   False,
                                                 'complete':  True,
                                                 'expand':    'after'}
                                                 )
    stack.add_link(x=X_KEYS[-1], y=Y_KEYS, views=nets_mapper, weights=weights)

    nets_mapper = qp.ViewMapper(template={'method': qp.QuantipyViews().frequency,
                                          'kwargs': {'iterators': {'rel_to': rel_to},
                                                     'groups': 'Nets'}})
    from operator import sub
    kwargs = {'calc_only': False,
              'calc': {'text': {u'en-GB': u'Net YES'},
              'Net agreement': ('Net: Yes', sub, 'Net: No')},
              'axis': 'x',
              'logic': [{'text': {u'en-GB': 'Net: No'}, 'Net: No': [1, 2]},
                        {'text': {u'en-GB': 'Net: Yes'}, 'Net: Yes': [4, 5]}]}
    nets_mapper.add_method(name='NPS', kwargs=kwargs)
    kwargs = {'calc_only': True,
              'calc': {'text': {u'en-GB': u'Net YES'},
              'Net agreement (only)': ('Net: Yes', sub, 'Net: No')},
              'axis': 'x',
              'logic': [{'text': {u'en-GB': 'Net: No'}, 'Net: No': [1, 2]},
                        {'text': {u'en-GB': 'Net: Yes'}, 'Net: Yes': [4, 5]}]}
    nets_mapper.add_method(name='NPSonly', kwargs=kwargs)
    stack.add_link(x=X_KEYS[0], y=Y_KEYS, views=nets_mapper, weights=weights)

    stats = ['mean', 'stddev', 'median', 'var', 'varcoeff', 'sem', 'lower_q', 'upper_q']
    for stat in stats:
        options = {'stats': stat,
                   'source': None,
                   'rescale': None,
                   'drop': False,
                   'exclude': None,
                   'axis': 'x',
                   'text': ''}
        view = qp.ViewMapper()
        view.make_template('descriptives')
        view.add_method('stat', kwargs=options)
        stack.add_link(x=X_KEYS, y=Y_KEYS, views=view, weights=weights)

    if TESTS:
        test_view = qp.ViewMapper().make_template('coltests')
        view_name = 'test'
        options = {'level': 0.8,
                'metric': 'props',
                'test_total': True,
                'flag_bases': [30, 100]
                }
        test_view.add_method(view_name, kwargs=options)
        stack.add_link(x=X_KEYS, y=Y_KEYS, views=test_view, weights=weights)


        test_view = qp.ViewMapper().make_template('coltests')
        view_name = 'test'
        options = {'level': 0.8, 'metric': 'means',
                   'test_total': True,
                   'flag_bases': [30, 100]
                  }
        test_view.add_method(view_name, kwargs=options)
        stack.add_link(x=X_KEYS, y=Y_KEYS, views=test_view, weights=weights)

    #stack.describe().to_csv('d.csv'); stop()

    VIEW_KEYS = ('x|f|x:|||cbase',
                 'x|f|x:||%s|cbase' % WEIGHT,
                 'x|f|x:|||cbase_gross',
                 'x|f|x:||%s|cbase_gross' % WEIGHT,
                 'x|f|x:|||ebase',
                 'x|f|x:||%s|ebase' % WEIGHT,
                 ('x|f|:||%s|counts' % WEIGHT,
                  'x|f|:|y|%s|c%%' % WEIGHT,
                  'x|f|:|x|%s|r%%' % WEIGHT,
                  'x|t.props.Dim.80+@|:||%s|test' % WEIGHT),
                 ('x|f|x[{1,2,3}]:||%s|No' % WEIGHT,
                  'x|f|x[{1,2,3}]:|y|%s|No' % WEIGHT,
                  'x|f|x[{1,2,3}]:|x|%s|No' % WEIGHT,
                  'x|t.props.Dim.80+@|x[{1,2,3}]:||%s|test' % WEIGHT),
                 ('x|f|x[{4,5,97}]:||%s|Yes' % WEIGHT,
                  'x|f|x[{4,5,97}]:|y|%s|Yes' % WEIGHT,
                  'x|f|x[{4,5,97}]:|x|%s|Yes' % WEIGHT,
                  'x|t.props.Dim.80+@|x[{4,5,97}]:||%s|test' % WEIGHT),
                 ('x|f.c:f|x[{4,5}-{1,2}]:||%s|NPSonly' % WEIGHT,
                  'x|f.c:f|x[{4,5}-{1,2}]:|y|%s|NPSonly' % WEIGHT,
                  'x|f.c:f|x[{4,5}-{1,2}]:|x|%s|NPSonly' % WEIGHT,
                  'x|t.props.Dim.80+@|x[{4,5}-{1,2}]:||%s|test' % WEIGHT),
                 ('x|f.c:f|x[{1,2}],x[{4,5}],x[{4,5}-{1,2}]:||%s|NPS' % WEIGHT,
                  'x|f.c:f|x[{1,2}],x[{4,5}],x[{4,5}-{1,2}]:|y|%s|NPS' % WEIGHT,
                  'x|f.c:f|x[{1,2}],x[{4,5}],x[{4,5}-{1,2}]:|x|%s|NPS' % WEIGHT,
                  'x|t.props.Dim.80+@|x[{1,2}],x[{4,5}],x[{4,5}-{1,2}]:||%s|test' % WEIGHT),
                 ('x|d.mean|x:||%s|stat' % WEIGHT,
                  'x|t.means.Dim.80+@|x:||%s|test' % WEIGHT),
                  'x|d.stddev|x:||%s|stat' % WEIGHT,
                  'x|d.median|x:||%s|stat' % WEIGHT,
                  'x|d.var|x:||%s|stat' % WEIGHT,
                  'x|d.varcoeff|x:||%s|stat' % WEIGHT,
                  'x|d.sem|x:||%s|stat' % WEIGHT,
                  'x|d.lower_q|x:||%s|stat' % WEIGHT,
                  'x|d.upper_q|x:||%s|stat' % WEIGHT,
                 ('x|f.c:f|x:||%s|counts_sum' % WEIGHT,
                  'x|f.c:f|x:|y|%s|c%%_sum' % WEIGHT),
                 #('x|f.c:f|x++:||%s|counts_cumsum' % WEIGHT,
                 # 'x|f.c:f|x++:|y|%s|c%%_cumsum' % WEIGHT)
                )

    if CA1:
        chains = ChainManager(stack)

        chains.get(data_key=DATA_KEY, filter_key=FILTER_KEY,
                   x_keys=X_KEYS[:-1], y_keys=Y_KEYS,
                   views=VIEW_KEYS, orient=ORIENT,
                   )
        VIEW_KEYS = ('x|f|x:|||cbase',
                     'x|f|x:||%s|cbase' % WEIGHT,
                     'x|f|x:|||cbase_gross',
                     'x|f|x:||%s|cbase_gross' % WEIGHT,
                     'x|f|x:|||ebase',
                     'x|f|x:||%s|ebase' % WEIGHT,
                     ('x|f|x[{1,2}+],x[{4,5}+]*:||%s|BLOCK' % WEIGHT,
                      'x|f|x[{1,2}+],x[{4,5}+]*:|y|%s|BLOCK' % WEIGHT,
                      'x|f|x[{1,2}+],x[{4,5}+]*:|x|%s|BLOCK' % WEIGHT,
                      'x|t.props.Dim.80+@|x[{1,2}+],x[{4,5}+]*:||%s|test' % WEIGHT),
                     #('x|d.mean|x:||%s|stat' % WEIGHT,
                     # 'x|t.means.Dim.80+@|x:||%s|test' % WEIGHT),
                     # 'x|d.stddev|x:||%s|stat' % WEIGHT,
                     # 'x|d.median|x:||%s|stat' % WEIGHT,
                     # 'x|d.var|x:||%s|stat' % WEIGHT,
                     # 'x|d.varcoeff|x:||%s|stat' % WEIGHT,
                     # 'x|d.sem|x:||%s|stat' % WEIGHT,
                     # 'x|d.lower_q|x:||%s|stat' % WEIGHT,
                     # 'x|d.upper_q|x:||%s|stat' % WEIGHT,
                     ('x|f.c:f|x:||%s|counts_sum' % WEIGHT,
                      'x|f.c:f|x:|y|%s|c%%_sum' % WEIGHT),
                     #('x|f.c:f|x++:||%s|counts_cumsum' % WEIGHT,
                     # 'x|f.c:f|x++:|y|%s|c%%_cumsum' % WEIGHT)
                    )

        chains.get(data_key=DATA_KEY, filter_key=FILTER_KEY,
                   x_keys=X_KEYS[-1], y_keys=Y_KEYS,
                   views=VIEW_KEYS, orient=ORIENT,
                   )

        chains.paint_all(transform_tests='full', 
                         text_key='fake',
                         text_loc_x='x edits',
                         text_loc_y='y edits',
                        )

        # how to attach to single chain
        # 1. Add to tables
        # 2. Add formatting
        chains[0].annotations.set('Headder Title -- no reason', category='header', position='title')
        chains[0].annotations.set('Header Left -- explanation text', category='header', position='left')
        chains[0].annotations.set('Header Center -- mask text', category='header', position='center')
        chains[0].annotations.set('Notes -- base text', category='notes')
        chains[1].annotations.set('Header Center -- mask text', category='header', position='center')
        chains[1].annotations.set('Notes -- base text', category='notes')
        chains[2].annotations.set('Headder Title -- no reason', category='header', position='title')

    # ------------------------------------------------------------ dataframe
    if OEC:
        open_ends = data.loc[:, ['RecordNo', 'gender', 'age', 'q8', 'q8a', 'q9', 'q9a']]
        open_chain = ChainManager(stack)
        open_chain = open_chain.add(open_ends,
                                    meta_from=(DATA_KEY, FILTER_KEY),
                                    name='Open Ends')
        #open_chain.paint_all(text_key='en-GB', sep='. ', na_rep='__NA__')
        open_chain.paint_all(text_key='en-GB', sep='. ', na_rep='-')

        #open_ends = data.loc[:, ['RecordNo', 'gender', 'age', 'q2']]
        #open_chain = open_chain.add(open_ends,
        #                            meta_from=(DATA_KEY, FILTER_KEY),
        #                            )
        #
        #for x in iter(open_chain):
        #    print '\n', x

        #open_chain.paint_all(text_key='en-GB', sep='. ')
        #
        #for x in iter(open_chain):
        #    print '\n', x
    # ------------------------------------------------------------

    # ------------------------------------------------------------ arr. summaries
    stack.add_link(x='q5', y='@', views=VIEWS, weights=weights)
    stack.add_link(x='@', y='q5', views=VIEWS, weights=weights)

    nets_mapper = qp.ViewMapper(template={'method': qp.QuantipyViews().frequency,
                                          'kwargs': {'iterators': {'rel_to': rel_to},
                                                     'groups': 'Nets'}})
    nets_mapper.add_method(name='No', kwargs={'axis':    'x',
                                              'logic':   [{'No': [1, 2, 3]}],
                                              'text':    'Net: No',
                                              'combine': False})
    stack.add_link(x='q5', y='@', views=nets_mapper, weights=weights)
    stack.add_link(x='@', y='q5', views=nets_mapper, weights=weights)

    nets_mapper = qp.ViewMapper(template={'method': qp.QuantipyViews().frequency,
                                          'kwargs': {'iterators': {'rel_to': rel_to},
                                                     'groups': 'Nets'}})
    nets_mapper.add_method(name='Yes', kwargs={'axis':    'x',
                                               'logic':   [{'Yes': [4, 5, 97]}],
                                               'text':    'Net: Yes',
                                               'combine': False})
    stack.add_link(x='q5', y='@', views=nets_mapper, weights=weights)
    stack.add_link(x='@', y='q5', views=nets_mapper, weights=weights)

    kwargs = {'calc_only': False,
              'calc': {'text': {u'en-GB': u'Net YES'},
              'Net agreement': ('Net: Yes', sub, 'Net: No')},
              'axis': 'x',
              'logic': [{'text': {u'en-GB': 'Net: No'}, 'Net: No': [1, 2]},
                        {'text': {u'en-GB': 'Net: Yes'}, 'Net: Yes': [4, 5]}]}
    nets_mapper.add_method(name='NPS', kwargs=kwargs)
    kwargs = {'calc_only': True,
              'calc': {'text': {u'en-GB': u'Net YES'},
              'Net agreement (only)': ('Net: Yes', sub, 'Net: No')},
              'axis': 'x',
              'logic': [{'text': {u'en-GB': 'Net: No'}, 'Net: No': [1, 2]},
                        {'text': {u'en-GB': 'Net: Yes'}, 'Net: Yes': [4, 5]}]}
    nets_mapper.add_method(name='NPSonly', kwargs=kwargs)
    stack.add_link(x='q5', y='@', views=nets_mapper, weights=weights)
    stack.add_link(x='@', y='q5', views=nets_mapper, weights=weights)

    stats = ['mean', 'stddev', 'median', 'var', 'varcoeff', 'sem', 'lower_q', 'upper_q']

    for stat in stats:
        options = {'stats': stat,
                   'source': None,
                   'rescale': None,
                   'drop': False,
                   'exclude': None,
                   'axis': 'x',
                   'text': ''}
        view = qp.ViewMapper()
        view.make_template('descriptives')
        view.add_method('stat', kwargs=options)
        stack.add_link(x='@', y='q5', views=view, weights=weights)
        stack.add_link(x='q5', y='@', views=view, weights=weights)

    nets_mapper = qp.ViewMapper(template={'method': qp.QuantipyViews().frequency,
                                          'kwargs': {'iterators': {'rel_to': rel_to},
                                                     'groups': 'Nets'}})
    nets = [{'N1': [1, 2], 'text': {'en-GB': 'Waves 1 & 2 (NET)'}, 'expand': 'after'},
            {'N2': [4, 5], 'text': {'en-GB': 'Waves 4 & 5 (NET)'}, 'expand': 'after'}]
    nets_mapper.add_method(name='BLOCK', kwargs={'axis':      'x',
                                                 'logic':     nets,
                                                 'text':      'Net: ',
                                                 'combine':   False,
                                                 'complete':  True,
                                                 'expand':    'after'}
                                                 )
    stack.add_link(x='q5', y='@', views=nets_mapper, weights=weights)
    stack.add_link(x='@', y='q5', views=nets_mapper, weights=weights)

    VIEW_KEYS = ('x|f|x:|||cbase',
                 'x|f|x:||%s|cbase' % WEIGHT,
                 ('x|f|:||%s|counts' % WEIGHT,
                  'x|f|:|y|%s|c%%' % WEIGHT),
                 ('x|f|x[{1,2,3}]:||%s|No' % WEIGHT,
                  'x|f|x[{1,2,3}]:|y|%s|No' % WEIGHT),
                  #'x|f|x[{1,2,3}]:|x|%s|No' % WEIGHT),
                 ('x|f|x[{4,5,97}]:||%s|Yes' % WEIGHT,
                  'x|f|x[{4,5,97}]:|y|%s|Yes' % WEIGHT),
                  #'x|f|x[{4,5,97}]:|x|%s|Yes' % WEIGHT),
                 ('x|f.c:f|x[{4,5}-{1,2}]:||%s|NPSonly' % WEIGHT,
                  'x|f.c:f|x[{4,5}-{1,2}]:|y|%s|NPSonly' % WEIGHT),
                  #'x|f.c:f|x[{4,5}-{1,2}]:|x|%s|NPSonly' % WEIGHT),
                 ('x|f.c:f|x[{1,2}],x[{4,5}],x[{4,5}-{1,2}]:||%s|NPS' % WEIGHT,
                  'x|f.c:f|x[{1,2}],x[{4,5}],x[{4,5}-{1,2}]:|y|%s|NPS' % WEIGHT),
                  #'x|f.c:f|x[{1,2}],x[{4,5}],x[{4,5}-{1,2}]:|x|%s|NPS' % WEIGHT),
                 'x|d.mean|x:||%s|stat' % WEIGHT,
                 'x|d.stddev|x:||%s|stat' % WEIGHT,
                 'x|d.median|x:||%s|stat' % WEIGHT,
                 'x|d.var|x:||%s|stat' % WEIGHT,
                 'x|d.varcoeff|x:||%s|stat' % WEIGHT,
                 'x|d.sem|x:||%s|stat' % WEIGHT,
                 'x|d.lower_q|x:||%s|stat' % WEIGHT,
                 'x|d.upper_q|x:||%s|stat' % WEIGHT,
                 ('x|f.c:f|x:||%s|counts_sum' % WEIGHT,
                  'x|f.c:f|x:|y|%s|c%%_sum' % WEIGHT),
                )

    #VIEW_KEYS = ('x|f|x:|||cbase',
    #             'x|f|x:||%s|cbase' % WEIGHT,
    #             'x|f|:|y|%s|c%%' % WEIGHT,
    #             'x|f|x[{1,2,3}]:|y|%s|No' % WEIGHT,
    #             'x|f|x[{4,5,97}]:|y|%s|Yes' % WEIGHT,
    #             'x|f.c:f|x[{4,5}-{1,2}]:|y|%s|NPSonly' % WEIGHT,
    #             'x|f.c:f|x[{1,2}],x[{4,5}],x[{4,5}-{1,2}]:|y|%s|NPS' % WEIGHT,
    #             'x|d.mean|x:||%s|stat' % WEIGHT,
    #             'x|d.stddev|x:||%s|stat' % WEIGHT,
    #             'x|d.median|x:||%s|stat' % WEIGHT,
    #             'x|d.var|x:||%s|stat' % WEIGHT,
    #             'x|d.varcoeff|x:||%s|stat' % WEIGHT,
    #             'x|d.sem|x:||%s|stat' % WEIGHT,
    #             'x|d.lower_q|x:||%s|stat' % WEIGHT,
    #             'x|d.upper_q|x:||%s|stat' % WEIGHT,
    #            )

    if AC1:
        arr_chains_1 = ChainManager(stack)

        arr_chains_1.get(data_key=DATA_KEY,
                       filter_key=FILTER_KEY,
                       x_keys=['@'],
                       y_keys=['q5'],
                       views=VIEW_KEYS,
                      )

        arr_chains_1.paint_all()

    if AC0:
        arr_chains_0 = ChainManager(stack)

        arr_chains_0.get(data_key=DATA_KEY,
                       filter_key=FILTER_KEY,
                       x_keys=['q5'],
                       y_keys=['@'],
                       views=VIEW_KEYS,
                      )
        arr_chains_0.paint_all()
    # ------------------------------------------------------------

    # ------------------------------------------------------------ arr. summaries - block nets

    VIEW_KEYS = ('x|f|x:|||cbase',
                 'x|f|x:||%s|cbase' % WEIGHT,
                 ('x|f|x[{1,2}+],x[{4,5}+]*:||%s|BLOCK' % WEIGHT,
                  'x|f|x[{1,2}+],x[{4,5}+]*:|y|%s|BLOCK' % WEIGHT),
                 'x|d.mean|x:||%s|stat' % WEIGHT,
                 'x|d.stddev|x:||%s|stat' % WEIGHT,
                 'x|d.median|x:||%s|stat' % WEIGHT,
                 'x|d.var|x:||%s|stat' % WEIGHT,
                 'x|d.varcoeff|x:||%s|stat' % WEIGHT,
                 'x|d.sem|x:||%s|stat' % WEIGHT,
                 'x|d.lower_q|x:||%s|stat' % WEIGHT,
                 'x|d.upper_q|x:||%s|stat' % WEIGHT,
                 ('x|f.c:f|x:||%s|counts_sum' % WEIGHT,
                  'x|f.c:f|x:|y|%s|c%%_sum' % WEIGHT),
                 #('x|f.c:f|x++:||%s|counts_cumsum' % WEIGHT,
                 # 'x|f.c:f|x++:|y|%s|c%%_cumsum' % WEIGHT)
                )

    if ACB1:
        arr_chains_block_1 = ChainManager(stack)
        arr_chains_block_1.get(data_key=DATA_KEY,
                               filter_key=FILTER_KEY,
                               x_keys=['@'],
                               y_keys=['q5'],
                               views=VIEW_KEYS,
                              )
        arr_chains_block_1.paint_all()

    if ACB0:
        arr_chains_block_0 = ChainManager(stack)
        arr_chains_block_0.get(data_key=DATA_KEY,
                               filter_key=FILTER_KEY,
                               x_keys=['q5'],
                               y_keys=['@'],
                               views=VIEW_KEYS,
                              )
        arr_chains_block_0.paint_all()
    # ------------------------------------------------------------

    # ------------------------------------------------------------ arr. summaries - mean
    VIEW_KEYS = ('x|f|x:|||cbase',
                 'x|f|x:||%s|cbase' % WEIGHT,
                 'x|d.mean|x:||%s|stat' % WEIGHT,
                 'x|d.stddev|x:||%s|stat' % WEIGHT,
                 'x|d.median|x:||%s|stat' % WEIGHT,
                 'x|d.var|x:||%s|stat' % WEIGHT,
                 'x|d.varcoeff|x:||%s|stat' % WEIGHT,
                 'x|d.sem|x:||%s|stat' % WEIGHT,
                 'x|d.lower_q|x:||%s|stat' % WEIGHT,
                 'x|d.upper_q|x:||%s|stat' % WEIGHT,
                )

    if ACM1:
        arr_chains_mean_1 = ChainManager(stack)

        arr_chains_mean_1.get(data_key=DATA_KEY,
                              filter_key=FILTER_KEY,
                              x_keys=['@'],
                              y_keys=['q5'],
                              views=VIEW_KEYS,
                             )

        arr_chains_mean_1.paint_all()

    if ACM0:
        arr_chains_mean_0 = ChainManager(stack)

        arr_chains_mean_0.get(data_key=DATA_KEY,
                              filter_key=FILTER_KEY,
                              x_keys=['q5'],
                              y_keys=['@'],
                              views=VIEW_KEYS,
                             )

        arr_chains_mean_0.paint_all()
    # ------------------------------------------------------------

    # table props - check editability
    table_properties = {
                            ### global properties

                            ### y
                            'bold_y': True,
                            'bg_color_y': '#B9FFCC',
                            'font_color_y': 'gray',
                            'font_name_y': 'Courier',
                            'font_size_y': 12,
                            'italic_y': True,
                            'text_v_align_y': 3,
                            'text_h_align_y': 1,

                            ### label
                            'bold_label': True,
                            'bg_color_label': 'red',
                            'font_color_label': '#FFB6C1',
                            'font_name_label': 'Calibri',
                            'font_size_label': 11,
                            'italic_label': True,
                            'text_v_align_label': 1,
                            'text_h_align_label': 3,

                            ### u_c_base text
                            'bold_u_c_base_text': True,
                            'bg_color_u_c_base_text': 'green',
                            'font_color_u_c_base_text': '#AB94FF',
                            'font_name_u_c_base_text': 'Helvetica',
                            'font_size_u_c_base_text': 11,
                            'italic_u_c_base_text': True,
                            'text_v_align_u_c_base_text': 3,
                            'text_h_align_u_c_base_text': 2,

                            ### u_c_base
                            'bold_u_c_base': True,
                            'bg_color_u_c_base': '#AB94FF',
                            'font_color_u_c_base': 'green',
                            'font_name_u_c_base': 'Helvetica',
                            'font_size_u_c_base': 11,
                            'italic_u_c_base': True,
                            'text_v_align_u_c_base': 3,
                            'text_h_align_u_c_base': 3,

                            ### c_base text
                            'bold_c_base_text': True,
                            'bg_color_c_base_text': '#AB94FF',
                            'font_color_c_base_text': 'green',
                            'font_name_c_base_text': 'Broadway',
                            'font_size_c_base_text': 10,
                            'italic_c_base_text': True,
                            'text_v_align_c_base_text': 1,
                            'text_h_align_c_base_text': 1,

                            ### c_base
                            'bold_c_base': True,
                            'bg_color_c_base': 'green',
                            'font_color_c_base': '#AB94FF',
                            'font_name_c_base': 'Broadway',
                            'font_size_c_base': 10,
                            'italic_c_base': True,
                            'text_v_align_c_base': 1,
                            'text_h_align_c_base': 1,

                            ### u_c_base_gross text
                            'bold_u_c_base_gross_text': True,
                            'bg_color_u_c_base_gross_text': '#DAF7A6',
                            'font_color_u_c_base_gross_text': '#7DCEA0',
                            'font_name_u_c_base_gross_text': 'Helvetica',
                            'font_size_u_c_base_gross_text': 11,
                            'italic_u_c_base_gross_text': True,
                            'text_v_align_u_c_base_gross_text': 3,
                            'text_h_align_u_c_base_gross_text': 2,

                            ### u_c_base_gross
                            'bold_u_c_base_gross': True,
                            'bg_color_u_c_base_gross': '#7DCEA0',
                            'font_color_u_c_base_gross': '#DAF7A6',
                            'font_name_u_c_base_gross': 'Helvetica',
                            'font_size_u_c_base_gross': 11,
                            'italic_u_c_base_gross': True,
                            'text_v_align_u_c_base_gross': 3,
                            'text_h_align_u_c_base_gross': 3,

                            ### c_base_gross text
                            'bold_c_base_gross_text': True,
                            'bg_color_c_base_gross_text': '#7DCEA0',
                            'font_color_c_base_gross_text': '#DAF7A6',
                            'font_name_c_base_gross_text': 'Broadway',
                            'font_size_c_base_gross_text': 10,
                            'italic_c_base_gross_text': True,
                            'text_v_align_c_base_gross_text': 1,
                            'text_h_align_c_base_gross_text': 1,

                            ### c_base_gross
                            'bold_c_base_gross': True,
                            'bg_color_c_base_gross': '#DAF7A6',
                            'font_color_c_base_gross': '#7DCEA0',
                            'font_name_c_base_gross': 'Broadway',
                            'font_size_c_base_gross': 10,
                            'italic_c_base_gross': True,
                            'text_v_align_c_base_gross': 1,
                            'text_h_align_c_base_gross': 1,

                            ### u_e_base text
                            'bold_u_e_base_text': True,
                            'bg_color_u_e_base_text': '#839192',
                            'font_color_u_e_base_text': '#E5F315',
                            'font_name_u_e_base_text': 'Helvetica',
                            'font_size_u_e_base_text': 11,
                            'italic_u_e_base_text': True,
                            'text_v_align_u_e_base_text': 3,
                            'text_h_align_u_e_base_text': 2,

                            ### u_e_base
                            'bold_u_e_base': True,
                            'bg_color_u_e_base': '#E5F315',
                            'font_color_u_e_base': '#839192',
                            'font_name_u_e_base': 'Helvetica',
                            'font_size_u_e_base': 11,
                            'italic_u_e_base': True,
                            'text_v_align_u_e_base': 3,
                            'text_h_align_u_e_base': 3,

                            ### e_base text
                            'bold_e_base_text': True,
                            'bg_color_e_base_text': '#E5F315',
                            'font_color_e_base_text': '#839192',
                            'font_name_e_base_text': 'Broadway',
                            'font_size_e_base_text': 10,
                            'italic_e_base_text': True,
                            'text_v_align_e_base_text': 1,
                            'text_h_align_e_base_text': 1,

                            ### e_base
                            'bold_e_base': True,
                            'bg_color_e_base': '#839192',
                            'font_color_e_base': '#E5F315',
                            'font_name_e_base': 'Broadway',
                            'font_size_e_base': 10,
                            'italic_e_base': True,
                            'text_v_align_e_base': 1,
                            'text_h_align_e_base': 1,

                            ### counts text
                            'bold_counts_text': True,
                            'bg_color_counts_text': '#8B4513',
                            'font_color_counts_text': '#CD853F',
                            'font_name_counts_text': 'FreeSerif',
                            'font_size_counts_text': 13,
                            'italic_counts_text': True,
                            'text_v_align_counts_text': 3,
                            'text_h_align_counts_text': 3,

                            ### counts
                            'bold_counts': True,
                            'bg_color_counts': '#CD853F',
                            'font_color_counts': '#8B4513',
                            'font_name_counts': 'FreeSerif',
                            'font_size_counts': 12,
                            'italic_counts': True,
                            'text_v_align_counts': 3,
                            'text_h_align_counts': 3,

                            'view_border_counts': None, # experimental

                            ### c_pct text
                            'bold_c_pct_text': True,
                            'bg_color_c_pct_text': '#CD853F',
                            'font_color_c_pct_text': '#8B4513',
                            'font_name_c_pct_text': 'FreeSerif',
                            'font_size_c_pct_text': 12,
                            'italic_c_pct_text': True,
                            'text_v_align_c_pct_text': 1,
                            'text_h_align_c_pct_text': 1,

                            ### c_pct
                            'bold_c_pct': True,
                            'bg_color_c_pct': '#8B4513',
                            'font_color_c_pct': '#CD853F',
                            'font_name_c_pct': 'FreeSerif',
                            'font_size_c_pct': 13,
                            'italic_c_pct': True,
                            'text_v_align_c_pct': 1,
                            'text_h_align_c_pct': 1,

                            ### r_pct text
                            'bold_r_pct_text': True,
                            'bg_color_r_pct_text': '#8B4513',
                            'font_text_color_r_pct': '#CD853F',
                            'font_text_name_r_pct': 'FreeSerif',
                            'font_size_r_pct_text': 12,
                            'it_textalic_r_pct': True,
                            'text_v_align_r_pct_text': 1,
                            'text_h_align_r_pct_text': 1,

                            ### r_pct
                            'bold_r_pct': True,
                            'bg_color_r_pct':  '#CD853F',
                            'font_color_r_pct': '#8B4513',
                            'font_name_r_pct': 'FreeSerif',
                            'font_size_r_pct': 13,
                            'italic_r_pct': True,
                            'text_v_align_r_pct': 1,
                            'text_h_align_r_pct': 1,

                            ### propstest text
                            'bold_propstest_text': True,
                            'bg_color_propstest_text': '#98FB98',
                            'font_color_propstest_text': '#7DCEA0',
                            'font_name_propstest_text': 'Liberation Sans Narrow',
                            'font_size_propstest_text': 11,
                            'italic_propstest_text': True,
                            'text_v_align_propstest_text': 1,
                            'text_h_align_propstest_text': 1,

                            ### propstest
                            'bold_propstest': True,
                            'bg_color_propstest': '#7DCEA0',
                            'font_color_propstest': '#98FB98',
                            'font_name_propstest': 'Liberation Sans Narrow',
                            'font_size_propstest': 10,
                            'italic_propstest': True,
                            'text_v_align_propstest': 1,
                            'text_h_align_propstest': 3,

                            ### net counts text
                            'bold_net_counts_text': True,
                            'bg_color_net_counts_text': '#B2DFEE',
                            'font_color_net_counts_text': '#FF5733',
                            'font_name_net_counts_text': 'Century Schoolbook L',
                            'font_size_net_counts_text': 11,
                            'italic_net_counts_text': True,
                            'text_v_align_net_counts_text': 1,
                            'text_h_align_net_counts_text': 1,

                            ### net counts
                            'bold_net_counts': True,
                            'bg_color_net_counts': '#B2DFEE',
                            'font_color_net_counts': '#FF5733',
                            'font_name_net_counts': 'Century Schoolbook L',
                            'font_size_net_counts': 13,
                            'italic_net_counts': True,
                            'text_v_align_net_counts': 1,
                            'text_h_align_net_counts': 1,

                            ### net c pct text
                            'bold_net_c_pct_text': True,
                            'bg_color_net_c_pct_text': '#FF5733',
                            'font_color_net_c_pct_text': '#B2DFEE',
                            'font_name_net_c_pct_text': 'Century Schoolbook L',
                            'font_size_net_c_pct_text': 11,
                            'italic_net_c_pct_text': True,
                            'text_v_align_net_c_pct_text': 1,
                            'text_h_align_net_c_pct_text': 1,

                            ### net c pct
                            'bold_net_c_pct': True,
                            'bg_color_net_c_pct': '#FF5733',
                            'font_color_net_c_pct': '#B2DFEE',
                            'font_name_net_c_pct': 'Century Schoolbook L',
                            'font_size_net_c_pct': 13,
                            'italic_net_c_pct': True,
                            'text_v_align_net_c_pct': 1,
                            'text_h_align_net_c_pct': 1,

                            ### net r pct text
                            'bold_net_r_pct_text': True,
                            'bg_color_net_r_pct_text': '#B2DFEE',
                            'font_color_net_r_pct_text': '#FF5733',
                            'font_name_net_r_pct_text': 'Century Schoolbook L',
                            'font_size_net_r_pct_text': 11,
                            'italic_net_r_pct_text': True,
                            'text_v_align_net_r_pct_text': 1,
                            'text_h_align_net_r_pct_text': 1,

                            ### net r pct
                            'bold_net_r_pct': True,
                            'bg_color_net_r_pct': '#B2DFEE',
                            'font_color_net_r_pct': '#FF5733',
                            'font_name_net_r_pct': 'Century Schoolbook L',
                            'font_size_net_r_pct': 13,
                            'italic_net_r_pct': True,
                            'text_v_align_net_r_pct': 1,
                            'text_h_align_net_r_pct': 1,

                            ### net_propstest text
                            'bold_net_propstest_text': True,
                            'bg_color_net_propstest_text': '#FF5733',
                            'font_color_net_propstest_text': '#B2DFEE',
                            'font_name_net_propstest_text': 'Century Schoolbook L',
                            'font_size_net_propstest_text': 11,
                            'italic_net_propstest_text': True,
                            'text_v_align_net_propstest_text': 1,
                            'text_h_align_net_propstest_text': 1,

                            ### net_propstest
                            'bold_net_propstest': True,
                            'bg_color_net_propstest': '#FF5733',
                            'font_color_net_propstest': '#B2DFEE',
                            'font_name_net_propstest': 'Century Schoolbook L',
                            'font_size_net_propstest': 13,
                            'italic_net_propstest': True,
                            'text_v_align_net_propstest': 1,
                            'text_h_align_net_propstest': 1,

                            ### block_calc_net_counts text
                            'bold_block_calc_net_counts_text': True,
                            'bg_color_block_calc_net_counts_text': '#839192',
                            'font_color_block_calc_net_counts_text': '#F8C471F',
                            'font_name_block_calc_net_counts_text': 'Century Schoolbook L',
                            'font_size_block_calc_net_counts_text': 11,
                            'italic_block_calc_net_counts_text': True,
                            'text_v_align_block_calc_net_counts_text': 1,
                            'text_h_align_block_calc_net_counts_text': 1,

                            ###block_calc_net_counts
                            'bold_block_calc_net_counts': True,
                            'bg_color_block_calc_net_counts': '#F8C471F',
                            'font_color_block_calc_net_counts': '#839192',
                            'font_name_block_calc_net_counts': 'Century Schoolbook L',
                            'font_size_block_calc_net_counts': 13,
                            'italic_block_calc_net_counts': True,
                            'text_v_align_block_calc_net_counts': 1,
                            'text_h_align_block_calc_net_counts': 1,

                            ### block_calc_net_c_pct text
                            'bold_block_calc_net_c_pct_text': True,
                            'bg_color_block_calc_net_c_pct_text': '#F8C471F',
                            'font_color_block_calc_net_c_pct_text': '#839192',
                            'font_name_block_calc_net_c_pct_text': 'Century Schoolbook L',
                            'font_size_block_calc_net_c_pct_text': 11,
                            'italic_block_calc_net_c_pct_text': True,
                            'text_v_align_block_calc_net_c_pct_text': 1,
                            'text_h_align_block_calc_net_c_pct_text': 1,

                            ### block_calc_net_c_pct
                            'bold_block_calc_net_c_pct': True,
                            'bg_color_block_calc_net_c_pct': '#839192',
                            'font_color_block_calc_net_c_pct': '#F8C471F',
                            'font_name_block_calc_net_c_pct': 'Century Schoolbook L',
                            'font_size_block_calc_net_c_pct': 13,
                            'italic_block_calc_net_c_pct': True,
                            'text_v_align_block_calc_net_c_pct': 1,
                            'text_h_align_block_calc_net_c_pct': 1,

                            ### block_calc_net_r_pct text
                            'bold_block_calc_net_r_pct_text': True,
                            'bg_color_block_calc_net_r_pct_text': '#839192',
                            'font_color_block_calc_net_r_pct_text': '#F8C471F',
                            'font_name_block_calc_net_r_pct_text': 'Century Schoolbook L',
                            'font_size_block_calc_net_r_pct_text': 11,
                            'italic_block_calc_net_r_pct_text': True,
                            'text_v_align_block_calc_net_r_pct_text': 1,
                            'text_h_align_block_calc_net_r_pct_text': 1,

                            ### block_calc_net_r_pct
                            'bold_block_calc_net_r_pct': True,
                            'bg_color_block_calc_net_r_pct': '#F8C471F',
                            'font_color_block_calc_net_r_pct': '#839192',
                            'font_name_block_calc_net_r_pct': 'Century Schoolbook L',
                            'font_size_block_calc_net_r_pct': 13,
                            'italic_block_calc_net_r_pct': True,
                            'text_v_align_block_calc_net_r_pct': 1,
                            'text_h_align_block_calc_net_r_pct': 1,

                            ### block_calc_net_propstest text
                            'bold_block_calc_net_propstest_text': True,
                            'bg_color_block_calc_net_propstest_text': '#F8C471F',
                            'font_color_block_calc_net_propstest_text': '#839192',
                            'font_name_block_calc_net_propstest_text': 'Century Schoolbook L',
                            'font_size_block_calc_net_propstest_text': 11,
                            'italic_block_calc_net_propstest_text': True,
                            'text_v_align_block_calc_net_propstest_text': 1,
                            'text_h_align_block_calc_net_propstest_text': 1,

                            ### block_calc_net_propstest
                            'bold_block_calc_net_propstest': True,
                            'bg_color_block_calc_net_propstest': '#839192',
                            'font_color_block_calc_net_propstest': '#F8C471F',
                            'font_name_block_calc_net_propstest': 'Century Schoolbook L',
                            'font_size_block_calc_net_propstest': 13,
                            'italic_block_calc_net_propstest': True,
                            'text_v_align_block_calc_net_propstest': 1,
                            'text_h_align_block_calc_net_propstest': 1,

                            ### block_calc_counts text
                            'bold_block_calc_counts_text': True,
                            'bg_color_block_calc_counts_text': 'blue',
                            'font_color_block_calc_counts_text': 'red',
                            'font_name_block_calc_counts_text': 'Century Schoolbook L',
                            'font_size_block_calc_counts_text': 11,
                            'italic_block_calc_counts_text': True,
                            'text_v_align_block_calc_counts_text': 1,
                            'text_h_align_block_calc_counts_text': 1,

                            ###block_calc_counts
                            'bold_block_calc_counts': True,
                            'bg_color_block_calc_counts': 'red',
                            'font_color_block_calc_counts': 'blue',
                            'font_name_block_calc_counts': 'Century Schoolbook L',
                            'font_size_block_calc_counts': 13,
                            'italic_block_calc_counts': True,
                            'text_v_align_block_calc_counts': 1,
                            'text_h_align_block_calc_counts': 1,

                            ### block_calc_c_pct text
                            'bold_block_calc_c_pct_text': True,
                            'bg_color_block_calc_c_pct_text': 'red',
                            'font_color_block_calc_c_pct_text': 'blue',
                            'font_name_block_calc_c_pct_text': 'Century Schoolbook L',
                            'font_size_block_calc_c_pct_text': 11,
                            'italic_block_calc_c_pct_text': True,
                            'text_v_align_block_calc_c_pct_text': 1,
                            'text_h_align_block_calc_c_pct_text': 1,

                            ### block_calc_c_pct
                            'bold_block_calc_c_pct': True,
                            'bg_color_block_calc_c_pct': 'blue',
                            'font_color_block_calc_c_pct': 'red',
                            'font_name_block_calc_c_pct': 'Century Schoolbook L',
                            'font_size_block_calc_c_pct': 13,
                            'italic_block_calc_c_pct': True,
                            'text_v_align_block_calc_c_pct': 1,
                            'text_h_align_block_calc_c_pct': 1,

                            ### block_calc_r_pct text
                            'bold_block_calc_r_pct_text': True,
                            'bg_color_block_calc_r_pct_text': 'blue',
                            'font_color_block_calc_r_pct_text': 'red',
                            'font_name_block_calc_r_pct_text': 'Century Schoolbook L',
                            'font_size_block_calc_r_pct_text': 11,
                            'italic_block_calc_r_pct_text': True,
                            'text_v_align_block_calc_r_pct_text': 1,
                            'text_h_align_block_calc_r_pct_text': 1,

                            ### block_calc_r_pct
                            'bold_block_calc_r_pct': True,
                            'bg_color_block_calc_r_pct': 'red',
                            'font_color_block_calc_r_pct': 'blue',
                            'font_name_block_calc_r_pct': 'Century Schoolbook L',
                            'font_size_block_calc_r_pct': 13,
                            'italic_block_calc_r_pct': True,
                            'text_v_align_block_calc_r_pct': 1,
                            'text_h_align_block_calc_r_pct': 1,

                            ### block_calc_propstest text
                            'bold_block_calc_propstest_text': True,
                            'bg_color_block_calc_propstest_text': 'red',
                            'font_color_block_calc_propstest_text': 'blue',
                            'font_name_block_calc_propstest_text': 'Century Schoolbook L',
                            'font_size_block_calc_propstest_text': 11,
                            'italic_block_calc_propstest_text': True,
                            'text_v_align_block_calc_propstest_text': 1,
                            'text_h_align_block_calc_propstest_text': 1,

                            ### block_calc_propstest
                            'bold_block_calc_propstest': True,
                            'bg_color_block_calc_propstest': 'blue',
                            'font_color_block_calc_propstest': 'red',
                            'font_name_block_calc_propstest': 'Century Schoolbook L',
                            'font_size_block_calc_propstest': 13,
                            'italic_block_calc_propstest': True,
                            'text_v_align_block_calc_propstest': 1,
                            'text_h_align_block_calc_propstest': 1,

                            ### block_net text
                            'bold_block_net_text': True,
                            'bg_color_block_net_text': '#15F3BB',
                            'font_color_block_net_text': '#F31588',
                            'font_name_block_net_text': 'Century Schoolbook L',
                            'font_size_block_net_text': 11,
                            'italic_block_net_text': True,
                            'text_v_align_block_net_text': 1,
                            'text_h_align_block_net_text': 1,

                            ### block_net
                            'bold_block_net': True,
                            'bg_color_block_net': '#F31588',
                            'font_color_block_net': '#15F3BB',
                            'font_name_block_net': 'Century Schoolbook L',
                            'font_size_block_net': 13,
                            'italic_block_net': True,
                            'text_v_align_block_net': 1,
                            'text_h_align_block_net': 1,

                            ### block_expanded text
                            'bold_block_expanded_text': True,
                            'bg_color_block_expanded_text': '#F08080',
                            'font_color_block_expanded_text': '#FCF3CF',
                            'font_name_block_expanded_text': 'Century Schoolbook L',
                            'font_size_block_expanded_text': 11,
                            'italic_block_expanded_text': True,
                            'text_v_align_block_expanded_text': 1,
                            'text_h_align_block_expanded_text': 1,

                            ### blockexpanded_
                            'bold_block_expanded': True,
                            'bg_color_block_expanded': '#FCF3CF',
                            'font_color_block_expanded': '#F08080',
                            'font_name_block_expanded': 'Century Schoolbook L',
                            'font_size_block_expanded': 13,
                            'italic_block_expanded': True,
                            'text_v_align_block_expanded': 1,
                            'text_h_align_block_expanded': 1,

                            ### block_normal text
                            'bold_block_normal_text': True,
                            'bg_color_block_normal_text': '#00BFFF',
                            'font_color_block_normal_text': '#F08080',
                            'font_name_block_normal_text': 'Century Schoolbook L',
                            'font_size_block_normal_text': 11,
                            'italic_block_normal_text': True,
                            'tnormalign_block_expanded_text': 1,
                            'tnormalign_block_expanded_text': 1,

                            ### block_normal
                            'bold_block_normal': True,
                            'bg_color_block_normal': '#F08080',
                            'font_color_block_normal': '#00BFFF',
                            'font_name_block_normal': 'Century Schoolbook L',
                            'font_size_block_normal': 13,
                            'italic_block_normal': True,
                            'text_v_align_block_normal': 1,
                            'text_h_align_block_normal': 1,

                            ### mean text
                            'bold_mean_text': True,
                            'bg_color_mean_text': '#FF69B4',
                            'font_color_mean_text': '#00E5EE',
                            'font_name_mean_text': 'MathJax_SanSerif',
                            'font_size_mean_text': 13,
                            'italic_mean_text': True,
                            'text_v_align_mean_text': 3,
                            'text_h_align_mean_text': 3,

                            ### mean
                            'bold_mean': True,
                            'bg_color_mean': '#FF69B4',
                            'font_color_mean': '#00E5EE',
                            'font_name_mean': 'MathJax_SanSerif',
                            'font_size_mean': 11,
                            'italic_mean': True,
                            'text_v_align_mean': 3,
                            'text_h_align_mean': 3,

                            ### stddev text
                            'bold_stddev_text': True,
                            'bg_color_stddev_text': '#FF69B4',
                            'font_color_stddev_text': '#00E5EE',
                            'font_name_stddev_text': 'MathJax_SanSerif',
                            'font_size_stddev_text': 13,
                            'italic_stddev_text': True,
                            'text_v_align_stddev_text': 3,
                            'text_h_align_stddev_text': 3,

                            ### stddev
                            'bold_stddev': True,
                            'bg_color_stddev': '#FF69B4',
                            'font_color_stddev': '#00E5EE',
                            'font_name_stddev': 'MathJax_SanSerif',
                            'font_size_stddev': 11,
                            'italic_stddev': True,
                            'text_v_align_stddev': 3,
                            'text_h_align_stddev': 3,

                            ### median text
                            'bold_median_text': True,
                            'bg_color_median_text': '#FF69B4',
                            'font_color_median_text': '#00E5EE',
                            'font_name_median_text': 'MathJax_SanSerif',
                            'font_size_median_text': 13,
                            'italic_median_text': True,
                            'text_v_align_median_text': 3,
                            'text_h_align_median_text': 3,

                            ### median
                            'bold_median': True,
                            'bg_color_median': '#FF69B4',
                            'font_color_median': '#00E5EE',
                            'font_name_median': 'MathJax_SanSerif',
                            'font_size_median': 11,
                            'italic_median': True,
                            'text_v_align_median': 3,
                            'text_h_align_median': 3,

                            ### var text
                            'bold_var_text': True,
                            'bg_color_var_text': '#FF69B4',
                            'font_color_var_text': '#00E5EE',
                            'font_name_var_text': 'MathJax_SanSerif',
                            'font_size_var_text': 13,
                            'italic_var_text': True,
                            'text_v_align_var_text': 3,
                            'text_h_align_var_text': 3,

                            ### var
                            'bold_var': True,
                            'bg_color_var': '#FF69B4',
                            'font_color_var': '#00E5EE',
                            'font_name_var': 'MathJax_SanSerif',
                            'font_size_var': 11,
                            'italic_var': True,
                            'text_v_align_var': 3,
                            'text_h_align_var': 3,

                            ### varcoeff text
                            'bold_varcoeff_text': True,
                            'bg_color_varcoeff_text': '#FF69B4',
                            'font_color_varcoeff_text': '#00E5EE',
                            'font_name_varcoeff_text': 'MathJax_SanSerif',
                            'font_size_varcoeff_text': 13,
                            'italic_varcoeff_text': True,
                            'text_v_align_varcoeff_text': 3,
                            'text_h_align_varcoeff_text': 3,

                            ### varcoeff
                            'bold_varcoeff': True,
                            'bg_color_varcoeff': '#FF69B4',
                            'font_color_varcoeff': '#00E5EE',
                            'font_name_varcoeff': 'MathJax_SanSerif',
                            'font_size_varcoeff': 11,
                            'italic_varcoeff': True,
                            'text_v_align_varcoeff': 3,
                            'text_h_align_varcoeff': 3,

                            ### sem text
                            'bold_sem_text': True,
                            'bg_color_sem_text': '#FF69B4',
                            'font_color_sem_text': '#00E5EE',
                            'font_name_sem_text': 'MathJax_SanSerif',
                            'font_size_sem_text': 13,
                            'italic_sem_text': True,
                            'text_v_align_sem_text': 3,
                            'text_h_align_sem_text': 3,

                            ### sem
                            'bold_sem': True,
                            'bg_color_sem': '#FF69B4',
                            'font_color_sem': '#00E5EE',
                            'font_name_sem': 'MathJax_SanSerif',
                            'font_size_sem': 11,
                            'italic_sem': True,
                            'text_v_align_sem': 3,
                            'text_h_align_sem': 3,

                            ### lower_q text
                            'bold_lower_q_text': True,
                            'bg_color_lower_q_text': '#FF69B4',
                            'font_color_lower_q_text': '#00E5EE',
                            'font_name_lower_q_text': 'MathJax_SanSerif',
                            'font_size_lower_q_text': 13,
                            'italic_lower_q_text': True,
                            'text_v_align_lower_q_text': 3,
                            'text_h_align_lower_q_text': 3,

                            ### lower_q
                            'bold_lower_q': True,
                            'bg_color_lower_q': '#FF69B4',
                            'font_color_lower_q': '#00E5EE',
                            'font_name_lower_q': 'MathJax_SanSerif',
                            'font_size_lower_q': 11,
                            'italic_lower_q': True,
                            'text_v_align_lower_q': 3,
                            'text_h_align_lower_q': 3,

                            ### upper_q text
                            'bold_upper_q_text': True,
                            'bg_color_upper_q_text': '#FF69B4',
                            'font_color_upper_q_text': '#00E5EE',
                            'font_name_upper_q_text': 'MathJax_SanSerif',
                            'font_size_upper_q_text': 13,
                            'italic_upper_q_text': True,
                            'text_v_align_upper_q_text': 3,
                            'text_h_align_upper_q_text': 3,

                            ###upper_q
                            'bold_upper_q': True,
                            'bg_color_upper_q': '#FF69B4',
                            'font_color_upper_q': '#00E5EE',
                            'font_name_upper_q': 'MathJax_SanSerif',
                            'font_size_upper_q': 11,
                            'italic_upper_q': True,
                            'text_v_align_upper_q': 3,
                            'text_h_align_upper_q': 3,

                            ### meanstest text
                            'bold_meanstest_text': True,
                            'bg_color_meanstest_text': '#00E5EE',
                            'font_color_meanstest_text': '#FF69B4',
                            'font_name_meanstest_text': 'MathJax_SanSerif',
                            'font_size_meanstest_text': 11,
                            'italic_meanstest_text': True,
                            'text_v_align_meanstest_text': 3,
                            'text_h_align_meanstest_text': 3,

                            ### meanstest
                            'bold_meanstest': True,
                            'bg_color_meanstest': '#00E5EE',
                            'font_color_meanstest': '#FF69B4',
                            'font_name_meanstest': 'MathJax_SanSerif',
                            'font_size_meanstest': 13,
                            'italic_meanstest': True,
                            'text_v_align_meanstest': 3,
                            'text_h_align_meanstest': 3,

                            ### counts_sum text
                            'bold_counts_sum_text': True,
                            'bg_color_counts_sum_text': '#34495E',
                            'font_color_counts_sum_text': '#D4AC0D',
                            'font_name_counts_sum_text': 'URW Gothic L',
                            'font_size_counts_sum_text': 8,
                            'italic_counts_sum_text': True,
                            'text_v_align_counts_sum_text': 1,
                            'text_h_align_counts_sum_text': 1,

                            ### counts_sum
                            'bold_counts_sum': True,
                            'bg_color_counts_sum': '#34495E',
                            'font_color_counts_sum': '#D4AC0D',
                            'font_name_counts_sum': 'URW Gothic L',
                            'font_size_counts_sum': 10,
                            'italic_counts_sum': True,
                            'text_v_align_counts_sum': 1,
                            'text_h_align_counts_sum': 3,

                            ### c_pct_sum text
                            'bold_c_pct_sum_text': True,
                            'bg_color_c_pct_sum_text': '#D4AC0D',
                            'font_color_c_pct_sum_text': '#34495E',
                            'font_name_c_pct_sum_text': 'URW Gothic L',
                            'font_size_c_pct_sum_text': 8,
                            'italic_c_pct_sum_text': True,
                            'text_v_align_c_pct_sum_text': 1,
                            'text_h_align_c_pct_sum_text': 1,

                            ### c_pct_sum
                            'bold_c_pct_sum': True,
                            'bg_color_c_pct_sum': '#D4AC0D',
                            'font_color_c_pct_sum': '#34495E',
                            'font_name_c_pct_sum': 'URW Gothic L',
                            'font_size_c_pct_sum': 10,
                            'italic_c_pct_sum': True,
                            'text_v_align_c_pct_sum': 1,
                            'text_h_align_c_pct_sum': 3,

                           }

    table_properties_group = {
                              ### label
                              'bold_label': True,

                              ### u_base text
                              'bold_u_base_text': True,
                              'font_color_u_base_text': '#808080',
                              ### u_base
                              'font_color_u_base': '#808080',

                              ### base text
                              'bold_base_text': True,
                              'font_color_base_text': '#632523',
                              ### base
                              'font_color_base': '#632523',

                              ### c_base_gross text
                              'bold_c_base_gross_text': True,
                              'bg_color_c_base_gross_text': 'yellow',
                              'font_color_c_base_gross_text': 'pink',
                              ### c_base_gross text
                              'bold_c_base_gross': False,
                              'bg_color_c_base_gross': 'gray',
                              'font_color_c_base_gross': 'yellow',

                              ### freq
                              'italic_freq_text': True,
                              'font_color_freq_text': 'blue',
                              'font_color_freq': 'blue',
                              'view_border_freq': False,

                              # net
                              'font_color_net_text': '#FF0000',
                              'font_color_net': '#FF0000',

                              # stat
                              'font_color_stat_text': '#FF0000',
                              'font_color_stat': '#FF0000',

                              # sum
                              'bg_color_sum_text': '#333333',
                              'font_color_sum_text': '#FFA500',
                              'italic_sum': True,

                              # block
                              'bold_block_net_text': True,
                              'italic_block_expanded_text': True,
                              'italic_block_normal_text': False,

                              # header - left
                              'bold_header_left': True,
                              'font_color_header_left': '#FFFFFF',
                              'text_h_align_header_left': 1,
                              'bg_color_header_left': '#AF8272',

                              # header - center
                              'bold_header_center': True,
                              'font_color_header_center': '#FFFFFF',
                              'text_h_align_header_center': 1,
                              'bg_color_header_center': '#85AD6E',

                              # header - title
                              'bold_header_title': True,
                              'font_color_header_title': '#265E1A',
                              'text_h_align_header_title': 1,
                              'bg_color_header_title': '#DFF442',

                              # notes 
                              'bold_notes': True,
                              'font_color_notes': '#FF6DB8',
                              'text_h_align_notes': 1,
                              'bg_color_notes': '#6DFFFC',

                              # mask label
                              'bold_mask_label': True,
                              'bg_color_mask_label': '#BDB1D8',
                              'font_color_mask_label': '#33A59E',
                              'font_name_mask_label': 'Calibri',
                              'font_size_mask_label': 11,
                              'italic_mask_label': True,
                              'text_v_align_mask_label': 1,
                              'text_h_align_mask_label': 3,

                             }


    sheet_properties = dict()

    #test = 1
    #test = 2
    test = 3

    if test == 1:
        sheet_properties = dict(dummy_tests=True,
                                alternate_bg=False,
                                #alternate_bg=True,
                                start_row=7,
                                start_column=2,
                               )
        custom_vg = {
                'block_normal_counts': 'block_normal',
                'block_normal_c_pct': 'block_normal',
                'block_normal_r_pct': 'block_normal',
                'block_normal_propstest': 'block_normal'}
        #custom_vg = {}
        tp = table_properties
        image = None
    elif test == 2:
        sheet_properties = dict(dummy_tests=True,
                                alternate_bg=True,
                               )
        custom_vg = {'r_pct': 'sum',
                     'stddev': 'base',
                     #'net_c_pct': 'freq'
                     }
        tp = table_properties_group
        image = None
    elif test == 3:
        sheet_properties = dict(alternate_bg=True,
                                freq_0_rep=':',
                                stat_0_rep='#',
                                y_header_height=20,
                                y_row_height=40)
        custom_vg = {
                'block_expanded_counts': 'freq',
                'block_expanded_c_pct': 'freq',
                'block_expanded_r_pct': 'freq',
                'block_expanded_propstest': 'freq',
                'block_net_counts': 'freq',
                'block_net_c_pct': 'freq',
                'block_net_r_pct': 'freq',
                'block_net_propstest': 'freq',
                }
        tp = {'bg_color_freq': 'gray'}
        image = {'img_name': 'logo',
                 'img_url': './qplogo_invert.png',
                 'img_size': [110, 120],
                 'img_insert_x': 4,
                 'img_insert_y': 0,
                 'img_x_offset': 3,
                 'img_y_offset': 6}

    # -------------
    x = Excel('basic_excel.xlsx',
              views_groups=custom_vg,
              italicise_level=50,
              decimals=dict(N=0, P=2, D=1),
              #decimals=2,
              details=True,
              image=image,
              **tp)

    if CA1:
        x.add_chains(chains,
                     'S H E E T',
                     annotations=['Ann. 1', 'Ann. 2', 'Ann. 3', 'Ann. 4'],
                     **sheet_properties
                    )
    if AC1:
        x.add_chains(arr_chains_1,
                     'array summary 1',
                     #annotations=['Ann. 1', 'Ann. 2', 'Ann. 3', 'Ann. 4'],
                     **sheet_properties
                    )
    if ACB1:
        x.add_chains(arr_chains_block_1,
                     'block summary 1',
                     annotations=['Ann. 1', 'Ann. 2', 'Ann. 3', 'Ann. 4'],
                     **sheet_properties
                    )
    if ACM1:
        x.add_chains(arr_chains_mean_1,
                     'means summary 1',
                     annotations=['Ann. 1', 'Ann. 2', 'Ann. 3', 'Ann. 4'],
                     **sheet_properties
                    )
    if AC0:
        x.add_chains(arr_chains_0,
                     'array summary 0',
                     annotations=['Ann. 1', 'Ann. 2', 'Ann. 3', 'Ann. 4'],
                     **sheet_properties
                    )
    if ACB0:
        x.add_chains(arr_chains_block_0,
                     'block summary 0',
                     annotations=['Ann. 1', 'Ann. 2', 'Ann. 3', 'Ann. 4'],
                     **sheet_properties
                    )
    if ACM0:
        x.add_chains(arr_chains_mean_0,
                     'means summary 0',
                     annotations=['Ann. 1', 'Ann. 2', 'Ann. 3', 'Ann. 4'],
                     **sheet_properties
                    )
    if OEC:
        x.add_chains(open_chain,
                     'Open_Ends',
                     annotations=['Ann. 1', 'Ann. 2', 'Ann. 3', 'Ann. 4'],
                     **sheet_properties
                    )
    x.close()
    # -------------
