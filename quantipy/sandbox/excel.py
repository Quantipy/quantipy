# -* coding: utf-8 -*-

import os
import re
import numpy as np
import pandas as pd
import quantipy as qp

from quantipy.core.tools.qp_decorators import lazy_property
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
import warnings

warnings.simplefilter('ignore')

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
                       arrow_color_high='#2EB08C',
                       arrow_rep_high=u'\u25B2',
                       arrow_color_low='#FC8EAC',
                       arrow_rep_low=u'\u25BC',
                       column_width=9,
                       column_width_label=35,
                       column_width_frame=15,
                       dummy_tests=False,
                       freq_0_rep='-',
                       img_insert_x=0,
                       img_insert_y=0,
                       img_size=[130, 130],
                       img_x_offset=0,
                       img_y_offset=0,
                       in_memory=False,
                       row_height_label=12.75,
                       start_column=0,
                       start_row=0,
                       stat_0_rep='-',
                       y_header_height=33.75,
                       y_row_height=50
                       )


class Excel(Workbook):
    # TODO: docstring

    def __init__(self,
                 filename,
                 toc=False,
                 views_groups=None,
                 italicise_level=None,
                 details=False,
                 in_memory=False,
                 decimals=None,
                 image=None,
                 **formats):
        super(Excel, self).__init__()
        self.filename = filename
        self.toc = toc
        self.italicise_level = italicise_level
        self.details = details
        self.in_memory = in_memory
        self._views_groups = views_groups
        self._decimals = decimals
        self._image = image

        self._formats = ExcelFormats(self.views_groups, **formats)

    def __repr__(self):
        return 'Excel(%r)' % self.filename

    def __str__(self):
        return '%s' % self.filename

    def __del__(self):
        del self

    @lazy_property
    def views_groups(self):
        if self._views_groups:
            return dict([(k, self._views_groups.get(k, v))
                        for k, v in _VIEWS_GROUPS.iteritems()])
        return _VIEWS_GROUPS

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
            image.thumbnail(self._image.get('img_size',
                                            _SHEET_DEFAULTS['img_size']),
                            Image.ANTIALIAS)
            image.save(os.path.basename(self._image['img_url']))
        return self._image

    def add_chains(self, chains, sheet_name=None, annotations=None, **kwargs):
        # TODO: docstring
        warning_message = ('quantipy.ChainManager has folders, '
                           'sheet_name will be ignored')
        if chains.folders:
            if sheet_name:
                print UserWarning(warning_message)
            for chain in chains:
                if isinstance(chain, dict):
                    sheet_name = chain.keys()[0]
                    self._write_chains(chain[sheet_name], sheet_name,
                                       annotations, **kwargs)
                else:
                    self._write_chains((chain, ), chain.name,
                                       annotations, **kwargs)
        else:
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

    @lazy_property
    def image(self):
        return self.excel.image

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

        if self.image:

            self.insert_image(self.image.get('img_insert_x', self.img_insert_x),
                              self.image.get('img_insert_y', self.img_insert_y),
                              self.image['img_url'],
                              dict(x_offset=self.image.get('img_x_offset',
                                                           self.img_x_offset),
                                   y_offset=self.image.get('img_y_offset',
                                                           self.img_y_offset)))

    def _set_columns(self, columns):
        self.set_column(self._column, self._column, self.column_width_label)
        self.set_column(self._column + 1, self._column + columns.size,
                        self.column_width)

    def _set_freeze_loc(self, columns):
        if list(columns.labels[0]).count(0) == 1:
            offset = 1
        else:
            offset = 0
        self._freeze_loc = ((self._row + columns.nlevels),
                            (self._column + offset + 1))

    def _set_row(self, row):
        self.set_row(row, self.row_height_label)


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
            self.sheet.set_column(column, column, self.sheet.column_width_frame)
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

        for i in xrange(rel_x):
            self.sheet._set_row(self.sheet._row + i)

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
        for i in xrange(rel_x):
            self.sheet._set_row(self.sheet._row + i)
        self.sheet._row += rel_x

    def _write_annotations(self, names):
        for name in names:
            anno = getattr(self, name)
            if anno:
                self.sheet.write(self.sheet._row, self.sheet._column,
                                 anno[0], self.excel._formats[name])
                self._format_row(self.excel._formats[name])
                self.sheet._set_row(self.sheet._row)
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
        if isinstance(self.data, basestring):
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

