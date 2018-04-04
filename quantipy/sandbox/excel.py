# -* coding: utf-8 -*-

import numpy as np
import pandas as pd
import quantipy as qp
import cPickle as cp

from PIL import ImageFont
from excel_formats import ExcelFormats, _Format
from excel_formats_constants import _DEFAULTS, _VIEWS_GROUPS
from difflib import SequenceMatcher
from os.path import basename
from PIL import Image
from quantipy.core.tools.qp_decorators import lazy_property
from re import sub as rsub
from xlsxwriter import Workbook
from xlsxwriter.worksheet import Worksheet
from xlsxwriter.utility import xl_rowcol_to_cell

try:
    from functools import lru_cache
except ImportError:
    from functools32 import lru_cache

# Initialization data to pass to the worksheet.
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
               'constant_memory')

# Defaults for _Sheet.
_SHEET_DEFAULTS = dict(alternate_bg             = True,
                       arrow_color_high         = '#2EB08C',
                       arrow_rep_high           = u'\u25B2',
                       arrow_color_low          = '#FC8EAC',
                       arrow_rep_low            = u'\u25BC',
                       column_width             = 9,
                       column_width_label       = 35,
                       column_width_frame       = 15,
                       column_width_specific    = dict(),
                       dummy_tests              = False,
                       freq_0_rep               = '-',
                       img_insert_x             = 0,
                       img_insert_y             = 0,
                       img_size                 = [130, 130],
                       img_x_offset             = 0,
                       img_y_offset             = 0,
                       row_height_label         = 12.75,
                       start_column             = 0,
                       start_row                = 0,
                       start_column_annotations = 0,
                       start_row_annotations    = 0,
                       stat_0_rep               = '-',
                       y_header_height          = 33.75,
                       y_row_height             = 50)

class Excel(Workbook):
    """
    A class for writing ChainManager to Excel XLSX files.
    Uses the xlsxwriter lib.

    Parameters
    ----------
    filename : ``str``
        The path of the target XLSX file.
    italicise_level : ``int``, default ``None``
        Italicise column values with column base below given level.
    details : ``bool``, default ``False``
        A summary of the aggregated data displayed in the sheet.
        If any views contain column proportion tests against the
        total column, there are 2 additional cells to explain the
        representation of ``higher/ lower than`` result.
    annotations ``list-like``/ ``dict``: , default ``None``
        1. A ``list`` of annotation items. These items can be ``str`` or a
           ``list-like`` with ``str`` and ``dict``. The ``str`` are written
           before any chains are written in row ``start_row`` and column
           ``start_column`` using default formats or format spec provided in
           ``dict``. These will be applied to every sheet.
        2. A `dict`` with keys as ``Sheet`` names and a ``list-like`` as
           described in 1.
        see notes for more information on formats.
    decimals : ``int``/ ``dict``, default None
        The digits to round ``float`` data to in the underlying data.
        An ``int`` implies all ``float`` data to be rounded with the same
        digits. You can specify the following keys in a dict to use specific
        digits for different view types...
            N - count frequency views
            P - percentage frequency views
            D - descriptive views
    image : ``dict``, default None
        Describes the location, sizing and location of an image to insert
        in all ``Sheets``.
        The following key-vaue pairs must be provided:
            img_name - ``str`` (Arbitrary image name)
            img_url - ``str`` (The path to the image. PNG/ JPEG/ BMP format)
        The following key-value pairs are optional:
            img_insert_x - ``int``, default 0 (the cell row (zero indexed))
            img_insert_y - ``int``, default 0 (the cell column (zero indexed))
            img_size - ``list``, default [130, 130] (resize to [width, height])
            img_x_offset - ``int``, default 0 (offset by pixels - x axis)
            img_y_offset - ``int``, default 0 (offset by pixels - y axis)
    sheet_properties : ``dict``, default None
        Optional sheet properties (see Notes 1.).
    views_groups ``dict``: , default ``None``
        When modifying the default formatting, view types can be grouped
        to share a custom format (see Notes 2.).
    **formats : ``dict``
        The format specifications by view type (see Notes 2.).

    Notes
    -----

    1. ``sheet_properties``

        The following table shows the default sheet properties:

        Property                     Default value   Definition
        ~~~~~~~~                     ~~~~~~~~~~~~~   ~~~~~~~~~~
        ``alternate_bg``             ``True``        Alternate bg_color in freq views
        ``arrow_color_high``         ``'#2EB08C'``   High arrow color
        ``arrow_rep_high``           ``u'\u25B2'``   High arrow representation for column
                                                     proportion tests
        ``arrow_color_low``          ``'#FC8EAC'``   Low arrow color
        ``arrow_rep_low``            ``u'\u25BC'``   Low arrow representation for column
                                                     proportion tests
        ``column_width``             ``9``           Column width - column 1, ..., N
        ``column_width_label``       ``35``          Column width - column 0
        ``column_width_frame``       ``15``          Column width - Chain.structure
        ``column_width_specific``    ``dict``        Column width - specific column override
        ``dummy_tests``              ``False``       If column proportion/ mean tests in
                                                     views add dummy test rows for view
        ``freq_0_rep``               ``'-'``         Representation of zero data in
                                                     frequency views
        ``img_insert_x``             ``0``           Cell row to insert image
                                                     (zero indexed)
        ``img_insert_y``             ``0``           Cell column to insert image
                                                     (zero indexed)
        ``img_size``                 ``[130, 130]``  Resize image to [width, height]
        ``img_x_offset``             ``0``           Offset image by pixels - x axis
        ``img_y_offset``             ``0``           Offset image by pixels - y axis
        ``row_height_label``         ``12.75``       Row height for view rows
        ``start_column``             ``0``           Start column (zero indexed)
        ``start_row``                ``0``           Start row (zero indexed)
        ``start_column_annotations`` ``0``           Start column for annotations (zero indexed)
        ``start_row_annotations``    ``0``           Start row for annotations (zero indexed)
        ``stat_0_rep``               ``'-'``         Representation of zero data in
                                                     descriptive views
        ``y_header_height``          ``33.75``       y row hieght - header (pands.DataFrame.index.level #0)
        ``y_row_height``             ``50``          y row height (pands.DataFrame.index.level #1)

        To change the values pass a dict with the property / new value as
        key/ value pair.

        To modify a property for a specific sheet, pass a dict to the kwargs
        parameter in the ``quantipy.Excel.add_chains`` method.

        When you have a ``quantipy.ChainManager`` with many sheets the kwargs
        dict should contain a key with the sheet name and a value of the sheet
        properties as previously described. Otherwise you are simply updateing
        the property for every sheet.

        If ``start_row_annotations`` + number of annotations is greater than
        ``start_row`` then the start_row will become the row after the last
        annotation.

    2. ``formats``

        The following cell properties can be modified, globally or by view type:

        ``bg_color``
        ``bold``
        ``border_color``
        ``bottom``
        ``bottom_color``
        ``font_color``
        ``font_name``
        ``font_size``
        ``font_script``
        ``italic``
        ``left``
        ``left_color``
        ``num_format``
        ``right``
        ``right_color``
        ``text_v_align``
        ``text_h_align``
        ``text_wrap``
        ``top``
        ``top_color``

        (See xlsxwriter.readthedocs.io/format.html for more information on these)

        Add the keys above to set a global property, which ignores element/
        view type.

        For element/ view type, the key will start with one of the properties above, and must
        be concatenated with either an ``element`` or ``view type`` (or
        ``view group``) name, for example:

        To make the rows that contain base views italic, ``formats`` will
        contain a key "font_italic_base" with value ``True``.

        ~ Elements ~
        ~~~~~~~~~~~~
        The following elements refer to the parts of the Excel tables that
        are not part of the ``Chain.dataframe`` rows:

        ``y`` - for column headers
        ``label`` - question labels
        ``mask_label`` - mask labels, which use ``label`` otherwise
        ``data_header`` - column headers in a sheet from ``Chain.structure``
        ``header_left`` - ``Chain.annotation`` Header Left
        ``header_center`` - ``Chain.annotation`` Header Center
        ``header_title`` - ``Chain.annotation`` Header Title
        ``notes`` - ``Chain.annotation`` Notes

        ~ View types/ groups ~
        ~~~~~~~~~~~~~~~~~~~~~~
        These refer to the values in a row of a ``Chain.dataframe``, since a row
        will be one view type, such as ``cbase`` or ``counts``.

        To modify more than one view type in the same way, either use the
        default group (specified below) or a group name that is passed to the
        ``view_groups`` parameter.

        Each of the view types/ groups also have a corresponding text
        option for formatting the row label, such as ``cbase_text`` or
        ``net_text``.

        TYPE                            GROUP
        ~~~~                            ~~~~~
        ``default``                     ``default``
        ``label``                       ``label``
        ``mask_label``                  ``label``
        ``c_base``                      ``base``
        ``u_c_base``                    ``u_base``
        ``c_base_gross``                ``base``
        ``u_c_base_gross``              ``u_base``
        ``e_base``                      ``base``
        ``u_e_base``                    ``u_base``
        ``u_r_base``                    ``base``
        ``r_base``                      ``u_base``
        ``counts``                      ``freq``
        ``c_pct``                       ``freq``
        ``res_c_pct``                   ``freq``
        ``r_pct``                       ``freq``
        ``block_normal_counts``         ``freq``
        ``block_normal_c_pct``          ``freq``
        ``block_normal_r_pct``          ``freq``
        ``block_normal_propstest``      ``freq``
        ``propstest``                   ``freq``
        ``net_counts``                  ``net``
        ``net_c_pct``                   ``net``
        ``net_r_pct``                   ``net``
        ``net_propstest``               ``net``
        ``block_calc_net_counts``       ``net``
        ``block_calc_net_c_pct``        ``net``
        ``block_calc_net_r_pct``        ``net``
        ``block_calc_net_propstest``    ``net``
        ``block_calc_counts``           ``net``
        ``block_calc_c_pct``            ``net``
        ``block_calc_r_pct``            ``net``
        ``block_calc_propstest``        ``net``
        ``block_expanded_counts``       ``block_expanded``
        ``block_expanded_c_pct``        ``block_expanded``
        ``block_expanded_r_pct``        ``block_expanded``
        ``block_expanded_propstest``    ``block_expanded``
        ``block_net_counts``            ``block_net``
        ``block_net_c_pct``             ``block_net``
        ``block_net_r_pct``             ``block_net``
        ``block_net_propstest``         ``block_net``
        ``mean``                        ``stat``
        ``stddev``                      ``stat``
        ``min``                         ``stat``
        ``max``                         ``stat``
        ``median``                      ``stat``
        ``var``                         ``stat``
        ``varcoeff``                    ``stat``
        ``sem``                         ``stat``
        ``lower_q``                     ``stat``
        ``upper_q``                     ``stat``
        ``meanstest``                   ``stat``
        ``counts_sum``                  ``sum``
        ``c_pct_sum``                   ``sum``
        ``counts_cumsum``               ``sum``
        ``c_pct_cumsum``                ``sum``

    There is one exception to the above, "view_border_<view type>".
    Views which can have more than one row, e.g. counts or c%, will have an
    internal border between these rows. To remove these internal borders set
    this to ``None``.

    Examples
    --------
    annotations = dict(sheet_1=[('Text. 1', dict(font_size=8,
                                                 font_color='yellow',
                                                 bg_color='gray'))],
                       sheet_2=[('Text 2.a', dict(font_size=10,
                                                  font_color='pink',
                                                  bg_color='gray')),
                                ('Text 2.b', dict(font_size=12,
                                                  font_color='pink',
                                                  bg_color='blue')) ])

    view_groups = dict(block_expanded_counts='freq',
                       block_expanded_c_pct='freq',
                       block_expanded_r_pct='freq',
                       block_expanded_propstest='freq',
                       block_net_counts='freq',
                       block_net_c_pct='freq',
                       block_net_r_pct='freq',
                       block_net_propstest='freq')

    decimals = dict(N=0, P=2, D=1)

    image = dict(img_name='name',
                 img_url='/path/to/image.png',
                 img_size=[110, 120],
                 img_insert_x=4,
                 img_insert_y=0,
                 img_x_offset=3,
                 img_y_offset=6)

    sheet_properties = dict(alternate_bg=True,
                            dummy_tests=True,
                            row_height_label=15)

    formats = dict(bg_color_freq='gray',
                   bold_propstest=True,
                   font_size=8)

    excel = Excel('/path/to/file.xlsx',
                  italicise_level=50,
                  details=True,
                  annotations=annotations,
                  views_groups=view_groups,
                  decimals=decimals,
                  image=image,
                  sheet_properties=sheet_properties,
                  **formats)
    """
    def __init__(self, filename, italicise_level=None, details=False,
                 annotations=None, decimals=None, image=None,
                 sheet_properties=None, views_groups=None, **formats):
        super(Excel, self).__init__()
        self.filename = filename
        self.italicise_level = italicise_level
        self.details = details
        self.annotations = annotations
        self._views_groups = views_groups
        self._decimals = decimals
        self._image = image
        self._sheet_properties = sheet_properties

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
                                            self.sheet_properties['img_size']),
                            Image.ANTIALIAS)
            image.save(basename(self._image['img_url']))
        return self._image

    @lazy_property
    def sheet_properties(self):
        if self._sheet_properties:
            return dict([(key, self._sheet_properties.get(key, value))
                         for key, value in _SHEET_DEFAULTS.iteritems()])
        return dict(_SHEET_DEFAULTS)

    def _get_annotations(self, annotations, sheet_name):
        error_messaage = ('"annotations" passed to Excel() must either be a '
                          'list, for all sheets, or a dict, key=sheet name '
                          ': value=list of annotations')
        if self.annotations:
            if isinstance(self.annotations, list):
                return annotations or self.annotations
            elif isinstance(self.annotations, dict):
                sheet_anno = self.annotations.get(sheet_name,  annotations)
                return annotations or sheet_anno
            else:
                raise TypeError(error_message)
        return annotations

    def add_chains(self, chains, sheet_name=None, annotations=None, **kwargs):
        """
        Add a ChainManager to be written to Excel.Sheet(s). If the
        ChainManager has no folders the Chain objects are written into a single
        sheet. When there are folders, a Chain not in a folder is written to a
        single sheet and folders are written to a sheet.

        Parameters
        ----------
        chains : ``ChainManager``
            The object containing the ``Chain`` objects to be written.
        sheet_name : ``str``, default None
            The name of the sheet for a ChainManager without folders.
        annotations ``list-like``/ ``dict``: , default ``None``
            Same as in the ``Excel`` constructor but for a can be used here to
            overwrite the global annotations.
        **kwargs : ``dict``
            Optional arguments to update sheet properties for a given
            sheet.
        """
        warning_message = ('quantipy.ChainManager has folders, '
                           'sheet_name will be ignored')
        for key in self.sheet_properties.iterkeys():
            if key in kwargs:
                self.sheet_properties[key] = kwargs.pop(key)
        if chains.folders:
            if sheet_name:
                print UserWarning(warning_message)
            for chain in chains:
                this_sheet_properties = dict(self.sheet_properties)
                if isinstance(chain, dict):
                    sheet_name = chain.keys()[0]
                    if sheet_name in kwargs:
                        this_sheet_properties.update(kwargs[sheet_name])
                    self._write_chains(chain[sheet_name], sheet_name,
                                       self._get_annotations(annotations,
                                                             sheet_name),
                                       **this_sheet_properties)
                else:
                    if chain.name in kwargs:
                        this_sheet_properties.update(kwargs[chain.name])
                    self._write_chains((chain, ), chain.name,
                                       self._get_annotations(annotations,
                                                             chain.name),
                                       **this_sheet_properties)
        else:
            this_sheet_properties = dict(self.sheet_properties)
            if sheet_name in kwargs:
                this_sheet_properties.update(kwargs[sheet_name])
            self._write_chains(chains, sheet_name,
                               self._get_annotations(annotations, sheet_name),
                               **this_sheet_properties)

    def _write_chains(self, chains, sheet_name, annotations, **kwargs):
        worksheet = _Sheet(self, chains, sheet_name, annotations, **kwargs)

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
    def _add_format(self, **format_):
        return self.add_format(format_)


class _Sheet(Worksheet):

    def __init__(self, excel, chains, sheet_name, annotations, **kwargs):
        super(_Sheet, self).__init__()
        self.excel = excel
        self.chains = chains
        self.sheet_name = sheet_name
        self.annotations = annotations

        for attr, value in kwargs.iteritems():
            setattr(self, attr, value)

        self._row = self.start_row
        self._column = self.start_column
        self._row_annotations = self.start_row_annotations
        self._column_annotations = self.start_column_annotations
        self._formats = None
        self._freeze_loc = None
        self._columns = None
        self._test_letters = None
        self._column_edges = None
        self._view_keys = None
        self._group_order = None
        self._columns_set = None

    @property
    def formats(self):
        return self.excel._formats

    @lazy_property
    def test_letters(self):
        return self.chains[0].sig_test_letters

    @lazy_property
    def image(self):
        return self.excel.image

    @lazy_property
    def default_annotation_format(self):
        exclude_keys = ('border', 'border_style_int', 'border_style_ext',
                        'num_format_counts', 'num_format_default',
                        'num_format_c_pct', 'num_format_mean')
        format_spec = dict([(k, v)
                            for k, v in _DEFAULTS.iteritems()
                            if k not in exclude_keys])
        return self.excel._add_format(**_Format(**format_spec))

    @property
    def column_edges(self):
        if self._column_edges is None:
            self._column_edges = []
        return self._column_edges

    @property
    def columns_set(self):
        if self._columns_set is None:
            self._columns_set = set()
        return self._columns_set

    def write(self, *args):
        if isinstance(args[-1], dict):
            args = args[:-1] + (self.excel._add_format(**args[-1]), )
        super(_Sheet, self).write(*args)

    def write_rich_string(self, *args):
        args, rich_text = ((args[0], args[1]), ), args[2:]
        for arg in rich_text:
            if isinstance(arg, dict):
                args = args + (self.excel._add_format(**arg), )
            else:
                args = args + (arg, )
        args = (xl_rowcol_to_cell(*args[0]), ) + args[1:]
        super(_Sheet, self).write_rich_string(*args)

    def merge_range(self, *args):
        args = args[:-1] + (self.excel._add_format(**args[-1]), )
        super(_Sheet, self).merge_range(*args)

    def write_chains(self):
        write = self.write
        write_rich_string = self.write_rich_string

        if self.annotations:
            for annotation in self.annotations:
                try:
                    annotation, format_spec = annotation
                    format_ = self.excel._add_format(**_Format(**format_spec))
                except ValueError:
                    format_ = self.default_annotation_format

                write(self._row_annotations, self._column_annotations,
                      annotation, format_)
                self._row_annotations += 1
            if self._row_annotations > self._row:
                self._row = self._row_annotations

        for i, chain in enumerate(self.chains):

            try:
                columns = chain.dataframe.columns

                # make y-axis writing availbale to all chains
                if i == 0:
                    self._set_freeze_loc(columns)
                    self._set_column(columns)

            except AttributeError:
                columns = chain.structure.columns

            # write frame
            box = _Box(self, chain, self._row, self._column)
            box.to_sheet(columns=(i==0))

            del box

        if self._freeze_loc:
            self.freeze_panes(*self._freeze_loc)

        self.hide_gridlines(2)

        if self.excel.details and all(c.structure is None for c in self.chains):
            format_ = self.formats._cell_details
            cd = None
            arrow_descriptions = None
            for chain in self.chains:
                cds = chain.cell_details
                if len(cds) == 3 and not arrow_descriptions:
                    arrow_descriptions = cds[1:]
                if cd is None:
                    cd = cds[0]
                else:
                    if cd != cds[0]:
                        lists = (cd, cds[0])
                        long_ = max(lists, key=len)
                        short = min(lists, key=len)
                        sm = SequenceMatcher(None, long_, short)
                        for tag, i1, i2, j1, j2 in sm.get_opcodes():
                            if tag == 'insert':
                                long_ = long_[:i1] + short[j1:j2] + long_[i2:]
                        cd = long_

            write(self._row + 1, self._column + 1, cd, format_)
            if arrow_descriptions:
                arrow_format = _Format(**{'font_color': self.arrow_color_high})
                arrow_format = self.excel._add_format(**arrow_format)
                write_rich_string(self._row + 2, self._column + 1,
                                  arrow_format, self.arrow_rep_high,
                                  format_, cds[1], format_)
                arrow_format = _Format(**{'font_color': self.arrow_color_low})
                arrow_format = self.excel._add_format(**arrow_format)
                write_rich_string(self._row + 3, self._column + 1,
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

        if self.column_width_specific:
            for column, width in self.column_width_specific.iteritems():
                self.set_column(column, column, width)

    def set_column(self, *args):
        first_col, last_col = args[:2]
        if first_col == last_col:
            if first_col in self.columns_set:
                return
            self.columns_set.add(first_col)
        else:
            for column in xrange(first_col, last_col + 1):
                if column in self.columns_set:
                    return
            self.columns_set.add(list(xrange(first_col, last_col + 1)))
        super(_Sheet, self).set_column(*args)

    def _set_column(self, columns):
        column = self._column
        width = self.column_width_specific.get(column, self.column_width_label)
        self.set_column(column, column, width)
        for idx in xrange(columns.size):
            relative = column + idx + 1
            width = self.column_width_specific.get(relative, self.column_width)
            self.set_column(relative, relative, width)

    @lazy_property
    def truetype(self):
        fn = self.excel._formats.default_attributes['font_name']
        fs = self.excel._formats.default_attributes['font_size']
        return ImageFont.truetype('%s.ttf' % fn.lower(), fs)

    def set_row(self, row, height, label=None, font_name=None, font_size=None):
        padding = 5
        units_to_pixels = 4.0 / 3.0

        if isinstance(label, basestring):

            if font_name is None:
                font_name = self.excel._formats.default_attributes['font_name']

            if font_size is None:
                font_size = self.excel._formats.default_attributes['font_size']

            dimensions = self.truetype.getsize(label)

            if (dimensions[1] * units_to_pixels) - padding > height:
                # text too tall
                return

            if (dimensions[0] * units_to_pixels) > self._size_col(self.start_column):
                # text too long
                return

        super(_Sheet, self).set_row(row, height)

    def _set_freeze_loc(self, columns):
        if list(columns.labels[-1]).count(0) == 1:
            offset = 1
        else:
            offset = 0
        self._freeze_loc = ((self._row + columns.nlevels),
                            (self._column + offset + 1))


class _Box(object):

    __slots__ = ('sheet', 'chain', '_formats', '_single_columns','_column_edges',
                 '_columns', '_italic', '_lazy_excel', '_lazy_index',
                 '_lazy_columns', '_lazy_values', '_lazy_contents',
                 '_lazy_is_weighted', '_lazy_shape', '_lazy_has_tests',
                 '_lazy_arrow_rep', '_lazy_arrow_color', '_lazy_header_left',
                 '_lazy_header_center', '_lazy_header_title', '_lazy_notes')

    def __init__(self, sheet, chain, row, column):
        self.sheet = sheet
        self.chain = chain
        self._formats = None
        self._single_columns = None
        self._column_edges = None
        self._columns = []
        self._italic = []

    @property
    def formats(self):
        return self.excel._formats

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
        contents = cp.loads(cp.dumps(self.chain.contents, cp.HIGHEST_PROTOCOL))

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
        write = self.sheet.write
        merge_range = self.sheet.merge_range
        format_ = self.formats._data_header

        for rel_y, label in enumerate(self.chain.structure.columns):
            column = self.sheet._column + rel_y
            merge_range(self.sheet._row, column,
                        self.sheet._row + 1, column,
                        label, format_)

            width = self.sheet.column_width_specific.get(
                column,
                self.sheet.column_width_frame)
            self.sheet.set_column(column, column, width)
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
            format_ = self.formats[name]
            if data != data:
                data = ''
            write(self.sheet._row + rel_x,
                  self.sheet._column + rel_y,
                  data, format_)
            rel_x, rel_y = flat.coords

        for i in xrange(rel_x):
            self.sheet.set_row(self.sheet._row + i,
                               self.sheet.row_height_label)

    def _write_columns(self):
        contents = dict()
        format_ = self.formats._y
        column = self.sheet._column + 1
        nlevels = self.columns.nlevels
        write = self.sheet.write
        merge_range = self.sheet.merge_range
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
                        level = -(1 + self.has_tests)
                        lowest_label = self.columns.get_level_values(level)[left]
                        if lowest_label == 'Total':
                            self.single_columns.append(left)
                    self.column_edges.append(right + 1)
                if left not in self.single_columns:
                    if group_sizes and not is_values:
                        r = 0
                        while r != right:
                            merge_range(row, column + group_sizes[0][0],
                                        row, column + group_sizes[0][1],
                                        data, format_)
                            _, r = group_sizes.pop(0)
                    elif left == right:
                        write(row, column + left, data, format_)
                    else:
                        merge_range(row, column + left,
                                    row, column + right,
                                    data, format_)
                    if is_values:
                        group_sizes.append((left, right))
                data = next_
                left = right + 1
                if next_ is None:
                    break

            has_tests = self.has_tests
            if not has_tests or (((level_id + 1) != nlevels) and has_tests):
                if (row % 2) == 0:
                    self.sheet.set_row(row, self.sheet.y_header_height)
                else:
                    self.sheet.set_row(row, self.sheet.y_row_height)

        for cindex in self.single_columns:
            level = -(1 + self.has_tests)
            data = self._cell(self.columns.get_level_values(level)[cindex],
                              **contents)
            merge_range(row - nlevels + 1, column + cindex,
                        row, column + cindex,
                        data, format_)

        self.sheet._row = row + 1

    def _write_rows(self):
        write = self.sheet.write
        write_rich_string = self.sheet.write_rich_string

        if self.chain.annotations:
            cat_pos = ['header_left', 'header_center', 'header_title']
            self._write_annotations(cat_pos)

        column = self.sheet._column

        levels = self.index.get_level_values

        if self.chain._is_mask_item:
            write(self.sheet._row, column,
                                   levels(0).unique().values[0],
                                   self.formats['mask_label'])
            self._format_row(self.formats['mask_label'])
        else:
            write(self.sheet._row, column,
                  levels(0).unique().values[0], self.formats['label'])
            self._format_row(self.formats['label'])
        self.sheet._row += 1

        if self.notes:
            self._write_annotations(['notes'])

        if self.sheet.dummy_tests and self.has_tests:
            level_1, values, contents = self._get_dummies(levels(1).values,
                                                          self.values)
        else:
            level_1, values, contents = (levels(1).values,
                                         self.values,
                                         self.contents)

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
                base = self._is('base', **x_contents)
                if rel_y and self.chain.array_style == 0:
                    if base and  not x_contents['is_weighted']:
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
                        if base and not x_contents['is_weighted']:
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
                arrow_format = {"'@L'": arrow_high_format,
                                "'@H'": arrow_low_format}.get(arrow)

                if arrow_format is _None:
                    arrow_format = self._format_x(name, rel_x, rel_y,
                                                  row_max,
                                                  x_contents.get('dummy'),
                                                  use_bg, view_border,
                                                  border_from,
                                                  **{'font_color': arrow_color})
                write_rich_string(self.sheet._row + rel_x,
                                  self.sheet._column + rel_y,
                                  arrow_format, arrow_rep,
                                  format_, ' ' + cell_data, format_)
            else:
                write(self.sheet._row + rel_x,
                      self.sheet._column + rel_y,
                      cell_data, format_)
            nxt_x, nxt_y = flat.coords
            rel_x, rel_y = nxt_x, nxt_y
            if rel_y == 0:
                border_from = name
        for i in xrange(rel_x):
            self.sheet.set_row(self.sheet._row + i,
                               self.sheet.row_height_label,
                               label=level_1[i])
        self.sheet._row += rel_x

    def _write_annotations(self, names):
        for name in names:
            anno = getattr(self, name)
            if anno:
                format_ = self.formats[name]
                label = anno[0]
                self.sheet.write(self.sheet._row, self.sheet._column,
                                 label, format_)
                self._format_row(format_)
                self.sheet.set_row(self.sheet._row,
                                   self.sheet.row_height_label,
                                   label=label,
                                   font_name=format_.get('font_name'),
                                   font_size=format_.get('font_size'))

                self.sheet._row += 1

    def _format_row(self, format_):
        write = self.sheet.write
        row = self.sheet._row
        column = self.sheet._column
        for rel_y in xrange(1, self.values.shape[1] + 1):
            write(row, column + rel_y, '', format_)

    def _alternate_bg(self, name, bg):
        freq_view_group = self.excel.views_groups.get(name, '') == 'freq'
        is_freq_test = any(_ in name
                           for _ in ('counts', 'pct', 'propstest'))
        is_mean = 'mean' in name
        not_net_sum = all(_ not in name for _ in ('net', 'sum'))
        if ((is_freq_test and not_net_sum) or freq_view_group) or \
                (not is_mean and self.chain.array_style == 0):
            return not bg, bg
        return self.sheet.alternate_bg, True

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
        if self.chain.array_style > -1:
            format_name += '_array_style_%s' % self.sheet.sheet_name
        format_ = self.formats[format_name]
        if kwargs:
            format_ = cp.loads(cp.dumps(format_, cp.HIGHEST_PROTOCOL))
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
        append = dummy_idx.append
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
                        append(ndx + len(dummy_idx))
                    if not self._is('base', **self.contents[idx]):
                        group = next_
                    dummy = True
                idx, data = ndx, next_
            except StopIteration:
                if group and dummy:
                    append(ndx + len(dummy_idx) + 1)
                break
        dummy_arr = np.array([[u'' for _ in xrange(len(values[0]))]],
                             dtype=str)
        for idx in dummy_idx:
            try:
                index = np.insert(index, idx, u'')
                values = np.vstack((values[:idx, :],
                                   dummy_arr,
                                   values[idx:, :]))
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

    def _cell(self, value, **contents):
        normalize, vtype, nan_rep = self._cell_args(**contents)
        return _Cell(value, normalize,
                     self.excel.decimals.get(vtype),
                     nan_rep).__repr__()

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
    def _is(name, **contents):
        return any(name in _ for _ in list(filter(contents.get, contents)))

class _Cell(object):

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
            return rsub(r'#pad-\d+', str(), self.data)
        if self.normalize:
            if self.decimals is not None:
                if isinstance(self.data, (float, np.float64)):
                    return round(self.data / 100., self.decimals + 2)
            return self.data / 100.
        if self.decimals is not None:
            if isinstance(self.data, (float, np.float64)):
                return round(self.data, self.decimals)
        return self.data
