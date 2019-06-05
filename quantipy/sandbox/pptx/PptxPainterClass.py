# encoding: utf-8

import re
from lxml import etree
import warnings

# Imports from Python-PPTX
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.util import (
    Emu,
    Pt,
    Cm,
    Inches)

try:
    from pptx import table
except:
    from pptx.shapes import table

from pptx.chart.data import ChartData
from pptx.enum.chart import XL_CHART_TYPE
from pptx.dml.color import RGBColor

from .enumerations import (
    fill_type_dct,
    data_label_pos_dct,
    legend_pos_dct,
    tick_label_pos_dct,
    tick_mark_pos_dct,
    vertical_alignment_pos_dct,
    paragraph_alignment_pos_dct,
    theme_color_index_dct,
    chart_type_dct
)

from .PptxDefaultsClass import PptxDefaults
from .PptxChainClass import float2String
import pandas as pd
import copy


def chartdata_from_dataframe(df, number_format="0%", xl_number_format='0.00%'):
    """
    Return a CategoryChartData instance from the given Pandas DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe instance from which ChartData will be created.
    number_format : str, default="0%"
        The pptx number format for the intended ChartData. See:
        http://python-pptx.readthedocs.io/en/latest/api/chart-data.html?highlight=number_format#pptx.chart.data.CategoryChartData.number_format
    xl_number_format : str, default="0.00%"
        The xlsx number format for the Excel sheet behind the intended Chart. See:

    Returns
    -------
    cd : pptx.chart.data.CategoryChartData
        The ChartData instance created from the given dataframe.
    """

    def get_parent(sub_categories, line, pos):
        """
        Return the sub_category's parent given its lineage position.
        """

        for subcat in sub_categories:
            if subcat.label == line[pos]:
                return subcat

    cd = CategoryChartData(number_format=number_format)

    if isinstance(df.index, pd.MultiIndex):
        cats = []
        for line in df.index.unique().tolist():
            for l, lvl in enumerate(line):
                if l == 0:
                    if not any([lvl == cat.label for cat in cats]):
                        cats.append(cd.add_category(lvl))
                else:
                    parent = get_parent(cats, line, 0)
                    if l > 1:
                        for i in range(1, l):
                            parent = get_parent(parent.sub_categories, line, i)
                    sub_categories = parent.sub_categories
                    seen = [lvl == subcat.label for subcat in sub_categories]
                    if not any(seen):
                        parent.add_sub_category(lvl)
    else:
        categories = tuple(df.index.values.tolist())
        cd.categories = categories

    for col in df.columns:
        values = [
            value if value == value else None
            for value in df[col].values.tolist()
        ]
        series = (col, tuple(values))
        cd.add_series(*series, number_format=xl_number_format)

    return cd


def return_slide_layout_by_name(pptx, slide_layout_name):
    """
    Loop over the slide layout object and find slide layout by name, return slide layout
    object.

    example: myslide = get_slide_layout_by_name(prs, 'Inhaltsverzeichnis')
             slide = prs.slides.add_slide(myslide)
    """

    for slide_layout in pptx.slide_layouts:
        if slide_layout.name == slide_layout_name:
            return slide_layout
    else:
        raise Exception(
            'Slide layout: {sld_layout} not found\n'.format(
                sld_layout=slide_layout_name))


def convertable(obj, func):
    """
    Returns True if obj can be converted by func without an error.
    """

    try:
        func(obj)
        return True
    except ValueError:
        return False


class PptxPainter(object):
    """
    A convenience wrapper around the python-pptx library

    Makes a Presentation instance and also defines a default slide layout if specified.

    Parameters
    ----------
    path_to_presentation: str
        Full path to PowerPoint template
    slide_layout: int
        A PowerPoint slide layout.
        To see available Slide Layouts in a PPTX, select the Viev menu and click Slide Master.
    shape_properties: quantipy.sandbox.pptx.PptxDefaultsClass.PptxDefaults
        An instance of PptxDefaults

    """

    def __init__(self, path_to_presentation, slide_layout=None, shape_properties=None):

        self.presentation  = Presentation(path_to_presentation) # TODO PptxPainter - Path checking # type: Presentation
        if slide_layout is None:
            self.default_slide_layout = None
        else:
            self.default_slide_layout = self.set_slide_layout(slide_layout)

        # Add all the dafault dicts to the class -
        if shape_properties:
            self._shape_properties = shape_properties
        else:
            self._shape_properties = PptxDefaults()

        self.textbox = self._shape_properties.textbox
        self.textbox_header = self._shape_properties.textbox_header
        self.textbox_footer = self._shape_properties.textbox_footer
        self.chart = self._shape_properties.chart
        self.table = self._shape_properties.table
        self.side_table = self._shape_properties.side_table

        charts = self._shape_properties.charts
        self.chart_bar = charts['bar']
        self.chart_bar_stacked100 = charts['bar_stacked100']
        self.chart_line = charts['line']
        self.chart_column = charts['column']
        self.chart_pie = charts['pie']

        self.slide_kwargs = {
                    'textboxs': {},
                    'charts': {},
                    'tables': {},
                    'side_tables': {},
                    }

    @staticmethod
    def get_plot_values(plot):
        """
        Return a list of dicts with serie name as dict-key and serie values as dict-value

        Parameters
        ----------
        plot: pptx.chart.plot._BasePlot

        Returns
        -------
        list

        """
        series = [
            {series.name: [str(s) for s in series.values]}
            for series in plot.series
        ]

        return series

    def show_data_labels(self, plot, decimals=0):
        """
        Explicitly sets datalabels to allow for datalabel editing.

        Parameters
        ----------
        plot: pptx.chart.plot._BasePlot
            The plot object for which datalabels need should be shown.
        decimals: the number of decimals to show

        Returns
        -------
        None
        """

        # Get number format and font from data labels
        data_labels = plot.data_labels
        number_format = data_labels.number_format  # '0%'
        font = data_labels.font

        plot_values = self.get_plot_values(plot)
        for s, series in enumerate(plot_values):
            values = [
                value
                for value in list(series.values())[0]
                if convertable(value, float)
            ]
            for v, value in enumerate(values):
                if value is not None:
                    if number_format == '0%':
                        value = round(float(value) * 100, decimals)

                        str_value = float2String(value) + '%'
                    else:
                        str_value = str(value)
                else:
                    str_value = ""
                point = plot.series[s].points[v]
                data_label = point.data_label
                frame = data_label.text_frame
                frame.text = str_value
                pgraph = frame.paragraphs[0]
                for run in pgraph.runs:
                    run.font.bold = font.bold
                    # run.font.color.rgb = font.color.rgb
                    # run.font.fill.fore_color.rgb = font.fill.fore_color.rgb

                    run.font.italic = font.italic
                    run.font.name = font.name
                    run.font.size = font.size
                    run.font.underline = font.underline

    def edit_datalabel(self, plot, series, point, text, prepend=False, append=False, rgb=None):
        """
        Add/append data label text.

        Parameters
        ----------
        plot: pptx.chart.plot._BasePlot
            An instance of a Chart object.
        serie: int
            The serie where the data label should be edited
            chart.series[serie]
        point: int
            The point where the data label should be edited
            chart.series[serie].points[point]
        text: basestring
            The text to add/append to data label
        prepend: bool
            Set to True to prepend text to existing data label
        append: bool
            Set to True to append text to existing data label
        rgb: tuple
            Tuple with three ints defining each RGB color

        Returns
        -------
        None

        """
        data_label = plot.series[series].points[point].data_label
        frame = data_label.text_frame

        run = frame.paragraphs[0].runs[0]
        original_text = frame.text
        if prepend:
            run.text = '{}{}'.format(text, original_text)
        elif append:
            run.text = '{}{}'.format(original_text, text)
        else:
            run.text = text
        if rgb is not None:
            run.font.color.rgb = RGBColor(*rgb)

    def queue_slide_items(self, pptx_chain, slide_items,
                          decimal_separator='.',
                          pct_decimals=0,
                          decimals=2,
                          ):
        """
        Helper function to queue a full automated slide.
        Includes queueing of header with question text, a table or chart with optional side table,
        and footer with base description.

        Parameters
        ----------
        pptx_chain: quantipy.sandbox.pptx.PptxChainClass.PptxChain
            An instance of a PptxChain
        slide_items: basestring
            A string of slide items with cell types in the form 'slide_item:cell_types'.
            Available slide items are:
                'chart'
                'sidetable'
                'table'
            Every slide item needs a comma separated list of cell types to include in the chart or table.
            Available cell types are: 'pct, net, mean, stats, tests'

            A slide_items_string could look like: 'chart:pct,net'
            Separate multiple slide items with '+' eg. : 'chart:pct+side_table:net'
            One slide item type can only appear once in a slide_items_string.

        Returns
        -------
        None
            calls:
            self.draft_textbox_header()
            self.draft_textbox_footer()
            self.queue_textbox()
            self.draft_table()
            self.queue_table()
            self.draft_side_table()
            self.queue_side_table()
            self.draft_autochart()
            self.queue_chart()
            self._check_shapes()

        """

        valid_slide_items = ['chart','table','side_table']
        slide_items = re.sub(' +', '', slide_items)
        # TODO check for valid slide_items input

        # Question text
        draft = self.draft_textbox_header(pptx_chain.question_text)
        self.queue_textbox(settings=draft)

        # Base description
        draft = self.draft_textbox_footer(pptx_chain.base_text)
        self.queue_textbox(settings=draft)

        slide_items = slide_items.split('+')

        for slide_item in slide_items:
            if slide_item.startswith('table'):
                cell_items = slide_item.split(':')[1]
                pptx_frame = pptx_chain.chart_df.get(cell_items).to_table(pct_decimals=pct_decimals,
                                                                          decimals=decimals,
                                                                          decimal_separator=decimal_separator,
                                                                          )
                if not pptx_frame().empty:
                    table_draft = self.draft_table(pptx_frame())
                    self.queue_table(settings=table_draft)
            if slide_item.startswith('side_table'):
                cell_items = slide_item.split(':')[1]
                pptx_frame = pptx_chain.chart_df.get(cell_items).to_table(pct_decimals=pct_decimals,
                                                                          decimals=decimals,
                                                                          decimal_separator=decimal_separator,
                                                                          )
                if not pptx_frame().empty:
                    side_table_draft = self.draft_side_table(pptx_frame())
                    pct_index = [index for index, value in enumerate(pptx_frame.cell_items) if 'is_c_pct' in value]
                    if pct_index:
                        side_table_draft['values_suffix'] = '%'
                        side_table_draft['values_suffix_columns'] = pct_index
                    self.queue_side_table(settings=side_table_draft)
            if slide_item.startswith('chart'):
                sig_test = False
                cell_items = slide_item.split(':')[1]

                ''' 
                Makes no sense to actually have 'test' as a cell_item.
                Will remove it from cell_items and set flag sig_test as True
                '''
                cell_items = cell_items.split(',')
                if 'test' in cell_items:
                    sig_test = True
                    pptx_chain.add_test_letter_to_column_labels()
                    pptx_chain.chart_df = pptx_chain.prepare_dataframe()
                    cell_items.remove('test')
                cell_items = ','.join(cell_items)

                pptx_frame = pptx_chain.chart_df.get(cell_items)
                if not pptx_frame().empty:
                    chart_draft = self.draft_autochart(pptx_frame(), pptx_chain.chart_type)
                    if sig_test:
                        chart_draft['sig_test_visible'] = True
                        chart_draft['sig_test_results'] = pptx_chain.sig_test

                    self.queue_chart(settings=chart_draft)

        self._check_shapes()

        return None

    def _check_shapes(self, adjust='chart'):
        """
        Purpose is to check and adjust all queued items for any collisions:
        Currently checks if charts and side_tables colide and then adjust the chart.

        Parameters
        ----------
        adjust: str
            A not implemented future option to select what to adjust.

        Returns
        -------
        None
            edits self.slide_kwargs

        """

        # Find the side_table with the lowest 'left' number
        table_max_left=12240000
        table_width=0
        for table, table_settings in self.slide_kwargs['side_tables'].items():
            if table_settings['left'] < table_max_left:
                table_max_left = table_settings['left']
                table_width = table_settings['width']

        # If any charts overlay a side_table then adjust width of chart
        for chart, chart_settings in self.slide_kwargs['charts'].items():
            if chart_settings['left'] + chart_settings['width'] > table_max_left:
                chart_settings['width'] -= table_width

    def clear_tables(self):
        """
        Initilalize the slide_kwargs "tables" dict

        Returns
        -------
        None
            Removes all keys from self.slide_kwargs['tables']
        """
        self.clear_queue('tables')

    def clear_side_tables(self):
        """
        Initilalize the slide_kwargs "side_tables" dict

        Returns
        -------
        None
            Removes all keys from self.slide_kwargs['side_tables']
        """
        self.clear_queue('side_tables')

    def clear_charts(self):
        """
        Initilalize the slide_kwargs "charts" dict

        Returns
        -------
        None
            Removes all keys from self.slide_kwargs['charts']
        """
        self.clear_queue('charts')

    def clear_textboxes(self):
        """
        Initilalize the slide_kwargs "text_boxes" dict

        Returns
        -------
        None
            Removes all keys from self.slide_kwargs['text_boxes']
        """
        self.clear_queue('textboxs')

    def clear_queue(self, key):
        """
        Initialize the requested shape dict in slide_kwargs

        Parameters
        ----------
        key: str
            'all', 'charts', textboxes','tables', 'side_tables'

        Returns
        -------
        None
            Removes requested keys from self.slide_kwargs

        """
        if key=='all':
            for item in list(self.slide_kwargs.keys()):
                self.slide_kwargs[item].clear()
        elif key=='charts':
            self.slide_kwargs['charts'].clear()
        elif key=='textboxes':
            self.slide_kwargs['textboxs'].clear()
        elif key=='tables':
            self.slide_kwargs['tables'].clear()
        elif key=='side_tables':
            self.slide_kwargs['side_tables'].clear()

    def set_slide_layout(self, slide_layout):
        """
        Method to set a Slide Layout.

        Parameters
        ----------
        slide_layout: int or str
            To see available Slide Layouts in a PPTX, select the Viev menu and click Slide Master.

        Returns
        -------
        pptx.slide.SlideLayout
            Instance of SlideLayout set to the specified slide layout

        """
        if isinstance(slide_layout, int):
            return self.presentation.slide_layouts[slide_layout]
        else:
            return return_slide_layout_by_name(self.presentation, slide_layout)

    def add_slide(self, slide_layout=None):
        """
        Method that creates a Slide instance.

        Parameters
        ----------
        slide_layout: int or str
            To see available Slide Layouts in a PPTX, select the Viev menu and click Slide Master.

        Returns
        -------

        """
        if slide_layout is None:
            if self.default_slide_layout is None:
                raise ValueError('No slide layout found! Specify a slide layout or set a default slide layout')
            else:
                slide_layout = self.default_slide_layout
        else:
            slide_layout = self.set_slide_layout(slide_layout=slide_layout)

        return self.presentation.slides.add_slide(slide_layout)

    def draft_textbox(self, settings=None, text=None):
        """
        Method for drafting a textboc

        Parameters
        ----------
        settings: dict
            A dict of textbox settings, see dict default_textbox in pptx_defaults.py
        text: basestring
            Text to show in textbox
        Returns: self.textbox
        -------
        """
        if settings:
            draft = copy.deepcopy(settings)
        else:
            draft = copy.deepcopy(self.textbox)
        draft['text'] = text
        return draft

    def draft_textbox_header(self, text=None):
        """
        Simplified method for drafting a header textbox that wont require the settings dict,
        but will instead pick the default textbox_header setting

        Parameters
        ----------
        text: basestring
            Text to show in textbox
        Returns: dict
            Returns settings for a textbox, which can be used for method queue_textbox
        -------
        """

        draft = copy.deepcopy(self.textbox_header)
        draft['text'] = text

        return draft

    def draft_textbox_footer(self, text=None):
        """
        Simplified method for drafting a footer textbox that wont require the settings dict,
        but will instead pick the default textbox_footer setting

        Parameters
        ----------
        text: basestring
            Text to show in textbox
        Returns: dict
            Returns settings for a textbox, which can be used for method queue_textbox
        -------
        """

        draft = copy.deepcopy(self.textbox_footer)
        draft['text'] = text

        return draft

    def draft_chart(self, dataframe, settings=None):
        """
        Sets attribute self.chart

        Parameters
        ----------
        settings: dict
            A dict of chart settings, see dict default_chart in pptx_defaults.py
        dataframe: pandas.core.frame.DataFrame
            the pandas.dataframe to chart
        Returns: self.chart
        -------
        """

        if settings:
            draft = copy.deepcopy(settings)
        else:
            draft = copy.deepcopy(self.chart)
        draft['dataframe'] = dataframe
        return draft

    def draft_autochart(self, dataframe, chart_type):
        """
        Simplified caller for method draft_chart that wont require the settings dict,
        but will instead pick the default chart setting for the chart type requested

        Parameters
        ----------
        dataframe: pandas.core.frame.DataFrame
        chart_type: str
            A string corresponding to the keys in Dict "chart_type_dct" from "enumerations.py"
        Returns: self.chart
        -------
        """
        valid_chart_types = ['pie',
                             'bar_clustered',
                             'bar_stacked_100',
                             'bar',
                             'column',
                             'column_clustered',
                             'line',
                             ]
        # Validate the user-provided chart types.
        if not isinstance(chart_type, str):
            raise ValueError('The chart_type argument must be a string')
        if chart_type not in valid_chart_types:
                error_msg ='Invalid chart_type {}. Valid chart types are {}'
                raise ValueError(error_msg.format(chart_type, valid_chart_types))

        # Make draft
        if chart_type == 'pie':
            draft = copy.deepcopy(self.chart_pie)
        elif chart_type == 'bar_clustered' or chart_type == 'bar':
            draft = copy.deepcopy(self.chart_bar)
            if len(dataframe.columns) > 1:
                draft['has_legend'] = True
        elif chart_type == 'column_clustered' or chart_type == 'column':
            draft = copy.deepcopy(self.chart_column)
            if len(dataframe.columns) > 1:
                draft['has_legend'] = True
        elif chart_type == 'line':
            draft = copy.deepcopy(self.chart_line)
            if len(dataframe.columns) > 1:
                draft['has_legend'] = True
        elif chart_type == 'bar_stacked_100':
            draft = copy.deepcopy(self.chart_bar_stacked100)
        else:
            draft = copy.deepcopy(self.chart_bar)

        draft['dataframe'] = dataframe
        return draft

    def draft_table(self, dataframe, text=None, settings=None):
        """
        Sets attribute self.tables

        Parameters
        ----------
        settings: dict
            A dict of chart settings, see dict default_chart in pptx_defaults.py
        dataframe: pandas.core.frame.DataFrame
            the pandas.dataframe to make a table of
        Returns: self.tables
        -------
        """

        if settings:
            draft = copy.deepcopy(settings)
        else:
            draft = copy.deepcopy(self.table)
        draft['dataframe'] = dataframe
        draft['text'] = text
        return draft

    def draft_side_table(self, dataframe):
        """
        Draft a table

        Parameters
        ----------
        dataframe: pandas.core.frame.DataFrame
            the pandas.dataframe to make a table of
        Returns: self.tables
        -------
        """

        draft = copy.deepcopy(self.side_table)
        draft['dataframe'] = dataframe

        # set table width based on number of columns in dataframe
        cols = len(draft['dataframe'].columns)
        col_width = draft['values_column_width']
        table_width = cols * col_width

        # position table
        rows = len(draft['dataframe'].index)
        draft['left'] = draft['left'] + draft['width'] - table_width
        draft['width'] = table_width
        height = draft['height']
        top = draft['top']
        top_member_row_height = draft['top_member_row_height']

        if rows == 1:
            top = top + int(round(height * (0.3 / rows)) - top_member_row_height)
            draft['values_cell_kwargs']['margin_top'] = Cm(1.6)
        else:
            top = top + int(round(height * (0.45 / rows)) - top_member_row_height)
            draft['values_cell_kwargs']['margin_top'] = Cm(0.1)

        draft['top'] = top

        plot_area = int(round(height * 0.86))
        value_row_height = int(round(plot_area / rows))
        height = top_member_row_height + rows * value_row_height

        draft['height'] = height
        draft['values_row_height'] = value_row_height

        return draft

    def queue_chart(self, settings, name=None):
        """
        Will add a chart to the Slide properties Dict
        :param
            settings: A dictionary of chart settings, default is self.chart
            name: Optionally give the chart a name. If none the chart will be named 'chart[n]'
        :return:
            None, Adds a key to self.slide_kwargs['charts']
        """

        self._add('chart', settings, name=name)

    def queue_textbox(self, settings, name=None):
        """
        Will add a textbox to the Slide properties Dict
        :param
            settings:  A dictionary of textbox settings, deafult is self.textbox
            name: Optionally give the textbox a name. If none the textbox will be named 'textbox[n]'
        :return:
            None, adds a key to self.slide_kwargs['textboxes']
        """

        self._add('textbox', settings, name=name)

    def queue_table(self, settings, name=None):
        """
        Will add a table to the Slide properties Dict
        :param
            settings: A dictionary of table settings, default is self.table
            name: Optionally give the table a name. If none the table will be named 'table[n]'
        :return:
            None, Adds a key to self.slide_kwargs['tables']
        """

        self._add('table', settings, name=name)

    def queue_side_table(self, settings, name=None):
        """
        Will add a table to the Slide properties Dict
        :param
            settings: A dictionary of table settings, default is self.table
            name: Optionally give the table a name. If none the table will be named 'table[n]'
        :return:
            None, Adds a key to self.slide_kwargs['tables']
        """

        self._add('side_table', settings, name=name)

    def add_slide_from_queue(self, slide_layout=None):
        """
        Method that creates a Slide instance and inserts
        textboxes and charts as queued in self.slide_kwargs
        :param
            slide_layout: Int
                The Slide Layout to use.
                To see available Slide Layouts in a PPTX, select the Viev menu and click Slide Master.
                If no Slide layout is specified then self.default_slide_layout will be used
        :return:
        """
        slide= self.add_slide(slide_layout=slide_layout)
        kwargs = self.slide_kwargs
        for _type, draft in kwargs.items():
            # Add text boxes
            if _type == 'textboxs':
                for name, settings in draft.items():
                    textbox=self.add_textbox(slide, **settings)
            # Add charts
            if _type == 'charts':
                for name, settings in draft.items():
                    chart=self.add_chart(slide, **settings)
            # Add tables
            if _type == 'tables':
                for name, settings in draft.items():
                    table=self.add_table(slide, **settings)
            # Add side tables
            if _type == 'side_tables':
                for name, settings in draft.items():
                    side_table=self.add_table(slide, **settings)
        return slide

    def _add(self, shape, settings, name=None):
        """
        Internal for adding a new shape (textbox or chart) to self.slide_kwargs
        :param
            shape: String ('chart', 'textbox', 'table')
            settings: A dictionary of settings
            name:
        :return:
        """

        shapes='{}s'.format(shape)

        if name is None:
            name='{}{}'.format(shape, len(self.slide_kwargs[shapes]) + 1)

        self.slide_kwargs[shapes][name] = settings.copy()

    @staticmethod
    def set_cell_properties(cell,  # type: table._Cell
                            margin_left=Cm(0.1),
                            margin_right=Cm(0.1),
                            margin_top=Cm(0.1),
                            margin_bottom=Cm(0.1),
                            vertical_alignment='top',
                            shading=False,
                            shading_color=(0,0,0)):

        cell.margin_left = margin_left
        cell.margin_right = margin_right
        cell.margin_top = margin_top
        cell.margin_bottom = margin_bottom
        cell.vertical_anchor = vertical_alignment_pos_dct[vertical_alignment]
        fill = cell.fill
        if shading:
            fill.solid()
            fill.fore_color.rgb = RGBColor(*shading_color)
        else:
            fill.background()

    @staticmethod
    def add_table(slide,
                  dataframe=None,
                  text=None,
                  left=Cm(4), top=Cm(8), width=Cm(5), height=Cm(8),

                  show_side_member=True,
                  side_member_column_width=Cm(1),
                  side_member_textframe_kwargs=None, # type: dict
                  side_member_cell_kwargs=None, # type: dict

                  show_top_member=True,
                  top_member_row_height=Cm(1),
                  top_member_textframe_kwargs=None, # type: dict
                  top_member_cell_kwargs=None,  # type: dict

                  values_row_height=Cm(1),
                  values_column_width=Cm(2),
                  values_textframe_kwargs=None, # type: dict
                  values_cell_kwargs=None,  # type: dict
                  values_prefix=None,
                  values_prefix_columns='all', # or a list of column indexes
                  values_suffix=None,
                  values_suffix_columns = 'all',  # or a list of column indexes

                  top_left_corner_textframe_kwargs=None, # type: dict
                  top_left_corner_cell_kwargs=None,  # type: dict
                  ):
        # -------------------------------------------------------------------------

        rows = len(dataframe.index)
        if show_top_member: rows +=1
        cols = len(dataframe.columns)
        if show_side_member: cols +=1

        shapes = slide.shapes
        table = shapes.add_table(rows, cols, left, top, width, height).table

        # row specific properties
        if show_side_member:
            table.columns[0].width = side_member_column_width
            row_labels = list(dataframe.index)
            for idx, row_label in enumerate(row_labels):
                row_idx = idx + 1 if show_top_member else idx
                cell = table.cell(row_idx, 0)
                PptxPainter.set_cell_properties(cell,**side_member_cell_kwargs)
                PptxPainter.add_textframe(cell, row_label, **side_member_textframe_kwargs)

        # col specific properties
        if show_top_member:
            table.rows[0].height = top_member_row_height
            col_labels = list(dataframe.columns)
            for idx, col_label in enumerate(col_labels):
                col_idx = idx + 1 if show_side_member else idx
                cell = table.cell(0, col_idx)
                PptxPainter.set_cell_properties(cell,**top_member_cell_kwargs)
                PptxPainter.add_textframe(cell, col_label, **top_member_textframe_kwargs)

        # add values
        df_cols_range = list(range(len(dataframe.columns)))
        if values_prefix_columns =='all': values_prefix_columns = df_cols_range
        if values_suffix_columns == 'all': values_suffix_columns = df_cols_range
        table_values = dataframe.values
        datatypes = dataframe.dtypes.tolist()
        for i, val in enumerate(table_values):
            row_idx = i +1 if show_top_member else i
            table.rows[row_idx].height = values_row_height
            for x, subval in enumerate(val):
                col_idx = x + 1 if show_side_member else x
                table.columns[col_idx].width = values_column_width
                cell = table.cell(row_idx, col_idx)
                PptxPainter.set_cell_properties(cell, **values_cell_kwargs)
                if subval != '':
                    prefix = None
                    if x in values_prefix_columns: prefix = values_prefix
                    suffix = None
                    if x in values_suffix_columns: suffix = values_suffix
                    if datatypes[x].kind == 'i':
                        subval = int(subval)
                    subval = (prefix or '') + str(subval) + (suffix or '')
                PptxPainter.add_textframe(cell, subval, **values_textframe_kwargs)

        # for i in range(len(dataframe.columns)):
        #     col_idx = i + 1 if show_side_member else i
        #     table.columns[col_idx].width = values_column_width
        #     for x in range(len(dataframe.index)):
        #         row_idx = x + 1 if show_top_member else x
        #         table.rows[row_idx].height = values_row_height
        #         cell = table.cell(row_idx, col_idx)
        #         PptxPainter.set_cell_properties(cell, **values_cell_kwargs)
        #         prefix = None
        #         if i in values_prefix_columns: prefix = values_prefix
        #         suffix = None
        #         if i in values_suffix_columns: suffix = values_suffix
        #         value = dataframe.iloc[x,i]
        #         subval = (prefix or '') + str(value) + (suffix or '')
        #         PptxPainter.add_textframe(cell, subval, **values_textframe_kwargs)

        # add question label
        if show_side_member and show_top_member:
            cell = table.cell(0, 0)
            PptxPainter.set_cell_properties(cell,**top_left_corner_cell_kwargs)
            PptxPainter.add_textframe(cell, text, **top_left_corner_textframe_kwargs)

        return table

    def add_chart(self, slide,
                  dataframe=None,
                  chart_type='bar_stacked_100',
                  left=Cm(2.31), top=Cm(5.93), width=Cm(29.23), height=Cm(10.5),
                  chart_style=2,

                  # Title
                  has_chart_title=False,
                  titletext=None,
                  textframe_kwargs=None,  # type: dict

                  # Legend properties
                  has_legend=True,
                  legend_position='right',
                  legend_in_layout=False,  # will we ever set this True?
                  legend_horz_offset=0.1583,
                  legend_font_kwargs=None,  # type: dict

                  # Category axis properties
                  caxis_visible=True,
                  caxis_tick_label_position='next_to_axis',
                  caxis_tick_labels_offset=730,
                  caxis_has_major_gridlines=False,
                  caxis_has_minor_gridlines=False,
                  caxis_major_tick_mark='outside',
                  caxis_minor_tick_mark='none',
                  caxis_font_kwargs=None,  # type: dict

                  # Value axis properties
                  vaxis_visible=True,
                  vaxis_tick_label_position='low',
                  vaxis_has_major_gridlines=True,
                  vaxis_has_minor_gridlines=False,
                  vaxis_major_tick_mark='outside',
                  vaxis_minor_tick_mark='none',
                  vaxis_max_scale=1.0,
                  vaxis_min_scale=0.0,
                  vaxis_major_unit=0.1,
                  vaxis_minor_unit=None,
                  vaxis_tick_labels_num_format='0%',
                  vaxis_tick_labels_num_format_is_linked=False,
                  vaxis_font_kwargs=None,  # type: dict

                  # Fix yaxis (False, 'center', ?). Currently only an option for bar chart
                  fix_yaxis = False,

                  # Datalabel properties
                  plot_has_data_labels=True,
                  data_labels_position='center',
                  data_labels_num_format='0%',
                  data_labels_num_format_is_linked=False,
                  data_labels_font_kwargs=None,  # type: dict

                  # Plot properties
                  plot_vary_by_cat=False,
                  plot_gap_width=150,
                  plot_overlap=100,
                  smooth_line=False,

                  # Number format
                  number_format='0.00%',
                  xl_number_format='0.00%',

                  # Sig test
                  sig_test_visible = False,
                  sig_test_results = None,
                  ):
        """
        Adds a chart to the given slide and sets all properties for the chart
        :param
            slide:
            dataframe:
            chart_type:
            left:
            top:
            width:
            height:
            chart_style:
            has_chart_title:
            titletext:
            textframe_kwargs:
            has_legend:
            legend_position:
            legend_in_layout:
            legend_horz_offset:
            legend_font_kwargs:
            caxis_visible:
            caxis_tick_label_position:
            caxis_tick_labels_offset:
            caxis_has_major_gridlines:
            caxis_has_minor_gridlines:
            caxis_major_tick_mark:
            caxis_minor_tick_mark:
            caxis_font_kwargs:
            vaxis_visible:
            vaxis_tick_label_position:
            vaxis_has_major_gridlines:
            vaxis_has_minor_gridlines:
            vaxis_major_tick_mark:
            vaxis_minor_tick_mark:
            vaxis_max_scale:
            vaxis_min_scale:
            vaxis_major_unit:
            vaxis_minor_unit:
            vaxis_tick_labels_num_format:
            vaxis_tick_labels_num_format_is_linked:
            vaxis_font_kwargs:
            plot_has_data_labels:
            data_labels_position:
            data_labels_num_format:
            data_labels_num_format_is_linked:
            data_labels_font_kwargs:
            plot_vary_by_cat:
            plot_gap_width:
            plot_overlap:
            smooth_line:
            number_format:
            xl_number_format:
        :return:
        """

        # Switch rows and columns if bar chart
        if chart_type == "bar_clustered":
            dataframe = dataframe[::-1]
            dataframe = dataframe[dataframe.columns[::-1]]
        # Switch rows if bar stacked
        if chart_type == "bar_stacked_100":
            dataframe = dataframe[::-1]

        # =============================== chart data from pandas dataframe
        chart_data = chartdata_from_dataframe(dataframe, number_format=number_format, xl_number_format=number_format)

        # =============================== Create chart
        # For all chart types
        x, y, cx, cy = left, top, width, height
        graphic_frame = slide.shapes.add_chart(
            chart_type_dct[chart_type], x, y, cx, cy, chart_data)

        chart = graphic_frame.chart
        chart.chart_style = chart_style

        # =============================== Fixed y axis
        # TODO Will affect legend, will have to adjust legend manually (mainly a grid summary issue)
        if fix_yaxis:
            if has_legend:
                self.fix_yaxis(chart, 'center', legend=legend_position)
            else:
                self.fix_yaxis(chart, 'center')

        # =============================== Chart title
        # Hmm, added but not sure if ever needed
        # For all chart types
        if has_chart_title:
            charttitle = chart.chart_title
            self.add_textframe(charttitle, titletext, **textframe_kwargs)

        # =============================== legend properties
        # For all chart types
        chart.has_legend = has_legend
        if has_legend:
            legend = chart.legend
            legend.include_in_layout = legend_in_layout
            legend.position = legend_pos_dct[legend_position]
            legend.horz_offset = legend_horz_offset
            self.set_font(legend.font, **legend_font_kwargs)

        # ================================ category axis (horizontal axis) properties
        # Not relevant for Pie
        try:
            category_axis = chart.category_axis
        except ValueError:
            pass
        else:
            category_axis.has_major_gridlines = caxis_has_major_gridlines
            category_axis.has_minor_gridlines = caxis_has_minor_gridlines
            category_axis.major_tick_mark = tick_mark_pos_dct[caxis_major_tick_mark]
            category_axis.minor_tick_mark = tick_mark_pos_dct[caxis_minor_tick_mark]
            category_axis.tick_label_position = tick_label_pos_dct[caxis_tick_label_position]

            category_axis.visible = caxis_visible
            if caxis_visible:
                caxis_tick_labels = category_axis.tick_labels
                caxis_tick_labels.offset = caxis_tick_labels_offset
                self.set_font(caxis_tick_labels.font, **caxis_font_kwargs)

        # ================================= value axis (vertical axis) properties
        # Not relevant for Pie
        try:
            value_axis = chart.value_axis
        except ValueError:
            pass
        else:
            value_axis.has_major_gridlines = vaxis_has_major_gridlines
            value_axis.has_minor_gridlines = vaxis_has_minor_gridlines
            value_axis.minor_tick_mark = tick_mark_pos_dct[vaxis_minor_tick_mark]
            value_axis.major_tick_mark = tick_mark_pos_dct[vaxis_major_tick_mark]
            value_axis.maximum_scale = vaxis_max_scale
            value_axis.minimum_scale = vaxis_min_scale
            value_axis.major_unit = vaxis_major_unit
            value_axis.minor_unit = vaxis_minor_unit
            value_axis.tick_label_position = tick_label_pos_dct[vaxis_tick_label_position]

            value_axis.visible = vaxis_visible
            if vaxis_visible:
                vaxis_tick_labels = value_axis.tick_labels
                self.set_font(vaxis_tick_labels.font, **vaxis_font_kwargs)

                if vaxis_tick_labels_num_format is not None:
                    vaxis_tick_labels.number_format = vaxis_tick_labels_num_format
                vaxis_tick_labels.number_format_is_linked = vaxis_tick_labels_num_format_is_linked

        # ================================= set plot area properties
        # Not relevant for Pie
        plot = chart.plots[0]
        plot.vary_by_categories = plot_vary_by_cat
        plot.has_data_labels = plot_has_data_labels
        plot.gap_width = plot_gap_width
        plot.overlap = plot_overlap

        # ================================= data labels
        # For all chart types
        if plot_has_data_labels:
            data_labels = plot.data_labels
            data_labels.position = data_label_pos_dct[data_labels_position]
            self.set_font(data_labels.font, **data_labels_font_kwargs)

            if data_labels_num_format is not None:
                data_labels.number_format = data_labels_num_format
            data_labels.number_format_is_linked = data_labels_num_format_is_linked

            if not sig_test_results: sig_test_visible = False
            if len(dataframe.columns) == 1: sig_test_visible = False
            if sig_test_visible:
                self.show_data_labels(plot, decimals=0)
                for serie, column in enumerate(sig_test_results[::-1]):
                    for point, test_result in enumerate(column[::-1]):
                        if not isinstance(test_result, str): continue
                        for text in ['*.',
                                     '*',
                                     '**.',
                                     '**',
                                     '\'@L\'.',
                                     '\'@L\'',
                                     '\'@H\'.',
                                     '\'@H\'',
                                     ]:
                            test_result = test_result.replace(text,'')
                        if test_result == '': continue
                        text =  ' ({})'.format(test_result)
                        self.edit_datalabel(plot, serie, point, text, prepend=False, append=True)

        # # ================================ series
        # for i, ser in enumerate(dataframe.columns):
        #     ser = plot.series[i]
        #     try:
        #         ser.smooth = smooth_line
        #     except:
        #         pass

        return chart

    def add_text(self, text):
        """
        Adds text to self.textbox['text']
        :param
            text: Text to show in self.textbox
        :return: None, sets self.textbox
        """
        self.textbox['text'] = text

    @staticmethod
    def add_textbox(slide,
                    text=None,
                    left=Cm(2.33), top=Cm(3.35), width=Cm(29.21), height=Cm(1.78),
                    rotation=0,
                    # Are next three below needed?
                    textbox_fill_solid=False,
                    textbox_color=(100, 0, 0),
                    textbox_color_brightness=0,
                    textframe_kwargs=None, # type: dict
                    ):
        """
        Adds a text box to the given slide and sets all properties for the text box
        :param slide:
        :param text:
        :param left:
        :param top:
        :param width:
        :param height:
        :param rotation:
        :param textbox_fill_solid:
        :param textbox_color:
        :param textbox_color_brightness:
        :param textframe_kwargs:
        :return:
        """

        # ============================== Text Box
        # Add and Position text box
        textbox = slide.shapes.add_textbox(
            left, top, width, height)

        # Clockwise rotation of text box
        textbox.rotation = rotation

        # Adds Solid fill to text box
        if textbox_fill_solid:
            txfill = textbox.fill
            txfill.solid()
            txfill.fore_color.rgb = RGBColor(*textbox_color)
            txfill.fore_color.brightness = textbox_color_brightness

        PptxPainter.add_textframe(textbox, text, **textframe_kwargs)

        return textbox

    @staticmethod
    def add_textframe(textbox,
                      text=None,
                      fit_text=True,
                      margin_left=Cm(0.25),
                      margin_right=Cm(0.25),
                      margin_top=Cm(0.13),
                      margin_bottom=Cm(0.13),
                      vertical_alignment='middle',
                      horizontal_alignment='left',
                      font_kwargs=None # type: dict
                      ):
        """
        Adds a textframe to the given textbox and sets all properties for the text frame

        :param textbox:
        :param text:
        :param fit_text:
        :param margin_left:
        :param margin_right:
        :param margin_top:
        :param margin_bottom:
        :param vertical_alignment:
        :param horizontal_alignment:
        :return: textframe
        """

        textframe = textbox.text_frame

        # Text to show
        if text is not None:
            textframe.text = text

        # Vertical alignment of text in text frame
        textframe.vertical_anchor = vertical_alignment_pos_dct[vertical_alignment]

        # Text margin in text frame
        textframe.margin_left = margin_left
        textframe.margin_bottom = margin_bottom
        textframe.margin_right = margin_right
        textframe.margin_top = margin_top

        # If textbox has no attribute width we will not be able to use textframe.fit_text
        try:
            textbox.width
        except:
            fit_text = False

        # ============================== Paragraph
        # Access the only paragraph in the text frame
        paragraph = textframe.paragraphs[0]

        # Vertical alignment of paragraph
        paragraph.alignment = paragraph_alignment_pos_dct[horizontal_alignment]

        font = paragraph.font

        # If no text we will not be able to do fit_text
        if text is None or text == '':
            fit_text = False

        PptxPainter.set_font(font,
                             textframe=textframe,
                             fit_text=fit_text,
                             **font_kwargs)

        return textframe

    @staticmethod
    def set_font(font_obj,
                 textframe=None,
                 fit_text=False,
                 font_name='Trebuchet MS',
                 font_size=12,
                 font_bold=False,
                 font_italic=False,
                 font_underline=False,
                 font_color=(0, 0, 0),
                 font_color_brightness=0,
                 font_color_theme=None):

        '''
        Does all font settings on the Font object
        :param
        font_obj: Object, The Font Instance
        textframe: Need a textframe object, if fit_text==True
        fit_text: Bool - True = Scale the font size to fit the text box. Requires a textframe
        font_name: Str
        font_file: Str - Relative path to a font file
        font_size: Int
        font_italic: Bool
        font_underline: Bool
        font_color: Tuple of Int, (R, G, B)
        :return: font, The Font Instance
        '''

        font = font_obj

        # Resize text to fit textframe
        if not textframe == None and fit_text == True:
            # if font_name == "Raleway":
            #     font_file = os.path.join(thisdir, default_font_file)
            #     textframe.fit_text(font_family=font_name, max_size=font_size, bold=font_bold,
            #                        italic=font_italic, font_file=font_file)
            # else:
            textframe.fit_text(font_family=font_name, max_size=font_size, bold=font_bold,
                               italic=font_italic)

        # ============================== Font
        # Font color
        font.color.rgb = RGBColor(*font_color)
        font.underline = font_underline

        # Font name/size/bold/italic if not set with textframe.fit_text
        if fit_text == False:
            font.name = font_name
            font.size = Pt(font_size)
            font.bold = font_bold
            font.italic = font_italic

        # Add color theme
        if font_color_theme is not None:
            font.color.theme_color = theme_color_index_dct[font_color_theme]

        # Font color brightness
        font.color.brightness = font_color_brightness

        return font

    @staticmethod
    def fix_yaxis(chart, fix_point, legend=None):
        """
        Method to fix the vertical axis in a charts plotArea
        :param
            chart:      An instance of a Chart object
            fix_point:  Where to fix the vertical axis (Not implemented - TODO)
        :return: None, will edit the plotArea in place
        """

        if legend=='right':
            xml_string = """<c:layout xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart">
            <c:manualLayout>
            <c:layoutTarget val="inner"/>
            <c:xMode val="edge"/>
            <c:yMode val="edge"/>
            <c:x val="0.41682566853056413"/>
            <c:y val="3.3743961352657004E-2"/>
            <c:w val="0.45661636045494312"/>
            <c:h val="0.89697898550724642"/>
            </c:manualLayout>
            </c:layout>"""
        else:
            xml_string = """<c:layout xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart">
            <c:manualLayout>
            <c:layoutTarget val="inner"/>
            <c:xMode val="edge"/>
            <c:yMode val="edge"/>
            <c:x val="0.41682566853056413"/>
            <c:y val="3.3743961352657004E-2"/>
            <c:w val="0.55661636045494312"/>
            <c:h val="0.89697898550724642"/>
            </c:manualLayout>
            </c:layout>"""

        xml_insert = etree.fromstring(xml_string)
        chart._element.plotArea.append(xml_insert)

    @staticmethod
    def add_net(slide,
                df,
                left=Cm(27.55), top=Cm(3.69), width=Cm(1.8), # width for 1 column
                height=Cm(10.68),
                margin_left=0.0,
                margin_right=0.0,
                margin_top=0.0,
                margin_bottom=0.0,

                # side_member_font_size=8,
                # side_member_font_name='Trebuchet MS',
                # side_member_font_bold=False,
                # side_member_font_italic=False,
                # side_member_font_color=(109,111,113),
                # side_member_font_para_alignment=PP_ALIGN.LEFT,
                # side_member_vert_alignment=MSO_ANCHOR.MIDDLE,
                # sidemember_shading=True,
                # sidemember_shading_color=(255,255,255),

                top_member_font_size=8,
                top_member_font_name='Trebuchet MS',
                top_member_font_bold=False,
                top_member_font_italic=False,
                top_member_font_color=(0,0,0),
                top_member_font_para_alignment=paragraph_alignment_pos_dct['center'],
                top_member_vert_alignment=vertical_alignment_pos_dct['middle'],
                top_member_shading=True,
                top_member_shading_color=(255,255,255),

                values_font_size=8,
                values_font_name='Trebuchet MS',
                values_font_bold=False,
                values_font_italic=False,
                values_font_color=(0,0,0),
                values_font_para_alignment=paragraph_alignment_pos_dct['center'],
                values_vert_alignment=vertical_alignment_pos_dct['top'],
                values_shading=True,
                values_shading_shading_color=(255,255,255),
                ):
        #-------------------------------------------------------------------------

        rows = len(df.index)
        cols = len(df.columns)

        # Height of table and rows
        header_row_height = Cm(1.7)
        if rows == 1:
            top = top + int(round(height * (0.3 / rows)) - header_row_height)
        else:
            top = top + int(round(height* (0.45/rows)) - header_row_height)
        plot_area = int(round(height * 0.9)) - Cm(0.4)
        value_row_height = int(round(plot_area / rows))
        height = header_row_height + rows * value_row_height

        # Width of table
        width = width * cols

        shapes = slide.shapes
        table = shapes.add_table(rows+1, cols, left, top, width, height).table
        table.horz_banding = True
        #isolate seperate sections of a table
        row_labels = list(df.index)
        col_labels = list(df.columns)
        table_values = df.values
        #question_label = df.index.get_level_values(level=0)[0]

        #table specific properties
        for i in range(0, rows + 1):
            if i == 0:
                table.rows[i].height = header_row_height
            else:
                table.rows[i].height = value_row_height

            for x in range(0, cols):

                cell = table.cell(i, x)

                cell.margin_left= Cm(margin_left)
                cell.margin_right = Cm(margin_right)
                cell.margin_top = Cm(margin_top)
                cell.margin_bottom = Cm(margin_bottom)

        #add col labels
        for idx, col_label in enumerate(col_labels):

            #table.columns[0].width = Emu(first_column_width)

            cell = table.cell(0, idx)
            cell.vertical_anchor = top_member_vert_alignment

            if top_member_shading:
                if top_member_shading_color == "No fill":
                    fill = cell.fill
                    fill.background()
                else:
                    cfill = cell.fill
                    cfill.solid()
                    cfill.fore_color.rgb = RGBColor(*top_member_shading_color)
                    #cfill.fore_color.brightness = textbox_color_brightness

            textframe = cell.text_frame
            paragraph = textframe.paragraphs[0]
            paragraph.font.size = Pt(top_member_font_size)
            paragraph.font.name = top_member_font_name
            paragraph.font.bold = top_member_font_bold
            paragraph.font.italic = top_member_font_italic
            paragraph.font.color.rgb = RGBColor(*top_member_font_color)
            paragraph.alignment = top_member_font_para_alignment

            cell.text = col_label

        #add values
        for i, val in enumerate(table_values):
            for x, subval in enumerate(val):

                cell = table.cell(i+1, x)
                cell.vertical_anchor = values_vert_alignment
                cell.margin_top = Cm(0.1)

                if values_shading:
                    if values_shading_shading_color == "No fill":
                        fill = cell.fill
                        fill.background()
                    else:
                        cfill = cell.fill
                        cfill.solid()
                        cfill.fore_color.rgb = RGBColor(*values_shading_shading_color)

                textframe = cell.text_frame
                paragraph = textframe.paragraphs[0]
                paragraph.font.size = Pt(values_font_size)
                paragraph.font.name = values_font_name
                paragraph.font.bold = values_font_bold
                paragraph.font.italic = values_font_italic
                paragraph.font.color.rgb = RGBColor(*values_font_color)
                paragraph.alignment = values_font_para_alignment
                #paragraph.line_spacing = Pt(6)
                cell.text = str(subval)

