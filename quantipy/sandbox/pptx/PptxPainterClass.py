# encoding: utf-8

from lxml import etree

# Imports from Python-PPTX
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.util import (
    Emu,
    Pt,
    Cm,
    Inches)

from pptx.chart.data import ChartData
from pptx.enum.chart import XL_CHART_TYPE
from pptx.dml.color import RGBColor

from enumerations import (
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

from PptxDefaultsClass import PptxDefaults
import pandas as pd


# chartdata_from_dataframe taken from topy.core.pandas_pptx.py
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


# return_slide_layout_by_name is taken from quantipy.core.builds.powerpoint.visual_editor.py
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


class PptxPainter(object):
    """
    A convenience wrapper around the python-pptx library
    """

    def __init__(self, path_to_presentation, slide_layout=None, shape_properties=None):
        """
        Makes a Presentation instance and also defines a default slide layout if specified.

        Parameters
        ----------
        path_to_presentation: str
            Full path to PowerPoint book
        slide_layout: int
            A PowerPoint slide layout.
            To see available Slide Layouts in a PPTX, select the Viev menu and click Slide Master.
        shape_properties: quantipy.sandbox.pptx.PptxDefaultsClass.PptxDefaults
            An instance of PptxDefaults
        """

        self.presentation  = Presentation(path_to_presentation) # TODO PptxPainter - Path checking # type: Presentation
        if slide_layout is None:
            self.default_slide_layout = None
        else:
            self.default_slide_layout = self.set_slide_layout(slide_layout)

        # Add all the dafault dicts to the class -
        if shape_properties:
            self._shape_properties = shape_properties
        else:
            self._defaults = PptxDefaults()
            self._shape_properties = self._defaults.shapes

        self.textbox = self._shape_properties.textbox
        self.chart = self._shape_properties.chart
        self.table = self._shape_properties.table

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
                    }

    def queue_slide_items(self, pptx_chain, items):
        """
        Helper function to queue a full automated slide.
        Includes queueing of header with question text, chart and footer with base description

        Parameters
        ----------
        pptx_chain: quantipy.sandbox.pptx.PptxChainClass.PptxChain
            An instance of a PptxChain
        items: basestring
            A string of slide items, separated with +, eg. 'basic+nets+means-table'
            Supported items are 'basic', 'nets', 'means-table', 'means-line', 'nets-table',
        Returns
        -------
        None
        """

        # Question text
        draft = self.draft_textbox(self._shape_properties.textbox_header, pptx_chain.question_text)
        self.queue_textbox(settings=draft)

        draft = self.draft_textbox(self._shape_properties.textbox_footer, pptx_chain.base_text)
        self.queue_textbox(settings=draft)

        shape_items = items.split('+')
        for shape_item in shape_items:
            if shape_item == 'basic':
                chart_df = pptx_chain.chart_df.get_cpct()
                draft = self.draft_autochart(chart_df(), pptx_chain.chart_type)
                self.queue_chart(settings=draft)
            if shape_item == 'basic_nets':
                chart_df = pptx_chain.chart_df.get('c_pct,net')
                draft = self.draft_autochart(chart_df(), pptx_chain.chart_type)
                self.queue_chart(settings=draft)
            if shape_item == 'table':
                chart_df = pptx_chain.chart_df.get('c_pct')
                draft = self.draft_table(self.table, chart_df())
                self.queue_table(settings=draft)
        return None

    def clear_tables(self):
        """
        Initilalize the slide_kwargs "tables" dict
        :return: None, removes all keys from self.slide_kwargs['tables']
        """
        self.clear_queue('tables')

    def clear_charts(self):
        """
        Initilalize the slide_kwargs "charts" dict
        :return: None, removes all keys from self.slide_kwargs['charts']
        """
        self.clear_queue('charts')

    def clear_textboxes(self):
        """
        Initilalize the slide_kwargs "txtboxes" dict
        :return: None, removes all keys from self.slide_kwargs['txtboxes']
        """
        self.clear_queue('textboxs')

    def clear_queue(self, key):
        """
        Initialize the shape dicts in slide_kwargs
        :param key: String ('all', 'charts', textboxes','tables')
        :return: None, removes all keys from requested dict in self.slide_kwargs
        """
        if key=='all':
            for item in self.slide_kwargs.keys():
                self.slide_kwargs[item].clear()
        elif key=='charts':
            self.slide_kwargs['charts'].clear()
        elif key=='textboxes':
            self.slide_kwargs['textboxs'].clear()
        elif key=='tables':
            self.slide_kwargs['tables'].clear()

    def set_slide_layout(self, slide_layout):
        """
        Method to set a Slide Layout
        :param
            slide_layout: Int
                To see available Slide Layouts in a PPTX, select the Viev menu and click Slide Master.
        :return: Instance of SlideLayout set to the specified slide layout
        """
        if isinstance(slide_layout, int):
            return self.presentation.slide_layouts[slide_layout]
        else:
            return return_slide_layout_by_name(self.presentation, slide_layout)

    def add_slide(self, slide_layout=None):
        """
        Method that creates a Slide instance
        :param
            slide_layout: Int
                The Slide Layout to use.
                To see available Slide Layouts in a PPTX, select the Viev menu and click Slide Master.
                If no Slide layout is specified then self.default_slide_layout will be used

        :return: None - sets the self.slide property
        """
        if slide_layout is None:
            if self.default_slide_layout is None:
                raise ValueError('No slide layout found! Specify a slide layout or set a default slide layout')
            else:
                slide_layout = self.default_slide_layout
        else:
            slide_layout = self.set_slide_layout(slide_layout=slide_layout)

        return self.presentation.slides.add_slide(slide_layout)

    def draft_textbox(self, settings, text=''):
        """
        Sets attribute self.textbox

        Parameters
        ----------
        settings: dict
            A dict of textbox settings, see dict default_textbox in pptx_defaults.py
        text: basestring
            Text to show in textbox
        Returns: self.textbox
        -------
        """
        self.textbox = settings.copy()
        self.textbox['text'] = text
        return self.textbox

    def draft_chart(self, settings, dataframe):
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

        self.chart = settings.copy()
        self.chart['dataframe'] = dataframe
        return self.chart

    def draft_autochart(self, dataframe, chart_type):
        """
        Sets self.chart['chart_type'] and self.chart['dataframe']

        Parameters
        ----------
        dataframe: pandas.core.frame.DataFrame
        chart_type: str
            A string corresponding to the keys in Dict "chart_type_dct" from "enumerations.py"
        Returns: self.chart
        -------
        """
        # TODO Class PptxPainter - Method draft_chart: check for correct chart_type input
        if chart_type == 'pie':
            self.chart = self.chart_pie.copy()
        elif chart_type == 'bar_clustered' or chart_type == 'bar':
            self.chart = self.chart_bar.copy()
            if len(dataframe.columns) > 1:
                self.chart['has_legend'] = True
        elif chart_type == 'bar_stacked_100':
            self.chart = self.chart_bar_stacked100.copy()
        else:
            self.chart = self.chart_bar.copy()

        self.chart['dataframe'] = dataframe

        return self.chart

    def draft_table(self, settings, dataframe):
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

        self.table = settings.copy()
        self.table['dataframe'] = dataframe
        return self.table

    def queue_chart(self, settings=None, name=None):
        """
        Will add a chart to the Slide properties Dict
        :param
            settings: A dictionary of chart settings, default is self.chart
            name: Optionally give the chart a name. If none the chart will be named 'chart[n]'
        :return:
            None, Adds a key to self.slide_kwargs['charts']
        """
        if settings is None:
            settings = self.chart.copy()
        self._add('chart', settings, name=name)

    def queue_textbox(self, settings=None, name=None):
        """
        Will add a textbox to the Slide properties Dict
        :param
            settings:  A dictionary of textbox settings, deafult is self.textbox
            name: Optionally give the textbox a name. If none the textbox will be named 'textbox[n]'
        :return:
            None, adds a key to self.slide_kwargs['textboxes']
        """
        if settings is None:
            settings = self.textbox.copy()

        self._add('textbox', settings, name=name)

    def queue_table(self, settings=None, name=None):
        """
        Will add a table to the Slide properties Dict
        :param
            settings: A dictionary of table settings, default is self.table
            name: Optionally give the table a name. If none the table will be named 'table[n]'
        :return:
            None, Adds a key to self.slide_kwargs['tables']
        """
        if settings is None:
            settings = self.table.copy()
        self._add('table', settings, name=name)

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
        for _type, draft in kwargs.iteritems():
            # Add text boxes
            if _type == 'textboxs':
                for name, settings in draft.iteritems():
                    textbox=self.add_textbox(slide, **settings)
            # Add charts
            if _type == 'charts':
                for name, settings in draft.iteritems():
                    chart=self.add_chart(slide, **settings)
            # Add tables
            if _type == 'tables':
                for name, settings in draft.iteritems():
                    chart=self.add_table(slide, **settings)

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

    def add_table(self,
                  slide,
                  dataframe,
                  question_text=None,
                  left=Cm(4), top=Cm(8), width=Cm(5), height=Cm(8),
                  margin_left=Cm(0.5),
                  margin_right=Cm(0.5),
                  margin_top=Cm(0.5),
                  margin_bottom=Cm(0.5),

                  show_side_member=True,
                  side_member_column_width=Cm(1),
                  side_member_font_size=10,
                  side_member_font_name='Verdana',
                  side_member_font_bold=False,
                  side_member_font_italic=False,
                  side_member_font_color=(0, 0, 0),
                  side_member_font_para_alignment='left',
                  side_member_vert_alignment='top',
                  side_member_shading=False,
                  side_member_shading_color='No fill', # (0, 0, 128)

                  show_top_member=True,
                  top_member_row_height=Cm(1),
                  top_member_font_size=10,
                  top_member_font_name='Verdana',
                  top_member_font_bold=False,
                  top_member_font_italic=False,
                  top_member_font_color=(0, 0, 0),
                  top_member_font_para_alignment='center',
                  top_member_vert_alignment='bottom',
                  top_member_shading=False,
                  top_member_shading_color='No fill', # (0, 0, 128)

                  values_font_size=10,
                  values_font_name='Verdana',
                  values_font_bold=False,
                  values_font_italic=False,
                  values_font_color=(0, 0, 0),
                  values_font_para_alignment='right',
                  values_vert_alignment='top',
                  values_shading=True,
                  values_shading_shading_color='No fill', # (0, 0, 128)

                  question_box_font_size=10,
                  question_box_font_name='Verdana',
                  question_box_font_bold=False,
                  question_box_font_italic=False,
                  question_box_font_color=(0, 0, 0),
                  question_box_vert_alignment='bottom',
                  question_box_para_alignment='left',
                  question_box_shading=False,
                  question_box_shading_color='No fill' # (0, 0, 128)
                  ):
        # -------------------------------------------------------------------------

        rows = len(dataframe.index) + 1
        cols = len(dataframe.columns) + 1

        shapes = slide.shapes
        table = shapes.add_table(rows, cols, left, top, width, height).table

        # isolate seperate sections of a table
        row_labels = list(dataframe.index)
        col_labels = list(dataframe.columns)
        table_values = dataframe.values

        # table specific properties
        for i in range(0, rows):
            for x in range(0, cols):
                cell = table.cell(i, x)

                cell.margin_left = margin_left
                cell.margin_right = margin_right
                cell.margin_top = margin_top
                cell.margin_bottom = margin_bottom

        # row specific properties
        for idx, row_label in enumerate(row_labels):

            cell = table.cell(idx + 1, 0)
            cell.vertical_anchor = vertical_alignment_pos_dct[side_member_vert_alignment]

            if side_member_shading:

                if side_member_shading_color == "No fill":
                    fill = cell.fill
                    fill.background()
                else:
                    cfill = cell.fill
                    cfill.solid()
                    cfill.fore_color.rgb = RGBColor(*side_member_shading_color)
                    # cfill.fore_color.brightness = textbox_color_brightness

            textframe = cell.text_frame
            paragraph = textframe.paragraphs[0]
            paragraph.font.size = Pt(side_member_font_size)
            paragraph.font.name = side_member_font_name
            paragraph.font.color.rgb = RGBColor(*side_member_font_color)
            paragraph.font.bold = side_member_font_bold
            paragraph.font.italic = side_member_font_italic
            paragraph.alignment = paragraph_alignment_pos_dct[side_member_font_para_alignment]

            cell.text = row_label

        # add col labels
        for idx, col_label in enumerate(col_labels):

            table.columns[0].width = side_member_column_width

            cell = table.cell(0, idx + 1)
            cell.vertical_anchor = vertical_alignment_pos_dct[top_member_vert_alignment]

            if top_member_shading:
                if top_member_shading_color == "No fill":
                    fill = cell.fill
                    fill.background()
                else:
                    cfill = cell.fill
                    cfill.solid()
                    cfill.fore_color.rgb = RGBColor(*top_member_shading_color)
                    # cfill.fore_color.brightness = textbox_color_brightness

            textframe = cell.text_frame
            paragraph = textframe.paragraphs[0]
            paragraph.font.size = Pt(top_member_font_size)
            paragraph.font.name = top_member_font_name
            paragraph.font.bold = top_member_font_bold
            paragraph.font.italic = top_member_font_italic
            paragraph.font.color.rgb = RGBColor(*top_member_font_color)
            paragraph.alignment = paragraph_alignment_pos_dct[top_member_font_para_alignment]

            cell.text = col_label

        # add values
        for i, val in enumerate(table_values):
            for x, subval in enumerate(val):

                cell = table.cell(i + 1, x + 1)

                cell.vertical_anchor = vertical_alignment_pos_dct[values_vert_alignment]

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
                paragraph.alignment = paragraph_alignment_pos_dct[values_font_para_alignment]

                cell.text = str(subval)

        # add question label
        if question_text is not None:
            cell = table.cell(0, 0)
            cell.vertical_anchor = vertical_alignment_pos_dct[question_box_vert_alignment]

            if question_box_shading:
                if top_member_shading_color == "No fill":
                    fill = cell.fill
                    fill.background()
                else:
                    cfill = cell.fill
                    cfill.solid()
                    cfill.fore_color.rgb = RGBColor(*question_box_shading_color)

            textframe = cell.text_frame
            paragraph = textframe.paragraphs[0]
            paragraph.font.size = Pt(question_box_font_size)
            paragraph.font.name = question_box_font_name
            paragraph.font.bold = question_box_font_bold
            paragraph.font.italic = question_box_font_italic
            paragraph.font.color.rgb = RGBColor(*question_box_font_color)
            paragraph.alignment = paragraph_alignment_pos_dct[question_box_para_alignment]

            cell.text = question_text

        return table

    def add_chart(self, slide,
                  dataframe=None,
                  chart_type='bar_stacked_100',
                  left=Cm(2.31), top=Cm(5.93), width=Cm(29.23), height=Cm(10.5),
                  chart_style=2,

                  # Title
                  has_chart_title=False,
                  titletext=None,
                  textframe_kwargs=None, # type: dict

                  # Legend properties
                  has_legend=True,
                  legend_position='right',
                  legend_in_layout=False,  # will we ever set this True?
                  legend_horz_offset=0.1583,
                  legend_font_kwargs=None, # type: dict

                  # Category axis properties
                  caxis_visible=True,
                  caxis_tick_label_position='next_to_axis',
                  caxis_tick_labels_offset=730,
                  caxis_has_major_gridlines=False,
                  caxis_has_minor_gridlines=False,
                  caxis_major_tick_mark='outside',
                  caxis_minor_tick_mark='none',
                  caxis_font_kwargs=None, # type: dict

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
                  vaxis_font_kwargs=None, # type: dict

                  # Fix yaxis (False, 'center', ?). Currently only an option for bar chart
                  fix_yaxis = False,

                  # Datalabel properties
                  plot_has_data_labels=True,
                  data_labels_position='center',
                  data_labels_num_format='0%',
                  data_labels_num_format_is_linked=False,
                  data_labels_font_kwargs=None, # type: dict

                  # Plot properties
                  plot_vary_by_cat=False,
                  plot_gap_width=150,
                  plot_overlap=100,
                  smooth_line=False,

                  # Number format
                  number_format='0.00%',
                  xl_number_format='0.00%'
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
                    text="",
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
                      text="",
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
        if text == '':
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

    def fix_yaxis(self, chart, fix_point, legend=None):
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

    def add_net(self,
                slide,
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