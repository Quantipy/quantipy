# encoding: utf-8

from collections import OrderedDict

import pandas as pd

# ----------------------------------------------------------------------------
# Font defaults

default_font_name='Trebuchet MS'
# default_font_file='fonts\Raleway-Regular.ttf'

default_font = OrderedDict([
    ('font_name', default_font_name),
    ('font_size', 9),
    ('font_bold', False),
    ('font_italic', False),
    ('font_underline', False),
    ('font_color', (89, 89, 89)),
    ('font_color_brightness', 0),
    ('font_color_theme', None),
])

# Chart Legend
default_font_legend = default_font.copy()

# Chart Category axis
default_font_caxis = default_font.copy()

# Chart Value axis
default_font_vaxis = default_font.copy()
default_font_vaxis['font_bold'] = True

# Chart data labels
default_font_data_label = default_font.copy()
default_font_data_label['font_size'] = 8.5
default_font_data_label['font_color'] = (0, 0, 0)

# ----------------------------------------------------------------------------
# Textframe defaults

default_textframe = OrderedDict([
    ('fit_text', True),
    ('margin_left', 90000),  # Cm(0.25)
    ('margin_right', 90000),  # Cm(0.25)
    ('margin_top', 46800),  # Cm(0.13)
    ('margin_bottom', 46800),  # Cm(0.13)
    ('vertical_alignment', 'top'),
    ('horizontal_alignment', 'left'),
    ('font_kwargs', default_font.copy()),
])

# ---------------------------------------------------------------------------
# Textbox defaults
default_textbox = OrderedDict([
    ('text', ''),
    ('left', 831600),  # Cm(2.31)
    ('top', 1170000),  # Cm(3.25)
    ('width', 10522800),  # Cm(29.23)
    ('height', 630000),  # Cm(1.75)
    ('rotation', 0),
    ('textbox_fill_solid', False),
    ('textbox_color', (100, 0, 0)),
    ('textbox_color_brightness', 0),
    ('textframe_kwargs', default_textframe.copy()),
])

# ---------------------------------------------------------------------------
# Table cell defaults
default_cell_kwargs = OrderedDict([
    ('margin_left', 36000),  # Cm(0.1)
    ('margin_right', 36000),  # Cm(0.1)
    ('margin_top', 36000),  # Cm(0.1)
    ('margin_bottom', 36000),  # Cm(0.1)
    ('vertical_alignment', 'top'),
    ('shading', True),
    ('shading_color', (255,255,255)),
])

# -----------------------------------------------------------------------------
# Table defaults
default_table = OrderedDict([
    ('dataframe', pd.DataFrame()),
    ('text', None),
    ('left', 838800), ('top', 1476000), ('width', 10515600), ('height', 4140000),

    ('show_side_member', True),
    ('side_member_column_width', 2160000),  # Cm(6)
    ('side_member_textframe_kwargs', default_textframe.copy()),
    ('side_member_cell_kwargs', default_cell_kwargs.copy()),

    ('show_top_member', True),
    ('top_member_row_height', 360000),  # Cm(1)
    ('top_member_textframe_kwargs', default_textframe.copy()),
    ('top_member_cell_kwargs', default_cell_kwargs.copy()),

    ('values_row_height', 360000),  # Cm(1)
    ('values_column_width', 720000),  # Cm(2)
    ('values_textframe_kwargs', default_textframe.copy()),
    ('values_cell_kwargs', default_cell_kwargs.copy()),
    ('values_prefix', None),
    ('values_prefix_columns', 'all'),

    ('values_suffix', None),
    ('values_suffix_columns', 'all'),

    ('top_left_corner_textframe_kwargs', default_textframe.copy()),
    ('top_left_corner_cell_kwargs', default_cell_kwargs.copy()),
])

default_side_table = default_table.copy()
default_side_table['show_side_member'] = False
default_side_table['values_column_width'] = 648000  # Cm(1.8)
default_side_table['top_member_row_height'] = 612000  # Cm(1.7)
default_side_table['values_textframe_kwargs'] = default_textframe.copy()
default_side_table['values_textframe_kwargs']['horizontal_alignment'] = 'center'
default_side_table['values_textframe_kwargs']['font_kwargs'] = default_font.copy()
default_side_table['values_textframe_kwargs']['font_kwargs']['font_size'] = 8

default_side_table['top_member_textframe_kwargs'] = default_textframe.copy()
default_side_table['top_member_textframe_kwargs']['horizontal_alignment'] = 'center'
default_side_table['top_member_textframe_kwargs']['font_kwargs'] = default_font.copy()
default_side_table['top_member_textframe_kwargs']['font_kwargs']['font_size'] = 8

default_side_table['top_member_cell_kwargs'] = default_cell_kwargs.copy()
default_side_table['top_member_cell_kwargs']['vertical_alignment'] = 'middle'

default_side_table['side_member_textframe_kwargs']['font_kwargs'] = default_font.copy()
default_side_table['side_member_textframe_kwargs']['font_kwargs']['font_size'] = 8

# -------------------------------------------------------------------------------------------
# Chart defaults (Bar_clustered 1 series)

default_chart = OrderedDict([
    ('dataframe', pd.DataFrame()),
    ('chart_type', 'bar_clustered'),
    ('left', 838800), ('top', 1476000), ('width', 10515600), ('height', 4140000),
    ('chart_style', 2),

    # Title
    ('has_chart_title', False),
    ('titletext', None),
    ('textframe_kwargs', default_textframe.copy()),

    # Legend properties
    ('has_legend', False),
    ('legend_position', 'right'),
    ('legend_in_layout', False),  # will we ever set this True?
    ('legend_horz_offset', 0.1583),
    ('legend_font_kwargs', default_font_legend.copy()),

    # Category axis properties
    ('caxis_visible', True),
    ('caxis_tick_label_position', 'low'),
    ('caxis_tick_labels_offset', 730),
    ('caxis_has_major_gridlines', False),
    ('caxis_has_minor_gridlines', False),
    ('caxis_major_tick_mark', 'outside'),
    ('caxis_minor_tick_mark', 'none'),
    ('caxis_font_kwargs', default_font_caxis.copy()),

    # Value axis properties
    ('vaxis_visible', True),
    ('vaxis_tick_label_position', 'low'),
    ('vaxis_has_major_gridlines', True),
    ('vaxis_has_minor_gridlines', False),
    ('vaxis_major_tick_mark', 'outside'),
    ('vaxis_minor_tick_mark', 'none'),
    ('vaxis_max_scale', 1.0),
    ('vaxis_min_scale', 0.0),
    ('vaxis_major_unit', 0.1),
    ('vaxis_minor_unit', None),
    ('vaxis_tick_labels_num_format', '0%'),
    ('vaxis_tick_labels_num_format_is_linked', False),
    ('vaxis_font_kwargs', default_font_vaxis.copy()),

    # Datalabel properties
    ('plot_has_data_labels', True),
    ('data_labels_position', 'outside_end'),
    ('data_labels_num_format', '0%'),
    ('data_labels_num_format_is_linked', False),
    ('data_labels_font_kwargs', default_font_data_label.copy()),

    # Plot properties
    ('plot_vary_by_cat', False),
    ('plot_gap_width', 150),
    ('plot_overlap', -10),

    # Number format
    ('number_format', '0.00%'),
    ('xl_number_format', '0.00%'),
    # Sig_testing
    ('sig_test_visible', False),
    ('sig_test_results', None),
])

# --------------------------
# Bar_Clustered
default_chart_bar = default_chart.copy()
# fix y axis (False, 'center'), Currently only an option for bar chart, legend is not available if used.
default_chart_bar['fix_yaxis'] = False

# --------------------------
# Bar_Stacked_100%
default_chart_bar_stacked100 = default_chart.copy()
default_chart_bar_stacked100['chart_type'] = 'bar_stacked_100'
# Legend properties
default_chart_bar_stacked100['has_legend'] = True
default_chart_bar_stacked100['legend_horz_offset'] = 0
# Datalabel properties
default_chart_bar_stacked100['data_labels_position'] = 'center'
# Plot properties
# default_chart_bar_stacked100['plot_vary_by_cat'] = True
default_chart_bar_stacked100['plot_overlap'] = 100

# ---------------------------
# Line
default_chart_line = default_chart.copy()
default_chart_line['chart_type'] = 'line'
# Legend properties
default_chart_line['legend_position'] = 'bottom'
default_chart_line['legend_horz_offset'] = 0
# Category axis properties
default_chart_line['caxis_tick_label_position'] = 'low'
default_chart_line['caxis_tick_labels_offset'] = 100
# Datalabel properties
default_chart_line['data_labels_position'] = 'above'
# Plot properties
default_chart_line['plot_overlap'] = 10
default_chart_line['smooth_line'] = False

# ----------------------------
# Column
default_chart_column = default_chart.copy()
default_chart_column['chart_type'] = 'column_clustered'
# Legend properties
default_chart_column['legend_position'] = 'bottom'
default_chart_column['legend_horz_offset'] = 0
# Category axis properties
default_chart_column['caxis_tick_label_position'] = 'next_to_axis'
default_chart_column['caxis_tick_labels_offset'] = 100
# Plot properties
default_chart_column['plot_overlap'] = -10

# ---------------------------
# Pie
default_chart_pie = default_chart.copy()
default_chart_pie['chart_type'] = 'pie'
# Legend properties
default_chart_pie['has_legend'] = True
default_chart_pie['legend_position'] = 'bottom'
default_chart_pie['legend_horz_offset'] = 0
# Plot properties
default_chart_pie['plot_vary_by_cat'] = True

# ----------------------------
# Question_text-dict for default_slide_kwargs
header_shape = default_textbox.copy()
header_shape['left'] = 838800
header_shape['top'] = 1026000
header_shape['width'] = 10515600
header_shape['height'] = 468000
header_shape['textframe_kwargs'] = default_textframe.copy()
header_shape['textframe_kwargs']['font_kwargs'] = default_font.copy()
header_shape['textframe_kwargs']['font_kwargs']['font_size']=11
header_shape['textframe_kwargs']['font_kwargs']['font_bold']=True
header_shape['textframe_kwargs']['font_kwargs']['font_italic']=True

# ----------------------------
# Base_description-dict for default_slide_kwargs
footer_shape = default_textbox.copy()
footer_shape['left'] = 838800
footer_shape['top'] = 5652000
footer_shape['width'] = 10515600
footer_shape['height'] = 396000
footer_shape['textframe_kwargs'] = default_textframe.copy()
footer_shape['textframe_kwargs']['font_kwargs'] = default_font.copy()
footer_shape['textframe_kwargs']['font_kwargs']['font_bold'] = True

shapes = {'charts' : {'bar': default_chart_bar,
                      'bar_stacked100': default_chart_bar_stacked100,
                      'line': default_chart_line,
                      'column': default_chart_column,
                      'pie': default_chart_pie},
          'textboxes' : {'header': header_shape,
                         'footer': footer_shape},
          'tables' : {'table' : default_table,
                      'side_table': default_side_table},
          }

