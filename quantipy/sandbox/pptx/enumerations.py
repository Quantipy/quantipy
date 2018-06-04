# encoding: utf-8

'''
@author: Majeed.sahebzadha
'''

from pptx.enum.chart import(
  XL_CHART_TYPE, 
  XL_LABEL_POSITION, 
  XL_LEGEND_POSITION, 
  XL_TICK_MARK, 
  XL_TICK_LABEL_POSITION
  )
from pptx.enum.dml import(
  MSO_THEME_COLOR, 
  MSO_COLOR_TYPE,
  MSO_FILL
  )
from pptx.enum.text import(
  PP_ALIGN,
  MSO_AUTO_SIZE, 
  MSO_ANCHOR
  )
 
#------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

'''
See: http://python-pptx.readthedocs.io/en/latest/api/enum/index.html
TO DO: add individual description for each enum
'''

# Specifies the type of bitmap used for the fill of a shape
fill_type_dct = {
    'background': MSO_FILL.BACKGROUND,
    'gradient': MSO_FILL.GRADIENT,
    'group': MSO_FILL.GROUP,
    'patterned': MSO_FILL.PATTERNED,
    'picture': MSO_FILL.PICTURE,
    'solid': MSO_FILL.SOLID,
    'textured': MSO_FILL.TEXTURED
    }

# Specifies where the data label is positioned
data_label_pos_dct = {
    'above': XL_LABEL_POSITION.ABOVE,
    'below': XL_LABEL_POSITION.BELOW,
    'best_fit': XL_LABEL_POSITION.BEST_FIT,
    'center': XL_LABEL_POSITION.CENTER,
    'inside_base': XL_LABEL_POSITION.INSIDE_BASE,
    'inside_end': XL_LABEL_POSITION.INSIDE_END,
    'left': XL_LABEL_POSITION.LEFT,
    'mixed': XL_LABEL_POSITION.MIXED,
    'outside_end': XL_LABEL_POSITION.OUTSIDE_END,
    'right': XL_LABEL_POSITION.RIGHT
    }

# Specifies the position of the legend on a chart
legend_pos_dct = {
    'bottom': XL_LEGEND_POSITION.BOTTOM,
    'corner': XL_LEGEND_POSITION.CORNER,
    'custom': XL_LEGEND_POSITION.CUSTOM,
    'left': XL_LEGEND_POSITION.LEFT,
    'right': XL_LEGEND_POSITION.RIGHT,
    'top': XL_LEGEND_POSITION.TOP
    }

# Specifies the position of tick-mark labels on a chart axis
tick_label_pos_dct = {
    'high': XL_TICK_LABEL_POSITION.HIGH,
    'low': XL_TICK_LABEL_POSITION.LOW,
    'next_to_axis': XL_TICK_LABEL_POSITION.NEXT_TO_AXIS,
    'none': XL_TICK_LABEL_POSITION.NONE
    }

# Specifies a type of axis tick for a chart.
tick_mark_pos_dct = {
    'cross': XL_TICK_MARK.CROSS,            #Tick mark crosses the axis
    'inside': XL_TICK_MARK.INSIDE,          #Tick mark appears inside the axis
    'none': XL_TICK_MARK.NONE,              #No tick mark
    'outside': XL_TICK_MARK.OUTSIDE         #Tick mark appears outside the axis
    }

# Specifies the vertical alignment of text in a text frame
vertical_alignment_pos_dct = {
    'top': MSO_ANCHOR.TOP,
    'middle': MSO_ANCHOR.MIDDLE,
    'bottom': MSO_ANCHOR.BOTTOM,
    'mixed': MSO_ANCHOR.MIXED
    }

# Specifies the horizontal alignment for one or more paragraphs
paragraph_alignment_pos_dct = {
    'center': PP_ALIGN.CENTER,
    'distribute': PP_ALIGN.DISTRIBUTE,
    'justify': PP_ALIGN.JUSTIFY,
    'justify_low': PP_ALIGN.JUSTIFY_LOW,
    'left': PP_ALIGN.LEFT,
    'right': PP_ALIGN.RIGHT,
    'thai_distribute': PP_ALIGN.THAI_DISTRIBUTE,
    'mixed': PP_ALIGN.MIXED
    }

# Indicates the Office theme color, one of those shown in the color gallery on the formatting ribbon
theme_color_index_dct = {
    'not_theme_color': MSO_THEME_COLOR.NOT_THEME_COLOR,
    'accent_1': MSO_THEME_COLOR.ACCENT_1,
    'accent_2': MSO_THEME_COLOR.ACCENT_2,
    'accent_3': MSO_THEME_COLOR.ACCENT_3,
    'accent_4': MSO_THEME_COLOR.ACCENT_4,
    'accent_5': MSO_THEME_COLOR.ACCENT_5,
    'accent_6': MSO_THEME_COLOR.ACCENT_6,
    'background_1': MSO_THEME_COLOR.BACKGROUND_1,
    'background_2': MSO_THEME_COLOR.BACKGROUND_2,
    'dark_1': MSO_THEME_COLOR.DARK_1,
    'dark_2': MSO_THEME_COLOR.DARK_2,
    'followed_hyperlink': MSO_THEME_COLOR.FOLLOWED_HYPERLINK,
    'hyperlink': MSO_THEME_COLOR.HYPERLINK,
    'light_1': MSO_THEME_COLOR.LIGHT_1,
    'light_2': MSO_THEME_COLOR.LIGHT_2,
    'text_1': MSO_THEME_COLOR.TEXT_1,
    'text_2': MSO_THEME_COLOR.TEXT_2,
    'mixed': MSO_THEME_COLOR.MIXED
    }

# Specifies the type of a chart
chart_type_dct = {
    'area': XL_CHART_TYPE.AREA,
    'area_stacked': XL_CHART_TYPE.AREA_STACKED,
    'area_stacked_100': XL_CHART_TYPE.AREA_STACKED_100,
    'bar_clustered': XL_CHART_TYPE.BAR_CLUSTERED,
    'bar_stacked': XL_CHART_TYPE.BAR_STACKED,
    'bar_stacked_100': XL_CHART_TYPE.BAR_STACKED_100,
    'column_clustered': XL_CHART_TYPE.COLUMN_CLUSTERED,
    'column_stacked': XL_CHART_TYPE.COLUMN_STACKED,
    'column_stacked_100': XL_CHART_TYPE.COLUMN_STACKED_100,
    'doughnut': XL_CHART_TYPE.DOUGHNUT,
    'doughnut_exploded': XL_CHART_TYPE.DOUGHNUT_EXPLODED,
    'line': XL_CHART_TYPE.LINE,
    'line_markers': XL_CHART_TYPE.LINE_MARKERS,
    'line_markers_stacked': XL_CHART_TYPE.LINE_MARKERS_STACKED,
    'line_markers_stacked_100': XL_CHART_TYPE.LINE_MARKERS_STACKED_100,
    'line_stacked': XL_CHART_TYPE.LINE_STACKED,
    'line_stacked_100': XL_CHART_TYPE.LINE_STACKED_100,
    'pie': XL_CHART_TYPE.PIE,
    'pie_exploded': XL_CHART_TYPE.PIE_EXPLODED,
    'radar': XL_CHART_TYPE.RADAR,
    'radar_filled': XL_CHART_TYPE.RADAR_FILLED,
    'radar_markers': XL_CHART_TYPE.RADAR_MARKERS,
    'xy_scatter': XL_CHART_TYPE.XY_SCATTER,
    'xy_scatter_lines': XL_CHART_TYPE.XY_SCATTER_LINES,
    'xy_scatter_lines_no_markers': XL_CHART_TYPE.XY_SCATTER_LINES_NO_MARKERS,
    'xy_scatter_smooth': XL_CHART_TYPE.XY_SCATTER_SMOOTH,
    'xy_scatter_smooth_no_markers': XL_CHART_TYPE.XY_SCATTER_SMOOTH_NO_MARKERS,
    'bubble': XL_CHART_TYPE.BUBBLE,
    'bubble_three_d_effect': XL_CHART_TYPE.BUBBLE_THREE_D_EFFECT
    }
