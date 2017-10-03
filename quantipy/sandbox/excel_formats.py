"""
Default cell formats
"""

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


class _Format(dict):

    __attributes__ = ('font_name', 'font_size', 'font_color', 'bold', 'italic',
                      'font_script', 'num_format', 'align', 'valign',
                      'text_v_align', 'text_h_align', 'text_wrap', 'bg_color',
                      'border', 'bottom', 'top', 'left', 'right',
                      'border_color', 'bottom_color', 'top_color',
                      'left_color', 'right_color')

    __slots__ = __attributes__

    def __init__(self, **kwargs):
        for name in self.__attributes__:
            if name in kwargs.keys():
                self[name] = kwargs[name]
        for name in kwargs:
            if name not in self.__attributes__:
                raise Exception(name)

    def __hash__(self):
        return hash(repr(self))

class _ExcelFormats(object):

    __slots__ = ('num_format_pct', 'bold_nets', 'border_style_ext',
                 'format_label_row', 'font_name_nets', 'num_format_n',
                 'bg_color_label', 'bold_ubase', 'border_color',
                 'bold_ubase_text','font_size_nets', 'dummy_tests',
                 'italicise_nets', 'img_insert_x', 'img_insert_y',
                 'font_color_descriptives', 'start_column', 'bold_base_text',
                 'frequency_0_rep', 'img_y_offset', 'img_x_offset',
                 'y_header_height', 'row_wrap_trigger',
                 'font_size_descriptives', 'row_height',
                 'font_color_base_text', 'border_color_descriptives_top',
                 'num_format_descriptives', 'df_nan_rep',
                 'font_name_descriptives', 'font_name_str', 'font_color_str',
                 'column_width_str', 'img_url', 'test_seperator',
                 'border_color_nets_top', 'font_size_tests', 'bold_base',
                 'img_size', 'bold', 'font_color_tests', 'font_color_label',
                 'bold_descriptives', 'no_logo', 'font_size',
                 'font_color_ubase_text', 'num_format_default', 'bg_color',
                 'bold_tests', 'img_name', 'y_row_height', 'font_color_base',
                 'font_name', 'start_row', 'descriptives_0_rep',
                 'font_color_ubase', 'font_color_nets', 'bold_x', 'bold_y',
                 'font_size_str', 'arrow_color_high', 'arrow_color_low',
                 'display_test_level', 'border_style_int', 'bg_color_nets',
                 'font_color', 'font_name_tests', 'font_super_tests',
                 'bg_color_tests')

    _ATTRIBUTES =  {'start_row': 8,
                    'start_column': 2,
                    'row_height': 12.75,
                    'row_wrap_trigger': 44,
                    'y_header_height': 33.75,
                    'y_row_height': 50,
                    'no_logo': False,
                    'img_name': 'qplogo_invert_lg.png',
                    'img_url': 'logo/qplogo_invert_lg.png',
                    'img_size': [130, 130],
                    'img_insert_x': 0,
                    'img_insert_y': 0,
                    'img_x_offset': 0,
                    'img_y_offset': 0,
                    'frequency_0_rep': '-',
                    'descriptives_0_rep': 0.00,
                    'df_nan_rep': '__NA__',
                    'test_seperator': '.',
                    'font_name': 'Arial',
                    'font_size': 9,
                    'font_color': '#000000',
                    'font_color_label': '#000000',
                    'bold': False,
                    'bold_y': False,
                    'bold_x': False,
                    'font_color_ubase': '#000000',
                    'font_color_ubase_text': '#000000',
                    'font_color_base': '#000000',
                    'font_color_base_text': '#000000',
                    'bold_ubase_text': False,
                    'bold_ubase': False,
                    'bold_base_text': False,
                    'bold_base': False,
                    'font_name_nets': 'Arial',
                    'font_size_nets': 9,
                    'font_color_nets': '#000000',
                    'bold_nets': False,
                    'italicise_nets': False,
                    'font_name_descriptives': 'Arial',
                    'font_size_descriptives': 9,
                    'font_color_descriptives': '#000000',
                    'bold_descriptives': False,
                    'font_name_tests': 'Arial',
                    'font_size_tests': 9,
                    'font_color_tests': '#000000',
                    'bold_tests': False,
                    'font_super_tests': True,
                    'display_test_level': True,
                    'dummy_tests': False,
                    'arrow_color_high': '#2EB08C',
                    'arrow_color_low': '#FC8EAC',
                    'font_name_str': 'Arial',
                    'font_size_str': 9,
                    'font_color_str': '#000000',
                    'column_width_str': 10,
                    'format_label_row': False,
                    'border_color': '#D9D9D9',
                    'border_color_nets_top': '#D9D9D9',
                    'border_color_descriptives_top': '#D9D9D9',
                    'border_style_ext': 5,
                    'border_style_int': 1,
                    'bg_color': '#F2F2F2',
                    'bg_color_nets': '#FFFFFF',
                    'bg_color_tests': '#F2F2F2',
                    'bg_color_label': '#FFFFFF',
                    'num_format_n': '0',
                    'num_format_pct': '0%',
                    'num_format_descriptives': '0.00',
                    'num_format_default': '0.00'}

    def __init__(self, **kwargs):
        for name in self._ATTRIBUTES.keys():
            value_or_default = kwargs.get(name, self._ATTRIBUTES[name])
            setattr(self, name, value_or_default)

class ExcelFormats(_ExcelFormats):
    
    __slots__ = ('_lazy_y',  
                 '_lazy_x_left_bold', 
                 '_lazy_x_right_base',
                 '_lazy_x_right_ubase', 
                 '_lazy_x_right_tests', 
                 '_lazy_x_right_descriptives', 
                 '_lazy_x_right_nets', 
                 '_lazy_cell_details', 
                 '_lazy_x_right_italic', 
                 '_lazy_x_right_bold', 
                 '_lazy_x_right')

    def __init__(self, **kwargs):
        super(ExcelFormats, self).__init__(**kwargs)

    @lazy_property
    def y(self):
        format_ = _Format(**{'font_name': self.font_name,
                            'font_size': self.font_size,
                            'bold': self.bold_y,
                            'text_v_align': 2,
                            'text_h_align': 2,
                            'text_wrap': True,
                            'left': self.border_style_ext,
                            'top': self.border_style_ext,
                            'right': self.border_style_ext,
                            'bottom': self.border_style_ext,
                            'left_color': self.border_color,
                            'top_color': self.border_color,
                            'right_color': self.border_color,
                            'bottom_color': self.border_color})
        return format_

    @lazy_property
    def tests(self):
        format_ = _Format(**{'bg_color': '#FFFFFF',
                            'font_name': self.font_name_tests,
                            'font_size': self.font_size_tests,
                            'font_color': self.font_color_tests,
                            'bold': self.bold_y,
                            'text_v_align': 2,
                            'text_h_align': 2,
                            'text_wrap': True,
                            'left': self.border_style_ext,
                            'top': self.border_style_ext,
                            'right': self.border_style_ext,
                            'bottom': self.border_style_ext,
                            'left_color': self.border_color,
                            'top_color': self.border_color,
                            'right_color': self.border_color,
                            'bottom_color': self.border_color})
        return format_

    @lazy_property
    def x_left_bold(self):
        format_ = _Format(**{'font_name': self.font_name,
                            'font_size': self.font_size,
                            'font_color': self.font_color_label,
                            'bold': self.bold_x,
                            'bg_color': self.bg_color_label,
                            'text_v_align': 2,
                            'text_h_align': 1,
                            'text_wrap': True})
        return format_

    @lazy_property
    def x_right(self):
        format_ = _Format(**{'font_name': self.font_name,
                            'font_size': self.font_size,
                            'text_v_align': 2,
                            'text_h_align': 3,
                            'text_wrap': True})
        return format_

    @lazy_property
    def x_right_bold(self):
        format_ = _Format(**{'font_name': self.font_name,
                            'font_size': self.font_size,
                            'text_v_align': 2,
                            'text_h_align': 3,
                            'text_wrap': True,
                            'bold': True})
        return format_

    @lazy_property
    def x_right_italic(self):
        format_ = _Format(**{'font_name': self.font_name,
                            'font_size': self.font_size,
                            'text_v_align': 2,
                            'text_h_align': 3,
                            'text_wrap': True,
                            'italic': True})
        return format_

    @lazy_property
    def cell_details(self):
        format_ = _Format(**{'font_name': self.font_name_tests,
                            'font_size': self.font_size,
                            'text_v_align': 2,
                            'text_h_align': 1})
        return format_

    @lazy_property
    def x_right_nets(self):
        format_ = _Format(**{'bold': self.bold_nets,
                            'bg_color': self.bg_color_nets,
                            'italic': self.italicise_nets,
                            'font_name': self.font_name_nets,
                            'font_size': self.font_size_nets,
                            'font_color': self.font_color_nets,
                            'text_v_align': 2,
                            'text_h_align': 3,
                            'text_wrap': True})
        return format_

    @lazy_property
    def x_right_descriptives(self):
        format_ = _Format(**{'font_name': self.font_name_descriptives,
                            'font_size': self.font_size_descriptives,
                            'font_color': self.font_color_descriptives,
                            'bold': self.bold_descriptives,
                            'text_v_align': 2,
                            'text_h_align': 3,
                            'text_wrap': True})
        return format_

    @lazy_property
    def x_right_tests(self):
        format_ = _Format(**{'num_format': '0.00',
                            'font_name': self.font_name_tests,
                            'font_size': self.font_size_tests,
                            'font_color': self.font_color_tests,
                            'font_script': self.font_super_tests,
                            'bold': self.bold_tests,
                            'text_v_align': 2,
                            'text_h_align': 3,
                            'text_wrap': True})
        return format_

    @lazy_property
    def x_right_ubase(self):
        format_ = _Format(**{'font_name': self.font_name,
                            'font_size': self.font_size,
                            'font_color': self.font_color_ubase_text,
                            'bold': self.bold_ubase_text,
                            'text_v_align': 2,
                            'text_h_align': 3,
                            'text_wrap': True})
        return format_

    @lazy_property
    def x_right_base(self):
        format_ = _Format(**{'font_name': self.font_name,
                            'font_size': self.font_size,
                            'font_color': self.font_color_base_text,
                            'bold': self.bold_base_text,
                            'text_v_align': 2,
                            'text_h_align': 3,
                            'text_wrap': True})
        return format_

