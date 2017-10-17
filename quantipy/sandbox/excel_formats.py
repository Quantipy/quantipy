"""
Excel cell formats
"""

from excel_formats_constants import ATTRIBUTES, DEFAULT_ATTRIBUTES


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

    __attributes__ = ATTRIBUTES
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

    __default_attributes__ = DEFAULT_ATTRIBUTES.keys()
    __slots__ = __default_attributes__

    def __init__(self, **kwargs):
        for name in self.__default_attributes__:
            value_or_default = kwargs.get(name, DEFAULT_ATTRIBUTES[name]) 
            setattr(self, name, value_or_default)


class ExcelFormats(_ExcelFormats):

    __slots__ = ('_lazy_y',  
                 '_lazy_template',
                 '_lazy_cell_details', 
                 '_lazy_x_left_bold', 
                 '_lazy_x_right_base',
                 '_lazy_x_right_ubase', 
                 '_lazy_x_right_test', 
                 '_lazy_x_right_stat', 
                 '_lazy_x_right_net', 
                 '_lazy_x_right_italic', 
                 '_lazy_x_right_bold', 
                 '_lazy_x_right',
                 '_lazy_x_right_count',
                 '_lazy_x_right_pct',
                 )

    def __init__(self, **kwargs):
        super(ExcelFormats, self).__init__(**kwargs)

    @lazy_property
    def template(self):
        return dict([(a, getattr(self, a)) for a in _Format.__attributes__])
            
    @lazy_property
    def y(self):
        format_ = self.template.copy()
        
        format_.update(dict(bold=self.bold_y,
                            left=self.border_style_ext,
                            top=self.border_style_ext,
                            right=self.border_style_ext,
                            bottom=self.border_style_ext))

        return _Format(**format_)

    @lazy_property
    def test(self):
        format_ = self.template.copy()

        format_.update(dict(font_name=self.font_name_test,
                            font_size=self.font_size_test,
                            font_color=self.font_color_test,
                            bold=self.bold_y,
                            left=self.border_style_ext,
                            top=self.border_style_ext,
                            right=self.border_style_ext,
                            bottom=self.border_style_ext))

        return _Format(**format_)

    @lazy_property
    def x_left_bold(self):
        format_ = self.template.copy()

        format_.update(dict(font_color=self.font_color_label,
                            bold=self.bold_x,
                            text_h_align=1,
                            bg_color=self.bg_color_label))

        return _Format(**format_)

    @lazy_property
    def x_right(self):
        format_ = self.template.copy()

        format_.update(dict(text_h_align=3))
        
        return _Format(**format_)

    @lazy_property
    def x_right_count(self):
        return self.x_right

    @lazy_property
    def x_right_pct(self):
        return self.x_right

    @lazy_property
    def x_right_bold(self):
        format_ = self.template.copy()

        format_.update(dict(bold=True, text_h_align=3))
                            
        return _Format(**format_)

    @lazy_property
    def x_right_italic(self):
        format_ = self.template.copy()

        format_.update(dict(italic=True, text_h_align=3))

        return _Format(**format_)

    @lazy_property
    def cell_details(self):
        format_ = self.template.copy()

        format_.update(dict(font_name=self.font_name_test, text_h_align=1))

        return _Format(**format_)

    @lazy_property
    def x_right_net(self):
        format_ = self.template.copy()

        format_.update(dict(font_name=self.font_name_net,
                            font_size=self.font_size_net,
                            font_color=self.font_color_net,
                            bold=self.bold_net,
                            italic=self.italicise_net,
                            text_h_align=3,
                            bg_color=self.bg_color_net))
        
        return _Format(**format_)

    @lazy_property
    def x_right_stat(self):
        format_ = self.template.copy()

        format_.update(dict(font_name=self.font_name_stat,
                            font_size=self.font_size_stat,
                            font_color=self.font_color_stat,
                            bold=self.bold_stat,
                            text_h_align=3))

        return _Format(**format_)

    @lazy_property
    def x_right_test(self):
        format_ = self.template.copy()

        format_.update(dict(font_name=self.font_name_test,
                            font_size=self.font_size_test,
                            font_color=self.font_color_test,
                            font_script=self.font_super_test,
                            bold=self.bold_test,
                            text_h_align=3,
                            num_format='0.00'))

        return _Format(**format_)

    @lazy_property
    def x_right_ubase(self):
        format_ = self.template.copy()

        format_.update(dict(font_color=self.font_color_ubase_text,
                            bold=self.bold_ubase_text,
                            text_h_align=3))

        return _Format(**format_)

    @lazy_property
    def x_right_base(self):
        format_ = self.template.copy()

        format_.update(dict(font_color=self.font_color_base_text,
                            bold=self.bold_base_text,
                            text_h_align=3))

        return _Format(**format_)

    def _left(self):
        return dict(left=self.border_style_ext)

    def _right(self):
        return dict(left=self.border_style_int,
                    right=self.border_style_ext)

    def _top(self):
        return dict(top=self.border_style_ext)

    def _bottom(self):
        return dict(bottom=self.border_style_etxt)

    def _interior(self):
        return dict(left=self.border_style_int)

    def _base(self):
        return dict(font_color=self.font_color_base,
                    bold=self.bold_base)

    def _ubase(self):
        return dict(font_color=self.font_color_ubase,
                    bold=self.bold_ubase)

    def _count(self):
        return dict(num_format=self.num_format_count) 

    def _stat(self):
        return dict(font_name=self.font_name_stat,
                    font_size=self.font_size_stat,
                    font_color=self.font_color_stat,
                    bold=self.bold_stat,
                    num_format=self.num_format_stat)

    def _test(self):
        return dict(font_name=self.font_name_test,
                    font_size=self.font_size_test,
                    font_color=self.font_color_test,
                    bold=self.bold_test,
                    font_script=self.font_super_test)

    def get(self, name):
        try:
            return getattr(self, name)
        except AttributeError:
            return self._get(name)

    def _get(self, name):
        format_ = self.template.copy()

        print name,
        for part in name.split('_'):
            updates = getattr(self, '_' + part)()
            if ('left' in name) and (part == 'right'):
                updates.pop('left')
            format_.update(updates)
            print format_

        return _Format(**format_)

