"""
Excel cell formats
"""

from quantipy.core.tools.qp_decorators import lazy_property

from excel_formats_constants import _ATTRIBUTES, _DEFAULT_ATTRIBUTES


class _Format(dict):

    __attributes__ = _ATTRIBUTES
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

    __default_attributes__ = _DEFAULT_ATTRIBUTES.keys()
    __slots__ = __default_attributes__

    def __init__(self, **kwargs):
        for name in self.__default_attributes__:
            value_or_default = kwargs.get(name, _DEFAULT_ATTRIBUTES[name]) 
            setattr(self, name, value_or_default)


class ExcelFormats(_ExcelFormats):

    __slots__ = ('_lazy__background',
                 '_lazy__base',
                 '_lazy__bottom',
                 '_lazy_cell_details', 
                 '_lazy__count',
                 '_lazy__interior',
                 '_lazy__left',
                 '_lazy__net',
                 '_lazy__pct',
                 '_lazy__right',
                 '_lazy__stat',
                 '_lazy__stattest',
                 '_lazy__sum',
                 '_lazy_template',
                 '_lazy__test',
                 '_lazy__top',
                 '_lazy__ubase',
                 '_lazy_x_label',
                 '_lazy_x_right',
                 '_lazy_x_base',
                 '_lazy_x_bold',
                 '_lazy_x_italic',
                 '_lazy_x_net',
                 '_lazy_x_stat',
                 '_lazy_x_stattest',
                 '_lazy_x_sum',
                 '_lazy_x_test',
                 '_lazy_x_ubase',
                 '_lazy_y',
                 '_format_builder'
                 )

    def __init__(self, **kwargs):
        super(ExcelFormats, self).__init__(**kwargs)

    def __getattr__(self, name):
        if name.startswith('x_') and name not in dir(self):
            return self.x_right
        return self.__getattribute__(name)

    def __getitem__(self, name):
        try:
            return getattr(self, name)
        except AttributeError, e:
            
            print "AttributeError: %s" % e

            format_ = self.template
            
            parts = name.split('_no_')
            name, no = parts[0], parts[1:]

            for part in name.split('_'):
                updates = getattr(self, '_' + part)
                if ('left' in name) and (part == 'right'):
                    updates = {k: v for k, v in updates.iteritems() 
                               if k != 'left'}
                format_.update(updates)

            for attr in no:
                try:
                    format_.pop(attr)
                except KeyError:
                    pass

            return _Format(**format_)

    def _format_builder(self, method):
        attrs =  ('bold', 'bg_color', 'font_color', 'font_name', 
                  'font_size', 'italic', 'text_v_align', 'text_h_align')

        def _format_attributes(self):
            return dict([(attr, getattr(self, attr + '_' + method))
                         for attr in attrs])

        return _format_attributes(self)        

    @property
    def template(self):
        return dict([(a, getattr(self, a)) for a in _Format.__attributes__])
            
    @lazy_property
    def cell_details(self):
        format_ = self.template

        format_.update(dict(font_name=self.font_name_test, text_h_align=1))

        return _Format(**format_)

    @lazy_property
    def y(self):
        format_ = self.template
        
        format_.update(dict(left=self.border_style_ext,
                            top=self.border_style_ext,
                            right=self.border_style_ext,
                            bottom=self.border_style_ext,
                            ))
        format_.update(self._format_builder('y'))

        return _Format(**format_)

    @lazy_property
    def test(self):
        format_ = self.template

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
    def x_label(self):
        format_ = self.template

        format_.update(self._format_builder('label'))

        return _Format(**format_)

    @lazy_property
    def x_right(self):
        format_ = self.template

        format_.update(dict(text_h_align=3))
        
        return _Format(**format_)

    @lazy_property
    def x_bold(self):
        format_ = self.template

        format_.update(dict(bold=True, text_h_align=3))
                            
        return _Format(**format_)

    @lazy_property
    def x_italic(self):
        format_ = self.template

        format_.update(dict(italic=True, text_h_align=3))

        return _Format(**format_)

    @lazy_property
    def x_net(self):
        format_ = self.template

        format_.update(self._format_builder('net_text'))
                            
        return _Format(**format_)

    @lazy_property
    def x_stat(self):
        format_ = self.template

        format_.update(self._format_builder('stat_text'))
                            
        return _Format(**format_)

    @lazy_property
    def x_stattest(self):
        format_ = self.template

        format_.update(self._format_builder('stattest_text'))
                            
        return _Format(**format_)

    @lazy_property
    def x_test(self):
        format_ = self.template

        format_.update(dict(font_name=self.font_name_test,
                            font_size=self.font_size_test,
                            font_color=self.font_color_test,
                            font_script=self.font_super_test,
                            bold=self.bold_test,
                            text_h_align=3,
                            num_format='0.00'))

        return _Format(**format_)

    @lazy_property
    def x_base(self):
        format_ = self.template

        format_.update(self._format_builder('base_text'))
                            
        return _Format(**format_)

    @lazy_property
    def x_ubase(self):
        format_ = self.template

        format_.update(self._format_builder('ubase_text'))
                            
        return _Format(**format_)

    @lazy_property
    def x_sum(self):
        format_ = self.template

        format_.update(self._format_builder('sum_text'))
                            
        return _Format(**format_)

    @lazy_property
    def _left(self):
        return dict(left=self.border_style_ext)

    @lazy_property
    def _right(self):
        return dict(left=self.border_style_int,
                    right=self.border_style_ext)

    @lazy_property
    def _top(self):
        return dict(top=self.border_style_ext)

    @lazy_property
    def _bottom(self):
        return dict(bottom=self.border_style_ext)

    @lazy_property
    def _interior(self):
        return dict(left=self.border_style_int)

    @lazy_property
    def _base(self):
        format_ = self._format_builder('base')
        format_.update(dict(bottom=self.border_style_int))
        return format_

    @lazy_property
    def _ubase(self):
        format_ = self._format_builder('ubase')
        format_.update(dict(bottom=self.border_style_int))
        return format_

    @lazy_property
    def _count(self):
        return dict(num_format=self.num_format_count,
                    bg_color=self.bg_color_default) 

    @lazy_property
    def _pct(self):
        return dict(num_format=self.num_format_pct, 
                    bg_color=self.bg_color_default) 

    @lazy_property
    def _net(self):
        return self._format_builder('net')

    @lazy_property
    def _stat(self):
        return self._format_builder('stat')

    @lazy_property
    def _stattest(self):
        return self._format_builder('stattest')

    @lazy_property
    def _test(self):
        return dict(font_name=self.font_name_test,
                    font_size=self.font_size_test,
                    font_color=self.font_color_test,
                    bold=self.bold_test,
                    font_script=self.font_super_test,
                    bg_color=self.bg_color_test)

    @lazy_property
    def _sum(self):
        return self._format_builder('sum')

