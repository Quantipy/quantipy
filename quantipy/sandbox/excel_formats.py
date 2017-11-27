"""
Excel cell formats
"""

import re
from excel_formats_constants import _ATTRIBUTES, _DEFAULT_ATTRIBUTES, _VIEWS_GROUPS
from quantipy.core.tools.qp_decorators import lazy_property
try:
    from functools import lru_cache
except ImportError:
    from functools32 import lru_cache


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

    __default_attributes__ = tuple(_DEFAULT_ATTRIBUTES.keys())
    __slots__ = __default_attributes__ + ('_view_or_group', )

    def __init__(self, views_groups, **kwargs):
        for name in self.__default_attributes__:
            value_or_default = kwargs.get(name, self._view_or_group(name, views_groups, kwargs))
            setattr(self, name, value_or_default)
    
    def _view_or_group(self, name, views_groups, kwargs):
        for view, group in views_groups.iteritems():
            pattern = r'(\w+)(%s)(_text|$)' % view
            match = re.match(pattern, name)
            if match:
                groups = match.groups()
                attr = groups[0] + group + groups[2]
                if attr in kwargs:
                    return kwargs[attr]
        return _DEFAULT_ATTRIBUTES[name]


class ExcelFormats(_ExcelFormats):

    __slots__ = ('_lazy__base',
                 '_lazy__bottom',
                 '_lazy__cell_details',
                 '_lazy__interior',
                 '_lazy__left',
                 '_lazy__meanstest',
                 '_lazy__net_propstest',
                 '_lazy__propstest',
                 '_lazy__right',
                 '_lazy__top',
                 '_lazy__ubase',
                 '_lazy__y',
                 '_lazy_slots',
                 '_view_border',
                 '_format_builder',
                 '_method',
                 '_template')

    def __init__(self, views_groups, **kwargs):
        super(ExcelFormats, self).__init__(views_groups, **kwargs)

    def __getitem__(self, name):
        return self._format_builder(name)

    @lru_cache()
    def _format_builder(self, name):
        format_ = self._template

        parts = name.split('_no_')
        methods, no = parts[0].split('^'), parts[1:]

        for item in methods:
            item = item.split('.')
            if len(item) > 1:
                method, alt = item
            else:
                method, alt = item[0], None

            if method in ('bottom', 'interior', 'left', 'right', 'top'):
                updates = getattr(self, '_' + method)
                if ('left' in name) and (method == 'right'):
                    updates = {k: v for k, v in updates.iteritems()
                               if k != 'left'}
                format_.update(updates)
            else:
                try:
                    format_.update(self._method(method))
                except AttributeError:
                    pass
                
                if '_lazy__' + method in self.slots:
                    format_.update(getattr(self, '_' + method))

                if '_' + method in self.slots:
                    format_.update(getattr(self, '_' + method)(methods[-1], alt))

            if 'num_format_' + method in self.slots:
                format_['num_format'] = getattr(self, 'num_format_' + method)

        for attr in no:
            try:
                format_.pop(attr)
            except KeyError:
                pass

        return _Format(**format_)

    @lazy_property
    def slots(self):
        return self.__slots__ + tuple(super(ExcelFormats, self).__slots__)

    @lru_cache()
    def _method(self, method):
        return dict(bold=getattr(self, 'bold_' + method),
                    bg_color=getattr(self, 'bg_color_' + method),
                    font_color=getattr(self, 'font_color_' + method),
                    font_name=getattr(self, 'font_name_' + method),
                    font_size=getattr(self, 'font_size_' + method),
                    italic=getattr(self, 'italic_' + method),
                    text_v_align=getattr(self, 'text_v_align_' + method),
                    text_h_align=getattr(self, 'text_h_align_' + method),
                    text_wrap=self.text_wrap)

    @property
    def _template(self):
        return dict([(a, getattr(self, a)) for a in _Format.__attributes__])

    @lazy_property
    def _cell_details(self):
        format_ = self._template

        format_.update(dict(font_name=self.font_name_propstest, text_h_align=1))

        return _Format(**format_)

    @lazy_property
    def _y(self):
        format_ = self._template

        format_.update(dict(left=self.border_style_ext,
                            top=self.border_style_ext,
                            right=self.border_style_ext,
                            bottom=self.border_style_ext,
                            ))
        format_.update(self._method('y'))

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
    def _propstest(self):
        return dict(font_script=self.font_script_propstest)

    @lazy_property
    def _net_propstest(self):
        return dict(font_script=self.font_script_net_propstest)

    @lazy_property
    def _meanstest(self):
        return dict(font_script=self.font_script_meanstest)

    @lru_cache()
    def _view_border(self, name, alt):
        border = getattr(self, 'view_border_' + name)
        if border or alt is None:
            return dict(top=border)
        return dict(top=getattr(self, 'view_border_' + alt))

