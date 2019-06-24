"""
Excel cell formats
"""

import re
from .excel_formats_constants import (_ATTRIBUTES,
                                     _DEFAULT_ATTRIBUTES,
                                     _VIEWS_GROUPS)
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
            if name in list(kwargs.keys()):
                self[name] = kwargs[name]
        for name in kwargs:
            if name not in self.__attributes__:
                raise Exception(name)

    def __hash__(self):
        return hash(repr(self))


class _ExcelFormats(object):

    __default_attributes__ = tuple(_DEFAULT_ATTRIBUTES.keys())
    __slots__ = __default_attributes__ + ('default_attributes',)

    def __init__(self, views_groups, **kwargs):
        self.default_attributes = dict(_DEFAULT_ATTRIBUTES)

        # update default for globals, e.g. "font_size"
        for x in kwargs:
            if x in _ATTRIBUTES:
                for y in self.default_attributes:
                    if y.startswith(x):
                        self.default_attributes[y] = kwargs[x]

        # set attributes
        for name in self.__default_attributes__:
            view_or_group = self._view_or_group(name, _VIEWS_GROUPS,
                                                views_groups, kwargs)
            value_or_default = kwargs.get(name, view_or_group)
            setattr(self, name, value_or_default)

    def _view_or_group(self, name, implicit, explicit, kwargs):
        if self._extract_from(name, explicit, kwargs) != '#':
            return self._extract_from(name, explicit, kwargs)

        if self._extract_from(name, implicit, kwargs) != '#':
            return self._extract_from(name, implicit, kwargs)

        return self.default_attributes[name]

    @staticmethod
    def _extract_from(name, source, kwargs):
        for view, group in source.items():
            pattern = r'(\w+)(%s)(_text|$)' % view
            match = re.match(pattern, name)
            if match:
                groups = match.groups()
                attr = groups[0] + group + groups[2]
                if attr in kwargs:
                    return kwargs[attr]
        return '#'

class ExcelFormats(_ExcelFormats):

    def __init__(self, views_groups, **kwargs):
        super(ExcelFormats, self).__init__(views_groups, **kwargs)

    def __getitem__(self, name):
        value = getattr(self, name, None)
        if value is None:
            value = self._format_builder(name)
            setattr(self, name, value)
        return value

    def _format_builder(self, name):
        format_ = self._template

        if '_array_style_' in name:
            name = name.split('_array_style_')[0]

        parts = name.split('_no_')
        methods, no = parts[0].split('^'), parts[1:]

        for item in methods:
            item = item.split('.')
            if len(item) > 1:
                method, alt = item
            else:
                method, alt = item[0], None

            if method in ('bottom', 'interior', 'left', 'right', 'top'):
                updates = getattr(self, '_' + method)()
                if ('left' in name) and (method == 'right'):
                    updates = {k: v for k, v in updates.items()
                               if k != 'left'}
                format_.update(updates)
            else:
                try:
                    format_.update(self._method(method))
                except AttributeError:
                    pass

                if '_' + method in dir(self):
                    try:
                        updates = getattr(self, '_' + method)(methods[-1], alt)
                    except TypeError:
                        updates = getattr(self, '_' + method)()
                    format_.update(updates)

            if 'num_format_' + method in dir(self):
                format_['num_format'] = getattr(self, 'num_format_' + method)

        for attr in no:
            try:
                format_.pop(attr)
            except KeyError:
                pass

        return _Format(**format_)

    @property
    def _template(self):
        return dict([(a, getattr(self, a)) for a in _Format.__attributes__])

    def _method(self, method):
        result = dict()
        for name in ('bold', 'bg_color', 'font_color', 'font_name', 'font_size',
                     'italic', 'text_v_align', 'text_h_align', 'text_wrap'):
            attr = getattr(self, name + '_' + method, None)
            if attr is not None:
                result[name] = attr
        return result

    def _view_border(self, name, alt):
        border = getattr(self, 'view_border_' + name)
        if border or alt is None:
            return dict(top=border)
        return dict(top=getattr(self, 'view_border_' + alt))

    @lazy_property
    def _cell_details(self):
        format_ = self._template

        format_.update(dict(text_wrap=False, text_h_align=1))

        return _Format(**format_)

    @lazy_property
    def _y(self):
        format_ = self._template

        format_.update(dict(left=self.border_style_ext,
                            top=self.border_style_ext,
                            right=self.border_style_ext,
                            bottom=self.border_style_ext))
        format_.update(self._method('y'))

        return _Format(**format_)

    def _header_left(self):
        format_ = self._template

        format_.update(self._method('header_left'))

        return _Format(**format_)

    def _header_center(self):
        format_ = self._template

        format_.update(self._method('header_center'))

        return _Format(**format_)

    def _header_title(self):
        format_ = self._template

        format_.update(self._method('header_title'))

        return _Format(**format_)

    def _notes(self):
        format_ = self._template

        format_.update(self._method('notes'))

        return _Format(**format_)

    @lazy_property
    def _data_header(self):
        format_ = self._template

        format_.update(dict(left=self.border_style_ext,
                            top=self.border_style_ext,
                            right=self.border_style_ext,
                            bottom=self.border_style_ext))
        format_.update(self._method('data_header'))

        return _Format(**format_)

    def _data(self):
        return dict(text_wrap=False)

    def _left(self):
        return dict(left=self.border_style_ext)

    def _right(self):
        return dict(left=self.border_style_int,
                    right=self.border_style_ext)

    def _top(self):
        return dict(top=self.border_style_ext)

    def _bottom(self):
        return dict(bottom=self.border_style_ext)

    def _interior(self):
        return dict(left=self.border_style_int)

    def _propstest(self):
        return dict(font_script=self.font_script_propstest)

    def _net_propstest(self):
        return dict(font_script=self.font_script_net_propstest)

    def _block_calc_net_propstest(self):
        return dict(font_script=self.font_script_block_calc_net_propstest)

    def _block_calc_propstest(self):
        return dict(font_script=self.font_script_block_calc_propstest)

    def _block_expanded_propstest(self):
        return dict(font_script=self.font_script_block_expanded_propstest)

    def _block_net_propstest(self):
        return dict(font_script=self.font_script_block_net_propstest)

    def _block_normal_propstest(self):
        return dict(font_script=self.font_script_block_normal_propstest)

    def _meanstest(self):
        return dict(font_script=self.font_script_meanstest)
