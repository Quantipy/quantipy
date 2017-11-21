"""
Excel cell formats
"""

from excel_formats_constants import _ATTRIBUTES, _DEFAULT_ATTRIBUTES
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
                 '_lazy__counts',
                 '_lazy__interior',
                 '_lazy__left',
                 '_lazy__net',
                 '_lazy__netpropstest',
                 '_lazy__c_pct',
                 '_lazy__right',
                 '_lazy__stat',
                 '_lazy__meanstest',
                 '_lazy__sum',
                 '_lazy_template',
                 '_lazy__propstest',
                 '_lazy__top',
                 '_lazy__ubase',
                 '_lazy_x_label',
                 '_lazy_x_right',
                 '_lazy_x_base',
                 '_lazy_x_bold',
                 '_lazy_x_counts',
                 '_lazy_x_italic',
                 '_lazy_x_net',
                 '_lazy_x_netpropstest',
                 '_lazy_x_c_pct',
                 '_lazy_x_stat',
                 '_lazy_x_meanstest',
                 '_lazy_x_sum',
                 '_lazy_x_propstest',
                 '_lazy_x_ubase',
                 '_lazy__y',
                 '_format_builder'
                 '_method'
                 )

    def __init__(self, **kwargs):
        super(ExcelFormats, self).__init__(**kwargs)

    def __getattr__(self, name):
    #     # if name.startswith('x_') and name not in dir(self):
    #     #     return self.x_right
        return self.__getattribute__(name)

    def __getitem__(self, name):
        return self._format_builder(name)

    @lru_cache()
    def _format_builder(self, name):
        format_ = self.template
        
        parts = name.split('_no_')
        name, no = parts[0], parts[1:]
        
        for method in name.split('^'):
            if method in ('bottom', 'interior', 'left', 'right', 'top'):
                updates = getattr(self, '_' + method)
                if ('left' in name) and (method == 'right'):
                    updates = {k: v for k, v in updates.iteritems() 
                               if k != 'left'}
                format_.update(updates)
            else:
                format_.update(self._method(method))
                try:
                    format_.update(getattr(self, '_' + method))
                except AttributeError:
                    pass
            
            try:
                format_['num_format'] = getattr(self, 'num_format_' + method)
            except  AttributeError:
                pass

        for attr in no:
            try:
                format_.pop(attr)
            except KeyError:
                pass
             
        return _Format(**format_)

    @lru_cache()
    def _method(self, method):
        attrs =  ('bold', 'bg_color', 'font_color', 'font_name', 
                  'font_size', 'italic', 'text_v_align', 'text_h_align')

        def _attributes(self):
            return dict([(attr, getattr(self, attr + '_' + method, self.template[attr]))
                         for attr in attrs])

        return _attributes(self)        

    @property
    def template(self):
        return dict([(a, getattr(self, a)) for a in _Format.__attributes__])
            
    @lazy_property
    def cell_details(self):
        format_ = self.template

        format_.update(dict(font_name=self.font_name_propstest, text_h_align=1))

        return _Format(**format_)

    @lazy_property
    def _y(self):
        format_ = self.template
        
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
    def _cbase(self):
        return dict(bottom=self.border_style_int)

    @lazy_property
    def _u_cbase(self):
        return dict(bottom=self.border_style_int)

