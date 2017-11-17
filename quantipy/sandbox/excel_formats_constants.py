
_ATTRIBUTES = ('bg_color',
               'bold',
               'border_color',
               'bottom',
               'bottom_color',
               'font_color',
			   'font_name',
               'font_size',
               'font_script',
               'italic',
               'left',
               'left_color',
               'num_format',
               'right',
               'right_color',
               'text_v_align',
               'text_h_align',
               'text_wrap',
               'top',
               'top_color')

_DEFAULTS = dict(bg_color='#FFFFFF',
                 bold=False,
                 border=None,
                 border_color='#D9D9D9',
                 border_style_ext=5,
                 border_style_int=1,
                 font_color='#000000',
                 font_name='Arial',
                 font_size=9,
                 italic=False,
                 num_format='0',
                 num_format_count='0',
                 num_format_default='0.00',
                 num_format_pct='0%',
                 num_format_stat='0.00',
                 text_wrap=True)

_DEFAULT_GENERAL = dict(bg_color=_DEFAULTS['bg_color'],
                        bold=_DEFAULTS['bold'],
                        border_color=_DEFAULTS['border_color'],
                        border_color_net_top=_DEFAULTS['border_color'],
                        border_color_stat_top=_DEFAULTS['border_color'],
                        border_style_ext=_DEFAULTS['border_style_ext'],
                        border_style_int=_DEFAULTS['border_style_int'],
                        bottom=_DEFAULTS['border'],
                        bottom_color=_DEFAULTS['border_color'],
                        font_color=_DEFAULTS['font_color'],
                        font_color_str=_DEFAULTS['font_color'],
                        font_name=_DEFAULTS['font_name'],
                        font_name_str=_DEFAULTS['font_name'],
                        font_script=False,
                        font_size=_DEFAULTS['font_size'],
                        font_size_str=_DEFAULTS['font_size'],
                        font_super_test=True,
                        font_super_nettest=True,
                        font_super_stattest=True,
                        italic=_DEFAULTS['italic'],
                        left=_DEFAULTS['border'],
                        left_color=_DEFAULTS['border_color'],
                        num_format=_DEFAULTS['num_format'],
                        num_format_count=_DEFAULTS['num_format_count'],
                        num_format_default=_DEFAULTS['num_format_default'],
                        num_format_pct=_DEFAULTS['num_format_pct'],
                        num_format_stat=_DEFAULTS['num_format_stat'],
                        right=_DEFAULTS['border'],
                        right_color=_DEFAULTS['border_color'],
                        text_v_align=2,
                        text_h_align=2,
                        text_h_align_label=1,
                        text_wrap=_DEFAULTS['text_wrap'],
                        top=_DEFAULTS['border'],
                        top_color=_DEFAULTS['border_color'])

_DEFAULT_ALIGN = dict(text_v_align=2, text_h_align=2, text_v_align_text=2, text_h_align_text=3)

_CELL_ATTRIBUTES = ('bg_color', 'bold', 'font_color', 'font_name', 'font_size', 'italic')

_VIEWS = ('base', 'count', 'pct', 'net', 'nettest', 'stat', 'stattest', 'sum', 'test', 'ubase')

_CELLS = ('y', 'label')
for view in _VIEWS:
	_CELLS = _CELLS + (view, view + '_text')

_DEFAULT_CELL = dict()
for cell in _CELLS:
	attrs = [(attr + '_' + cell, _DEFAULTS[attr]) for attr in _CELL_ATTRIBUTES]
	for attr in ('text_v_align', 'text_h_align'):
		cell_attr = attr + '_' + cell
		if cell_attr not in _DEFAULT_GENERAL:
			if 'text' in cell:
				_DEFAULT_GENERAL.update(dict([(cell_attr, _DEFAULT_ALIGN[attr + '_text'])]))
			else:
				_DEFAULT_GENERAL.update(dict([(cell_attr, _DEFAULT_ALIGN[attr])]))
	_DEFAULT_CELL.update(dict(attrs))


_DEFAULT_ATTRIBUTES = dict([item for item in (_DEFAULT_CELL.items() + _DEFAULT_GENERAL.items())])
