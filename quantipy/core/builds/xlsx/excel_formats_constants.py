"""
Excel formats constants.
"""

_ATTRIBUTES = (
    'bg_color',
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

_DEFAULTS = dict(
    bg_color='#FFFFFF',
    bold=False,
    border=None,
    border_color='#D9D9D9',
    border_style_ext=5,
    border_style_int=1,
    font_color='#000000',
    font_name='Arial',
    font_size=9,
    font_script=1,
    italic=False,
    num_format='0',
    num_format_counts='0',
    num_format_default='0.00',
    num_format_c_pct='0%',
    num_format_mean='0.00',
    text_wrap=True)

_DEFAULT_GENERAL = dict(
    bg_color=_DEFAULTS['bg_color'],
    bold=_DEFAULTS['bold'],
    border_color=_DEFAULTS['border_color'],
    border_style_ext=_DEFAULTS['border_style_ext'],
    border_style_int=_DEFAULTS['border_style_int'],
    bottom=_DEFAULTS['border'],
    bottom_color=_DEFAULTS['border_color'],
    font_color=_DEFAULTS['font_color'],
    font_color_str=_DEFAULTS['font_color'],
    font_name=_DEFAULTS['font_name'],
    font_name_str=_DEFAULTS['font_name'],
    font_script=False,
    font_script_block_calc_propstest=_DEFAULTS['font_script'],
    font_script_block_calc_net_propstest=_DEFAULTS['font_script'],
    font_script_block_expanded_propstest=_DEFAULTS['font_script'],
    font_script_block_net_propstest=_DEFAULTS['font_script'],
    font_script_block_normal_propstest=_DEFAULTS['font_script'],
    font_script_meanstest=_DEFAULTS['font_script'],
    font_script_net_propstest=_DEFAULTS['font_script'],
    font_script_propstest=_DEFAULTS['font_script'],
    font_size=_DEFAULTS['font_size'],
    font_size_str=_DEFAULTS['font_size'],
    italic=_DEFAULTS['italic'],
    left=_DEFAULTS['border'],
    left_color=_DEFAULTS['border_color'],
    num_format=_DEFAULTS['num_format'],
    num_format_u_c_base=_DEFAULTS['num_format_counts'],
    num_format_c_base=_DEFAULTS['num_format_counts'],
    num_format_r_base=_DEFAULTS['num_format_counts'],
    num_format_u_r_base=_DEFAULTS['num_format_counts'],
    num_format_e_base=_DEFAULTS['num_format_counts'],
    num_format_u_e_base=_DEFAULTS['num_format_counts'],
    num_format_counts=_DEFAULTS['num_format_counts'],
    num_format_net_counts=_DEFAULTS['num_format_counts'],
    num_format_default=_DEFAULTS['num_format_default'],
    num_format_c_pct=_DEFAULTS['num_format_c_pct'],
    num_format_r_pct=_DEFAULTS['num_format_c_pct'],
    num_format_net_c_pct=_DEFAULTS['num_format_c_pct'],
    num_format_net_r_pct=_DEFAULTS['num_format_c_pct'],
    num_format_block_calc_counts=_DEFAULTS['num_format_counts'],
    num_format_block_calc_net_counts=_DEFAULTS['num_format_counts'],
    num_format_block_calc_c_pct=_DEFAULTS['num_format_c_pct'],
    num_format_block_calc_net_c_pct=_DEFAULTS['num_format_c_pct'],
    num_format_block_calc_r_pct=_DEFAULTS['num_format_c_pct'],
    num_format_block_calc_net_r_pct=_DEFAULTS['num_format_c_pct'],
    num_format_block_expanded_counts=_DEFAULTS['num_format_counts'],
    num_format_block_expanded_c_pct=_DEFAULTS['num_format_c_pct'],
    num_format_block_expanded_r_pct=_DEFAULTS['num_format_c_pct'],
    num_format_block_net_counts=_DEFAULTS['num_format_counts'],
    num_format_block_net_c_pct=_DEFAULTS['num_format_c_pct'],
    num_format_block_net_r_pct=_DEFAULTS['num_format_c_pct'],
    num_format_block_normal_counts=_DEFAULTS['num_format_counts'],
    num_format_block_normal_c_pct=_DEFAULTS['num_format_c_pct'],
    num_format_block_normal_r_pct=_DEFAULTS['num_format_c_pct'],
    num_format_block_calc_normal_counts=_DEFAULTS['num_format_counts'],
    num_format_block_calc_normal_c_pct=_DEFAULTS['num_format_c_pct'],
    num_format_block_calc_normal_r_pct=_DEFAULTS['num_format_c_pct'],
    num_format_mean=_DEFAULTS['num_format_mean'],
    num_format_stddev=_DEFAULTS['num_format_mean'],
    num_format_min=_DEFAULTS['num_format_mean'],
    num_format_max=_DEFAULTS['num_format_mean'],
    num_format_var=_DEFAULTS['num_format_mean'],
    num_format_varcoeff=_DEFAULTS['num_format_mean'],
    num_format_sem=_DEFAULTS['num_format_mean'],
    num_format_lower_q=_DEFAULTS['num_format_mean'],
    num_format_upper_q=_DEFAULTS['num_format_mean'],
    num_format_counts_sum=_DEFAULTS['num_format_counts'],
    num_format_c_pct_sum=_DEFAULTS['num_format_c_pct'],
    right=_DEFAULTS['border'],
    right_color=_DEFAULTS['border_color'],
    text_v_align=2,
    text_h_align=2,
    text_h_align_label=1,
    text_h_align_mask_label=1,
    text_wrap=_DEFAULTS['text_wrap'],
    top=_DEFAULTS['border'],
    top_color=_DEFAULTS['border_color']
)

_DEFAULT_ALIGN = dict(
    text_v_align=2,
    text_h_align=2,
    text_v_align_text=2,
    text_h_align_text=3)

_CELL_ATTRIBUTES = (
    'bg_color',
    'bold',
    'font_color',
    'font_name',
    'font_size',
    'italic',
    'text_wrap')

_VIEWS_GROUPS = dict(
    default='default',
    label='label',
    mask_label='label',
    c_base='base',
    u_c_base='u_base',
    c_base_gross='base',
    u_c_base_gross='u_base',
    e_base='base',
    u_e_base='u_base',
    u_r_base='base',
    r_base='u_base',
    counts='freq',
    c_pct='freq',
    res_c_pct='freq',
    r_pct='freq',
    block_normal_counts='freq',
    block_normal_c_pct='freq',
    block_normal_r_pct='freq',
    block_normal_propstest='freq',
    block_calc_normal_counts='freq',
    block_calc_normal_c_pct='freq',
    block_calc_normal_r_pct='freq',
    block_calc_normal_propstest='freq',
    propstest='freq',
    net_counts='net',
    net_c_pct='net',
    net_r_pct='net',
    net_propstest='net',
    block_calc_net_counts='net',
    block_calc_net_c_pct='net',
    block_calc_net_r_pct='net',
    block_calc_net_propstest='net',
    block_calc_counts='net',
    block_calc_c_pct='net',
    block_calc_r_pct='net',
    block_calc_propstest='net',
    block_expanded_counts='block_expanded',
    block_expanded_c_pct='block_expanded',
    block_expanded_r_pct='block_expanded',
    block_expanded_propstest='block_expanded',
    block_net_counts='block_net',
    block_net_c_pct='block_net',
    block_net_r_pct='block_net',
    block_net_propstest='block_net',
    mean='stat',
    stddev='stat',
    min='stat',
    max='stat',
    median='stat',
    var='stat',
    varcoeff='stat',
    sem='stat',
    lower_q='stat',
    upper_q='stat',
    meanstest='stat',
    counts_sum='sum',
    c_pct_sum='sum',
    counts_cumsum='sum',
    c_pct_cumsum='sum')

_NO_TEXT = ('label', 'mask_label')

_CELLS = (
    'y',
    'data_header',
    'header_left',
    'header_center',
    'header_title',
    'notes',
    'data')

for view in _VIEWS_GROUPS.keys():
    _CELLS += (view, ) if view in _NO_TEXT else (view, view + '_text')

_DEFAULT_CELL = dict()

for cell in _CELLS:
    attrs = [(attr + '_' + cell, _DEFAULTS[attr]) for attr in _CELL_ATTRIBUTES]
    _DEFAULT_CELL.update(dict(attrs))
    for attr in ('text_v_align', 'text_h_align'):
        cell_attr = attr + '_' + cell
        if cell_attr not in _DEFAULT_GENERAL:
            if 'text' in cell:
                attr += '_text'
            _DEFAULT_GENERAL.update(dict([(cell_attr, _DEFAULT_ALIGN[attr])]))
    _DEFAULT_GENERAL['view_border_' + cell] = _DEFAULTS['border_style_int']

items = _DEFAULT_CELL.items() + _DEFAULT_GENERAL.items()
_DEFAULT_ATTRIBUTES = dict([item for item in items])
