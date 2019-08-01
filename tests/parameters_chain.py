# -*- coding: utf-8 -*-

import json
import numpy as np

from itertools import izip_longest, product
from itertools import chain as ichain
from operator import add

NAN = np.NaN
AST = '*'
EMPTY = ''
NAP = "Not applicable - I won't ever go to a sporting goods store"
DK = "Don't know"
TCP = "To check prices"
TBSFM = "To buy something for myself"
TBSFSE = "To buy something for somebody else"
TBSFSEO = "To buy something for somebody else, Other"
TATSFSA = "To ask the staff for sporting advice"
TBSFMTCP = "To buy something for myself, To check prices"
TBSFSETCP = "To buy something for somebody else, To check prices"
TBSFSETATSFSATCP = (
    "To buy something for somebody else, To ask the staff for sporting advice,"
    " To check prices")
TBSFMTATSFSA = (
    "To buy something for myself, To ask the staff for sporting advice")
TBSFMTBSFSETCP = (
    "To buy something for myself, To buy something for somebody else, "
    "To check prices")
TBSFMTBSFSETCPO = (
    "To buy something for myself, To buy something for somebody else, "
    "To check prices, Other")
TBSFMTATSFSATCP = (
    "To buy something for myself, To ask the staff for sporting advice, "
    "To check prices")
TBSFSETATSFSA = (
    "To buy something for somebody else, To ask the staff for sporting advice")
TBSFMTBSFSE = "To buy something for myself, To buy something for somebody else"
TBSFMTBSFSETATSFSATCP = (
    "To buy something for myself, To buy something for somebody else, To ask "
    "the staff for sporting advice, To check prices")


def czip(*args):
    return list(ichain(_zip(*args)))


def lzip(*args):
    return list(_zip(*args))


def _zip(question, values):
    return izip_longest((question, ), values, fillvalue=question)


def get_values(column, values, paint=False):
    def _get_values():
        if column['parent']:
            return values[column['parent'].keys()[0].split('@')[1]]
        return column['values']
    if paint:
        return [u'%s' % item['text']['en-GB'] for item in _get_values()]
    return [long(item['value']) for item in _get_values()]


def get_label(column, paint=False):
    if paint:
        return u'%s. %s' % (column['name'], column['text']['en-GB'])
    return u'%s' % column['name']


def build_x_index(column, lib_values, paint=False):
    prepend = [u'Base'] if paint else [u'All']
    append = [u'Mean', u'Median'] if paint else [u'mean', u'median']
    return lzip(
        get_label(column, paint=paint),
        (prepend + get_values(column, lib_values, paint=paint) + append))


def build_y_index(columns, lib_values, x, y_keys, paint=False):
    max_ = max(map(lambda x: len(x.split('>')), y_keys))
    pad_id = 0
    total = u'Total' if paint else u'@'
    result = []
    level = [(get_label(columns[x], paint=paint), total)]
    if max_ > 1:
        level = pad(level, (max_ - 1), pad_id)
        pad_id += 1
    result.append(level)
    for y in y_keys:
        nest_size = len(y.split('>'))
        if nest_size > 1:
            level = nest(columns, lib_values, y, paint)
        else:
            level = czip(get_label(columns[y], paint=paint),
                         get_values(columns[y], lib_values, paint=paint))
        if max_ > nest_size:
            level = pad(level, (max_ - nest_size), pad_id)
            pad_id += 1
        result.append(level)
    return list(ichain(*result))


def nest(columns, lib_values, y_keys, paint):
    result = []
    for y in y_keys.split(' > '):
        result.append(czip(get_label(columns[y], paint=paint),
                           get_values(columns[y], lib_values, paint=paint)))
    return [sum(item, ()) for item in product(*result)]


def pad(level, size, pad_id):
    pad = tuple(u'#pad-%d' % pad_id for _ in xrange(size * 2))
    return [add(pad, item) for item in level]


COMPLEX_CHAIN_STR = [
    (
        u'Chain...'
        u'\nSource:          native'
        u'\nName:            q5_1'
        u'\nOrientation:     x'
        u'\nX:               [\'q5_1\']'
        u'\nY:               [\'@\', \'q4\']'
        u'\nNumber of views: OrderedDict([(\'x|f|x:|||cbase\', 1), '
        u'(\'x|f|:|||counts\', 7), (\'x|d.mean|x:|||mean\', 1), '
        u'(\'x|d.median|x:|||median\', 1)])'),
    (
        u'Chain...'
        u'\nSource:          native'
        u'\nName:            q5_1'
        u'\nOrientation:     x'
        u'\nX:               [\'q5_1\']'
        u'\nY:               [\'@\', \'q4 > gender\']'
        u'\nNumber of views: OrderedDict([(\'x|f|x:|||cbase\', 1), '
        u'(\'x|f|:|||counts\', 7), (\'x|d.mean|x:|||mean\', 1), '
        u'(\'x|d.median|x:|||median\', 1)])'),
    (
        u'Chain...'
        u'\nSource:          native'
        u'\nName:            q5_1'
        u'\nOrientation:     x'
        u'\nX:               [\'q5_1\']'
        u'\nY:               [\'@\', \'q4 > gender > Wave\']'
        u'\nNumber of views: OrderedDict([(\'x|f|x:|||cbase\', 1), '
        u'(\'x|f|:|||counts\', 7), (\'x|d.mean|x:|||mean\', 1), '
        u'(\'x|d.median|x:|||median\', 1)])'),
    (
        u'Chain...'
        u'\nSource:          native'
        u'\nName:            q5_1'
        u'\nOrientation:     x'
        u'\nX:               [\'q5_1\']'
        u'\nY:               [\'@\', \'q4 > gender > Wave\', '
        u'\'q5_1\', \'q4 > gender\']'
        u'\nNumber of views: OrderedDict([(\'x|f|x:|||cbase\', 1), '
        u'(\'x|f|:|||counts\', 7), (\'x|d.mean|x:|||mean\', 1), '
        u'(\'x|d.median|x:|||median\', 1)])')]


with open('./tests/Example Data (A).json', 'r') as f:
    meta = json.load(f)

X_INDEX = build_x_index(meta['columns']['q5_1'], meta['lib']['values'])

X_INDEX_PAINTED = build_x_index(
    meta['columns']['q5_1'], meta['lib']['values'], paint=True)

X1 = (
    [
        [
            [250.0, 81.0, 169.0],
            [11.0, 4.0, 7.0],
            [20.0, 5.0, 15.0],
            [74.0, 30.0, 44.0],
            [0.0, 0.0, 0.0],
            [74.0, 24.0, 50.0],
            [10.0, 4.0, 6.0],
            [61.0, 14.0, 47.0],
            [30.364, 24.493827160493826, 33.17751479289941],
            [5.0, 5.0, 5.0]
        ],
        X_INDEX,
        build_y_index(meta['columns'], meta['lib']['values'], 'q5_1', ['q4']),
        X_INDEX_PAINTED,
        build_y_index(
            meta['columns'], meta['lib']['values'], 'q5_1', ['q4'],
            paint=True),
        COMPLEX_CHAIN_STR[0]
    ],
)

X2 = (
    [
        [
            [250.0, 53.0, 28.0, 81.0, 88.0],
            [11.0, 2.0, 2.0, 5.0, 2.0],
            [20.0, 2.0, 3.0, 7.0, 8.0],
            [74.0, 19.0, 11.0, 21.0, 23.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [74.0, 19.0, 5.0, 21.0, 29.0],
            [10.0, 4.0, 0.0, 2.0, 4.0],
            [61.0, 7.0, 7.0, 25.0, 22.0],
            [
                30.364, 23.245283018867923, 26.857142857142858,
                34.95061728395062, 31.545454545454547],
            [5.0, 5.0, 3.0, 5.0, 5.0]
        ],
        X_INDEX,
        build_y_index(
            meta['columns'], meta['lib']['values'], 'q5_1', ['q4 > gender']),
        X_INDEX_PAINTED,
        build_y_index(
            meta['columns'], meta['lib']['values'], 'q5_1', ['q4 > gender'],
            paint=True),
        COMPLEX_CHAIN_STR[1]
    ],
)

X3 = (
    # values
    [
        [
            [
                250.0, 12.0, 15.0, 7.0, 8.0, 11.0, 3.0, 6.0, 6.0, 6.0, 7.0,
                20.0, 12.0, 16.0, 17.0, 16.0, 21.0, 23.0, 21.0, 9.0, 14.0],
            [
                11.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            [
                20.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0,
                1.0, 3.0, 0.0, 2.0, 3.0, 3.0, 2.0, 0.0, 0.0],
            [
                74.0, 1.0, 7.0, 3.0, 4.0, 4.0, 2.0, 4.0, 2.0, 1.0, 2.0, 4.0,
                6.0, 5.0, 4.0, 2.0, 5.0, 6.0, 5.0, 2.0, 5.0],
            [
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [
                74.0, 6.0, 3.0, 1.0, 2.0, 7.0, 1.0, 0.0, 1.0, 1.0, 2.0, 7.0,
                2.0, 3.0, 6.0, 3.0, 8.0, 7.0, 5.0, 2.0, 7.0],
            [
                10.0, 1.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
            [
                61.0, 4.0, 1.0, 2.0, 0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0, 7.0,
                2.0, 4.0, 5.0, 7.0, 4.0, 5.0, 7.0, 5.0, 1.0],
            [
                30.364, 43.5, 22.066666666666666, 30.142857142857142, 15.125,
                4.2727272727272725, 3.6666666666666665, 18.666666666666668,
                34.833333333333336, 34.333333333333336, 30.571428571428573,
                36.8, 18.916666666666668, 26.8125, 37.05882352941177, 50.5625,
                26.19047619047619, 28.130434782608695, 39.42857142857143,
                56.22222222222222, 17.5],
            [
                5.0, 5.0, 3.0, 3.0, 3.0, 5.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0,
                3.0, 3.0, 5.0, 51.0, 5.0, 5.0, 5.0, 98.0, 5.0]
        ],
        # index
        X_INDEX,
        # columns
        build_y_index(
            meta['columns'], meta['lib']['values'], 'q5_1',
            ['q4 > gender > Wave']),
        # pindex
        X_INDEX_PAINTED,
        # pcolumns
        build_y_index(
            meta['columns'], meta['lib']['values'], 'q5_1',
            ['q4 > gender > Wave'], paint=True),
        # chain_str
        COMPLEX_CHAIN_STR[2]
    ],
)

X4 = (
    [
        [
            [
                250.0, 12.0, 15.0, 7.0, 8.0, 11.0, 3.0, 6.0, 6.0, 6.0, 7.0,
                20.0, 12.0, 16.0, 17.0, 16.0, 21.0, 23.0, 21.0, 9.0, 14.0,
                11.0, 20.0, 74.0, 0.0, 74.0, 10.0, 61.0, 53.0, 28.0, 81.0,
                88.0],
            [
                11.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 11.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 5.0, 2.0],
            [
                20.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0,
                1.0, 3.0, 0.0, 2.0, 3.0, 3.0, 2.0, 0.0, 0.0, 0.0, 20.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 2.0, 3.0, 7.0, 8.0],
            [
                74.0, 1.0, 7.0, 3.0, 4.0, 4.0, 2.0, 4.0, 2.0, 1.0, 2.0, 4.0,
                6.0, 5.0, 4.0, 2.0, 5.0, 6.0, 5.0, 2.0, 5.0, 0.0, 0.0, 74.0,
                0.0, 0.0, 0.0, 0.0, 19.0, 11.0, 21.0, 23.0],
            [
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [
                74.0, 6.0, 3.0, 1.0, 2.0, 7.0, 1.0, 0.0, 1.0, 1.0, 2.0, 7.0,
                2.0, 3.0, 6.0, 3.0, 8.0, 7.0, 5.0, 2.0, 7.0, 0.0, 0.0, 0.0,
                0.0, 74.0, 0.0, 0.0, 19.0, 5.0, 21.0, 29.0],
            [
                10.0, 1.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 10.0, 0.0, 4.0, 0.0, 2.0, 4.0],
            [
                61.0, 4.0, 1.0, 2.0, 0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0, 7.0,
                2.0, 4.0, 5.0, 7.0, 4.0, 5.0, 7.0, 5.0, 1.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 61.0, 7.0, 7.0, 25.0, 22.0],
            [
                30.364, 43.5, 22.066666666666666, 30.142857142857142, 15.125,
                4.2727272727272725, 3.6666666666666665, 18.666666666666668,
                34.833333333333336, 34.333333333333336, 30.571428571428573,
                36.8, 18.916666666666668, 26.8125, 37.05882352941177, 50.5625,
                26.19047619047619, 28.130434782608695, 39.42857142857143,
                56.22222222222222, 17.5, 1.0, 2.0, 3.0, NAN, 5.0, 97.0, 98.0,
                23.245283018867923, 26.857142857142858, 34.95061728395062,
                31.545454545454547],
            [
                5.0, 5.0, 3.0, 3.0, 3.0, 5.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0,
                3.0, 3.0, 5.0, 51.0, 5.0, 5.0, 5.0, 98.0, 5.0, 1.0, 2.0, 3.0,
                0.0, 5.0, 97.0, 98.0, 5.0, 3.0, 5.0, 5.0]
        ],
        X_INDEX,
        build_y_index(
            meta['columns'], meta['lib']['values'], 'q5_1',
            ['q4 > gender > Wave', 'q5_1', 'q4 > gender']),
        X_INDEX_PAINTED,
        build_y_index(
            meta['columns'], meta['lib']['values'], 'q5_1',
            ['q4 > gender > Wave', 'q5_1', 'q4 > gender'],
            paint=True),
        COMPLEX_CHAIN_STR[3]
    ],
)

X5_SIG_SIMPLE = (
    # values
    [
        [250.0, 134.0, 116.0, 81.0, 169.0],
        [
            4.3999999999999995, 5.223880597014925, 3.4482758620689653,
            4.938271604938271, 4.142011834319527],
        ['None', "'@L'.B", "'@H'", 'D', 'None'],
        [
            8.0, 6.7164179104477615, 9.482758620689655, 6.172839506172839,
            8.875739644970414],
        ['None', "'@H'", "'@L'.A", "'@H'", "'@L'.C"],
        [
            29.599999999999998, 29.850746268656714, 29.310344827586203,
            37.03703703703704, 26.035502958579883],
        ['None', 'None', 'None', "'@L'.D", "'@H'"],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        ['None', 'None', 'None', 'None', 'None'],
        [
            29.599999999999998, 29.850746268656714, 29.310344827586203,
            29.629629629629626, 29.585798816568047],
        ['None', 'None', 'None', 'None', 'None'],
        [
            4.0, 4.477611940298507, 3.4482758620689653, 4.938271604938271,
            3.5502958579881656],
        ['None', 'B', "'@H'", "'@L'.D", 'None'],
        [24.4, 23.88059701492537, 25.0, 17.28395061728395, 27.810650887573964],
        ['None', 'None', 'None', "'@H'", "'@L'.C"],
        [
            30.364, 30.32089552238806, 30.413793103448278, 24.493827160493826,
            33.17751479289941],
        ['None', 'None', 'None', "'@H'", "'@L'.C"],
        [100.0, 100.0, 100.0, 100.0, 100.0]
    ],
    # column index
    [
        (u'q5_1', '@', '@'),
        ('gender', 1L, 'A'),
        ('gender', 2L, 'B'),
        ('q4', 1L, 'C'),
        ('q4', 2L, 'D')
    ],
    # .sig_test_letters
    ['@', 'A', 'B', 'C', 'D']
)

CONTENTS = {
    0: {
        'is_block': False,
        'is_c_base': True,
        'is_c_base_gross': False,
        'is_c_pct': False,
        'is_c_pct_cumsum': False,
        'is_c_pct_sum': False,
        'is_counts': True,
        'is_counts_cumsum': False,
        'is_counts_sum': False,
        'is_default': False,
        'is_e_base': False,
        'is_max': False,
        'is_mean': False,
        'is_meanstest': False,
        'is_median': False,
        'is_min': False,
        'is_variance': False,
        'is_sem': False,
        'is_varcoeff': False,
        'is_percentile': False,
        'is_net': False,
        'is_propstest': False,
        'is_r_base': False,
        'is_r_pct': False,
        'is_res_c_pct': False,
        'is_stat': False,
        'is_stddev': False,
        'is_weighted': False,
        'siglevel': None,
        'stat': None,
        'weight': None,
        'is_calc_only': False,
        'is_viewlike': False},
    1: {
        'is_block': False,
        'is_c_base': False,
        'is_c_base_gross': False,
        'is_c_pct': False,
        'is_c_pct_cumsum': False,
        'is_c_pct_sum': False,
        'is_counts': True,
        'is_counts_cumsum': False,
        'is_counts_sum': False,
        'is_default': False,
        'is_e_base': False,
        'is_max': False,
        'is_mean': False,
        'is_meanstest': False,
        'is_median': False,
        'is_min': False,
        'is_variance': False,
        'is_sem': False,
        'is_varcoeff': False,
        'is_percentile': False,
        'is_net': False,
        'is_propstest': False,
        'is_r_base': False,
        'is_r_pct': False,
        'is_res_c_pct': False,
        'is_stat': False,
        'is_stddev': False,
        'is_weighted': False,
        'siglevel': None,
        'stat': None,
        'weight': None,
        'is_calc_only': False,
        'is_viewlike': False},
    2: {
        'is_block': False,
        'is_c_base': False,
        'is_c_base_gross': False,
        'is_c_pct': False,
        'is_c_pct_cumsum': False,
        'is_c_pct_sum': False,
        'is_counts': True,
        'is_counts_cumsum': False,
        'is_counts_sum': False,
        'is_default': False,
        'is_e_base': False,
        'is_max': False,
        'is_mean': False,
        'is_meanstest': False,
        'is_median': False,
        'is_min': False,
        'is_variance': False,
        'is_sem': False,
        'is_varcoeff': False,
        'is_percentile': False,
        'is_net': False,
        'is_propstest': False,
        'is_r_base': False,
        'is_r_pct': False,
        'is_res_c_pct': False,
        'is_stat': False,
        'is_stddev': False,
        'is_weighted': False,
        'siglevel': None,
        'stat': None,
        'weight': None,
        'is_calc_only': False,
        'is_viewlike': False},
    3: {
        'is_block': False,
        'is_c_base': False,
        'is_c_base_gross': False,
        'is_c_pct': False,
        'is_c_pct_cumsum': False,
        'is_c_pct_sum': False,
        'is_counts': True,
        'is_counts_cumsum': False,
        'is_counts_sum': False,
        'is_default': False,
        'is_e_base': False,
        'is_max': False,
        'is_mean': False,
        'is_meanstest': False,
        'is_median': False,
        'is_min': False,
        'is_variance': False,
        'is_sem': False,
        'is_varcoeff': False,
        'is_percentile': False,
        'is_net': False,
        'is_propstest': False,
        'is_r_base': False,
        'is_r_pct': False,
        'is_res_c_pct': False,
        'is_stat': False,
        'is_stddev': False,
        'is_weighted': False,
        'siglevel': None,
        'stat': None,
        'weight': None,
        'is_calc_only': False,
        'is_viewlike': False},
    4: {
        'is_block': False,
        'is_c_base': False,
        'is_c_base_gross': False,
        'is_c_pct': False,
        'is_c_pct_cumsum': False,
        'is_c_pct_sum': False,
        'is_counts': True,
        'is_counts_cumsum': False,
        'is_counts_sum': False,
        'is_default': False,
        'is_e_base': False,
        'is_max': False,
        'is_mean': False,
        'is_meanstest': False,
        'is_median': False,
        'is_min': False,
        'is_variance': False,
        'is_sem': False,
        'is_varcoeff': False,
        'is_percentile': False,
        'is_net': False,
        'is_propstest': False,
        'is_r_base': False,
        'is_r_pct': False,
        'is_res_c_pct': False,
        'is_stat': False,
        'is_stddev': False,
        'is_weighted': False,
        'siglevel': None,
        'stat': None,
        'weight': None,
        'is_calc_only': False,
        'is_viewlike': False},
    5: {
        'is_block': False,
        'is_c_base': False,
        'is_c_base_gross': False,
        'is_c_pct': False,
        'is_c_pct_cumsum': False,
        'is_c_pct_sum': False,
        'is_counts': True,
        'is_counts_cumsum': False,
        'is_counts_sum': False,
        'is_default': False,
        'is_e_base': False,
        'is_max': False,
        'is_mean': False,
        'is_meanstest': False,
        'is_median': False,
        'is_min': False,
        'is_variance': False,
        'is_sem': False,
        'is_varcoeff': False,
        'is_percentile': False,
        'is_net': False,
        'is_propstest': False,
        'is_r_base': False,
        'is_r_pct': False,
        'is_res_c_pct': False,
        'is_stat': False,
        'is_stddev': False,
        'is_weighted': False,
        'siglevel': None,
        'stat': None,
        'weight': None,
        'is_calc_only': False,
        'is_viewlike': False},
    6: {
        'is_block': False,
        'is_c_base': False,
        'is_c_base_gross': False,
        'is_c_pct': False,
        'is_c_pct_cumsum': False,
        'is_c_pct_sum': False,
        'is_counts': True,
        'is_counts_cumsum': False,
        'is_counts_sum': False,
        'is_default': False,
        'is_e_base': False,
        'is_max': False,
        'is_mean': False,
        'is_meanstest': False,
        'is_median': False,
        'is_min': False,
        'is_variance': False,
        'is_sem': False,
        'is_varcoeff': False,
        'is_percentile': False,
        'is_net': False,
        'is_propstest': False,
        'is_r_base': False,
        'is_r_pct': False,
        'is_res_c_pct': False,
        'is_stat': False,
        'is_stddev': False,
        'is_weighted': False,
        'siglevel': None,
        'stat': None,
        'weight': None,
        'is_calc_only': False,
        'is_viewlike': False},
    7: {
        'is_block': False,
        'is_c_base': False,
        'is_c_base_gross': False,
        'is_c_pct': False,
        'is_c_pct_cumsum': False,
        'is_c_pct_sum': False,
        'is_counts': True,
        'is_counts_cumsum': False,
        'is_counts_sum': False,
        'is_default': False,
        'is_e_base': False,
        'is_max': False,
        'is_mean': False,
        'is_meanstest': False,
        'is_median': False,
        'is_min': False,
        'is_variance': False,
        'is_sem': False,
        'is_varcoeff': False,
        'is_percentile': False,
        'is_net': False,
        'is_propstest': False,
        'is_r_base': False,
        'is_r_pct': False,
        'is_res_c_pct': False,
        'is_stat': False,
        'is_stddev': False,
        'is_weighted': False,
        'siglevel': None,
        'stat': None,
        'weight': None,
        'is_calc_only': False,
        'is_viewlike': False},
    8: {
        'is_block': False,
        'is_c_base': False,
        'is_c_base_gross': False,
        'is_c_pct': False,
        'is_c_pct_cumsum': False,
        'is_c_pct_sum': False,
        'is_counts': False,
        'is_counts_cumsum': False,
        'is_counts_sum': False,
        'is_default': False,
        'is_e_base': False,
        'is_max': False,
        'is_mean': True,
        'is_meanstest': False,
        'is_median': False,
        'is_min': False,
        'is_variance': False,
        'is_sem': False,
        'is_varcoeff': False,
        'is_percentile': False,
        'is_net': False,
        'is_propstest': False,
        'is_r_base': False,
        'is_r_pct': False,
        'is_res_c_pct': False,
        'is_stat': True,
        'is_stddev': False,
        'is_weighted': False,
        'siglevel': None,
        'stat': 'mean',
        'weight': None,
        'is_calc_only': False,
        'is_viewlike': False},
    9: {
        'is_block': False,
        'is_c_base': False,
        'is_c_base_gross': False,
        'is_c_pct': False,
        'is_c_pct_cumsum': False,
        'is_c_pct_sum': False,
        'is_counts': False,
        'is_counts_cumsum': False,
        'is_counts_sum': False,
        'is_default': False,
        'is_e_base': False,
        'is_max': False,
        'is_mean': False,
        'is_meanstest': False,
        'is_median': True,
        'is_variance': False,
        'is_sem': False,
        'is_varcoeff': False,
        'is_percentile': True,
        'is_min': False,
        'is_net': False,
        'is_propstest': False,
        'is_r_base': False,
        'is_r_pct': False,
        'is_res_c_pct': False,
        'is_stat': True,
        'is_stddev': False,
        'is_weighted': False,
        'siglevel': None,
        'stat': 'median',
        'weight': None,
        'is_calc_only': False,
        'is_viewlike': False}
}

CHAIN_STRUCT_COLUMNS = ['record_number', 'age', 'gender', 'q9', 'q9a']
CHAIN_STRUCT_VALUES = [
    [
        2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
        40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
        58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
        76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93,
        94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108,
        109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122,
        123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136,
        137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150,
        151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164,
        165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178,
        179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192,
        193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206,
        207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220,
        221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234,
        235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248,
        249, 250, 251
    ],
    [
        22, 22, 28, 31, 38, 41, 20, 19, 20, 48, 22, 44, 32, 40, 19, 40, 26, 44,
        38, 25, 42, 19, 48, 39, 29, 46, 22, 42, 19, 30, 42, 48, 43, 27, 34, 30,
        24, 35, 40, 22, 27, 35, 22, 33, 48, 49, 47, 42, 30, 38, 48, 28, 41, 49,
        39, 34, 40, 49, 39, 21, 19, 47, 48, 37, 32, 23, 26, 28, 31, 26, 47, 19,
        32, 19, 44, 37, 35, 22, 36, 22, 23, 38, 37, 40, 27, 31, 23, 29, 26, 48,
        42, 43, 27, 21, 36, 46, 28, 30, 38, 37, 25, 25, 20, 19, 31, 48, 29, 26,
        22, 25, 29, 32, 36, 23, 35, 31, 19, 45, 42, 26, 37, 35, 49, 31, 35, 36,
        35, 36, 30, 35, 40, 47, 35, 25, 32, 19, 43, 43, 19, 46, 30, 47, 37, 24,
        24, 42, 23, 41, 31, 41, 47, 46, 35, 23, 25, 38, 32, 41, 30, 26, 47, 44,
        45, 33, 20, 46, 38, 40, 24, 33, 36, 29, 44, 45, 33, 22, 37, 28, 37, 38,
        42, 42, 43, 31, 20, 36, 44, 22, 22, 32, 31, 36, 27, 34, 47, 24, 21, 47,
        21, 39, 34, 32, 42, 31, 36, 33, 36, 27, 38, 30, 33, 30, 38, 44, 30, 31,
        37, 20, 24, 29, 40, 24, 31, 49, 41, 30, 32, 38, 27, 41, 43, 29, 40, 32,
        28, 45, 44, 36, 25, 27, 34, 43, 23, 32, 40, 27, 33, 22, 19, 26
    ],
    [
        1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2,
        1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 1, 2, 2, 1, 2, 2,
        2, 2, 2, 1, 1, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 1, 2, 1, 2,
        2, 1, 1, 2, 1, 1, 1, 2, 2, 1, 2, 2, 1, 1, 2, 2, 2, 1, 2, 2, 1, 2, 1, 1,
        1, 2, 1, 2, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2, 1, 2, 1, 1, 2, 1, 2, 2,
        1, 1, 1, 1, 1, 2, 1, 2, 2, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 2,
        1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1,
        2, 1, 2, 2, 1, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1,
        2, 1, 2, 1, 1, 2, 1, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 1, 2, 1, 1, 1, 1, 1,
        2, 2, 2, 1, 1, 1, 1, 2, 2, 2
    ],
    [
        '99;', '1;4;', '98;', '1;4;', '99;', '98;', '1;4;', '99;', '99;',
        '99;', '99;', '99;', '1;4;', '1;2;4;', '1;4;', '1;', '1;3;4;', '1;2;',
        '2;', '1;', '99;', '99;', '1;4;', '99;', '99;', '1;', '98;', '2;',
        '99;', '1;2;4;96;', '1;2;', '1;2;4;', '99;', '99;', '4;', '99;',
        '1;4;', '96;', '2;', '2;', '99;', '99;', '2;', '98;', '99;', '99;',
        '99;', '99;', '99;', '1;', '99;', '99;', '1;2;4;', '1;4;', '1;3;',
        '98;', '98;', '99;', '1;3;', '1;', '1;2;4;', '99;', '1;4;', '99;',
        '98;', '99;', '96;', '1;2;', '99;', '1;2;', '99;', '99;', '98;', '2;',
        '4;', '1;', '1;', '1;2;3;4;', '96;', '99;', '1;2;3;4;', '99;', '99;',
        '99;', '1;', '1;2;', '99;', '1;', '1;4;', '98;', '1;2;3;4;', '1;3;4;',
        '99;', '1;2;3;4;', '99;', '1;4;', '99;', '1;2;', '99;', '99;', '98;',
        '96;', '99;', '98;', '98;', '1;', '3;', '1;', '99;', '1;', '1;4;',
        '98;', '99;', '1;', '99;', '99;', '1;3;4;', '99;', '99;', '1;4;',
        '99;', '1;2;4;', '99;', '98;', '1;2;4;', '98;', '98;', '2;4;', '1;',
        '1;2;4;', '99;', '99;', '99;', '99;', '2;4;', '99;', '1;2;4;', '98;',
        '99;', '99;', '99;', '99;', '1;', '1;2;3;4;', '1;4;', '99;', '99;',
        '1;2;3;4;', '99;', '1;2;4;96;', '99;', '1;4;', '99;', '99;', '99;',
        '99;', '1;', '1;', '99;', '99;', '99;', '99;', '1;3;4;', '99;', '99;',
        '98;', '99;', '96;', '99;', '98;', '98;', '1;', '99;', '1;4;',
        '1;2;3;4;', '96;', '1;2;4;', '98;', '4;', '1;2;4;', '1;4;', '4;',
        '96;', '4;', '1;2;', '99;', '1;3;', '99;', '99;', '99;', '1;2;4;',
        '1;2;3;4;', '99;', '1;', '98;', '1;', '4;', '98;', '1;2;4;', '4;',
        '96;', '1;4;', '99;', '98;', '99;', '98;', '99;', '98;', '99;', '98;',
        '99;', '1;4;', '1;2;3;4;', '1;2;4;', '96;', '99;', '99;', '99;', '99;',
        '1;2;4;', '1;2;3;4;', '1;2;', '1;', '1;2;3;4;', '2;', '1;3;4;', '99;',
        '99;', '3;', '2;', '98;', '1;2;3;4;', '4;', '98;', '2;96;', '2;',
        '98;', '99;', '98;', '1;4;', '2;3;4;', '99;', '98;', '2;4;', '98;',
        '2;3;', '99;', '98;', '1;2;4;', '2;'
    ],
    [
        AST, AST, AST, AST, AST, AST, AST, AST, AST, AST, AST, AST, AST, AST,
        AST, AST, AST, AST, AST, AST, AST, AST, AST, AST, AST, AST, AST, AST,
        AST,
        (
            'Cat directed dachshund jokingly alas this disagreeably perfect'
            ' quaint.'),
        AST, AST, AST, AST, AST, AST, AST, 'Alongside ravenously near far.',
        AST, AST, AST, AST, AST, AST, AST, AST, AST, AST, AST, AST, AST, AST,
        AST, AST, AST, AST, AST, AST, AST, AST, AST, AST, AST, AST, AST, AST,
        'Owing cowered gosh spontaneously heron thus more insistent.',
        AST, AST, AST, AST, AST, AST, AST, AST, AST, AST, AST,
        (
            'But flashy began less clenched a other and amid ostrich '
            'infinitesimally jeepers.'),
        AST, AST, AST, AST, AST, AST, AST, AST, AST, AST, AST, AST, AST, AST,
        AST, AST, AST, AST, AST, AST, AST, AST,
        'Until beneath and flailed a.',
        AST, AST, AST, AST, AST, AST, AST, AST, AST, AST, AST, AST, AST, AST,
        AST, AST, AST, AST, AST, AST, AST, AST, AST, AST, AST, AST, AST, AST,
        AST, AST, AST, AST, AST, AST, AST, AST, AST, AST, AST, AST, AST, AST,
        AST, AST, AST, AST, AST,
        'Impala koala.',
        AST, AST, AST, AST, AST, AST, AST, AST, AST, AST, AST, AST, AST, AST,
        AST, AST, AST,
        'One.',
        AST, AST, AST, AST, AST, AST, AST,
        (
            'One incorrect public irrespective next much underneath however '
            'invoked.'),
        AST, AST, AST, AST, AST, AST,
        (
            'Much however much porcupine far rooster irresistible salamander '
            'some and less seal.'),
        AST, AST, AST, AST, AST, AST, AST, AST, AST, AST, AST, AST, AST, AST,
        AST, AST, AST,
        (
            'Stank withdrew one much indifferently well so trimly goat '
            'jeepers from dear.'),
        AST, AST, AST, AST, AST, AST, AST, AST, AST, AST, AST, AST, AST,
        'Outside tryingly goldfinch.',
        AST, AST, AST, AST, AST, AST, AST, AST, AST, AST, AST, AST, AST, AST,
        AST, AST, AST, AST, AST,
        'To ouch lubber jeepers when rapid balefully hey insect.',
        AST, AST, AST, AST, AST, AST, AST, AST, AST, AST, AST, AST, AST, AST,
        AST
    ]
]

CHAIN_STRUCT_COLUMNS_PAINTED = [
    'record_number. Record number',
    'age. Age',
    'gender. What is your gender?',
    (
        'q9. For what reasons might you visit a sporting goods store in the '
        'future?'),
    'q9a. Other specify (q9)']

CHAIN_STRUCT_COLUMNS_REPAINTED = [
    'record_number* Record number',
    'age* Age',
    'gender* What is your gender?',
    (
        'q9* For what reasons might you visit a sporting goods store in the '
        'future?'),
    'q9a* Other specify (q9)']

CHAIN_STRUCT_VALUES_PAINTED = [
    CHAIN_STRUCT_VALUES[0],
    CHAIN_STRUCT_VALUES[1],
    [
        'Male', 'Female', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male',
        'Female', 'Male', 'Male', 'Female', 'Female', 'Male', 'Female',
        'Female', 'Female', 'Female', 'Female', 'Male', 'Female', 'Female',
        'Male', 'Female', 'Male', 'Male', 'Male', 'Male', 'Female', 'Male',
        'Male', 'Female', 'Male', 'Female', 'Female', 'Female', 'Male',
        'Female', 'Female', 'Female', 'Male', 'Female', 'Male', 'Female',
        'Female', 'Male', 'Female', 'Female', 'Female', 'Female', 'Female',
        'Male', 'Male', 'Female', 'Male', 'Male', 'Male', 'Female', 'Female',
        'Female', 'Male', 'Male', 'Male', 'Male', 'Male', 'Female', 'Female',
        'Female', 'Male', 'Female', 'Male', 'Female', 'Female', 'Male', 'Male',
        'Female', 'Male', 'Male', 'Male', 'Female', 'Female', 'Male', 'Female',
        'Female', 'Male', 'Male', 'Female', 'Female', 'Female', 'Male',
        'Female', 'Female', 'Male', 'Female', 'Male', 'Male', 'Male', 'Female',
        'Male', 'Female', 'Female', 'Male', 'Male', 'Male', 'Male', 'Female',
        'Male', 'Male', 'Male', 'Female', 'Female', 'Female', 'Male', 'Female',
        'Male', 'Male', 'Female', 'Male', 'Female', 'Female', 'Male', 'Male',
        'Male', 'Male', 'Male', 'Female', 'Male', 'Female', 'Female', 'Male',
        'Female', 'Male', 'Male', 'Male', 'Female', 'Male', 'Female', 'Male',
        'Male', 'Male', 'Male', 'Male', 'Female', 'Female', 'Male', 'Male',
        'Male', 'Male', 'Male', 'Male', 'Female', 'Female', 'Male', 'Male',
        'Female', 'Male', 'Male', 'Female', 'Male', 'Male', 'Female', 'Female',
        'Female', 'Female', 'Female', 'Female', 'Male', 'Male', 'Female',
        'Male', 'Female', 'Female', 'Male', 'Male', 'Female', 'Female',
        'Female', 'Female', 'Female', 'Male', 'Female', 'Female', 'Female',
        'Male', 'Female', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male',
        'Male', 'Female', 'Female', 'Female', 'Female', 'Female', 'Female',
        'Male', 'Male', 'Male', 'Female', 'Male', 'Male', 'Male', 'Male',
        'Male', 'Male', 'Male', 'Male', 'Male', 'Female', 'Female', 'Male',
        'Male', 'Male', 'Female', 'Male', 'Female', 'Male', 'Male', 'Female',
        'Male', 'Female', 'Female', 'Female', 'Female', 'Female', 'Male',
        'Male', 'Male', 'Female', 'Female', 'Male', 'Female', 'Male', 'Male',
        'Male', 'Male', 'Male', 'Female', 'Female', 'Female', 'Male', 'Male',
        'Male', 'Male', 'Female', 'Female', 'Female'
    ],
    [
        NAP, TBSFMTCP, DK, TBSFMTCP, NAP, DK, TBSFMTCP, NAP, NAP, NAP, NAP,
        NAP, TBSFMTCP, TBSFMTBSFSETCP, TBSFMTCP, TBSFM, TBSFMTATSFSATCP,
        TBSFMTBSFSE, TBSFSE, TBSFM, NAP, NAP, TBSFMTCP, NAP, NAP, TBSFM, DK,
        TBSFSE, NAP, TBSFMTBSFSETCPO, TBSFMTBSFSE, TBSFMTBSFSETCP, NAP, NAP,
        TCP, NAP, TBSFMTCP, 'Other', TBSFSE, TBSFSE, NAP, NAP, TBSFSE, DK, NAP,
        NAP, NAP, NAP, NAP, TBSFM, NAP, NAP, TBSFMTBSFSETCP, TBSFMTCP,
        TBSFMTATSFSA, DK, DK, NAP, TBSFMTATSFSA, TBSFM, TBSFMTBSFSETCP, NAP,
        TBSFMTCP, NAP, DK, NAP, 'Other', TBSFMTBSFSE, NAP, TBSFMTBSFSE, NAP,
        NAP, DK, TBSFSE, TCP, TBSFM, TBSFM, TBSFMTBSFSETATSFSATCP, 'Other',
        NAP, TBSFMTBSFSETATSFSATCP, NAP, NAP, NAP, TBSFM, TBSFMTBSFSE, NAP,
        TBSFM, TBSFMTCP, DK, TBSFMTBSFSETATSFSATCP, TBSFMTATSFSATCP, NAP,
        TBSFMTBSFSETATSFSATCP, NAP, TBSFMTCP, NAP, TBSFMTBSFSE, NAP, NAP, DK,
        'Other', NAP, DK, DK, TBSFM, TATSFSA, TBSFM, NAP, TBSFM, TBSFMTCP, DK,
        NAP, TBSFM, NAP, NAP, TBSFMTATSFSATCP, NAP, NAP, TBSFMTCP, NAP,
        TBSFMTBSFSETCP, NAP, DK, TBSFMTBSFSETCP, DK, DK, TBSFSETCP, TBSFM,
        TBSFMTBSFSETCP, NAP, NAP, NAP, NAP, TBSFSETCP, NAP, TBSFMTBSFSETCP, DK,
        NAP, NAP, NAP, NAP, TBSFM, TBSFMTBSFSETATSFSATCP, TBSFMTCP, NAP, NAP,
        TBSFMTBSFSETATSFSATCP, NAP, TBSFMTBSFSETCPO, NAP, TBSFMTCP, NAP, NAP,
        NAP, NAP, TBSFM, TBSFM, NAP, NAP, NAP, NAP, TBSFMTATSFSATCP, NAP, NAP,
        DK, NAP, 'Other', NAP, DK, DK, TBSFM, NAP, TBSFMTCP,
        TBSFMTBSFSETATSFSATCP, 'Other', TBSFMTBSFSETCP, DK, TCP,
        TBSFMTBSFSETCP, TBSFMTCP, TCP, 'Other', TCP, TBSFMTBSFSE, NAP,
        TBSFMTATSFSA, NAP, NAP, NAP, TBSFMTBSFSETCP, TBSFMTBSFSETATSFSATCP,
        NAP, TBSFM, DK, TBSFM, TCP, DK, TBSFMTBSFSETCP, TCP, 'Other', TBSFMTCP,
        NAP, DK, NAP, DK, NAP, DK, NAP, DK, NAP, TBSFMTCP,
        TBSFMTBSFSETATSFSATCP, TBSFMTBSFSETCP, 'Other', NAP, NAP, NAP, NAP,
        TBSFMTBSFSETCP, TBSFMTBSFSETATSFSATCP, TBSFMTBSFSE, TBSFM,
        TBSFMTBSFSETATSFSATCP, TBSFSE, TBSFMTATSFSATCP, NAP, NAP, TATSFSA,
        TBSFSE, DK, TBSFMTBSFSETATSFSATCP, TCP, DK, TBSFSEO, TBSFSE, DK, NAP,
        DK, TBSFMTCP, TBSFSETATSFSATCP, NAP, DK, TBSFSETCP, DK, TBSFSETATSFSA,
        NAP, DK, TBSFMTBSFSETCP, TBSFSE],
    CHAIN_STRUCT_VALUES[4],
]
