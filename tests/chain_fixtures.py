# -*- coding: utf-8 -*-

import json
import numpy as np

from itertools import izip_longest, product
from itertools import chain as ichain
from operator import add

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
    return lzip(get_label(column, paint=paint),
                (prepend +
                 get_values(column, lib_values, paint=paint) +
                 append))

def build_y_index(columns, lib_values, x, y_keys, paint=False):
    max_ = max(map(lambda x: len(x.split('>')), y_keys))
    pad_id = 0
    total = u'Total' if paint else u'@'
    result = []
    level = [(get_label(columns[x], paint=paint), total)]
    if max_ > 1:
        level = pad(level, (max_-1), pad_id)
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
            level = pad(level, (max_-nest_size), pad_id)
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

# BASIC_CHAIN_STR = (u'Chain...\nName:            chain\nOrientation:     None'
#                    u'\nX:               None\nY:               None'
#                    u'\nNumber of views: None')

COMPLEX_CHAIN_STR = [(u'Chain...'
                      u'\nSource:          native'
                      u'\nName:            q5_1'
                      u'\nOrientation:     x'
                      u'\nX:               [\'q5_1\']'
                      u'\nY:               [\'@\', \'q4\']'
                      u'\nNumber of views: OrderedDict([(\'x|f|x:|||cbase\', '
                      u'1), (\'x|f|:|||counts\', 7), (\'x|d.mean|x:|||mean\','
                      u' 1), (\'x|d.median|x:|||median\', 1)])'),
                     (u'Chain...'
                      u'\nSource:          native'
                      u'\nName:            q5_1'
                      u'\nOrientation:     x'
                      u'\nX:               [\'q5_1\']'
                      u'\nY:               [\'@\', \'q4 > gender\']'
                      u'\nNumber of views: OrderedDict([(\'x|f|x:|||cbase\', '
                      u'1), (\'x|f|:|||counts\', 7), (\'x|d.mean|x:|||mean\','
                      u' 1), (\'x|d.median|x:|||median\', 1)])'),
                     (u'Chain...'
                      u'\nSource:          native'
                      u'\nName:            q5_1'
                      u'\nOrientation:     x'
                      u'\nX:               [\'q5_1\']'
                      u'\nY:               [\'@\', \'q4 > gender > Wave\']'
                      u'\nNumber of views: OrderedDict([(\'x|f|x:|||cbase\', '
                      u'1), (\'x|f|:|||counts\', 7), (\'x|d.mean|x:|||mean\','
                      u' 1), (\'x|d.median|x:|||median\', 1)])'),
                     (u'Chain...'
                      u'\nSource:          native'
                      u'\nName:            q5_1'
                      u'\nOrientation:     x'
                      u'\nX:               [\'q5_1\']'
                      u'\nY:               [\'@\', \'q4 > gender > Wave\', '
                      u'\'q5_1\', \'q4 > gender\']'
                      u'\nNumber of views: OrderedDict([(\'x|f|x:|||cbase\', '
                      u'1), (\'x|f|:|||counts\', 7), (\'x|d.mean|x:|||mean\','
                      u' 1), (\'x|d.median|x:|||median\', 1)])')]


with open('./tests/Example Data (A).json', 'r') as f:
    meta = json.load(f)

X_INDEX = build_x_index(meta['columns']['q5_1'], meta['lib']['values'])

X_INDEX_PAINTED = build_x_index(meta['columns']['q5_1'], meta['lib']['values'],
                                paint=True)

X1 = ([[[250.0, 81.0, 169.0], [11.0, 4.0, 7.0], [20.0, 5.0, 15.0],
        [74.0, 30.0, 44.0], [0.0, 0.0, 0.0], [74.0, 24.0, 50.0],
        [10.0, 4.0, 6.0], [61.0, 14.0, 47.0],
        [30.364, 24.493827160493826, 33.17751479289941],
        [5.0, 5.0, 5.0]],
       X_INDEX,
       build_y_index(meta['columns'], meta['lib']['values'], 'q5_1', ['q4']),
       X_INDEX_PAINTED,
       build_y_index(meta['columns'], meta['lib']['values'], 'q5_1', ['q4'],
                     paint=True),
       COMPLEX_CHAIN_STR[0]],
     )

X2 = ([[[250.0, 53.0, 28.0, 81.0, 88.0], [11.0, 2.0, 2.0, 5.0, 2.0],
        [20.0, 2.0, 3.0, 7.0, 8.0], [74.0, 19.0, 11.0, 21.0, 23.0],
        [0.0, 0.0, 0.0, 0.0, 0.0], [74.0, 19.0, 5.0, 21.0, 29.0],
        [10.0, 4.0, 0.0, 2.0, 4.0],  [61.0, 7.0, 7.0, 25.0, 22.0],
        [30.364, 23.245283018867923, 26.857142857142858,
         34.95061728395062, 31.545454545454547],
        [5.0, 5.0, 3.0, 5.0, 5.0]],
       X_INDEX,
       build_y_index(meta['columns'], meta['lib']['values'], 'q5_1',
                     ['q4 > gender']),
       X_INDEX_PAINTED,
       build_y_index(meta['columns'], meta['lib']['values'], 'q5_1',
                     ['q4 > gender'], paint=True),
       COMPLEX_CHAIN_STR[1]],
     )

X3 = ([[[250.0, 12.0, 15.0, 7.0, 8.0, 11.0, 3.0, 6.0, 6.0, 6.0, 7.0, 20.0,
         12.0, 16.0, 17.0, 16.0, 21.0, 23.0, 21.0, 9.0, 14.0],
        [11.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 1.0, 1.0, 1.0,
         1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        [20.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 3.0,
         0.0, 2.0, 3.0, 3.0, 2.0, 0.0, 0.0],
        [74.0, 1.0, 7.0, 3.0, 4.0, 4.0, 2.0, 4.0, 2.0, 1.0, 2.0, 4.0, 6.0, 5.0,
         4.0, 2.0, 5.0, 6.0, 5.0, 2.0, 5.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [74.0, 6.0, 3.0, 1.0, 2.0, 7.0, 1.0, 0.0, 1.0, 1.0, 2.0, 7.0, 2.0, 3.0,
         6.0, 3.0, 8.0, 7.0, 5.0, 2.0, 7.0],
        [10.0, 1.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
        [61.0, 4.0, 1.0, 2.0, 0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0, 7.0, 2.0, 4.0,
         5.0, 7.0, 4.0, 5.0, 7.0, 5.0, 1.0],
        [30.364, 43.5, 22.066666666666666, 30.142857142857142, 15.125,
         4.2727272727272725, 3.6666666666666665, 18.666666666666668,
         34.833333333333336, 34.333333333333336, 30.571428571428573, 36.8,
         18.916666666666668, 26.8125, 37.05882352941177, 50.5625,
         26.19047619047619, 28.130434782608695, 39.42857142857143,
         56.22222222222222, 17.5],
        [5.0, 5.0, 3.0, 3.0, 3.0, 5.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 3.0, 3.0,
         5.0, 51.0, 5.0, 5.0, 5.0, 98.0, 5.0]],
       X_INDEX,
       build_y_index(meta['columns'], meta['lib']['values'], 'q5_1',
                     ['q4 > gender > Wave']),
       X_INDEX_PAINTED,
       build_y_index(meta['columns'], meta['lib']['values'], 'q5_1',
                     ['q4 > gender > Wave'], paint=True),
       COMPLEX_CHAIN_STR[2]],
     )

X4 = ([[[250.0, 12.0, 15.0, 7.0, 8.0, 11.0, 3.0, 6.0, 6.0, 6.0, 7.0, 20.0,
         12.0, 16.0, 17.0, 16.0, 21.0, 23.0, 21.0, 9.0, 14.0, 11.0, 20.0, 74.0,
         0.0, 74.0, 10.0, 61.0, 53.0, 28.0, 81.0, 88.0],
        [11.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 1.0, 1.0, 1.0,
         1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 11.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         2.0, 2.0, 5.0, 2.0],
        [20.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 3.0,
         0.0, 2.0, 3.0, 3.0, 2.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         2.0, 3.0, 7.0, 8.0],
        [74.0, 1.0, 7.0, 3.0, 4.0, 4.0, 2.0, 4.0, 2.0, 1.0, 2.0, 4.0, 6.0, 5.0,
         4.0, 2.0, 5.0, 6.0, 5.0, 2.0, 5.0, 0.0, 0.0, 74.0, 0.0, 0.0, 0.0, 0.0,
          19.0, 11.0, 21.0, 23.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0],
        [74.0, 6.0, 3.0, 1.0, 2.0, 7.0, 1.0, 0.0, 1.0, 1.0, 2.0, 7.0, 2.0, 3.0,
         6.0, 3.0, 8.0, 7.0, 5.0, 2.0, 7.0, 0.0, 0.0, 0.0, 0.0, 74.0, 0.0, 0.0,
         19.0, 5.0, 21.0, 29.0],
        [10.0, 1.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0,
         4.0, 0.0, 2.0, 4.0],
        [61.0, 4.0, 1.0, 2.0, 0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0, 7.0, 2.0, 4.0,
         5.0, 7.0, 4.0, 5.0, 7.0, 5.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 61.0,
         7.0, 7.0, 25.0, 22.0],
        [30.364, 43.5, 22.066666666666666, 30.142857142857142, 15.125,
         4.2727272727272725, 3.6666666666666665, 18.666666666666668,
         34.833333333333336, 34.333333333333336, 30.571428571428573, 36.8,
         18.916666666666668, 26.8125, 37.05882352941177, 50.5625,
         26.19047619047619, 28.130434782608695, 39.42857142857143,
         56.22222222222222, 17.5, 1.0, 2.0, 3.0, np.NaN, 5.0, 97.0, 98.0,
         23.245283018867923, 26.857142857142858, 34.95061728395062,
         31.545454545454547],
        [5.0, 5.0, 3.0, 3.0, 3.0, 5.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 3.0, 3.0,
         5.0, 51.0, 5.0, 5.0, 5.0, 98.0, 5.0, 1.0, 2.0, 3.0, 0.0, 5.0, 97.0,
         98.0, 5.0, 3.0, 5.0, 5.0]],
       X_INDEX,
       build_y_index(meta['columns'], meta['lib']['values'], 'q5_1',
                     ['q4 > gender > Wave', 'q5_1', 'q4 > gender']),
       X_INDEX_PAINTED,
       build_y_index(meta['columns'], meta['lib']['values'], 'q5_1',
                     ['q4 > gender > Wave', 'q5_1', 'q4 > gender'],
                     paint=True),
       COMPLEX_CHAIN_STR[3]],
     )



X5_SIG_SIMPLE = (# values
                 [[250.0, 134.0, 116.0, 81.0, 169.0],
                  [4.3999999999999995, 5.223880597014925, 3.4482758620689653,
                   4.938271604938271, 4.142011834319527],
                  ['None', "'@L'.B", "'@H'", 'D', 'None'],
                  [8.0, 6.7164179104477615, 9.482758620689655, 6.172839506172839,
                   8.875739644970414],
                  ['None', "'@H'", "'@L'.A", "'@H'", "'@L'.C"],
                  [29.599999999999998, 29.850746268656714, 29.310344827586203,
                   37.03703703703704, 26.035502958579883],
                  ['None', 'None', 'None', "'@L'.D", "'@H'"],
                  [0.0, 0.0, 0.0, 0.0, 0.0],
                  ['None', 'None', 'None', 'None', 'None'],
                  [29.599999999999998, 29.850746268656714, 29.310344827586203,
                   29.629629629629626, 29.585798816568047],
                  ['None', 'None', 'None', 'None', 'None'],
                  [4.0, 4.477611940298507, 3.4482758620689653, 4.938271604938271,
                   3.5502958579881656],
                  ['None', 'B', "'@H'", "'@L'.D", 'None'],
                  [24.4, 23.88059701492537, 25.0, 17.28395061728395, 27.810650887573964],
                  ['None', 'None', 'None', "'@H'", "'@L'.C"],
                  [30.364, 30.32089552238806, 30.413793103448278, 24.493827160493826,
                   33.17751479289941],
                  ['None', 'None', 'None', "'@H'", "'@L'.C"],
                  [100.0, 100.0, 100.0, 100.0, 100.0]],
                 # column index
                 [(u'q5_1', '@', '@'),
                  ('gender', 1L, 'A'),
                  ('gender', 2L, 'B'),
                  ('q4', 1L, 'C'),
                  ('q4', 2L, 'D')],
                 # .sig_test_letters
                 ['@', 'A', 'B', 'C', 'D'])

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
     'is_calc_only': False},
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
     'is_calc_only': False},
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
     'is_calc_only': False},
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
     'is_calc_only': False},
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
     'is_calc_only': False},
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
     'is_calc_only': False},
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
     'is_calc_only': False},
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
     'is_calc_only': False},
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
     'is_calc_only': False},
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
     'is_calc_only': False}
      }
