
import re
import os
import sys
import pytest
import numpy as np
import quantipy as qp

from quantipy.sandbox.sandbox import ChainManager
from quantipy.sandbox.excel import Excel
from zipfile import ZipFile, BadZipfile, LargeZipFile


PATH_DATA = '../../tests/'
NAME_PROJ = 'Example Data (A)'
NAME_META = 'Example Data (A).json'
NAME_DATA = 'Example Data (A).csv'
PATH_META = os.path.join(PATH_DATA, NAME_META)
PATH_DATA = os.path.join(PATH_DATA, NAME_DATA)

DATA_KEY = ORIENT = 'x'
FILTER_KEY = 'no_filter'
# X_KEYS = ['q5_1']
X_KEYS = ['q5_1', 'q4', 'gender', 'Wave']
#Y_KEYS = ['@', 'q4']                                        # 1.
#Y_KEYS = ['@', 'q4', 'q5_2', 'gender', 'Wave']              # 2.
#Y_KEYS = ['@', 'q4 > gender']                               # 3.
#Y_KEYS = ['@', 'q4 > gender > Wave']                        # 4.
Y_KEYS = ['@', 'q4 > gender', 'q4 > gender > Wave', 'q5_1'] # 5.
TESTS = True

# WEIGHT = None
WEIGHT = 'weight_a'

VIEWS = ('cbase',
         'cbase_gross',
         #'rbase',
         'ebase',
         'counts',
         'c%',
         'r%',
         #'mean',
         #'stddev',
         #'median',
         #'variance',
         #'varcoeff',
         #'sem',
         #'lower_q',
         #'upper_q',
         'counts_sum',
         'c%_sum',
         #'counts_cumsum',
         #'c%_cumsum',
         )

VIEW_KEYS = ('x|f|x:|||cbase',
             'x|f|x:||%s|cbase' % WEIGHT,
             'x|f|x:|||cbase_gross',
             'x|f|x:||%s|cbase_gross' % WEIGHT,
             'x|f|x:|||ebase',
             'x|f|x:||%s|ebase' % WEIGHT,
             'x|f|:||%s|counts' % WEIGHT,
             'x|f|:|y|%s|c%%' % WEIGHT,
             'x|f|:|x|%s|r%%' % WEIGHT,
             'x|t.props.Dim.80+@|:||%s|test' % WEIGHT,
             'x|f|x[{1,2,3}]:||%s|No' % WEIGHT,
             'x|f|x[{1,2,3}]:|y|%s|No' % WEIGHT,
             'x|t.props.Dim.80+@|x[{1,2,3}]:||%s|test' % WEIGHT,
             'x|f|x[{4,5,97}]:||%s|Yes' % WEIGHT,
             'x|f|x[{4,5,97}]:|y|%s|Yes' % WEIGHT,
             'x|t.props.Dim.80+@|x[{4,5,97}]:||%s|test' % WEIGHT,
             'x|d.mean|x:||%s|mean' % WEIGHT,
             'x|t.means.Dim.80+@|x:||%s|test' % WEIGHT,
             'x|d.stddev|x:||%s|stddev' % WEIGHT,
             'x|d.median|x:||%s|median' % WEIGHT,
             'x|f.c:f|x:||%s|counts_sum' % WEIGHT,
             'x|f.c:f|x:|y|%s|c%%_sum' % WEIGHT,
             #'x|f.c:f|x++:||%s|counts_cumsum' % WEIGHT,
             #'x|f.c:f|x++:|y|%s|c%%_cumsum' % WEIGHT,
            )

weights = [None]
if WEIGHT is not None:
    VIEW_KEYS = ('x|f|x:|||cbase', ) + VIEW_KEYS
    weights.append(WEIGHT)


# RUN RUN RUN RUN RUN RUN RUN RUN RUN RUN RUN RUN RUN RUN RUN RUN
CA1 = True
AC1 = True
ACB1 = True
ACM1 = True
AC0 = True
ACB0 = True
ACM0 = True
OEC = True

dataset = qp.DataSet(NAME_PROJ, dimensions_comp=False)
dataset.read_quantipy(PATH_META, PATH_DATA)
meta, data = dataset.split()

# text key modifications ------------------------------------------
meta['columns']['Wave']['text']['fail'] = 'from text > fake'
meta['columns']['Wave']['text']['x edits'] = {'fail': 'from text > x edits > fail'}

meta['columns']['gender']['text']['y edits'] = {'fake': 'from text > y edits > fake'}
meta['columns']['gender']['text']['x edits'] = {'en-GB': 'from text > x edits > en-GB'}

meta['columns']['q5_1']['text']['x edits'] = {'fake': 'from text > x edits > fake'}
meta['columns']['q5_1']['properties'] = {'base_text': {'fake': 'Base: fake',
                                                       'en-GB': 'Base: en-GB'}}

meta['columns']['q4']['text']['fake'] = 'from text > fake'
meta['columns']['q4']['text']['y edits'] = {'fake': 'from text > y edits > fake'}
meta['columns']['q4']['properties'] = {'base_text': 'Base: Text'}
# -----------------------------------------------------------------

#data = data.head(250)
data.loc[30:,'q5_2'] = np.NaN
data.loc[30:,'q5_4'] = np.NaN

stack = qp.Stack(NAME_PROJ, add_data={DATA_KEY: {'meta': meta, 'data': data}})
stack.add_link(x=X_KEYS, y=Y_KEYS, views=VIEWS, weights=weights)

rel_to = []
if 'counts' in VIEWS:
    rel_to.append(None)
if 'c%' in VIEWS:
    rel_to.append('y')
if 'r%' in VIEWS:
    rel_to.append('x')

nets_mapper = qp.ViewMapper(template={'method': qp.QuantipyViews().frequency,
                                      'kwargs': {'iterators': {'rel_to': rel_to},
                                                 'groups': 'Nets'}})
nets_mapper.add_method(name='No', kwargs={'axis':    'x',
                                          'logic':   [{'No': [1, 2, 3]}],
                                          'text':    'Net: No',
                                          'combine': False})
stack.add_link(x=X_KEYS[0], y=Y_KEYS, views=nets_mapper, weights=weights)

nets_mapper = qp.ViewMapper(template={'method': qp.QuantipyViews().frequency,
                                      'kwargs': {'iterators': {'rel_to': rel_to},
                                                 'groups': 'Nets'}})
nets_mapper.add_method(name='Yes', kwargs={'axis':    'x',
                                           'logic':   [{'Yes': [4, 5, 97]}],
                                           'text':    'Net: Yes',
                                           'combine': False})
stack.add_link(x=X_KEYS[0], y=Y_KEYS, views=nets_mapper, weights=weights)

nets_mapper = qp.ViewMapper(template={'method': qp.QuantipyViews().frequency,
                                      'kwargs': {'iterators': {'rel_to': rel_to},
                                                 'groups': 'Nets'}})
nets = [{'N1': [1, 2], 'text': {'en-GB': 'Waves 1 & 2 (NET)'}, 'expand': 'after'},
        {'N2': [4, 5], 'text': {'en-GB': 'Waves 4 & 5 (NET)'}, 'expand': 'after'}]
nets_mapper.add_method(name='BLOCK', kwargs={'axis':      'x',
                                             'logic':     nets,
                                             'text':      'Net: ',
                                             'combine':   False,
                                             'complete':  True,
                                             'expand':    'after'}
                                             )
stack.add_link(x=X_KEYS[-1], y=Y_KEYS, views=nets_mapper, weights=weights)

nets_mapper = qp.ViewMapper(template={'method': qp.QuantipyViews().frequency,
                                      'kwargs': {'iterators': {'rel_to': rel_to},
                                                 'groups': 'Nets'}})
from operator import sub
kwargs = {'calc_only': False,
          'calc': {'text': {u'en-GB': u'Net YES'},
          'Net agreement': ('Net: Yes', sub, 'Net: No')},
          'axis': 'x',
          'logic': [{'text': {u'en-GB': 'Net: No'}, 'Net: No': [1, 2]},
                    {'text': {u'en-GB': 'Net: Yes'}, 'Net: Yes': [4, 5]}]}
nets_mapper.add_method(name='NPS', kwargs=kwargs)
kwargs = {'calc_only': True,
          'calc': {'text': {u'en-GB': u'Net YES'},
          'Net agreement (only)': ('Net: Yes', sub, 'Net: No')},
          'axis': 'x',
          'logic': [{'text': {u'en-GB': 'Net: No'}, 'Net: No': [1, 2]},
                    {'text': {u'en-GB': 'Net: Yes'}, 'Net: Yes': [4, 5]}]}
nets_mapper.add_method(name='NPSonly', kwargs=kwargs)
stack.add_link(x=X_KEYS[0], y=Y_KEYS, views=nets_mapper, weights=weights)

stats = ['mean', 'stddev', 'median', 'var', 'varcoeff', 'sem', 'lower_q', 'upper_q']
for stat in stats:
    options = {'stats': stat,
               'source': None,
               'rescale': None,
               'drop': False,
               'exclude': None,
               'axis': 'x',
               'text': ''}
    view = qp.ViewMapper()
    view.make_template('descriptives')
    view.add_method('stat', kwargs=options)
    stack.add_link(x=X_KEYS, y=Y_KEYS, views=view, weights=weights)

if TESTS:
    test_view = qp.ViewMapper().make_template('coltests')
    view_name = 'test'
    options = {'level': 0.8,
            'metric': 'props',
            'test_total': True,
            'flag_bases': [30, 100]
            }
    test_view.add_method(view_name, kwargs=options)
    stack.add_link(x=X_KEYS, y=Y_KEYS, views=test_view, weights=weights)


    test_view = qp.ViewMapper().make_template('coltests')
    view_name = 'test'
    options = {'level': 0.8, 'metric': 'means',
               'test_total': True,
               'flag_bases': [30, 100]
              }
    test_view.add_method(view_name, kwargs=options)
    stack.add_link(x=X_KEYS, y=Y_KEYS, views=test_view, weights=weights)

#stack.describe().to_csv('d.csv'); stop()

VIEW_KEYS = ('x|f|x:|||cbase',
             'x|f|x:||%s|cbase' % WEIGHT,
             'x|f|x:|||cbase_gross',
             'x|f|x:||%s|cbase_gross' % WEIGHT,
             'x|f|x:|||ebase',
             'x|f|x:||%s|ebase' % WEIGHT,
             ('x|f|:||%s|counts' % WEIGHT,
              'x|f|:|y|%s|c%%' % WEIGHT,
              'x|f|:|x|%s|r%%' % WEIGHT,
              'x|t.props.Dim.80+@|:||%s|test' % WEIGHT),
             ('x|f|x[{1,2,3}]:||%s|No' % WEIGHT,
              'x|f|x[{1,2,3}]:|y|%s|No' % WEIGHT,
              'x|f|x[{1,2,3}]:|x|%s|No' % WEIGHT,
              'x|t.props.Dim.80+@|x[{1,2,3}]:||%s|test' % WEIGHT),
             ('x|f|x[{4,5,97}]:||%s|Yes' % WEIGHT,
              'x|f|x[{4,5,97}]:|y|%s|Yes' % WEIGHT,
              'x|f|x[{4,5,97}]:|x|%s|Yes' % WEIGHT,
              'x|t.props.Dim.80+@|x[{4,5,97}]:||%s|test' % WEIGHT),
             ('x|f.c:f|x[{4,5}-{1,2}]:||%s|NPSonly' % WEIGHT,
              'x|f.c:f|x[{4,5}-{1,2}]:|y|%s|NPSonly' % WEIGHT,
              'x|f.c:f|x[{4,5}-{1,2}]:|x|%s|NPSonly' % WEIGHT,
              'x|t.props.Dim.80+@|x[{4,5}-{1,2}]:||%s|test' % WEIGHT),
             ('x|f.c:f|x[{1,2}],x[{4,5}],x[{4,5}-{1,2}]:||%s|NPS' % WEIGHT,
              'x|f.c:f|x[{1,2}],x[{4,5}],x[{4,5}-{1,2}]:|y|%s|NPS' % WEIGHT,
              'x|f.c:f|x[{1,2}],x[{4,5}],x[{4,5}-{1,2}]:|x|%s|NPS' % WEIGHT,
              'x|t.props.Dim.80+@|x[{1,2}],x[{4,5}],x[{4,5}-{1,2}]:||%s|test' % WEIGHT),
             ('x|d.mean|x:||%s|stat' % WEIGHT,
              'x|t.means.Dim.80+@|x:||%s|test' % WEIGHT),
              'x|d.stddev|x:||%s|stat' % WEIGHT,
              'x|d.median|x:||%s|stat' % WEIGHT,
              'x|d.var|x:||%s|stat' % WEIGHT,
              'x|d.varcoeff|x:||%s|stat' % WEIGHT,
              'x|d.sem|x:||%s|stat' % WEIGHT,
              'x|d.lower_q|x:||%s|stat' % WEIGHT,
              'x|d.upper_q|x:||%s|stat' % WEIGHT,
             ('x|f.c:f|x:||%s|counts_sum' % WEIGHT,
              'x|f.c:f|x:|y|%s|c%%_sum' % WEIGHT),
             #('x|f.c:f|x++:||%s|counts_cumsum' % WEIGHT,
             # 'x|f.c:f|x++:|y|%s|c%%_cumsum' % WEIGHT)
            )

if CA1:
    chains = ChainManager(stack)

    chains.get(data_key=DATA_KEY, filter_key=FILTER_KEY,
               x_keys=X_KEYS[:-1], y_keys=Y_KEYS,
               views=VIEW_KEYS, orient=ORIENT,
               )
    VIEW_KEYS = ('x|f|x:|||cbase',
                 'x|f|x:||%s|cbase' % WEIGHT,
                 'x|f|x:|||cbase_gross',
                 'x|f|x:||%s|cbase_gross' % WEIGHT,
                 'x|f|x:|||ebase',
                 'x|f|x:||%s|ebase' % WEIGHT,
                 ('x|f|x[{1,2}+],x[{4,5}+]*:||%s|BLOCK' % WEIGHT,
                  'x|f|x[{1,2}+],x[{4,5}+]*:|y|%s|BLOCK' % WEIGHT,
                  'x|f|x[{1,2}+],x[{4,5}+]*:|x|%s|BLOCK' % WEIGHT,
                  'x|t.props.Dim.80+@|x[{1,2}+],x[{4,5}+]*:||%s|test' % WEIGHT),
                 #('x|d.mean|x:||%s|stat' % WEIGHT,
                 # 'x|t.means.Dim.80+@|x:||%s|test' % WEIGHT),
                 # 'x|d.stddev|x:||%s|stat' % WEIGHT,
                 # 'x|d.median|x:||%s|stat' % WEIGHT,
                 # 'x|d.var|x:||%s|stat' % WEIGHT,
                 # 'x|d.varcoeff|x:||%s|stat' % WEIGHT,
                 # 'x|d.sem|x:||%s|stat' % WEIGHT,
                 # 'x|d.lower_q|x:||%s|stat' % WEIGHT,
                 # 'x|d.upper_q|x:||%s|stat' % WEIGHT,
                 ('x|f.c:f|x:||%s|counts_sum' % WEIGHT,
                  'x|f.c:f|x:|y|%s|c%%_sum' % WEIGHT),
                 #('x|f.c:f|x++:||%s|counts_cumsum' % WEIGHT,
                 # 'x|f.c:f|x++:|y|%s|c%%_cumsum' % WEIGHT)
                )

    chains.get(data_key=DATA_KEY, filter_key=FILTER_KEY,
               x_keys=X_KEYS[-1], y_keys=Y_KEYS,
               views=VIEW_KEYS, orient=ORIENT,
               )

    chains.paint_all(transform_tests='full',
                     text_key='fake',
                     text_loc_x='x edits',
                     text_loc_y='y edits',
                    )

    # how to attach to single chain
    # 1. Add to tables
    # 2. Add formatting
    chains[0].annotations.set('Headder Title -- no reason', category='header', position='title')
    chains[0].annotations.set('Header Left -- explanation text', category='header', position='left')
    chains[0].annotations.set('Header Center -- mask text', category='header', position='center')
    chains[0].annotations.set('Notes -- base text', category='notes')
    chains[1].annotations.set('Header Center -- mask text', category='header', position='center')
    chains[1].annotations.set('Notes -- base text', category='notes')
    chains[2].annotations.set('Headder Title -- no reason', category='header', position='title')

# ------------------------------------------------------------ dataframe
if OEC:
    open_ends = data.loc[:, ['RecordNo', 'gender', 'age', 'q8', 'q8a', 'q9', 'q9a']]
    open_chain = ChainManager(stack)
    open_chain = open_chain.add(open_ends,
                                meta_from=(DATA_KEY, FILTER_KEY),
                                name='Open Ends')
    #open_chain.paint_all(text_key='en-GB', sep='. ', na_rep='__NA__')
    open_chain.paint_all(text_key='en-GB', sep='. ', na_rep='-')

    #open_ends = data.loc[:, ['RecordNo', 'gender', 'age', 'q2']]
    #open_chain = open_chain.add(open_ends,
    #                            meta_from=(DATA_KEY, FILTER_KEY),
    #                            )
    #
    #for x in iter(open_chain):
    #    print '\n', x

    #open_chain.paint_all(text_key='en-GB', sep='. ')
    #
    #for x in iter(open_chain):
    #    print '\n', x
# ------------------------------------------------------------

# ------------------------------------------------------------ arr. summaries
stack.add_link(x='q5', y='@', views=VIEWS, weights=weights)
stack.add_link(x='@', y='q5', views=VIEWS, weights=weights)

nets_mapper = qp.ViewMapper(template={'method': qp.QuantipyViews().frequency,
                                      'kwargs': {'iterators': {'rel_to': rel_to},
                                                 'groups': 'Nets'}})
nets_mapper.add_method(name='No', kwargs={'axis':    'x',
                                          'logic':   [{'No': [1, 2, 3]}],
                                          'text':    'Net: No',
                                          'combine': False})
stack.add_link(x='q5', y='@', views=nets_mapper, weights=weights)
stack.add_link(x='@', y='q5', views=nets_mapper, weights=weights)

nets_mapper = qp.ViewMapper(template={'method': qp.QuantipyViews().frequency,
                                      'kwargs': {'iterators': {'rel_to': rel_to},
                                                 'groups': 'Nets'}})
nets_mapper.add_method(name='Yes', kwargs={'axis':    'x',
                                           'logic':   [{'Yes': [4, 5, 97]}],
                                           'text':    'Net: Yes',
                                           'combine': False})
stack.add_link(x='q5', y='@', views=nets_mapper, weights=weights)
stack.add_link(x='@', y='q5', views=nets_mapper, weights=weights)

kwargs = {'calc_only': False,
          'calc': {'text': {u'en-GB': u'Net YES'},
          'Net agreement': ('Net: Yes', sub, 'Net: No')},
          'axis': 'x',
          'logic': [{'text': {u'en-GB': 'Net: No'}, 'Net: No': [1, 2]},
                    {'text': {u'en-GB': 'Net: Yes'}, 'Net: Yes': [4, 5]}]}
nets_mapper.add_method(name='NPS', kwargs=kwargs)
kwargs = {'calc_only': True,
          'calc': {'text': {u'en-GB': u'Net YES'},
          'Net agreement (only)': ('Net: Yes', sub, 'Net: No')},
          'axis': 'x',
          'logic': [{'text': {u'en-GB': 'Net: No'}, 'Net: No': [1, 2]},
                    {'text': {u'en-GB': 'Net: Yes'}, 'Net: Yes': [4, 5]}]}
nets_mapper.add_method(name='NPSonly', kwargs=kwargs)
stack.add_link(x='q5', y='@', views=nets_mapper, weights=weights)
stack.add_link(x='@', y='q5', views=nets_mapper, weights=weights)

stats = ['mean', 'stddev', 'median', 'var', 'varcoeff', 'sem', 'lower_q', 'upper_q']

for stat in stats:
    options = {'stats': stat,
               'source': None,
               'rescale': None,
               'drop': False,
               'exclude': None,
               'axis': 'x',
               'text': ''}
    view = qp.ViewMapper()
    view.make_template('descriptives')
    view.add_method('stat', kwargs=options)
    stack.add_link(x='@', y='q5', views=view, weights=weights)
    stack.add_link(x='q5', y='@', views=view, weights=weights)

nets_mapper = qp.ViewMapper(template={'method': qp.QuantipyViews().frequency,
                                      'kwargs': {'iterators': {'rel_to': rel_to},
                                                 'groups': 'Nets'}})
nets = [{'N1': [1, 2], 'text': {'en-GB': 'Waves 1 & 2 (NET)'}, 'expand': 'after'},
        {'N2': [4, 5], 'text': {'en-GB': 'Waves 4 & 5 (NET)'}, 'expand': 'after'}]
nets_mapper.add_method(name='BLOCK', kwargs={'axis':      'x',
                                             'logic':     nets,
                                             'text':      'Net: ',
                                             'combine':   False,
                                             'complete':  True,
                                             'expand':    'after'}
                                             )
stack.add_link(x='q5', y='@', views=nets_mapper, weights=weights)
stack.add_link(x='@', y='q5', views=nets_mapper, weights=weights)

VIEW_KEYS = ('x|f|x:|||cbase',
             'x|f|x:||%s|cbase' % WEIGHT,
             ('x|f|:||%s|counts' % WEIGHT,
              'x|f|:|y|%s|c%%' % WEIGHT),
             ('x|f|x[{1,2,3}]:||%s|No' % WEIGHT,
              'x|f|x[{1,2,3}]:|y|%s|No' % WEIGHT),
              #'x|f|x[{1,2,3}]:|x|%s|No' % WEIGHT),
             ('x|f|x[{4,5,97}]:||%s|Yes' % WEIGHT,
              'x|f|x[{4,5,97}]:|y|%s|Yes' % WEIGHT),
              #'x|f|x[{4,5,97}]:|x|%s|Yes' % WEIGHT),
             ('x|f.c:f|x[{4,5}-{1,2}]:||%s|NPSonly' % WEIGHT,
              'x|f.c:f|x[{4,5}-{1,2}]:|y|%s|NPSonly' % WEIGHT),
              #'x|f.c:f|x[{4,5}-{1,2}]:|x|%s|NPSonly' % WEIGHT),
             ('x|f.c:f|x[{1,2}],x[{4,5}],x[{4,5}-{1,2}]:||%s|NPS' % WEIGHT,
              'x|f.c:f|x[{1,2}],x[{4,5}],x[{4,5}-{1,2}]:|y|%s|NPS' % WEIGHT),
              #'x|f.c:f|x[{1,2}],x[{4,5}],x[{4,5}-{1,2}]:|x|%s|NPS' % WEIGHT),
             'x|d.mean|x:||%s|stat' % WEIGHT,
             'x|d.stddev|x:||%s|stat' % WEIGHT,
             'x|d.median|x:||%s|stat' % WEIGHT,
             'x|d.var|x:||%s|stat' % WEIGHT,
             'x|d.varcoeff|x:||%s|stat' % WEIGHT,
             'x|d.sem|x:||%s|stat' % WEIGHT,
             'x|d.lower_q|x:||%s|stat' % WEIGHT,
             'x|d.upper_q|x:||%s|stat' % WEIGHT,
             ('x|f.c:f|x:||%s|counts_sum' % WEIGHT,
              'x|f.c:f|x:|y|%s|c%%_sum' % WEIGHT),
            )

#VIEW_KEYS = ('x|f|x:|||cbase',
#             'x|f|x:||%s|cbase' % WEIGHT,
#             'x|f|:|y|%s|c%%' % WEIGHT,
#             'x|f|x[{1,2,3}]:|y|%s|No' % WEIGHT,
#             'x|f|x[{4,5,97}]:|y|%s|Yes' % WEIGHT,
#             'x|f.c:f|x[{4,5}-{1,2}]:|y|%s|NPSonly' % WEIGHT,
#             'x|f.c:f|x[{1,2}],x[{4,5}],x[{4,5}-{1,2}]:|y|%s|NPS' % WEIGHT,
#             'x|d.mean|x:||%s|stat' % WEIGHT,
#             'x|d.stddev|x:||%s|stat' % WEIGHT,
#             'x|d.median|x:||%s|stat' % WEIGHT,
#             'x|d.var|x:||%s|stat' % WEIGHT,
#             'x|d.varcoeff|x:||%s|stat' % WEIGHT,
#             'x|d.sem|x:||%s|stat' % WEIGHT,
#             'x|d.lower_q|x:||%s|stat' % WEIGHT,
#             'x|d.upper_q|x:||%s|stat' % WEIGHT,
#            )

if AC1:
    arr_chains_1 = ChainManager(stack)

    arr_chains_1.get(data_key=DATA_KEY,
                   filter_key=FILTER_KEY,
                   x_keys=['@'],
                   y_keys=['q5'],
                   views=VIEW_KEYS,
                  )

    arr_chains_1.paint_all()

    arr_chains_1[0].annotations.set('Headder Title -- no reason', category='header', position='title')
    arr_chains_1[0].annotations.set('Header Left -- explanation text', category='header', position='left')
    arr_chains_1[0].annotations.set('Header Center -- mask text', category='header', position='center')
    arr_chains_1[0].annotations.set('Notes -- base text', category='notes')

if AC0:
    arr_chains_0 = ChainManager(stack)

    arr_chains_0.get(data_key=DATA_KEY,
                   filter_key=FILTER_KEY,
                   x_keys=['q5'],
                   y_keys=['@'],
                   views=VIEW_KEYS,
                  )
    arr_chains_0.paint_all()

    arr_chains_0[0].annotations.set('Headder Title -- no reason', category='header', position='title')
    arr_chains_0[0].annotations.set('Header Left -- explanation text', category='header', position='left')
    arr_chains_0[0].annotations.set('Header Center -- mask text', category='header', position='center')
    arr_chains_0[0].annotations.set('Notes -- base text', category='notes')
# ------------------------------------------------------------

# ------------------------------------------------------------ arr. summaries - block nets

VIEW_KEYS = ('x|f|x:|||cbase',
             'x|f|x:||%s|cbase' % WEIGHT,
             ('x|f|x[{1,2}+],x[{4,5}+]*:||%s|BLOCK' % WEIGHT,
              'x|f|x[{1,2}+],x[{4,5}+]*:|y|%s|BLOCK' % WEIGHT),
             'x|d.mean|x:||%s|stat' % WEIGHT,
             'x|d.stddev|x:||%s|stat' % WEIGHT,
             'x|d.median|x:||%s|stat' % WEIGHT,
             'x|d.var|x:||%s|stat' % WEIGHT,
             'x|d.varcoeff|x:||%s|stat' % WEIGHT,
             'x|d.sem|x:||%s|stat' % WEIGHT,
             'x|d.lower_q|x:||%s|stat' % WEIGHT,
             'x|d.upper_q|x:||%s|stat' % WEIGHT,
             ('x|f.c:f|x:||%s|counts_sum' % WEIGHT,
              'x|f.c:f|x:|y|%s|c%%_sum' % WEIGHT),
             #('x|f.c:f|x++:||%s|counts_cumsum' % WEIGHT,
             # 'x|f.c:f|x++:|y|%s|c%%_cumsum' % WEIGHT)
            )

if ACB1:
    arr_chains_block_1 = ChainManager(stack)
    arr_chains_block_1.get(data_key=DATA_KEY,
                           filter_key=FILTER_KEY,
                           x_keys=['@'],
                           y_keys=['q5'],
                           views=VIEW_KEYS,
                          )
    arr_chains_block_1.paint_all()

if ACB0:
    arr_chains_block_0 = ChainManager(stack)
    arr_chains_block_0.get(data_key=DATA_KEY,
                           filter_key=FILTER_KEY,
                           x_keys=['q5'],
                           y_keys=['@'],
                           views=VIEW_KEYS,
                          )
    arr_chains_block_0.paint_all()
# ------------------------------------------------------------

# ------------------------------------------------------------ arr. summaries - mean
VIEW_KEYS = ('x|f|x:|||cbase',
             'x|f|x:||%s|cbase' % WEIGHT,
             'x|d.mean|x:||%s|stat' % WEIGHT,
             'x|d.stddev|x:||%s|stat' % WEIGHT,
             'x|d.median|x:||%s|stat' % WEIGHT,
             'x|d.var|x:||%s|stat' % WEIGHT,
             'x|d.varcoeff|x:||%s|stat' % WEIGHT,
             'x|d.sem|x:||%s|stat' % WEIGHT,
             'x|d.lower_q|x:||%s|stat' % WEIGHT,
             'x|d.upper_q|x:||%s|stat' % WEIGHT,
            )

if ACM1:
    arr_chains_mean_1 = ChainManager(stack)

    arr_chains_mean_1.get(data_key=DATA_KEY,
                          filter_key=FILTER_KEY,
                          x_keys=['@'],
                          y_keys=['q5'],
                          views=VIEW_KEYS,
                         )

    arr_chains_mean_1.paint_all()

if ACM0:
    arr_chains_mean_0 = ChainManager(stack)

    arr_chains_mean_0.get(data_key=DATA_KEY,
                          filter_key=FILTER_KEY,
                          x_keys=['q5'],
                          y_keys=['@'],
                          views=VIEW_KEYS,
                         )

    arr_chains_mean_0.paint_all()
# ------------------------------------------------------------

# table props - check editability
table_properties = {
                        ### global properties

                        ### y
                        'bold_y': True,
                        'bg_color_y': '#B9FFCC',
                        'font_color_y': 'gray',
                        'font_name_y': 'Courier',
                        'font_size_y': 12,
                        'italic_y': True,
                        'text_v_align_y': 3,
                        'text_h_align_y': 1,

                        ### label
                        'bold_label': True,
                        'bg_color_label': 'red',
                        'font_color_label': '#FFB6C1',
                        'font_name_label': 'Calibri',
                        'font_size_label': 11,
                        'italic_label': True,
                        'text_v_align_label': 1,
                        'text_h_align_label': 3,

                        ### u_c_base text
                        'bold_u_c_base_text': True,
                        'bg_color_u_c_base_text': 'green',
                        'font_color_u_c_base_text': '#AB94FF',
                        'font_name_u_c_base_text': 'Helvetica',
                        'font_size_u_c_base_text': 11,
                        'italic_u_c_base_text': True,
                        'text_v_align_u_c_base_text': 3,
                        'text_h_align_u_c_base_text': 2,

                        ### u_c_base
                        'bold_u_c_base': True,
                        'bg_color_u_c_base': '#AB94FF',
                        'font_color_u_c_base': 'green',
                        'font_name_u_c_base': 'Helvetica',
                        'font_size_u_c_base': 11,
                        'italic_u_c_base': True,
                        'text_v_align_u_c_base': 3,
                        'text_h_align_u_c_base': 3,

                        ### c_base text
                        'bold_c_base_text': True,
                        'bg_color_c_base_text': '#AB94FF',
                        'font_color_c_base_text': 'green',
                        'font_name_c_base_text': 'Broadway',
                        'font_size_c_base_text': 10,
                        'italic_c_base_text': True,
                        'text_v_align_c_base_text': 1,
                        'text_h_align_c_base_text': 1,

                        ### c_base
                        'bold_c_base': True,
                        'bg_color_c_base': 'green',
                        'font_color_c_base': '#AB94FF',
                        'font_name_c_base': 'Broadway',
                        'font_size_c_base': 10,
                        'italic_c_base': True,
                        'text_v_align_c_base': 1,
                        'text_h_align_c_base': 1,

                        ### u_c_base_gross text
                        'bold_u_c_base_gross_text': True,
                        'bg_color_u_c_base_gross_text': '#DAF7A6',
                        'font_color_u_c_base_gross_text': '#7DCEA0',
                        'font_name_u_c_base_gross_text': 'Helvetica',
                        'font_size_u_c_base_gross_text': 11,
                        'italic_u_c_base_gross_text': True,
                        'text_v_align_u_c_base_gross_text': 3,
                        'text_h_align_u_c_base_gross_text': 2,

                        ### u_c_base_gross
                        'bold_u_c_base_gross': True,
                        'bg_color_u_c_base_gross': '#7DCEA0',
                        'font_color_u_c_base_gross': '#DAF7A6',
                        'font_name_u_c_base_gross': 'Helvetica',
                        'font_size_u_c_base_gross': 11,
                        'italic_u_c_base_gross': True,
                        'text_v_align_u_c_base_gross': 3,
                        'text_h_align_u_c_base_gross': 3,

                        ### c_base_gross text
                        'bold_c_base_gross_text': True,
                        'bg_color_c_base_gross_text': '#7DCEA0',
                        'font_color_c_base_gross_text': '#DAF7A6',
                        'font_name_c_base_gross_text': 'Broadway',
                        'font_size_c_base_gross_text': 10,
                        'italic_c_base_gross_text': True,
                        'text_v_align_c_base_gross_text': 1,
                        'text_h_align_c_base_gross_text': 1,

                        ### c_base_gross
                        'bold_c_base_gross': True,
                        'bg_color_c_base_gross': '#DAF7A6',
                        'font_color_c_base_gross': '#7DCEA0',
                        'font_name_c_base_gross': 'Broadway',
                        'font_size_c_base_gross': 10,
                        'italic_c_base_gross': True,
                        'text_v_align_c_base_gross': 1,
                        'text_h_align_c_base_gross': 1,

                        ### u_e_base text
                        'bold_u_e_base_text': True,
                        'bg_color_u_e_base_text': '#839192',
                        'font_color_u_e_base_text': '#E5F315',
                        'font_name_u_e_base_text': 'Helvetica',
                        'font_size_u_e_base_text': 11,
                        'italic_u_e_base_text': True,
                        'text_v_align_u_e_base_text': 3,
                        'text_h_align_u_e_base_text': 2,

                        ### u_e_base
                        'bold_u_e_base': True,
                        'bg_color_u_e_base': '#E5F315',
                        'font_color_u_e_base': '#839192',
                        'font_name_u_e_base': 'Helvetica',
                        'font_size_u_e_base': 11,
                        'italic_u_e_base': True,
                        'text_v_align_u_e_base': 3,
                        'text_h_align_u_e_base': 3,

                        ### e_base text
                        'bold_e_base_text': True,
                        'bg_color_e_base_text': '#E5F315',
                        'font_color_e_base_text': '#839192',
                        'font_name_e_base_text': 'Broadway',
                        'font_size_e_base_text': 10,
                        'italic_e_base_text': True,
                        'text_v_align_e_base_text': 1,
                        'text_h_align_e_base_text': 1,

                        ### e_base
                        'bold_e_base': True,
                        'bg_color_e_base': '#839192',
                        'font_color_e_base': '#E5F315',
                        'font_name_e_base': 'Broadway',
                        'font_size_e_base': 10,
                        'italic_e_base': True,
                        'text_v_align_e_base': 1,
                        'text_h_align_e_base': 1,

                        ### counts text
                        'bold_counts_text': True,
                        'bg_color_counts_text': '#8B4513',
                        'font_color_counts_text': '#CD853F',
                        'font_name_counts_text': 'FreeSerif',
                        'font_size_counts_text': 13,
                        'italic_counts_text': True,
                        'text_v_align_counts_text': 3,
                        'text_h_align_counts_text': 3,

                        ### counts
                        'bold_counts': True,
                        'bg_color_counts': '#CD853F',
                        'font_color_counts': '#8B4513',
                        'font_name_counts': 'FreeSerif',
                        'font_size_counts': 12,
                        'italic_counts': True,
                        'text_v_align_counts': 3,
                        'text_h_align_counts': 3,

                        'view_border_counts': None, # experimental

                        ### c_pct text
                        'bold_c_pct_text': True,
                        'bg_color_c_pct_text': '#CD853F',
                        'font_color_c_pct_text': '#8B4513',
                        'font_name_c_pct_text': 'FreeSerif',
                        'font_size_c_pct_text': 12,
                        'italic_c_pct_text': True,
                        'text_v_align_c_pct_text': 1,
                        'text_h_align_c_pct_text': 1,

                        ### c_pct
                        'bold_c_pct': True,
                        'bg_color_c_pct': '#8B4513',
                        'font_color_c_pct': '#CD853F',
                        'font_name_c_pct': 'FreeSerif',
                        'font_size_c_pct': 13,
                        'italic_c_pct': True,
                        'text_v_align_c_pct': 1,
                        'text_h_align_c_pct': 1,

                        ### r_pct text
                        'bold_r_pct_text': True,
                        'bg_color_r_pct_text': '#8B4513',
                        'font_text_color_r_pct': '#CD853F',
                        'font_text_name_r_pct': 'FreeSerif',
                        'font_size_r_pct_text': 12,
                        'it_textalic_r_pct': True,
                        'text_v_align_r_pct_text': 1,
                        'text_h_align_r_pct_text': 1,

                        ### r_pct
                        'bold_r_pct': True,
                        'bg_color_r_pct':  '#CD853F',
                        'font_color_r_pct': '#8B4513',
                        'font_name_r_pct': 'FreeSerif',
                        'font_size_r_pct': 13,
                        'italic_r_pct': True,
                        'text_v_align_r_pct': 1,
                        'text_h_align_r_pct': 1,

                        ### propstest text
                        'bold_propstest_text': True,
                        'bg_color_propstest_text': '#98FB98',
                        'font_color_propstest_text': '#7DCEA0',
                        'font_name_propstest_text': 'Liberation Sans Narrow',
                        'font_size_propstest_text': 11,
                        'italic_propstest_text': True,
                        'text_v_align_propstest_text': 1,
                        'text_h_align_propstest_text': 1,

                        ### propstest
                        'bold_propstest': True,
                        'bg_color_propstest': '#7DCEA0',
                        'font_color_propstest': '#98FB98',
                        'font_name_propstest': 'Liberation Sans Narrow',
                        'font_size_propstest': 10,
                        'italic_propstest': True,
                        'text_v_align_propstest': 1,
                        'text_h_align_propstest': 3,

                        ### net counts text
                        'bold_net_counts_text': True,
                        'bg_color_net_counts_text': '#B2DFEE',
                        'font_color_net_counts_text': '#FF5733',
                        'font_name_net_counts_text': 'Century Schoolbook L',
                        'font_size_net_counts_text': 11,
                        'italic_net_counts_text': True,
                        'text_v_align_net_counts_text': 1,
                        'text_h_align_net_counts_text': 1,

                        ### net counts
                        'bold_net_counts': True,
                        'bg_color_net_counts': '#B2DFEE',
                        'font_color_net_counts': '#FF5733',
                        'font_name_net_counts': 'Century Schoolbook L',
                        'font_size_net_counts': 13,
                        'italic_net_counts': True,
                        'text_v_align_net_counts': 1,
                        'text_h_align_net_counts': 1,

                        ### net c pct text
                        'bold_net_c_pct_text': True,
                        'bg_color_net_c_pct_text': '#FF5733',
                        'font_color_net_c_pct_text': '#B2DFEE',
                        'font_name_net_c_pct_text': 'Century Schoolbook L',
                        'font_size_net_c_pct_text': 11,
                        'italic_net_c_pct_text': True,
                        'text_v_align_net_c_pct_text': 1,
                        'text_h_align_net_c_pct_text': 1,

                        ### net c pct
                        'bold_net_c_pct': True,
                        'bg_color_net_c_pct': '#FF5733',
                        'font_color_net_c_pct': '#B2DFEE',
                        'font_name_net_c_pct': 'Century Schoolbook L',
                        'font_size_net_c_pct': 13,
                        'italic_net_c_pct': True,
                        'text_v_align_net_c_pct': 1,
                        'text_h_align_net_c_pct': 1,

                        ### net r pct text
                        'bold_net_r_pct_text': True,
                        'bg_color_net_r_pct_text': '#B2DFEE',
                        'font_color_net_r_pct_text': '#FF5733',
                        'font_name_net_r_pct_text': 'Century Schoolbook L',
                        'font_size_net_r_pct_text': 11,
                        'italic_net_r_pct_text': True,
                        'text_v_align_net_r_pct_text': 1,
                        'text_h_align_net_r_pct_text': 1,

                        ### net r pct
                        'bold_net_r_pct': True,
                        'bg_color_net_r_pct': '#B2DFEE',
                        'font_color_net_r_pct': '#FF5733',
                        'font_name_net_r_pct': 'Century Schoolbook L',
                        'font_size_net_r_pct': 13,
                        'italic_net_r_pct': True,
                        'text_v_align_net_r_pct': 1,
                        'text_h_align_net_r_pct': 1,

                        ### net_propstest text
                        'bold_net_propstest_text': True,
                        'bg_color_net_propstest_text': '#FF5733',
                        'font_color_net_propstest_text': '#B2DFEE',
                        'font_name_net_propstest_text': 'Century Schoolbook L',
                        'font_size_net_propstest_text': 11,
                        'italic_net_propstest_text': True,
                        'text_v_align_net_propstest_text': 1,
                        'text_h_align_net_propstest_text': 1,

                        ### net_propstest
                        'bold_net_propstest': True,
                        'bg_color_net_propstest': '#FF5733',
                        'font_color_net_propstest': '#B2DFEE',
                        'font_name_net_propstest': 'Century Schoolbook L',
                        'font_size_net_propstest': 13,
                        'italic_net_propstest': True,
                        'text_v_align_net_propstest': 1,
                        'text_h_align_net_propstest': 1,

                        ### block_calc_net_counts text
                        'bold_block_calc_net_counts_text': True,
                        'bg_color_block_calc_net_counts_text': '#839192',
                        'font_color_block_calc_net_counts_text': '#F8C471F',
                        'font_name_block_calc_net_counts_text': 'Century Schoolbook L',
                        'font_size_block_calc_net_counts_text': 11,
                        'italic_block_calc_net_counts_text': True,
                        'text_v_align_block_calc_net_counts_text': 1,
                        'text_h_align_block_calc_net_counts_text': 1,

                        ###block_calc_net_counts
                        'bold_block_calc_net_counts': True,
                        'bg_color_block_calc_net_counts': '#F8C471F',
                        'font_color_block_calc_net_counts': '#839192',
                        'font_name_block_calc_net_counts': 'Century Schoolbook L',
                        'font_size_block_calc_net_counts': 13,
                        'italic_block_calc_net_counts': True,
                        'text_v_align_block_calc_net_counts': 1,
                        'text_h_align_block_calc_net_counts': 1,

                        ### block_calc_net_c_pct text
                        'bold_block_calc_net_c_pct_text': True,
                        'bg_color_block_calc_net_c_pct_text': '#F8C471F',
                        'font_color_block_calc_net_c_pct_text': '#839192',
                        'font_name_block_calc_net_c_pct_text': 'Century Schoolbook L',
                        'font_size_block_calc_net_c_pct_text': 11,
                        'italic_block_calc_net_c_pct_text': True,
                        'text_v_align_block_calc_net_c_pct_text': 1,
                        'text_h_align_block_calc_net_c_pct_text': 1,

                        ### block_calc_net_c_pct
                        'bold_block_calc_net_c_pct': True,
                        'bg_color_block_calc_net_c_pct': '#839192',
                        'font_color_block_calc_net_c_pct': '#F8C471F',
                        'font_name_block_calc_net_c_pct': 'Century Schoolbook L',
                        'font_size_block_calc_net_c_pct': 13,
                        'italic_block_calc_net_c_pct': True,
                        'text_v_align_block_calc_net_c_pct': 1,
                        'text_h_align_block_calc_net_c_pct': 1,

                        ### block_calc_net_r_pct text
                        'bold_block_calc_net_r_pct_text': True,
                        'bg_color_block_calc_net_r_pct_text': '#839192',
                        'font_color_block_calc_net_r_pct_text': '#F8C471F',
                        'font_name_block_calc_net_r_pct_text': 'Century Schoolbook L',
                        'font_size_block_calc_net_r_pct_text': 11,
                        'italic_block_calc_net_r_pct_text': True,
                        'text_v_align_block_calc_net_r_pct_text': 1,
                        'text_h_align_block_calc_net_r_pct_text': 1,

                        ### block_calc_net_r_pct
                        'bold_block_calc_net_r_pct': True,
                        'bg_color_block_calc_net_r_pct': '#F8C471F',
                        'font_color_block_calc_net_r_pct': '#839192',
                        'font_name_block_calc_net_r_pct': 'Century Schoolbook L',
                        'font_size_block_calc_net_r_pct': 13,
                        'italic_block_calc_net_r_pct': True,
                        'text_v_align_block_calc_net_r_pct': 1,
                        'text_h_align_block_calc_net_r_pct': 1,

                        ### block_calc_net_propstest text
                        'bold_block_calc_net_propstest_text': True,
                        'bg_color_block_calc_net_propstest_text': '#F8C471F',
                        'font_color_block_calc_net_propstest_text': '#839192',
                        'font_name_block_calc_net_propstest_text': 'Century Schoolbook L',
                        'font_size_block_calc_net_propstest_text': 11,
                        'italic_block_calc_net_propstest_text': True,
                        'text_v_align_block_calc_net_propstest_text': 1,
                        'text_h_align_block_calc_net_propstest_text': 1,

                        ### block_calc_net_propstest
                        'bold_block_calc_net_propstest': True,
                        'bg_color_block_calc_net_propstest': '#839192',
                        'font_color_block_calc_net_propstest': '#F8C471F',
                        'font_name_block_calc_net_propstest': 'Century Schoolbook L',
                        'font_size_block_calc_net_propstest': 13,
                        'italic_block_calc_net_propstest': True,
                        'text_v_align_block_calc_net_propstest': 1,
                        'text_h_align_block_calc_net_propstest': 1,

                        ### block_calc_counts text
                        'bold_block_calc_counts_text': True,
                        'bg_color_block_calc_counts_text': 'blue',
                        'font_color_block_calc_counts_text': 'red',
                        'font_name_block_calc_counts_text': 'Century Schoolbook L',
                        'font_size_block_calc_counts_text': 11,
                        'italic_block_calc_counts_text': True,
                        'text_v_align_block_calc_counts_text': 1,
                        'text_h_align_block_calc_counts_text': 1,

                        ###block_calc_counts
                        'bold_block_calc_counts': True,
                        'bg_color_block_calc_counts': 'red',
                        'font_color_block_calc_counts': 'blue',
                        'font_name_block_calc_counts': 'Century Schoolbook L',
                        'font_size_block_calc_counts': 13,
                        'italic_block_calc_counts': True,
                        'text_v_align_block_calc_counts': 1,
                        'text_h_align_block_calc_counts': 1,

                        ### block_calc_c_pct text
                        'bold_block_calc_c_pct_text': True,
                        'bg_color_block_calc_c_pct_text': 'red',
                        'font_color_block_calc_c_pct_text': 'blue',
                        'font_name_block_calc_c_pct_text': 'Century Schoolbook L',
                        'font_size_block_calc_c_pct_text': 11,
                        'italic_block_calc_c_pct_text': True,
                        'text_v_align_block_calc_c_pct_text': 1,
                        'text_h_align_block_calc_c_pct_text': 1,

                        ### block_calc_c_pct
                        'bold_block_calc_c_pct': True,
                        'bg_color_block_calc_c_pct': 'blue',
                        'font_color_block_calc_c_pct': 'red',
                        'font_name_block_calc_c_pct': 'Century Schoolbook L',
                        'font_size_block_calc_c_pct': 13,
                        'italic_block_calc_c_pct': True,
                        'text_v_align_block_calc_c_pct': 1,
                        'text_h_align_block_calc_c_pct': 1,

                        ### block_calc_r_pct text
                        'bold_block_calc_r_pct_text': True,
                        'bg_color_block_calc_r_pct_text': 'blue',
                        'font_color_block_calc_r_pct_text': 'red',
                        'font_name_block_calc_r_pct_text': 'Century Schoolbook L',
                        'font_size_block_calc_r_pct_text': 11,
                        'italic_block_calc_r_pct_text': True,
                        'text_v_align_block_calc_r_pct_text': 1,
                        'text_h_align_block_calc_r_pct_text': 1,

                        ### block_calc_r_pct
                        'bold_block_calc_r_pct': True,
                        'bg_color_block_calc_r_pct': 'red',
                        'font_color_block_calc_r_pct': 'blue',
                        'font_name_block_calc_r_pct': 'Century Schoolbook L',
                        'font_size_block_calc_r_pct': 13,
                        'italic_block_calc_r_pct': True,
                        'text_v_align_block_calc_r_pct': 1,
                        'text_h_align_block_calc_r_pct': 1,

                        ### block_calc_propstest text
                        'bold_block_calc_propstest_text': True,
                        'bg_color_block_calc_propstest_text': 'red',
                        'font_color_block_calc_propstest_text': 'blue',
                        'font_name_block_calc_propstest_text': 'Century Schoolbook L',
                        'font_size_block_calc_propstest_text': 11,
                        'italic_block_calc_propstest_text': True,
                        'text_v_align_block_calc_propstest_text': 1,
                        'text_h_align_block_calc_propstest_text': 1,

                        ### block_calc_propstest
                        'bold_block_calc_propstest': True,
                        'bg_color_block_calc_propstest': 'blue',
                        'font_color_block_calc_propstest': 'red',
                        'font_name_block_calc_propstest': 'Century Schoolbook L',
                        'font_size_block_calc_propstest': 13,
                        'italic_block_calc_propstest': True,
                        'text_v_align_block_calc_propstest': 1,
                        'text_h_align_block_calc_propstest': 1,

                        ### block_net text
                        'bold_block_net_text': True,
                        'bg_color_block_net_text': '#15F3BB',
                        'font_color_block_net_text': '#F31588',
                        'font_name_block_net_text': 'Century Schoolbook L',
                        'font_size_block_net_text': 11,
                        'italic_block_net_text': True,
                        'text_v_align_block_net_text': 1,
                        'text_h_align_block_net_text': 1,

                        ### block_net
                        'bold_block_net': True,
                        'bg_color_block_net': '#F31588',
                        'font_color_block_net': '#15F3BB',
                        'font_name_block_net': 'Century Schoolbook L',
                        'font_size_block_net': 13,
                        'italic_block_net': True,
                        'text_v_align_block_net': 1,
                        'text_h_align_block_net': 1,

                        ### block_expanded text
                        'bold_block_expanded_text': True,
                        'bg_color_block_expanded_text': '#F08080',
                        'font_color_block_expanded_text': '#FCF3CF',
                        'font_name_block_expanded_text': 'Century Schoolbook L',
                        'font_size_block_expanded_text': 11,
                        'italic_block_expanded_text': True,
                        'text_v_align_block_expanded_text': 1,
                        'text_h_align_block_expanded_text': 1,

                        ### blockexpanded_
                        'bold_block_expanded': True,
                        'bg_color_block_expanded': '#FCF3CF',
                        'font_color_block_expanded': '#F08080',
                        'font_name_block_expanded': 'Century Schoolbook L',
                        'font_size_block_expanded': 13,
                        'italic_block_expanded': True,
                        'text_v_align_block_expanded': 1,
                        'text_h_align_block_expanded': 1,

                        ### block_normal text
                        'bold_block_normal_text': True,
                        'bg_color_block_normal_text': '#00BFFF',
                        'font_color_block_normal_text': '#F08080',
                        'font_name_block_normal_text': 'Century Schoolbook L',
                        'font_size_block_normal_text': 11,
                        'italic_block_normal_text': True,
                        'tnormalign_block_expanded_text': 1,
                        'tnormalign_block_expanded_text': 1,

                        ### block_normal
                        'bold_block_normal': True,
                        'bg_color_block_normal': '#F08080',
                        'font_color_block_normal': '#00BFFF',
                        'font_name_block_normal': 'Century Schoolbook L',
                        'font_size_block_normal': 13,
                        'italic_block_normal': True,
                        'text_v_align_block_normal': 1,
                        'text_h_align_block_normal': 1,

                        ### mean text
                        'bold_mean_text': True,
                        'bg_color_mean_text': '#FF69B4',
                        'font_color_mean_text': '#00E5EE',
                        'font_name_mean_text': 'MathJax_SanSerif',
                        'font_size_mean_text': 13,
                        'italic_mean_text': True,
                        'text_v_align_mean_text': 3,
                        'text_h_align_mean_text': 3,

                        ### mean
                        'bold_mean': True,
                        'bg_color_mean': '#FF69B4',
                        'font_color_mean': '#00E5EE',
                        'font_name_mean': 'MathJax_SanSerif',
                        'font_size_mean': 11,
                        'italic_mean': True,
                        'text_v_align_mean': 3,
                        'text_h_align_mean': 3,

                        ### stddev text
                        'bold_stddev_text': True,
                        'bg_color_stddev_text': '#FF69B4',
                        'font_color_stddev_text': '#00E5EE',
                        'font_name_stddev_text': 'MathJax_SanSerif',
                        'font_size_stddev_text': 13,
                        'italic_stddev_text': True,
                        'text_v_align_stddev_text': 3,
                        'text_h_align_stddev_text': 3,

                        ### stddev
                        'bold_stddev': True,
                        'bg_color_stddev': '#FF69B4',
                        'font_color_stddev': '#00E5EE',
                        'font_name_stddev': 'MathJax_SanSerif',
                        'font_size_stddev': 11,
                        'italic_stddev': True,
                        'text_v_align_stddev': 3,
                        'text_h_align_stddev': 3,

                        ### median text
                        'bold_median_text': True,
                        'bg_color_median_text': '#FF69B4',
                        'font_color_median_text': '#00E5EE',
                        'font_name_median_text': 'MathJax_SanSerif',
                        'font_size_median_text': 13,
                        'italic_median_text': True,
                        'text_v_align_median_text': 3,
                        'text_h_align_median_text': 3,

                        ### median
                        'bold_median': True,
                        'bg_color_median': '#FF69B4',
                        'font_color_median': '#00E5EE',
                        'font_name_median': 'MathJax_SanSerif',
                        'font_size_median': 11,
                        'italic_median': True,
                        'text_v_align_median': 3,
                        'text_h_align_median': 3,

                        ### var text
                        'bold_var_text': True,
                        'bg_color_var_text': '#FF69B4',
                        'font_color_var_text': '#00E5EE',
                        'font_name_var_text': 'MathJax_SanSerif',
                        'font_size_var_text': 13,
                        'italic_var_text': True,
                        'text_v_align_var_text': 3,
                        'text_h_align_var_text': 3,

                        ### var
                        'bold_var': True,
                        'bg_color_var': '#FF69B4',
                        'font_color_var': '#00E5EE',
                        'font_name_var': 'MathJax_SanSerif',
                        'font_size_var': 11,
                        'italic_var': True,
                        'text_v_align_var': 3,
                        'text_h_align_var': 3,

                        ### varcoeff text
                        'bold_varcoeff_text': True,
                        'bg_color_varcoeff_text': '#FF69B4',
                        'font_color_varcoeff_text': '#00E5EE',
                        'font_name_varcoeff_text': 'MathJax_SanSerif',
                        'font_size_varcoeff_text': 13,
                        'italic_varcoeff_text': True,
                        'text_v_align_varcoeff_text': 3,
                        'text_h_align_varcoeff_text': 3,

                        ### varcoeff
                        'bold_varcoeff': True,
                        'bg_color_varcoeff': '#FF69B4',
                        'font_color_varcoeff': '#00E5EE',
                        'font_name_varcoeff': 'MathJax_SanSerif',
                        'font_size_varcoeff': 11,
                        'italic_varcoeff': True,
                        'text_v_align_varcoeff': 3,
                        'text_h_align_varcoeff': 3,

                        ### sem text
                        'bold_sem_text': True,
                        'bg_color_sem_text': '#FF69B4',
                        'font_color_sem_text': '#00E5EE',
                        'font_name_sem_text': 'MathJax_SanSerif',
                        'font_size_sem_text': 13,
                        'italic_sem_text': True,
                        'text_v_align_sem_text': 3,
                        'text_h_align_sem_text': 3,

                        ### sem
                        'bold_sem': True,
                        'bg_color_sem': '#FF69B4',
                        'font_color_sem': '#00E5EE',
                        'font_name_sem': 'MathJax_SanSerif',
                        'font_size_sem': 11,
                        'italic_sem': True,
                        'text_v_align_sem': 3,
                        'text_h_align_sem': 3,

                        ### lower_q text
                        'bold_lower_q_text': True,
                        'bg_color_lower_q_text': '#FF69B4',
                        'font_color_lower_q_text': '#00E5EE',
                        'font_name_lower_q_text': 'MathJax_SanSerif',
                        'font_size_lower_q_text': 13,
                        'italic_lower_q_text': True,
                        'text_v_align_lower_q_text': 3,
                        'text_h_align_lower_q_text': 3,

                        ### lower_q
                        'bold_lower_q': True,
                        'bg_color_lower_q': '#FF69B4',
                        'font_color_lower_q': '#00E5EE',
                        'font_name_lower_q': 'MathJax_SanSerif',
                        'font_size_lower_q': 11,
                        'italic_lower_q': True,
                        'text_v_align_lower_q': 3,
                        'text_h_align_lower_q': 3,

                        ### upper_q text
                        'bold_upper_q_text': True,
                        'bg_color_upper_q_text': '#FF69B4',
                        'font_color_upper_q_text': '#00E5EE',
                        'font_name_upper_q_text': 'MathJax_SanSerif',
                        'font_size_upper_q_text': 13,
                        'italic_upper_q_text': True,
                        'text_v_align_upper_q_text': 3,
                        'text_h_align_upper_q_text': 3,

                        ###upper_q
                        'bold_upper_q': True,
                        'bg_color_upper_q': '#FF69B4',
                        'font_color_upper_q': '#00E5EE',
                        'font_name_upper_q': 'MathJax_SanSerif',
                        'font_size_upper_q': 11,
                        'italic_upper_q': True,
                        'text_v_align_upper_q': 3,
                        'text_h_align_upper_q': 3,

                        ### meanstest text
                        'bold_meanstest_text': True,
                        'bg_color_meanstest_text': '#00E5EE',
                        'font_color_meanstest_text': '#FF69B4',
                        'font_name_meanstest_text': 'MathJax_SanSerif',
                        'font_size_meanstest_text': 11,
                        'italic_meanstest_text': True,
                        'text_v_align_meanstest_text': 3,
                        'text_h_align_meanstest_text': 3,

                        ### meanstest
                        'bold_meanstest': True,
                        'bg_color_meanstest': '#00E5EE',
                        'font_color_meanstest': '#FF69B4',
                        'font_name_meanstest': 'MathJax_SanSerif',
                        'font_size_meanstest': 13,
                        'italic_meanstest': True,
                        'text_v_align_meanstest': 3,
                        'text_h_align_meanstest': 3,

                        ### counts_sum text
                        'bold_counts_sum_text': True,
                        'bg_color_counts_sum_text': '#34495E',
                        'font_color_counts_sum_text': '#D4AC0D',
                        'font_name_counts_sum_text': 'URW Gothic L',
                        'font_size_counts_sum_text': 8,
                        'italic_counts_sum_text': True,
                        'text_v_align_counts_sum_text': 1,
                        'text_h_align_counts_sum_text': 1,

                        ### counts_sum
                        'bold_counts_sum': True,
                        'bg_color_counts_sum': '#34495E',
                        'font_color_counts_sum': '#D4AC0D',
                        'font_name_counts_sum': 'URW Gothic L',
                        'font_size_counts_sum': 10,
                        'italic_counts_sum': True,
                        'text_v_align_counts_sum': 1,
                        'text_h_align_counts_sum': 3,

                        ### c_pct_sum text
                        'bold_c_pct_sum_text': True,
                        'bg_color_c_pct_sum_text': '#D4AC0D',
                        'font_color_c_pct_sum_text': '#34495E',
                        'font_name_c_pct_sum_text': 'URW Gothic L',
                        'font_size_c_pct_sum_text': 8,
                        'italic_c_pct_sum_text': True,
                        'text_v_align_c_pct_sum_text': 1,
                        'text_h_align_c_pct_sum_text': 1,

                        ### c_pct_sum
                        'bold_c_pct_sum': True,
                        'bg_color_c_pct_sum': '#D4AC0D',
                        'font_color_c_pct_sum': '#34495E',
                        'font_name_c_pct_sum': 'URW Gothic L',
                        'font_size_c_pct_sum': 10,
                        'italic_c_pct_sum': True,
                        'text_v_align_c_pct_sum': 1,
                        'text_h_align_c_pct_sum': 3,

                       }

table_properties_group = {
                          ### label
                          'bold_label': True,

                          ### u_base text
                          'bold_u_base_text': True,
                          'font_color_u_base_text': '#808080',
                          ### u_base
                          'font_color_u_base': '#808080',

                          ### base text
                          'bold_base_text': True,
                          'font_color_base_text': '#632523',
                          ### base
                          'font_color_base': '#632523',

                          ### c_base_gross text
                          'bold_c_base_gross_text': True,
                          'bg_color_c_base_gross_text': 'yellow',
                          'font_color_c_base_gross_text': 'pink',
                          ### c_base_gross text
                          'bold_c_base_gross': False,
                          'bg_color_c_base_gross': 'gray',
                          'font_color_c_base_gross': 'yellow',

                          ### freq
                          'italic_freq_text': True,
                          'font_color_freq_text': 'blue',
                          'font_color_freq': 'blue',
                          'view_border_freq': False,

                          # net
                          'font_color_net_text': '#FF0000',
                          'font_color_net': '#FF0000',

                          # stat
                          'font_color_stat_text': '#FF0000',
                          'font_color_stat': '#FF0000',

                          # sum
                          'bg_color_sum_text': '#333333',
                          'font_color_sum_text': '#FFA500',
                          'italic_sum': True,

                          # block
                          'bold_block_net_text': True,
                          'italic_block_expanded_text': True,
                          'italic_block_normal_text': False,

                          # header - left
                          'bold_header_left': True,
                          'font_color_header_left': '#FFFFFF',
                          'text_h_align_header_left': 1,
                          'bg_color_header_left': '#AF8272',

                          # header - center
                          'bold_header_center': True,
                          'font_color_header_center': '#FFFFFF',
                          'text_h_align_header_center': 1,
                          'bg_color_header_center': '#85AD6E',

                          # header - title
                          'bold_header_title': True,
                          'font_color_header_title': '#265E1A',
                          'text_h_align_header_title': 1,
                          'bg_color_header_title': '#DFF442',

                          # notes
                          'bold_notes': True,
                          'font_color_notes': '#FF6DB8',
                          'text_h_align_notes': 1,
                          'bg_color_notes': '#6DFFFC',

                          # mask label
                          'bold_mask_label': True,
                          'bg_color_mask_label': '#BDB1D8',
                          'font_color_mask_label': '#33A59E',
                          'font_name_mask_label': 'Calibri',
                          'font_size_mask_label': 11,
                          'italic_mask_label': True,
                          'text_v_align_mask_label': 1,
                          'text_h_align_mask_label': 3,

                         }


sheet_properties = dict()

#test = 1
#test = 2
test = 3

if test == 1:
    sheet_properties = dict(dummy_tests=True,
                            alternate_bg=False,
                            #alternate_bg=True,
                            start_row=7,
                            start_column=2,
                           )
    custom_vg = {
            'block_normal_counts': 'block_normal',
            'block_normal_c_pct': 'block_normal',
            'block_normal_r_pct': 'block_normal',
            'block_normal_propstest': 'block_normal'}
    #custom_vg = {}
    tp = table_properties
    image = None
elif test == 2:
    sheet_properties = dict(dummy_tests=True,
                            alternate_bg=True,
                           )
    custom_vg = {'r_pct': 'sum',
                 'stddev': 'base',
                 #'net_c_pct': 'freq'
                 }
    tp = table_properties_group
    image = None
elif test == 3:
    sheet_properties = dict(alternate_bg=True,
                            freq_0_rep=':',
                            stat_0_rep='#',
                            y_header_height=20,
                            y_row_height=40)
    custom_vg = {
            'block_expanded_counts': 'freq',
            'block_expanded_c_pct': 'freq',
            'block_expanded_r_pct': 'freq',
            'block_expanded_propstest': 'freq',
            'block_net_counts': 'freq',
            'block_net_c_pct': 'freq',
            'block_net_r_pct': 'freq',
            'block_net_propstest': 'freq',
            }
    tp = {'bg_color_freq': 'gray'}
    image = {'img_name': 'logo',
             'img_url': './qplogo_invert.png',
             #'img_size': [110, 120],
             #'img_insert_x': 4,
             #'img_insert_y': 0,
             #'img_x_offset': 3,
             #'img_y_offset': 6
             }

# -------------
x = Excel('basic_excel.xlsx',
          views_groups=custom_vg,
          italicise_level=50,
          decimals=dict(N=0, P=2, D=1),
          #decimals=2,
          details=True,
          image=image,
          **tp)

if CA1:
    x.add_chains(chains,
                 'S H E E T',
                 annotations=['Ann. 1', 'Ann. 2', 'Ann. 3', 'Ann. 4'],
                 **sheet_properties
                )
if AC1:
    x.add_chains(arr_chains_1,
                 'array summary 1',
                 #annotations=['Ann. 1', 'Ann. 2', 'Ann. 3', 'Ann. 4'],
                 **sheet_properties
                )
if ACB1:
    x.add_chains(arr_chains_block_1,
                 'block summary 1',
                 annotations=['Ann. 1', 'Ann. 2', 'Ann. 3', 'Ann. 4'],
                 **sheet_properties
                )
if ACM1:
    x.add_chains(arr_chains_mean_1,
                 'means summary 1',
                 annotations=['Ann. 1', 'Ann. 2', 'Ann. 3', 'Ann. 4'],
                 **sheet_properties
                )
if AC0:
    x.add_chains(arr_chains_0,
                 'array summary 0',
                 annotations=['Ann. 1', 'Ann. 2', 'Ann. 3', 'Ann. 4'],
                 **sheet_properties
                )
if ACB0:
    x.add_chains(arr_chains_block_0,
                 'block summary 0',
                 annotations=['Ann. 1', 'Ann. 2', 'Ann. 3', 'Ann. 4'],
                 **sheet_properties
                )
if ACM0:
    x.add_chains(arr_chains_mean_0,
                 'means summary 0',
                 annotations=['Ann. 1', 'Ann. 2', 'Ann. 3', 'Ann. 4'],
                 **sheet_properties
                )
if OEC:
    x.add_chains(open_chain,
                 'Open_Ends',
                 annotations=['Ann. 1', 'Ann. 2', 'Ann. 3', 'Ann. 4'],
                 **sheet_properties
                )
x.close()
# -------------

# # -----------------------------------------------------------------------------
# PATH_DATA  = '../../tests/'
# NAME_PROJ  = 'Example Data (A)'
# NAME_META  = 'Example Data (A).json'
# NAME_DATA  = 'Example Data (A).csv'
# PATH_META  = os.path.join(PATH_DATA, NAME_META)
# PATH_DATA  = os.path.join(PATH_DATA, NAME_DATA)
# DATA_KEY   = 'x'
# FILTER_KEY = 'no_filter'
# VIEWS      = ('cbase', 'counts', 'c%', 'mean', 'median')
# VIEW_KEYS  = ('x|f|x:|||cbase', 'x|f|:|||counts', 'x|d.mean|x:|||mean',
#               'x|d.median|x:|||median', 'x|f.c:f|x:|||counts_sum')
# ORIENT     = 'x'
# TEST_FILE  = 'test_excel.xlsx'
# ISO8601    = r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)'
# # -----------------------------------------------------------------------------

# def _load_zip(path):
#     try:
#         z = ZipFile(path, 'r')
#     except (BadZipfile, LargeZipFile):
#         raise BadZipfile('%s: %s' % (path, sys.exc_info()[1]))
#     else:
#         return z

# def _read_file(zipf, filename):
#     try:
#         f = zipf.read(filename)
#     except KeyError:
#         print 'ERROR: Did not find %s in zip file' % filename
#     else:
#         return re.sub(ISO8601, '', f)

# @pytest.fixture(scope='module')
# def dataset():
#     _dataset = qp.DataSet(NAME_PROJ, dimensions_comp=False)
#     _dataset.read_quantipy(PATH_META, PATH_DATA)
#     yield _dataset.split()
#     del _dataset

# @pytest.fixture(scope='class')
# def stack(dataset):
#     meta, data = dataset
#     data = data.head(250)
#     _stack = qp.Stack(NAME_PROJ,
#                       add_data={DATA_KEY: {'meta': meta,
#                                            'data': data.head(250)}})
#     yield _stack
#     del _stack

# @pytest.fixture(scope='function')
# def excel(stack, x_keys, y_keys):
#     stack.add_link(x=x_keys, y=y_keys, views=VIEWS)
#     chain = Chain(stack, name='chain')
#     chains = chain.get(data_key=DATA_KEY, filter_key=FILTER_KEY,
#                        x_keys=x_keys, y_keys=y_keys, views=VIEW_KEYS,
#                        orient=ORIENT)

#     chains = [c.paint() for c in chains]
#     _excel = Excel(TEST_FILE)
#     _excel.add_chains(chains, 'S H E E T')
#     _excel.close()
#     return _excel.filename

# test_data = [(['q5_1', 'q4', 'gender', 'Wave'],
#               ['@', 'q4 > gender > Wave', 'q5_1', 'q4 > gender'],
#               'test_exp_complex_nest.xlsx'),
#             ]

# @pytest.fixture(scope='class', params=test_data, ids=['complex nest'])
# def params(request):
#     yield request.param

# @pytest.fixture()
# def cleandir():
#     if os.path.exists(TEST_FILE):
#         os.remove(TEST_FILE)

# @pytest.mark.usefixtures('cleandir')
# class TestExcel:

#     def test_xml(self, stack, params):

#         x_keys, y_keys, expected = params

#         got = excel(stack, x_keys, y_keys)

#         zip_got, zip_exp = _load_zip(got), _load_zip(expected)

#         assert zip_got.namelist() == zip_exp.namelist()

#         for filename in zip_got.namelist():
#             xml_got = _read_file(zip_got, filename)
#             xml_exp = _read_file(zip_exp, filename)
#             err = ' ... %s ...\nGOT: %s\nEXPECTED: %s'
#             assert xml_got == xml_exp, err % (filename, xml_got, xml_exp)
