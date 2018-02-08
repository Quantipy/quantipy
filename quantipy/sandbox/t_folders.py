
import json


import quantipy as qp
from quantipy.sandbox.sandbox import ChainManager
from quantipy.sandbox.excel import Excel
from quantipy.core.view_generators.view_specs import ViewManager

meta = qp.core.tools.dp.io.load_json('../../tests/Example Data (A).json')
data = qp.core.tools.dp.io.load_csv('../../tests/Example Data (A).csv')

stack = qp.Stack('#')
stack.add_data(data_key='#', data=data, meta=meta)

stack.add_link(data_keys='#',
               filters='no_filter',
               x=['q2', 'q2b', 'q3', 'q4',
                  'q5_1', 'q5_2', 'q5_3', 'q5_4', 'q5_5', 'q5_6',
                  'q8', 'q9'],
               y=['@', 'gender', 'locality'],
               views=['cbase', 'counts'])

vm = ViewManager(stack)
vm.get_views(cell_items='counts', weight=None, bases='auto').group()

qp.set_option('new_chains', True)

cm = ChainManager(stack)
cm.get(data_key='#',
       filter_key='no_filter',
       x_keys=['q2', 'q2b', 'q3', 'q4',
               'q5_1', 'q5_2', 'q5_3', 'q5_4', 'q5_5', 'q5_6'],
       y_keys=['@', 'gender', 'locality'],
       views=vm.views,
       orient='x',
       prioritize=True,
       folder=None
       )

print cm.describe(True)
cm.fold('FOLDER: q5', ['q5_1', 'q5_2', 'q5_3', 'q5_4', 'q5_5', 'q5_6'])
cm.reorder([0, 1, 'FOLDER: q5', 2, 3])
print cm.describe(True)

cm.add(data.loc[:, ['q8a', 'q9a']],
       meta_from=('#', 'no_filter'),
       name='Open Ends')
print cm.describe(True)

cm.paint_all()

x = Excel('fold.xlsx')
x.add_chains(cm, 'LOL')
x.close()
