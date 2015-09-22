import unittest
import os.path
import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal
import test_helper
import copy

from operator import lt, le, eq, ne, ge, gt

from pandas.core.index import Index
__index_symbol__ = {
    Index.union: ',',
    Index.intersection: '&',
    Index.difference: '~',
    Index.sym_diff: '^'
}

from collections import defaultdict, OrderedDict
from quantipy.core.stack import Stack
from quantipy.core.chain import Chain
from quantipy.core.link import Link
from quantipy.core.view_generators.view_mapper import ViewMapper
from quantipy.core.view_generators.view_maps import QuantipyViews
from quantipy.core.view import View
from quantipy.core.helpers import functions
from quantipy.core.helpers.functions import load_json
from quantipy.core.tools.dp.prep import (
    frange,
    frequency,
    crosstab
)

class TestRules(unittest.TestCase):

    def setUp(self):
        self.path = './tests/'
#         self.path = ''
        project_name = 'Example Data (A)'

        # Load Example Data (A) data and meta into self
        name_data = '%s.csv' % (project_name)
        path_data = '%s%s' % (self.path, name_data)
        self.example_data_A_data = pd.DataFrame.from_csv(path_data)
        name_meta = '%s.json' % (project_name)
        path_meta = '%s%s' % (self.path, name_meta)
        self.example_data_A_meta = load_json(path_meta)

        # Variables by type for Example Data A
        self.dk = 'Example Data (A)'
        self.fk = 'no_filter'
        self.single = ['gender', 'locality', 'ethnicity', 'religion', 'q1']
        self.delimited_set = ['q2', 'q3', 'q8', 'q9']
        self.q5 = ['q5_1', 'q5_2', 'q5_3']
                 
    def test_slicex(self):
         
        meta = self.example_data_A_meta
        data = self.example_data_A_data
         
        col_x = 'q5_1'
        df = crosstab(meta, data, col_x, col_x)
        natural_x = str_index_values(df.index)
         
        col_y = 'q5_1'
        df = crosstab(meta, data, col_y, col_y)
        natural_y = str_index_values(df.columns)
         
        ################## values        
        meta['columns'][col_x]['rules'] = {
            'x': {'slicex': {'values': [1, 3, 5, 98]}},
            'y': {'slicex': {'values': [2, 4, 97]}}
        }
         
        rules_x = index_items(
            col_x, 
            all=True,
            values=[1, 3, 5, 98]
        )
         
        rules_y = index_items(
            col_y, 
            all=True,
            values=[2, 4, 97]
        )
         
        confirm_crosstabs(
            self,
            meta, data, 
            col_x, col_y,
            natural_x, rules_x,
            natural_y, rules_y
        )
                     
    def test_sortx(self):
         
        meta = self.example_data_A_meta
        data = self.example_data_A_data
         
        col_x = 'q1'
        df = crosstab(meta, data, col_x, col_x)
        natural_x = str_index_values(df.index)
         
        col_y = 'Wave'
        df = crosstab(meta, data, col_y, col_y)
        natural_y = str_index_values(df.columns)
         
        ################## sort_on - default
        meta['columns'][col_x]['rules'] = {
            'x': {'sortx': {}}
        }  
        meta['columns'][col_y]['rules'] = {
            'y': {'sortx': {}}
        }      
  
        rules_x = index_items(
            col_x, 
            all=True,
            values=[4, 3, 7, 6, 2, 99, 1, 5, 8, 98, 96, 9]
        )
         
        rules_y = index_items(
            col_y, 
            all=True,
            values=[3, 2, 5, 4, 1]
        )
 
        confirm_crosstabs(
            self,
            meta, data, 
            col_x, col_y,
            natural_x, rules_x,
            natural_y, rules_y
        )
         
        ################## sort_on - '@'
        meta['columns'][col_x]['rules'] = {
            'x': {'sortx': {'sort_on': '@'}}
        }  
        meta['columns'][col_y]['rules'] = {
            'y': {'sortx': {'sort_on': '@'}}
        }      
  
        rules_x = index_items(
            col_x, 
            all=True,
            values=[4, 3, 7, 6, 2, 99, 1, 5, 8, 98, 96, 9]
        )
         
        rules_y = index_items(
            col_y, 
            all=True,
            values=[3, 2, 5, 4, 1]
        )
         
        confirm_crosstabs(
            self,
            meta, data, 
            col_x, col_y,
            natural_x, rules_x,
            natural_y, rules_y
        )
         
        ################## sort_on - int
        ##################
        # Sorting on anything other than
        # @ is currently unsupported.
        ##################
        
#         meta['columns'][col_x]['rules'] = {
#             'x': {'sortx': {'sort_on': 2}}
#         }  
#         meta['columns'][col_y]['rules'] = {
#             'y': {'sortx': {'sort_on': 6}}
#         }      
#          
#         rules_x = index_items(
#             col_x, 
#             all=True,
#             values=[4, 3, 7, 6, 99, 2, 1, 5, 8, 96, 98, 9]
#         )
#          
#         rules_y = index_items(
#             col_y, 
#             all=True,
#             values=[4, 2, 3, 1, 5]
#         )
#         
#         confirm_crosstabs(
#             self,
#             meta, data, 
#             col_x, col_y,
#             natural_x, rules_x,
#             natural_y, rules_y
#         )
         
        ################## fixed   
        meta['columns'][col_x]['rules'] = {
            'x': {'sortx': {'fixed': [1, 98]}}
        }  
        meta['columns'][col_y]['rules'] = {
            'y': {'sortx': {'fixed': [2, 4]}}
        }          
         
        rules_x = index_items(
            col_x, 
            all=True,
            values=[4, 3, 7, 6, 2, 99, 5, 8, 96, 9, 1, 98]
        )
         
        rules_y = index_items(
            col_y, 
            all=True,
            values=[3, 5, 1, 2, 4]
        )
         
        confirm_crosstabs(
            self,
            meta, data, 
            col_x, col_y,
            natural_x, rules_x,
            natural_y, rules_y
        )
              
    def test_rules_frequency(self):
         
        meta = self.example_data_A_meta
        data = self.example_data_A_data
         
        col_x = 'q5_1'
        df = frequency(meta, data, col_x)
        natural_x = str_index_values(df.index)
         
        natural_y = [('q5_1', '@')]
         
        ################## slicex
        meta['columns'][col_x]['rules'] = {
            'x': {'slicex': {'values': frange('5-1')}},
            'y': {'slicex': {'values': frange('1-5')}}
        }
         
        rules_x = index_items(
            col_x, 
            all=True,
            values=frange('5-1')
        )
         
        rules_y = natural_y
         
        confirm_frequencies(
            self,
            meta, data, 
            col_x,
            natural_x, rules_x,
            natural_y, rules_y
        )
         
        ################## sortx
        meta['columns'][col_x]['rules'] = {
            'x': {'sortx': {'fixed': [98]}},
            'y': {'sortx': {'fixed': [1]}}
        }        
         
        rules_x = index_items(
            col_x, 
            all=True,
            values=[3, 5, 2, 1, 97, 4, 98]
        )
         
        rules_y = natural_y
         
        confirm_frequencies(
            self,
            meta, data, 
            col_x,
            natural_x, rules_x,
            natural_y, rules_y
        )
         
        ################## dropx
        meta['columns'][col_x]['rules'] = {
            'x': {'dropx': {'values': [98]}},
            'y': {'dropx': {'values': [1]}}
        }
         
        rules_x = index_items(
            col_x, 
            all=True,
            values=frange('1-5, 97')
        )
         
        rules_y = natural_y
         
        confirm_frequencies(
            self,
            meta, data, 
            col_x,
            natural_x, rules_x,
            natural_y, rules_y
        )
         
        ################## slicex + sortx
        meta['columns'][col_x]['rules'] = {                    
            'x': {
                'slicex': {'values': frange('5-1')},
                'sortx': {'fixed': [5]}
            },
            'y': {
                'slicex': {'values': frange('1-5')},
                'sortx': {'fixed': [1]}
            }
        }        
         
        rules_x = index_items(
            col_x, 
            all=True,
            values=[3, 2, 1, 4, 5]
        )
         
        rules_y = natural_y
         
        confirm_frequencies(
            self,
            meta, data, 
            col_x,
            natural_x, rules_x,
            natural_y, rules_y
        )
         
 
        ################## slicex + dropx
        col_x = 'q5_1'
         
        meta['columns'][col_x]['rules'] = {                    
            'x': {
                'slicex': {'values': frange('5-1')},
                'dropx': {'values': [5]}
            },
            'y': {
                'slicex': {'values': frange('1-5')},
                'dropx': {'values': [1]}
            }
        }
         
        rules_x = index_items(
            col_x, 
            all=True,
            values=frange('4-1')
        )
         
        rules_y = natural_y
         
        confirm_frequencies(
            self,
            meta, data, 
            col_x,
            natural_x, rules_x,
            natural_y, rules_y
        )
         
        ################## sortx + dropx
        col_x = 'q5_1'
         
        meta['columns'][col_x]['rules'] = {                    
            'x': {
                'sortx': {'fixed': [5]},
                'dropx': {'values': [1]}
            },
            'y': {
                'sortx': {'fixed': [1]},
                'dropx': {'values': [5]}
            }
        }        
         
        rules_x = index_items(
            col_x, 
            all=True,
            values=[3, 98, 2, 97, 4, 5]
        )
         
        rules_y = natural_y
         
        confirm_frequencies(
            self,
            meta, data, 
            col_x,
            natural_x, rules_x,
            natural_y, rules_y
        )
 
        ################## slicex + sortx + dropx
        col_x = 'q5_1'
         
        meta['columns'][col_x]['rules'] = {                    
            'x': {
                'slicex': {'values': frange('5-1')},
                'sortx': {'fixed': [5]},
                'dropx': {'values': [1]}
            },
            'y': {
                'slicex': {'values': frange('1-5')},
                'sortx': {'fixed': [1]},
                'dropx': {'values': [5]}
            }
        }        
         
        rules_x = index_items(
            col_x, 
            all=True,
            values=[3, 2, 4, 5]
        )
         
        rules_y = natural_y
         
        confirm_frequencies(
            self,
            meta, data, 
            col_x,
            natural_x, rules_x,
            natural_y, rules_y
        )
                

    def test_rules_crosstab(self):
        
        meta = self.example_data_A_meta
        data = self.example_data_A_data
        
        col_x = 'q5_1'
        df = crosstab(meta, data, col_x, col_x)
        natural_x = str_index_values(df.index)
        
        col_y = 'q5_1'
        df = crosstab(meta, data, col_y, col_y)
        natural_y = str_index_values(df.columns)
        
        ################## slicex
        meta['columns'][col_x]['rules'] = {
            'x': {'slicex': {'values': frange('5-1')}},
            'y': {'slicex': {'values': frange('1-5')}}
        }
         
        rules_x = index_items(
            col_x, 
            all=True,
            values=frange('5-1')
        )
         
        rules_y = index_items(
            col_y, 
            all=True,
            values=frange('1-5')
        )
         
        confirm_crosstabs(
            self,
            meta, data, 
            col_x, col_y,
            natural_x, rules_x,
            natural_y, rules_y
        )
        
        ################## sortx 
        meta['columns'][col_x]['rules'] = {
            'x': {'sortx': {'fixed': [98]}},
            'y': {'sortx': {'fixed': [1]}}
        }        
        
        rules_x = index_items(
            col_x, 
            all=True,
            values=[3, 5, 2, 1, 97, 4, 98]
        )
        
        rules_y = index_items(
            col_y, 
            all=True,
            values=[3, 5, 98, 2, 97, 4, 1]
        )
        
        confirm_crosstabs(
            self,
            meta, data, 
            col_x, col_y,
            natural_x, rules_x,
            natural_y, rules_y
        )
        
        ################## dropx
        meta['columns'][col_x]['rules'] = {
            'x': {'dropx': {'values': [98]}},
            'y': {'dropx': {'values': [1]}}
        }
         
        rules_x = index_items(
            col_x, 
            all=True,
            values=frange('1-5, 97')
        )
         
        rules_y = index_items(
            col_y, 
            all=True,
            values=frange('2-5, 97, 98')
        )
         
        confirm_crosstabs(
            self,
            meta, data, 
            col_x, col_y,
            natural_x, rules_x,
            natural_y, rules_y
        )

        ################## slicex + sortx
        meta['columns'][col_x]['rules'] = {                    
            'x': {
                'slicex': {'values': frange('5-1')},
                'sortx': {'fixed': [5]}
            },
            'y': {
                'slicex': {'values': frange('1-5')},
                'sortx': {'fixed': [1]}
            }
        }        
        
        rules_x = index_items(
            col_x, 
            all=True,
            values=[3, 2, 1, 4, 5]
        )
        
        rules_y = index_items(
            col_y, 
            all=True,
            values=[3, 5, 2, 4, 1]
        )
        
        confirm_crosstabs(
            self,
            meta, data, 
            col_x, col_y,
            natural_x, rules_x,
            natural_y, rules_y
        )
        
        ################## slicex + dropx
        meta['columns'][col_x]['rules'] = {                    
            'x': {
                'slicex': {'values': frange('5-1')},
                'dropx': {'values': [5]}
            },
            'y': {
                'slicex': {'values': frange('1-5')},
                'dropx': {'values': [1]}
            }
        }
         
        rules_x = index_items(
            col_x, 
            all=True,
            values=frange('4-1')
        )
         
        rules_y = index_items(
            col_y, 
            all=True,
            values=frange('2-5')
        )
         
        confirm_crosstabs(
            self,
            meta, data, 
            col_x, col_y,
            natural_x, rules_x,
            natural_y, rules_y
        )
        
        ################## sortx + dropx
        meta['columns'][col_x]['rules'] = {                    
            'x': {
                'sortx': {'fixed': [5]},
                'dropx': {'values': [1]}
            },
            'y': {
                'sortx': {'fixed': [1]},
                'dropx': {'values': [5]}
            }
        }        
        
        rules_x = index_items(
            col_x, 
            all=True,
            values=[3, 98, 2, 97, 4, 5]
        )
        
        rules_y = index_items(
            col_y, 
            all=True,
            values=[3, 98, 2, 97, 4, 1]
        )
        
        confirm_crosstabs(
            self,
            meta, data, 
            col_x, col_y,
            natural_x, rules_x,
            natural_y, rules_y
        )

        ################## slicex + sortx + dropx
        meta['columns'][col_x]['rules'] = {                    
            'x': {
                'slicex': {'values': frange('5-1')},
                'sortx': {'fixed': [5]},
                'dropx': {'values': [1]}
            },
            'y': {
                'slicex': {'values': frange('1-5')},
                'sortx': {'fixed': [1]},
                'dropx': {'values': [5]}
            }
        }        
        
        rules_x = index_items(
            col_x, 
            all=True,
            values=[3, 2, 4, 5]
        )
        
        rules_y = index_items(
            col_y, 
            all=True,
            values=[3, 2, 4, 1]
        )
        
        confirm_crosstabs(
            self,
            meta, data, 
            col_x, col_y,
            natural_x, rules_x,
            natural_y, rules_y
        )
          
        
##################### Helper functions #####################

      
def index_items(col, values, all=False):
    """
    Return a correctly formed list of tuples to matching an index.
    """
    
    items = [
        (col, str(i))
        for i in values
    ]
    
    if all: items = [(col, 'All')] + items
    
    return items

def confirm_frequencies(self, meta, data, col_x,
                        natural_x, rules_x,
                        natural_y, rules_y):        
    """
    Confirms all variations of rules applied with frequency.
    """
    
    # rules=True
    df = frequency(meta, data, col_x, rules=True)
    confirm_index_columns(self, df, rules_x, rules_y)
    
    # rules=False
    df = frequency(meta, data, col_x, rules=False)
    confirm_index_columns(self, df, natural_x, natural_y)
    
    # rules=x
    df = frequency(meta, data, col_x, rules=['x'])
    confirm_index_columns(self, df, rules_x, natural_y)
    
    # rules=y
    df = frequency(meta, data, col_x, rules=['y'])
    confirm_index_columns(self, df, natural_x, rules_y)
    
    # rules=xy
    df = frequency(meta, data, col_x, rules=['x', 'y'])
    confirm_index_columns(self, df, rules_x, rules_y)    
    
def confirm_crosstabs(self, meta, data,
                      col_x, col_y,
                      natural_x, rules_x,
                      natural_y, rules_y):        
    """
    Confirms all variations of rules applied with frequency.
    """
    
    # rules=True
    df = crosstab(meta, data, col_x, col_y, rules=True)
    confirm_index_columns(self, df, rules_x, rules_y)
    
    # rules=False
    df = crosstab(meta, data, col_x, col_y, rules=False)
    confirm_index_columns(self, df, natural_x, natural_y)
    
    # rules=x
    df = crosstab(meta, data, col_x, col_y, rules=['x'])
    confirm_index_columns(self, df, rules_x, natural_y)
    
    # rules=y
    df = crosstab(meta, data, col_x, col_y, rules=['y'])
    confirm_index_columns(self, df, natural_x, rules_y)
    
    # rules=xy
    df = crosstab(meta, data, col_x, col_y, rules=['x', 'y'])
    confirm_index_columns(self, df, rules_x, rules_y)    

def str_index_values(index):
    """
    Make sure level 1 of the multiindex are all strings
    """
    values = index.values.tolist()
    values = zip(*[zip(*values)[0], [str(i) for i in zip(*values)[1]]])
    return values
        
def confirm_index_columns(self, df, expected_x, expected_y):
    """
    Confirms index and columns are as expected.
    """    
    actual_x = str_index_values(df.index)
    actual_y = str_index_values(df.columns)
    
    self.assertEqual(actual_x, expected_x)
    self.assertEqual(actual_y, expected_y)
        
def setup_stack_Example_Data_A(self, **kwargs):        
    self.stack = self.get_stack_Example_Data_A(**kwargs)


def get_stack_Example_Data_A(self, name=None, fk=None, xk=None, yk=None, views=None, weights=None):
    if name is None:
        name = 'Example Data (A)'
    if fk is None:
        fk = ['no_filter']
    if xk is None:
        xk = self.minimum
    if yk is None:
        yk = ['@'] + self.minimum
    if views is None:
        views = ['default', 'counts']
    if weights is None:
        weights = self.weights

    stack = Stack(name=name)
    stack.add_data(
        data_key=stack.name, 
        meta=self.example_data_A_meta, 
        data=self.example_data_A_data
    )

    for weight in weights:
        stack.add_link(
            data_keys=stack.name,
            filters=fk,
            x=xk,
            y=yk,
            views=QuantipyViews(views),
            weights=weights
        )

    return stack    
            