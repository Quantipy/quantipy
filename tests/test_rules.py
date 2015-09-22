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

COUNTER = 0

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
             
        col_x = 'religion'
        col_y = 'ethnicity'
           
        ################## values        
        meta['columns'][col_x]['rules'] = {
            'x': {'slicex': {'values': [1, 3, 5, 7, 9, 11, 13, 15]}}}
                
        meta['columns'][col_y]['rules'] = {
            'y': {'slicex': {'values': [2, 4, 6, 8, 10, 12, 14, 16]}}}
   
        rules_values_x = {
            'unwtd': index_items(col_x, all=True, 
                values=[1, 3, 5, 7, 9, 11, 13, 15]),
            'iswtd': index_items(col_x, all=True, 
                values=[1, 3, 5, 7, 9, 11, 13, 15])
        }
           
        rules_values_y = {
            'unwtd': index_items(col_y, all=True, 
                values=[2, 4, 6, 8, 10, 12, 14, 16]),
            'iswtd': index_items(col_y, all=True, 
                values=[2, 4, 6, 8, 10, 12, 14, 16])
        }
           
        confirm_crosstabs(
            self,
            meta, data, 
            [None, 'weight_a'],
            col_x, col_y,
            rules_values_x,
            rules_values_y)
            
    def test_sortx(self):
             
        meta = self.example_data_A_meta
        data = self.example_data_A_data
           
        col_x = 'religion'
        col_y = 'ethnicity'
  
        ################## sort_on - default
        meta['columns'][col_x]['rules'] = {'x': {'sortx': {}}}  
        meta['columns'][col_y]['rules'] = {'y': {'sortx': {}}}      
      
        rules_values_x = {
            'unwtd': index_items(col_x, all=True, 
                values=[2, 1, 3, 15, 4, 5, 16, 6, 10, 12, 14, 11, 7, 13, 8, 9]),
            'iswtd': index_items(col_x, all=True, 
                values=[2, 1, 3, 15, 4, 5, 16, 6, 12, 10, 14, 11, 7, 13, 9, 8])
        }
          
        rules_values_y = {
            'unwtd': index_items(col_y, all=True, 
                values=[1, 2, 16, 7, 15, 12, 3, 11, 14, 6, 8, 10, 9, 5, 4, 13]),
            'iswtd': index_items(col_y, all=True, 
                values=[1, 2, 16, 7, 12, 11, 3, 15, 8, 9, 10, 14, 5, 6, 4, 13])
        }
          
        confirm_crosstabs(
            self,
            meta, data, 
            [None, 'weight_a'],
            col_x, col_y,
            rules_values_x,
            rules_values_y)
            
        ################## sort_on - '@'
        meta['columns'][col_x]['rules'] = {
            'x': {'sortx': {'sort_on': '@'}}
        }  
        meta['columns'][col_y]['rules'] = {
            'y': {'sortx': {'sort_on': '@'}}
        }      
      
        rules_values_x = {
            'unwtd': index_items(col_x, all=True, 
                values=[2, 1, 3, 15, 4, 5, 16, 6, 10, 12, 14, 11, 7, 13, 8, 9]),
            'iswtd': index_items(col_x, all=True, 
                values=[2, 1, 3, 15, 4, 5, 16, 6, 12, 10, 14, 11, 7, 13, 9, 8])
        }
          
        rules_values_y = {
            'unwtd': index_items(col_y, all=True, 
                values=[1, 2, 16, 7, 15, 12, 3, 11, 14, 6, 8, 10, 9, 5, 4, 13]),
            'iswtd': index_items(col_y, all=True, 
                values=[1, 2, 16, 7, 12, 11, 3, 15, 8, 9, 10, 14, 5, 6, 4, 13])
        }
          
        confirm_crosstabs(
            self,
            meta, data, 
            [None, 'weight_a'],
            col_x, col_y,
            rules_values_x,
            rules_values_y)
            
        ################## fixed   
        meta['columns'][col_x]['rules'] = {
            'x': {'sortx': {'fixed': [5, 1, 3]}}
        }  
        meta['columns'][col_y]['rules'] = {
            'y': {'sortx': {'fixed': [6, 2, 4]}}
        }          
             
        rules_values_x = {
            'unwtd': index_items(col_x, all=True, 
                values=[2, 15, 4, 16, 6, 10, 12, 14, 11, 7, 13, 8, 9, 5, 1, 3]),
            'iswtd': index_items(col_x, all=True, 
                values=[2, 15, 4, 16, 6, 12, 10, 14, 11, 7, 13, 9, 8, 5, 1, 3])
        }
          
        rules_values_y = {
            'unwtd': index_items(col_y, all=True, 
                values=[1, 16, 7, 15, 12, 3, 11, 14, 8, 10, 9, 5, 13, 6, 2, 4]),
            'iswtd': index_items(col_y, all=True, 
                values=[1, 16, 7, 12, 11, 3, 15, 8, 9, 10, 14, 5, 13, 6, 2, 4])
        }
          
        confirm_crosstabs(
            self,
            meta, data, 
            [None, 'weight_a'],
            col_x, col_y,
            rules_values_x,
            rules_values_y)
                   
    def test_dropx(self):
 
        meta = self.example_data_A_meta
        data = self.example_data_A_data
 
        col_x = 'religion'
        col_y = 'ethnicity'
 
        ################## values        
        meta['columns'][col_x]['rules'] = {
            'x': {'dropx': {'values': [1, 3, 5, 7, 9, 11, 13, 15]}}}
  
        meta['columns'][col_y]['rules'] = {
            'y': {'dropx': {'values': [2, 4, 6, 8, 10, 12, 14, 16]}}}
  
        rules_values_x = {
            'unwtd': index_items(col_x, all=True, 
                values=[2, 4, 6, 8, 10, 12, 14, 16]),
            'iswtd': index_items(col_x, all=True, 
                values=[2, 4, 6, 8, 10, 12, 14, 16])
        }
 
        rules_values_y = {
            'unwtd': index_items(col_y, all=True, 
                values=[1, 3, 5, 7, 9, 11, 13, 15]),
            'iswtd': index_items(col_y, all=True, 
                values=[1, 3, 5, 7, 9, 11, 13, 15])
        }
 
        confirm_crosstabs(
            self,
            meta, data, 
            [None, 'weight_a'],
            col_x, col_y,
            rules_values_x,
            rules_values_y)
           
    def test_rules_frequency(self):
            
        meta = self.example_data_A_meta
        data = self.example_data_A_data
            
        col = 'religion'
          
        ################## slicex
        meta['columns'][col]['rules'] = {
            'x': {'slicex': {'values': [1, 3, 5, 7, 9, 10, 11, 13, 15]}},
            'y': {'slicex': {'values': [2, 4, 6, 8, 10, 12, 14, 16]}}}
                
        rules_values_x = {
            'unwtd': index_items(col, all=True, 
                values=[1, 3, 5, 7, 9, 10, 11, 13, 15]),
            'iswtd': index_items(col, all=True, 
                values=[1, 3, 5, 7, 9, 10, 11, 13, 15])
        }
          
        rules_values_y = {
            'unwtd': index_items(col, all=True, 
                values=[2, 4, 6, 8, 10, 12, 14, 16]),
            'iswtd': index_items(col, all=True, 
                values=[2, 4, 6, 8, 10, 12, 14, 16])
        }
           
        confirm_frequencies(
            self,
            meta, data, 
            [None, 'weight_a'],
            col,
            rules_values_x,
            rules_values_y)
              
        ################## sortx
        meta['columns'][col]['rules'] = {
            'x': {'sortx': {'fixed': [5, 1, 3]}},
            'y': {'sortx': {'fixed': [6, 2, 4]}}}          
             
        rules_values_x = {
            'unwtd': index_items(col, all=True, 
                values=[2, 15, 4, 16, 6, 10, 12, 14, 11, 7, 13, 8, 9, 5, 1, 3]),
            'iswtd': index_items(col, all=True, 
                values=[2, 15, 4, 16, 6, 12, 10, 14, 11, 7, 13, 9, 8, 5, 1, 3])
        }
          
        rules_values_y = {
            'unwtd': index_items(col, all=True, 
                values=[1, 3, 15, 5, 16, 10, 12, 14, 11, 7, 13, 8, 9, 6, 2, 4]),
            'iswtd': index_items(col, all=True, 
                values=[1, 3, 15, 5, 16, 12, 10, 14, 11, 7, 13, 9, 8, 6, 2, 4])
        }
          
        confirm_frequencies(
            self,
            meta, data, 
            [None, 'weight_a'],
            col,
            rules_values_x,
            rules_values_y)
 
        ################## dropx     
        meta['columns'][col]['rules'] = {
            'x': {'dropx': {'values': [1, 3, 5, 7, 9, 11, 13, 15]}},
            'y': {'dropx': {'values': [2, 4, 6, 8, 10, 12, 14, 16]}}}
  
        rules_values_x = {
            'unwtd': index_items(col, all=True, 
                values=[2, 4, 6, 8, 10, 12, 14, 16]),
            'iswtd': index_items(col, all=True, 
                values=[2, 4, 6, 8, 10, 12, 14, 16])
        }
 
        rules_values_y = {
            'unwtd': index_items(col, all=True, 
                values=[1, 3, 5, 7, 9, 11, 13, 15]),
            'iswtd': index_items(col, all=True, 
                values=[1, 3, 5, 7, 9, 11, 13, 15])
        }
         
        confirm_frequencies(
            self,
            meta, data, 
            [None, 'weight_a'],
            col,
            rules_values_x,
            rules_values_y)
 
        ################## slicex + sortx
        meta['columns'][col]['rules'] = {                    
            'x': {
                'slicex': {'values': frange('4-13')},
                'sortx': {'fixed': [1, 2]}},
            'y': {
                'slicex': {'values': frange('7-16')},
                'sortx': {'fixed': [15, 16]}}}        
           
        rules_values_x = {
            'unwtd': index_items(col, all=True, 
                values=[4, 5, 6, 10, 12, 11, 7, 13, 8, 9, 1, 2]),
            'iswtd': index_items(col, all=True, 
                values=[4, 5, 6, 12, 10, 11, 7, 13, 9, 8, 1, 2])
        }
 
        rules_values_y = {
            'unwtd': index_items(col, all=True, 
                values=[10, 12, 14, 11, 7, 13, 8, 9, 15, 16]),
            'iswtd': index_items(col, all=True, 
                values=[12, 10, 14, 11, 7, 13, 9, 8, 15, 16])
        }
         
        confirm_frequencies(
            self,
            meta, data, 
            [None, 'weight_a'],
            col,
            rules_values_x,
            rules_values_y)
 
        ################## slicex + dropx
        meta['columns'][col]['rules'] = {                    
            'x': {
                'slicex': {'values': [1, 3, 5, 7, 9, 11, 13, 15]},
                'dropx': {'values': [3, 7, 11, 15]}},
            'y': {
                'slicex': {'values': [2, 4, 6, 8, 10, 12, 14, 16]},
                'dropx': {'values': [2, 6, 10, 14]}}}        
           
        rules_values_x = {
            'unwtd': index_items(col, all=True, 
                values=[1, 5, 9, 13]),
            'iswtd': index_items(col, all=True, 
                values=[1, 5, 9, 13])
        }
 
        rules_values_y = {
            'unwtd': index_items(col, all=True, 
                values=[4, 8, 12, 16]),
            'iswtd': index_items(col, all=True, 
                values=[4, 8, 12, 16])
        }
         
        confirm_frequencies(
            self,
            meta, data, 
            [None, 'weight_a'],
            col,
            rules_values_x,
            rules_values_y)
         
        ################## sortx + dropx
        meta['columns'][col]['rules'] = {                    
            'x': {
                'sortx': {'fixed': [1, 2]},
                'dropx': {'values': [5, 11, 13]}},
            'y': {
                'sortx': {'fixed': [15, 16]},
                'dropx': {'values': [7, 13, 14]}}}
           
        rules_values_x = {
            'unwtd': index_items(col, all=True, 
                values=[3, 15, 4, 16, 6, 10, 12, 14, 7, 8, 9, 1, 2]),
            'iswtd': index_items(col, all=True, 
                values=[3, 15, 4, 16, 6, 12, 10, 14, 7, 9, 8, 1, 2])
        }
 
        rules_values_y = {
            'unwtd': index_items(col, all=True, 
                values=[2, 1, 3, 4, 5, 6, 10, 12, 11, 8, 9, 15, 16]),
            'iswtd': index_items(col, all=True, 
                values=[2, 1, 3, 4, 5, 6, 12, 10, 11, 9, 8, 15, 16])
        }
         
        confirm_frequencies(
            self,
            meta, data, 
            [None, 'weight_a'],
            col,
            rules_values_x,
            rules_values_y)
 
        ################## slicex + sortx + dropx
        meta['columns'][col]['rules'] = {                    
            'x': {
                'slicex': {'values': frange('4-13')},
                'sortx': {'fixed': [11, 13]},
                'dropx': {'values': [7]}},
            'y': {
                'slicex': {'values': frange('7-16')},
                'sortx': {'fixed': [15, 16]},
                'dropx': {'values': [7, 13]}}}
           
        rules_values_x = {
            'unwtd': index_items(col, all=True, 
                values=[4, 5, 6, 10, 12, 8, 9, 11, 13]),
            'iswtd': index_items(col, all=True, 
                values=[4, 5, 6, 12, 10, 9, 8, 11, 13])
        }
 
        rules_values_y = {
            'unwtd': index_items(col, all=True, 
                values=[10, 12, 14, 11, 8, 9, 15, 16]),
            'iswtd': index_items(col, all=True, 
                values=[12, 10, 14, 11, 9, 8, 15, 16])
        }
         
        confirm_frequencies(
            self,
            meta, data, 
            [None, 'weight_a'],
            col,
            rules_values_x,
            rules_values_y)
 
    def test_rules_crosstab(self):
           
        meta = self.example_data_A_meta
        data = self.example_data_A_data
           
        col_x = 'religion'
        col_y = 'ethnicity'
         
        ################## slicex
        meta['columns'][col_x]['rules'] = {
            'x': {'slicex': {'values': [1, 3, 5, 7, 9, 10, 11, 13, 15]}}}
  
        meta['columns'][col_y]['rules'] = {
            'y': {'slicex': {'values': [2, 4, 6, 8, 10, 12, 14, 16]}}}
  
        rules_values_x = {
            'unwtd': index_items(col_x, all=True, 
                values=[1, 3, 5, 7, 9, 10, 11, 13, 15]),
            'iswtd': index_items(col_x, all=True, 
                values=[1, 3, 5, 7, 9, 10, 11, 13, 15])
        }
          
        rules_values_y = {
            'unwtd': index_items(col_y, all=True, 
                values=[2, 4, 6, 8, 10, 12, 14, 16]),
            'iswtd': index_items(col_y, all=True, 
                values=[2, 4, 6, 8, 10, 12, 14, 16])
        }
           
        confirm_crosstabs(
            self,
            meta, data, 
            [None, 'weight_a'],
            col_x, col_y,
            rules_values_x,
            rules_values_y)
           
        ################## sortx
        meta['columns'][col_x]['rules'] = {
            'x': {'sortx': {'fixed': [5, 1, 3]}}}
  
        meta['columns'][col_y]['rules'] = {
            'y': {'sortx': {'fixed': [6, 2, 4]}}}
  
        rules_values_x = {
            'unwtd': index_items(col_x, all=True, 
                values=[2, 15, 4, 16, 6, 10, 12, 14, 11, 7, 13, 8, 9, 5, 1, 3]),
            'iswtd': index_items(col_x, all=True, 
                values=[2, 15, 4, 16, 6, 12, 10, 14, 11, 7, 13, 9, 8, 5, 1, 3])
        }
           
        rules_values_y = {
            'unwtd': index_items(col_y, all=True, 
                values=[1, 16, 7, 15, 12, 3, 11, 14, 8, 10, 9, 5, 13, 6, 2, 4]),
            'iswtd': index_items(col_y, all=True, 
                values=[1, 16, 7, 12, 11, 3, 15, 8, 9, 10, 14, 5, 13, 6, 2, 4])
        }
 
        confirm_crosstabs(
            self,
            meta, data, 
            [None, 'weight_a'],
            col_x, col_y,
            rules_values_x,
            rules_values_y)
 
        ################## dropx   
        meta['columns'][col_x]['rules'] = {
            'x': {'dropx': {'values': [1, 3, 5, 7, 9, 11, 13, 15]}}}
  
        meta['columns'][col_y]['rules'] = {
            'y': {'dropx': {'values': [2, 4, 6, 8, 10, 12, 14, 16]}}}
  
        rules_values_x = {
            'unwtd': index_items(col_x, all=True, 
                values=[2, 4, 6, 8, 10, 12, 14, 16]),
            'iswtd': index_items(col_x, all=True, 
                values=[2, 4, 6, 8, 10, 12, 14, 16])
        }
  
        rules_values_y = {
            'unwtd': index_items(col_y, all=True, 
                values=[1, 3, 5, 7, 9, 11, 13, 15]),
            'iswtd': index_items(col_y, all=True, 
                values=[1, 3, 5, 7, 9, 11, 13, 15])
        }
          
        confirm_crosstabs(
            self,
            meta, data, 
            [None, 'weight_a'],
            col_x, col_y,
            rules_values_x,
            rules_values_y)
            
        ################## slicex + sortx
        meta['columns'][col_x]['rules'] = {
            'x': {
                'slicex': {'values': frange('4-13')},
                'sortx': {'fixed': [4, 7, 3]}}}
  
        meta['columns'][col_y]['rules'] = {
            'y': {
                'slicex': {'values': frange('7-16')},
                'sortx': {'fixed': [7, 11, 13]}}}
  
        rules_values_x = {
            'unwtd': index_items(col_x, all=True, 
                values=[5, 6, 10, 12, 11, 13, 8, 9, 4, 7, 3]),
            'iswtd': index_items(col_x, all=True, 
                values=[5, 6, 12, 10, 11, 13, 9, 8, 4, 7, 3])
        }
           
        rules_values_y = {
            'unwtd': index_items(col_y, all=True, 
                values=[16, 15, 12, 14, 8, 10, 9, 7, 11, 13]),
            'iswtd': index_items(col_y, all=True, 
                values=[16, 12, 15, 8, 9, 10, 14, 7, 11, 13])
        }
 
        confirm_crosstabs(
            self,
            meta, data, 
            [None, 'weight_a'],
            col_x, col_y,
            rules_values_x,
            rules_values_y)
            
        ################## slicex + dropx
        meta['columns'][col_x]['rules'] = {               
            'x': {
                'slicex': {'values': [1, 3, 5, 7, 9, 11, 13, 15]},
                'dropx': {'values': [3, 7, 11, 15]}}}
 
        meta['columns'][col_y]['rules'] = {
            'y': {
                'slicex': {'values': [2, 4, 6, 8, 10, 12, 14, 16]},
                'dropx': {'values': [2, 6, 10, 14]}}}      
            
        rules_values_x = {
            'unwtd': index_items(col_x, all=True, 
                values=[1, 5, 9, 13]),
            'iswtd': index_items(col_x, all=True, 
                values=[1, 5, 9, 13])
        }
  
        rules_values_y = {
            'unwtd': index_items(col_y, all=True, 
                values=[4, 8, 12, 16]),
            'iswtd': index_items(col_y, all=True, 
                values=[4, 8, 12, 16])
        }
 
        confirm_crosstabs(
            self,
            meta, data, 
            [None, 'weight_a'],
            col_x, col_y,
            rules_values_x,
            rules_values_y)
           
        ################## sortx + dropx
        meta['columns'][col_x]['rules'] = {
            'x': {
                'sortx': {'fixed': [4, 7, 3]},
                'dropx': {'values': [5, 10]}}}
        
        meta['columns'][col_y]['rules'] = {
            'y': {
                'sortx': {'fixed': [7, 11, 13]},
                'dropx': {'values': [4, 12]}}}
 
        rules_values_x = {
            'unwtd': index_items(col_x, all=True, 
                values=[2, 1, 15, 16, 6, 12, 14, 11, 13, 8, 9, 4, 7, 3]),
            'iswtd': index_items(col_x, all=True, 
                values=[2, 1, 15, 16, 6, 12, 14, 11, 13, 9, 8, 4, 7, 3])
        }
          
        rules_values_y = {
            'unwtd': index_items(col_y, all=True, 
                values=[1, 2, 16, 15, 3, 14, 6, 8, 10, 9, 5, 7, 11, 13]),
            'iswtd': index_items(col_y, all=True, 
                values=[1, 2, 16, 3, 15, 8, 9, 10, 14, 5, 6, 7, 11, 13])
        }

        confirm_crosstabs(
            self,
            meta, data, 
            [None, 'weight_a'],
            col_x, col_y,
            rules_values_x,
            rules_values_y)
           
        ################## slicex + sortx + dropx
        meta['columns'][col_x]['rules'] = {
            'x': {
                'slicex': {'values': frange('4-13')},
                'sortx': {'fixed': [4, 7, 3]},
                'dropx': {'values': [6, 11]}}}
  
        meta['columns'][col_y]['rules'] = {
            'y': {
                'slicex': {'values': frange('7-16')},
                'sortx': {'fixed': [7, 11, 13]},
                'dropx': {'values': [11, 16]}}}
  
        rules_values_x = {
            'unwtd': index_items(col_x, all=True, 
                values=[5, 10, 12, 13, 8, 9, 4, 7, 3]),
            'iswtd': index_items(col_x, all=True, 
                values=[5, 12, 10, 13, 9, 8, 4, 7, 3])
        }
           
        rules_values_y = {
            'unwtd': index_items(col_y, all=True, 
                values=[15, 12, 14, 8, 10, 9, 7, 13]),
            'iswtd': index_items(col_y, all=True, 
                values=[12, 15, 8, 9, 10, 14, 7, 13])
        }
 
        confirm_crosstabs(
            self,
            meta, data, 
            [None, 'weight_a'],
            col_x, col_y,
            rules_values_x,
            rules_values_y)
           
#     def test_rules_get_dataframe(self):
#         
#         meta = self.example_data_A_meta
#         data = self.example_data_A_data
#         
#         col_x = 'q5_1'
#         df = crosstab(meta, data, col_x, col_x)
#         natural_x = str_index_values(df.index)
#         
#         col_y = 'q5_1'
#         df = crosstab(meta, data, col_y, col_y)
#         natural_y = str_index_values(df.columns)
#         
#         ################## slicex
#         meta['columns'][col_x]['rules'] = {
#             'x': {'slicex': {'values': frange('5-1')}},
#             'y': {'slicex': {'values': frange('1-5')}}
#         }
        
    
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

def confirm_frequencies(self, meta, data, 
                        weights,
                        col,
                        rules_values_x,
                        rules_values_y):        
    """
    Confirms all variations of rules applied with frequency.
    """
    
    df = frequency(meta, data, x=col)
    natural_x = str_index_values(df.index)
    natural_y = natural_x
    
    frequ_x = [(col, '@')]
    frequ_y = frequ_x
    
    for weight in weights:
        
        if weight is None:
            rules_x = rules_values_x['unwtd']
            rules_y = rules_values_y['unwtd']
        else:
            rules_x = rules_values_x['iswtd']
            rules_y = rules_values_y['iswtd']
            
        # rules=True
        fx = frequency(meta, data, x=col, weight=weight, rules=True)
        fy = frequency(meta, data, y=col, weight=weight, rules=True)
#         print fx
#         print zip(*rules_x)[1]
#         print zip(*rules_y)[1]
        confirm_index_columns(self, fx, rules_x, frequ_x)
        confirm_index_columns(self, fy, frequ_x, rules_y)
        
        # rules=False
        fx = frequency(meta, data, x=col, weight=weight, rules=False)
        fy = frequency(meta, data, y=col, weight=weight, rules=False)
        confirm_index_columns(self, fx, natural_x, frequ_x)
        confirm_index_columns(self, fy, frequ_x, natural_y)
        
        # rules=x
        fx = frequency(meta, data, x=col, weight=weight, rules=['x'])
        fy = frequency(meta, data, y=col, weight=weight, rules=['x'])
        confirm_index_columns(self, fx, rules_x, frequ_x)
        confirm_index_columns(self, fy, frequ_x, natural_y)
        
        # rules=y
        fx = frequency(meta, data, x=col, weight=weight, rules=['y'])
        fy = frequency(meta, data, y=col, weight=weight, rules=['y'])
        confirm_index_columns(self, fx, natural_x, frequ_x)
        confirm_index_columns(self, fy, frequ_x, rules_y)
        
        # rules=xy
        fx = frequency(meta, data, x=col, weight=weight, rules=['x', 'y'])
        fy = frequency(meta, data, y=col, weight=weight, rules=['x', 'y'])
        confirm_index_columns(self, fx, rules_x, frequ_x)  
        confirm_index_columns(self, fy, frequ_x, rules_y)      
    
def confirm_crosstabs(self, meta, data, 
                      weights,
                      col_x, col_y,
                      rules_values_x,
                      rules_values_y):        
    """
    Confirms all variations of rules applied with frequency.
    """
    
    fx = frequency(meta, data, x=col_x)
    natural_x = str_index_values(fx.index)
      
    fy = frequency(meta, data, y=col_y)
    natural_y = str_index_values(fy.columns)
    
    for weight in weights:
        
        if weight is None:
            rules_x = rules_values_x['unwtd']
            rules_y = rules_values_y['unwtd']
        else:
            rules_x = rules_values_x['iswtd']
            rules_y = rules_values_y['iswtd']
        
        # rules=True
        df = crosstab(meta, data, col_x, col_y, weight=weight, rules=True)
#         print df
#         print zip(*rules_x)[1]
#         print zip(*rules_y)[1]
        confirm_index_columns(self, df, rules_x, rules_y)
        
        # rules=False
        df = crosstab(meta, data, col_x, col_y, weight=weight, rules=False)
        confirm_index_columns(self, df, natural_x, natural_y)
        
        # rules=x
        df = crosstab(meta, data, col_x, col_y, weight=weight, rules=['x'])
        confirm_index_columns(self, df, rules_x, natural_y)
        
        # rules=y
        df = crosstab(meta, data, col_x, col_y, weight=weight, rules=['y'])
        confirm_index_columns(self, df, natural_x, rules_y)
        
        # rules=xy
        df = crosstab(meta, data, col_x, col_y, weight=weight, rules=['x', 'y'])
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
    global COUNTER
    
    actual_x = str_index_values(df.index)
    actual_y = str_index_values(df.columns)
    
    self.assertEqual(actual_x, expected_x)
    self.assertEqual(actual_y, expected_y)
    
    COUNTER = COUNTER + 2
#     print COUNTER
        
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
        views = ['counts']
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
            