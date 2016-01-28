
# import pdb;
import unittest
import os.path
# import json
import numpy as np
import pandas as pd

from collections import OrderedDict
from quantipy.core.view_generators.view_maps import QuantipyViews
from quantipy.core.view import View
from quantipy.core.stack import Stack
from quantipy.core.view_generators.view_mapper import ViewMapper
from quantipy.core.tools.dp import io
from quantipy.core.helpers.functions import load_json
from quantipy.core.tools.view.logic import (
    union, _union,
    intersection, _intersection,
    difference, _difference,
    sym_diff, _sym_diff,
    has_any, _has_any, 
    has_all, _has_all,
    not_any, _not_any, 
    not_all, _not_all,
    has_count, _has_count,
    not_count, _not_count,
    is_lt, _is_lt,
    is_le, _is_le,
    is_eq, _is_eq,
    is_ne, _is_ne,
    is_ge, _is_ge,
    is_gt, _is_gt,
    get_logic_key_chunk,
    get_logic_index
)

class TestViewObject(unittest.TestCase):

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
        self.int = ['record_number', 'unique_id', 'age', 'birth_day', 'birth_month']
        self.float = ['weight', 'weight_a', 'weight_b']
        self.single = ['gender', 'locality', 'ethnicity', 'religion', 'q1']
        self.delimited_set = ['q2', 'q3', 'q8', 'q9']
        self.string = ['q8a', 'q9a']
        self.date = ['start_time', 'end_time']
        self.time = ['duration']
        self.array = ['q5', 'q6', 'q7']

        # The minimum list of variables required to populate a stack with all single*delimited set variations
        self.minimum = ['Wave', 'ethnicity', 'q2', 'gender']

        # Set up the expected weight iterations
        self.weights = [None, 'weight_a']
        
#         # Set up example stack
#         self.setup_stack_Example_Data_A()
        
        # Set up the net views ViewMapper
        self.net_views = ViewMapper(
            template={
                'method': QuantipyViews().frequency,
                'kwargs': {
                    'axis': 'x',
                    'groups': ['Nets'],
                    'iterators': {
                        'rel_to': [None, 'y'],
                        'weights': self.weights
                    }
                }
            })


    def test_simple_or(self):
        
        # Test single x stored as int64 and float64
        for xk in ['Wave', 'ethnicity']:
            # Initial setup
            values = [1, 2, 3]
            yks = ['@', 'ethnicity', 'q2', 'gender']
            
            # Set up basic stack
            self.setup_stack_Example_Data_A()
            
            # Test net created using an OR list with 'codes'
            method_name = 'codes_list'
            self.net_views.add_method(
                name=method_name, 
                kwargs={'text': 'Ever', 'logic': values, 'axis': 'x'}
            )
            self.stack.add_link(x=xk, views=self.net_views.subset([method_name]))
            relation = 'x[{1,2,3}]:'
            self.verify_net_values_single_x(self.stack, xk, yks, values, relation, method_name)
             
            # Test net created using an OR list with 'logic'
            method_name = 'logic_list'
            self.net_views.add_method(
                name=method_name, 
                kwargs={'text': 'Ever', 'logic': values, 'axis': 'x'}
            )
            self.stack.add_link(x=xk, views=self.net_views.subset([method_name]))
            relation = 'x[{1,2,3}]:'
            self.verify_net_values_single_x(self.stack, xk, yks, values, relation, method_name)
             
            # Test net created using has_any() logic
            method_name = 'has_any'
            self.net_views.add_method(
                name=method_name, 
                kwargs={'text': 'Ever', 'logic': has_any([1,2,3]), 'axis': 'x'}
            )
            self.stack.add_link(x=xk, views=self.net_views.subset([method_name]))
            relation = 'x[{1,2,3}]:'
            self.verify_net_values_single_x(self.stack, xk, yks, values, relation, method_name)
            
            # Test net created using has_count() logic
            method_name = 'has_count'
            self.net_views.add_method(
                name=method_name, 
                kwargs={'text': 'Ever',
                        'logic': has_count([is_ge(1), [1,2,3]]), 'axis': 'x'}
            )
            self.stack.add_link(x=xk, views=self.net_views.subset([method_name]))
            relation = 'x[(1,2,3){>=1}]:'
            self.verify_net_values_single_x(self.stack, xk, yks, values, relation, method_name)
        
        
    def verify_net_values_single_x(self, stack, xk, yks, values, relation, method_name):
        dk = 'Example Data (A)'
        fk = 'no_filter'
        dvk = 'x|default|:|||default'
        dvkw = 'x|default|:||weight_a|default'
        
        for yk in yks:
            if yk == '@':
                yk = xk
            # Get comparison figures
            def_df = stack[dk][fk][xk][yk][dvk].dataframe[yk]
            defw_df = stack[dk][fk][xk][yk][dvkw].dataframe[yk]
            # Get the figures to be tested
            vk = 'x|f|%s|||%s' % (relation, method_name)
            vkw = 'x|f|%s||weight_a|%s' % (relation, method_name)
            df = stack[dk][fk][xk][yk][vk].dataframe[yk]
            dfw = stack[dk][fk][xk][yk][vkw].dataframe[yk]
            # Verify all crosstab net figures
            for y_val in def_df.columns.intersection(df.columns):
                net = 0
                netw = 0
                for x_val in values:
                    net += def_df.xs([xk, x_val])[y_val]
                    netw += defw_df.xs([xk, x_val])[y_val]
#                 if np.round(net, 10) != np.round(df.xs([xk, method_name])[y_val], 10):
#                     print ''
                self.assertTrue(
                    np.allclose(net, df.xs([xk, method_name])[y_val])
                )
                self.assertTrue(
                    np.allclose(netw, dfw.xs([xk, method_name])[y_val])
                )
        
        
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
   