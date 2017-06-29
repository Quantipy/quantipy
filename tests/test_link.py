import unittest
import os.path
import pandas as pd
# import numpy as np
from quantipy.core.link import Link
from quantipy.core.stack import Stack
from quantipy.core.helpers.functions import load_json
from quantipy.core.view_generators.view_maps import QuantipyViews

class TestLinkObject(unittest.TestCase):

# stack.add_link(x='q1', y=y, views=mean_views.subset('m1to6'), weights=weight)

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
        
        # The minimum list of variables required to populate a stack with all single*delimited set variations
        self.minimum = ['q2b', 'Wave', 'q2', 'q3', 'q5_1']
        
        self.setup_stack_Example_Data_A()
        
    def test_link_is_a_subclassed_dict(self):        
        dk = self.stack.name
        fk = 'no_filter'
        xk = self.minimum
        yk = ['@'] + self.minimum
        
        for x in xk:
            for y in yk:
                link = self.stack[dk][fk][x][y]
                self.assertIsInstance(link, dict)
                self.assertIsInstance(link, Link)
    
    def test_link_behaves_like_a_dict(self):
        
        dk = self.stack.name
        fk = 'no_filter'
        xk = self.minimum
        yk = ['@'] + self.minimum
        
        key = "some_key_name"
        value = "some_value"
        
        for x in xk:
            for y in yk:
                link = self.stack[dk][fk][x][y]
                link[key] = value
                self.assertIn(
                    key,
                    link.keys(), 
                    msg="Link should have key {data_key}, but has {link_keys}".format(
                        data_key=key, 
                        link_keys=link.keys()
                    )
                )   
    
    def test_get_meta(self):
        
        dk = self.stack.name
        fk = 'no_filter'
        xk = self.minimum
        yk = ['@'] + self.minimum
        
        #test returned meta against stack meta
        for x in xk:
            for y in yk:
                link = self.stack[dk][fk][x][y]
                self.assertEqual(link.get_meta(), self.stack[dk].meta)
    
    def test_get_data(self):
        
        dk = self.stack.name
        fk = 'no_filter'
        xk = self.minimum
        yk = ['@'] + self.minimum
        
        stack_data = self.stack[dk][fk].data
        
        #test returned data against stack data
        for x in xk:
            for y in yk:
                link_data = self.stack[dk][fk][x][y].get_data()                
                self.assertTrue(link_data is stack_data)
        
    @classmethod
    def tearDownClass(self):
        self.stack = Stack("StackName")
        filepath ='./tests/'+self.stack.name+'.stack'
        if os.path.exists(filepath):
            os.remove(filepath)

    def is_empty(self, any_structure):
        if any_structure:
            #print('Structure is not empty.')
            return False
        else:
            #print('Structure is empty.')
            return True

    def create_key_stack(self, branch_pos="data"):
        """ Creates a dictionary that has the structure of the keys in the Stack
            It is used to loop through the stack without affecting it.
        """
        key_stack = {}
        for data_key in self.stack:
            key_stack[data_key] = {}
            for the_filter in self.stack[data_key][branch_pos]:
                key_stack[data_key][the_filter] = {}
                for x in self.stack[data_key][branch_pos][the_filter]:
                    key_stack[data_key][the_filter][x] = []
                    for y in self.stack[data_key][branch_pos][the_filter][x]:
                        link = self.stack[data_key][branch_pos][the_filter][x][y]
                        if not isinstance(link, Link):
                            continue
                        key_stack[data_key][the_filter][x].append(y)
        return key_stack

    def setup_stack_Example_Data_A(self, fk=None, xk=None, yk=None, views=None, weights=None):
        if fk is None:
            fk = [
                'no_filter',
                'Wave == 1'
            ]
        if xk is None:
            xk = self.minimum
        if yk is None:
            yk = ['@'] + self.minimum
        if views is None:
            views = ['default']
        if not isinstance(weights, list):
            weights = [weights]
         
        self.stack = Stack(name="Example Data (A)")
        self.stack.add_data(
            data_key=self.stack.name, 
            meta=self.example_data_A_meta, 
            data=self.example_data_A_data
        )
        
        for weight in weights:
            self.stack.add_link(
                data_keys=self.stack.name,
                filters=fk,
                x=xk, 
                y=yk, 
                views=QuantipyViews(views),
                weights=weight
            )            

if __name__ == '__main__':
    unittest.main()