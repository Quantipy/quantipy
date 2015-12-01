import unittest
import os.path
import test_helper
import pandas as pd
from pandas.util.testing import assert_frame_equal

from quantipy.core.stack import Stack
from quantipy.core.chain import Chain
from quantipy.core.link import Link
from quantipy.core.helpers.functions import load_json
from quantipy.core.view_generators.view_maps import QuantipyViews
from quantipy.core.helpers import functions
 
class TestChainObject(unittest.TestCase):

    def setUp(self):        
        self.path = './tests/'
        self.path_chain = './temp.chain'.format(self.path)
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
        
        self.setup_chains_Example_Data_A()
        
    def test_save_chain(self):
        
        self.setup_chains_Example_Data_A()
        
        for chain in self.chains:
        
            chain.save(path=self.path_chain)
     
            loaded_chain = Chain.load(self.path_chain)
     
            # Create a dictionary with the attribute structure of the chain
            chain_attributes = test_helper.create_attribute_dict(chain)
     
            # Create a dictionary with the attribute structure of the chain
            loaded_chain_attributes = test_helper.create_attribute_dict(loaded_chain)
     
            # Ensure that we are not comparing the same variable (in memory)
            self.assertNotEqual(id(chain), id(loaded_chain))
     
            # Make sure that this is working by altering the loaded_stack_attributes
            # and comparing the result. (It should fail)
     
            # Change a 'value' in the dict
            loaded_chain_attributes['__dict__']['name'] = 'SomeOtherName'
            with self.assertRaises(AssertionError):
                self.assertEqual(chain_attributes, loaded_chain_attributes)
     
            # reset the value
            loaded_chain_attributes['__dict__']['name'] = chain_attributes['__dict__']['name']
            self.assertEqual(chain_attributes, loaded_chain_attributes)
     
            # Change a 'key' in the dict
            del loaded_chain_attributes['__dict__']['name']
            loaded_chain_attributes['__dict__']['new_name'] = chain_attributes['__dict__']['name']
            with self.assertRaises(AssertionError):
                self.assertEqual(chain_attributes, loaded_chain_attributes)
     
            # reset the value
            del loaded_chain_attributes['__dict__']['new_name']
            loaded_chain_attributes['__dict__']['name'] = chain_attributes['__dict__']['name']
            self.assertEqual(chain_attributes, loaded_chain_attributes)
     
            # Remove a key/value pair
            del loaded_chain_attributes['__dict__']['name']
            with self.assertRaises(AssertionError):
                self.assertEqual(chain_attributes, loaded_chain_attributes)
     
            # Cleanup
            if os.path.exists('./tests/{0}.chain'.format(chain.name)):
                os.remove('./tests/{0}.chain'.format(chain.name))
    
    def test_validate_x_y_combination(self):
        
        fk = 'no_filter'
        xk = self.minimum
        yk = ['@'] + self.minimum
        views = ['cbase', 'counts', 'c%']
                  
        # check the correct error message is returned, irrespective of orientation...
        
        # error #1
        expected_message = "If the number of keys for both x and y are greater than 1, whether or not you have specified the x and y values, orient_on must be either 'x' or 'y'."
        with self.assertRaises(ValueError) as error_message:
            _ = self.stack.get_chain(
                name='y', 
                data_keys=self.stack.name, 
                filters=fk, 
                x=xk, 
                y=yk, 
                views=views,
                post_process=True
            )
        self.assertEqual(error_message.exception[0], expected_message)

    def test_lazy_name(self):
        
        fk = 'no_filter'
        xk = self.minimum
        yk = ['@'] + self.minimum
        views = ['cbase', 'counts', 'c%']
    
        # get chain but do not name - y orientation
        chain_y = self.stack.get_chain(
                    data_keys=self.stack.name, 
                    filters=fk, 
                    x=xk, 
                    y=yk[0],  
                    views=views, 
                    post_process=False
                )
        
        # get chain but do not name - x orientation
        chain_x = self.stack.get_chain(
                    data_keys=self.stack.name,
                    filters=fk,  
                    x=xk[0], 
                    y=yk,  
                    views=views, 
                    post_process=False
                )
  
        # check lazy_name is working as it should be
        self.assertEqual(chain_y.name, 'y.@.q2b.Wave.q2.q3.q5_1.cbase.counts.c%')   
        self.assertEqual(chain_x.name, 'x.q2b.@.q2b.Wave.q2.q3.q5_1.cbase.counts.c%')         

    def test_dervie_attributes(self):
                
        # check chain attributes 
        self.assertEqual(self.chains[0].name, '@')
        self.assertEqual(self.chains[0].orientation, 'y')
        self.assertEqual(self.chains[0].source_name, '@')
        self.assertEqual(self.chains[0].len_of_axis, 5)
        self.assertEqual(self.chains[0].content_of_axis, ['q2b', 'Wave', 'q2', 'q3', 'q5_1'])
        self.assertEqual(self.chains[0].views, ['x|frequency|x:y|||cbase', 'x|frequency||||counts', 'x|frequency||y||c%'])
        self.assertEqual(self.chains[0].data_key, 'Example Data (A)')
        self.assertEqual(self.chains[0].filter, 'no_filter')
        self.assertEqual(self.chains[0].source_type, None)

        self.assertEqual(self.chains[-1].name, 'q5_1')
        self.assertEqual(self.chains[-1].orientation, 'x')
        self.assertEqual(self.chains[-1].source_name, 'q5_1')
        self.assertEqual(self.chains[-1].len_of_axis, 6)
        self.assertEqual(self.chains[-1].content_of_axis, ['@', 'q2b', 'Wave', 'q2', 'q3', 'q5_1'])
        self.assertEqual(self.chains[-1].views, ['x|frequency|x:y|||cbase', 'x|frequency||||counts', 'x|frequency||y||c%'])
        self.assertEqual(self.chains[-1].data_key, 'Example Data (A)')
        self.assertEqual(self.chains[-1].filter, 'no_filter')
        self.assertEqual(self.chains[-1].source_type, None)
    
    def test_post_process_shapes(self):
        
        # check chain attributes after post_processing 
        self.assertEqual(self.chains[0].x_new_order,  [[], [], [], [], [], [], [], [], [], [], [], [], [], [], []])
        self.assertEqual(self.chains[0].x_hidden_codes, [[], [], [], [], [], [], [], [], [], [], [], [], [], [], []])
        self.assertEqual(self.chains[0].y_new_order, None)
        self.assertEqual(self.chains[0].y_hidden_codes, None)        
        self.assertEqual(self.chains[0].props_tests, [])
        self.assertEqual(self.chains[0].props_tests_levels, [])
        self.assertEqual(self.chains[0].has_props_tests, False)
        self.assertEqual(self.chains[0].means_tests, [])
        self.assertEqual(self.chains[0].means_tests_levels, [])
        self.assertEqual(self.chains[0].has_means_tests, False)
        self.assertEqual(self.chains[0].view_sizes, [[(1, 1), (3, 1), (3, 1)], [(1, 1), (5, 1), (5, 1)], [(1, 1), (8, 1), (8, 1)], [(1, 1), (9, 1), (9, 1)], [(1, 1), (7, 1), (7, 1)]])
        self.assertEqual(self.chains[0].view_lengths, [[1, 3, 3], [1, 5, 5], [1, 8, 8], [1, 9, 9], [1, 7, 7]])
        self.assertEqual(self.chains[0].source_length, 1)
        
        self.assertEqual(self.chains[-1].x_new_order,  [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []])
        self.assertEqual(self.chains[-1].x_hidden_codes, [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []])
        self.assertEqual(self.chains[-1].y_new_order, None)
        self.assertEqual(self.chains[-1].y_hidden_codes, None)        
        self.assertEqual(self.chains[-1].props_tests, [])
        self.assertEqual(self.chains[-1].props_tests_levels, [])
        self.assertEqual(self.chains[-1].has_props_tests, False)
        self.assertEqual(self.chains[-1].means_tests, [])
        self.assertEqual(self.chains[-1].means_tests_levels, [])
        self.assertEqual(self.chains[-1].has_means_tests, False)
        self.assertEqual(self.chains[-1].view_sizes, [[(1, 1), (7, 1), (7, 1)], [(1, 3), (7, 3), (7, 3)], [(1, 5), (7, 5), (7, 5)], [(1, 8), (7, 8), (7, 8)], [(1, 9), (7, 9), (7, 9)], [(1, 7), (7, 7), (7, 7)]])
        self.assertEqual(self.chains[-1].view_lengths, [[1, 7, 7], [1, 7, 7], [1, 7, 7], [1, 7, 7], [1, 7, 7], [1, 7, 7]])
        self.assertEqual(self.chains[-1].source_length, 9)
    
    def test_describe(self):
        
        fk = 'no_filter'
        
        for chain in self.chains:
            
            chain_described = chain.describe()
            
            #test describe() returns a dataframe
            self.assertIsInstance(chain_described, pd.DataFrame)
            
            #test descibe() returns the expected dataframe - *no args* 
            if chain.orientation == 'y':
                keys = chain[self.stack.name][fk].keys()
                views = chain[self.stack.name][fk][keys[0]][chain.source_name].keys()
                data = [self.stack.name]*(len(keys)*len(views))
                filters = [fk]*(len(keys)*len(views))
                x = []
                for key in keys:
                    x.extend([key]*len(views))
                y = [chain.source_name]*(len(keys)*len(views))
                view = [v for v in views]*len(keys)
                ones = [1]*(len(keys)*len(views))
                df = pd.DataFrame({'data': data, 
                                   'filter': filters, 
                                   'x': x, 
                                   'y': y, 
                                   'view': view, 
                                   '#': ones})
                df = df[chain_described.columns.tolist()]
                assert_frame_equal(chain_described, df)
            elif chain.orientation == 'x':
                
                keys = chain[self.stack.name][fk][chain.source_name].keys()
                views = chain[self.stack.name][fk][chain.source_name][keys[0]].keys()
                data = [self.stack.name]*(len(keys)*len(views))
                filters = [fk]*(len(keys)*len(views))
                y = []
                for key in keys:
                    y.extend([key]*len(views))
                x = [chain.source_name]*(len(keys)*len(views))
                view = [v for v in views]*len(keys)
                ones = [1]*(len(keys)*len(views))
                df = pd.DataFrame({'data': data, 
                                   'filter': filters, 
                                   'x': x, 
                                   'y': y, 
                                   'view': view, 
                                   '#': ones})
                df = df[chain_described.columns.tolist()]
                assert_frame_equal(chain_described, df)
    
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
            fk = 'no_filter'
        if xk is None:
            xk = self.minimum
        if yk is None:
            yk = ['@'] + self.minimum
        if views is None:
            views = ['default', 'cbase', 'counts', 'c%'] 
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
        
    def setup_chains_Example_Data_A(self, fk=None, xk=None, yk=None, views=None, orient_on=None):
                    
        if fk is None:
            fk = 'no_filter'
        if xk is None:
            xk = self.minimum
        if yk is None:
            yk = ['@'] + self.minimum
        if views is None:
            views = [
                'x|frequency|x:y|||cbase',
                'x|frequency||||counts',
                'x|frequency||y||c%'
            ] 
        
        self.chains = []
        
        for y in yk:
            self.chains.append(
                self.stack.get_chain(
                    name=y, 
                    data_keys=self.stack.name, 
                    filters='no_filter', 
                    x=xk, 
                    y=y, 
                    views=views,
                    post_process=True
                )
            )
            
        for x in xk:
            self.chains.append(
                self.stack.get_chain(
                    name=x, 
                    data_keys=self.stack.name, 
                    filters='no_filter', 
                    x=x, 
                    y=yk, 
                    views=views,
                    post_process=True
                )
            )
        
if __name__ == '__main__':
    unittest.main()
