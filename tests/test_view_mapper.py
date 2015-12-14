import pdb;
import unittest
import os.path
import numpy as np
import pandas as pd
import json

from quantipy.core.view_generators.view_maps import QuantipyViews
from quantipy.core.stack import Stack
from quantipy.core.link import Link
from quantipy.core.view_generators.view_mapper import ViewMapper
from quantipy.core.helpers import functions
from quantipy.core.helpers.functions import load_json

class TestViewObject(unittest.TestCase):

    def setUp(self):
        self.path = './tests/'
#         self.path = ''
        self.filepath = '%sengine_B_data.csv' % (self.path)
        self.metapath = '%sengine_B_meta.json' % (self.path)
        self.stack = Stack("StackName")
        self.stack.seperator = ','
        self.stack.decoding = "UTF-8"
        self.data = pd.DataFrame.from_csv(self.filepath)
        self.meta = load_json(self.metapath)
        self.stack.add_data(data_key="Jan", meta=self.meta, data=self.data)
#         self.x_names=['age', 'cost_breakfast', 'age_group', 'endtime', 'name', 'q4'],
        self.x_names = ['age', 'cost_breakfast', 'age_group', 'q4']
#         self._types = ['int', 'float', 'single', 'date', 'string', 'delimited set']
        self.x_types = ['int', 'float', 'single', 'delimited set']
        self.y_names = ['profile_gender']
        
    def test_add_method(self):
            views = ViewMapper()
    
            #At first there should not be method 'plus' in views
            self.assertRaises(KeyError, lambda: views['plus'])
            self.assertNotIn('plus', views.keys())
            
            #Plus method added:
            views.add_method('plus', lambda x: 2)
            
            #Checked for the existance of the plus method
            self.assertIsNotNone(views['plus']['method'])
            self.assertIsNotNone(views['plus']['kwargs'])
            self.assertIn('plus', views.keys())
    
            #Check to see wether a method is overridden properly
            views.add_method('plus', 'me')
            
            #The method should have changed from lambda x:2 to me
            self.assertEqual(views['plus']['method'], 'me')
            
#             Check to see wether a method is overridden properly
            views.add_method('minus', 'x', {'key': 'value'})
            self.assertEqual(views['minus']['method'], 'x')
            
    def test__custom_methods(self):
        views = ViewMapper()
        
        views.add_method('minus_x', 'x', {'key': 'value'})
        views.add_method('minus_y', 'y')
        views.add_method('minus_z', 'z')
        
        for view in ['minus_x', 'minus_y', 'minus_z']:
            self.assertIn(view, views._custom_methods())
            
        for view in ['X', 'Y', 'Z']:
            self.assertNotIn(view, views._custom_methods())
        
    def test__get_method_types(self):
        
        #not checked - time & string - need to implemented 
        #examples: time   - "endtime"
        #          string - "name"
        
        views = QuantipyViews(['default'])
        
        self.stack.add_link(data_keys="Jan",
                            x=self.x_names, 
                            y=self.y_names, 
                            views=views)
        
        for x_type, x_name in zip(self.x_types, self.x_names):
            link = self.stack['Jan']['no_filter'][x_name]['profile_gender']    
            self.assertEqual(x_type, views._get_method_types(link)[0])

    def test__apply_to(self):
        
        #not checked - time & string - need to implemented 
        #examples: time   - "endtime"
        #          string - "name"
        
        views = QuantipyViews(['cbase', 'counts', 'mean'])
        
        # This matches the view method names to their resulting view key
        # based on the new view notation
        notation = {
            'default': 'x|default|:|||default',
            'cbase': 'x|f|x:|||cbase',
            'counts': 'x|f|:|||counts',
            'mean': 'x|d.mean|:|||mean'
        }
        
        x_names = ['profile_gender', 'age_group', 'q4']
        y_names = x_names
        
        self.stack.add_link(data_keys="Jan",
                            x=x_names, 
                            y=y_names)
        
        for a_x in self.stack['Jan']['no_filter'].keys():
            for a_y in self.stack['Jan']['no_filter'][a_x].keys():
                link = self.stack['Jan']['no_filter'][a_x][a_y]
                if not link.y == "@":
                    views._apply_to(link)

        for view in views.keys():
            for a_x in self.stack['Jan']['no_filter'].keys():
                for a_y in self.stack['Jan']['no_filter'][a_x].keys():
                    if views[view]['method'] == 'descriptives':
                        if a_x == 'q4':
                            self.assertNotIn(notation[view], self.stack['Jan']['no_filter'][a_x][a_y].keys())
                        else:                        
                            self.assertIn(notation[view], self.stack['Jan']['no_filter'][a_x][a_y].keys())
                    else:
                        self.assertIn(notation[view], self.stack['Jan']['no_filter'][a_x][a_y].keys())

    def test_get_view_iterations(self):
        
        views = QuantipyViews(['default'])
        iterators = {'A': [1, 2], 'B': [3, 4, 5], 'C': [6, 7, 8, 9]}
        iterations = views.__get_view_iterations__(iterators)
        
        # Test that iterations has the anticipated number of items
        # It should be the multiplication of the len of all of 
        # iterators lists, in this case 2x3x4=24
        self.assertEqual(len(iterations), 24)
        
        self.assertEqual(iterations[0], {'A': 1, 'B': 3, 'C': 6})
        self.assertEqual(iterations[1], {'A': 1, 'B': 4, 'C': 6})
        self.assertEqual(iterations[2], {'A': 1, 'B': 5, 'C': 6})
        self.assertEqual(iterations[3], {'A': 1, 'B': 3, 'C': 7})
        self.assertEqual(iterations[4], {'A': 1, 'B': 4, 'C': 7})
        self.assertEqual(iterations[5], {'A': 1, 'B': 5, 'C': 7})
        self.assertEqual(iterations[6], {'A': 1, 'B': 3, 'C': 8})
        self.assertEqual(iterations[7], {'A': 1, 'B': 4, 'C': 8})
        self.assertEqual(iterations[8], {'A': 1, 'B': 5, 'C': 8})
        self.assertEqual(iterations[9], {'A': 1, 'B': 3, 'C': 9})
        self.assertEqual(iterations[10], {'A': 1, 'B': 4, 'C': 9})
        self.assertEqual(iterations[11], {'A': 1, 'B': 5, 'C': 9})
        self.assertEqual(iterations[12], {'A': 2, 'B': 3, 'C': 6})
        self.assertEqual(iterations[13], {'A': 2, 'B': 4, 'C': 6})
        self.assertEqual(iterations[14], {'A': 2, 'B': 5, 'C': 6})
        self.assertEqual(iterations[15], {'A': 2, 'B': 3, 'C': 7})
        self.assertEqual(iterations[16], {'A': 2, 'B': 4, 'C': 7})
        self.assertEqual(iterations[17], {'A': 2, 'B': 5, 'C': 7})
        self.assertEqual(iterations[18], {'A': 2, 'B': 3, 'C': 8})
        self.assertEqual(iterations[19], {'A': 2, 'B': 4, 'C': 8})
        self.assertEqual(iterations[20], {'A': 2, 'B': 5, 'C': 8})
        self.assertEqual(iterations[21], {'A': 2, 'B': 3, 'C': 9})
        self.assertEqual(iterations[22], {'A': 2, 'B': 4, 'C': 9})
        self.assertEqual(iterations[23], {'A': 2, 'B': 5, 'C': 9})

    def test_iterations_object(self):
        
        # Set up path to example files
        path_tests = self.path
        project_name = 'Example Data (A)'

        # Load Example Data (A) data and meta
        name_data = '%s.csv' % (project_name)
        path_data = '%s%s' % (path_tests, name_data)
        example_data_A_data = pd.DataFrame.from_csv(path_data)
        
        name_meta = '%s.json' % (project_name)
        path_meta = '%s%s' % (path_tests, name_meta)
        example_data_A_meta = load_json(path_meta)
 
        # Variables by type for Example Data A
        eda_int = ['record_number', 'unique_id', 'age', 'birth_day', 'birth_month']
        eda_float = ['weight', 'weight_a', 'weight_b']
        eda_single = ['gender', 'locality', 'ethnicity', 'religion', 'q1']
        eda_delimited_set = ['q2', 'q3', 'q8', 'q9']
        eda_string = ['q8a', 'q9a']
        eda_date = ['start_time', 'end_time']
        eda_time = ['duration']
        eda_array = ['q5', 'q6', 'q7']       
        eda_minimum = ['q2b', 'Wave', 'q2', 'q3', 'q5_1']
        
        # Create basic stack
        stack = Stack(name=project_name)
        stack.add_data(project_name, example_data_A_data, example_data_A_meta)
        stack.add_link(
            data_keys=project_name,
            filters=['no_filter'],
            x=eda_minimum,
            y=['@'],
            views=QuantipyViews(['default', 'cbase', 'counts', 'c%']),
            weights=[None, 'weight_a', 'weight_b']
        )
    
        # Get list of views created
        views_present = stack.describe(index=['view'])
        
        # Test that weighted an unweighted versions of all basic views
        # were created
        self.assertIn('x|default|:|||default', views_present)
        self.assertIn('x|default|:||weight_a|default', views_present)
        self.assertIn('x|default|:||weight_b|default', views_present)
        
        self.assertIn('x|f|x:|||cbase', views_present)
        self.assertIn('x|f|x:||weight_a|cbase', views_present)
        self.assertIn('x|f|x:||weight_b|cbase', views_present)
        
        self.assertIn('x|f|:|y||c%', views_present)
        self.assertIn('x|f|:|y|weight_a|c%', views_present)
        self.assertIn('x|f|:|y|weight_b|c%', views_present)
        
        self.assertIn('x|f|:|||counts', views_present)
        self.assertIn('x|f|:||weight_a|counts', views_present)
        self.assertIn('x|f|:||weight_b|counts', views_present)

        # Create a ViewMapper using the iterator object in a template
        xnets = ViewMapper(
            template={
                'method': QuantipyViews().frequency,
                'kwargs': {
                    'axis': 'x',
                    'groups': ['Nets'],
                    'iterators': {
                        'rel_to': [None, 'y'],
                        'axis': 'x',
                        'weights': [None, 'weight_a']
                    }
                }
            })
        
        # Add a method to the xnets ViewMapper, then use it to generate additional
        # views which include N/c% and unweighted/weighted
        xnets.add_method(name='ever', kwargs={'text': 'Ever', 'logic': [1, 2]})
        stack.add_link(x='q2b', y=['@'], views=xnets.subset(['ever']))
        
        # Get list of views created
        views_present = stack.describe(index=['view'])
        
        # Test that the expected views were all created
        self.assertIn('x|f|x[{1,2}]:|||ever', views_present)
        self.assertIn('x|f|x[{1,2}]:|y||ever', views_present)
        self.assertIn('x|f|x[{1,2}]:||weight_a|ever', views_present)
        self.assertIn('x|f|x[{1,2}]:|y|weight_a|ever', views_present)
        
        # Add another method to the xnets ViewMapper, but then override the weights
        # in the iterator object using the stack.add_link(weights) parameter
        stack.add_link(x='q2b', y=['@'], views=xnets.subset(['ever']), weights='weight_b')
        
        # Get list of views created
        views_present = stack.describe(index=['view'])
        
        # Test that the expected views were all created
        self.assertIn('x|f|x[{1,2}]:||weight_b|ever', views_present)
        self.assertIn('x|f|x[{1,2}]:|y|weight_b|ever', views_present)
        
        # Add two methods and apply them at the same time, make sure all expected iterations 
        # of both were created
        xnets.add_method(name='ever (multi test)', kwargs={'text': 'Ever', 'logic': [1, 2]})
        xnets.add_method(name='never (multi test)', kwargs={'text': 'Never', 'logic': [2, 3]})
        stack.add_link(x='q2b', y=['@'], views=xnets.subset(['ever (multi test)', 'never (multi test)']))
        
        # Get list of views created
        views_present = stack.describe(index=['view'])
        
        # Test that the expected views were all created
        self.assertIn('x|f|x[{1,2}]:|||ever (multi test)', views_present)
        self.assertIn('x|f|x[{1,2}]:|y||ever (multi test)', views_present)
        self.assertIn('x|f|x[{1,2}]:||weight_a|ever (multi test)', views_present)
        self.assertIn('x|f|x[{1,2}]:|y|weight_a|ever (multi test)', views_present)
        self.assertIn('x|f|x[{2,3}]:|||never (multi test)', views_present)
        self.assertIn('x|f|x[{2,3}]:|y||never (multi test)', views_present)
        self.assertIn('x|f|x[{2,3}]:||weight_a|never (multi test)', views_present)
        self.assertIn('x|f|x[{2,3}]:|y|weight_a|never (multi test)', views_present)
        
        # Add two methods and apply them at the same time, make sure all expected iterations 
        # of both were created, in this case that the weights arg for stack.add_link() overrides
        # what the iterator object is asking for
        xnets.add_method(name='ever (weights test)', kwargs={'text': 'Ever', 'logic': [1, 2]})
        xnets.add_method(name='never (weights test)', kwargs={'text': 'Never', 'logic': [2, 3]})
        stack.add_link(
            x='q2b', y=['@'], 
            views=xnets.subset(['ever (weights test)', 'never (weights test)']), 
            weights=['weight_b']
        )
        
        # Get list of views created
        views_present = stack.describe(index=['view'])
        
        # Test that the expected views were all created
        self.assertNotIn('x|f|x[{1,2}]:|||ever (weights test)', views_present)
        self.assertNotIn('x|f|x[{1,2}]:|y||ever (weights test)', views_present)
        self.assertIn('x|f|x[{1,2}]:||weight_b|ever (weights test)', views_present)
        self.assertIn('x|f|x[{1,2}]:|y|weight_b|ever (weights test)', views_present)
        self.assertNotIn('x|f|x[{2,3}]:|||never (weights test)', views_present)
        self.assertNotIn('x|f|x[{2,3}]:|y||never (weights test)', views_present)
        self.assertIn('x|f|x[{2,3}]:||weight_b|never (weights test)', views_present)
        self.assertIn('x|f|x[{2,3}]:|y|weight_b|never (weights test)', views_present)
        
if __name__ == '__main__':
    unittest.main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    