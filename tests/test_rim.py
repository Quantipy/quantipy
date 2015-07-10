import pdb;
import unittest
import os.path
import numpy
import pandas as pd
from core.weights.rim import Rim
from core.weights.weight_engine import WeightEngine


class TestScheme(unittest.TestCase):

    def setUp(self):
        self.analog_scheme = Rim('analog')

    def test_constructor(self):
        #Check to see if everything initialised correctly
        self.assertIsInstance(self.analog_scheme, Rim)
        self.assertEqual(self.analog_scheme.name, 'analog')
        self.assertEqual(self.analog_scheme.lists, [])
        self.assertEqual(self.analog_scheme.max_iterations, 1000)
        self.assertEqual(self.analog_scheme.convcrit, 0.01)
        self.assertEqual(self.analog_scheme.cap, 0)
        self.assertTrue(self.analog_scheme.dropna)
        self.assertIsInstance(self.analog_scheme._impute_method_specific, dict)
        self.assertIsNone(self.analog_scheme.weight_column_name)
        self.assertEqual(self.analog_scheme.total, 0)


    def test_renamefunc(self):
        # Test the rename function
        self.analog_scheme.lists = ['gender', 'column1c', 'q06', 'sta_wo', 'abschluss', 'q04']

        # This command should throw (raise) an exception not print to stdout (Maybe!)
        with self.assertRaises(ValueError):
            self.analog_scheme.rename_list(find='somethingnotinthelist', replace='With_this')

        rename_successfull = self.analog_scheme.rename_list(find='gender', replace='sex')
        self.assertTrue(rename_successfull)
        self.assertIn('sex', self.analog_scheme.lists)

    def test_cap(self):
        # Test the cap functionality
        self.assertEqual(0, self.analog_scheme.cap)
        self.assertFalse(self.analog_scheme.use_cap())

        self.analog_scheme.cap = 5
        self.assertEqual(5, self.analog_scheme.cap)
        self.assertTrue(self.analog_scheme.use_cap())

        #Check to see if it reverts properly to not using cap
        self.analog_scheme.cap = 0
        self.assertFalse(self.analog_scheme.use_cap())

    def test_groups(self):
        scheme = Rim('digital')
        self.assertIsInstance(scheme.groups, dict)
        self.assertIsNone(scheme.groups['__default_group_name__']['report'])
        self.assertIsNone(scheme.groups['__default_group_name__']['filters'])

        scheme.lists = ['gender', 'column1c', 'q06', 'sta_wo', 'abschluss', 'q04']

        scheme.add_group(name='Apple', filter='ownership==1')
        self.assertIn('Apple', scheme.groups)
        scheme.add_group(name='Samsung', filter={'ownership': 2})
        self.assertIn('Samsung', scheme.groups)
        
        # Try an invalid filter
        # It should not contain any filters since it was called incorrectly
        with self.assertRaises(Exception):
            scheme.add_group(name='Motorola', filter='ownership3')

        # Check to see if the available methods to add filter are equal
        self.assertEqual(scheme.groups['Samsung']['filters']['ownership'], 2)
        self.assertEqual(scheme.groups['Apple']['filters']['ownership'], 1)

        # Test for changes to valid filter
        scheme.group_filter(group_name='Apple', filter='ownership==8')
        self.assertEqual(scheme.groups['Apple']['filters']['ownership'], 8)

        # Test for changes to an non-existing filter, NOTE: the group_filter function actually creates the group motorola
        scheme.group_filter(group_name='Motorola', filter='ownership==12')
        self.assertEqual(scheme.groups['Motorola']['filters']['ownership'], 12)

        #Test for invalid filter type
        #Test for invalid group with dictionary
        with self.assertRaises(ValueError):
            scheme.group_filter(group_name='Motorola', filter=[])

        with self.assertRaises(ValueError):
            scheme.group_filter(group_name='Ferrari', filter={'ownership': 3})

        # Try to change a filter for invalid group
        self.assertNotIn('doesnotexist', scheme.groups.keys())
        # Try to make illegal changes to filter in a valid group
        self.assertNotEqual(scheme.groups['Apple']['filters']['ownership'], 1)
        with self.assertRaises(Exception):
            scheme.group_filter(group_name='Apple', filter='ownership1')

        #The targets should be empty lists
        for key in scheme.groups['Apple']['targets']:
            self.assertEqual(scheme.groups['Apple']['targets'][key], [])

        #Test for incorrect target change
        with self.assertRaises(ValueError):
            scheme.set_targets(group_name='Apple', targets={'gender': 1234})

        #Test for setting targets to a list that wasn't given
        # with self.assertRaises(ValueError):
        #     scheme.set_targets(group_name='Apple', targets={'doesnotexist': [80, 20]})
                
        #Set valid targets
        valid_targets={
            'gender': [50, 50],
            'column1c': [20, 18, 25, 21, 16],
            'q04': [20, 55, 12.5, 12.5],
            'sta_wo': [50, 50],
            'abschluss': [60, 40],
            'q06': [20, 20, 20, 20, 20]
        }

        scheme.set_targets(group_name='Apple', targets=valid_targets)

        #Test that only the most recently set targets are still in the scheme
        self.assertNotIn('doesnotexist', scheme.groups['Apple']['targets'].keys())

        #Test that the targets were applied to the lists corrected
        for key in scheme.groups['Apple']['targets']:
            self.assertEqual(scheme.groups['Apple']['targets'][key], valid_targets[key])
        
        #add group_targets
        scheme.group_targets(
            {
                "Apple": 30,
                "Samsung": 40,
                "Motorola": 30
            }
        )
        
        self.assertItemsEqual(scheme._group_targets.keys(), ['Motorola', 'Apple', 'Samsung'])
        self.assertItemsEqual(scheme._group_targets.values(), [0.3, 0.3, 0.4])