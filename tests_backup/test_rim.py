import pdb;
import unittest
import os.path
import numpy
import pandas as pd
from quantipy.core.weights.rim import Rim
from quantipy.core.weights.weight_engine import WeightEngine


class TestScheme(unittest.TestCase):

    def setUp(self):
        self.analog_scheme = Rim('analog')

    def test_constructor(self):
        #Check to see if everything initialised correctly
        self.assertIsInstance(self.analog_scheme, Rim)
        self.assertEqual(self.analog_scheme.name, 'analog')
        self.assertEqual(self.analog_scheme.max_iterations, 1000)
        self.assertEqual(self.analog_scheme.convcrit, 0.01)
        self.assertEqual(self.analog_scheme.cap, 0)
        self.assertTrue(self.analog_scheme.dropna)
        self.assertIsInstance(self.analog_scheme._specific_impute, dict)
        self.assertIsNone(self.analog_scheme.weight_column_name)
        self.assertEqual(self.analog_scheme.total, 0)

    def test_cap(self):
        # Test the cap functionality
        self.assertEqual(0, self.analog_scheme.cap)
        self.assertFalse(self.analog_scheme._use_cap())

        self.analog_scheme.cap = 5
        self.assertEqual(5, self.analog_scheme.cap)
        self.assertTrue(self.analog_scheme._use_cap())

        #Check to see if it reverts properly to not using cap
        self.analog_scheme.cap = 0
        self.assertFalse(self.analog_scheme._use_cap())

    def test_groups(self):
        scheme = Rim('digital')
        self.assertIsInstance(scheme.groups, dict)
        self.assertIsNone(scheme.groups['_default_name_']['report'])
        self.assertIsNone(scheme.groups['_default_name_']['filters'])

        scheme.lists = ['gender', 'column1c', 'q06', 'sta_wo', 'abschluss', 'q04']

        scheme.add_group(name='Apple', filter_def='ownership==1')
        self.assertIn('Apple', scheme.groups)
        scheme.add_group(name='Samsung', filter_def='ownership==2')
        self.assertIn('Samsung', scheme.groups)
        
        # Check to see if the available methods to add filter are equal
        self.assertEqual(scheme.groups['Samsung']['filters'], 'ownership==2')
        self.assertEqual(scheme.groups['Apple']['filters'], 'ownership==1')
        
        #The targets should be empty lists
        for key in scheme.groups['Apple']['targets']:
            self.assertEqual(scheme.groups['Apple']['targets'][key], [])

        #Test for incorrect target change
        #with self.assertRaises(ValueError):
        scheme.set_targets(group_name='Apple', targets=[
            {'gender': {1: 1234, 2:200}}
            ])
                
        #Set valid targets
        valid_targets=[
            {'gender': {code: prop for code, prop
                        in enumerate([50, 50], start=1)}},
            {'column1c': {code: prop for code, prop
                          in enumerate([20, 18, 25, 21, 16], start=1)}},
            {'q04': {code: prop for code, prop
                     in enumerate([20, 55, 12.5, 12.5], start=1)}},
            {'sta_wo': {code: prop for code, prop
                        in enumerate([50, 50], start=1)}},
            {'abschluss': {code: prop for code, prop 
                           in enumerate([60, 40], start=1)}},
            {'q06': {code: prop for code, prop
                     in enumerate([20, 20, 20, 20, 20], start=1)}}
        ]

        scheme.set_targets(group_name='Apple', targets=valid_targets)

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