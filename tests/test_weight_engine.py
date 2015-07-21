import pdb;
import unittest
import os.path
import numpy
import pandas as pd
import json
from core.weights.rim import Rim
from core.weights.weight_engine import WeightEngine


class TestEngine(unittest.TestCase):

    def setUp(self):
        ''' Simple engine without meta - engine_A
        '''
        path_data = 'tests/engine_A_data.csv'
        data = pd.read_csv(path_data)

        # Setup engine_A
        self.engine_A = WeightEngine(data=data)

        self.scheme_name_A1 = 'scheme_name_A1'
        self.scheme_name_A2 = 'scheme_name_A2'
        self.scheme_name_A3 = 'scheme_name_A3'

        # Setup schemes to use in tests
        self.scheme_A1 = Rim(self.scheme_name_A1)
        self.scheme_A1.lists = ['column1', 'column2']
        self.scheme_A1.add_group(name='Senior Type 1', filter='column3==1', 
            targets={
                'column1': [32.00, 31.00, 37.00],
                'column2': [23.13, 14.32, 4.78, 4.70, 2.65, 2.61, 3.47, 31.04, 13.3]
            })
        self.scheme_A1.add_group(name='Senior Type 2', filter='column3==1', 
            targets={
                'column1': [33.40, 33.40, 33.20],
                'column2': [11.11, 11.11, 11.11, 11.11, 11.11, 11.11, 11.11, 11.11, 11.12]
            })
        self.scheme_A1.add_group(name='Senior Type 3', filter='column3==3',
            targets={
                'column1': [33.2, 29.7, 37.1],
                'column2': [23.13, 14.32, 4.78, 4.70, 2.65, 2.61, 3.47, 31.04, 13.3]
            })
        self.scheme_A1.add_group(name='Senior Type 4', filter='column3==4',
            targets={
                'column1': [33.2, 29.7, 37.1],
                'column2': [23.13, 14.32, 4.78, 4.70, 2.65, 2.61, 3.47, 32.34, 12.00]
            })

        self.scheme_A2 = Rim(self.scheme_name_A2)
        self.scheme_A2.lists = ['column1', 'column2']
        self.scheme_A2.add_group(name='Senior Type 1', filter='column3==1', 
            targets={
                'column1': [37.00, 32.00, 31.00],
                'column2': [13.3, 23.13, 14.32, 4.78, 4.70, 2.65, 2.61, 3.47, 31.04]
            })
        self.scheme_A2.add_group(name='Senior Type 2', filter='column3==1', 
            targets={
                'column1': [33.2, 33.40, 33.40],
                'column2': [11.11, 11.11, 11.11, 11.11, 11.11, 11.11, 11.11, 11.11, 11.12]
            })
        self.scheme_A2.add_group(name='Senior Type 3', filter='column3==3',
            targets={
                'column1': [37.1, 33.2, 29.7],
                'column2': [13.3, 23.13, 14.32, 4.78, 4.70, 2.65, 2.61, 3.47, 31.04]
            })
        self.scheme_A2.add_group(name='Senior Type 4', filter='column3==4',
            targets={
                'column1': [37.1, 33.2, 29.7],
                'column2': [12.00, 23.13, 14.32, 4.78, 4.70, 2.65, 2.61, 3.47, 32.34]
            })

        self.scheme_A3 = Rim(self.scheme_name_A3)
        self.scheme_A3.lists = ['profile_gender']
        self.scheme_A3.targets = {'profile_gender' : [47, 53]}
        self.scheme_A3.add_group(
            name='11-19', filter='age_group=2', targets=self.scheme_A3.targets
        )
        self.scheme_A3.add_group(
            name='31-39', filter='age_group=4', targets=self.scheme_A3.targets
        )
        self.scheme_A3.add_group(
            name='41-49', filter='age_group=5', targets=self.scheme_A3.targets
        )
        self.scheme_A3.add_group(
            name='51-59', filter='age_group=6', targets=self.scheme_A3.targets
        )
        self.scheme_A3.group_targets({
             '11-19': 25, 
             '31-39': 25, 
             '41-49': 25, 
             '51-59': 25
        })
        
        ''' Complex engine with meta - engine_B
        '''
        path_data = 'tests/engine_B_data.csv'
        path_meta = 'tests/engine_B_meta.json'

        data = pd.read_csv(path_data)
        meta = json.load(file(path_meta))

        self.scheme_name_B1 = 'scheme_name_B1'

        engine_B = WeightEngine(data=data, meta=meta)
        
        # Setup schemes to use in tests
        self.scheme_B1 = Rim(self.scheme_name_B1)
        self.scheme_B1.lists = ['profile_gender', 'age_group']
        # self.scheme_B1.set_targets()

    def test_constructor(self):
        path_data = 'tests/engine_B_data.csv'
        path_meta = 'tests/engine_B_meta.json'

        data = pd.read_csv(path_data)
        meta = json.load(file(path_meta))
        
        engine_B = WeightEngine(data=data, meta=meta)

        self.assertIsNotNone(engine_B._df)
        self.assertTrue(engine_B.dropna)
        self.assertEqual(engine_B.schemes, {})
        self.assertIsInstance(engine_B.schemes, dict)

    def test_add_scheme_and_dataframe(self):
        #A list of scheme names used in setUp used for comparison
        scheme_names = [self.scheme_name_A1, self.scheme_name_A2]

        self.engine_A.add_scheme(scheme=self.scheme_A2, key='identity')
        # Should now contain a dict with scheme_name_A2 as the first key
        self.assertEqual(self.engine_A.schemes.keys()[0], self.scheme_name_A2)

        self.engine_A.add_scheme(scheme=self.scheme_A1, key='identity')
        # Should now contain a dict with scheme_name_A2 and scheme_name_A1 as keys
        for key in self.engine_A.schemes:
            self.assertIn(key, scheme_names)
            self.assertIn('identity', self.engine_A.schemes[key]['key'])

        # Sets weights_scheme_name_A1 and weights_scheme_name_A2 to ones
        self.engine_A._df[self.scheme_A1.weight_name()] = pd.np.ones(len(self.engine_A._df))
        self.engine_A._df[self.scheme_A2.weight_name()] = pd.np.ones(len(self.engine_A._df))

        for key in self.engine_A.schemes:
            weight_scheme = self.engine_A._df['weights_'+key]
            boolean_vector = (weight_scheme == pd.np.ones(len(weight_scheme)))
            self.assertTrue(boolean_vector.all())
            self.engine_A.run(schemes=[key])
            boolean_vector = (weight_scheme == pd.np.ones(len(weight_scheme)))
            self.assertFalse(boolean_vector.all())

    def test_add_scheme_no_key(self):
        self.engine_A.add_scheme(scheme=self.scheme_A1, key='identity')
        self.assertIsNotNone(self.engine_A.schemes[self.scheme_name_A1]['key'])

    def test_weight_lazy(self):
        return
        self.engine_A.add_scheme(scheme=self.scheme_A2, key='identity')
        self.engine_A.add_scheme(scheme=self.scheme_A1, key='identity')
        self.assertNotIn('weights_scheme_name_A2', self.engine_A._df.columns)
        self.engine_A.weight()

        self.assertIn('weights_%s' % self.scheme_name_A1, self.engine_A._df.columns)
        self.assertIn('weights_%s' % self.scheme_name_A2, self.engine_A._df.columns)

    def test_group_targets(self):
        path_data = 'tests/engine_B_data.csv'
        path_meta = 'tests/engine_B_meta.json'

        data = pd.read_csv(path_data)
        meta = json.load(file(path_meta))
        
        weight = '_'.join(
            ['weights', 
             self.scheme_name_A3]
        )
        
        # Run weights for scheme_A3
        engine_B = WeightEngine(data=data, meta=meta)
        engine_B.add_scheme(scheme=self.scheme_A3, key='identity')
        engine_B.run()

        data_A3 = engine_B.dataframe("scheme_name_A3")
        
        # check identical weighted column frequencies
        df = data_A3.pivot_table(
            values=[weight], 
            index=['profile_gender'], 
            columns=['age_group'], 
            aggfunc='sum'
        )  
        for column in df.columns.tolist():
            self.assertTrue(
                numpy.allclose(df[column].values, numpy.array([1.645, 1.855]))
            ) 
        
        #check the weight column counts & sum are equal to index length (14)
        a = numpy.asscalar(data_A3[weight].count())
        b = numpy.asscalar(data_A3[weight].sum())
        c = data_A3.shape[0]
        self.assertTrue(int(a) == int(b) == int(c))

        # check weighted group frequencies have euqal proportions
        values = data_A3.pivot_table(
            values=[weight], 
            index=['age_group'], 
            aggfunc='sum'
        ).values
        self.assertTrue(numpy.allclose(values, 3.5))
