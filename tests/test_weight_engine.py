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
        self.scheme_A1.target_cols = ['column1', 'column2']
        self.scheme_A1.add_group(name='Senior Type 1', filter_def='column3==1', 
            targets=[
                {'column1': {code: prop for code, prop
                             in enumerate([32.00, 31.00, 37.00], start=1)}},
                {'column2': {code: prop for code, prop
                             in enumerate([23.13, 14.32, 4.78, 4.70, 2.65,
                                           2.61, 3.47, 31.04, 13.3], start=1)}}
            ])
        self.scheme_A1.add_group(name='Senior Type 2', filter_def='column3==1', 
            targets=[
                {'column1': {code: prop for code, prop
                             in enumerate([33.40, 33.40, 33.20], start=1)}},
                {'column2': {code: prop for code, prop
                             in enumerate([11.11, 11.11, 11.11, 11.11, 11.11,
                                           11.11, 11.11, 11.11, 11.12], start=1)}}
            ])
        self.scheme_A1.add_group(name='Senior Type 3', filter_def='column3==3',
            targets=[
                {'column1': {code: prop for code, prop
                             in enumerate([33.2, 29.7, 37.1], start=1)}},
                {'column2': {code: prop for code, prop
                             in enumerate([23.13, 14.32, 4.78, 4.70, 2.65,
                                           2.61, 3.47, 31.04, 13.3], start=1)}}
            ])
        self.scheme_A1.add_group(name='Senior Type 4', filter_def='column3==4',
            targets=[
                {'column1': {code: prop for code, prop
                             in enumerate([33.2, 29.7, 37.1], start=1)}},
                {'column2': {code: prop for code, prop
                             in enumerate([23.13, 14.32, 4.78, 4.70, 2.65,
                                           2.61, 3.47, 32.34, 12.00], start=1)}}
            ])

        self.scheme_A2 = Rim(self.scheme_name_A2)
        self.scheme_A2.target_cols = ['column1', 'column2']
        self.scheme_A2.add_group(name='Senior Type 1', filter_def='column3==1', 
            targets=[
                {'column1': {code: prop for code, prop
                             in enumerate([37.00, 32.00, 31.00], start=1)}},
                {'column2': {code: prop for code, prop
                             in enumerate([13.3, 23.13, 14.32, 4.78, 4.70,
                                           2.65, 2.61, 3.47, 31.04], start=1)}}
            ])
        self.scheme_A2.add_group(name='Senior Type 2', filter_def='column3==1', 
            targets=[
                {'column1': {code: prop for code, prop
                             in enumerate([33.2, 33.40, 33.40], start=1)}},
                {'column2': {code: prop for code, prop
                             in enumerate([11.11, 11.11, 11.11, 11.11, 11.11,
                                           11.11, 11.11, 11.11, 11.12], start=1)}}
            ])
        self.scheme_A2.add_group(name='Senior Type 3', filter_def='column3==3',
            targets=[
                {'column1': {code: prop for code, prop
                             in enumerate([37.1, 33.2, 29.7], start=1)}},
                {'column2': {code: prop for code, prop
                             in enumerate([13.3, 23.13, 14.32, 4.78, 4.70,
                                           2.65, 2.61, 3.47, 31.04], start=1)}}
            ])
        self.scheme_A2.add_group(name='Senior Type 4', filter_def='column3==4',
            targets=[
                {'column1': {code: prop for code, prop
                             in enumerate([37.1, 33.2, 29.7], start=1)}},
                {'column2': {code: prop for code, prop
                             in enumerate([12.00, 23.13, 14.32, 4.78, 4.70,
                                           2.65, 2.61, 3.47, 32.34], start=1)}}
            ])

        self.scheme_A3 = Rim(self.scheme_name_A3)
        self.scheme_A3.target_cols = ['profile_gender']
        self.scheme_A3.targets = [{'profile_gender' : {1: 47, 2: 53}}]
        self.scheme_A3.add_group(
            name='11-19', filter_def='age_group==2', targets=self.scheme_A3.targets
        )
        self.scheme_A3.add_group(
            name='31-39', filter_def='age_group==4', targets=self.scheme_A3.targets
        )
        self.scheme_A3.add_group(
            name='41-49', filter_def='age_group==5', targets=self.scheme_A3.targets
        )
        self.scheme_A3.add_group(
            name='51-59', filter_def='age_group==6', targets=self.scheme_A3.targets
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
        self.scheme_B1.target_cols = ['profile_gender', 'age_group']
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
        self.engine_A._df[self.scheme_A1._weight_name()] = pd.np.ones(len(self.engine_A._df))
        self.engine_A._df[self.scheme_A2._weight_name()] = pd.np.ones(len(self.engine_A._df))

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

    # def test_group_targets(self):
    #     path_data = 'tests/engine_B_data.csv'
    #     path_meta = 'tests/engine_B_meta.json'

    #     data = pd.read_csv(path_data)
    #     meta = json.load(file(path_meta))
        
    #     weight = '_'.join(
    #         ['weights', 
    #          self.scheme_name_A3]
    #     )
        

    #     # Run weights for scheme_A3
    #     engine_B = WeightEngine(data=data, meta=meta)
    #     engine_B.add_scheme(scheme=self.scheme_A3, key='identity')
    #     engine_B.run()

    #     data_A3 = engine_B.dataframe("scheme_name_A3")
        
    #     # check identical weighted column frequencies
    #     df = data_A3.pivot_table(
    #         values=[weight], 
    #         index=['profile_gender'], 
    #         columns=['age_group'], 
    #         aggfunc='sum'
    #     )

    #     for column in df.columns.tolist():
    #         self.assertTrue(
    #             numpy.allclose(df[column].values, numpy.array([1.645, 1.855]))
    #         ) 
        
    #     #check the weight column counts & sum are equal to index length (14)
    #     a = numpy.asscalar(data_A3[weight].count())
    #     b = numpy.asscalar(data_A3[weight].sum())
    #     c = data_A3.shape[0]
    #     self.assertTrue(int(a) == int(b) == int(c))

    #     # check weighted group frequencies have euqal proportions
    #     values = data_A3.pivot_table(
    #         values=[weight], 
    #         index=['age_group'], 
    #         aggfunc='sum'
    #     ).values
    #     self.assertTrue(numpy.allclose(values, 3.5))

    def test_vaidate_targets(self):
        path_data = 'tests/Example Data (A).csv'
        data = pd.read_csv(path_data)
        engine = WeightEngine(data)

        targets_gender = [45.6, 54.4]
        targets_locality = [10, 15, 20, 25, 30]
        weight_targets = [
                          {'gender': {code: prop for code, prop 
                                      in enumerate(targets_gender, start=1)}},
                          {'locality': {code: prop for code, prop
                                        in enumerate(targets_locality, start=1)}}
                          ]
        
        scheme = Rim('missing_data')
        scheme.set_targets(weight_targets)
        engine.add_scheme(scheme, key='unique_id')

        validate_df = scheme.validate()
        self.assertTrue(validate_df.columns.tolist() == ['missing', 'mean',
                                                         'mode', 'median'])
        self.assertTrue(validate_df.index.tolist() == ['gender', 'locality'])
        self.assertTrue(validate_df.values.tolist() == [[0.0, 2.0, 2.0, 2.0],
                                                        [177.0, 2.0, 1.0, 2.0]])


    def test_wdf_structure(self):
        path_data = 'tests/Example Data (A).csv'
        data = pd.read_csv(path_data)
        engine = WeightEngine(data)

        targets_gender = [45.6, 54.4]
        targets_locality = [10, 15, 20, 25, 30]
        weight_targets =  [
                          {'gender': {code: prop for code, prop 
                                      in enumerate(targets_gender, start=1)}},
                          {'locality': {code: prop for code, prop
                                        in enumerate(targets_locality, start=1)}}
                          ]
        
        scheme = Rim('complex_filter')
        
        scheme.add_group(name='W1, male',
                         filter_def='Wave==1 & religion==1',
                         targets=weight_targets)
        scheme.add_group(name='W2, female',
                         filter_def='Wave==2 & religion==2',
                         targets=weight_targets)
        
        engine.add_scheme(scheme, key='unique_id')
        engine.run()


        wdf = engine.dataframe('complex_filter')

        self.assertTrue(wdf.columns.tolist() == ['unique_id', 'gender',
                                                 'locality',
                                                 'weights_complex_filter',
                                                 'religion', 'Wave'])
        self.assertTrue(len(wdf.index) == 596)

