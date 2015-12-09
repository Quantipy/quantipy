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
    get_logic_index,
    get_logic_key
)

class TestStackObject(unittest.TestCase):

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
        self.single = ['gender', 'locality', 'ethnicity', 'religion', 'q1']
        self.delimited_set = ['q2', 'q3', 'q8', 'q9']
        

    def test_has_not_any(self):
        # Test has version
        test_values = [1, 3, 5]
        func, values, exclusive = has_any(test_values)
        self.assertEqual(func, _has_any)
        self.assertEqual(values, test_values)
        self.assertEqual(exclusive, False)

        # Test not version
        test_values = [1, 3, 5]
        func, values, exclusive = not_any(test_values)
        self.assertEqual(func, _not_any)
        self.assertEqual(values, test_values)
        self.assertEqual(exclusive, False)


    def test_has_not_any_errors(self):
        
        # Test values not given as a list
        gender = self.example_data_A_data['gender']
        for test_values in [1, 'wrong!']:
            # Test has version
            with self.assertRaises(TypeError) as error:
                # Test _has_all raises TypeError
                func, values = has_any(test_values)
            self.assertEqual(
                error.exception.message[:54],
                "The values given to has_any() must be given as a list."
            )
            # Test not version
            with self.assertRaises(TypeError) as error:
                # Test _has_all raises TypeError
                func, values = not_any(test_values)
            self.assertEqual(
                error.exception.message[:54],
                "The values given to not_any() must be given as a list."
            )
 
        # Test values inside the values are not int
        gender = self.example_data_A_data['gender']
        for test_values in [['1'], [1.9, 2, 3]]:
            # Test has version
            with self.assertRaises(TypeError) as error:
                # Test _has_all raises TypeError
                func, values = has_any(test_values)
            self.assertEqual(
                error.exception.message[:54],
                "The values given to has_any() are not correctly typed."
            )
            # Test not version
            with self.assertRaises(TypeError) as error:
                # Test _has_all raises TypeError
                func, values = not_any(test_values)
            self.assertEqual(
                error.exception.message[:54],
                "The values given to not_any() are not correctly typed."
            )
 
 
    def test__has_not_any(self):
         
        # Test _has_any on delimited set
        q2 = self.example_data_A_data['q2']
        q2_verify_unchanged = q2.copy()
        for test_values in [[1, 3, 5], [2, 4, 6], [999]]:
            # Make string versions of test_values
            str_test_values = [str(v) for v in test_values]            
            # Test _has_any returns correct results
            idx = _has_any(q2, test_values)
            self.assertTrue(
                all([
                     any([tv in v for tv in str_test_values])
                     for v in q2[idx]
                ])
            )
            # Test inverse index produced by not version
            not_idx = _not_any(q2, test_values)
            self.confirm_inverse_index(q2, idx, not_idx)
        self.assertTrue((q2.fillna(0)==q2_verify_unchanged.fillna(0)).all())
   
         
        # Test _has_any on single that is stored as int64
        gender = self.example_data_A_data['gender']
        gender_verify_unchanged = gender.copy()
        for test_values in [[1], [1, 2], [999]]:
            # Test _has_any returns correct results
            idx = _has_any(gender, test_values)
            filter_idx = gender.isin(test_values)
            self.assertTrue(
                all(gender[idx].index==gender[filter_idx].index)
            )
            # Test inverse index produced by not version
            not_idx = _not_any(gender, test_values)
            self.confirm_inverse_index(gender, idx, not_idx)
        self.assertTrue((gender.fillna(0)==gender_verify_unchanged.fillna(0)).all())
         
        # Test _has_any on single that is stored as float64
        locality = self.example_data_A_data['locality']
        locality_verify_unchanged = locality.copy()
        for test_values in [[1], [1, 2], [999]]:
            # Test _has_any returns correct results
            idx = _has_any(locality, test_values)
            filter_idx = locality.isin(test_values)
            self.assertTrue(
                all(locality[idx].index==locality[filter_idx].index)
            )
            # Test inverse index produced by not version
            not_idx = _not_any(locality, test_values)
            self.confirm_inverse_index(locality, idx, not_idx)
        self.assertTrue((locality.fillna(0)==locality_verify_unchanged.fillna(0)).all())
         
        # Test _has_any using exclusivity
        q2 = self.example_data_A_data['q2']
        q2_verify_unchanged = q2.copy()
        for test_values in [[1, 3, 5], [2, 4, 6]]:
            # Make string versions of test_values
            str_test_values = [str(v) for v in test_values]            
            # Test _has_all returns correct results
            idx = _has_any(q2, test_values, True)
            resulting_columns = q2[idx].astype('object').str.get_dummies(';').columns
            self.assertTrue(all([
                int(col) in test_values
                for col in resulting_columns
            ]))
            # Test inverse index produced by not version
            not_idx = _not_any(q2, test_values, True)
            self.confirm_inverse_index(q2, idx, not_idx)
        self.assertTrue((q2.fillna(0)==q2_verify_unchanged.fillna(0)).all())
         
 
    def test__has_not_any_errors(self):
         
        # Test unsupported dtype series is given
        start_time = self.example_data_A_data['start_time']
        start_time = start_time.astype(np.datetime64)
        # Test has version
        with self.assertRaises(TypeError) as error:
            test_values = [1, 2]
            # Test _has_all raises TypeError
            idx = _has_any(start_time, test_values)
        self.assertEqual(
            error.exception.message[:56],
            "The series given to has_any() must be a supported dtype."
        )
        # Test not version
        with self.assertRaises(TypeError) as error:
            test_values = [1, 2]
            # Test _has_all raises TypeError
            idx = _not_any(start_time, test_values)
        self.assertEqual(
            error.exception.message[:56],
            "The series given to not_any() must be a supported dtype."
        )
 

    def test_has_all(self):
        # Test has version
        test_values = [1, 3, 5]
        func, values, exclusive = has_all(test_values)
        self.assertEqual(func, _has_all)
        self.assertEqual(values, test_values)
        self.assertEqual(exclusive, False)
 
        # Test not version
        test_values = [1, 3, 5]
        func, values, exclusive = not_all(test_values)
        self.assertEqual(func, _not_all)
        self.assertEqual(values, test_values)
        self.assertEqual(exclusive, False)


    def test_has_all_errors(self):
         
        # Test values not given as a list
        gender = self.example_data_A_data['gender']
        for test_values in [1, 'wrong!']:
            # Test has version
            with self.assertRaises(TypeError) as error:
                func, values = has_all(test_values)
            self.assertEqual(
                error.exception.message[:54],
                "The values given to has_all() must be given as a list."
            )
            # Test not version
            with self.assertRaises(TypeError) as error:
                func, values = not_all(test_values)
            self.assertEqual(
                error.exception.message[:54],
                "The values given to not_all() must be given as a list."
            )
 
        # Test values inside the values are not int
        gender = self.example_data_A_data['gender']
        for test_values in [['1'], [1.9, 2, 3]]:
            # Test has version
            with self.assertRaises(TypeError) as error:
                func, values = has_all(test_values)
            self.assertEqual(
                error.exception.message[:54],
                "The values given to has_all() are not correctly typed."
            )
            # Test not version
            with self.assertRaises(TypeError) as error:
                func, values = not_all(test_values)
            self.assertEqual(
                error.exception.message[:54],
                "The values given to not_all() are not correctly typed."
            )
 
 
    def test__has_not_all(self):
         
        # Test _has_all on delimited set
        q2 = self.example_data_A_data['q2']
        q2_verify_unchanged = q2.copy()
        for test_values in [[1, 3, 5], [2, 4, 6], [999]]:
            # Make string versions of test_values
            str_test_values = [str(v) for v in test_values]            
            # Test _has_all returns correct results
            idx = _has_all(q2, test_values)
            self.assertTrue(
                all([
                     all([tv in v for tv in str_test_values])
                     for v in q2[idx]
                ])
            )
            # Test inverse index produced by not version
            not_idx = _not_all(q2, test_values)
            self.confirm_inverse_index(q2, idx, not_idx)
        self.assertTrue((q2.fillna(0)==q2_verify_unchanged.fillna(0)).all())
         
        # Test _has_all on single that is stored as int64
        gender = self.example_data_A_data['gender']
        gender_verify_unchanged = gender.copy()
        for test_values in [[1], [999]]:
            # Test _has_all returns correct results
            idx = _has_all(gender, test_values)
            filter_idx = gender.isin(test_values)
            self.assertTrue(
                all(gender[idx].index==gender[filter_idx].index)
            )
            # Test inverse index produced by not version
            not_idx = _not_all(gender, test_values)
            self.confirm_inverse_index(gender, idx, not_idx)
        self.assertTrue((gender.fillna(0)==gender_verify_unchanged.fillna(0)).all())
         
        # Test _not_any on single that is stored as float64
        locality = self.example_data_A_data['locality']
        locality_verify_unchanged = locality.copy()
        for test_values in [[1], [999]]:
            # Test _not_any returns correct results
            idx = _has_all(locality, test_values)
            filter_idx = locality.isin(test_values)
            self.assertTrue(
                all(locality[idx].index==locality[filter_idx].index)
            )
            # Test inverse index produced by not version
            not_idx = _not_any(locality, test_values, True)
            self.confirm_inverse_index(locality, idx, not_idx)
        self.assertTrue((locality.fillna(0)==locality_verify_unchanged.fillna(0)).all())
        
        # Test _has_all using exclusivity
        q2 = self.example_data_A_data['q2']
        q2_verify_unchanged = q2.copy()
        for test_values in [[1, 3, 5], [2, 4, 6]]:
            # Make string versions of test_values
            str_test_values = [str(v) for v in test_values]            
            # Test _has_all returns correct results
            idx = _has_all(q2, test_values, True)
            resulting_columns = q2[idx].str.get_dummies(';').columns
            self.assertItemsEqual(
                [int(col) for col in resulting_columns],
                test_values
            )
            # Test inverse index produced by not version
            not_idx = _not_all(q2, test_values, True)
            self.confirm_inverse_index(q2, idx, not_idx)
        self.assertTrue((q2.fillna(0)==q2_verify_unchanged.fillna(0)).all())
         
        
    def test__has_not_all_errors(self):
         
        # Test unsupported dtype series is given
        start_time = self.example_data_A_data['start_time']
        start_time = start_time.astype(np.datetime64)
        # Test has version
        with self.assertRaises(TypeError) as error:
            test_values = [1, 2]
            idx = _has_all(start_time, test_values)
        self.assertEqual(
            error.exception.message[:56],
            "The series given to has_all() must be a supported dtype."
        )
        # Test not version
        with self.assertRaises(TypeError) as error:
            test_values = [1, 2]
            idx = _not_all(start_time, test_values)
        self.assertEqual(
            error.exception.message[:56],
            "The series given to not_all() must be a supported dtype."
        )
 
 
    def test_has_not_count(self):
        # Test has versions
        func, values, exclusive = has_count(1)
        self.assertEqual(func, _has_count)
        self.assertEqual(values, [1])
        self.assertEqual(exclusive, False)
 
        func, values, exclusive = has_count([1])
        self.assertEqual(func, _has_count)
        self.assertEqual(values, [1])
        self.assertEqual(exclusive, False)
 
        test_values = [1, 3]
        func, values, exclusive = has_count(test_values)
        self.assertEqual(func, _has_count)
        self.assertEqual(values, test_values)
        self.assertEqual(exclusive, False)
 
        test_values = [1, 3, [1, 2, 3]]
        func, values, exclusive = has_count(test_values)
        self.assertEqual(func, _has_count)
        self.assertEqual(values, test_values)
        self.assertEqual(exclusive, False)
 
        test_values = [1, 3]
        func, values, exclusive = has_count(test_values)
        self.assertEqual(func, _has_count)
        self.assertEqual(values, test_values)
        self.assertEqual(exclusive, False)
 
        test_values = [1, 3, [1, 2, 3]]
        func, values, exclusive = has_count(test_values)
        self.assertEqual(func, _has_count)
        self.assertEqual(values, test_values)
        self.assertEqual(exclusive, False)
 
        for op_func in [is_lt, is_le, is_eq, is_ne, is_ge, is_gt]:
            
            test_values = [op_func(3)]
            func, values, exclusive = has_count(test_values)
            self.assertEqual(func, _has_count)
            self.assertEqual(values, test_values)
            self.assertEqual(exclusive, False)
     
            test_values = [op_func(3), [1, 2, 3]]
            func, values, exclusive = has_count(test_values)
            self.assertEqual(func, _has_count)
            self.assertEqual(values, test_values)
            self.assertEqual(exclusive, False)

        # Test not versions
        func, values, exclusive = not_count(1)
        self.assertEqual(func, _not_count)
        self.assertEqual(values, [1])
        self.assertEqual(exclusive, False)
 
        func, values, exclusive = not_count([1])
        self.assertEqual(func, _not_count)
        self.assertEqual(values, [1])
        self.assertEqual(exclusive, False)
 
        test_values = [1, 3]
        func, values, exclusive = not_count(test_values)
        self.assertEqual(func, _not_count)
        self.assertEqual(values, test_values)
        self.assertEqual(exclusive, False)
 
        test_values = [1, 3, [1, 2, 3]]
        func, values, exclusive = not_count(test_values)
        self.assertEqual(func, _not_count)
        self.assertEqual(values, test_values)
        self.assertEqual(exclusive, False)
 
        test_values = [1, 3]
        func, values, exclusive = not_count(test_values)
        self.assertEqual(func, _not_count)
        self.assertEqual(values, test_values)
        self.assertEqual(exclusive, False)
 
        test_values = [1, 3, [1, 2, 3]]
        func, values, exclusive = not_count(test_values)
        self.assertEqual(func, _not_count)
        self.assertEqual(values, test_values)
        self.assertEqual(exclusive, False)
 
        for op_func in [is_lt, is_le, is_eq, is_ne, is_ge, is_gt]:
            
            test_values = [op_func(3)]
            func, values, exclusive = not_count(test_values)
            self.assertEqual(func, _not_count)
            self.assertEqual(values, test_values)
            self.assertEqual(exclusive, False)
     
            test_values = [op_func(3), [1, 2, 3]]
            func, values, exclusive = not_count(test_values)
            self.assertEqual(func, _not_count)
            self.assertEqual(values, test_values)
            self.assertEqual(exclusive, False)


    def test_has_not_count_errors(self):

        responses_tests = [
            [],
            [1, 2, 3, 4],
        ]
        for responses in responses_tests:
            # Test has version
            with self.assertRaises(IndexError) as error:
                func, values, exclusive = has_count(responses)
            self.assertEqual(
                error.exception.message[:85],
                "The responses list given to has_count() must have "
                "either 1, 2 or 3 items in the form:"
            )
            # Test not version
            with self.assertRaises(IndexError) as error:
                # Test _has_all raises TypeError
                func, values, exclusive = not_count(responses)
            self.assertEqual(
                error.exception.message[:85],
                "The responses list given to not_count() must have "
                "either 1, 2 or 3 items in the form:"
            )
        
        responses_tests = [
            '1',
            None,
            1.5,
            ['1'],
        ]
        for responses in responses_tests:
            # Test has version
            with self.assertRaises(TypeError) as error:
                func, values = has_count(responses)
            self.assertEqual(
                error.exception.message[:59],
                "The count target given to has_count() is "
                "incorrectly typed."
            )
            # Test not version
            with self.assertRaises(TypeError) as error:
                func, values = not_count(responses)
            self.assertEqual(
                error.exception.message[:59],
                "The count target given to not_count() is "
                "incorrectly typed."
            )
          
        responses_tests = [
            ['1', 2],
            [1, '2'],
            [None, 2],
            [2, None],
            [1.5, 2],
            [1, 2.5],
            [lambda x: x, 2],
            [1, is_lt]
        ]          
        for responses in responses_tests:
            # Test has version
            with self.assertRaises(TypeError) as error:
                func, values = has_count(responses)
            self.assertEqual(
                error.exception.message[:63],
                "The values subset given to has_count() are "
                "not correctly typed."
            )
            # Test not version
            with self.assertRaises(TypeError) as error:
                func, values = not_count(responses)
            self.assertEqual(
                error.exception.message[:63],
                "The values subset given to not_count() are "
                "not correctly typed."
            )
        
        values_tests = copy.copy(responses_tests)
        [r.append([1, 2, 3]) for r in values_tests]      
        for responses in values_tests:
            # Test has version
            with self.assertRaises(TypeError) as error:
                func, values = has_count(responses)
            self.assertEqual(
                error.exception.message[:63],
                "The values subset given to has_count() are "
                "not correctly typed."
            )
            # Test not version
            with self.assertRaises(TypeError) as error:
                func, values = not_count(responses)
            self.assertEqual(
                error.exception.message[:63],
                "The values subset given to not_count() are "
                "not correctly typed."
            )
        
        
        values_tests = [
            [1, 2, ['1', 2, 3]],
            [1, 2, [1, '2', 3]],
            [1, 2, [1, 2, '3']],
            [1, 2, [1.5, 2, 3]],
            [1, 2, [1, 2.5, 3]],
            [1, 2, [1, 2, 3.5]],
            [1, 2, [None, 2, 3]],
            [1, 2, [1, None, 3]],
            [1, 2, [1, 2, None]],
        ]
        for responses in values_tests:
            # Test has version
            with self.assertRaises(TypeError) as error:
                func, values = has_count(responses)
            self.assertEqual(
                error.exception.message[:63],
                "The values subset given to has_count() are"
                " not correctly typed."
            )
            # Test not version
            with self.assertRaises(TypeError) as error:
                func, values = not_count(responses)
            self.assertEqual(
                error.exception.message[:63],
                "The values subset given to not_count() are"
                " not correctly typed."
            )
         
 
    def test__has_not_count(self):
        
        test_vars = [
            'q2',       # Test on delimited set
            'gender',   # Test on single stored as int64
            'locality'  # Test on single stored as float64
        ]
        
        # Test non-operator-lead logical comparisons
        for var_name in test_vars:
            test_var = self.example_data_A_data[var_name]
            response_tests = [
                [1],
                [1, 2],
                [1, 3],
                [1, 3, [1, 2, 3]]
            ]
            for test_responses in response_tests:
                
                # Test _has_count returns correct results
                idx = _has_count(test_var, test_responses)
                
                # Determine min/max values for logic and
                # slice dummies column-wise for targeted values subset
                dummies, _min, _max = self.get_count_nums(
                    test_var[idx], 
                    test_responses
                )
                
                # Count the number of resposnes per row
                test_var_counts = dummies.sum(axis=1).unique()
                
                if len(test_responses)==1:
                    # Test single targeted response count
                    self.assertEqual(test_var_counts, [_min])
                else:
                    value_range = range(_min, _max+1)
                    # Positive test range of response count
                    self.assertTrue(all([
                        c in value_range
                        for c in test_var_counts
                    ]))
                    
                # Test inverse index produced by not version
                not_idx = _not_count(test_var, test_responses)
                self.confirm_inverse_index(test_var, idx, not_idx)
        
        # Test operator-lead logical comparisons
        __op_symbol__ = {
            _is_lt: '<', _is_le: '<=', 
            _is_eq: '', _is_ne: '!=', 
            _is_ge: '>=', _is_gt: '>'
        }
        __op_map__ = {
            _is_lt: lt, _is_le: le,
            _is_eq: eq, _is_ne: ne,
            _is_ge: ge, _is_gt: gt
        }
        for op_func in [_is_lt, _is_le, _is_eq, _is_ne, _is_ge, _is_gt]:
            key_part = __op_symbol__[op_func]
            
            for var_name in test_vars:
                test_var = self.example_data_A_data[var_name]
                response_tests = [
                    [(op_func, 3)],
                    [(op_func, 3), [1, 2, 3]]
                ]
                for test_responses in response_tests:
                    
                    # Test _has_count returns correct results
                    idx = _has_count(test_var, test_responses)
                    
                    # Determine min/max values for logic and
                    # slice dummies column-wise for targeted values subset
                    dummies, dum_func, _max = self.get_count_nums(
                        test_var[idx], 
                        test_responses
                    )
                    numerator = dum_func[1]
                    
                    try:
                        values = test_responses[1]
                        values = [str(v) for v in values if str(v) in dummies.columns]
                        dummies = dummies[values]
                    except:
                        pass
                    
                    # Count the number of resposnes per row
                    test_var_counts = dummies.sum(axis=1).unique()
                    
                    # Positive test range of response count
                    self.assertTrue(all(__op_map__[op_func](
                        test_var_counts, 
                        numerator
                    )))
                    
                    if op_func in [_is_ge, _is_eq] and numerator > 0:
                        incl_na = False
                    elif op_func in [_is_gt]:
                        incl_na = False
                    else:
                        incl_na = True
                    
                    # Test inverse index produced by not version
                    not_idx = _not_count(test_var, test_responses)
                    self.confirm_inverse_index(
                        test_var, 
                        idx, 
                        not_idx, 
                        incl_na
                    )
               
        
        # Test non-operator-lead logical comparisons with 
        # exclusivity
        for var_name in test_vars:
            test_var = self.example_data_A_data[var_name]
            response_tests = [
                [1, 3, [1, 2, 3]]
            ]
            for test_responses in response_tests:
                
                # Test _has_count returns correct results
                idx = _has_count(test_var, test_responses, True)
                
                # Determine min/max values for logic and
                # slice dummies column-wise for targeted values subset
                dummies, _min, _max = self.get_count_nums(
                    test_var[idx], 
                    test_responses
                )
                
                # Count the number of resposnes per row
                test_var_counts = dummies.sum(axis=1).unique()
                
                value_range = range(_min, _max+1)
                # Positive test range of response count
                self.assertTrue(all([
                    c in value_range
                    for c in test_var_counts
                ]))
                # Negative test for exclusivity
                all_dummies = test_var.astype('object').str.get_dummies(';')
                other_cols = [
                    c for c in all_dummies.columns 
                    if not c in dummies.columns
                ]
                other_dummies = all_dummies[other_cols]
                other_any_mask = other_dummies.any(axis=1)
                other_dummies = other_dummies[other_any_mask]
                self.assertEqual(
                    other_dummies.index.intersection(dummies.index).size,
                    0
                )
                
                # Test inverse index produced by not version
                not_idx = _not_count(test_var, test_responses, True)
                self.confirm_inverse_index(test_var, idx, not_idx)


    def test__has_not_count_errors(self):
          
        # Test unsupported dtype series is given
        start_time = self.example_data_A_data['start_time']
        start_time = start_time.astype(np.datetime64)
        
        # Test has version
        with self.assertRaises(TypeError) as error:
            test_values = [1, 2]
            # Test _has_count raises TypeError
            idx = _has_count(start_time, test_values)
        self.assertEqual(
            error.exception.message[:58],
        "The series given to has_count() must be a supported dtype."
        )
        
        # Test not version
        with self.assertRaises(TypeError) as error:
            test_values = [1, 2]
            # Test _has_count raises TypeError
            idx = _not_count(start_time, test_values)
        self.assertEqual(
            error.exception.message[:58],
        "The series given to not_count() must be a supported dtype."
        )
      
    def test_is_lt(self):
        test_value = 5
        func, value = is_lt(test_value)
        self.assertEqual(func, _is_lt)
        self.assertEqual(value, test_value)
        
        
    def test___lt(self):
        test_value = 30
        age = self.example_data_A_data['age']
        idx = _is_lt(age, test_value)        
        self.assertTrue(all(age[idx] < test_value))
        

    def test_is_le(self):
        test_value = 5
        func, value = is_le(test_value)
        self.assertEqual(func, _is_le)
        self.assertEqual(value, test_value)
        
        
    def test___le(self):
        test_value = 30
        age = self.example_data_A_data['age']
        idx = _is_le(age, test_value)        
        self.assertTrue(all(age[idx] <= test_value))
        

    def test_is_eq(self):
        test_value = 5
        func, value = is_eq(test_value)
        self.assertEqual(func, _is_eq)
        self.assertEqual(value, test_value)
        
        
    def test___eq(self):
        test_value = 30
        age = self.example_data_A_data['age']
        idx = _is_eq(age, test_value)        
        self.assertTrue(all(age[idx] == test_value))
        

    def test_is_ne(self):
        test_value = 5
        func, value = is_ne(test_value)
        self.assertEqual(func, _is_ne)
        self.assertEqual(value, test_value)
        
        
    def test___ne(self):
        test_value = 30
        age = self.example_data_A_data['age']
        idx = _is_ne(age, test_value)        
        self.assertTrue(all(age[idx] != test_value))
        

    def test_is_ge(self):
        test_value = 5
        func, value = is_ge(test_value)
        self.assertEqual(func, _is_ge)
        self.assertEqual(value, test_value)
        
        
    def test___ge(self):
        test_value = 30
        age = self.example_data_A_data['age']
        idx = _is_ge(age, test_value)        
        self.assertTrue(all(age[idx] >= test_value))
        

    def test_is_gt(self):
        test_value = 5
        func, value = is_gt(test_value)
        self.assertEqual(func, _is_gt)
        self.assertEqual(value, test_value)
        
        
    def test___gt(self):
        test_value = 30
        age = self.example_data_A_data['age']
        idx = _is_gt(age, test_value)        
        self.assertTrue(all(age[idx] > test_value))
        

    def test_union(self):
        q2 = self.example_data_A_data['q2']       
        test_logic = (has_all([1, 2]), Index.union, has_any([3, 4]))
        idx1, vkey1 = get_logic_index(q2, test_logic[0])
        idx2, vkey2 = get_logic_index(q2, test_logic[2])
        idx, vkey = get_logic_index(q2, test_logic)
        self.assertItemsEqual(
            idx,
            idx1.union(idx2)
        )
        self.assertEqual(
            vkey,
            'x[({1&2},{3,4})]:y'
        )
         
         
    def test_intersection(self):
        q2 = self.example_data_A_data['q2']       
        test_logic = (has_all([1, 2]), Index.intersection, has_any([3, 4]))
        idx1, vkey1 = get_logic_index(q2, test_logic[0])
        idx2, vkey2 = get_logic_index(q2, test_logic[2])
        idx, vkey = get_logic_index(q2, test_logic)
        self.assertItemsEqual(
            idx,
            idx1.intersection(idx2)
        )
        self.assertEqual(
            vkey,
            'x[({1&2}&{3,4})]:y'
        )
         
         
    def test_difference(self):
        q2 = self.example_data_A_data['q2']       
        test_logic = (has_all([1, 2]), Index.difference, has_any([3, 4]))
        idx1, vkey1 = get_logic_index(q2, test_logic[0])
        idx2, vkey2 = get_logic_index(q2, test_logic[2])
        idx, vkey = get_logic_index(q2, test_logic)
        self.assertItemsEqual(
            idx,
            idx1.difference(idx2)
        )
        self.assertEqual(
            vkey,
            'x[({1&2}~{3,4})]:y'
        )
         
         
    def test_sym_diff(self):
        q2 = self.example_data_A_data['q2']       
        test_logic = (has_all([1, 2]), Index.sym_diff, has_any([3, 4]))
        idx1, vkey1 = get_logic_index(q2, test_logic[0])
        idx2, vkey2 = get_logic_index(q2, test_logic[2])
        idx, vkey = get_logic_index(q2, test_logic)
        self.assertItemsEqual(
            idx,
            idx1.sym_diff(idx2)
        )
        self.assertEqual(
            vkey,
            'x[({1&2}^{3,4})]:y'
        )


    def test_wildcards(self):
        q2 = self.example_data_A_data['q2']  
        q3 = self.example_data_A_data['q3']  
        test_logic = {'q3': has_all([1, 2, 3])}
        idx, vkey = get_logic_index(q2, test_logic, self.example_data_A_data)
        idx_q3, vkey_q3 = get_logic_index(q3, has_all([1, 2, 3]))
        idx_q3 = q2.dropna().index.intersection(idx_q3)
        self.assertItemsEqual(
            idx,
            idx_q3
        )
        self.assertEqual(
            vkey,
            'x[q3={1&2&3}]:y'
        )

        q2 = self.example_data_A_data['q2']  
        test_logic = (has_any([1, 2]), Index.intersection, {'q3': has_all([1, 2, 3])})
        idx, vkey = get_logic_index(q2, test_logic, self.example_data_A_data)        
        idx_q2, vkey_q2 = get_logic_index(q2, has_any([1, 2]))        
        q3 = self.example_data_A_data['q3']  
        idx_q3, vkey_q3 = get_logic_index(q3, has_all([1, 2, 3]))
        idx_q3 = idx_q2.intersection(idx_q3)        
        self.assertItemsEqual(
            idx,
            idx_q3
        )
        self.assertEqual(
            vkey,
            'x[({1,2}&q3={1&2&3})]:y'
        )
         
        q2 = self.example_data_A_data['q2']  
        test_logic = ({'q3': has_all([1, 2, 3])}, Index.difference, has_any([1, 2]))
        idx, vkey = get_logic_index(q2, test_logic, self.example_data_A_data)        
        idx_q2, vkey_q2 = get_logic_index(q2, has_any([1, 2]))        
        q3 = self.example_data_A_data['q3']  
        idx_q3, vkey_q3 = get_logic_index(q3, has_all([1, 2, 3]))
        idx_q3 = q2.dropna().index.intersection(idx_q3.difference(idx_q2))        
        self.assertItemsEqual(
            idx,
            idx_q3
        )
        self.assertEqual(
            vkey,
            'x[(q3={1&2&3}~{1,2})]:y'
        )
         
         
    def test_nested_logic(self):
        q2 = self.example_data_A_data['q2']       
        test_logic = (
            (
                has_all([1, 2]), 
                Index.union, 
                has_any([3, 4])
            ), 
            Index.intersection,
            has_any([5, 6])
        )
        idx, vkey = get_logic_index(q2, test_logic)        
        idx1, vkey1 = get_logic_index(q2, test_logic[0])
        idx2, vkey2 = get_logic_index(q2, test_logic[2])
        self.assertItemsEqual(
            idx,
            idx1.intersection(idx2)
        )
        self.assertEqual(
            vkey,
            'x[(({1&2},{3,4})&{5,6})]:y'
        )
           
        q2 = self.example_data_A_data['q2']  
        test_logic = (
            (
                has_any([1, 2]), 
                Index.intersection, 
                {'q3': has_all([1, 2, 3])}
            ),
            Index.intersection,
            has_any([5, 6])
        )
        idx, vkey = get_logic_index(q2, test_logic, self.example_data_A_data)        
        idx_q2_a, vkey_q2_a = get_logic_index(q2, has_any([1, 2]))        
        q3 = self.example_data_A_data['q3']  
        idx_q3, vkey_q3 = get_logic_index(q3, has_all([1, 2, 3]))
        idx_q3 = idx_q2_a.intersection(idx_q3)        
        idx_q2_b, vkey_q2_a = get_logic_index(q2, has_any([5, 6]))
        idx_q2_b = idx_q3.intersection(idx_q2_b)        
        self.assertItemsEqual(
            idx,
            idx_q2_b
        )
        self.assertEqual(
            vkey,
            'x[(({1,2}&q3={1&2&3})&{5,6})]:y'
        )
    
        
    def test_logic_list(self):
        q2 = self.example_data_A_data['q2']       
        test_logic = union([
            has_all([1, 2]),
            has_any([3, 4]),
            has_count([3])
        ])
        idx, vkey = get_logic_index(q2, test_logic)  
        idx1, vkey1 = get_logic_index(q2, test_logic[1][0])
        idx2, vkey2 = get_logic_index(q2, test_logic[1][1])
        idx3, vkey3 = get_logic_index(q2, test_logic[1][2])
        self.assertItemsEqual(
            idx,
            idx1.union(idx2).union(idx3)
        )
        self.assertEqual(
            vkey,
            'x[({1&2},{3,4},{3})]:y'
        )

        q2 = self.example_data_A_data['q2']       
        test_logic = intersection([
            has_all([1, 2]),
            has_any([3, 4]),
            has_count([3])
        ])
        idx, vkey = get_logic_index(q2, test_logic)        
        idx1, vkey1 = get_logic_index(q2, test_logic[1][0])
        idx2, vkey2 = get_logic_index(q2, test_logic[1][1])
        idx3, vkey3 = get_logic_index(q2, test_logic[1][2])
        self.assertItemsEqual(
            idx,
            idx1.intersection(idx2).intersection(idx3)
        )
        self.assertEqual(
            vkey,
            'x[({1&2}&{3,4}&{3})]:y'
        )

        q2 = self.example_data_A_data['q2']       
        test_logic = difference([
            has_all([1, 2]),
            has_any([3, 4]),
            has_count([3])
        ])
        idx, vkey = get_logic_index(q2, test_logic)        
        idx1, vkey1 = get_logic_index(q2, test_logic[1][0])
        idx2, vkey2 = get_logic_index(q2, test_logic[1][1])
        idx3, vkey3 = get_logic_index(q2, test_logic[1][2])
        self.assertItemsEqual(
            idx,
            idx1.difference(idx2).difference(idx3)
        )
        self.assertEqual(
            vkey,
            'x[({1&2}~{3,4}~{3})]:y'
        )
        
        q2 = self.example_data_A_data['q2']       
        test_logic = sym_diff([
            has_all([1, 2]),
            has_any([3, 4]),
            has_count([3])
        ])
        idx, vkey = get_logic_index(q2, test_logic)        
        idx1, vkey1 = get_logic_index(q2, test_logic[1][0])
        idx2, vkey2 = get_logic_index(q2, test_logic[1][1])
        idx3, vkey3 = get_logic_index(q2, test_logic[1][2])
        self.assertItemsEqual(
            idx,
            idx1.sym_diff(idx2).sym_diff(idx3)
        )
        self.assertEqual(
            vkey,
            'x[({1&2}^{3,4}^{3})]:y'
        )
          
        
    def test_nested_logic_list(self):
        q2 = self.example_data_A_data['q2']       
        test_logic = intersection([
            union([
                has_all([1, 2]),
                has_any([3, 4])
            ]),
            has_count([3])
        ])
        idx, vkey = get_logic_index(q2, test_logic)
        idx1, vkey1 = get_logic_index(q2, has_all([1, 2]))
        idx2, vkey2 = get_logic_index(q2, has_any([3, 4]))
        idx3, vkey3 = get_logic_index(q2, has_count([3]))
        self.assertItemsEqual(
            idx,
            idx1.union(idx2).intersection(idx3)
        )
        self.assertEqual(
            vkey,
            'x[(({1&2},{3,4})&{3})]:y'
        )


    def test_get_logic_key_chunk(self):

        func = _has_any
        values = [1, 2, 3]
        chunk = get_logic_key_chunk(func, values)
        self.assertEqual(
            chunk, '{1,2,3}'
        )
        chunk = get_logic_key_chunk(func, values, True)
        self.assertEqual(
            chunk, 'e{1,2,3}'            
        )
        
        func = _not_any
        values = [1, 2, 3]
        chunk = get_logic_key_chunk(func, values)
        self.assertEqual(
            chunk, '~{1,2,3}'
        )
        chunk = get_logic_key_chunk(func, values, True)
        self.assertEqual(
            chunk, '~e{1,2,3}'            
        )
        
        func = _has_all
        values = [1, 2, 3]
        chunk = get_logic_key_chunk(func, values)
        self.assertEqual(
            chunk, '{1&2&3}'            
        )
        chunk = get_logic_key_chunk(func, values, True)
        self.assertEqual(
            chunk, 'e{1&2&3}'
        )
        
        func = _not_all
        values = [1, 2, 3]
        chunk = get_logic_key_chunk(func, values)
        self.assertEqual(
            chunk, '~{1&2&3}'            
        )
        chunk = get_logic_key_chunk(func, values, True)
        self.assertEqual(
            chunk, '~e{1&2&3}'
        )
        
        func = _has_count
        values = [1]
        chunk = get_logic_key_chunk(func, values)
        self.assertEqual(
            chunk, '{1}'            
        )
        
        func = _has_count
        values = [1, 3]
        chunk = get_logic_key_chunk(func, values)
        self.assertEqual(
            chunk, '{1-3}'            
        )
        
        func = _has_count
        values = [1, 3, [5, 6, 7, 8, 9]]
        chunk = get_logic_key_chunk(func, values)
        self.assertEqual(
            chunk, '(5,6,7,8,9){1-3}'            
        )
        chunk = get_logic_key_chunk(func, values, True)
        self.assertEqual(
            chunk, 'e(5,6,7,8,9){1-3}'            
        )
        
        func = _not_count
        values = [1]
        chunk = get_logic_key_chunk(func, values)
        self.assertEqual(
            chunk, '~{1}'            
        )
        
        func = _not_count
        values = [1, 3]
        chunk = get_logic_key_chunk(func, values)
        self.assertEqual(
            chunk, '~{1-3}'            
        )
        
        func = _not_count
        values = [1, 3, [5, 6, 7, 8, 9]]
        chunk = get_logic_key_chunk(func, values)
        self.assertEqual(
            chunk, '(5,6,7,8,9)~{1-3}'            
        )
        chunk = get_logic_key_chunk(func, values, True)
        self.assertEqual(
            chunk, 'e(5,6,7,8,9)~{1-3}'            
        )
        
        __op_symbol__ = {
            _is_lt: '<', _is_le: '<=', 
            _is_eq: '', _is_ne: '!=', 
            _is_ge: '>=', _is_gt: '>'
        }
        for op_func in [_is_lt, _is_le, _is_eq, _is_ne, _is_ge, _is_gt]:
            key_part = __op_symbol__[op_func]
            
            values = [(op_func, 3)]
            chunk = get_logic_key_chunk(_has_count, values)
            self.assertEqual(
                chunk, '{%s3}' % (key_part)
            )
            
            values = [(op_func, 3), [5, 6, 7, 8, 9]]
            chunk = get_logic_key_chunk(_has_count, values)
            self.assertEqual(
                chunk, '(5,6,7,8,9){%s3}' % (key_part)
            )
            chunk = get_logic_key_chunk(_has_count, values, True)
            self.assertEqual(
                chunk, 'e(5,6,7,8,9){%s3}' % (key_part)
            )

            chunk = get_logic_key_chunk(op_func, 5)
            self.assertEqual(
                chunk, 
                '(%s5)' % (__op_symbol__[op_func])
            )


    def test_get_logic_key(self):
        
        logic = has_all([1, 2, 3], True)
        self.assertEqual(
            get_logic_key(logic),
            'x[e{1&2&3}]:y'
        )
        
        logic = has_count([is_ge(1), [5, 6, 7, 8, 9]])
        self.assertEqual(
            get_logic_key(logic),
            'x[(5,6,7,8,9){>=1}]:y'
        )
        
        logic = not_count([is_ge(1), [5, 6, 7, 8, 9]])
        self.assertEqual(
            get_logic_key(logic),
            'x[(5,6,7,8,9)~{>=1}]:y'
        )
        
        logic = union([
            has_any([1, 2]), 
            has_all([3, 4]), 
            not_any([5, 6])
        ])
        self.assertEqual(
            get_logic_key(logic),
            'x[({1,2},{3&4},~{5,6})]:y'
        )
        
        logic = union([
            intersection([
                has_any([1, 2]),
                not_any([3])
            ]),
            {'Wave': has_any([1,2])}
        ])
        self.assertEqual(
            get_logic_key(logic, self.example_data_A_data),
            'x[(({1,2}&~{3}),Wave={1,2})]:y'
        )
        
        
##################### Helper functions #####################

    def confirm_inverse_index(self, series, idx_a, idx_b, incl_na=False):
                
        self.assertEqual(
            len(idx_a.intersection(idx_b)),
            0
        )
        if incl_na:
            self.assertItemsEqual(
                series.index,
                idx_a.union(idx_b)
            )
        else:
            self.assertItemsEqual(
                series.dropna().index,
                idx_a.union(idx_b)
            )


    def get_count_nums(self, series, test_responses):
    
        dummies = series.astype('object').str.get_dummies(';')
        
        _min = test_responses[0]
        
        if len(test_responses)<2:
            _max = None
        else:
            _max = test_responses[1]
        
        if len(test_responses)<3:
            test_values = None
            str_test_values = None
            cols = []
        else:
            test_values = test_responses[2]
            # Make string versions of test_values                
            str_test_values = [str(v) for v in test_values]
            cols = [
                col 
                for col in dummies.columns 
                if col in str_test_values
            ]
               
        # Slice dummies column-wise for targeted values subset
        if cols:
            dummies = dummies[cols]
            
        return dummies, _min, _max
            
         