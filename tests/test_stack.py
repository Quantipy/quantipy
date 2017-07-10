import unittest
import os.path
import numpy
import pandas as pd
from pandas.util.testing import assert_frame_equal
import test_helper
import copy

from collections import defaultdict, OrderedDict
from quantipy.tests.test_batch import _get_batch
from quantipy.core.stack import Stack
from quantipy.core.chain import Chain
from quantipy.core.link import Link
from quantipy.core.view_generators.view_mapper import ViewMapper
from quantipy.core.view_generators.view_maps import QuantipyViews
from quantipy.core.view_generators.view_specs import (net, calc)
from quantipy.core.view import View
from quantipy.core.helpers import functions
from quantipy.core.helpers.functions import load_json
from quantipy.core.cache import Cache

CBASE = "x|f|x:|||cbase"
COUNTS = "x|f|:|||counts"
DEFAULT = "x|default|:|||default"

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
        self.int = ['record_number', 'unique_id', 'age', 'birth_day', 'birth_month']
        self.float = ['weight', 'weight_a', 'weight_b']
        self.single = ['gender', 'locality', 'ethnicity', 'religion', 'q1']
        self.delimited_set = ['q2', 'q3', 'q8', 'q9']
        self.string = ['q8a', 'q9a']
        self.date = ['start_time', 'end_time']
        self.time = ['duration']
        self.array = ['q5', 'q6', 'q7']

        # The minimum list of variables required to populate a stack with all single*delimited set variations
        self.minimum = ['q2b', 'Wave', 'q2', 'q3', 'q5_1']

        # Set up example stack
        self.setup_stack_Example_Data_A()

    def test_stack_is_a_subclassed_dict(self):
        self.assertIsInstance(self.stack, dict)
        self.assertIsInstance(self.stack, Stack)
 
    def test_stack_behaves_like_a_dict(self):
        key = "some_key_name"
        value = "some_value"
        self.stack[key] = value
        self.assertIn(
            key,
            self.stack.keys(),
            msg="Stack should have key {data_key}, but has {stack_key}".format(
                data_key=key,
                stack_key=self.stack.keys()
            )
        )
 
    def test_cache_is_created(self):
        name = 'cache'
        fk = ['no_filter']
        xk = self.minimum
        yk = ['@'] + self.minimum
        views = ['default']
        weight = None
 
        # Init a stack
        stack = Stack(name=name)
        stack.add_data(
            data_key=stack.name,
            meta=self.example_data_A_meta,
            data=self.example_data_A_data
        )
 
        # Assert that it has a Cache that is empty
        self.assertIn('cache', stack[name].__dict__.keys())
        self.assertIsInstance(stack[name].cache, Cache)
        self.assertEqual(Cache(), stack[name].cache)
 
        # Run the Aggregations
        stack.add_link(
            data_keys=name,
            filters=fk,
            x=xk,
            y=yk,
            views=QuantipyViews(views),
            weights=weight
        )
 
        # Assert that it has a Cache that is NOT empty
        self.assertIn('cache', stack[name].__dict__.keys())
        self.assertIsInstance(stack[name].cache, Cache)
        self.assertNotEqual([], stack[name].cache.keys())
 
        # Manually remove the cache
        del stack[name].cache
        stack[name].cache = Cache()
        self.assertEqual(Cache(), stack[name].cache)
 
        # ReRun the Aggregations
        stack.add_link(
            data_keys=name,
            filters=fk,
            x=xk,
            y=yk,
            views=QuantipyViews(views),
            weights=weight
        )
 
        # Assert that it has a Cache has been recreated
        self.assertIn('cache', stack[name].__dict__.keys())
        self.assertIsInstance(stack[name].cache, Cache)
        self.assertNotEqual([], stack[name].cache.keys())
 
    def test_add_data(self):
 
        key_error_message = "Stack should have key {data_key}, but has {stack_key}"
 
        # Test data_key errors
        for data_key in [1, 1.1, [1], {}]:
            self.temp_stack = Stack()
            with self.assertRaises(TypeError) as error:
                self.temp_stack.add_data(data_key=data_key)
            self.assertEqual(
                error.exception.message[:50],
                "All data keys must be one of the following types: "
            )
 
        # Test data errors
        for data in [1, pd.Series([1, 2, 3, 4, 5]), 'data']:
            self.temp_stack = Stack()
            with self.assertRaises(TypeError) as error:
                self.temp_stack.add_data(data_key=self.stack.name, data=data)
            self.assertEqual(
                error.exception.message[:73],
                "The 'data' given to Stack.add_data() must be one of the following types: "
            )
 
        # Test meta errors
        for meta in [1, [1], 'meta']:
            self.temp_stack = Stack()
            with self.assertRaises(TypeError) as error:
                self.temp_stack.add_data(data_key=self.stack.name, meta=meta)
            self.assertEqual(
                error.exception.message[:73],
                "The 'meta' given to Stack.add_data() must be one of the following types: "
            )
 
        # Test proxy data key
        self.temp_stack = Stack()
        self.temp_stack.add_data(data_key=self.stack.name)
        self.assertTrue(self.stack.name in self.temp_stack, msg=key_error_message.format(
            data_key=data_key,
            stack_key=self.temp_stack.keys()
        ))
 
        # Test meta-only data key
        self.temp_stack = Stack()
        self.temp_stack.add_data(data_key=self.stack.name, meta=self.example_data_A_meta)
        self.assertTrue(self.stack.name in self.temp_stack, msg=key_error_message.format(
            data_key=data_key,
            stack_key=self.temp_stack.keys()
        ))
 
        # Test data-only data key
        self.temp_stack = Stack()
        self.temp_stack.add_data(data_key=self.stack.name, data=self.example_data_A_data)
        self.assertTrue(self.stack.name in self.temp_stack, msg=key_error_message.format(
            data_key=data_key,
            stack_key=self.temp_stack.keys()
        ))
 
        # Test meta+data data key
        self.temp_stack = Stack()
        self.temp_stack.add_data(
            data_key=self.stack.name,
            meta=self.example_data_A_meta,
            data=self.example_data_A_data
            )
        self.assertTrue(self.stack.name in self.temp_stack, msg=key_error_message.format(
            data_key=self.stack.name,
            stack_key=self.temp_stack.keys()
        ))
 
        # The data and meta keys exist
        self.assertTrue('meta' in self.temp_stack[self.stack.name].__dict__)
        self.assertTrue('data' in self.temp_stack[self.stack.name].__dict__)
 
        # The data and meta attributes exist (after using stack.add_data())
        self.assertTrue(hasattr(self.temp_stack[self.stack.name], 'meta'))
        self.assertTrue(hasattr(self.temp_stack[self.stack.name], 'data'))
 
        # The data and meta attributes should be the correct instance type
        self.assertIsInstance(self.temp_stack[self.stack.name].meta, (dict, OrderedDict))
        self.assertIsInstance(self.temp_stack[self.stack.name].data, pd.DataFrame)
 
    def test_add_data_as_arg(self):
 
        key_error_message = "Stack should have key {data_key}, but has {stack_key}"
 
        # Test data_key errors
        for data_key in [1, 1.1]:
            with self.assertRaises(TypeError) as error:
                self.temp_stack = Stack(
                    add_data={data_key: {'data': data_key}}
                )
            self.assertEqual(
                error.exception.message[:50],
                "All data keys must be one of the following types: "
            )
 
        # Test data errors
        for data in [1, pd.Series([1, 2, 3, 4, 5]), 'data']:
            with self.assertRaises(TypeError) as error:
                self.temp_stack = Stack(
                    add_data={self.stack.name: {'data': data}}
                )
            self.assertEqual(
                error.exception.message[:73],
                "The 'data' given to Stack.add_data() must be one of the following types: "
            )
 
        # Test meta errors
        for meta in [1, [1], 'meta']:
            with self.assertRaises(TypeError) as error:
                self.temp_stack = Stack(
                    add_data={self.stack.name: {'meta': meta}}
                )
            self.assertEqual(
                error.exception.message[:73],
                "The 'meta' given to Stack.add_data() must be one of the following types: "
            )
 
        # Test proxy data key
        self.temp_stack = Stack(
            add_data={self.stack.name: {}}
        )
        self.assertTrue(self.stack.name in self.temp_stack, msg=key_error_message.format(
            data_key=data_key,
            stack_key=self.temp_stack.keys()
        ))
 
        # Test meta-only data key
        self.temp_stack = Stack(
            add_data={self.stack.name: {'meta': self.example_data_A_meta}}
        )
        self.assertTrue(self.stack.name in self.temp_stack, msg=key_error_message.format(
            data_key=data_key,
            stack_key=self.temp_stack.keys()
        ))
 
        # Test data-only data key
        self.temp_stack = Stack(
            add_data={self.stack.name: {'data': self.example_data_A_data}}
        )
        self.assertTrue(self.stack.name in self.temp_stack, msg=key_error_message.format(
            data_key=data_key,
            stack_key=self.temp_stack.keys()
        ))
 
        # Test meta+data data key
        self.temp_stack = Stack(
            add_data={
                self.stack.name: {
                    'data': self.example_data_A_data,
                    'meta': self.example_data_A_meta
                }
            }
        )
        self.assertTrue(self.stack.name in self.temp_stack, msg=key_error_message.format(
            data_key=self.stack.name,
            stack_key=self.temp_stack.keys()
        ))
 
        # The data and meta keys exist
        self.assertTrue('meta' in self.temp_stack[self.stack.name].__dict__)
        self.assertTrue('data' in self.temp_stack[self.stack.name].__dict__)
 
        # The data and meta attributes exist (after using stack.add_data())
        self.assertTrue(hasattr(self.temp_stack[self.stack.name], 'meta'))
        self.assertTrue(hasattr(self.temp_stack[self.stack.name], 'data'))
 
        # The data and meta attributes should be the correct instance type
        self.assertIsInstance(self.temp_stack[self.stack.name].meta, (dict, OrderedDict))
        self.assertIsInstance(self.temp_stack[self.stack.name].data, pd.DataFrame)
 
        # Create an array of data sources
        # using dicts and tuples as data_key values) & create stack
        df_1 = self.example_data_A_data.copy().query('Wave == 1')
        df_2 = self.example_data_A_data.copy().query('Wave == 2')
        df_3 = self.example_data_A_data.copy().query('Wave == 3')
        df_4 = self.example_data_A_data.copy().query('Wave == 4')
        data_sources_A = {
            'Wave 1': {'data': df_1,
                       'meta': self.example_data_A_meta},
            'Wave 2': {'data': df_2,
                       'meta': self.example_data_A_meta},
            'Wave 3': {'data': df_3,
                       'meta': self.example_data_A_meta},
            'Wave 4': {'data': df_4,
                       'meta': self.example_data_A_meta},
        }
        data_sources_B = {
            'Wave 1': (df_1,
                       self.example_data_A_meta),
            'Wave 2': (df_2,
                       self.example_data_A_meta),
            'Wave 3': (df_3,
                       self.example_data_A_meta),
            'Wave 4': (df_4,
                       self.example_data_A_meta),
        }
 
        for data_sources in [data_sources_A, data_sources_B]:
 
            self.temp_stack = Stack(add_data=data_sources)
 
            # Test meta+data data key for array of data keys
            for data_source in data_sources.keys():
                self.assertTrue(
                    data_source in self.temp_stack, msg=key_error_message.format(
                        data_key=data_source,
                        stack_key=self.temp_stack.keys()
                    )
                )
 
                self.assertTrue('meta' in self.temp_stack[data_source].__dict__)
                self.assertTrue('data' in self.temp_stack[data_source].__dict__)
 
                # The data and meta attributes exist (after using stack.add_data())
                self.assertTrue(hasattr(self.temp_stack[data_source], 'meta'))
                self.assertTrue(hasattr(self.temp_stack[data_source], 'data'))
 
                # The data and meta attributes should be the correct instance type
                self.assertIsInstance(
                    self.temp_stack[data_source].meta, (dict, OrderedDict)
                )
                self.assertIsInstance(
                    self.temp_stack[data_source].data, pd.DataFrame
                )
 
        # Test data_key values for array of data keys
        data_sources = {
            'Wave 1': (df_1,
                       self.example_data_A_meta),
            'Wave 2': [df_2,
                       self.example_data_A_meta]
        }
        with self.assertRaises(TypeError) as error:
            self.temp_stack = Stack(
                add_data=data_sources
            )
            self.assertEqual(
                error.exception.message[:55],
                "All data_key values must be one of the following types: "
            )
 
    def test_add_link_generates_links_and_views(self):
        self.setup_stack_Example_Data_A()
 
        contents = self.stack.describe()
        dks = contents['data'].unique()
        fks = contents['filter'].unique()
        xks = contents['x'].unique()
        yks = contents['y'].unique()
        vks = contents['view'].unique()
 
        # Test that a Link objects sits behind all y-keys
        self.verify_links_and_views_exist_in_nest(self.stack)
 
    def test_reduce(self):
        filters = ["no_filter", "Wave == 1", "age > 30"]
        self.setup_stack_Example_Data_A(fk=filters, views=['default', 'counts'])
 
        old_contents = self.stack.describe()
        old_dk = old_contents['data'].unique()
        old_fk = old_contents['filter'].unique()
        old_xk = old_contents['x'].unique()
        old_yk = old_contents['y'].unique()
        old_vk = old_contents['view'].unique()
 
        # Remove a filter from the stack, check that other views are unaffected
        self.stack.reduce(filters=[old_fk[-1]])
        new_fk = self.stack.describe()['filter'].unique()
        self.assertNotIn(old_fk[-1], new_fk)
        for fk in old_fk[:-1]:
            self.assertIn(fk, new_fk)
 
        # Remove a x-key from the stack, check that other views are unaffected
        self.stack.reduce(x=[old_xk[-1]])
        new_xk = self.stack.describe()['x'].unique()
        self.assertNotIn(old_xk[-1], new_xk)
        for xk in old_xk[:-1]:
            self.assertIn(xk, new_xk)
 
        # Remove a y-key from the stack, check that other views are unaffected
        self.stack.reduce(y=[old_yk[0]])
        new_yk = self.stack.describe()['y'].unique()
        self.assertNotIn(old_yk[0], new_yk)
        for yk in old_yk[1:]:
            self.assertIn(yk, new_yk)
 
        # Remove a view from the stack, check that other views are unaffected
        self.stack.reduce(views=[old_vk[-1]])
        new_vk = self.stack.describe()['view'].unique()
        self.assertNotIn(old_vk[-1], new_vk)
        for vk in old_vk[:-1]:
            self.assertIn(vk, new_vk)
 
        # Reset self.stack after the deletions
        self.setup_stack_Example_Data_A(fk=filters, views=['default', 'counts'])
        old_contents = self.stack.describe()
        old_xk = old_contents['x'].unique()
        old_yk = old_contents['y'].unique()
 
        # Remove a variable (x and y) from the stack, check that other x and ys are unaffected
        self.stack.reduce(variables=['q3'])
        new_contents = self.stack.describe()
        new_xk = new_contents['x'].unique()
        new_yk = new_contents['y'].unique()
        self.assertNotIn('q3', new_xk)
        self.assertNotIn('q3', new_yk)
        for xk in old_xk[old_xk!='q3']:
            self.assertIn(xk, new_xk)
        for yk in old_yk[old_yk!='q3']:
            self.assertIn(yk, new_yk)
 
        # Test error handling for non-existant keys
        non_key = 'this key does not exist'
        with self.assertRaises(ValueError):
            self.stack.reduce(filters=[non_key])
            self.stack.reduce(x=[non_key])
            self.stack.reduce(y=[non_key])
            self.stack.reduce(views=[non_key])
            self.stack.reduce(variables=[non_key])
 
    def test_getting_1D_views(self):
 
        dk = self.stack.name
        fk = 'no_filter'
 
        # Test that x='@' links produce Views
        self.setup_stack_Example_Data_A(xk=['@', 'Wave'], yk=['Wave'])
        self.assertIsInstance(self.stack[dk][fk]['@']['Wave']['x|default|:|||default'], View)
 
        # Test that y='@' links produce Views
        self.setup_stack_Example_Data_A(xk=['Wave'], yk=['@', 'Wave'])
        self.assertIsInstance(self.stack[dk][fk]['Wave']['@']['x|default|:|||default'], View)
 
    def test_filters(self):
        filters = [
            "no_filter",
            "Wave == 1",
            "Wave == 2 and age > 30",
            "age > 30",
        ]
        # Create a test stack
        x_keys = ['Wave', 'age']
        y_keys = ['@'] + x_keys
        self.setup_stack_Example_Data_A(fk=filters, xk=x_keys, yk=y_keys)
        # Test all the populated filter keys exist in the stack
        for filter in filters:
            for x in x_keys:
                for y in y_keys:
                    self.assertIsInstance(
                        self.stack[self.stack.name][filter][x][y]['x|default|:|||default'].dataframe,
                        pd.DataFrame
                    )
 
        # Test the filters have calculated correctly
        # no_filter
        df = self.stack[self.stack.name]["no_filter"]['Wave']['@']['x|default|:|||default'].dataframe
        self.assertEqual(df.iloc[(0,0)], self.example_data_A_data.shape[0])
        # Other filters
        for filter_def in filters[1:]:
            df = self.stack[self.stack.name][filter_def]['Wave']['@']['x|default|:|||default'].dataframe
            self.assertEqual(df.iloc[(0,0)], self.example_data_A_data.query(filter_def).shape[0])
 
    def test_add_link_exceptions(self):
        self.setup_stack_Example_Data_A(xk=self.single, yk=self.single)
 
        variables = self.delimited_set
        x = self.single
        y = self.single
 
        # Test that providing 'variables' along with 'x' and/or 'y' raises a ValueError
        self.assertRaises(ValueError, self.stack.add_link, variables=variables, x=x)
        self.assertRaises(ValueError, self.stack.add_link, variables=variables, y=y)
        self.assertRaises(ValueError, self.stack.add_link, variables=variables, x=x, y=y)
 
    def test_add_link_x_y_equal(self):
        self.setup_stack_Example_Data_A()
        dk = self.stack.name
        fk = 'no_filter'
        vk = 'default'
 
        # Test that x==y links have populated correctly
        for xy in self.minimum:
            # Test x==y requests produce Link objects
            link = self.stack[dk][fk][xy][xy]
            self.assertIsInstance(link, Link)
            # Test x==y requests produce View objects
            view = link['x|default|:|||default']
            self.assertIsInstance(view, View)
            # Test x==y requests produce dataframes where index and columns are the same
            # (with the execption of the "All"-margin)
            df = view.dataframe
            self.assertTrue(df.index.get_level_values(1).tolist() == df.columns.get_level_values(1).tolist())
 
    def test_add_link_lazy(self):
        dk = self.stack.name
        fk = 'no_filter'
        xk = self.single
        yk = ['@'] + self.delimited_set
        vk = ['default']
 
        # Test adding new views fills all links
        self.setup_stack_Example_Data_A(
            xk=xk,
            yk=yk,
            views=['default'])
 
        old_contents = self.stack.describe()
        old_xk = old_contents['x'].unique()
        old_yk = old_contents['y'].unique()
        old_vk = old_contents['view'].unique()
 
        self.assertEqual(old_vk, ['x|default|:|||default'])
        self.stack.add_link(
            x=old_xk,
            y=old_yk,
            views=['counts'])
 
        new_vk = self.stack.describe()['view'].unique()
        self.assertEqual(new_vk.tolist(), ['x|f|:|||counts', 'x|default|:|||default'])
 
        # Test lazy y-keys when 1 x key is given
        self.stack.add_link(x=xk[0], views=['cbase'])
        lazy_y = self.stack.describe(
            index=['y'],
            query="x=='%s' and view=='x|f|x:|||cbase'" % xk[0]
        ).index.tolist()
        self.assertItemsEqual(yk, lazy_y)
 
        # Test lazy x-keys when 1 y key is given
        self.stack.add_link(y=yk[0], views=['cbase'], weights=['weight_a'])
        lazy_x = self.stack.describe(
            index=['x'],
            query="y=='%s' and view=='x|f|x:||weight_a|cbase'" % yk[0]
        ).index.tolist()
        self.assertItemsEqual(xk, lazy_x)
 
        # TO DO (these features are not yet supported)
        # - lazy providing only data keys
        # - lazy providing only filters
        # - lazy providing only x
        # - lazy providing only y
 
    def test_describe(self):
        dk = [self.stack.name]
        fk = ['no_filter', 'Wave == 1']
        xk = self.single
        yk = self.delimited_set
        vk = ['default', 'cbase', 'rbase', 'counts']
        vk_notation = ['x|default|:|||default', 'x|f|:y|||rbase', 'x|f|:|||counts',
                       'x|f|x:|||cbase']
        self.setup_stack_Example_Data_A(
            fk=fk,
            xk=xk,
            yk=yk,
            views=vk)
 
        # Test describe returns a pandas DataFrame
        contents = self.stack.describe()
        self.assertIsInstance(contents, pd.DataFrame)
 
        # Test describe returns df with required columns
        expected_columns = ['data', 'filter', 'x', 'y', 'view', '#']
        actual_columns = contents.columns.tolist()
        self.assertEqual(actual_columns, expected_columns)
 
        # Test desribe returns df with expected number of rows
        expected_rows = len(dk) * len(fk) * len(self.single) * len(self.delimited_set) * len(vk)
        actual_rows = contents.shape[0]
        self.assertEqual(actual_rows, expected_rows)
 
        # Test the returned df contains everything expected and nothing unexpected
        self.verify_contains_expected_not_unexpected(contents, dk, fk, xk, yk, vk_notation)
 
        # Test index & column parameters
        column_names = ['data', 'filter', 'x', 'y', 'view']
 
        #check index column parameters
        for column in column_names:
            described_index = self.stack.describe(index=[column])
            described_column = self.stack.describe(columns=[column])
            self.assertIn(column, described_index.index.names)
            self.assertIn(column, described_column.index.names)
 
        for index, column in zip(column_names, column_names):
            if index == column:
                pass
            else:
                described = self.stack.describe(index=[index], columns=[column])
                self.assertIn(index, described.index.names)
                self.assertIn(column, described.columns.names)
 
        # Test query parameter
        xk = self.single[:2]
        yk = self.delimited_set[:2]
        query = "x in {x} and y in {y} and view=={v}".format(
            x=str(xk),
            y=str(yk),
            v="'x|default|:|||default'")
 
        contents = self.stack.describe(query=query)
 
        # Test the returned df contains everything expected and nothing unexpected
        self.verify_contains_expected_not_unexpected(contents, xk=xk, yk=yk)
 
        # Finally, bulk-test the entire queried result
        expected_contents = contents.query(query)
        assert_frame_equal(expected_contents, contents)
 
        # Test that query can be used in conjunction with index
        contents = self.stack.describe(index=['view'], query="x=='gender'")
        self.assertItemsEqual(
            contents.index.tolist(),
            [
                'x|default|:|||default',
                'x|f|x:|||cbase',
                'x|f|:|||counts',
                'x|f|:y|||rbase',
            ])
 
        # Test that query can be used in conjunction with columns
        contents = self.stack.describe(columns=['y'], query="x=='gender'")
        self.assertItemsEqual(contents.index.tolist(), ['q2', 'q3', 'q8', 'q9'])
 
        # Test that query can be used in conjunction with index AND columns
        contents = self.stack.describe(index=['view'], columns=['y'], query="x=='gender'")
        self.assertItemsEqual(contents.columns.tolist(), ['q2', 'q3', 'q8', 'q9'])
        self.assertItemsEqual(
            contents.index.tolist(),
            [
                'x|default|:|||default',
                'x|f|x:|||cbase',
                'x|f|:|||counts',
                'x|f|:y|||rbase',
            ])
 
    def test_get_chain_generates_chains(self):
        dk = self.stack.name
        fk = 'no_filter'
        xk = self.single
        yk = self.delimited_set
        vk = ['default']
        self.setup_stack_Example_Data_A(xk=xk, yk=yk)
 
        # Test auto-orient_on x
        for x in xk:
            chain = self.stack.get_chain(data_keys=dk, filters=fk, x=x, y=yk, views=['x|default|:|||default'])
            self.assertIsInstance(chain, Chain)
            self.verify_links_and_views_exist_in_nest(chain)
            # Test the chain contains everything expected and nothing unexpected
            contents = chain.describe()
            self.verify_contains_expected_not_unexpected(contents, dk, fk, x, yk, 'x|default|:|||default')
 
        # Test auto-orient_on y
        for y in yk:
            chain = self.stack.get_chain(data_keys=dk, filters=fk, x=xk, y=y, views=['x|default|:|||default'])
            self.assertIsInstance(chain, Chain)
            self.verify_links_and_views_exist_in_nest(chain)
            # Test the chain contains everything expected and nothing unexpected
            contents = chain.describe()
            self.verify_contains_expected_not_unexpected(contents, dk, fk, xk, y, 'x|default|:|||default')
 
        # Test orient_on x
        chains = self.stack.get_chain(data_keys=dk, filters=fk, x=xk, y=yk, views=['x|default|:|||default'], orient_on='x')
        for i, chain in enumerate(chains):
            self.assertIsInstance(chain, Chain)
            self.verify_links_and_views_exist_in_nest(chain)
            # Test the chain contains everything expected and nothing unexpected
            contents = chain.describe()
            self.verify_contains_expected_not_unexpected(contents, dk, fk, xk[i], yk, 'x|default|:|||default')
 
        # Test orient_on y
        chains = self.stack.get_chain(data_keys=dk, filters=fk, x=xk, y=yk, views=['x|default|:|||default'], orient_on='y')
        for i, chain in enumerate(chains):
            self.assertIsInstance(chain, Chain)
            self.verify_links_and_views_exist_in_nest(chain)
            # Test the chain contains everything expected and nothing unexpected
            contents = chain.describe()
            self.verify_contains_expected_not_unexpected(contents, dk, fk, xk, yk[i], 'x|default|:|||default')
 
    def test_get_chain_orient_on_gives_correct_orientation(self):
        self.setup_stack_Example_Data_A()
        dk = self.stack.name
        fk = 'no_filter'
        xk = self.minimum
        yk = ['@']+self.minimum
        vk = [COUNTS]
 
        # Test orient_on x
        chains = self.stack.get_chain(data_keys=dk, x=xk, y=yk, views=vk, orient_on='x')
        for i, chain in enumerate(chains):
            self.assertEqual(chain.orientation, 'x')
            self.assertEqual(chain.content_of_axis, yk)
            self.assertEqual(chain.source_name, xk[i])
 
        # Test orient_on y
        chains = self.stack.get_chain(data_keys=dk, x=xk, y=yk, views=vk, orient_on='y')
        for i, chain in enumerate(chains):
            self.assertEqual(chain.orientation, 'y')
            self.assertEqual(chain.content_of_axis, xk)
            self.assertEqual(chain.source_name, yk[i])
 
    def test_get_chain_preserves_link_orientation(self):
        self.setup_stack_Example_Data_A()
        dk = self.stack.name
        fk = 'no_filter'
        xk = 'Wave'
        yk = ['@', 'q2']
        vk = DEFAULT
 
        chain = self.stack.get_chain(data_keys=dk, x=xk, y=yk, views=[DEFAULT])
        # the index part of the dataframe should be 'Wave'
        self.assertEqual(chain[dk][fk][xk][yk[0]][DEFAULT].dataframe.index[0][0], 'Wave')
 
    def test_get_chain_lazy(self):
        self.setup_stack_Example_Data_A()
        dk = self.stack.name
        xk = ['Wave']
        vk = [COUNTS]
 
        # Test lazy y-keys
        chain = self.stack.get_chain(data_keys=dk, x=xk, views=vk)
        self.assertIsInstance(chain, Chain)
        self.assertEqual(chain.name, '.'.join([chain.orientation, chain.source_name] + chain.content_of_axis + vk))
 
        # Test lazy data keys - lazy data keys currently picks up only the last data-key
        # Intended behaviour for this has not been properly described so although the test
        # passes, it will remain inactive for now
        self.stack.add_data(
            data_key="DK2",
            meta=self.example_data_A_meta,
            data=self.example_data_A_data
        )
        self.stack.add_link(data_keys=["DK2"], x=self.minimum, y=['@']+self.minimum)
        chain = self.stack.get_chain(x=xk, views=vk)
        self.assertIsInstance(chain, Chain)
        self.assertEqual(chain.data_key, 'DK2')

    def test_refresh(self):
        all_filters = ['Wave==1', 'no_filter']
        all_x = ['q1', 'q2', 'q2b', 'q3', 'q4']
        all_y = ['@', 'gender', 'locality', 'ethnicity']
        weights = [None, 'weight_a']

        stack = Stack()
        stack.add_data(data_key='old_key', data=self.example_data_A_data,
                       meta=self.example_data_A_meta)

        stack.add_link(x=all_x, y=all_y, weights=weights, filters=all_filters,
                       views=['counts'])
        stack.add_link(x=['q2'], y=['gender'], weights=None, views=['c%'])
        stack.add_link(x=['q1', 'q3'], y=['@', 'locality'], weights='weight_a',
                       filters=['Wave==1'], views=['cbase'])

        before_refresh = stack.describe(columns='data', index='view')

        stack.refresh(data_key='old_key', new_data_key='new_key',
                      new_weight='weight_b')

        after_refresh = stack.describe(columns='data', index='view')
        self.assertTrue(before_refresh.values.sum() == 85.0)
        self.assertTrue(after_refresh['old_key'].sum() == 85.0)
        
        self.assertTrue(after_refresh['new_key'].sum() == 130.0)

        stack.reduce(data_keys='new_key')

        mod_data = self.example_data_A_data.copy().head(1000)
        stack.refresh(data_key='old_key', new_data_key='new_key',
                      new_data=mod_data)

        after_refresh = stack.describe(columns='data', index='view')
        self.assertTrue(before_refresh.values.sum() == 85.0)
        self.assertTrue(after_refresh['old_key'].sum() == 85.0)
        self.assertTrue(after_refresh['new_key'].sum() == 85.0)
        self.assertTrue(after_refresh.index.tolist() ==
                        before_refresh.index.tolist())

        stack.reduce(data_keys='new_key')
        stack.refresh(data_key='old_key', new_data_key='new_key',
                      new_data=mod_data, new_weight='weight_b')

        after_refresh = stack.describe(columns='data', index='view')
        self.assertTrue(before_refresh.values.sum() == 85.0)
        self.assertTrue(after_refresh['old_key'].sum() == 85.0)
        self.assertTrue(after_refresh['new_key'].sum() == 129.0)

    def test_refresh_remove_weight(self):
        all_filters = ['Wave==1', 'no_filter']
        all_x = ['q1', 'q2', 'q2b', 'q3', 'q4']
        all_y = ['@', 'gender', 'locality', 'ethnicity']
        weights = ['weight_a']

        stack = Stack()
        stack.add_data(data_key='old_key', data=self.example_data_A_data,
                       meta=self.example_data_A_meta)
        stack.add_link(x=all_x, y=all_y, weights=weights, filters=all_filters,
                       views=['counts'])
        stack.add_link(x=['q2'], y=['gender'], weights=None, views=['c%'])
        stack.add_link(x=['q1', 'q3'], y=['@', 'locality'], weights='weight_a',
                       filters=['Wave==1'], views=['cbase'])

        before_refresh = stack.describe(columns='data', index='view')

        stack.refresh(data_key='old_key', new_data_key='new_key',
                      new_weight='')

        after_refresh = stack.describe(columns='data', index='view')

        self.assertTrue(before_refresh.values.sum() == 45.0)
        self.assertTrue(after_refresh['old_key'].sum() == 45.0)
        self.assertTrue(after_refresh['new_key'].sum() == 89.0)

    def test_save_and_load_with_and_without_cache(self):
        """ This tests that the cache is stored and loaded with
            and without the cache
        """
        key = 'Example Data (A)'
        path_stack = '%s%s.stack' % (self.path, self.stack.name)
        path_cache = '%s%s.cache' % (self.path, self.stack.name)
        compressiontype = [None, "gzip"]

        if os.path.exists(path_stack):
            os.remove(path_stack)
        if os.path.exists(path_cache):
            os.remove(path_cache)

        self.assertFalse(os.path.exists(path_stack), msg="Saved stack exists but should NOT have been created yet")
        self.assertFalse(os.path.exists(path_cache), msg="Saved cache exists but should NOT have been created yet")
        self.setup_stack_Example_Data_A()

        caches = {}
        for key in self.stack.keys():
            self.assertIn('cache', self.stack[key].__dict__.keys())
            self.assertIsInstance(self.stack[key].cache, Cache)
            self.assertNotEqual(Cache(), self.stack[key].cache)
            caches[key] = self.stack[key].cache

        for compression in compressiontype:
            # Save the stack WITHOUT the cache
            self.stack.save(path_stack=path_stack, compression=compression, store_cache=False)
            self.assertTrue(os.path.exists(path_stack), msg="File {file} should exist".format(file=path_stack))
            self.assertFalse(os.path.exists(path_cache), msg="File {file} should NOT exist".format(file=path_cache))

            new_stack = Stack.load(path_stack, compression=compression)
            key_error_message = "Stack should have key {data_key}, but has {stack_key}"
            self.assertTrue(key in new_stack, msg=key_error_message.format(data_key=key, stack_key=self.stack.keys()))

            # Ensure that there is NO cache
            for key in new_stack.keys():
                self.assertDictEqual(Cache(), new_stack[key].cache)

            self.stack.save(path_stack=path_stack, compression=compression, store_cache=True)
            self.assertTrue(os.path.exists(path_stack), msg="File {file} should exist".format(file=path_stack))
            self.assertTrue(os.path.exists(path_cache), msg="File {file} should exist".format(file=path_cache))

            new_stack = Stack.load(path_stack, compression=compression, load_cache=True)
            key_error_message = "Stack should have key {data_key}, but has {stack_key}"
            self.assertTrue(key in new_stack, msg=key_error_message.format(data_key=key, stack_key=self.stack.keys()))

            # Ensure that there IS a cache
            for key in caches:
                self.assertIn(key, new_stack)
                self.assertTrue('matrices' in caches[key].keys())
                self.assertTrue('weight_vectors' in caches[key].keys())
                for sect_def in caches[key]['matrices']:
                    mat1, codes1 = caches[key]['matrices'][sect_def]
                    mat2, codes2 = new_stack[key].cache['matrices'][sect_def]
                    self.assertIsInstance(mat1, numpy.ndarray)
                    self.assertIsInstance(mat2, numpy.ndarray)
                    self.assertTrue(numpy.array_equal(mat1, mat2))
                    self.assertIsInstance(codes1, list)
                    self.assertIsInstance(codes2, list)
                    self.assertTrue(numpy.array_equal(codes1, codes2))
                self.assertNotEqual(id(caches[key]), id(new_stack[key].cache), msg="The matrix cache should be equal but not the same.")
            if os.path.exists(path_stack):
                os.remove(path_stack)
            if os.path.exists(path_cache):
                os.remove(path_cache)

    def test_save_and_load_stack_path_expectations(self):
        """ This tests makes sure the path expectations for
        stack.save() and stack.load() are working and are
        triggering the in-built user feedback when ignored.
        """
        name_stack = '%s.stack' % (self.stack.name)
        path_stack = '%s%s' % (self.path, name_stack)
 
        if os.path.exists(path_stack):
            os.remove(path_stack)
 
        self.assertFalse(os.path.exists(path_stack), msg="Saved stack exists but should not have been created yet")
        self.setup_stack_Example_Data_A()
 
        with self.assertRaises(ValueError):
            # Test fails when a folder path is given
            self.stack.save(path_stack='./tests/')
            # Test fails when the path doesn't end with '.stack'
            self.stack.save(path_stack=path_stack.replace('.stack', ''))
            # Test fails when the path is truncated at all
            self.stack.save(path_stack=path_stack[:-1])
 
        # This should not cause any errors
        self.stack.save(path_stack=path_stack)
 
        with self.assertRaises(ValueError):
            # Test fails when a folder path is given
            self.stack = Stack.load(path_stack='./tests/')
            # Test fails when the path doesn't end with '.stack'
            self.stack = Stack.load(path_stack=path_stack.replace('.stack', ''))
            # Test fails when the path is truncated at all
            self.stack = Stack.load(path_stack=path_stack[:-1])
 
        # This should not cause any errors
        self.stack = Stack.load(path_stack=path_stack)
 
        if os.path.exists(path_stack):
            os.remove(path_stack)
 
    def test_save_and_load_stack(self):
        """ This tests saves the stack and loads it,
            then it does the checks
        """
        key = 'Example Data (A)'
        path_stack = '%s%s.stack' % (self.path, self.stack.name)
        compressiontype = [None, "gzip"]
 
        if os.path.exists(path_stack):
            os.remove(path_stack)
 
        self.assertFalse(os.path.exists(path_stack), msg="Saved stack exists but should not have been created yet")
        self.setup_stack_Example_Data_A()
 
        for compression in compressiontype:
            self.stack.save(path_stack=path_stack, compression=compression)
            self.assertTrue(os.path.exists(path_stack), msg="File {file} should exist".format(file=path_stack))
            new_stack = Stack.load(path_stack, compression=compression)
 
            key_error_message = "Stack should have key {data_key}, but has {stack_key}"
            self.assertTrue(key in new_stack, msg=key_error_message.format(data_key=key, stack_key=self.stack.keys()))
 
            for key in new_stack.keys():
                # Verify that the x Variables are the same in the loaded file and the original stack
                self.assertItemsEqual(new_stack[key].keys(), self.stack[key].keys())
                for a_filter in new_stack[key].keys():
                    for x in new_stack[key][a_filter]:
                        # Verify that the y Variables are the same in the loaded file and the original stack
                        self.assertItemsEqual(new_stack[key][a_filter][x].keys(), self.stack[key][a_filter][x].keys())
                        for y in new_stack[key][a_filter][x]:
                            self.assertItemsEqual(new_stack[key][a_filter][x][y].keys(), self.stack[key][a_filter][x][y].keys())
                            for view in new_stack[key][a_filter][x][y]:
                            # Verify that the content of the dataframes is the same
                                self.assertTrue(new_stack[key][a_filter][x][y][view].dataframe.equals(self.stack[key][a_filter][x][y][view].dataframe),
                                                "FATAL ERROR: A data-frame in the loaded stack is not the same as the one in the created stack.")
            # Cleanup for next compressiontype
            if os.path.exists(path_stack):
                os.remove(path_stack)
 
#     def test_load_stack(self):
#         key = 'Jan'
#         data = 'tests/example.csv'
#         filepath = './tests/'+self.stack.name+'.stack'
#         compressiontype = [None, "gzip"]
#
#         self.stack.link_data(data_key=key, filename=data)
#         self.stack.add_link()
#         for compression in compressiontype:
#             self.stack.save(path='./tests/', compression=compression)
#             self.assertTrue(os.path.exists(filepath), msg="File {file} should exist".format(file=filepath))
#             new_stack = Stack.load(filepath, compression=compression)
#
    def test_save_load_stack_improved(self):
        # This tests save/load methods using dataframes and
        # verifies that the source data is still intact after load
        # and that the structure, meta and data attributes and views are intact.
 
        path_stack = '%s%s.stack' % (self.path, self.stack.name)
        self.setup_stack_Example_Data_A()
 
        # Ensure that the stack has the correct structure, attributes
        # and views.
        for data_key in self.stack.keys():
            # Does the loaded stack actually have the data and meta attributes
            self.assertTrue(hasattr(self.stack[data_key], 'data'))
            self.assertTrue(hasattr(self.stack[data_key], 'meta'))
 
            # Is the attribute a Pandas DataFrame ?
            self.assertIsInstance(self.stack[data_key].data, pd.DataFrame)
 
        # Save and load the stack and test that everything is still available,
        # the loaded stack has the same views, attributes as the generated one
        compressiontype = [None, "gzip"]
        for compression in compressiontype:
            if os.path.exists(path_stack):
                os.remove(path_stack)
 
            self.assertFalse(os.path.exists(path_stack), msg="File {file} should NOT exist".format(file=path_stack))
            self.stack.save(path_stack=path_stack, compression=compression)
            self.assertTrue(os.path.exists(path_stack), msg="File {file} should exist".format(file=path_stack))
            loaded_stack = Stack.load(path_stack, compression=compression)
 
            # Ensure that it is not the same stack (in memory)
            self.assertFalse(id(loaded_stack) == id(self.stack), msg="The stacks must not be in the same memory location")
 
            # Test all of the keys in the loaded stack
            for data_key in loaded_stack:
                # Does the loaded stack actually have the data and meta attributes
                self.assertTrue(hasattr(loaded_stack[data_key], 'data'))
                self.assertTrue(hasattr(loaded_stack[data_key], 'meta'))
 
                # Is the attribute a Pandas DataFrame ?
                self.assertIsInstance(loaded_stack[data_key].data, pd.DataFrame)
 
                # Verify that the metadata is also loaded
                self.assertEqual(loaded_stack[data_key].meta, self.stack[data_key].meta)
                self.assertFalse(id(loaded_stack[data_key].meta) == id(self.stack[data_key].meta),
                                 msg="The meta must not be in the same memory location")
 
                for the_filter in loaded_stack[data_key]:
                    for x in loaded_stack[data_key][the_filter]:
                        for y in loaded_stack[data_key][the_filter][x]:
                            for view in loaded_stack[data_key][the_filter][x][y]:
                                result_origin = self.stack[data_key][the_filter][x][y][view]
                                result_loaded = loaded_stack[data_key][the_filter][x][y][view]
 
                                self.assertEqual(type(result_origin), type(result_loaded))
 
                                if isinstance(result_loaded, Link):
                                    # Does the Link have the 'dataframe' attributre
                                    self.assertTrue(hasattr(loaded_stack[data_key], 'dataframe'))
 
                                    # Is the attribute a Pandas DataFrame ?
                                    self.assertIsInstance(result_loaded.dataframe, pd.DataFrame)
 
                                    # Verify that it is the same after the load
                                    self.assertEqual(result_loaded.dataframe, result_origin.dataframe)
                                    self.assertFalse(id(result_loaded.dataframe) == id(result_origin.dataframe),
                                                     msg="The dataframes must not be in the same memory location")
            if os.path.exists(path_stack):
                os.remove(path_stack)
 
    def test_save_dataset(self):
        # This tests save/load methods using the dataset
        # parameter.
 
        path_stack = '%s%s.stack' % (self.path, self.stack.name)
        self.setup_stack_Example_Data_A()
 
        self.stack.save(path_stack=path_stack, dataset=True)
 
        for key in self.stack.keys():
            path_json = path_stack.replace(
                '.stack',
                ' [{}].json'.format(key))
            path_csv = path_stack.replace(
                '.stack',
                ' [{}].csv'.format(key))
            self.assertTrue(os.path.exists(path_json))
            self.assertTrue(os.path.exists(path_csv))
 
            os.remove(path_json)
            os.remove(path_csv)
  
    def test_save_describe(self):
        # This tests save/load methods using the describe
        # parameter.
 
        path_stack = '%s%s.stack' % (self.path, self.stack.name)
        self.setup_stack_Example_Data_A()
 
        self.stack.save(path_stack=path_stack, describe=True)
 
        path_describe = path_stack.replace('.stack', '.xlsx')
        self.assertTrue(os.path.exists(path_describe))
        os.remove(path_describe)

    def test_stack_aggregate(self):
        b1, ds = _get_batch('test1', full=True)
        b2, ds = _get_batch('test2', ds, False)
        b3, ds = _get_batch('test3', ds, False)
        b1.add_x(['q1', 'q6', 'age'])
        b1.add_y(['gender', 'q2'])
        b1.add_filter('men only', {'gender': 1})
        b1.extend_filter({'q1':{'age': [20, 21, 22]}})
        b1.set_weights('weight_a')
        b2.add_x(['q1', 'q6'])
        b2.add_y(['gender', 'q2'])
        b2.set_weights('weight_b')
        b2.transpose_arrays('q6')
        b3.add_x(['q1', 'q7'])
        b3.add_y(['q2b'])
        b3.add_y_on_y('y_on_y')
        b3.make_summaries(None)
        b3.set_weights(['weight_a', 'weight_b'])
        stack = ds.populate()
        stack.aggregate(['cbase', 'counts', 'c%'], True, 
                        'age', ['test1', 'test2'], verbose=False)
        stack.aggregate(['cbase', 'counts', 'c%', 'counts_sum', 'c%_sum'], 
                        False, None, ['test3'], verbose=False)
        index = ['x|f.c:f|x:|y|weight_a|c%_sum', 'x|f.c:f|x:|y|weight_b|c%_sum', 
                 'x|f.c:f|x:||weight_a|counts_sum', 'x|f.c:f|x:||weight_b|counts_sum', 
                 'x|f|:|y|weight_a|c%', 'x|f|:|y|weight_b|c%', 'x|f|:||weight_a|counts', 
                 'x|f|:||weight_b|counts', 'x|f|x:||weight_a|cbase', 
                 'x|f|x:||weight_b|cbase', 'x|f|x:|||cbase']
        cols = ['@', 'age', 'q1', 'q2b', 'q6', u'q6_1', u'q6_2', u'q6_3', u'q7_1', 
                u'q7_2', u'q7_3', u'q7_4', u'q7_5', u'q7_6']
        values = [['NONE', 'NONE', 2.0, 2.0, 'NONE', 'NONE', 'NONE', 'NONE', 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], 
                  ['NONE', 'NONE', 2.0, 2.0, 'NONE', 'NONE', 'NONE', 'NONE', 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], 
                  ['NONE', 'NONE', 2.0, 2.0, 'NONE', 'NONE', 'NONE', 'NONE', 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], 
                  ['NONE', 'NONE', 2.0, 2.0, 'NONE', 'NONE', 'NONE', 'NONE', 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], 
                  ['NONE', 3.0, 5.0, 2.0, 1.0, 3.0, 3.0, 3.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], 
                  [1.0, 'NONE', 4.0, 2.0, 'NONE', 3.0, 3.0, 3.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], 
                  ['NONE', 3.0, 5.0, 2.0, 1.0, 3.0, 3.0, 3.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], 
                  [1.0, 'NONE', 4.0, 2.0, 'NONE', 3.0, 3.0, 3.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], 
                  ['NONE', 3.0, 5.0, 2.0, 1.0, 3.0, 3.0, 3.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], 
                  [1.0, 'NONE', 4.0, 2.0, 'NONE', 3.0, 3.0, 3.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], 
                  [1.0, 3.0, 6.0, 'NONE', 'NONE', 6.0, 6.0, 6.0, 'NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE']]                
        describe = stack.describe('view', 'x').replace(numpy.NaN, 'NONE')
        self.assertEqual(describe.index.tolist(), index)
        self.assertEqual(describe.columns.tolist(), cols)
        self.assertEqual(describe.values.tolist(), values)  

    def test_cumulative_sum(self):
        b, ds = _get_batch('test1', full=True)
        stack = ds.populate()
        stack.aggregate(['cbase', 'counts', 'c%'], batches='all', verbose=False)
        stack.cumulative_sum(['q1', 'q6'], 'all', verbose=False)
        describe = stack.describe('view', 'x').replace(numpy.NaN, 'NONE')
        index = ['x|f.c:f|x++:|y|weight_a|c%_cumsum', 'x|f.c:f|x++:||weight_a|counts_cumsum', 
                 'x|f|:|y|weight_a|c%', 'x|f|:||weight_a|counts', 'x|f|x:||weight_a|cbase', 'x|f|x:|||cbase']
        cols = ['age', 'q1', 'q2', 'q6', u'q6_1', u'q6_2', u'q6_3']
        values = [['NONE', 3.0, 'NONE', 1.0, 3.0, 3.0, 3.0], 
                  ['NONE', 3.0, 'NONE', 1.0, 3.0, 3.0, 3.0], 
                  ['NONE', 'NONE', 3.0, 'NONE', 'NONE', 'NONE', 'NONE'], 
                  ['NONE', 'NONE', 3.0, 'NONE', 'NONE', 'NONE', 'NONE'], 
                  [3.0, 3.0, 3.0, 1.0, 3.0, 3.0, 3.0], 
                  [3.0, 3.0, 3.0, 'NONE', 3.0, 3.0, 3.0]]
        self.assertEqual(describe.index.tolist(), index)
        self.assertEqual(describe.columns.tolist(), cols)
        self.assertEqual(describe.values.tolist(), values)  

    def test_add_nets(self):
        b, ds = _get_batch('test1', full=True)
        stack = ds.populate()
        stack.aggregate(['cbase', 'counts', 'c%'], batches='all', verbose=False)
        calcu = calc((2, '-', 1), 'difference', 'en-GB')
        stack.add_nets(['q1', 'q6'], [{'Net1': [1, 2]}, {'Net2': [3, 4]}], 'after',
                       calcu, _batches='all', verbose=False)
        index = ['x|f.c:f|x[{1,2}+],x[{3,4}+],x[{3,4}-{1,2}]*:|y|weight_a|net', 
                 'x|f.c:f|x[{1,2}+],x[{3,4}+],x[{3,4}-{1,2}]*:||weight_a|net', 
                 'x|f|:|y|weight_a|c%', 'x|f|:||weight_a|counts', 
                 'x|f|x:||weight_a|cbase', 'x|f|x:|||cbase']
        cols = ['age', 'q1', 'q2', 'q6', u'q6_1', u'q6_2', u'q6_3']
        values = [['NONE', 3.0, 'NONE', 1.0, 3.0, 3.0, 3.0], 
                  ['NONE', 3.0, 'NONE', 1.0, 3.0, 3.0, 3.0], 
                  ['NONE', 'NONE', 3.0, 'NONE', 'NONE', 'NONE', 'NONE'], 
                  ['NONE', 'NONE', 3.0, 'NONE', 'NONE', 'NONE', 'NONE'], 
                  [3.0, 3.0, 3.0, 1.0, 3.0, 3.0, 3.0], 
                  [3.0, 3.0, 3.0, 'NONE', 3.0, 3.0, 3.0]]
        describe = stack.describe('view', 'x').replace(numpy.NaN, 'NONE')
        self.assertEqual(describe.index.tolist(), index)
        self.assertEqual(describe.columns.tolist(), cols)
        self.assertEqual(describe.values.tolist(), values)

    def test_add_stats(self):
        b, ds = _get_batch('test1', full=True)
        stack = ds.populate()
        stack.aggregate(['cbase', 'counts', 'c%'], batches='all', verbose=False)
        stack.add_stats('q6', ['mean'], rescale={1:3, 2:2, 3:1}, factor_labels=False,
                        _batches='all', verbose=False)
        stack.add_stats('q1', ['mean'], 'age', factor_labels=False, verbose=False,
                        _batches='all')
        index = ['x|d.mean|age:||weight_a|stat', 'x|d.mean|x[{3,2,1}]:||weight_a|stat', 
                 'x|f|:|y|weight_a|c%', 'x|f|:||weight_a|counts', 
                 'x|f|x:||weight_a|cbase', 'x|f|x:|||cbase']
        cols = ['age', 'q1', 'q2', 'q6', u'q6_1', u'q6_2', u'q6_3']
        values = [['NONE', 3.0, 'NONE', 'NONE', 'NONE', 'NONE', 'NONE'], 
                  ['NONE', 'NONE', 'NONE', 1.0, 3.0, 3.0, 3.0], 
                  ['NONE', 3.0, 3.0, 1.0, 3.0, 3.0, 3.0], 
                  ['NONE', 3.0, 3.0, 1.0, 3.0, 3.0, 3.0], 
                  [3.0, 3.0, 3.0, 1.0, 3.0, 3.0, 3.0], 
                  [3.0, 3.0, 3.0, 'NONE', 3.0, 3.0, 3.0]]
        describe = stack.describe('view', 'x').replace(numpy.NaN, 'NONE')
        self.assertEqual(describe.index.tolist(), index)
        self.assertEqual(describe.columns.tolist(), cols)
        self.assertEqual(describe.values.tolist(), values) 

    def test_factor_labels(self):
        def _factor_on_values(values, axis = 'x'):
            return all(v['text']['{} edits'.format(axis)]['en-GB'].endswith(
                        '[{}]'.format(v['value'])) for v in values)

        b1, ds = _get_batch('test1', full=True)
        b1.add_x(['q1', 'q2b', 'q6'])
        b1.set_variable_text('q1', 'some new text1')
        b1.set_variable_text('q6', 'some new text1')
        b2, ds = _get_batch('test2', ds, True)
        b2.add_x(['q1', 'q2b', 'q6'])
        b2.set_variable_text('q1', 'some new text2')
        stack = ds.populate()
        stack.aggregate(['cbase', 'counts', 'c%'], batches='all', verbose=False)
        stack.add_stats(['q1', 'q2b', 'q6'], ['mean'], _batches='all', verbose=False)
        for dk in stack.keys():
            meta = stack[dk].meta
            # q1, both batches have meta_edits
            values = meta['sets']['batches']['test1']['meta_edits']['q1']['values']
            self.assertTrue(_factor_on_values(values))
            values = meta['sets']['batches']['test2']['meta_edits']['q1']['values']
            self.assertTrue(_factor_on_values(values))
            values = meta['columns']['q1']['values']
            self.assertTrue(all('x_edits' not in v['text'] for v in values))
            # q2b, no batch has meta_edits
            values = meta['columns']['q2b']['values']
            self.assertTrue(_factor_on_values(values))
            self.assertTrue(all('q2b' not in b['meta_edits'] 
                                for n, b in meta['sets']['batches'].items()))
            # q6, one batch with meta_edits and one without
            values = meta['sets']['batches']['test1']['meta_edits']['lib']['q6']
            self.assertTrue(_factor_on_values(values))
            self.assertTrue(_factor_on_values(values), 'y')
            values = meta['lib']['values']['q6']
            self.assertTrue(_factor_on_values(values))
            self.assertTrue(_factor_on_values(values), 'y')
            self.assertTrue('q6' not in meta['sets']['batches']['test2']['meta_edits'])

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

    def yield_links(self, nest):
        ''' Yields all the Links in nest, which could be either a stack
        or a chain.
        '''
        for dk in nest.keys():
            filters = nest[dk]
            for fk in filters.keys():
                xks = nest[dk][fk]
                for xk in xks.keys():
                    yks = nest[dk][fk][xk]
                    for yk in yks.keys():
                        link = nest[dk][fk][xk][yk]
                        yield link

    def yield_views(self, nest):
        ''' Yields all the Views in nest, which could be either a stack
        or a chain.
        '''
        for link in self.yield_links(nest):
            for vk in link.keys():
                view = link[vk]
                yield view

    def verify_links_and_views_exist_in_nest(self, nest):
        ''' Verifies that Links and Views sit in the appropriate
        levels of the given nest, which could be either a stack
        or a chain.
        '''
        for link in self.yield_links(nest):
            if isinstance(nest, Stack):
                self.assertIsInstance(link, Link)
            for vk in link.keys():
                view = link[vk]
                self.assertIsInstance(view, View)
                self.assertIsInstance(view.dataframe, pd.DataFrame)

    def verify_contains_expected_not_unexpected(self, contents, dk=None, fk=None, xk=None, yk=None, vk=None):
        ''' Verifies that contents (a stack/chain.describe() result) contains all the keys
        passed and no keys that weren't passed.
        '''
        if not dk is None:
            has_dks = contents['data'].unique()
            if isinstance(dk, (str, unicode)): dk = [dk]
            self.assertItemsEqual(has_dks, dk)

        if not fk is None:
            has_fks = contents['filter'].unique()
            if isinstance(fk, (str, unicode)): fk = [fk]
            self.assertItemsEqual(has_fks, fk)

        if not xk is None:
            has_xks = contents['x'].unique()
            if isinstance(xk, (str, unicode)): xk = [xk]
            self.assertItemsEqual(has_xks, xk)

        if not yk is None:
            has_yks = contents['y'].unique()
            if isinstance(yk, (str, unicode)): yk = [yk]
            self.assertItemsEqual(has_yks, yk)

        if not vk is None:
            has_vks = contents['view'].unique()
            if isinstance(vk, (str, unicode)): vk = [vk]
            self.assertItemsEqual(has_vks, vk)

    def setup_stack_Example_Data_A(self, **kwargs):
        self.stack = self.get_stack_Example_Data_A(**kwargs)

    def get_stack_Example_Data_A(self, name=None, fk=None, xk=None, yk=None, views=None, weights=None):
        if name is None:
            name = "Example Data (A)"
        if fk is None:
            fk = ['no_filter']
        if xk is None:
            xk = self.minimum
        if yk is None:
            yk = ['@'] + self.minimum
        if views is None:
            views = ['default']
        if not isinstance(weights, list):
            weights = [weights]

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
                weights=weight
            )

        return stack

if __name__ == '__main__':
    unittest.main()














