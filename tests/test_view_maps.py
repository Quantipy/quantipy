
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
from quantipy.core.tools.view.query import get_dataframe
from quantipy.core.helpers.functions import emulate_meta

class TestViewObject(unittest.TestCase):

    ''' Data file check '''
    def setUp(self):
        path = os.path.dirname(os.path.abspath(__file__))
        casedata = '/Example Data (A).csv'
        metadata = '/Example Data (A).json'
        try:
            os.path.exists(path+casedata)
        except:
            raise Exception("THERE IS NO Example Data (A).CSV FILE !!!")
        try:
            os.path.exists(path+metadata)
        except:
            raise Exception("THERE IS NO Example Data (A).JSON FILE !!!")

    ''' default view: Test that verify single @-links are working correctly '''
    def test_default_int_at_no_w(self):
        views = QuantipyViews(['default'])
        x = 'age'
        y = '@'
        self.setup_stack(
                        views=views,
                        x=x,
                        y=y
                        )
        viewkeys = self.stack['testing']['no_filter'][x][y].keys()
        df = self.stack['testing']['no_filter'][x][y][viewkeys[0]].dataframe
        meta = self.stack['testing']['no_filter'][x][y][viewkeys[0]].meta()

        self.assertTrue(viewkeys[0] == 'x|default|:|||default')
        # changed when margins dropped from default numeric links
#         self.assertEqual(round(np.nansum(df.values), 6), 8467.848047)
        self.assertEqual(round(np.nansum(df.values), 6), 212.848047 )

        self.assertTrue(meta['agg']['name'] == 'default')
        self.assertTrue(meta['agg']['fullname'] == 'x|default|:|||default')
        self.assertTrue(meta['agg']['is_weighted'] == False)
        self.assertTrue(meta['agg']['weights'] == None)
        self.assertTrue(meta['agg']['method'] == 'default')

        self.assertTrue(meta['x']['name'] == 'age')
        self.assertTrue(meta['x']['is_multi'] == False)
        self.assertTrue(meta['x']['is_nested'] == False)

        self.assertTrue(meta['y']['name'] == '@')
        self.assertTrue(meta['y']['is_multi'] == False)
        self.assertTrue(meta['y']['is_nested'] == False)

        # changed when margins dropped from default numeric links
#         self.assertTrue(meta['shape'] ==  (8, 1))
        self.assertTrue(meta['shape'] ==  (7, 1))

    def test_default_int_at_w(self):
        views = QuantipyViews(['default'])
        x = 'age'
        y = '@'
        self.setup_stack(
                        views=views,
                        x=x,
                        y=y,
                        weights='weight_a'
                        )
        viewkeys = self.stack['testing']['no_filter'][x][y].keys()
        df = self.stack['testing']['no_filter'][x][y][viewkeys[0]].dataframe
        meta = self.stack['testing']['no_filter'][x][y][viewkeys[0]].meta()

        self.assertTrue(viewkeys[0] == 'x|default|:||weight_a|default')
        # changed when margins dropped from default numeric links
#         self.assertEqual(round(np.nansum(df.values), 6), 8467.821229)
        self.assertEqual(round(np.nansum(df.values), 6), 212.821229)

        self.assertTrue(meta['agg']['name'] == 'default')
        self.assertTrue(meta['agg']['fullname'] == 'x|default|:||weight_a|default')
        self.assertTrue(meta['agg']['is_weighted'] == True)
        self.assertTrue(meta['agg']['weights'] == 'weight_a')
        self.assertTrue(meta['agg']['method'] == 'default')

        self.assertTrue(meta['x']['name'] == 'age')
        self.assertTrue(meta['x']['is_multi'] == False)
        self.assertTrue(meta['x']['is_nested'] == False)

        self.assertTrue(meta['y']['name'] == '@')
        self.assertTrue(meta['y']['is_multi'] == False)
        self.assertTrue(meta['y']['is_nested'] == False)

        # changed when margins dropped from default numeric links
#         self.assertTrue(meta['shape'] ==  (8, 1))
        self.assertTrue(meta['shape'] ==  (7, 1))

    def test_default_float_at_no_w(self):
        views = QuantipyViews(['default'])
        x = 'weight_b'
        y = '@'
        self.setup_stack(
                        views=views,
                        x=x,
                        y=y
                        )

        viewkeys = self.stack['testing']['no_filter'][x][y].keys()
        df = self.stack['testing']['no_filter'][x][y][viewkeys[0]].dataframe
        meta = self.stack['testing']['no_filter'][x][y][viewkeys[0]].meta()
        self.assertTrue(viewkeys[0] == 'x|default|:|||default')

        # changed when margins dropped from default numeric links
#         self.assertEqual(round(np.nansum(df.values), 6), 8107.098562)
        self.assertEqual(round(np.nansum(df.values), 6), 9.098562)

        self.assertTrue(meta['agg']['name'] == 'default')
        self.assertTrue(meta['agg']['fullname'] == 'x|default|:|||default')
        self.assertTrue(meta['agg']['is_weighted'] == False)
        self.assertTrue(meta['agg']['weights'] == None)
        self.assertTrue(meta['agg']['method'] == 'default')

        self.assertTrue(meta['x']['name'] == 'weight_b')
        self.assertTrue(meta['x']['is_multi'] == False)
        self.assertTrue(meta['x']['is_nested'] == False)

        self.assertTrue(meta['y']['name'] == '@')
        self.assertTrue(meta['y']['is_multi'] == False)
        self.assertTrue(meta['y']['is_nested'] == False)

        # changed when margins dropped from default numeric links
#         self.assertTrue(meta['shape'] ==  (8, 1))
        self.assertTrue(meta['shape'] ==  (7, 1))

    def test_default_float_at_w(self):
        views = QuantipyViews(['default'])
        x = 'weight_b'
        y = '@'
        self.setup_stack(
                        views=views,
                        x=x,
                        y=y,
                        weights='weight_a'
                        )

        viewkeys = self.stack['testing']['no_filter'][x][y].keys()
        df = self.stack['testing']['no_filter'][x][y][viewkeys[0]].dataframe
        meta = self.stack['testing']['no_filter'][x][y][viewkeys[0]].meta()

        self.assertTrue(viewkeys[0] == 'x|default|:||weight_a|default')

        # changed when margins dropped from default numeric links
#         self.assertEqual(round(np.nansum(df.values), 6), 8035.373694)
        self.assertEqual(round(np.nansum(df.values), 6), 11.513689)

        self.assertTrue(meta['agg']['name'] == 'default')
        self.assertTrue(meta['agg']['fullname'] == 'x|default|:||weight_a|default')
        self.assertTrue(meta['agg']['is_weighted'] == True)
        self.assertTrue(meta['agg']['weights'] == 'weight_a')
        self.assertTrue(meta['agg']['method'] == 'default')

        self.assertTrue(meta['x']['name'] == 'weight_b')
        self.assertTrue(meta['x']['is_multi'] == False)
        self.assertTrue(meta['x']['is_nested'] == False)

        self.assertTrue(meta['y']['name'] == '@')
        self.assertTrue(meta['y']['is_multi'] == False)
        self.assertTrue(meta['y']['is_nested'] == False)

        # changed when margins dropped from default numeric links
#         self.assertTrue(meta['shape'] ==  (8, 1))
        self.assertTrue(meta['shape'] ==  (7, 1))

    def test_default_single_at_no_w(self):
        views = QuantipyViews(['default'])
        x = 'gender'
        y = '@'
        self.setup_stack(
                        views=views,
                        x=x,
                        y=y,
                        )

        viewkeys = self.stack['testing']['no_filter'][x][y].keys()
        df = self.stack['testing']['no_filter'][x][y][viewkeys[0]].dataframe
        meta = self.stack['testing']['no_filter'][x][y][viewkeys[0]].meta()


        self.assertTrue(viewkeys[0] == 'x|default|:|||default')
        self.assertEqual(round(np.nansum(df.values), 6), 16510.0)

        self.assertTrue(meta['agg']['name'] == 'default')
        self.assertTrue(meta['agg']['fullname'] == 'x|default|:|||default')
        self.assertTrue(meta['agg']['is_weighted'] == False)
        self.assertTrue(meta['agg']['weights'] == None)
        self.assertTrue(meta['agg']['method'] == 'default')

        self.assertTrue(meta['x']['name'] == 'gender')
        self.assertTrue(meta['x']['is_multi'] == False)
        self.assertTrue(meta['x']['is_nested'] == False)

        self.assertTrue(meta['y']['name'] == '@')
        self.assertTrue(meta['y']['is_multi'] == False)
        self.assertTrue(meta['y']['is_nested'] == False)

        self.assertTrue(meta['shape'] ==  (3, 1))

    def test_default_single_at_w(self):
        views = QuantipyViews(['default'])
        x = 'gender'
        y = '@'
        self.setup_stack(
                        views=views,
                        x=x,
                        y=y,
                        weights='weight_a'
                        )

        viewkeys = self.stack['testing']['no_filter'][x][y].keys()
        df = self.stack['testing']['no_filter'][x][y][viewkeys[0]].dataframe
        meta = self.stack['testing']['no_filter'][x][y][viewkeys[0]].meta()

        self.assertTrue(viewkeys[0] == 'x|default|:||weight_a|default')

        self.assertEqual(round(np.nansum(df.values), 6), 16510.0)

        self.assertTrue(meta['agg']['name'] == 'default')
        self.assertTrue(meta['agg']['fullname'] == 'x|default|:||weight_a|default')
        self.assertTrue(meta['agg']['is_weighted'] == True)
        self.assertTrue(meta['agg']['weights'] == 'weight_a')
        self.assertTrue(meta['agg']['method'] == 'default')

        self.assertTrue(meta['x']['name'] == 'gender')
        self.assertTrue(meta['x']['is_multi'] == False)
        self.assertTrue(meta['x']['is_nested'] == False)

        self.assertTrue(meta['y']['name'] == '@')
        self.assertTrue(meta['y']['is_multi'] == False)
        self.assertTrue(meta['y']['is_nested'] == False)

        self.assertTrue(meta['shape'] ==  (3, 1))

    def test_default_delimited_at_no_w(self):
        views = QuantipyViews(['default'])
        x = 'q9'
        y = '@'
        self.setup_stack(
                        views=views,
                        x=x,
                        y=y,
                        )

        viewkeys = self.stack['testing']['no_filter'][x][y].keys()
        df = self.stack['testing']['no_filter'][x][y][viewkeys[0]].dataframe
        meta = self.stack['testing']['no_filter'][x][y][viewkeys[0]].meta()

        self.assertTrue(viewkeys[0] == 'x|default|:|||default')
        self.assertEqual(round(np.nansum(df.values), 6), 20504.0)

        self.assertTrue(meta['agg']['name'] == 'default')
        self.assertTrue(meta['agg']['fullname'] == 'x|default|:|||default')
        self.assertTrue(meta['agg']['is_weighted'] == False)
        self.assertTrue(meta['agg']['weights'] == None)
        self.assertTrue(meta['agg']['method'] == 'default')

        self.assertTrue(meta['x']['name'] == 'q9')
        self.assertTrue(meta['x']['is_multi'] == True)
        self.assertTrue(meta['x']['is_nested'] == False)

        self.assertTrue(meta['y']['name'] == '@')
        self.assertTrue(meta['y']['is_multi'] == False)
        self.assertTrue(meta['y']['is_nested'] == False)

        self.assertTrue(meta['shape'] ==  (8, 1))

    def test_default_delimited_at_w(self):
        views = QuantipyViews(['default'])
        x = 'q9'
        y = '@'
        self.setup_stack(
                        views=views,
                        x=x,
                        y=y,
                        weights='weight_a'
                        )

        viewkeys = self.stack['testing']['no_filter'][x][y].keys()
        df = self.stack['testing']['no_filter'][x][y][viewkeys[0]].dataframe
        meta = self.stack['testing']['no_filter'][x][y][viewkeys[0]].meta()

        self.assertTrue(viewkeys[0] == 'x|default|:||weight_a|default')
        self.assertEqual(round(np.nansum(df.values), 6), 20622.967226)

        self.assertTrue(meta['agg']['name'] == 'default')
        self.assertTrue(meta['agg']['fullname'] == 'x|default|:||weight_a|default')
        self.assertTrue(meta['agg']['is_weighted'] == True)
        self.assertTrue(meta['agg']['weights'] == 'weight_a')
        self.assertTrue(meta['agg']['method'] == 'default')

        self.assertTrue(meta['x']['name'] == 'q9')
        self.assertTrue(meta['x']['is_multi'] == True)
        self.assertTrue(meta['x']['is_nested'] == False)

        self.assertTrue(meta['y']['name'] == '@')
        self.assertTrue(meta['y']['is_multi'] == False)
        self.assertTrue(meta['y']['is_nested'] == False)

        self.assertTrue(meta['shape'] ==  (8, 1))

    ''' default view: Test that verify x=y-links are working correctly '''
    def test_default_int_on_int_no_w(self):
        views = QuantipyViews(['default'])
        x = 'age'
        y = 'age'
        self.setup_stack(
                        views=views,
                        x=x,
                        y=y,
                        )
        viewkeys = self.stack['testing']['no_filter'][x][y].keys()
        df = self.stack['testing']['no_filter'][x][y][viewkeys[0]].dataframe
        meta = self.stack['testing']['no_filter'][x][y][viewkeys[0]].meta()
        results_all =[[8255.0], [33.9061175045427], [8.941929888811542], [19.0], [26.0], [34.0], [42.0], [49.0]]
        results_intersect = [[26]]
#         self.assertTrue(np.allclose(df.xs('All', level=1, axis=1).values, results_all))
        self.assertTrue(np.array_equal(df.xs('mean', axis=0, level=1).xs(26, axis=1, level=1).values, results_intersect))

        self.assertTrue(meta['x'].values() == meta['y'].values())

    def test_default_delimited_on_delimited_w(self):
        views = QuantipyViews(['default'])
        x = 'q9'
        y = 'q9'
        self.setup_stack(
                        views=views,
                        x=x,
                        y=y,
                        weights='weight_a'
                        )

        viewkeys = self.stack['testing']['no_filter'][x][y].keys()
        df = self.stack['testing']['no_filter'][x][y][viewkeys[0]].dataframe
        meta = self.stack['testing']['no_filter'][x][y][viewkeys[0]].meta()

        results_x_all = [[8255.000000000611, 3052.249139193268, 1723.1099695522712, 896.6852717638529, 2280.3900812423453,
                        341.12706352421003, 1304.518495129215, 2769.887205952252]]
        results_y_all = [[8255.000000000611], [3052.2491391932563], [1723.1099695522678], [896.685271763852],
                        [2280.390081242337], [341.1270635242103], [1304.518495129213], [2769.8872059522414]]
        results_x_axis_codes = ['All', 1, 2, 3, 4, 96, 98, 99]

        self.assertTrue(np.allclose(df.xs('All', level=1, axis=0).values, results_x_all))
        self.assertTrue(np.allclose(df.xs('All', level=1, axis=1).values, results_y_all))
        self.assertEqual(df.index.get_level_values(1).tolist(), results_x_axis_codes)

    ''' non-default views: Test that verify mixed-type-links are working correctly '''
    def test_bases_float_on_single_w(self):
        views = QuantipyViews(['cbase', 'rbase'])
        x = 'weight_b'
        y = 'gender'
        self.setup_stack(
                        views=views,
                        x=x,
                        y=y,
                        weights='weight_a'
                        )

        viewkeys = self.stack['testing']['no_filter'][x][y].keys()

        # check view key naming
        self.assertTrue('x|f|x:||weight_a|cbase' in viewkeys)
        self.assertTrue('x|f|:y||weight_a|rbase' in viewkeys)

        # test for column bases
        df = self.stack['testing']['no_filter'][x][y]['x|f|x:||weight_a|cbase'].dataframe
        meta = self.stack['testing']['no_filter'][x][y]['x|f|x:||weight_a|cbase'].meta()
        results_bases = [[3854.951079386677, 4168.90892510124]]

        self.assertTrue(np.allclose(df.values, results_bases))

        self.assertTrue(meta['agg']['name'] == 'cbase')
        self.assertTrue(meta['agg']['is_weighted'] == True)
        self.assertTrue(meta['agg']['weights'] == 'weight_a')
        self.assertTrue(meta['agg']['method'] == 'frequency')

        self.assertTrue(meta['x']['name'] == 'weight_b')
        self.assertTrue(meta['x']['is_multi'] == False)
        self.assertTrue(meta['x']['is_nested'] == False)

        self.assertTrue(meta['y']['name'] == 'gender')
        self.assertTrue(meta['y']['is_multi'] == False)
        self.assertTrue(meta['y']['is_nested'] == False)

        self.assertTrue(meta['shape'] ==  (1, 2))

        self.assertTrue(meta['agg']['fullname'] == 'x|f|x:||weight_a|cbase')
        self.assertTrue(meta['agg']['text'] == 'Base')

        #test for row bases
        df = self.stack['testing']['no_filter'][x][y]['x|f|:y||weight_a|rbase'].dataframe
        meta = self.stack['testing']['no_filter'][x][y]['x|f|:y||weight_a|rbase'].meta()
        results_bases_first_10 = [[0.606200775342], [0.228206355215], [0.229625249702], [0.233572172205], [4.410317090241001],
                                  [1.29664815927], [1.055148517124], [4.68784645512], [0.572517025908], [0.8659851290640002]]
        self.assertTrue(np.allclose(df.head(10).values, results_bases_first_10))

        self.assertTrue(meta['agg']['name'] == 'rbase')
        self.assertTrue(meta['agg']['is_weighted'] == True)
        self.assertTrue(meta['agg']['weights'] == 'weight_a')
        self.assertTrue(meta['agg']['method'] == 'frequency')

        self.assertTrue(meta['x']['name'] == 'weight_b')
        self.assertTrue(meta['x']['is_multi'] == False)
        self.assertTrue(meta['x']['is_nested'] == False)

        self.assertTrue(meta['y']['name'] == 'gender')
        self.assertTrue(meta['y']['is_multi'] == False)
        self.assertTrue(meta['y']['is_nested'] == False)
        self.assertTrue(meta['shape'] ==  (727, 1))

        self.assertTrue(meta['agg']['fullname'] == 'x|f|:y||weight_a|rbase')

        self.assertTrue(meta['agg']['text'] == 'Base')

    def test_frequencies_single_on_delimited_w(self):
        views = QuantipyViews(['counts', 'cbase', 'rbase', 'c%', 'r%'])
        x = 'q1'
        y = 'q3'
        self.setup_stack(
                        views=views,
                        x=x,
                        y=y,
                        weights='weight_a'
                        )

        viewkeys = self.stack['testing']['no_filter'][x][y].keys()
        # check view key naming
        self.assertTrue('x|f|:||weight_a|counts' in viewkeys)
        self.assertTrue('x|f|:|y|weight_a|c%' in viewkeys)
        self.assertTrue('x|f|:|x|weight_a|r%' in viewkeys)

        # test for counts
        df = self.stack['testing']['no_filter'][x][y]['x|f|:||weight_a|counts'].dataframe
        meta = self.stack['testing']['no_filter'][x][y]['x|f|:||weight_a|counts'].meta()

        results_last_4_codes = [[0.0, 0.0, 3.44709414359, 0.0],
                                [0.821382495122, 22.558078157094002, 0.0, 1.732273332958],
                                [0.955723639461, 8.362635087097, 0.826341200496, 24.219637692783998],
                                [0.573290188245, 11.229184465293999, 1.8703201042210003, 39.487432678395]]

        self.assertTrue(np.allclose(df.tail(4).T.tail(4).T.values, results_last_4_codes))

        self.assertTrue(meta['agg']['name'] == 'counts')
        self.assertTrue(meta['agg']['is_weighted'] == True)
        self.assertTrue(meta['agg']['weights'] == 'weight_a')
        self.assertTrue(meta['agg']['method'] == 'frequency')

        self.assertTrue(meta['x']['name'] == 'q1')
        self.assertTrue(meta['x']['is_multi'] == False)
        self.assertTrue(meta['x']['is_nested'] == False)

        self.assertTrue(meta['y']['name'] == 'q3')
        self.assertTrue(meta['y']['is_multi'] == True)
        self.assertTrue(meta['y']['is_nested'] == False)
        self.assertTrue(meta['shape'] ==  (12, 9))

        self.assertTrue(meta['agg']['fullname'] == 'x|f|:||weight_a|counts')
        self.assertTrue(meta['agg']['text'] == '')

        # test for column percentages
        df = self.stack['testing']['no_filter'][x][y]['x|f|:|y|weight_a|c%'].dataframe
        meta = self.stack['testing']['no_filter'][x][y]['x|f|:|y|weight_a|c%'].meta()

        results_first_4_codes = [[3.4164893569353234, 3.417954270304682, 3.492018770073583, 3.8586355507159444],
                                [4.227870164831371, 4.578461082166348, 4.584324860290225, 3.1928372306033808],
                                [23.56828772697806, 22.987572077142847, 26.587049664959217, 17.579103923262892],
                                [41.25716641855391, 43.16460410688994, 38.3839681942993, 50.59512968991407]]

        self.assertTrue(np.allclose(df.head(4).T.head(4).T.values, results_first_4_codes))

        self.assertTrue(meta['agg']['name'] == 'c%')
        self.assertTrue(meta['agg']['is_weighted'] == True)
        self.assertTrue(meta['agg']['weights'] == 'weight_a')
        self.assertTrue(meta['agg']['method'] == 'frequency')

        self.assertTrue(meta['x']['name'] == 'q1')
        self.assertTrue(meta['x']['is_multi'] == False)
        self.assertTrue(meta['x']['is_nested'] == False)

        self.assertTrue(meta['y']['name'] == 'q3')
        self.assertTrue(meta['y']['is_multi'] == True)
        self.assertTrue(meta['y']['is_nested'] == False)
        self.assertTrue(meta['shape'] ==  (12, 9))

        self.assertTrue(meta['agg']['fullname'] == 'x|f|:|y|weight_a|c%')
        self.assertTrue(meta['agg']['text'] == '')

        # test for row percentages
        df = self.stack['testing']['no_filter'][x][y]['x|f|:|x|weight_a|r%'].dataframe
        meta = self.stack['testing']['no_filter'][x][y]['x|f|:|x|weight_a|r%'].meta()
        results_last_row = [[64.12450762534469, 36.81913905767139, 80.14448727523266, 2.8203344805561996,
                            0.3708073255480199, 0.1906190441644346, 3.733704942819817, 0.6218816192189421, 13.129575262234159]]

        self.assertTrue(np.allclose(df.tail(1).values, results_last_row))

        self.assertTrue(meta['agg']['name'] == 'r%')
        self.assertTrue(meta['agg']['is_weighted'] == True)
        self.assertTrue(meta['agg']['weights'] == 'weight_a')
        self.assertTrue(meta['agg']['method'] == 'frequency')

        self.assertTrue(meta['x']['name'] == 'q1')
        self.assertTrue(meta['x']['is_multi'] == False)
        self.assertTrue(meta['x']['is_nested'] == False)

        self.assertTrue(meta['y']['name'] == 'q3')
        self.assertTrue(meta['y']['is_multi'] == True)
        self.assertTrue(meta['y']['is_nested'] == False)
        self.assertTrue(meta['shape'] ==  (12, 9))

        self.assertTrue(meta['agg']['fullname'] == 'x|f|:|x|weight_a|r%')
        self.assertTrue(meta['agg']['text'] == '')

    def test_frequencies_delimited_on_delimited_no_w(self):
        views = QuantipyViews(['counts', 'cbase', 'rbase', 'c%', 'r%'])
        x = 'q2'
        y = 'q3'
        self.setup_stack(
                        views=views,
                        x=x,
                        y=y,
                        )

        viewkeys = self.stack['testing']['no_filter'][x][y].keys()
        # check view key naming
        self.assertTrue('x|f|:|||counts' in viewkeys)
        self.assertTrue('x|f|:|y||c%' in viewkeys)
        self.assertTrue('x|f|:|x||r%' in viewkeys)

        # test for counts
        df = self.stack['testing']['no_filter'][x][y]['x|f|:|||counts'].dataframe
        meta = self.stack['testing']['no_filter'][x][y]['x|f|:|||counts'].meta()
        results_last_4_codes = [[27.0, 156.0, 40.0, 3.0],
                                [14.0, 116.0, 33.0, 8.0],
                                [2.0, 78.0, 19.0, 14.0],[0.0, 6.0, 1.0, 3.0]]

        self.assertTrue(np.allclose(df.tail(4).T.tail(4).T.values, results_last_4_codes))

        self.assertTrue(meta['agg']['name'] == 'counts')
        self.assertTrue(meta['agg']['is_weighted'] == False)
        self.assertTrue(meta['agg']['weights'] == None)
        self.assertTrue(meta['agg']['method'] == 'frequency')

        self.assertTrue(meta['x']['name'] == 'q2')
        self.assertTrue(meta['x']['is_multi'] == True)
        self.assertTrue(meta['x']['is_nested'] == False)

        self.assertTrue(meta['y']['name'] == 'q3')
        self.assertTrue(meta['y']['is_multi'] == True)
        self.assertTrue(meta['y']['is_nested'] == False)
        self.assertTrue(meta['shape'] ==  (8, 9))

        self.assertTrue(meta['agg']['fullname'] == 'x|f|:|||counts')
        self.assertTrue(meta['agg']['text'] == '')

        # test for column percentages
        df = self.stack['testing']['no_filter'][x][y]['x|f|:|y||c%'].dataframe
        meta = self.stack['testing']['no_filter'][x][y]['x|f|:|y||c%'].meta()
        results_first_4_codes = [[40.55727554179567, 39.472349061390155, 38.291380625476734, 46.83098591549296],
                                [46.351172047766475, 46.93049213597158, 46.3768115942029, 46.12676056338028],
                                [60.76957098628925, 61.54236428209031, 57.93287566742944, 65.14084507042254],
                                [23.087129588677577, 23.642820903094876, 22.578184591914567, 31.690140845070424]]

        self.assertTrue(np.allclose(df.head(4).T.head(4).T.values, results_first_4_codes))
        self.assertTrue(meta['agg']['name'] == 'c%')
        self.assertTrue(meta['agg']['is_weighted'] == False)
        self.assertTrue(meta['agg']['weights'] == None)
        self.assertTrue(meta['agg']['method'] == 'frequency')

        self.assertTrue(meta['x']['name'] == 'q2')
        self.assertTrue(meta['x']['is_multi'] == True)
        self.assertTrue(meta['x']['is_nested'] == False)

        self.assertTrue(meta['y']['name'] == 'q3')
        self.assertTrue(meta['y']['is_multi'] == True)
        self.assertTrue(meta['y']['is_nested'] == False)
        self.assertTrue(meta['shape'] ==  (8, 9))

        self.assertTrue(meta['agg']['fullname'] == 'x|f|:|y||c%')
        self.assertTrue(meta['agg']['text'] == '')

        # test for row percentages
        df = self.stack['testing']['no_filter'][x][y]['x|f|:|x||r%'].dataframe

        meta = self.stack['testing']['no_filter'][x][y]['x|f|:|x||r%'].meta()
        results_last_row = [[69.81132075471697, 45.28301886792453, 75.47169811320755, 7.547169811320755,
                             0.0, 0.0, 11.320754716981133, 1.8867924528301887, 5.660377358490567]]

        self.assertTrue(np.allclose(df.tail(1).values, results_last_row))

        self.assertTrue(meta['agg']['name'] == 'r%')
        self.assertTrue(meta['agg']['is_weighted'] == False)
        self.assertTrue(meta['agg']['weights'] == None)
        self.assertTrue(meta['agg']['method'] == 'frequency')

        self.assertTrue(meta['x']['name'] == 'q2')
        self.assertTrue(meta['x']['is_multi'] == True)
        self.assertTrue(meta['x']['is_nested'] == False)

        self.assertTrue(meta['y']['name'] == 'q3')
        self.assertTrue(meta['y']['is_multi'] == True)
        self.assertTrue(meta['y']['is_nested'] == False)
        self.assertTrue(meta['shape'] ==  (8, 9))

        self.assertTrue(meta['agg']['fullname'] == 'x|f|:|x||r%')
        self.assertTrue(meta['agg']['text'] == '')

    def test_frequencies_single_on_single_no_w(self):
        views = QuantipyViews(['counts', 'cbase', 'rbase', 'c%', 'r%'])
        x = 'religion'
        y = 'q1'
        self.setup_stack(
                        views=views,
                        x=x,
                        y=y,
                        )

        viewkeys = self.stack['testing']['no_filter'][x][y].keys()
        # check view key naming
        self.assertTrue('x|f|:|||counts' in viewkeys)
        self.assertTrue('x|f|:|y||c%' in viewkeys)
        self.assertTrue('x|f|:|x||r%' in viewkeys)

        # test for counts
        df = self.stack['testing']['no_filter'][x][y]['x|f|:|||counts'].dataframe
        meta = self.stack['testing']['no_filter'][x][y]['x|f|:|||counts'].meta()
        results_last_4_codes = [[0.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 3.0],
                                [0.0, 2.0, 2.0, 20.0],
                                [1.0, 1.0, 2.0, 8.0]]

        self.assertTrue(np.allclose(df.tail(4).T.tail(4).T.values, results_last_4_codes))

        self.assertTrue(meta['agg']['name'] == 'counts')
        self.assertTrue(meta['agg']['is_weighted'] == False)
        self.assertTrue(meta['agg']['weights'] == None)
        self.assertTrue(meta['agg']['method'] == 'frequency')

        self.assertTrue(meta['x']['name'] == 'religion')
        self.assertTrue(meta['x']['is_multi'] == False)
        self.assertTrue(meta['x']['is_nested'] == False)

        self.assertTrue(meta['y']['name'] == 'q1')
        self.assertTrue(meta['y']['is_multi'] == False)
        self.assertTrue(meta['y']['is_nested'] == False)
        self.assertTrue(meta['shape'] ==  (16, 12))

        self.assertTrue(meta['agg']['fullname'] == 'x|f|:|||counts')
        self.assertTrue(meta['agg']['text'] == '')

        # test for column percentages
        df = self.stack['testing']['no_filter'][x][y]['x|f|:|y||c%'].dataframe
        meta = self.stack['testing']['no_filter'][x][y]['x|f|:|y||c%'].meta()
        results_first_4_codes = [[34.437086092715234, 31.41592920353982, 32.68416596104996, 30.646193218170186],
                                [37.086092715231786, 39.38053097345133, 38.01862828111769, 39.027511196417144],
                                [11.258278145695364, 6.637168141592921, 10.160880609652837, 10.17274472168906],
                                [7.28476821192053, 1.7699115044247788, 3.302286198137172, 5.502239283429303]]

        self.assertTrue(np.allclose(df.head(4).T.head(4).T.values, results_first_4_codes))

        self.assertTrue(meta['agg']['name'] == 'c%')
        self.assertTrue(meta['agg']['is_weighted'] == False)
        self.assertTrue(meta['agg']['weights'] == None)
        self.assertTrue(meta['agg']['method'] == 'frequency')

        self.assertTrue(meta['x']['name'] == 'religion')
        self.assertTrue(meta['x']['is_multi'] == False)
        self.assertTrue(meta['x']['is_nested'] == False)

        self.assertTrue(meta['y']['name'] == 'q1')
        self.assertTrue(meta['y']['is_multi'] == False)
        self.assertTrue(meta['y']['is_nested'] == False)
        self.assertTrue(meta['shape'] ==  (16, 12))

        self.assertTrue(meta['agg']['fullname'] == 'x|f|:|y||c%')
        self.assertTrue(meta['agg']['text'] == '')

        # test for row percentages
        df = self.stack['testing']['no_filter'][x][y]['x|f|:|x||r%'].dataframe
        meta = self.stack['testing']['no_filter'][x][y]['x|f|:|x||r%'].meta()
        results_last_row = [[1.2658227848101267, 5.063291139240507, 35.44303797468354, 30.37974683544304,
                            1.2658227848101267, 3.79746835443038, 6.329113924050633, 1.2658227848101267,
                            1.2658227848101267, 1.2658227848101267, 2.5316455696202533, 10.126582278481013]]

        self.assertTrue(np.allclose(df.tail(1).values, results_last_row))

        self.assertTrue(meta['agg']['name'] == 'r%')
        self.assertTrue(meta['agg']['is_weighted'] == False)
        self.assertTrue(meta['agg']['weights'] == None)
        self.assertTrue(meta['agg']['method'] == 'frequency')

        self.assertTrue(meta['x']['name'] == 'religion')
        self.assertTrue(meta['x']['is_multi'] == False)
        self.assertTrue(meta['x']['is_nested'] == False)

        self.assertTrue(meta['y']['name'] == 'q1')
        self.assertTrue(meta['y']['is_multi'] == False)
        self.assertTrue(meta['y']['is_nested'] == False)
        self.assertTrue(meta['shape'] ==  (16, 12))

        self.assertTrue(meta['agg']['fullname'] == 'x|f|:|x||r%')
        self.assertTrue(meta['agg']['text'] == '')

    ''' num_stats views: Test that verify numerical summary statistics are calculated correctly '''
    def test_simple_means_all_types_no_w(self):
        '''
        Support for means (and other descriptive measures) has been removed from Quantipy.
        This tests only considers (categorical) int/float on X and (categorical) int on Y.
        Tests for float on Y could be added, but they are extremly expensive while not being used
        in the real world.
        '''
        views = QuantipyViews(['mean'])
        x = ['religion', 'age', 'weight_b']
        y = ['religion', 'q9']
        self.setup_stack(
                        views=views,
                        x=x,
                        y=y,
                        )

        results_first_15_means = {
            ('age', 'q9'): [[34.034, 33.66126013724267, 33.94308943089431, 34.05971520440974, 33.23160762942779, 33.92709867452135, 33.91605966007631]],
            ('age', 'religion'): [[33.96017699115044, 33.86948297604035, 33.506637168141594, 34.101063829787236, 34.888, 33.984375, 34.23529411764706, 39.0, 34.0,
                                34.27272727272727, 34.421052631578945, 33.490196078431374, 36.18181818181818, 36.458333333333336, 32.646511627906975]],
            ('religion', 'q9'): [[3.2643908969210176, 3.3703703703703702, 3.360189573459716, 3.3056603773584907, 3.4771573604060912, 3.314327485380117, 3.3649588867805185]] ,
            ('religion', 'religion'): [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]] ,
            ('weight_b', 'q9'): [[1.0182296770611803, 1.0758356686570276, 1.0418269358274441, 1.0456914765196268, 0.9312216701981225, 0.9610482592712275, 0.9597691403990088]],
            ('weight_b', 'religion'): [[0.9100697643462116, 1.0207477587602642, 1.0697000971715473, 0.8942976339536192, 1.0811501419933254, 0.8132031171672097, 0.9227091789183529,
                                        0.731571892767, 1.2594213912240002, 0.8741697110032545, 0.9767286348968889, 1.0179290882704313, 1.0400800039273637, 0.911321417807913, 0.8929007681113169]],
            }

        for X in x:
            for Y in y:
                df = self.stack['testing']['no_filter'][X][Y]['x|d.mean|x:|||mean'].dataframe
                self.assertTrue(np.allclose(df.T.head(15).T.values, results_first_15_means[(X, Y)]))

    def test_exclude_and_rescale_on_means_categorical_w(self):
        views = QuantipyViews(['mean'])
        views['mean']['kwargs']['exclude'] = [2, 3]
        views['mean']['kwargs']['rescale'] = {4:300, 7:900, 187:555, 99:900}
        x = ['religion']
        y = ['religion', 'q9']
        self.setup_stack(
                        views=views,
                        x=x,
                        y=y,
                        weights='weight_a'
                        )

        results_first_15_means = {
            ('religion', 'religion'): [[1.0, 0.00, 0.00, 299.99999999999994, 5.0, 6.0, 900.00, 8.000,
                                        9.00, 10.00, 11.00, 12.00, 13.00, 14.0, 15.0]],
            ('religion', 'q9'):  [[42.551728834710325, 39.24544561891599, 27.051829048758677, 33.24349179295309, 37.32054819314336, 28.864524650046256, 35.661188879722964]]
            }

        viewkey_store = []
        for X in x:
            for Y in y:
                viewkeys = self.stack['testing']['no_filter'][X][Y].keys()
                for viewkey in viewkeys:
                    if not viewkey == 'x|default|:||weight_b|default':
                        df = self.stack['testing']['no_filter'][X][Y][viewkey].dataframe
                        self.assertTrue(np.allclose(np.nan_to_num(df.T.head(15).T.values), results_first_15_means[(X, Y)]))
                        viewkey_store.append(viewkey)
        self.assertTrue('x|d.mean|x[{1,300,5,6,900,8,9,10,11,12,13,14,15,16}]:||weight_a|mean' in viewkey_store)

    ''' combined_codes views: Test that verify combined_codes views are calculated correctly '''
    def test_combined_codes_delimited_on_single_w(self):
        views = QuantipyViews(['counts'])
        x = 'q9'
        y = 'gender'
        self.setup_stack(
                        views=views,
                        x=x,
                        y=y,
                        weights='weight_a'
                        )

        net_view = ViewMapper()
        net_view.add_method('net',
                        QuantipyViews().frequency,
                        kwargs={
                                'axis': 'x',
                                'logic': [1, 2, 3, 8761]
                                })
        self.stack.add_link(data_keys='testing', x=x, y=y, views=net_view, weights='weight_a')
        self.assertTrue('x|f|x[{1,2,3,8761}]:||weight_a|net' in self.stack['testing']['no_filter'][x][y].keys())

        df = self.stack['testing']['no_filter'][x][y]['x|f|x[{1,2,3,8761}]:||weight_a|net'].dataframe

        results_nets = [[1673.71012585387, 2019.67696380492]]
        self.assertTrue(np.allclose(df.values, results_nets))

    ''' nps views: Test that verify nps views are calculated correctly '''
    def test_nps_single_on_delimited_no_w(self):
        from operator import add, sub
        views = QuantipyViews(['counts', 'cbase'])
        x = 'q1'
        y = 'q9'
        self.setup_stack(
                        views=views,
                        x=x,
                        y=y
                        )

        nps_view = ViewMapper()
        groups = [{'A': [1, 2, 3]}, {'B': [4]}, {'C': [5, 6]}]
        operation =  {'score': ('A', sub, 'C')}

        nps_view.add_method('nps',
                        QuantipyViews().frequency,
                        kwargs={
                                'rel_to': 'y',
                                'logic': groups,
                                'axis': 'x',
                                'calc': operation
                                })
        self.stack.add_link(data_keys='testing', x=x, y=y, views=nps_view)
        self.assertTrue('x|f.c:f|x[{1,2,3}],x[{4}],x[{5,6}],x[{1,2,3}-{5,6}]:|y||nps' in self.stack['testing']['no_filter'][x][y].keys())

        df = self.stack['testing']['no_filter'][x][y]['x|f.c:f|x[{1,2,3}],x[{4}],x[{5,6}],x[{1,2,3}-{5,6}]:|y||nps'].dataframe
        results_nps_groups_and_score = [[34.46666666666667, 34.6849656893325, 33.681765389082464, 33.210840606339, 35.694822888283376, 37.187039764359355, 38.60561914672216],
                                        [40.400000000000006, 39.30131004366812, 39.60511033681765, 40.652273771244836, 34.05994550408719, 34.75699558173785, 32.327436697884146],
                                        [7.766666666666666, 8.484092326887087, 8.362369337979095, 7.808911345888838, 9.536784741144414, 8.541973490427099, 8.047173083593478],
                                        [26.700000000000003, 26.200873362445414, 25.319396051103364, 25.401929260450164, 26.158038147138964, 28.64506627393225, 30.558446063128685]]

        self.assertTrue(np.allclose(df.values, results_nps_groups_and_score))

    ''' check cumulative sums'''
    def test_cumulative_sum_counts(self):
        views = QuantipyViews(['counts_cumsum'])
        x = 'q8'
        y = 'gender'
        self.setup_stack(
            views=views,
            x=x,
            y=y)
        df = self.stack['testing']['no_filter']['q8']['gender']['x|f.c:f|x++:|||counts_cumsum'].dataframe
        results = [[  473,   476],
                   [  585,   580],
                   [  878,   882],
                   [ 1385,  1345],
                   [ 2021,  1944],
                   [ 2186,  2062],
                   [ 2212,  2085]]
        self.assertTrue(np.allclose(df.values, results))

    def test_cumulative_sum_cpercent(self):
        views = QuantipyViews(['c%_cumsum'])
        x = 'q5'
        y = '@'
        self.setup_stack(
            views=views,
            x=x,
            y=y)

        df = self.stack['testing']['no_filter']['q5']['@']['x|f.c:f|x++:|y||c%_cumsum'].dataframe
        results = [[   4.39733495,   12.87704422,   44.34887947,   45.85099939,
                      75.79648698,   78.21926105,  100.        ],
                   [   8.29800121,   24.07026045,   58.21926105,   59.04300424,
                      79.80617807,   82.14415506,  100.        ],
                   [   7.41368867,   22.4954573 ,   55.22713507,   56.02665051,
                      76.74136887,   80.48455482,  100.        ],
                   [   6.66262871,   16.93519079,   38.691702  ,   39.5517868 ,
                      61.35675348,   69.04906118,  100.        ],
                   [  33.28891581,   46.94124773,   73.50696548,   73.91883707,
                      76.6444579 ,   81.07813446,  100.        ],
                   [   5.29376136,   13.50696548,   43.98546336,   45.24530588,
                      73.03452453,   75.59055118,  100.        ]]
        self.assertTrue(np.allclose(df.values, results))

    ''' "effective base (ebase) test" '''
    def test_ebase(self):
        views = QuantipyViews(['ebase'])
        x = 'gender'
        y = ['@', 'q8']
        self.setup_stack(views=views, x=x, y=y, weights='weight_a')
        total_l = self.stack['testing']['no_filter'][x]['@']
        cross_l = self.stack['testing']['no_filter'][x]['q8']
        vk = 'x|f|x:||weight_a|ebase'
        actual_total_ebase = total_l[vk].dataframe.round(1).values.tolist()
        actual_cross_ebase = cross_l[vk].dataframe.round(1).values.tolist()
        expected_total = [[5473.2]]
        expected_cross = [[641.0, 132.8, 417.5, 658.9, 823.5, 187.9, 35.1]]
        self.assertEqual(actual_total_ebase, expected_total)
        self.assertEqual(actual_cross_ebase, expected_cross)

    ''' "source" kwarg/descriptives(): check if swapped axis behaves correctly '''
    def test_source_kwarg_descriptives(self):
        views = QuantipyViews(['counts', 'cbase'])
        x = 'q2b'
        y = 'q2b'
        self.setup_stack(
            views=views,
            x=x,
            y=y,
            weights='weight_b'
            )

        foreign_stats = ViewMapper(
            template={
                'method': QuantipyViews().descriptives,
                'kwargs': {'iterators': {'stats': ['mean', 'stddev']}}
                })
        foreign_stats.add_method(name='foreign_stats',
                         kwargs={'source': 'weight_a',
                                 'axis': 'x'})

        self.stack.add_link(data_keys='testing', x=x, y=y,
                            views=foreign_stats, weights='weight_b')

        l = self.stack['testing']['no_filter'][x][y]
        mean = 'x|d.mean|weight_a:||weight_b|foreign_stats'
        stddev = 'x|d.stddev|weight_a:||weight_b|foreign_stats'
        mean_result = l[mean].dataframe.values.tolist()
        stddev_result = l[stddev].dataframe.values.tolist()

        foreign_means = [[1.4874813073223419,
                          1.4484446605625558,
                          1.57250526358392]]
        foreign_stddevs = [[0.922248166021751,
                            0.902477800807479,
                            0.9259579927078846]]

        self.assertEqual(mean_result, foreign_means)
        self.assertEqual(stddev_result, foreign_stddevs)


    def test_source_kwarg_descriptives_sigtest(self):
        views = QuantipyViews(['counts', 'cbase'])
        x = 'q2b'
        y = 'q2b'
        self.setup_stack(
            views=views,
            x=x,
            y=y,
            weights='weight_b'
            )

        foreign_stats = ViewMapper(
            template={
                'method': QuantipyViews().descriptives,
                'kwargs': {}
                })
        foreign_stats.add_method(name='foreign_stats',
                         kwargs={'source': 'weight_a',
                                 'axis': 'x'})

        self.stack.add_link(data_keys='testing', x=x, y=y,
                            views=foreign_stats, weights='weight_b')

        mean_sig = ViewMapper(
            template={
                'method': QuantipyViews().coltests,
                'kwargs': {
                    'metric': 'means',
                    'stack': self.stack,
                    'iterators': {
                        'level': [0.10]
                        }
                    }
                })
        mean_sig.add_method(name='source_sig_test',
                            kwargs={'text': 'sigtest source'})

        self.stack.add_link(data_keys='testing', x=x, y=y,
                            views=mean_sig, weights='weight_b')

        l = self.stack['testing']['no_filter'][x][y]
        mean_test = 'x|t.means.Dim.10|weight_a:||weight_b|source_sig_test'
        mean_test_result = l[mean_test].dataframe.replace(np.NaN, 'NONE')
        mean_test_result = mean_test_result.values.tolist()

        expected_result = [['NONE', 'NONE', '[2]']]
        self.assertEqual(mean_test_result, expected_result)

    ''' means_test views: Tests that tests of mean significance are yielding the correct results '''
    def test_means_test_level_5_weighted_all_codes(self):
        views = QuantipyViews(['counts', 'cbase'])
        x = 'q5_1'
        y = 'locality'
        self.setup_stack(
            views=views,
            x=x,
            y=y,
            weights='weight_a'
            )

        means = ViewMapper(
            template={
                'method': QuantipyViews().descriptives,
                'kwargs': {}
                })
        means.add_method(name='all',
                         kwargs={'text': '(all codes))',
                                 'axis': 'x'})
        self.stack.add_link(data_keys='testing', x=x, y=y,
                            views=means.subset('all'), weights='weight_a')

        mean_sig = ViewMapper(
            template={
                'method': QuantipyViews().coltests,
                'kwargs': {
                    'metric': 'means',
                    'stack': self.stack,
                    'iterators': {
                        'level': [0.05]
                        }
                    }
                })
        mean_sig.add_method(name='DIM_means_test',
                            kwargs={'text': 'SIG (means)'})

        self.stack.add_link(data_keys='testing', x=x, y=y,
                            views=mean_sig, weights='weight_a')
        view = self.stack['testing']['no_filter'][x][y]['x|t.means.Dim.05|x:||weight_a|DIM_means_test']
        meta = self.stack['testing']['no_filter'][x][y]['x|t.means.Dim.05|x:||weight_a|DIM_means_test'].meta()

        sig_result = [['NONE', 'NONE', 'NONE', '[1, 2]', '[1, 2, 3]']]
        meta_siglevel = 0.05

        self.assertEqual(view.dataframe.replace(np.NaN, 'NONE').values.tolist(),
                         sig_result)
        self.assertEqual(view.is_meanstest(), meta_siglevel)

    def test_means_test_level_20_weighted_no_missings(self):
        views = QuantipyViews(['counts', 'cbase'])
        x = 'q5_1'
        y = 'locality'
        self.setup_stack(
            views=views,
            x=x,
            y=y,
            weights='weight_a'
            )

        means = ViewMapper(
            template={
                'method': QuantipyViews().descriptives,
                'kwargs': {}
                })
        means.add_method(name='excl_9798',
                         kwargs={'text': '(no missings))',
                                 'exclude': [97, 98],
                                 'axis': 'x'})
        self.stack.add_link(data_keys='testing', x=x, y=y,
                            views=means.subset('excl_9798'), weights='weight_a')

        mean_sig = ViewMapper(
            template={
                'method': QuantipyViews().coltests,
                'kwargs': {
                    'metric': 'means',
                    'stack': self.stack,
                    'iterators': {
                        'level': [0.20]
                        }
                    }
                })
        mean_sig.add_method(name='DIM_means_test',
                            kwargs={'text': 'SIG (means)'})

        self.stack.add_link(data_keys='testing', x=x, y=y,
                            views=mean_sig, weights='weight_a')

        view = self.stack['testing']['no_filter'][x][y]['x|t.means.Dim.20|x[{1,2,3,4,5}]:||weight_a|DIM_means_test']
        meta = self.stack['testing']['no_filter'][x][y]['x|t.means.Dim.20|x[{1,2,3,4,5}]:||weight_a|DIM_means_test'].meta()

        sig_result = [['NONE', '[1, 3]', 'NONE', 'NONE', 'NONE']]
        meta_siglevel = 0.20

        self.assertEqual(view.dataframe.replace(np.NaN, 'NONE').values.tolist(),
                         sig_result)
        self.assertEqual(view.is_meanstest(), meta_siglevel)

    def test_means_test_level_10_unweighted_ovlp_no_missings(self):
        views = QuantipyViews(['counts', 'cbase'])
        x = 'q5_1'
        y = 'q3'
        self.setup_stack(
            views=views,
            x=x,
            y=y
            )

        means = ViewMapper(
            template={
                'method': QuantipyViews().descriptives,
                'kwargs': {}
                })
        means.add_method(name='excl_9798',
                         kwargs={'text': '(no missings))',
                                 'exclude': [97, 98],
                                 'axis': 'x'})
        self.stack.add_link(data_keys='testing', x=x, y=y,
                            views=means.subset('excl_9798'))

        mean_sig = ViewMapper(
            template={
                'method': QuantipyViews().coltests,
                'kwargs': {
                    'metric': 'means',
                    'stack': self.stack,
                    'iterators': {
                        'level': ['low']
                        }
                    }
                })
        mean_sig.add_method(name='DIM_means_test',
                            kwargs={'text': 'SIG (means)'})

        self.stack.add_link(data_keys='testing', x=x, y=y,
                            views=mean_sig)

        view = self.stack['testing']['no_filter'][x][y]['x|t.means.Dim.10|x[{1,2,3,4,5}]:|||DIM_means_test']
        meta = self.stack['testing']['no_filter'][x][y]['x|t.means.Dim.10|x[{1,2,3,4,5}]:|||DIM_means_test'].meta()

        sig_result = [['[5]', '[5]', '[1, 2, 5, 6, 7, 97]', '[5]', 'NONE', '[5]', '[5]', '[5]', 'NONE']]
        meta_siglevel = 0.10

        self.assertEqual(view.dataframe.replace(np.NaN, 'NONE').values.tolist(),
                         sig_result)
        self.assertEqual(view.is_meanstest(), meta_siglevel)


    def test_means_test_level_high_askia_unweighted_all_codes(self):
        views = QuantipyViews(['counts', 'cbase'])
        x = 'q5_1'
        y = 'locality'
        self.setup_stack(
            views=views,
            x=x,
            y=y,
            weights=None
            )

        means = ViewMapper(
            template={
                'method': QuantipyViews().descriptives,
                'kwargs': {}
                })
        means.add_method(name='all',
                         kwargs={'text': '(all codes))',
                                  'axis': 'x'})
        self.stack.add_link(data_keys='testing', x=x, y=y,
                            views=means.subset('all'), weights=None)

        mean_sig = ViewMapper(
            template={
                'method': QuantipyViews().coltests,
                'kwargs': {
                    'metric': 'means',
                    'stack': self.stack,
                    'iterators': {
                        'level': ['high']
                        }
                    }
                })
        mean_sig.add_method(name='askia_means_test',
                            kwargs={'text': 'SIG (means)',
                                    'mimic': 'askia'})

        self.stack.add_link(data_keys='testing', x=x, y=y,
                            views=mean_sig, weights=None)

        view = self.stack['testing']['no_filter'][x][y]['x|t.means.askia.01|x:|||askia_means_test']
        meta = self.stack['testing']['no_filter'][x][y]['x|t.means.askia.01|x:|||askia_means_test'].meta()

        sig_result = [['NONE', '[1]', '[1]', '[1, 2, 3]', '[1, 2, 3, 4]']]
        meta_siglevel = 0.01

        self.assertEqual(view.dataframe.replace(np.NaN, 'NONE').values.tolist(),
                         sig_result)
        self.assertEqual(view.is_meanstest(), meta_siglevel)

    ''' props_test views: Tests that tests of proportion significance are yielding the correct results '''
    def test_props_test_level_20_weighted(self):
        views = QuantipyViews(['counts', 'cbase'])
        x = 'q1'
        y = 'locality'
        self.setup_stack(
            views=views,
            x=x,
            y=y,
            weights='weight_a'
            )

        prop_sig = ViewMapper(
            template={
                'method': QuantipyViews().coltests,
                'kwargs': {
                    'rel_to': 'y',
                    'metric': 'props',
                    'stack': self.stack,
                    'iterators': {
                        'level': [0.20]
                        }
                    }
                })

        prop_sig.add_method(name='DIM_props_test',
                            kwargs={'text': 'sig without overlap'})

        self.stack.add_link(data_keys='testing', x=x, y=y,
                            views=prop_sig, weights='weight_a')

        view = self.stack['testing']['no_filter'][x][y]['x|t.props.Dim.20|:|y|weight_a|DIM_props_test']
        meta = self.stack['testing']['no_filter'][x][y]['x|t.props.Dim.20|:|y|weight_a|DIM_props_test'].meta()

        sig_result = [['[2, 3, 4, 5]', '[4]', 'NONE', 'NONE', 'NONE'],
                      ['NONE', 'NONE', 'NONE', 'NONE', 'NONE'],
                      ['NONE', '[1, 4]', 'NONE', 'NONE', '[4]'],
                      ['[2, 5]', 'NONE', '[5]', '[2, 5]', 'NONE'],
                      ['NONE', 'NONE', 'NONE', 'NONE', 'NONE'],
                      ['NONE', '[1]', '[1]', 'NONE', 'NONE'],
                      ['NONE', 'NONE', 'NONE', '[1]', '[1]'],
                      ['NONE', 'NONE', 'NONE', 'NONE', 'NONE'],
                      ['NONE', 'NONE', 'NONE', 'NONE', 'NONE'],
                      ['NONE', 'NONE', 'NONE', 'NONE', 'NONE'],
                      ['[2, 3]', 'NONE', 'NONE', 'NONE', '[2, 3]'],
                      ['[3]', '[1, 3]', 'NONE', '[3]', '[3]']]



        meta_agg_text = 'sig without overlap'
        meta_siglevel = 0.2

        self.assertEqual(view.dataframe.replace(np.NaN, 'NONE').values.tolist(), sig_result)
        self.assertEqual(meta['agg']['text'], meta_agg_text)
        self.assertEqual(view.is_propstest(), meta_siglevel)

    def test_props_test_level_5_ovlp_unweighted(self):
        views = QuantipyViews(['counts', 'cbase'])
        x = 'q5_1'
        y = 'q3'
        self.setup_stack(
            views=views,
            x=x,
            y=y
            )

        prop_sig = ViewMapper(
            template={
                'method': QuantipyViews().coltests,
                'kwargs': {
                    'rel_to': 'y',
                    'stack': self.stack,
                    'iterators': {
                        'level': ['mid']
                        }
                    }
                })

        prop_sig.add_method(name='DIM_props_test',
                            kwargs={'text': 'SIG (props)'})

        self.stack.add_link(data_keys='testing', x=x, y=y,
                            views=prop_sig, weights=None)

        view = self.stack['testing']['no_filter'][x][y]['x|t.props.Dim.05|:|y||DIM_props_test']
        meta = self.stack['testing']['no_filter'][x][y]['x|t.props.Dim.05|:|y||DIM_props_test'].meta()

        sig_result = [['[2, 3]', 'NONE', 'NONE', 'NONE', '[1, 2, 3, 4, 6, 7, 8, 97]', 'NONE', '[2, 3]', '[2, 3, 97]', 'NONE'],
                      ['[3, 97]', '[1, 3, 97]', '[97]', '[3, 97]', '[1, 2, 3, 4, 7, 97]', '[1, 2, 3, 4, 7, 97]', '[1, 3, 97]', '[3, 97]', 'NONE'],
                      ['[2, 3, 4, 8, 97]', '[8, 97]', '[2, 4, 8, 97]', '[97]', '[97]', '[97]', '[2, 4, 8, 97]', '[97]', 'NONE'],
                      ['[3]', '[3]', 'NONE', '[1, 2, 3, 5, 6, 7, 97]', 'NONE', 'NONE', 'NONE', '[1, 2, 3, 4, 5, 6, 7, 97]', '[1, 2, 3, 7]'],
                      ['[97]', '[97]', '[97]', '[97]', '[97]', '[97]', '[97]', '[97]', 'NONE'],
                      ['NONE', '[1, 5, 7]', '[1, 5, 7]', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE', '[1, 2, 3, 4, 5, 6, 7, 8]'],
                      ['[5, 6]', '[1, 5, 6, 7, 8]', '[1, 2, 4, 5, 6, 7, 8]', '[6]', 'NONE', 'NONE', '[6]', 'NONE', '[1, 2, 3, 4, 5, 6, 7, 8]']]

        meta_agg_text = 'SIG (props)'
        meta_siglevel = 0.05

        self.assertEqual(view.dataframe.replace(np.NaN, 'NONE').values.tolist(), sig_result)
        self.assertEqual(meta['agg']['text'], meta_agg_text)
        self.assertEqual(view.is_propstest(), meta_siglevel)

    def test_props_test_level_1_ovlp_weighted(self):
        views = QuantipyViews(['counts', 'cbase'])
        x = 'q9'
        y = 'q8'
        self.setup_stack(
            views=views,
            x=x,
            y=y,
            weights='weight_a'
            )

        prop_sig = ViewMapper(
            template={
                'method': QuantipyViews().coltests,
                'kwargs': {
                    'rel_to': 'y',
                    'metric': 'props',
                    'stack': self.stack,
                    'iterators': {
                        'level': [0.01]
                        }
                    }
                })

        prop_sig.add_method(name='DIM_props_test',
                            kwargs={'text': 'SIG (props, strict)'})

        self.stack.add_link(data_keys='testing', x=x, y=y,
                            views=prop_sig, weights='weight_a')

        view = self.stack['testing']['no_filter'][x][y]['x|t.props.Dim.01|:|y|weight_a|DIM_props_test']
        meta = self.stack['testing']['no_filter'][x][y]['x|t.props.Dim.01|:|y|weight_a|DIM_props_test'].meta()

        sig_result = [['NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE'],
                      ['NONE', 'NONE', '[1, 96]', '[96]', '[96]', 'NONE', 'NONE'],
                      ['NONE', 'NONE', '[4]', 'NONE', 'NONE', 'NONE', 'NONE'],
                      ['NONE', 'NONE', '[98]', 'NONE', 'NONE', 'NONE', 'NONE'],
                      ['NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE'],
                      ['NONE', '[3]', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE'],
                      ['NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE']]


        meta_agg_text = 'SIG (props, strict)'
        meta_siglevel = 0.01

        self.assertEqual(view.dataframe.replace(np.NaN, 'NONE').values.tolist(), sig_result)
        self.assertEqual(meta['agg']['text'], meta_agg_text)
        self.assertEqual(view.is_propstest(), meta_siglevel)

    def test_props_means_tests_incl_total(self):
        views = ['counts', 'mean']
        x, y = 'q7_1', 'q8'
        self.setup_stack(
            views=views,
            x=x,
            y=y,
            weights='weight_a'
            )

        tests = ViewMapper()
        tests.make_template('coltests', iterators={'metric': ['props', 'means']})
        tests.add_method(name='total_tests', kwargs={'test_total': True})
        self.stack.add_link(data_keys='testing', x=x, y=y,
                            views=tests, weights='weight_a')

        link = self.stack['testing']['no_filter'][x][y]
        props_view = link['x|t.props.Dim.10+@|:||weight_a|total_tests']
        means_view = link['x|t.means.Dim.10+@|x:||weight_a|total_tests']

        props_results = [['NONE', "['@L', 1, 3, 4, 5, 96]", 'NONE', '[5]', "['@H']", 'NONE', 'NONE'],
                         ['NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE'],
                         ['NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE'],
                         ['NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE'],
                         ['[4]', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE'],
                         ['NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE'],
                         ['NONE', 'NONE', 'NONE', 'NONE', '[96]', 'NONE', 'NONE'],
                         ['[5]', '[5]', 'NONE', '[5]', "['@H']", "['@L', 5]", "['@L', 1, 3, 4, 5]"],
                         ['NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE']]

        means_result = [['NONE', 'NONE', 'NONE', 'NONE', 'NONE', "['@L', 2]", 'NONE']]

        self.assertEqual(props_view.dataframe.replace(np.NaN, 'NONE').values.tolist(),
                         props_results)
        self.assertEqual(means_view.dataframe.replace(np.NaN, 'NONE').values.tolist(),
                         means_result)

    def test_means_tests_code_exclusion_incl_total(self):
        views = ['counts']
        x, y = 'q7_1', 'q8'
        self.setup_stack(
            views=views,
            x=x,
            y=y,
            weights='weight_a'
            )

        mean = ViewMapper()
        mean.make_template('descriptives', iterators={'stats': ['mean']})
        mean.add_method(name='excl. 6,8', kwargs={'exclude': [6, 8], 'axis': 'x'})
        self.stack.add_link(data_keys='testing', x=x, y=y,
                            views=mean, weights=None)

        tests = ViewMapper()
        tests.make_template('coltests', iterators={'metric': ['means']})
        tests.add_method(name='total_tests', kwargs={'test_total': True})
        self.stack.add_link(data_keys='testing', x=x, y=y,
                            views=tests, weights=None)

        link = self.stack['testing']['no_filter'][x][y]
        means_view = link['x|t.means.Dim.10+@|x[{1,2,3,4,5,7,9}]:|||total_tests']

        means_result = [["['@L', 98]", '[98]', '[98]', "['@L', 98]", '[98]', '[98]', "['@H']"]]
        self.assertEqual(means_view.dataframe.replace(np.NaN, 'NONE').values.tolist(),
                         means_result)

    def test_means_tests_code_exclusion_base_flags_incl_total(self):
        views = ['counts']
        x, y = 'q7_1', 'q8'
        self.setup_stack(
            views=views,
            x=x,
            y=y,
            weights='weight_a'
            )

        mean = ViewMapper()
        mean.make_template('descriptives', iterators={'stats': ['mean']})
        mean.add_method(name='excl. 6,8', kwargs={'exclude': [6, 8],
                                                  'axis': 'x'})
        self.stack.add_link(data_keys='testing', x=x, y=y,
                            views=mean, weights=None)

        tests = ViewMapper()
        tests.make_template('coltests', iterators={'metric': ['means']})
        tests.add_method(name='total_tests_flags',
                         kwargs={'test_total': True,
                                 'flag_bases': [30, 100]})
        self.stack.add_link(data_keys='testing', x=x, y=y,
                            views=tests, weights=None)

        link = self.stack['testing']['no_filter'][x][y]
        means_view = link['x|t.means.Dim.10+@|x[{1,2,3,4,5,7,9}]:|||total_tests_flags']

        means_result = [["['@L']", '*', 'NONE', "['@L']", 'NONE', '*', '**']]
        self.assertEqual(means_view.dataframe.replace(np.NaN, 'NONE').values.tolist(),
                         means_result)

    def test_props_blocknet_calc_incl_total(self):
        from operator import add, sub
        views = ['counts']
        x, y = 'q7_1', 'q8'
        self.setup_stack(
            views=views,
            x=x,
            y=y,
            weights='weight_a'
            )
        nets = ViewMapper()
        nets.make_template('frequency')
        nets_def = [{'Z': [1, 2, 3], 'expand': 'after',
                     'text': {'en-GB': 'some text1'}},
                    {'A': [4, 5],
                     'text': {'en-GB': 'some text2'}},
                    {'F': [6, 7, 8], 'expand': 'before',
                     'text': {'en-GB': 'some text3'}}]
        calc = {'my_calc': ('Z', sub, 'F')}
        nets.add_method(name='blocknet',
                        kwargs={'logic': nets_def,
                                'axis': 'x',
                                'complete': True,
                                'calc': calc})
        self.stack.add_link(data_keys='testing', x=x, y=y,
                            views=nets, weights='weight_a')

        tests = ViewMapper()
        tests.make_template('coltests', iterators={'metric': ['props']})
        tests.add_method(name='total_tests_blocks',
                         kwargs={'test_total': True})
        self.stack.add_link(data_keys='testing', x=x, y=y,
                            views=tests, weights='weight_a')

        link = self.stack['testing']['no_filter'][x][y]
        nets_view = link['x|t.props.Dim.10+@|x[{1,2,3}+],x[{4,5}],x[+{6,7,8}],x[{1,2,3}-{6,7,8}]*:||weight_a|total_tests_blocks']

        nets_result = [["['@H']", 'NONE', '[1]', 'NONE', 'NONE', 'NONE', 'NONE'],
                       ['NONE', "['@L', 1, 3, 4, 5, 96]", 'NONE', '[5]', "['@H']", 'NONE', 'NONE'],
                       ['NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE'],
                       ['NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE'],
                       ['NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE'],
                       ['[4]', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE'],
                       ['[4]', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE'],
                       ['NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE'],
                       ['NONE', 'NONE', 'NONE', 'NONE', '[96]', 'NONE', 'NONE'],
                       ['[5]', '[5]', 'NONE', '[5]', "['@H']", "['@L', 5]", "['@L', 1, 3, 4, 5]"],
                       ['NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE'],
                       ['NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE'],
                       ['NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE']]
        self.assertEqual(nets_view.dataframe.replace(np.NaN, 'NONE').values.tolist(),
                         nets_result)

    def test_props_changed_meta_nets_incl_total(self):
        from copy import deepcopy
        x = 'q5_1'
        y = 'q8'
        self.setup_stack(
            views=None,
            x=x,
            y=y,
            weights=None
            )
        # re-ordering meta in a random fashion to test robustness
        meta = deepcopy(self.stack['testing'].meta)
        meta = emulate_meta(meta, meta)
        values = meta['columns']['q5_1']['values']
        slicer = [98, 1, 2, 5, 4, 3, 97]
        meta['columns']['q5_1']['values'] = self.slice_values_meta(values, slicer)
        values = meta['columns']['q8']['values']
        slicer = [98, 96, 5, 1, 2, 3, 4]
        meta['columns']['q8']['values'] = self.slice_values_meta(values, slicer)
        self.stack['testing'].meta = meta
        self.stack.add_link(x='q5_1', y='q8', views=QuantipyViews(['counts']))

        nets = ViewMapper()
        nets.make_template('frequency')
        nets_def = [{'Z': [98, 97], 'expand': 'after',
                                    'text': {'en-GB': 'some text1'}},
                    {'A': [4, 1],  'expand': 'before',
                                   'text': {'en-GB': 'some text2'}}]

        nets.add_method(name='blocknet',
                        kwargs={'logic': nets_def,
                                'axis': 'x',
                                'complete': True})
        self.stack.add_link(x='q5_1', y='q8', views=nets, weights=None)

        tests = ViewMapper()
        tests.make_template('coltests')
        tests.add_method(name='tests',
                         kwargs={'level': 0.1,
                                'test_total': True})
        self.stack.add_link(x='q5_1', y='q8', views=tests, weights=None)

        link = self.stack['testing']['no_filter']['q5_1']['q8']
        nets_test_view = link['x|t.props.Dim.10+@|x[{98,97}+],x[+{4,1}]*:|||tests']
        test_resuts = [['[5, 1, 2, 3, 4]', "['@H', 2]", "['@H']", "['@H']", "['@H']", "['@H']", "['@H']"],
                       ['[5, 1, 2, 3, 4]', "['@H', 2]", "['@H']", "['@H']", "['@H']", "['@H']", "['@H']"],
                       ['NONE', '[5]', "['@H']", "['@H', 5]", "['@H']", '[5]', "['@H']"],
                       ['NONE', 'NONE', "['@L']", "['@L']", "['@L', 96, 3, 4]", 'NONE', 'NONE'],
                       ['NONE', "['@H']", '[96]', '[96]', "['@H']", '[96]', '[96, 2]'],
                       ['NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE', '[96, 1, 3]'],
                       ['NONE', 'NONE', 'NONE', "['@H']", "['@L', 96, 5, 1, 3, 4]", '[1]', '[1]'],
                       ['NONE', "['@H']", '[1]', "['@H']", '[96, 5, 1, 3]', 'NONE', '[96, 1]'],
                       ['NONE', "['@L', 5, 4]", "['@L']", "['@L', 4]", "['@L']", "['@L']", "['@L']"]]
        self.assertEqual(nets_test_view.dataframe.replace(np.NaN, 'NONE').values.tolist(),
                         test_resuts)

        net_agg = link['x|f|x[{98,97}+],x[+{4,1}]*:|||blocknet']
        net_agg_idx = net_agg.dataframe.index.get_level_values(1).tolist()
        net_agg_cols = net_agg.dataframe.columns.get_level_values(1).tolist()
        net_test_idx = nets_test_view.dataframe.index.get_level_values(1).tolist()
        net_test_cols = nets_test_view.dataframe.columns.get_level_values(1).tolist()

        self.assertEqual(net_agg_idx, ['Z', 98, 97, 2, 5, 4, 1, 'A', 3])
        self.assertEqual(net_agg_cols, [98, 96, 5, 1, 2, 3, 4])
        self.assertEqual(net_agg_idx, net_test_idx)
        self.assertEqual(net_agg_cols, net_test_cols)

    def test_props_test_level_low_askia_weighted(self):
        views = QuantipyViews(['counts', 'cbase'])
        x = 'q9'
        y = 'q8'
        self.setup_stack(
            views=views,
            x=x,
            y=y,
            weights='weight_a'
            )

        prop_sig = ViewMapper(
            template={
                'method': QuantipyViews().coltests,
                'kwargs': {
                    'rel_to': 'y',
                    'stack': self.stack,
                    'iterators': {
                        'level': ['low']
                        }
                    }
                })

        prop_sig.add_method(name='askia_props_test',
                            kwargs={'text': 'SIG (props, askia_low)',
                                    'mimic': 'askia'})

        self.stack.add_link(data_keys='testing', x=x, y=y,
                            views=prop_sig, weights='weight_a')

        view = self.stack['testing']['no_filter'][x][y]['x|t.props.askia.10|:|y|weight_a|askia_props_test']
        meta = self.stack['testing']['no_filter'][x][y]['x|t.props.askia.10|:|y|weight_a|askia_props_test'].meta()

        sig_result = [['[98]', '[98]', '[2, 5, 98]', '[2, 98]', '[98]', '[98]', 'NONE'],
                      ['[96, 98]', '[96, 98]', '[1, 96, 98]', '[96, 98]', '[96, 98]', 'NONE', 'NONE'],
                      ['[98]', '[98]', '[1, 2, 4, 5, 96, 98]', '[98]', '[98]', '[98]', 'NONE'],
                      ['[98]', '[98]', '[4, 5, 98]', '[98]', '[98]', '[98]', 'NONE'],
                      ['[2]', 'NONE', '[2, 4]', 'NONE', '[2]', '[2, 4]', 'NONE'],
                      ['NONE', '[1, 3, 4, 5, 96]', 'NONE', 'NONE', '[3, 4]', 'NONE', 'NONE'],
                      ['NONE', 'NONE', 'NONE', 'NONE', 'NONE', '[2, 3, 5]', '[1, 2, 3, 4, 5]']]


        meta_agg_text = 'SIG (props, askia_low)'
        meta_siglevel = 0.10

        self.assertEqual(view.dataframe.replace(np.NaN, 'NONE').values.tolist(), sig_result)
        self.assertEqual(meta['agg']['text'], meta_agg_text)
        self.assertEqual(view.is_propstest(), meta_siglevel)


    def setup_stack(
        self,
        key='testing',
        data=io.load_csv(os.path.dirname(os.path.abspath(__file__))+'/Example Data (A).csv'),
        meta=io.load_json(os.path.dirname(os.path.abspath(__file__))+'/Example Data (A).json'),
        filters=None,
        x=None,
        y=None,
        views=None,
        weights=None):
        self.stack = Stack('Test')
        self.stack.add_data(data_key=key, data=data, meta=meta)
        self.stack.add_link(data_keys=key, x=x, y=y, views=views, weights=weights)

    def slice_values_meta(self, values, slicer):
        """
        Return a slice of the given values meta using slicer.
        """
        new_values = [
            value
            for i in slicer
            for value in values
            if value['value']==i]
        return new_values