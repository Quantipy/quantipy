
import pytest
import os
import time
import pandas as pd
import numpy as np
import quantipy as qp
from itertools import count, izip

from quantipy.sandbox.sandbox import Chain

from pandas.util.testing import assert_frame_equal, assert_index_equal

import chain_fixtures as fixture

# CONSTANTS ## fixtures?
# -----------------------------------------------------------------------------
PATH_DATA = './tests/'
NAME_PROJ = 'Example Data (A)'
NAME_META = 'Example Data (A).json'
NAME_DATA = 'Example Data (A).csv'
PATH_META = os.path.join(PATH_DATA, NAME_META)
PATH_DATA = os.path.join(PATH_DATA, NAME_DATA)

DATA_KEY = 'x'
FILTER_KEY = 'no_filter'
# -----------------------------------------------------------------------------


@pytest.fixture(scope='module')
def dataset():
    _dataset = qp.DataSet(NAME_PROJ, dimensions_comp=False)
    _dataset.read_quantipy(PATH_META, PATH_DATA)
    yield _dataset.split()
    del _dataset

@pytest.fixture(scope='class')
def stack(dataset):
    meta, data = dataset
    _stack = qp.Stack(NAME_PROJ,
                      add_data={DATA_KEY: {'meta': meta,
                                           'data': data.head(250)}})
    yield _stack
    del _stack

@pytest.fixture(scope='class')
def basic_chain(stack):
    _chain = Chain(stack, name='chain')
    yield _chain
    del _chain

@pytest.fixture(scope='function')
def complex_chain(stack, x_keys, y_keys, views, view_keys, orient, incl_tests,
                  incl_sum):
    # Custom view methods...
    # ---SIG
    sigtest_props_l80_total = qp.ViewMapper().make_template('coltests')
    view_name = 't_p_80'
    options = {'level': 0.8, 'metric': 'props', 'test_total': True}
    sigtest_props_l80_total.add_method(view_name, kwargs=options)

    sigtest_means_l80_total = qp.ViewMapper().make_template('coltests')
    view_name = 't_m_80'
    options = {'level': 0.8, 'metric': 'means', 'test_total': True}
    sigtest_means_l80_total.add_method(view_name, kwargs=options)

    sig_views = [sigtest_props_l80_total, sigtest_means_l80_total]

    # ------------------------------------------------------------------------

    stack.add_link(x=x_keys, y=y_keys, views=views)

    if incl_tests:
        for v in sig_views:
            stack.add_link(x=x_keys, y=y_keys, views=v)

    if incl_sum:
        stack.add_link(x=x_keys, y=y_keys, views=['counts_sum', 'c%_sum'])

    _chain = Chain(stack, name='chain')
    _chains = _chain.get(data_key='x',
                         filter_key='no_filter',
                         x_keys=x_keys,
                         y_keys=y_keys,
                         views=view_keys,
                         orient=orient)

    if isinstance(_chains, Chain): # single chain
        _chains = [_chains]
    return _chains

@pytest.fixture(scope='function')
def multi_index(tuples):
    names = ['Question', 'Values'] * (len(tuples[0]) / 2)
    _index = pd.MultiIndex.from_tuples(tuples, names=names)
    return _index

@pytest.fixture(scope='function')
def frame(values, index, columns):
    _frame = pd.DataFrame(values, index=index, columns=columns)
    return _frame

class TestChainConstructor:
    def test_init(self, basic_chain):
        assert basic_chain.name == 'chain'
        assert isinstance(basic_chain.stack, qp.Stack)

    def test_str(self, basic_chain):
        assert str(basic_chain) == fixture.BASIC_CHAIN_STR

    def test_repr(self, basic_chain):
        assert repr(basic_chain) == str(basic_chain)

    def test_len(self, basic_chain):
        assert len(basic_chain) == 0

class TestChainExceptions:
    def test_get_non_existent_columns(self, basic_chain):
        expected = "Expecting ValueError"
        with pytest.raises(ValueError, message=expected) as excinfo:
            basic_chain.get(data_key=DATA_KEY,
                            filter_key=FILTER_KEY,
                            x_keys=['erdbeer', 'bananana'],
                            y_keys=['@'],
                            views=['cbase', 'counts', 'c%', 'mean', 'median'],
                            orient='x')
        assert excinfo.match(r'.* "erdbeer", "bananana" .*')

    @pytest.mark.xfail(raises=NotImplementedError, reason="Method not complete")
    def test_bank(self, basic_chain):
        basic_chain.bank(to_bank=None)


@pytest.yield_fixture(
    scope='class',
    params=[
        (['q5_1'], ['@', 'q4'], fixture.X1),
        (['q5_1'], ['@', 'q4 > gender'], fixture.X2),
        (['q5_1'], ['@', 'q4 > gender > Wave'], fixture.X3),
        (['q5_1'], ['@', 'q4 > gender > Wave', 'q5_1', 'q4 > gender'], fixture.X4),
    ]
)
def params_getx(request):
    return request.param

class TestChainGet:
    _VIEWS = ('cbase', 'counts', 'c%', 'mean', 'median', 'c%_sum')

    _VIEW_KEYS = ('x|f|x:|||cbase', 'x|f|:|||counts', 'x|d.mean|x:|||mean',
                  'x|d.median|x:|||median', 'x|f.c:f|x:|||counts_sum')

    _VIEW_SIG_KEYS = ['x|f|x:|||cbase',
                      ('x|f|:|y||c%', 'x|t.props.Dim.80+@|:|||t_p_80'),
                      ('x|d.mean|x:|||mean', 'x|t.means.Dim.80+@|x:|||t_m_80'),
                       'x|f.c:f|x:|y||c%_sum']

    def test_get_x_orientation(self, stack, params_getx):
        x, y, expected = params_getx

        chains = complex_chain(stack, x, y, self._VIEWS, self._VIEW_KEYS, 'x',
                               incl_tests=False, incl_sum=False)

        for chain, args in izip(chains, expected):

            values, index, columns, pindex, pcolumns, chain_str = args

            expected_dataframe = frame(values,
                                       multi_index(index),
                                       multi_index(columns))
            painted_index = multi_index(pindex)
            painted_columns = multi_index(pcolumns)

            ### Test Chain.dataframe is Chain._frame
            assert chain.dataframe is chain._frame

            ### Test Chain attributes
            assert chain.orientation is 'x'

            ### Test Chain.get
            assert_frame_equal(chain.dataframe, expected_dataframe)

            ### Test Chain.paint
            chain.paint()
            assert_index_equal(chain.dataframe.index, painted_index)
            assert_index_equal(chain.dataframe.columns, painted_columns)

            ### Test Chain.toggle_labels
            chain.toggle_labels()
            assert_frame_equal(chain.dataframe, expected_dataframe)
            chain.toggle_labels()
            assert_index_equal(chain.dataframe.index, painted_index)
            assert_index_equal(chain.dataframe.columns, painted_columns)

            ### Test Chain str/ len
            assert str(chain) == chain_str

            ### Test Contents
            assert chain.contents == fixture.CONTENTS

    def test_sig_transformation_simple(self, stack):
        x, y = 'q5_1', ['@', 'gender', 'q4']
        chains = complex_chain(stack, x, y, self._VIEWS, self._VIEW_SIG_KEYS,
                               'x', incl_tests=True, incl_sum=True)
        chain_df = chains[0].paint().dataframe.replace(np.NaN, 'None')
        actual = pd.DataFrame(values.values.tolist())
        expected = pd.DataFrame(fixture.X5_SIG_SIMPLE[0])
        assert_frame_equal(expected, actual)




    def test_sig_transformation_large(self, stack):
        pass

# TODO: test_get_y_orientation
