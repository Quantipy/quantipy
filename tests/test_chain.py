
import pytest
import os
import time
import pandas as pd
import quantipy as qp
from itertools import count

from quantipy.sandbox.sandbox import Chain

from pandas.util.testing import assert_frame_equal, assert_index_equal

import chain_fixtures as fixtures

# PATHS
# -----------------------------------------------------------------------------
PATH_DATA = './tests/'
NAME_PROJ = 'Example Data (A)'
NAME_META = 'Example Data (A).json'
NAME_DATA = 'Example Data (A).csv'
PATH_META = os.path.join(PATH_DATA, NAME_META)
PATH_DATA = os.path.join(PATH_DATA, NAME_DATA)
# -----------------------------------------------------------------------------


@pytest.fixture(scope='session', autouse=True)
def dataset():
    ds = qp.DataSet(NAME_PROJ, dimensions_comp=False)
    ds.read_quantipy(PATH_META, PATH_DATA)
    meta, data = ds.split()
    return meta, data.head(250)

@pytest.fixture(scope='class', autouse=True)
def stack(request, dataset):
    meta, data = dataset
    obj = qp.Stack(NAME_PROJ, add_data={'#': {'meta': meta, 'data': data}})

    @request.addfinalizer
    def teardown():
        obj.pop('#')

    return obj

@pytest.fixture()
def chain(stack):
    return Chain(stack, name='chain')

class TestChainConstructor:

    def test_init(self, chain):
        assert chain.name == 'chain'
        assert isinstance(chain.stack, qp.Stack)

    def test_str(self, chain):
        assert str(chain) == fixtures.BASIC_CHAIN_STR

    def test_repr(self, chain):
        assert repr(chain) == str(chain)

    def test_len(self, chain):
        assert len(chain) == 0

class TestChainExceptions:

    def test_get_non_existent_columns(self, chain):
        expected = "Expecting ValueError"
        with pytest.raises(ValueError, message=expected) as excinfo:
            chain.get(data_key='#',
                      filter_key='no_filter',
                      x_keys=['erdbeer', 'bananana'],
                      y_keys=['@'],
                      views=['cbase', 'counts', 'c%', 'mean', 'median'],
                      orient='x')
        assert excinfo.match(r'.* "erdbeer", "bananana" .*')

    @pytest.mark.xfail(raises=NotImplementedError,
                       reason="Method not complete")
    def test_bank(self, chain):
        chain.bank(to_bank=None)


def set_up_chains(stack, x_keys, y_keys, views, view_keys, orient):
    stack.add_link(x=x_keys, y=y_keys, views=views)
    chain = Chain(stack, name='chain')
    chains = chain.get(data_key='#', filter_key='no_filter',
                       x_keys=x_keys, y_keys=y_keys,
                       views=view_keys, orient=orient)
    if isinstance(chains, Chain):
        return [chains]
    return chains

def create_index(tuples):
    names = ['Question', 'Values'] * (len(tuples[0]) / 2)
    index = pd.MultiIndex.from_tuples(tuples, names=names)
    return index

def create_frame(values, index, columns):
    frame = pd.DataFrame(values,
                         index=create_index(index),
                         columns=create_index(columns))
    return frame

@pytest.yield_fixture(
    scope='class',
    params=[
        (['q5_1'], ['@', 'q4'], fixtures.EXPECTED_X_BASIC),
        (['q5_1'], ['@', 'q4 > gender'], fixtures.EXPECTED_X_NEST_1),
        (['q5_1'], ['@', 'q4 > gender > Wave'], fixtures.EXPECTED_X_NEST_2),
        (['q5_1'], ['@', 'q4 > gender > Wave', 'q5_1', 'q4 > gender'], fixtures.EXPECTED_X_NEST_3),
    ]
)
def params_getx(request):
    return request.param

class TestChainGetX:

    views = ['cbase', 'counts', 'c%', 'mean', 'median']

    view_keys = ['x|f|x:|||cbase', 'x|f|:|||counts', 'x|d.mean|x:|||mean',
                 'x|d.median|x:|||median', 'x|f.c:f|x:|||counts_sum']

    # TODO: add sig testing

    def test_get_x(self, stack, params_getx):
        x_keys, y_keys, expected = params_getx
        chains = set_up_chains(stack, x_keys, y_keys, self.views,
                               self.view_keys, 'x')

        for chain, args in zip(chains, expected):

            values, index, columns, pindex, pcolumns = args

            expected_dataframe = create_frame(values, index, columns)
            painted_index = create_index(pindex)
            painted_columns = create_index(pcolumns)


            ### Test Chain.dataframe is Chain._frame
            assert chain.dataframe is chain._frame

            ### Test Chain.get
            assert_frame_equal(chain.dataframe, expected_dataframe)

            # ### Test Chain.paint
            chain.paint()
            assert_index_equal(chain.dataframe.index, painted_index)
            assert_index_equal(chain.dataframe.columns, painted_columns)

            # # ### Test Chain.toggle_labels
            chain.toggle_labels()
            assert_frame_equal(chain.dataframe, expected_dataframe)
            chain.toggle_labels()
            assert_index_equal(chain.dataframe.index, painted_index)
            assert_index_equal(chain.dataframe.columns, painted_columns)

            ### Test Chain str/ len

            ### Test Contents


"""
# -*- coding: utf-8 -*-

import operator
import pytest

from foobar import Package, Woman, Man

PACKAGES = [
    Package('requests', 'Apache 2.0'),
    Package('django', 'BSD'),
    Package('pytest', 'MIT'),
]


@pytest.fixture(params=PACKAGES, ids=operator.attrgetter('name'))
def python_package(request):
    return request.param


@pytest.mark.parametrize('person', [
    Woman('Audrey'), Woman('Brianna'),
    Man('Daniel'), Woman('Ola'), Man('Kenneth')
])
def test_become_a_programmer(person, python_package):
    person.learn(python_package.name)
    assert person.looks_like_a_programmer


def test_is_open_source(python_package):
    assert python_package.is_open_source
"""
