
import pytest
import os
import time
import pandas as pd
import numpy as np
import quantipy as qp
from itertools import count, izip

from quantipy.sandbox.sandbox import ChainManager

from pandas.util.testing import assert_frame_equal, assert_index_equal

import chain_fixtures as fixture

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
                                           'data': data.copy().head(250)}})
    yield _stack
    del _stack

@pytest.fixture(scope='class')
def basic_chain(stack):
    _chain = ChainManager(stack)
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

    _chain = ChainManager(stack)
    _chain.get(data_key='x',
               filter_key='no_filter',
               x_keys=x_keys,
               y_keys=y_keys,
               views=view_keys,
               orient=orient)
    _chains = _chain
    # if isinstance(_chains, Chain): # single chain
    #     _chains = [_chains]
    return _chains

@pytest.fixture(scope='class')
def unnamed_chain_for_structure(dataset, basic_chain):
    _, data = dataset
    columns = ['record_number', 'age']
    _frame = data.head(1).loc[:, columns]
    basic_chain.add(_frame, DATA_KEY)
    yield basic_chain
    del basic_chain

@pytest.fixture(scope='class')
def chain_for_structure(dataset, basic_chain):
    _, data = dataset
    columns = ['record_number', 'age', 'gender', 'q9', 'q9a']
    _frame = data.head(250).loc[:, columns]
    basic_chain.add(_frame, DATA_KEY, name='open')
    yield basic_chain
    del basic_chain

@pytest.fixture(scope='function')
def chain_structure(chain_for_structure, paint=False, sep=None):
    if paint:
        chain_for_structure.paint_all(sep=sep or '. ',
                                      text_key='en-GB',
                                      na_rep=fixture.AST)
    it = iter(chain_for_structure)
    return next(it)

@pytest.fixture(scope='function')
def expected_structure(values, columns, paint=False):
    _expected = pd.DataFrame(np.array(values).T, columns=columns)
    _expected.iloc[:, 0] = pd.to_numeric(_expected.iloc[:, 0])
    _expected.iloc[:, 1] = pd.to_numeric(_expected.iloc[:, 1])
    if not paint:
        _expected.iloc[:, 2] = pd.to_numeric(_expected.iloc[:, 2])
    return _expected

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
        assert isinstance(basic_chain.stack, qp.Stack)

    def test_str(self, basic_chain):
        assert str(basic_chain) == ""

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

@pytest.yield_fixture(
    scope='class',
    params=[
        (['q5_1'], ['@', 'q4'], fixture.X1),
        (['q5_1'], ['@', 'q4 > gender'], fixture.X2),
        (['q5_1'], ['@', 'q4 > gender > Wave'], fixture.X3),
        (
            ['q5_1'],
            ['@', 'q4 > gender > Wave', 'q5_1', 'q4 > gender'],
            fixture.X4
        ),
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
        chain_df = (chains[0].transform_tests()
                             .dataframe.replace(np.NaN, 'None')
                   )
        # all tests results converted correctly from numbers to letters?
        actual_vals = pd.DataFrame(chain_df.values.tolist())
        expected_vals = pd.DataFrame(fixture.X5_SIG_SIMPLE[0])
        assert_frame_equal(expected_vals, actual_vals)
        # correctly added third index level with letter row?
        actual_cols = chain_df.columns.tolist()
        expected_cols = fixture.X5_SIG_SIMPLE[1]
        assert expected_cols == actual_cols
        # sig_test_letters property updated correctly?
        actual_letters = chains[0].sig_test_letters
        expected_letters = fixture.X5_SIG_SIMPLE[2]
        assert expected_letters == actual_letters

    def test_annotations_fields(self, stack):
        x, y = 'q5_1', ['@', 'gender', 'q4']
        chains = complex_chain(stack, x, y, self._VIEWS, self._VIEW_SIG_KEYS,
                               'x', incl_tests=True, incl_sum=True)
        annot = chains[0].annotations
        annot.set('header-title', 'header', 'title')
        annot.set('header-left', 'header', 'left')
        annot.set('header-center', 'header', 'center')
        annot.set('header-right', 'header', 'right')
        annot.set('footer-title', 'footer', 'title')
        annot.set('footer-left', 'footer', 'left')
        annot.set('footer-center', 'footer', 'center')
        annot.set('footer-right', 'footer', 'right')
        annot.set('notes', 'notes', None)
        # are all attributes populated correctly...?
        # (text, as list, props & dict)
        assert annot.header_title == ['header-title'] == annot.header['title']
        assert annot.header_left == ['header-left'] == annot.header['left']
        assert annot.header_center == ['header-center'] == annot.header['center']
        assert annot.header_right == ['header-right'] == annot.header['right']
        assert annot.footer_title == ['footer-title'] == annot.footer['title']
        assert annot.footer_left == ['footer-left'] == annot.footer['left']
        assert annot.footer_center == ['footer-center'] == annot.footer['center']
        assert annot.footer_right == ['footer-right'] == annot.footer['right']
        assert annot.notes == ['notes'] == annot.notes

    def test_annotations_populated(self, stack):
        x, y = 'q5_1', ['@', 'gender', 'q4']
        chains = complex_chain(stack, x, y, self._VIEWS, self._VIEW_SIG_KEYS,
                               'x', incl_tests=True, incl_sum=True)
        annot = chains[0].annotations
        annot.set('header-center', 'header', 'center')
        annot.set('footer-left', 'footer', 'left')
        annot.set('footer-center', 'footer', 'center')
        annot.set('notes', 'notes', None)
        # are the populated fields returned in sorted order?
        expected = ['footer-center', 'footer-left', 'header-center', 'notes']
        assert annot.populated == expected

    def test_annotations_list_append(self, stack):
        x, y = 'q5_1', ['@', 'gender', 'q4']
        chains = complex_chain(stack, x, y, self._VIEWS, self._VIEW_SIG_KEYS,
                               'x', incl_tests=True, incl_sum=True)
        annot = chains[0].annotations
        annot.set('header-title1', 'header', 'title')
        annot.set('footer-right1', 'footer', 'right')
        annot.set('notes1', 'notes', None)
        annot.set('header-title2', 'header', 'title')
        annot.set('footer-right2', 'footer', 'right')
        annot.set('notes2', 'notes', None)
        assert annot.header_title == ['header-title1', 'header-title2']
        assert annot.footer_right == ['footer-right1', 'footer-right2']
        assert annot.notes == ['notes1', 'notes2']

    def test_sig_transformation_large(self, stack):
        pass

@pytest.yield_fixture(
    scope='class',
    params=[
        (
            False,
            fixture.CHAIN_STRUCT_COLUMNS,
            fixture.CHAIN_STRUCT_VALUES,
            False,
            None
        ),
        (
            True,
            fixture.CHAIN_STRUCT_COLUMNS_PAINTED,
            fixture.CHAIN_STRUCT_VALUES_PAINTED,
            fixture.CHAIN_STRUCT_COLUMNS_REPAINTED,
            '* '
        )
    ]
)
def params_structure(request):
    return request.param

class TestChainUnnamedAdd:
    def test_unnamed(self, unnamed_chain_for_structure):
        _chain = chain_structure(unnamed_chain_for_structure)
        assert _chain.name == 'record_number.age'

class TestChainAdd:
    def test_named(self, chain_for_structure):
        _chain = chain_structure(chain_for_structure)
        assert _chain.name == 'open'

    def test_str(self, chain_for_structure, params_structure):
        paint, columns, values, repaint_columns, _ = params_structure

        _chain = chain_structure(chain_for_structure, paint=paint)
        _expected_structure = expected_structure(values, columns, paint=paint)

        assert_frame_equal(_chain.structure.fillna('*'), _expected_structure)

class TestChainAddRepaint:
    def test_str(self, chain_for_structure, params_structure):
        paint, columns, values, repaint_columns, sep = params_structure

        if repaint_columns:
            _chain = chain_structure(chain_for_structure, paint=paint)
            _expected_structure = expected_structure(values,
                                                     repaint_columns,
                                                     paint=paint)

            _chain.paint(sep=sep, text_key='en-GB', na_rep=fixture.AST)

            assert_frame_equal(_chain.structure.fillna('*'),
                               _expected_structure)

