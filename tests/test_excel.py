
import os
import re
import pytest
import collections
from zipfile import ZipFile, BadZipfile, LargeZipFile

import quantipy as qp
from quantipy.sandbox.sandbox import ChainManager
from quantipy.sandbox.excel import Excel
from quantipy.core.view_generators.view_specs import ViewManager

import parameters_excel as parameters

# -----------------------------------------------------------------------------
PATH_DATA  = './tests/'
NAME_PROJ  = 'Example Data (A)'
NAME_META  = 'Example Data (A).json'
NAME_DATA  = 'Example Data (A).csv'
PATH_META  = os.path.join(PATH_DATA, NAME_META)
PATH_DATA  = os.path.join(PATH_DATA, NAME_DATA)

DATA_KEY   = 'x'
FILTER_KEY = 'no_filter'
ISO8601    = r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)'
# -----------------------------------------------------------------------------

def _load_zip(path):
    try:
        z = ZipFile(path, 'r')
    except (BadZipfile, LargeZipFile):
        raise BadZipfile('%s: %s' % (path, sys.exc_info()[1]))
    else:
        return z

def _read_file(zipf, filename):
    try:
        f = zipf.read(filename)
    except KeyError:
        print 'ERROR: Did not find %s in zip file' % filename
    else:
        return re.sub(ISO8601, '', f)

class Chain_Manager:
    def __init__(self, stack):
        self.basic = self.basic_chain_manager(stack)
        # self.complex = complex(stack)

    def __getitem__(self, value):
        return getattr(self, value)

    @staticmethod
    def flatten(l):
        res = []
        for i, el in enumerate(l):
            if isinstance(el, basestring):
                res.append(el)
            else:
                res.extend(el)
        return res

    def basic_chain_manager(self, stack):

        x_keys = ['q2', 'q2b', 'q3', 'q4',
                 ['q5_1', 'q5_2', 'q5_3', 'q5_4', 'q5_5', 'q5_6'],
                 'q8', 'q9']
        y_keys = ['@', 'gender', 'locality']
        views = ['cbase', 'counts']
        opens = ['q8a', 'q9a']
        cells = 'counts'
        weight = None

        stack.add_link(data_keys=DATA_KEY,
                       filters=FILTER_KEY,
                       x=self.flatten(x_keys),
                       y=y_keys,
                       views=views)

        vm = ViewManager(stack)
        vm.get_views(cell_items=cells,
                     weight=None,
                     bases='auto').group()

        _chain_manager_basic = ChainManager(stack)
        xks = []
        for item in x_keys:
            if isinstance(item, basestring):
                _chain_manager_basic.get(data_key=DATA_KEY,
                                         filter_key=FILTER_KEY,
                                         x_keys=item,
                                         y_keys=y_keys,
                                         views=vm.views,
                                         orient='x',
                                         prioritize=True,
                                         folder=None)
            else:
                _chain_manager_basic.get(data_key=DATA_KEY,
                                         filter_key=FILTER_KEY,
                                         x_keys=item,
                                         y_keys=y_keys,
                                         views=vm.views,
                                         orient='x',
                                         prioritize=True,
                                         folder='FOLDER_%s' % str(x_keys.index(item)))

        _chain_manager_basic.add(stack[DATA_KEY].data.loc[:, opens],
                                 meta_from=(DATA_KEY, FILTER_KEY),
                                 name='Open Ends')

        _chain_manager_basic.paint_all()

        return _chain_manager_basic

@pytest.fixture(scope='module')
def dataset():
    _dataset = qp.DataSet(NAME_PROJ, dimensions_comp=False)
    _dataset.read_quantipy(PATH_META, PATH_DATA)
    yield _dataset.split()
    del _dataset

@pytest.fixture(scope='module')
def stack(dataset):
    meta, data = dataset
    store = {DATA_KEY: {'meta': meta, 'data': data.copy()}}
    _stack = qp.Stack(NAME_PROJ, add_data=store)
    yield _stack
    del _stack

@pytest.fixture(scope='function')
def excel(chain_manager):
    x = Excel('tmp.xlsx')
    x.add_chains(chain_manager)
    x.close()

@pytest.fixture(scope='class')
def chain_manager(stack):
    return Chain_Manager(stack)

@pytest.yield_fixture(
    scope='class',
    params=[('basic', parameters.PATH_BASIC)]
)
def params(request):
    return request.param


class TestExcel:
    teardown = False

    @staticmethod
    def cleandir():
        if os.path.exists('./tmp.xlsx'):
            os.remove('./tmp.xlsx')

    @classmethod
    def setup_class(cls):
        cls.cleandir()

    @classmethod
    def teardown_class(cls):
        if cls.teardown:
            cls.cleandir()

    def test_structure(self, chain_manager, params):

        complexity, path_expected = params

        excel(chain_manager[complexity])

        zip_got, zip_exp = _load_zip('tmp.xlsx'), _load_zip(path_expected)

        assert zip_got.namelist() == zip_exp.namelist()

        for filename in zip_got.namelist():
            xml_got = _read_file(zip_got, filename)
            xml_exp = _read_file(zip_exp, filename)
            err = ' ... %s ...\nGOT: %s\nEXPECTED: %s'
            assert xml_got == xml_exp, err % (filename, xml_got, xml_exp)

        TestExcel.teardown = True
