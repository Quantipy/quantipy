
import re
import os
import sys
import pytest
import quantipy as qp

from quantipy.sandbox.sandbox import Chain
from quantipy.sandbox.excel import Excel
from zipfile import ZipFile, BadZipfile, LargeZipFile


# -----------------------------------------------------------------------------
PATH_DATA  = '../../tests/'
NAME_PROJ  = 'Example Data (A)'
NAME_META  = 'Example Data (A).json'
NAME_DATA  = 'Example Data (A).csv'
PATH_META  = os.path.join(PATH_DATA, NAME_META)
PATH_DATA  = os.path.join(PATH_DATA, NAME_DATA)
DATA_KEY   = 'x'
FILTER_KEY = 'no_filter'
VIEWS      = ('cbase', 'counts', 'c%', 'mean', 'median')
VIEW_KEYS  = ('x|f|x:|||cbase', 'x|f|:|||counts', 'x|d.mean|x:|||mean',
              'x|d.median|x:|||median', 'x|f.c:f|x:|||counts_sum')
ORIENT     = 'x'
TEST_FILE  = 'test_excel.xlsx'
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

@pytest.fixture(scope='module')
def dataset():
    _dataset = qp.DataSet(NAME_PROJ, dimensions_comp=False)
    _dataset.read_quantipy(PATH_META, PATH_DATA)
    yield _dataset.split()
    del _dataset

@pytest.fixture(scope='class')
def stack(dataset):
    meta, data = dataset
    data = data.head(250)
    _stack = qp.Stack(NAME_PROJ,
                      add_data={DATA_KEY: {'meta': meta,
                                           'data': data.head(250)}})
    yield _stack
    del _stack

@pytest.fixture(scope='function')
def excel(stack, x_keys, y_keys):
    stack.add_link(x=x_keys, y=y_keys, views=VIEWS)
    chain = Chain(stack, name='chain')
    chains = chain.get(data_key=DATA_KEY, filter_key=FILTER_KEY,
                       x_keys=x_keys, y_keys=y_keys, views=VIEW_KEYS,
                       orient=ORIENT)

    chains = [c.paint() for c in chains]
    _excel = Excel(TEST_FILE)
    _excel.add_chains(chains, 'S H E E T')
    _excel.close()
    return _excel.filename

test_data = [(['q5_1', 'q4', 'gender', 'Wave'],
              ['@', 'q4 > gender > Wave', 'q5_1', 'q4 > gender'],
              'test_exp_complex_nest.xlsx'),
            ]

@pytest.fixture(scope='class', params=test_data, ids=['complex nest'])
def params(request):
    yield request.param

@pytest.fixture()
def cleandir():
    if os.path.exists(TEST_FILE):
        os.remove(TEST_FILE)

@pytest.mark.usefixtures('cleandir')
class TestExcel:

    def test_xml(self, stack, params):

        x_keys, y_keys, expected = params

        got = excel(stack, x_keys, y_keys)

        zip_got, zip_exp = _load_zip(got), _load_zip(expected)

        assert zip_got.namelist() == zip_exp.namelist()

        for filename in zip_got.namelist():
            xml_got = _read_file(zip_got, filename)
            xml_exp = _read_file(zip_exp, filename)
            err = ' ... %s ...\nGOT: %s\nEXPECTED: %s'
            assert xml_got == xml_exp, err % (filename, xml_got, xml_exp)

