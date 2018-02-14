
import os
import re
import pytest
import collections
from operator import sub
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
        self.complex = self.complex_chain_manager(stack)

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

        stack.add_link(x=self.flatten(x_keys), y=y_keys,
                       views=views, weights=weight)

        vm = ViewManager(stack)
        vm.get_views(cell_items=cells, weight=None, bases='auto').group()

        _basic = ChainManager(stack)
        for item in x_keys:
            folder = None if isinstance(item, basestring) \
                        else 'FOLDER_%s' % str(x_keys.index(item))
            _basic.get(data_key=DATA_KEY,
                       filter_key=FILTER_KEY,
                       x_keys=item,
                       y_keys=y_keys,
                       views=vm.views,
                       orient='x',
                       prioritize=True,
                       folder=folder)

        _basic.add(stack[DATA_KEY].data.loc[:, opens],
                   meta_from=(DATA_KEY, FILTER_KEY),
                   name='Open Ends')

        _basic.paint_all()

        return _basic

    @staticmethod
    def net_mapper(name, logic, text, mapper=None, **kwargs):
        if mapper is None:
            freq = qp.QuantipyViews().frequency
            iters = dict(iterators=dict(rel_to=[None, 'y', 'x'], groups='Nets'))
            mapper = qp.ViewMapper(template=dict(method=freq, kwargs=iters))
        kwargs.update(dict(axis='x', logic=logic, text=text))
        mapper.add_method(name=name, kwargs=kwargs)
        return mapper

    def complex_chain_manager(self, stack):
        x_keys = ['q5_1', 'q4', 'gender', 'Wave']
        y_keys = ['@', 'q4 > gender', 'q4 > gender > Wave', 'q5_1']
        views = ['cbase', 'cbase_gross', 'ebase', 'counts', 'c%', 'r%',
                 'counts_sum', 'c%_sum']
        opens = ['RecordNo', 'gender', 'age', 'q8', 'q8a', 'q9', 'q9a']
        cells = ['counts_colpct_rowpct']
        weight = 'weight_a'

        stack.add_link(x=x_keys, y=y_keys, views=views, weights=[None, weight])
        stack.add_link(x='q5', y='@', views=views, weights=weight)
        stack.add_link(x='@', y='q5', views=views, weights=weight)

        kwargs = dict(combine=False)
        mapper = self.net_mapper('No', [dict(No=[1, 2, 3])],
                                 'Net: No', **kwargs)
        stack.add_link(x=x_keys[0], y=y_keys, views=mapper, weights=weight)
        stack.add_link(x='q5', y='@', views=mapper, weights=weight)
        stack.add_link(x='@', y='q5', views=mapper, weights=weight)

        mapper = self.net_mapper('Yes', [dict(Yes=[4, 5, 97])],
                                 'Net: Yes', **kwargs)
        stack.add_link(x=x_keys[0], y=y_keys, views=mapper, weights=weight)
        stack.add_link(x='q5', y='@', views=mapper, weights=weight)
        stack.add_link(x='@', y='q5', views=mapper, weights=weight)

        logic = [dict(N1=[1, 2],
                      text={'en-GB': 'Waves 1 & 2 (NET)'},
                      expand='after'),
                 dict(N2=[4, 5],
                     text={'en-GB': 'Waves 4 & 5 (NET)'},
                      expand='after')]
        kwargs = dict(combine=False, complete=True, expand='after')
        mapper = self.net_mapper('BLOCK', logic,
                                 'Net: ', **kwargs)
        stack.add_link(x=x_keys[-1], y=y_keys, views=mapper, weights=weight)

        logic = [{'text': {u'en-GB': 'Net: No'}, 'Net: No': [1, 2]},
                 {'text': {u'en-GB': 'Net: Yes'}, 'Net: Yes': [4, 5]}]
        kwargs = {'calc_only': False,
                  'calc': {'text': {u'en-GB': u'Net YES'},
                           'Net agreement': ('Net: Yes', sub, 'Net: No')}}
        mapper = self.net_mapper('NPS', logic,
                                 'Net: ', **kwargs)

        logic = [{'text': {u'en-GB': 'Net: No'}, 'Net: No': [1, 2]},
                 {'text': {u'en-GB': 'Net: Yes'}, 'Net: Yes': [4, 5]}]
        kwargs = {'calc_only': True,
                  'calc': {'text': {u'en-GB': u'Net YES'},
                           'Net agreement (only)': ('Net: Yes', sub, 'Net: No')}}

        mapper = self.net_mapper('NPSonly', logic, 'Net: ',
                                 mapper=mapper, **kwargs)
        stack.add_link(x=x_keys[-1], y=y_keys, views=mapper, weights=weight)

        options = dict(stats=None, source=None, rescale=None, drop=False,
                       exclude=None, axis='x', text='') 
        stats = ['mean', 'stddev', 'median', 'var',
                 'varcoeff', 'sem', 'lower_q', 'upper_q']
        for stat in stats:
            options['stat'] = stat
            mapper = qp.ViewMapper()
            mapper.make_template('descriptives')
            mapper.add_method(stat, kwargs=options)
            stack.add_link(x=x_keys, y=y_keys, views=mapper, weights=weight)
            stack.add_link(x='q5', y='@', views=mapper, weights=weight)
            stack.add_link(x='@', y='q5', views=mapper, weights=weight)

        test_view = qp.ViewMapper().make_template('coltests')
        options = dict(level=0.8, metric='props',
                       test_total=True, flag_bases=[30, 100])
        test_view.add_method('test', kwargs=options)
        stack.add_link(x=x_keys, y=y_keys, views=mapper, weights=weight)

        test_view = qp.ViewMapper().make_template('coltests')
        options = dict(level=0.8, metric='means',
                       test_total=True, flag_bases=[30, 100])
        test_view.add_method('test', kwargs=options)
        stack.add_link(x=x_keys, y=y_keys, views=mapper, weights=weight)

        vm = ViewManager(stack)
        vm.get_views(cell_items=cells, weight=None, bases='auto').group()

        _complex = ChainManager(stack)
        _complex.get(data_key=DATA_KEY, filter_key=FILTER_KEY,
                     x_keys=x_key, y_keys=y_keys,
                     views=vm.views, orient='x',
                     folder='Main')

        _complex.add(stack[DATA_KEY].data.loc[:, opens],
                     meta_from=(DATA_KEY, FILTER_KEY),
                     name='Open Ends')

        return _complex


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
