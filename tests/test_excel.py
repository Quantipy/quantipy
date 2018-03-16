
import os
import re
import pytest
import collections
from operator import sub
from zipfile import ZipFile, BadZipfile, LargeZipFile
import numpy as np
import quantipy as qp
from quantipy.sandbox.sandbox import ChainManager
from quantipy.sandbox.excel import Excel
from quantipy.sandbox.excel_formats_constants import _DEFAULT_ATTRIBUTES
from quantipy.core.view_generators.view_specs import ViewManager

import parameters_excel as p

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
        self.stack = stack
        self.basic = None
        self.complex = None

    def __getitem__(self, value):
        attr = getattr(self, value)
        if attr:
            return attr
        attr = getattr(self, value + '_chain_manager')(self.stack)
        setattr(self, value, attr)
        return attr

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
        stack.add_link(x=self.flatten(p.XKEYS_BASIC), y=p.YKEYS_BASIC,
                       views=p.VIEWS_BASIC, weights=p.WEIGHT_BASIC)

        vm = ViewManager(stack)
        vm.get_views(cell_items=p.CELLS_BASIC,
                     weight=p.WEIGHT_BASIC,
                     bases='auto').group()

        _basic = ChainManager(stack)
        for item in p.XKEYS_BASIC:
            folder = None if isinstance(item, basestring) \
                        else 'FOLDER_%s' % str(p.XKEYS_BASIC.index(item))
            _basic.get(data_key=DATA_KEY,
                       filter_key=FILTER_KEY,
                       x_keys=item,
                       y_keys=p.YKEYS_BASIC,
                       views=vm.views,
                       orient='x',
                       prioritize=True,
                       folder=folder)

        _basic.add(stack[DATA_KEY].data.loc[:, p.OPENS_BASIC],
                   meta_from=DATA_KEY,
                   name='Open Ends')

        _basic.paint_all()

        return _basic

    def complex_chain_manager(self, stack):
        weight = [None, p.WEIGHT_COMPLEX]
        for x, y in [(p.XKEYS_COMPLEX, p.YKEYS_COMPLEX), ('q5', '@'), ('@', 'q5')]:
            stack.add_link(x=x, y=y, views=p.VIEWS_COMPLEX, weights=weight)

        kwargs = dict(combine=False)
        mapper = self.net_mapper('No', [dict(No=[1, 2, 3])],
                                 'Net: No', **kwargs)
        stack.add_link(x=p.XKEYS_COMPLEX[0], y=p.YKEYS_COMPLEX,
                       views=mapper, weights=p.WEIGHT_COMPLEX)
        stack.add_link(x='q5', y='@', views=mapper, weights=p.WEIGHT_COMPLEX)
        stack.add_link(x='@', y='q5', views=mapper, weights=p.WEIGHT_COMPLEX)

        mapper = self.net_mapper('Yes', [dict(Yes=[4, 5, 97])],
                                 'Net: Yes', **kwargs)
        stack.add_link(x=p.XKEYS_COMPLEX[0], y=p.YKEYS_COMPLEX,
                       views=mapper, weights=p.WEIGHT_COMPLEX)
        stack.add_link(x='q5', y='@', views=mapper, weights=p.WEIGHT_COMPLEX)
        stack.add_link(x='@', y='q5', views=mapper, weights=p.WEIGHT_COMPLEX)

        logic = [dict(N1=[1, 2],
                      text={'en-GB': 'Waves 1 & 2 (NET)'},
                      expand='after'),
                 dict(N2=[4, 5],
                     text={'en-GB': 'Waves 4 & 5 (NET)'},
                      expand='after')]
        kwargs = dict(combine=False, complete=True, expand='after')
        mapper = self.net_mapper('BLOCK', logic, 'Net: ', **kwargs)
        stack.add_link(x=p.XKEYS_COMPLEX[-1], y=p.YKEYS_COMPLEX,
                       views=mapper, weights=p.WEIGHT_COMPLEX)

        mapper = self.new_net_mapper()
        kwargs = {'calc_only': False,
                  'calc': {'text': {u'en-GB': u'Net YES'},
                           'Net agreement': ('Net: Yes', sub, 'Net: No')},
                  'axis': 'x',
                  'logic': [{'text': {u'en-GB': 'Net: No'}, 'Net: No': [1, 2]},
                            {'text': {u'en-GB': 'Net: Yes'}, 'Net: Yes': [4, 5]}]}
        mapper.add_method(name='NPS', kwargs=kwargs)
        kwargs = {'calc_only': True,
                  'calc': {'text': {u'en-GB': u'Net YES'},
                           'Net agreement (only)': ('Net: Yes', sub, 'Net: No')},
                  'axis': 'x',
                  'logic': [{'text': {u'en-GB': 'Net: No'}, 'Net: No': [1, 2]},
                            {'text': {u'en-GB': 'Net: Yes'}, 'Net: Yes': [4, 5]}]}
        mapper.add_method(name='NPSonly', kwargs=kwargs)
        stack.add_link(x=p.XKEYS_COMPLEX[0], y=p.YKEYS_COMPLEX,
                       views=mapper, weights=p.WEIGHT_COMPLEX)
        stack.add_link(x='q5', y='@', views=mapper, weights=p.WEIGHT_COMPLEX)
        stack.add_link(x='@', y='q5', views=mapper, weights=p.WEIGHT_COMPLEX)

        options = dict(stats=None, source=None, rescale=None, drop=False,
                       exclude=None, axis='x', text='')
        stats = ['mean', 'stddev', 'median', 'var',
                 'varcoeff', 'sem', 'lower_q', 'upper_q']
        for stat in stats:
            options = {'stats': stat,
                       'source': None,
                       'rescale': None,
                       'drop': False,
                       'exclude': None,
                       'axis': 'x',
                       'text': ''}
            mapper = qp.ViewMapper()
            mapper.make_template('descriptives')
            mapper.add_method('stat', kwargs=options)
            stack.add_link(x=p.XKEYS_COMPLEX, y=p.YKEYS_COMPLEX,
                           views=mapper, weights=p.WEIGHT_COMPLEX)
            stack.add_link(x='q5', y='@', views=mapper,
                           weights=p.WEIGHT_COMPLEX)
            stack.add_link(x='@', y='q5', views=mapper,
                           weights=p.WEIGHT_COMPLEX)

        mapper = qp.ViewMapper().make_template('coltests')
        options = dict(level=0.8, metric='props',
                       test_total=True, flag_bases=[30, 100])
        mapper.add_method('test', kwargs=options)
        stack.add_link(x=p.XKEYS_COMPLEX, y=p.YKEYS_COMPLEX,
                       views=mapper, weights=p.WEIGHT_COMPLEX)

        mapper = qp.ViewMapper().make_template('coltests')
        options = dict(level=0.8, metric='means',
                       test_total=True, flag_bases=[30, 100])
        mapper.add_method('test', kwargs=options)
        stack.add_link(x=p.XKEYS_COMPLEX, y=p.YKEYS_COMPLEX,
                       views=mapper, weights=p.WEIGHT_COMPLEX)

        _complex = ChainManager(stack)
        _complex.get(data_key=DATA_KEY, filter_key=FILTER_KEY,
                     x_keys=p.XKEYS_COMPLEX[:-1], y_keys=p.YKEYS_COMPLEX,
                     views=p.VIEWS_COMPLEX_MAIN, orient='x',
                     folder='Main')

        _complex.get(data_key=DATA_KEY, filter_key=FILTER_KEY,
                     x_keys=p.XKEYS_COMPLEX[-1], y_keys=p.YKEYS_COMPLEX,
                     views=p.VIEWS_COMPLEX_WAVE, orient='x',
                     folder='Main')

        _complex.get(data_key=DATA_KEY, filter_key=FILTER_KEY,
                     x_keys='q5', y_keys='@',
                     views=p.VIEWS_COMPLEX_ARRAY,
                     folder='arr_0')

        _complex.get(data_key=DATA_KEY, filter_key=FILTER_KEY,
                     x_keys='@', y_keys='q5',
                     views=p.VIEWS_COMPLEX_ARRAY,
                     folder='arr_1')

        _complex.get(data_key=DATA_KEY, filter_key=FILTER_KEY,
                     x_keys='q5', y_keys='@',
                     views=p.VIEWS_COMPLEX_MEAN,
                     folder='mean_0')

        _complex.get(data_key=DATA_KEY, filter_key=FILTER_KEY,
                     x_keys='@', y_keys='q5',
                     views=p.VIEWS_COMPLEX_MEAN,
                     folder='mean_1')

        _complex.add(stack[DATA_KEY].data.loc[:, p.OPENS_COMPLEX],
                     meta_from=DATA_KEY,
                     name='Open Ends')

        _complex.paint_all(transform_tests='full')

        self.add_annotations(_complex)

        return _complex

    def net_mapper(self, name, logic, text, mapper=None, **kwargs):
        if mapper is None:
            mapper = self.new_net_mapper()
        kwargs.update(dict(axis='x', logic=logic, text=text))
        mapper.add_method(name=name, kwargs=kwargs)
        return mapper

    @staticmethod
    def new_net_mapper():
        freq = qp.QuantipyViews().frequency
        iters = dict(iterators=dict(rel_to=[None, 'y', 'x'], groups='Nets'))
        return qp.ViewMapper(template=dict(method=freq, kwargs=iters))

    @staticmethod
    def add_annotations(chain_manager):
        chains = chain_manager.chains
        for i in (0, 4, 5, 6, 7):
            chains[i].annotations.set('Headder Title -- no reason',
                                      category='header', position='title')
            chains[i].annotations.set('Header Left -- explanation text',
                                      category='header', position='left')
            chains[i].annotations.set('Header Center -- mask text',
                                      category='header', position='center')
            chains[i].annotations.set('Notes -- base text', category='notes')


@pytest.fixture(scope='module')
def dataset():
    _dataset = qp.DataSet(NAME_PROJ, dimensions_comp=False)
    _dataset.read_quantipy(PATH_META, PATH_DATA)
    yield _dataset.split()
    del _dataset

@pytest.fixture(scope='module')
def stack(dataset):
    meta, data = dataset
    data.loc[30:,'q5_2'] = np.NaN
    data.loc[30:,'q5_4'] = np.NaN
    store = {DATA_KEY: {'meta': meta, 'data': data.copy()}}
    _stack = qp.Stack(NAME_PROJ, add_data=store)
    yield _stack
    del _stack

@pytest.fixture(scope='function')
def excel(chain_manager, sheet_properties, views_groups, italicise_level,
          details, decimals, image, formats, annotations, properties):
    kwargs = formats if formats else dict()
    x = Excel('tmp.xlsx',
              views_groups=views_groups,
              italicise_level=italicise_level,
              details=details,
              decimals=decimals,
              image=image,
              annotations=annotations,
              sheet_properties=properties if properties else dict(),
              **kwargs)
    x.add_chains(chain_manager, **(sheet_properties if sheet_properties else dict()))
    x.close()

@pytest.fixture(scope='class')
def chain_manager(stack):
    return Chain_Manager(stack)

@pytest.yield_fixture(
    scope='class',
    params=[
        (
           'basic', p.PATH_BASIC,
           p.SHEET_PROPERTIES_BASIC, None, None, False, None, None,
           p.FORMATS_BASIC, None, p.SHEET_PROPERTIES_EXCEL_BASIC
        ),
        (
           'complex', p.PATH_COMPLEX_0,
           None, None, None, False, None, None, None, None, None
        ),
        (
           'complex', p.PATH_COMPLEX_1, p.SHEET_PROPERTIES_1, p.VIEW_GROUPS_1,
           None, False, p.DECIMALS_1, p.IMAGE_1, p.FORMATS_1, None, None
        ),
        (
            'complex', p.PATH_COMPLEX_2, p.SHEET_PROPERTIES_2, p.VIEW_GROUPS_2,
            None, False, None, None, p.FORMATS_2, p.ANNOTATIONS_2, None
        ),
        (
           'complex', p.PATH_COMPLEX_3, p.SHEET_PROPERTIES_3, p.VIEW_GROUPS_3,
           p.ITALICISE_LEVEL_3 , p.DETAILS_3, p.DECIMALS_3, None, p.FORMATS_3,
           p.ANNOTATIONS_3, None
        )
    ]
)
def params(request):
    return request.param


class TestExcel:
    teardown = False

    @staticmethod
    def cleandir():
        for x in ('./tmp.xlsx', './qplogo_invert.png'):
            if os.path.exists(x):
                os.remove(x)

    @classmethod
    def setup_class(cls):
        cls.cleandir()

    @classmethod
    def teardown_class(cls):
        if cls.teardown:
            cls.cleandir()

    def test_structure(self, chain_manager, params):

        complexity, path_expected, sp, vg, il, dt, dc, im, fm, an, pt = params

        excel(chain_manager[complexity], sp, vg, il, dt, dc, im, fm, an, pt)

        zip_got, zip_exp = _load_zip('tmp.xlsx'), _load_zip(path_expected)

        assert zip_got.namelist() == zip_exp.namelist()

        for filename in zip_got.namelist():
            xml_got = _read_file(zip_got, filename)
            xml_exp = _read_file(zip_exp, filename)
            err = ' ... %s ...\nGOT: %s\nEXPECTED: %s'
            assert xml_got == xml_exp, err % (filename, xml_got, xml_exp)

        TestExcel.teardown = False
