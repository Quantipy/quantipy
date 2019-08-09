
import os
import re
import sys
import pytest
import numpy as np
from operator import sub
from zipfile import ZipFile, BadZipfile, LargeZipFile

from quantipy.core.stack import Stack
from quantipy.core.dataset import DataSet
from quantipy.core.chainmanager import ChainManager
from quantipy.core.builds.xlsx.excel import Excel

# from quantipy.sandbox.excel import Excel
# from quantipy.sandbox.excel_formats_constants import _DEFAULT_ATTRIBUTES
from quantipy.core.view_generators.view_specs import ViewManager
from quantipy.core.view_generators.view_mapper import ViewMapper
from quantipy.core.view_generators.view_maps import QuantipyViews

from .parameters_excel import (
    PATH_BASIC,
    XKEYS_BASIC,
    YKEYS_BASIC,
    VIEWS_BASIC,
    OPENS_BASIC,
    CELLS_BASIC,
    WEIGHT_BASIC,
    SHEET_PROPERTIES_BASIC,
    SHEET_PROPERTIES_EXCEL_BASIC,
    FORMATS_BASIC,
    PATH_COMPLEX_0,
    PATH_COMPLEX_1,
    PATH_COMPLEX_2,
    PATH_COMPLEX_3,
    XKEYS_COMPLEX,
    YKEYS_COMPLEX,
    VIEWS_COMPLEX,
    OPENS_COMPLEX,
    WEIGHT_COMPLEX,
    VIEWS_COMPLEX_MAIN,
    VIEWS_COMPLEX_WAVE,
    VIEWS_COMPLEX_ARRAY,
    VIEWS_COMPLEX_MEAN,
    FORMATS_0,
    SHEET_PROPERTIES_1,
    VIEW_GROUPS_1,
    FORMATS_1,
    IMAGE_1,
    DECIMALS_1,
    SHEET_PROPERTIES_2,
    VIEW_GROUPS_2,
    FORMATS_2,
    ANNOTATIONS_2,
    SHEET_PROPERTIES_3,
    VIEW_GROUPS_3,
    FORMATS_3,
    ANNOTATIONS_3,
    ITALICISE_LEVEL_3,
    DECIMALS_3,
    DETAILS_3)

# -----------------------------------------------------------------------------
PATH_DATA = './tests/'
NAME_PROJ = 'Example Data (A)'
NAME_META = 'Example Data (A).json'
NAME_DATA = 'Example Data (A).csv'
PATH_META = os.path.join(PATH_DATA, NAME_META)
PATH_DATA = os.path.join(PATH_DATA, NAME_DATA)

DATA_KEY = 'dk'
FILTER_KEY = 'no_filter'
ISO8601 = r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)'
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
    ds = DataSet(NAME_PROJ, dimensions_comp=False)
    ds.read_quantipy(PATH_META, PATH_DATA)
    return ds


@pytest.fixture(scope='module')
def stack(dataset):
    meta = dataset._meta
    data = dataset._data.copy()
    data.loc[30:, 'q5_2'] = np.NaN
    data.loc[30:, 'q5_4'] = np.NaN
    return Stack(NAME_PROJ, add_data={DATA_KEY: {'meta': meta, 'data': data}})


@pytest.fixture(scope='class')
def chain_manager(stack):
    return Chain_Manager(stack)


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
        stack.add_link(
            x=self.flatten(XKEYS_BASIC),
            y=YKEYS_BASIC,
            views=VIEWS_BASIC,
            weights=WEIGHT_BASIC)

        locality = stack[DATA_KEY].meta['columns']['locality']
        for idx in xrange(1, 4):
            locality['values'][idx]['text'] = {'en-GB': '.'}

        vm = ViewManager(stack)
        vm.get_views(cell_items=CELLS_BASIC,
                     weight=WEIGHT_BASIC,
                     bases='auto').group()

        _basic = ChainManager(stack)
        for x, item in enumerate(XKEYS_BASIC):
            if isinstance(item, basestring):
                folder = None
            else:
                folder = 'FOLDER_{}'.format(x)
            _basic.get(
                data_key=DATA_KEY,
                filter_key=FILTER_KEY,
                x_keys=item,
                y_keys=YKEYS_BASIC,
                views=vm.views,
                orient='x',
                prioritize=True,
                folder=folder)

        _basic.add_df(
            stack[DATA_KEY].data.loc[:, OPENS_BASIC],
            meta_from=DATA_KEY,
            name='Open Ends')

        _basic.paint_all()
        return _basic

    def complex_chain_manager(self, stack):
        weight = [None, WEIGHT_COMPLEX]
        for x, y in [(XKEYS_COMPLEX, YKEYS_COMPLEX), ('q5', '@'), ('@', 'q5')]:
            stack.add_link(x=x, y=y, views=VIEWS_COMPLEX, weights=weight)

        kwargs = dict(combine=False)
        mapper = self.net_mapper(
            'No', [dict(No=[1, 2, 3])], 'Net: No', **kwargs)
        stack.add_link(
            x=XKEYS_COMPLEX[0],
            y=YKEYS_COMPLEX,
            views=mapper,
            weights=WEIGHT_COMPLEX)
        stack.add_link(x='q5', y='@', views=mapper, weights=WEIGHT_COMPLEX)
        stack.add_link(x='@', y='q5', views=mapper, weights=WEIGHT_COMPLEX)

        mapper = self.net_mapper(
            'Yes', [dict(Yes=[4, 5, 97])], 'Net: Yes', **kwargs)
        stack.add_link(
            x=XKEYS_COMPLEX[0],
            y=YKEYS_COMPLEX,
            views=mapper,
            weights=WEIGHT_COMPLEX)
        stack.add_link(x='q5', y='@', views=mapper, weights=WEIGHT_COMPLEX)
        stack.add_link(x='@', y='q5', views=mapper, weights=WEIGHT_COMPLEX)

        logic = [
            dict(
                N1=[1, 2],
                text={'en-GB': 'Waves 1 & 2 (NET)'},
                expand='after'),
            dict(
                N2=[4, 5],
                text={'en-GB': 'Waves 4 & 5 (NET)'},
                expand='after')]
        kwargs = dict(combine=False, complete=True, expand='after')
        mapper = self.net_mapper('BLOCK', logic, 'Net: ', **kwargs)
        stack.add_link(
            x=XKEYS_COMPLEX[-1],
            y=YKEYS_COMPLEX,
            views=mapper,
            weights=WEIGHT_COMPLEX)

        mapper = self.new_net_mapper()
        kwargs = {
            'calc_only': False,
            'calc': {
                'text': {u'en-GB': u'Net YES'},
                'Net agreement': ('Net: Yes', sub, 'Net: No')},
            'axis': 'x',
            'logic': [
                {'text': {u'en-GB': 'Net: No'}, 'Net: No': [1, 2]},
                {'text': {u'en-GB': 'Net: Yes'}, 'Net: Yes': [4, 5]}]}
        mapper.add_method(name='NPS', kwargs=kwargs)
        kwargs = {
            'calc_only': True,
            'calc': {
                'text': {u'en-GB': u'Net YES'},
                'Net agreement (only)': ('Net: Yes', sub, 'Net: No')},
            'axis': 'x',
            'logic': [
                {'text': {u'en-GB': 'Net: No'}, 'Net: No': [1, 2]},
                {'text': {u'en-GB': 'Net: Yes'}, 'Net: Yes': [4, 5]}]}
        mapper.add_method(name='NPSonly', kwargs=kwargs)
        stack.add_link(
            x=XKEYS_COMPLEX[0],
            y=YKEYS_COMPLEX,
            views=mapper,
            weights=WEIGHT_COMPLEX)
        stack.add_link(x='q5', y='@', views=mapper, weights=WEIGHT_COMPLEX)
        stack.add_link(x='@', y='q5', views=mapper, weights=WEIGHT_COMPLEX)

        options = dict(
            stats=None,
            source=None,
            rescale=None,
            drop=False,
            exclude=None,
            axis='x',
            text='')
        stats = [
            'mean', 'stddev', 'median', 'var', 'varcoeff', 'sem', 'lower_q',
            'upper_q']
        for stat in stats:
            options = {
                'stats': stat,
                'source': None,
                'rescale': None,
                'drop': False,
                'exclude': None,
                'axis': 'x',
                'text': ''}
            mapper = ViewMapper()
            mapper.make_template('descriptives')
            mapper.add_method('stat', kwargs=options)
            stack.add_link(
                x=XKEYS_COMPLEX,
                y=YKEYS_COMPLEX,
                views=mapper,
                weights=WEIGHT_COMPLEX)
            stack.add_link(x='q5', y='@', views=mapper, weights=WEIGHT_COMPLEX)
            stack.add_link(x='@', y='q5', views=mapper, weights=WEIGHT_COMPLEX)

        mapper = ViewMapper().make_template('coltests')
        options = dict(
            level=0.8, metric='props', test_total=True, flag_bases=[30, 100])
        mapper.add_method('test', kwargs=options)
        stack.add_link(
            x=XKEYS_COMPLEX,
            y=YKEYS_COMPLEX,
            views=mapper,
            weights=WEIGHT_COMPLEX)

        mapper = ViewMapper().make_template('coltests')
        options = dict(
            level=0.8, metric='means', test_total=True, flag_bases=[30, 100])
        mapper.add_method('test', kwargs=options)
        stack.add_link(
            x=XKEYS_COMPLEX,
            y=YKEYS_COMPLEX,
            views=mapper,
            weights=WEIGHT_COMPLEX)

        _complex = ChainManager(stack)
        _complex.get(
            data_key=DATA_KEY,
            filter_key=FILTER_KEY,
            x_keys=XKEYS_COMPLEX[:-1],
            y_keys=YKEYS_COMPLEX,
            views=VIEWS_COMPLEX_MAIN,
            orient='x',
            folder='Main')

        _complex.get(
            data_key=DATA_KEY,
            filter_key=FILTER_KEY,
            x_keys=XKEYS_COMPLEX[-1],
            y_keys=YKEYS_COMPLEX,
            views=VIEWS_COMPLEX_WAVE,
            orient='x',
            folder='Main')

        _complex.get(
            data_key=DATA_KEY,
            filter_key=FILTER_KEY,
            x_keys='q5',
            y_keys='@',
            views=VIEWS_COMPLEX_ARRAY,
            folder='arr_0')

        _complex.get(
            data_key=DATA_KEY,
            filter_key=FILTER_KEY,
            x_keys='@',
            y_keys='q5',
            views=VIEWS_COMPLEX_ARRAY,
            folder='arr_1')

        _complex.get(
            data_key=DATA_KEY,
            filter_key=FILTER_KEY,
            x_keys='q5',
            y_keys='@',
            views=VIEWS_COMPLEX_MEAN,
            folder='mean_0')

        _complex.get(
            data_key=DATA_KEY,
            filter_key=FILTER_KEY,
            x_keys='@',
            y_keys='q5',
            views=VIEWS_COMPLEX_MEAN,
            folder='mean_1')

        _complex.add_df(
            stack[DATA_KEY].data.loc[:, OPENS_COMPLEX],
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
        freq = QuantipyViews().frequency
        iters = dict(iterators=dict(rel_to=[None, 'y', 'x'], groups='Nets'))
        return ViewMapper(template=dict(method=freq, kwargs=iters))

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


def excel(cm, sheet_properties, views_groups, italicise_level, details,
          decimals, image, formats, annotations, properties):
    kwargs = formats if formats else dict()
    x = Excel(
        'tmp.xlsx',
        views_groups=views_groups,
        italicise_level=italicise_level,
        details=details,
        decimals=decimals,
        image=image,
        annotations=annotations,
        sheet_properties=properties if properties else dict(),
        **kwargs)
    x.add_chains(cm, **(sheet_properties if sheet_properties else dict()))
    x.close()


@pytest.yield_fixture(
    scope='class',
    params=[
        (
            'basic', PATH_BASIC,
            SHEET_PROPERTIES_BASIC, None, None, False, None, None,
            FORMATS_BASIC, None, SHEET_PROPERTIES_EXCEL_BASIC
        ),
        (
            'complex', PATH_COMPLEX_0, None, None, None, False, None,
            None, FORMATS_0, None, None
        ),
        (
            'complex', PATH_COMPLEX_1, SHEET_PROPERTIES_1, VIEW_GROUPS_1,
            None, False, DECIMALS_1, IMAGE_1, FORMATS_1, None, None
        ),
        (
            'complex', PATH_COMPLEX_2, SHEET_PROPERTIES_2, VIEW_GROUPS_2,
            None, False, None, None, FORMATS_2, ANNOTATIONS_2, None
        ),
        (
            'complex', PATH_COMPLEX_3, SHEET_PROPERTIES_3, VIEW_GROUPS_3,
            ITALICISE_LEVEL_3, DETAILS_3, DECIMALS_3, None, FORMATS_3,
            ANNOTATIONS_3, None
        )
    ]
)
def params(request):
    return request.param


class TestExcel:

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
