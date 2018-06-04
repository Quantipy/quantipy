"""
Created on 20 Nov 2014

@author: JamesG
"""

import json
import numpy as np
import pandas as pd
import quantipy as qp
import warnings
from StringIO import StringIO
from lxml import etree
import sqlite3
import re
from quantipy.core.helpers.functions import load_json
import json

DAYS_TO_MS = 24 * 60 * 60 * 1000
DDF_TYPES_MAP = {
    'X': 'string',
    'L': 'int',
    'D': 'float',
    'T': 'date',
    'C1': 'single',
    'S': 'delimited set',
    'B': 'boolean'
}
MDD_TYPES_MAP = {
    '0': 'unknown (Unitialized or empty)',
    '1': 'int',
    '2': 'string',
    '3': 'categorical',
    '4': 'unknown (Object subtype)',
    '5': 'date',
    '6': 'float',
    '7': 'boolean'
}
RE_GRID_SLICES = "[^{.]+(?=[}]|$|\[)"
XPATH_DEFINITION = '//definition'
XPATH_VARIABLES = '//design//fields//variable'
XPATH_LOOPS = '//design//fields//loop'
XPATH_GRIDS = '//design//fields//grid'
XPATH_CATEGORYMAP = '//categorymap'


def ddf_to_pandas(path_ddf):
    """ Returns a dict of pandas DataFrames from the given Dimensions
    case data file (DDF), which is a sqlite file.
    """

    with sqlite3.connect(path_ddf) as conn:
        tables_df = pd.read_sql('SELECT * FROM sqlite_master;', conn)
        sql = {
            table_name: pd.read_sql('SELECT * FROM '+table_name+';', conn)
            for table_name in tables_df['tbl_name'].values
            if table_name.startswith('L')
        }
        table_info = {}
        for table_name in sql.keys():
            table_info[table_name] = pd.read_sql(
                "PRAGMA table_info('"+table_name+"');",
                conn
            )
        sql['table_info'] = table_info

    if 'Levels' in sql:
        sql['Levels'].set_index(['TableName'], drop=True, inplace=True)
    else:
        raise KeyError(
            "The 'Levels' table was not found. Your DDF may be empty"
            " or corrupt."
        )

    ddf = {
        'table_info': sql['table_info'].copy(),
        'Levels': sql['Levels'].copy(),
        'HDATA': sql['L1'].copy()
    }

    levels = sql['Levels']
    table_name_map = dict(levels['DSCTableName'])
    table_name_map['L1'] = 'HDATA'
    level_id_map = {}
    new_levels_index = ['HDATA']
    for table_name in levels.index[1:]:
        new_table_name = levels.ix[table_name,'DSCTableName']
        ddf[new_table_name] = sql[table_name]
        new_levels_index.append(new_table_name)

    ddf['Levels'].index = pd.Index(new_levels_index, name='table_name')
    ddf['Levels'].drop('DSCTableName', axis=1, inplace=True)
    ddf['Levels']['ParentName'] = ddf['Levels']['ParentName'].map(
        table_name_map
    )
    ddf['Levels']['ParentName'] = ['None'] + [v for v in ddf['Levels']['ParentName'][1:]]

    return ddf


def timestamp_to_ISO8610(timestamp, offset_date="1900-01-01",
                         as_string=False, adjuster=None):

    offset = np.datetime64(offset_date).astype("float") * DAYS_TO_MS
    day = timestamp * DAYS_TO_MS
    date = (day + offset).astype("datetime64[ms]")
    if not adjuster is None:
        date = date - adjuster
    if as_string:
        date = str(date)

    return date


def get_datetime_values(var_df, adjuster, as_string=True):

    dates = var_df.astype(float).apply(
        timestamp_to_ISO8610, args=(
            "1899-12-30",
            as_string,
            np.timedelta64(adjuster,'m')
        )
    )
    if as_string:
        return list(dates.str.encode('utf-8').values)
    else:
        return dates


def quantipy_clean(ddf):

    clean = {}
    data_table_keys = [
        k for k in ddf.keys()
        if not k in ['table_info','Levels']
    ]

    for n_tab in data_table_keys:

        if ddf[n_tab].shape[0] > 0:

            # Map parent columns for heirarchical tables
            p_cols = ['id_'+n_tab]
            child = n_tab
            while True:
                parent = ddf['Levels'].loc[child,'ParentName']
                if parent=='None':
                    break
                id = 'id_'+parent
                p_cols = [id] + p_cols
                child = parent

            # Identify non-parent columns in the table,
            # skip to next table if none
            num_np_cols = ddf[n_tab].columns.size-(len(p_cols)+1)
            if num_np_cols > 0:
                np_cols = list(ddf[n_tab].columns[len(p_cols)+1:])
            else:
                np_cols = []

            # Apply parent-mapped column names, set index using
            # table id and sort the index
            ddf[n_tab].columns = p_cols + ['LevelId_'+n_tab] + np_cols
            ddf[n_tab].set_index([p_cols[-1]], drop=False, inplace=True)
            if pd.__version__ == '0.19.2':
                ddf[n_tab].sort_index()
            else:
                ddf[n_tab].sort()

            # Generate Dimensions type to Quantipy type reference
            # dataframe
            if num_np_cols > 0:
                types_df = pd.DataFrame(
                    pd.Series(ddf[n_tab].columns).str.split(':').tolist(),
                    columns=['column', 'type']
                )
                types_df['type'] = types_df['type'].map(DDF_TYPES_MAP)
                types_df.set_index(['type'], drop=True, inplace=True)
                ddf[n_tab].columns = types_df['column']

                # Coerce column dtypes for expected Quantipy usage
                # methods and functions by type
                if 'single' in types_df.index:
                    columns = types_df.ix['single','column']
                    if isinstance(columns, str):
                        columns = [columns]
                    for column in columns:
                        if not ddf[n_tab][column].dtype in [
                                np.int64, np.float64
                                ]:
                            str_col = ddf[n_tab][column].str.strip(";")
                            if pd.__version__ == '0.19.2':
                                num_col = pd.to_numeric(str_col, errors='coerce')
                            else:
                                num_col = str_col.convert_objects(
                                    convert_numeric=True
                                )
                            ddf[n_tab][column] = num_col
                    ddf[n_tab][column].replace(-1, np.NaN, inplace=True)

                if 'date' in types_df.index:
                    columns = types_df.ix['date','column']
                    if isinstance(columns, str):
                        columns = [columns]
                    for column in columns:
                        ddf[n_tab][column] = get_datetime_values(
                            ddf[n_tab][column],
                            adjuster=0,
                            as_string=False
                        )

                if 'boolean' in types_df.index:
                    columns = types_df.ix['boolean','column']
                    if isinstance(columns, str):
                        columns = [columns]
                    for column in columns:
                        ddf[n_tab][column] = ddf[n_tab][column].astype('bool')

            clean.update({n_tab: ddf[n_tab]})

    return clean, ddf['Levels']


def force_single_from_delimited(data):

    data = data.apply(lambda x: x.str.replace(';', ''))
    data = data.convert_objects(convert_numeric=True)
    return data


def as_L1(child, parent=None, force_single=False):

    if parent is None:

        child_as_L1 = child.copy()
        id_L1 = ['id_HDATA']
        level_ids = [
            c for c in child_as_L1.columns
            if c.startswith('LevelId')
        ]
        np_cols = [
            c for c in child_as_L1.columns
            if not c.startswith('id')
            and not c.startswith('LevelId')
        ]
        for level_id in level_ids:
            grid_name = level_id[8:]
            child_as_L1[level_id] = (
                grid_name+'~'+child_as_L1[level_id].astype('str')
            )
        child_as_L1 = child_as_L1[id_L1+level_ids+np_cols].set_index(
            id_L1+level_ids,
            drop=True,
            inplace=False
        ).unstack(1)
        if force_single:
            child_as_L1 = force_single_from_delimited(child_as_L1)

    else:

        p_cols = [
            c for c in parent.columns
            if c.startswith('id')
        ]
        level_id = [
            c for c in parent.columns
            if c.startswith('LevelId')
        ]
        np_cols = [
            c for c in child.columns
            if not c.startswith('id')
            and not c.startswith('LevelId')
        ]

        parent_level_id = parent[p_cols+level_id]
        parent_level_id.set_index(p_cols, drop=True, inplace=True)
        index_name = child.index.name
        child.set_index(p_cols, drop=False, inplace=True)
        child = child.join(parent_level_id[level_id])
        child.set_index(index_name, drop=False, inplace=True)

        id_L1 = ['id_HDATA']
        level_ids = [c for c in child.columns if c.startswith('LevelId')]

        child_as_L1 = child[id_L1+level_ids+np_cols].copy()
        for level_id in level_ids:
            grid_name = level_id[8:]
            new_grid_name = grid_name+'~'+child_as_L1[level_id].astype('str')
            child_as_L1[level_id] = new_grid_name
        child_as_L1.set_index(id_L1+level_ids, drop=True, inplace=True)
        child_as_L1 = child_as_L1.unstack([2,1])
        if force_single:
            child_as_L1 = force_single_from_delimited(child_as_L1)

    return child_as_L1


def get_var_type(var):

    mdd_type = MDD_TYPES_MAP[var.get('type')]
    if mdd_type=='categorical':
        if var.get('max')=='1':
            mdd_type = 'single'
        else:
            mdd_type = 'delimited set'

    return mdd_type


def get_text_dict(source):

    text = {
        l.get('{http://www.w3.org/XML/1998/namespace}lang'): l.text
        for l in source
    }
    for tk in text.keys():
        if text[tk] is None:
            text[tk] = ""
    return text


def get_meta_values(xml, column, data, map_values=True):

    if '.' in column['name']:
        grid_elem, var_name = column['name'].split('.')
        grid_name = grid_elem.split('[')[0]
        is_grid = True
    else:
        is_grid = False
        var_name = column['name']

    column_values = []
    column_factors = []

    if is_grid:
        # this protects against the scenario where multiple grids
        # contain the same-named field. in this situation the variable
        # (the field) needs to be identified by its ref id as taken
        # from the grid/loop definition
        xpath_grid = "//design//grid[@name='{}']".format(grid_name)
        if not xml.xpath(xpath_grid):
            xpath_grid = "//design//loop[@name='{}']".format(grid_name)
        xpath_field = xpath_grid+"//variable[@name='{}']".format(var_name)
        field = xml.xpath(xpath_field)[0]
        field_ref = field.get('ref')
        xpath_var = XPATH_DEFINITION+"//variable[@id='"+field_ref+"']"
        xpath_categories = xpath_var+"//categories//category"

    else:
        xpath_var = XPATH_DEFINITION+"//variable[@name='"+var_name+"']"
        xpath_categories = xpath_var+"//categories//category"

    categories = xml.xpath(xpath_categories)

    # First, figure out the most appropriate way to derive the
    # values, attempting in this order byName, byProperty.NativeValue
    # byProperty.Value, byPosition
    byName = True
    byName_values = []
    byProperty = True
    byProperty_key = None
    byProperty_values = []
    for cat in categories:
        cat_name = cat.get('name')
        mapped_value = re.search('a(minus)?[0-9]+$', cat_name)
        if mapped_value is None:
            byName = False
        else:
            byName_values.append(mapped_value.group(0)[1:])

        xpath_category = xpath_var+"//categories//category[@name='"+cat_name+"']"
        xpath_properties = xpath_category+"//properties//property"
        properties = xml.xpath(xpath_properties)
        if properties is None or len(properties) == 0:
            byProperty = False
        else:
            byProperty_values.append({
                prop.get('name'): prop.get('value')
                for prop in properties
            })

    if byName:
        if len(byName_values) != len(set(byName_values)):
            byName = False
        try:
            byName_values = [int(v.replace('minus', '-')) for v in byName_values]
        except:
            byName = False

    if not byName and byProperty:
        if all(['NativeValue' in bpv for bpv in byProperty_values]):
            byProperty_key = 'NativeValue'
            byProperty_values = [bpv['NativeValue'] for bpv in byProperty_values]
            if len(byProperty_values) != len(set(byProperty_values)):
                byProperty = False
                byProperty_values = []
                byProperty_key = None                
        elif all(['Value' in bpv for bpv in byProperty_values]):
            byProperty_key = 'Value'
            byProperty_values = [bpv['Value'] for bpv in byProperty_values]
            if len(byProperty_values) != len(set(byProperty_values)):
                byProperty = False
                byProperty_values = []
                byProperty_key = None
        else:
            byProperty = False
            byProperty_values = []
            byProperty_key = None

    if byName:
        values = [int(v) for v in byName_values]
        msg = 'Category values for {} will be taken byName.'.format(var_name)
    elif byProperty:
        values = [int(v) for v in byProperty_values]
        msg = 'Category values for {} will be taken byProperty using {}.'.format(
                var_name, byProperty_key)
    else:
        values = range(1, len(categories)+1)
        msg = 'Category values for {} will be taken byPosition'.format(var_name)
        warnings.warn(msg)

    # handy trouble-shooting printout for figuring out where category values
    # have come from.
    # print msg, values
    # print values
    value_map = {}
    for i, cat in enumerate(categories):
        value = {}
        cat_name = cat.get('name')

        try:
            value['factor'] = float(cat.get('factor-value'))
        except:
            pass

        xpath_category = xpath_categories+"[@name='"+cat_name+"']"
        xpath_category_label_text = xpath_category+"//labels//text"
        value['text'] = get_text_dict(xml.xpath(xpath_category_label_text))
        value['properties'] = get_meta_properties(xml, xpath_category)

        cat_name_lower = cat_name.encode('utf-8').lower().decode('utf-8')
        xpath_categoryid_lower = (
            XPATH_CATEGORYMAP+"//categoryid[@name='"+cat_name_lower+"']")
        xpath_categoryid = (
            XPATH_CATEGORYMAP+"//categoryid[@name='"+cat_name+"']")
        try:
            category = xml.xpath(xpath_categoryid_lower)[0]
        except IndexError:
            category = xml.xpath(xpath_categoryid)[0]

        ddf_value = int(category.get('value'))
        if not map_values:
            value['value'] = ddf_value
        else:
            value['value'] = values[i]

        value_map[ddf_value] = value['value']
        column_values.append(value)

    return column_values, value_map


def remap_values(data, column, value_map):
    if column['type'] in ['single']:
        missing = [
            value
            for value in data[column['name']].dropna().unique()
            if value not in value_map.keys()
            and value not in [-1]]
        if missing:
            msg = (
                "Unknown category ids {} for '{}' found in the ddf."
                " The data for these category ids will not be converted "
                "because there is no corresponding metadata.").format
            warnings.warn(msg(missing, column['name']))

        data[column['name']] = data[column['name']].map(value_map)

        return data[column['name']].copy()

    elif column['type'] in ['delimited set']:
        temp = data[column['name']][data[column['name']].notnull()]
        if temp.size>0:
            value_map = {str(k): str(v) for k, v in value_map.iteritems()}
            temp = temp.apply(
                lambda x: map_delimited_values(x, value_map, column['name']))
            data[column['name']].update(temp)

        return data[column['name']].copy()

    return False


def map_delimited_values(y, value_map, col_name):
    """
    Map the delimited values using the given mapper, dropping unknown responses.
    """

    msg = (
        "Unknown category id '{}' for '{}' found in the ddf."
        " The data for this category id will not be converted "
        "because there is no corresponding metadata.").format

    if y == ';':
        return y

    # add artificial leading ; to help secure matching
    y = ';{}'.format(y)

    # seek ;-bound matches
    # replace with ;_X_; bounded secure edits (prevents compound edits)
    seek = ';{};'.format
    repl = ';_{}_;'.format

    for value in y.split(';')[1:-1]:
        if value in value_map:
            # replace values with secure edits
            y = y.replace(seek(value), repl(value_map[value]))
        else:
            warnings.warn(msg(value, col_name))
            # tag all data to be removed
            y = y.replace(seek(value), ';X;')

    # remove compounded edit security bounds
    y = y.replace('_', '')
    # remove deleted data
    y = y.replace('X;', '')
    # remove aritifial leading ; if there are any responses left
    if y.startswith(';') and len(y) > 1: y = y[1:]

    return y


def begin_column(xml, col_name, data):

    column = {}

    xpath_var = XPATH_DEFINITION+"//variable[@name='"+col_name+"']"
    var = xml.xpath(xpath_var.decode('utf-8'))[0]
    column['name'] = col_name
    column['properties'] = get_meta_properties(xml, xpath_var)
    xpath__col_text = xpath_var+"//labels"
    column['text'] = get_text_dict(xml.xpath(xpath__col_text.decode('utf-8'))[0].getchildren())
    column['parent'] = {}
    column['type'] = get_var_type(var)
    if column['type'] in ['delimited set']:
        xpath_categories = xpath_var+"//categories//category"
        categories = xml.xpath(xpath_categories)
        if len(categories)==1:
            column['type'] = 'single'

    return column


def get_meta_properties(xml, xpath_var, exclude=None):

    if exclude is None:
        exclude = [
            'SqlClmnName',
            'TableName',
            '__SYS',
            'NativeValue',
            'Value'
        ]

    try:
        properties = {
            e.get('name'): e.get('value')
            for e in xml.xpath(xpath_var+"//properties".decode('utf-8'))[0]
            if not e.get('name') in exclude
        }
    except IndexError:
        properties = {}

    return properties


def map_cols_from_grid(xml, data):

    needs_mapping = False
    mapped_columns = {}
    for c in data.columns:
        if isinstance(c, tuple):
            if len(c)==2:
                l1 = c[1].split("~")
                l1_grid_name = l1[0]
                l1_element_name = xml.xpath(
                    "//categorymap//categoryid[@value='%s']" % l1[1].rstrip(';')
                )[0].get('name')
                field_name = c[0]
                mapped_columns[c] = '%s[{%s}].%s' % (
                    l1_grid_name,
                    l1_element_name,
                    field_name
                )
            elif len(c)==3:
                l1 = c[1].split("~")
                l1_grid_name = l1[0]
                l1_element_name = xml.xpath(
                    "//categorymap//categoryid[@value='%s']" % l1[1].rstrip(';')
                )[0].get('name')
                l2 = c[2].split("~")
                l2_grid_name = l2[0]
                l2_element_name = xml.xpath(
                    "//categorymap//categoryid[@value='%s']" % l2[1].rstrip(';')
                )[0].get('name')
                field_name = c[0]
                mapped_columns[c] = '%s[{%s}].%s[{%s}].%s' % (
                    l1_grid_name,
                    l1_element_name,
                    l2_grid_name,
                    l2_element_name,
                    field_name
                )
            if not needs_mapping: needs_mapping = True
        else:
            mapped_columns[c] = c

    if needs_mapping:
        data.columns = pd.Series(data.columns).map(mapped_columns)

    return data


def get_mdd_xml(path_mdd):

    with open(path_mdd, 'r+') as f:
        xml_text = f.read()
    recovering_parser = etree.XMLParser(recover=True)
    xml = etree.parse(StringIO(xml_text), parser=recovering_parser)

    return xml


def get_grid_elements(xml, grid_name):

    xpath_elements = XPATH_LOOPS+"[@name='"+grid_name+"']//categories"
    categories = xml.xpath(xpath_elements)
    if categories:
        elements = categories[0].getchildren()
    else:
        xpath_elements = XPATH_GRIDS+"[@name='"+grid_name+"']//categories"
        elements = xml.xpath(xpath_elements)[0].getchildren()

    return elements, xpath_elements


def get_columns_meta(xml, meta, data, map_values=True):

    columns = {}

    for col_name in data.columns[1:]:

        if '[{' in col_name:

            tmap = re.findall(RE_GRID_SLICES, col_name)
            mm_name = '.'.join(tmap[::2])

            column = begin_column(xml, tmap[-1], data)
            column['name'] = col_name

            if not mm_name in meta['lib']['values']:
                column_values, value_map = get_meta_values(
                    xml, column, data, map_values
                )
                meta['lib']['values'][mm_name] = column_values
                if map_values:
                    meta['lib']['values']['ddf'][mm_name] = value_map

            values_mapper = 'lib@values@%s' % mm_name
            column['values'] = values_mapper
            parent_map = {'masks@{}'.format(mm_name): {'type': 'array'}}
            column['parent'] = parent_map

            if map_values and column['type'] in ['single', 'delimited set']:
                data[column['name']] = remap_values(
                    data, column, meta['lib']['values']['ddf'][mm_name]
                )

            if not mm_name in meta['masks']:
#                 xpath_grid = "//design//grid[@name='%s']" % mm_name
                xpath_grid = "//design//grid[@name='%s']" % mm_name.split('.')[0]
                if not xml.xpath(xpath_grid):
                    xpath_grid = "//design//loop[@name='%s']" % mm_name.split('.')[0]
                xpath_grid_text = '%s//labels//text' % xpath_grid
                try:
                    texts = [
                        text
                        for text in xml.xpath(xpath_grid_text)
                        if text.getparent().getparent().tag in ['grid', 'loop']
                    ]
                    grid_text = get_text_dict(texts)
                except:
                    grid_text = column['text']
                meta['masks'].update({
                    mm_name: {
                        'text': grid_text,
                        'type': 'array',
                        'subtype': column['type'],
                        'items': [],
                        'values': values_mapper,
                        'properties': get_meta_properties(xml, xpath_grid)
                        }
                    })

            try:
                xpath_properties = "//design//category[@name='%s']//properties" % (tmap[1])
                xpath_elements = xpath_grid+"//categories//category"
                elem_name = None
                for element in xml.xpath(xpath_elements):
                    if element.get('name').lower() == tmap[1].lower():
                        elem_name = element.get('name')
                if elem_name is None:
                    raise KeyError("Grid element '{}' not gound in grid '{}'.".format(tmap[1], col_name))
                xpath_labels = xpath_elements+"[@name='%s']//labels//text" % (elem_name)
                sources = xml.xpath(xpath_labels)
                element_text = {
                    source.get('{http://www.w3.org/XML/1998/namespace}lang'):
                    "" if source.text is None else source.text
                    for source in sources}
            except:
                element_text = tmap[1]
            if element_text is None:
                element_text = ""
            meta['masks'][mm_name]['items'].append({
                'source': 'columns@%s' % col_name,
                'text': element_text,
                'properties': get_meta_properties(xml, xpath_properties)
            })

        else:
            column = begin_column(xml, col_name, data)
            if column['type'] in ['single', 'delimited set']:
                # if get_values:
                column_values, value_map = get_meta_values(
                    xml, column, data, map_values
                )
                column['values'] = column_values
                if map_values:
                    meta['lib']['values']['ddf'][column['name']] = value_map
                    data[column['name']] = remap_values(data, column, value_map)

        columns[col_name] = column

    return meta, columns, data


def mdd_to_quantipy(path_mdd, data, map_values=True):

    meta = {}

    meta['type'] = 'pandas.DataFrame'

    meta['info'] = {}
    meta['info']['from_source'] = path_mdd,
    meta['info']['name'] = path_mdd.split('/')[-1].split('.')[0]

    meta['lib'] = {}
    meta['lib']['values'] = {}
    meta['lib']['values']['ddf'] = {}

    meta['masks'] = {}

    xml = get_mdd_xml(path_mdd)
    data = map_cols_from_grid(xml, data)

    default_text = xml.xpath('//languages')[0].get('base').split('-')
    meta['lib']['default text'] = '-'.join([
        default_text[0].lower(),
        default_text[1]
    ])

    meta, columns, data = get_columns_meta(xml, meta, data, map_values=True)

    meta['columns'] = columns

    meta['sets'] = {}

    array_masks= {
        k: v
        for k, v in meta['masks'].iteritems()
        if v['type']=='array'
    }

    for k in array_masks.keys():

        array_set = []
        tmap = k.split('.')

        # try:
        xpath_grid_text = ''.join([
            XPATH_GRIDS,
            "[@name='"+tmap[0]+"']//labels//text"]
        )
        sources = xml.xpath(xpath_grid_text)
        # except IndexError:
        if not sources:
            xpath_grid_text = ''.join([
                XPATH_LOOPS,
                "[@name='"+tmap[0]+"']//labels//text"]
            )
            sources = xml.xpath(xpath_grid_text)

        texts = [
            text
            for text in sources
            if text.getparent().getparent().tag in ['grid', 'loop']
        ]
        grid_text = get_text_dict(texts)

        if len(tmap)==2:

            l1_elements, xpath_l1_categories = get_grid_elements(xml, tmap[0])

            for l1_element in [e for e in l1_elements if e.tag=='category']:
                l1_element_name = l1_element.get('name')
                xpath_category_label_text = (
                    "%s//category[@name='%s']//labels//text" % (
                        xpath_l1_categories,
                        l1_element_name
                    )
                )
                l1_element_text = get_text_dict(
                    xml.xpath(xpath_category_label_text)
                )

                full_name = '%s[{%s}].%s' % (
                    tmap[0],
                    l1_element_name.lower(),
                    tmap[1]
                )
                array_set.append('columns@%s' % full_name)

                if not full_name in data.columns:
                    data[full_name] = np.NaN
                    meta['columns'][full_name] = {
                        'name': full_name,
                        'values': 'lib@values@%s' % (k),
                        'type': 'single'
                    }

                compound_text = {
                    k: ' - '.join([grid_text[k], l1_element_text[k]])
                    for k in grid_text.keys()
                    if k in l1_element_text
                }
                for key in compound_text.keys():
                    if compound_text[key] is None:
                        compound_text[key] = ""
                meta['columns'][full_name]['text'] = compound_text

        elif len(tmap)==3:

            l1_elements, xpath_l1_categories = get_grid_elements(xml, tmap[0])
            l2_elements, xpath_l2_categories = get_grid_elements(xml, tmap[1])

            for l1_element in [e for e in l1_elements if e.tag=='category']:
                l1_element_name = l1_element.get('name')
                l1_element_text = xml.xpath(
                    xpath_l1_categories,
                    "//category[@name='"+l1_element_name+"']//labels//text"
                )[0].text

                for l2_element in [e for e in l2_elements if e.tag=='category']:
                    l2_element_name = l2_element.get('name')
                    l2_element_text = xml.xpath(
                        xpath_l2_categories,
                        "//category[@name='"+l2_element_name+"']//labels//text"
                    )[0].text

                    full_name = '%s[{%s}].%s[{%s}].%s' % (
                        tmap[0],
                        l1_element_name.lower(),
                        tmap[1],
                        l2_element_name.lower(),
                        tmap[2]
                    )
                    array_set.append('columns@%s' % full_name)

                    if not full_name in data.columns:
                        data[full_name] = np.NaN
                        meta['columns'][full_name] = {
                            'name': full_name,
                            'values': 'lib@values@%s' % (k),
                            'type': 'single'
                        }
                    meta['columns'][full_name]['text'] = " - ".join([
                        grid_text, l1_element_text, l2_element_text
                    ])

        meta['sets'][k] = {'items': array_set}

    meta['sets']['data file'] = {}
    meta['sets']['data file']['text'] = 'As per the data file'
    design_set = [
        e.get('name')
        for e in xml.xpath('//design//fields')[0].getchildren()
        if e.tag in ('variable','loop','grid')
        and (
            e.get('name') in data.columns
            or any([
                m.startswith(e.get('name'))
                for m in meta['masks'].keys()
            ])
        )
    ]
    updated_design_set = []
    for name in design_set:
        if name in data.columns:
            updated_design_set.append('columns@%s' % name)
        for k in meta['masks'].keys():
            if k.startswith('%s.' % name):
                updated_design_set.append('masks@%s' % k)
                set_items = meta['sets'][k]['items']
                mask_items = meta['masks'][k]['items']
                mask_items = [
                    item['source']
                    for item in mask_items]
                meta['sets'][k]['items'] = [
                    item
                    for item in set_items
                    if item in mask_items]
                meta['masks'][k]['items'] = [
                    get_mask_item(meta['masks'][k], item, k)
                    for item in meta['sets'][k]['items']
                    if item in mask_items
                ]
#                 meta['masks'][k]['items'] = [
#                     {'source': i}
#                     for i in meta['sets'][k]['items']
#                 ]

    meta['sets']['data file']['items'] = updated_design_set

    data = order_by_meta(
        data,
        meta['sets']['data file']['items'],
        meta['masks']
    )

    return meta, data


def get_mask_item(mask, source, k):
    for item in mask['items']:
        if item['source']==source:
            return item


def quantipy_from_dimensions(path_mdd, path_ddf, fields='all', grids=None):

    ddf, levels = quantipy_clean(ddf_to_pandas(path_ddf))
    L1 = ddf['HDATA'].copy()
    L1.drop('LevelId_HDATA', axis=1, inplace=True)
#     L1.dropna(axis=1, how='all', inplace=True)

    if isinstance(fields, (list, tuple)):
        L1 = L1[['id_HDATA']+fields]

    if grids is None:
        grids = levels.query("ParentName=='HDATA'").index.tolist()

    if grids is not None:
        single_level = []
        two_level = []
        empty_grids = []
        for grid_name in grids:
            if not any(levels['ParentName'].isin([grid_name])):
                parent_name = levels.loc[grid_name, 'ParentName']
                if grid_name in ddf.keys():
                    single_level.append(as_L1(child=ddf[grid_name]))
                else:
                    empty_grids.append(grid_name)
            else:
                child_name = levels[levels['ParentName']==grid_name].index[0]
                if grid_name in ddf.keys():
                    two_level.append(as_L1(
                        child=ddf[child_name],
                        parent=ddf[grid_name],
                        force_single=True)
                    )
                else:
                    empty_grids.append(grid_name)
        if single_level:
            L1 = L1.join(pd.concat(single_level, axis=1))
        if two_level:
            L1 = L1.join(pd.concat(two_level, axis=1))
        if empty_grids:
            print '\n*** Empty grids %s ignored ***\n' % (', '.join(empty_grids))

    meta, ddf = mdd_to_quantipy(path_mdd, data=L1)

    for mask in meta['masks'].keys():
        meta['masks'][mask]['items'] = [
            item
            for item in meta['masks'][mask]['items']
            if not item is None
        ]

    for key, col in meta['columns'].iteritems():
        if col['type']=='string' and key in ddf:
            ddf[key] = ddf[key].apply(qp.core.tools.dp.io.unicoder)
        if col['type']=='int' and key in ddf:
            ddf[key] = ddf[key].replace('null', 0)

    mdd, ddf = verify_columns(meta, ddf)

    return meta, ddf


def verify_columns(mdd, ddf):
    """
    Ensure all columns in the data appear in the meta.
    """

    droplist = []
    for col in mdd['columns'].keys():
        if not col in ddf.columns:
            droplist.append(col)
            del mdd['columns'][col]

    if droplist:
        print 'Found in data, not in meta (columns removed):', droplist

    return mdd, ddf


def order_by_meta(data, columns, masks):
    """
    Check and re-order data.columns against meta['sets']['data file']['items'].
    """
    def _get_column_items(columns, masks):
        result = []
        for item in columns:
            column = item.split('@')[1]
            if column in masks:
                items = [item['source'] for item in masks[column]['items']]
                result.extend(_get_column_items(items, []))
            else:
                result.append(column)
        return result
    new_order = ["id_L1"]
    new_order.extend(_get_column_items(columns, masks))
    data = data.ix[:, new_order]
    return data
