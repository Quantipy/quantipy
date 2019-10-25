#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Created on 20 Nov 2014

@author: JamesG

Cleanup: KMUE, Oct 2019
"""

from ....__imports__ import *  # noqa

from ...meta import Meta

logger = get_logger(__name__)

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
RE_GRID_SLICES = "[^{.]+(?=[}]|$|\[)"  # noqa
XPATH_DEFINITION = '//definition'
XPATH_VARIABLES = '//design//fields//variable'
XPATH_LOOPS = '//design//fields//loop'
XPATH_GRIDS = '//design//fields//grid'
XPATH_CATEGORYMAP = '//categorymap'


def ddf_to_pandas(path_ddf):
    """
    Returns a dict of pandas DataFrames from the given Dimensions
    case data file (DDF), which is a sqlite file.
    """
    with sqlite3.connect(path_ddf) as conn:
        tables_df = pd.read_sql('SELECT * FROM sqlite_master;', conn)
        sql = {
            table_name: pd.read_sql('SELECT * FROM ' + table_name + ';', conn)
            for table_name in tables_df['tbl_name'].values
            if table_name.startswith('L')
        }
        table_info = {}
        for table_name in sql.keys():
            table_info[table_name] = pd.read_sql(
                "PRAGMA table_info('" + table_name + "');",
                conn
            )
        sql['table_info'] = table_info

    if 'Levels' in sql:
        sql['Levels'].set_index(['TableName'], drop=True, inplace=True)
    else:
        err = (
            "The 'Levels' table was not found. Your DDF may be empty or "
            "corrupt.")
        logger.error(err); raise KeyError(err)

    ddf = {
        'table_info': sql['table_info'].copy(),
        'Levels': sql['Levels'].copy(),
        'HDATA': sql['L1'].copy()
    }

    levels = sql['Levels']
    table_name_map = dict(levels['DSCTableName'])
    table_name_map['L1'] = 'HDATA'
    new_levels_index = ['HDATA']
    for table_name in levels.index[1:]:
        new_table_name = levels.ix[table_name, 'DSCTableName']
        ddf[new_table_name] = sql[table_name]
        new_levels_index.append(new_table_name)

    ddf['Levels'].index = pd.Index(new_levels_index, name='table_name')
    ddf['Levels'].drop('DSCTableName', axis=1, inplace=True)
    ddf['Levels']['ParentName'] = ddf['Levels']['ParentName'].map(
        table_name_map)
    ddf['Levels']['ParentName'] = ['None'] + [
        v for v in ddf['Levels']['ParentName'][1:]]
    return ddf


def timestamp_to_ISO8610(timestamp, offset_date="1900-01-01", as_string=False,
                         adjuster=None):

    offset = np.datetime64(offset_date).astype("float") * DAYS_TO_MS
    day = timestamp * DAYS_TO_MS
    date = (day + offset).astype("datetime64[ms]")
    if adjuster is not None:
        date = date - adjuster
    if as_string:
        date = str(date)
    return date


def get_datetime_values(var_df, adjuster, as_string=True):

    dates = var_df.astype(float).apply(
        timestamp_to_ISO8610, args=(
            "1899-12-30", as_string, np.timedelta64(adjuster, 'm'))
    )
    if as_string:
        return list(dates.str.encode('utf-8').values)
    else:
        return dates


def quantipy_clean(ddf):
    clean = {}
    data_table_keys = [
        k for k in ddf.keys() if k not in ['table_info', 'Levels']]
    for n_tab in data_table_keys:
        if ddf[n_tab].shape[0] > 0:

            # Map parent columns for heirarchical tables
            p_cols = ['id_' + n_tab]
            child = n_tab
            while True:
                parent = ddf['Levels'].loc[child, 'ParentName']
                if parent == 'None':
                    break
                id = 'id_' + parent
                p_cols = [id] + p_cols
                child = parent

            # Identify non-parent columns in the table,
            # skip to next table if none
            num_np_cols = ddf[n_tab].columns.size - (len(p_cols) + 1)
            if num_np_cols > 0:
                np_cols = list(ddf[n_tab].columns[len(p_cols) + 1:])
            else:
                np_cols = []

            # Apply parent-mapped column names, set index using
            # table id and sort the index
            ddf[n_tab].columns = p_cols + ['LevelId_' + n_tab] + np_cols
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
                    columns = types_df.ix['single', 'column']
                    if isinstance(columns, str):
                        columns = [columns]
                    for column in columns:
                        if not ddf[n_tab][column].dtype in [np.int64,
                                                            np.float64]:
                            str_col = ddf[n_tab][column].str.strip(";")
                            if pd.__version__ == '0.19.2':
                                num_col = pd.to_numeric(
                                    str_col, errors='coerce')
                            else:
                                num_col = str_col.convert_objects(
                                    convert_numeric=True
                                )
                            ddf[n_tab][column] = num_col
                    ddf[n_tab][column].replace(-1, np.NaN, inplace=True)

                if 'date' in types_df.index:
                    columns = types_df.ix['date', 'column']
                    if isinstance(columns, str):
                        columns = [columns]
                    for column in columns:
                        ddf[n_tab][column] = get_datetime_values(
                            ddf[n_tab][column],
                            adjuster=0,
                            as_string=False
                        )

                if 'boolean' in types_df.index:
                    columns = types_df.ix['boolean', 'column']
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
            if c.startswith('LevelId')]
        np_cols = [
            c for c in child_as_L1.columns
            if not (c.startswith('id') or c.startswith('LevelId'))]
        for level_id in level_ids:
            grid_name = level_id[8:]
            child_as_L1[level_id] = (
                grid_name + '~' + child_as_L1[level_id].astype('str'))
        child_as_L1 = child_as_L1[id_L1 + level_ids + np_cols].set_index(
            id_L1 + level_ids, drop=True, inplace=False).unstack(1)
        if force_single:
            child_as_L1 = force_single_from_delimited(child_as_L1)

    else:
        p_cols = [c for c in parent.columns if c.startswith('id')]
        level_id = [c for c in parent.columns if c.startswith('LevelId')]
        np_cols = [
            c for c in child.columns
            if not (c.startswith('id') or c.startswith('LevelId'))]

        parent_level_id = parent[p_cols + level_id]
        parent_level_id.set_index(p_cols, drop=True, inplace=True)
        index_name = child.index.name
        child.set_index(p_cols, drop=False, inplace=True)
        child = child.join(parent_level_id[level_id])
        child.set_index(index_name, drop=False, inplace=True)

        id_L1 = ['id_HDATA']
        level_ids = [c for c in child.columns if c.startswith('LevelId')]

        child_as_L1 = child[id_L1 + level_ids + np_cols].copy()
        for level_id in level_ids:
            grid_name = level_id[8:]
            new_grid_name = grid_name + '~' + child_as_L1[level_id]
            child_as_L1[level_id] = new_grid_name.astype('str')
        child_as_L1.set_index(id_L1 + level_ids, drop=True, inplace=True)
        child_as_L1 = child_as_L1.unstack([2, 1])
        if force_single:
            child_as_L1 = force_single_from_delimited(child_as_L1)

    return child_as_L1


def get_values(xml, name, data, map_values=True):
    if '.' in name:
        mask, field = name.split(".")
        mask = mask.split('[')[0]
        xpath_grid = "//design//grid[@name='{}']".format(mask)
        if not xml.xpath(xpath_grid):
            xpath_grid = "//design//loop[@name='{}']".format(mask)
        xpath_field = "{}//variable[@name='{}']".format(xpath_grid, field)
        f_ref = xml.xpath(xpath_field)[0].get('ref')
        xpath_var = "{}//variable[@id='{}']".format(XPATH_DEFINITION, f_ref)
    else:
        xpath_var = "{}//variable[@name='{}']".format(XPATH_DEFINITION, name)

    xpath_categories = "{}//categories//category".format(xpath_var)
    categories = xml.xpath(xpath_categories)

    # First, figure out the most appropriate way to derive the
    # values, attempting in this order byName, byProperty.NativeValue
    # byProperty.Value, byPosition
    byName = []
    byProp = []
    # byProp_key = None
    for cat in categories:
        cat_name = cat.get('name')
        # byName?
        mapped_value = re.search('a(minus)?[0-9]+$', cat_name)
        if mapped_value:
            byName.append(mapped_value.group(0)[1:])
        # byProperty
        xpath_category = "{}//categories//category[@name='{}']".format(
            xpath_var, cat_name)
        xpath_properties = "{}//properties//property".format(xpath_category)
        props = xml.xpath(xpath_properties)
        if props:
            byProp.append({p.get('name'): p.get('value') for p in props})

    if byName:
        if len(byName) != len(set(byName)):
            byName = []
        try:
            byName = [int(v.replace('minus', '-')) for v in byName]
        except TypeError:
            byName = []
    elif byProp:
        if all(['NativeValue' in bp for bp in byProp]):
            # byProp_key = 'NativeValue'
            byProp = [int(bp['NativeValue']) for bp in byProp]
        elif all(['Value' in bp for bp in byProp]):
            # byProp_key = 'Value'
            byProp = [int(bp['Value']) for bp in byProp]
        else:
            byProp = []
        if len(byProp) != len(set(byProp)):
            byProp = []
            # byProp_key = None
    values = byName or byProp
    if not values:
        values = range(1, len(categories) + 1)
        msg = 'Category values for {} will be taken byPosition'
        logger.warning(msg.format(name))

    column_values = []
    value_map = {}
    for val, cat in zip(values, categories):
        value = {}
        cat_name = cat.get('name')
        try:
            value['factor'] = float(cat.get('factor-value'))
        except TypeError:
            pass
        xpath_category = "{}[@name='{}']".format(xpath_categories, cat_name)
        xpath_category_label_text = "{}//labels//text".format(xpath_category)
        value['text'] = get_text_dict(xml.xpath(xpath_category_label_text))
        value['properties'] = get_properties(xml, xpath_category)
        value['value'] = val
        try:
            cat_name_lower = cat_name.lower()
            xpath_catid_lower = "{}//categoryid[@name='{}']".format(
                XPATH_CATEGORYMAP, cat_name_lower)
            category = xml.xpath(xpath_catid_lower)[0]
        except IndexError:
            xpath_catid = "{}//categoryid[@name='{}']".format(
                XPATH_CATEGORYMAP, cat_name)
            category = xml.xpath(xpath_catid)[0]
        value_map[int(category.get('value'))] = val
        column_values.append(value)

    return column_values, value_map


def remap_values(data, name, qtype, value_map):
    if qtype in ['single']:
        missing = [
            value for value in data[column['name']].dropna().unique()
            if value not in value_map.keys() and value not in [-1]]
        if missing:
            msg = (
                "Unknown category ids {} for '{}' found in the ddf."
                " The data for these category ids will not be converted "
                "because there is no corresponding metadata.").format
            logger.warning(msg(missing, column['name']))
        data[name] = data[name].map(value_map)

    elif qtype in ['delimited set']:
        temp = data[name][data[name].notnull()]
        if temp.size > 0:
            value_map = {str(k): str(v) for k, v in value_map.items()}
            temp = temp.apply(
                lambda x: map_delimited_values(x, value_map, name))
            data[name].update(temp)


def map_delimited_values(y, value_map, col_name):
    msg = (
        "Unknown category id '{}' for '{}' found in the ddf."
        " The data for this category id will not be converted "
        "because there is no corresponding metadata.").format

    if y == ';':
        return y

    y = ';{}'.format(y)
    seek = ';{};'.format
    repl = ';_{}_;'.format

    for value in y.split(';')[1:-1]:
        if value in value_map:
            y = y.replace(seek(value), repl(value_map[value]))
        else:
            logger.warning(msg(value, col_name))
            y = y.replace(seek(value), ';X;')

    # remove compounded edit security bounds
    y = y.replace('_', '')
    # remove deleted data
    y = y.replace('X;', '')
    # remove aritifial leading ; if there are any responses left
    if y.startswith(';') and len(y) > 1:
        y = y[1:]
    return y


def get_text_dict(source):
    text = {
        l.get('{http://www.w3.org/XML/1998/namespace}lang'): l.text or ""
        for l in source
    }
    return text


def get_properties(xml, xpath_var, exclude=None):
    if not exclude:
        exclude = ['SqlClmnName', 'TableName', '__SYS', 'NativeValue', 'Value']
    xml_props = xml.xpath("{}//properties".format(xpath_var))
    properties = {}
    if len(xml_props) > 0:
        for e in xml_props[0]:
            if e.get("name", "") not in exclude:
                properties[e.get("name")] = e.get("value")
    return properties


def map_cols_from_grid(xml, data):

    def _extract(obj):
        mask, value = obj.split("~")
        item = xml.xpath(
            "//categorymap//categoryid[@value='{}']".format(
                value.rstrip(';')))[0].get('name')
        return mask, item

    mapping = {}
    for col in data.columns:
        if isinstance(col, tuple):
            # mapping needed for array items
            if len(col) == 2:
                field = col[0]
                mask, item = _extract(col[1])
                mapping[col] = '{mask}[{{{item}}}].{field}'.format(
                    mask=mask, item=item, field=field)
            elif len(col) == 3:
                field = col[0]
                mask1, item1 = _extract(col[1])
                mask2, item2 = _extract(col[2])
                mapping[col] = '{m1}[{{{i1}}}].{m2}[{{{i2}}}].{f}'.format(
                    m1=mask1, i1=item1, m2=mask2, i2=item2, f=field)
        else:
            # no mapping needed for non array items
            mapping[col] = col
    data.columns = pd.Series(data.columns).map(mapping)


def add_columns(xml, meta, data):
    for name in data.columns[1:]:
        if '[{' in name:
            tmap = re.findall(RE_GRID_SLICES, name)
            xpath_var = "{}//variable[@name='{}']".format(
                XPATH_DEFINITION, tmap[-1])
        else:
            xpath_var = "{}//variable[@name='{}']".format(
                XPATH_DEFINITION, name)
        # column properties
        props = get_properties(xml, xpath_var)
        # column label
        xpath_col_text = "{}//labels".format(xpath_var)
        texts = xml.xpath(xpath_col_text)[0].getchildren()
        label = get_text_dict(texts)
        # type
        var = xml.xpath(xpath_var)[0]
        qtype = MDD_TYPES_MAP[var.get('type')]
        # values
        if qtype == 'categorical':
            values, value_map = get_values(xml, name, data)
            qtype = 'single' if len(values) == 1 else "delimited set"
            data[name] = remap_values(data, name, qtype, value_map)
        else:
            values = False
        column = meta.start_column(name, qtype, "", values=values, prop=props)
        column["text"] = label
        meta["columns"][name] = column
    meta.create_set("data file", [data.columns.tolist()[1:]], overwrite=True)


def add_masks(xml, meta, data):
    masks = {}
    for col in data.columns[1:]:
        if '[{' in col:
            tmap = re.findall(RE_GRID_SLICES, col)
            mask = '.'.join(tmap[::2])
            if mask not in masks:
                masks[mask] = []
            masks[mask].append(col)

    for name, sources in masks.items():
        mask = name.split(".")[0]
        meta.to_array(name, sources, "")
        xpath_grid = "//design//grid[@name='{}']".format(mask)
        if not xml.xpath(xpath_grid):
            xpath_grid = "//design//loop[@name='{}']".format(mask)
        xpath_grid_text = '{}//labels//text'.format(xpath_grid)
        try:
            texts = [
                text for text in xml.xpath(xpath_grid_text)
                if text.getparent().getparent().tag in ['grid', 'loop']]
            labels = get_text_dict(texts)
        except ValueError:
            labels = {}
        for tk, txt in labels.items():
            meta.set_text(name, txt, tk)
        props = get_properties(xml, xpath_grid)
        for prop, val in props.items():
            meta.set_properties(name, prop, val, True)


def order_by_meta(data, meta):
    """
    Check and re-order data.columns against meta['sets']['data file']['items'].
    """
    new_order = ["id_L1"]
    new_order.extend(meta.unroll(meta.variables_from_set("data file")))
    data = data.ix[:, new_order]
    return data


def mdd_to_quantipy(path_mdd, data, map_values=True):
    recovering_parser = etree.XMLParser(recover=True)
    xml = etree.parse(path_mdd, parser=recovering_parser)

    # adjust data columns
    map_cols_from_grid(xml, data)

    # initialize  meta
    meta = Meta()
    meta.dimensions_comp = True
    meta["info"].update({
        'from_source': path_mdd,
        "name": path_mdd.split('/')[-1].split('.')[0]
    })
    # text_key
    tk = xml.xpath('//languages')[0].get('base').split('-')
    meta.text_key = "{}-{}".format(tk[0].lower(), tk[1].upper())
    # fill meta from xml information
    add_columns(xml, meta, data)
    add_masks(xml, meta, data)

    data = order_by_meta(data, meta)
    return meta, data


def quantipy_from_dimensions(path_mdd, path_ddf, fields='all', grids=None):

    ddf, levels = quantipy_clean(ddf_to_pandas(path_ddf))
    L1 = ddf['HDATA'].copy()
    L1.drop('LevelId_HDATA', axis=1, inplace=True)

    if isinstance(fields, (list, tuple)):
        L1 = L1[['id_HDATA'] + fields]

    if not grids:
        grids = levels.query("ParentName=='HDATA'").index.tolist()
    if grids:
        single_level = []
        two_level = []
        empty_grids = []
        for grid_name in grids:
            if not any(levels['ParentName'].isin([grid_name])):
                if grid_name in ddf.keys():
                    single_level.append(as_L1(child=ddf[grid_name]))
                else:
                    empty_grids.append(grid_name)
            else:
                child_name = levels[levels['ParentName'] == grid_name].index[0]
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
            msg = '*** Empty grids {} ignored ***'.format(
                ', '.join(empty_grids))
            logger.info(msg)

    meta, data = mdd_to_quantipy(path_mdd, data=L1)

    for col in meta.ints:
        data[col] = data[col].replace('null', 0)

    meta.to_json("test.json")

    return meta, data
