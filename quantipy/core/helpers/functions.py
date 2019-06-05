import pandas as pd
import numpy as np
import json
import pickle
import re
import copy
import itertools
import math
import re, string

from collections import OrderedDict, defaultdict
from .constants import DTYPE_MAP
from .constants import MAPPED_PATTERN
from itertools import product
import quantipy as qp

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def load_json(path_json, hook=OrderedDict):
    ''' Returns a python object from the json file located at path_json
    '''

    with open(path_json) as f:
        obj = json.load(f, object_pairs_hook=hook)

        return obj

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def loads_json(json_text, hook=OrderedDict):
    ''' Returns a python object from the json string json_text
    '''

    obj = json.loads(json_text, object_pairs_hook=hook)

    return obj

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def save_json(obj, path_json):

    with open(path_json, 'w+') as f:
        json.dump(obj, f)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def df_to_browser(df, path_html='df.html', **kwargs):

    import webbrowser

    with open(path_html, 'w') as f:
        f.write(df.to_html(**kwargs))

    webbrowser.open(path_html, new=2)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_delimited_value_map(ds, ds_split=None, sep=';'):
    ''' Returns a sorted list of unique values found in ds, being a series
    storing delimited set data separated by sep

    ds - (pandas.Series) a series storing delimited set data
    ds_split - (pandas.DataFrame, optional) an Excel-style text-to-columns
        version of ds
    sep - (str, optional) the character/s to use to delimit ds
    '''

    if ds_split is None:
        ds_split = ds.dropna().str.split(sep)

    delimited = pd.DataFrame(ds_split.tolist())
    value_map = pd.unique(delimited.values.ravel())
    value_map = np.sort(value_map[value_map.nonzero()])

    return value_map


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def verify_dtypes_vs_meta(data, meta):
    ''' Returns a df showing the pandas dtype for each column in data compared
    to the type indicated for that variable name in meta plus a 'verified'
    column indicating if quantipy determines the comparison as viable.

    data - (pandas.DataFrame)
    meta - (dict) quantipy meta object
    '''

    dtypes = data.dtypes
    dtypes.name = 'dtype'
    var_types = pd.DataFrame({k: v['type'] for k, v in meta['columns'].items()}, index=['meta']).T
    df = pd.concat([var_types, dtypes.astype(str)], axis=1)

    missing = df.loc[df['dtype'].isin([np.NaN])]['meta']
    if missing.size>0:
        print('\nSome meta not paired to data columns was found (these may be special data types):\n', missing, '\n')

    df = df.dropna(how='any')
    df['verified'] = df.apply(lambda x: x['dtype'] in DTYPE_MAP[x['meta']], axis=1)

    return df

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def coerce_dtypes_from_meta(data, meta):

    data = data.copy()
    verified = verify_dtypes_vs_meta(data, meta)
    for idx in verified[~verified['verified']].index:
        meta = verified.loc[idx]['meta']
        dtype = verified.loc[idx]['dtype']
        if meta in ["int", "single"]:
            if dtype in ["object"]:
                data[idx] = data[idx].convert_objects(convert_numeric=True)
            data[idx] = data[idx].replace(np.NaN, 0).astype(int)

    return data

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def index_to_dict(index):

    if isinstance(index, pd.MultiIndex):
        levels = index.levels
        names = index.names
        index_dict = {names[i]: levels[i] for i in range(len(names))}
    else:
        index_dict = {None: index.tolist()}

    return index_dict

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def has_collapsed_axis(df, axis=0):
    # agg_func =  ('cBase', 'cMean', 'rBase', 'rMean', 'net', 'Promoters' , 'Net')
    agg_func = ('cbase', 'rbase', 'effbase', 'mean', 'net', 'promoters')
    if axis == 0:
        if df.index.get_level_values(1)[0].startswith(agg_func):
            return True
    else:
        if df.T.index.get_level_values(1)[0].startswith(agg_func):
            return True

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_view_slicer(meta, col, values=None):

    if values is None:
        slicer = [
            (col, val['value'])
            for val in emulate_meta(meta, meta['columns'][col]['values'])]
    else:
        slicer = [
            (col, val)
            for val in values]

    return slicer

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def paint_index(meta,
                index,
                text_key,
                display_names=False,
                transform_names=None,
                grp_text_map=None):

    single_row = len(index.values)==1
    levels = get_index_levels(index)
    col = levels[0]
    values = list(levels[1])
    if not col in meta['columns']:
        return index
    else:
        col_text = paint_col_text(
            meta, col, text_key, display_names, transform_names)
        values_text = paint_col_values_text(
            meta, col, values, text_key, grp_text_map)

        new_index = build_multiindex_from_tuples(
            col_text,
            values_text,
            ['Question', 'Values'],
            single_row)

        return new_index

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def paint_view(meta, view, text_key=None, display_names=None,
               transform_names=False, axes=['x', 'y']):

    if text_key is None: text_key = finish_text_key(meta, {})
    if display_names is None: display_names = ['x', 'y']

    is_array = any(view.meta()[axis]['is_array'] for axis in ['x', 'y'])

    if is_array:
        df = paint_array(
            meta, view, text_key, display_names, transform_names, axes)
    else:
        df = view.dataframe.copy()
        grp_text_map = view.meta()['agg']['grp_text_map']
        df = paint_dataframe(
            meta, df, text_key, display_names, transform_names, axes,
            grp_text_map)

    return df

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def paint_dataframe(meta, df, text_key=None, display_names=None,
                    transform_names=False, axes=['x', 'y'],
                    grp_text_map=None):

    if text_key is None: text_key = finish_text_key(meta, {})
    if display_names is None: display_names = ['x', 'y']

    if 'x' in axes:
        display_x_names = 'x' in display_names

        if len(df.index.levels[0])>1:
            order = []
            for x in df.index.labels[0]:
                if x not in order:
                    order.append(x)
            levels = df.index.levels[0]
            it = sorted(zip(levels, order), key=lambda x: x[1])
            df.index = pd.concat([
                paint_dataframe(
                    meta, df.ix[[level], :], text_key, display_names,
                    transform_names, 'x', grp_text_map)
                for level, _ in it],
                axis=0).index
        else:
            df.index = paint_index(
                meta, df.index, text_key['x'],
                display_x_names, transform_names, grp_text_map)

    if 'y' in axes:
        display_y_names = 'y' in display_names

        if len(df.columns.levels[0])>1:
            df.columns = pd.concat([
                paint_dataframe(
                    meta, df.ix[:, [level]], text_key, display_names,
                    transform_names, 'y', grp_text_map)
                for level in df.columns.levels[0]],
                axis=1).columns
        else:
            df.columns = paint_index(
                meta, df.columns, text_key['y'],
                display_y_names, transform_names)

    return df

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def paint_array(meta, view, text_key, display_names, transform_names, axes):

    df = view.dataframe.copy()
    grp_text_map = view.meta()['agg']['grp_text_map']
    columns_on_x = view.meta()['x']['is_array']
    axes_x = {True: 'x',
              False: 'y'}

    if 'x' in axes:
        display_x_names = axes_x.get(columns_on_x) in display_names
        index = paint_array_items_index(
            meta,
            df.index if columns_on_x else df.columns,
            text_key['x'],
            display_x_names)
    if 'y' in axes:
        display_y_names = axes_x.get(not columns_on_x) in display_names
        columns = paint_array_values_index(
            meta,
            df.columns if columns_on_x else df.index,
            text_key['y'],
            display_y_names,
            grp_text_map)

    df.index = index if columns_on_x else columns
    df.columns = columns if columns_on_x else index

    return df

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_index_levels(index):

    levels = []
    idx_values = index.values
    single_row = len(idx_values)==1
    if single_row:
        unzipped = [idx_values[0]]
        levels.append(unzipped[0][0])
        levels.append([unzipped[0][1]])
    else:
        unzipped = list(zip(*index.values))
        levels.append(unzipped[0][0])
        levels.append(unzipped[1])

    return levels

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def paint_col_text(meta, col, text_key, display_names, transform_names):

    col_meta = emulate_meta(meta, meta['columns'][col])
    if display_names:
        try:
            col_name = col
            if transform_names: col_name = transform_names.get(col, col)
            col_text = '{}. {}'.format(
                col_name, get_text(col_meta['text'], text_key))
        except UnicodeEncodeError:
            col_text = '{}. {}'.format(
                col_name, qp.core.tools.dp.io.unicoder(
                    get_text(col_meta['text'], text_key),
                    like_ascii=True))
    else:
        col_text = get_text(col_meta['text'], text_key)

    return col_text

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def paint_add_text_map(meta, add_text_map, text_key):

    if add_text_map is None:
        add_text_map = {}
    else:
        try:
            add_text_map = {
                key: get_text(text, text_key)
                for key, text in add_text_map.items()}
        except UnicodeEncodeError:
            add_text_map = {
                key: qp.core.tools.dp.io.unicoder(
                    get_text(text, text_key, like_ascii=True))
                for key, text in add_text_map.items()}

    return add_text_map

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def paint_col_values_text(meta, col, values, text_key, add_text_map=None):
    add_text_map = paint_add_text_map(meta, add_text_map, text_key)
    num_col = meta['columns'][col]['type'] in ['int', 'float']
    try:
        has_all = 'All' in values
        if has_all: values.remove('All')
        if not num_col:
            try:
                values_map = {
                    val['value']: get_text(val['text'], text_key)
                    for val in meta['columns'][col]['values']}
            except UnicodeEncodeError:
                values_map = {
                    val['value']: qp.core.tools.dp.io.unicoder(
                        get_text(val['text'], text_key, like_ascii=True))
                    for val in meta['columns'][col]['values']}
        else:
            values_map = {}
        values_map.update(add_text_map)
        values_text = [values_map[v] for v in values]
    except KeyError:
        values_text = values
    except ValueError:
        values_text = values
    if has_all:
        values_text = ['All'] + values_text

    return values_text

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def paint_mask_text(meta, mask, text_key, display_names):

    mask_meta = meta['masks'][mask]
    if display_names:
        try:
            mask_text = '{}. {}'.format(
                mask, get_text(mask_meta['text'], text_key))
        except UnicodeEncodeError:
            mask_text = '{}. {}'.format(
                mask, qp.core.tools.dp.io.unicoder(
                    get_text(mask_meta['text'], text_key),
                    like_ascii=True))
    else:
        mask_text = get_text(mask_meta['text'], text_key)

    return mask_text

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def paint_array_items_text(meta, mask, items, text_key):

    try:
        has_all = 'All' in items
        items = [i for i in items if not i=='All']
        items_map = {}
        try:
            for item in meta['masks'][mask]['items']:
                if isinstance(item['text'], dict):
                    text = get_text(item['text'], text_key)
                else:
                    source = item['source'].split('@')[-1]
                    text = get_text(meta['columns'][source]['text'], text_key)
                    text = text.replace(
                        '{} - '.format(
                            get_text(meta['masks'][mask]['text'],
                            text_key)),
                        '')
                items_map.update({item['source'].split('@')[-1]: text})
        except UnicodeEncodeError:
            for item in meta['masks'][mask]['items']:
                if isinstance(item['text'], dict):
                    text = qp.core.tools.dp.io.unicoder(
                        get_text(item['text'], text_key),
                        like_ascii=True)
                else:
                    source = item['source'].split('@')[-1]
                    text = qp.core.tools.dp.io.unicoder(
                        get_text(meta['columns'][source]['text'], text_key),
                        like_ascii=True)
                    text = qp.core.tools.dp.io.unicoder(
                        text.replace(
                            '{} - '.format(
                                get_text(meta['masks'][mask]['text'],
                                text_key)),
                            ''),
                        like_ascii=True)
                items_map.update({item['source'].split('@')[-1]: text})
        items_text = [items_map[i] for i in items]
        if has_all:
            items_text = ['All'] + items_text
    except KeyError:
        items_text = items
    except ValueError:
        items_text = items

    return items_text

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def paint_array_values_text(meta, mask, values, text_key, add_text_map=None):

    add_text_map = paint_add_text_map(meta, add_text_map, text_key)

    # Values text
    values_meta = emulate_meta(meta, meta['masks'][mask]['values'])
    try:
        has_all = 'All' in values
        if has_all: values.remove('All')
        try:
            values_map = {
                val['value']: get_text(val['text'], text_key)
                for val in values_meta}
        except UnicodeEncodeError:
            values_map = {
                val['value']: qp.core.tools.dp.io.unicoder(
                    get_text(val['text'], text_key,
                    like_ascii=True))
                for val in values_meta}
        values_map.update(add_text_map)
        values_text = [values_map[v] for v in values]
        if has_all:
            values_text = ['All'] + values_text
    except KeyError:
        values_text = values
    except ValueError:
        values_text = values

    return values_text

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def build_multiindex_from_tuples(l0_text, l1_text, names, single_row):

    if single_row:
        new_index = pd.MultiIndex.from_tuples(
            [(l0_text, l1_text[0])], names=names)
    else:
        new_index = pd.MultiIndex.from_product(
            [[l0_text], l1_text], names=names)

    return new_index

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def paint_array_items_index(meta, index, text_key, display_names):

    single_row = len(index.values)==1
    levels = get_index_levels(index)
    mask = levels[0]
    items = levels[1]

    mask_text = paint_mask_text(meta, mask, text_key, display_names)
    items_text = paint_array_items_text(meta, mask, items, text_key)

    new_index = build_multiindex_from_tuples(
        mask_text,
        items_text,
        ['Array', 'Questions'],
        single_row)

    return new_index

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def paint_array_values_index(meta, index, text_key, display_names,
                             grp_text_map=None):

    single_row = len(index.values)==1
    levels = get_index_levels(index)
    mask = levels[0]
    values = levels[1]

    mask_text = paint_mask_text(meta, mask, text_key, display_names)

    values_text = paint_array_values_text(
        meta, mask, values, text_key, grp_text_map)

    new_index = build_multiindex_from_tuples(
        mask_text,
        values_text,
        ['Question', 'Values'],
        single_row)

    return new_index

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_rules(meta, col, axis):
    if col=='@':
        return None
    try:
        if col in meta['columns']:
            rules = meta['columns'][col]['rules'][axis]
        elif col in meta['masks']:
            rules = meta['masks'][col]['rules'][axis]
        return rules
    except:
        return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_rules_slicer(f, rules, copy=True):

    if copy:
        f = f.copy()

    if 'slicex' in rules:
        kwargs = rules['slicex']
        values = kwargs.get('values', None)
#         if not values is None:
#             kwargs['values'] = [val for val in values]
        f = qp.core.tools.view.query.slicex(f, **kwargs)

    if 'sortx' in rules:
        kwargs = rules['sortx']
        fixed = kwargs.get('fixed', None)
        sort_on = kwargs.get('sort_on', '@')
#         if not fixed is None:
#             kwargs['fixed'] = [fix for fix in fixed]
        f = qp.core.tools.view.query.sortx(f, **kwargs)

    if 'dropx' in rules:
        kwargs = rules['dropx']
        values = kwargs.get('values', None)
#         if not values is None:
#             kwargs['values'] = [v for v in values]
        f = qp.core.tools.view.query.dropx(f, **kwargs)
    return f.index.values.tolist()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def apply_rules(df, meta, rules):
    """
    Applies custom rules to df
    """

    # Get names of x and y columns
    col_x = meta['columns'][df.index.levels[0][0]]
    col_y = meta['columns'][df.columns.levels[0][0]]

    # If True was given to rules apply both x and y rules
    if isinstance(rules, bool):
        rules = ['x', 'y']

    if 'x' in rules and df.index.levels[1][0]!='@' and 'rules' in col_x:

        # Get x rules for the x column
        rx = col_x['rules'].get('x', None)

        if not rx is None:

            if 'slicex' in rx:
                kwargs = rx['slicex']
                values = kwargs.get('values', None)
                if not values is None:
                    kwargs['values'] = [str(v) for v in values]
                df = qp.core.tools.view.query.slicex(df, **kwargs)

            if 'sortx' in rx:
                kwargs = rx['sortx']
                fixed = kwargs.get('fixed', None)
                if not fixed is None:
                    kwargs['fixed'] = [str(f) for f in fixed]
                df = qp.core.tools.view.query.sortx(df, **kwargs)

            if 'dropx' in rx:
                kwargs = rx['dropx']
                values = kwargs.get('values', None)
                if not values is None:
                    kwargs['values'] = [str(v) for v in values]
                df = qp.core.tools.view.query.dropx(df, **kwargs)

    if 'y' in rules and df.columns.levels[1][0]!='@' and 'rules' in col_y:

        # Get y rules for the y column
        ry = col_y['rules'].get('y', None)

        if not ry is None:

            if 'slicex' in ry:
                kwargs = ry['slicex']
                values = kwargs.get('values', None)
                if not values is None:
                    kwargs['values'] = [str(v) for v in values]
                df = qp.core.tools.view.query.slicex(df.T, **kwargs).T

            if 'sortx' in ry:
                kwargs = ry['sortx']
                fixed = kwargs.get('fixed', None)
                if not fixed is None:
                    kwargs['fixed'] = [str(f) for f in fixed]
                df = qp.core.tools.view.query.sortx(df.T, **kwargs).T

            if 'dropx' in ry:
                kwargs = ry['dropx']
                values = kwargs.get('values', None)
                if not values is None:
                    kwargs['values'] = [str(v) for v in values]
                df = qp.core.tools.view.query.dropx(df.T, **kwargs).T

    return df

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def rule_viable_axes(meta, vk, x, y):
    viable_axes = ['x', 'y']
    condensed_x = False
    condensed_y = False

    array_summary = (x in meta['masks'] and y == '@')
    transposed_summary = (y in meta['masks'] and x == '@')
    v_method = vk.split('|')[1]
    relation = vk.split('|')[2]
    s_name = vk.split('|')[-1]
    descriptive = v_method.startswith('.d')
    exp_net = '}+]' in relation
    array_sum_freqs = array_summary and s_name in ['counts', 'c%', 'r%']


    if transposed_summary:
        x, y = y, x

    if (relation.split(":")[0].startswith('x') and not exp_net) or descriptive:
        if not array_summary:
            condensed_x = True
    elif relation.split(":")[1].startswith('y'):
        condensed_y = True
    else:
        if re.search('x\[.+:y$', relation) != None:
            condensed_x = True
        elif re.search('x:y\[.+', relation) != None:
            condensed_y = True
        if re.search('y\[.+:x$', relation) != None:
            condensed_y = True
        elif re.search('y:x\[.+', relation) != None:
            condensed_x = True

    if condensed_x or x=='@': viable_axes.remove('x')
    if condensed_y or (y=='@' and not array_sum_freqs): viable_axes.remove('y')

    return viable_axes

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_text(text, text_key, axis=None):
    """ Uses text_key on text if it is a dictionary, pulling out the targeted
    text. Either way, the resulting text (given directly or pulled out a
    dictionary) is type-checked to ensure <str> or <unicode>

    Parameters
    ----------
    text : <dict>, <OrderedDict>, <str> or <unicode>
    text_key : <str>

    Returns
    ----------
    <str>
    """

    if text is None:
        text = ''

    if isinstance(text, str):
        return text

    elif isinstance(text, (dict, OrderedDict)):

        if axis is None:

            if isinstance(text_key, str):
                if text_key in text:
                    return text[text_key]
            else:
                for key in text_key:
                    if key in text:
                        return text[key]
        else:
            if axis in list(text_key.keys()):
                for key in text_key[axis]:
                    if key in text:
                        return text[key]

        raise KeyError(
            "No matching text key from the list {} was not found in the"
            " text object: {}".format(text_key, text)
        )

    else:
        raise TypeError(
            "The value set into a 'text' object must either be"
            " <str> or <unicode>, or <dict> or <collections.OrderedDict>"
            " of <str> or <unicode>. Found: {}".format(text)
        )

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def finish_text_key(meta, text_key):

    #add default to text_key
    default_text = meta['lib'].get('default text', 'None')
    if text_key is None:
        text_key = {}
    for key in ['x', 'y']:
        if key in list(text_key.keys()):
            if isinstance(text_key[key], str):
                text_key[key] = [text_key[key], default_text]
            elif isinstance(text_key[key], list):
                text_key[key].append(default_text)
            else:
                raise TypeError(
                    "text_key items must be <str> or <list>\n"
                    "Found: %s" % (type(text_key[key]))
                )
        else:
            text_key[key] = [default_text]

    return text_key

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_values(var_meta, meta):

    values = []
    for value in var_meta['values']:

        if isinstance(value, dict):
            values.append(value)

        elif isinstance(value, str):
            values += get_mapped_meta(meta, value)

    return values

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def flatten_list(the_list, deep_flatten=False):

    if deep_flatten:
        flat = list(itertools.chain.from_iterable(the_list))
    else:
        flat = []
        for item in the_list:
            if isinstance(item, (list)):
                for subitem in item:
                    flat.append(subitem)
            else:
                flat.append(item)
        # flat = [item for sublist in the_list for item in sublist]

    return flat

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def is_mapped_meta(item):
    """ Returns True if item is a string conforming to recognised mapped meta
    syntax.
    """

    if isinstance(item, str):
        if item.split('@')[0] in ['lib', 'columns', 'masks', 'info', 'sets']:
            if re.match(MAPPED_PATTERN, item):
                return True

    return False

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_mapped_meta(meta, mapped):
    """ Returns a subset of the meta object as indicated by the given mapped
    meta syntax.

    mapped should be a mapped meta string.
    """

    steps = mapped.split('@')
    key = steps.pop()
    for step in steps:
        if isinstance(meta, list):
            step = int(step)
        meta = meta[step]

    if key in meta:
        if isinstance(meta[key], (dict, OrderedDict)):
            meta = {key: meta[key]}
        else:
            meta = meta[key]

    return meta

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def create_multi_index(item_1, item_2, names=None):
    return pd.MultiIndex.from_product([[item_1], item_2], names=names)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def apply_multi_index_items(view, multi_index_items=None, names=None):
    if multi_index_items is not None:
        for key in multi_index_items:
            item = multi_index_items[key]
            multi_index = create_multi_index(item[0], item[1], names=names)
            setattr(view, key, multi_index)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def emulate_meta(meta, item):
    """ Returns a fully emulated version of item with all mapped meta
    components recursively discovered and replaced.

    item can be either a meta object or a mapped meta string.
    """


    if is_mapped_meta(item):
        item = get_mapped_meta(meta, item)
        item = emulate_meta(meta, item)
        return item

    elif isinstance(item, (list, tuple, set)):
        for n, i in enumerate(item):
            item[n] = emulate_meta(meta, item[n])
        item = flatten_list(item)
        return item

    elif isinstance(item, (dict, OrderedDict)):
        for k in list(item.keys()):
            item[k] = emulate_meta(meta, item[k])
        return item

    else:
        return item

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def as_datetime64(data, date_format='dmy', date_sep='/', time_format='hm', time_sep=':'):
    """ Converts data, a dtype 'object' Series storing date or datetime
    as text, to a numpy.datetime64 Series object.

    The default argument values work for a Series in the format:
        "02/07/2014 16:58" as dtype object
    and will return:
        "2014-07-02 15:58:00" as dtype numpy.datetime64

    There are still some issues with UTC that will need to be ironed out.
    """

    has_time = ' ' in data[0]
    date_reorder = [date_format.index(p) for p in 'ymd']

    if has_time:
        if 's' not in time_format:
            data = data + ':00'
        date_time = list(zip(*(data.str.split(' '))))
        time = pd.Series(date_time[1])
        date = pd.Series(date_time[0])

    date = date.str.split(date_sep).apply(lambda x: '-'.join([x[p] for p in date_reorder]))

    if has_time:
        date_time = (date +' '+ time).astype(np.datetime64)

    return date_time

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_time_from_datetime(data):
    """ Returns a timedelta64 series the form "hh:mm:ss" from data, being a
    datetime64 series
    """

    time = data.apply(lambda x: np.timedelta64(x.hour, 'h') + np.timedelta64(x.minute, 'm') + np.timedelta64(x.second, 's'))
    return time

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def summarize_time(data, raw_output=True):
    """ Summary statistics function for timedelta64, returns a DataFrame
    with the following index:

    min
    median
    mean
    mode
    max
    """

    index_list = ['min', 'median', 'mean', 'mode', 'max']
    summary = [
        data.min(),
        data.median(),
        data.mean(),
        data.mode(),
        data.max()
    ]

    s_describe = data.describe()
    df = pd.DataFrame(pd.concat(summary))
    df[0] = df[0].apply(lambda x: pd.tslib.repr_timedelta64(x))
    df.index = index_list
    df = pd.concat([s_describe, df])
    df.columns = ['Total']

    if not raw_output:
        df.index = pd.MultiIndex.from_product([[data.name], df.index], names=['Question', 'Values'])
    return df

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def single_value_counts(data):
    ''' Replacement for the pandas value_counts() dataframe method. Supports
    aggregation over a 'values' vector to either produce the abs. unweighted distribution (=1)
    or an abs. weighted distribution (=weight factors).
    '''
    margin = data.dropna()[data.columns[1]].sum()
    data = data.dropna()
    df = pd.DataFrame({c: data[data[data.columns[0]] == c][data.columns[1]].sum() for c in data[data.columns[0]].astype(int).unique()}, index = ['@1'])
    if margin == 0:
        df = pd.DataFrame(0, ['nan'], ['nan']).T
        df = df.T
    df['All'] = margin
    return df

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def single_value_counts_groups(data, groupby):
    ''' Wrapper for groupby sub-sample aggregation.
    '''
    if groupby:
        grouped = data.dropna().groupby(groupby)
        groups = sorted(grouped.groups.keys())
        if list(grouped.groups.keys()):
            df = pd.concat([single_value_counts(group_data[1][[data.columns[0], data.columns[2]]]) for group_data in grouped], axis=0).T
            df.columns = [int(g) for g in groups]
        else:
            df = pd.DataFrame(0, ['nan'], ['nan']).T
            df['All'] = 0
            df = df.T
        return df
    else:
        pass


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def calc_margins(data, multi=False):
    '''
    For future development: method that is used to compute the correct margins for the
    default meta-related aggregations in base.py. In particular, the following must be covered:
        - compute both col and row margins
        - compute weighted margins
        - handle multi-code and array (sets, hierarchical structures, ...) data correctly
    Currently only used for testing in base.py
    '''
    if multi:
        pass
    else:
        #margin_values = data
        margin_values = data[data[data.columns[0]].notnull()][data.columns[1]].sum()
        #margin_values = data[data[data.columns[0]].notnull()][data.columns[1]].sum()

    return margin_values

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def delimited_set_value_counts(data, value_map=None):
    ''' Now supports weighted aggregation.
    '''
    if value_map is None:
        value_map = get_delimited_value_map(data[data.columns[0]])
    margin = data.dropna()[data.columns[1]].sum()
    data = data.dropna()
    df = pd.DataFrame({k: data[data[data.columns[0]].str.contains("(^{k};)|(;{k};)".format(k=k), regex=True)][data.columns[1]].sum() for k in value_map}, index=['@1'])
    df['All'] = margin
    return df.T

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def delimited_set_value_counts_groups(data, groupby, value_map=None):
    ''' Wrapper for groupby sub-sample aggregation.
    '''
    grouped = data.dropna().groupby(groupby)
    groups = sorted(grouped.groups.keys())
    if list(grouped.groups.keys()):
        df = pd.concat([delimited_set_value_counts(group_data[1][[data.columns[0], data.columns[2]]], value_map) for group_data in grouped], axis=1)
        df.columns = [int(g) for g in groups]
        if len(df.index) == 1:
            df = pd.DataFrame(0, ['nan'], ['nan'])
            df['All'] = 0
            df = df.T
    else:
        df = pd.DataFrame(0, ['nan'], ['nan']).T
        df['All'] = 0
        df = df.T
    return df

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def dichotomous_set_value_counts(data):
    ''' Now supports weighted aggregation.
    '''
    cdata = data[data.columns[:-1]].replace(2, np.NAN).mul(data[data.columns[-1]], axis=0)
    df = pd.DataFrame(pd.concat([cdata.sum(), pd.Series({'All': cdata.T.count().count()})]))
    df.columns = ['@1']
    return df

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def array_value_counts(data, array_name, array_items, weight='@1', aggfunc='sum'):

    ndf = pd.concat([data.pivot_table(values=weight, index=[p], margins=True, aggfunc=aggfunc) for p in array_items], axis=1, keys=array_items)
    ndf.index = pd.MultiIndex.from_product([[array_name], ndf.index], names=['Array', 'Values'])
    ndf.columns = pd.MultiIndex.from_product([[array_name], ndf.columns], names=['Array', 'Item'])
    return ndf

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_views(qp_structure):
    ''' Generator replacement for nested loops to return all view objects
        stored in a given qp container structure.
        Currently supports chain-classed shapes and cluster objects natively.
        To return views from a stack object instance provide input container as per
        qp_structure = < stack[data_key]['data'] >
    '''

    for k, v in qp_structure.items():
        if not isinstance(v, qp.View):
            for item in get_views(v):
                yield item
        else:
            yield v

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_links(qp_structure):
    ''' Generator replacement for nested loops to return all link objects
        stored in a given qp container structure.
        Currently supports chain-classed shapes and cluster objects natively.
        To return views from a stack object instance provide input container as per
        qp_structure = < stack[data_key]['data'] >
    '''

    for k, v in qp_structure.items():
        if not isinstance(v, qp.Link):
            for item in get_links(v):
                yield item
        else:
            yield v

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def create_combinations_from_array(array):
    """ Takes an array and creates a list of combinations from it.

        Unlike itertools.combinations, this creates both xs of combinations
        example:
            >>>list(itertools.combinations(['A', 'B', 'C'])):
            [('A', 'B'),
             ('A', 'C'),
             ('B', 'C')]
            >>>create_combinations_from_array(['A', 'B', 'C'])
            [('A', 'B'),
             ('A', 'C'),
             ('B', 'A'),
             ('B', 'C'),
             ('C', 'A'),
             ('C', 'B')]
    """
    return [ (item_1, item_2) for item_1 in array for item_2 in array if item_1 is not item_2]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def array_items_from_meta(meta, variable):
    if variable in meta['masks']:
        return [var['source'].split('@')[1] for var in meta['masks'][variable]['items'] if 'source' in var ]
    else:
        return []

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_numeric_indexcodes(source_df, axis='x'):
    codes = source_df.index.get_level_values(-1) if axis == 'x' else source_df.T.index.get_level_values(-1)
    if codes.all() == 'total':
        codes = [1]
    elif codes.all() == 'None':
        return False
    else:
        codes = codes.astype(int).tolist()
    return codes

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def default_meta(link, view_df, dtypes, weights=None):
    ''' Used to define the view meta information organized as a dictionary.
    Gets called in base.py. Will ( --- currently --- ) only work if there is
    meta data related to the qp input dataframe.
    '''

    if dtypes[0] in ['single', 'dichotomous set', 'categorical set', 'delimited set']:
        datatype = 'categorical'
        method = 'default'
    elif dtypes[0] in ['float', 'int']:
        datatype = 'numeric'
        method = 'default'
    elif dtypes[0] in ['array']:
        datatype = 'array'
        method = 'default'
    else:
        datatype = dtypes[0]
        method = 'default'

    default_meta = {
                    'agg': {'method': 'default',
                            'name': 'default',
                            'datatype': datatype,
                            'viewtype': 'quantipy.DefaultView',
                            'is_weighted': True if not weights is None else False,
                            'weights': weights
                            },
                    'x': {
                        'name': link.x,
                        'is_multi': True if dtypes[0] in ['dichotomous set', 'categorical set', 'delimited set'] and not link.x == '@' else False,
                        'is_nested': True if ">" in link.x else False
                        },
                    'y': {
                        'name': link.y,
                        'is_multi': True if dtypes[1] in ['dichotomous set', 'categorical set', 'delimited set'] and not link.y == '@' else False,
                        'is_nested': True if ">" in link.y else False
                        },
                    'shape' : view_df.shape
                    }

    return default_meta

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def update_view_meta(view_df, meta, method_name, name, fullname, method_type, label='', source=None):
    ''' Updates the view meta information based on the selected view methods
    used in the QuantipyViewss class. Gets called in core.viewgenerators.view_maps.py.
    Will only work if there is meta data related
    to the qp input dataframe.
    '''
    meta['agg']['method'] = method_name
    meta['agg']['name'] = name
    meta['agg']['fullname'] = fullname
    meta['agg']['viewtype'] = method_type
    meta['agg']['label'] = label
    meta['shape'] = view_df.shape

    return meta

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def create_NA_view(x, y):
    ''' Creates an empty view without data and meta.
    '''
    df = pd.DataFrame(data = ['N/A'], index = [x], columns=[y])
    view = View(df, meta = 'unavailable')
    return view

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def describe(data, x, weights=None):
    ''' Replacment of (wrapper around) the df.describe() method that can deal with
    weighted data. Weight vectors are allowed to be non-normalized, i.e.
    sum of weights <> number of cases in sample. Quartile information currently
    dropped from output, variance is unbiased variance.

    Calculations are identical to SPSS Statistics/Professional.
    '''
    data = data.copy().dropna(subset=[x])
    desc_df = data[x].describe()
    desc_df.rename(
        {
        'count': 'Count',
        'min': 'Min',
        'max': 'Max',
        'mean': 'Mean',
        'std': 'StdDev',
        '25%': 'Lower quartile',
        '50%': 'Median',
        '75%': 'Upper quartile'
        },
        inplace=True)
    # percentile information (incorrect for weighted data!) excluded for now...
    # desc_df.drop(['Lower quartile', 'Median', 'Upper quartile'], inplace=True)
    if not len(data.index) == 0:
        if not weights == '@1':
            count = data[weights].sum()
            norm_wvector_coef = 1 if len(data.index) == count else len(data.index)/count
            w_squared_sum = (data[weights]**2).sum()
            eff_count = count**2/w_squared_sum
            mean = data[x].mul(data[weights].mul(norm_wvector_coef)).mean()
            var = data[weights].mul((data[x].sub(mean))**2).sum()/(data[weights].sum()-1)
            try:
                stddev = math.sqrt(var)
                if abs(stddev) == 0.00:
                    stddev = np.NaN
            except:
                stddev = np.NaN
            desc_df['Count'] = count
            desc_df['Eff. count'] = eff_count
            desc_df['Weights squared sum'] = w_squared_sum
            desc_df['Mean'] = mean
            desc_df['StdDev'] = stddev
        else:
            desc_df['Eff. count'] = desc_df['Count']
            desc_df['Weights squared sum'] = 1.00

        desc_df['Efficiency'] = desc_df['Eff. count']/desc_df['Count']*100

    return pd.DataFrame(desc_df[['Count', 'Eff. count', 'Min', 'Max', 'Mean', 'StdDev', 'Weights squared sum', 'Efficiency']])

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_var_nest_combo(var):
    ''' Returns the y var and it's nest if nested

        example a:
            var, nest = get_var_nest_combo(var="A>B>C>D")
            # var is 'A'
            # nest is 'B>C>D'

        example b:
            var, nest = get_var_nest_combo(var="A")
            # var is 'A'
            # nest is None
    '''
    if '>' in var:
        split = var.split('>')
        return (split[0], '>'.join(split[1:]))
    else:
        return (var, None)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def create_multiindex_from_length(name, length, names=['Question', 'Values'], margins=True):
    if margins:
        multiindex = pd.MultiIndex(levels=[[name], [str(s) for s in range(1, length + 1)] + ['All']],
                                   labels=[[0]*(length + 1), list(range(length + 1))],
                                    names=names)
    else:
        multiindex = pd.MultiIndex(levels=[[name], [str(s) for s in range(1, length + 1)]],
                                   labels=[[0]*length, list(range(length))],
                                    names=names)
    return multiindex

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def cast_index_levels_as_str(df, x_eq_y=False):
    index_levels = [[str(item) for item in level] for level in df.index.levels]
    df.index.set_levels(index_levels, inplace=True)

    column_levels = index_levels if x_eq_y else [[str(item) for item in level] for level in df.columns.levels]
    df.columns.set_levels(column_levels, inplace=True)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def create_nested_multiindex_from_array(array, data, meta, names=['Question', 'Values'], margins=True):
    if len(names) <= len(array):
        names = names * len(array)
    product = [list(item) for sublist in [[[x], get_values_from_categorical(meta['columns'][x]['values'], meta)] for x in array] for item in sublist]
    if margins:
        product[-1].append('All')

    return pd.MultiIndex.from_product(product, names=names)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def create_nest_meta(x, y, data, meta=None):
    """ Creates dictionary with information about unique sizes of variables in the data

        Note : Skip var if `var not in data.columns`

        Example:
            input:
                x = "A>B>C"
                y = "D>E>F"
                data = <<pandas.DataFrame>>

            output: (random values)
                dict =
                {
                    'A':{'items': [1, 2, 3, 4]}
                    'B':{'items': ['Jon', 'Atli']}
                    'C':{'items': [1, 2, 3, 4, 5]}
                    'D':{'items': ['Jan', 'Feb', 'Mar']}
                    'E':{'items': [1, 2, 3, 4]}
                    'F':{'items': [1, 2]}
                }
    """
#    return {var:{'items':sorted(data[var].unique())
#                 'length':len(data[var].unique())} for var in x.split('>') + y.split('>') }
    if meta is None:
        return None
    if x in list(meta['columns'].keys()) and not meta['columns'][x]['type'] in ['int', 'float'] and not meta['columns'][y]['type'] in ['int', 'float']:
        return {var:{'items': get_values_from_categorical(meta['columns'][var]['values'], meta)} for var in x.split('>') + y.split('>') if var in data}
    else:
        return {var:{'items': sorted(data[var].unique())} for var in x.split('>') + y.split('>') if var in data}
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def is_nested_with_arrays(link):
    if '>' in link.y + link.x:
        meta = link.get_meta()
        masks = meta['masks']
        variables = link.y.split('>') + link.x.split('>')
        variable_masks = [mask for mask in [key for key in masks if 'array' in masks[key]['type']] if mask in variables]
        return len(variable_masks) > 0
    return False

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def deep_drop(df, targets, axes=[0, 1]):

    if not isinstance(targets, (list, tuple)):
        targets = [targets]

    if not isinstance(axes, (list, tuple)):
        axes = [axes]

    levels = (len(df.index.levels), len(df.columns.levels))

    for axis in axes:
        for level in range(1, levels[axis])[::2]:
            for target in targets:
                df = df.drop(target, axis=axis, level=level)

    return df

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_unique_level_values(index):
    """
    Returns the unique values for all levels of an Index object, in the
    correct order.

    Parameters
    ----------
    index : pandas.core.index/ pandas.core.index.MultiIndex
        Index object

    Returns
    -------
    list of lists
    """
    return [
        index.get_level_values(i).unique().tolist()
        for i in range(len(index.levels))]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def translate(items, text_keys):

    transmap = qp.View()._metric_name_map()
    translations = []
    for item in items:
        found = False
        for text_key in text_keys:
            if not found:
                try:
                    translation = transmap[text_key][item]
                    found = True
                except:
                    translation = item
        translations.append(translation)

    return translations

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def recode_into(data, col_from, col_to, assignment, multi=False):
    ''' Recodes one column based on the values of another column
    codes = [([10, 11], 1), ([8, 9], 2), ([1, 2, 3, 5, 6, 7, ], 3)]
    data = recode_into(data, 'CONNECTIONS4', 'CONNECTIONS4_nps', codes)
    '''

    s = pd.Series()
    for group in assignment:
        for val in group[0]:
            data[col_to] = np.where(data[col_from] == val, group[1], np.NaN)
            s = s.append(data[col_to].dropna())
    data[col_to] = s
    return data

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def create_column(name, type_name, text='', values=None):
    ''' Returns a column object that can be stored into a Quantipy meta
    document.
    '''

    column = {
        'name': name,
        'type': type_name,
        'text': text
    }

    if not values is None:
        column['values'] = values

    return column

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def find_variable_level(levels, variable):
    """ Returns the first level found for variable in a MultiIndexLevel

        Example:
            >> df.columns.levels
                FrozenList([[u'aldurflo'], [u'1.0', u'2.0', u'3.0', u'All']])
            >> level = find_variable_level(df.columns.levels, 'All')
               # level is 1

       Warning: It will crash if the variable is not in the MultiIndex.Level
    """
    for idx, level in enumerate(levels):
        if variable in level:
            return idx
    return None
    # return [idx for idx, level in enumerate(levels) if variable in level][0]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def categorical_value_counts(df, is_single=False, x_is_multi=False, y_is_multi=False):
    """ Given a pd.DataFrame with x/y variables and a values vector (i.e. @1 vs. weights),
    this method produces basic frequency tables for single- and multi-coded categorical data.
    If the three optional arguments are omitted, x/y/values with single-coded data columns is assumed.

    The method requires a metadata file associated with the input dataframe.
    It is used in base.py with all arguments set accordingly to the respective aggregation types.

    Parameters
    ----------
    df : pandas.DataFrame
        df consisting of either x, values columns (for @1 aggregation, single links)
        or x, y, values columns (for cross-tabulated bivariate links).
        Example: df[['q1','gender','weight']]

    is_single : boolean, optional
        Indicates creation of single links.

    x_is_multi : boolean, optional
        Indicates a multi-coded question on the index axis of the dataframe.

    y_is_multi : boolean, optional
        Indicates a multi_coded question on the column axis of the dataframe.

    Returns
    -------
    quantipy-multiindexed dataframe following the Question/Values convention
    """

    # 1D links, @1 aggregates
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_single:
        if len(df.dropna().index) == 0:
            ct = pd.DataFrame(0,['@1'],['nan']).T
            margin = 0
        else:
            margin = df.dropna()[df.columns[-1]].sum()
            if x_is_multi:
                ct = pd.DataFrame(df[df.columns[0]].str.get_dummies(';').astype(int).mul(df[df.columns[-1]], axis=0).sum(axis=0),
                                  columns=['@1'])
            else:
                ct = pd.DataFrame(pd.get_dummies(df[df.columns[0]].dropna().astype(int)).mul(df[df.columns[-1]], axis=0).sum(axis=0),
                                  columns=['@1'])


    # 2D links, bivariate aggregates
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    else:
        if len(df.dropna().index) == 0:
            ct = pd.DataFrame(0,['nan'],['nan']).T
            margin = 0
        else:
            if x_is_multi and y_is_multi:
                df = df.dropna()
                dummy_x_df = df[df.columns[0]].str.get_dummies(';').astype(int).mul(df[df.columns[-1]], axis=0)
                dummy_y_df = df[df.columns[1]].str.get_dummies(';').astype(int)
                dummy_y_df_columns = dummy_y_df.columns
                dummy_y_df.columns = [df.columns[1] + '_' + code for code in dummy_y_df.columns]
                dummy_full_df = pd.concat([dummy_x_df,dummy_y_df, df[df.columns[-1]]], axis=1)
                margin = [dummy_full_df[dummy_full_df[code] == 1][df.columns[-1]].sum(axis=0) for code in dummy_y_df.columns]
                ct = pd.concat([dummy_full_df[dummy_full_df[code]==1][dummy_x_df.columns].sum(axis=0) for code in dummy_y_df.columns], axis=1)
                ct.columns = dummy_y_df_columns

            elif y_is_multi:
                dummy_x_df = pd.DataFrame(pd.get_dummies(df[df.columns[0]]).mul(df[df.columns[-1]], axis=0))
                dummy_y_df = df[df.columns[1]].str.get_dummies(';').astype(int)
                dummy_y_df_columns = dummy_y_df.columns
                dummy_y_df.columns = [df.columns[1] + '_' + code for code in dummy_y_df.columns]
                dummy_full_df =  pd.concat([dummy_x_df,dummy_y_df], axis=1)
                ct = pd.concat([dummy_full_df[dummy_full_df[code]==1][dummy_x_df.columns].sum(axis=0) for code in dummy_y_df.columns], axis=1)
                ct.index = ct.index.astype(int)
                margin = ct.sum(axis=0).values
                ct.columns = dummy_y_df_columns

            elif x_is_multi:
                df = df.dropna()
                dummy_x_df = df[df.columns[0]].str.get_dummies(';').astype(int).mul(df[df.columns[-1]], axis=0)
                dummy_y_df = pd.DataFrame(pd.get_dummies(df[df.columns[1]].astype(int)))
                dummy_y_df_columns = dummy_y_df.columns
                dummy_y_df.columns = [df.columns[1] + '_' + str(code) for code in dummy_y_df.columns]
                dummy_full_df = pd.concat([dummy_x_df,dummy_y_df, df[df.columns[-1]]], axis=1)
                margin = [dummy_full_df[dummy_full_df[code] == 1][df.columns[-1]].sum(axis=0) for code in dummy_y_df.columns]
                ct = pd.concat([dummy_full_df[dummy_full_df[code]==1][dummy_x_df.columns].sum(axis=0) for code in dummy_y_df.columns], axis=1)
                ct.columns = dummy_y_df_columns

            else:
                df = df.dropna()
                ct = pd.crosstab(index=df[df.columns[0]].astype(int), columns=df[df.columns[1]].astype(int), values=df[df.columns[2]],aggfunc='sum')
                margin = ct.sum(axis=0)

    # create All = margin index/column
    ct = ct.T
    ct['All'] = margin
    ct = ct.T
    ct['All'] = ct.sum(axis=1) # the row margins are currently incorrect for all multicode data

    # apply multiindex confirming the Question/Values convention for both index and column axis
    ct.index = pd.MultiIndex.from_product([[df.columns[0]], ct.index.astype(str)], names=['Question','Values'])
    ct.columns = pd.MultiIndex.from_product([[df.columns[-2]], ct.columns.astype(str)], names=['Question','Values'])

    return ct

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def flatten(stack, separator='-', prefix=''):
    """ Will return a one-level dictionary of all stack keys
    seperated by '-' matched to their concluding value.
    Example: data_key-data-no_filter-x_key-y_key-view_key : View.View
    (This is straight from stackoverflow.com)
    """
    return {prefix + separator + k if prefix else k : v
            for kk, vv in list(stack.items())
            for k, v in list(flatten(vv, separator, kk).items())
            } if isinstance(stack, dict) else {prefix : stack}

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def create_range_as_list(rng, as_type=str):
    """

    Parameters
    ----------
    range : str
        The range to create as a list.
        e.g.
            range="1-4" --> [1, 2, 3, 4]
            range="1-3,5,7-9,1001" --> [1, 2, 3, 5, 7, 8, 9, 1001]
    as_type : type
        map list to as_type. default = str

    Returns
    -------
    list
    """
    res = []
    for sub_rng in rng.split(','):
        if '-' in sub_rng:
            lo, hi = sub_rng.split('-')
            res.extend([i for i in range(int(lo), int(hi)+1)])
        else:
            res.append(sub_rng)
    if as_type==str:
        return res
    return list(map(as_type, res))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def add_combined_codes_view(create_method, source, combine_codes, label):
    """
    Returns a view object

    Parameters
    ----------
    create_method : quantipy.core.viewgenerators.view.QuantipyViews() method
        method used to create view
    source : str
        source name
    combine_codes: list
        list of codes (int) to combine
    label: str
        view label

    Returns
    -------
    quantipy.ViewMapper
    """
    view=ViewMapper()
    if isinstance(combine_codes[0], list):
        for i in range(max(len(combine_codes), len(label))):
            name = 'net_%s_%s_%s_%s' % (combine_codes[i][0],
                                        combine_codes[i][-1],
                                        source,
                                        re.sub(r'['+string.punctuation+']', '',label[i]))
            view.add_method(name.replace(' ', ''),
                            create_method,
                            kwargs={'source': source,
                                    'combine_codes': combine_codes[i],
                                    'label': label[i]})
    else:
        name = 'net_%s_%s_%s_%s' % (combine_codes[0],
                                    combine_codes[-1],
                                    source,
                                    re.sub(r'['+string.punctuation+']', '',label))
        view.add_method(name.replace(' ', ''),
                        create_method,
                        kwargs={'source': source,
                                'combine_codes': combine_codes,
                                'label': label})
    return view

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def hide_codes_x(x, codes_to_hide, views, exclude_from_base=False):
    """

    Parameters
    ----------
    x : quantipy.core.stack.Stack
        e.g. stack[data_key]['data']['no_filter']['Q1']
    codes_to_hide : str
        range of codes (int) to hide
    views : list
        list of views (str)
    exclude_from_base : bool
        0 - do no recalculate base
        1 - recalculate base
    *** NOT IMPLEMENTED YET ***

    Returns
    -------
    """
    for y in x:
        for view in views:
            if view in x[y]:
                x[y][view].meta.update({'x_hidden_codes': create_range_as_list(codes_to_hide),
                                             'x_hidden_in_views': views,
                                             'x_exclude_from_base': exclude_from_base})
            else:
                print('hide_codes_x(): %s not found --> ignored.' % (view))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def reorder_codes_x(x, new_order, views):
    """

    Parameters
    ----------
    x : quantipy.core.stack.Stack
        e.g. stack[data_key]['data']['no_filter']['Q1']
    new_order : list
        new order (int) to hide
    views : list
        list of views (str)

    Returns
    -------
    """
    for y in x:
        for view in views:
            if view in x[y]:
                x[y][view].meta.update({'x_new_order': create_range_as_list(new_order),
                                             'x_new_order_in_views': views})

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def define_multicodes(varlist, meta):
    multicodes = {}
    for var in varlist:
        multicodes.update({var: [mrs_q for mrs_q in meta['columns'] if mrs_q.startswith(var + '_')]})

    return multicodes

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def create_delimited_from_single(data, meta = None, mrs_spec = {}):
    df = data.copy()
    mrs_data = []
    mrs_meta = None
    for name, definition in list(mrs_spec.items()):

        mrs_name = name
        mrs_q = sorted(definition, key=lambda num: int(num.split('_')[-1]))

        mrs_df = df[mrs_q]
        for col in mrs_df.columns:
            mrs_df[col].replace(np.NaN, 0, inplace=True)
            mrs_df[col].replace(1, col.split('_')[1]+';', inplace=True)
        mrs_df[mrs_name] = mrs_df.replace(0, '').sum(axis=1)

        mrs_data.append(mrs_df[mrs_name])

        if not meta is None:
            q_lab = meta['columns'][mrs_q[0]]['text'].split('-', 2)[1].strip()
            values = []

            for q in mrs_q:
                value = int(q.split('_')[-1].strip())
                text = meta['columns'][q]['text'].split('-', 2)[-1].strip()
                values.append({'value': value, 'text': text})

            meta['columns'][mrs_name] = create_column(mrs_name, type_name='delimited set', text=q_lab, values = values)

    data = pd.concat([df] + mrs_data, axis=1)
    return data, meta

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def slice_stack(stack, dks=None, fks=None, xks=None, yks=None,
                          vks=None, att_xks=False, att_yks=True):
    ''' This function will return a copy of stack, keeping only the parts of
    it that were specified, given a lazy rule of everything when nothing is
    specified.

    The att_xks and att_kys parameters are used to allow control over '@' keys
    in the x and y positions of the stack, the default being to remove for x
    and keep for y.
    '''

    stack = copy.deepcopy(stack)

    all_dks = dks is None
    all_fks = fks is None
    all_xks = xks is None
    all_yks = yks is None
    all_vks = vks is None

    if all_dks:
        dks = list(stack.keys())

    for d in dks:

        if not d in list(stack.keys()):
            del stack[d]
        else:
            if all_fks:
                fks = list(stack[d]['data'].keys())

            for f in list(stack[d]['data'].keys()):
                if not f in fks:
                    del stack[d]['data'][f]
                else:
                    if all_xks:
                        xks = list(stack[d]['data'][f].keys())

                    if att_xks and not '@' in xks:
                        xks.append('@')

                    for x in list(stack[d]['data'][f].keys()):
                        if x not in xks:
                            del stack[d]['data'][f][x]
                        else:
                            if all_yks:
                                yks = list(stack[d]['data'][f][x].keys())

                            if att_yks and not '@' in yks:
                                yks.append('@')

                            for y in list(stack[d]['data'][f][x].keys()):
                                if y not in yks:
                                    del stack[d]['data'][f][x][y]
                                else:
                                    if all_vks:
                                        vks = list(stack[d]['data'][f][x][y].keys())

                                    for v in list(stack[d]['data'][f][x][y].keys()):
                                        if v not in vks:
                                            del stack[d]['data'][f][x][y][v]

    return stack

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def map_multiindex(index, levels_map=None, names_map=None):
    ''' This function will map the levels and names of the given MultiIndex
    using the maps provided. It is used to map the index labels and names to
    new labels and names. Note that in Pandas the MultiIndex 'levels' are
    what we would normally think of as the labels.

    Parameters
    ----------
    index : pandas.MultiIndex instance to be mapped
    levels_map : list of dict, the indicies of which corresponding to the
        MultiIndex level for which the given dict will be used as a map.
    names_map : dict to be used to map existing level names to new level
        level names

    Returns : pd.MultiIndex
    -------

    Example:

    >> df

    Question                  profile_gender
    Values                                 1         2
    Question       Values
    profile_bpcage cbase eff        0.846005  0.841076

    df.index = map_multiindex(
        index=df.index,
        levels_map=[{}, {'cbase eff': 'Efficiency'}]
    )

    df.columns = map_multiindex(
        index=df.columns,
        levels_map=[{}, {'1': 'Male'}]
    )

    Question                   profile_gender
    Values                               Male         2
    Question       Values
    profile_bpcage Efficiency        0.846005  0.841076
    '''

    if not isinstance(index, pd.MultiIndex):
        raise TypeError("Index passed is not a MultiIndex, it is '%s'" % type(index))
    else:

        if not levels_map:
            mapped_levels = index.levels
        else:
            if not len(levels_map)==len(index.levels):
                raise IndexError((
                    "The levels_map passed is not the same length as "
                    "the target levels (given %s, expected %s)"
                    ) % (len(levels_map), len(index.levels)))
            else:
                mapped_levels = [[levels_map[l][i] if i in list(levels_map[l].keys()) else i for i in level] for l, level in enumerate(index.levels)]

        if not names_map:
            mapped_names = index.names
        else:
            mapped_names = [names_map[n] if n in list(names_map.keys()) else n for n in index.names]

        mapped_index = pd.MultiIndex(
            levels=mapped_levels,
            labels=index.labels,
            names=mapped_names
        )

        return mapped_index

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def cat_to_dummies(data, limit_to=None, as_df=False):
    '''
    Creates a dichotomously 1/0-coded version of the incoming pd.Series with the answer codes
    found in the data being transformend into column names. This representation of the data situationally
    can be easier to work with if multicoded variables are being fed into computations (i.e. numpy).
    The function unites the two .get_dummies() functions found in pandas, see also:
    - pandas.core.reshape.get_dummies
    - pandas.core.strings.StringMethods.get_dummies

    Parameters
    ----------
    data : pd.Series
        Contanis the single- or multicoded variable that
        should be transformed into the dummy representation.

    limit_to : list, default=None
        List of codes that should be used in the transformation.
        If None (=default), the function will transform all data.

    as_df : bool, default=False
        Controls if the returned output is a pd.DataFrame (=True) or
        a tuple of the DataFrame's values and the column codes (=False).

    Returns
    -------
    tuple : np.array, list of column codes
    OR
    dummy_df : pd.DataFrame
    '''
    if data.dtype == 'object':
        # i.e. Quantipy multicode data
        dummy_df = data.str.get_dummies(';')
        dummy_df.columns = [int(col) for col in dummy_df.columns]
        dummy_df.sort_index(axis=1).rename(columns={col: str(col) for col in dummy_df.columns}, inplace=True)
    else:
        data = data.copy().dropna()
        dummy_df = pd.get_dummies(data)
        dummy_df.rename(columns={col: str(int(col)) if float(col).is_integer() else str(col) for col in dummy_df.columns}, inplace=True)

    if limit_to:
        dummy_df = limit_dummy_df(dummy_df, limit_to)
    if as_df:
        return dummy_df
    else:
        return dummy_df.values, dummy_df.columns.tolist()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#if limit_to is None else data.str.get_dummies(';')[[str(code) for code in limit_to]]
# if limit_to is None else pd.get_dummies(data)[limit_to]
def limit_dummy_df(dummy_df, codes):
    '''
    Limits a 1/0 dummy pd.DataFrame to the given list of column codes.
    Also checks if none existing codes have been passed and removes
    those from the lists if necessary.

    Parameters
    ----------
    dummy_df : pd.DataFrame (dummy-transformed)

    codes : list of integers
        Column codes to keep

    Returns
    -------
    limited : pd.DataFrame
    '''
    if len(dummy_df.index) > 0:
        if not sorted(codes) == sorted(dummy_df.columns):
            codes = [code for code in codes if code in dummy_df.columns]
        return dummy_df[codes]
    else:
        return dummy_df

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def df_to_value_matrix(data, x, y=None, limit_x=None, limit_y=None, weights=None):
    '''
    Transforms a pd.DataFrame into a np.array representation consiting of:
    1. a value column storing either only 1s or weight factors
    2. a dichotomous version of a Quantipy link's x axis' data as an 1/0 matrix
    3. if present: a dichotomous version of a Quantipy link's y axis' data as an 1/0 matrix

    Caution: The np.array contains no naming information, so column data must be inferred from
    a list of codes that matches the structure of the value matrix when working with this function's output.

    Parameters
    ----------
    data : pd.DataFrame
        E.g. Quantipy casedata

    x, y : str, (y: default=False)
        E.g. Quantipy link x and y axes (or other data columns inside a pd.DataFrame).

    limit_x, limit_y : list, default=None
        List of codes that should be ignored in the transformation.

    weighted :  bool, default=False
        Controls if the first column of the matrix contains only 1s or weight factors.

    Returns
    -------
    tuple : value_matrix as np.array, list of x codes, list of y codes
    '''
    values = weights if weights else '@1'
    if not y is None:
        # two variable case, x and y specified
        data = data.copy().replace('', np.NaN).dropna(subset=[x, y])
        wg_vec = data[values].values.reshape(len(data.index), 1)
        x_matrix, x_codes = cat_to_dummies(data = data[x], limit_to=limit_x)
        y_matrix, y_codes = cat_to_dummies(data = data[y], limit_to=limit_y)
        if weights:
            value_matrix = np.concatenate((wg_vec, x_matrix*wg_vec, y_matrix), axis=1)
        else:
            value_matrix = np.concatenate((wg_vec, x_matrix, y_matrix), axis=1)
    else:
        # single variable case, only x specified
        data = data.copy().replace('', np.NaN).dropna(subset=[x])
        wg_vec = data[values].values.reshape(len(data.index), 1)
        x_matrix, x_codes = cat_to_dummies(data[x], limit_to=limit_x)
        y_codes = None
        if weights:
            value_matrix = np.concatenate((wg_vec, x_matrix*wg_vec), axis=1)
        else:
            value_matrix = np.concatenate((wg_vec, x_matrix), axis=1)

    return value_matrix, x_codes, y_codes

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def aggregate_matrix(value_matrix,  x_def, y_def, calc_bases=True, as_df=True):
    '''
    Uses a np.array containing dichotomous values and lists of column codes
    to aggregate frequency tables (and bases if requested) to create basic categorical
    aggregations of uni- or bivariate cell frequencies.

    Parameters
    ----------
    value_matrix : np.array with 1/0 coded values
        I.e. as returned from qp.helpers.aggregation.df_to_value_matrix().

    x_def, y_def : lists of column codes

    calc_bases : bool, default=True
        Controls if the output contains base calculations
        for column, row and total base figures (cb, rb, tb).

    as_df : bool, default=True
        Controls if the output is returned as pd.DataFrame with regular axis indexing
        (and base rows/columns ['All'] if requested).

    Returns
    -------
    agg_df : pd.DataFrame of frequency and base figures
    OR
    tuple : freqs as np.array, list of base figures (column, row, total)
    '''
    # handling empty matrices
    if np.size(value_matrix) == 0:
        empty = True
        freq, cb, rb, tb = np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)
    else:
        empty = False
        xcodes = len(x_def)+1
        if not y_def is None:
            # bivariate calculation (cross-tabulation)
            ycodes = reversed(range(1, len(y_def)+1))
            freq = np.array([np.sum(value_matrix[value_matrix[:, -ycode] == 1][:, 1:xcodes], axis=0)
                       for ycode in ycodes])
            if calc_bases:
                ycodes = reversed(range(1, len(y_def)+1))
                cb = np.array([np.sum(value_matrix[value_matrix[:, -ycode] == 1][:, :1])
                            for ycode in ycodes])
                rb = np.sum(value_matrix[:, 1:xcodes], axis=0)
                tb = np.sum(value_matrix[:, [0]], axis=0)
        else:
            # univariate calculation (frequency table)
            freq = np.sum(value_matrix[:, 1:xcodes], axis=0)
            if calc_bases:
                cb = np.array(np.sum(value_matrix[:, :1]))
                rb = freq
                tb = cb
    # output creation: tuple of np.arrays vs. pd.DataFrame
    if as_df:
        if not empty:
            ixnames = x_def
            colnames = y_def if y_def else ['@']
        else:
            ixnames = ['None']
            colnames = ['None']
        freq_df = pd.DataFrame(data=freq.T, index = ixnames, columns=colnames)
        if calc_bases:
            cb_df = pd.DataFrame(data=[cb], index=['All'], columns=colnames)
            agg_df = pd.concat([freq_df, cb_df], axis=0)
            agg_df['All'] = np.append(rb, tb)
        else:
            agg_df = freq_df

        return agg_df
    else:
        if calc_bases:
            return freq, [cb, rb, tb]
        else:
            return freq

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def apply_viewdf_layout(df, x, y):
    '''
    Takes a pd.DataFrames and applies Quantipy's Question/Values
    layout to it by creating a multiindex on both axes.

    Parameters
    ----------
    df : pd.DataFrame

    x, y : str
        Variable names from the processed case data input,
        i.e. the link definition.

    Returns
    -------
    df : pd.Dataframe (multiindexed)
    '''
    axis_labels = ['Question', 'Values']
    df.index = pd.MultiIndex.from_product([[x], df.index], names=axis_labels)
    if y is None:
        df.columns = pd.MultiIndex.from_product([[x], df.columns], names=axis_labels)
    elif y == '@':
        df.columns = pd.MultiIndex.from_product([[x], '@'], names=axis_labels)
    else:
        df.columns = pd.MultiIndex.from_product([[y], df.columns], names=axis_labels)

    return df

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def make_default_cat_view(data, x, y=None, weights=None):
    '''
    This function is creates Quantipy's default categorical aggregations:
    The x axis has to be a catgeorical single or multicode variable, the y axis
    can be generated from either categorical (single or multicode) or numeric
    (int/float). Numeric y axes are categorized into unique column codes.

    Acts as a wrapper around df_to_value_matrix(), aggregate_matrix() and
    set_qp_multiindex().

    Parameters
    ----------
    data : pd.DataFrame

    x, y : str
        Variable names from the procesFsed case data input,
        i.e. the link definition.

    weighted : bool
        Controls if the aggregation is performed on weighted or weighted data.

    Returns
    -------
    view_df : pd.Dataframe (multiindexed)
    '''
    matrix, x_ref, y_ref = df_to_value_matrix(data=data, x=x, y=y, weights=weights)
    df = aggregate_matrix(matrix, x_ref, y_ref)
    view_df = set_qp_multiindex(df, x, y)

    return view_df

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def make_default_num_view(data, x, y=None, weights=None, get_only=None):
    '''
    This function is creates Quantipy's default numeric aggregations:
    The x axis has to be a numeric variable of type int or float, the y axis
    can be generated from either categorical (single or multicode) or numeric
    (int/float) as well. Numeric y axes are categorized into unique column codes.

    Acts as a wrapper around describe() and set_qp_multiindex().

    Parameters
    ----------
    data : pd.DataFrame

    x, y : str
        Variable names from the processed case data input,
        i.e. the link definition.

    weighted : bool
        Controls if the aggregation is performed on weighted or weighted data.

    Returns
    -------
    view_df : pd.Dataframe (multiindexed)
    '''
    weight = weights if not weights is None else '@1'
    if y is None or y == '@':
        df = describe(data, x, weight)
        df.columns = ['@']
    else:
        data = data[[x, y, weight]].copy().dropna()
        if len(data.index) == 0:
            df = describe(data, x, weight)
            df.columns = ['None']
        else:
            # changing column naming for x==y aggregations
            if not data.columns.is_unique:
                data.columns = [x, y+'_', weight]
            if data[y].dtype == 'object':
                # for Quantipy multicoded data on the y axis
                dummy_y = cat_to_dummies(data[y], as_df=True)
                dummy_y_data = pd.concat([data[[x, weight]], dummy_y], axis=1)
                df = pd.concat([describe(dummy_y_data[dummy_y_data[y_code] == 1], x, weight) for y_code in dummy_y.columns], axis=1)
                df.columns = dummy_y.columns
            else:
                y_codes =  sorted(data[y].unique())
                df = pd.concat([describe(data[data[y] == y_code], x, weight) for y_code in y_codes], axis=1)
                df.columns = [str(int(y_code)) if float(y_code).is_integer() else str(y_code) for y_code in y_codes]

    if get_only is None:
        df['All'] = describe(data, x, weight).values
        c_margin = df.xs('Count')
        df = df.T
        df['All'] = c_margin
        df = df.T
        view_df = set_qp_multiindex(df, x, y)

        return view_df
    else:
        return df.T[get_only].T

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def set_names_to_values(df, names, axis='x'):
    '''
    Changes the inner index's elements to the names specified in the view method definition.
    Helpful to update the 'Values' layer of a multiindexed Quantipy view DataFrame
    after an axis-collapsing aggregation has been applied.

    Parameters
    ----------
    df : Quantipy view dataframe

    name : str or list of strings
        The names to be applied as index values.
        If string is passed, the method converts to list automatically.

    axis : str, default=x
        The link's axis to set the name to.

    Returns
    -------
    df : pd.MultiIndex
    '''
    if not isinstance(names, list):
        names = [names]
    if axis == 'x':
        for name in enumerate(names):
            df.rename(index={df.index.get_level_values(-1)[name[0]]: name[1]}, inplace=True)
        return df.index
    else:
        for name in enumerate(names):
            df.rename(columns={df.columns.get_level_values(-1)[name[0]]: name[1]}, inplace=True)
        return df.columns



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def partition_view_df(view, values=False, data_only=False, axes_only=False):
    '''
    Disassembles a view dataframe object into its
    inner-most index/columns parts (by dropping the first level)
    and the actual data.

    Parameters
    ----------
    view : Quantipy view

    values : boolean, optional
        If True will return the np.array
        containing the df values instead of a dataframe

    data_only : boolean, optional
        If True will only return the data component of the view dataframe

    axes_only : boolean, optional
        If True will only return the inner-most index and columns component
        of the view dataframe.

    Returns
    -------
    data, index, columns : dataframe (or np.array of values), index, columns
    '''
    df = view.copy()
    index = df.index.droplevel() if isinstance(df.index, pd.MultiIndex) else df.index
    columns = df.columns.droplevel() if isinstance(df.columns, pd.MultiIndex) else df.columns
    data = df if not values else df.values

    if data_only:
        return data
    elif axes_only:
        return index.tolist(), columns.tolist()
    else:
        return data, index.tolist(), columns.tolist()

def inherit_axis_codes(df, from_source, to='y'):
    '''
    Renames a dataframe's index or column codes to match the ones
    from the passed source.

    Parameters
    ----------

    df : pd.DataFrame
        The dataframe to be renamed.

    from_source : pd.DataFrame
        The dataframe that acts as the reference.

    to : str, default='y'
        The link axis to that the renaming applies.
        Can be 'x', 'y', 'both'.

    Returns
    -------
    df : pd.DataFrame
    '''
    codes = partition_view_df(from_source, axes_only=True)
    if to == 'x':
        df.rename(index={code[0]: code[1] for code in enumerate(codes[0])}, inplace=True)
        df.rename(columns={code[0]: code[1] for code in enumerate(codes[1])}, inplace=True)
    elif to == 'y':
        df.rename(columns={code[0]: code[1] for code in enumerate(codes[1])}, inplace=True)
    elif to == 'both':
        df.rename(index={code[0]: code[1] for code in enumerate(codes[0])}, inplace=True)
        df.rename(columns={code[0]: code[1] for code in enumerate(codes[1])}, inplace=True)

    return df

def set_qp_multiindex(df, x, y):
    '''
    Takes a pd.DataFrames and applies Quantipy's Question/Values
    layout to it by creating a multiindex on both axes.

    Parameters
    ----------
    df : pd.DataFrame

    x, y : str
        Variable names from the processed case data input,
        i.e. the link definition.

    Returns
    -------
    df : pd.Dataframe (Quantipy convention, multiindexed)
    '''
    axis_labels = ['Question', 'Values']
    df.index = pd.MultiIndex.from_product([[x], df.index], names=axis_labels)
    if y is None:
        df.columns = pd.MultiIndex.from_product([[x], df.columns], names=axis_labels)
    elif y == '@':
        df.columns = pd.MultiIndex.from_product([[x], '@'], names=axis_labels)
    else:
        df.columns = pd.MultiIndex.from_product([[y], df.columns], names=axis_labels)

    return df

def set_view_df_layout(df, x, y, new_names=None, names_to=None, inherit_codes_from=None, codes_to=None):
    '''
    Main function to rebuild the Quantipy view dataframe structure after
    calculations have been processed inside a view method. This functions is
    an all-in-one solution for setting new Values names (e.g. Top2Box), resp.
    renaming axis codes ranges (e.g. 1, 2, 3, 4, 5 instead of 0, 1, 2, 3, 4) and
    applying the Question/Values multiindex convention.
    See also:
    - partition_view_df()
    - set_names_to_values()
    - inherit_axis_codes()
    - set_qp_multiindex()

    Parameters
    ----------
    df : pd.DataFrame

    x, y : str
        Variable names from the processed case data input,
        i.e. the link definition.

    new_names : str or list of str, optional, default=None
        Specification of new names for the Values index.
        Must be matched 1-to-1 by position.

    names_to : str, optional, default=None
        Specification of the link axis new_names is placed to.
        Can be either 'x' or 'y'.

    inherit_codes_from : pd.DataFrame, optional, default=None
        Reference DataFrame that contains index or columns codes that should be applied
        to the new view_df.

    codes_to : str, optional, default=None
        Specifies if the index/column codes from the inherited_codes_from DataFrame
        will be placed on 'x', 'y' or 'both'.

    Returns
    -------
    layouted_df : pd.DataFrame (Quantipy convention, multiindexed)
    '''
    if not new_names is None:
        set_names_to_values(df, new_names, names_to)
    if not inherit_codes_from is None:
        df = inherit_axis_codes(df, inherit_codes_from, codes_to)

    layouted_df = set_qp_multiindex(df, x, y)

    return layouted_df

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_default_num_stat(default_num_view, stat, drop_bases=True, as_df=True):
    '''
    Is used to extract a specific statistical figure from
    a given numerical default aggregation.

    Parameters
    ----------
    default_num_view : Quantipy default view
        (Numerical aggregation case)

    stat : string
        States the figure to extract.

    drop_bases : boolean, optional, default = True
        Controls if the base [= 'All'] column figure is excluded

    as_df : boolean, optional, default = True
        If True will only return as pd.DataFrame, otherwise as np.array.

    Returns
    -------
    pd.DataFrame
    OR
    np.array
    '''
    df = partition_view_df(default_num_view, values=False, data_only=True)
    if drop_bases:
        df = df.drop('All', axis=1).drop('All', axis=0)
    df = df.T[[stat]].T

    if as_df:
        return df
    else:
        df.values

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def calc_net_values(link, source_view, combine_codes, force_raw_sum=False):
    '''
    Used to compute (categorical) net code figures from a given Quantipy link definition,
    a reference view dataframe and a list of codes to build from.
    If the link's aggregation x axis is single coded categorical type, the calculation is
    a simple addition over the qualifying x codes. If x is of type multicode, the result is
    calculated using the value matrix approach (as long force_raw_sum is not set to True).
    See also:
    - cat_to_dummies(), df_to_value_matrix(), aggregate_matrix()
    - make_default_cat_view()

    Parameters
    ----------
    link : Quantipy Link object

    source_view : Quantipy View object
        I.e. a count or pct aggregation

    combine_codes : list of integers
        The list of codes to combine.

    force_raw_sum : bool, optional, default=False
        Controls if the calculation is performed on raw source_view figures.
        This effectively treats every categorical aggregation as single coded and is useful when
        needing to calculate the total responses given insetad of effective qualifying answers.

    Returns
    -------
    net_values : np.array
        Stores the calculated net values
    '''
    if not source_view.meta['x']['is_multi'] or force_raw_sum:
        boolmask = [int(index_val[1]) in combine_codes
                    for index_val in source_view.index
                    if not (isinstance(index_val[1], str)
                    and index_val[1] == 'None')]
        if any(boolmask):
            net_values = np.array(source_view[boolmask].values.sum(axis=0))
        else:
            net_values = np.zeros(1)
    else:
        if not link.y == '@':
            matrix, x_def, y_def = df_to_value_matrix(data=link.get_data(), x=link.x, y=link.y,
                                                      limit_x=combine_codes,
                                                      weights=source_view.meta['agg']['weights'])
            xcodes = len(x_def)+1
            ycodes = reversed(range(1, len(y_def)+1))
            net_values = np.array([matrix[(matrix[:, 1:xcodes].sum(axis=1) > 0) & (matrix[:, -ycode] == 1)][:, 0].sum() for ycode in ycodes])
        else:
            matrix, x_def, y_def = df_to_value_matrix(data=link.get_data(), x=link.x, y=None,
                                                      limit_x=combine_codes,
                                                      weights=source_view.meta['agg']['weights'])
            xcodes = len(x_def)+1
            net_values = np.sum(matrix[matrix[:, 1:xcodes].sum(axis=1) > 0][:, 0])
        if net_values.size == 0:
            net_values = np.zeros(1)

    return net_values

def calc_pct(source, base):
    return pd.DataFrame(np.divide(source.values, base.values)*100)

def get_variable_types(data, meta):
    """ Returns a dict of variable types to lists of variable names.

    Parameters
    ----------
    data : Pandas.DataFrame

    meta : Quantipy meta object pared to data

    Returns
    ----------
    Dict in the form {type_name: [variable_names], ...}

    """
    types = {
        'int': [],
        'float': [],
        'single': [],
        'delimited set': [],
        'string': [],
        'date': [],
        'time': [],
        'array': []
    }

    for col in data.columns[1:]:
        types[meta['columns'][col]['type']].append(col)

    for mask in list(meta['masks'].keys()):
        types[meta['masks'][mask]['type']].append(mask)

    return types

def make_delimited_from_dichotmous(df, use_col_values=False):
    """ Returns a delimited set from the incoming dichotomous
    set dataframe.
    """

    def make_delimited_from_series(s):
        s = s.dropna().astype(int).astype(str)
        delimited = ';'.join(s.tolist()) + ';'
        if delimited == ";":
            delimited = np.NaN
        return delimited

    if use_col_values:
        for i, col in enumerate(df.columns, start=1):
            df[col] = df[col].replace(1, col)
    else:
        for i, col in enumerate(df.columns, start=1):
            df[col] = df[col].replace(1, i)

    delimited_series = df.replace(0, np.NaN).apply(
        make_delimited_from_series,
        axis=1
    )

    return delimited_series

def filtered_set(meta, based_on, masks=True, included=None, excluded=None,
                 strings=None):
    if included is None and excluded is None:
        included = []
        for set_item in meta['sets'][based_on]['items']:
            name = set_item.split('@')[-1]
            if name in meta['columns']:
                included.append(name)
            elif name in meta['masks']:
                for mask_item in meta['masks'][name]['items']:
                    included.append(mask_item['source'].split('@')[1])

    if included is None:
        included = []
    elif isinstance(included, str):
        included = [included]
    if not isinstance(included, (list, tuple, set)):
        raise ValueError (
            "'included' must be either a string or a list, tuple or"
            " set of strings."
        )

    if excluded is None:
        excluded = []
    elif isinstance(excluded, str):
        excluded = [excluded]
    elif not isinstance(excluded, (list, tuple, set)):
        raise ValueError (
            "'excluded' must be either a string or a list, tuple or"
            " set of strings."
        )

    if strings is None:
        strings = 'keep'
    else:
        if not strings in ['keep', 'drop', 'only']:
            raise ValueError (
                "'strings' must be either None, 'keep', 'drop' or"
                "'only'."
            )

    pattern = "\[(.*?)\]"

    items = []
    for item in set(included) - set(excluded) - set(['@']):
        # Account for special strings instruction
        if strings=='keep':
            allow = True
        elif item in meta['columns']:
            is_string = meta['columns'][item]['type']=='string'
            if not is_string and not strings=='only':
                allow = True
            elif not is_string and strings=='only':
                allow = False
            elif is_string and strings=='drop':
                allow = False
            elif is_string and strings=='only':
                allow = True

        if not allow:
            continue

        if 'columns@{}'.format(item) in meta['sets'][based_on]['items']:
            items.append('columns@{}'.format(item))
        elif 'masks@{}'.format(item) in meta['sets'][based_on]['items']:
            items.append('masks@{}'.format(item))
        # what is this else-branch supposed to achieve?
        else:
            try:
                if item in meta['columns'] and meta['columns'][item]['parent']:
                    items.append(list(meta['columns'][item]['parent'].keys())[0])
            except:
                if 'masks@{}'.format(re.sub(pattern, '', item)) in meta['sets'][based_on]['items']:
                    items.append('masks@{}'.format(re.sub(pattern, '', item)))

    fset = {'items': []}
    for item in meta['sets'][based_on]['items']:
        if item in items:
            if item.startswith('masks') and not masks:
                for mask_item in meta['masks'][item.split('@')[1]]['items']:
                    fset['items'].append(mask_item['source'])
            else:
                fset['items'].append(item)
    return fset

def cpickle_copy(obj):
    copy = pickle.loads(pickle.dumps(obj, pickle.HIGHEST_PROTOCOL))
    return copy

def parrot():
    from IPython.display import Image
    from IPython.display import display
    import os
    filename = os.path.dirname(__file__) + '\\parrot.gif'
    try:
        return display(Image(filename=filename, format='png'))
    except:
        print(':sad_parrot: Looks like the parrot is not available!')