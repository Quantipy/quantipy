import pandas as pd
import numpy as np
import json
import re
import copy
import itertools
import math
import re, string
import sqlite3
import sys

from ftfy import fix_text

from collections import OrderedDict
from quantipy.core.helpers.constants import DTYPE_MAP
from quantipy.core.helpers.constants import MAPPED_PATTERN
from itertools import product

from quantipy.core.tools.dp.dimensions.reader import quantipy_from_dimensions
from quantipy.core.tools.dp.dimensions.writer import dimensions_from_quantipy
from quantipy.core.tools.dp.decipher.reader import quantipy_from_decipher
from quantipy.core.tools.dp.spss.reader import parse_sav_file
from quantipy.core.tools.dp.spss.writer import save_sav
from quantipy.core.tools.dp.ascribe.reader import quantipy_from_ascribe
import importlib

def make_like_ascii(text):
    """
    Replaces any non-ascii unicode with ascii unicode.
    """

    unicode_ascii_mapper = {
        '\u2022': '-',         # http://www.fileformat.info/info/unicode/char/2022/index.htm
        '\u2013': '-',         # http://www.fileformat.info/info/unicode/char/2013/index.htm
        '\u2018': '\u0027',    # http://www.fileformat.info/info/unicode/char/2018/index.htm
        '\u2019': '\u0027',    # http://www.fileformat.info/info/unicode/char/2019/index.htm
        '\u201c': '\u0022',    # http://www.fileformat.info/info/unicode/char/201C/index.htm
        '\u201d': '\u0022',    # http://www.fileformat.info/info/unicode/char/201D/index.htm
        '\u00a3': 'GBP ',      # http://www.fileformat.info/info/unicode/char/a3/index.htm
        '\u20AC': 'EUR ',      # http://www.fileformat.info/info/unicode/char/20aC/index.htm
        '\u2026': '\u002E\u002E\u002E', # http://www.fileformat.info/info/unicode/char/002e/index.htm
    }

    for old, new in unicode_ascii_mapper.items():
        text = text.replace(old, new)

    return text

def unicoder(obj, decoder='UTF-8', like_ascii=False):
    """
    Decodes all the text (keys and strings) in obj.

    Recursively mines obj for any str objects, whether keys or values,
    converting any str objects to unicode and then correcting the
    unicode (which may have been decoded incorrectly) using ftfy.

    Parameters
    ----------
    obj : object
        The object to be mined.

    Returns
    -------
    obj : object
        The recursively decoded object.
    """

    if isinstance(obj, list):
        obj = [
            unicoder(item, decoder, like_ascii)
            for item in obj]
    if isinstance(obj, tuple):
        obj = tuple([
            unicoder(item, decoder, like_ascii)
            for item in obj])
    elif isinstance(obj, (dict)):
        obj = {
            key: unicoder(value, decoder, like_ascii)
            for key, value in obj.items()}
    elif isinstance(obj, str):
        obj = fix_text(str(obj))
    elif isinstance(obj, str):
        obj = fix_text(obj)

    if like_ascii and isinstance(obj, str):
        obj = make_like_ascii(obj)

    return obj

def encoder(obj, encoder='UTF-8'):
    """
    Encodes all the text (keys and strings) in obj.

    Recursively mines obj for any str objects, whether keys or values,
    encoding any str objects.

    Parameters
    ----------
    obj : object
        The object to be mined.

    Returns
    -------
    obj : object
        The recursively decoded object.
    """

    if isinstance(obj, list):
        obj = [
            unicoder(item)
            for item in obj
        ]
    if isinstance(obj, tuple):
        obj = tuple([
            unicoder(item)
            for item in obj
        ])
    elif isinstance(obj, (dict)):
        obj = {
            key: unicoder(value)
            for key, value in obj.items()
        }
    elif isinstance(obj, str):
        obj = obj.endoce(encoder)

    return obj

def enjson(obj, indent=4, encoding='UTF-8'):
    """
    Dumps unicode json allowing non-ascii characters encoded as needed.
    """
    return json.dumps(obj, indent=indent, ensure_ascii=False).encode(encoding)

def load_json(path_json, hook=OrderedDict):
    ''' Returns a python object from the json file located at path_json
    '''

    with open(path_json) as f:
        obj = unicoder(json.load(f, object_pairs_hook=hook))

        return obj

def loads_json(json_text, hook=OrderedDict):
    ''' Returns a python object from the json string json_text
    '''

    obj = json.loads(json_text, object_pairs_hook=hook)

    return obj

def load_csv(path_csv):

    data = pd.read_csv(path_csv)
    return data

def save_json(obj, path_json, decode_str=False, decoder='UTF-8'):

    if decode_str:
        obj = unicoder(obj, decoder)

    def represent(obj):
        if isinstance(obj, np.generic):
            return np.asscalar(obj)
        else:
            return "Unserializable object: %s" % (str(type(obj)))

    with open(path_json, 'w+') as f:
        json.dump(obj, f, default=represent, sort_keys=True)

def df_to_browser(df, path_html='df.html', **kwargs):

    import webbrowser

    with open(path_html, 'w') as f:
        f.write(df.to_html(**kwargs))

    webbrowser.open(path_html, new=2)

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

def read_ddf(path_ddf, auto_index_tables=True):
    ''' Returns a raw version of the DDF in the form of a dict of
    pandas DataFrames (one for each table in the DDF).

    Parameters
    ----------
    path_ddf : string, the full path to the target DDF

    auto_index_tables : boolean (optional)
        if True, will set the index for all returned DataFrames using the most
        meaningful candidate column available. Columns set into the index will
        not be dropped from the DataFrame.

    Returns
    ----------
    dict of pandas DataFrames
    '''

    # Read in the DDF (which is a sqlite file) and retain all available
    # information in the form of pandas DataFrames.
    with sqlite3.connect(path_ddf) as conn:
        ddf = {}
        ddf['sqlite_master'] = pd.read_sql(
            'SELECT * FROM sqlite_master;',
            conn
        )
        ddf['tables'] = {
            table_name:
            pd.read_sql('SELECT * FROM %s;' % (table_name), conn)
            for table_name in ddf['sqlite_master']['tbl_name'].values
            if table_name.startswith('L')
        }
        ddf['table_info'] = {
            table_name:
            pd.read_sql("PRAGMA table_info('%s');" % (table_name), conn)
            for table_name in list(ddf['tables'].keys())
        }

    # If required, set the index for the expected Dataframes that should
    # result from the above operation.
    if auto_index_tables:
        try:
            ddf['sqlite_master'].set_index(
                ['name'],
                drop=False,
                inplace=True
            )
        except:
            print (
                "Couldn't set 'name' into the index for 'sqlite_master'."
            )
        for table_name in list(ddf['table_info'].keys()):
            try:
                ddf['table_info'][table_name].set_index(
                    ['name'],
                    drop=False,
                    inplace=True
                )
            except:
                print((
                    "Couldn't set 'name' into the index for '%s'."
                ) % (table_name))

        for table_name in list(ddf['tables'].keys()):
            index_col = 'TableName' if table_name=='Levels' else ':P0'
            try:
                ddf['table_info'][table_name].set_index(
                    ['name'],
                    drop=False,
                    inplace=True
                )
            except:
                print((
                    "Couldn't set '%s' into the index for the '%s' "
                    "Dataframe."
                ) % (index_col, table_name))

    return ddf

def read_dimensions(path_mdd, path_ddf):

    meta, data = quantipy_from_dimensions(path_mdd, path_ddf)
    return meta, data

def write_dimensions(meta, data, path_mdd, path_ddf, text_key=None,
                     CRLF="CR", run=True, clean_up=True):

    default_stdout = sys.stdout
    default_stderr = sys.stderr
    importlib.reload(sys)
    sys.setdefaultencoding("cp1252")
    sys.stdout = default_stdout
    sys.stderr = default_stderr

    out = dimensions_from_quantipy(meta, data, path_mdd, path_ddf,
                                   text_key, CRLF, run, clean_up)

    default_stdout = sys.stdout
    default_stderr = sys.stderr
    importlib.reload(sys)
    sys.setdefaultencoding("utf-8")
    sys.stdout = default_stdout
    sys.stderr = default_stderr
    return out

def read_decipher(path_json, path_txt, text_key='main'):

    meta, data = quantipy_from_decipher(path_json, path_txt, text_key)
    return meta, data

def read_spss(path_sav, **kwargs):

    meta, data = parse_sav_file(path_sav, **kwargs)
    return meta, data

def write_spss(path_sav, meta, data, index=True, text_key=None,
               mrset_tag_style='__', drop_delimited=True, from_set=None,
               verbose=False):

    save_sav(
        path_sav,
        meta,
        data,
        index=index,
        text_key=text_key,
        mrset_tag_style=mrset_tag_style,
        drop_delimited=drop_delimited,
        from_set=from_set,
        verbose=verbose
    )

def read_ascribe(path_xml, path_txt, text_key='main'):

    meta, data = quantipy_from_ascribe(path_xml, path_txt, text_key)
    return meta, data

def read_quantipy(path_json, path_csv):
    """
    Load Quantipy meta and data from disk.
    """

    meta = load_json(path_json)
    data = load_csv(path_csv)

    for col in list(meta['columns'].keys()):
        if meta['columns'][col]['type']=='date':
            data[col] = pd.to_datetime(data[col])

    return meta, data

def write_quantipy(meta, data, path_json, path_csv):
    """
    Save Quantipy meta and data to disk.
    """

    save_json(meta, path_json)
    data.to_csv(path_csv)
