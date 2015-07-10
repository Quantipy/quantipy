import pandas as pd
import numpy as np
import json
import re
import copy
import itertools
import math
import re, string
import sqlite3

from collections import OrderedDict
from quantipy.core.helpers.constants import DTYPE_MAP
from quantipy.core.helpers.constants import MAPPED_PATTERN
from itertools import product
from quantipy.core.view import View
from quantipy.core.view_generators.view_mapper import ViewMapper
from quantipy.core.helpers import functions

from quantipy.core.tools.dp.dimensions.reader import quantipy_from_dimensions
from quantipy.core.tools.dp.decipher.reader import quantipy_from_decipher
from quantipy.core.tools.dp.spss.reader import parse_sav_file
from quantipy.core.tools.dp.spss.writer import save_sav
from quantipy.core.tools.dp.ascribe.reader import quantipy_from_ascribe

def load_json(path_json, hook=OrderedDict):
    ''' Returns a python object from the json file located at path_json
    '''

    with open(path_json) as f:
        obj = json.load(f, object_pairs_hook=hook)

        return obj

def loads_json(json_text, hook=OrderedDict):
    ''' Returns a python object from the json string json_text
    '''

    obj = json.loads(json_text, object_pairs_hook=hook)

    return obj

def load_csv(path_csv):
    
    return pd.DataFrame.from_csv(path_csv)

def save_json(obj, path_json):

    def represent(obj):
        if isinstance(obj, np.generic):
            return np.asscalar(obj)
        else:
            return "Unserializable object: %s" % (str(type(obj)))
    
    with open(path_json, 'w+') as f:
        json.dump(obj, f, default=represent)

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
    var_types = pd.DataFrame({k: v['type'] for k, v in meta['columns'].iteritems()}, index=['meta']).T
    df = pd.concat([var_types, dtypes.astype(str)], axis=1)

    missing = df.loc[df['dtype'].isin([np.NaN])]['meta']
    if missing.size>0:
        print '\nSome meta not paired to data columns was found (these may be special data types):\n', missing, '\n'

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
            for table_name in ddf['tables'].keys()
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
        for table_name in ddf['table_info'].keys():
            try:
                ddf['table_info'][table_name].set_index(
                    ['name'],
                    drop=False,
                    inplace=True 
                )
            except:
                print (
                    "Couldn't set 'name' into the index for '%s'."
                ) % (table_name)
 
        for table_name in ddf['tables'].keys():
            index_col = 'TableName' if table_name=='Levels' else ':P0' 
            try:
                ddf['table_info'][table_name].set_index(
                    ['name'],
                    drop=False,
                    inplace=True 
                )
            except:
                print (
                    "Couldn't set '%s' into the index for the '%s' "
                    "Dataframe."
                ) % (index_col, table_name)
   
    return ddf

def read_dimensions(path_mdd, path_ddf):
    
    meta, data = quantipy_from_dimensions(path_mdd, path_ddf)
    return meta, data

def read_decipher(path_json, path_txt, text_key='main'):
    
    meta, data = quantipy_from_decipher(path_json, path_txt, text_key)
    return meta, data

def read_spss(path_sav, **kwargs):
    
    meta, data = parse_sav_file(path_sav, **kwargs)
    return meta, data

def write_spss(path_sav, meta, data, index=True, text_key=None, mrset_tag_style='__', drop_delimited=True):
    
    save_sav(
        path_sav, 
        meta, 
        data, 
        index=index, 
        text_key=text_key, 
        mrset_tag_style=mrset_tag_style,
        drop_delimited=drop_delimited
    )

def read_ascribe(path_xml, path_txt, text_key='main'):
    
    meta, data = quantipy_from_ascribe(path_xml, path_txt, text_key)
    return meta, data
