import pandas as pd
import numpy as np

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

def define_multicodes(varlist, meta):
    multicodes = {}
    for var in varlist:
        multicodes.update({var: [mrs_q for mrs_q in meta['columns'] if mrs_q.startswith(var + '_')]})

    return multicodes

def dichotomous_from_delimited(ds, value_map=None, sep=';', trailing_sep=True,
                               dichotom=[1, 2]):
    ''' Returns a dichotomous set DataFrame from ds, being a series storing
    delimited set data separated by 'sep'

    ds - (pandas.Series) a series storing delimited set data
    value_map - (list-like, optional)  the values to be anticipated as unique
        in ds
    sep - (str, optional) the character/s to use to delimit ds
    trailing_sep - (bool, optional) is sep trailing all items in ds?
    dichotom - (list-like, optional) the dochotomous values to use [yes, no]
    '''

    ds_split = ds.dropna().str.split(';')

    if value_map is None:
        value_map = get_delimited_value_map(ds, ds_split, sep)
        
    df = pd.DataFrame(data=dichotom[1], index=ds.index, columns=value_map)

    for idx in ds_split.index:
        if trailing_sep:
            cols = ds_split.loc[idx][:-1]
        else:
            cols = ds_split.loc[idx][:]
        df.loc[idx][cols] = dichotom[0]

    return df

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

def stack_column_group(data, cols, levelid_name='levelid', data_name='data',
                       dropna=True, levelid_map=None):
    ''' Stacks the given columns from data, optionally renaming the 
    resultiong levelid and data columns, mapping the values found in 
    the levelid column, and appending the levelid column onto the index. 

    Parameters
    ----------
    data : pandas.DataFrame
        The data from which the hierarchical groups are being drawn.

    cols : list
        A list column names that need to be stacked from the source
        data.

    levelid_name : str
        The name to be given to the levelid series that results from
        the pandas.DataFrame.stack() operation.

    data_name : str
        The name to be given to the data series that results from
        the pandas.DataFrame.stack() operation.

    dropna: boolean (optional; default=True)
        Passed through to the pandas.DataFrame.stack() operation.

    levelid_map: list (optional; default=None)
        The list of values/labels used to identify each resulting 
        stacked row. Using a mapper allows multi-question hierarchies
        to be merged together because the resulting MultiIndexes will
        match.
    '''

    # For multi-level hierarchies, capture the new level number about
    # to be added|
    if isinstance(data.index, pd.MultiIndex):
        new_level = len(data.index.levels)
    else:
        new_level = 1
    
    df = data[cols].stack(dropna=dropna).reset_index(level=[new_level])
    df.columns = [levelid_name, data_name]
    
    if not levelid_map is None:
        df[levelid_name] = df[levelid_name].map(levelid_map)

    df.set_index([levelid_name], append=True, drop=True, inplace=True)

    return df


def stack_hierarchy(data, structure, levelid_name, levelid_mapper=None,
                    others=None, dropna=True):
    ''' Returns a list of dicts suitable for use with the
    'hierarchy_spec' parameter of the transpose_grid() function.
    
    Parameters
    ----------
    data : pandas.DataFrame
        The data from which the hierarchical groups are being drawn.

    structure : list of dicts
        A list of dicts matching where the new column names are keys to
        to lists of source columns. 

    levelid_name : str
        The name to be given to the levelid series that results from
        the pandas.DataFrame.stack() operation.

    levelid_mapper: list (optional; default=None)
        The list of values/labels used to identify each resulting 
        stacked row. Using a mapper allows multi-question hierarchies
        to be merged together because the resulting MultiIndexes will
        match. 

    others: list (optional; default=None)
        A list of additional columns from the source data to be appended
        to the end of the resulting stacked dataframe.

    dropna: boolean (optional; default=True)
        Passed through to the pandas.DataFrame.stack() operation.

    Returns
    ----------
    df : pandas.DataFrame
        The stacked dataframe.  
    '''

    # For multi-level hierarchies, capture the new level number about
    # to be added|
    if isinstance(data.index, pd.MultiIndex):
        new_level = len(data.index.levels)
    else:
        new_level = 1

    # Collect all of the stacked column groups into a list
    dfs = []
    for question_group in structure:
        question_name = question_group.keys()[0]
        question_columns = question_group.values()[0]
        df = stack_column_group(
            data=data, 
            cols=question_columns, 
            levelid_name=levelid_name, 
            data_name=question_name,
            dropna=dropna,
            levelid_map=dict(zip(question_columns, levelid_mapper))
        )
        dfs.append(df)

    # Join all of the stacked dataframes together
    df = pd.concat(dfs, axis=1)

    if not others is None:
        # Merge in additional columns from the source data
        df.reset_index(level=[new_level], inplace=True)
        df = df.join(data[others])
        df.set_index([levelid_name], append=True, drop=True, inplace=True)

    return df

def start_meta(name='', text_key='main'):
    """ Starts the Quantipy meta document for a project converted 
    from Decipher.

    Parameters
    ----------
    name : str, default='?'

    Returns
    -------
    meta : dict
        Quantipy meta object
    """

    meta = {
        'info': {
            'text': ''
        },
        'lib': {
            'default text': text_key,
            'values': {}
        },
        'columns': {},
        'masks': {},
        'sets': {
            'data file': {
                'text': {'main': 'Variable order in source file'},
                'items': []
            }
        },
        'type': 'pandas.DataFrame'
    }

    return meta

def condense_dichotomous_set(df, values_from_labels=True, sniff_single=False):
    """
    Condense the given dichotomous columns to a delimited set series.

    Parameters
    ----------
    df : pandas.DataFrame
        The column/s in the dichotomous set. This may be a single-column
        DataFrame, in which case a non-delimited set will be returned.
    values_from_labels : bool, default=True
        Should the values used for each response option be taken from
        the dichotomous column names using the rule name.split('_')[-1]?
        If not then the values will be sequential starting from 1.
    sniff_single : bool, default=False
        Should the returned series be given as dtype 'int' if the 
        maximum number of responses for any row is 1?

    Returns
    -------
    series: pandas.series
        The converted series
    """
    
    # Convert to delimited set
    df_str = df.astype('str')
    for v, col in enumerate(df_str.columns, start=1):
        if values_from_labels:
            v = col.split('_')[-1]
        else:
            v = str(v)
        # Convert to categorical set
        df_str[col].replace(
            {
                '1': v, 
                '1.0': v, 
                'nan': 'nan', 
                '0.0': 'nan'
            }, 
            inplace=True
        )
    # Concatenate the rows
    series = df_str.apply(
        lambda x: ';'.join([
            v 
            for v in x.tolist() 
            if v != 'nan'
        ]),
        axis=1
    )
    # Use NaNs to represent emtpy
    series.replace(
        {'': np.NaN}, 
        inplace=True
    )
    
    if df.dropna().size==0:
        # No responses are known, return filled with NaN
        return series
    
    if sniff_single and df.sum(axis=1).max()==1:
        # Convert to float
        series = series.str.replace(';','').astype('float')
        return series
    else:
        # Append final delimiting character
        series = series + ';'
    
    return series

def split_series(series, sep, columns=None):
    """
    Splits all the items of a series using the given delimiter.

    Splits each item in series using the given delimiter and returns
    a DataFrame (as per Excel text-to-columns). Optionally, you can
    pass in a list of column names that should be used to name the 
    resulting columns.
    
    Parameters
    ----------
    series : pandas.Series
        The series that should be split.
    sep : str
        The separator that should be used to split the series.
    columns : list-list, default=None
        A list of names that should be set into the resulting DataFrame
        columns.

    Returns
    -------
    df : pandas.DataFrame
        Series, split by sep, returned as a DataFrame.
    """

    df = pd.DataFrame(series.astype('str').str.split(sep).tolist())
    if not columns is None:
        df.columns = columns
    return df
