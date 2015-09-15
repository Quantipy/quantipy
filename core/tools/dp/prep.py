import numpy as np
import pandas as pd
import quantipy as qp
import copy

from quantipy.core.helpers.functions import emulate_meta
from quantipy.core.helpers.functions import (
    create_full_index_dataframe,
    paint_dataframe
)

from quantipy.core.tools.view.logic import (
    has_any,
    get_logic_index
)

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
                'nan': 'nan', 
                '0.0': 'nan',
                '0': 'nan'
            }, 
            inplace=True
        )
        df_str[col].replace(
            {
                '1': v, 
                '1.0': v
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

def frange(range_def, sep=','):
    """
    Return the full, unabbreviated list of ints suggested by range_def. 

    This function takes a string of abbreviated ranges, possibly
    delimited by a comma (or some other character) and extrapolates
    its full, unabbreviated list of ints.

    Parameters
    ----------
    range_def : str
        The range string to be listed in full. 
    sep : str, default=','
        The character that should be used to delimit discrete entries in
        range_def.
        
    Returns
    -------
    res : list
        The exploded list of ints indicated by range_def.
    """
    
    res = []
    for item in range_def.split(sep):
        if '-' in item:
            a, b = item.split('-')
            a, b = int(a), int(b)
            lo = min([a, b])
            hi = max([a, b])
            ints = range(lo, hi+1)
            if b <= a:
                ints = list(reversed(ints))
            res.extend(ints)
        else:
            res.append(int(item))
    return res

def crosstab(meta, data, x, y, get='count', decimals=1, weight=None,
             show='values', rules=False, full=False):
    """
    Return a type-appropriate crosstab of x and y.

    This function uses the given meta and data to create a 
    type-appropriate cross-tabulation (pivot table) of the named x and y
    variables. The result may be either counts or column percentages, 
    weighted or unweighted.

    Parameters
    ----------
    meta : dict
        Quantipy meta document.    
    data : pandas.DataFrame
        Data accompanying the given meta document. 
    x : str
        The variable that should be placed into the x-position.
    y : str
        The variable that should be placed into the y-position.
    get : str, default='count'
        Control the type of data that is returned. 'count' will return
        absolute counts and 'normalize' will return column percentages.
    decimals : int, default=1
        Control the number of decimals in the returned dataframe.
    weight : str, default=None
        The name of the weight variable that should be used on the data,
        if any.
    show : str, default='values'
        How the index and columns should be displayed. 'values' returns 
        the raw value indexes. 'text' returns the text associated with 
        each value, according to the text key 
        meta['lib']['default text']. Any other str value is assumed to
        be a non-default text_key.  
    rules : bool or list-like, default=False
        If True then all rules that are found will be applied. If 
        list-like then rules with those keys will be applied. 
    full : bool, default=False
        If True, the returned dataframe will have a full index applied.
        Note that rules=True requires a full index be applied and so
        makes this argument redundant.

    Returns
    -------
    df : pandas.DataFrame
        The crosstab as a pandas DataFrame.
    """
    
    stack = qp.Stack(name='ct', add_data={'ct': {'meta': meta, 'data': data}})
    stack.add_link(x=x, y=y)
    link = stack['ct']['no_filter'][x][y]
    q = qp.Quantity(link, weight=weight).count()
    if weight is None: weight = ''
    if get=='count':
        df = q.result
        vk = 'x|frequency|||{}|counts'.format(weight)
    elif get=='normalize':
        df = q.normalize().result
        vk = 'x|frequency||y|{}|c%'.format(weight)
    else:
        raise ValueError(
           "The value for 'get' was not recognized. Should be 'count' or "
           "'normalize'."
        )
    
    df = np.round(df, decimals=decimals)
    df = show_df(df, meta, show, rules, full, link, vk)

    return df
 
def show_df(df, meta, show='values', rules=False, full=False, link=None,
            vk=None):
    """
    """

    if rules:
        
        full = True

        xk = link.x
        yk = link.y
        
        # Determine if sorting is required on x or y
        x_sortx = has_sorting_rules(meta, xk)
        y_sortx = has_sorting_rules(meta, yk)
        
        # If sorting is required then the 'All' row/column
        # needs to be appended (if it isn't already there),
        # otherwise there's no way to sort on 'total'.
        if x_sortx or y_sortx:
            needs_x_margin = not (xk, 'All') in df.index
            needs_y_margin = not (yk, 'All') in df.columns
            
            if needs_x_margin or needs_y_margin:                
                # Get the link under which df was found and
                # find its weight (if any) 
                weight = vk.split("|")[-2]
                if weight=='': weight = None    
                # Add the missing margins to df
                df = add_margins(
                    df, link, weight,
                    x_margin=needs_x_margin,
                    y_margin=needs_y_margin
                )

    if show=='values' and not rules and not full:
        pass

    elif show=='values' and not rules and full:
        df = create_full_index_dataframe(df, meta, rules=None)

    elif show=='values' and rules and (full or not full):
        df = create_full_index_dataframe(df, meta, rules=rules)

    else:
        if show=='text':
            df = paint_dataframe(
                df, meta, 
                create_full_index=full, 
                rules=rules
            )
        else:
            text_key = {'x': [show], 'y': [show]}
            df = paint_dataframe(
                df, meta, 
                text_key=text_key, 
                create_full_index=full, 
                rules=rules
            )

    if rules:
        
        # If the original dataframe didn't have any margins
        # to begin with, but now there are some due to the need
        # to apply sorting, then they should now be removed.
        if x_sortx or y_sortx:
            if needs_x_margin:
                df.drop(
                    (df.index.levels[0][0], 'All'), 
                    inplace=True
                )
            if needs_y_margin:
                df.drop(
                    (df.columns.levels[0][0], 'All'), 
                    inplace=True, 
                    axis=1
                )
        
    # Make sure that all the margins, if present, 
    # appear first on their respective axes
    df = prepend_margins(df)

    return df

def add_margins(df, link, weight, x_margin, y_margin):
    """
    Add missing margins to the view result df.

    This function uses a Quantity instance based on link and weight
    to add the index and column margins onto df.
    """

    xk = link.x
    yk = link.y
    
    if xk=='@':
        xk = df.index.levels[0][0]
        
    if yk=='@':
        yk = df.index.levels[0][0]
    
    q = qp.Quantity(link, weight=weight)
    
    # Extract the x, y and xy margins using
    # using Quantity methods
    x_all = q._col_n()[0]
    xy_all = [x_all[-1]]
    x_all = x_all if len(x_all)==1 else list(x_all[:-1])
    y_all = [item[0] for item in q._row_n()]
    
    # There are three possibilities:
    # 1. y has a margin but x doesn't
    # 2. x has a margin by y doesn't
    # 3. Neither x nor y has a margin

    # The x and y margins need to be concatenated
    # with the xy margin based on where in the target
    # index any perpendicular margin may already exist.

    if x_margin and not y_margin:
        # 1. y has a margin but x doesn't
        idx = df.columns.tolist().index((yk, 'All'))
        df = df.T
        if idx==0:
            # Perpendicular margin is first
            df[(xk, 'All')] = xy_all + x_all
        else:
            # Perpendicular margin is last
            df[(xk, 'All')] = x_all + xy_all
        df = df.T

    elif y_margin and not x_margin:
        # 2. x has a margin by y doesn't
        idx = df.index.tolist().index((xk, 'All'))
        if idx==0:
            # Perpendicular margin is first
            df[(yk, 'All')] = xy_all + y_all
        else:
            # Perpendicular margin is last
            df[(yk, 'All')] = y_all + xy_all
        
    elif x_margin and y_margin:
        # 3. Neither x nor y has a margin
        df[(yk, 'All')] = y_all
            # Perpendicular margin is last
        df = df.T
        df[(xk, 'All')] = list(x_all) + list(xy_all)
        df = df.T

    return df

def prepend_margins(df):
    """
    Ensures that the margins in df appear first on each axis. 
    """

    x_col = df.index.levels[0][0]
    if not (x_col, '@') in df.index:
        margin = (x_col, 'All')
        if margin in df.index:
            if not df.index[0] == margin:
                margin = [margin]
                others = [c for c in df.index if c[1] != 'All']
                df = df.T[margin+others].T

    y_col = df.columns.levels[0][0]
    if not (y_col, '@') in df.columns:
        margin = (y_col, 'All')
        if margin in df.columns:
            if not df.columns[0] == margin:
                margin = [margin]
                others = [c for c in df.columns if c[1] != 'All']
                df = df[margin+others]

    return df

def has_sorting_rules(meta, col_name):
    """
    Return if the named column has any sortx rules defined.
    """
        
    if col_name=='@':
        return False
        
    # Determine if sorting is required on x
    has_sortx = False
    col = meta['columns'][col_name]
    if 'rules' in col:
        rules = col['rules'].get('x', None)
        if not rules is None:
            has_sortx = 'sortx' in rules
    
    return has_sortx

def frequency(meta, data, x, **kwargs):
    """
    Return a type-appropriate frequency of x.

    This function uses the given meta and data to create a 
    type-appropriate frequency table of the named x variable.
    The result may be either counts or column percentages, weighted 
    or unweighted.

    Parameters
    ----------
    meta : dict
        Quantipy meta document.    
    data : pandas.DataFrame
        Data accompanying the given meta document. 
    x : str
        The variable that should be placed into the x-position.
    kwargs : kwargs
        All remaining keyword arguments will be passed along to the
        crosstab function.

    Returns
    -------
    f : pandas.DataFrame
        The frequency as a pandas DataFrame.
    """
    
    f = crosstab(meta, data, x, '@', **kwargs)
    return f

def get_index_mapper(meta, data, mapper, default=None):
    """
    Convert a {value: logic} map to a {value: index} map.

    This function takes a mapper of {key: logic} entries and resolves
    the logic statements using the given meta/data to return a mapper
    of {key: index}. The indexes returned can be used on data to isolate
    the cases described by arbitrarily complex logical statements.

    Parameters
    ----------
    meta : dict
        Quantipy meta document.
    data : pandas.DataFrame
        Data accompanying the given meta document.       
    mapper : dict
        A mapper of {key: logic}
    default : str
        The column name to default to in cases where unattended lists
        are given as logic, where an auto-transformation of {key: list}
        to {key: {default: list}} is provided.

    Returns
    -------
    index_mapper : dict
        A mapper of {key: index}
    """
    
    if default is None:
        # Check that mapper isn't in a default-requiring
        # format
        for key, val in mapper.iteritems():
            if not isinstance(val, (dict, tuple)):
                raise TypeError(
                    "'%s' recode definition appears to be using "
                    "default-shorthand but no value for 'default'"
                    "was given." % (key)
                )
        keyed_mapper = mapper
    else:
        # Use default to correct the form of the mapper
        # where un-keyed value lists were given
        # Creates: {value: {source: logic}}
        keyed_mapper = {
            key: 
            {default: has_any(val)}
            if isinstance(val, list)
            else {default: val}
            for key, val in mapper.iteritems()
        }
    
    # Create temp series with a full data index 
    series = pd.Series(1, index=data.index)
    
    # Return indexes from logic statements
    # Creates: {value: index}
    index_mapper = {
        key: get_logic_index(series, logic, data)[0]
        for key, logic in keyed_mapper.iteritems()
    }
    
    return index_mapper 

def join_delimited_set_series(ds1, ds2, append=True):
    """
    Item-wise join of two delimited sets.

    This function takes a mapper of {key: logic} entries and resolves
    the logic statements using the given meta/data to return a mapper
    of {key: index}. The indexes returned can be used on data to isolate
    the cases described by arbitrarily complex logical statements.

    Parameters
    ----------
    ds1 : pandas.Series
        First delimited set series to join.
    ds2 : pandas.Series
        Second delimited set series to join.
    append : bool
        Should the data in ds1 (where found) be appended to items from
        ds1? If False, data from ds1 (where found) will overwrite
        whatever was found for that item in ds1 instead.

    Returns
    -------
    joined : pandas.Series
        The joined result of ds1 and ds2.
    """
    
    df = pd.concat([ds1, ds2], axis=1)
    df.fillna('', inplace=True)
    if append:
        df['joined'] = df[0] + df[1]
    else:
        df['joined'] = df[0].copy()
        df[1] = df[1].replace('', np.NaN)
        df['joined'].update(df[1].dropna())
    
    joined = df['joined'].replace('', np.NaN)
    return joined

def recode_from_index_mapper(meta, series, index_mapper, append):
    """
    Convert a {value: logic} map to a {value: index} map.

    This function takes a mapper of {key: logic} entries and resolves
    the logic statements using the given meta/data to return a mapper
    of {key: index}. The indexes returned can be used on data to isolate
    the cases described by arbitrarily complex logical statements.

    Parameters
    ----------
    meta : dict
        Quantipy meta document.
    series : pandas.Series
        The series in which the recoded data will be stored and 
        returned.
    index_mapper : dict
        A mapper of {key: index}
    append : bool
        Should the new recodd data be appended to items already found
        in series? If False, data from series (where found) will
        overwrite whatever was found for that item in ds1 instead.

    Returns
    -------
    series : pandas.Series
        The series in which the recoded data will be stored and 
        returned.
    """
    
    qtype = meta['columns'][series.name]['type']
    
    if qtype in ['delimited set']:
        if series.dtype in ['int64', 'float64']:
            not_null = series.notnull()
            if len(not_null) > 0:
                series.loc[not_null] = series.loc[not_null].map(str) + ';'
        cols = [str(c) for c in sorted(index_mapper.keys())]
        ds = pd.DataFrame(0, index=series.index, columns=cols)
        for key, idx in index_mapper.iteritems():
            ds[str(key)].loc[idx] = 1
        ds = condense_dichotomous_set(ds)
        series = join_delimited_set_series(series, ds, append)
        
    elif qtype in ['single', 'int', 'float']:
        for key, idx in index_mapper.iteritems():
            series.loc[idx] = key
    else:
        raise TypeError(
            "Can't recode '{col}'. Recoding for '{typ}' columns is not"
            " yet supported.".format(col=series.name, typ=qtype) 
        )
        
    return series

def recode(meta, data, target, mapper, append=False, default=None):
    """
    Return a new or copied series from data, recoded using a mapper.

    This function takes a mapper of {key: logic} entries and injects the
    key into the target column where its paired logic is True. The logic
    may be arbitrarily complex and may refer to any other variable or 
    variables in data. Where a pre-existing column has been used to 
    start the recode, the injected values can replace or be appended to 
    any data found there to begin with. Note that this function does
    not edit the target column, it returns a recoded copy of the target
    column. The recoded data will always comply with the column type
    indicated for the target column according to the meta.

    Parameters
    ----------
    meta : dict
        Quantipy meta document.    
    data : pandas.DataFrame
        Data accompanying the given meta document. 
    target : str
        The column name that is the target of the recode. If target
        is not found in meta['columns'] this will fail with an error.
        If target is not found in data.columns the recode will start
        from an empty series with the same index as data. If target
        is found in data.columns the recode will start from a copy
        of that column.
    mapper : dict
        A mapper of {key: logic} entries.
    append : bool, default=False
        Should the new recodd data be appended to values already found
        in the series? If False, data from series (where found) will
        overwrite whatever was found for that item instead.
    default : str, default=None
        The column name to default to in cases where unattended lists
        are given in your logic, where an auto-transformation of 
        {key: list} to {key: {default: list}} is provided. Note that
        lists in logical statements are themselves a form of shorthand
        and this will ultimately be interpreted as:
        {key: {default: has_any(list)}}.

    Returns
    -------
    series : pandas.Series
        The series in which the recoded data is stored.
    """

    # Error handling
   
    # Check meta, data
    if not isinstance(meta, dict):
        raise ValueError("'meta' must be a dictionary.")
    if not isinstance(data, pd.DataFrame):
        raise ValueError("'data' must be a pandas.DataFrame.")
        
    # Check mapper
    if not isinstance(mapper, dict):
        raise ValueError("'mapper' must be a dictionary.")

    # Check copy_of
    if not isinstance(target, (str, unicode)):
        raise ValueError("The value for 'target' must be a string.")
    if not target in meta['columns']:
        raise ValueError("'%s' not found in meta['columns']." % (target))
    
    # Check append
    if not isinstance(append, bool):
        raise ValueError("'append' must be boolean.")
            
    # Check default
    if not default is None:
        if not isinstance(default, (str, unicode)):
            raise ValueError("The value for 'default' must be a string.")
        if not default in meta['columns']:
            raise ValueError("'%s' not found in meta['columns']." % (default))
        
    # Resolve the logic to a mapper of {key: index} 
    index_mapper = get_index_mapper(meta, data, mapper, default)
    
    # Get/create recode series
    if target in data.columns:
        series = data[target].copy()
    else:
        series = pd.Series(np.NaN, index=data.index, name=target)

    # Use the index mapper to edit the target series
    series = recode_from_index_mapper(meta, series, index_mapper, append)
    
    return series  

def hmerge(dataset_left, dataset_right, how='left', **kwargs):
    """
    Merge Quantipy datasets together using an index-wise identifer.

    This function merges two Quantipy datasets (meta and data) together,
    updating variables that exist in the left dataset and appending 
    others. New variables will be appended in the order indicated by
    the 'data file' set if found, otherwise they will be appended in
    alphanumeric order. Packed kwargs will be passed on to the
    pandas.DataFrame.merge() method call.

    Parameters
    ----------
    dataset_left : tuple
        A tuple of the left dataset in the form (meta, data).
    dataset_right : tuple
        A tuple of the right dataset in the form (meta, data). 
    how : str
        As per pandas.DataFrame.merge(how).
    **kwargs : various
        As per pandas.DataFrame.merge().
        
    Returns
    -------
    meta, data : dict, pandas.DataFrame
        Updated Quantipy dataset.
    """

    meta_left = copy.deepcopy(dataset_left[0])
    data_left = dataset_left[1].copy()

    meta_right = copy.deepcopy(dataset_right[0])
    data_right = dataset_right[1].copy()

    print '\n', 'Checking metadata...'
    if 'data file' in meta_right['sets']:
        print (
            "New columns will be appended in the order found in"
            " meta['sets']['data file']."
        )
        col_names = [
            item.split('@')[-1]
            for item in meta_right['sets']['data file']['items']
        ]
    else:
        print (
            "No 'data file' set was found, new columns will be appended"
            " alphanumerically."
        )
        col_names = meta_right['columns'].keys().sort(key=str.lower)

    print '\n', 'Merging meta...'
    col_updates = []
    for col_name in col_names:
        print '...', col_name
        if col_name in meta_left['columns'] and col_name in data_left.columns:
            col_updates.append(col_name)
        meta_left['columns'][col_name] = emulate_meta(
            meta_right, 
            meta_right['columns'][col_name]
        )

    if 'how' not in kwargs:
        kwargs['how'] = 'left'

    if 'left_on' not in kwargs and 'left_index' not in kwargs:
        kwargs['left_index'] = True

    if 'right_on' not in kwargs and 'right_index' not in kwargs:
        kwargs['right_index'] = True

    print '\n', 'Merging data...'
    if col_updates:
        if 'left_on' in kwargs:
            updata_left = data_left.set_index(
                [kwargs['left_on']]
            )[col_updates].copy()
        else:
            updata_left = data_left.copy()

        if 'right_on' in kwargs:
            updata_right = data_right.set_index(
                [kwargs['right_on']]
            )[col_updates].copy()
        else:
            updata_right = data_right.copy()

        print '...updating data for known columns'
        # print updata_left.head()
        # print updata_right.head()
        updata_left.update(updata_right)
        for update_col in col_updates:
            print "...", update_col
            data_left[update_col] = updata_left[update_col].copy()

    print '...', 'appending new columns'
    new_cols = [col for col in col_names if not col in col_updates]
    data_left = data_left.merge(data_right[new_cols], **kwargs)
    for col_name in new_cols:
        print '...', col_name

    return meta_left, data_left
