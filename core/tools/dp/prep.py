import numpy as np
import pandas as pd
import quantipy as qp
import copy
import re

from quantipy.core.helpers.functions import emulate_meta
from quantipy.core.helpers.functions import cpickle_copy

from quantipy.core.tools.view.logic import (
    has_any,
    get_logic_index,
    intersection
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

def derotate_column_group(data, cols, rotation_name='rotation', 
                          data_name='data', dropna=True, 
                          rotation_map=None):
    ''' Stacks the given columns from data, optionally renaming the 
    resultiong rotation and data columns, mapping the values found in 
    the rotation column, and appending the rotation column onto the index. 

    Parameters
    ----------
    data : pandas.DataFrame
        The data from which the hierarchical groups are being drawn.

    cols : list
        A list column names that need to be stacked from the source
        data.

    rotation_name : str
        The name to be given to the rotation series that results from
        the pandas.DataFrame.stack() operation.

    data_name : str
        The name to be given to the data series that results from
        the pandas.DataFrame.stack() operation.

    dropna: boolean (optional; default=True)
        Passed through to the pandas.DataFrame.stack() operation.

    rotation_map: list (optional; default=None)
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
    df.columns = [rotation_name, data_name]
    
    if not rotation_map is None:
        df[rotation_name] = df[rotation_name].map(rotation_map)

    df.set_index([rotation_name], append=True, drop=True, inplace=True)

    return df


def derotate(data, input_mapper, output_mapper, others=None, dropna=True):
    """
    Derotate data using the given input_mapper, and appending others.

    This function derotates data using the specification defined in
    input_mapper, which is a list of dicts of lists, describing how 
    columns from data can be read as a heirarchical structure.  
    
    Parameters
    ----------
    data : pandas.DataFrame
        The data from which the hierarchical groups are being drawn.

    input_mapper : list of dicts of lists
        A list of dicts matching where the new column names are keys to
        to lists of source columns. 

    output_mapper : dict
        The name and values to be given to the rotation index in the 
        output dataframe.

    others: list (optional; default=None)
        A list of additional columns from the source data to be appended
        to the end of the resulting stacked dataframe.

    dropna: boolean (optional; default=True)
        Passed through to the pandas.DataFrame.stack() operation.

    Returns
    ----------
    df : pandas.DataFrame
        The stacked dataframe.  
    """

    # For multi-level hierarchies, capture the new level number about
    # to be added|
    if isinstance(data.index, pd.MultiIndex):
        new_level = len(data.index.levels)
    else:
        new_level = 1

    rotation_name = output_mapper.keys()[0]
    rotation_index = output_mapper[rotation_name]

    # Collect all of the stacked column groups into a list
    dfs = []
    for question_group in input_mapper:
        question_name = question_group.keys()[0]
        question_columns = question_group.values()[0]
        df = derotate_column_group(
            data=data, 
            cols=question_columns, 
            rotation_name=rotation_name, 
            data_name=question_name,
            dropna=dropna,
            rotation_map=dict(zip(question_columns, rotation_index))
        )
        dfs.append(df)

    # Join all of the stacked dataframes together
    df = pd.concat(dfs, axis=1)

    if not others is None:
        # Merge in additional columns from the source data
        df.reset_index(level=[new_level], inplace=True)
        df = df.join(data[others])
        df.set_index([rotation_name], append=True, drop=True, inplace=True)

    return df

def start_meta(text_key='main'):
    """ 
    Starts a new Quantipy meta document.

    Parameters
    ----------
    text_key : str, default='main'
        The default text key to be set into the new meta document.

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
                'text': {text_key: 'Variable order in source file'},
                'items': []
            }
        },
        'type': 'pandas.DataFrame'
    }

    return meta

def condense_dichotomous_set(df, values_from_labels=True, sniff_single=False,
                             yes=1, no=0):
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
                '{}.0'.format(no): 'nan',
                '{}'.format(no): 'nan'
            }, 
            inplace=True
        )
        df_str[col].replace(
            {
                '{}'.format(yes): v, 
                '{}.0'.format(yes): v
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

def frequency(meta, data, x=None, y=None, **kwargs):
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
    x : str, default=None
        The column of data for which a frequency should be generated
        on the x-axis.
    y : str, default=None
        The column of data for which a frequency should be generated
        on the y-axis.
    kwargs : kwargs
        All remaining keyword arguments will be passed along to the
        crosstab function.

    Returns
    -------
    f : pandas.DataFrame
        The frequency as a pandas DataFrame.
    """
    
    if x is None and y is None:
        raise ValueError(
            "You must provide a value for either x or y."
        )
    elif not x is None and not y is None:
        raise ValueError(
            "You may only provide a value for either x or y, and not"
            " both, when generating a frequency."
        )
        
    if x is None:
        x = '@'
    else:
        y = '@'
        
    f = crosstab(meta, data, x, y, **kwargs)
    return f

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
 
# def show_df(df, meta, show='values', rules=False, full=False, link=None,
#             vk=None):
#     """
#     """

#     expand_axes = ['x', 'y']
#     relation = vk.split('|')[2]
    
#     condensed_x = False
#     condensed_y = False
    
#     if relation=='x:y':
#         condensed_x = True
#         expand_axes.remove('x')  
#     elif relation=='y:x':
#         condensed_y = True
#         expand_axes.remove('y')
#     else: 
#         if re.search('x\[.+:y$', relation) != None:
#             condensed_x = True
#             expand_axes.remove('x')
#         elif re.search('x:y\[.+', relation) != None:
#             condensed_y = True
#             expand_axes.remove('x')
#             expand_axes.remove('y')
            
#         if re.search('y\[.+:x$', relation) != None:
#             condensed_y = True
#             expand_axes.remove('y')
#         elif re.search('y:x\[.+', relation) != None:
#             condensed_x = True
#             expand_axes.remove('y')
#             expand_axes.remove('x')

#     has_rules = []
#     try:
#         if len(meta['columns'][link.x]['rules']['x']) > 0:
#             has_rules.append('x')
#     except:
#         pass
#     try:
#         if len(meta['columns'][link.y]['rules']['y']) > 0:
#             has_rules.append('y')
#     except:
#         pass

#     if rules is True:
#         rules = [
#             axis 
#             for axis in expand_axes 
#             if axis in has_rules]
#     elif isinstance(rules, list):
#         rules = [
#             axis 
#             for axis in expand_axes 
#             if axis in rules 
#             and axis in has_rules]
#     else:
#         rules = False

#     if rules:
        
#         full = True

#         xk = link.x
#         yk = link.y
        
#         weight = vk.split('|')[4]
#         weight = None if weight=='' else weight

#         rules_slicer_x = None
#         if xk=='@':
#             xk = df.index.levels[0][0]
#         elif 'x' in rules:
#             try:
#                 rules_x = meta['columns'][link.x]['rules']['x']
#                 with_weight = rules_x['sortx']['with_weight']
#             except:
#                 with_weight = weight
#             if 'sortx' in rules_x:
#                 fx = frequency(
#                     meta, 
#                     link.stack[link.data_key].data, 
#                     x=link.x, 
#                     rules=False,
#                     weight=with_weight
#                 )
#             else:
#                 fx = df
#             fx = create_full_index_dataframe(fx, meta, rules=rules, axes=['x'])
#             rules_slicer_x = fx.index.values.tolist()
#             if not (link.x, 'All') in df.index:
#                 try:
#                     rules_slicer_x.remove((link.x, 'All'))
#                 except:
#                     pass
            
#         rules_slicer_y = None
#         if yk=='@':
#             yk = df.columns.levels[0][0]
#         elif 'y' in rules:
#             try:
#                 rules_y = meta['columns'][link.y]['rules']['y']
#                 with_weight = rules_y['sortx']['with_weight']
#             except:
#                 with_weight = weight
#             if 'sortx' in rules_y:
#                 fy = frequency(
#                     meta, 
#                     link.stack[link.data_key].data, 
#                     y=link.y, 
#                     rules=False,
#                     weight=with_weight
#                 )
#             else:
#                 fy = df
#             fy = create_full_index_dataframe(fy, meta, rules=rules, axes=['y'])
#             rules_slicer_y = fy.columns.values.tolist()
#             if not (link.y, 'All') in df.columns:
#                 try:
#                     rules_slicer_y.remove((link.y, 'All'))
#                 except:
#                     pass
            
#     if show=='values' and not rules and not full:
#         pass

#     elif show=='values' and not rules and full:
#         df = create_full_index_dataframe(df, meta, rules=None, axes=expand_axes)

#     elif show=='values' and rules and (full or not full):
#         df = create_full_index_dataframe(df, meta, rules=False, axes=expand_axes)
#         if not rules_slicer_x is None:
#             df = df.loc[rules_slicer_x]
#         if not rules_slicer_y is None:
#             df = df[rules_slicer_y]

#         if 'y' in rules:
#             if df.columns.levels[1][0]!='@':
#                 if vk.split('|')[1].startswith('tests.'):
#                     df = verify_test_results(df)

#     else:
#         if show=='text':
#             df = paint_dataframe(
#                 df, meta, 
#                 create_full_index=full, 
#                 rules=rules
#             )
#         else:
#             text_key = {'x': [show], 'y': [show]}
#             df = paint_dataframe(
#                 df, meta, 
#                 text_key=text_key, 
#                 create_full_index=full, 
#                 rules=rules
#             )

#     # Make sure that all the margins, if present, 
#     # appear first on their respective axes
#     df = prepend_margins(df)

#     return df

def verify_test_results(df):
    """ 
    Verify tests results in df are consistent with existing columns. 
    
    This function verifies that all of the test results present in df
    only refer to column headings that actually exist in df. This is
    needed after rules have been applied at which time some columns
    may have been dropped.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The view dataframe showing column tests results.

    Returns
    -------
    df : pandas.DataFrame
        The view dataframe showing edited column tests results.
    """
      
    def verify_test_value(value):
        """
        Verify a specific test value.
        """
        if isinstance(value, str):
            len_value = len(value)
            if len(value)==1:
                value = set(value)
            else:
                value = set([int(i) for i in list(value[1:-1].split(','))])
            value = cols.intersection(value)
            if not value:
                return np.NaN
            elif len(value)==1:
                return str(list(value))
            else:
                return str(sorted(list(value)))
        else:
            return value
    
    cols = set([int(v) for v in zip(*[c for c in df.columns])[1]])
    df = df.applymap(verify_test_value)
    
    return df

# def prepend_margins(df):
#     """
#     Ensures that the margins in df appear first on each axis. 
#     """

#     x_col = df.index.levels[0][0]
#     if not (x_col, '@') in df.index:
#         margin = (x_col, 'All')
#         if margin in df.index:
#             if not df.index[0] == margin:
#                 margin = [margin]
#                 others = [c for c in df.index if c[1] != 'All']
#                 df = df.T[margin+others].T

#     y_col = df.columns.levels[0][0]
#     if not (y_col, '@') in df.columns:
#         margin = (y_col, 'All')
#         if margin in df.columns:
#             if not df.columns[0] == margin:
#                 margin = [margin]
#                 others = [c for c in df.columns if c[1] != 'All']
#                 df = df[margin+others]

#     return df

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
        Should the data in ds2 (where found) be appended to items from
        ds1? If False, data from ds2 (where found) will overwrite
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
        ds2 = condense_dichotomous_set(ds)
        series = join_delimited_set_series(series, ds2, append)
        ## Remove potential duplicate values
        ds = series.str.get_dummies(';')
        # Make sure columns are in numeric order
        ds.columns = [int(float(c)) for c in ds.columns]
        cols = sorted(ds.columns.tolist())
        ds = ds[cols] 
        ds.columns = [str(i) for i in ds.columns]
        # Reconstruct the dichotomous set
        series = condense_dichotomous_set(ds)
        
    elif qtype in ['single', 'int', 'float']:
        for key, idx in index_mapper.iteritems():
            series.loc[idx] = key
    else:
        raise TypeError(
            "Can't recode '{col}'. Recoding for '{typ}' columns is not"
            " yet supported.".format(col=series.name, typ=qtype) 
        )
        
    return series

def recode(meta, data, target, mapper, default=None, append=False,
           intersect=None, initialize=None, fillna=None):
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
    default : str, default=None
        The column name to default to in cases where unattended lists
        are given in your logic, where an auto-transformation of 
        {key: list} to {key: {default: list}} is provided. Note that
        lists in logical statements are themselves a form of shorthand
        and this will ultimately be interpreted as:
        {key: {default: has_any(list)}}.
    append : bool, default=False
        Should the new recodd data be appended to values already found
        in the series? If False, data from series (where found) will
        overwrite whatever was found for that item instead.
    intersect : logical statement, default=None
        If a logical statement is given here then it will be used as an
        implied intersection of all logical conditions given in the
        mapper.
    initialize : str or np.NaN, default=None
        If not None, a copy of the data named column will be used to
        populate the target column before the recode is performed.
        Alternatively, initialize can be used to populate the target
        column with np.NaNs (overwriting whatever may be there) prior
        to the recode.
    fillna : int, default=None
        If not None, the value passed to fillna will be used on the
        recoded series as per pandas.Series.fillna().

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

    # Check target
    if not isinstance(target, (str, unicode)):
        raise ValueError("The value for 'target' must be a string.")
    if not target in meta['columns']:
        raise ValueError("'%s' not found in meta['columns']." % (target))
    
    # Check append
    if not isinstance(append, bool):
        raise ValueError("'append' must be boolean.")

    # Check column type vs append
    if append and meta['columns'][target]['type']!="delimited set":
        raise TypeError("'{}' is not a delimited set, cannot append.")

    # Check default
    if not default is None:
        if not isinstance(default, (str, unicode)):
            raise ValueError("The value for 'default' must be a string.")
        if not default in meta['columns']:
            raise ValueError("'%s' not found in meta['columns']." % (default))

    # Check initialize
    initialize_is_string = False
    if not initialize is None:
        if isinstance(initialize, (str, unicode)):
            initialize_is_string = True
            if not initialize in meta['columns']:
                raise ValueError("'%s' not found in meta['columns']." % (target))
        elif not np.isnan(initialize):
            raise ValueError(
                "The value for 'initialize' must either be"
                " a string naming an existing column or np.NaN.")
    
    # Apply any implied intersection
    if not intersect is None:
        print ''
        mapper = {
            key: intersection([
                intersect, 
                value if isinstance(value, dict) else {default: value}])
            for key, value in mapper.iteritems()
        }

    # Resolve the logic to a mapper of {key: index}
    index_mapper = get_index_mapper(meta, data, mapper, default)
    
    # Get/create recode series
    if not initialize is None:
        if initialize_is_string:
            # Start from a copy of another existing column
            series = data[initialize].copy()
        else:
            # Ignore existing series for target, start with NaNs
            series = pd.Series(np.NaN, index=data.index, copy=True)
    elif target in data.columns:
        # Start with existing target column
        series = data[target].copy()
    else:
        # Start with NaNs
        series = pd.Series(np.NaN, index=data.index, copy=True)
    
    # Name the recoded series
    series.name = target

    # Use the index mapper to edit the target series
    series = recode_from_index_mapper(meta, series, index_mapper, append)

    # Rename the recoded series
    series.name = target

    if not fillna is None:
        col_type = meta['columns'][series.name]['type']
        if col_type=='single':
            series.fillna(fillna, inplace=True)
        elif col_type=='delimited set':
            series.fillna('{};'.format(fillna))
            
    return series  

def merge_text_meta(left_text, right_text, overwrite=False):
    """
    Merge known text keys from right to left, add unknown text_keys.
    """
    if overwrite:
        left_text.update(right_text)
    else:
        for text_key in right_text.keys():
            if not text_key in left_text:
                left_text[text_key] = right_text[text_key]

    return left_text

def merge_values_meta(left_values, right_values, overwrite=False):
    """
    Merge known left values from right to left, add unknown values.
    """
    for val_right in right_values:
        found = False
        for i, val_left in enumerate(left_values):
            if val_left['value']==val_right['value']:
                found = True
                left_values[i]['text'] = merge_text_meta(
                    val_left['text'], 
                    val_right['text'],
                    overwrite=overwrite)
        if not found:
            left_values.append(val_right)
            
    return left_values

def merge_column_metadata(left_column, right_column, overwrite=False):
    """
    Merge the metadata from the right column into the left column.
    """

    left_column['text'] = merge_text_meta(
        left_column['text'], 
        right_column['text'],
        overwrite=overwrite)
    if 'values' in left_column:
        left_column['values'] = merge_values_meta(
            left_column['values'], 
            right_column['values'],
            overwrite=overwrite)

    return left_column

def merge_meta(meta_left, meta_right, columns, from_set,
               overwrite_text=False, verbose=True):

    if verbose:
        print '\n', 'Merging meta...'
    col_updates = []
    for col_name in columns:
        if verbose:
            print '...', col_name
        # emulate the right meta
        right_column = emulate_meta(
            meta_right, 
            meta_right['columns'][col_name])
        if col_name in meta_left['columns'] and col_name in columns:
            col_updates.append(col_name)
            # emulate the left meta
            left_column = emulate_meta(
                meta_left,
                meta_left['columns'][col_name])
            # merge the eumlated metadata
            meta_left['columns'][col_name] = merge_column_metadata(
                left_column, 
                right_column,
                overwrite=overwrite_text)
        else:
            # add metadata
            meta_left['columns'][col_name] = right_column
        mapper = 'columns@{}'.format(col_name)
        if not mapper in meta_left['sets'][from_set]['items']:
            meta_left['sets'][from_set]['items'].append(
                'columns@{}'.format(col_name))

    return meta_left

def get_columns_from_mask(meta, mask_name):
    """
    Recursively retrieve the columns indicated by the named mask.
    """

    cols = []
    for item in meta['masks'][mask_name]['items']:
        source, name = item['source'].split('@')
        if source=='columns':
            cols.append(name)
        elif source=='masks':
            cols.extend(get_columns_from_mask(meta, name))
        elif source=='sets':
            cols.extend(get_columns_from_set(meta, name))
        else:
            raise KeyError(
                "Unsupported meta-mapping: {}".format(item))

    return cols    

def get_columns_from_set(meta, set_name):
    """
    Recursively retrieve the columns indicated by the named set.
    """

    cols = []
    for item in meta['sets'][set_name]['items']:
        source, name = item.split('@')
        if source=='columns':
            cols.append(name)
        elif source=='masks':
            cols.extend(get_columns_from_mask(meta, name))
        elif source=='sets':
            cols.extend(get_columns_from_set(meta, name))
        else:
            raise KeyError(
                "Unsupported meta-mapping: {}".format(item))
    
    return cols    

def get_masks_from_mask(meta, mask_name):
    """
    Recursively retrieve the masks indicated by the named mask.
    """

    masks = []
    for item in meta['masks'][mask_name]['items']:
        source, name = item['source'].split('@')
        if source=='masks':
            masks.append(name)
        elif source=='columns':
            pass
        elif source=='sets':
            masks.extend(get_masks_from_set(meta, name))
        else:
            raise KeyError(
                "Unsupported meta-mapping: {}".format(item))

    return masks

def get_masks_from_set(meta, set_name):
    """
    Recursively retrieve the masks indicated by the named set.
    """

    masks = []
    for item in meta['sets'][set_name]['items']:
        source, name = item.split('@')
        if source=='masks':
            masks.append(name)
        elif source=='columns':
            pass
        elif source=='sets':
            masks.extend(get_masks_from_mask(meta, name))
        else:
            raise KeyError(
                "Unsupported meta-mapping: {}".format(item))
    
    return masks    

def get_sets_from_mask(meta, mask_name):
    """
    Recursively retrieve the sets indicated by the named mask.
    """

    sets = []
    for item in meta['masks'][mask_name]['items']:
        source, name = item['source'].split('@')
        if source=='sets':
            sets.append(name)
        elif source=='columns':
            pass
        elif source=='masks':
            sets.extend(get_sets_from_mask(meta, name))
        else:
            raise KeyError(
                "Unsupported meta-mapping: {}".format(item))

    return sets

def get_sets_from_set(meta, set_name):
    """
    Recursively retrieve the sets indicated by the named set.
    """

    sets = []
    for item in meta['sets'][set_name]['items']:
        source, name = item.split('@')
        if source=='sets':
            sets.append(name)
        elif source=='columns':
            pass
        elif source=='masks':
            sets.extend(get_sets_from_mask(meta, name))
        else:
            raise KeyError(
                "Unsupported meta-mapping: {}".format(item))
    
    return sets    

def hmerge(dataset_left, dataset_right, on=None, left_on=None, right_on=None,
           overwrite_text=False, from_set=None, verbose=True):
    """
    Merge Quantipy datasets together using an index-wise identifer.

    This function merges two Quantipy datasets (meta and data) together,
    updating variables that exist in the left dataset and appending 
    others. New variables will be appended in the order indicated by
    the 'data file' set if found, otherwise they will be appended in
    alphanumeric order. This merge happend horizontally (column-wise).
    Packed kwargs will be passed on to the pandas.DataFrame.merge() 
    method call, but that merge will always happen using how='left'.

    Parameters
    ----------
    dataset_left : tuple
        A tuple of the left dataset in the form (meta, data).
    dataset_right : tuple
        A tuple of the right dataset in the form (meta, data). 
    on : str, default=None
        The column to use as a join key for both datasets.
    left_on : str, default=None
        The column to use as a join key for the left dataset.
    right_on : str, default=None
        The column to use as a join key for the right dataset.
    overwrite_text : bool, default=False
        If True, text_keys in the left meta that also exist in right 
        meta will be overwritten instead of ignored.
    from_set : str, default=None
        Use a set defined in the right meta to control which columns are
        merged from the right dataset.
    verbose : bool, default=True
        Echo progress feedback to the output pane.
        
    Returns
    -------
    meta, data : dict, pandas.DataFrame
        Updated Quantipy dataset.
    """

    # This will be passed into pd.DataFrame.merge()
    kwargs = {}

    if all([kwarg is None for kwarg in [on, left_on, right_on]]):
        raise TypeError(
            "You must provide a column name for either 'on' or both"
            " 'left_on' AND 'right_on'")

    if not on is None:
        if not left_on is None or not right_on is None:
            raise ValueError(
                "You cannot provide a value for both 'on' and either/"
                "both 'left_on'/'right_on'.") 
        left_on = on
        right_on = on

    meta_left = cpickle_copy(dataset_left[0])
    data_left = dataset_left[1].copy()

    meta_right = cpickle_copy(dataset_right[0])
    data_right = dataset_right[1].copy()

    if verbose:
        print '\n', 'Checking metadata...'

    if from_set is None:
        from_set = 'data file'

    if from_set in meta_right['sets']:
        if verbose:
            print (
                "New columns will be appended in the order found in"
                " meta['sets']['{}'].".format(from_set)
            )
        # Collect columns for merge
        cols = get_columns_from_set(meta_right, from_set)
        # Collect masks for merge
        masks = get_masks_from_set(meta_right, from_set)
        masks = [key for key in masks if not key in meta_left['masks']]
        if masks:
            for mask_name in sorted(masks):
                if verbose:
                    print "Adding meta['masks']['{}']".format(mask_name)
                meta_left['masks'][mask_name] = meta_right['masks'][mask_name]
        # Collect sets for merge
        sets = get_sets_from_set(meta_right, from_set)
        sets = [key for key in sets if not key in meta_left['sets']]
        if sets:
            for set_name in sorted(sets):
                if verbose:
                    print "Adding meta['sets']['{}']".format(set_name)
                meta_left['sets'][set_name] = meta_right['sets'][set_name]
    else:
        if verbose:
            print (
                "No '{}' set was found, new columns will be appended"
                " alphanumerically.".format(from_set)
            )
        cols = meta_right['columns'].keys().sort(key=str.lower)

    # Find th columns that are being updated rather than added
    col_updates = list(set(meta_left['columns'].keys()).intersection(set(cols)))

    # Merge the right meta into the left meta
    meta_left = merge_meta(
        meta_left, meta_right, 
        cols, from_set, 
        overwrite_text, verbose)
    
    kwargs['left_on'] = left_on
    kwargs['right_on'] = right_on
    
    # col_updates exception when left_on==right_on
    if left_on==right_on and not left_on is None:
        col_updates.remove(left_on)

    # hmerge must operate on a 'left' basis
    kwargs['how'] = 'left'

    if verbose:
        print '\n', 'Merging data...'
    if col_updates:
        if not left_on is None:
            updata_left = data_left.set_index(
                [left_on]
            )[col_updates].copy()
        else:
            updata_left = data_left.copy()

        if not right_on is None:
            updata_right = data_right.set_index(
                [right_on]
            )[col_updates].copy()
        else:
            updata_right = data_right.copy()

        if verbose:
            print '------ updating data for known columns'
        # print updata_left.head()
        # print updata_right.head()
        updata_left.update(updata_right)
        for update_col in col_updates:
            if verbose:
                print "..{}".format(update_col)
            data_left[update_col] = updata_left[update_col].astype(
                data_left[update_col].dtype).values

    if verbose:
        print '------ appending new columns'
    new_cols = [col for col in cols if not col in col_updates]
    data_left = data_left.merge(data_right[new_cols], **kwargs)
    if verbose:
        for col_name in new_cols:
            print '..{}'.format(col_name)

    return meta_left, data_left

def vmerge(dataset_left, dataset_right, on=None, left_on=None, right_on=None,
           row_id_name=None, left_id=None, right_id=None,
           overwrite_text=False, from_set=None, reset_index=True, verbose=True):
    """
    Merge Quantipy datasets together by appending rows.

    This function merges two Quantipy datasets (meta and data) together,
    updating variables that exist in the left dataset and appending 
    others. New variables will be appended in the order indicated by
    the 'data file' set if found, otherwise they will be appended in
    alphanumeric order. This merge happens vertically (row-wise).

    Parameters
    ----------
    dataset_left : tuple
        A tuple of the left dataset in the form (meta, data).
    dataset_right : tuple
        A tuple of the right dataset in the form (meta, data). 
    on : str, default=None
        The column to use to identify unique rows in both datasets.
    left_on : str, default=None
        The column to use to identify unique in the left dataset.
    right_on : str, default=None
        The column to use to identify unique in the right dataset.
    overwrite_text : bool, default=False
        If True, text_keys in the left meta that also exist in right 
        meta will be overwritten instead of ignored.
    from_set : str, default=None
        Use a set defined in the right meta to control which columns are
        merged from the right dataset.
    verbose : bool, default=True
        Echo progress feedback to the output pane.
    **kwargs : various
        As per pandas.DataFrame.merge().
        
    Returns
    -------
    meta, data : dict, pandas.DataFrame
        Updated Quantipy dataset.
    """

    if on is None and left_on is None and right_on is None:
        blind_append = True
    else:
        blind_append = False
        if on is None:
            if left_on is None or right_on is None:
                raise ValueError(
                    "You may not provide a value for only one of"
                    "'left_on'/'right_on'.")
        else:
            if not left_on is None or not right_on is None:
                raise ValueError(
                    "You cannot provide a value for both 'on' and either/"
                    "both 'left_on'/'right_on'.") 
            left_on = on
            right_on = on

    meta_left = cpickle_copy(dataset_left[0])
    data_left = dataset_left[1].copy()
    
    if not blind_append:
        if not left_on in data_left.columns:
            raise KeyError(
                "'{}' not found in the left data.".format(left_on))
        if not left_on in meta_left['columns']:
            raise KeyError(
                "'{}' not found in the left meta.".format(left_on))

    meta_right = cpickle_copy(dataset_right[0])
    data_right = dataset_right[1].copy()
    
    if not blind_append:
        if not right_on in data_left.columns:
            raise KeyError(
                "'{}' not found in the right data.".format(right_on))
        if not right_on in meta_left['columns']:
            raise KeyError(
                "'{}' not found in the right meta.".format(right_on))

    if not row_id_name is None:
        if left_id is None or right_id is None:
            raise TypeError(
                "When indicating a 'row_id_name' you must also"
                " provide both 'left_id' and 'right_id'.")
            
        if not row_id_name in meta_left['columns']:
            left_id_int = isinstance(left_id, (int, np.int64))
            right_id_int = isinstance(right_id, (int, np.int64))
            if left_id_int and right_id_int:
                id_type = 'int'
            else:
                left_id_float = isinstance(left_id, (float, np.float64))
                right_id_float = isinstance(right_id, (float, np.float64))
                if (left_id_int or left_id_float) and (right_id_int or right_id_float):
                    id_type = 'float'
                    left_id = float(left_id)
                    right_id = float(right_id)
                else:
                    id_type = 'str'
                    left_id = str(left_id)
                    right_id = str(right_id)
            if verbose:
                print (
                    "'{}' was not found in the left meta so a new"
                    " column definition will be created for it. Based"
                    " on the given 'left_id' and 'right_id' types this"
                    " new column will be given the type '{}'".format(
                        row_id_name,
                        id_type))
            text_key = meta_left['lib']['default text']
            meta_left['columns'][row_id_name] = {
                'name': row_id_name,
                'type': id_type,
                'text': {text_key: 'vmerge row id'}}
            id_mapper = "columns@{}".format(row_id_name)
            if not id_mapper in meta_left['sets']['data file']['items']:
                meta_left['sets']['data file']['items'].append(id_mapper)
                
            # Add the left and right id values
            if row_id_name in data_left.columns:
                left_id_rows = data_left[row_id_name].isnull()
                data_left.ix[left_id_rows, row_id_name] = left_id
            else:
                data_left[row_id_name] = left_id
            data_right[row_id_name] = right_id

    if verbose:
        print '\n', 'Checking metadata...'

    if from_set is None:
        from_set = 'data file'

    if from_set in meta_right['sets']:
        if verbose:
            print (
                "New columns will be appended in the order found in"
                " meta['sets']['{}'].".format(from_set)
            )
        # Collect columns for merge
        cols = get_columns_from_set(meta_right, from_set)
        # Collect masks for merge
        masks = get_masks_from_set(meta_right, from_set)
        masks = [key for key in masks if not key in meta_left['masks']]
        if masks:
            for mask_name in sorted(masks):
                if verbose:
                    print "Adding meta['masks']['{}']".format(mask_name)
                meta_left['masks'][mask_name] = meta_right['masks'][mask_name]
        # Collect sets for merge
        sets = get_sets_from_set(meta_right, from_set)
        sets = [key for key in sets if not key in meta_left['sets']]
        if sets:
            for set_name in sorted(sets):
                if verbose:
                    print "Adding meta['sets']['{}']".format(set_name)
                meta_left['sets'][set_name] = meta_right['sets'][set_name]
    else:
        if verbose:
            print (
                "No '{}' set was found, new columns will be appended"
                " alphanumerically.".format(from_set)
            )
        cols = meta_right['columns'].keys().sort(key=str.lower)

    # Merge the right meta into the left meta
    meta_left = merge_meta(
        meta_left, meta_right, 
        cols, from_set, 
        overwrite_text, verbose)
    
    if not blind_append:
        vmerge_slicer = data_right[left_on].isin(data_left[right_on])
        data_right = data_right.loc[~vmerge_slicer]
        
    vdata = pd.concat([
        data_left,
        data_right
    ])
    
    col_slicer = data_left.columns.tolist() + [
        col for col in data_right.columns.tolist()
        if not col in data_left.columns]
    
    vdata = vdata[col_slicer]
    
    if reset_index:
        vdata.reset_index(inplace=True)
        idx_col = vdata.columns[0]
        vdata.drop(idx_col, axis=1, inplace=True)
    
    return meta_left, vdata

def subset_dataset(meta, data, columns):
    """
    Get a subset of the given meta
    """
    
    sdata = data[columns].copy()
    
    smeta = start_meta(text_key='en-GB')
    
    for col in columns:
        smeta['columns'][col] = meta['columns'][col]
    
    for col_mapper in meta['sets']['data file']['items']:
        if col_mapper.split('@')[-1] in columns:
            smeta['sets']['data file']['items'].append(col_mapper)
    
    return smeta, sdata
