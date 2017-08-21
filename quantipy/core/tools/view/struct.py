import pandas as pd
import numpy as np
import json
import re
import copy
import itertools
import math
import re, string

from collections import OrderedDict
from quantipy.core.helpers.constants import DTYPE_MAP
from quantipy.core.helpers.constants import MAPPED_PATTERN
from itertools import product
from quantipy.core.view import View
from quantipy.core.view_generators.view_mapper import ViewMapper
from quantipy.core.helpers import functions

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

def deep_drop(df, targets, axes=[0, 1]):
    '''
    Drops all columns given in the targets list from the defined
    axes of the passed dataframe. The dataframe is allowed to be
    multiindexed on both axes.

    Parameters
    ---------
    df : pd.DataFrame

    targets : string or sequence of strings
        Columns to be dropped.

    axes : list of int, default = [0, 1]
        Specification of the axes to drop from.
        Will perform the drop on both axes by default.

    Returns
    -------
    df : pd.Dataframe
    '''
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
    codes = _partition_view_df(from_source, axes_only=True)
    if to == 'x':
        df.rename(index={code[0]: code[1] for code in enumerate(codes[0])}, inplace=True)
        df.rename(columns={code[0]: code[1] for code in enumerate(codes[1])}, inplace=True)
    elif to == 'y':
        df.rename(columns={code[0]: code[1] for code in enumerate(codes[1])}, inplace=True)
    elif to == 'both':
        df.rename(index={code[0]: code[1] for code in enumerate(codes[0])}, inplace=True)
        df.rename(columns={code[0]: code[1] for code in enumerate(codes[1])}, inplace=True)

    return df

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
        df.columns = pd.MultiIndex.from_product([[x], df.columns], names=axis_labels)
    else:
        df.columns = pd.MultiIndex.from_product([[y], df.columns], names=axis_labels)

    return df


def _partition_view_df(view, values=False, data_only=False, axes_only=False):
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
    if isinstance(df.index, pd.MultiIndex):
        df.index = df.index.droplevel()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel()
    index = df.index
    columns = df.columns
    data = df if not values else df.values
    
    if data_only:
        return data
    elif axes_only:
        return index.tolist(), columns.tolist()
    else:
        return data, index.tolist(), columns.tolist()

