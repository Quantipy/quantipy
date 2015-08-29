import numpy as np
import quantipy as qp
from quantipy.core.helpers.functions import create_full_index_dataframe

def set_fullname(pos, method_name, relation, rel_to, weights, view_name):
    '''
    Sets the view's fullname: the fullname is the key for the view 
    under which it can be queried from the stack. It differs from the short view_name
    that can be chosen arbitrarily by the user by pre-fixing systematic information from
    kwargs parameters. The naming convention is as follows:
    
    pos|method_name|relation|rel_to|weights|view_name
    
    pos:            position of the view method result given the Link defintion
    method_name:    name of the originating method
    relation:       relationship of the calculation given the Link's x and y codes 
    rel_to:         indicates wether the calculation is in absolute terms 
                    or relative to another view
    weights:        the name of the weight (if any)
    view_name:      the short name of the view

    Examples
    --------
    x|frequency|||weight|counts
    pos:            results sits on the x axis
    method_name:    freq
    relation:       empty; the calculation is for x value TO y value
    rel_to:         empty; the calculation is not relative to anything, absolute counts
    weights:        the name of the weight variable has been "weight"
    counts:         short-name for the counts aggregation

    x|frequency||y|weight|c%
    pos:            results sits on the x axis
    method_name:    freq
    relation:       empty; the calculation is for x value TO y value
    rel_to:         the calculation is expressed relative to the y axis (the column base)
    weights:        the name of the weight variable has been "weight"
    c%:             short-name for the column percentage view

    x|means|x:y||weight|cMean
    pos:            results sits on the x axis
    method_name:    means
    relation:       calculation of x FOR EACH y value
    rel_to:         empty: the mean is not relative to anything
    weights:        the name of the weight variable has been "weight"
    cMean:          short-name for the column mean 

    Parameters
    ----------
    pos, method_name, 
    relation, rel_to, 
    weights, view_name : str

    Returns
    -------
    view fullname key : str
    '''
    if weights is None:
        weights = ''
    if rel_to is None:
        rel_to = ''
    if relation is None:
        relation = ''

    return '%s|%s|%s|%s|%s|%s' %(pos, method_name, relation, rel_to, weights, view_name)
        
def get_std_kwargs(kwargs_dict):
    '''
    Used to return a tuple of the mandatory kwargs Quantipy's view 
    methods are required to know. 

    Parameters
    ----------
    kwargs_dict : dictionary

    Returns
    -------
    tuple of str : pos, relation, rel_to, weights, groups
    '''
    return (
        kwargs_dict.get('pos', 'x'), 
        kwargs_dict.get('relation', None),
        kwargs_dict.get('rel_to', None), 
        kwargs_dict.get('weights', None),
        kwargs_dict.get('groups', None),
        kwargs_dict.get('text', '')
        )
            
def get_source_key(pos, rel_to, weights):
    '''
    Used to return the the fullname of a view to operaten on inside a view method.

    CAUTION: Currently only works for simple "freq" views.

    Parameters
    ----------
    pos, rel_to, weights : str

    Returns
    -------
    view fullname key : str
    '''
    if weights is None:
        weights = ''
    if rel_to is None:
        rel_to = ''
        name = 'counts'
    elif rel_to == 'y':
        name = 'c%'
    else:
        name = 'r%'

    return '%s|frequency|||%s|counts' %(pos, weights)
        
def get_rel_to_key(rel_to, weights):
    '''
    Used to return the the fullname of a view that serves as 
    the reference view for an operation that used the rel_to
    parameter.

    CAUTION: Currently only works for simple "base" views.

    Parameters
    ----------
    rel_to, weights : str

    Returns
    -------
    view fullname key : str
    '''
    if weights is None:
        weights = ''
    if rel_to == 'y':
        return 'x|frequency|x:y||%s|cbase' %(weights)
    else:
        return 'y|frequency|y:x||%s|rbase' %(weights)

def set_num_stats_relation(link, exclude, rescale):
    '''
    Used to implement the relation-part of the view name
    notation while means are still restricted to appearing
    on x.

    CAUTION: Only works for mean on x.

    Parameters
    ----------
    link : Link

    Returns
    ----------
    relation-part notation : str
    '''
    try:
        if '[{' in link.x:
            set_name = link.x.split('[{')[0] + link.x.split('}]')[-1]
            x_values = [int(x['value']) for x in link.get_meta()['lib']['values'][set_name]]
        else:
            x_values = [int(x['value']) for x in link.get_meta()['columns'][link.x]['values']]
        if exclude:
            x_values = [x for x in x_values if not x in exclude]
        if rescale:
            x_values = [x if not x in rescale else rescale[x] for x in x_values]
        if exclude or rescale:
            relation = 'x%s:y' % (str(x_values).replace(' ', ''))
        else:
            relation = 'x:y'
    except:
        relation = 'x:y'
    
    return relation

def get_num_stats_fullname_from_subset(link, subset, weights):
    '''
    '''
    if weights is None:
        weights = ''
    subset_name = ''.join(subset.keys())
    kwargs = subset[subset_name]['kwargs']
    exclude = kwargs.get('exclude', None)
    rescale = kwargs.get('rescale', None)
    relation = set_num_stats_relation(link, exclude, rescale)
    return 'x|mean|%s||%s|%s' %(relation, weights, subset_name)

def get_num_stats_relation_from_fullname(fullname):
    '''
    '''
    return fullname.split('|',3)[2]

def slicex(df, values, keep_margins=True):
    """
    Return an index-wise slice of df, keeping margins if desired.
    
    Assuming a Quantipy-style view result this function takes an index
    slice of df as indicated by values and returns the result.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe that should be sliced along the index.
    values : list-like
        A list of index values that should be sliced from df.
    keep_margins : bool, default=True
        If True and the margins index row exists, it will be kept. 
    
    Returns
    -------
    df : list
        The sliced dataframe. 
    """
    
    name_x = df.index.levels[0][0]
    slicer = [(name_x, value) for value in values]
    if keep_margins and (name_x, 'All') in df.index:
        slicer = [(name_x, 'All')] + slicer

    df = df.loc[slicer]

    return df

def sortx(df, sort_on='All', ascending=False, fixed=None):
    """
    Sort the index of df on a column, keeping margins and fixing values.
    
    This function sorts df, which is assumed to be a Quantipy-style
    view result with appropriate index/column structure, using
    a given column, while maintaining the position of margins if
    they exist, and also optionally fixing certain values at the
    bottom of the result without sorting them. Note that nested
    variable view results are not yet supported.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The Quantipy-style view result to be sorted
    sort_on : str or int, default='All'
        The column (on the innermost level of the column's
        MultiIndex) on which to sort.
    ascending : bool, default=False
        Sort ascending vs. descending. Default descending for
        easier application to MR use cases.
    fixed : list-like, default=None
        A list of index values that should appear underneath
        the sorted index values.
    
    Returns
    -------
    df : pandas.DataFrame
        The sorted df. 
    """
    
    # Get question names for index and columns from the
    # index/column level 0 values
    name_x = df.index.levels[0][0]
    name_y = df.columns.levels[0][0]
    
    if (name_x, 'All') in df.index:
        # Get the margin slicer
        s_all = [(name_x, 'All')]
        # Get non-margin index slicer for the sort
        # (if fixed has been used it will be edited)
        s_sort = df.drop((name_x, 'All')).index.tolist()
    else:
        s_all = []
        s_sort = df.index.tolist()
    
    # Get fixed slicer
    if fixed is None:
        s_fixed = []
    else:
        s_fixed = [(name_x, value) for value in fixed]
        # Drop fixed tuples from the sort slicer
        s_sort = [t for t in s_sort if not t in s_fixed]
    
    # Get sorted slicer
    if (name_y, sort_on) in df.columns:
        sort_col = (name_y, sort_on)
    elif (name_y, str(sort_on)) in df.columns:
        sort_col = (name_y, str(sort_on))
    df_sorted = df.loc[s_sort].sort_index(0, sort_col, ascending)
    s_sort = df_sorted.index.tolist()
    
    df = df.loc[s_all+s_sort+s_fixed]
    
    return df

def dropx(df, values):
    """
    Return df after dropping values from the index.
    
    Assuming a Quantipy-style view result this function drops index
    rows indicated by values and returns the result.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe that should have some index rows dropped.
    values : list-like
        A list of index values that should be dropped from the index.
    
    Returns
    -------
    df : list
        The edited dataframe. 
    """
    
    name_x = df.index.levels[0][0]
    slicer = [(name_x, value) for value in values]

    if not all([s in df.index for s in slicer]):
        raise KeyError (
            "Some of of the values from the list %s cannot be dropped"
            " from the dataframe because they were not found in %s."
            " Be careful that you are not both slicing and/or sorting"
            " any values that you are also trying to drop." % (
                values,
                df.index.tolist()
            )
        )

    df = df.drop(slicer)

    return df

def get_dataframe(obj, described=None, loc=None, keys=None, 
                  full=False, show='values', rules=False, 
                  verbose=False):
    """
    Convenience function for extracting a single dataframe from a stack.
    
    This function either uses a string of keys sliced out of 
    obj.describe() or the result of same, or a string of keys provided
    by the user, to return a single, targeted dataframe from obj. 
    Optionally, the exact keys used in the extraction can be printed to
    the output window for verification.
    
    Parameters
    ----------
    obj : quantipy.Stack or quantipy.Chain
        The stack or chain from which the dataframe should be extracted.
    described : pandas.DataFrame, default=None
        If given, this will be used with loc to identify the string of
        targeted keys. This parameter is provided to reduce repeated
        calls to obj.describe() when this function is being used in a
        loop.
    loc : int, default=None
        The .loc[] indexer that can be used on described or 
        obj.describe() to isolate the targeted dataframe.
    keys : list-like, default=None
        A list of five keys (dk, fk, xk, yk, vk) that can be used on obj
        to isolate the targeted dataframe.
    full : bool, default=False
        If True, the returned dataframe will have a full index applied.
        Note that rules will NOT be applied if full=True.
    show : str, default='values'
        How the index and columns should be displayed. 'values' returns 
        the raw value indexes. 'text' returns the text associated with 
        each value, according to the text key 
        meta['lib']['default text']. Any other str value is assumed to
        be a non-default text_key.  
    rules : bool, default=False
        Should any applicable rules (slicex, sortx, dropx) be applied to
        the dataframe before it is returned?
    verbose : bool, default=False
        If True, the keys used in the extraction will be printed to the
        output window.
        
    Returns
    -------
    df : pandas.DataFrame
        The targeted dataframe. 
    """

    # Error handling for both loc and keys being None
    if all([arg is None for arg in [loc, keys]]):
        raise ValueError (
            "You must provide a value for either loc or keys."
        )
    if not described is None:
        if not isinstance(described, pd.DataFrame):
            raise TypeError (
                "The describe argument must be a pandas.DataFrame."
            )
    # Error handling for both loc and keys being provided
    if all([not arg is None for arg in [loc, keys]]):
        raise ValueError (
            "You should not provide values for both loc and keys."
        )
    
    if not loc is None:
        # Use loc to generate keys
        if described is None:
            described = obj.describe()
        keys = described.loc[loc]

    # Split out pathway to the target dataframe
    dk = keys[0]
    fk = keys[1]
    xk = keys[2]
    yk = keys[3]
    vk = keys[4]
            
    if show:
        print 'dk:\t', dk
        print 'fk:\t', fk
        print 'xk:\t', xk
        print 'yk:\t', yk
        print 'vk:\t', vk
        print ''
        
    if not dk in obj.keys():
        raise KeyError('dk not found: {}'.format(dk))
    if not fk in obj[dk].keys():
        raise KeyError('fk not found: {}'.format(fk))
    if not xk in obj[dk][fk].keys():
        raise KeyError('xk not found: {}'.format(xk))
    if not yk in obj[dk][fk][xk].keys():
        raise KeyError('yk not found: {}'.format(yk))
    if not vk in obj[dk][fk][xk][yk].keys():
        raise KeyError('vk not found: {}'.format(vk))
    
    try:
        df = obj[dk][fk][xk][yk][vk].dataframe
    except:
        raise AttributeError (
            "The aggregation for this view must have failed,"
            " expected View instance under a view key that"
            " did already exist but found a Stack instead."
        )

    if full:
        # Use the show function to apply rules and return
        # full index values or text as requested.
        meta = obj[dk].meta
        df = qp.core.tools.dp.prep.show_df(df, meta, show, rules)
        
    if rules:
        
        # Determine if sorting is required on x or y
        meta = obj[dk].meta
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
                link = obj[dk][fk][xk][yk]  
                weight = vk.split("|")[-2]
                if weight=='': weight = None    
                # Add the missing margins to df
                df = add_margins(
                    df, link, weight,
                    x_margin=needs_x_margin,
                    y_margin=needs_y_margin
                )

        # Use the show function to apply rules and return
        # full index values or text as requested.
        df = qp.core.tools.dp.prep.show_df(df, meta, show, rules)
        
        # If the original dataframe didn't have any margins
        # to begin with, but now there are some due to the need
        # to apply sorting, then they should now be removed.
        if x_sortx or y_sortx:
            if needs_x_margin:
                df.drop((df.index.levels[0][0], 'All'), inplace=True)
            if needs_y_margin:
                df.drop((df.columns.levels[0][0], 'All'), inplace=True, axis=1)
            
    return df

def has_sorting_rules(meta, col_name):
    """
    Return if the named column has any sortx rules defined.
    """
        
    # Determine if sorting is required on x
    has_sortx = False
    col = meta['columns'][col_name]
    if 'rules' in col:
        rules = col['rules'].get('x', None)
        if not rules is None:
            has_sortx = 'sortx' in rules
    
    return has_sortx

def add_margins(df, link, weight, x_margin, y_margin):
    """
    Add missing margins to the view result df.

    This function uses a Quantity instance based on link and weight
    to add the index and column margins onto df.
    """

    xk = link.x
    yk = link.y
    q = qp.Quantity(link, weight=weight)
    
    # Extract the x, y and xy margins using
    # using Quantity methods
    x_all = q._col_n()[0]
    xy_all = [x_all[-1]]
    x_all = list(x_all[:-1])
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
            df[(yk, 'All')] = xy_all + x_all
        else:
            # Perpendicular margin is last
            df[(yk, 'All')] = x_all + xy_all
        
    elif x_margin and y_margin:
        # 3. Neither x nor y has a margin
        df[(yk, 'All')] = y_all
            # Perpendicular margin is last
        df = df.T
        df[(xk, 'All')] = x_all + xy_all
        df = df.T

    return df
