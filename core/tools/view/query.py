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

def slicer(question, values, margins=True):
    """
    Return the tuple product of a and b, optionally including margins.
    
    This function uses a and b to generate a list of tuples that can
    be used to slice a Quantipy-style view result by index or columns,
    optionally adding in a tuple for the margins if desired.
    
    Parameters
    ----------
    question : str
        The first part of the desired product operation.
    values : list-like
        The second (iterable) part of the desired product operation.
    margins : bool, default=True
        If True, the tuple (question, 'All') will prepended to the 
        result.
    
    Returns
    -------
    slicer : list
        The list of tuples. 
    """

    slicer = [(question, value) for value in values]
    if margins:
        slicer = [(question, 'All')] + slicer

    return slicer

def sortx(df, sort_col='All', ascending=False, fixed=None):
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
    sort_col : str or int, default='All'
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
    il0 = df.index.levels[0][0]
    cl0 = df.columns.levels[0][0]
    
    # Get the margin slicer
    if (il0, 'All') in df.index:
        s_all = [(il0, 'All')]
    else:
        s_all = []
    
    # Get non-margin index slicer for the sort
    # (if fixed has been used it will be edited)
    s_sort = df.drop((il0, 'All')).index.tolist()
    
    # Get fixed slicer
    if fixed is None:
        s_fixed = []
    else:
        s_fixed = slicer(il0, fixed, margins=False)
        # Drop fixed tuples from the sort slicer
        s_sort = [t for t in s_sort if not t in s_fixed]
    
    # Get sorted slicer
    s_sort = df.loc[s_sort].sort_index(0, (cl0, sort_col), ascending).index.tolist()
    
    df = df.loc[s_all + s_sort + s_fixed]
    
    return df
