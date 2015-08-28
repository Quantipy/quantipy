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

def get_dataframe(obj, described=None, loc=None, keys=None, show=False):
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
        The .loc[] indexer that can be used on obj.describe() to isolate
        the targeted dataframe.
    keys : list-like, default=None
        A list of five yes (dk, fk, xk, yk, vk) that can be used on obj
        to isolate the targeted dataframe.
    show : bool, default=False
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
        if loc is None:
            raise ValueError (
                "When providing describe you must also provide loc."
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
         
    return df

