def view_meta(link, viewdf, weight, method_name, method_type,
              name, fullname, text='', groups=None):
    # Get file meta info on link definition data types
    x = link.x if not link.x == '@' else link.y
    y = link.y if not link.y == '@' else link.x
    filemeta = link.get_meta()

    if x in filemeta['columns']:
        xtype = filemeta['columns'][x]['type']
    elif link.x in filemeta['masks']:
        xtype = filemeta['masks'][x]['type']
    if y in filemeta['columns']:
        ytype = filemeta['columns'][y]['type']
    elif y in filemeta['masks']:
        ytype = filemeta['masks'][y]['type']
    mc = ['dichotomous set', 'categorical set', 'delimited set']
    num = ['float', 'int']
    cat = mc + ['single']
    arr = ['array']
    if xtype in num:
        datatype = 'numeric'
    elif xtype in cat:
        datatype = 'categorical'
    elif xtype in arr:
        datatype = 'array'
    # Structure view meta information in regular dict format
    viewmeta = {
                'agg':
                {
                 'datatype': datatype,
                 'viewtype': method_type,
                 'is_weighted': True if weight is not None else False,
                 'weights': weight,
                 'method': method_name,
                 'name': name,
                 'fullname': fullname,
                 'text': text,
                 'groups': groups,
                 },
                'x':
                {
                 'name': link.x,
                 'is_multi': True if xtype in mc else False,
                 'is_nested': True if ">" in link.x else False
                 },
                'y':
                {
                 'name': link.y,
                 'is_multi': True if ytype in mc else False,
                 'is_nested': True if ">" in link.y else False
                 },
                'shape': viewdf.shape
                }
    
    return viewmeta

def default_meta(link, view_df, dtypes, weights=None):
    '''
    Sets up the initial meta information for a Quantipy view dataframe.
    The information is derived in the generating process of the default 
    view in core.view_generators.view_maps. The information is structured
    inside a nested dict with the following keys:
    - agg       
    - x 
    - y
    - shape

    Parameters
    ----------
    link : Quantipy Link object
    
    view_df : pd:DataFrame

    dtypes : tuple of str
        The file meta type information on the
        x and y axis of the link.

    weights : str, default = None
        Name of the weight variable.
        Used to derive weight information.

    Parameters
    ----------
    default_meta : dict (nested)
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
                        'is_nested': True if ">" in link.x else False,
                        },
                    'y': {
                        'name': link.y,
                        'is_multi': True if dtypes[1] in ['dichotomous set', 'categorical set', 'delimited set'] and not link.y == '@' else False,
                        'is_nested': True if ">" in link.y else False
                        },
                    'shape' : view_df.shape
                    }

    return default_meta

def update_view_meta(view_df, meta, method_name, method_type, name, fullname, text='', groups=None):
    '''
    Updates the view meta information based on the selected view methods
    used in the QuantipyViews class from core.view_generators.view_maps.

    Parameters
    ----------
    view_df : pd.DataFrame

    meta : dict (nested)

    method_name, method_type,
    name, fullname,
    text : str

    groups : list
    
    Returns
    -------
    meta (updated) : dict (nested)
    '''
    meta['agg']['method'] = method_name
    meta['agg']['name'] = name
    meta['agg']['fullname'] = fullname
    meta['agg']['viewtype'] = method_type
    meta['agg']['text'] = text
    meta['agg']['groups'] = groups
    meta['shape'] = view_df.shape

    return meta

def full_num_stat_text(stat, text=''):
    '''
    Creates the full text (=label) meta for num_stats() view aggregations.
    The full text consists of the name of the figure and the passed suffix from
    the num_stats() view method's "text" kwarg.

    Parameters
    ----------
    stat, text : str
        Names of the stat. figure and
        the view method kwarg "text".

    Returns
    -------
    fulltext : str
        The text that is passed into the
        meta component of a Quantipy View.
    '''
    stat_labs = {
        'mean': 'Mean',
        'sem': 'Std. err. of mean',
        'median': 'Median',
        'stddev': 'Std. dev.',
        'var': 'Sample variance',
        'varcoeff': 'Coefficient of variation'
        }

    if text == '':
        return stat_labs[stat]
    else:
        return '%s %s' % (stat_labs[stat], text)

def full_multivariate_text(stat, text=''):
    '''
    UNITE WITH full_num_stats_text
    '''
    stat_labs = {
        'cov': 'Covariance',
        'corr': 'Correlation coefficient'
        }

    if text == '':
        return stat_labs[stat]
    else:
        return '%s %s' % (stat_labs[stat], text)

def set_num_stats_meta(exclude, rescale, meta):
    '''
    Applies additional meta information specifically for the
    .num_stats() view method's code exclusion and rescaling
    functionality.

    Parameters
    ----------
    exclude : list
        List fo codes that have been excluded.
    rescale : dict
        Dictionary mapping of old code: new code
    meta : dict
        The meta information generated/updated by
        the .num_stats().

    Returns
    -------
    None
    '''
    meta['x']['exclude'] = exclude
    meta['x']['rescale'] = rescale
        
    return None

    