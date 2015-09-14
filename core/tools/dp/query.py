import quantipy as qp

def get_views(qp_structure):
    ''' Generator replacement for nested loops to return all view objects
        stored in a given qp container structure.
        Currently supports chain-classed shapes and cluster objects natively.
        To return views from a stack object instance provide input container as per
        qp_structure = < stack[data_key]['data'] >
    '''

    for k, v in qp_structure.iteritems():
        if not isinstance(v, qp.View):
            for item in get_views(v):
                yield item
        else:
            yield v

def get_variable_types(data, meta):
    """ Returns a dict of variable types to lists of variable names.

    Parameters
    ----------
    data : Pandas.DataFrame

    meta : Quantipy meta object pared to data

    Returns
    ----------
    Dict in the form {type_name: [variable_names], ...}

    """
    types = {
        'int': [],
        'float': [],
        'single': [],
        'delimited set': [],
        'string': [],
        'date': [],
        'time': [],
        'array': []
    }

    not_found = []
    for col in data.columns[1:]:
        try:
            types[meta['columns'][col]['type']].append(col)
        except:
            not_found.append(col)                

    for mask in meta['masks'].keys():
        types[meta['masks'][mask]['type']].append(mask)

    if not_found:
        print '%s not found in meta file. Ignored.' %(not_found)
    
    return types

def request_views(stack, weight=None, nets=True, descriptives=["mean"], 
                  coltests=True, mimic='Dim', sig_levels=[".05"]):
    """
    Get structured, request-ready views from the stack.

    This function uses the given parameters to inspect the view keys in
    a stack and return them in a standard, structured form suitable for
    use with ``stack.get_chain()`` and 
    ``qp.ExcelPainter(..., grouped_views)``. Configurations for 
    counts-only (``'c'``), percentages-only (``'p'``) and combintation 
    counts+percentages (``'cp'``) are returned.

    Parameters
    ----------
    stack : quantipy.Stack
        The stack instance view keys will be drawn from.
    weight : str, default=None
        The name of the weight variable to be used when requesting
        weighted views.
    nets : bool, default=True
        If True, net views (frequency with logic) will be included.
    descriptives : list-like or None, default=["mean"]
        If list-like, the given descriptive statistics will be included,
        (e.g. ["mean", "stddev", "stderr"]. If the list is empty or None
        is given instead, no descriptive statistics will be included.
    coltests : bool, default=True
        If True, column tests (proportions and means) will be included.
    mimic : str
        The mimic type to be targeted when finding coltests.
    sig_levels : list-like, default=[".05"]
        The level/s of significance being requested, e.g. [".05", ".10"]

    Returns
    -------
    requested_views : dict
        The returned object is a nested dict in the form :

    ::

        requested_views = {
            'get_chain': {
                'c': [...],
                'p': [...],
                'cp': [...]
            },
            'grouped_views': {
                'c': [
                    [...],
                    [...],
                    ...
                ],
                'p': [
                    [...],
                    [...],
                    ...
                ],
                'cp': [
                    [...],
                    [...],
                    ...
                ]
            }
        }

    """

    all_views = stack.describe(columns='view').index

    requested_views = {
        'get_chain': {
            'c': [],
            'p': [],
            'cp': []
        },
        'grouped_views': {
            'c': [],
            'p': [],
            'cp': []
        }
    }

    # Base views
    bases = ['x|frequency|x:y|||cbase']
    if weight is None:
        weight = ''
    else:
        bases.append('x|frequency|x:y||%s|cbase' % (weight))

    # Main views
    cs = ['x|frequency|||%s|counts' % (weight)]
    ps = ['x|frequency||y|%s|c%%' % (weight)]
    cps = cs[:] + ps [:]
    
    # Column tests for main views
    if coltests:
        for level in sig_levels:
            # Main test views
            props_test_views = [
                v for v in all_views 
                if 'tests.props.{}{}|||'.format(
                    mimic,
                    level
                ) in v
                and v.split('|')[4]==weight
            ]            
            cs.extend(props_test_views)
            ps.extend(props_test_views)
            cps.extend(props_test_views)

    # Net views
    if nets:
        net_cs = [
            [v] for v in all_views
            if v.split('|')[1]=='frequency'
            and v.split('|')[2].startswith('x[')
            and v.split('|')[3]==''
            and v.split('|')[4]==weight
        ]
        net_ps = [
            [v] for v in all_views
            if v.split('|')[1]=='frequency'
            and v.split('|')[2].startswith('x[')
            and v.split('|')[3]=='y'
            and v.split('|')[4]==weight
        ]
        net_cps = []
        for vc in net_cs:
            for vp in net_ps:
                if  vc[0] == vp[0].replace('|y|', '||'):
                    net_cps.append([vc[0], vp[0]])
                    break 
    
        # Column tests
        if coltests:
            net_test_views = []
            for level in sig_levels:
                if nets:
                    # Net test views
                    net_test_views.extend([
                        v for v in all_views 
                        if v.split('|')[1]=='tests.props.{}{}'.format(
                            mimic,
                            level
                        )
                        and v.split('|')[2].startswith('x[')
                        and v.split('|')[4]==weight
                    ])
            for i, vc in enumerate(net_cs):
                for vt in net_test_views:
                    eq_relation = vc[0].split('|')[2]  == vt.split('|')[2]
                    eq_weight = vc[0].split('|')[4] == vt.split('|')[4]
                    if eq_relation and eq_weight:
                        net_cs[i].append(vt)
                        net_ps[i].append(vt)
                        net_cps[i].append(vt)
                        
    # Descriptive statistics views
    if descriptives:
        views = {}
        for descriptive in descriptives:
            views[descriptive] = [
                [v] for v in all_views
                if v.split('|')[1]==descriptive
                and v.split('|')[4]==weight
            ]
                        
            # Column tests
            if descriptive=='mean' and coltests:
                means_test_views = []
                for level in sig_levels:
                    # Means test views
                    means_test_views.extend([
                        v for v in all_views 
                        if v.split('|')[1]=='tests.means.{}{}'.format(
                            mimic,
                            level
                        )
                        and v.split('|')[4]==weight
                    ])

        base_desc =  descriptives[0]
        if coltests:
            for i, vbd in enumerate(views[base_desc]):
                for vt in means_test_views:
                    eq_relation = vbd[0].split('|')[2]  == vt.split('|')[2]
                    eq_weight = vbd[0].split('|')[4] == vt.split('|')[4]
                    if eq_relation and eq_weight:
                        views[base_desc][i].append(vt)    
                        
        if len(descriptives) > 1:
            for i, vbd in enumerate(views[base_desc]):
                for rem_desc in descriptives[1:]:
                    for vrd in views[rem_desc]:
                        eq_relation = vbd[0].split('|')[2]  == vrd[0].split('|')[2]
                        eq_weight = vbd[0].split('|')[4] == vrd[0].split('|')[4]
                        if eq_relation and eq_weight:
                            views[base_desc][i].append(vrd[0])
    
          
        desc = views[base_desc]

    # Construct request object
    requested_views['get_chain']['c'] = bases + cs
    requested_views['grouped_views']['c'] = [bases, cs]

    requested_views['get_chain']['p'] = bases + ps
    requested_views['grouped_views']['p'] = [bases, ps]

    requested_views['get_chain']['cp'] = bases + cps
    requested_views['grouped_views']['cp'] = [bases, cps]
        
    if nets:
        requested_views['get_chain']['c'].extend([v for item in net_cs for v in item ])
        requested_views['get_chain']['p'].extend([v for item in net_ps for v in item ])
        requested_views['get_chain']['cp'].extend([v for item in net_cps for v in item])
        
        requested_views['grouped_views']['c'].extend(net_cs)
        requested_views['grouped_views']['p'].extend(net_ps)
        requested_views['grouped_views']['cp'].extend(net_cps)
        
    if descriptives: 
        requested_views['get_chain']['c'].extend([v for item in desc for v in item ])
        requested_views['get_chain']['p'].extend([v for item in desc for v in item ])
        requested_views['get_chain']['cp'].extend([v for item in desc for v in item ])
        
        requested_views['grouped_views']['c'].extend(desc)
        requested_views['grouped_views']['p'].extend(desc)
        requested_views['grouped_views']['cp'].extend(desc)
    
    # Remove bases and lists with one element
    for key in requested_views['grouped_views'].iterkeys():
        requested_views['grouped_views'][key].pop(0)
        for idx, item in enumerate(requested_views['grouped_views'][key]):
            if len(item) < 2:
                requested_views['grouped_views'][key].pop(idx)
        
    return requested_views

def reorder_set_keys(view_set):
    """
    Enforces an ordered convention of view keys in given keyed lists.

    Re-orders the view keys for all lists found in the given dict 
    view_set, achieved by calling reorder_test_keys() on each.
    
    Parameters
    ----------
    view_set : dict
        The dict of column names keyed to lists of view keys.

    Returns
    -------
    view_set : dict
        The now-ordered dict of column names keys to lists of view keys.
    """

    for key, value in view_set['items'].iteritems():
        old_order = view_set['items'][key]
        new_order = reorder_test_keys(value)
        view_set['items'][key] = new_order
        
    return view_set
    
def reorder_test_keys(views):
    """
    Enforces an ordered convention of view keys in given keyed lists.

    When view keys are stored as a dict of column names keyed to a list
    of view keys, significance test keys often end up in a group on
    their own at the end of the list. These significance test keys need 
    to be inserted after their paired targets. This function will take
    all the lists and make sure this convnetion is followed accordingly.
    
    .. note:: Important note (if any).

    Parameters
    ----------
    name : type, default=
        Description

    Returns
    -------
    name : type, default=
        Description
    """
    
    new_order = []
    for vk1 in views:
        pos1, agg1, relation1, rel_to1, weight1, name1 = vk1.split('|')
        if agg1=='frequency':
            new_order.append(vk1)
            for vk2 in views:
                pos2, agg2, relation2, rel_to2, weight2, name2 = vk2.split('|')
                if 'tests.props.Dim' in agg2:
                    if relation1==relation2:
                        new_order.append(vk2)
        elif agg1=='mean':
            new_order.append(vk1)
            for vk2 in views:
                pos2, agg2, relation2, rel_to2, weight2, name2 = vk2.split('|')
                if 'tests.mean.Dim' in agg2:
                    if relation1==relation2:
                        new_order.append(vk2)
        elif agg1 in ['stddev', 'sem', 'nps']:
            new_order.append(pos)
    
    return new_order
