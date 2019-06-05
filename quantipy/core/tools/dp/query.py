import pandas as pd
import quantipy as qp
import re

def get_views(qp_structure):
    ''' Generator replacement for nested loops to return all view objects
        stored in a given qp container structure.
        Currently supports chain-classed shapes and cluster objects natively.
        To return views from a stack object instance provide input container as per
        qp_structure = < stack[data_key]['data'] >
    '''

    for k, v in qp_structure.items():
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

    for mask in list(meta['masks'].keys()):
        types[meta['masks'][mask]['type']].append(mask)

    if not_found:
        print('%s not found in meta file. Ignored.' %(not_found))

    return types

def uniquify_list(l):
    # De-dupe keys so far:
    # Credit: Dave Kirby's order preserving uniqueifying list function
    # http://www.peterbe.com/plog/uniqifiers-benchmark
    seen = set()
    seen_add = seen.add
    l = [x for x in l if x not in seen and not seen_add(x)]
    return l

def get_tests_slicer(s, reverse=False):
    """
    Returns the slicer needed to get tests in order from high to low.
    """
    tests_mapper = {}
    for idx_test in s.index:
        if s[idx_test].startswith('t.'):
            tests_mapper[float(s[idx_test][-3:])] = idx_test
    tests_slicer = [
        tests_mapper[level]
        for level in sorted(tests_mapper.keys())
    ]
    return tests_slicer

def shake(l):
    """
    De-dupe and reorder view keys in l for request_views.
    """

    s = pd.Series(uniquify_list(l))
    df = pd.DataFrame(s.str.split('|').tolist())
    df.insert(0, 'view', s)
    if pd.__version__ == '0.19.2':
        df.sort_values(by=[2, 1], inplace=True)
    else:
        df.sort_index(by=[2, 1], inplace=True)
    return df

def shake_nets(l):
    """
    De-dupe and reorder net view keys in l for request_views.
    """

    l = shake(l)['view'].values.tolist()
    return l

def shake_descriptives(l, descriptives):
    """
    De-dupe and reorder descriptives view keys in l for request_views.
    """

    df = shake(l)

    grouped = df.groupby(2)

    slicer = []
    for name, group in grouped:
        s = group[1]

        for i, desc in enumerate(descriptives):
            mean_found = False
            tests_done = False
            for idx in s.index:
                if s[idx]=='d.{}'.format(desc):
                    slicer.append(idx)
                    if desc=='mean':
                        mean_found = True
                if desc=='mean' and mean_found and not tests_done:
                    tests_slicer = get_tests_slicer(s)
                    slicer.extend(tests_slicer)
                    tests_done = True

    s = df.loc[slicer]['view']
    l = s.values.tolist()

    return l

def request_views(stack, data_key=None, filter_key=None, weight=None,
                  frequencies=True, default=False, nets=True,
                  descriptives=["mean"], sums=None, coltests=True,
                  mimic='Dim', sig_levels=[".05"], x=None, y=None, by_x=False):
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
    sums : {'mid', 'bottom'}, deafult None
        Get any frequency summing views and place them at the bottom or
        in the middle between nets and descriptives.
    coltests : bool, default=True
        If True, column tests (proportions and means) will be included.
    mimic : str
        The mimic type to be targeted when finding coltests.
    sig_levels : list-like, default=[".05"]
        The level/s of significance being requested, e.g. [".05", ".1"]
        or any of ["low", "mid", "high"] for [".10", ".05", ".01"]
        respectively.
    x : str, default=None
        The x-keys to which the results should be restricted.
    y : str, default=None
        The y-keys to which the results should be restricted.
    by_x : bool, default=False
        If True, the get_chain object in the returned dict will be
        structured as a dict of x-keys.

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

    described = stack.describe()

    if not data_key is None:
        if not isinstance(data_key, (list, tuple)):
            data_key = [data_key]
        described = described.loc[described['data'].isin(data_key)]

    if not filter_key is None:
        if not isinstance(filter_key, (list, tuple)):
            filter_key = [filter_key]
        described = described.loc[described['filter'].isin(filter_key)]

    if not x is None:
        if not isinstance(x, (list, tuple)):
            x = [x]
        described = described.loc[described['x'].isin(x)]

    if not y is None:
        if not isinstance(y, (list, tuple)):
            y = [y]
        described = described.loc[described['y'].isin(y)]

    all_views = sorted(described['view'].dropna().unique().tolist())

    if by_x:
        xks = described['x'].unique().tolist()
        requested_views = {
            'get_chain': {
                xk: {'c': [], 'p': [], 'cp': []}
                for xk in xks
            },
            'grouped_views': {'c': [], 'p': [], 'cp': []}
        }
        xks_views = {
            xk: [vk for vk in described.loc[described['x']==xk]['view']]
            for xk in xks
        }
    else:
        requested_views = {
            'get_chain': {'c': [], 'p': [], 'cp': []},
            'grouped_views': {'c': [], 'p': [], 'cp': []}
        }

    # Base views
    bases = ['x|f|x:|||cbase_gross', 'x|f|x:|||cbase']
    if weight is None:
        weight = ''
    else:
        bases.append('x|f|x:||%s|cbase_gross' % (weight))
        bases.append('x|f|x:||%s|cbase' % (weight))
        bases.append('x|f|x:||%s|ebase' % (weight))

    # Main views
    if frequencies:
        cs = ['x|f|:||%s|counts' % (weight)]
        ps = ['x|f|:|y|%s|c%%' % (weight)]
        cps = cs[:] + ps [:]
        csc = ['x|f.c:f|x++:||%s|counts_cumsum' % (weight)]
        psc = ['x|f.c:f|x++:|y|%s|c%%_cumsum' % (weight)]
        cpsc = csc[:] + psc[:]
    else:
        cs = []
        ps = []
        cps = []
        csc = []
        psc = []
        cpsc = []

    if default:
        dcs = ['x|default|:||%s|default' % (weight)]
        dps = ['x|default|:||%s|default' % (weight)]
        dcps = cs[:] + ps [:]

        cs.extend(dcs)
        ps.extend(dps)
        cps.extend(dcps)

    levels_ref = {
        "low": ".10",
        "mid": ".05",
        "high": ".01"
    }

    if not isinstance(sig_levels, (list, tuple)):
        sig_levels = [sig_levels]
    lvls = []
    for level in sig_levels:
        # Remove leading 0
        if not isinstance(level, str):
            level = str(level)
        if level[0]=='0': level = level[1:]
        if level in list(levels_ref.keys()):
            lvls.append(levels_ref[level])
        elif not re.match('\.[0-9]$', level) is None:
            lvls.append('{}0'.format(level))
        else:
            lvls.append(level)
    sig_levels = [str(i)[-3:] for i in sorted([float(s) for s in lvls])]
    sig_levels = [
        s if s.startswith('.') else '{}{}'.format(s[1:], 0)
        for s in sig_levels]

    # Column tests for main views
    if coltests:
        for level in sig_levels:
            # Main regular test views
            props_test_views = [
                v for v in all_views
                if 't.props.{}{}'.format(
                    mimic,
                    level
                ) in v
                and v.split('|')[2]==':'
                and v.split('|')[4]==weight
            ]
            cs.extend(props_test_views)
            ps.extend(props_test_views)
            cps.extend(props_test_views)

        for level in sig_levels:
            # Main cumulative test views
            props_test_views = [
                v for v in all_views
                if 't.props.{}{}'.format(
                    mimic,
                    level
                ) in v
                and v.split('|')[2]=='x++:'
                and v.split('|')[4]==weight
            ]
            csc.extend(props_test_views)
            psc.extend(props_test_views)
            cpsc.extend(props_test_views)

    # Net views
    if nets:
        net_cs = [
            [v] for v in all_views
            if v.split('|')[1].startswith('f')
            and v.split('|')[2].startswith('x[')
            and v.split('|')[3]==''
            and v.split('|')[4]==weight
        ]
        net_ps = [
            [v] for v in all_views
            if v.split('|')[1].startswith('f')
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
                        if v.split('|')[1]=='t.props.{}{}'.format(
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
    else:
        net_cs = False
        net_ps = False
        net_cps = False

    # Sum views
    if sums:
        sums_cs = [
            [v] for v in all_views
            if v.split('|')[3] == ''
            and v.split('|')[4] == weight
            and v.split('|')[-1].endswith('_sum')
        ]
        sums_ps = [
            [v] for v in all_views
            if v.split('|')[3] == 'y'
            and v.split('|')[4] == weight
            and v.split('|')[-1].endswith('_sum')
        ]

        if sums_cs:
            sums_cps = [[sums_cs[0][0], sums_ps[0][0]]]
            sums_cs_flat = sums_cs[0] if sums_cs else []
            sums_ps_flat = sums_ps[0] if sums_ps else []
            sums_cps_flat = sums_cs_flat[:] + sums_ps_flat[:]
        else:
            sums_cps = []
            sums_cs_flat = []
            sums_ps_flat = []
            sums_cps_flat = []

        sum_chains = [sums_cs_flat, sums_ps_flat, sums_cps_flat]
        sum_gvs = [sums_cs, sums_ps, sums_cps]


    # Descriptive statistics views
    if descriptives:
        views = {}
        for descriptive in descriptives:
            views[descriptive] = [
                [v] for v in all_views
                if v.split('|')[1] == 'd.{}'.format(descriptive)
                and v.split('|')[4] == weight
            ]

            # Column tests
            if descriptive=='mean' and coltests:
                means_test_views = []
                for level in sig_levels:

                    # Means test views
                    means_test_views.extend([
                        v for v in all_views
                        if v.split('|')[1]=='t.means.{}{}'.format(
                            mimic,
                            level
                        )
                        and v.split('|')[4]==weight
                    ])

        base_desc =  descriptives[0]
        if 'mean' in descriptives and coltests:
            for i, vbd in enumerate(views['mean']):
                for vt in means_test_views:
                    eq_relation = vbd[0].split('|')[2] == vt.split('|')[2]
                    eq_weight = vbd[0].split('|')[4] == vt.split('|')[4]
                    if eq_relation and eq_weight:
                        views['mean'][i].append(vt)

        if len(descriptives) > 1:
            for i, vbd in enumerate(views[base_desc]):
                for rem_desc in descriptives[1:]:
                    for vrd in views[rem_desc]:
                        eq_relation = vbd[0].split('|')[2]  == vrd[0].split('|')[2]
                        eq_weight = vbd[0].split('|')[4] == vrd[0].split('|')[4]
                        if eq_relation and eq_weight:
                            views[base_desc][i].extend(vrd)

        desc = views[base_desc]
    else:
        desc = False

    # Construct request object
    if by_x:
        xks = described['x'].unique().tolist()
        all_views = {
            xk: described.loc[described['x']==xk]['view'].unique().tolist()
            for xk in xks
        }

    if by_x:
        for xk in xks:
            requested_views['get_chain'][xk]['c'] = bases + cs + csc
            requested_views['get_chain'][xk]['p'] = bases + ps + psc
            requested_views['get_chain'][xk]['cp'] = bases + cps + cpsc
    else:
        requested_views['get_chain']['c'] = bases + cs + csc
        requested_views['get_chain']['p'] = bases + ps + psc
        requested_views['get_chain']['cp'] = bases + cps + cpsc

    requested_views['grouped_views']['c'] = [bases, cs, csc]
    requested_views['grouped_views']['p'] = [bases, ps, psc]
    requested_views['grouped_views']['cp'] = [bases, cps, cpsc]

    if nets and net_cs and net_ps and net_cps:

        net_cs_flat = shake_nets([v for item in net_cs for v in item])
        net_ps_flat = shake_nets([v for item in net_ps for v in item])
        net_cps_flat = shake_nets([v for item in net_cps for v in item])

        if by_x:
            for xk in xks:
                requested_views['get_chain'][xk]['c'].extend([
                    v for v in net_cs_flat
                    if v in xks_views[xk]])
                requested_views['get_chain'][xk]['p'].extend([
                    v for v in net_ps_flat
                    if v in xks_views[xk]])
                requested_views['get_chain'][xk]['cp'].extend([
                    v for v in net_cps_flat
                    if v in xks_views[xk]])
        else:
            requested_views['get_chain']['c'].extend(net_cs_flat)
            requested_views['get_chain']['p'].extend(net_ps_flat)
            requested_views['get_chain']['cp'].extend(net_cps_flat)


        requested_views['grouped_views']['c'].extend(net_cs)
        requested_views['grouped_views']['p'].extend(net_ps)
        requested_views['grouped_views']['cp'].extend(net_cps)

    if sums == 'mid':
        for ci, sum_chain in zip(['c', 'p', 'cp'], sum_chains):
            requested_views['get_chain'][ci].extend(sum_chain)
        for ci, sum_gv in zip(['c', 'p', 'cp'], sum_gvs):
            requested_views['grouped_views'][ci].extend(sum_gv)


    if descriptives and desc:

        desc_flat = shake_descriptives(
            [v for item in desc for v in item],
            descriptives)

        if by_x:
            for xk in xks:
                requested_views['get_chain'][xk]['c'].extend([
                    v for v in desc_flat
                    if v in xks_views[xk]])
                requested_views['get_chain'][xk]['p'].extend([
                    v for v in desc_flat
                    if v in xks_views[xk]])
                requested_views['get_chain'][xk]['cp'].extend([
                    v for v in desc_flat
                    if v in xks_views[xk]])
        else:
            requested_views['get_chain']['c'].extend(desc_flat)
            requested_views['get_chain']['p'].extend(desc_flat)
            requested_views['get_chain']['cp'].extend(desc_flat)

        requested_views['grouped_views']['c'].extend(desc)
        requested_views['grouped_views']['p'].extend(desc)
        requested_views['grouped_views']['cp'].extend(desc)

    if sums == 'bottom':
        for ci, sum_chain in zip(['c', 'p', 'cp'], sum_chains):
            requested_views['get_chain'][ci].extend(sum_chain)
        for ci, sum_gv in zip(['c', 'p', 'cp'], sum_gvs):
            requested_views['grouped_views'][ci].extend(sum_gv)

    # Remove bases and lists with one element
    for key in ['c', 'p', 'cp']:
        requested_views['grouped_views'][key].pop(0)
        requested_views['grouped_views'][key] = [
            item
            for item in requested_views['grouped_views'][key]
            if len(item) > 1
        ]
        for i, item in enumerate(requested_views['grouped_views'][key]):
            requested_views['grouped_views'][key][i] = [
                vk
                for vk in item
                if vk.split('|')[1] not in ['d.median', 'd.stddev',
                                            'd.sem', 'd.max', 'd.min']
            ]
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

    for key, value in view_set['items'].items():
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
        if agg1 == 'f':
            new_order.append(vk1)
            for vk2 in views:
                pos2, agg2, relation2, rel_to2, weight2, name2 = vk2.split('|')
                if 't.props.Dim' in agg2:
                    if relation1==relation2:
                        new_order.append(vk2)
        elif agg1 == 'd.mean':
            new_order.append(vk1)
            for vk2 in views:
                pos2, agg2, relation2, rel_to2, weight2, name2 = vk2.split('|')
                if 't.means.Dim' in agg2:
                    if relation1==relation2:
                        new_order.append(vk2)
        elif agg1 in ['d.stddev', 'd.sem', 'nps']:
            new_order.append(pos)

    return new_order
