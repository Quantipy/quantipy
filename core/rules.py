import re

def get_axis_slicer(link, all_rules_axes, rules_axis, rules_weight):
        if rules_axis == 'x' and 'x' not in all_rules_axes:
            return None
        elif rules_axis == 'y' and 'y' not in all_rules_axes:
            return None

        k, f, x, y = link.data_key, link.filter, link.x, link.y
        meta = link.stack[k].meta

        array_summary = _is_array_summary(meta, x, y)
        transposed_summary = _is_transposed_summary(meta, x, y)

        axis_slicer = None

        if rules_axis == 'x':
            if not array_summary and not transposed_summary:
                axis_slicer = _compute_slicer(link, x=x, weight=rules_weight)
            elif array_summary:
                axis_slicer = _compute_slicer(
                    dk, the_filter, x=x, y='@', weight=rules_weight,
                    slice_array_items=True)
            elif transposed_summary:
                axis_slicer = _compute_slicer(
                    dk, the_filter, x='@', y=y, weight=rules_weight)
        elif rules_axis == 'y':
            if not array_summary and not transposed_summary:
                axis_slicer = _compute_slicer(
                    dk, the_filter, y=y, weight=rules_weight)
            elif array_summary:
                axis_slicer = _compute_slicer(
                    dk, the_filter, x=x, y='@', weight=rules_weight,
                    slice_array_items=False)
            elif transposed_summary:
                axis_slicer = _compute_slicer(
                    dk, the_filter, x='@', y=y, weight=rules_weight)

        return axis_slicer


def _is_array_summary(meta, x, y):
    return x in meta['masks']

def _is_transposed_summary( meta, x, y):
    return x == '@' and y in meta['masks']

def _get_frequency_via_stack(link, col, weight=None):
    weight_notation = '' if weight is None else weight
    vk = 'x|f|:||{}|counts'.format(weight_notation)
    try:
        f = link.stack[link.data_key][link.filter][col]['@'][vk].dataframe
    except (KeyError, AttributeError) as e:
        try:
            f = link.stack[link.data_key][link.filter]['@'][col][vk].dataframe.T
        except (KeyError, AttributeError) as e:
            print 'THIS IS UNSUPPORTED RIGHT NOW!'
            print 'FREQ/CROSSBREAK FUNCTION MUST WORK HERE!!!!!!!!!!!!'
            f = frequency(self[data_key].meta, self[data_key].data, x=col, weight=weight)
    return f

def _compute_slicer(link, x=None, y=None, weight=None, slice_array_items=False):

        k, f, x, y = link.data_key, link.filter, link.x, link.y
        meta = link.stack[k].meta
        array_summary = _is_array_summary(meta, x, y)
        transposed_summary = _is_transposed_summary(meta, x, y)

        rules = None

        if not array_summary and not transposed_summary:
            if not x is None:
                try:
                    rules = meta['columns'][x]['rules']['x']
                    col = x
                except:
                    pass
            elif not y is None:
                try:
                    rules = meta['columns'][y]['rules']['y']
                    col = y
                except:
                    pass

        elif array_summary:
            if slice_array_items:
                try:
                    rules = meta['masks'][x]['rules']['x']
                    col = x
                except:
                    pass
            else:
                try:
                    rules = meta['masks'][x]['rules']['y']
                    col = x
                except:
                    pass

        elif transposed_summary:
                try:
                    rules = meta['masks'][y]['rules']['x']
                    col = y
                except:
                    pass

        if not rules: return None

        # views = self[data_key][the_filter][col]['@'].keys()
        views = link.stack[k][f][col]['@'].keys()

        w = '' if weight is None else weight
        expanded_net = [v for v in views if '}+]' in v
                        and v.split('|')[-2] == w
                        and v.split('|')[1] == 'f' and
                        not v.split('|')[3] == 'x']
        if expanded_net:
            if len(expanded_net) > 1:
                if len(expanded_net) == 2:
                    if expanded_net[0].split('|')[2] == expanded_net[1].split('|')[2]:
                        expanded_net = expanded_net[0]
                else:
                    msg = "Multiple 'expand' using views found for '{}'. Unable to sort!"
                    raise RuntimeError(msg.format(col))
            else:
                expanded_net = expanded_net[0]
        if 'sortx' in rules:
            on_mean = rules['sortx'].get('sort_on', '@') == 'mean'
        else:
            on_mean = False
        if 'sortx' in rules and on_mean:
            f = self.get_descriptive_via_stack(
                data_key, the_filter, col, weight=weight)
        elif 'sortx' in rules and expanded_net:
            within = rules['sortx'].get('within', False)
            between = rules['sortx'].get('between', False)
            fix = rules['sortx'].get('fixed', False)
            ascending = rules['sortx'].get('ascending', False)
            view = self[data_key][the_filter][col]['@'][expanded_net]
            f = self.sort_expanded_nets(view, between=between, within=within,
                                        ascending=ascending, fix=fix)
        else:
            f = _get_frequency_via_stack(link, col, weight=weight)


        if transposed_summary or (not slice_array_items and array_summary):
            rules_slicer = get_rules_slicer(f.T, rules)
        else:
            if not expanded_net or ('sortx' in rules and on_mean):
                rules_slicer = get_rules_slicer(f, rules)
            else:
                rules_slicer = f.index.values.tolist()
        try:
            rules_slicer.remove((col, 'All'))
        except:
            pass
        return rules_slicer



def get_rules_slicer(f, rules, copy=True):

    if copy:
        f = f.copy()

    if 'slicex' in rules:
        kwargs = rules['slicex']
        values = kwargs.get('values', None)
#         if not values is None:
#             kwargs['values'] = [val for val in values]
        f = qp.core.tools.view.query.slicex(f, **kwargs)

    if 'sortx' in rules:
        kwargs = rules['sortx']
        fixed = kwargs.get('fixed', None)
        sort_on = kwargs.get('sort_on', '@')
#         if not fixed is None:
#             kwargs['fixed'] = [fix for fix in fixed]
        f = qp.core.tools.view.query.sortx(f, **kwargs)

    if 'dropx' in rules:
        kwargs = rules['dropx']
        values = kwargs.get('values', None)
#         if not values is None:
#             kwargs['values'] = [v for v in values]
        f = dropx(f, **kwargs)
    return f.index.values.tolist()

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

    # If the index is from a frequency then the rule
    # should be skipped
    if df.index.levels[1][0]=='@':
        return df

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


def rule_viable_axes(link, vk):
    viable_axes = ['x', 'y']
    condensed_x = False
    condensed_y = False

    meta = link.stack[link.data_key].meta
    x, y = link.x, link.y

    array_summary = (x in meta['masks'] and y == '@')
    transposed_summary = (y in meta['masks'] and x == '@')
    v_method = vk.split('|')[1]
    relation = vk.split('|')[2]
    s_name = vk.split('|')[-1]
    descriptive = v_method.startswith('.d')
    exp_net = '}+]' in relation
    array_sum_freqs = array_summary and s_name in ['counts', 'c%', 'r%']


    if transposed_summary:
        x, y = y, x

    if (relation.split(":")[0].startswith('x') and not exp_net) or descriptive:
        if not array_summary:
            condensed_x = True
    elif relation.split(":")[1].startswith('y'):
        condensed_y = True
    else:
        if re.search('x\[.+:y$', relation) != None:
            condensed_x = True
        elif re.search('x:y\[.+', relation) != None:
            condensed_y = True
        if re.search('y\[.+:x$', relation) != None:
            condensed_y = True
        elif re.search('y:x\[.+', relation) != None:
            condensed_x = True

    if condensed_x or x=='@': viable_axes.remove('x')
    if condensed_y or (y=='@' and not array_sum_freqs): viable_axes.remove('y')

    return viable_axes