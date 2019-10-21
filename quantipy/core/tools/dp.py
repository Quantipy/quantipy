

def frequency(meta, data, x=None, y=None, weight=None, rules=False, **kwargs):
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
        raise ValueError("You must provide a value for either x or y.")
    elif not x is None and not y is None:
        raise ValueError(
            "You may only provide a value for either x or y, and not"
            " both, when generating a frequency.")

    if rules and isinstance(rules, bool):
        rules = ['x', 'y']

    if x is None:
        x = '@'
        col = y
        if rules:
            rules_axis = 'y'
            transpose = True
            if not 'y' in rules:
                rules = False
    else:
        y = '@'
        col = x
        if rules:
            rules_axis = 'x'
            transpose = False
            if not 'x' in rules:
                rules = False
    if rules:
        try:
            if col in meta['columns']:
                rules = meta['columns'][col]['rules'][rules_axis]
            elif col in meta['masks']:
                rules = meta['masks'][col]['rules'][rules_axis]
        except KeyError:
            rules = False
        with_weight = weight
    else:
        with_weight = weight

    f = crosstab(
        meta, data, x, y,
        weight=with_weight,
        rules=False,
        xtotal=False,
        **kwargs)

    if rules:
        f = crosstab(
            meta, data, x, y,
            weight=with_weight,
            rules=True,
            xtotal=False,
            **kwargs)

    return f

def crosstab(meta, data, x, y, get='count', decimals=1, weight=None,
             show='values', rules=False, xtotal=False):
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
    xtotal : bool, default=False
        If True, the first column of the returned dataframe will be the
        regular frequency of the x column.

    Returns
    -------
    df : pandas.DataFrame
        The crosstab as a pandas DataFrame.
    """
    stack = Stack(name='ct', add_data={'ct': {'meta': meta, 'data': data}})
    stack.add_link(x=x, y=y)
    link = stack['ct']['no_filter'][x][y]
    q = Quantity(link, weight=weight).count()
    weight_notation = '' if weight is None else weight
    if get=='count':
        df = q.result
        vk = 'x|f|:||{}|counts'.format(weight_notation)
    elif get=='normalize':
        df = q.normalize().result
        vk = 'x|f|:|y|{}|c%'.format(weight_notation)
    else:
        raise ValueError(
           "The value for 'get' was not recognized. Should be 'count' or "
           "'normalize'."
        )
    df = np.round(df, decimals=decimals)
    if rules and isinstance(rules, bool):
        rules = ['x', 'y']

    if rules:
        view = View(link, vk)
        view.dataframe = df
        link[vk] = view
        rulesobj = Rules(link, vk, axes=rules)
        rulesobj.apply()
        if rulesobj.x_rules and 'x' in rules:
            idx = rulesobj.rules_df().index
            if not 'All' in idx.get_level_values(1).tolist():
                df_index =  [(link.x, 'All')] + idx.values.tolist()
            else:
                df_index = idx.values.tolist()
            df = df.loc[df_index]
        if rulesobj.y_rules and 'y' in rules:
            idx = rulesobj.rules_df().columns
            if not 'All' in idx.get_level_values(1).tolist():
                df_columns = [(link.y, 'All')] + idx.values.tolist()
            else:
                df_columns = idx.values.tolist()
            df = df[df_columns]

    if show != 'values':
        if show == 'text':
            text_key = meta['lib']['default text']
        else:
            text_key = show
        if not isinstance(text_key, dict):
            text_key = {'x': text_key, 'y': text_key}
        df = paint_dataframe(meta, df, text_key)

    if xtotal:
        try:
            f = frequency(
                meta, data, x,
                get=get, decimals=decimals, weight=weight, show=show)
            f = f.loc[df.index.values]
        except:
            pass
        df = pd.concat([f, df], axis=1)

    if q._get_type() == 'array':
        df = df.T
    return df
