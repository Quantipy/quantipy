
def condense_dichotomous_set(df, values_from_labels=True, sniff_single=False,
                             yes=1, no=0, values_regex=None):
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

    # Anything not counted as yes or no should be treated as no
    df = df.applymap(lambda x: x if x in [yes, no] else no)
    # Convert to delimited set
    df_str = df.astype('str')
    for v, col in enumerate(df_str.columns, start=1):
        if values_from_labels:
            if values_regex is None:
                v = col.split('_')[-1]
            else:
                try:
                    v = str(int(re.match(values_regex, col).groups()[0]))
                except AttributeError:
                    raise AttributeError(
                        "Your values_regex may have failed to find a match"
                        " using re.match('{}', '{}')".format(
                            values_regex, col))
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

    # Add trailing delimiter
    series = series + ';'

    # Use NaNs to represent emtpy
    series.replace(
        {';': np.NaN},
        inplace=True
    )

    if df.dropna().size==0:
        # No responses are known, return filled with NaN
        return series

    if sniff_single and df.sum(axis=1).max()==1:
        # Convert to float
        series = series.str.replace(';','').astype('float')
        return series

    return series

def index_mapper(meta, data, mapper, default=None, intersect=None):
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

    # Apply any implied intersection
    if not intersect is None:
        keyed_mapper = {
            key: intersection([
                intersect,
                value if isinstance(value, dict) else {default: value}])
            for key, value in keyed_mapper.iteritems()
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
    if pd.__version__ == '0.19.2':
        df = pd.concat([ds1, ds2], axis=1, ignore_index=True)
    else:
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
        if index_mapper:
            cols = [str(c) for c in sorted(index_mapper.keys())]
        else:
            vals = meta['columns'][series.name]['values']
            codes = [c['value'] for c in vals]
            cols = [str(c) for c in codes]
        ds = pd.DataFrame(0, index=series.index, columns=cols)
        for key, idx in index_mapper.iteritems():
            ds[str(key)].loc[idx] = 1
        ds2 = condense_dichotomous_set(ds)
        org_name = series.name
        series = join_delimited_set_series(series, ds2, append)
        ## Remove potential duplicate values
        if series.dropna().empty:
            warn_msg = 'Could not recode {}, found empty data column dependency!'.format(org_name)
            warnings.warn(warn_msg)
            return series
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

    # Resolve the logic to a mapper of {key: index}
    index_map = index_mapper(meta, data, mapper, default, intersect)

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
    series = recode_from_index_mapper(meta, series, index_map, append)

    # Rename the recoded series
    series.name = target

    if not fillna is None:
        col_type = meta['columns'][series.name]['type']
        if col_type=='single':
            series.fillna(fillna, inplace=True)
        elif col_type=='delimited set':
            series.fillna('{};'.format(fillna), inplace=True)

    return series

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
