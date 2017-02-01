import pandas as pd

from operator import lt, le, eq, ne, ge, gt
__op_symbol__ = {lt: '<', le: '<=', eq: '', ne: '!=', ge: '>=', gt: '>'}

from pandas.core.index import Index
__index_symbol__ = {
    Index.union: ',',
    Index.intersection: '&',
    Index.difference: '~',
}
if pd.__version__ == '0.19.2':
    __index_symbol__[Index.symmetric_difference] = '^'
else:
    __index_symbol__[Index.sym_diff] = '^'

def verify_logic_values(values, func_name):
    """ Verifies that the values given are a list of ints.

    Parameters
    ----------
    values : list-like
        The values to be tested

    func_name : string ('any', 'all' ONLY)
        The name of the logic being used

    Returns
    -------
    None
    """
    if isinstance(values, (list, tuple)):
        for value in values:
            if not isinstance(value, int):
                raise TypeError(
                    "The values given to %s() are not correctly "
                    "typed. Expected list of <int>, found a %s." % (
                        func_name,
                        type(value)
                    )
                )
    else:
        raise TypeError(
            "The values given to %s() must be given as a list. "
            "Expected a <list>, found a %s" % (
                func_name,
                type(values)
            )
        )


def verify_logic_series(series, func_name):
    """ Verifies that the series given is a compatible type (object,
    int64 or float64).

    Parameters
    ----------
    series : pandas.Series
        The data to be queried

    func_name : string ('any', 'all' ONLY)
        The name of the logic being used

    Returns
    -------
    None
    """
    if not series.dtype in ['object', 'int64', 'float64']:
        raise TypeError(
            "The series given to %s() must be a supported dtype. "
            "Expected 'object', 'int64' or 'float64', found a '%s'." % (
                func_name,
                series.dtype
            )
        )


def verify_count_responses(responses, func_name):
    """ Verifies that the responses given are well formed.

    Parameters
    ----------
    responses : int OR list-like
        If an int, the exact number of responses targeted.
        If list-like, the first two elements are the minimum and maximum
        (inclusive) range of responses targeted.
        If a third item is in the list it must be a list of values from
        which the range of target responses is being restricted.

    func_name : str
        Name of the function input being verified

    Returns
    -------
    None
    """

    if not isinstance(responses, list):
        responses = [responses]

    if not len(responses) in [1, 2, 3]:
        raise IndexError (
            "The responses list given to %s() must have "
            "either 1, 2 or 3 items in the form: "
            "[target_count] or "
            "[min, max] or "
            "[min, max, [values subset]] or "
            "[<operator_function>, numerator] or "
            "[<operator_function>, numerator, [values subset]]. "
            "Found %s." % (func_name, responses)
        )

    if isinstance(responses[0], tuple):
        if not responses[0][0] in [_is_lt, _is_le, _is_eq, _is_ne, _is_ge, _is_gt]:
            raise TypeError (
                "The binary function given to %s() is not recognized "
                "Found %s." % (func_name, responses)
            )
        if not isinstance(responses[0][1], int):
            raise TypeError (
                "The numerator given to %s() is "
                "incorrectly typed. It must be <int>. "
                "Found %s." % (func_name, responses)
            )

        if len(responses)==2:
            for value in responses[1]:
                if not isinstance(value, int):
                    raise TypeError (
                        "The values subset given to %s() are"
                        " not correctly typed. Each value must be "
                        "<int>. Found %s." % (func_name, responses[1])
                    )

        return responses

    if len(responses)==1:
        if not isinstance(responses[0], int):
            raise TypeError (
                "The count target given to %s() is "
                "incorrectly typed. It must be <int>. "
                "Found %s." % (func_name, responses)
            )
        return responses

    if len(responses) in [2, 3]:
        for value in responses[:2]:
            if not isinstance(value, int):
                raise TypeError (
                    "The values subset given to %s() are"
                    " not correctly typed. Each value must be "
                    "<int>. Found %s." % (func_name, responses[1])
                )

    if len(responses)==3:

        if len(responses)==3:
            for value in responses[2]:
                if not isinstance(value, int):
                    raise TypeError (
                        "The values subset given to %s() are"
                        " not correctly typed. Each value must be "
                        "<int>. Found %s." % (func_name, responses)
                    )

    return responses


def verify_numeric(value, func_name):
    """ Verifies that the value is numeric (int or float).

    Parameters
    ----------
    value : numeric (int or float)
        The value to be tested

    func_name : string ('lt', 'le', 'eq', 'ne', 'ge' or 'gt' ONLY)
        The name of the logic being used

    Returns
    -------
    None
    """
    try:
        test = float(value)
    except ValueError:
        raise ValueError(
            "The value given to is_%s() must be numeric. Found %s." % (
                func_name,
                type(value)
            )
        )

    return value


def _any_all(series, values, func_name, exclusive=False, _not=False):
    """ Returns the index of rows from series containing any/all of the
    given values as requested by func_name.

    Parameters
    ----------
    series : pandas.Series
        The data to be queried

    values : list-like
        The values to be tested

    func_name : string ('any' or 'all' ONLY)
        The name of the logic being used

    Returns
    -------
    index : pandas.index
        The index of series for rows containing any/all of the given values.

    """
    if series.dtype=='object':
        # Get the dichotomous version of series
        dummies = series.str.get_dummies(';')
        # Slice the dummies column-wise for only the targeted values
        values = [str(v) for v in values]
        cols = [col for col in dummies.columns if col in values]
        # If no valid columns are availabe, the result is no rows
        if not cols:
            if _not:
                if exclusive:
                    return series.dropna().index
                else:
                    return series.index
            else:
                return pd.Index([])
        else:
            if exclusive:
                other_cols = [
                    col
                    for col in
                    dummies.columns if not col in values
                ]
                other_dummies = dummies[other_cols]
                other_dummies = other_dummies[(other_dummies.T==1).any()]
            dummies = dummies[cols]
        # Slice the dummies row-wise for only rows with any/all of
        # the targeted responses
        if 'any' in func_name:
            # Apply 'any' logic
            dummies = dummies[(dummies.T!=0).any()]
            if exclusive:
                if _not:
                    exclusive_idx = other_dummies.index.difference(dummies.index)
                    return exclusive_idx
                else:
                    exclusive_idx = dummies.index.difference(other_dummies.index)
                    dummies = dummies.loc[exclusive_idx]
        if 'all' in func_name:
            # Apply 'all' logic
            dummies = dummies[(dummies.T==1).all()]
            if exclusive:
                if _not:
                    exclusive_idx = other_dummies.index.difference(dummies.index)
                    return exclusive_idx
                else:
                    exclusive_idx = dummies.index.difference(other_dummies.index)
                    dummies = dummies.loc[exclusive_idx]

        if _not:
            dummies = series.loc[series.index.difference(dummies.index)]
            if exclusive:
                dummies = dummies.dropna()

        # Return the index
        return dummies.index

    elif series.dtype in ['int64', 'float64']:
        # Slice the series row-wise for only rows with any/all of the
        # targets responses
        if func_name=='any' or (func_name=='all' and len(values)==1):
            result = series[series.isin(values)].dropna()
        else:
            # has_all() for multiple values is being requested on a
            # single-type variable, so the result will be none
            if _not:
                if exclusive:
                    return series.dropna().index
                else:
                    return series.index
            else:
                return pd.Index([])

        if _not:
            result = series.loc[series.index.difference(result.index)]
            if exclusive:
                result = result.dropna()

        # Return the index
        return result.index

    else:
        raise TypeError(
            "The dtype '%s' of series is incompatible with has_%s()" %
                series.dtype,
                func_name
        )


def has_any(values, exclusive=False):
    """ Convenience for managing 'any' part of the 'logic' instructions
    provided in a freq method's kwargs.

    Parameters
    ----------
    values : list-like
        List of values on which an 'any' condition will be used

    Returns
    -------
    _has_any : function
        The function that will be used to evaluate the 'any' condition

    values : list-like
        List of values on which an 'any' condition will be used
    """
    verify_logic_values(values, 'has_any')
    return _has_any, values, exclusive


def _has_any(series, values, exclusive=False):
    """ Returns the index of rows from series containing any of the
    given values.

    Parameters
    ----------
    series : pandas.Series
        The data to be queried

    values : list-like
        The values to be tested

    Returns
    -------
    index : pandas.index
        The index of series for rows containing any of the given values.

    """
    verify_logic_series(series, 'has_any')
    return _any_all(series, values, "any", exclusive)


def not_any(values, exclusive=False):
    """ Convenience for managing 'any' part of the 'logic' instructions
    provided in a freq method's kwargs.

    Parameters
    ----------
    values : list-like
        List of values on which an 'any' condition will be used

    Returns
    -------
    _not_any : function
        The function that will be used to evaluate the 'any' condition

    values : list-like
        List of values on which an 'any' condition will be used
    """
    verify_logic_values(values, 'not_any')
    return _not_any, values, exclusive


def _not_any(series, values, exclusive=False):
    """ Returns the index of rows from series containing any of the
    given values.

    Parameters
    ----------
    series : pandas.Series
        The data to be queried

    values : list-like
        The values to be tested

    Returns
    -------
    index : pandas.index
        The index of series for rows containing any of the given values.

    """
    verify_logic_series(series, 'not_any')
    return _any_all(series, values, "any", exclusive, True)


def has_all(values, exclusive=False):
    """ Convenience for managing 'all' part of the 'logic' instructions
    provided in a freq method's kwargs.

    Parameters
    ----------
    values : list-like
        List of values on which an 'all' condition will be used

    Returns
    -------
    _has_all : function
        The function that will be used to evaluate the 'all' condition

    values : list-like
        List of values on which an 'any' condition will be used
    """
    verify_logic_values(values, 'has_all')
    return _has_all, values, exclusive


def _has_all(series, values, exclusive=False):
    """ Returns the index of rows from series containing all of the given
    values.

    Parameters
    ----------
    series : pandas.Series
        The data to be queried

    values : list-like
        The values to be tested

    Returns
    -------
    index : pandas.index
        The index of series for rows containing all of the given values.
    """
    verify_logic_series(series, 'has_all')
    return _any_all(series, values, "all", exclusive)


def not_all(values, exclusive=False):
    """ Convenience for managing 'all' part of the 'logic' instructions
    provided in a freq method's kwargs.

    Parameters
    ----------
    values : list-like
        List of values on which an 'all' condition will be used

    Returns
    -------
    _not_all : function
        The function that will be used to evaluate the 'all' condition

    values : list-like
        List of values on which an 'any' condition will be used
    """
    verify_logic_values(values, 'not_all')
    return _not_all, values, exclusive


def _not_all(series, values, exclusive=False):
    """ Returns the index of rows from series containing all of the given
    values.

    Parameters
    ----------
    series : pandas.Series
        The data to be queried

    values : list-like
        The values to be tested

    Returns
    -------
    index : pandas.index
        The index of series for rows containing all of the given values.
    """
    verify_logic_series(series, 'not_all')
    return _any_all(series, values, "all", exclusive, True)


def has_count(responses, exclusive=False):
    """ Convenience for managing the 'count of responses' part of the
    'logic' instructions provided in a freq method's kwargs.

    Parameters
    ----------
    responses : int OR list-like
        If an int, the exact number of responses targeted.
        If list-like, the first two elements are the minimum and maximum
        (inclusive) range of responses targeted.
        If a third item is in the list it must be a list of values from
        which the range of target responses is being restricted.

    Returns
    -------
    _has_count : function
        The function that will be used to evaluate the 'count of
        responses' condition

    responses :
        List of values on which a 'count of responses' condition will
        be used
    """
    responses = verify_count_responses(responses, 'has_count')
    return _has_count, responses, exclusive


def _has_count(series, responses, exclusive=False):
    """ Returns the index of rows from series containing the targeted number
    or range of responses.

    Parameters
    ----------
    series : pandas.Series
        The data to be queried

    responses : int OR list-like
        If an int, the exact number of responses targeted.
        If list-like, the first two elements are the minimum and maximum
        (inclusive) range of responses targeted.
        If a third item is in the list it must be a list of values from
        which the range of target responses is being restricted.

    Returns
    -------
    index : pandas.index
        The index of series for rows containing the given number of
        responses
    """
    verify_logic_series(series, 'has_count')
    return _count(series, responses, exclusive)


def not_count(responses, exclusive=False):
    """ Convenience for managing the 'count of responses' part of the
    'logic' instructions provided in a freq method's kwargs.

    Parameters
    ----------
    responses : int OR list-like
        If an int, the exact number of responses targeted.
        If list-like, the first two elements are the minimum and maximum
        (inclusive) range of responses targeted.
        If a third item is in the list it must be a list of values from
        which the range of target responses is being restricted.

    Returns
    -------
    _has_count : function
        The function that will be used to evaluate the 'count of
        responses' condition

    responses :
        List of values on which a 'count of responses' condition will
        be used
    """
    responses = verify_count_responses(responses, 'not_count')
    return _not_count, responses, exclusive


def _not_count(series, responses, exclusive=False):
    """ Returns the index of rows from series containing the targeted number
    or range of responses.

    Parameters
    ----------
    series : pandas.Series
        The data to be queried

    responses : int OR list-like
        If an int, the exact number of responses targeted.
        If list-like, the first two elements are the minimum and maximum
        (inclusive) range of responses targeted.
        If a third item is in the list it must be a list of values from
        which the range of target responses is being restricted.

    Returns
    -------
    index : pandas.index
        The index of series for rows containing the given number of
        responses
    """
    verify_logic_series(series, 'not_count')
    return _count(series, responses, exclusive, True)


def _count(series, responses, exclusive=False, _not=False):
    """ Returns the index of rows from series containing the targeted number
    or range of responses.

    Parameters
    ----------
    series : pandas.Series
        The data to be queried

    responses : int OR list-like
        If an int, the exact number of responses targeted.
        If list-like, the first two elements are the minimum and maximum
        (inclusive) range of responses targeted.
        If a third item is in the list it must be a list of values from
        which the range of target responses is being restricted.

    Returns
    -------
    index : pandas.index
        The index of series for rows containing the given number of
        responses
    """
    if series.dtype in ['object', 'int64', 'float64']:

        # Get the dichotomous version of series
        dummies = series.astype('object').str.get_dummies(';')
        if dummies.columns.dtype=='object':
            dummies.columns = [int(float(col)) for col in dummies.columns]
        try:
            if isinstance(responses[0], tuple):
                values = responses[1]
            else:
                values = responses[2]
            # Slice the dummies column-wise for only the targeted values
            cols = [col for col in dummies.columns if col in values]
            # If not valid columns are availabe, the result is no rows
            if not cols:
                if _not:
                    return series.dropna().index
                else:
                    return pd.Index([])
            else:
                if exclusive:
                    other_cols = [
                        col
                        for col in
                        dummies.columns if not col in values
                    ]
                    other_dummies = dummies[other_cols]
                    other_dummies = other_dummies[(other_dummies.T==1).any()]
                dummies = dummies[cols]
        except:
            pass

        # Get a count of the number of responses
        count = dummies.sum(axis=1)

        # Get the min and max targeted responses
        if isinstance(responses[0], tuple):
            op_func = responses[0][0]
            numerator = responses[0][1]
            # Get a boolean slicing mask for use on dummies
            mask = (op_func(count, numerator))
        else:
            op_func = None
            _min = responses[0]
            try:
                _max = responses[1]
                # Get a boolean slicing mask for use on dummies
                mask = (count>=_min) & (count<=_max)
            except:
                _max = None
                # Get a boolean slicing mask for use on dummies
                mask = count==_min

        # Slice the dummies row-wise for only rows with the targeted
        # count of responses
        dummies = dummies.loc[mask]
        if exclusive:
            exclusive_idx = dummies.index.difference(other_dummies.index)
            dummies = dummies.loc[exclusive_idx]

        if _not:
            dummies = series.loc[series.index.difference(dummies.index)].dropna()

        # Return the index
        return dummies.index

    else:
        raise TypeError(
            "The series given to has_count() must be a supported dtype."
        )


def is_lt(value):
    """ Convenience for managing 'less than' part of the 'logic'
    instructions provided in a freq method's kwargs.

    Parameters
    ----------
    value : numeric (int or float)
        The value on which the 'less than' condition will be used

    Returns
    -------
    _lt : function
        The function that will be used to evaluate the 'less than'
        condition

    value : numeric (int or float)
        The value on which the 'less than' condition will be used
    """
    value = verify_numeric(value, 'lt')
    return _is_lt, value


def _is_lt(series, value):
    """ Returns the index of rows from series where series < value.

    Parameters
    ----------
    series : pandas.Series
        The data to be queried

    value : list-like
        The values to be tested

    Returns
    -------
    index : pandas.index
        The index of series for rows where series < value.
    """
    series = series[series.lt(value)]
    return series.index


def is_le(value):
    """ Convenience for managing 'less than or equal to' part of the
    'logic' instructions provided in a freq method's kwargs.

    Parameters
    ----------
    value : numeric (int or float)
        The value on which the 'less than or equal to' condition will be
        used.

    Returns
    -------
    _le : function
        The function that will be used to evaluate the 'less than or
        equal to' condition

    value : numeric (int or float)
        The value on which the 'less than or equal to' condition will be
        used
    """
    value = verify_numeric(value, 'le')
    return _is_le, value


def _is_le(series, value):
    """ Returns the index of rows from series where series <= value.

    Parameters
    ----------
    series : pandas.Series
        The data to be queried

    value : list-like
        The values to be tested

    Returns
    -------
    index : pandas.index
        The index of series for rows where series <= value.
    """
    series = series[series.le(value)]
    return series.index


def is_eq(value):
    """ Convenience for managing 'equal to' part of the 'logic'
    instructions provided in a freq method's kwargs.

    Parameters
    ----------
    value : numeric (int or float)
        The value on which the 'equal to' condition will be used

    Returns
    -------
    _eq : function
        The function that will be used to evaluate the 'equal to'
        condition

    value : numeric (int or float)
        The value on which the 'equal to' condition will be used
    """
    value = verify_numeric(value, 'eq')
    return _is_eq, value


def _is_eq(series, value):
    """ Returns the index of rows from series where series == value.

    Parameters
    ----------
    series : pandas.Series
        The data to be queried

    value : list-like
        The values to be tested

    Returns
    -------
    index : pandas.index
        The index of series for rows where series == value.
    """
    series = series[series.eq(value)]
    return series.index


def is_ne(value):
    """ Convenience for managing 'not equaL to' part of the 'logic'
    instructions provided in a freq method's kwargs.

    Parameters
    ----------
    value : numeric (int or float)
        The value on which the 'less than condition will be used

    Returns
    -------
    _ne : function
        The function that will be used to evaluate the 'not equal to'
        condition

    value : numeric (int or float)
        The value on which the 'not equal to' condition will be used
    """
    value = verify_numeric(value, 'ne')
    return _is_ne, value


def _is_ne(series, value):
    """ Returns the index of rows from series where series != value.

    Parameters
    ----------
    series : pandas.Series
        The data to be queried

    value : list-like
        The values to be tested

    Returns
    -------
    index : pandas.index
        The index of series for rows where series != value.
    """
    series = series[series.ne(value)]
    return series.index


def is_ge(value):
    """ Convenience for managing 'greater than or equal to' part of the
    'logic' instructions provided in a freq method's kwargs.

    Parameters
    ----------
    value : numeric (int or float)
        The value on which the 'greater than or equal to' condition will
        be used.

    Returns
    -------
    _ge : function
        The function that will be used to evaluate the 'greater than or
        equal to' condition

    value : numeric (int or float)
        The value on which the 'greater than or equal to' condition will be
        used
    """
    value = verify_numeric(value, 'ge')
    return _is_ge, value


def _is_ge(series, value):
    """ Returns the index of rows from series where series >= value.

    Parameters
    ----------
    series : pandas.Series
        The data to be queried

    value : list-like
        The values to be tested

    Returns
    -------
    index : pandas.index
        The index of series for rows where series >= value.
    """
    series = series[series.ge(value)]
    return series.index


def is_gt(value):
    """ Convenience for managing 'greater than' part of the 'logic'
    instructions provided in a freq method's kwargs.

    Parameters
    ----------
    value : numeric (int or float)
        The value on which the 'greater than' condition will be used

    Returns
    -------
    _lt : function
        The function that will be used to evaluate the 'greater than'
        condition

    value : numeric (int or float)
        The value on which the 'greater than' condition will be used
    """
    value = verify_numeric(value, 'gt')
    return _is_gt, value


def _is_gt(series, value):
    """ Returns the index of rows from series where series > value.

    Parameters
    ----------
    series : pandas.Series
        The data to be queried

    value : list-like
        The values to be tested

    Returns
    -------
    index : pandas.index
        The index of series for rows where series > value.
    """
    series = series[series.gt(value)]
    return series.index


def union(logic_list):
    """ Convenience for managing union logic provided in a freq
    method's kwargs.

    Parameters
    ----------
    logic_list : list
        The list of logical conditions the be chained with a union
        operation.

    Returns
    -------
    _union : function
        The function that will be used to evaluate the union

    logic_list : list
        The list of logical conditions to be chained with a union
        operation.
    """
    return _union, logic_list


def _union(idxs):
    """ Returns the chained union of the indexes given.

    Parameters
    ----------
    idxs : list
        List of pandas.Index objects.

    Returns
    -------
    idx : pandas.Index
        The result of the chained union of the indexes
        given.
    """
    idx = idxs[0]
    for idx_part in idxs[1:]:
        idx = idx.union(idx_part)
    return idx


def intersection(logic_list):
    """ Convenience for managing intersection logic provided in a freq
    method's kwargs.

    Parameters
    ----------
    logic_list : list
        The list of logical conditions the be chained with a
        intersection operation.

    Returns
    -------
    _intersection : function
        The function that will be used to evaluate the intersection

    logic_list : list
        The list of logical conditions to be chained with a
        intersection operation.
    """
    return _intersection, logic_list


def _intersection(idxs):
    """ Returns the chained intersection of the indexes given.

    Parameters
    ----------
    idxs : list
        List of pandas.Index objects.

    Returns
    -------
    idx : pandas.Index
        The result of the chained intersection of the indexes
        given.
    """
    idx = idxs[0]
    for idx_part in idxs[1:]:
        idx = idx.intersection(idx_part)
    return idx


def difference(logic_list):
    """ Convenience for managing difference logic provided in a freq
    method's kwargs.

    Parameters
    ----------
    logic_list : list
        The list of logical conditions the be chained with a difference
        operation.

    Returns
    -------
    _difference : function
        The function that will be used to evaluate the difference

    logic_list : list
        The list of logical conditions to be chained with a difference
        operation.
    """
    return _difference, logic_list


def _difference(idxs):
    """ Returns the chained difference of the indexes given.

    Parameters
    ----------
    idxs : list
        List of pandas.Index objects.

    Returns
    -------
    idx : pandas.Index
        The result of the chained difference of the indexes
        given.
    """
    idx = idxs[0]
    for idx_part in idxs[1:]:
        idx = idx.difference(idx_part)
    return idx


def symmetric_difference(logic_list):
    """ Convenience for managing symmetrical difference logic provided
    in a freq method's kwargs.

    Parameters
    ----------
    logic_list : list
        The list of logical conditions the be chained with a symmetrical
        difference operation.

    Returns
    -------
    _symmetrical difference : function
        The function that will be used to evaluate the symmetrical
        difference

    logic_list : list
        The list of logical conditions to be chained with a symmetrical
        difference operation.
    """
    return _symmetric_difference, logic_list


def _symmetric_difference(idxs):
    """ Returns the chained symmetrical difference of the indexes given.

    Parameters
    ----------
    idxs : list
        List of pandas.Index objects.

    Returns
    -------
    idx : pandas.Index
        The result of the chained symmetrical difference of the indexes
        given.
    """
    idx = idxs[0]
    for idx_part in idxs[1:]:
        if pd.__version__ == '0.19.2':
            idx = idx.symmetric_difference(idx_part)
        else:
            idx = idx.sym_diff(idx_part)
    return idx


def apply_set_theory(func, series, logic_list, data):
    """ Returns the result of chained binary operations defined in
    logic_list using the given function.

    Parameters
    ----------
    func : function [_union, _intersection, _difference, _symmetric_difference]
        The set theory operator to be used.

    series : pandas.Series
        The defaul data to be queried.

    logic_list : list
        The list of logical conditions to be chained together.

    data : pandas.DataFrame
        The source data from which wildcard variable references may
        be drawn.

    Returns
    -------
    idx : pandas.Index
        The index slice of series representing the given logic.

    vkey : str
        The relationship-part of the view key that represents this
        logical block.
    """
    idxs = []
    vkeys = []
    for logic in logic_list:
        idx, vkey = resolve_logic(series, logic, data)
        idxs.append(idx)
        vkeys.append(vkey)
    idx = func(idxs)
    __set_symbol__ = {
        _union: ',', _intersection: '&',
        _difference: '~', _symmetric_difference: '^'
    }
    vkey = '(%s)' % (
        __set_symbol__[func].join(vkeys)
    )
    return idx, vkey


def get_logic_key_chunk(func, values, exclusive=False):
    """ Derives the relationship view key chunk describing the
    combination of the given function and values.

    Parameters
    ----------
    func : function
        The logical function (lt, le, eq, ne, ge, gt,
        has_any, not_any, has_all, not_all or has_count).

    values : int or list
        The values associated with the logical function.

    Returns
    -------
    chunk : str
        The relationship-part of the view key that represents this
        combination of view function and values.
    """

    _not = '~' if func in [_not_any, _not_all, _not_count] else ''
    excl = 'e' if exclusive else ''

    if func in [_is_lt, _is_le, _is_eq, _is_ne, _is_ge, _is_gt]:
        __op_symbol__ = {
            _is_lt: '<', _is_le: '<=',
            _is_eq: '', _is_ne: '!=',
            _is_ge: '>=', _is_gt: '>'
        }
        op_func_name = func.__name__[1:]
        chunk = '(%s%s)' % (
            __op_symbol__[func],
            values
        )

    elif func in [_has_any, _not_any]:
        values = [str(v) for v in values]
        chunk = '%s%s{%s}' % (
            _not,
            excl,
            ','.join(values)
        )

    elif func in [_has_all, _not_all]:
        values = [str(v) for v in values]
        chunk = '%s%s{%s}' % (
            _not,
            excl,
            '&'.join(values)
        )

    elif func in [_has_count, _not_count]:
        # Get the min, max and targeted responses
        _min = values[0]

        if isinstance(_min, tuple):
            _max = None
            op_func = _min[0]
            numerator = _min[1]
            __op_symbol__ = {
                _is_lt: '<', _is_le: '<=',
                _is_eq: '', _is_ne: '!=',
                _is_ge: '>=', _is_gt: '>'
            }
            min_max = '%s%s' % (
                __op_symbol__[op_func],
                numerator
            )
            if len(values)==2:
                values = values[1]
                values = [str(v) for v in values]
            else:
                values = None
        else:
            op_func = None
            if len(values)>1:
                _max = values[1]
            else:
                _max = None
                min_max = _min
            if len(values)==3:
                values = values[2]
                values = [str(v) for v in values]
            else:
                values = None

        if not _max is None:
            if _min==_max:
                min_max = _min
                max = None
            else:
                if op_func is None:
                    min_max = '%s-%s' % (_min, _max)

        if values is None:
            chunk = '%s{%s}' % (_not, min_max)
        else:
            chunk = '%s(%s)%s{%s}' % (
                excl,
                ','.join(values),
                _not,
                min_max
            )

    return chunk


def resolve_func_logic(series, logic):
    """ Uses the given complex logic block to return a slice of series.

    Parameters
    ----------
    series : pandas.Series
        The series on which the logic should be applied.

    logic : list or tuple
        The complex logic block to be applied. Must be well-formed.

    Returns
    -------
    idx : pandas.Index
        The index slice of series representing the given logic.

    vkey : str
        The relationship-part of the view key that represents this
        logical block.
    """

    func, values, exclusive = (logic)
    idx = func(series, values, exclusive)
    vkey = get_logic_key_chunk(func, values, exclusive)

    return idx, vkey


def resolve_logic(series, logic, data):
    """ Uses the given complex logic block to return a slice of series.

    Parameters
    ----------
    series : pandas.Series
        The series on which the logic should be applied.

    logic : list or tuple
        The complex logic block to be applied. Must be well-formed.

    data : pandas.DataFrame
        The source data in full.

    Returns
    -------
    idx : pandas.Index
        The index slice of series representing the given logic.

    vkey : str
        The relationship-part of the view key that represents this
        logical block.
    """

    if isinstance(logic, dict):
        wildcard, logic = logic.keys()[0], logic.values()[0]
        if isinstance(logic, (str, unicode)):
            idx = data[data[wildcard]==logic].index
            vkey = logic
        else:
            if isinstance(logic, list):
                logic = has_any(logic)
            idx, vkey = resolve_logic(data[wildcard], logic, data)
        idx = series.dropna().index.intersection(idx)
        vkey = '%s=%s' % (wildcard, vkey)

    else:

        if isinstance(logic, int):
            logic = has_any([logic])

        if logic[0] in [
                _has_any, _not_any,
                _has_all, _not_all,
                _has_count, _not_count
            ]:
            idx, vkey = resolve_func_logic(series, logic)

        elif logic[0] in [_is_lt, _is_le, _is_eq, _is_ne, _is_ge, _is_gt]:
            func = logic[0]
            value = logic[1]
            idx = func(series, value)
            vkey = get_logic_key_chunk(func, value)

        elif logic[0] in [_union, _intersection, _difference, _symmetric_difference]:
            set_func = logic[0]
            idx, vkey = apply_set_theory(set_func, series, logic[1], data)

        elif isinstance(logic[0], (tuple, dict)):
            idx1, vkey1 = resolve_logic(series, logic[0], data)
            index_func = logic[1]
            idx2, vkey2 = resolve_logic(series, logic[2], data)

            idx = index_func(idx1, idx2)
            vkey = '(%s%s%s)' % (
                vkey1,
                __index_symbol__[index_func],
                vkey2
            )

    return idx, vkey


def get_logic_index(series, logic, data=None):
    """ Uses the given complex logic block to return a slice of series.

    Parameters
    ----------
    series : pandas.Series
        The series on which the logic should be applied.

    logic : list or tuple
        The complex logic block to be applied. Must be well-formed.

    data : pandas.DataFrame
        The source data in full.

    Returns
    -------
    idx : pandas.Index
        The index slice of series representing the given logic.

    vkey : str
        The relationship-part of the view key that represents this
        logical block.
    """

    if isinstance(logic, list):
        logic = (_has_any, logic, False)
        idx, vkey = resolve_logic(series, logic, data)

    if isinstance(logic, (tuple, dict)):
        idx, vkey = resolve_logic(series, logic, data)

    else:
        raise TypeError (
            "get_logic_index() recieved a non-tuple logical chunk. "
            "%s" % (logic)
        )

    vkey = 'x[%s]:y' % (vkey)

    return idx, vkey


def get_logic_key(logic, data=None):
    """ Uses the given complex logic block to return the matched view
    key.

    Parameters
    ----------
    logic : list or tuple
        The complex logic block to be applied. Must be well-formed.

    data : (optional) pandas.DataFrame
        The source data in full, required if any wildcards have been
        used in the logic definition.

    Returns
    -------
    vkey : str
        The relationship-part of the view key that represents this
        logical block.
    """

    idx, vkey = get_logic_index(pd.Series([]), logic, data)
    return vkey