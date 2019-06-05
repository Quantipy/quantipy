import pandas as pd
import numpy as np
import json
import re
import copy
import itertools
import math
import re, string

from collections import OrderedDict, defaultdict
from quantipy.core.helpers.constants import DTYPE_MAP
from quantipy.core.helpers.constants import MAPPED_PATTERN
from itertools import product, combinations
from scipy.stats.stats import _ttest_finish as get_pval
from operator import add, sub, mul, truediv

from quantipy.core.view import View
from quantipy.core.view_generators.view_mapper import ViewMapper
from quantipy.core.helpers import functions
from quantipy.core.tools.view import struct
import quantipy.core.tools.dp.prep

def describe(data, x, weights=None):
    ''' Replacment of (wrapper around) the df.describe() method that can deal with
    weighted data. Weight vectors are allowed to be non-normalized, i.e.
    sum of weights <> number of cases in sample. Quartile information currently
    dropped from output, variance is unbiased variance.
    Calculations are identical to SPSS Statistics/Professional.
    '''
    data = data.copy().dropna(subset=[x])
    desc_df = data[x].describe()
    desc_df.rename(
        {
            'std': 'stdDev',
            '25%': 'lower quartile',
            '50%': 'median',
            '75%': 'upper quartile'
        },
        inplace=True
    )
    # percentile information (incorrect for weighted data!) excluded for now...
    # desc_df.drop(['Lower quartile', 'Median', 'Upper quartile'], inplace=True)
    if not len(data.index) == 0:
        if not weights == '@1':
            count = data[weights].sum()
            norm_wvector_coef = 1 if len(data.index) == count else len(data.index)/count
            w_squared_sum = (data[weights]**2).sum()
            eff_count = count**2/w_squared_sum
            mean = data[x].mul(data[weights].mul(norm_wvector_coef)).mean()
            var = data[weights].mul((data[x].sub(mean))**2).sum()/(data[weights].sum()-1)
            try:
                stddev = math.sqrt(var)
                if abs(stddev) == 0.00:
                    stddev = np.NaN
            except:
                stddev = np.NaN
            desc_df['count'] = count
            desc_df['eff. count'] = eff_count
            desc_df['weights squared sum'] = w_squared_sum
            desc_df['mean'] = mean
            desc_df['stdDev'] = stddev
        else:
            desc_df['eff. count'] = desc_df['count']
            desc_df['weights squared sum'] = 1.00

        desc_df['efficiency'] = desc_df['eff. count']/desc_df['count']*100

    return pd.DataFrame(desc_df[['count', 'eff. count', 'min', 'max', 'mean', 'stdDev', 'weights squared sum', 'efficiency']])

def make_default_cat_view(link, weights=None):
    '''
    This function is creates Quantipy's default categorical aggregations:
    The x axis has to be a catgeorical single or multicode variable, the y axis
    can be generated from either categorical (single or multicode) or numeric
    (int/float). Numeric y axes are categorized into unique column codes.

    Acts as a wrapper around _df_to_value_matrix(), _aggregate_matrix() and
    set_qp_multiindex().

    Parameters
    ----------
    data : pd.DataFrame

    x, y : str
        Variable names from the processed case data input,
        i.e. the link definition.

    weighted : bool
        Controls if the aggregation is performed on weighted or weighted data.

    Returns
    -------
    view_df : pd.Dataframe (multiindexed)
    '''
    mat, xdef, ydef = get_matrix(link, weights)
    mat = weight_matrix(mat, xdef)
    df = _default_cat_df(mat, xdef, ydef)
    view_df = struct.set_qp_multiindex(df, link.x, link.y)

    return view_df


def make_default_str_view(data, x, y=None):

    df = pd.DataFrame({x: data[x]})

    return df



def make_default_num_view(data, x, y=None, weights=None, drop=None, rescale=None, get_only=None):
    '''
    This function is creates Quantipy's default numeric aggregations:
    The x axis has to be a numeric variable of type int or float, the y axis
    can be generated from either categorical (single or multicode) or numeric
    (int/float) as well. Numeric y axes are categorized into unique column codes.

    Acts as a wrapper around describe() and set_qp_multiindex().

    Parameters
    ----------
    data : pd.DataFrame

    x, y : str
        Variable names from the processed case data input,
        i.e. the link definition.

    weights : str
        Controls if the aggregation is performed on weighted or weighted data.

    Returns
    -------
    view_df : pd.Dataframe (multiindexed)
    '''
    if not drop is None:
        _exclude_codes(data[x], drop)
    if not rescale is None:
        _rescale_codes(data[x], rescale)
    weight = weights if not weights is None else '@1'
    if y is None or y == '@':
        df = describe(data, x, weight)
        df.columns = ['@']
    else:
        data = data[[x, y, weight]].copy().dropna()
        if len(data.index) == 0:
            df = describe(data, x, weight)
            df.columns = ['None']
        else:
            # changing column naming for x==y aggregations
            if not data.columns.is_unique:
                data.columns = [x, y+'_', weight]
            if data[y].dtype == 'object':
                # for Quantipy multicoded data on the y axis
                dummy_y = _cat_to_dummies(data[y], as_df=True)
                dummy_y_data = pd.concat([data[[x, weight]], dummy_y], axis=1)
                df = pd.concat(
                    [
                        describe(dummy_y_data[dummy_y_data[y_code] == 1], x, weight)
                        for y_code in dummy_y.columns
                    ],
                    axis=1
                )
                df.columns = dummy_y.columns
            else:
                y_codes =  sorted(data[y].unique())
                df = pd.concat(
                    [
                        describe(data[data[y] == y_code], x, weight)
                        for y_code in y_codes
                    ],
                    axis=1
                )
                df.columns = [
                    str(int(y_code)) if float(y_code).is_integer() else str(y_code)
                    for y_code in y_codes
                ]

    if get_only is None:
        df['All'] = describe(data, x, weight).values
        c_margin = df.xs('count')
        df = df.T
        df['All'] = c_margin
        df = df.T
        view_df = struct.set_qp_multiindex(df, x, y)

        return view_df
    else:
        return df.T[get_only].T

def calc_nets(casedata, link, source_view, combine_codes,
              use_logic=False, force_raw_sum=False):
    '''
    Used to compute (categorical) net code figures from a given Quantipy link
    definition, a reference view dataframe and a list of codes to build from.
    If the link's aggregation x axis is single coded categorical type, the
    calculation is a simple addition over the qualifying x codes. If x is type
    multicode, the result is calculated using the value matrix approach
    (as long force_raw_sum is not set to True).
    See also:
    - _cat_to_dummies(), _df_to_value_matrix(), _aggregate_matrix()
    - make_default_cat_view()

    Parameters
    ----------
    link : Quantipy Link object

    source_view : Quantipy View object
        I.e. a count or pct aggregation

    combine_codes : list of integers
        The list of codes to combine.

    force_raw_sum : bool, optional, default=False
        Controls if the calculation is performed on raw source_view figures.
        This effectively treats every categorical aggregation as single coded
        and is useful when needing to calculate the total responses given
        instead of effective qualifying answers.

    Returns
    -------
    net_values : np.array
        Stores the calculated net values
    '''
    if not use_logic and (not source_view.meta['x']['is_multi']
                          or force_raw_sum):
        boolmask = [
            int(index_val[1]) in combine_codes
            for index_val in source_view.index
            if not (
                isinstance(index_val[1], str)
                and index_val[1] == 'None'
            )
        ]
        if any(boolmask):
            net_values = np.array(source_view[boolmask].values.sum(axis=0))
        else:
            net_values = np.zeros(link['default'].dataframe.shape[1]-1)
    else:
        if not link.y == '@':
            matrix, xdef, ydef = get_matrix(
                link, weights=source_view.meta['agg']['weights'], data=casedata)
            matrix = weight_matrix(matrix, xdef)
            matrix = missingfy_matrix(matrix, xdef, combine_codes, keep=True)
            ycodes = reversed(range(1, len(ydef)+1))
            net_values = np.array([np.nansum(
                matrix[:, [0]]*matrix[:, [-ycode]])
                for ycode in ycodes])
        else:
            matrix, xdef, ydef = get_matrix(
                link, weights=source_view.meta['agg']['weights'], data=casedata)
            matrix = weight_matrix(matrix, xdef)
            matrix = missingfy_matrix(matrix, xdef, combine_codes, keep=True)
            net_values = np.nansum(matrix[:, [0]])
        if net_values.size == 0:
            net_values = np.zeros(link['default'].dataframe.shape[1]-1)

    return net_values


def _exclude_codes(matrix, dropped):
    '''
    Used to drop columns from a numeric matrix representation
    of a Link. This will prevent unwanted values from feeding into
    the statistical calculations.
    Parameters
    ----------
    matrix : pd.DataFrame of dummy-transformed data.
        As produced by _cat_to_dummies().

    dropped: int or list of int (or floats)
        The codes that should be dropped.
        If str is passed the function automatically converts to a list
        of a single element.

    Returns
    -------
    matrix : pd.DataFrame (modified)
    '''
    if not isinstance(dropped, list):
        dropped = [dropped]
    dropped = [code for code in dropped if code in matrix.columns]

    return matrix.drop(dropped, axis=1, inplace=True)

def _rescale_codes(matrix, scaling):
    '''
    Used to orient statistical figures produced by numerical aggregation
    on a new scale, e.g. to produce means and stddev that range between 0 and 100
    instead of the original survey codes that might have been 1,2,3,4,5.

    Parameters
    ----------
    matrix : pd.DataFrame of dummy-transformed data.
        As produced by _cat_to_dummies.

    scaling: dict
        A 1-on-1 mapping of old values to new values.

    Returns
    -------
    data : pd.DataFrame (modified)
    '''

    return matrix.rename(columns=scaling, inplace=True)

def calc_pct(source, base):
    return pd.DataFrame(np.divide(source.values, base.values)*100)

def get_default_num_stat(default_num_view, stat, drop_bases=True, as_df=True):
    '''
    Is used to extract a specific statistical figure from
    a given numerical default aggregation.

    Parameters
    ----------
    default_num_view : Quantipy default view
        (Numerical aggregation case)

    stat : string
        States the figure to extract.

    drop_bases : boolean, optional, default = True
        Controls if the base [= 'All'] column figure is excluded

    as_df : boolean, optional, default = True
        If True will only return as pd.DataFrame, otherwise as np.array.

    Returns
    -------
    pd.DataFrame
    OR
    np.array
    '''
    df = struct._partition_view_df(default_num_view, values=False, data_only=True)
    if drop_bases:
        df = df.drop('All', axis=1).drop('All', axis=0)
    df = df.T[[stat]].T

    if as_df:
        return df
    else:
        df.values

def _aggregate_matrix(value_matrix, x_def, y_def, calc_bases=True, as_df=True):
    '''
    Uses a np.array containing dichotomous values and lists of column codes
    to aggregate frequency tables (and bases if requested) to create basic categorical
    aggregations of uni- or bivariate cell frequencies.

    Parameters
    ----------
    value_matrix : np.array with 1/0 coded values
        I.e. as returned from qp.helpers.aggregation._df_to_value_matrix().

    x_def, y_def : lists of column codes

    calc_bases : bool, default=True
        Controls if the output contains base calculations
        for column, row and total base figures (cb, rb, tb).

    as_df : bool, default=True
        Controls if the output is returned as pd.DataFrame with regular axis indexing
        (and base rows/columns ['All'] if requested).

    Returns
    -------
    agg_df : pd.DataFrame of frequency and base figures
    OR
    tuple : freqs as np.array, list of base figures (column, row, total)
    '''
    # handling empty matrices
    if np.size(value_matrix) == 0:
        empty = True
        freq, cb, rb, tb = np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)
    else:
        empty = False
        xcodes = len(x_def)+1
        if not y_def is None:
            # bivariate calculation (cross-tabulation)
            ycodes = reversed(range(1, len(y_def)+1))
            freq = np.array([
                np.sum(
                    value_matrix[value_matrix[:, -ycode] == 1][:, 1:xcodes],
                    axis=0
                )
                for ycode in ycodes
            ])
            if calc_bases:
                ycodes = reversed(range(1, len(y_def)+1))
                cb = np.array([
                    np.sum(
                        value_matrix[value_matrix[:, -ycode] == 1][:, [0]])
                        for ycode in ycodes
                ])
                rb = np.sum(value_matrix[:, 1:xcodes], axis=0)
                tb = np.sum(value_matrix[:, [0]], axis=0)
        else:
            # univariate calculation (frequency table)
            freq = np.sum(value_matrix[:, 1:xcodes], axis=0)
            if calc_bases:
                cb = np.array(np.sum(value_matrix[:, :1]))
                rb = freq
                tb = cb
    # output creation: tuple of np.arrays vs. pd.DataFrame
    if as_df:
        if not empty:
            ixnames = x_def
            colnames = y_def if y_def else ['@']
        else:
            ixnames = ['None']
            colnames = ['None']
        freq_df = pd.DataFrame(data=freq.T, index = ixnames, columns=colnames)
        if calc_bases:
            cb_df = pd.DataFrame(data=[cb], index=['All'], columns=colnames)
            agg_df = pd.concat([freq_df, cb_df], axis=0)
            agg_df['All'] = np.append(rb, tb)
        else:
            agg_df = freq_df

        return agg_df
    else:
        if calc_bases:
            return freq, [cb, rb, tb]
        else:
            return freq

def aggregate_matrix(mat, xdef, ydef, get='full'):
    if np.size(mat) == 0:
        empty = True
    else:
        empty = False
    if get == 'freqs':
        return _cell_n(mat, xdef, ydef) if not empty else np.zeros(1)
    if get == 'cbase':
        return _col_n(mat, xdef, ydef) if not empty else np.zeros(1)
    if get == 'rbase':
        return _row_n(mat, xdef) if not empty else np.zeros(1)
    if get == 'tbase':
        return _total_n(mat) if not empty else np.zeros(1)
    if get == 'ebase':
        return _effective_n(mat, ydef) if not empty else np.zeros(1)
    if get == 'full':
        if not empty:
            return (_cell_n(mat, xdef, ydef), _col_n(mat, xdef, ydef),
                    _row_n(mat, xdef), _total_n(mat))
        else:
            return np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)


def _default_cat_df(mat, xdef, ydef):
    counts, cb, rb, tb = aggregate_matrix(mat, xdef, ydef, get='full')
    if tb.sum() == 0.0:
        row_codes, col_codes = ['None'], ['None']
    else:
        row_codes = xdef
        col_codes = ydef if ydef is not None else ['@']

    all = np.hstack((cb, tb))
    cat_df = pd.DataFrame(data=counts, index=row_codes, columns=col_codes)

    cat_df['All'] = rb
    cat_df = cat_df.T
    cat_df['All'] = all

    return cat_df.T


def _df_to_value_matrix(data, x, y=None, limit_x=None, limit_y=None, weights=None):
    '''
    Transforms a pd.DataFrame into a np.array representation consiting of:
    1. a value column storing either only 1s or weight factors
    2. a dichotomous version of a Quantipy link's x axis' data as an 1/0 matrix
    3. if present: a dichotomous version of a Quantipy link's y axis' data as an 1/0 matrix

    Caution: The np.array contains no naming information, so column data must be inferred from
    a list of codes that matches the structure of the value matrix when working with this function's output.

    Parameters
    ----------
    data : pd.DataFrame
        E.g. Quantipy casedata

    x, y : str, (y: default=False)
        E.g. Quantipy link x and y axes (or other data columns inside a pd.DataFrame).

    limit_x, limit_y : list, default=None
        List of codes that should be ignored in the transformation.

    weighted :  bool, default=False
        Controls if the first column of the matrix contains only 1s or weight factors.

    Returns
    -------
    tuple : value_matrix as np.array, list of x codes, list of y codes
    '''
    values = weights if weights else '@1'
    if not y is None:
        # two variable case, x and y specified
        data.replace('', np.NaN, inplace=True)
        data.dropna(subset=[x, y], inplace=True)
        #data = data.copy().replace('', np.NaN).dropna(subset=[x, y])
        wg_vec = data[values].values.reshape(len(data.index), 1)
        x_matrix, x_codes = _cat_to_dummies(data = data[x], limit_to=limit_x)
        y_matrix, y_codes = _cat_to_dummies(data = data[y], limit_to=limit_y)
        if weights:
            value_matrix = np.concatenate((wg_vec, x_matrix*wg_vec, y_matrix), axis=1)
        else:
            value_matrix = np.concatenate((wg_vec, x_matrix, y_matrix), axis=1)
    else:
        # single variable case, only x specified
        #data = data.copy().replace('', np.NaN).dropna(subset=[x])
        data.replace('', np.NaN, inplace=True)
        data.dropna(subset=[x], inplace=True)
        wg_vec = data[values].values.reshape(len(data.index), 1)
        x_matrix, x_codes = _cat_to_dummies(data[x], limit_to=limit_x)
        y_codes = None
        if weights:
            value_matrix = np.concatenate((wg_vec, x_matrix*wg_vec), axis=1)
        else:
            value_matrix = np.concatenate((wg_vec, x_matrix), axis=1)

    return value_matrix, x_codes, y_codes


def _cat_to_dummies(data, limit_to=None, style='freq', as_df=False):
    '''
    Creates a dichotomously 1/0-coded version of the incoming pd.Series with the answer codes
    found in the data being transformend into column names. This representation of the data situationally
    can be easier to work with if multicoded variables are being fed into computations (i.e. Numpy).
    The function unites the two .get_dummies() functions found in pandas, see also:
    - pandas.core.reshape.get_dummies
    - pandas.core.strings.StringMethods.get_dummies

    Parameters
    ----------
    data : pd.Series
        Contanis the single- or multicoded variable that
        should be transformed into the dummy representation.

    limit_to : list, default=None
        List of codes that should be used in the transformation.
        If None (=default), the function will transform all data.

    as_df : bool, default=False
        Controls if the returned output is a pd.DataFrame (=True) or
        a tuple of the DataFrame's values and the column codes (=False).

    Returns
    -------
    tuple : np.array, list of column codes
    OR
    dummy_df : pd.DataFrame
    '''
    if data.dtype == 'object':
        # i.e. Quantipy multicode data
        dummy_df = data.str.get_dummies(';')
        dummy_df.columns = [int(col) for col in dummy_df.columns]
        if style == 'freq':
            dummy_df.sort_index(axis=1, inplace=True)
            dummy_df.rename(
                columns={col: str(col) for col in dummy_df.columns},
                inplace=True
                )
        else:
            dummy_df.sort_index(axis=1, inplace=True)
    else:
        data.dropna(inplace=True)
        dummy_df = pd.get_dummies(data)
        if style == 'freq':
            dummy_df.rename(
                columns={
                    col: str(int(col))
                    if float(col).is_integer()
                    else str(col)
                    for col in dummy_df.columns
                },
                inplace=True
            )

    if limit_to:
        dummy_df = _limit_dummy_df(dummy_df, limit_to)
    if as_df:
        return dummy_df
    else:
        return dummy_df.values, dummy_df.columns.tolist()

def _limit_dummy_df(dummy_df, codes):
    '''
    Limits a 1/0 dummy pd.DataFrame to the given list of column codes.
    Also checks if none existing codes have been passed and removes
    those from the lists if necessary.

    Parameters
    ----------
    dummy_df : pd.DataFrame (dummy-transformed)

    codes : list of integers
        Column codes to keep

    Returns
    -------
    dummy_df : pd.DataFrame (dummy-transformed, with limited columns)
    '''
    if len(dummy_df.index) > 0:
        if not sorted(codes) == sorted(dummy_df.columns):
            codes = [code for code in codes if code in dummy_df.columns]
        return dummy_df[codes]
    else:
        return dummy_df

def _df_to_num_matrix(data, x, y=None, exclude=None, rescale=None, weights=None):
    values = weights if weights else '@1'
    data.replace('', np.NaN, inplace=True)
    if y == '@':
        y = None
    x_matrix = _cat_to_dummies(data = data[x], style='num', as_df=True)
    if exclude:
        _exclude_codes(x_matrix, exclude)
    if rescale:
        _rescale_codes(x_matrix, rescale)
    x_codes = x_matrix.columns.tolist()
    if not y is None:
        data.dropna(subset=[x,y], inplace=True)
        y_matrix = _cat_to_dummies(data = data[y], style='num', as_df=True)
        y_codes = y_matrix.columns.tolist()
        num_matrix = pd.concat(
            [
                data[[values]].T,
                x_matrix.T.mul(x_codes, axis=0).mul(data[values], axis=1),
                x_matrix.T.mul(data[values], axis=1),
                y_matrix.T
            ],
            axis=0
        ).T.values

    else:
        y_codes = None
        data.dropna(subset=[x], inplace=True)
        num_matrix = pd.concat(
            [
                data[[values]].T,
                x_matrix.T.mul(x_codes, axis=0).mul(data[values], axis=1),
                x_matrix.T.mul(data[values], axis=1)
            ],
            axis=0
        ).T.values

    return num_matrix, x_codes, y_codes

def _mean_from_mat(num_matrix, x_def, y_def):
    if y_def is not None:
        ycodes = reversed(range(1, len(y_def)+1))
        means = np.array([
            np.true_divide(
                np.nansum(
                    num_matrix[num_matrix[:, -ycode] == 1][:, 1:len(x_def)+1]),
                np.nansum(
                    num_matrix[num_matrix[:, -ycode] == 1][:, 1 +
                                                         len(x_def):1 +
                                                         len(x_def)*2])
            ) for ycode in ycodes
        ])
    else:
        means = np.true_divide(
            np.nansum(num_matrix[:, 1:len(x_def)+1]),
            np.nansum(num_matrix[:, 1+len(x_def):1+len(x_def)*2])
        )

    return means

def _dispersion_from_mat(num_matrix, x_def, y_def, measure='stddev', return_mean=False):
    '''
    Will calculate measures of dispersion from a given Link's numeric matrix
    representation. Will either return the standard deviation
    or the sample variance. Both measures are unbiased (N-1).
    Supports exclusion and rescaling of codes when used from inside
    the num_stats() view method.

    Parameters
    ----------
    num_matrix : np.array
        I.e. as returned from _df_to_num_matrix().
    x_def, y_def :  list
        x and y codes information as generated by _df_to_num_matrix()
        for indexing the numeric matrix.
    measure : str, default="stddev"
        Controls the dispersion measure returned...
            - stddev: Standard deviation
            - var: Unbiased sample variance
            - varcoeff: Coefficient of variation
    return_mean : bool, optional, default=False
        If True will also return the mean.

    Returns
    -------
    stddev OR var OR varcoeff
    OR
    mean, stddev : np.array or tuple of np.arrays
        A Numpy array of the variance measure
        or a tuple of means and related standard deviations.
    '''
    means = _mean_from_mat(num_matrix, x_def, y_def)
    num_matrix_no_w = num_matrix
    num_matrix_no_w[:, 1: 1+len(x_def)*2] = (
        num_matrix[:, 1:1+len(x_def)*2] /
        num_matrix[:, [0]])
    wv_mask = np.nansum(num_matrix_no_w[:, 1:len(x_def)+1], axis=1) > 0

    num_matrix[num_matrix == 0.] = np.NaN
    num_matrix_no_w[num_matrix_no_w == 0.] = np.NaN

    if y_def is not None:
        ycodes = reversed(range(1, len(y_def)+1))

        var = [np.true_divide(
            np.nansum(
                (num_matrix[num_matrix[:, -ycode] == 1][:, [0]] *
                 (num_matrix_no_w[num_matrix_no_w[:, -ycode] == 1][:, 1:len(x_def)+1] -
                 means[-ycode + (len(y_def))]) ** 2)),
            np.nansum(num_matrix[(num_matrix_no_w[:, -ycode] == 1) &
                                 (wv_mask)][:, [0]]) - 1)
               for ycode in ycodes]
    else:
        var = np.true_divide(
            np.nansum(
                (num_matrix[wv_mask][:, [0]] *
                 (num_matrix_no_w[wv_mask][:, 1:len(x_def)+1] - means) ** 2)),
            np.nansum(
                num_matrix[wv_mask][:, [0]]) - 1)

    var = np.array(var)
    var[var < 0] = 0

    if return_mean and measure == 'stddev':
        return means, np.sqrt(var)
    elif not return_mean and measure == 'stddev':
        return np.sqrt(var)
    elif measure == 'varcoeff':
        return np.sqrt(var) / means
    else:
        return var


def _effbase_from_mat(num_matrix, x_def, y_def):
    '''
    Used to calculate the effective base sizes for a given Link's
    x and y axis definition. The effective base calculation mirrors
    SPSS Dimensions.

    Parameters
    ----------
    num_matrix : np.array
        I.e. as returned from _df_to_num_matrix().
    x_def, y_def :  list
        x and y codes information as generated by _df_to_num_matrix()
        for indexing the numeric matrix.

    Returns
    -------
    effbases : np.array
        A Numpy array of effective column base sizes.
    '''
    wv_mask = np.nansum(num_matrix[:, 1:len(x_def)+1], axis=1) > 0
    if y_def is not None:
        ycodes = reversed(range(1, len(y_def)+1))
        effbases = np.array(
            [
                np.true_divide(
                    np.sum(
                        num_matrix[(num_matrix[:, -ycode] == 1)&(wv_mask)][:, [0]]) ** 2,
                    np.sum(
                        num_matrix[(num_matrix[:, -ycode] == 1)&(wv_mask)][:, [0]] ** 2)
                )
                for ycode in ycodes
            ]
        )
    else:
        effbases = np.true_divide(
            np.sum(num_matrix[wv_mask][:, [0]])**2,
            np.sum(num_matrix[wv_mask][:, [0]]**2)
        )

    return effbases

def _sum_w_squared_from_mat(num_matrix, x_def, y_def, base_ratio=True):
    '''
    Computes the sum of squared weights (sws) for the Link's matrix
    representation. Will by default return the ratio between the
    sws to the base sizes.

    Parameters
    ----------
    num_matrix : np.array
        I.e. as returned from _df_to_num_matrix().
    x_def, y_def : list
        x and y codes information as generated by _df_to_num_matrix()
        for indexing the numeric matrix.
    base_ratio : bool, default=True
        If False will only return the sws.

    Returns
    -------
    base_ratio
    OR
    sws : np.array
        A Numpy array of sws/bases or raw sws.
    '''
    wv_mask = np.nansum(num_matrix[:, 1:len(x_def)+1], axis=1) > 0
    if y_def is not None:
        ycodes = reversed(range(1, len(y_def)+1))
        sws = np.array(
            [ np.sum(
                    num_matrix[(num_matrix[:, -ycode] == 1)&(wv_mask)][:, [0]]**2)
                for ycode in ycodes
            ]
        )
    else:
        sws = np.sum(num_matrix[wv_mask][:, [0]]**2)

    if base_ratio:
        bases =  _base_from_mat(num_matrix, x_def, y_def, effbase=False)
        return sws / bases
    else:
        return sws


def _base_from_mat(num_matrix, x_def, y_def, effbase=True):
    '''
    Used to calculate the column bases for a given Link's
    x and y axis definition. The effective base calculation mirrors
    SPSS Dimensions.

    Parameters
    ----------
    num_matrix : np.array
        I.e. as returned from _df_to_num_matrix().
    x_def, y_def :  list
        x and y codes information as generated by _df_to_num_matrix()
        for indexing the numeric matrix.
    effbase : bool, optional, default=True
        Will also return the effective base sizes if True.

    Returns
    -------
    bases, effbases
    OR
    bases : tuple of np.array or np.array
        Numpy array(s) of unweighted/weighted base sizes and
        effective base meassures.
    '''
    wv_mask = np.nansum(num_matrix[:, 1:len(x_def)+1], axis=1) > 0
    if y_def is not None:
        ycodes = reversed(range(1, len(y_def)+1))
        bases = np.array(
            [ np.sum(
                    num_matrix[(num_matrix[:, -ycode] == 1)&(wv_mask)][:, [0]])
                for ycode in ycodes
            ]
        )
    else:
        bases = np.sum(num_matrix[wv_mask][:, [0]])

    if effbase:
        effbases =  _effbase_from_mat(num_matrix, x_def, y_def)
        return bases, effbases
    else:
        return bases


def calc_stat_from_mat(mat, xdef, ydef, stat):
    '''
    This function links to various helper functions to calculate distribution
    statistics from a matrix representation of Quantipy data.
    It accepts a pre-defined matrix defintion given as a tuple of
    matrix/x_def/y_def (np.array, list, list) or a pd.DataFrame. If a DataFrame
    is passed the functions will convert it to the value matrix according to
    the rest of parameter settings (x, y, exclude, rescale, weights).

    Parameters
    ----------
    data : (np.array, list, list) or pd.DataFrame
        The input data representation as a matrix
        defintion as from _df_to_num_matrix() or
        a pd.DataFrame.
    stat : str
        The statistic to be computed.
        - mean: Mean
        - stddev: Standard deviation
        - var: Sample variance
        - varcoeff: Coefficient of variation
        - mean_stddev: Mean + standard dev.
        - effbase: Effective base
        - base: Column base (+ effbase by default)
        - sws: Sum of weights squared (+ ratio to base by default)
    x, y : str, optional, required when data is df
        Link x and y defintion.
    exclude : list, optional
        List of codes to exclude.
    rescale : dict
        Mapping of {old: new} codes.
    weights : str, optional
        Weight variable to use for weighted aggregation.
    Returns
    -------
    stats : np.array
        Numpy array storing the calculation results.
    '''
    # Check for input: matrix vs. df
    # if isinstance(data, tuple):
    #     matrix, x_def, y_def = data[0], data[1], data[2]
    # elif isinstance(data, pd.DataFrame):
    #     matrix, x_def, y_def = _df_to_num_matrix(
    #         data, x, y, exclude, rescale, weights)
    # Pass matrix to helper and calculate stat
    if stat == 'mean':
        stats = _mean(mat, xdef, ydef)
    elif stat == 'stddev':
        stats = _dispersion(mat, xdef, ydef)
    elif stat == 'var':
        stats = _dispersion(mat, xdef, ydef,
                                     measure='var')
    elif stat == 'mean_stddev':
        stats = _dispersion(mat, xdef, ydef,
                                     return_mean=True)
    elif stat == 'varcoeff':
        stats = _dispersion(mat, xdef, ydef,
                                     measure='vcoef')
    # elif stat == 'effbase':
    #     stats = _effbase_from_mat(matrix, x_def, y_def)
    # elif stat == 'base':
    #     stats = _base_from_mat(matrix, x_def, y_def,
    #                            effbase=True)
    elif stat == 'ssw':
        stats = _sum_sq_w(mat, xdef, ydef,
                    base_ratio=True)

    return stats



def _get_mv_matrices(data, x, y, weights):
    '''
    Generate the input matrix inputs for the analytics view method.
    In contrast to the "regular" view method's matrices, the
    output of this function is a two-element tuple of all the matrices
    and matrix defintions for the elements found in x and y
    (a tuple of tuple of tuples).

    Parameters
    ----------
    data : pd.DataFrame
        The raw data associated with the Link defintiton from the view
        method.
    x, y : list-like
        Lists of the input variables for the x and y axis, e.g. a variable
        of type array.
    weights : str, optional
        Weight variable to use for weighted aggregation.

    Returns
    -------
    xmats, ymats : tuple of tuples of tuples
        xmats will store all matrices/matrix defintions for the
        elements in x, ymats vice versa.
    '''
    xmats = ()
    ymats = ()
    weight = weights if weights is not None else '@1'
    var_list = [weight] + x + y
    data = data.copy()[var_list].dropna()
    if x == y:
        y = [y + '_' for y in y]
    data.columns = [weight] + x + y
    wv = data[[weight]].values
    for var in x:
        xmat = _make_dummies(data[var], style='num', as_df=False)
        xfactors = xmat[0] * xmat[1]
        xmats += ((np.hstack((wv, xfactors*wv, xmat[0])), xmat[1], None), )
    for var in y:
        ymat = _make_dummies(data[var], style='num', as_df=False)
        yfactors = ymat[0] * ymat[1]
        ymats += ((np.hstack((wv, yfactors*wv, ymat[0])), ymat[1], None), )

    return xmats, ymats


def _deviations_from_mean(matrix, x_def, y_def, known_mean=None):
    '''
    Returns the (weighted) mean-centered values held by the input matrix.

    Parameters
    ----------
    matrix : np.array
        I.e. as returned from _get_mv_matrices().
    x_def, y_def : list
        x and y codes information for indexing the matrix.
    known_mean : float, optional
        It is possible to inject a pre-defined mean.

    Returns
    -------
    mean-centered matrix : np.array
    '''
    if known_mean is None:
        mean = qp.v.agg._mean_from_mat(matrix, x_def, y_def)
    else:
        mean = known_mean

    matrix_no_w = matrix
    matrix_no_w[:, 1: 1+len(x_def)*2] = (matrix[:, 1:1+len(x_def)*2] /
                                         matrix[:, [0]])

    matrix_no_w[matrix_no_w == 0.] = np.NaN

    return np.nansum(matrix_no_w[:, 1:len(x_def)+1] - mean, axis=1)

def _xproduct_of_deviations(xmat, ymat, known_means=None):
    '''
    Returns the (weighted) cross-product of the mean-centered values
    found in the marices for x and y.

    Parameters
    ----------
    xmat, ymat : tuples of...
        ...Matrix, definition for x/y variable, None
    known_mean : float, optional
        It is possible to inject pre-defined means to compute the
        mean-centered values for the x and y matrix.
    Returns
    -------
    xprod : np.array
        The cross-product of the deviations from the means inside the x
        and the y variables' values.
    '''
    x_mat, xx_def, xy_def = xmat[0].copy(), xmat[1], xmat[2]
    y_mat, yx_def, yy_def = ymat[0].copy(), ymat[1], ymat[2]
    if known_means is not None:
        xmean, ymean = known_means
    else:
        xmean, ymean = (None, None)
    x_devi = _deviations_from_mean(x_mat, xx_def, xy_def, known_mean=xmean)
    y_devi = _deviations_from_mean(y_mat, yx_def, yy_def, known_mean=ymean)

    return np.nansum(np.transpose(x_mat[:, [0]])*x_devi*y_devi)


def _calc_mv_n(xdata, ydata):
    '''
    PLAEHOLDER: WILL COMPUTE PAIRWISE N FOR ANALYTICS.
    '''
    return np.nansum(xdata[:, [0]])

def _covariance(x_inputs, y_inputs, known_means=None):
    '''
    Returns the (weighted) covariances for all combinations
    of the x and y variables that have been fed into the
    analytics view method.

    Parameters
    ----------

    Returns
    -------

    '''
    xres = []
    res = []
    if isinstance(x_inputs[0], np.ndarray):
        x_inputs = (x_inputs,)
    if isinstance(y_inputs[0], np.ndarray):
        y_inputs = (y_inputs,)
    for x in x_inputs:
        xres = []
        xmean = _mean_from_mat(x[0].copy(), x[1], x[2])
        for y in y_inputs:
            ymean = _mean_from_mat(y[0].copy(), y[1], y[2])
            means = (xmean, ymean)
            xprod = _xproduct_of_deviations(x, y, known_means = means)
            xres.append(round(xprod / (np.sum(x[0][:, [0]])-1), 6))
        res.append(xres)
    return res

def _corr(x_inputs, y_inputs):
    '''
    Returns the (weighted) Pearson r for all combinations
    of the x and y variables that have been fed into the
    analytics view method.

    Parameters
    ----------

    Returns
    -------

    '''
    xres = []
    res = []
    for x in x_inputs:
        xres = []
        xmean, xstddev = _dispersion_from_mat(x[0].copy(), x[1], x[2], return_mean=True)
        for y in y_inputs:
            ymean, ystddev = _dispersion_from_mat(y[0].copy(), y[1], y[2], return_mean=True)
            means = (xmean, ymean)
            xres.append(round((_covariance(x, y, known_means=means)/(xstddev*ystddev))[0][0], 6))
        res.append(xres)

    return res




# def _mask_misvals(matrix, x_def, mis_def):
#     matrix = matrix.copy()
#     matrix[:,[4]] = 0.0
#     mask = np.nansum(matrix[:, 1:len(x_def)+1], axis=1) > 0.
#     matrix[mask][:, [0]]= 0.0

#     return matrix

# def _mask_wv(matrix, x_def):
#     matrix=matrix.copy()

#     valmask = np.nansum(matrix[:, 1:len(x_def)+1], axis=1) == 0.00
#     matrix[valmask][:, [0]] = np.NaN

#     return matrix


def calc_multivariate_from_mat(data, stat):
    xdata = data[0]
    ydata = data[1]

    if stat == 'cov':
        stats = _covariance(xdata, ydata)
    if stat == 'corr':
        stats = _corr(xdata, ydata)
    if stat == 'corr':
        stats = _corr(xdata, ydata)
    return stats


def _get_paired_columns_df(df):
    '''
    Creates a copy of the passed df with all unique y-code
    pairs next to each other following the order of the
    original DataFrame.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    paired_columns_df : pd.DataFrame
    '''
    return pd.concat([df[list(pair)]
                     for pair in combinations(df.columns, 2)], axis=1)



def _calc_column_operation(df, operator):
    '''
    Calculates the row value sums, differences, products or quotients
    between each pair of consecutive columns.
    If the number of columns is uneven, the last column is ignored.

    Parameters
    ----------
    df : pd.DataFrame
    operator : mathematical operator from the operators library
        Function supports the following mathematical operations...
            - add, addition
            - sub, subtraction
            - mul, multiplication
            - div, division

    Returns
    -------
    pair_operation_results : pd.DataFrame
    '''
    cols = df.columns.tolist()

    return pd.concat(
        [operator(
            df[pair[0]].T.drop_duplicates().T,
            df[pair[1]].T.drop_duplicates().T.values)
            for pair in np.array_split(cols, len(cols)/2)],
        axis=1)


def _convert_test_statistic(test_statistic,
                            overlaps, effbases, package='Dim'):
    if package =='Dim':
        effbases = _calc_paired_effbase_correctors(effbases)[0]
        dof = effbases - overlaps - 2
        t_stat = _get_pvals(test_statistic, dof)
    elif package == 'askia':
        t_stat = abs(test_statistic)
    if t_stat.shape[0] == 1:
        return t_stat[0]
    else:
        return t_stat


def _convert_test_level(level, package):
    if isinstance(level, str):
        if level == 'low':
            if package == 'Dim':
                comparelevel = siglevel = 0.10
            elif package == 'askia':
                comparelevel = 1.65
                siglevel = 0.10
        elif level == 'mid':
            if package == 'Dim':
                comparelevel = siglevel = 0.05
            elif package == 'askia':
                comparelevel = 1.96
                siglevel = 0.05
        elif level == 'high':
            if package == 'Dim':
                comparelevel = siglevel = 0.01
            elif package == 'askia':
                comparelevel = 2.576
                siglevel = 0.01
    else:
        if package == 'Dim':
            comparelevel = siglevel = level
        elif package == 'askia':
            comparelevel = 1.65
            siglevel = 0.10

    return comparelevel, siglevel


def _get_pvals(test_statistic, dof):
    '''
    This function computes a DataFrame of p-values for each unique column pair.
    It can be used to derive stat. sig. differences in proportions or means.
    Alternatively, the DataFrame of t-values can be returned.

    Parameters
    ----------
    test_statistic: np.array

    dof : np.array
        Numpy array of degrees of freedom per unique column pair.

    Returns
    -------
    pvals : np.array
    '''
    return get_pval(dof, test_statistic)[1]


def _z_score(counts, bases, return_diffs=True):
    '''
    Computes the z-score statistic for two-sample differences between
    all column code pairings of a given Link definition. Will return the
    proportion differences by default as well.

    Parameters
    ----------
    freqs, bases : pd.DataFrame
        DataFrame representations of raw cell frequencies and base sizes.
    return_diffs : bool, default = True
        Controls if the function also returns the proportion differnces.

    Returns
    -------
    z_scores_pairs, prop_diff_pairs : tuple of np.arrays
    OR
    z_scores_pairs : np.array
        z-score statistic and proportion differences
        for all column pairings.
    '''
    prop_diff_pairs = _props_diff_pairs(counts, bases)

    unp_sd_pairs = _props_unpooled_sd_pairs(counts, bases)

    if return_diffs:
        return prop_diff_pairs/unp_sd_pairs, prop_diff_pairs
    else:
        return prop_diff_pairs/unp_sd_pairs


def _calc_sd_unpooled_props_pairs(counts, bases):
    props = counts.values/bases.values[0]
    unp_sd = (props*(1-props))/bases.values

    return np.hstack([np.sqrt(unp_sd[:,[cat1]]+unp_sd[:,[cat2]])
                           for cat1, cat2
                           in combinations(range(0,unp_sd.shape[1]), 2)])

def _props_unpooled_sd_pairs(counts, bases):
        props = counts / bases
        var = (props * (1 - props) / bases).T

        return np.array([np.sqrt(var1 + var2)
                         for var1, var2 in combinations(var, 2)]).T

def _means_unpooled_sd_pairs(means, stddevs, bases):
    sd_base_ratio = stddevs / bases

    return np.array([np.sqrt(sd_b_r1 + sd_b_r2)
                     for sd_b_r1, sd_b_r2 in combinations(sd_base_ratio, 2)])




def _calc_se_pooled_props_pairs(counts, bases, effbases, overlap_bases):
    '''
    Returns the pooled sample standard error for all unique
    column pairings' proportions.

    Parameters
    ----------
    counts,
    bases, effbases, overlap_bases,
    ratios : np.array
        Inputs of standard deviations, bases,
        effective basses, overlap_bases.

    Returns
    -------
    pooled_se : np.array
        Numpy array storing each column pair's pooled
        standard error.
    '''
    paired_effb_c = _calc_paired_effbase_correctors(effbases)[1]
    paired_ovlp_c = _calc_paired_overlap_correctors(overlap_bases, effbases)

    pooled_props = _calc_pooled_props_pairs(counts, bases)

    return np.sqrt(pooled_props*(1-pooled_props)*(np.array(paired_effb_c - paired_ovlp_c)))

def _calc_se_pooled_means_pairs(stddevs, bases, effbases, overlap_bases, ratios):
    '''
    Returns the pooled sample standard error for all unique
    column pairings' means.

    Parameters
    ----------
    stddevs,
    bases, effbases, overlap_bases,
    ratios : np.array
        Inputs of standard deviations, bases,
        effective basses, overlap_bases
        and sum(w**2)/base measures.

    Returns
    pooled_se : np.array
        Numpy array storing each column pair's pooled
        standard error.
    '''
    paired_effb_c = _calc_paired_effbase_correctors(effbases)[1]
    paired_ovlp_c = _calc_paired_overlap_correctors(overlap_bases, effbases)

    enum = (stddevs ** 2) * (bases - 1)
    denom = bases - ratios

    enum_pairs = np.array([x + y for x, y in combinations(enum, 2)])
    denom_pairs = np.array([x + y for x, y in combinations(denom, 2)])

    return np.sqrt((enum_pairs / (denom_pairs)) * (paired_effb_c - paired_ovlp_c))


def _cwi(default, threshold=5):
    '''
    Will filter a dataframe of weighted or unweghted absolute cell frequencies
    according to the "counts when independent" computation used in the survey
    analysis software "Askia".
    Further documentation can be found via:
    http://blog.askia.com/tag/counts-when-independent/

    Parameters
    ----------
    default : Quantipy default view df
        Crossed x, y aggregation; y!=@
    threshold : int, default = 5
        The threshold that applies before a given cell cwi value
        is filtered out of the cell frequency dataframe.

    Returns
    -------
    freqs_cwi : pd.DataFrame
        Only contains cell values above the threshold
        defined for the cwi measure.
    '''

    freqs, idx_names, col_names = struct._partition_view_df(
        default, values=True, data_only=False)

    t_col_n = freqs[-1, -1]
    t_cell_n = freqs[:-1, [-1]]
    c_col_n = freqs[-1, :-1]
    c_cell_n = freqs[:-1, :-1]

    np.place(t_col_n, t_col_n == 0, np.NaN)
    np.place(t_cell_n, t_cell_n == 0, np.NaN)
    np.place(c_col_n, c_col_n == 0, np.NaN)
    np.place(c_cell_n, c_cell_n == 0, np.NaN)

    cwi = (t_cell_n * c_col_n) / t_col_n
    cwi[cwi < threshold] = np.NaN

    return pd.DataFrame(c_cell_n + cwi - cwi,
                        index=idx_names[:-1], columns=col_names[:-1])


def _calc_paired_overlap_correctors(overlap_bases, eff_bases):
    '''
    Function to compute the overlap correction factor
    used in the Dimensions-like freqs. sig. testing algorithm.

    Parameters
    ----------
    overlap_bases : np.array
        Numpy array of overlap bases for all unique column pairs.
    eff_bases : np.array
        Numpy array of effective bases per column.

    Returns
    -------
    overlap_corrector : np.array
        Numpy array of correction factors per column pair.
    '''

    return (2*overlap_bases)/[x*y for x, y in combinations(eff_bases, 2)]


def _calc_paired_effbase_correctors(eff_bases):
    '''
    Function to compute the effective base correction factor
    used in the Dimensions-like freqs. sig. testing algorithm.
    Will also return the effective sample size that is used
    in the calculation of the degrees of freedom of the test
    statistics.

    Parameters
    ----------
    eff_bases : np.array
        Numpy array of effective bases per y-column.

    Returns
    -------
    eff_sample_size,
    eff_base_correction : tuple of np.arrays
        The correction factors and the effective sample sizes
        for all column pairs.
    '''
    eff_ssize = [x + y for x, y in combinations(eff_bases, 2)]

    eff_base_correction = [1/x + 1/y
                           for x, y in combinations(eff_bases, 2)]

    return np.array(eff_ssize), np.array(eff_base_correction)



def _filter_for_sigs(diff, test_df, level, package='Dim'):
    '''
    This method filters the t-values contained in test_df
    by the value differences per category intersection.
    After comparing the relevant t-values against the sig. level
    provided, it produces a DataFrames showing only the stat.
    sig. results for both the positive and negative direction of the
    difference.

    Parameters
    ----------
    diff, test_df : pd.DataFrame
        Note: The DataFrame inputs must be identically indexed
        on both axes.

    level : float
        The desired level of significance.

    package : str, default="Dim"
        The statistical software mimicked.
    Returns
    -------
    masked_diff : pd.DataFrame
    '''
    if package == 'Dim':
        return diff[(diff != 0)&(test_df < level)]
    elif package == 'askia':
        return diff[(diff != 0)&(test_df > level)]


def _make_sigtest_df(sig_res):
    col_res = defaultdict(list)
    row_res = {}
    res_collec = []

    sigs = sig_res.T.to_dict()
    for row, colpair_res in list(sigs.items()):
        col_res.clear()
        for colpair, result in list(colpair_res.items()):
            if result < 0:
                col_res[int(colpair[1])].append(int(colpair[0]))
                col_res[int(colpair[0])].append(-1)
            elif result > 0:
                col_res[int(colpair[0])].append(int(colpair[1]))
                col_res[int(colpair[1])].append(-1)
            else:
                col_res[int(colpair[1])].append(-1)
                col_res[int(colpair[0])].append(-1)
        row_res = {int(col): str(sorted(list(set(res)))).replace('-1, ', '')
                   for col, res in list(col_res.items())}
        res_collec.append(pd.DataFrame(row_res, index=[int(row)]))

    sigtest = pd.concat(res_collec).replace('[-1]', np.NaN).sort_index()

    return sigtest


def _overlap(mat, xdef, ydef):
    '''

    '''
    mat = mat[:, [0]] * mat[:, len(xdef)+1:]
    mat[mat == 0] = np.NaN

    w_sum_sq_paired = np.hstack(
        [np.nansum(mat[:, [col1]] + mat[:, [col2]], axis=0)**2
         for col1, col2 in combinations(range(0, mat.shape[1]), 2)])
    w_sq_sum_paired = np.hstack(
        [np.nansum(mat[:, [col1]]**2 + mat[:, [col2]]**2)
         for col1, col2 in combinations(range(0, mat.shape[1]), 2)])

    return np.nan_to_num((w_sum_sq_paired/w_sq_sum_paired)/2)


def _props_diff_pairs(counts, bases):
    '''
    Function to calculate the pct. point differences for all
    unique column pairs from counts and bases View dataframes.

    Parameters
    ----------
    counts, bases : pd.DataFrame
        As produced per Quantipy's freq and base view methods.

    Returns
    -------
    pct_diff : np.array
        Numpy array of pct. point differences.
    '''
    props = (counts / bases).T
    return np.array([p1 - p2 for p1, p2 in combinations(props, 2)]).T

def _means_diff_pairs(means):
    '''
    Function to calculate the differences between mean figures
    for all unique column pairs of a Quantipy Link holding a
    mean aggregations View.

    Parameters
    ----------
    means : np.array
        As produced by _mean_from_mat()

    Returns
    -------
    mean_diff : np.array
        Numpy array of mean differences for
        all unique column pairs.
    '''
    return np.array([x - y for x, y in combinations(means, 2)])


def _calc_pooled_props_pairs(counts, bases):
    '''
    Function to calculate the pooled percentages inside all
    unique column pairs from counts and bases View dataframes.

    Parameters
    ----------
    counts, bases : pd.DataFrame
        As produced per Quantipy's freq and base view methods.

    Returns
    -------
    pooled_pct : np.array
        Numpy array of pooled column percentages.
    '''
    #return np.array([p1 - p2 for p1, p2 in combinations(props, 2)]).T

    counts_sum_pairs = np.array([x + y for x, y in combinations(counts.T, 2)])
    bases_sum_pairs = np.array([x + y for x, y in combinations(bases, 2)])
    return (counts_sum_pairs/bases_sum_pairs.reshape(bases_sum_pairs.shape[0],1)).T

# def get_matrix(link, weights=None, style='freq',
#                limit_to=None, exclude=None, rescale=None, data=None):
#     '''
#     '''
#     weight = weights if weights is not None else '@1'
#     if data is None:
#         data = link.get_data().copy()
#     if link.y == '@' or link.x == '@':
#         var = link.x if not link.x == '@' else link.y
#         data = data[
#             [weight, var]
#             ].replace('', np.NaN).dropna(subset=([var]))
#         xmat = _make_dummies(data[data.columns[1]],
#                              limit_to=limit_to, exclude=exclude, rescale=rescale,
#                              style=style, as_df=False)
#         w = data[[weight]].values

#         if style == 'num':
#             factors = xmat[0]*xmat[1]
#             return (np.hstack((w, w*factors, xmat[0]*w)),
#                     xmat[1], None)
#         elif style == 'freq':
#             return (np.hstack((w, xmat[0]*w)),
#                     xmat[1], None)
#     else:
#         data = data[
#             [weight, link.x, link.y]
#             ].replace('', np.NaN).dropna(subset=([link.x, link.y]))
#         if not data.columns.is_unique:
#             data.columns = [weight, link.x, link.y + '_']
#         xmat = _make_dummies(data[data.columns[1]],
#                              limit_to=limit_to, exclude=exclude, rescale=rescale,
#                              style=style, as_df=False)
#         ymat = _make_dummies(data[data.columns[2]], style=style, as_df=False)
#         w = data[[weight]].values

#         if style == 'num':
#             factors = xmat[0]*xmat[1]
#             return (np.hstack((w, w*factors, xmat[0]*w, ymat[0])),
#                     xmat[1], ymat[1])
#         elif style == 'freq':
#             return (np.hstack((w, xmat[0]*w, ymat[0])),
#                     xmat[1], ymat[1])
#     print _effective_n(matrix, ydef)

''' Basic matrix creation, retrievel and manipulation functions:
'''

def get_matrix(link, weights, data=None):
    '''

    Example A - unweighted, x=single, y=multi:

    | wv | x section | y section |
    ------------------------------
    | 0  | 1 2 3 4 5 | 1  2  3   |
    ------------------------------
    | 1  | 0 1 0 0 0 | 0  1  0   |
    | 0  | 0 0 1 0 0 | 1  1  1   |
    | 1  | 0 0 0 0 1 | 0  0  1   |
    | 1  | 0 0 0 0 1 | 1  1  0   |
    | 1  | 1 0 0 0 0 | 0  1  0   |
    | 0  | 0 0 0 1 0 | 0  1  1   |
    ...


    '''
    if data is None:
        data = link.get_data().copy()
    weight = weights if weights is not None else '@1'
    if link.x == '@' or link.y == '@':
        var = link.x if not link.x == '@' else link.y
        data = data[[weight, var]].replace('', np.NaN)
        #data.dropna(inplace=True)
        wv = data[[weight]].values
        xm, xdef = _make_dummies(data[var])
        ydef = None

        mat = np.concatenate((wv, xm), axis=1)
        mat = _clean_matrix(mat, xdef, ydef)

    else:
        data = data[[weight, link.x, link.y]].replace('', np.NaN)
        #data.dropna(inplace=True)
        wv = data[[weight]].values
        if not data.columns.is_unique:
            data.columns = [weight, link.x, link.y + '_']
        xm, xdef = _make_dummies(data[data.columns[1]])
        ym, ydef = _make_dummies(data[data.columns[2]])

        mat = np.concatenate((wv, xm, ym), axis=1)
        mat = _clean_matrix(mat, xdef, ydef)

    return mat, xdef, ydef

def _make_dummies(data, limit_to=None, exclude=None, rescale=None, style='freq', as_df=False):
    '''
    Creates a dichotomously 1/0-coded version of the incoming pd.Series with the answer codes
    found in the data being transformend into column names. This representation of the data situationally
    can be easier to work with if multicoded variables are being fed into computations (i.e. Numpy).
    The function unites the two .get_dummies() functions found in pandas, see also:
    - pandas.core.reshape.get_dummies
    - pandas.core.strings.StringMethods.get_dummies

    Parameters
    ----------
    data : pd.Series
        Contanis the single- or multicoded variable that
        should be transformed into the dummy representation.

    limit_to : list, default=None
        List of codes that should be used in the transformation.
        If None (=default), the function will transform all data.

    as_df : bool, default=False
        Controls if the returned output is a pd.DataFrame (=True) or
        a tuple of the DataFrame's values and the column codes (=False).

    Returns
    -------
    tuple : np.array, list of column codes
    OR
    dummy_df : pd.DataFrame
    '''
    if data.dtype == 'object':
        # i.e. Quantipy multicode data
        dummy_df = data.str.get_dummies(';')
        dummy_df.columns = [int(col) for col in dummy_df.columns]
        # if style == 'freq':
        #     dummy_df.sort_index(axis=1, inplace=True)
        #     dummy_df.rename(
        #         columns={col: str(col) for col in dummy_df.columns},
        #         inplace=True
        #         )
        # else:
        dummy_df.sort_index(axis=1, inplace=True)
    else:
        #data.dropna(inplace=True)
        dummy_df = pd.get_dummies(data)
        if style == 'freq':
            dummy_df.rename(
                columns={
                    col: int(col)
                    if float(col).is_integer()
                    else col
                    for col in dummy_df.columns
                },
                inplace=True
            )

    if limit_to:
        dummy_df = _limit_dummy_df(dummy_df, limit_to)
    if exclude:
        _exclude_codes(dummy_df, exclude)
    if rescale:
        _rescale_codes(dummy_df, rescale)
    if as_df:
        return dummy_df
    else:
        return dummy_df.values, dummy_df.columns.tolist()

def _clean_matrix(mat, xdef, ydef):
    '''
    Returns a copy of the input matrix that only contains rows
    that are populated both on the x and the y section according
    to its y and y defintion.

    Parameters
    ----------
    mat : np.array
        1/0 representation of a Link data defintiton.
        Produced by tools.view.agg.get_matrix().
    xdef, ydef : list
        x and y section defintion of the input matrix.
        Produced by tools.view.agg.get_matrix().

    Returns
    -------
    mat : np.array
        Clean copy of the input matrix with only fully populated
        x and y section rows.
    '''
    mat = mat.copy()
    xmask = (np.sum(mat[:, 1:len(xdef)+1], axis=1) > 0)
    if ydef is not None:
        ymask = (np.sum(mat[:, len(xdef)+1:], axis=1) > 0)
        return mat[xmask&ymask]
    else:
        return mat[xmask]


def weight_matrix(mat, xdef):
    '''
    Returns a copy of the input matrix which contains the x section's
    1-value entries multiplied by the the weight vector.

    Parameters
    ----------
    mat : np.array
        1/0 representation of a Link data defintiton.
        Produced by tools.view.agg.get_matrix().
    xdef: list
        x section defintion of the input matrix.
        Produced by tools.view.agg.get_matrix().

    Returns
    -------
    mat : np.array
        Weighted copy of the input matrix with regard to the x section.
    '''
    mat = mat.copy()
    mat[:, 1:len(xdef)+1] = mat[:, 1:len(xdef)+1]*mat[:, [0]]

    return mat


def _unweight_matrix(mat, xdef):
    '''
    Returns a copy of the (weighted) input matrix without a
    weight vector multiplied x section.

    Parameters
    ----------
    mat : np.array
        1/0 representation of a Link data defintiton.
        Produced by tools.view.agg.get_matrix().
    xdef : list
        x section defintion of the input matrix.
        Produced by tools.view.agg.get_matrix().

    Returns
    -------
    mat : np.array
        Unweighted copy of the input matrix with regard to the x section.
    '''
    mat = mat.copy()
    mat[:, 1: len(xdef)+1] = (mat[:, 1:len(xdef)+1] / mat[:, [0]])

    return mat


def _get_drop_idx(xdef, codes, keep):
    '''
    Produces a list of indices refering to the input matrix's x section in
    order to erase data entries.
    See also: tools.view.agg.missingfy_matrix().

    Parameters
    ----------
    xdef :  list
        x section defintion of the input matrix.
        Produced by tools.view.agg.get_matrix().
    codes : list
        Data entry codes that should be dropped from or kept in the matrix.
    keep : boolean
        Controls if the the passed code defintion is interpreted as
        "codes to keep" or "codes to drop".

    Returns
    -------
    drop_idx : list
        List of x section matrix indices.
    '''
    if codes is None:
        return None
    elif keep:
        return [xdef.index(code) for code in xdef if code not in codes]
    else:
        return [xdef.index(code) for code in codes if code in xdef]

def missingfy_matrix(mat, xdef, codes, keep=False):
    '''
    '''
    mis_ix = _get_drop_idx(xdef, codes, keep)
    if mis_ix is not None:
        mat = mat.copy()
        for ix in mis_ix:
            np.place(mat[:, ix+1], mat[:, ix+1] > 0, np.NaN)
        if not keep:
            wv_mask = (np.sum(mat[:, 1:len(xdef)+1], axis=1) > 0)
        else:
            wv_mask = (np.nansum(mat[:, 1:len(xdef)+1], axis=1) > 0)
        mat[~wv_mask, [0]] = np.NaN

    return mat

def _factorize_matrix(mat, xdef, scaling=None):
    mat = mat.copy()
    mat[:, 1:len(xdef)+1] = mat[:, 1:len(xdef)+1]*xdef

    return mat

def _refactor_xdef(xdef, scaling):
    clean_scaling = {old_code: new_code for old_code, new_code
                     in list(scaling.items())
                     if old_code in xdef}

    return [clean_scaling[code] if code in list(clean_scaling.keys())
            else code for code in xdef]

def _reduce_xsect(mat, xdef):
    mat = mat.copy()
    redx = np.nansum(mat[:, 1:len(xdef)+1], axis=1).reshape(mat.shape[0], 1)

    return np.concatenate((mat[:, [0]], redx, mat[:, 1+len(xdef):]), axis=1)

def _get_ysect(mat, ydef):
    mat = mat.copy()
    if ydef is not None:
        ysec = reversed(range(1, len(ydef)+1))
        return [mat[mat[:, -y] == 1] for y in ysec]
    else:
        return [mat]

''' Aggregation functions that produce raw frequencies from a
    prepared input matrix and code defintiton:
'''

def _cell_n(mat, xdef, ydef):
    if ydef is not None:
        xcodes = range(1, len(xdef)+1)
        return np.vstack(
            [np.sum(
                mat[:, len(xdef)+1:]*mat[:, [xcode]], axis=0)
                for xcode in xcodes])

    else:
        return np.sum(mat[:, 1:len(xdef)+1], axis=0)

def _col_n(mat, xdef, ydef):
    if ydef is not None:
        ycodes = reversed(range(1, len(ydef)+1))
        return np.array([np.nansum(mat[:, [0]]*mat[:, [-ycode]])
                        for ycode in ycodes])
    else:
        return np.nansum(mat[:, [0]])

def _row_n(mat, xdef):
    return np.sum(mat[:, 1:len(xdef)+1], axis=0)


def _total_n(mat):
    return np.nansum(mat[:, [0]], axis=0)

def _effective_n(mat, ydef):
    if ydef is not None:
        ycodes = reversed(range(1, len(ydef)+1))

        return np.array([np.nansum(mat[:,[0]]*mat[:,[-ycode]])**2 /
                         np.nansum((mat[:,[0]]*mat[:,[-ycode]])**2)
                         for ycode in ycodes])
    else:
        return np.array(np.nansum(mat[:,[0]])**2 /
                        np.nansum(mat[:,[0]]**2))


''' Functions to generate statistical summary information / descriptives
    from a prepared input matrix and definition
'''

# def _mean(mat, xdef, ydef):
#     f_mat = _factorize_matrix(mat.copy(), xdef)
#     if ydef is not None:
#         ycodes = reversed(xrange(1, len(ydef)+1))

#         return np.array([np.nansum(f_mat[:, 1:len(xdef)+1]*mat[:, [-ycode]]) /
#                          np.nansum(mat[:, 1:len(xdef)+1]*mat[:, [-ycode]])
#                          for ycode in ycodes])
#     else:
#         return np.array(np.nansum(f_mat[:, 1:len(xdef)+1]) /
#                         np.nansum(mat[:, 1:len(xdef)+1]))

def _mean(mat, xdef, ydef):
    '''
    Computes the arithmetic mean from the incoming distribution given as per
    the data's matrix definition.
    '''
    mat = _factorize_matrix(mat, xdef)
    mat = _reduce_xsect(mat, xdef)
    ysects = _get_ysect(mat, ydef)

    return np.array([np.nansum(mat[:, 1] /
                     np.nansum(mat[:, 0]))
                    for mat in ysects])


def _percentile(mat, xdef, ydef, perc=0.5):
    '''
    Computes percentiles from the incoming distribution given as per
    the data's matrix definition and the requested percentile value.
    The implementation mirrors the algorithm used in SPSS Dimensions and
    the EXAMINE procedure in SPSS Statistics. Weighted data supported.
    It based on the percentile defintion #6 in:
    >>> Hyndman, Rob J. and Fan, Yanan (1996) -
    "Sample Quantiles in Statistical Packages",
    The American Statistician, 50, No. 4, 361-365.

    Parameters
    ----------

    perc : float, default=0.5
        Defines the percentile to be computed. Defaults to 0.5,
        the sample median.

    Returns
    -------
    '''
    percs = []
    mat = _unweight_matrix(mat, xdef)
    mat = _factorize_matrix(mat, xdef)
    mat = _reduce_xsect(mat, xdef)
    ysects = _get_ysect(mat, ydef)

    for mat in ysects:
        sortidx = np.argsort(mat[:, 1])
        mat = np.take(mat, sortidx, axis=0)
        wsum = np.sum(mat[:,0], axis=0)
        wcsum = np.cumsum(mat[:, 0], axis=0)
        k = (wsum+1)*perc

        if wcsum[0] > k:
            wcsum_k = wcsum[0]
            percs.append(mat[0, 1])
        elif wcsum[-1] < k:
            percs.append(mat[-1, 1])
        else:
            wcsum_k  = wcsum[wcsum <= k][-1]
            p_k_idx = np.searchsorted(np.ndarray.flatten(wcsum), wcsum_k)
            p_k = mat[p_k_idx, 1]
            p_k1 = mat[p_k_idx+1, 1]
            w_k1 = mat[p_k_idx+1, 0]
            excess = k - wcsum_k
            if excess >= 1.0:
                percs.append(p_k1)
            else:
                if w_k1 >= 1.0:
                    percs.append((1.0-excess)*p_k + excess*p_k1)
                else:
                    percs.append((1.0-excess/w_k1)*p_k + (excess/w_k1)*p_k1)

    return np.array(percs)

def _dispersion(mat, xdef, ydef, measure='sd', return_mean=False):
    means = _mean(mat, xdef, ydef)
    unbiased_n = _col_n(mat, xdef, ydef) - 1
    if type(unbiased_n) == np.float64:
        print('THIS IS A STUPID CHECK: FIX COL_N')
        unbiased_n = [unbiased_n]
    mat = _unweight_matrix(mat, xdef)
    mat = _factorize_matrix(mat, xdef)
    mat = _reduce_xsect(mat, xdef)
    np.place(mat[:, 1],
             mat[:, 1] == 0, np.NaN)
    ysects = _get_ysect(mat, ydef)

    var = np.array([(np.nansum(mat[:, 0] *
                              (mat[:, 1] - means[idx]) ** 2)) /
                              unbiased_n[idx]
                    for idx, mat in enumerate(ysects)])

    var[var < 0] = 0

    if measure == 'sd':
        if return_mean:
            return means, np.array(np.sqrt(var))
        else:
            return np.array(np.sqrt(var))
    elif measure == 'vcoef':
        if return_mean:
            return means, np.array(np.sqrt(var)/means)
        else:
            return np.array(np.sqrt(var)/means)
    else:
        if return_mean:
            return means, var
        else:
            return var

def _sum_sq_w(mat, xdef, ydef, base_ratio=True):
    if ydef is not None:
        ycodes = reversed(range(1, len(ydef)+1))
        ssw = np.array([np.nansum((mat[:, [0]]*mat[:, [-ycode]])**2)
                        for ycode in ycodes])
    else:
        ssw =  np.array(np.nansum((mat[:, [0]])**2))

    if base_ratio:
        cb = _col_n(mat, xdef, ydef)
        return np.array(ssw/cb)
    else:
        return np.array(ssw)



def verify_logic_values(values, func_name):
    """ Verifies that the values given are a list of ints.

    Parameters
    ----------
    values : list-like
        The values to be tested

    func_name : string ('any' or 'all' ONLY)
        The name of the logic being used

    Returns
    -------
    None
    """
    if isinstance(values, (list, tuple)):
        for value in values:
            if not isinstance(value, int):
                raise TypeError(
                    "The values given to has_%s() are not correctly "
                    "typed. Expected list of <int>, found a %s." % (
                        func_name,
                        type(value)
                    )
                )
    else:
        raise TypeError(
            "The values given to has_%s() must be given as a list. "
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

    func_name : string ('any' or 'all' ONLY)
        The name of the logic being used

    Returns
    -------
    None
    """
    if not series.dtype in ['object', 'int64', 'float64']:
        raise TypeError(
            "The series given to has_%s() must be a supported dtype. "
            "Expected 'object', 'int64' or 'float64', found a '%s'." % (
                func_name,
                series.dtype
            )
        )


def verify_count_responses(responses):
    """ Verifies that the responses given are well formed.

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
    None
    """

    if isinstance(responses, int):
        responses = [responses]
    elif isinstance(responses, (list, tuple)):
        if not len(responses) in [2, 3]:
            raise IndexError (
                "The responses list given to has_count() is must have "
                "either 2 or 3 items in the form: "
                "[min, max, [values subset]]. Found %s." % (responses)
            )
        valid_types = [int, int, (list, tuple)]
        for r, response in enumerate(responses):
            if not isinstance(response, valid_types[r]):
                raise TypeError (
                    "The responses list given to has_count() has "
                    "incorrectly typed items. It must be either 2 or 3 "
                    "items in the form: [int, int, list/tuple]. "
                    "Found %s." % (responses)
                )
            if r==3:
                for value in response:
                    if not isinstance(value, int):
                        raise TypeError (
                            "The values subset given as the third item "
                            "in has_count(responses) is not correctly "
                            "typed. Each value must be int. "
                            "Found %s." % (response)
                        )

    return responses

def _any_all_none(series, values, func_name):
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
        # If not valid columns are availabe, the result is no rows
        if not cols:
            return []
        else:
            dummies = dummies[cols]
        # Slice the dummies row-wise for only rows with any/all/none of
        # the targeted responses
        if func_name=='any':
            # Apply 'any' logic
            dummies = dummies[(dummies.T!=0).any()]
        elif func_name=='all':
            # Apply 'all' logic
            dummies = dummies[(dummies.T!=0).all()]
        else:
            # Apply 'none' logic
            dummies = dummies[(dummies.T==0).all()]

        # Return the index
        return dummies.index

    elif series.dtype in ['int64', 'float64']:
        # Slice the series row-wise for only rows with any/all of the
        # targets responses
        if func_name=='any' or (func_name=='all' and len(values)==1):
            series = series[series.isin(values)].dropna()
        elif func_name=='none':
            series = series[~series.isin(values)]
        else:
            # has_all() for multiple values is being requested on a
            # single-type variable, so the result will be none
            return []

        # Return the index
        return series.index

    else:
        raise TypeError(
            "The dtype '%s' of series is incompatible with has_%s()" %
                series.dtype,
                func_name
        )


def has_any(values):
    """ Convenience for managing 'any' part of the 'logic' instructions
    provided in a freq method's kwargs.

    Parameters
    ----------
    values : list-like
        List of values on which an 'any' condition will be used

    Returns
    -------
    _any : function
        The function that will be used to evaluate the 'any' condition

    values : list-like
        List of values on which an 'any' condition will be used
    """
    verify_logic_values(values, 'any')
    return _any, values


def _any(series, values):
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
    verify_logic_series(series, 'any')
    return _any_all_none(series, values, "any")


def has_all(values):
    """ Convenience for managing 'all' part of the 'logic' instructions
    provided in a freq method's kwargs.

    Parameters
    ----------
    values : list-like
        List of values on which an 'all' condition will be used

    Returns
    -------
    _all : function
        The function that will be used to evaluate the 'all' condition

    values : list-like
        List of values on which an 'any' condition will be used
    """
    verify_logic_values(values, 'all')
    return _all, values


def _all(series, values):
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
    verify_logic_series(series, 'all')
    return _any_all_none(series, values, "all")


def has_none(values):
    """ Convenience for managing 'none' part of the 'logic' instructions
    provided in a freq method's kwargs.

    Parameters
    ----------
    values : list-like
        List of values on which an 'none' condition will be used

    Returns
    -------
    _none : function
        The function that will be used to evaluate the 'none' condition

    values : list-like
        List of values on which an 'any' condition will be used
    """
    verify_logic_values(values, 'none')
    return _none, values


def _none(series, values):
    """ Returns the index of rows from series containing none of the given
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
        The index of series for rows containing none of the given values.
    """
    verify_logic_series(series, 'none')
    return _any_all_none(series, values, "none")


def has_count(responses):
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
    _count : function
        The function that will be used to evaluate the 'count of
        responses' condition

    responses :
        List of values on which a 'count of responses' condition will
        be used
    """
    responses = verify_count_responses(responses)
    return _count, responses


def _count(series, responses):
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
    verify_logic_series(series, 'none')

    if series.dtype in ['object', 'int64', 'float64']:

        # Get the dichotomous version of series
        dummies = series.str.get_dummies(';')
        try:
            # Slice the dummies column-wise for only the targeted values
            values = [str(v) for v in responses[2]]
            cols = [col for col in dummies.columns if col in values]
            # If not valid columns are availabe, the result is no rows
            if not cols:
                return []
            else:
                dummies = dummies[cols]
        except:
            pass

        # Get a count of the number of responses
        count = dummies.sum(axis=1)

        # Get the min and max targeted responses
        min_responses = responses[0]
        try:
            max_responses = responses[1]
        except:
            max_responses = None

        # Get a boolean slicing mask for use on dummies
        if max_responses is None:
            mask = count==min_responses
        else:
            mask = (count>=min_responses) & (count<=min_responses)

        # Slice the dummies row-wise for only rows with the targeted
        # count of responses
        dummies = dummies.loc[mask]

        # Return the index
        return dummies.index

    else:
        raise TypeError(
            "The dtype '%s' of series is incompatible with has_%s()" %
                series.dtype,
                func_name
        )

def get_logic_key_chunk(has_func, values):
    """ Derives the relationship view key chunk describing the combination
    of the given function and values.

    Parameters
    ----------
    has_func : function
        The logical function (has_any, has_all, has_none or has_count).

    values : int or list
        The values associated with the logical function.

    Returns
    -------
    chunk : str
        The relationship-part of the view key that represents this
        combination of view function and values.
    """

    if has_func is _any:
        values = [str(v) for v in values]
        chunk = '%s' % (','.join(values))

    elif has_func is _all:
        values = [str(v) for v in values]
        chunk = '%s' % ('&'.join(values))

    elif has_func is _none:
        values = [str(v) for v in values]
        chunk = '~(%s)' % (','.join(values))

    elif has_func is _count:
        # Get the min, max and targeted responses
        min_responses = values[0]
        try:
            max_responses = values[1]
        except:
            max_responses = None
        try:
            values = values[2]
            values = [str(v) for v in values]
        except:
            values = None

        if not max_responses is None:
            if min_responses==max_responses:
                min_max = min_responses
            else:
                min_max = '%s-%s' % (min_responses, max_responses)

        if values is None:
            if max_responses is None:
                chunk = '{%s}' % (min_responses)
            else:
                chunk = '{%s}' % (min_max)
        else:
            chunk = '(%s){%s}' % (','.join(values), min_max)

    return chunk


def get_logic_index(series, logic):
    """ Uses the given complex logic block to return a slice of series.

    Parameters
    ----------
    series : pandas.Series
        The series on which the logic should be applied.

    logic : list or tuple
        The complex logic block to be applied. Must be well-formed.

    Returns
    -------
    series : pandas.Series
        The logical slice of the incoming series.

    relation : str
        The relationship-part of the view key that represents this
        logical block.
    """

    if isinstance(logic, list):
        return _any(series, logic)

    elif isinstance(logic, tuple):

        has_func, values = (logic)
        idx = has_func(series, values)

        vkey = 'x[%s]:y' % (
            get_logic_key_chunk(has_func, values)
        )

    return idx, vkey
