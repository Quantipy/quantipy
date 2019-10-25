#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import json
import pickle
import numpy as np
import pandas as pd
from collections import OrderedDict


# -----------------------------------------------------------------------------
# i/o
# -----------------------------------------------------------------------------
def cpickle_copy(obj):
    copy = pickle.loads(pickle.dumps(obj, pickle.HIGHEST_PROTOCOL))
    return copy


def load_json(path_json, hook=OrderedDict):
    with open(path_json) as f:
        obj = json.load(f, object_pairs_hook=hook)
        return obj


def loads_json(json_text, hook=OrderedDict):
    obj = json.loads(json_text, object_pairs_hook=hook)
    return obj


def represent(obj):
    if isinstance(obj, np.generic):
        return np.asscalar(obj)
    else:
        return "Unserializable object: {}".format(type(obj))

def save_json(obj, path_json):
    with open(path_json, 'w+') as f:
        json.dump(obj, f, default=represent, sort_keys=True)


def load_csv(path_csv):
    data = pd.DataFrame.from_csv(path_csv)
    return data


# -----------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------
def _merge_delimited_sets(x, y):
    codes = []
    x = str(x) + str(y).replace("nan", "")
    for c in x.split(';'):
        if not c:
            continue
        if not c in codes:
            codes.append(c)
    if not codes:
        return np.NaN
    else:
        return ';'.join(sorted(codes)) + ';'

def remove_codes(x, remove):
    if any([x is np.NaN, x in remove, x == ""]):
        x = np.NaN
    elif ';' in str(x):
        remove = [str(r) for r in remove]
        x = str(x).split(';')
        x = [y for y in x if y not in remove]
        x = ';'.join(x) or np.NaN
    return x

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

# -----------------------------------------------------------------------------
# lists
# -----------------------------------------------------------------------------
def frange(range_def, sep=','):
    """
    Return the full, unabbreviated list of ints suggested by range_def.

    This function takes a string of abbreviated ranges, possibly
    delimited by a comma (or some other character) and extrapolates
    its full, unabbreviated list of ints.

    Parameters
    ----------
    range_def : str
        The range string to be listed in full.
    sep : str, default=','
        The character that should be used to delimit discrete entries in
        range_def.

    Returns
    -------
    res : list
        The exploded list of ints indicated by range_def.
    """
    res = []
    for item in range_def.split(sep):
        if '-' in item:
            a, b = item.split('-')
            a, b = int(a), int(b)
            lo = min([a, b])
            hi = max([a, b])
            ints = range(lo, hi+1)
            if b <= a:
                ints = list(reversed(ints))
            res.extend(ints)
        else:
            res.append(int(item))
    return res


def flatten_list(the_list):
    flat = []
    for item in the_list:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)
    return flat


def ensure_list(obj):
    if not obj:
        obj = []
    elif not isinstance(obj, list):
        obj = [obj]
    return obj


def dupes_and_unique(the_list):
    unique = []
    dupes = []
    for item in the_list:
        if item in unique:
            dupes.append(item)
        else:
            unique.append(item)
    return unique, dupes


def uniquify_list(the_list):
    unique, _ = dupes_and_unique(the_list)
    return unique


def dupes_in_list(the_list):
    _, dupes = dupes_and_unique(the_list)
    return dupes


def insert_by_anchor(the_list, incl):
    """
    Include items into an existing list and position by anchor.

    Parameters
    ----------
    the_list : list
        The list to be extended.
    incl : dict/ list
        Keys of a provided dict are the anchor for the given value.
        The anchor can either be the item or the index. The new items are
        always added before the anchor.
        Given list without an anchor is added at the end.

    Returns
    -------
    the_list: extended list

    Note
    ----
    Only string items and lists of str are supported.
    """
    if not all(isinstance(i, str) for i in the_list):
        msg = "Only string items are supported!"
        logger.error(msg); raise ValueError(msg)
    if not isinstance(incl, (list, dict)):
        msg = "'incl' must either be dict or list!"
        logger.error(msg); raise ValueError(msg)
    if isinstance(incl, list):
        the_list.extend(incl)
        return the_list

    for v in flatten_list(list(incl.values())):
        while v in the_list:
            the_list.remove(v)

    ext = incl.pop(-1, [])
    if not isinstance(ext, list):
        ext = [ext]
    the_list.extend(ext)
    for k in list(incl.keys()):
        if isinstance(k, str):
            incl[the_list.index(k)] = incl.pop(k)
            k = the_list.index(k)
        if not isinstance(incl[k], list):
            incl[k] = [incl[k]]
    for k in reversed(sorted(incl.keys())):
        the_list = the_list[:k] + incl[k] + the_list[k:]
    return the_list
