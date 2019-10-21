#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import json
import pickle
import numpy as np
import pandas as pd


def set_encoding(encoding):
    """
    Hack sys.setdefaultencoding() to escape ASCII hell.

    Parameters
    ----------
    encoding : str
        The name of the encoding to default to.
    """
    default_stdout = sys.stdout
    default_stderr = sys.stderr
    reload(sys)
    sys.setdefaultencoding(encoding)
    sys.stdout = default_stdout
    sys.stderr = default_stderr


def make_like_ascii(text):
    """
    Replaces any non-ascii unicode with ascii unicode.

    http://www.fileformat.info/info/unicode/char/
    """
    unicode_ascii_mapper = {
        u'\u2022': u'-',
        u'\u2013': u'-',
        u'\u2018': u'\u0027',
        u'\u2019': u'\u0027',
        u'\u201c': u'\u0022',
        u'\u201d': u'\u0022',
        u'\u00a3': u'GBP',
        u'\u20AC': u'EUR',
        u'\u2026': u'\u002E\u002E\u002E',
    }
    for old, new in unicode_ascii_mapper.iteritems():
        text = text.replace(old, new)
    return text


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


def save_json(obj, path_json, decode_str=False, decoder='UTF-8'):
    if decode_str:
        obj = unicoder(obj, decoder)

    def represent(obj):
        if isinstance(obj, np.generic):
            return np.asscalar(obj)
        else:
            return "Unserializable object: {}".format(type(obj))
    with open(path_json, 'w+') as f:
        json.dump(obj, f, default=represent, sort_keys=True)


def load_csv(path_csv):
    data = pd.DataFrame.from_csv(path_csv)
    return data


# -----------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------
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


def _dupes_in_list(the_list):
    unique = []
    dupes = []
    for item in the_list:
        if item in unique:
            dupes.append(item)
        else:
            unique.append(item)
    return unique, dupes


def uniquify_list(the_list):
    unique, _ = _dupes_in_list(the_list)
    return unique


def dupes_in_list(the_list):
    _, dupes = _dupes_in_list(the_list)
    return dupes
