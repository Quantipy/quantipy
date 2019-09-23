
import numpy as np
import pandas as pd
import quantipy as qp
from quantipy.core.helpers.functions import emulate_meta
import savReaderWriter as srw
import copy
import json

def write_sav(path_sav, data, **kwargs):
    """
    Write the given records to a SAV file at path_sav.

    Using the various definitions indicated by the packed kwargs, write
    the given set of records to a SAV file at the location indicated by
    path_sav.

    For a full explanation of the kwargs used please see the
    savReaderWriter library documentation here:
    http://pythonhosted.org/savReaderWriter/

    Parameters
    ----------
    path_sav : str
        The full path, including extension, indicating where the output
        file should be saved.
    records : list
        A list of records (so a list of lists) holding the row data to
        be saved in the output SAV file.
    **kwargs : various
        Remaining keyword arguments passed to
        savReaderWriter.SavWriter().

    Returns
    -------
    None
    """
    with srw.SavWriter(path_sav, ioUtf8=True, **kwargs) as writer:
        records = data.fillna(writer.sysmis).values.tolist()
        for record in records:
            writer.writerow(record)


def split_series(series, sep, columns=None):
    """
    Splits all the items of a series using the given delimiter.

    Splits each item in series using the given delimiter and returns
    a DataFrame (as per Excel text-to-columns). Optionally, you can
    pass in a list of column names that should be used to name the
    resulting columns.

    Parameters
    ----------
    series : pandas.Series
        The series that should be split.
    sep : str
        The separator that should be used to split the series.
    columns : list-list, default=None
        A list of names that should be set into the resulting DataFrame
        columns.

    Returns
    -------
    df : pandas.DataFrame
        Series, split by sep, returned as a DataFrame.
    """

    df = pd.DataFrame(series.astype('str').str.split(sep).tolist())
    if not columns is None:
        df.columns = columns
    return df


def get_savwriter_integer_format(series):
    """
    Derive the required SAV format value for the given integer series.

    savReaderWriter requires specific instructions for each variable
    in order to correctly create target variables. This function
    determines the width of the maximum integer in the given series
    and constructs that format string accordingly.

    Parameters
    ----------
    series : pandas.Series
        The series for which the format string is being generated.

    Returns
    -------
    fmt : str
        The format string describing the given integer series.
    """

    fmt = 'F%s' % (
        len(str(series.dropna().astype('int').max()))
    )
    return fmt


def get_savwriter_float_format(series):
    """
    Derive the required SAV format value for the given float series.

    savReaderWriter requires specific instructions for each variable
    in order to correctly create target variables. This function
    determines the width of the maximum float in the given series
    and constructs that format string accordingly.

    Parameters
    ----------
    series : pandas.Series
        The series for which the format string is being generated.

    Returns
    -------
    fmt : str
        The format string describing the given float series.
    """

    if series.dropna().shape[0]==0:
        # If there's no data in the series it's impossible to predict
        # what sort of data may be expected later on, so a generic
        # interpretation of three whole numbers and two decimals is
        # applied.
        fmt = 'F5.2'
    else:
        df = split_series(
            series.dropna(),
            sep='.',
            columns=['int', 'dec']
        )
        w_int = len(str(df['int'].max()))
        w_dec = len(str(df['dec'].max()))
        if df['dec'].max()!='0':
            fmt = 'F%s.%s' %(
                w_int + w_dec,
                w_dec
            )
        else:
            fmt = 'F%s' %(
                w_int
            )
    return fmt


def get_value_text(values, value, text_key):
    """
    Get the text for the given value, using the given text_key.

    Values is a meta object in the form of a list of dicts. This
    function is used to find in that list the dict that has the given
    value, and from it return the text for that value using the given
    text_key.

    Parameters
    ----------
    values : list
        A Quantipy values meta object.
    value : int
        The value that should targeted in the list of values.
    text_key : str
        The key from which the the value text should be taken.

    Returns
    -------
    text : str
        The text associated with the targeted value and text_key.
    """

    for val in values:
        if val['value']==value:
            try:
                text = val['text'][text_key]
                return text
            except KeyError:
                print((
                    "The text key '%s' was not found in the text object: %s" % (
                        text_key,
                        json.dumps(val)
                    )
                ))
    raise ValueError(
            "The value '%s' was not found in the values object: %s" % (
                value,
                json.dumps(values)
            )
        )


def list_known_columns(meta, from_set):
    """
    Get an ordered list of columns from meta's from_set set.

    The from_set set may include items that point to masks. This
    function will replace mask-references with the names of the columns
    those masks point to so that what you get in return is an ordered
    list of column names that definitely exist in meta['columns'].

    .. note:: The meta document must have a set called from_set.

    Parameters
    ----------
    meta : dict
        The meta document.
    from_set : str
        The set name from which the export should be drawn.

    Returns
    -------
    column_names : list
        The column names.
    """

    column_names = []
    for col in meta['sets'][from_set ]['items']:
        pointer, name = col.split('@')
        if pointer=='columns':
            if name in meta['columns'] and not name in column_names:
                column_names.append(name)
        elif pointer=='masks':
            for item in meta['masks'][name]['items']:
                name =  item['source'].split('@')[1]
                if name in meta['columns'] and not name in column_names:
                    column_names.append(name)
    return column_names


def stringify_dates(dates):
    """
    Convert a datetime64 series to string in the form 'YYYY-M-D'.

    Parameters
    ----------
    dates : pandas.Series
        The numpy.datetime64 dates.

    Returns
    -------
    series : pandas.Series
        The string dates.
    """

    def stringify_date(date):

        try:
            date = ' '.join([
                '-'.join([
                    str(date.year),
                    str(date.month).zfill(2),
                    str(date.day).zfill(2)]),
                ':'.join([
                    str(date.hour).zfill(2),
                    str(date.minute).zfill(2),
                    str(date.second).zfill(2)])])
        except:
            pass

        return date

    series = dates.apply(stringify_date)
    series = series.astype('str')

    return series

def fix_label(label):
    label = label.replace('\n', '')
    return label

def save_sav(path_sav, meta, data, index=False, text_key=None,
             mrset_tag_style='__', drop_delimited=True, from_set=None,
             verbose=True):
    """
    One-sentence description.

    More detailed description.

    .. note:: Important note (if any).

    Parameters
    ----------
    path_sav : str
        The full path, including extension, indicating where the output
        file should be saved.
    meta : dict
        The meta document to be converted to SAV.
    data : pandas.DataFrame
        The data to be converted to SAV.
    index : bool, default=False
        Should the index be inserted into the dataframe before the
        conversion happens?
    text_key : str, default=None
        The text_key that should be used when taking labels from the
        source meta. If the given text_key is not found for any
        particular text object, the default text key (as indicated under
        meta['lib']['default text']) will be used instead.
    mrset_tag_style : str, default='__'
        The delimiting character/string to use when naming dichotomous
        set variables. The mrset_tag_style will appear between the
        name of the variable and the dichotomous variable's value name,
        as taken from the delimited set value that dichotomous
        variable represents.
    drop_delimited : bool, default=True
        Should Quantipy's delimited set variables be dropped from
        the export after being converted to dichotomous sets/mrsets?
    from_set : str
        The set name from which the export should be drawn.

    Returns
    -------
    None
    """
    # This function will make edits to the meta and data objects, so
    # they should be copied first.
    meta = copy.deepcopy(meta)
    data = data.copy()

    if from_set is None:
        from_set = 'data file'
    if from_set not in meta['sets']:
        raise KeyError(
            "The set '{}' was not found in meta.".format(from_set)
        )

    # There is an issue converting numpy dates to SAV so dates
    # are currently being turned into strings in the form 'Y-M-D'
    date_cols = [
        col for col in meta['columns']
        if meta['columns'][col]['type'] == 'date']
    for date_col in date_cols:
        data[date_col] = stringify_dates(data[date_col])
        meta['columns'][date_col]['type'] = 'string'

        # This code can be used to instead simply remove all
        # dates from the dataset before conversion
#         del meta['columns'][date_col]
#         mapper = 'columns@{}'.format(date_col)
#         try:
#             meta['sets'][from_set]['items'].remove(mapper)
#         except:
#             pass

    for key, val in meta['columns'].items():
        if val['type'] == 'string':
            if key in data.columns:
                data[key].fillna('', inplace=True)

    if index:
        if data.index.name not in data.columns:
            # Put the index into the first column of data
            data.insert(0, data.index.name, data.index)
        mapper = 'columns@%s' % (data.index.name)
        if mapper not in meta['sets'][from_set]['items']:
            # Add the index meta-mapper to the set
            meta['sets'][from_set]['items'].insert(0, mapper)

    if text_key is None:
        # Get default text key instead
        text_key = meta['lib']['default text']

    # Remove columns from data not found in meta
    known_columns = list_known_columns(meta, from_set)
    for col in data.columns:
        if col not in known_columns and col:
            if col != '@1':
                if verbose:
                    print((
                        "Data column '{}' not included in"
                        " the '{}' set, it will be excluded"
                        " from the SAV file."
                    ).format(col, from_set))
            data.drop(columns=col, axis=1, inplace=True)

    # Remove columns from meta not found in data
    for col in known_columns:
        if col not in data.columns:
            if verbose:
                print((
                    'Meta column "%s" not found in data, it will '
                    'be excluded from the SAV file.'
                ) % (col))
            if col in meta['columns']:
                del meta['columns'][col]

    # Create the varNames definition for the savWriter
    # Part of this process is to identify variable names that contain
    # illegal characters (as far as SPSS is concerned) and replace them
    # with an underscore ('_').
    varNames = list_known_columns(meta, from_set)
    new_names = []
    column_mapper = {}
    for varName in varNames:
        new_name = varName
        if varName[0] in [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]:
            new_name = '_%s' % (new_name)
        for i, char in enumerate(new_name):
            if char in ['[', ']', '{', '}', '.']:
                new_name = '%s_%s' % (
                    new_name[:i],
                    new_name[i+1:]
                )
        new_names.append(new_name)
        column_mapper[varName] = new_name
    for old_name, new_name in column_mapper.items():
        if new_name not in meta['columns']:
            meta['columns'][new_name] = meta['columns'].pop(old_name)
            idx = varNames.index(old_name)
            varNames = varNames[:idx] + [new_name] + varNames[idx+1:]
    data.columns = pd.Series(data.columns).map(column_mapper)

    # Create the multRespDefs definition for the savWriter
    delimited_sets = [
        c
        for c in varNames
        if meta['columns'][c]['type'] == 'delimited set'
    ]
    multRespDefs = {}
    for ds_name in delimited_sets:
        values = [
            val['value']
            for val in emulate_meta(meta, meta['columns'][ds_name]['values'])
        ]
        if data[ds_name].dtype == 'float':
            # The delimited set has no responses
            dichot = pd.DataFrame(np.NaN, index=data.index, columns=values)
        else:
            dichot = data[ds_name].str.get_dummies(sep=';')
            if '0' in dichot.columns and 0 not in values:
                # A '0' column here indicates rows that had no answers
                # in the delimited set, that column can be discarded.
                dichot.drop('0', axis=1, inplace=True)
            # Make sure all known values from the meta appear in the
            # dichotomous dataframe.
            for val in values:
                if val not in dichot.columns.astype('int'):
                    dichot[val] = 0
            no_responses = dichot.sum(axis=1) == 0
            dichot.loc[no_responses, :] = np.NaN

        dichot.columns = dichot.columns.astype(int)
        dichot.sort_index(axis=1, inplace=True)
        dsNames = ['%s%s%s' % (ds_name, mrset_tag_style, val) for val in values]
        ds_index = varNames.index(ds_name)
        varNames[ds_index+1:ds_index+1] = dsNames
        varNames.pop(ds_index)

        cols = [
            '%s%s%s' % (ds_name, mrset_tag_style, c) for c in dichot.columns]
        dichot.columns = cols
        # Synch dichotomous columns with varName order
        dichot = dichot[dsNames]
        # Find the position of the original delimited set in the source
        # dataframe's columns.
        ds_name_idx = data.columns.tolist().index(ds_name)
        # Insert the columns from the dichotomous dataframe after the
        # position of the delimited set.
        for i, col in enumerate(dichot.columns, start=1):
            data.insert(ds_name_idx+i, col, dichot[col])
        # Add the column metadata for each dichotomous column
        for dichName in dsNames:
            meta['columns'][dichName] = {
                'type': 'single',
                'values': [
                    {'value': 0, 'text': {text_key: 'No'}},
                    {'value': 1, 'text': {text_key: 'Yes'}}
                ],
                'text': {
                    text_key: get_value_text(
                        emulate_meta(meta, meta['columns'][ds_name]['values']),
                        int(dichName.split('_')[-1]),
                        text_key)
                }
            }

        # Add the savWriter-required definition of the mrset
        varLabel = fix_label(meta['columns'][ds_name]['text'].get(text_key, ''))
        if len(varLabel) > 120:
            varLabel = varLabel[:120]
        multRespDefs[ds_name] = {
            'varNames': dsNames,
            'label': qp.core.tools.dp.io.unicoder(varLabel, like_ascii=True),
            'countedValue': 1,
            'setType': 'D'
        }

        if drop_delimited:
            data.drop(ds_name, axis=1, inplace=True)

    # Create the varLabels definition for the savWriter
    varLabels = {
        v: fix_label(meta['columns'][v]['text'].get(text_key, ''))
        for v in varNames
    }

    for v in varLabels:
        if len(varLabels[v]) > 120:
            varLabels[v] = varLabels[v][:120]

    # Create the valueLabels definition for the savWriter
    # This will now catch all of the newly added dichotomous set columns
    singles = [v for v in varNames if meta['columns'][v]['type'] == 'single']
    valueLabels = {
        var: {
            int(val['value']): fix_label(val['text'].get(text_key, ''))
            for val in emulate_meta(meta, meta['columns'][var]['values'])
        }
        for var in singles
    }

    # Create the varTypes definition for the savWriter
    varTypes = {
        v:
        0
        if meta['columns'][v]['type'] in ['single', 'int', 'float']
        else 1000
        for v in varNames
    }

    # Create the formats definition for the savWriter
    numerics = [v for v, t in varTypes.items() if t == 0]
    strings = [
        v
        for v in list(varTypes.keys())
        if meta['columns'][v]['type'] in ['string', 'delimited set']
    ]
    dates = [
        v
        for v in list(varTypes.keys())
        if meta['columns'][v]['type'] in ['date']
    ]

    sav_formatter = {
        'single': get_savwriter_integer_format,
        'int': get_savwriter_integer_format,
        'float': get_savwriter_float_format
    }
    numeric_formats = {
        v: sav_formatter[meta['columns'][v]['type']](data[v])
        for v in numerics
    }
    string_formats = {
        s: 'A1000'
        for s in strings
    }
    date_formats = {
        d: 'EDATE40'
        for d in dates
    }
    formats = {}
    formats.update(numeric_formats)
    formats.update(string_formats)
    formats.update(date_formats)

    data = data[varNames]

    write_sav(
        path_sav,
        data,
        varNames=varNames,
        varTypes=varTypes,
        formats=formats,
        varLabels=varLabels,
        valueLabels=valueLabels,
        multRespDefs=multRespDefs
    )
