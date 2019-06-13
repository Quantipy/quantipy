import numpy as np
import pandas as pd
import savReaderWriter as sr

from collections import defaultdict
from quantipy.core.tools.dp.prep import start_meta, condense_dichotomous_set
import os

def parse_sav_file(filename, path=None, name="", ioLocale="en_US.UTF-8", ioUtf8=True, dichot=None,
                   dates_as_strings=False, text_key="en-GB"):
    """ Parses a .sav file and returns a touple of Data and Meta

        Parameters
        ----------
        filename : str, name of sav file
            path : str, the path to the sav file
            name : str, a name for the sav (stored in the meta)
        ioLocale : str, the locale that SavReaderWriter uses
          ioUtf8 : bool, Boolean that indicates the mode in which text
                         communicated to or from the I/O Module will be.
          dichot : dicit, default=None
                   The values to use for True/False in dichotomous sets
dates_as_strings : bool, default=False
                   If True then all dates from the input SAV will be treated as
                   Quantipy strings.
        text_key : str, default="main"
                   The text_key that all labels should be stored under.

        Returns
        -------
        (data, meta) : The data is a Pandas Dataframe
                     : The meta is a JSON dictionary
    """
    filepath="{}{}".format(path or '', filename)
    filepath = os.path.abspath(filepath)
    data = extract_sav_data(filepath, ioLocale=ioLocale, ioUtf8=ioUtf8)
    meta, data = extract_sav_meta(filepath, name="", data=data, ioLocale=ioLocale,
                                  ioUtf8=ioUtf8, dichot=dichot, dates_as_strings=dates_as_strings,
                                  text_key=text_key)
    return (meta, data)

def extract_sav_data(sav_file, ioLocale='en_US.UTF-8', ioUtf8=True):
    """ see parse_sav_file doc """
    with sr.SavReader(sav_file, returnHeader=True, ioLocale=ioLocale, ioUtf8=ioUtf8) as reader:
        thedata = [x for x in reader]
        header = thedata[0]
        dataframe = pd.DataFrame.from_records(thedata[1:], coerce_float=False)
        dataframe.columns = header
        for column in header:
            if isinstance(dataframe[column].dtype, np.object):
                # Replace None with NaN because SRW returns None if casting dates fails (dates are of type np.object))
                values = dataframe[column].dropna().values
                if len(values) > 0:
                    if isinstance(values[0], str):
                        dataframe[column] = dataframe[column].dropna().map(str.strip)
                    elif isinstance(values[0], str):
                        # savReaderWriter casts dates to str
                        dataframe[column] = dataframe[column].dropna().map(str.strip)
                        # creating DATETIME objects should happen here
        return dataframe

def extract_sav_meta(sav_file, name="", data=None, ioLocale='en_US.UTF-8',
                     ioUtf8=True, dichot=None, dates_as_strings=False,
                     text_key="en-GB"):

    if dichot is None: dichot = {'yes': 1, 'no': 0}

    """ see parse_sav_file doc """
    with sr.SavHeaderReader(sav_file, ioLocale=ioLocale, ioUtf8=ioUtf8) as header:
        # Metadata Attributes
        # ['valueLabels', 'varTypes', 'varSets', 'varAttributes', 'varRoles',
        #  'measureLevels', 'caseWeightVar', 'varNames', 'varLabels', 'formats',
        #  'multRespDefs', 'columnWidths', 'fileAttributes', 'alignments',
        #  'fileLabel', 'missingValues']
        metadata = header.dataDictionary(True)

    meta = start_meta(text_key=text_key)
    meta['info']['text'] = 'Converted from SAV file {}.'.format(name)
    meta['info']['from_source'] = {'pandas_reader':'sav'}
    meta['sets']['data file']['items'] = [
        'columns@{}'.format(varName)
        for varName in metadata.varNames]

    # This should probably be somewhere in the metadata
    # weight_variable_name = metadata.caseWeightVar

    # Descriptions of attributes in metadata are are located here :
    # http://pythonhosted.org/savReaderWriter/#savwriter-write-spss-system-files
    for column in metadata.varNames:
        meta['columns'][column] = {}
        meta['columns'][column]['name'] = column
        meta['columns'][column]['parent'] = {}
        if column in metadata.valueLabels:
            # ValueLabels is type = 'single' (possibry 1-1 map)
            meta['columns'][column]['values'] = []
            meta['columns'][column]['type'] = "single"
            for value, text in metadata.valueLabels[column].items():
                values = {'text': {text_key: str(text)},
                          'value': int(value)}
                meta['columns'][column]['values'].append(values)
        else:
            if column in metadata.formats:
                f = metadata.formats[column]
                if 'DATETIME' in f:
                    if dates_as_strings:
                        # DATETIME fields from SPSS are currently
                        # being read in as strings because there's an
                        # as-yet undetermined discrepancy between the
                        # input and output dates if datetime64 is used
                        meta['columns'][column]['type'] = 'string'
                    else:
                        meta['columns'][column]['type'] = 'date'
                        data[column] = pd.to_datetime(data[column])
                elif f.startswith('A'):
                    meta['columns'][column]['type'] = 'string'
                elif '.' in f:
                    meta['columns'][column]['type'] = "float"
                else:
                    meta['columns'][column]['type'] = "int"
            else:
                # Infer meta from data
                if data is not None:
                    # print "VAR '{}' NOT IN value_labels".format(column)
                    column_values = data[column].dropna()
                    if len(column_values) > 0:
                        # Get the first "not nan" value from the column
                        value = column_values.values[0]
                        if isinstance(value, pd.np.float64):
                            # Float AND Int because savReaderWriter loads them both as float64
                            meta['columns'][column]['text'] = {text_key: [column]}
                            meta['columns'][column]['type'] = "float"
                            if (data[column].dropna() % 1).sum() == 0:
                                if (data[column].dropna() % 1).unique() == [0]:
                                    try:
                                        data[column] = data[column].astype('int')
                                    except:
                                        pass
                                    meta['columns'][column]['type'] = "int"

                        elif isinstance(value, str) or isinstance(value, str):
                            # Strings
                            meta['columns'][column]['text'] = {text_key: [column]}
                            meta['columns'][column]['type'] = "string"

        if column in metadata.varTypes:
            pass

        if column in metadata.varSets:
            pass

        if column in metadata.varAttributes:
            pass

        if column in metadata.varRoles:
            pass

        if column in metadata.measureLevels:
            pass

        # Some labels are empty strings.
        if column in metadata.varLabels:
            meta['columns'][column]['text'] = {text_key: metadata.varLabels[column]}

    for mrset in metadata.multRespDefs:
        # meta['masks'][mrset] = {}
        # 'D' is "multiple dichotomy sets" in SPSS
        # 'C' is "multiple category sets" in SPSS
        varNames = metadata.multRespDefs[mrset]['varNames']
        # Find the index where there delimited set should be inserted
        # into data, which is immediately prior to the start of the
        # dichotomous set columns
        dls_idx = data.columns.tolist().index(varNames[0])
        if metadata.multRespDefs[mrset]['setType'] == 'C':
            # Raise if value object of columns is not equal
            if not all(meta['columns'][v]['values'] == meta['columns'][varNames[0]]['values']
                       for v in varNames):
                msg = 'Columns must have equal values to be combined in a set: {}'
                raise ValueError(msg.format(varNames))
            # Concatenate columns to set
            df_str = data[varNames].astype('str')
            dls = df_str.apply(lambda x: ';'.join([
                v.replace('.0', '') for v in x.tolist()
                if not v in ['nan', 'None']]),
                axis=1) + ';'
            dls.replace({';': np.NaN}, inplace=True)
            # Get value object
            values = meta['columns'][varNames[0]]['values']

        elif metadata.multRespDefs[mrset]['setType'] == 'D':
            # Generate the delimited set from the dichotomous set
            dls = condense_dichotomous_set(data[varNames], values_from_labels=False, **dichot)
            # Get value object
            values = [{
                        'text': {text_key: metadata.varLabels[varName]},
                        'value': int(v)
                    }
                    for v, varName in enumerate(varNames, start=1)]
        else:
            continue
        # Insert the delimited set into data
        data.insert(dls_idx, mrset, dls)
        # Generate the column meta for the new delimited set
        meta['columns'][mrset] = {
            'name': mrset,
            'type': 'delimited set',
            'text': {text_key: metadata.multRespDefs[mrset]['label']},
            'parent': {},
            'values': values}
        # Add the new delimited set to the 'data file' set
        df_items = meta['sets']['data file']['items']
        df_items.insert(
            df_items.index('columns@{}'.format(varNames[0])),
            'columns@{}'.format(mrset))

        data = data.drop(varNames, axis=1)
        for varName in varNames:
            df_items.remove('columns@{}'.format(varName))
            del meta['columns'][varName]

    return meta, data
