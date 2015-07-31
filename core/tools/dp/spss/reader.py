import numpy as np
import pandas as pd
import savReaderWriter as sr

from collections import defaultdict
from quantipy.core.tools.dp.prep import start_meta, condense_dichotomous_set

def parse_sav_file(filename, path=None, name="", ioLocale="en_US.UTF-8", ioUtf8=True):
    """ Parses a .sav file and returns a touple of Data and Meta

        Parameters
        ----------
        filename : str, name of sav file
            path : str, the path to the sav file
            name : str, a name for the sav (stored in the meta)
        ioLocale : str, the locale that SavReaderWriter uses
          ioUtf8 : bool, Boolean that indicates the mode in which text
                         communicated to or from the I/O Module will be.

        Returns
        -------
        (data, meta) : The data is a Pandas Dataframe
                     : The meta is a JSON dictionary
    """
    filepath="{}{}".format(path or '', filename)
    data = extract_sav_data(filepath, ioLocale=ioLocale, ioUtf8=ioUtf8)
    meta, data = extract_sav_meta(filepath, name="", data=data, ioLocale=ioLocale, ioUtf8=ioUtf8)
    return (meta, data)

def extract_sav_data(sav_file, ioLocale='en_US.UTF-8', ioUtf8=True):
    """ see parse_sav_file doc """
    with sr.SavReader(sav_file, returnHeader=True, ioLocale=ioLocale, ioUtf8=ioUtf8) as reader:
        header = next(reader)
        dataframe = pd.DataFrame.from_records(reader, coerce_float=False)
        dataframe.columns = header
        for column in header:
            if isinstance(dataframe[column].dtype, np.object):
                # Replace None with NaN because SRW returns None if casting dates fails (dates are of type np.object))
                values = dataframe[column].dropna().values
                if len(values) > 0:
                    if isinstance(values[0], unicode):
                        dataframe[column] = dataframe[column].dropna().map(unicode.strip)
                    elif isinstance(values[0], str):
                        # savReaderWriter casts dates to str
                        dataframe[column] = dataframe[column].dropna().map(str.strip)
                        # creating DATETIME objects should happen here
        return dataframe

def extract_sav_meta(sav_file, name="", data=None, ioLocale='en_US.UTF-8', ioUtf8=True):
    """ see parse_sav_file doc """
    with sr.SavHeaderReader(sav_file, ioLocale=ioLocale, ioUtf8=ioUtf8) as header:
        # Metadata Attributes
        # ['valueLabels', 'varTypes', 'varSets', 'varAttributes', 'varRoles',
        #  'measureLevels', 'caseWeightVar', 'varNames', 'varLabels', 'formats',
        #  'multRespDefs', 'columnWidths', 'fileAttributes', 'alignments',
        #  'fileLabel', 'missingValues']
        metadata = header.dataDictionary(True)

    meta = start_meta(name=name)
    meta['info']['text'] = 'Converted from SAV file %s.' % (name)
    meta['info']['from_source'] = {'pandas_reader':'sav'}
    meta['sets']['data file']['items'] = [
        'columns@%s' % (varName)
        for varName in metadata.varNames
    ]

    # This should probably be somewhere in the metadata
    # weight_variable_name = metadata.caseWeightVar

    # Descriptions of attributes in metadata are are located here :
    # http://pythonhosted.org/savReaderWriter/#savwriter-write-spss-system-files
    for column in metadata.varNames:
        meta['columns'][column] = {}

        if column in metadata.valueLabels:
            # ValueLabels is type = 'single' (possibry 1-1 map)
            meta['columns'][column]['values'] = []
            meta['columns'][column]['type'] = "single"
            for value, text in metadata.valueLabels[column].iteritems():
                values = {'text': {'main': unicode(text)},
                          'value': unicode(int(value))}
                meta['columns'][column]['values'].append(values)
        else:
            if column in metadata.formats:
                f = metadata.formats[column]
                if '.' in f:
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
                            meta['columns'][column]['text'] = {'main': [column]}
                            meta['columns'][column]['type'] = "float"
                            if (data[column].dropna() % 1).sum() == 0:
                                if (data[column].dropna() % 1).unique() == [0]:
                                    try:
                                        data[column] = data[column].astype('int')
                                    except:
                                        pass
                                    meta['columns'][column]['type'] = "int"

                        elif isinstance(value, unicode) or isinstance(value, str):
                            # Strings
                            meta['columns'][column]['text'] = {'main': [column]}
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
            meta['columns'][column]['text'] = {'main': metadata.varLabels[column]}

    for mrset in metadata.multRespDefs:
        # meta['masks'][mrset] = {}
        # 'D' is "multiple dichotomy sets" in SPSS
        # 'C' is "multiple category sets" in SPSS
        if metadata.multRespDefs[mrset]['setType'] == 'C':
            'C'
#             meta['masks'][mrset]['type'] = "categorical set"
        elif metadata.multRespDefs[mrset]['setType'] == 'D':
            'D'
#             meta['masks'][mrset]['type'] = "dichotomous set"
#             meta['masks'][mrset]['countedValue'] = metadata.multRespDefs[mrset]['countedValue']
            varNames = metadata.multRespDefs[mrset]['varNames']
#             meta, data[mrset] = delimited_from_dichotomous(meta, data[varNames], mrset)
            data[mrset] = condense_dichotomous_set(data[varNames], values_from_labels=False)
            meta['columns'][mrset] = {
                'type': 'delimited set',
                'text': {'main': metadata.multRespDefs[mrset]['label']},
                'values': [
                    {
                        'text': {'main': metadata.varLabels[varName]},
                        'value': v
                    }
                    for v, varName in enumerate(varNames, start=1)
                ]
            }
            idx = meta['sets']['data file']['items'].index('columns@%s' % (varNames[0]))
            items = meta['sets']['data file']['items']
            meta['sets']['data file']['items'] = items[:idx] + ['columns@%s' % (mrset)] + items[idx+len(varNames):]

            data = data.drop(varNames, axis=1)
            for varName in varNames:                
                del meta['columns'][varName]

#         meta['masks'][mrset]['text'] = [metadata.multRespDefs[mrset]['label']]
#         meta['masks'][mrset]['items'] = []
#         for var_name in metadata.multRespDefs[mrset]['varNames']:
#             meta['masks'][mrset]['items'].append({'source':"columns@{0}".format(var_name)})

        # df = make_delimited_from_dichotmous(data[common_vars[var]])

    return meta, data

