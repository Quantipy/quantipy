
import pandas as pd
import json
import xmltodict
import warnings
from quantipy.core.tools.dp.prep import start_meta, condense_dichotomous_set


def quantipy_from_ascribe(path_xml, path_txt, text_key='main'):

    # Read the AScribe data (tab-delimited)
    meta_ascribe = xmltodict.parse(open(path_xml))
    data_ascribe = pd.DataFrame.from_csv(
        path_txt,
        sep='\t',
        header=0,
        encoding='utf-16'
    )
    data_ascribe[data_ascribe.index.name] = data_ascribe.index

    # Start a Quantipy meta document
    meta = start_meta(text_key=text_key)
    meta['columns']['responseid'] = {
        'name': 'responseid',
        'type': 'int',
        'text': {text_key: 'responseid'},
        'parent': {}
    }
    meta['sets']['data file']['items'] = ['columns@responseid']

    MultiForm = meta_ascribe['CodedQuestions']['MultiForm']
    if not isinstance(MultiForm, list):
        meta_ascribe['CodedQuestions']['MultiForm'] = [MultiForm]

    # Container to record the names, in order, of the resulting
    # coded columns
    coded_names = []

    for var in meta_ascribe['CodedQuestions']['MultiForm']:
        name = var['Name']
        if var['Answers'] is None:
            msg = ("The variable '%s' has no answer codes "
                   "and will be skipped.") % (name)
            warnings.warn(msg)
            continue
        coded_names.append(name)
        coded_from = var['FormTexts']['FormText']['Title']
        var_text = var['FormTexts']['FormText']['Text']
        var_text = '' if var_text is None else var_text.replace('\n', ' ')
        if var_text is None: var_text = 'Label not provided'
        var_text = {text_key: var_text}
        columns = []
        values = []
        for val in var['Answers']['Answer']:
            value = int(val['@Precode'])
            if value==0:
                msg = (
                    "The value 0 has been assigned to a code for the "
                    "variable '%s'."
                ) % (name)
                warnings.warn(msg)
            val_text = val['Texts']['Text']['#text']
            val_text = '' if val_text is None else val_text.replace('\n', ' ')
            if val_text is None: val_text = 'Label not provided'
            val_text = {text_key: val_text}
            values.append({'value': value, 'text': val_text})
            columns.append('%s_%s' % (name, value))

        # Create a single series from the dichotomous set
        data_ascribe[name] = condense_dichotomous_set(
            data_ascribe[columns],
            sniff_single=True
        )

        # Determine the Quantipy type of the returned
        # series from its dtype (see 'sniff_sinlge' in
        # condense_dichotomous_set()
        if data_ascribe[columns].sum(axis=1).max()==1:
            col_type = 'single'
        else:
            col_type = 'delimited set'

        # Create the new Quantipy column meta
        column = {
            'name': name,
            'type': col_type,
            'text': var_text,
            'values': values,
            'parent': {}
        }

        # Add the newly defined column to the Quantipy meta
        meta['columns'][name] = column

    meta['sets']['data file']['items'].extend([
        'columns@%s' % (col_name)
        for col_name in coded_names
    ])

    # Keep only the slice that has been converted.
    data = data_ascribe[[data_ascribe.index.name]+coded_names]

    return meta, data
