
import numpy as np
import pandas as pd
import json
import copy
import re
import itertools
from quantipy.core.helpers.functions import load_json, get_mapped_meta
from quantipy.core.tools.dp.prep import start_meta

    
def manage_decipher_quota_variables(meta, data, quotas):
    """ Uses the collected dichotomous quota variables found in the 
    decipher metadata and combines them into single columns based on
    their specified variable group (as defined by Decipher).

    Parameters
    ----------
    meta : dict
        The meta document paired to the data being converted

    data : pandas.DataFrame
        The data being converted

    quotas: dict
        The dict containing details of the quota types being managed

    Returns
    -------
    meta : dict
        The meta document paired to the data being converted

    data : pandas.DataFrame
        The data being converted
    """

    quota_types = ['Quotas', 'Over-quotas']
    text_key = meta['lib']['default text']

    for qt, qtable in enumerate(['vqtable', 'voqtable']):
        if quotas[qtable]:
            for key, item in sorted(quotas[qtable].items()):
                
                # Create column meta
                meta['columns'][key] = {
                    'type': 'delimited set',
                    'text': {text_key: '%s: %s' % (quota_types[qt], key)},
                    'values': [
                        {'value': int(q), 'text': {text_key: quota['title']}}
                        for q, quota in enumerate(item, start=1)
                    ]       
                }
                cols = [i['label'] for i in item]
                # Convert dichotomous to delimited
                meta, data[key] = delimited_from_dichotomous(
                    meta, data[cols], key
                )            
                # Remove dichotomous columns from meta
                for col in cols: del meta['columns'][col]                         
                # Remove dichotomous columns from data
                data.drop(cols, axis=1, inplace=True)
        
        meta['sets'][quota_types[qt]] = {
            'text': {text_key: '%s columns' % (quota_types[qt])},
            'items': [
                col_mapper
                for col_mapper in meta['sets']['data file']['items']
                if qtable in col_mapper
            ]
        }

    return meta, data


def get_decipher_values(values, text_key):
    """ Converts a Decipher values collection into a Quantipy values
    collection.

    Parameters
    ----------
    values : list
        The values object from a block of Decipher metadata

    Returns
    -------
    values : list
        The Quantipy values object 
    """

    values = [
        {
            'value': int(v['value']),
            'text': {text_key: v['title']}
        }
        for v in values
    ]

    return values


def get_vgroups(variables):
    """ Returns a list of unique vgroup found in the given variables
    object.

    Parameters
    ----------
    variables : list
        Decipher variables object, which is a list of dictionaries

    Returns
    -------
    vgroups : list
        List of unique vgroups found in variables
    """


    vgroups = []  
    for var in variables:
        if not var['vgroup'] in vgroups:
            vgroups.append(var['vgroup'])
    
    return vgroups
    
    
def get_vgroup_variables(vgroups, variables):
    """ Returns a list of variable lists belonging to each vgroup
    provided in vgroups.

    Parameters
    ----------
    vgroups : list
        The vgroups for which lists of variable objects should be 
        returned

    variables : list
        Decipher variables object, which is a list of dictionaries

    Returns
    -------
    vgroup_variables : list
        List of lists of variable objects, positionally matched to their
        vgroup in the vgroups list
    """
    
    vgroup_variables = [
        [
            var
            for var in variables
            if vgroup == var['vgroup']
        ]
        for vgroup in vgroups
    ]
    
    return vgroup_variables


def get_vgroup_types(vgroups, variables):
    """ Returns a list of type names belonging to each vgroup provided
    in vgroups.

    Parameters
    ----------
    vgroups : list
        The vgroups for which lists of variable objects should be 
        returned

    variables : list
        Decipher variables object, which is a list of dictionaries

    Returns
    -------
    vgroup_types : list
        List of type names, positionally matched to their vgroup in the
        vgroups list
    """

    vgroup_types = {}
    for vgroup in vgroups:
        vgroup_types[vgroup] = [
            var['type']
            for var in variables
            if vgroup == var['vgroup']
        ]
        if all([
            t == vgroup_types[vgroup][0]
            for t in vgroup_types[vgroup]
        ]):
            vgroup_types[vgroup] = vgroup_types[vgroup][0]
        else:
            print("vgroup '%s' has mixed types: %s" % (
                vgroup, vgroup_types[vgroup]
            ))
    
    return vgroup_types


def delimited_from_dichotomous(meta, df, name, sniff_single=False):
    """ Takes df, which should contain one or more columns of 
    dichotomous data (as 0s/1s) related to the same set of response
    options, and returns a single series. The returned series will be a
    delimited set if necessary, but if there is only 1 column in df or
    the responses indicated in the data are mutually exclusive then a
    normal 'single' series will be returned instead and the meta type
    for that column will be adjusted to single. 

    Parameters
    ----------
    meta : dict
        The meta document paired to the data being converted

    df : pandas.DataFrame
        The column/s in the dichotomous set. This may be a single-column
        DataFrame, in which case a non-delimited set will be returned.

    name : str
        The relevant key name for the resulting column in meta['columns']

    Returns
    -------
    meta : dict
        The meta document paired to the data being converted

    series: pandas.series
        The converted series
    """
    
    if sniff_single and df.shape[1]==1:
        # The set has only 1 possible response
        # Convert to single
        series = df.iloc[:,0].replace(0, np.NaN)
        # Replace data file set item
        old_set_item = 'columns@{}'.format(series.name)
        new_set_item = 'columns@{}'.format(name)
        idx = meta['sets']['data file']['items'].index(old_set_item)
        meta['sets']['data file']['items'].insert(idx, new_set_item)
        meta['sets']['data file']['items'].remove(old_set_item)
        # Rename the series
        series.name = name
        # Update type in meta
        meta['columns'][name]['type'] = 'single'
        return meta, series
    
    elif sniff_single and all([v<=1 for v in df.sum(axis=1)]):
        # The set values are mutually exclusive  
        # Convert to single
        df = df.copy()
        # Replace data file set item
        old_set_item = 'columns@{}'.format(df.columns[0])
        new_set_item = 'columns@{}'.format(name)
        idx = meta['sets']['data file']['items'].index(old_set_item)
        meta['sets']['data file']['items'].insert(idx, new_set_item)
        for col in df.columns:
            old_set_item = 'columns@{}'.format(col)
            meta['sets']['data file']['items'].remove(old_set_item)
        # Transform the dataframe
        for v, col in enumerate(df.columns, start=1):
            # Convert to categorical set
            df[v] = df[col].replace(1, v)
            del df[col]
        series = df.sum(axis=1).replace(0, np.NaN)
        # Rename the series
        series.name = name
        # Update type in meta
        meta['columns'][name]['type'] = 'single'
        return meta, series
    
    else:
        series = condense_dichotomous_set(
            df, values_from_labels=True, values_regex='^.+r([0-9]+)[c0-9]*$')
        series.name = name
        # Replace data file set item
        old_set_item = 'columns@{}'.format(df.columns[0])
        new_set_item = 'columns@{}'.format(name)
        idx = meta['sets']['data file']['items'].index(old_set_item)
        meta['sets']['data file']['items'].insert(idx, new_set_item)
        for col in df.columns:
            old_set_item = 'columns@{}'.format(col)
            meta['sets']['data file']['items'].remove(old_set_item)
        
        return meta, series


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

    # Convert to delimited set
    df_str = df.astype('str')
    for v, col in enumerate(df_str.columns, start=1):
        if values_from_labels:
            if values_regex is None:
                val = col.split('_')[-1]
            else:
                
                try:
                    val = get_delimited_value(col, v, as_str=True)
                except AttributeError:
                    raise AttributeError(
                        "Your values_regex may have failed to find a match"
                        " using re.match('{}', '{}')".format(
                            values_regex, col))
        else:
            val = str(val)
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
                '{}'.format(yes): val, 
                '{}.0'.format(yes): val
            }, 
            inplace=True
        )
    # Concatenate the rows
    series = df_str.apply(
        lambda x: ';'.join([
            val 
            for val in x.tolist() 
            if val != 'nan'
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


def get_delimited_value(raw_value, i, as_str=False):
    
    matches = re.match('^.+r([0-9]*).*$', raw_value)
    if not matches is None:
        val = matches.groups()[0]
        if not as_str:
            val = int(val)
    else:
        if raw_value.endswith('none'):
            val = 999
        elif raw_value.endswith('other'):
            val = 995
        else:
            val = i
    
    if as_str:
        val = str(val)
        
    return val


def get_delimited_values(raw_values, as_str=False):
    
    values = []
    for i, val in enumerate(raw_values, start=1):
        v = get_delimited_value(val, i, as_str)
        if v in values:
            raise ValueError(
                "The conversion of the multiple-choice"
                " variable '{}' must be partly inferred"
                " due to non-numeric category values"
                " in the Decipher metadata. In this case"
                " the value '{}' has been asked for"
                " twice. To prevent silent errors in the"
                " conversion this error has been raised.")
        else:
            values.append(v)
    
    return values
    
    
def make_delimited_set(meta, data, question):
    """ Given the question object, converts the Decipher multiple type
    from dichotomous to delimited, updating the meta document
    accordingly.

    Parameters
    ----------
    meta : dict
        The Quantipy meta document paired to data

    data : pandas.DataFrame
        The Quantipy data paired to meta

    question : dict
        Decipher question meta object 

    Returns
    -------
    meta : dict
        The Quantipy meta document paired to data

    data : pandas.DataFrame
        The Quantipy data paired to meta

    vgroups : list
        List of unique vgroups found in variables

    vgroup_variables : list
        List of lists of variable objects, positionally matched to their
        vgroup in the vgroups list
    """
    
    text_key = meta['lib']['default text']

    # Find the number of variable groups in the set
    vgroups = get_vgroups(question['variables'])
    
    # Determine if the question is a compound multi
    # in which case special rules apply because the
    # vgroups are organized around the grid structure
    # rather than the multiple choice structure
#     compound_multi = not re.match('^.+([r|c][0-9]+)$', vgroups[0]) is None
    rowwise_compound_multi = not re.match('^.+(r[0-9]+)$', vgroups[0]) is None
    colwise_compound_multi = not re.match('^.+(c[0-9]+)$', vgroups[0]) is None
    
    # Get the variable type for each vgroup
    vgroup_types = get_vgroup_types(vgroups, question['variables'])
    
    # For each variable group, get its members
    vgroup_variables = get_vgroup_variables(vgroups, question['variables'])
    
    if rowwise_compound_multi:
        # This is a multi-choice array that is stored in single-row grids
        
        # Find the row values (excluding any open-ended rows in the group)
        rs = [
            re.match('^.+(r[0-9]+).*$', g).groups()[0] 
            for g in vgroups 
            if not g.endswith('oe')]
        # Find the column values
        cs = [
            re.match('^.+(c[0-9]+)$', v['label']).groups()[0] 
            for v in vgroup_variables[0]]
        
        # Find the grid, row and column labels
        rowTitles = [vgv[0]['rowTitle'] for vgv in vgroup_variables]
        colTitles = [v['colTitle'] for v in vgroup_variables[0]]
        qtitle = vgroup_variables[0][0]['qtitle']
        # Find the broad name of the entire group
        qname = vgroups[0][:len(rs[0])*-1]
        
        # Arrange the data columns that make up this broad group
        cols = ['{}{}'.format(qname, ''.join(i[::-1])) for i in itertools.product(cs, rs)]
        cgroups = {c: [col for col in cols if col.endswith(c)] for c in cs}
        # Generate a values object for the array
        raw_values = [r.split('r')[-1] for r in rs]
        ds_values = get_delimited_values(raw_values, as_str=False)
        values = [
            {'value': val, 'text': {text_key: rowTitle}} 
            for val, rowTitle in zip(ds_values, rowTitles)]
        meta['lib']['values'][qname] = values
        
        # Generate the new column meta (since this data
        # is originally stored as single-response dichotomous
        # grids they need to be reconstructed to columns
        # of delimited sets).
        values_mapper = 'lib@values@{}'.format(qname)
        for c, colTitle in zip(cs, colTitles):
            cgroup_cols = cgroups[c]
            col_name = '{}{}'.format(qname, c)
            meta['columns'][col_name] = {
                'type': 'delimited set',
                'text': {
                    text_key: '%s - %s' % (
                        colTitle, 
                        qtitle
                    )
                },
                'values': values_mapper
            }
            # Convert dichotomous to delimited
            meta, data[col_name] = delimited_from_dichotomous(
                meta, data[cgroup_cols], col_name
            )
            
        # Create the array mask
        mask = meta['masks'][qname] = {
            'type': 'array',
            'item type': 'delimited set',
            'text': {text_key: qtitle},
            'items': [{
                'source': 'columns@{}'.format(
                    '{}{}'.format(qname, c)),
                'text': {text_key: colTitles[i]}}
                for i, c in enumerate(cs)],
            'values': values_mapper}
        
        # Remove dichotomous columns from meta
        for col in cols: del meta['columns'][col]            
        # Remove dichotomous columns from data
        data.drop(cols, axis=1, inplace=True)
    
    else:
        # Create a delimited set from each variable group
        for vgroup, vars in zip(vgroups, vgroup_variables):
            
            if not vgroup_types[vgroup] == 'multiple':
                # It's possible the question is a combination of multiple
                # and another type, in which case the non-multiple
                # parts of it need to be left as they are
                continue
            
            rmatches = re.match('^.+(r[0-9]+)$', vgroup)
            cmatches = re.match('^.+(c[0-9]+)$', vgroup)
            
            if not rmatches is None:
                # This should never happen, because it is a
                # compound multi which has been dealt with above
                raise TypeError(
                    "Unexpected compound multi found: {}.".format(vgroup))
                
            else:
    #             print 'type 3', vgroup
                raw_values = [var['label'] for var in vars]
                ds_values = get_delimited_values(raw_values, as_str=False)
                values = [
                    {
                        'value': val, 
                        'text': {text_key: var['rowTitle']}} 
                    for var, val in zip(vars, ds_values)]
            
            # Create column meta
            meta['columns'][vgroup] = {
                'type': 'delimited set',
                'text': {
                    text_key: '%s - %s' % (
                        vars[0]['colTitle'], 
                        vars[0]['qtitle']
                    )
                },
                'values': values
            }
            
            # Get dichotomous set column names
            # Ignore non-'multiple'-type parts
            # of the set (Decipher Question
            # objects can include text variables
            # for compound-type Questions, which
            # are similar to a Quantipy set)
            cols = [
                v['label'] 
                for v in vars 
                if v['type']=='multiple'
            ]
            # Convert dichotomous to delimited
            meta, data[vgroup] = delimited_from_dichotomous(
                meta, data[cols], vgroup
            )
            # Remove dichotomous columns from meta
            for col in cols: del meta['columns'][col]            
            # Remove dichotomous columns from data
            data.drop(cols, axis=1, inplace=True)
        
        if colwise_compound_multi:
            # This is a multi-choice array that is stored in single-column grids
        
            # Find the row values (excluding any open-ended rows in the group)
            rs = [
                re.match('^.+(r[0-9]+).*$', v['label']).groups()[0] 
                for v in vgroup_variables[0]
                if not v['label'].endswith('oe')]
            # Find the column values
            cs = [
                re.match('^.+(c[0-9]+)$', g).groups()[0]
                for g in vgroups]
            
            # Find the grid, row and column labels
            rowTitles = [v['rowTitle'] for v in vgroup_variables[0]]
            colTitles = [vgv[0]['colTitle'] for vgv in vgroup_variables]
            qtitle = vgroup_variables[0][0]['qtitle']            
            # Find the broad name of the entire group
            qname = vgroups[0][:len(cs[0])*-1]
        
            # Create the array mask
            values_mapper = 'lib@values@{}'.format(qname)
            mask = meta['masks'][qname] = {
                'type': 'array',
                'item type': 'delimited set',
                'text': {text_key: qtitle},
                'items': [{
                    'source': 'columns@{}'.format(
                        '{}{}'.format(qname, c)),
                    'text': {text_key: colTitles[i]}}
                    for i, c in enumerate(cs)],
                'values': values_mapper}
            
            values = meta['columns']['{}{}'.format(qname, cs[0])]['values'][:]
            meta['lib']['values'][qname] = values
            for c in cs:
                col = '{}{}'.format(qname, c)
                meta['columns'][col]['values'] = values_mapper

    return meta, data, vgroups, vgroup_variables

                        
def quantipy_from_decipher(decipher_meta, decipher_data, text_key='main'): 
    """ Converts the given Decipher data (which must have been exported
    in tab-delimited format) to Quantipy-ready meta and data.
    
    Parameters
    ----------
    decipher_meta : str or dict
        Either the path to the Decipher meta document saved as JSON or
        said document read into memory

    decipher_data : str or pandas.DataFrame
        Either the path to the Decipher data saved as tab-delimited text
        said file read into memory

    Returns
    -------
    meta : dict
        The Quantipy meta document

    data : pandas.DataFrame
        The converted data
    """

    # If they're not already in memory, read in the Decipher meta and
    # data files
    if isinstance(decipher_meta, str):
        dmeta = load_json(decipher_meta)
    if isinstance(decipher_data, str):
        data = pd.DataFrame.from_csv(decipher_data, sep='\t')
        data[data.index.name] = data.index

    meta = start_meta(text_key=text_key)

    quotas = {
        'vqtable': {}, 
        'voqtable': {}
    }

    types_map = {
        'text': 'string',
        'number': 'int',
        'float': 'float',
        'single': 'single',
        'multiple': 'delimited set'
    }

    # Create generator for compound questions
    compound_questions = [
        question 
        for question in dmeta['questions'] 
        if len(question['variables']) > 1]
    
    # Get basic variables
    for var in dmeta['variables']:
        
        # Collect quota variables
        # These will be dealt with later
        for qtable in ['vqtable', 'voqtable']:
            if qtable in var['vgroup']:
                if not var['vgroup'] in quotas[qtable]:
                    quotas[qtable][var['vgroup']] = []
                quotas[qtable][var['vgroup']].append(var)
                continue
        
        # Start the column meta for the current variable
        var_name = var['label']
        column = meta['columns'][var_name] = {
            'type': types_map[var['type']],
            'text': {text_key: var['title']}
        }
        
        # Add meta-mapped path for current column to the 'data file' set
        # object so that the original order of the variables is known
        set_item = 'columns@%s' % (var_name)
        if not set_item in meta['sets']['data file']['items']:
            meta['sets']['data file']['items'].append(set_item)
        
        if var['type']=='single':
            # Get the response values
            column['values'] = get_decipher_values(var['values'], text_key)

    # Manage compound variables (delimited sets, arrays, mixed-type 
    # sets)
    for question in compound_questions:

        if question['type']=='multiple':

            # Construct delimited set
            meta, data, vgroups, vgroup_variables = make_delimited_set(
                meta, data, question
            )
            
            # If there's only 1 vgroup then this is a basic multiple-
            # choice question and doesn't require construction as an
            # array or set
            if len(vgroups)==1:
                continue

        else:
            # vgroups indicate how many groups of discrete variables sit
            # in the question
            
            # Find the number of variable groups in the set
            vgroups = get_vgroups(question['variables'])        
            
            # For each variable group, get its members
            vgroup_variables = get_vgroup_variables(
                vgroups, question['variables']
            )
        
        # vgroup_types is used to keep track of the types used in the
        # variable group. This will help us identify mixed-type
        # question groups which are not arrays.            
        vgroup_types = get_vgroup_types(vgroups, question['variables'])
        unique_vgroup_types = set(vgroup_types.values())
        
        # Note if the vgroups use more than one variable type
        mixed_types = len(unique_vgroup_types) > 1
        
        if mixed_types:
            # A set should be creted to bind mixed-type variables 
            # together

            vgroup = vgroups[0]
            
            # Create the set
            mask = meta['sets'][vgroup] = {
                'item type': 'mixed',
                'text': {text_key: question['qtitle']},
                'items': [
                    'columns@%s' % (var['label'])
                    for var in question['variables']
                ]
            }        

        if 'multiple' in list(vgroup_types.values()):
            # This is a multiple grid
            # vgroup and vgroup_variables needs to be
            # edited to make them useable in the next step
            # This is related to the structure of multiple
            # response variables in Decipher
            multiple_vgroups = [
                vgroup
                for vgroup in vgroups
                if vgroup_types[vgroup] == 'multiple'
            ]
            vgroup_variables = [copy.copy(vgroups)]
            new_vgroup_match = re.match('(^.+)(?=[c|r][0-9]+)', vgroups[0])
            if new_vgroup_match is None:
                continue
            else:
                vgroups = [new_vgroup_match.group(0)]
                vgroup_types[vgroups[0]] = 'multiple'
        
        # Extract only the vgroups that contain multiple variables
        # so that an array mask can be created for each of them
        array_vgroups = [
            (vgroup, vars)
            for vgroup, vars in zip(vgroups, vgroup_variables)
            if len(vars) > 1
        ]
        
        # If there are any array-like groups of variables inside the
        # question, add an array mask/s accordingly
        for vgroup, vars in array_vgroups:
        
            if vgroup in meta['masks']:
                # This was a multiple-choice grid and has
                # already been converted
                continue
        
            # It's possible the vgroup is in the 'data file' set
            # and needs to be replaced with the name of the group's
            # component vars. This happens with compound questions
            # that are arrays with added open-ends variables
            mapped_vgroup = 'columns@%s' % (vgroup)
            df_items = meta['sets']['data file']['items']
            if mapped_vgroup in df_items:
                mapped_vars = [('columns@%s' % v['label']) for v in vars]
                idx = meta['sets']['data file']['items'].index(mapped_vgroup)
                df_items = df_items[:idx] + mapped_vars + df_items[idx+1:]
                meta['sets']['data file']['items'] = df_items
                    
            # Create the array mask
            mask = meta['masks'][vgroup] = {
                'type': 'array',
                'item type': types_map[vgroup_types[vgroup]],
                'text': {text_key: (
                    '{} - {}'.format(
                        vars[0]['rowTitle'], 
                        question['qtitle']
                    )
                    if vgroup_types[vgroup] in ['number', 'float', 'text']
                    else question['qtitle']
                )},
                'items': [{
                    'source': 'columns@{}'.format(var['label']),
                    'text': {text_key: var['rowTitle']}}
                    for var in vars
                ]}
    
            if vgroup_types[vgroup] in ['single', 'multiple']:
                # Create lib values entry
                values_mapper = 'lib@values@%s' % (vgroup)
                meta['masks'][vgroup]['values'] = values_mapper
                if vgroup_types[vgroup] == 'single':
                    values = get_decipher_values(question['values'], text_key)
                elif vgroup_types[vgroup] == 'multiple':
                    values = copy.deepcopy(meta['columns'][vars[0]]['values'])
                meta['lib']['values'][vgroup] = values
                
                # Use meta-mapped values reference for single or 
                # multiple array variables
                for item in mask['items']:
                    col = item['source'].split('@')[-1]
                    if col in meta['columns']:
                        if 'values' in meta['columns'][col]:
                            meta['columns'][col]['values'] = values_mapper
    
    # Construct quota columns (meta+data)
    meta, data = manage_decipher_quota_variables(meta, data, quotas)

    # Confirm that all meta columns exist in the data
    for col in list(meta['columns'].keys()):
        if not col in data.columns:
            print((
                "Unpaired data warning: {} found in meta['columns']"
                " but not in data.columns. Removing it.".format(col)))
            del meta['columns'][col]
            set_item = 'columns@{}'.format(col)
            if set_item in meta['sets']['data file']['items']:
                idx = meta['sets']['data file']['items'].remove(set_item)

    # Confirm that all data columns exist in the meta
    for col in data.columns:
        if not col in meta['columns']:
            print((
                "Unpaired meta warning: {} found in data.columns"
                " but not in meta['columns']. Removing it.".format(col)))
            data.drop(col, axis=1, inplace=True)

    return meta, data
