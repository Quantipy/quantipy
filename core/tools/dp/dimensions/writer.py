"""
Created on 20 June 2016

@author: AlexBuchhammer
"""
import numpy as np
import pandas as pd
import quantipy as qp
from quantipy.core.helpers.functions import (
    is_mapped_meta,
    get_mapped_meta,
    get_text
)
from quantipy.core.helpers.functions import load_json
import os
import json

QTYPES = {
    'single': 'mr.Categorical',
    'delimited set': 'mr.Categorical',
    'int': 'mr.Long',
    'float': 'mr.Double',
    'string': 'mr.Text',
    'date': 'mr.Date',
    'boolean': 'mr.Boolean'
}


def tab(tabs):
    return '' if tabs == 0 else '\t' * tabs

def vetlab(label):
    text = label.replace('"', "'")
    text = text.replace('\n', "")
    return text

def AddLanguage(lang):
    add_lang = 'MDM.Languages.Add("{}")'.format(lang)
    set_lang = 'MDM.Languages.Current = "{}"'.format(lang)
    return '\n'.join([add_lang, set_lang])

def Dim(*args):
    text = 'Dim {}'.format(
        ', '.join(*args))
    return text

def SetMDM():
    text = u'Set MDM = CreateObject("MDM.Document")'
    return text

def section_break(n):
    text = u"\n'{}".format('#'*n)
    return text

def comment(tabs, text):
    text = u"{t}' {tx}".format(
        t=tab(tabs),
        tx=text)
    return text

def CreateVariable(tabs, name, label):
    text = u'{t}Set newVar = MDM.CreateVariable("{n}", "{l}")'.format(
        t=tab(tabs),
        n=name,
        l=vetlab(label))
    return text

def DataType(tabs, parent, dtype):
    text = u'{t}{p}.DataType = {dt}'.format(
        t=tab(tabs),
        p=parent,
        dt=dtype)
    return text

def MaxValue(tabs, parent, mval):
    text = u'{t}{p}.MaxValue = {mv}'.format(
        t=tab(tabs),
        p=parent,
        mv=mval)
    return text

def CreateElement(tabs, name, label):
    text = u'{t}Set newElement = MDM.CreateElement("{n}", "{l}")'.format(
        t=tab(tabs),
        n=name,
        l=vetlab(label))
    return text

def ElementType(tabs):
    text = u'{t}newElement.Type = 0'.format(
        t=tab(tabs))
    return text

def ElementExpression(tabs, expression):
    text = u'{t}newElement.Expression = {e}'.format(
        t=tab(tabs),
        e=expression)
    return text

def AddElement(tabs, parent, child):
    text = u'{t}{p}.Elements.Add({c})'.format(
        t=tab(tabs),
        p=parent,
        c=child)
    return text

def AddField(tabs, parent, child):
    text = u'{t}{p}.Fields.Add({c})'.format(
        t=tab(tabs),
        p=parent,
        c=child)
    return text

def CreateGrid(tabs, name, label):
    text = u'{t}Set newGrid = MDM.CreateGrid("{n}", "{l}")'.format(
        t=tab(tabs),
        n=name,
        l=vetlab(label))
    return text

def MDMSave(tabs, path_mdd):
    text = u'{t}MDM.Save("{p}")'.format(
        t=tab(tabs),
        p=path_mdd)
    return text

def _dedupe_datafile_items_set(items_list, keep_first=False):
    # NOTE:
    #-------------------------------------------------------------------------
    # reverse and re-reverse because I need the last (!) deduped items
    # for correct order
    items_list = list(reversed(items_list))
    items_list_deduped = {}
    return reversed([items_list_deduped.setdefault(i, i) for i in items_list
                     if i not in items_list_deduped])

def create_mdd(meta, data, path_mrs, path_mdd, mdm_lang):
    text_key = meta['lib']['default text']
    mrs = [
        Dim(['MDM', 'newVar', 'newElement', 'newGrid']),
        SetMDM()]
    mrs.append(AddLanguage(mdm_lang))
    all_items = []
    for item in meta['sets']['data file']['items']:
        all_items.append(item.split('@')[-1])
    all_items = _dedupe_datafile_items_set(all_items)
    for name in all_items:
        if name in meta['columns']:
            col = meta['columns'][name]
            # NOTE:
            #-----------------------------------------------------------------
            # patched: some meta does not have the 'name' object, so I am
            # deriving it from the column name...
            # how is the missing 'name' key possible?
            if not 'name' in col: col['name'] = name
            mrs.extend(
            col_to_mrs(
                meta=meta,
                col=col,
                text_key=text_key))
        elif name in meta['masks']:
            mask = meta['masks'][name]
            mrs.extend(
                mask_to_mrs(
                    meta=meta,
                    mask=mask,
                    name=name,
                    text_key=text_key))
    mrs.extend([
        section_break(20),
        comment(0, 'Save MDD'),
        MDMSave(0, path_mdd)])

    with open(path_mrs, 'w') as f:
        f.write(u'\n'.join(mrs).encode('utf-8', errors='replace'))

def get_categories_mrs(meta, vtype, vvalues, child, child_name, text_key):
    if is_mapped_meta(vvalues):
        vvalues = get_mapped_meta(meta, vvalues)
    var_code = []
    mval = 1 if vtype == 'single' else len(vvalues)
    var_code.append(MaxValue(0, child, mval))
    for value in vvalues:
        name = '{}a{}'.format(child_name, value['value'])
        vlabel = get_text(value['text'], text_key)
        var_code.extend([
            CreateElement(0, name, vlabel),
            ElementType(0),
            AddElement(0, child, 'newElement')])
    return var_code

def col_to_mrs(meta, col, text_key):
    # try:
    # NOTE:
    #-------------------------------------------------------------------------
    # .replace() is only for US weight vars that can contain line breaks
    clabel = get_text(col['text'], text_key).replace('\n', ' ')
    col_code = [
        section_break(20),
        comment(0, '{}'.format(col['name'])),
        CreateVariable(0, col['name'], clabel),
        DataType(0, 'newVar', QTYPES[col['type']]),
    ]
    if col['type'] in ['single', 'delimited set']:
        col_code.extend(
            get_categories_mrs(
                meta=meta,
                vtype=col['type'],
                vvalues=col['values'],
                child='newVar',
                child_name=col['name'],
                text_key=text_key))
    col_code.append(AddField(0, 'MDM', 'newVar'))
    return col_code
    # except Exception, e:
    #     # print e
    #     print col['name']
    #     return section_break(20)

def mask_to_mrs(meta, mask, name, text_key):
    mask_code = []
    mask_name = name.split('.')[0]
    field_name = '{}_grid'.format(mask_name)
    mitem0_name = mask['items'][0]['source'].split('@')[-1]
    mitem0 = get_mapped_meta(meta, mask['items'][0]['source'])[mitem0_name]
    mtype = mitem0['type']

    if mtype in ['single', 'delimited set']:
        mvalues = mask['values']

    mlabel = get_text(mask['text'], text_key)
    mask_code.extend([
        section_break(20),
        comment(0, '{}'.format(mask_name)),
        CreateGrid(0, mask_name, mlabel)])
    for i, item in enumerate(mask['items'], start=0):
        name = mask['items'][i]['source'].split('@')[-1].split('.')[0]
        name = name.replace('[{', '').replace('}]', '').replace(mask_name, '', 1)
        ilabel = get_text(mask['items'][i]['text'], text_key)
        mask_code.extend([
            CreateElement(0, name, ilabel),
            AddElement(0, 'newGrid', 'newElement')])

    mask_code.extend([
        CreateVariable(0, field_name, mlabel),
        DataType(0, 'newVar', QTYPES[mtype])])

    if mtype in ['single', 'delimited set']:
        mask_code.extend(
            get_categories_mrs(
                meta=meta,
                vtype=mtype,
                vvalues=mvalues,
                child='newVar',
                child_name=mask_name,
                text_key=text_key))

    mask_code.extend([
        AddField(0, 'newGrid', 'newVar'),
        AddField(0, 'MDM', 'newGrid')])

    return mask_code

def create_ddf(master_input, path_dms):
    dms_dummy_path = os.path.dirname(__file__)
    define_input_name = '#define MASTER_INPUT "{}"'.format(master_input)
    dms = open(os.path.join(dms_dummy_path, '_create_ddf.dms'), 'r')
    full_dms = [define_input_name] + [line.replace('\n', '') for line in dms]
    # NOTE:
    #-------------------------------------------------------------------------
    # dropping the second "line" which is an invisible line-break char
    del full_dms[1]
    with open(path_dms, 'w') as f:
        f.write('\n'.join(full_dms))

def _paired_empty_csv(meta, data):
    """
    """
    empty_csv = data.copy()
    cols = meta['sets']['data file']['items']
    cols = [c.split('@')[-1] for c in cols]
    # cols = [col for col in cols if col in data.columns]
    cols = _dedupe_datafile_items_set(cols)
    paired_cols = []
    for col in cols:
        if col in meta['columns']:
            paired_cols.append(col)
        elif col in meta['masks']:
            mask = meta['masks'][col]
            items = [i['source'].split('@')[-1] for i in mask['items']]
            paired_cols.extend(items)
    empty_csv = empty_csv[paired_cols]
    empty_csv[paired_cols] = np.NaN
    return empty_csv

def _datastore_csv(meta, data, columns):
    """
    """
    datastore = data.copy()
    categoricals, texts = [], []
    for col in columns:
        col_type = meta['columns'][col]['type']
        if col_type in ['single', 'delimited set']:
            datastore[col] = convert_categorical(datastore[col])
        elif col_type == 'int':
            datastore[col].replace(np.NaN, 'NULL', inplace=True)
            try:
                # Note:
                #-------------------------------------------------------------
                # I am converting to int32 (if possible) to prevent type
                # conflicts
                datastore[col] = datastore[col].astype('int32')
            except:
                pass
        elif col_type == 'float':
            datastore[col].replace(np.NaN, 'NULL', inplace=True)
        elif col_type == 'string':
            datastore[col] = replace_comma_in_string(datastore[col])
    return datastore

def _extract_grid_element_name(gridslice):
    return gridslice.split('.')[0].split('[{')[-1].replace('}]', '')

def get_case_data_inputs(meta, data, path_paired_csv, path_datastore):
    """
    """
    empty_csv = _paired_empty_csv(meta, data)
    paired_cols = empty_csv.columns
    datastore_csv = _datastore_csv(meta, data, paired_cols)
    # NOTE:
    #-------------------------------------------------------------------------
    # This check for consistency between the paired/empty csv, the datastore
    # and the mdd columns (derived from the qp data file items set) is central
    # and should be moved into a method/happen in one place...
    # it's a bit scattered now!
    invalids = ['id_L1', 'id_L1.1', '@1']
    invalids.extend([col for col in datastore_csv.columns
                     if col not in paired_cols])
    for invalid in invalids:
        if invalid in datastore_csv.columns:
            datastore_csv.drop(invalid, axis=1, inplace=True)
    empty_csv.to_csv(path_paired_csv, index=False, sep='\t')
    datastore_csv.to_csv(path_datastore, index=False)

def replace_comma_in_string(string):
    """
    """
    s = string.copy()
    s = s.apply(lambda x: str(x).replace(',', '>_>_>'))
    s.replace('nan', '', inplace=True)
    return s

def convert_categorical(categorical):
    """
    """
    cat = categorical.copy()
    is_gridslice = '{' in cat.name
    if is_gridslice:
        resp_prefix = cat.name.split('[{')[0] + 'a'
    else:
        resp_prefix = categorical.name + 'a'
    if not cat.dtype == 'object':
        cat = cat.apply(lambda x:
                        '{}{}'.format(resp_prefix, int(x))
                        if not np.isnan(x) else np.NaN)
    else:
        cat = cat.apply(lambda x: str(x).split(';')[:-1])
        cat = cat.apply(lambda x: ['{}{}'.format(resp_prefix, code)
                                   for code in x])
        cat = cat.apply(lambda x: str(x).replace('[', '').replace(']', ''))
        cat = cat.apply(lambda x: x.replace("'", '').replace(', ', ';'))
    return cat

def dimensions_from_quantipy(meta, data, path_mdd, path_ddf, text_key=None,
                             mdm_lang = 'ENG',
                             run=True, clean_up=True):
    """
    DESCP

    Parameters
    ----------

    Returns
    -------

    """
    name = path_mdd.split('/')[-1].split('.')[0]
    path =  '/'.join(path_mdd.split('/')[:-1])
    if '/' in path_mdd: path = path + '/'
    path_mrs = '{}create_mdd [{}].mrs'.format(path, name)
    path_dms = '{}create_ddf [{}].dms'.format(path, name)
    path_paired_csv = '{}{}_paired.csv'.format(path, name)
    path_datastore = '{}{}_datastore.csv'.format(path, name)
    all_paths = (path_dms, path_mrs, path_datastore, path_paired_csv)

    create_mdd(meta, data, path_mrs, path_mdd, mdm_lang)
    create_ddf(name, path_dms)
    get_case_data_inputs(meta, data, path_paired_csv, path_datastore)
    print 'Case and meta data validated and transformed.'
    if run:
        from subprocess import check_output, STDOUT, CalledProcessError
        try:
            print 'Converting to .ddf/.mdd...'
            command = 'mrscriptcl "{}"'.format(path_mrs)
            check_output(command, stderr=STDOUT, shell=True)
            print '.mdd file generated successfully.'
            command = 'DMSRun "{}"'.format(path_dms)
            check_output(command, stderr=STDOUT, shell=True)
            print '.ddf file generated successfully.\n'
            print 'Conversion completed!'
        except CalledProcessError as exc:
            print '\nERROR:\n', exc.output
            if clean_up:
                for file_loc in all_paths:
                    os.remove(file_loc)
            return exc.returncode
        if clean_up:
            for file_loc in all_paths:
                os.remove(file_loc)
    return None
