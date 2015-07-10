# -*- coding: utf-8 -*-

'''
Created on 02 Oct 2014

@author: Majeed.sahebzadha
'''

from __future__ import unicode_literals
import numpy as np
import pandas as pd
from math import ceil
from ast import literal_eval
import re
import operator

''' Simplified access to, and manipulation of, the pandas dataframe.
    Contains various helper functions. 

    Simplified access to, and manipulation of python-pptx shapes/objects.
    
    Key differences between hubs:
    
    mtd type       |   Country    |   base type       | description location  |  sld title location
    ---------------|--------------|-------------------|-----------------------|---------------------
    1-Omni         |   UK         |   unweightedbase  |     table             |      annotation 
    21-Omni        |   DE         |   Basis Netto     |     table             |      annotation
    31-Omni        |   FR         |   weightedbase    |     table             |      annotation
    41-Omni        |   USA        |   unweightedbase  |     annotation        |      annotation
    [11/12/13/14]  |   Nordic     |   weightedbase    |     annotation        |      annotation
'''

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

def clean_df_values(dframe, replace_this, replace_with_that, regex_bol, as_type):

    df = dframe.replace(replace_this, replace_with_that, regex=regex_bol).astype(as_type)
    
    return df

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

def changeencode(data, cols):
    for col in cols:
        data[col] = data[col].str.decode('iso-8859-1').str.encode('utf-8')
    return data   

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

def meta_filter(df, meta, prop, prop_value, axis=1, delete=True):

    to_del = meta[meta[prop] == prop_value].index.tolist()
    
    if delete == True:
        clean_dframe = df.drop(to_del,axis)
        return clean_dframe
    
    elif delete == False:
        return to_del

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

def df_meta_filter(df, meta, property_type, property_value):
    
    '''Allows you to select a subsection of a df by given meta property.
    example useage: df_meta_filter(df_raw,side_meta,['Category'])
    '''
    #pull rows from meta which has a given property value from the given property type. 
    #in other words: in the given property type, look for the given property value and pull the rows. 
    to_keep = meta[meta[property_type].isin(property_value)].index.tolist()
    
    if to_keep:
        if any(df.index.isin(case_insensitive_matcher(to_keep, df.index))):
            to_keep = case_insensitive_matcher(to_keep, df.index)
            clean_dframe = df.loc[to_keep]
        else:
            to_keep = case_insensitive_matcher(to_keep, df.columns)
            clean_dframe = df[to_keep]
            
        return clean_dframe
    else:
        #return a blank/empty df 
        return pd.DataFrame()
        
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

def case_insensitive_matcher(check_these, against_this):
    '''
    performs the case-insensitive search of given list of items against df rows or columns and pulls out
    matched items from the df.  
    '''
    
    matched = [v for x,d in enumerate(check_these) for i,v in enumerate(against_this) if v.lower() == d.lower()]
    return matched

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'        

def drop_hidden_cols(df, meta, axis=1):
    
    if 'ShownOnTable' in meta:
        to_del = meta[meta['ShownOnTable'] == 'false'].index.tolist()
        col_list = list(df)
        df = df.drop(case_insensitive_matcher(to_del, col_list), axis)
    
    return df

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

def drop_hidden_rows(df, meta, axis=0):
    
    if 'ShownOnTable' in meta:
        to_del = meta[meta['ShownOnTable'] == 'false'].index.tolist()
        row_list = list(df.index)
        df = df.drop(case_insensitive_matcher(to_del, row_list), axis)
        
    return df

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

def ignored_types(df, meta, type_val, axis=1):
    #index values/labels need to be unique otherwise it may delete unintended rows.
    to_del = meta[meta['Type'].isin(type_val)].index.tolist()
    
    clean_df = df.drop(to_del, axis)
    
    return clean_df

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

def remove_percentage_sign(df):
    
    df = df.replace('%','',regex=True).astype(np.float64)
    
    return df

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

def drop_null_rows(df, axis_type=1):
    
    ''' drop rows with all columns having value 0 '''
    
    df = df.loc[(df!=0).any(axis=axis_type)]
    
    return df

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
                                     
def split_df_on_type(df, side_meta):

    df_base  = ignored_types(df, side_meta, ['Category'], axis=0)
    df_table = ignored_types(df, side_meta, ['UnweightedBase', 'Base'], axis=0)
    
    return df_base, df_table

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

def auto_sort(df, meta, fixed_categories, column_position=0, ascend=True):
    ''' sorts df whilst excluding fixed rows/categories '''
    
    if fixed_categories:
        #standardise casing for side meta 
        meta.index = map(str.lower, meta.index)
        #standardise casing for row labels 
        df.index = map(str.lower, df.index)
        #convert string into a proper list
        fixed_categories = literal_eval(fixed_categories)
        #now standardise the casing for all elements in fixed category list
        fixed_categories = map(str.lower, fixed_categories)
        
        #lets check the elements from the fixed_categories list against the df's index. 
        #we want to check if the fixed elements are located at the bottom of the df. 
        #if try, pull these out. We do this by, looking at a splice of the df from the bottom 
        #to the lenth of the fixed_categories, then check if the items in the df are in the fixed_categories list,
        #if true then return the names as in a list. 
        fixed_elements = df[-len(fixed_categories):][df.index[-len(fixed_categories):].isin(fixed_categories)].index.tolist()

        #build df which contains items from fixed_categories only
        excluded_cats = df.loc[fixed_elements]
        #build df which exclude items from fixed_categories
        included_cats = df[~df.index.isin(fixed_elements)]
        #sort the included df based on the position of a column
        sorted_cats = included_cats.sort(columns=df.columns[column_position], ascending=ascend)
        #combine sorted and excluded dfs
        df = pd.concat([sorted_cats, excluded_cats])

    else:
        df = df.sort(columns=df.columns[column_position], ascending=ascend)
    
    return df

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

def append_type_to_end(df, meta, type_value):
    
    df_exp = df_meta_filter(df, meta, 'Type', [type_value])
    
    if not df_exp.empty:
        df_without_exp = df[~df.index.isin(list(df_exp.index), 0)]
        
        df_merged = pd.concat([df_without_exp, df_exp])
        
        return df_merged
    else:
        return df

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

def append_Net_by_label_to_end(df):
    
    res = df.index.tolist()
    nets = []
    for i in df.index:
        if 'Net: ' in i:
            nets.append(res.pop(res.index(i)))

    res.extend(nets)
    df = df.reindex(res)
    return df 

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

def drop_exp(df, meta):
    
    df_exp = df_meta_filter(df, meta, 'Type', ['Expression'])

    if not df_exp.empty:
        df_without_exp = df[~df.index.isin(list(df_exp.index), 0)]
        return df_without_exp
    else:
        return df
    
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

def rename_df_columns(df, columns_name=None):
    #Rename address column to addr1
    #example: df = df.rename(columns={'address': 'addr1'})
    df = df.rename(columns={columns_name})
    return df

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

def all_same(numpy_list):
    
    val = numpy_list.tolist()
    
    return all(x == val[0] for x in val)

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

def find_dups(df, orientation='Side'):
    '''
    Looks for duplicate labels in a df. Convers axis labels to a list and then returns 
    duplicate index from list. If the list contains duplicates then a statememnt is returned. 
    '''
    
    if orientation == 'Side':
        mylist = list(df.index.values)
        axis = 'row'
    else:
        mylist = list(df.columns.values)
        axis = 'column'

    dup_idx = [i for i, x in enumerate(mylist) if mylist.count(x) > 1]
    
    if dup_idx:
        statement = ("\n{indent:>10}*Warning: This table/chart contains duplicate "
                    "{orientation} labels".format(
                                             indent='', 
                                             orientation=axis))
    else:
        statement = ''
        
    return statement
    
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

def df_splitter(df, min_rows, max_rows):
    ''' returns a list of dataframes sliced as evenly as possible ''' 
                                         
    row_count = len(df.index)

    maxs = pd.Series(range(min_rows, max_rows+1))
    rows = pd.Series([row_count]*maxs.size)
    mods = rows % maxs
    splitter = maxs[mods >= min_rows].max()

    if row_count <= max_rows:
        splitter = 1
    else:
        splitter = ceil(row_count/float(splitter))
    
    size = int(ceil(float(len(df)) / splitter))

    return [df[i:i + size] for i in range(0, len(df), size)]

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

def strip_HTML_tags(string):
    """
    Strip HTML tags from any string and transform special entities
    """
    text = string

    # apply rules in given order!
    rules = [
             { r'>\s+' : u'>'},                                 # remove spaces after a tag opens or closes
             { r'\s+' : u' '},                                  # replace consecutive spaces
             { r'\s*<br\s*/?>\s*' : u'\n'},                     # newline after a <br>
             { r'[ \t]*<[^<]*?/?>' : u'' },                     # remove remaining tags
             { r'^\s+' : u'' },                                 # remove spaces at the beginning
             { r'\.([a-zA-Z])' : r'. \1' },                     # add space after a full stop
             { r'\,([a-zA-Z])' : r', \1' }                      # add space after a comma
             ]
    for rule in rules:
        for (k,v) in rule.items():
            regex = re.compile (k)
            text  = regex.sub (v, text)
            
    # replace special strings
    special = {
               '&nbsp;' : ' ', '&amp;' : '&', '&quot;' : '"',
               '&lt;'   : '<', '&gt;'  : '>', '**' : '',
               }
    for (k,v) in special.items():
        text = text.replace (k, v)
        
    return text

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

def format_question_label(question_name, question_text, mtd_origin=None, grid_summary=False):
    '''constructs and formats a given question label in a mixture of ways depending on 
    the given question type and mtd origin.     
    '''

    question_text = strip_HTML_tags(question_text)
    regex = re.compile(r'(.*?)\[.*?\]\.(.*)')
    matches = regex.findall(question_name)
    
    #check if both grid_summary and matches are TRUE
    if grid_summary:
        
        if mtd_origin in ["1-Omni", "41-Omni"]:
            #get everything in between "." and "?"            
            question_text = question_text[question_text.find(".")+1:question_text.find("?")+1].strip()
            question_text = matches[0][0] + ". " + question_text
        
        elif mtd_origin == "21-Omni": 
            #get everything after ":" in the question label
            question_text = question_text.split(" : ")[-1]
            
        elif mtd_origin in ["11", "12", "13", "14"]:
            if "?" in question_text:
                question_text = question_text[:question_text.find("?")+1].strip()
            elif "Bas: " in question_text:
                question_text = question_text[:question_text.find("Bas: ")].strip()
            elif "Base: " in question_text: 
                question_text = question_text[:question_text.find("Base: ")].strip()

    else:
        #UK / FR
        if mtd_origin in ["1-Omni", "41-Omni"]:
            #check if it's a grid element question
            if (len(matches) > 0 and len(matches[0]) == 2):
                #get grid statement
                grid_statement = question_text.split(' - ', -1)[-1]
                #get table name
                table_name = question_text.split('. ', 1)[0]
                #combine the 2 vars to build a grid element question 
                question_text = table_name + '. ' + grid_statement
            else:
                question_text = question_text
        #DE    
        elif mtd_origin == "21-Omni": 
            question_text = question_text
        #NORDIC
        elif mtd_origin in ["11", "12", "13", "14"]:
            question_text = question_text
    
    #if question text has empty then assign the string "Blank" to it. 
    if not question_text:
        question_text = "Blank"
        
    return question_text


'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

def get_base(df, base_description, mtd_origin="1-Omni", grid_summary=False):
    

    numofcols = len(df.columns)
    numofrows = len(df.index)

    if mtd_origin in ["1", "1-Omni", "31-Omni", "21-Omni", "41-Omni"]:   
         
        top_members = df.columns.values
        base_values = df.values
        if not base_description:
            base_description = df.index.values[0] 
         
        #single series format
        if numofcols == 1:
            base_text = base_description.strip() + " (" + base_values[0][0] +") " 
         
        #multi series format
        elif numofcols > 1:
            if all_same(base_values[0]):
                base_text = base_description.strip() + " (" +  base_values[0][0] + ") "
            else:
                base_text = base_description.strip() + " - " + ", ".join(['{} ({})'.format(str(x),str(y)) for x,y in zip(top_members, base_values[0])]) 

    elif mtd_origin in ["11", "12", "13", "14"]:
        
        if grid_summary:
            side_members = df.index
            base_values = df.values
            if not base_description:
                base_description = 'Base'
                
            #base_text = base_description.strip() + ": " + ", ".join(['{} ({})'.format(x,y[0]) for x,y in zip(side_members, base_values)])
            base_text = base_description.strip() + " (" +  base_values[0][0] + ") "
            
        else:
            top_members = df.columns.values
            base_values = df.values
            if not base_description:
                base_description = df.index.values[0]
 
            if numofcols == 1:
                base_text = base_description.strip() + " (" + base_values[0][0] +") " 
            elif numofcols > 1:
#                 if all_same(base_values[0]):
#                     base_text = base_description.strip() + " (" +  base_values[0][0] + ") "
#                 else:
                base_text = base_description.strip() + ": " + ", ".join(['{} ({})'.format(str(x),str(y)) for x,y in zip(top_members, base_values[0])])

    return base_text

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

# def color_setter(numofseries, color_order='reverse'):
# 
#     import operator
# 
#     #all series colours are derived from 6 color themes outlined below:
#     color_set = [(147,208,35), (83,172,175), (211,151,91), (17,124,198), (222,231,5), (136,87,136)]
#     colour_incrementor = [(16,11,16), (26,26,1), (3,15,23), (24,18,11), (10,56,20), (17,24,5)]
#     
#     if numofseries <= 6:
#         color_set = color_set[0:numofseries]
# 
#     else:
#         for x in range(6, numofseries):
#             base_color_index = x % 6 #base colour
#             multiply_by = x // 5 #offset value
#             increment_by = [tup*multiply_by for tup in colour_incrementor[base_color_index]]
#             series_color = tuple(map(operator.add, color_set[base_color_index], increment_by))
#             color_set.append(series_color)
#   
#     if color_order == 'reverse':
#         return color_set[::-1]
#     else:
#         return color_set

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

def color_setter(numofseries, color_order='reverse'):
    
    color_set = [(147,208,35), (83,172,175), (211,151,91), (17,124,198), (222,231,5), (136,87,136),
                 (88,125,21), (49,104,106), (143,91,38), (10,74,119), (133,139,3), (82,52,82),
                 (171,224,72), (117,189,191), (220,172,124), (38,155,236), (242,250,40), (165,115,165),
                 (118,166,28), (66,138,141), (190,121,51), (14,99,158), (178,185,4), (109,70,109),
                 (192,232,118), (152,205,207), (229,193,157), (92, 180,241), (245,252,94), (188,150,188),
                 (74,104,18), (41,87,88), (119,75,32), (9,62,99), (111,116,3), (68,44,68),
                 (197,227,168), (176,209,210), (229,199,178), (166,188,222), (235,240,166), (193,177,193),
                 (202,229,175), (182,212,213), (231,203,184), (174,194,224), (237,241,174), (198,183,198)]
    
    color_set = color_set[0:numofseries]
    if color_order == 'reverse':
        return color_set[::-1]
    else:
        return color_set
    

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

def get_slide_title(dGroup):

    annotation_block = dGroup['TitleFooter'].strip()
    if annotation_block:
        if '/' in annotation_block:
            title_text = annotation_block.split('/')[0]
            
            if title_text.startswith('slide_title:', 0):
                raw = title_text.split('slide_title:')[-1].strip().decode("utf-8")
                if raw: 
                    return raw 
                else:
                    return "Click to add title"
        else:
            return "Click to add title"

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'  
        
def has_reporting_tags(stack, folder_name, key):
    '''
    checks if any reporting tags for a given folder exists before building a pptx.
    '''
    
    for file_name, data in stack.iteritems():
        if file_name!='meta':
            if file_name==folder_name:
                for filter_name, subset in data.iteritems():
                    if filter_name!='meta':
                        for side_name, side in subset.iteritems():
                            if side_name!='meta':
    
                                meta_props = stack[file_name]['meta']['Side SubAxes'].loc[side_name].dropna() 
                                if key in meta_props:
                                    return True
                                    break
                        return False
 
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'                 

def format_grid_statements(text, mtd_origin):

    if mtd_origin == "1-Omni":
        text = text.split(" - ")[-1]
    elif mtd_origin == "21-Omni":
        text = text.split(" : ")[0]
    return text
        