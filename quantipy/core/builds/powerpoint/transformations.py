# encoding: utf-8

'''
@author: Majeed.sahebzadha
'''


import numpy as np
import pandas as pd
from math import ceil
import re
import operator
from quantipy.core.helpers import functions as helpers

''' Simplified access to, and manipulation of, the pandas dataframe.
    Contains various helper functions.
'''

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

def clean_df_values(old_df, replace_this, replace_with_that, regex_bol, as_type):
    '''

    '''

    new_df = old_df.replace(replace_this,
                            replace_with_that,
                            regex=regex_bol).astype(as_type)

    return new_df

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

def case_insensitive_matcher(check_these, against_this):
    '''
    performs the case-insensitive search of given list of items against df rows or columns and pulls out
    matched items from the df.
    '''
    matched = [v
               for x,d in enumerate(check_these)
                   for i,v in enumerate(against_this)
                        if v.lower() == d.lower()
                        ]
    return matched

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

def remove_percentage_sign(old_df):
    '''

    '''

    new_df = old_df.replace('%','',regex=True).astype(np.float64)

    return new_df

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

def drop_null_rows(old_df, axis_type=1):
    '''
    drop rows with all columns having value 0
    '''

    new_df = old_df.loc[(df!=0).any(axis=axis_type)]

    return new_df

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

def auto_sort(df, fixed_categories=[], column_position=0, ascend=True):
    '''
    Sorts a flattened (non multiindexed) panda dataframe whilst excluding given rows

    Params:
    -------

    df: pandas dataframe
    fixed_categories: list
        list of row labels
        example: [u'Other', u"Don't know/ can't recall", u'Not applicable']
    column_position: int
    ascend: boolean
        Sort ascending vs. descending
    '''

    # ensure df is not empty
    if not df.empty:

        # ensure df is not multiindexed
        nblevels = df.index.nlevels
        if nblevels == 2:
            raise Exception('Expected Flat DF, got multiindex DF')

        # ensure fixed_categories is not empty
        if fixed_categories:

            #reindex df because it might contain duplicates
            df = df.reset_index()

            #df with no fixed categories, then sort.
            df_without_fc = df.loc[~df[df.columns[0]].isin(fixed_categories)]
            if pd.__version__ == '0.19.2':
                df_without_fc = df_without_fc.sort_values(by=df.columns[column_position+1], ascending=ascend)
            else:
                df_without_fc = df_without_fc.sort(columns=df.columns[column_position+1], ascending=ascend)

            #put each row as a tuple in a list
            tups = []
            for x in df_without_fc.itertuples():
                tups.append(x)

            #get fixed categories as a df
            df_fc = df[~df.index.isin(df_without_fc.index)]

            #convert fixed categories to rows of tuples,
            #then insert row to tups list in a specific index
            for x in df_fc.itertuples():
                tups.insert(x[0], x)

            #remove the indexes from the list of tuples
            filtered_tups = [x[1:] for x in tups]

            #put all the items in the tups list together to build a df
            new_df = pd.DataFrame(filtered_tups, columns=list(df.columns.values))
            new_df = new_df.set_index(df.columns[0])

        else:
            if pd.__version__ == '0.19.2':
                new_df = df.sort_values(by=df.columns[column_position], ascending=ascend)
            else:
                new_df = df.sort(columns=df.columns[column_position], ascending=ascend)

    return new_df

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

def all_same(numpy_list):
    '''

    '''

    val = numpy_list.tolist()

    return all(x == val[0] for x in val)

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

def find_dups(df, orientation='Side'):
    '''
    Looks for duplicate labels in a df. Converts axis
    labels to a list and then returns duplicate index from list.
    If the list contains duplicates then a statememnt is returned.
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
    '''
    returns a list of dataframes sliced as evenly as possible
    '''

    #ensure the indexs are strings not ints or floats
    if not isinstance(df.index, str):
        df.index = df.index.map(str)

    row_count = len(df.index)

    maxs = pd.Series(list(range(min_rows, max_rows+1)))
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

def strip_html_tags(text):
    '''
    Strip HTML tags from any string and transform special entities
    '''

    rules = [
             {r'<[^<]+?>': ''},                # remove remaining tags
             {r'^\s+' : '' },                  # remove spaces at the beginning
             {r'\,([a-zA-Z])': r', \1'},        # add space after a comma
             {r'\s+' : ' '}                    # replace consecutive spaces
             ]
    for rule in rules:
        for (k,v) in list(rule.items()):
            regex = re.compile(k)
            text = regex.sub(v, text)

    # replace special strings
    special = {
               '&nbsp;': ' ',
               '&amp;': '&',
               '&quot;': '"',
               '&lt;': '<',
               '&gt;': '>',
               '**': '',
               "’": "'"

               }
    for (k,v) in list(special.items()):
        text = text.replace(k, v)

    return text

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

def clean_axes_labels(df):
    '''
    Cleans dataframe labels. Strips html code, double white spaces and so on.

    Params:
    -------

    df: pandas dataframe
    '''

    #standardise all index/column elements as unicode
    df_index_labels = df.index.map(str)
    df_col_labels = df.columns.map(str)

#     df_index_labels = [unicode(w)
#                        if not isinstance(w, unicode) and not isinstance(w, str)
#                        else w
#                        for w in df.index.values]

#     df_col_labels = df.columns.values

    col_labels = []
    index_labels = []

    for ctext in df_col_labels:
        ctext = strip_html_tags(ctext)
        col_labels.append(ctext)

    for indtext in df_index_labels:
        indtext = strip_html_tags(indtext)
        index_labels.append(indtext)

    df.columns = col_labels
    df.index = index_labels

    return df

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

def color_setter(numofseries, color_order='reverse'):
    '''

    '''

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

def place_vals_in_labels(old_df, base_position=0, orientation='side', drop_position=True):
    '''
    Takes values from a given column or row and inserts it to the df's row or column labels.
    Normally used to insert base values in row or column labels.
    '''

    if orientation == 'side':
        #grab desired column's values, normally index 0
        col_vals = old_df.ix[:,[base_position]].values
        #col_vals returns a list of list which needs flattening
        flatten_col_vals = [item for sublist in col_vals for item in sublist]
        #grab row labels
        index_labels = old_df.index

        new_labels_list = {}
        for x,y in zip(index_labels, flatten_col_vals):
            new_labels_list.update({x : x + " (n=" + str(y) +")"})

        new_df = old_df.rename(index=new_labels_list, inplace=False)

        if drop_position:
            new_df = new_df.drop(new_df.columns[[base_position]], axis=1, inplace=False)

    else:
        #grab desired row's values, normally index 0
        row_vals = old_df.ix[[base_position],:].values
        #row_vals returns a list of list which needs flattening
        flatten_col_vals = [item for sublist in row_vals for item in sublist]
        #grab row labels
        col_labels = df.columns

        #rename rows one by one.
        new_labels_list = {}
        for x,y in zip(index_labels, flatten_col_vals):
            new_labels_list.update({x : x + " (n=" + str(y) +")"})

        new_df = old_df.rename(columns=new_labels_list, inplace=False)

        if drop_position:
            new_df = new_df.drop(new_df.index[[base_position]], axis=0, inplace=False)

    return new_df

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

def get_qestion_labels(cluster_name, meta, table_name=None):
    '''

    '''

    question_label_dict ={}

    text_key = meta['lib']['default text']

    table_list = list(cluster_name.keys())

    for table in table_list:
        view = cluster_name[table][cluster_name[table].data_key][cluster_name[table].filter][table][cluster_name[table].content_of_axis[0]][cluster_name[table].views[1]]

        vdf = view.dataframe
#         vdf = drop_hidden_codes(vdf, meta)
#         df = index_from_meta(vdf, meta, vdf)
#         question_label = df.index.get_level_values(level=0)[0]

#         question_label_dict[table] = question_label

        qname = vdf.index.get_level_values(0).tolist()[0]
        vdf_meta = meta['columns'].get(qname, '%s not in the columns set in the meta' % (qname))

        question_label_dict[table] = vdf_meta['text'][text_key]

#         question_label_dict[table] = vdf_meta['text'][text_key]

    if table_name:
        return question_label_dict[table_name]
    else:
        return question_label_dict

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

def validate_cluster_orientations(cluster):
    '''
    Make sure that the chains follow the rule:
        - All chains must have the same orientation, x or y.
    '''
    if len(set([
        cluster[chain_name].orientation
        for chain_name in list(cluster.keys())
    ])) != 1:
        raise Exception(
            "Chain orientations must be consistent. Please review chain "
            "specification"
        )

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

def get_base(df, base_description, is_mask):
    '''
    Constructs base text for any kind of chart, single, multiple, grid.

    Params:
    -------

    df: pandas dataframe
    base_description: str
    '''
    num_to_str = lambda string: str(int(round(string)))
    base_text_format = lambda txt, num: '{} ({})'.format(txt, num_to_str(num))

    #standardise all index/column elements as unicode
    df_index_labels = df.index.map(str)
    df_col_labels = df.columns.map(str)

    # get col labels and row values
    top_members = df.columns.values
    base_values = df.values

    # count row/col
    numofcols = len(df.columns)
    numofrows = len(df.index)

    #if base description is empty then
    if base_description:
        #example of what the base description would look like - 'Base: Har minst ett plagg'
        #remove the word "Base:" from the description
        description = base_description.split(': ')[-1]
        #grab the label for base from the df
        base_label = df.index[0]
        #put them together
        base_description = '{}: {}'.format(base_label, description)
    else:
        base_description = df.index.values[0]
    base_description = base_description.strip()

    #single series format
    if numofcols == 1:
        base_text = base_text_format(base_description, base_values[0][0])

    #multi series format
    elif numofcols > 1:
        # if all_same(base_values[0]):
        #     base_text = base_text_format(base_description, base_values[0][0])
        # else:
        if not is_mask:
            it = list(zip(top_members, base_values[0]))
            base_texts = ', '.join([base_text_format(x, y) for x, y in it])
            base_text = ' - '.join([base_description, base_texts])
        else:
            base_text = base_text_format(base_description, base_values[0][0])
    return base_text


'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

def replace_decimal_point_with(df, replacer=","):
    '''

    '''

    for col in df.columns:
        df[col] = pd.Series(["{0}".format(val) for val in df[col]], index = df.index)
    return df

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

def set_column_sequence(dataframe, seq):
    '''
    Takes a dataframe and a subsequence of its columns, returns dataframe with seq as first columns
    '''

    cols = seq[:] # copy so we don't mutate seq
    for x in dataframe.columns:
        if x not in cols:
            cols.append(x)

    return dataframe[cols]

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

def round_df_cells(df, decimal_points):
    '''

    '''

    if decimal_points == 0:
        df = df.applymap(lambda x: int(round(x)))
    else:
        df = df.applymap(lambda x: round(x, decimal_points))

    return df

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

def reverse_order(old_df, orientation='side'):
    '''
    Will reverse the order of rows or columns
    '''

    if orientation == 'side':
        df = old_df.iloc[::-1]
    else:
        df = old_df[old_df.columns[::-1]]

    return df

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

def get_selection_by_index(df, position, orientation='side'):
    '''
    Grabs and returns a single column or row.
    example: myrow = get_selection_by_index(mydf, 2, 'side')
    grabs row 2 from df.
    '''

    if orientation == 'side':
        #will return a single row
        df = df.ix[[position],:]
    else:
        #will return a single col
        df = df.ix[:,[position]]

    return df

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

def del_by_label(df, label_to_del, orientation='side'):
    '''
    deletes a single or multiple row or col labels from df
    param - label_to_del: takes a list of labels
    '''

    #if what's passed into label_to_del is not in a list then
    #put it in a list
    if not isinstance(label_to_del, list):
        label_to_del = [label_to_del]

    if orientation=='side':
        orientation=0
    else:
        orientation=1

    df = df.drop([label_to_del], axis=orientation, inplace=True)

    return df

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

def rename_label(df, old_label, new_label, orientation='side'):
    '''
    Renames a single row or cols label
    '''

    if orientation == 'side':
        df.rename(index={old_label: new_label}, inplace=True)
    else:
        df.rename(columns={old_label: new_label}, inplace=True)

    return df

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

def drop_hidden_codes(view):
    '''

    '''

    #drop hidden codes
    if 'x_hidden_codes' in view.meta():
        vdf = helpers.deep_drop(
                view.dataframe,
                view.meta()['x_hidden_codes'],
                axes=0)
    else:
        vdf = view.dataframe

    return vdf

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

def paint_df(vdf, view, meta, text_key):
    '''

    '''

    #add question and value labels to df
    if 'x_new_order' in view.meta():
        df = helpers.paint_dataframe(
                df=vdf.copy(),
                meta=meta,
                ridx=view.meta()['x_new_order'],
                text_key=text_key)
    else:
        df = helpers.paint_dataframe(
                df=vdf.copy(),
                meta=meta,
                text_key=text_key)

    return df

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

def partition_view_df(view, values=False, data_only=False, axes_only=False):
    '''
    Disassembles a view dataframe object into its
    inner-most index/columns parts (by dropping the first level)
    and the actual data.

    Parameters
    ----------
    view : Quantipy view

    values : boolean, optional
        If True will return the np.array
        containing the df values instead of a dataframe

    data_only : boolean, optional
        If True will only return the data component of the view dataframe

    axes_only : boolean, optional
        If True will only return the inner-most index and columns component
        of the view dataframe.

    Returns
    -------
    data, index, columns : dataframe (or np.array of values), index, columns
    '''
    df = view.copy()
    if isinstance(df.index, pd.MultiIndex):
        df.index = df.index.droplevel()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel()
    index = df.index
    columns = df.columns
    data = df if not values else df.values

    if data_only:
        return data
    elif axes_only:
        return index.tolist(), columns.tolist()
    else:
        return data, index.tolist(), columns.tolist()

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

def is_grid_element(table_name, table_pattern):
    '''
    Checks if a table is a grid element or not

    Parameters
    ----------

    '''

    matches = table_pattern.findall(table_name)

    if (len(matches)>0 and len(matches[0])==2):
        matched = True
    else:
        matched = False

    return matched
