'''
Created on 19 Nov 2014

@author: alasdaire
'''

import numpy as np
import pandas as pd
import quantipy as qp
from quantipy.core.cluster import Cluster
from quantipy.core.chain import Chain
from quantipy.core.helpers import functions as helpers
from quantipy.core.tools.dp.io import unicoder
from quantipy.core.builds.excel.formats.quantipy_basic import (
    STATIC_FORMATS
)
from quantipy.core.builds.excel.formats.quantipy_basic import (
    IMG_URL, 
    IMG_SIZE,
    FREQUENCY_0_REPR,
    DESCRIPTIVES_0_REPR,
    TEST_SEPARATOR
)
import quantipy.core.cluster

from xlsxwriter import Workbook
from xlsxwriter.utility import xl_rowcol_to_cell
import os
from string import ascii_uppercase
from collections import OrderedDict, Counter
from warnings import warn
from PIL import Image
import requests
from io import BytesIO

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

ROW_INDEX_ORIGIN = 8
COL_INDEX_ORIGIN = 1

DEFAULT_X_ROW_HEIGHT = 12.75
X_ROW_WRAP_TRIGGER = 44

DEFAULT_Y_HEAD_HEIGHT = 33.75
Y_HEAD_WRAP_TRIGGER = 11.25

DEFAULT_Y_ROW_HEIGHT = 50
Y_ROW_WRAP_TRIGGER = 44

TEST_SUFFIX = list(ascii_uppercase)
TEST_PREFIX = ['']+list(ascii_uppercase)


'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
def paint_box(worksheet, frames, format_dict, rows, cols, metas, 
              ceil=False, floor=False, testcol_map=None):
    '''
    Writes a "box" of data

    Parameters
    ----------
    worksheet : xlsxwriter.Workbook.worksheet
    frames : list
        list of pd.DataFrame objects
    format_dict : Format
        The dict of all formats
    rows : list
        Number of rows in each pd.DataFrame
    cols : list
        Number of cols in each pd.DataFrame
        Column range of box
    metas : list
        list of dict - view metas
    ceil : bool
        Whether ceiling view
    floor : bool
        Whether floor view
    '''
    if len(metas) == 0:
        rsize = rows[-1][1] - rows[0][0]
    else:
        rsize = rows[-1][1] - rows[0][0] + 1

    csize = cols[-1][1] - cols[0][0] + 1

    coords = [
        [rows[0][0] + (i // csize), cols[0][0] + (i % csize)] 
        for i in xrange(rsize * csize)
    ]

    coordsGenerator = (coord for coord in coords)
    for i, coord in enumerate(coordsGenerator):

        idxf = (i // csize) % len(frames)

        box_coord = [coord[0] - coords[0][0], coord[1] - coords[0][1]]

        # pick cell format
        cell_format = ''
        
        if len(metas) == 0:
            method = 'dataframe_columns'
        else:
            fullname, method, is_weighted = (
                metas[idxf]['agg']['fullname'],
                metas[idxf]['agg']['method'],
                metas[idxf]['agg']['is_weighted']
            )
            _, _, relation, rel_to, _, shortname  = fullname.split('|')

        # cell position
        if i % csize == 0:
            cell_format = 'left-'

        if i % csize == (csize - 1) or (cols[idxf][0] == cols[idxf][1]):
            cell_format = cell_format + 'right-'

        if cell_format == '':
            cell_format = cell_format + 'interior-'

        if ceil:
            if i < (csize):
                cell_format = cell_format + 'top-'
        if floor:            
            if i >= ((rsize * csize) - csize):
                cell_format = cell_format + 'bottom-'
        
        # additional format spec
        if method == 'dataframe_columns':

            cell_format = cell_format + 'STR'

        else:

            # background color (frequency/ coltests)
            cond_1 = method in ['frequency', 'coltests'] and len(relation) == 0
            cond_2 = method in ['default']
            if cond_1 or cond_2:
                if not shortname in ['cbase']:
                    if box_coord[0] == 0:
                        cell_format = cell_format + 'frow-bg-'
                    elif (box_coord[0] // len(frames)) % 2 == 0:
                        cell_format = cell_format + 'bg-'

            # first row (coltests - means)
            if method == 'coltests' and len(relation) > 0:
                if box_coord[0] == 0:
                    cell_format = cell_format + 'frow-'

            # choose view format type
            # base
            if shortname == 'cbase':
                if not ceil:
                    cell_format = cell_format + 'frow-N'
                else:
                    cell_format = cell_format + 'N'

            # frequency
            elif method == 'frequency':

                # counts
                if rel_to == '':

                    if len(relation) == 0:
                        cell_format = cell_format + 'N'

                    # complex logics
                    else:
                        if len(frames) == 1:
                            cell_format = cell_format + 'N-NET'
                        else:
                            if idxf == 0:
                                cell_format = cell_format + 'frow-N-NET'
                            elif idxf == len(frames)-1:
                                cell_format = cell_format + 'brow-N-NET'
                            else:
                                cell_format = cell_format + 'mrow-N-NET'
                                
                # %
                elif rel_to in ['x', 'y']:

                    if len(relation) == 0:
                        cell_format = cell_format + 'PCT'

                    # complex logics
                    else:
                        if len(frames) == 1:
                            cell_format = cell_format + 'PCT-NET'
                        else:
                            if idxf == 0:
                                cell_format = cell_format + 'frow-PCT-NET'
                            elif idxf == len(frames)-1:
                                cell_format = cell_format + 'brow-PCT-NET'
                            else:
                                cell_format = cell_format + 'mrow-PCT-NET'

            # descriptvies
            elif method == 'descriptives':
                if len(frames) == 1:
                    cell_format = cell_format + 'STATS'
                else:
                    if idxf == 0:
                        cell_format = cell_format + 'frow-STATS'
                    elif idxf == len(frames)-1:
                        cell_format = cell_format + 'brow-STATS'
                    else:
                        cell_format = cell_format + 'mrow-STATS'

            # coltests
            elif method == 'coltests':
                cell_format = cell_format + 'TESTS'

            # default 
            elif method == 'default':
                cell_format = cell_format + 'DEFAULT'

            # method not found...
            else:
                raise Exception(
                    "View method not recognised: %s" % (method)
                )

        # value to write into cell
        # dataframe
        if method == 'dataframe_columns':

            data = frames[idxf].head(
                box_coord[0] // len(frames)+1
            ).values[-1]

        # links
        else:

            data = frames[idxf].head(
                box_coord[0] // len(frames)+1
            ).values[-1][box_coord[1]]            
            
            # post-process cell data

            # ebase - convert numpy.inf
            if shortname == 'ebase':
                if data == np.inf:
                    data = str(np.inf)

            # % - divide data by 100 for formatting in Excel
            elif rel_to in ['x', 'y'] and not method in ['coltests',
                                                         'descriptives']:
                data = data / 100

            # coltests - convert NaN to '', otherwise get column letters
            elif method == 'coltests':    
                if pd.isnull(data) or data == 0:
                    data = ''   
                else:   
                    x = data.replace('[', '').replace(']', '')  
                    if len(x) == 1: 
                        data = testcol_map[x]    
                    else:   
                        data = ''   
                        for letter in x.split(', '):    
                            data += testcol_map[letter] + TEST_SEPARATOR  
                        data = data[:-1]

            # replace 0 with char
            try:
                if np.isclose([data], [0]):
                    if method == 'frequency':
                        data = FREQUENCY_0_REPR
                    elif method == 'descriptives':
                        data = DESCRIPTIVES_0_REPR
            except:
                pass

        # Check data for NaN and replace with '-'
        if not isinstance(data, (str, unicode)):
            if np.isnan(data):
                data = '-'

        # write data
        try:
            worksheet.write(
                coord[0], 
                coord[1], 
                data, 
                format_dict[cell_format]
            )
        except Exception, e:
            warn(
                '\n'.join(
                    ['Unable to write data to cell...',
                     '{0:<15}{1:<15}{2:<30}{3:<30}{4}'.format(
                        'DATA', 'CELL', 'FORMAT', 'VIEW FULLNAME', 'ERROR'
                     ),
                     '{0:<15}{1:<15}{2:<30}{3:<30}{4}'.format(  
                        data,
                        xl_rowcol_to_cell(coord[0], coord[1]),
                        cell_format,
                        fullname,
                        e
                    )]
                )
            )  
            
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
def set_row_height(worksheet, row_start, row_stop, text_size=1):
    '''
    Sets the row height for all rows in range.

    Parameters
    ----------
    worksheet : xlsxwriter.Workbook.worksheet
    row_start : int
        first row (index)
    row_stop : int
        last row (index)
    '''
    for row in xrange(row_start, row_stop+1):
        worksheet.set_row(row, DEFAULT_X_ROW_HEIGHT*text_size)

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
def write_column_labels(worksheet, labels, existing_format, row,
                        cols, levels=0):
    '''
    Writes column labels & headings.

    If any labels are greater than Y_ROW_WRAP_TRIGGER, the second format in the
    list will be used. One with text_v_align = 4 (vjustify) makes sense.

    Parameters
    ----------
    worksheet : xlsxwriter.Workbook.worksheet
    labels : list
        list of column labels. Last item is column heading
    existing_formats : list
        list of formats to use
    row : int
        row rto write column labels
    cols : list
        start and end column index
    levels : int
        number of additional levels due to nested ys
    '''
    try:
        if levels == 0:
            worksheet.set_column(cols[0], cols[1], 10)
            if cols[0] == cols[1]:
                worksheet.write_row(
                    row, cols[0], labels[0],  existing_format
                )
            else:
                worksheet.merge_range(
                    row, cols[0], row, cols[1], labels[0][0], existing_format
                )
            worksheet.write_row(row+1, cols[0], labels[1],  existing_format)
        elif levels > 0:
            worksheet.set_column(cols[0], cols[1], 10)
            if len(labels) == 2:
                worksheet.merge_range(
                    row, cols[0], row, cols[1], labels[0][0], existing_format
                )
                for i, col in enumerate(xrange(cols[0], cols[1]+1)):
                    worksheet.merge_range(
                        row+1, 
                        col, 
                        row+1+(levels*2), 
                        col, 
                        labels[1][i], 
                        existing_format
                    )
            elif len(labels) > 2:
                #last row size
                R = ((levels+1)*2)-len(labels)

                # header/ column size
                N = cols[1]-cols[0]+1
                n = N/len(labels[1])

                for lev in xrange((len(labels)/2)):
                    #update header/ column size
                    if lev > 0:
                        N = n
                        n = N / len(labels[(lev*2)+1])
                    # y repeat
                    X = (cols[1]-cols[0]+1) / N
                    for x in xrange(X):
                        #  write header(s)
                        worksheet.merge_range(
                            row+(lev*2), 
                            cols[0]+(N*x), 
                            row+(lev*2), 
                            cols[0]+(N*(x+1))-1, 
                            labels[(lev*2)][0], 
                            existing_format
                        )

                        #write columns
                        if n > 1:
                            for col in xrange(len(labels[(lev*2)+1])):
                                worksheet.merge_range(
                                    row+(lev*2)+1, 
                                    cols[0]+(N*x)+(n*col), 
                                    row+(lev*2)+1, 
                                    cols[0]+(N*x)+(n*col)+(n-1), 
                                    labels[(lev*2)+1][col], 
                                    existing_format
                                )
                        else:
                            if R == 0:
                                worksheet.write_row(
                                    row+(lev*2)+1, 
                                    cols[0], 
                                    labels[(lev*2)+1]*(
                                        (cols[1]-cols[0]+1)/len(labels[-1])
                                    ), 
                                    existing_format
                                )
                            else:
                                for col in xrange(len(labels[(lev*2)+1])):
                                    worksheet.merge_range(
                                        row+(lev*2)+1, 
                                        cols[0]+(N*x)+(n*col), 
                                        row+(lev*2)+R+1, 
                                        cols[0]+(N*x)+(n*col), 
                                        labels[(lev*2)+1][col], existing_format
                                    )
    except:
        pass

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
def write_category_labels(worksheet, labels, existing_format, row, col, 
                          group_size=1, set_heights=False):
    '''
    Writes category labels

    Parameters
    ----------
    worksheet : xlsxwriter.Workbook.worksheet
    labels : list
        list of column labels. Last item is column heading
    existing_formats : list
        list of formats to use
    row : int
        row to start writing labels
    col : int
        column to write labels
    '''
    try:
        for idx, lab in enumerate(labels):
            if group_size > 1 and set_heights:
                set_row_height(
                    worksheet, 
                    row+(idx*group_size)+1, 
                    row+(idx*group_size)+(group_size-1)
                )
            if isinstance(lab, float):
                worksheet.write_number(
                    row+(idx*group_size), 
                    col, 
                    lab, 
                    existing_format
                )
            else:
                if len(lab) < X_ROW_WRAP_TRIGGER:
                    set_row_height(
                        worksheet, 
                        row+(idx*group_size), 
                        row+(idx*group_size)
                    )
                worksheet.write(
                    row+(idx*group_size), 
                    col, 
                    lab, 
                    existing_format
                )
    except:
        pass

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
def write_question_label(worksheet, label, existing_format, row, col):
    '''
    Writes question labels

    Parameters
    ----------
    worksheet : xlsxwriter.Workbook.worksheet
    labels : list
        list of column labels. Last item is column heading
    existing_formats : list
        list of formats to use
    row : int
        row index
    col : int
        column index
    '''
    if len(label) < X_ROW_WRAP_TRIGGER:
        set_row_height(worksheet, row, row)
        worksheet.write(row, col, label, existing_format)
    else:
        worksheet.write(row, col, label, existing_format)

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
def chain_generator(cluster):
    '''Generate chains
    '''
    for chain_name in cluster.keys():
        yield cluster[chain_name]

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
def xy_generator(chain):
    '''x/ y generator
    '''
    for xy in chain.content_of_axis:
        yield xy

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
def view_generator(chain_views, grouped_views=[], ordered=False):
    '''View generator
    '''
    if grouped_views == []:
        for view in chain_views:
            yield [view]
    else:
        if all(isinstance(item, str) for item in grouped_views):
            non_grouped_views = [
                view for view in chain_views 
                if view not in grouped_views
            ]
            for view in non_grouped_views:
                yield [view]
            if all(view in chain_views for view in grouped_views):
                yield [view for view in grouped_views if view in chain_views]
            elif any(view in chain_views for view in grouped_views):
                for view in grouped_views:
                    if view in chain_views:
                        yield [view] 
        elif all(isinstance(item, list) for item in grouped_views):
            non_grouped_views = list(
                set(chain_views)-set([item 
                                      for sub_group in grouped_views 
                                      for item in sub_group])
            )
            for view in non_grouped_views:
                yield [view]
            for sub_group in grouped_views:
                if all(view in chain_views for view in sub_group):
                    yield [view for view in sub_group if view in chain_views]
        else:
            raise TypeError(
                "Grouped views objects must all be \n"
                "<str> or <list>, not mixed types\n"
                "Found: %s" % (
                    ', '.join([type(item) for item in grouped_views])
                )
            )

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
def get_view_offset(chain, offset_dict, grouped_views=[]):
    '''
    Creates a dictionary (x names as keys) of dictionaries
    (view names as keys) of realtive view positions, based on view sizes.

    Parameters
    ----------
    chain : Chain
        quantipy chain object
    offset_dict : dict
        dict of
    grouped_views : list
        views to group

    Returns
    -------
    offset_dict dictionary
    '''
    bumped_views = []
    
    x_iter = {
        'y': xy_generator(chain),
        'x': [chain.source_name]
    }
        
    for xy in x_iter[chain.orientation]:
        group_order = grouped_views[:]
        try:
            idxs = chain.content_of_axis.index(xy)
        except:
            idxs = 0
        if xy not in offset_dict.keys():
            offset_dict[xy] = OrderedDict()
        for view in view_generator(chain.views):
            if not view[0] in offset_dict[xy].keys():
                if view[0] in grouped_views:
                    idxv = chain.views.index(
                        group_order.pop(group_order.index(view[0]))
                    )
                else:
                    if (group_order==grouped_views) or len(group_order) == 0:
                        idxv = chain.views.index(view[0])
                    else:
                        bumped_views.append(view[0])
                if view[0] not in bumped_views:
                    if chain.view_lengths[idxs][idxv] > 0:
                        offset_dict[xy][view[0]] = 0
                        for idx in xrange(idxv):
                            if chain.views[idx] not in bumped_views:
                                temp = chain.view_lengths[idxs][idx]
                                offset_dict[xy][view[0]] += temp
                if len(bumped_views) > 0 and len(group_order) == 0:
                    for bv in bumped_views:
                        if bumped_views.index(bv) == 0:
                            pbv = next(reversed(offset_dict[xy]))
                            temp_a = offset_dict[xy][pbv]
                            pbv_index = chain.views.index(pbv)
                            temp_b = chain.view_lengths[idxs][pbv_index]
                            offset_dict[xy][bv] = temp_a + temp_b
                        else:
                            if chain.view_lengths[idxs][chain.views.index(bv)] > 0:
                                pbv = bumped_views[bumped_views.index(bv)-1]
                                temp_a = offset_dict[xy][pbv]
                                pbv_index = chain.views.index(pbv)
                                temp_b = chain.view_lengths[idxs][pbv_index]
                                offset_dict[xy][bv] = temp_a + temp_b
                    bumped_views = []

    return offset_dict

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
# def get_chain_gaps(cluster):
#     """
#     Creates a dictionary (chain names as keys) of dictionaries
#     (view names as keys) of missing view sizes for each chain.

#     Parameters
#     ----------
#     cluster : Cluster
#         quantipy chain object

#     Returns
#     -------
#     dictionary
#     """
#     res = dict()

#     view_counter = Counter(zip(*[
#         cluster[chain_name].views for chain_name in cluster.keys()
#     ]))

#     if sum([
#         0 
#         if view_counter[k] == cluster.__len__() 
#         else 1 
#         for k in view_counter.keys()
#     ]) == 0:
#         return None
#     else:
#         cluster_keys = cluster.keys()
#         for chain_name in cluster_keys:
#             chain = cluster[chain_name]
#             gap_views = list()
#             res[chain_name] = dict()
#             chains_rem = list(cluster_keys)
#             chains_rem.remove(chain_name)
#             for rem_name in chains_rem:
#                 chain_rem = cluster[rem_name]
#                 set_chain_rem = set(chain_rem.views)
#                 set_chain_views = set(chain.views)
#                 set_gap_views = set(gap_views)
#                 view_list = set_chain_rem - set_chain_views - set_gap_views
#                 for view in view_list:
#                     idxv = chain_rem.views.index(view)
#                     res[chain_name][view] = list()
#                     for x in chain.content_of_axis:
#                         if x in chain_rem.content_of_axis:
#                             idxs_rem = chain_rem.content_of_axis.index(x)
#                             res[chain_name][view].append(
#                                 chain_rem.view_lengths[idxs_rem][idxv]
#                             )
#                         else:
#                             res[chain_name][view].append(0)
#                     gap_views.append(view)
#         return res

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
def get_nest_levels(cluster):
    '''
    Returns the number of levels of nesting, or 0 if none.

    Parameters
    ----------
    cluster : quantipy.Cluster
        quantipy cluster object

    Returns
    -------
    int
    '''
    nest_levels = []
    for chain in chain_generator(cluster):
        if '>' in chain.source_name:
            nest_levels.append(len(chain.source_name.split('>'))-1)
    if nest_levels == []:
        return 0
    return max(nest_levels)

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
def validate_cluster_orientations(cluster):
    '''
    Make sure that the chains follow the rule:
        - All chains must have the same orientation, x or y.
    '''
    if len(set([
        cluster[chain_name].orientation 
        for chain_name in cluster.keys()
    ])) != 1:
        raise Exception(
            "Chain orientations must be consistent. Please review chain "
            "specification"
        )

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
def ExcelPainter(path_excel,
                 meta,
                 cluster,
                 grouped_views=None,
                 text_key=None,
                 annotations={},
                 display_names=None,
                 transform_names=None,
                 create_toc=False):
    '''
    Builds excel file (XLSX) from cluster, list of clusters, or 
    dictionary of clusters.

    Parameters
    ----------
    path_excel : str
        excel file path
    meta : dict
        metadata as dictionary used to paint datframes
    cluster : quantipy.Cluster /  list / dict
        container for cluster(s)
    grouped_views : list
        views to appear
    text_key : str
        language
    annotations : dict
        keys = cluster names, values = list of annotations for cells A1, A2, A3
    display_names : list
        list of axes <str> to append question numbers to labels 
    transform_names : dict
        keys as x/ y key names, values as names to display, if using 
        display_names arg
    create_toc : list | bool
        create a table for clusters (worksheets) in list, or all sheets if bool
    '''

    if path_excel.endswith('.xlsx'):
        path_excel = path_excel[:-5]
    elif path_excel.endswith('.xls'):
        path_excel = path_excel[:-4]

    if grouped_views is None:
        grouped_views = {}

    text_key = helpers.finish_text_key(meta, text_key)

    if display_names is None:
        display_names = []

    if create_toc is None:
        create_toc = list()

    workbook = Workbook(path_excel+'.xlsx', {'constant_memory': False})

    #create formats dictionary from STATIC_FORMATS dictionary
    formats = {
        key: workbook.add_format(STATIC_FORMATS[key]) 
        for key in STATIC_FORMATS.keys()
    }

    #render cluster
    names = []
    clusters = []
    if isinstance(cluster, Cluster):
        names.append(cluster.name)
        clusters.append(cluster)
    elif isinstance(cluster, list):
        for c in cluster:
            names.append(c.name)
            clusters.append(c)
    elif isinstance(cluster, dict):
        names_clusters_dict = cluster
        for sheet_name, c in cluster.iteritems():
            names.append(sheet_name)
            clusters.append(c)

    #create table of contents sheet
    toc_locs = []
    toc_names = []
    toc_labels = []  
    
    #transform banked chain specs to banked chains
    for cluster in clusters:
        for chain_name in cluster.keys():
            if cluster[chain_name].get('type')=='banked-chain':
                cluster[chain_name] = cluster.bank_chains(
                    cluster[chain_name], 
                    text_key)
        
    if create_toc:
        
        TOCsheet = workbook.add_worksheet('TOC')
        TOCsheet.write(2, 1, 'Table of Contents', formats['TOC-bold-14'])
        TOCsheet.set_column(0, 0, 1)
        
        if isinstance(create_toc, bool):
            sheet_idx = [i for i in xrange(len(clusters))]
        elif isinstance(create_toc, list):
            sheet_idx = [i for i, cl in enumerate(clusters) 
                         if cl.name in create_toc]
        else:
            raise Exception('create_toc arg must be of type bool/ list')
        
        for idx in sheet_idx:
            TOCsheet.set_column(
                1+sheet_idx.index(idx), 
                1+sheet_idx.index(idx), 
                10
            )
            TOCsheet.write(
                5, 
                1+sheet_idx.index(idx), 
                names[idx], 
                formats['TOC-bold-10']
            )
        TOCsheet.set_column(len(sheet_idx)+1, len(sheet_idx)+1, 1)
        TOCsheet.set_column(len(sheet_idx)+2, len(sheet_idx)+2, 125)
        TOCsheet.write(
            5, 
            len(sheet_idx)+2, 
            'Question Text', 
            formats['TOC-bold-center-10']
        )
        TOCsheet.freeze_panes(6,0)

    for sheet_name, cluster in zip(names, clusters):
        
        # get cluster's grouped views
        cluster_gv = grouped_views.get(sheet_name, list())

        #TOC
        if isinstance(create_toc, bool):
            toc_locs.append([])
            toc_names.append([])
            toc_labels.append([])
        else:
            if cluster.name in create_toc:
                toc_locs.append([])
                toc_names.append([])
                toc_labels.append([])
        
        #add worksheet
        worksheet = workbook.add_worksheet(sheet_name)
        
        #need a better way to identify "profile" tables...
        if all([
            isinstance(item, pd.DataFrame) 
            for item in cluster.itervalues()
        ]):           
            
            worksheet.set_row(5, 50)            
            
            for chain in chain_generator(cluster):                
                
#                 chain_format = chain.fillna('__NA__')
                chain_format = chain
                
                for column in chain_format.columns.tolist():
                    
                    frames = []
                    vmetas  = []
                    df_rows = []
                    df_cols = []
                    
                    worksheet.set_column(0, 0, 40)
                    
                    series = chain_format[column] 
                    
                    if meta['columns'][column]['type'] in ['single']:
                        categories = {
                            item['value']: item['text'][meta['lib']['default text']] 
                            for item in  meta['columns'][column]['values']
                        } 
                        series = series.map(categories.get, na_action='ignore')
                        series = series.fillna('__NA__')
                    elif meta['columns'][column]['type'] in ['delimited set']:
                        categories = {
                            str(item['value']): item['text'][meta['lib']['default text']] 
                            for item in  meta['columns'][column]['values']
                        }
                        series = series.str.split(';').apply(
                            pd.Series, 1
                        ).stack(dropna=False)
                        series = series.map(categories.get, 
                                            na_action='ignore').unstack()
#                         series.fillna('')
                        series[series.columns[0]] = series[series.columns[0]].str.cat(
                            [series[c] for c in series.columns[1:]], 
                            sep=', ',
                            na_rep=''
                        ).str.slice(0, -2)
                        series = series[series.columns[0]].replace(
                            to_replace='\, (?=\W|$)', value='', regex=True
                        )
                        series = series.replace(
                            to_replace='', value='__NA__'
                        )
                    else:
                        series = series.fillna('__NA__')
                        series = series.apply(unicoder)
                    
                    frames.append(series)

                    df_rows.append((7, 7+frames[-1].shape[0]))
                    
                    colmax = int(
                        0 
                        if worksheet.dim_colmax is None 
                        else worksheet.dim_colmax
                    )
                    df_cols.append((1+colmax, 1+colmax))
                    
                    worksheet.set_column(df_cols[-1][0], df_cols[-1][1], 10)
                    
                    try:
                        worksheet.write(
                            5, 
                            df_cols[-1][0], '. '.join(
                                [column, 
                                 meta['columns'][column]['text'][meta['lib']['default text']]]
                            ),
                            formats['y']
                        )
                    except:
                        worksheet.write(5, df_cols[-1][0], column, formats['y'])
                                            
                    paint_box(
                        worksheet=worksheet, 
                        frames=frames, 
                        format_dict=formats, 
                        rows=df_rows, 
                        cols=df_cols, 
                        metas=vmetas,
                        ceil=True,
                        floor=True
                    )

            worksheet.freeze_panes(6, 0)
                    
        else:
        
            #validate_cluster
            validate_cluster_orientations(cluster)
    
            #nesting sizes
            nest_levels = get_nest_levels(cluster)
    
            #initialise row and col indices
            current_position = {
                'x': ROW_INDEX_ORIGIN+(nest_levels*2),
                'y': COL_INDEX_ORIGIN,
                'test': COL_INDEX_ORIGIN+1
            }
            
            #update row index if freqs/ means tests?
            idxtestcol = 0
            testcol_maps = {}
            for chain in chain_generator(cluster):
                desc_val = chain.describe().values
                has_props_tests = any(
                    chain[d][f][x][y][v].is_propstest()
                    for d, f, x, y, v, _ in desc_val
                )
                has_means_tests = any(
                    chain[d][f][x][y][v].is_meanstest() 
                    for d, f, x, y, v, _ in desc_val
                )
                dk = chain.data_key
                fk = chain.filter
                if has_props_tests or has_means_tests:
                    if chain.orientation == 'y':
                        if chain.source_name != '@':
                            if chain.source_name not in testcol_maps:
                                testcol_maps[chain.source_name] = {}
                                for i in xrange(chain.source_length):
                                    pre = TEST_PREFIX[(idxtestcol+i) // 26]
                                    sur = TEST_SUFFIX[(idxtestcol+i) % 26]
                                    testcol_maps[chain.source_name][str(i+1)] = pre+sur
                                idxtestcol += chain.source_length
                    elif chain.orientation == 'x':
                        xk = chain.source_name
                        for idxc, column in enumerate(chain.content_of_axis):
                            if column != '@':
                                yk = column
                                vk = chain[dk][fk][xk][yk].keys()[0]
                                df = chain[dk][fk][xk][yk][vk].dataframe
                                if column not in testcol_maps:
                                    testcol_maps[column] = {}
                                    values = meta['columns'][column]['values']
                                    if helpers.is_mapped_meta(values):
                                        values = helpers.emulate_meta(meta, values)
                                    y_values = [int(v) for v in zip(*[c for c in df.columns])[1]]
                                    values = [
                                        [value for value in values if value['value']==v][0] 
                                        for v in y_values
                                    ]
                                    for i in xrange(chain.view_sizes[idxc][0][1]):
                                        pre = TEST_PREFIX[(idxtestcol+i) // 26]
                                        sur = TEST_SUFFIX[(idxtestcol+i) % 26]
                                        code = values[i]['value']
                                        # code = meta['columns'][column]['values'][i]['value']
                                        testcol_maps[column][str(code)] = pre+sur
                                idxtestcol += chain.view_sizes[idxc][0][1]
            testcol_labels = testcol_maps.keys()

            current_position['x'] += bool(testcol_maps)
    
            #dynamic coordinate map
            coordmap = {
                'x': {},
                'y': {}
            }
    
            #cluster offset
            # gaps = get_chain_gaps(cluster)
    
            #offset dict
            offset = dict()
    
            #column & headings size
            col_head_size = DEFAULT_Y_HEAD_HEIGHT
            set_row_height = True
    
            for chain in chain_generator(cluster):
                
                orientation = chain.orientation
                                
                #chain's view offset
                if not offset:
                    current_views = []
                else:
                    current_views = offset[offset.keys()[0]].keys()
    
                offset = get_view_offset(chain, offset, cluster_gv)
    
                new_views = set(offset[offset.keys()[0]].keys()) \
                    - set(current_views)
                
                if chain.source_name not in coordmap[orientation].keys():
    
                    if orientation == 'y':
                        coordmap['y'][chain.source_name] = [
                            current_position['y'],
                            current_position['y'] + chain.source_length - 1
                        ]
                    elif orientation == 'x':
                        if chain.source_name not in coordmap['x'].keys():
                            coordmap['x'][chain.source_name] = OrderedDict()
                            for view in offset[chain.source_name].keys():
                                idxv = chain.views.index(view)
                                coordmap['x'][chain.source_name][view] = [
                                    current_position['x'] \
                                        + offset[chain.source_name][view],
                                    current_position['x'] \
                                        + offset[chain.source_name][view] \
                                        + chain.view_lengths[0][idxv] \
                                        - 1
                                ]
    
                for xy in xy_generator(chain):
    
                    if orientation == 'y':
                        x, y = xy, chain.source_name
                    elif orientation == 'x':
                        y, x = xy, chain.source_name
    
                    idxs = chain.content_of_axis.index(xy)
    
                    # fill gaps
                    # if orientation == 'x':
                    #     if chain.name in gaps.keys():
                    #         gap = 0 if idxs == 0 else gap + sum([
                    #             gap_size[idxs-1] 
                    #             for gap_size in gaps[chain.name].values()
                    #         ])
    
                    #fill xs' ceil_floor
                    
                    ceiling, _ = min(offset[x].iteritems(), key=lambda o: o[1])
                    floor, _ = max(offset[x].iteritems(), key=lambda o: o[1])
                        
                    if orientation == 'y':
                        if x not in coordmap['x'].keys():
                            coordmap['x'][x] = OrderedDict()
                            for view in offset[x].keys():
                                idxv = chain.views.index(view)
                                coordmap['x'][x][view] = [
                                    current_position['x'] \
                                        + offset[x][view], \
                                        # + gap,
                                    current_position['x'] \
                                        + offset[x][view] \
                                        + chain.view_lengths[idxs][idxv] \
                                        # + gap \
                                        - 1
                                ]
                        else:
                            for view in offset[x].keys():
                                if view not in coordmap['x'][x]:
                                    idxv = chain.views.index(view)
                                    coordmap['x'][x][view] = [
                                        coordmap['x'][x].values()[-1][1] \
                                            + 1,
                                        coordmap['x'][x].values()[-1][1] \
                                            + chain.view_lengths[idxs][idxv]
                                    ]
                    elif orientation == 'x':
                        if y not in coordmap['y'].keys():
                            idxs = chain.content_of_axis.index(y)                            
                            coordmap['y'][y] = [
                                current_position['y'],
                                current_position['y'] \
                                    + chain.view_sizes[idxs][0][1] \
                                    - 1
                            ]
    
                    #loop views
                    
                    for views in view_generator(offset[x].keys(), cluster_gv):
                        
                        frames = []
                        vmetas  = []
                        vlevels = []
                        df_rows = []
                        df_cols = []
    
                        for idx, v in enumerate(views):
                            
                            view = chain[chain.data_key][chain.filter][x][y][v]

                            if not isinstance(view, qp.View):
                                raise Exception(
                                    ('\nA view in the chains, {vk}, '
                                     'does not exist in the stack for...\n'
                                     'cluster={cluster}\ndata_key={dk}\n'
                                     'filter={fk}\nx={xk}\ny={yk}\n').format(
                                        cluster=cluster.name,
                                        vk=v,
                                        dk=chain.data_key,
                                        fk=chain.filter,
                                        xk=x,
                                        yk=y
                                    )
                                )
                                
                            vmetas.append(view.meta())

                            if view.is_propstest():
                                vlevels.append(view.is_propstest())
                            elif view.is_meanstest():
                                vlevels.append(view.is_meanstest())
                            else:
                                vlevels.append(None)
                                
                            if view.meta()['agg']['method'] == 'frequency':
                                agg_name = view.meta()['agg']['name']
                                if agg_name in ['cbase', 'c%', 'counts']:
                                    df = helpers.paint_dataframe(
                                        df=view.dataframe.copy(),
                                        meta=meta, 
                                        text_key=text_key,
                                        display_names=display_names,
                                        transform_names=transform_names
                                    )
                                else:
                                    df = view.dataframe.copy()
                            else:
                                df = view.dataframe.copy()
    
                            #write column test labels
                            if 'test' in view.meta()['agg']['method']:
                                if view.meta()['y']['name'] in testcol_labels:
                                    tdf = view.dataframe
                                    y_values = [int(v) for v in zip(*[c for c in tdf.columns])[1]]
                                    code_idx = testcol_labels.index(
                                        view.meta()['y']['name']
                                    )
                                    for i, code in enumerate(y_values):
                                        worksheet.write(
                                            ROW_INDEX_ORIGIN+(nest_levels*2)-1, 
                                            current_position['test']+i,
                                            testcol_maps[view.meta()['y']['name']][str(code)], 
                                            formats['tests']
                                        )
                                    current_position['test'] += view.meta()['shape'][1]
                                    testcol_labels.remove(
                                        view.meta()['y']['name']
                                    )
    
                            #append frame to frames
                            frames.append(df)
    
                            #get dataframe and it's coordinates
                            df_rows.append(
                                coordmap['x'][x][view.meta()['agg']['fullname']]
                            )
                            df_cols.append(coordmap['y'][y])
                                    
                        #write data
                        is_ceil = vmetas[0]['agg']['fullname'] == ceiling
                        is_floor = vmetas[-1]['agg']['fullname'] == floor
                    
                        if view.meta()['y']['name'] in testcol_maps:
                            paint_box(
                                worksheet, 
                                frames, 
                                formats, 
                                df_rows, 
                                df_cols, 
                                vmetas, 
                                is_ceil, 
                                is_floor, 
                                testcol_maps[view.meta()['y']['name']]
                            )
                        else:
                            paint_box(
                                worksheet, 
                                frames, 
                                formats, 
                                df_rows, 
                                df_cols, 
                                vmetas, 
                                is_ceil, 
                                is_floor
                            )

                        x_name, y_name, shortname, \
                        fullname, text, method, is_weighted = (
                            vmetas[idx]['x']['name'],
                            vmetas[idx]['y']['name'],
                            vmetas[idx]['agg']['name'],
                            vmetas[idx]['agg']['fullname'],
                            vmetas[idx]['agg']['text'],
                            vmetas[idx]['agg']['method'],
                            vmetas[idx]['agg']['is_weighted']
                        )
                        relation = fullname.split('|')[2]
                        
                        #write y labels - NESTING WORKING FOR 2 LEVELS. NEEDS TO WORK FOR N LEVELS.
                        y_name = 'Total' if y_name == '@' else y_name
                            
                        if y_name == 'Total':
                            if coordmap['x'][x_name][fullname][0] == ROW_INDEX_ORIGIN+(nest_levels*2)+bool(testcol_maps):
                                #write column label(s) - multi-column y subaxis
                                worksheet.set_column(
                                    df_cols[idx][0], 
                                    df_cols[idx][1], 
                                    10
                                )
                                worksheet.merge_range(
                                    ROW_INDEX_ORIGIN-3, 
                                    df_cols[idx][0], 
                                    ROW_INDEX_ORIGIN+(nest_levels*2)-2, 
                                    df_cols[idx][1], 
                                    y_name, 
                                    formats['y']
                                )
                            if bool(testcol_maps):
                                worksheet.write(
                                    ROW_INDEX_ORIGIN+(nest_levels*2)-1, 
                                    1, 
                                    '', 
                                    formats['tests']
                                )
                        else:
                            if coordmap['x'][x_name][fullname][0] == ROW_INDEX_ORIGIN+(nest_levels*2)+bool(testcol_maps):
                                labels = helpers.get_unique_level_values(df.columns)
                                if max([len(lab) for lab in labels[-1]]) > Y_ROW_WRAP_TRIGGER:
                                    set_row_height = False
                                add_size = (((len(labels[0][0])-DEFAULT_Y_HEAD_HEIGHT)//Y_HEAD_WRAP_TRIGGER)+1)-len(labels[1])
                                if add_size > 0:
                                    col_head_size += (add_size*Y_HEAD_WRAP_TRIGGER)
                                if nest_levels == 0:
                                    write_column_labels(
                                        worksheet, 
                                        labels, 
                                        formats['y'],
                                        ROW_INDEX_ORIGIN-3, 
                                        df_cols[idx]
                                    )
                                elif nest_levels > 0:
                                    write_column_labels(worksheet, 
                                        labels, 
                                        formats['y'], 
                                        ROW_INDEX_ORIGIN-3, 
                                        df_cols[idx], 
                                        nest_levels
                                    )
    
                        #write x labels
                        if df_cols[0][0] == COL_INDEX_ORIGIN:
                            if fullname == ceiling:
                                
                                write_question_label(
                                    worksheet, 
                                    df.index[0][0], 
                                    formats['x_left_bold'], 
                                    df_rows[idx][0]-1,
                                    COL_INDEX_ORIGIN-1
                                )
                                
                                if create_toc:
                                    toc_locs[-1].append(
                                        (df_rows[idx][0]-1,  COL_INDEX_ORIGIN-1)
                                    )
                                    if transform_names:
                                        toc_names[-1].append(
                                            transform_names.get(x_name,
                                                                x_name))
                                    else:    
                                        toc_names[-1].append(x_name)
                                    if 'x' in display_names:                                        
                                        toc_labels[-1].append(
                                            df.index[0][0].split('. ')[1]
                                        )
                                    else:
                                        toc_labels[-1].append(df.index[0][0])

                        cond_1 = df_cols[0][0] == COL_INDEX_ORIGIN
                        cond_2 = fullname in new_views
                        if cond_1 or cond_2:                                    
                            if shortname == 'cbase':
                                if chain.has_weighted_views and not is_weighted:
                                    if len(text) > 0:
                                        labels = [''.join(['Unweighted ', 
                                                           text.lower()])]
                                    else:
                                        labels = [fullname]
                                    write_category_labels(
                                        worksheet, 
                                        labels, 
                                        formats['x_right_bold'], 
                                        df_rows[idx][0], 
                                        COL_INDEX_ORIGIN-1, 
                                        set_heights=True
                                    )
                                else:
                                    if len(text) > 0:
                                        if not chain.base_text is None:
                                            text = '{}: {}'.format(
                                                text,
                                                helpers.get_text(
                                                    chain.base_text,
                                                    text_key,
                                                    'x'))
                                        labels = [text]
                                    else:
                                        labels = [fullname]
                                    write_category_labels(
                                        worksheet, 
                                        labels, 
                                        formats['x_right_bold'], 
                                        df_rows[idx][0], 
                                        COL_INDEX_ORIGIN-1, 
                                        set_heights=True
                                    )
                            else:                            
                                if (vmetas[0]['agg']['method'] in ['descriptives'] or 
                                    (vmetas[0]['agg']['method'] in ['frequency'] and len(relation) > 0)):
                                    if len(frames) > 1:
                                        labels = []
                                        labels_written = []
                                        for idxdf, df in enumerate(frames):
                                            if vmetas[idxdf]['agg']['method'] == 'coltests':
                                                format_key = 'x_right_tests'
                                                labels = [vlevels[idxdf] for _ in df.index]
                                            else:
                                                format_key = 'x_right_stats'
                                                if len(vmetas[idxdf]['agg']['text']) > 0:
                                                    labels = [vmetas[idxdf]['agg']['text']]
                                                else:
                                                    labels = df.index.get_level_values(1)
                                            if all([label not in labels_written for label in labels]):
                                                write_category_labels(
                                                    worksheet, 
                                                    labels, 
                                                    formats[format_key], 
                                                    df_rows[0][0]+idxdf, 
                                                    COL_INDEX_ORIGIN-1, 
                                                    len(frames)
                                                )
                                                labels_written.extend(labels)
                                    else:
                                        format_key = 'x_right_stats'
                                        if len(frames[0].index) == 1:
                                            if len(vmetas[0]['agg']['text']) > 0:
                                                labels = [vmetas[0]['agg']['text']] 
                                            else:
                                                labels = [vmetas[0]['agg']['fullname']]
                                        else:
                                            labels = df.index.get_level_values(1)                                           
                                        write_category_labels(
                                            worksheet, 
                                            labels, 
                                            formats[format_key], 
                                            df_rows[0][0], 
                                            COL_INDEX_ORIGIN-1, 
                                            len(frames), 
                                            set_heights=True
                                        )
                                else:
                                    freq_view = False
                                    labels = []
                                    for idxdf, df in enumerate(frames):
                                        if vmetas[idxdf]['agg']['method'] == 'coltests':
                                            format_key = 'x_right_tests'
                                            labels = [vlevels[idxdf] for _ in df.index]
                                        elif vmetas[idxdf]['agg']['method'] == 'descriptives':
                                            format_key = 'x_right'
                                            labels = [df.index[idxf][idxf] for _ in df.index]
                                        else:
                                            format_key = 'x_right'
                                            if idxdf == 0 or (idxdf > 0 and not freq_view):
                                                freq_view = True
                                                labels = df.index.get_level_values(1)
                                            else:
                                                continue
                                        write_category_labels(
                                            worksheet, 
                                            labels, 
                                            formats[format_key], 
                                            df_rows[0][0]+idxdf, 
                                            COL_INDEX_ORIGIN-1, 
                                            len(frames)
                                        )
    
                    #increment row (only first occurrence of each x)
                    if orientation == 'y':
                        current_position['x'] += sum(
                            chain.view_lengths[idxs]
                        ) + 1
                    elif orientation == 'x':
                        current_position['y'] += (
                            coordmap['y'][xy][1]-coordmap['y'][xy][0]+1
                        )
    
                #increment col
                if orientation == 'y':
                    current_position['y'] += chain.source_length
                    
                elif orientation == 'x':
                    current_position['x'] += sum(chain.view_lengths[0])+1
            
            #set column widths
            worksheet.set_column(0, 0, 40)
    
            #set y axis height
            worksheet.set_row(ROW_INDEX_ORIGIN-3, col_head_size)
            if set_row_height:
                worksheet.set_row(
                    ROW_INDEX_ORIGIN+(nest_levels*2)-2, 
                    DEFAULT_Y_ROW_HEIGHT
                )        
            
            #freeze panes
            worksheet.freeze_panes(
                ROW_INDEX_ORIGIN+(nest_levels*2)+bool(testcol_maps)-1, 
                COL_INDEX_ORIGIN+1
            )

            
    #download image
    if IMG_URL:
        try:
            img_url_full = '\\'.join(
                [os.path.dirname(quantipy.__file__),
                'core\\builds\\excel\\formats',
                 IMG_URL]
            )
            if os.path.exists(img_url_full):
                img = Image.open(img_url_full)
                img.thumbnail(IMG_SIZE, Image.ANTIALIAS)
                img.save(os.path.basename(img_url_full))
                path_img = os.path.basename(img_url_full)
            else:
                response = requests.get(IMG_URL)
                img = Image.open(BytesIO(response.content))
                img.thumbnail(IMG_SIZE, Image.ANTIALIAS)
                img.save('img.png')
                path_img = 'img.png'
        except:
            pass

    #post-process non-TOC sheets
    for worksheet in workbook.worksheets_objs:
            
            #hide gridlines
            worksheet.hide_gridlines(2)

            if not worksheet.name == 'TOC':

                #write annotations to cells A1, A2, A3
                if annotations.get(worksheet.name):
                    worksheet.write(
                        0, 
                        0, 
                        annotations[worksheet.name][0], 
                        formats['x_left_bold']
                    )
                    worksheet.write(
                        1, 
                        0,
                        annotations[worksheet.name][1],
                        formats['x_left_bold']
                    )
                    worksheet.write(
                        2,
                        0,
                        annotations[worksheet.name][2],
                        formats['x_left_bold']
                    )

                #insert image
                try:
                    worksheet.insert_image(
                        3,
                        0,
                        path_img,
                        {'x_offset': 15, 'y_offset': 10}
                    )
                except:
                    pass

    #finish writing TOC 
    write_labels = all(name_list == toc_names[0] for name_list in toc_names)
    for i in xrange(len(toc_names)):
        for q in xrange(len(toc_names[i])):
            TOCsheet.write(
               6+q,
               1+i, 
               'internal:%s!%s' % (
                    names[i], 
                    xl_rowcol_to_cell(toc_locs[i][q][0], toc_locs[i][q][1])
                ),
               formats['TOC-url']
            ) 
            TOCsheet.write(
                6+q, 
                1+i, 
                toc_names[i][q], 
                formats['TOC-url']
            ) 
            if write_labels:
                if i == len(sheet_idx)-1:
                    TOCsheet.write(
                        6+q, 
                        3+i, 
                        toc_labels[i][q], 
                        formats['TOC-10']
                    )
        
    #close excel file
    workbook.close()
