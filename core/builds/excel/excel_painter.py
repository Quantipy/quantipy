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
from quantipy.core.builds.excel.formats.xlsx_formats import XLSX_Formats
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
import cPickle
import itertools

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
TEST_SUFFIX = list(ascii_uppercase)
TEST_PREFIX = ['']+list(ascii_uppercase)

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
def paint_box(worksheet, frames, format_dict, rows, cols, metas, formats_spec,
              has_weighted_views=False, y_italicise=dict(), ceil=False, floor=False,
              testcol_map=None, is_array=False, array_views=None):
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
        Whether ceiling view (this is overwritten for array tables)
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

        if is_array:
            ceil = (i // frames[idxf].shape[1])==0
            floor = (i // frames[idxf].shape[1])==frames[0].shape[0]-1

        box_coord = [coord[0] - coords[0][0], coord[1] - coords[0][1]]

        # pick cell format
        format_name = ''

        if len(metas) == 0:
            method = 'dataframe_columns'
        else:
            fullname, method, is_weighted, is_dummy = (
                metas[idxf]['agg']['fullname'],
                metas[idxf]['agg']['method'],
                metas[idxf]['agg']['is_weighted'],
                metas[idxf]['agg'].get('is_dummy', False)
            )
            _, _, relation, rel_to, _, shortname  = fullname.split('|')
            is_totalsum =  metas[idxf]['agg']['name'] in ['counts_sum', 'c%_sum']

        # cell position
        if is_array:
            if metas[0]['agg']['fullname']==array_views[0]:
                format_name = 'left-'
            elif metas[0]['agg']['fullname']==array_views[-1]:
                format_name = 'right-'
        else:
            if i % csize == 0:
                format_name = 'left-'
            elif i % csize == (csize - 1) or (cols[idxf][0] == cols[idxf][1]):
                format_name = format_name + 'right-'

        if format_name == '':
            format_name = format_name + 'interior-'

        if ceil:
            if is_array:
                format_name = format_name + 'top-'
            else:
                if i < (csize):
                    format_name = format_name + 'top-'
        if floor:
            if is_array:
                format_name = format_name + 'bottom-'
            else:
                if i >= ((rsize * csize) - csize):
                    format_name = format_name + 'bottom-'

        # additional format spec
        if method == 'dataframe_columns':

            format_name = format_name + 'STR'

        else:

            # background color (frequency/ coltests) / top border Totalsum
            if is_array:
                if (i // frames[idxf].shape[1]) % 2 == 0:
                    format_name = format_name + 'bg-'
            else:
                cond_1 = method in ['frequency', 'coltests'] and relation == ':'
                cond_2 = method in ['default']
                if cond_1 or cond_2:
                    if not shortname in ['cbase']:
                        if box_coord[0] == 0:
                            format_name = format_name + 'frow-bg-'
                        elif (box_coord[0] // len(frames)) % 2 == 0:
                            format_name = format_name + 'bg-'

            # first row (coltests - means)
            if method == 'coltests' and relation != ':':
                if box_coord[0] == 0:
                    format_name = format_name + 'frow-'

            # choose view format type
            # base
            if shortname == 'cbase':
                if is_array:
                    format_name = format_name + 'N'
                else:
                    if not ceil:
                        if is_weighted:
                            format_name = format_name + 'frow-BASE'
                        else:
                            if has_weighted_views:
                                format_name = format_name + 'frow-UBASE'
                            else:
                                format_name = format_name + 'frow-BASE'
                    else:
                        if is_weighted:
                            format_name = format_name + 'BASE'
                        else:
                            if has_weighted_views:
                                format_name = format_name + 'UBASE'
                            else:
                                format_name = format_name + 'BASE'

            # frequency
            elif method == 'frequency':

                # counts
                if rel_to == '':

                    if relation == ':' or is_array:
                        format_name = format_name + 'N'

                    elif is_totalsum:
                        if is_array:
                            format_name = format_name + 'N'
                        elif is_dummy:
                            format_name = format_name + 'N'
                        else:
                            if 'bottom' in format_name:
                                format_name = format_name + 'N'
                            else:
                                format_name = format_name + 'frow-N'

                    # complex logics
                    else:
                        if len(frames) == 1 or is_array:
                            format_name = format_name + 'N-NET'
                        else:
                            if idxf == 0:
                                format_name = format_name + 'frow-N-NET'
                            elif idxf == len(frames)-1:
                                format_name = format_name + 'brow-N-NET'
                            else:
                                format_name = format_name + 'mrow-N-NET'

                # %
                elif rel_to in ['x', 'y']:

                    if relation == ':' or is_array:
                        format_name = format_name + 'PCT'

                    elif is_totalsum:
                        if is_dummy:
                            format_name = format_name + 'PCT'
                        else:
                            if 'bottom' in format_name:
                                format_name = format_name + 'PCT'
                            else:
                                format_name = format_name + 'frow-PCT'

                    # complex logics
                    else:
                        if len(frames) == 1 or is_array:
                            format_name = format_name + 'PCT-NET'
                        else:
                            if idxf == 0:
                                format_name = format_name + 'frow-PCT-NET'
                            elif idxf == len(frames)-1:
                                format_name = format_name + 'brow-PCT-NET'
                            else:
                                format_name = format_name + 'mrow-PCT-NET'
            # descriptvies
            elif method == 'descriptives':
                if is_array:
                    format_name = format_name + 'DESCRIPTIVES-XT'
                elif len(frames) == 1:
                    format_name = format_name + 'DESCRIPTIVES'
                else:
                    if idxf == 0:
                        format_name = format_name + 'frow-DESCRIPTIVES'
                    elif idxf == len(frames)-1:
                        format_name = format_name + 'brow-DESCRIPTIVES'
                    else:
                        format_name = format_name + 'mrow-DESCRIPTIVES'

            # coltests
            elif method == 'coltests':
                format_name = format_name + 'TESTS'

            # default
            elif method == 'default':
                format_name = format_name + 'DEFAULT'

            # method not found...
            else:
                raise Exception(
                    "View method not recognised...\nView: {}\nMethod: {}" % (
                        metas[idxf]['agg']['fullname'],
                        method))

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

            # post-process cell data (if not dummy data)
            if not is_dummy:

                # convert numpy.inf
                # if data == np.inf:
                #     data = str(np.inf)

                # % - divide data by 100 for formatting in Excel
                if rel_to in ['x', 'y'] and not method in ['coltests',
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
                                data += testcol_map[letter] + formats_spec.test_seperator
                            data = data[:-len(formats_spec.test_seperator)]

                # replace 0 with char
                try:
                    if np.isclose([data], [0]):
                        if method == 'frequency':
                            # data = FREQUENCY_0_REPR
                            data = formats_spec.frequency_0_repr
                        elif method == 'descriptives':
                            # data = DESCRIPTIVES_0_REPR
                            data = formats_spec.descriptives_0_repr
                except:
                    pass

            # Check data for NaN and replace with '-'
            if not isinstance(data, (str, unicode)):
                if np.isnan(data) or np.isinf(data):
                    data = '-'

            # Italicise?
            if not format_name.endswith(('STR', 'TESTS')):
                if y_italicise.get(coord[1]):
                    x_ranges = y_italicise[coord[1]]
                    for x_range in x_ranges:
                        if coord[0] in range(*x_range):
                            format_name = format_name + '-italic'

        # write data
        try:
            worksheet.write(
                coord[0],
                coord[1],
                data,
                format_dict[format_name]
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
                        format_name,
                        fullname,
                        e
                    )]
                )
            )

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
def set_row_height(worksheet,
                   row_start,
                   row_stop,
                   row_height,
                   text_size=1):
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
        worksheet.set_row(row, row_height*text_size)

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
def write_column_labels(worksheet, labels, existing_format, row,
                        cols, levels=0, is_array=False):
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
            if not is_array:
                if (cols[0] == cols[1]):
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
def write_category_labels(worksheet,
                          labels,
                          existing_format,
                          row,
                          col,
                          row_height=None,
                          row_wrap_trigger=None,
                          group_size=1,
                          set_heights=False):
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
            try:
                lab_len = len(lab)
            except:
                lab_len = len(str(lab))
            if lab_len < row_wrap_trigger:
                if group_size > 1 and set_heights:
                    set_row_height(
                        worksheet=worksheet,
                        row_start=row+(idx*group_size),
                        row_stop=row+(idx*group_size)+(group_size-1),
                        row_height=row_height
                    )
                else:
                    set_row_height(
                        worksheet=worksheet,
                        row_start=row+(idx*group_size),
                        row_stop=row+(idx*group_size),
                        row_height=row_height
                    )
            elif group_size > 1 and set_heights:
                set_row_height(
                    worksheet=worksheet,
                    row_start=row+(idx*group_size)+1,
                    row_stop=row+(idx*group_size)+(group_size-1),
                    row_height=row_height
                )
            if isinstance(lab, float):
                worksheet.write_number(
                    row+(idx*group_size),
                    col,
                    lab,
                    existing_format
                )
            else:
                worksheet.write(
                    row+(idx*group_size),
                    col,
                    lab,
                    existing_format
                )
    except:
        pass

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
def write_question_label(worksheet,
                         label,
                         existing_format,
                         row,
                         col,
                         row_height,
                         row_wrap_trigger):
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
    if len(label) < row_wrap_trigger:
        set_row_height(worksheet, row, row, row_height)
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
            chained_grouped_views = list(itertools.chain(*grouped_views))
            non_grouped_views = [x for x in chain_views
                                 if not x in chained_grouped_views]
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
def get_view_offset(chain, offset_dict, grouped_views=[], dummy_tests=False):
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
    key_last = ''
    idx_last = 0
    len_last = 0
    dummy_rows = 0

    bumped_views = []

    x_iter = {
        'y': xy_generator(chain),
        'x': [chain.source_name]
    }

    view_sizes = chain.view_sizes()
    view_lengths = chain.view_lengths()

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
                    if view_lengths[idxs][idxv] > 0:
                        offset_dict[xy][view[0]] = 0
                        for idx in xrange(idxv):
                            if chain.views[idx] not in bumped_views:
                                temp = view_lengths[idxs][idx]
                                offset_dict[xy][view[0]] += temp
                if len(bumped_views) > 0 and len(group_order) == 0:
                    for bv in bumped_views:
                        if bumped_views.index(bv) == 0:
                            pbv = next(reversed(offset_dict[xy]))
                            temp_a = offset_dict[xy][pbv]
                            pbv_index = chain.views.index(pbv)
                            temp_b = view_lengths[idxs][pbv_index]
                            offset_dict[xy][bv] = temp_a + temp_b
                        else:
                            if view_lengths[idxs][chain.views.index(bv)] > 0:
                                pbv = bumped_views[bumped_views.index(bv)-1]
                                temp_a = offset_dict[xy][pbv]
                                pbv_index = chain.views.index(pbv)
                                temp_b = view_lengths[idxs][pbv_index]
                                offset_dict[xy][bv] = temp_a + temp_b
                    bumped_views = []
                elif len(bumped_views) > 0:
                    for bv in bumped_views:
                        pbv = next(reversed(offset_dict[xy]))
                        temp_a = offset_dict[xy][pbv]
                        pbv_index = chain.views.index(pbv)
                        temp_b = view_lengths[idxs][pbv_index]
                        offset_dict[xy][bv] = temp_a + temp_b
                    bumped_views = []

        if dummy_tests:

            exempt = []
            tests_loc = {'f': None, 'd': None}
            for group in grouped_views:
                v_type = group[0].split('|')[1][0]
                has_tests = any(v.split('|')[1].startswith('t') for v in group)
                if has_tests: exempt.extend(group)
                if not tests_loc[v_type]:
                    if has_tests:
                        for idx, view in enumerate(group):
                            if view.split('|')[1].startswith('t'):
                                tests_loc[v_type] = idx
                                continue
            dummy_rows = 0
            for vk in offset_dict[xy]:
                if not vk.endswith('cbase') and vk not in exempt:
                    v_type = vk.split('|')[1][0]
                    if vk in list(itertools.chain(*grouped_views)):
                        for group in grouped_views:
                            if vk in group:
                                if group.index(vk) == tests_loc[v_type]-1:
                                    idxvk = chain.views.index(vk)
                                    vk_size = view_lengths[idxs][idxvk]
                                    for ovk in offset_dict[xy].keys()[idxvk+1:]:
                                        offset_dict[xy][ovk] += vk_size
                    else:
                        idxvk = chain.views.index(vk)
                        vk_size = view_lengths[idxs][idxvk]
                        for ovk in offset_dict[xy].keys()[idxvk+1:]:
                            offset_dict[xy][ovk] += vk_size

    return offset_dict

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
def verify_grouped_views(grouped_views):

    if grouped_views is None:
        return True
    else:
        if not isinstance(grouped_views, (dict, OrderedDict)):
            return False
        for name in grouped_views.keys():
            if not isinstance(grouped_views[name], list):
                return False
            for block in grouped_views[name]:
                if not isinstance(block, list):
                    return False
                for vk in block:
                    if not isinstance(vk, (str, unicode)):
                        return False

        return True

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
def ExcelPainter(path_excel,
                 meta,
                 cluster,
                 grouped_views=None,
                 text_key=None,
                 annotations={},
                 display_names=None,
                 transform_names=None,
                 table_properties=None,
                 italicise_level=None,
                 create_toc=False):
    """
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
    table_properties : dict
        keys as format properties, values as change from default
    italicise_level : int
        italicise columns if base (unweighted) is below this threshold
    create_toc : list | bool
        create a table for clusters (worksheets) in list, or all sheets if bool
    """

    if path_excel.endswith('.xlsx'):
        path_excel = path_excel[:-5]
    elif path_excel.endswith('.xls'):
        path_excel = path_excel[:-4]

    if not verify_grouped_views(grouped_views):
        raise ValueError(
            "Either the value passed to 'grouped_views' or its structure is not"
            " valid. Please check it and again. The correct form is:"
            " {'name': [[vk1, vk2], [vk3, vk4], ...]}")

    if grouped_views is None:
        grouped_views = {}

    text_key = helpers.finish_text_key(meta, text_key)

    if display_names is None:
        display_names = []

    if create_toc is None:
        create_toc = list()

    workbook = Workbook(path_excel+'.xlsx', {'constant_memory': False})

    #create formats dictionary from STATIC_FORMATS dictionary
    if table_properties:
        formats_spec = XLSX_Formats(properties=table_properties)
    else:
        formats_spec = XLSX_Formats()
    formats_spec.create_formats_dict()
    formats = {
        key: workbook.add_format(formats_spec.format_dict[key])
        for key in formats_spec.format_dict.keys()}

    #create special formats dictionary for array tables
    if table_properties:
        formats_spec_arrays = XLSX_Formats(properties=table_properties)
    else:
        formats_spec_arrays = XLSX_Formats()
    formats_spec_arrays.set_bold_y(True)
    formats_spec_arrays.create_formats_dict()
    formats_arrays = {
        'array-{}'.format(key): workbook.add_format(formats_spec_arrays.format_dict[key])
        for key in formats_spec_arrays.format_dict.keys()}

    # Set starting row and column
    row_index_origin = formats_spec.start_row_idx+1
    col_index_origin = formats_spec.start_column_idx-1

    # Check the starting row/ column are not under the minimum
    # else apply the minimum
    if row_index_origin < 3: row_index_origin = 3
    if col_index_origin < 1: col_index_origin = 1

    # Render cluster
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

        # TOC
        if isinstance(create_toc, bool):
            toc_locs.append([])
            toc_names.append([])
            toc_labels.append([])
        else:
            if cluster.name in create_toc:
                toc_locs.append([])
                toc_names.append([])
                toc_labels.append([])

        # add worksheet
        worksheet = workbook.add_worksheet(sheet_name)

        #need a better way to identify "profile" tables...
        if all([
            isinstance(item, pd.DataFrame)
            for item in cluster.itervalues()
        ]):

            worksheet.set_row(4, formats_spec.y_header_height)
            worksheet.set_row(5, formats_spec.y_row_height)

            for chain in chain_generator(cluster):

#                 chain_format = chain.fillna('__NA__')
                chain_format = chain

                has_multiindex = any([
                    isinstance(idx, pd.MultiIndex)
                    for idx in [chain.index, chain.columns]])

                if has_multiindex:
                    df = helpers.paint_dataframe(meta, chain)
                    df.fillna('-', inplace=True)

                for column in chain_format.columns.tolist():

                    frames = []
                    vmetas  = []
                    df_rows = []
                    df_cols = []

                    worksheet.set_column(0, 0, 40)

                    if has_multiindex:
                        series = chain_format[column[0]][column[1]]
                    else:
                        series = chain_format[column]

                    if not has_multiindex:
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

                    colmax = int({True: 1, False: 0}.get(has_multiindex)
                                 if worksheet.dim_colmax in [None, 0]
                                 else worksheet.dim_colmax)
                    df_cols.append((1+colmax, 1+colmax))

                    if not has_multiindex:

                        worksheet.set_column(df_cols[-1][0],
                                             df_cols[-1][1],
                                             formats_spec.column_width_str)

                        try:
                            tk = meta['lib']['default text']
                            column_text = '. '.join(
                                [column,
                                 meta['columns'][column]['text'][tk]])
                            meta['columns'][column]['text'][tk]
                            worksheet.merge_range(4, df_cols[-1][0],
                                                  5, df_cols[-1][0],
                                                  column_text, formats['y'])
                        except:
                            worksheet.merge_range(4, df_cols[-1][0],
                                                  5, df_cols[-1][0],
                                                  column, formats['y'])

                    paint_box(
                        worksheet=worksheet,
                        frames=frames,
                        format_dict=formats,
                        rows=df_rows,
                        cols=df_cols,
                        metas=vmetas,
                        formats_spec=formats_spec,
                        ceil=True,
                        floor=True
                    )

                if has_multiindex:

                    worksheet.set_column(0, 0, 15)
                    worksheet.set_column(1, 1, 10)

                    lrow = 0
                    for level in df.index.levels[0]:
                        worksheet.write(7+lrow, 0, level, formats['x_left_bold'])
                        for idx in df.loc[level].index:
                            worksheet.write(7+lrow, 1, idx, formats['x_right'])
                            lrow += 1

                    lcol = 0
                    for level in df.columns.levels[0]:
                        worksheet.merge_range(4, 2+lcol,
                                              4, 2+lcol+len(df.loc[:, level].columns)-1,
                                              level, formats['y'])
                        for idx in df.loc[:, level].columns:
                            worksheet.write(5, 2+lcol, idx, formats['y'])
                            lcol += 1

            worksheet.freeze_panes(6, 0)

        else:

            #validate_cluster
            validate_cluster_orientations(cluster)

            #nesting sizes
            nest_levels = get_nest_levels(cluster)

            #initialise row and col indices
            current_position = {
                'x': row_index_origin+(nest_levels*2),
                'y': col_index_origin,
                'test': col_index_origin+1
            }

            #update row index if freqs/ means tests?
            idxtestcol = 0
            testcol_maps = {}
            for chain in chain_generator(cluster):

                view_sizes = chain.view_sizes()
                view_keys = chain.describe()['view'].values.tolist()
                has_props_tests = any([
                    '|t.props' in vk
                    for vk in view_keys])
                has_means_tests = any([
                    '|t.means' in vk
                    for vk in view_keys])
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
                                    for i in xrange(view_sizes[idxc][0][1]):
                                        pre = TEST_PREFIX[(idxtestcol+i) // 26]
                                        sur = TEST_SUFFIX[(idxtestcol+i) % 26]
                                        code = values[i]['value']
                                        # code = meta['columns'][column]['values'][i]['value']
                                        testcol_maps[column][str(code)] = pre+sur
                                idxtestcol += view_sizes[idxc][0][1]
            testcol_labels = testcol_maps.keys()

            current_position['x'] += bool(testcol_maps)

            #dynamic coordinate map
            coordmap = {'x': {}, 'y': {}}

            #offset dict
            offset = OrderedDict()

            #column & headings size
            set_row_height = True

            # italicise columns spec
            y_italicise = {}

            for chain in chain_generator(cluster):

                view_sizes = chain.view_sizes()
                view_lengths = chain.view_lengths()

                if chain.orientation=='x' and not chain.annotations is None:
                    len_chain_annotations = len(chain.annotations)
                    if len_chain_annotations > 0:
                        for ann in chain.annotations:
                            worksheet.write(
                                current_position['x']-1,
                                col_index_origin-1,
                                helpers.get_text(
                                    ann,
                                    text_key,
                                    'x'),
                                formats['x_left_bold']
                            )
                            current_position['x'] += +1
                else:
                    len_chain_annotations = 0

                orientation = chain.orientation

                #chain's view offset
                if not offset:
                    current_views = []
                else:
                    current_views = offset[offset.keys()[0]].keys()

                # Dummy tests needed?
                if grouped_views.get(sheet_name):
                    non_base_views = [vk for vk in chain.views if 'cbase' not in vk]
                    all_grouped_views = list(itertools.chain(*grouped_views[sheet_name]))
                    has_props_tests = any(['|t.props' in vk for vk in chain.views])
                    has_means_tests = any(['|t.means' in vk for vk in chain.views])
                    if all(vk in all_grouped_views for vk in non_base_views):
                        dummy_tests = False
                    else:
                        dummy_tests = (has_props_tests or has_means_tests) \
                                        and formats_spec.dummy_tests
                else:
                   dummy_tests = False

                offset = get_view_offset(chain,
                                         offset,
                                         cluster_gv,
                                         dummy_tests)

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
                            widths = {}
                            dk = chain.data_key
                            fk = chain.filter
                            xk = chain.source_name
                            yk = chain.content_of_axis[0]
                            link = chain[dk][fk][xk][yk]
                            for view in offset[chain.source_name].keys():
                                idxv = chain.views.index(view)
                                coordmap['x'][chain.source_name][view] = [
                                    current_position['x'] \
                                        + offset[chain.source_name][view],
                                    current_position['x'] \
                                        + offset[chain.source_name][view] \
                                        + view_lengths[0][idxv] \
                                        - 1
                                ]

                                # Needed for transforming array tables
                                widths[view] = link[view].dataframe.shape[1]

                            # If the chain holds an array then the coordmap
                            # needs to be transformed.
                            if link[view].meta()['x']['is_array']:

                                vks = coordmap['x'][xk].keys()

                                # Transform x coords
                                start_x = row_index_origin
                                end_x = start_x + (coordmap['x'][xk][vks[0]][-1] - coordmap['x'][xk][vks[0]][0])
                                coord_xs = [start_x, end_x]

                                # Transform y coords
                                coord_ys = OrderedDict()
                                for i, vk in enumerate(vks):
                                    if i==0:
                                        start_y = col_index_origin
                                    end_y = start_y + widths[vk] - 1
                                    coord_ys[vk] = [start_y, end_y]
                                    start_y = end_y + 1

                                coordmap = {'y': {yk: coord_ys}, 'x': {xk: coord_xs}}

                for xy in xy_generator(chain):

                    if orientation == 'y':
                        x, y = xy, chain.source_name
                    elif orientation == 'x':
                        y, x = xy, chain.source_name

                    idxs = chain.content_of_axis.index(xy)

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
                                        + view_lengths[idxs][idxv] \
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
                                            + view_lengths[idxs][idxv]
                                    ]
                    elif orientation == 'x':
                        if y not in coordmap['y'].keys():
                            idxs = chain.content_of_axis.index(y)
                            coordmap['y'][y] = [
                                current_position['y'],
                                current_position['y'] \
                                    + view_sizes[idxs][0][1] \
                                    - 1
                            ]

                    if dummy_tests: dummy_row_count = 0

                    #loop views
                    for vi, views in enumerate(view_generator(offset[x].keys(), cluster_gv)):

                        frames = []
                        vmetas  = []
                        vlevels = []
                        df_rows = []
                        df_cols = []

                        for idx, v in enumerate(views):

                            view = chain[chain.data_key][chain.filter][x][y][v]

                            is_array = view.meta()['x']['is_array']

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

                            if all([view.meta()['agg'][key]==value
                                   for key, value in [('name', 'cbase'),
                                                      ('is_weighted', False)]]):
                                a = view.dataframe.values[0]
                                for cbindex, cb in np.ndenumerate(a):
                                    if cb < italicise_level:
                                        xk = view.meta()['x']['name']
                                        xkc = coordmap['x'][xk].values()
                                        x_loc = list(itertools.chain(*xkc))
                                        x_range = [min(x_loc), max(x_loc)+1]
                                        yk = view.meta()['y']['name']
                                        y_loc = coordmap['y'][yk][0]+cbindex[0]
                                        if y_loc in y_italicise:
                                            y_italicise[y_loc].append(x_range)
                                        else:
                                            y_italicise.update(
                                                {y_loc: [x_range]}
                                            )

                            view.translate_metric(text_key['x'][-1], set_value='meta')
                            vmetas.append(view.meta())

                            if view.is_propstest():
                                vlevels.append(view.is_propstest())
                            elif view.is_meanstest():
                                vlevels.append(view.is_meanstest())
                            else:
                                vlevels.append(None)

                            if view.meta()['agg']['method'] == 'frequency':
                                agg_name = view.meta()['agg']['name']
                                if agg_name in ['cbase', 'c%', 'r%', 'counts']:
                                    axes = ['x', 'y']
                                    if chain.is_banked:
                                        axes.remove('x')
                                    df = helpers.paint_view(
                                        meta=meta,
                                        view=view,
                                        text_key=text_key,
                                        display_names=display_names,
                                        transform_names=transform_names,
                                        axes=axes
                                    )
                                elif agg_name.startswith('x_blocknet'):
                                    df = helpers.paint_view(
                                        meta=meta,
                                        view=view,
                                        text_key=text_key,
                                        display_names=display_names,
                                        transform_names=transform_names,
                                        axes=axes
                                    )
                                else:
                                    df = view.dataframe.copy()
                            else:
                                df = view.dataframe.copy()

                            #write column test labels
                            if 'test' in view.meta()['agg']['method']:
                                if view.meta()['y']['name'] in testcol_labels:
                                    tdf = view.dataframe
                                    y_values = [int(v) for v in zip(
                                        *[c for c in tdf.columns])[1]]
                                    code_idx = testcol_labels.index(
                                        view.meta()['y']['name']
                                    )
                                    for i, code in enumerate(y_values):
                                        worksheet.write(
                                            row_index_origin+(nest_levels*2)-1,
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

                            if is_array:
                                df_cols.append(
                                    coordmap['y'][y][view.meta()['agg']['fullname']]
                                )
                                df_rows.append(coordmap['x'][x])
                            else:
                                df_rows.append(
                                    coordmap['x'][x][view.meta()['agg']['fullname']]
                                )
                                df_cols.append(coordmap['y'][y])

                        # Add dummy dfs
                        if dummy_tests:
                            cond_1 = len(frames) == 1
                            cond_2 = (
                                len(frames) > 1 and not any(
                                    vm['agg']['method'] == 'coltests'
                                    for vm in vmetas
                                )
                            )
                            if cond_1 or cond_2:
                                if vmetas[0]['agg']['name'] != 'cbase':
                                    vmetas.append(cPickle.loads(cPickle.dumps(
                                        vmetas[0], cPickle.HIGHEST_PROTOCOL)))
                                    vmetas[-1]['agg']['is_dummy'] = True
                                    frames.append(pd.DataFrame(
                                        data=' ',
                                        index=frames[0].index,
                                        columns=frames[0].columns))
                                    len_rows = df_rows[0][1]-df_rows[0][0]+1
                                    df_rows.append([df_rows[-1][1]+1,
                                                    df_rows[-1][1]+len_rows])
                                    df_cols.append(coordmap['y'][y])
                                    dummy_row_count += len_rows

                        #write data
                        is_ceil = vmetas[0]['agg']['fullname'] == ceiling
                        is_floor = vmetas[-1]['agg']['fullname'] == floor

                        # has weighted views
                        sub_chain = chain[chain.data_key][chain.filter]
                        has_weighted_views = any(
                            sub_chain[xk][yk][vk].meta()['agg']['is_weighted']
                            for xk in sub_chain.keys()
                            for yk in sub_chain[xk].keys()
                            for vk in sub_chain[xk][yk].keys()
                        )

                        if view.meta()['y']['name'] in testcol_maps:
                            paint_box(
                                worksheet=worksheet,
                                frames=frames,
                                format_dict=formats,
                                rows=df_rows,
                                cols=df_cols,
                                metas=vmetas,
                                formats_spec=formats_spec,
                                has_weighted_views=has_weighted_views,
                                y_italicise=y_italicise,
                                ceil=is_ceil,
                                floor=is_floor,
                                testcol_map=testcol_maps[view.meta()['y']['name']]
                            )
                        else:
                            array_views = vks if is_array else None
                            paint_box(
                                worksheet=worksheet,
                                frames=frames,
                                format_dict=formats,
                                rows=df_rows,
                                cols=df_cols,
                                metas=vmetas,
                                formats_spec=formats_spec,
                                has_weighted_views=has_weighted_views,
                                y_italicise=y_italicise,
                                ceil=is_ceil,
                                floor=is_floor,
                                is_array=is_array,
                                array_views=array_views
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

                        if y_name == '@' and not is_array:
                            if coordmap['x'][x_name][fullname][0] == row_index_origin+(nest_levels*2) + bool(testcol_maps) + len_chain_annotations:
                                #write column label(s) - multi-column y subaxis
                                total_text = helpers.translate(['@'], text_key['y'])[0]

                                worksheet.set_column(
                                    df_cols[idx][0],
                                    df_cols[idx][1],
                                    10
                                )
                                worksheet.merge_range(
                                    row_index_origin-3,
                                    df_cols[idx][0],
                                    row_index_origin+(nest_levels*2)+bool(testcol_maps)+len_chain_annotations-2,
                                    df_cols[idx][1],
                                    total_text,
                                    formats['y']
                                )
                            if bool(testcol_maps):
                                worksheet.write(
                                    row_index_origin+(nest_levels*2)-1,
                                    1,
                                    '',
                                    formats['tests']
                                )

                        elif is_array:
                            labels = helpers.get_unique_level_values(df.columns)
                            if len(vmetas[0]['agg']['text']) > 0:
                                labels[1] = [vmetas[0]['agg']['text']]
                            else:
                                labels[1] = helpers.translate(labels[1], text_key['x'])
                            if nest_levels == 0:
                                write_column_labels(
                                    worksheet,
                                    labels,
                                    formats_arrays['array-y'],
                                    row_index_origin-3,
                                    df_cols[idx],
                                    is_array=True
                                )
                            elif nest_levels > 0:
                                write_column_labels(worksheet,
                                    labels,
                                    formats_arrays['arrays-y'],
                                    row_index_origin-3,
                                    df_cols[idx],
                                    nest_levels,
                                    is_array=True
                                )
                            if df_cols[idx][0] == col_index_origin:
                                worksheet.merge_range(
                                    row_index_origin-3,
                                    df_cols[idx][0],
                                    row_index_origin-3,
                                    df_cols[idx][0]+sum(
                                        [vs[1] for vs in view_sizes[0]])-1,
                                    ' ',
                                    formats['y'])
                        else:
                            if coordmap['x'][x_name][fullname][0] == row_index_origin+(nest_levels*2)+bool(testcol_maps) + len_chain_annotations:
                                labels = helpers.get_unique_level_values(df.columns)
                                labels[1] = helpers.translate(labels[1], text_key['y'])
                                if nest_levels == 0:
                                    write_column_labels(
                                        worksheet,
                                        labels,
                                        formats['y'],
                                        row_index_origin-3,
                                        df_cols[idx]
                                    )
                                elif nest_levels > 0:
                                    write_column_labels(worksheet,
                                        labels,
                                        formats['y'],
                                        row_index_origin-3,
                                        df_cols[idx],
                                        nest_levels
                                    )

                        #write x labels
                        if df_cols[0][0] == col_index_origin:
                            if fullname == ceiling:

                                write_question_label(
                                    worksheet,
                                    df.index[0][0],
                                    formats['x_left_bold'],
                                    df_rows[idx][0]-1,
                                    col_index_origin-1,
                                    formats_spec.row_height,
                                    formats_spec.row_wrap_trigger
                                )

                                if create_toc:
                                    toc_locs[-1].append(
                                        (df_rows[idx][0]-1,  col_index_origin-1)
                                    )
                                    if transform_names:
                                        toc_names[-1].append(
                                            transform_names.get(x_name,
                                                                x_name))
                                    else:
                                        toc_names[-1].append(x_name)
                                    if 'x' in display_names:
                                        toc_label_parts = df.index[0][0].split(
                                            '. ')
                                        if len(toc_label_parts) == 0:
                                            toc_label = toc_label_parts[0]
                                        else:
                                            toc_label = ''.join(
                                                toc_label_parts[1:])
                                        toc_labels[-1].append(toc_label)
                                    else:
                                        toc_labels[-1].append(df.index[0][0])

                        cond_1 = df_cols[0][0] == col_index_origin
                        cond_2 = fullname in new_views
                        cond_3 = not has_weighted_views and not is_weighted

                        if is_array :
                            if vi==0:
                                format_key = 'x_right'
                                labels = [df.index.levels[1][i] for i in df.index.labels[1]]
                                write_category_labels(
                                    worksheet=worksheet,
                                    labels=labels,
                                    existing_format=formats[format_key],
                                    row=df_rows[idx][0],
                                    col=col_index_origin-1,
                                    row_height=formats_spec.row_height,
                                    row_wrap_trigger=formats_spec.row_wrap_trigger,
                                    set_heights=True
                                )

                        elif cond_1 or cond_2:
                            if shortname == 'cbase':
                                if has_weighted_views and not is_weighted:
                                    if len(text) > 0:
                                        format_key = 'x_right_ubase'
                                        labels = [text]
                                        # labels = [''.join(['Unweighted ',
                                        #                    text.lower()])]
                                    else:
                                        format_key = 'x_right_base'
                                        labels = [fullname]
#                                     labels[1] = helpers.translate(labels[1], text_key['x'])
                                    write_category_labels(
                                        worksheet=worksheet,
                                        labels=labels,
                                        existing_format=formats[format_key],
                                        row=df_rows[idx][0],
                                        col=col_index_origin-1,
                                        row_height=formats_spec.row_height,
                                        row_wrap_trigger=formats_spec.row_wrap_trigger,
                                        set_heights=True
                                    )
                                else:
                                    if len(text) > 0:
                                        if not chain.base_text is None:
                                            text = '{}: {}'.format(
                                                text.split(' ')[-1].capitalize()
                                                    if cond_3 else text,
                                                helpers.get_text(
                                                    unicoder(chain.base_text,
                                                             like_ascii=True),
                                                    text_key,
                                                    'x'))
                                        elif cond_3:
                                            text = text.split(' ')[-1].capitalize()
                                        labels = [text]
                                    else:
                                        labels = [fullname]
                                    format_key = 'x_right_base'
                                    write_category_labels(
                                        worksheet=worksheet,
                                        labels=labels,
                                        existing_format=formats[format_key],
                                        row=df_rows[idx][0],
                                        col=col_index_origin-1,
                                        row_height=formats_spec.row_height,
                                        row_wrap_trigger=formats_spec.row_wrap_trigger,
                                        set_heights=True
                                    )
                            else:
                                if (vmetas[0]['agg']['method'] in ['descriptives'] or
                                    (vmetas[0]['agg']['method'] in ['frequency'] and relation != ':')):
                                    if len(frames) > 1:
                                        labels = []
                                        labels_written = []
                                        for idxdf, df in enumerate(frames):
                                            if vmetas[idxdf]['agg']['method'] == 'coltests':
                                                if not formats_spec.display_test_level:
                                                    continue
                                                format_key = 'x_right_tests'
                                                labels = [vlevels[idxdf] for _ in df.index]
                                            else:
                                                if vmetas[idxdf]['agg']['method'] == 'descriptives':
                                                    format_key = 'x_right_descriptives'
                                                else:
                                                    format_key = 'x_right_nets'
                                                if len(vmetas[idxdf]['agg']['text']) > 0 and \
                                                    not vmetas[idxdf]['agg']['is_block']:
                                                    labels = [vmetas[idxdf]['agg']['text']]
                                                else:
                                                    labels = df.index.get_level_values(1)
                                            if all([label not in labels_written for label in labels]):
                                                write_category_labels(
                                                    worksheet=worksheet,
                                                    labels=labels,
                                                    existing_format=formats[format_key],
                                                    row=df_rows[0][0]+idxdf,
                                                    col=col_index_origin-1,
                                                    row_height=formats_spec.row_height,
                                                    row_wrap_trigger=formats_spec.row_wrap_trigger,
                                                    group_size=len(frames),
                                                    set_heights=True
                                                )
                                                labels_written.extend(labels)
                                    else:
                                        if vmetas[0]['agg']['method'] == 'descriptives':
                                            format_key = 'x_right_descriptives'
                                        else:
                                            format_key = 'x_right_nets'
                                        if len(frames[0].index) == 1:
                                            if len(vmetas[0]['agg']['text']) > 0 and \
                                                not vmetas[0]['agg']['is_block']:
                                                labels = [vmetas[0]['agg']['text']]
                                            else:
                                                labels = df.index.get_level_values(1)
                                        else:
                                            labels = df.index.get_level_values(1)
                                        write_category_labels(
                                            worksheet=worksheet,
                                            labels=labels,
                                            existing_format=formats[format_key],
                                            row=df_rows[0][0],
                                            col=col_index_origin-1,
                                            row_height=formats_spec.row_height,
                                            row_wrap_trigger=formats_spec.row_wrap_trigger,
                                            group_size=len(frames),
                                            set_heights=True
                                        )
                                else:
                                    freq_view = False
                                    labels = []
                                    for idxdf, df in enumerate(frames):
                                        if vmetas[idxdf]['agg']['method'] == 'coltests':
                                            if not formats_spec.display_test_level:
                                                continue
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
                                            worksheet=worksheet,
                                            labels=labels,
                                            existing_format=formats[format_key],
                                            row=df_rows[0][0]+idxdf,
                                            col=col_index_origin-1,
                                            row_height=formats_spec.row_height,
                                            row_wrap_trigger=formats_spec.row_wrap_trigger,
                                            group_size=len(frames),
                                            set_heights=True
                                        )

#                     if is_array:
#                         # Merge the top of the array table and remove the merged text
#                         combined_width = sum([widths[vk] for vk in widths.keys()])
#                         worksheet.merge_range(5, 1, 5, combined_width, '', formats['y'])

                    #increment row (only first occurrence of each x)
                    if not is_array:
                        if orientation == 'y':
                            current_position['x'] += sum(
                                view_lengths[idxs]
                            ) + 1
                        elif orientation == 'x':
                            current_position['y'] += (
                                coordmap['y'][xy][1]-coordmap['y'][xy][0]+1
                            )

                #increment col
                if not is_array:
                    if orientation == 'y':
                        current_position['y'] += chain.source_length

                    elif orientation == 'x':
                        current_position['x'] += sum(view_lengths[0])+1
                        if dummy_tests:
                            current_position['x'] += dummy_row_count

            #set column widths
            worksheet.set_column(col_index_origin-1, col_index_origin-1, 40)

            #set y axis height
            worksheet.set_row(row_index_origin-3, formats_spec.y_header_height)
            worksheet.set_row(row_index_origin-2, formats_spec.y_row_height)

            #freeze panes
            worksheet.freeze_panes(
                row_index_origin+(nest_levels*2)+bool(testcol_maps)-1,
                col_index_origin+1
            )


    #download image
    # if IMG_URL:
    if formats_spec.img_url and not formats_spec.no_logo:
        try:
            img_url_full = '\\'.join(
                [os.path.dirname(quantipy.__file__),
                'core\\builds\\excel\\formats',
                 # IMG_URL
                 formats_spec.img_url]
            )
            if os.path.exists(img_url_full):
                img = Image.open(img_url_full)
                # img.thumbnail(IMG_SIZE, Image.ANTIALIAS)
                img.thumbnail(formats_spec.img_size, Image.ANTIALIAS)
                img.save(os.path.basename(img_url_full))
                path_img = os.path.basename(img_url_full)
            else:
                # response = requests.get(IMG_URL)
                response = requests.get(formats_spec.img_url)
                img = Image.open(BytesIO(response.content))
                # img.thumbnail(IMG_SIZE, Image.ANTIALIAS)
                img.thumbnail(formats_spec.img_size, Image.ANTIALIAS)
                img.save('img.png')
                path_img = 'img.png'
        except:
            pass

    #post-process non-TOC sheets
    for worksheet in workbook.worksheets_objs:

            #hide gridlines
            worksheet.hide_gridlines(2)

            if not worksheet.name == 'TOC':

                #write annotations to cells A1, A2, A3, ...
                if annotations.get(worksheet.name):
                    for annotation_spec in annotations[worksheet.name]:
                        if isinstance(annotation_spec, (str, unicode)):
                            annotation = annotation_spec
                            annotation_format = formats['x_left_bold']
                        else:
                            annotation = annotation_spec[0]
                            annotation_format = workbook.add_format(
                                annotation_spec[1]
                            )
                        worksheet.write(
                            annotations[worksheet.name].index(annotation_spec),
                            0,
                            annotation,
                            annotation_format
                        )

                #insert image
                try:
                    worksheet.insert_image(
                        formats_spec.img_insert_x,
                        formats_spec.img_insert_y,
                        path_img,
                        {'x_offset': formats_spec.img_x_offset,
                         'y_offset': formats_spec.img_y_offset}
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
