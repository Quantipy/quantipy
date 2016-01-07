# encoding: utf-8

'''
@author: Majeed.sahebzadha
'''

from __future__ import unicode_literals
import time
import re
import numpy as np
import pandas as pd
import quantipy as qp
from os import path
from collections import OrderedDict
from pptx import Presentation
from quantipy.core.cluster import Cluster
from quantipy.core.chain import Chain
from quantipy.core.helpers.functions import(
            paint_dataframe,
            finish_text_key
            )
from quantipy.core.builds.powerpoint.add_shapes import(
            chart_selector, 
            add_stacked_bar_chart,
            add_textbox
            )
from quantipy.core.builds.powerpoint.transformations import(
            sort_df, 
            is_grid_element,
            get_base,
            validate_cluster_orientations,
            drop_hidden_codes,
            partition_view_df,
            strip_html_tags,
            rename_label,
            df_splitter
            )
from quantipy.core.builds.powerpoint.visual_editor import(
            return_slide_layout_by_name
            )
            
thisdir = path.split(__file__)[0]

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

def chain_generator(cluster):
    '''
    Generate chains
    
    Parameters
    ----------
    cluster : quantipy.Cluster
        quantipy cluster object
    '''
    
    for chain_name in cluster.keys():
        yield cluster[chain_name]

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

def contains_weighted_freqs(chain):
    '''
    check if a qp.Chain contains weighted frequency views
    
    Parameters
    ----------
    chain : quantipy.Chain
        quantipy chain object
    '''

    for el in chain.views:
        e0, e1, e2, e3, e4, e5 = el.split('|')  
        if e0 == 'x' and e1 == 'f' and e3 == 'y' and e4:
            return True
    return False

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

def get_grid_el_label(df):
    '''
    grabs a grid element level label 
    
    Parameters
    ----------
    df : dataframe
        pandas dataframe object
    '''
 
    grid_el_label = strip_html_tags(df.index[0][0])
    if ' - ' in grid_el_label:
        label = grid_el_label.split(' - ')[-1].strip()
    elif '. ' in grid_el_label:
        label = grid_el_label.split('. ',1)[-1].strip()
    else:
        label = 'Label missing'
    return label

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

def PowerPointPainter(path_pptx,
                      meta, 
                      cluster,
                      path_pptx_template=None,
                      slide_layout='Blank',
                      text_key=None,
                      force_chart=True,
                      force_crossbreak=None,
                      base_type='weighted',
                      include_nets=True):
    
    '''
    Builds PowerPoint file (PPTX) from cluster, list of clusters, or 
    dictionary of clusters.

    Parameters
    ----------
    path_pptx : str
        PowerPoint file path
    meta : dict
        metadata as dictionary used to paint datframes
    cluster : quantipy.Cluster / list / dict
        container for cluster(s)
    path_pptx_template : str, optional
        full path to PowerPoint template 
    slide_layout : str / int, optional
        valid slide layout name or index
    text_key : str, optional
        language
    force_chart : boolean, optional
        ues default settings to produce a PowerPoint file
    force_crossbreak : str / list, optional
        use given crossbreaks to build a PowerPoint file
    base_type : str, optional
        use weighted or unweighted base
    include_nets : str, optional
        include, exclude net views in chart data
    '''
    
    #-------------------------------------------------------------------------  
    print('\n{ast}\n{ast}\n{ast}\nINITIALIZING POWERPOINT '
          'AUTOMATION SCRIPT...'.format(ast='*' * 80))
    
    #-------------------------------------------------------------------------     
    # check path extension
    if path_pptx.endswith('.pptx'):
        path_pptx = path_pptx[:-5]
    elif path_pptx.endswith('/') or path_pptx.endswith('\\'):
        raise Exception('File name not provided')

    #-------------------------------------------------------------------------  
    # check base type string  
    base_type = base_type.lower()
    
    #-------------------------------------------------------------------------  
    # render cluster
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

    #-------------------------------------------------------------------------  
    # default settings
    default_props = {'crossbreak': ['@'],
                     'chart_type': 'bar',
                     'sort_order': 'none',
                     'chart_color': 'green',
                     'fixed_categories': [],
                     'base_description': '',
                     'chart_layout': '1',
                     'slide_title_text': 'Click to add title',
                     'question_label': 'Unknown',
                     'copied_from': '',
                     'center_header': '',
                     'right_footer': '',
                     'title_footer': ''}
    
    #-------------------------------------------------------------------------  
    # Update 'crossbreak' key's value in default_props if 
    # force_crossbreak parameter is true
    if force_crossbreak:
        if isinstance(force_crossbreak, list):
            pass
        elif isinstance(force_crossbreak, str):
            force_crossbreak = [force_crossbreak]
        for c in force_crossbreak:
            default_props['crossbreak'].append(c)

    #-------------------------------------------------------------------------
    if not path_pptx_template:
        path_pptx_template = path.join(thisdir,
                                       'templates\default_template.pptx')

    #-------------------------------------------------------------------------
    # get the default text key if none provided
    if text_key is None:
        text_key = finish_text_key(meta, text_key)
        
    if text_key['x'][0] in ['da-DK', 'fi-FI', 'nb-NO', 'sv-SE']:
        country = 'nordic'
    else:
        country = 'uk'
        
    if country == 'nordic':
        
        header_params = {'font_size': 12,
                         'font_italic': True,
                         'left': 284400,
                         'top': 1007999,
                         'width': 8582400,
                         'height': 468000}

        footer_params = {'font_size': 10,
                         'left': 284400, 
                          'top': 1483200, 
                          'width': 8582400, 
                          'height': 396000}
        
        cht_params = {'top': 1879200,
                      'height': 3585600}
    else:

        header_params = {'font_size': 12,
                         'font_italic': True,
                         'left': 284400,
                         'top': 1007999,
                         'width': 8582400,
                         'height': 468000}
        
        footer_params = {'font_size': 10,
                          'left': 284400,
                          'top': 5652000,
                          'width': 8582400,
                          'height': 396000}
        
        cht_params = {'top': 1475999,
                     'height': 4140000}
        
    ############################################################################
    ############################################################################
    ############################################################################
    
    # loop over clusters, returns pptx for each cluster 
    for cluster_name, cluster in zip(names, clusters):
        
        print('\nPowerPoint minions are building your PPTX, ' 
              'please stand by...\n\n{indent:>2}Building '
              'PPTX for {file_name}').format(indent='',
                                             file_name=cluster_name)
        
        # log start time
        pptx_start_time = time.time()
        
        # check if cluster is empty
        if not cluster:
            raise Exception("'{}' cluster is empty".format(cluster_name))
        
        # ensure all chains in cluster have the same orientation
        validate_cluster_orientations(cluster)
        
        # pull orientation of chains in cluster
        orientation = cluster[cluster.keys()[0]].orientation

        # open pptx template file       
        prs = Presentation(path_pptx_template)
        
        # log slide number 
        slide_num = len(prs.slides)

        ############################################################################
        # X ORIENTATION CODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ############################################################################
        
        if orientation == 'x':
            
            # grid element storage dict
            grid_container = []
    
            # This section tries to finds, pull and build grid element 
            # dataframes by matching the downbreak name against the grid element name. 
            # Each downbreak is therefore checked against all keys in masks. 
            
            for chain in chain_generator(cluster):
                
                crossbreaks = chain.content_of_axis
                downbreak = chain.source_name

                '----PULL METADATA DETAILS ----------------------------------------'
                   
                if force_chart:
                    meta_props = []
                else:
                    if downbreak in meta['columns']:
                        if 'properties' in meta['columns'][downbreak]:
                            meta_props = meta['columns'][downbreak]['properties']
                        else:
                            meta_props = []
                    else:
                        meta_props = []
 
                chart_type = meta_props['chart_type'] if 'chart_type' in meta_props else default_props['chart_type']
                layout_type = meta_props['chart_layout'] if 'chart_layout' in meta_props else default_props['chart_layout']
                sort_order = meta_props['sort_order'] if 'sort_order' in meta_props else default_props['sort_order']
                fixed_categories = meta_props['fixed_categories'] if 'fixed_categories' in meta_props else default_props['fixed_categories']
                slide_title_text = meta_props['slide_title'] if 'slide_title' in meta_props else default_props['slide_title_text']
                copied_from = meta_props['copied_from'] if 'copied_from' in meta_props else default_props['copied_from'] 
                base_description = meta_props['base_text'] if 'base_text' in meta_props else default_props['base_description']   
                
                '----IF GRID THEN---------------------'
                
                # loop over items in masks 
                for grid in meta['masks']:
                    for x in range(0, len(meta['masks'][grid]['items'])):
                        gridname = meta['masks'][grid]['items'][x]['source'].split('columns@')[-1]
                        if downbreak == gridname:

                            # check if grid is in grid container, if it's not then continue                   
                            if not grid in grid_container:
                                grid_container += [grid]
                                
                                remaining_elements = [grid_element['source'].split('@')[1]
                                                      for grid_element in meta['masks'][grid]['items'][0:]]
                                
                                grouped_grid_views = []
                                
                                for grid_element_name in remaining_elements:
                                    grid_chain = cluster[grid_element_name]
                                    dk = grid_chain.data_key
                                    fk = grid_chain.filter
        
                                    '----PULL AND BUILD DATAFRAME--------------------------------------'
                                    
                                    # use weighted freq views if available
                                    use_weighted_freq_views = contains_weighted_freqs(grid_chain)
                                    
                                    views_on_chain = []
                                    
                                    # loop over views in chain
                                    for v in grid_chain.views:

                                        # only pull '@' based views as these will be concatenated together
                                        view = grid_chain[dk][fk][grid_element_name]['@'][v]
        
                                        # remove hidden rows and columns
                                        vdf = drop_hidden_codes(view)
        
                                        # raise error if df contains more than 1 column 
                                        if len(vdf.columns) > 1:
                                            raise ValueError("Invalid number of columns, "
                                                             "expected 1 got {}, " 
                                                             "for xk: '{}' cut by yk: '@'.".format(len(vdf.columns),
                                                                                                   downbreak))
        
                                        # paint the dataframe 
                                        df = paint_dataframe(meta=meta, df=vdf)
        
                                        # grab grid element label
                                        grid_el_label = get_grid_el_label(df)
        
                                        # check if painting the df replaced the inner label with NaN
                                        if len(df.index) == 1 and -1 in df.index.labels:
                                            original_labels = vdf.index.tolist()
                                            df_labels = df.index.tolist()
                                            new_idx = (df_labels[0][0], original_labels[0][1])
                                            df.index = pd.MultiIndex.from_tuples([new_idx], 
                                                                                 names=['Question', 'Values'])
        
                                        # flatten df - drop outter index/column levels 
                                        df = partition_view_df(df)[0]
        
                                        # we need two types of views: 1) pct and 2) base
                                        if view.is_pct():
                                            # use weighted or unweighted views
                                            if use_weighted_freq_views:
                                                if view.is_weighted():
                                                    if view.is_net():
                                                        if include_nets:
                                                            views_on_chain.append(df)
                                                    else:
                                                        views_on_chain.append(df)
                                            else:            
                                                if not view.is_weighted():
                                                    if view.is_net():
                                                        if include_nets:
                                                            views_on_chain.append(df)
                                                    else:
                                                        views_on_chain.append(df)
        
                                        # base views
                                        elif view.is_counts():
                                            if view.is_base():
                                                # pull weighted or unweighted base
                                                if base_type == 'weighted':
                                                    if view.is_weighted():
                                                        views_on_chain.append(df)
                                                elif base_type == 'unweighted':
                                                    if not view.is_weighted():
                                                        views_on_chain.append(df)
        
                                        # skip view if not pct or base
                                    
                                    # concat all the views together on a single chain
                                    mdf = pd.concat(views_on_chain, axis=0)
                                    mdf.rename(columns={mdf.columns[0]: grid_el_label}, inplace=True)
                                    grouped_grid_views.append(mdf)

                                #ensure all grid elements have the same number of views
                                el_len = [len(el) for el in grouped_grid_views]
                                if not all(x == el_len[0] for x in el_len):
                                    raise TypeError('cannot merge {} elements - uneven '
                                                    'number of element views.'.format(key))
                                    
                                # concat all grid chains in grouped_grid_views together
                                merged_grid_df = pd.concat(grouped_grid_views, axis=1)
                                merged_grid_df = merged_grid_df.fillna(0.0)

                                slide_num += 1
                                print('\n{indent:>5}Slide {num}. '
                                      'Adding a 100% STACKED BAR CHART '
                                      'for {qname} cut by '
                                      'Total{war_msg}'.format(indent='',
                                                              num=slide_num,
                                                              qname=grid,
                                                              war_msg=''))
                                 
                                # get base table 
                                df_grid_base = merged_grid_df.ix[:1, :]
                                 
                                # get chart table 
                                df_grid_table = merged_grid_df.ix[1:, :]
                                 
                                # get base text 
                                base_text = get_base(df_grid_base,
                                                     base_description)
                                 
                                # get question label 
                                question_label = meta['columns'][downbreak]['text']
                                if isinstance(question_label, dict):
                                    question_label = meta['columns'][downbreak]['text'][question_label.keys()[0]]
                                    question_label = question_label.split(" - ")[0]
                                question_label = '{}. {}'.format(grid,
                                                                 strip_html_tags(question_label))
                                        
                                ''' format table values '''
                                df_grid_table = df_grid_table/100
                                        
                                '----ADDPEND SLIDE TO PRES----------------------------------------------------'
                                if isinstance(slide_layout, int):
                                    slide_layout_obj = prs.slide_layouts[slide_layout]
                                else:
                                    slide_layout_obj = return_slide_layout_by_name(prs, slide_layout)
 
                                slide = prs.slides.add_slide(slide_layout_obj)
                                         
                                '----ADD SHAPES TO SLIDE------------------------------------------------------'
                                 
                                ''' title shape '''                                   
#                                 slide_title = add_textbox(slide,
#                                                           text=slide_title_text,
#                                                           font_color=(0,0,0),
#                                                           font_size=36,
#                                                           font_bold=False,
#                                                           vertical_alignment='middle',
#                                                           left=284400,
#                                                           top=309600,
#                                                           width=8582400,
#                                                           height=691200)
    
                                ''' sub title shape '''
                                sub_title_shp = add_textbox(slide, 
                                                            text=question_label, 
                                                            **header_params)
                                         
                                ''' chart shape '''
                                chart_shp = add_stacked_bar_chart(slide,
                                                                  df_grid_table,
                                                                  caxis_tick_label_position='low',
                                                                  **cht_params)
                                        
                                ''' footer shape '''   
                                base_text_shp = add_textbox(slide, 
                                                            text=base_text,
                                                            **footer_params)
                                             
                '----IF NOT GRID THEN--------------------------------------------------'
   
                if 'crossbreak' in meta_props:
                    if meta_props['crossbreak'] != '@':
                        target_crossbreaks = default_props['crossbreak'] + meta_props['crossbreak'].split(',')
                    else:
                        target_crossbreaks = meta_props['crossbreak'].split(',')
                else:
                    target_crossbreaks = default_props['crossbreak']
   
                for crossbreak in crossbreaks:
                    if crossbreak in target_crossbreaks:
 
                        '----BUILD DATAFRAME-----------------------------------------------'
 
                        use_weighted_freq_views = contains_weighted_freqs(chain)
                         
                        views_on_chain = []
 
                        for v in chain.views:
                            dk = chain.data_key
                            fk = chain.filter
                             
                            view = chain[dk][fk][downbreak][crossbreak][v]
                            vdf = drop_hidden_codes(view)
                         
                            if view.is_pct():
                                if use_weighted_freq_views:
                                    if view.is_weighted():
                                        if not view.is_net():
                                            # ignore questions if they are copied from another question 
                                            if not copied_from:
                                                # exclude fixed categories while sorting 
                                                if sort_order == 'ascending':
                                                    vdf = sort_df(vdf,
                                                                  fixed_categories,
                                                                  column_position=0,
                                                                  ascend=True)
                                                elif sort_order == 'descending':
                                                    vdf = sort_df(vdf,
                                                                  fixed_categories,
                                                                  column_position=0,
                                                                  ascend=False)
                                            df = paint_dataframe(meta=meta, df=vdf)   
                                            df = partition_view_df(df)[0]
                                            views_on_chain.append(df)
                                          
                                        # weighted net
                                        elif view.is_net():
                                            if include_nets:
                                                # check if painting the df replaced a label with NaN
                                                if len(df.index) == 1 and -1 in df.index.labels:
                                                    original_labels = vdf.index.tolist()
                                                    df = paint_dataframe(meta=meta, df=vdf) 
                                                    df_labels = df.index.tolist()
                                                    new_idx = (df_labels[0][0], original_labels[0][1])
                                                    df.index = pd.MultiIndex.from_tuples([new_idx], 
                                                                                         names=['Question', 'Values'])
                                                df = partition_view_df(df)[0]
                                                views_on_chain.append(df)
                                                 
                                if not use_weighted_freq_views:            
                                    if not view.is_weighted():
                                        # unweighted col %
                                        if not view.is_net():
                                            # ignore questions if they are copied from another question 
                                            if not copied_from:
                                                # exclude fixed categories while sorting 
                                                if sort_order == 'ascending':
                                                    vdf = sort_df(vdf,
                                                                  fixed_categories,
                                                                  column_position=0,
                                                                  ascend=True)
                                                elif sort_order == 'descending':
                                                    vdf = sort_df(vdf,
                                                                  fixed_categories,
                                                                  column_position=0,
                                                                  ascend=False)
                                            df = paint_dataframe(meta=meta, df=vdf)   
                                            df = partition_view_df(df)[0]
                                            views_on_chain.append(df)
                                          
                                        # unweighted net
                                        elif view.is_net():
                                            if include_nets:
                                                # check if painting the df replaced a label with NaN
                                                if len(df.index) == 1 and -1 in df.index.labels:
                                                    original_labels = vdf.index.tolist()
                                                    df = paint_dataframe(meta=meta, df=vdf) 
                                                    df_labels = df.index.tolist()
                                                    new_idx = (df_labels[0][0], original_labels[0][1])
                                                    df.index = pd.MultiIndex.from_tuples([new_idx], 
                                                                                         names=['Question', 'Values'])
                                                df = partition_view_df(df)[0]
                                                views_on_chain.append(df)
 
                            # base
                            elif view.is_counts():
                                if view.is_base():
                                    if base_type == 'weighted':
                                        if view.is_weighted():
                                            df = paint_dataframe(meta=meta, df=vdf) 
                                            df = partition_view_df(df)[0]
                                            views_on_chain.append(df)
                                    elif base_type == 'unweighted':
                                        if not view.is_weighted():
                                            df = paint_dataframe(meta=meta, df=vdf) 
                                            df = partition_view_df(df)[0]
                                            views_on_chain.append(df)
 
                        '----NON-GRID TABLES-----------------------------------------------'
                         
                        # merge views 
                        merged_non_grid_df = pd.concat(views_on_chain, axis=0)
                        merged_non_grid_df = merged_non_grid_df.fillna(0.0)
                     
                        # replace '@' with 'Total' 
                        merged_non_grid_df = rename_label(merged_non_grid_df, 
                                                          '@', 
                                                          'Total', 
                                                          orientation='Top')   
                        # get base table 
                        df_base = merged_non_grid_df.ix[:1, :]
                         
                        # get chart table 
                        df_table = merged_non_grid_df.ix[1:, :]
                         
                        # get base text 
                        base_text = get_base(df_base,
                                             base_description)
                         
                        # standardise table values 
                        df_table = df_table/100
                         
                        # get question label 
                        question_label = meta['columns'][downbreak]['text']
                        if isinstance(question_label, dict):
                            question_label = meta['columns'][downbreak]['text'][question_label.keys()[0]]
                        question_label = '{}. {}'.format(downbreak,
                                                         strip_html_tags(question_label))
 
                        # handle incorrect chart type assignment
                        if len(df_table.index) > 15 and chart_type=='pie':
                            chart_type='bar'
                         
                        '----SPLIT DFS & LOOP OVER THEM-------------------------------------'
                         
                        if not df_table.empty:
                             
                            #split large dataframes
                            collection_of_dfs = df_splitter(df_table,
                                                            min_rows=5,
                                                            max_rows=15)
                                    
                            for i, df_table_slice in enumerate(collection_of_dfs):
 
                                slide_num += 1
                                       
                                print('\n{indent:>5}Slide {slide_number}. '
                                      'Adding a {chart_name}'
                                      'CHART for {question_name} '
                                      'cut by {crossbreak_name} {x}'.format(indent='',
                                                                            slide_number=slide_num,
                                                                            chart_name=chart_type.upper(),
                                                                            question_name=downbreak,
                                                                            crossbreak_name='Total' if crossbreak == '@' else crossbreak,
                                                                            x='(cont ('+str(i)+'))' if i > 0 else ''))
 
                                '----ADDPEND SLIDE TO PRES----------------------------------------------------'
                                        
                                if isinstance(slide_layout, int):
                                    slide_layout_obj = prs.slide_layouts[slide_layout]
                                else:
                                    slide_layout_obj = return_slide_layout_by_name(prs, slide_layout)
                                     
                                slide = prs.slides.add_slide(slide_layout_obj)
                                        
                                '----ADD SHAPES TO SLIDE------------------------------------------------------'
                             
                                ''' title shape '''
#                                 if i > 0:
#                                     slide_title_text_cont = (
#                                         '%s (continued %s)' % 
#                                         (slide_title_text, i+1)) 
#                                 else:
#                                     slide_title_text_cont = slide_title_text
                                if i > 0:
                                    slide_title_text_cont = '%s (continued %s)' % (slide_title_text, i+1) 
                                    title_placeholder_shp = slide.placeholders[24]
                                    title_placeholder_shp.text = slide_title_text_cont
 
                                ''' sub title shape '''
                                sub_title_shp = add_textbox(slide,
                                                            text=question_label,
                                                            **header_params)
                                        
                                ''' chart shape '''
                                 
                                numofcols = len(df_table_slice.columns)
                                numofrows = len(df_table_slice.index)
                                 
                                # single series table with less than 3 categories = pie
                                if numofcols == 1 and numofrows <= 3:
                                    chart = chart_selector(slide,
                                                           df_table_slice,
                                                           chart_type='pie',
                                                           has_legend=True,
                                                           **cht_params)
                                     
                                # handle incorrect chart type requests - pie chart cannot handle more than 1 column    
                                elif chart_type == 'pie' and numofcols > 1:
                                    chart = chart_selector(slide,
                                                           df_table_slice,
                                                           chart_type='bar',
                                                           has_legend=True,
                                                           caxis_tick_label_position='low',
                                                           **cht_params)
 
                                # single series table with more than, equal to 4 categories and is not a 
                                # pie chart = chart type selected dynamically chart type with no legend
                                elif numofcols == 1 and chart_type != 'pie':
                                    chart = chart_selector(slide,
                                                           df_table_slice,
                                                           chart_type,
                                                           has_legend=False,
                                                           caxis_tick_label_position='low',
                                                           **cht_params)
                                     
                                else:
                                    # multi series tables = dynamic chart type with legend 
                                    chart = chart_selector(slide,
                                                           df_table_slice,
                                                           chart_type,
                                                           has_legend=True,
                                                           **cht_params)
                                            
                                ''' footer shape '''   
                                base_text_shp = add_textbox(slide, 
                                                            text=base_text,
                                                            **footer_params)
                        else:
                            print('\n{indent:>5}***Skipping {question_name}, '
                                  'no percentage based views found'.format(indent='',
                                                                           question_name=downbreak,))
                               
            prs.save('{}.pptx'.format(path_pptx))
 
        ############################################################################
        # Y ORIENTATION CODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ############################################################################
         
        if orientation == 'y': 
             
            # raise error is cluster is y orientated
            raise TypeError('y orientation not supported yet')
         
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
         
    pptx_elapsed_time = time.time() - pptx_start_time     
    print('\n{indent:>2}Presentation saved, '
          'time elapsed: {time:.2f} seconds\n'
          '\n{line}'.format(indent='',
                            time=pptx_elapsed_time,
                            line= '_' * 80))
