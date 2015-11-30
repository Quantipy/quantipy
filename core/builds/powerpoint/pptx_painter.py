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
from quantipy.core.helpers.functions import finish_text_key
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
            paint_df,
            strip_html_tags,
            partition_view_df,
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
    '''Generate chains
    '''
    for chain_name in cluster.keys():
        yield cluster[chain_name]

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

def view_validator(view):
    '''
    Checks if view is valid 
    '''
    
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
                     xk=downbreak,
                     yk=crossbreak))
        
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
    
    ''' Check path extension '''
    if path_pptx.endswith('.pptx'):
        path_pptx = path_pptx[:-5]
    elif path_pptx.endswith('/') or path_pptx.endswith('\\'):
        raise Exception('File name not provided')
        
    ''' Check base type string ''' 
    base_type = base_type.lower()

    ''' Render cluster '''
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
    ''' Default settings '''
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
    ''' Update default_props' crossbreak value if 
    force_crossbreak parameter is true '''
    if force_crossbreak:
        if isinstance(force_crossbreak, list):
            pass
        elif isinstance(force_crossbreak, str):
            force_crossbreak = [force_crossbreak]
        for c in force_crossbreak:
            default_props['crossbreak'].append(c)
            
    #-------------------------------------------------------------------------  
    print('\n{ast}\n{ast}\n{ast}\nINITIALIZING POWERPOINT '
          'AUTOMATION SCRIPT...'.format(ast='*' * 80))

    #-------------------------------------------------------------------------
    if not path_pptx_template:
        path_pptx_template = path.join(thisdir,
                                       'templates\default_template.pptx')

    #-------------------------------------------------------------------------
    # get the default text key if none provided
    if text_key is None:
        text_key = finish_text_key(meta, text_key)
    
    ############################################################################
    ############################################################################
    ############################################################################
    
    ''' loop over clusters, returns pptx for each cluster '''
    for cluster_name, cluster in zip(names, clusters):

        pptx_start_time = time.time()
        
        validate_cluster_orientations(cluster)
        orientation = cluster[cluster.keys()[0]].orientation

        print('\nPowerPoint minions are building your PPTX, ' 
              'please stand by...\n\n{indent:>2}Building '
              'PPTX for {file_name}').format(indent='',
                                             file_name=cluster_name)
              
        prs = Presentation(path_pptx_template)
        slide_num = len(prs.slides)
        groupofgrids = OrderedDict()

        ############################################################################
        # X ORIENTATION CODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ############################################################################
        
        if orientation == 'x':

            for chain in chain_generator(cluster):
                
                crossbreaks = chain.content_of_axis
                downbreak = chain.source_name
                
                '----BUILD DICT HOLDING GRID ELEMENT-----------'

                for grid in meta['masks']:
                    for x in range(0, len(meta['masks'][grid]['items'])):
                        gridname = meta['masks'][grid]['items'][x]['source'].split('columns@')[-1]
                        if downbreak == gridname:

                            for crossbreak in crossbreaks:
                                # only interested in the Total column for grid
                                if crossbreak == '@':

                                    '----BUILD DATAFRAME---------------------------------------------'
                                    # decide whether to use weight or unweight c% data for charts
                                    weighted_chart = [el 
                                                      for el in chain.views 
                                                      if el.startswith('x|frequency|') and el.split('|')[4]!='' and el.split('|')[3]=='y']

                                    views_on_var = []
                                    for v in chain.views:

                                        view = chain[chain.data_key][chain.filter][downbreak][crossbreak][v]
                                        view_validator(view)
                                        vdf = drop_hidden_codes(view)
                                        
                                        if len(vdf.columns) > 1:
                                            raise ValueError("Invalid number of columns, "
                                                             "expected 1 got {}, " 
                                                             "for xk: '{}' cut by yk: '@'.".format(len(vdf.columns),
                                                                                                   downbreak))
                                        
                                        if view.is_pct():
                                            
                                            if weighted_chart:
                                                if view.is_weighted():
                                                    # weighted col %
                                                    if not view.is_net():
                                                        df = paint_df(vdf, view, meta, text_key)  
                                                        # format question labels to grid index labels
                                                        grid_element_label = strip_html_tags(df.index[0][0])
                                                        if ' - ' in grid_element_label:
                                                            grid_element_label = grid_element_label.split(' - ')[-1].strip()
                                                        elif '. ' in grid_element_label:
                                                            grid_element_label = grid_element_label.split('. ',1)[-1].strip()
                                                             
                                                        df = partition_view_df(df)[0]
                                                        views_on_var.append(df)
                                                     
                                                    # weighted net
                                                    elif view.is_net():
                                                        if include_nets:
                                                            original_labels = vdf.index.tolist()
                                                            df = paint_df(vdf, view, meta, text_key)
                                                            df_labels = df.index.tolist()
                                                            new_idx = (df_labels[0][0], original_labels[0][1])
                                                            df.index = pd.MultiIndex.from_tuples([new_idx], 
                                                                                                 names=['Question', 'Values'])
                                                     
                                                            df = partition_view_df(df)[0]
                                                            views_on_var.append(df)
                                                            
                                            if not weighted_chart:            
                                                if not view.is_weighted():
                                                    # unweighted col %
                                                    if not view.is_net():
                                                        df = paint_df(vdf, view, meta, text_key)  
                                                        # format question labels to grid index labels
                                                        grid_element_label = strip_html_tags(df.index[0][0])
                                                        if ' - ' in grid_element_label:
                                                            grid_element_label = grid_element_label.split(' - ')[-1].strip()
                                                        if '. ' in grid_element_label:
                                                            grid_element_label = grid_element_label.split('. ',1)[-1].strip()
                                                             
                                                        df = partition_view_df(df)[0]
                                                        views_on_var.append(df)
                                                     
                                                    # unweighted net
                                                    elif view.is_net():
                                                        if include_nets:
                                                            original_labels = vdf.index.tolist()
                                                            df = paint_df(vdf, view, meta, text_key)
                                                            df_labels = df.index.tolist()
                                                            new_idx = (df_labels[0][0], original_labels[0][1])
                                                            df.index = pd.MultiIndex.from_tuples([new_idx], 
                                                                                                 names=['Question', 'Values'])
                                                     
                                                            df = partition_view_df(df)[0]
                                                            views_on_var.append(df)

                                        # base
                                        elif view.is_counts():
                                            
                                            if view.is_base():
                                                if base_type == 'weighted':
                                                    if view.is_weighted():
                                                        df = paint_df(vdf, view, meta, text_key)
                                                        df = partition_view_df(df)[0]
                                                        views_on_var.append(df)
                                                elif base_type == 'unweighted':
                                                    if not view.is_weighted():
                                                        df = paint_df(vdf, view, meta, text_key)
                                                        df = partition_view_df(df)[0]
                                                        views_on_var.append(df)

                                    '----POPULATE GRID DICT---------------------------------'
                                    
                                    ''' merge pct and base views '''
                                    mdf = pd.concat(views_on_var, axis=0)
                                                        
                                    ''' create a key '''    
                                    key = grid                                                          
                                    if not key in groupofgrids:
                                        groupofgrids[key] = []
                                            
                                    for col in mdf.columns:
                                        s = mdf[col].copy()
                                        s.name = grid_element_label
                                        groupofgrids[key].append(s)   
                                   
            '----CREATE NEW PRESENTATION --------------------------------------'
              
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
              
                '----IF GRID THEN--------------------------------------------------'

                for grid in meta['masks']:
                    for x in range(0, len(meta['masks'][grid]['items'])):
                        gridname = meta['masks'][grid]['items'][x]['source'].split('columns@')[-1]
                        if downbreak == gridname:

                            key = grid                                  
                            if key in groupofgrids.keys():

                                slide_num += 1
                                print('\n{indent:>5}Slide {num}. '
                                      'Adding a STACKED BAR CHART '
                                      'for {qname} cut by '
                                      'Total{war_msg}'.format(indent='',
                                                              num=slide_num,
                                                              qname=key,
                                                              war_msg=''))
                                
                                ''' merge grid element tables into a summary table '''
                                merged_grid_df = pd.concat(groupofgrids[key], axis=1)
                                merged_grid_df = merged_grid_df.fillna(0.0)
                                
                                ''' get base table '''
                                df_grid_base = merged_grid_df.ix[:1, :]
                                ''' get chart table '''
                                df_grid_table = merged_grid_df.ix[1:, :]
                                
                                ''' get base text '''
                                base_text = get_base(df_grid_base,
                                                     base_description)
                                
                                ''' get question label '''
                                question_label = meta['columns'][downbreak]['text']
                                if isinstance(question_label, dict):
                                    question_label = meta['columns'][downbreak]['text'][question_label.keys()[0]]
                                    question_label = question_label.split(" - ")[0]
                                question_label = '{}. {}'.format(key,
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
                                slide_title = add_textbox(slide,
                                                          text=slide_title_text,
                                                          font_color=(0,0,0),
                                                          font_size=36,
                                                          font_bold=False,
                                                          vertical_alignment='middle',
                                                          left=284400,
                                                          top=309600,
                                                          width=8582400,
                                                          height=691200)
   
                                ''' sub title shape '''
                                sub_title_shp = add_textbox(slide, 
                                                            text=question_label, 
                                                            font_size=12, 
                                                            font_italic=True,
                                                            left=284400, 
                                                            top=1007999, 
                                                            width=8582400, 
                                                            height=468000)
                                        
                                ''' chart shape '''
                                chart_shp = add_stacked_bar_chart(slide,
                                                                  df_grid_table,
                                                                  caxis_tick_label_position='low')
                                       
                                ''' footer shape '''   
                                base_text_shp = add_textbox(slide,
                                                            text=base_text,
                                                            font_size=8,
                                                            left=284400,
                                                            top=5652000,
                                                            width=8582400,
                                                            height=396000)
                                       
                                groupofgrids.pop(key)
              
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

                        '----BUILD DATAFRAME---------------------------------------------'
                        # decide whether to use weight or unweight c% data for charts
                        weighted_chart = [el 
                                          for el in chain.views 
                                          if el.startswith('x|frequency|') and el.split('|')[4]!='' and el.split('|')[3]=='y']

                        views_on_var = []

                        for v in chain.views:

                            view = chain[chain.data_key][chain.filter][downbreak][crossbreak][v]
                            view_validator(view)
                            vdf = drop_hidden_codes(view)
                            

                            if view.is_pct():

                                if weighted_chart:
                                    if view.is_weighted():
                                        # weighted col %
                                        if not view.is_net():
                                            df = paint_df(vdf, view, meta, text_key)  
                                            # format question labels to grid index labels
                                            grid_element_label = strip_html_tags(df.index[0][0])
                                            if ' - ' in grid_element_label:
                                                grid_element_label = grid_element_label.split(' - ')[-1].strip()
                                            if '. ' in grid_element_label:
                                                grid_element_label = grid_element_label.split('. ',1)[-1].strip()
                                                 
                                            df = partition_view_df(df)[0]
                                            views_on_var.append(df)
                                         
                                        # weighted net
                                        elif view.is_net():
                                            if include_nets:
                                                original_labels = vdf.index.tolist()
                                                df = paint_df(vdf, view, meta, text_key)
                                                df_labels = df.index.tolist()
                                                new_idx = (df_labels[0][0], original_labels[0][1])
                                                df.index = pd.MultiIndex.from_tuples([new_idx], 
                                                                                     names=['Question', 'Values'])
                                         
                                                df = partition_view_df(df)[0]
                                                views_on_var.append(df)
                                                
                                if not weighted_chart:            
                                    if not view.is_weighted():
                                        # weighted col %
                                        if not view.is_net():
                                            df = paint_df(vdf, view, meta, text_key)  
                                            # format question labels to grid index labels
                                            grid_element_label = strip_html_tags(df.index[0][0])
                                            if ' - ' in grid_element_label:
                                                grid_element_label = grid_element_label.split(' - ')[-1].strip()
                                            if '. ' in grid_element_label:
                                                grid_element_label = grid_element_label.split('. ',1)[-1].strip()
                                                 
                                            df = partition_view_df(df)[0]
                                            views_on_var.append(df)
                                         
                                        # weighted net
                                        elif view.is_net():
                                            if include_nets:
                                                original_labels = vdf.index.tolist()
                                                df = paint_df(vdf, view, meta, text_key)
                                                df_labels = df.index.tolist()
                                                new_idx = (df_labels[0][0], original_labels[0][1])
                                                df.index = pd.MultiIndex.from_tuples([new_idx], 
                                                                                     names=['Question', 'Values'])
                                         
                                                df = partition_view_df(df)[0]
                                                views_on_var.append(df)

                            # base
                            elif view.is_counts():
                                if view.is_base():
                                    if base_type == 'weighted':
                                        if view.is_weighted():
                                            df = paint_df(vdf, view, meta, text_key)
                                            df = partition_view_df(df)[0]
                                            views_on_var.append(df)
                                    elif base_type == 'unweighted':
                                        if not view.is_weighted():
                                            df = paint_df(vdf, view, meta, text_key)
                                            df = partition_view_df(df)[0]
                                            views_on_var.append(df)

                        '----IF NON-GRID TABLES---------------------------------------------'
                        
                        ''' merge views '''
                        merged_non_grid_df = pd.concat(views_on_var, axis=0)
                        merged_non_grid_df = merged_non_grid_df.fillna(0.0)
                        
                        ''' replace '@' with 'Total' '''
                        merged_non_grid_df = rename_label(merged_non_grid_df, 
                                                          '@', 
                                                          'Total', 
                                                          orientation='Top')   
                        ''' get base table '''
                        df_base = merged_non_grid_df.ix[:1, :]
                        ''' get chart table '''
                        df_table = merged_non_grid_df.ix[1:, :]
                        
                        ''' get base text '''
                        base_text = get_base(df_base,
                                             base_description)
                        
                        ''' standardise table values '''
                        df_table = df_table/100
                        
                        ''' get question label '''
                        question_label = meta['columns'][downbreak]['text']
                        if isinstance(question_label, dict):
                            question_label = meta['columns'][downbreak]['text'][question_label.keys()[0]]
                        question_label = '{}. {}'.format(downbreak,
                                                         strip_html_tags(question_label))
                        
                        '----SPLIT DFS & LOOP OVER THEM-------------------------------------'
                              
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

                            numofcols = len(df_table_slice.columns)
                            numofrows = len(df_table_slice.index)
                        
                            '----ADDPEND SLIDE TO PRES----------------------------------------------------'
                                   
                            if isinstance(slide_layout, int):
                                slide_layout_obj = prs.slide_layouts[slide_layout]
                            else:
                                slide_layout_obj = return_slide_layout_by_name(prs, slide_layout)
                                
                            slide = prs.slides.add_slide(slide_layout_obj)
                                   
                            '----ADD SHAPES TO SLIDE------------------------------------------------------'
                        
                            ''' title shape '''
                            if i > 0:
                                slide_title_text_cont = (
                                    '%s (continued %s)' % 
                                    (slide_title_text, i+1)) 
                            else:
                                slide_title_text_cont = slide_title_text
                                 
                            slide_title = add_textbox(slide,
                                                      text=slide_title_text_cont,
                                                      font_color=(0,0,0),
                                                      font_size=36,
                                                      font_bold=False,
                                                      vertical_alignment='middle',
                                                      left=284400,
                                                      top=309600,
                                                      width=8582400,
                                                      height=691200)
                        
                            ''' sub title shape '''
                            sub_title_shp = add_textbox(slide,
                                                        text=question_label,
                                                        font_size=12,
                                                        font_italic=True,
                                                        left=284400,
                                                        top=1007999,
                                                        width=8582400,
                                                        height=468000)
                                   
                            ''' chart shape '''
                            # single series table with less than 3 categories = pie
                            if numofcols == 1 and numofrows <= 3:
                                chart = chart_selector(slide,
                                                       df_table_slice,
                                                       'pie',
                                                       has_legend=True)
                                
                            # handle incorrect chart type requests - pie chart cannot handle more than 1 column    
                            elif chart_type == 'pie' and numofcols > 1:
                                chart = chart_selector(slide,
                                                       df_table_slice,
                                                       chart_type,
                                                       has_legend=True,
                                                       caxis_tick_label_position='low')
                                 
                            # single series table with more than, equal to 4 categories and is not a 
                            # pie chart = chart type selected dynamically chart type with no legend
                            elif numofcols == 1 and chart_type != 'pie':
                                chart = chart_selector(slide,
                                                       df_table_slice,
                                                       chart_type,
                                                       has_legend=False,
                                                       caxis_tick_label_position='low')
                                
                            else:
                                # multi series tables = dynamic chart type with legend 
                                chart = chart_selector(slide,
                                                       df_table_slice,
                                                       chart_type,
                                                       has_legend=True,
                                                       caxis_tick_label_position='low')
                                       
                            ''' footer shape '''   
                            base_text_shp = add_textbox(slide,
                                                        text=base_text,
                                                        font_size=8,
                                                        left=284400,
                                                        top=5652000,
                                                        width=8582400,
                                                        height=396000)
                              
            prs.save('{}.pptx'.format(path_pptx))
                                
        ############################################################################
        # Y ORIENTATION CODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ############################################################################
 
        elif orientation == 'y': 
             
            raise TypeError('y orientation not supported yet')
        

    pptx_elapsed_time = time.time() - pptx_start_time     
    print('\n{indent:>2}Presentation saved, '
        'time elapsed: {time:.2f} seconds\n\n{line}'.format(
        indent='',
        time=pptx_elapsed_time, 
        line= '_' * 80
        )
    )