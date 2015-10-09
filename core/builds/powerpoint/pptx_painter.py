'''
@author: Majeed.sahebzadha
'''

import time
import numpy as np
import pandas as pd
import quantipy as qp
from os import path
from collections import OrderedDict
from pptx import Presentation
from quantipy.core.cluster import Cluster
from quantipy.core.chain import Chain
from quantipy.core.helpers import functions as helpers
from quantipy.core.builds.powerpoint.add_shapes import *
from quantipy.core.builds.powerpoint.transformations import *

thisdir = path.split(__file__)[0]

pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
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
             'filter={fk}\nx={xk}\ny={yk}\n').format(cluster = cluster.name,
                                                     vk=v,
                                                     dk=chain.data_key,
                                                     fk=chain.filter,
                                                     xk=downbreak,
                                                     yk=crossbreak))
        
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

def is_grid_element(table_name, table_pattern):
    '''
    Checks if a table is a grid element or not
    '''
    
    matches = table_pattern.findall(table_name)

    if (len(matches)>0 and len(matches[0])==2): 
        return True
    else:
        return False

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

def auto_sort(df, fixed_categories, column_position=0, ascend=True):
    '''
    Sorts df whilst ignoring fixed categories
    '''
    
    if fixed_categories:
        
        nblevels = df.index.nlevels
        if nblevels == 1:
            pass
        elif nblevels == 2:
            
            outter = df.index[0][0]

            newl = [(outter, item) for item in fixed_categories]
            
            fixed_items = df[-len(newl):][df.index[-len(newl):].isin(newl)].index.tolist()
            
            excluded_cats = df.loc[fixed_items]
            
            included_cats = df[~df.index.isin(fixed_items)]
            
            sorted_cats = included_cats.sort(columns=df.columns[0], 
                                             ascending=False)
            
            df = pd.concat([sorted_cats, excluded_cats])
            
        return df

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

def PowerPointPainter(path_pptx_distination,
					  meta, 
					  cluster,
                      path_pptx_template=None,
                      text_key=None,
                      force_chart=True,
                      force_crossbreak=None):
    '''
    Builds PowerPoint file (PPTX) from cluster, list of clusters, or 
    dictionary of clusters.

    Parameters
    ----------
    path_pptx_distination : str
        distination path of PowerPoint file 
    meta : dict
        metadata as dictionary used to paint datframes
    cluster : quantipy.Cluster / list / dict
        container for cluster(s)
    path_pptx_template : str 
        full path to PowerPoint template 
    text_key : str
        language
    force_chart : boolean
        ues default settings to produce a PowerPoint file
    force_crossbreak : str / list
        use given crossbreaks to build a PowerPoint file
    '''

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
    print('\n{ast}\n{ast}\n{ast}\nINITIALIZING POWERPOINT '
          'AUTOMATION SCRIPT...'.format(ast='*' * 80))

    #-------------------------------------------------------------------------
    ''' table pattern to look for: xxx[{xxx}].xxx '''
    table_pattern = re.compile(r'(.*?)\[.*?\]\.(.*)')
    
    #-------------------------------------------------------------------------
    if not path_pptx_template:
        path_pptx_template = path.join(thisdir,
                                       'templates\yg_master_template_calibri.pptx')

    #-------------------------------------------------------------------------
    #get the default text key if none provided
    if text_key is None:
        text_key = helpers.finish_text_key(meta, text_key)
    
    ############################################################################
    ############################################################################
    ############################################################################
    
    ''' loop over clusters, returns pptx for each cluster. '''
    for cluster_name, cluster in zip(names, clusters):

        pptx_start_time = time.time()
        
        validate_cluster_orientations(cluster)
        orientation = cluster[cluster.keys()[0]].orientation

        print('\nPowerPoint minions are building your PPTX, ' 
              'please stand by...\n\n{indent:>2}Building '
              'PPTX for {file_name}').format(indent='',
                                             file_name=cluster_name)
              
        groupofgrids = OrderedDict()
        
        prs = Presentation(path_pptx_template)
        slide_num = 1 
        
        ############################################################################
        # X ORIENTATION CODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ############################################################################
        
        if orientation == 'x':
            
            for chain in chain_generator(cluster):
                
                crossbreaks = chain.content_of_axis
                downbreak = chain.source_name
                
                '----DETERMINE QUESTION TYPE - GRID OR NON-GRID QUESTION-----------'
                
                if is_grid_element(downbreak, table_pattern):
                    
                    for crossbreak in crossbreaks:
                        if crossbreak == '@':
                            
                            views_on_var = []
                            
                            for v in chain.views:
                                
                                '----BUILD DATAFRAME---------------------------------------------'

                                view = chain[chain.data_key][chain.filter][downbreak][crossbreak][v]
                                view_validator(view)

                                if view.is_weighted() and (view.is_pct() or view.is_base()):
                                    
                                    vdf = drop_hidden_codes(view)
                                    df = paint_df(vdf,view, meta, text_key)
                                    
                                    #format question labels to grid index labels
                                    grid_element_label = strip_html_tags(df.index[0][0])
                                    if ' : ' in grid_element_label:
                                        grid_element_label = grid_element_label.split(' : ')[0].strip()
                                    if '. ' in grid_element_label:
                                        grid_element_label = grid_element_label.rsplit('.',1)[-1].strip()

                                    df = partition_view_df(df)[0]
                                    
                                    views_on_var.append(df)
                                    
                            '----POPULATE GRID DICT---------------------------------'
                            
                            ''' merge pct and base views '''
                            mdf = pd.concat(views_on_var, axis=0)
                                                
                            ''' create a key '''
                            grid_inner_label = downbreak.split('[')[0]
                            grid_outter_label = downbreak.split('].')[-1]
                            key = '.'.join((grid_inner_label, grid_outter_label))                                                                  
                                
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
                    try:
                        meta_props = meta['columns'][downbreak]['properties']
                    except:
                        print 'meta properties not found for: ', downbreak
                        print 'use default instead'
                        meta_props = []
                    
                chart_type = meta_props['chart_type'] if 'chart_type' in meta_props else default_props['chart_type']
                layout_type = meta_props['chart_layout'] if 'chart_layout' in meta_props else default_props['chart_layout']
                sort_order = meta_props['sort_order'] if 'sort_order' in meta_props else default_props['sort_order']
                fixed_categories = meta_props['fixed_categories'] if 'fixed_categories' in meta_props else default_props['fixed_categories']
                slide_title_text = meta_props['slide_title'] if 'slide_title' in meta_props else default_props['slide_title_text']
                copied_from = meta_props['copied_from'] if 'copied_from' in meta_props else default_props['copied_from'] 
                base_description = meta_props['base_text'] if 'base_text' in meta_props else default_props['base_description']      
              
                '----FIND SPECIFIED CROSSBREAK FROM TABLE--------------------------'
  
                if 'crossbreak' in meta_props:
                    if meta_props['crossbreak'] != '@':
                        target_crossbreaks = default_props['crossbreak'] + meta_props['crossbreak'].split(',')
                    else:
                        target_crossbreaks = meta_props['crossbreak'].split(',')
                else:
                    target_crossbreaks = default_props['crossbreak']
  
                for crossbreak in crossbreaks:
                    if crossbreak in target_crossbreaks:
                          
                        views_on_var = []
  
                        for v in chain.views:
  
                            '----BUILD DATAFRAME---------------------------------------------'
            
                            view = chain[chain.data_key][chain.filter][downbreak][crossbreak][v]
                            view_validator(view)
                              
                            if view.is_weighted() and (view.is_pct() or view.is_base() or view.is_net()):
  
                                vdf = drop_hidden_codes(view)
      
                                if (view.is_pct() and view.is_weighted()) and not view.is_net():
                                    ''' ignore questions if they are copied from another question '''
                                    if not copied_from:
                                        ''' exclude fixed categories while sorting '''
                                        if sort_order == 'ascending':
                                            vdf = auto_sort(vdf, 
                                                            fixed_categories, 
                                                            column_position=0, 
                                                            ascend=True)
                                        elif sort_order == 'descending':
                                            vdf = auto_sort(vdf, 
                                                            fixed_categories, 
                                                            column_position=0, 
                                                            ascend=False)            
                                if view.is_net():
                                    ''' paint net df '''
                                    original_labels = vdf.index.tolist()
                                    df = paint_df(vdf, view, meta, text_key)
                                    df_labels = df.index.tolist()
                                    new_idx = (df_labels[0][0], original_labels[0][1])
                                    df.index = pd.MultiIndex.from_tuples([new_idx], 
                                                                         names=['Question', 'Values'])
                                else:
                                    df = paint_df(vdf, view, meta, text_key) 
                                    
                                views_on_var.append(df)
                              
                        '----IF GRID THEN--------------------------------------------------'

                        if is_grid_element(downbreak, table_pattern):
                            
                            grid_inner_label = downbreak.split('[')[0]
                            grid_outter_label = downbreak.split('].')[-1]
                            key = '.'.join((grid_inner_label, grid_outter_label))  

                            if key in groupofgrids.keys():
                                       
                                slide_num += 1
                                       
                                print('\n{indent:>5}Slide {num}. '
                                      'Adding a STACKED BAR CHART '
                                      'for {qname} cut by '
                                      'Total{war_msg}'.format(indent='',
                                                              num=slide_num,
                                                              qname=downbreak,
                                                              war_msg=''))
                                
                                ''' merge grid element tables into a summary table '''
                                merged_grid_df = pd.concat(groupofgrids[key], axis=1)
                                merged_grid_df = merged_grid_df.dropna()
                                
                                ''' get base table '''
                                df_grid_base = merged_grid_df.ix[:1, :]
                                ''' get chart table '''
                                df_grid_table = merged_grid_df.ix[1:, :]
                                
                                ''' construct base text '''
                                base_text = get_base(df_grid_base,
                                                     base_description)
                                
                                ''' get question label '''
                                question_label = meta['masks'][key]['text']['en-GB']
                                question_label = strip_html_tags(question_label)
                                       
                                ''' format table values '''
                                df_grid_table = df_grid_table/100
                                       
                                '----ADDPEND SLIDE TO PRES----------------------------------------------------'
                                        
                                slide_layout = prs.slide_masters[0].slide_layouts[1]
                                slide = prs.slides.add_slide(slide_layout)
                                        
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
                                 
                        '----IF NON-GRID TABLES---------------------------------------------'
                         
                        merged_non_grid_df = pd.concat(views_on_var, axis=0)
                        merged_non_grid_df = merged_non_grid_df.dropna()
                        
                        question_label = strip_html_tags(merged_non_grid_df.index[0][0])   
                        merged_non_grid_df = partition_view_df(merged_non_grid_df)[0]
                        merged_non_grid_df = rename_label(merged_non_grid_df, 
                                                          '@', 
                                                          'Total', 
                                                          orientation='Top')   
                        
                        df_base = merged_non_grid_df.ix[:1, :]
                        df_table = merged_non_grid_df.ix[1:, :]
                        
                        base_text = get_base(df_base,
                                             base_description)
                         
                        df_table = df_table/100
                        try:
                            question_label = strip_html_tags(meta['columns'][downbreak]['text']['en-GB'])
                        except:
                            print "Could not locate meta for: ", downbreak
                            print df
                            question_label = 'Question label not found'
                            
                              
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
                                   
                            slide_layout = prs.slide_masters[0].slide_layouts[1]
                            slide = prs.slides.add_slide(slide_layout)
                                   
                            '----ADD SHAPES TO SLIDE------------------------------------------------------'
   
                            ''' title shape '''
                            if i > 0:
                                slide_title_text_cont = (
                                    '%s (continued %s)' % (slide_title_text, i+1)) 
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
                            #single series table with less than 3 categories = pie
                            if numofcols == 1 and numofrows <= 3:
                                chart = chart_selector(slide,
                                                       df_table_slice,
                                                       'pie',
                                                       has_legend=True)
                                
                            #handle incorrect chart type requests - pie chart cannot handle more than 1 column    
                            elif chart_type == 'pie' and numofcols > 1:
                                chart = chart_selector(slide,
                                                       df_table_slice,
                                                       chart_type,
                                                       has_legend=True,
                                                       caxis_tick_label_position='low')
                                 
                            #single series table with more than, equal to 4 categories and is not a 
                            #pie chart = chart type selected dynamically chart type with no legend
                            elif numofcols == 1 and chart_type != 'pie':
                                chart = chart_selector(slide,
                                                       df_table_slice,
                                                       chart_type,
                                                       has_legend=False,
                                                       caxis_tick_label_position='low')
                                
                            else:
                                #multi series tables = dynamic chart type with legend 
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
                              
        prs.save('{pres_path}\\{pres_name}_'
                 '({cluster_name}).pptx'.format(pres_path=path_pptx_distination,
                                                pres_name=chain.data_key,
                                                cluster_name=cluster.name))
                                
        ############################################################################
        # Y ORIENTATION CODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ############################################################################
 
        if orientation == 'y': 
             
            numofdownbreaks = len(cluster[cluster.keys()[0]].content_of_axis)
 
            for downbreak_idx in range(0, numofdownbreaks):               
                for chain in chain_generator(cluster):
 
                    crossbreak = chain.source_name
                    downbreak = chain.content_of_axis[downbreak_idx]
                    matches = table_pattern.findall(downbreak)

                    if force_chart:
                        meta_props = []
                    else:
                        meta_props = meta['columns'][downbreak]['properties']
 
                    if (len(matches) > 0 and len(matches[0]) == 2):
                      
                        if crossbreak == '@':
                            for v in chain.views:
 
                                view = chain[chain.data_key][chain.filter][downbreak][crossbreak][v]
 
                                vkey = v.split('|')
                                func = vkey[1]
                                relation = vkey[2]
                                rel_to = vkey[3]
                                weight = vkey[4]
                                name = vkey[5]
 
                                #drop hidden codes
                                vdf = drop_hidden_codes(view) 
                                #add question and value labels to df
                                df = paint_df(vdf, view, meta, text_key)   
  
                                #format question labels to grid index labels
                                series_label = strip_html_tags(df.index[0][0])
                                series_label = series_label.split(' : ')[0]
                                series_label = series_label.rsplit('.',1)[-1]
                                 
                                df.columns = df.columns.droplevel(0)
                                df.index = df.index.droplevel(0)
                                df.rename(columns = {'@': 'Total'}, inplace=True)
      
                                '----BASE VIEWS--------------------------------------------------'
                                  
                                if view.is_base() and view.is_weighted():
      
                                    ''' create a key '''
                                    key = '.'.join(matches[0])                                                                   
                                      
                                    if not key in gridsbases:
                                        gridsbases[key] = []
                                          
                                    for col in df.columns:
                                        s = df[col].copy()
                                        s.name = series_label
                                        gridsbases[key].append(s)
      
                                '----PCT VIEWS---------------------------------------------------'
                                  
                                if view.is_pct() and view.is_weighted():
  
                                    ''' create a key '''
                                    key = '.'.join(matches[0])                                                                   
                                      
                                    if not key in groupofgrids:
                                        groupofgrids[key] = []
                                          
                                    for col in df.columns:
                                        s = df[col].copy()
                                        s.name = series_label
                                        groupofgrids[key].append(s)
                                        
                                         
            numofdownbreaks = len(cluster[cluster.keys()[0]].content_of_axis)
 
            for downbreak_idx in range(0, numofdownbreaks):               
                for chain in chain_generator(cluster):
 
                    crossbreak = chain.source_name
                    downbreak = chain.content_of_axis[downbreak_idx]
                    matches = table_pattern.findall(downbreak)
                    
                    '----PULL METADATA DETAILS FROM STACK------------------------------'
                    
                    if force_chart:
                        meta_props = []
                        chart_type = default_props['chart_type']
                        crossbreak = default_props['crossbreak']
                        sort_order = default_props['sort_order']
                    else:
                        meta_props = meta['columns'][downbreak]['properties']
                        chart_type = meta_props['chart_type'] if 'chart_type' in meta_props else default_props['chart_type']
                        layout_type = meta_props['chart_layout'] if 'chart_layout' in meta_props else default_props['chart_layout']
                        sort_order = meta_props['sort_order'] if 'sort_order' in meta_props else default_props['sort_order']
                        fixed_categories = meta_props['fixed_categories'] if 'fixed_categories' in meta_props else default_props['fixed_categories']
                        slide_title_text = meta_props['slide_title'] if 'slide_title' in meta_props else default_props['slide_title_text']
                        copied_from = meta_props['copied_from'] if 'copied_from' in meta_props else default_props['copied_from'] 
                        base_description = meta_props['base_text'] if 'base_text' in meta_props else default_props['base_description']                     
                    
                    '----FIND SPECIFIED CROSSBREAK FROM TABLE--------------------------'

                    if 'crossbreak' in meta_props:
                        if meta_props['crossbreak'] != '@':
                            target_crossbreaks = default_props['crossbreak'] + meta_props['crossbreak'].split(',')
                        else:
                            target_crossbreaks = meta_props['crossbreak'].split(',')
                    else:
                        target_crossbreaks = default_props['crossbreak']
                                        
                    if crossbreak in target_crossbreaks:
                        for v in chain.views:

                            '----BUILD DATAFRAME---------------------------------------------'
                            view = chain[chain.data_key][chain.filter][downbreak][crossbreak][v]
                            
                            vkey = v.split('|')
                            func = vkey[1]
                            relation = vkey[2]
                            rel_to = vkey[3]
                            weight = vkey[4]
                            name = vkey[5]
 
                            #drop hidden codes
                            vdf = drop_hidden_codes(view)
                            
                            if view.is_pct() and view.is_weighted():
                                if sort_order == 'ascending':
                                    vdf = auto_sort(vdf, fixed_categories, column_position=0, ascend=True)
                                elif sort_order == 'descending':
                                    vdf = auto_sort(vdf, fixed_categories, column_position=0, ascend=False)

                            #add question and value labels to df
                            df = paint_df(vdf, view, meta, text_key)   
                            #get base description
                            base_description = view.meta()['agg']['text'].strip()  
                                
                            question_label = strip_html_tags(df.index[0][0]) 
                            
                            df = partition_view_df(df)[0]

                            df = rename_label(df, '@', 'Total', orientation='Top')

                            '----BASE VIEWS--------------------------------------------------'
                             
                            if view.is_base() and view.is_weighted():
     
                                '----IF GRID THEN--------------------------------------------------'
                                 
                                matches = table_pattern.findall(downbreak)
 
                                if (len(matches) > 0 and len(matches[0]) == 2):    
                                    key = '.'.join(matches[0])
                                    if key in gridsbases.keys():
                                         
                                        df_grid_base = pd.concat(gridsbases[key], axis=1)
                                     
                                        base_text = get_base(
                                            df_grid_base, 
                                            base_description
                                        )
                                         
                                        gridsbases.pop(key)
                                 
                                '----IF NON-GRID THEN-----------------------------------------------'
                                 
                                base_text = get_base(
                                    df, 
                                    base_description
                                )
                             
                            '----WEIGHTED PCT VIEWS------------------------------------------'
                             
                            if view.is_pct() and view.is_weighted():
 
                                '----IF GRID THEN--------------------------------------------------'
                                 
                                matches = table_pattern.findall(downbreak)
 
                                if (len(matches) > 0 and len(matches[0]) == 2):    
                                    key = '.'.join(matches[0])
                                    if key in groupofgrids.keys():
                                         
                                        slide_num+=1
                                         
                                        print('\n{indent:>5}Slide {slide_number}. Adding a STACKED BAR CHART '
                                              'for {question_name} cut by Total{warning_msg}'.format(
                                                indent='',
                                                slide_number=slide_num, 
                                                question_name=downbreak,
                                                warning_msg=''
                                            )
                                        )
 
                                        df_grid_table = pd.concat(groupofgrids[key], axis=1)
                                         
                                        question_label = meta['masks'][key]['text']['en-GB']
                                        question_label = strip_html_tags(question_label)
                                         
                                        df_grid_table = df_grid_table/100
 
                                        '----ADDPEND SLIDE TO PRES----------------------------------------------------'
                                          
                                        slide_layout = prs.slide_masters[0].slide_layouts[1]
                                        slide = prs.slides.add_slide(slide_layout)
                                          
                                        '----ADD SHAPES TO SLIDE------------------------------------------------------'

                                        ''' sub title shape '''
                                        sub_title_shp = add_textbox(
                                            slide, 
                                            text=question_label, 
                                            font_size=12, 
                                            font_italic=True,
                                            left=284400, 
                                            top=1007999, 
                                            width=8582400, 
                                            height=468000
                                        )
                                           
                                        ''' chart shape '''
                                        chart_shp = add_stacked_bar_chart(
                                            slide, 
                                            df_grid_table,
                                            caxis_tick_label_position='low'
                                            )
                                         
                                        ''' footer shape '''   
                                        base_text_shp = add_textbox(
                                            slide, 
                                            text=base_text, 
                                            font_size=8,
                                            left=284400, 
                                            top=5652000, 
                                            width=8582400, 
                                            height=396000
                                        )
                                         
                                        groupofgrids.pop(key)
                                         
                                '----IF NON-GRID TABLES---------------------------------------------'
 
                                df_table = df
                                question_label = strip_html_tags(meta['columns'][downbreak]['text']['en-GB'])
                                 
                                '----SPLIT DFS & LOOP OVER THEM-------------------------------------'
                                 
                                collection_of_dfs = df_splitter(
                                    df_table, 
                                    min_rows=5, 
                                    max_rows=15
                                )
                                 
                                for i, df_table_slice in enumerate(collection_of_dfs):
                                     
                                    slide_num += 1
                                     
                                    print('\n{indent:>5}Slide {slide_number}. Adding a {chart_name}'
                                          'CHART for {question_name} cut by {crossbreak_name} {x}'.format(
                                            indent='',
                                            slide_number=slide_num, 
                                            chart_name=chart_type.upper(), 
                                            question_name=downbreak, 
                                            crossbreak_name=crossbreak,
                                            x='(cont ('+str(i)+'))' if i > 0 else ''
                                        )
                                    )
                                  
                                    numofcols = len(df_table_slice.columns)
                                    numofrows = len(df_table_slice.index)

                                    df_table_slice = df_table_slice/100
 
                                    '----ADDPEND SLIDE TO PRES----------------------------------------------------'
                                      
                                    slide_layout = prs.slide_masters[0].slide_layouts[1]
                                    slide = prs.slides.add_slide(slide_layout)
                                      
                                    '----ADD SHAPES TO SLIDE------------------------------------------------------'

                                    ''' title shape '''
                                    if i > 0:
                                        slide_title_text_cont = '%s (continued %s)' % (slide_title_text, i+1) 
                                        title_placeholder_shp = slide.placeholders[24]
                                        title_placeholder_shp.text = slide_title_text_cont

                                    ''' sub title shape '''
                                    sub_title_shp = add_textbox(
                                        slide, 
                                        text=question_label, 
                                        font_size=12, 
                                        font_italic=True,
                                        left=284400, 
                                        top=1007999, 
                                        width=8582400, 
                                        height=468000
                                    )
                                      
                                    ''' chart shape '''
                                    #single series table with less than 3 categories = pie
                                    if numofcols == 1 and numofrows <= 3:
                                        chart_type='pie'
                                    #handle incorrect chart type requests - e.g. pie chart cannot handle more than 1 column   
                                    elif chart_type == 'pie' and numofcols > 1:
                                        chart_type='bar'
                                    #turn legend off if table contains 1 series unless its a pie chart
                                    if numofcols == 1:
                                        legend_switch=False
                                        if chart_type == 'pie':
                                            legend_switch=True
                                    else:
                                        legend_switch=True

                                    if chart_type == 'bar':                                                                   
                                        label_split_switch='low'
                                    else:
                                        label_split_switch='none'
                                        
                                    #add chart    
                                    if chart_type == 'bar':
                                        chart_shp = chart_selector(slide, 
                                                                   df_table_slice, 
                                                                   chart_type, 
                                                                   has_legend=legend_switch,
                                                                   caxis_tick_label_position=label_split_switch)
  
                                    else:
                                        chart_shp = chart_selector(slide, 
                                                                   df_table_slice, 
                                                                   chart_type,
                                                                   has_legend=legend_switch)

                                    ''' footer shape '''   
                                    base_text_shp = add_textbox(
                                        slide, 
                                        text=base_text, 
                                        font_size=8,
                                        left=284400, 
                                        top=5652000, 
                                        width=8582400, 
                                        height=396000
                                    )
      
            prs.save('{pres_path}\\{pres_name}_({cluster_name}).pptx'.format(
                pres_path=path_pptx,
                pres_name=chain.data_key,
                cluster_name=cluster.name
                )
            )
        
    pptx_elapsed_time = time.time() - pptx_start_time     
    print('\n{indent:>2}Presentation saved, '
        'time elapsed: {time:.2f} seconds\n\n{line}'.format(
        indent='',
        time=pptx_elapsed_time, 
        line= '_' * 80
        )
    )

    