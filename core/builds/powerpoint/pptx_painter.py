
from pptx import Presentation

import numpy as np
import pandas as pd

from quantipy.core.helpers import functions as helpers
from quantipy.core.builds.powerpoint.add_shapes import *
from quantipy.core.builds.powerpoint.transformations import *

from os.path import ( 
    basename, 
    dirname
    )

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

def get_base(df, base_description, mtd_origin="1-Omni", grid_summary=False):
    

    numofcols = len(df.columns)
    numofrows = len(df.index)

    top_members = df.columns.values
    base_values = df.values
    if not base_description:
        base_description = df.index.values[0] 
     
    #single series format
    if numofcols == 1:
        base_text = base_description.strip() + " (" + str(int(base_values[0][0])) +") "
     
    #multi series format
    elif numofcols > 1:
        if all_same(base_values[0]):
            base_text = base_description.strip() + " (" +  str(int(base_values[0][0])) + ") "
        else:
            base_text = base_description.strip() + " - " + ", ".join([
                '{} ({})'.format(x,str(int(y))) 
                for x,y in zip(top_members, base_values[0])
            ]) 
    
    return base_text

'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

def PowerPointPainter(full_template_path,
					  pptx_output_path,
					  meta, 
					  clusters,
					  cut_by_these_crsbrks=['@'],
					  chart_type='bar',
                      text_key=None):
    """
    Builds PPTX file from cluster, list of clusters, or dictionary of
    clusters.

    param: full_template_path - path to YG pptx template
    param: pptx_output_path - path to where the newly generated pptx
        will be stored
    param: meta - metadata as dictionary used to paint
        datframes
    param: clusters - quantipy clusters, must be provided in a list
    """
    
    #check cluster type, if not a list, convert to list.
    if not isinstance(clusters, list):
        clusters = [clusters]
    
    #get the default text key if none provided
    if text_key is None:
        text_key = meta['lib']['default text']
    
    #loop over clusters, returns pptx for each cluster.
    for i, cluster in enumerate(clusters):
        
        save_prs = False
        
        prs = Presentation(full_template_path)
        
        target_crossbreaks = cut_by_these_crsbrks
        
        numofdownbreaks = len(cluster[cluster.keys()[0]].content_of_axis)
        
        print('\nBuilding PPTX for {file_name}').format(file_name=cluster.name)
    
        for x in range(0, numofdownbreaks):   
            for chk in cluster:

                chain = cluster[chk]
                side = chain.content_of_axis[x]
                top = chain.source_name

                if top in target_crossbreaks:
                    for idx, v in enumerate(chain.views):

                        view = chain[chain.data_key][chain.filter][side][top][v]
                        
                        vkey = v.split('|')
                        weight = vkey[0]
                        func = vkey[1]
                        relation = vkey[2]
                        rel_to = vkey[3]
                        weight = vkey[4]
                        name = vkey[5]

                        #drop hidden codes
                        if 'x_hidden_codes' in view.meta:
                            vdf = helpers.deep_drop(
                                view.dataframe, 
                                view.meta['x_hidden_codes'], 
                                axes=0
                            )
                        else:
                            vdf = view.dataframe
                             
                        #add question and value labels to df
                        if 'x_new_order' in view.meta:
                            df = helpers.paint_dataframe(
                                df=vdf.copy(), 
                                meta=meta, 
                                ridx=view.meta['x_new_order'], 
                                text_key=text_key
                            )
                        else:
                            df = helpers.paint_dataframe(
                                df=vdf.copy(), 
                                meta=meta, 
                                text_key=text_key
                            )
                        
                        #get question label
                        question_label = df.index[0][0]
                         
                        #remove nesting
                        df.columns = df.columns.droplevel(0)
                        df.index = df.index.droplevel(0)
                         
                        #rename @1 as Total
                        df.rename(columns = {'@1': 'Total'}, inplace=True)
                         
                        #base df
                        if v.startswith('x|freq|x:y||') and weight!='':
                             
                            base_description = view.meta['agg']['text'].strip()
                            base_text = get_base(
                                df, 
                                base_description, 
                                grid_summary=False
                            )   
                         
                        #percentage df
                        if v.startswith('x|freq||y|') and weight!='':
                            
                            ''' clean and round cells '''
                            df_table = df
                            df_table.columns = pd.Series(df_table.columns).str.replace('&', 'and')
                            df_table.index = pd.Series(df_table.index).str.replace('&', 'and')

                            '----SPLIT DFS & LOOP OVER THEM--------------------------------------------------'

                            collection_of_dfs = df_splitter(
                                df_table, 
                                min_rows=5, 
                                max_rows=15
                            )
                            
                            for i, df_table_slice in enumerate(collection_of_dfs):
                            
                                numofcols = len(df_table_slice.columns)
                                numofrows = len(df_table_slice.index)

                                df_table_slice = df_table_slice/100
                                
                                '----ADDPEND SLIDE TO PRES----------------------------------------------------'
                                
                                slide_layout = prs.slide_masters[0].slide_layouts[1]
                                slide = prs.slides.add_slide(slide_layout)
                                
                                '----ADD SHAPES TO SLIDE------------------------------------------------------'

                                ''' title shape '''
                                slide_title_text = "Click to add slide title"
                                
                                if i > 0:
                                    slide_title_text_cont = (
                                        '%s (continued %s)' % (
                                            slide_title_text, 
                                            i+1
                                        )
                                    ) 
                                else:
                                    slide_title_text_cont = slide_title_text
                                     
                                slide_title = add_textbox(
                                    slide, 
                                    text=slide_title_text_cont, 
                                    font_color=(0,0,0),
                                    font_size=36, 
                                    font_bold=False, 
                                    vertical_alignment='middle',
                                    left=284400, 
                                    top=309600, 
                                    width=8582400, 
                                    height=691200
                                )

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
                                    chart = chart_selector(
                                        slide, 
                                        df_table_slice, 
                                        'pie', 
                                        has_legend=True
                                    )

                                #handle incorrect chart type requests - pie chart cannot handle more than 1 column    
                                elif chart_type == 'pie' and numofcols > 1:
                                    chart = chart_selector(
                                        slide, 
                                        df_table_slice, 
                                        chart_type, 
                                        has_legend=True
                                    )
                                    #chart_type='bar'
                                #single series table with more than, equal to 4 categories and is not a 
                                #pie chart = chart type selected dynamically chart type with no legend
                                elif numofcols == 1 and chart_type != 'pie':
                                    chart = chart_selector(
                                        slide, 
                                        df_table_slice, 
                                        chart_type, 
                                        has_legend=False
                                    )
                                else:
                                    #multi series tables = dynamic chart type with legend 
                                    chart = chart_selector(
                                        slide, 
                                        df_table_slice, 
                                        chart_type, 
                                        has_legend=True
                                    )
                                    
                                ''' footer shape '''   
                                base_text_shape = add_textbox(
                                    slide, 
                                    text=base_text, 
                                    font_size=8,
                                    left=284400, 
                                    top=5652000, 
                                    width=8582400, 
                                    height=396000
                                )

                            save_prs = True

        if save_prs:
            prs.save('{pres_path}\\{pres_name}_({cluster_name}).pptx'.format(
                pres_path=pptx_output_path,
                pres_name=chain.data_key,
                cluster_name=cluster.name
            ))
        
            print('\nprs saved')

    print('\nScript Completed')
                                                                 
               