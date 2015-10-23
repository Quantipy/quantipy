import unittest
import os.path
import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal
import test_helper
import copy
import json

from operator import lt, le, eq, ne, ge, gt

from pandas.core.index import Index
__index_symbol__ = {
    Index.union: ',',
    Index.intersection: '&',
    Index.difference: '~',
    Index.sym_diff: '^'
}

from collections import defaultdict, OrderedDict
from quantipy.core.stack import Stack
from quantipy.core.chain import Chain
from quantipy.core.link import Link
from quantipy.core.view_generators.view_mapper import ViewMapper
from quantipy.core.view_generators.view_maps import QuantipyViews
from quantipy.core.view import View
from quantipy.core.helpers import functions
from quantipy.core.helpers.functions import load_json
from quantipy.core.tools.dp.prep import (
    start_meta,
    frange,
    frequency,
    crosstab
)
from quantipy.core.tools.view.query import get_dataframe


class TestMerging(unittest.TestCase):

    def setUp(self):
        self.path = './tests/'
#         self.path = ''
        project_name = 'Example Data (A)'

        # Load Example Data (A) data and meta into self
        name_data = '%s.csv' % (project_name)
        path_data = '%s%s' % (self.path, name_data)
        self.example_data_A_data = pd.DataFrame.from_csv(path_data)
        name_meta = '%s.json' % (project_name)
        path_meta = '%s%s' % (self.path, name_meta)
        self.example_data_A_meta = load_json(path_meta)

        # Variables by type for Example Data A
        self.dk = 'Example Data (A)'
        self.fk = 'no_filter'
        self.single = ['gender', 'locality', 'ethnicity', 'religion', 'q1']
        self.delimited_set = ['q2', 'q3', 'q8', 'q9']
        self.q5 = ['q5_1', 'q5_2', 'q5_3']
                  
    def test_hmerge(self):
                     
        meta = self.example_data_A_meta
        data = self.example_data_A_data
        
        meta_l, data_l = subset_dataset(
            meta, data,
            id='unique_id',
            columns=['gender', 'locality', 'ethnicity', 'q2', 'q3']
        )
        
#         print json.dumps(meta_l)
#         data_l.to_excel('./test.xlsx')
                                                
        ################## values    
        
        
        
        
# ##################### Helper functions #####################

def subset_dataset(meta, data, id, columns):
    
    all_columns = [id] + columns
    sdata = data[all_columns].copy()
    
    smeta = start_meta(text_key='en-GB')
    
    for col in all_columns:
        smeta['columns'][col] = meta['columns'][col]
    
    for col_mapper in meta['sets']['data file']['items']:
        if col_mapper.split('@')[-1] in all_columns:
            smeta['sets']['data file']['items'].append(col_mapper)
    
    return smeta, sdata
    

    