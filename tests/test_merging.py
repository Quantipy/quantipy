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
    crosstab,
    subset_dataset,
    hmerge,
    vmerge
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
   
    def test_subset_dataset(self):

        meta = self.example_data_A_meta
        data = self.example_data_A_data

        # Create left dataset
        subset_columns_l = [
            'unique_id', 'gender', 'locality', 'ethnicity', 'q2', 'q3']
        subset_rows_l = 10
        subset_cols_l = len(subset_columns_l)
        meta_l, data_l = subset_dataset(
            meta, data[:10],
            columns=subset_columns_l
        )
        self.assertItemsEqual(meta_l['columns'].keys(), subset_columns_l)
        datafile_items = meta_l['sets']['data file']['items']
        datafile_columns = [item.split('@')[-1]for item in datafile_items]
        self.assertItemsEqual(meta_l['columns'].keys(), datafile_columns)
        self.assertItemsEqual(data_l.columns.tolist(), datafile_columns)
        self.assertItemsEqual(data_l.columns.tolist(), subset_columns_l)
        self.assertEqual(data_l.shape, (subset_rows_l, subset_cols_l))
        dataset_left = (meta_l, data_l)
        
        # Create right dataset
        subset_columns_r = [
            'unique_id', 'gender', 'religion', 'q1', 'q2', 'q8', 'q9']
        subset_rows_r = 10
        subset_cols_r = len(subset_columns_r)
        meta_r, data_r = subset_dataset(
            meta, data[5:15],
            columns=subset_columns_r
        )
        self.assertItemsEqual(meta_r['columns'].keys(), subset_columns_r)
        datafile_items = meta_r['sets']['data file']['items']
        datafile_columns = [item.split('@')[-1]for item in datafile_items]
        self.assertItemsEqual(meta_r['columns'].keys(), datafile_columns)
        self.assertItemsEqual(data_r.columns.tolist(), datafile_columns)
        self.assertItemsEqual(data_r.columns.tolist(), subset_columns_r)
        self.assertEqual(data_r.shape, (subset_rows_r, subset_cols_r))
        dataset_right = (meta_r, data_r)
        
    def test_hmerge_basic(self):

        meta = self.example_data_A_meta
        data = self.example_data_A_data

        # Create left dataset
        subset_columns_l = [
            'unique_id', 'gender', 'locality', 'ethnicity', 'q2', 'q3']
        meta_l, data_l = subset_dataset(
            meta, data[:10],
            columns=subset_columns_l)
        dataset_left = (meta_l, data_l)
        
        # Create right dataset
        subset_columns_r = [
            'unique_id', 'gender', 'religion', 'q1', 'q2', 'q8', 'q9']
        meta_r, data_r = subset_dataset(
            meta, data[5:15],
            columns=subset_columns_r)
        dataset_right = (meta_r, data_r)
        
        # hmerge datasets
        meta_hm, data_hm = hmerge(
            dataset_left, dataset_right,
            left_on='unique_id', right_on='unique_id',
            verbose=False)
        
#         print data_l
#         print ''
#         print data_r
#         print ''
#         print data_hm
#         
#         actual = data_hm[data_l.columns].values
#         expected = data_l.values
#         self.assertItemsEqual(actual, expected)
        
        combined_columns = data_l.columns.union(data_r.columns)
        self.assertItemsEqual(meta_hm['columns'].keys(), combined_columns)
        self.assertItemsEqual(meta_hm['columns'].keys(), combined_columns)
        datafile_items = meta_hm['sets']['data file']['items']
        datafile_columns = [item.split('@')[-1]for item in datafile_items]
        self.assertItemsEqual(meta_hm['columns'].keys(), datafile_columns)
        self.assertEqual(data_hm.shape, (data_l.shape[0], len(combined_columns)))
        
        # hmerge datasets
        meta_hm, data_hm = hmerge(
            dataset_left, dataset_right,
            on='unique_id',
            verbose=False)
        
        combined_columns = data_l.columns.union(data_r.columns)
        self.assertItemsEqual(meta_hm['columns'].keys(), combined_columns)
        self.assertItemsEqual(meta_hm['columns'].keys(), combined_columns)
        datafile_items = meta_hm['sets']['data file']['items']
        datafile_columns = [item.split('@')[-1]for item in datafile_items]
        self.assertItemsEqual(meta_hm['columns'].keys(), datafile_columns)
        self.assertEqual(data_hm.shape, (data_l.shape[0], len(combined_columns)))
        
    def test_vmerge_basic(self):
 
        meta = self.example_data_A_meta
        data = self.example_data_A_data
 
        # Create left dataset
        subset_columns_l = [
            'unique_id', 'gender', 'locality', 'ethnicity', 'q2', 'q3']
        meta_l, data_l = subset_dataset(
            meta, data[:10],
            columns=subset_columns_l)
        dataset_left = (meta_l, data_l)
         
        # Create right dataset
        subset_columns_r = [
            'unique_id', 'gender', 'religion', 'q1', 'q2', 'q8', 'q9']
        meta_r, data_r = subset_dataset(
            meta, data[5:15],
            columns=subset_columns_r)
        dataset_right = (meta_r, data_r)
         
        # vmerge datasets using left_on/right_on
        dataset_left = (meta_l, data_l)
        meta_vm, data_vm = vmerge(
            dataset_left, dataset_right,
            left_on='unique_id', right_on='unique_id',
            verbose=False)
         
        combined_columns = data_l.columns.union(data_r.columns)
        self.assertItemsEqual(meta_vm['columns'].keys(), combined_columns)
        self.assertItemsEqual(meta_vm['columns'].keys(), combined_columns)
        datafile_items = meta_vm['sets']['data file']['items']
        datafile_columns = [item.split('@')[-1]for item in datafile_items]
        self.assertItemsEqual(meta_vm['columns'].keys(), datafile_columns)
        ids_left = data_l['unique_id']
        ids_right = data_r['unique_id']
        unique_ids = list(set(ids_left).union(set(ids_right)))
        self.assertEqual(data_vm.shape, (len(unique_ids), len(combined_columns)))
        self.assertItemsEqual(data_vm['unique_id'], unique_ids)
         
        # vmerge datasets using on
        dataset_left = (meta_l, data_l)
        meta_vm, data_vm = vmerge(
            dataset_left, dataset_right,
            on='unique_id',
            verbose=False)
 
        combined_columns = data_l.columns.union(data_r.columns)
        self.assertItemsEqual(meta_vm['columns'].keys(), combined_columns)
        self.assertItemsEqual(meta_vm['columns'].keys(), combined_columns)
        datafile_items = meta_vm['sets']['data file']['items']
        datafile_columns = [item.split('@')[-1]for item in datafile_items]
        self.assertItemsEqual(meta_vm['columns'].keys(), datafile_columns)
        ids_left = data_l['unique_id']
        ids_right = data_r['unique_id']
        unique_ids = list(set(ids_left).union(set(ids_right)))
        self.assertEqual(data_vm.shape, (len(unique_ids), len(combined_columns)))
        self.assertItemsEqual(data_vm['unique_id'], unique_ids)
         
    def test_vmerge_row_id(self):
 
        meta = self.example_data_A_meta
        data = self.example_data_A_data
 
        # Create left dataset
        subset_columns_l = [
            'unique_id', 'gender', 'locality', 'ethnicity', 'q2', 'q3']
        meta_l, data_l = subset_dataset(
            meta, data[:10],
            columns=subset_columns_l)
        dataset_left = (meta_l, data_l)
         
        # Create right dataset
        subset_columns_r = [
            'unique_id', 'gender', 'religion', 'q1', 'q2', 'q8', 'q9']
        meta_r, data_r = subset_dataset(
            meta, data[5:15],
            columns=subset_columns_r)
        dataset_right = (meta_r, data_r)
         
        # vmerge datasets indicating row_id
        dataset_left = (meta_l, data_l)
        meta_vm, data_vm = vmerge(
            dataset_left, dataset_right,
            on='unique_id',
            row_id_name='DataSource',
            left_id=1, right_id=2,
            verbose=False)
         
        expected = {'text': {'en-GB': 'vmerge row id'}, 'type': 'int', 'name': 'DataSource'}
        actual = meta_vm['columns']['DataSource']
        self.assertEqual(actual, expected)
        self.assertTrue(data_vm['DataSource'].dtype=='int64')
         
        combined_columns = list(data_l.columns.union(data_r.columns))
        combined_columns.append('DataSource')
        self.assertItemsEqual(meta_vm['columns'].keys(), combined_columns)
        self.assertItemsEqual(meta_vm['columns'].keys(), combined_columns)
        datafile_items = meta_vm['sets']['data file']['items']
        datafile_columns = [item.split('@')[-1]for item in datafile_items]
        self.assertItemsEqual(meta_vm['columns'].keys(), datafile_columns)
        ids_left = data_l['unique_id']
        ids_right = data_r['unique_id']
        unique_ids = list(set(ids_left).union(set(ids_right)))
        self.assertEqual(data_vm.shape, (len(unique_ids), len(combined_columns)))
        self.assertItemsEqual(data_vm['unique_id'], unique_ids)
         
        # vmerge datasets indicating row_id
        dataset_left = (meta_l, data_l)
        meta_vm, data_vm = vmerge(
            dataset_left, dataset_right,
            on='unique_id',
            row_id_name='DataSource',
            left_id=1, right_id=2.0,
            verbose=False)
         
        expected = {'text': {'en-GB': 'vmerge row id'}, 'type': 'float', 'name': 'DataSource'}
        actual = meta_vm['columns']['DataSource']
        self.assertEqual(actual, expected)
        self.assertTrue(data_vm['DataSource'].dtype=='float64')
         
        # vmerge datasets indicating row_id
        dataset_left = (meta_l, data_l)
        meta_vm, data_vm = vmerge(
            dataset_left, dataset_right,
            on='unique_id',
            row_id_name='DataSource',
            left_id='W1', right_id=2.0,
            verbose=False)
         
        expected = {'text': {'en-GB': 'vmerge row id'}, 'type': 'str', 'name': 'DataSource'}
        actual = meta_vm['columns']['DataSource']
        self.assertEqual(actual, expected)
        self.assertTrue(data_vm['DataSource'].dtype=='object')
         
    def test_hmerge_vmerge_basic(self):
 
        meta = self.example_data_A_meta
        data = self.example_data_A_data
 
        # Create left dataset
        subset_columns_l = [
            'unique_id', 'gender', 'locality', 'ethnicity', 'q2', 'q3']
        meta_l, data_l = subset_dataset(
            meta, data[:10],
            columns=subset_columns_l)
        dataset_left = (meta_l, data_l)
         
        # Create right dataset
        subset_columns_r = [
            'unique_id', 'gender', 'religion', 'q1', 'q2', 'q8', 'q9']
        meta_r, data_r = subset_dataset(
            meta, data[5:15],
            columns=subset_columns_r)
        dataset_right = (meta_r, data_r)
         
        # hmerge datasets
        meta_hm, data_hm = hmerge(
            dataset_left, dataset_right,
            left_on='unique_id', right_on='unique_id',
            verbose=False)
         
        # vmerge datasets
        dataset_left = (meta_hm, data_hm)
        meta_vm, data_vm = vmerge(
            dataset_left, dataset_right,
            left_on='unique_id', right_on='unique_id',
            verbose=False)
         
        combined_columns = data_l.columns.union(data_r.columns)
        self.assertItemsEqual(meta_hm['columns'].keys(), combined_columns)
        self.assertItemsEqual(meta_hm['columns'].keys(), combined_columns)
        datafile_items = meta_hm['sets']['data file']['items']
        datafile_columns = [item.split('@')[-1]for item in datafile_items]
        self.assertItemsEqual(meta_hm['columns'].keys(), datafile_columns)
        ids_left = data_l['unique_id']
        ids_right = data_r['unique_id']
        unique_ids = list(set(ids_left).union(set(ids_right)))
        self.assertEqual(data_vm.shape, (len(unique_ids), len(combined_columns)))
        
# ##################### Helper functions #####################


    