warning: LF will be replaced by CRLF in .pytest_cache/v/cache/nodeids.
The file will have its original line endings in your working directory
[1mdiff --git a/.pytest_cache/v/cache/nodeids b/.pytest_cache/v/cache/nodeids[m
[1mindex 48d40010..fc4d323c 100644[m
[1m--- a/.pytest_cache/v/cache/nodeids[m
[1m+++ b/.pytest_cache/v/cache/nodeids[m
[36m@@ -3,10 +3,10 @@[m
   "tests/test_banked_chains.py::TestBankedChains::test_banked_chain_structure_weighted", [m
   "tests/test_banked_chains.py::TestBankedChains::test_cluster_add_chain", [m
   "tests/test_banked_chains.py::TestBankedChains::test_verify_banked_chain", [m
[32m+[m[32m  "tests/test_batch.py::TestBatch::test_add_crossbreak",[m[41m [m
[32m+[m[32m  "tests/test_batch.py::TestBatch::test_add_downbreak",[m[41m [m
   "tests/test_batch.py::TestBatch::test_add_filter", [m
   "tests/test_batch.py::TestBatch::test_add_open_ends", [m
[31m-  "tests/test_batch.py::TestBatch::test_add_x", [m
[31m-  "tests/test_batch.py::TestBatch::test_add_y", [m
   "tests/test_batch.py::TestBatch::test_add_y_on_y", [m
   "tests/test_batch.py::TestBatch::test_as_addition", [m
   "tests/test_batch.py::TestBatch::test_copy", [m
[1mdiff --git a/tests/test_batch.py b/tests/test_batch.py[m
[1mindex 6fe7e989..dad3c6b6 100644[m
[1m--- a/tests/test_batch.py[m
[1m+++ b/tests/test_batch.py[m
[36m@@ -29,8 +29,8 @@[m [mdef _get_batch(name, dataset=None, full=False):[m
 	if not dataset: dataset = _get_dataset()[m
 	batch = qp.Batch(dataset, name)[m
 	if full:[m
[31m-		batch.add_x(['q1', 'q2', 'q6', 'age'])[m
[31m-		batch.add_y(['gender', 'q2'])[m
[32m+[m		[32mbatch.add_downbreak(['q1', 'q2', 'q6', 'age'])[m
[32m+[m		[32mbatch.add_crossbreak(['gender', 'q2'])[m
 		batch.add_open_ends(['q8a', 'q9a'], 'RecordNo')[m
 		batch.add_filter('men only', {'gender': 1})[m
 		batch.set_weights('weight_a')[m
[36m@@ -47,7 +47,7 @@[m [mclass TestBatch(unittest.TestCase):[m
 		batch1 = dataset.add_batch('batch1')[m
 		batch2 = dataset.add_batch('batch2', 'c', 'weight', .05)[m
 		self.assertTrue(isinstance(batch1, qp.Batch))[m
[31m-		self.assertEqual(len(_get_meta(batch1).keys()), 31)[m
[32m+[m		[32mself.assertEqual(len(_get_meta(batch1).keys()), 32)[m
 		b_meta = _get_meta(batch2)[m
 		self.assertEqual(b_meta['name'], 'batch2')[m
 		self.assertEqual(b_meta['cell_items'], ['c'])[m
[36m@@ -76,8 +76,8 @@[m [mclass TestBatch(unittest.TestCase):[m
 		batch1.hiding('q1', frange('8,9,96-99'))[m
 		batch1.slicing('q1', frange('9-4'))[m
 		batch2, ds = _get_batch('test2', ds)[m
[31m-		batch2.add_x('q1')[m
[31m-		batch2.add_y('Wave')[m
[32m+[m		[32mbatch2.add_downbreak('q1')[m
[32m+[m		[32mbatch2.add_crossbreak('Wave')[m
 		batch2.as_addition('test1')[m
 		n_ds = ds.from_batch('test1', 'RecordNo', 'de-DE', True, 'variables')[m
 		self.assertEqual(n_ds.codes('q1'), [7, 6, 5, 4])[m
[36m@@ -91,9 +91,9 @@[m [mclass TestBatch(unittest.TestCase):[m
 [m
 	########################## methods used in _get_batch ####################[m
 [m
[31m-	def test_add_x(self):[m
[32m+[m	[32mdef test_add_downbreak(self):[m
 		batch, ds = _get_batch('test')[m
[31m-		batch.add_x(['q1', 'q2', 'q2b', {'q3': 'q3_label'}, 'q4', {'q5': 'q5_label'}, 'q14_1'])[m
[32m+[m		[32mbatch.add_downbreak(['q1', 'q2', 'q2b', {'q3': 'q3_label'}, 'q4', {'q5': 'q5_label'}, 'q14_1'])[m
 		b_meta = _get_meta(batch)[m
 		self.assertEqual(b_meta['xks'], ['q1', 'q2', 'q2b', 'q3', 'q4', 'q5',[m
 		                 				 u'q5_1', u'q5_2', u'q5_3', u'q5_4', u'q5_5',[m
[36m@@ -115,13 +115,13 @@[m [mclass TestBatch(unittest.TestCase):[m
 							   (u'q14r10c01', ['@'])][m
 		self.assertEqual(b_meta['x_y_map'], x_y_map)[m
 [m
[31m-	def test_add_y(self):[m
[32m+[m	[32mdef test_add_crossbreak(self):[m
 		batch, ds = _get_batch('test')[m
[31m-		batch.add_y(['gender', 'q2b'])[m
[32m+[m		[32mbatch.add_crossbreak(['gender', 'q2b'])[m
 		b_meta = _get_meta(batch)[m
 		self.assertEqual(b_meta['yks'], ['@', 'gender', 'q2b'])[m
[31m-		self.assertRaises(KeyError, batch.add_y, ['@', 'GENDER'])[m
[31m-		batch.add_x('q1')[m
[32m+[m		[32mself.assertRaises(KeyError, batch.add_crossbreak, ['@', 'GENDER'])[m
[32m+[m		[32mbatch.add_downbreak('q1')[m
 		x_y_map = [('q1', ['@', 'gender', 'q2b'])][m
 		self.assertEqual(b_meta['x_y_map'], x_y_map)[m
 [m
[36m@@ -143,8 +143,8 @@[m [mclass TestBatch(unittest.TestCase):[m
 [m
 	def test_add_filter(self):[m
 		batch, ds = _get_batch('test', full=True)[m
[31m-		batch.add_x(['q1', 'q2b'])[m
[31m-		batch.add_y('gender')[m
[32m+[m		[32mbatch.add_downbreak(['q1', 'q2b'])[m
[32m+[m		[32mbatch.add_crossbreak('gender')[m
 		batch.add_filter('men only', {'gender': 1})[m
 		b_meta = _get_meta(batch)[m
 		self.assertEqual(b_meta['filter'], {'men only': {'gender': 1}})[m
[36m@@ -213,7 +213,7 @@[m [mclass TestBatch(unittest.TestCase):[m
 	def test_make_summaries_transpose_arrays(self):[m
 		batch, ds = _get_batch('test')[m
 		b_meta = _get_meta(batch)[m
[31m-		batch.add_x(['q5', 'q6', 'q14_2', 'q14_3', 'q14_1'])[m
[32m+[m		[32mbatch.add_downbreak(['q5', 'q6', 'q14_2', 'q14_3', 'q14_1'])[m
 		batch.make_summaries(None)[m
 		self.assertEqual(b_meta['summaries'], [])[m
 		batch.transpose_arrays(['q5', 'q6'], False)[m
[1mdiff --git a/tests/test_stack.py b/tests/test_stack.py[m
[1mindex e4d3a781..c3e50c83 100644[m
[1m--- a/tests/test_stack.py[m
[1m+++ b/tests/test_stack.py[m
[36m@@ -1108,17 +1108,17 @@[m [mclass TestStackObject(unittest.TestCase):[m
         b1, ds = _get_batch('test1', full=True)[m
         b2, ds = _get_batch('test2', ds, False)[m
         b3, ds = _get_batch('test3', ds, False)[m
[31m-        b1.add_x(['q1', 'q6', 'age'])[m
[31m-        b1.add_y(['gender', 'q2'])[m
[32m+[m[32m        b1.add_downbreak(['q1', 'q6', 'age'])[m
[32m+[m[32m        b1.add_crossbreak(['gender', 'q2'])[m
         b1.add_filter('men only', {'gender': 1})[m
         b1.extend_filter({'q1':{'age': [20, 21, 22]}})[m
         b1.set_weights('weight_a')[m
[31m-        b2.add_x(['q1', 'q6'])[m
[31m-        b2.add_y(['gender', 'q2'])[m
[32m+[m[32m        b2.add_downbreak(['q1', 'q6'])[m
[32m+[m[32m        b2.add_crossbreak(['gender', 'q2'])[m
         b2.set_weights('weight_b')[m
         b2.transpose_arrays('q6', True)[m
[31m-        b3.add_x(['q1', 'q7'])[m
[31m-        b3.add_y(['q2b'])[m
[32m+[m[32m        b3.add_downbreak(['q1', 'q7'])[m
[32m+[m[32m        b3.add_crossbreak(['q2b'])[m
         b3.add_y_on_y('y_on_y')[m
         b3.make_summaries(None)[m
         b3.set_weights(['weight_a', 'weight_b'])[m
[36m@@ -1255,11 +1255,11 @@[m [mclass TestStackObject(unittest.TestCase):[m
                         '[{}]'.format(v['value'])) for v in values)[m
 [m
         b1, ds = _get_batch('test1', full=True)[m
[31m-        b1.add_x(['q1', 'q2b', 'q6'])[m
[32m+[m[32m        b1.add_downbreak(['q1', 'q2b', 'q6'])[m
         b1.set_variable_text('q1', 'some new text1')[m
         b1.set_variable_text('q6', 'some new text1')[m
         b2, ds = _get_batch('test2', ds, True)[m
[31m-        b2.add_x(['q1', 'q2b', 'q6'])[m
[32m+[m[32m        b2.add_downbreak(['q1', 'q2b', 'q6'])[m
         b2.set_variable_text('q1', 'some new text2')[m
         stack = ds.populate(verbose=False)[m
         stack.aggregate(['cbase', 'counts', 'c%'], batches='all', verbose=False)[m
[1mdiff --git a/tests/test_view_manager.py b/tests/test_view_manager.py[m
[1mindex 64c8f914..3ac09f04 100644[m
[1m--- a/tests/test_view_manager.py[m
[1m+++ b/tests/test_view_manager.py[m
[36m@@ -33,8 +33,8 @@[m [mclass TestViewManager(unittest.TestCase):[m
         else:[m
             w = None[m
         batch = dataset.add_batch('viewmanager', weights=w, tests=[0.05] if tests else None)[m
[31m-        batch.add_x(x)[m
[31m-        batch.add_y(y)[m
[32m+[m[32m        batch.add_downbreak(x)[m
[32m+[m[32m        batch.add_crossbreak(y)[m
         stack = dataset.populate()[m
         basic_views = ['cbase', 'counts', 'c%', 'counts_sum', 'c%_sum'][m
         stack.aggregate(views=basic_views, verbose=False)[m
