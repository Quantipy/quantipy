.. toctree::
    :maxdepth: 5
    :hidden:

====================================
``DataSet`` Dimensions compatibility
====================================

DTO-downloaded and Dimensions converted variable naming conventions are following
specific rules for ``array`` names and corresponding ``ìtems``. ``DataSet``
offers a compatibility mode for Dimensions scenarios and handles the proper
renaming automatically. Here is what you should know...

----------------------
The compatibility mode
----------------------

A ``DataSet`` will (by default) support Dimensions-like ``array`` naming for its connected data files when constructed. An ``array`` ``masks`` meta defintition
of a variable called ``q5`` looking like this...::

    {u'items': [{u'source': u'columns@q5_1', u'text': {u'en-GB': u'Surfing'}},
      {u'source': u'columns@q5_2', u'text': {u'en-GB': u'Snowboarding'}},
      {u'source': u'columns@q5_3', u'text': {u'en-GB': u'Kite boarding'}},
      {u'source': u'columns@q5_4', u'text': {u'en-GB': u'Parachuting'}},
      {u'source': u'columns@q5_5', u'text': {u'en-GB': u'Cave diving'}},
      {u'source': u'columns@q5_6', u'text': {u'en-GB': u'Windsurfing'}}],
     u'subtype': u'single',
     u'text': {u'en-GB': u'How likely are you to do each of the following in the next year?'},
     u'type': u'array',
     u'values': u'lib@values@q5'}

...will be converted into its "Dimensions equivalent" as per:

>>> dataset = qp.DataSet(name_data, dimensions_comp=True)
>>> dataset.read_quantipy(path_data+name_data, path_data+name_data)
DataSet: ../Data/Quantipy/Example Data (A)
rows: 8255 - columns: 75
Dimensions compatibilty mode: True

>>> dataset.masks()
['q5.q5_grid', 'q6.q6_grid', 'q7.q7_grid']

>>> dataset._meta['masks']['q5.q5_grid']
{u'items': [{u'source': 'columns@q5[{q5_1}].q5_grid',
   u'text': {u'en-GB': u'Surfing'}},
  {u'source': 'columns@q5[{q5_2}].q5_grid',
   u'text': {u'en-GB': u'Snowboarding'}},
  {u'source': 'columns@q5[{q5_3}].q5_grid',
   u'text': {u'en-GB': u'Kite boarding'}},
  {u'source': 'columns@q5[{q5_4}].q5_grid',
   u'text': {u'en-GB': u'Parachuting'}},
  {u'source': 'columns@q5[{q5_5}].q5_grid',
   u'text': {u'en-GB': u'Cave diving'}},
  {u'source': 'columns@q5[{q5_6}].q5_grid',
   u'text': {u'en-GB': u'Windsurfing'}}],
 'name': 'q5.q5_grid',
 u'subtype': u'single',
 u'text': {u'en-GB': u'How likely are you to do each of the following in the next year?'},
 u'type': u'array',
 u'values': 'lib@values@q5.q5_grid'}

-------------------------------------
Accessing and creating ``array`` data
-------------------------------------

Since new names are converted automatically by ``DataSet`` methods, there is
no need to write down the full (DTO-like) Dimensions ``array`` name when adding
new metadata. However, querying variables is always requiring the proper name:

>>> name, qtype, label = 'array_var', 'single', 'ARRAY LABEL'
>>> cats =  ['A', 'B', 'C']
>>> items = ['1', '2', '3']
>>> dataset.add_meta(name, qtype, label, cats, items)

>>> dataset.masks()
['q5.q5_grid', 'array_var.array_var_grid', 'q6.q6_grid', 'q7.q7_grid']

>>> dataset.meta('array_var.array_var_grid')
single                                                                   items item texts  codes texts missing
array_var.array_var_grid: ARRAY LABEL
1                                      array_var[{array_var_1}].array_var_grid          1      1     A    None
2                                      array_var[{array_var_2}].array_var_grid          2      2     B    None
3                                      array_var[{array_var_3}].array_var_grid          3      3     C    None

>>> dataset['array_var.array_var_grid'].head(5)
   array_var[{array_var_1}].array_var_grid  array_var[{array_var_2}].array_var_grid  array_var[{array_var_3}].array_var_grid
0                                      NaN                                      NaN                                      NaN
1                                      NaN                                      NaN                                      NaN
2                                      NaN                                      NaN                                      NaN
3                                      NaN                                      NaN                                      NaN
4                                      NaN                                      NaN                                      NaN

As can been seen above, both the ``masks`` name as well as the ``array`` item
elements are being properly converted to match DTO/Dimensions
conventions.

When using ``rename()``, ``copy()`` or ``transpose()``, the same behaviour
applies:

>>> dataset.rename('q6.q6_grid', 'q6new')
>>> dataset.masks()
['q5.q5_grid', 'array_var.array_var_grid', 'q6new.q6new_grid', 'q7.q7_grid']

>>> dataset.copy('q6new.q6new_grid', suffix='q6copy')
>>> dataset.masks()
['q5.q5_grid', 'q6new_q6copy.q6new_q6copy_grid', 'array_var.array_var_grid', 'q6new.q6new_grid', 'q7.q7_grid']

>>> dataset.transpose('q6new_q6copy.q6new_q6copy_grid')
>>> dataset.masks()
['q5.q5_grid', 'q6new_q6copy_trans.q6new_q6copy_trans_grid', 'q6new_q6copy.q6new_q6copy_grid', 'array_var.array_var_grid', 'q6new.q6new_grid', 'q7.q7_grid']