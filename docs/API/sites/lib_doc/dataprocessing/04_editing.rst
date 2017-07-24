.. toctree::
 	:maxdepth: 5
	:includehidden:

================
Editing metadata
================

--------------------------
Creating meta from scratch
--------------------------
It is very easy to add new variable metadata to a ``DataSet`` via ``add_meta()``
which let's you create all supported variable types. Each new variable needs at
least a ``name``, ``qtype`` and ``label``. With this information a ``string``,
``int``, ``float`` or ``date`` variable can be defined, e.g.:

>>> ds.add_meta(name='new_int', qtype='int', label='My new int variable')
>>> ds.meta('new_int')
                              int
new_int: My new int variable  N/A

Using the ``categories`` parameter we can create categorical variables of type
``single`` or ``delimited set``. We can provide the ``categories`` in two
different ways:

>>> name, qtype, label = 'new_single', 'single', 'My new single variable'

**Providing a list of category labels** (codes will be enumerated starting
from ``1``):

>>> cats = ['Category A', 'Category B', 'Category C']

>>> ds.add_meta(name, qtype, label, categories=cats)
>>> ds.meta('new_single')
single                              codes       texts missing
new_single: My new single variable
1                                       1  Category A    None
2                                       2  Category B    None
3                                       3  Category C    None

**Providing a list of tuples pairing codes and labels**:

>>> cats = [(1, 'Category A'), (2, 'Category B'), (99, 'Category C')]

>>> ds.add_meta(name, qtype, label, categories=cats)
>>> ds.meta('new_single')
single                              codes       texts missing
new_single: My new single variable
1                                       1  Category A    None
2                                       2  Category B    None
3                                      99  Category C    None

.. note::
	``add_meta()`` is preventing you from adding ill-formed or
	inconsistent variable information, e.g. it is not possible to add ``categories``
	to an ``int``...

	>>> ds.add_meta('new_int', 'int', 'My new int variable', cats)
	ValueError: Numerical data of type int does not accept 'categories'.

	...and you must provide ``categories`` when trying to add categorical data:

	>>> ds.add_meta(name, 'single', label, categories=None)
	ValueError: Must provide 'categories' when requesting data of type single.

Similiar to the usage of the ``categories`` argument, ``items`` is controlling
the creation of an ``array``, i.e. specifying ``items`` is automatically
preparing the ``'masks'`` and ``'columns'`` metadata. The ``qtype`` argument
in this case always refers to the type of the corresponding ``'columns'``.

>>> name, qtype, label = 'new_array', 'single', 'My new array variable'
>>> cats = ['Category A', 'Category B', 'Category C']

Again, there are two alternatives to construct the ``items`` object:

**Providing a list of item labels** (item identifiers will be enumerated
starting from ``1``):

>>> items = ['Item A', 'Item B', 'Item C', 'Item D']


>>> ds.add_meta(name, qtype, label, cats, items=items)
>>> ds.meta('new_array')
single                                  items item texts codes       texts missing
new_array: My new array variable
1                                 new_array_1     Item A     1  Category A    None
2                                 new_array_2     Item B     2  Category B    None
3                                 new_array_3     Item C     3  Category C    None
4                                 new_array_4     Item D

**Providing a list of tuples pairing item identifiers and labels**:

>>> items = [(1, 'Item A'), (2, 'Item B'), (97, 'Item C'), (98, 'Item D')]

>>> ds.add_meta(name, qtype, label, cats, items)
>>> ds.meta('new_array')
single                                   items item texts codes       texts missing
new_array: My new array variable
1                                  new_array_1     Item A     1  Category A    None
2                                  new_array_2     Item B     2  Category B    None
3                                 new_array_97     Item C     3  Category C    None
4                                 new_array_98     Item D

.. note::
	For every created variable, ``add_meta()`` is also adding the relevant ``columns``
	into the ``pd.DataFrame`` case data component of the ``DataSet`` to keep
	it consistent:

	>>> ds['new_array'].head()
		   new_array_1  new_array_2  new_array_97  new_array_98
	0          NaN          NaN           NaN           NaN
	1          NaN          NaN           NaN           NaN
	2          NaN          NaN           NaN           NaN
	3          NaN          NaN           NaN           NaN
	4          NaN          NaN           NaN           NaN

--------
Renaming
--------
It is possible to attach new names to ``DataSet`` variables. Using the ``rename()``
method will replace all former variable ``keys`` and other mentions inside the
metadata document and exchange the ``DataFrame`` column names. For ``array``
variables only the ``'masks'`` name reference is updated by default -- to rename
the corresponding ``items`` a dict mapping item position number to new name can
be provided.

>>> ds.rename(name='q8', new_name='q8_with_a_new_name')

As mentioned, renaming a ``'masks'`` variable will leave the items untouched:

>>>

But we can simply provide their new names as per:

>>>

>>>


-------------------------------
Changing & adding ``text`` info
-------------------------------
All ``text``-related ``DataSet`` methods expose the ``text_key`` argument to
control to which language or context a label is added. For instance we can add
a German variable label to ``'q8'`` with ``set_variable_text()``:

>>> ds.set_variable_text(name='q8', new_text='Das ist ein deutsches Label', text_key='de-DE')

>>> ds.text('q8', 'en-GB')
Which of the following do you regularly skip?

>>> ds.text('q8', 'de-DE')
Das ist ein deutsches Label

To change the ``text`` inside the ``values`` or ``items`` metadata, we can
similarly use ``set_value_text`` and ``set_item_text()``:

>>>

When working with multiple language versions of the metadata, it might be required
to copy one language's ``text`` meta to another one's, for instance if there are
no fitting translations or the correct translation is missing. In such cases you
can use ``force_texts()`` to copy the meta of a source ``text_key`` (specified
in the ```copy_from`` parameter) to a target ``text_key`` (indicated via ``copy_to``).

>>>

>>>

With ``clean_texts()`` you also have the option to replace specific characters,
terms or formatting tags (i.e. ``html``) from all ``text`` metadata of the
``DataSet``:

>>>

-------------------------------
Extending the ``values`` object
-------------------------------
We can add new category defintitons to existing ``values`` meta with the
``extend_values()`` method. As when adding full metadata for categorical
variables, new ``values`` can be generated by either providing only labels or
tuples of codes and labels.

>>>

While the method will never allow adding duplicated numeric values for the
categories, setting ``safe`` to ``False`` will enable you to add duplicated ``text``
meta, i.e. ``values`` could contain both
``{'text': {'en-GB': 'No answer'}, 'value': 98}`` and
``{'text': {'en-GB': 'No answer'}, 'value': 99}``. By default, however,
the method will strictly prohibit any duplicates in the resulting ``values``.

>>>

--------------------------------
Reordering the ``values`` object
--------------------------------


----------------------------
Removing ``DataSet`` objects
----------------------------









