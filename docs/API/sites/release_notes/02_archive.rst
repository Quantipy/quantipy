.. toctree::
	:maxdepth: 5
	:includehidden:

=====================
Archive release notes
=====================

---------------
sd (17/05/2017)
---------------

**Update**: ``DataSet.set_variable_text(..., axis_edit=None)``, ``DataSet.set_value_texts(..., axis_edit=False)``

The new ``axis_edit`` argument can be used with one of ``'x'``, ``'y'`` or ``['x', 'y']`` to instruct a text metadata change that will only be visible in build exports.

.. warning::
	In a future version ``set_col_text_edit()`` and ``set_val_text_text()`` will
	be removed! The identical functionality is provided via this ``axis_edit`` parameter.

""""

**Update**: ``DataSet.replace_texts(..., text_key=None)``

The method loops over all meta text objects and replaces unwanted strings.
It is now possible to perform the replacement only for specified ``text_key``\s.
If ``text_key=None`` the method replaces the strings for all ``text_key``\s.

""""

**Update**: ``DataSet.force_texts(copy_to=None, copy_from=None, update_existing=False)``

The method is now only able to force texts for all meta text objects (for
single variables use the methods ``set_variable_text()`` and
``set_value_texts()``).

""""

**Bugfix**: ``DataSet.copy()``

Copied variables get the tag ``created`` and can be listed with
``t.list_variables(dataset, 'created')``.

""""

**Bugfix**: ``DataSet.hmerge()``, ``DataSet.vmerge()``

Array meta information in merged datafiles is now updated correctly.

---------------
sd (04/05/2017)
---------------

**New**: ``DataSet.var_exists()``

Returns True if the input variable/ list of variables are included in the
``DataSet`` instance, otherwise False.

""""

**New**: ``DataSet.remove_html()``, ``DataSet.replace_texts(replace)``

The ``DataSet`` method ``clean_texts()`` has been removed and split into two
methods to make usage more clear: ``remove_html()`` will strip all ``text``
metadata objects from any html and formatting tags. ``replace_texts()`` will
use a ``dict`` mapping of old to new ``str`` terms to change the matching
``text`` throughout the ``DataSet`` metadata.

""""

**New**: ``DataSet.item_no(name)``

This method will return the positional index number of an array item, e.g.:

>>> dataset.item_no('Q4A[{q4a_1}].Q4A_grid')
1

""""

**New**: ``QuantipyViews``: ``counts_cumsum``, ``c%_cumsum``

These two new views contain frequencies with cumulative sums which are computed
over the x-axis.

""""

**Update**: ``DataSet.text(name, shorten=True)``

The new parameter ``shorten`` is now controlling if the variable ``text`` metadata
of array masks will be reported in short format, i.e. without the corresponding
mask label text. This is now also the default behaviour.

""""

**Update**: ``DataSet.to_array()``

Created mask meta information now also contains keys ``parent`` and ``subtype``.
Variable names are compatible with crunch and dimensions meta:

Example in Dimensions modus:

>>> dataset.to_array('Q11', ['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], 'label')

The new grid is named ``'Q11.Q11_grid'`` and the source/column variables are
``'Q11[{Q1}].-Q11_grid'`` - ``'Q11[{Q5}].-Q11_grid'``.

""""

**Bugfix**: ``DataSet.derotate()``

Meta is now Crunch and Dimensions compatible. Also mask meta information are updated.

---------------
sd (24/04/2017)
---------------

**Update**: ``DataSet.hiding(..., hide_values=True)``

The new parameter ``hide_values`` is only necessary if the input variable is a
mask. If ``False``, mask items are hidden, if ``True`` mask values are hidden
for all mask items and for array summary sheets.

""""

**Bugfix**: ``DataSet.set_col_text_edit(name)``

If the input variable is an array item, the new column text is also added to
``meta['mask'][name]['items]``.


""""

**Bugfix**: ``DataSet.drop(name, ignore_items=False)``

If a mask is dropped, but the items are kept, all items are handled now as
individual variables and their meta information is not stored in ``meta['lib']``
anymore.

---------------
sd (06/04/2017)
---------------

Only small adjustments.

---------------
sd (29/03/2017)
---------------

**New**: ``DataSet.codes_in_data(name)``

This method returns a list of codes that exist in the data of a variable. This
information can be used for more complex recodes, for example copying a variable,
but keeping only all categories with more than 50 ratings, e.g.:

>>> valid_code = dataset.codes_in_data('varname')
>>> keep_code = [x for x in valid_code if dataset['varname'].value_counts()[x] > 49]
>>> dataset.copy('varname', 'rc', copy_only=keep_code)

""""

**Update**: ``DataSet.copy(..., copy_not=None)``

The new parameter ``copy_not`` takes a list of codes that should be ignored
for the copied version of the provided variable. The metadata of the copy will
be reduced as well.

""""

**Update**: ``DataSet.code_count()``

This method is now alligned with ``any()`` and ``all()`` in that it can be used
on ``'array'`` variables as well. In such a case, the resulting ``pandas.Series``
is reporting the number of answer codes found across all items per case data
row, i.e.:

>>> code_count = dataset.code_count('Q4A.Q4A_grid', count_only=[3, 4])
>>> check = pd.concat([dataset['Q4A.Q4A_grid'], code_count], axis=1)
>>> check.head(10)
   Q4A[{q4a_1}].Q4A_grid  Q4A[{q4a_2}].Q4A_grid  Q4A[{q4a_3}].Q4A_grid  0
0                    3.0                    3.0                    NaN  2
1                    NaN                    NaN                    NaN  0
2                    3.0                    3.0                    4.0  3
3                    5.0                    4.0                    2.0  1
4                    4.0                    4.0                    4.0  3
5                    4.0                    5.0                    4.0  2
6                    3.0                    3.0                    3.0  3
7                    4.0                    4.0                    4.0  3
8                    6.0                    6.0                    6.0  0
9                    4.0                    5.0                    5.0  1

""""

---------------
sd (20/03/2017)
---------------

**New**: ``qp.DataSet(dimensions_comp=True)``

The ``DataSet`` class can now be explicitly run in a Dimensions compatibility
mode to control the naming conventions of ``array`` variables ("grids"). This
is also the default behaviour for now. This comes with a few changes related to
meta creation and variable access using ``DataSet`` methods. Please see a brief
case study on this topic :doc:`here <../05_case_studies/dimensions_comp>`.

""""

**New**: enriched ``items`` / ``masks`` meta data

``masks`` will now also store the ``subtype`` (``single``, ``delimited set``, etc.)
while ``items`` elements will now contain a reference to the defining ``masks``
entrie(s) in a new ``parent`` object.

""""

**Update**: ``DataSet.weight(..., subset=None)``

Filters the dataset by giving a Quantipy complex logic expression and weights
only the remaining subset.

""""

**Update**: Defining categorical ``values`` meta and ``array`` items

Both ``values`` and ``items`` can now be created in three different ways when
working with the ``DataSet`` methods ``add_meta()``, ``extend_values()`` and
``derive()``: (1) Tuples that map element code to label, (2) only labels or (3)
only element codes. Please see quick guide on that :doc:`here <../04_how_to_snippets/create_categorical_metadata>`

---------------
sd (07/03/2017)
---------------

**Update**: ``DataSet.code_count(..., count_not=None)``

The new parameter ``count_not`` can be used to restrict the set of codes feeding
into the resulting ``pd.Series`` by exclusion (while ``count_only`` restricts
by inclusion).

""""

**Update**: ``DataSet.copy(..., copy_only=None)``

The new parameter ``copy_only`` takes a list of codes that should be included
for the copied version of the provided variable, all others will be ignored
and the metadata of the copy will be reduced as well.

""""

**Bugfix**: ``DataSet.band()``

There was a bug that was causing the method to crash for negative values. It is
now possible to create negative single value bands, while negative ranges
(lower and/or upper bound < 0) will raise a ``ValueError``.

""""

---------------
sd (24/02/2017)
---------------

*	Some minor bugfixes and updates. Please use latest version.

""""

---------------
sd (16/02/2017)
---------------

**New:** ``DataSet.derotate(levels, mapper, other=None, unique_key='identity', dropna=True)``

Create a derotated ("levelled", responses-to-cases) ``DataSet`` instance by
defining level variables, looped variables and other (simple) variables that
should be added.

View more information on the topic :doc:`here <../05_case_studies/derotate>`.

""""

**New:** ``DataSet.to_array(name, variables, label)``

Combine ``column`` variables with identical ``values`` objects to an ``array``
incl. all required ``meta['masks']`` information.

""""

**Update:** ``DataSet.interlock(..., variables)``

It is now possible to add ``dict``\s to ``variables``. In these ``dict``\s a
``derive()``-like mapper can be included which will then create a temporary
variable for the interlocked result. Example:

>>> variables = ['gender',
...              {'agegrp': [(1, '18-34', {'age': frange('18-34')}),
...                          (2, '35-54', {'age': frange('35-54')}),
...                          (3, '55+', {'age': is_ge(55)})]},
...              'region']
>>> dataset.interlock('new_var', 'label', variables)

""""

---------------
sd (04/01/2017)
---------------

**New:** ``DataSet.flatten(name, codes, new_name=None, text_key=None)``

Creates a new ``delimited set`` variable that groups ``grid item`` answers to
categories. The ``items`` become ``values`` of the new variable. If an
``item`` contains one of the ``codes`` it will be counted towards the categorical
case data of the new variable.

""""

**New:** ``DataSet.uncode(target, mapper, default=None, intersect=None, inplace=True)``

Remove codes from the ``target`` variable's data component if a logical
condition is satisfied.

""""

**New:** ``DataSet.text(var, text_key=None)``

Returns the question text label (per ``text_key``) of a variable.

""""

**New:** ``DataSet.unroll(varlist, keep=None, both=None)``

Replaces ``masks`` names inside ``varlist`` with their ``items``. Optionally,
individual ``masks`` can be excluded or kept inside the list.

""""

**New:** ``DataSet.from_stack(stack, datakey=None)``

Create a ``quantipy.DataSet`` from the ``meta``, ``data``, ``data_key`` and
``filter`` definition of a ``quantipy.Stack`` instance.

""""

---------------
sd (8/12/2016)
---------------

**New:**

``DataSet.from_excel(path_xlsx, merge=True, unique_key='identity')``

Returns a new ``DataSet`` instance with ``data`` from ``excel``. The ``meta``
for all variables contains ``type='int'``.

Example: ``new_ds = dataset.from_excel(path, True, 'identity')``

The function is able to modify ``dataset`` inplace by merging ``new_ds`` on
``identity``.

""""

**Update:**

``DataSet.copy(..., slicer=None)``

It is now possible to filter the data that statisfies the logical condition
provided in the ``slicer``.
Example:

>>> dataset.copy('q1', 'rec', True, {'q1': not_any([99])})

""""

---------------
sd (23/11/2016)
---------------

**Update:**

``DataSet.rename(name, new_name=None, array_item=None)``

The function is able to rename ``columns``, ``masks`` or ``mask items``.
``maks items`` are changed by position.

""""

**Update:**

``DataSet.categorize(..., categorized_name=None)``

Provide a custom name string for ``categorized_name`` will change the default
name of the categorized variable from ``OLD_NAME#`` to the passed string.


""""

---------------
sd (16/11/2016)
---------------

**New:**

``DataSet.check_dupe(name='identity')``

Returns a list with duplicated values for the variable provided via ``name``.
Identifies for example duplicated identities.

""""

**New:**

``DataSet.start_meta(text_key=None)``

Creates an empty QP meta data document blueprint to add variable definitions to.

""""

**Update:**

.. code-block:: python

	DataSet.create_set(setname='new_set', based_on='data file', included=None,
	...		  excluded=None, strings='keep', arrays='both', replace=None,
	...		  overwrite=False)

Add a new ``set`` to the ``meta['sets']`` object. Variables from an existing
``set`` (``based_on``) can be ``included`` to ``new_set`` or varibles can be
``excluded`` from ``based_on`` with customized lists of variables.
Control ``string`` variables and ``masks`` with the ``kwargs`` ``strings`` and
``arrays``. ``replace`` single variables in ``new_set`` with a ``dict`` .

""""

**Update:**

``DataSet.from_components(..., text_key=None)``

Will now accept a ``text_key`` in the method call. If querying a ``text_key``
from the meta component fails, the method will no longer crash, but raise a
``warning`` and set the ``text_key`` to ``None``.

""""

**Update:**

.. line-block::

	``DataSet.as_float()``
	``DataSet.as_int()``
	``DataSet.as_single()``
	``DataSet.as_delimited_set()``
	``DataSet.as_string()``
	``DataSet.band_numerical()``
	``DataSet.derive_categorical()``
	``DataSet.set_mask_text()``
	``DataSet.set_column_text()``

These methods will now print a ``UserWarning`` to prepare for the soon to
come removal of them.

""""

**Bugfix:**

``DataSet.__setitem__()``

Trying to set ``np.NaN`` was failing the test against meta data for categorical
variables and was raising a ``ValueError`` then. This is fixed now.

""""

---------------
sd (11/11/2016)
---------------

**New:**

.. line-block::

	``DataSet.columns``
	``DataSet.masks``
	``DataSet.sets``
	``DataSet.singles``
	``DataSet.delimited_sets``
	``DataSet.ints``
	``DataSet.floats``
	``DataSet.dates``
	``DataSet.strings``

New ``DataSet`` instance attributes to quickly return the list of ``columns``,
``masks`` and ``sets`` objects from the meta or query the variables by
``type``. Use this to check for variables, iteration, inspection, ect.

""""

**New:**

``DataSet.categorize(name)``

Create a categorized version of ``int/string/date`` variables. New variables
will be named as per ``OLD_NAME#``

""""

**New:**

``DataSet.convert(name, to)``

Wraps the individual ``as_TYPE()`` conversion methods. ``to`` must be one of
``'int', 'float', 'string', 'single', 'delimited set'``.

""""

**New:**

``DataSet.as_string(name)``

Only for completeness: Use ``DataSet.convert(name, to='string')`` instead.

Converts ``int/float/single/date`` typed variables into a ``string`` and
removes all categorical metadata.

""""

**Update:**

``DataSet.add_meta()``

Can now add ``date`` and ``text`` type meta data.

""""

**Bugfix:**

``DataSet.vmerge()``

If ``masks`` in the right ``dataset``, that also exist in the left ``dataset``,
have new ``items`` or ``values``, they are added to ``meta['masks']``,
``meta['lib']`` and ``meta['sets']``.

""""

---------------
sd (09/11/2016)
---------------

**New:**

``DataSet.as_float(name)``

Converts ``int/single`` typed variables into a ``float`` and removes
all categorical metadata.

""""

**New:**

``DataSet.as_int(name)``

Converts ``single`` typed variables into a ``int`` and removes
all categorical metadata.

""""

**New:**

``DataSet.as_single(name)``

Converts ``int`` typed variables into a ``single`` and adds numeric values as
categorical metadata.

""""

**New:**

``DataSet.create_set(name, variables, blacklist=None)``

Adds a new ``set`` to ``meta['sets']`` object. Create easily ``sets`` from
other ``sets`` while using customised ``blacklist``.

""""

**New:**

``DataSet.drop(name, ignore_items=False)``

Removes all metadata and data referenced to the variable. When passing an
``array mask``, ``ignore_items`` can be ste to ``True`` to keep the ``item
columns`` incl. their metadata.

""""

**New:**

``DataSet.compare(dataset=None, variables=None)``

Compare the metadata definition between the current and another ``dataset``,
optionally restricting to a pair of variables.

""""

**Update:**

``DataSet.__setitem__()``

``[..]``-Indexer now checks scalars against categorical meta.
