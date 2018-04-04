.. toctree::
	:maxdepth: 5
	:includehidden:

================
Upcoming (March)
================

**New**: Emptiness handlers in ``DataSet`` and ``Batch`` classes

* ``DataSet.empty(name, condition=None)``
* ``DataSet.empty_items(name, condition=None, by_name=True)``
* ``DataSet.hide_empty_items(condition=None, arrays=None)``
* ``Batch.hide_empty(xks=True, summaries=True)``

``empty()`` is used to test if regular variables are completely empty,
``empty_items()`` checks the same for the items of an array mask definition.
Both can be run on lists of variables. If a single variable is tested, the former
returns simply boolean, the latter will list all empty items. If lists are checked,
``empty()`` returns the sublist of empty variables, ``empty_items()`` is mapping
the list of empty items per array name. The ``condition`` parameter of these
methods takes a ``Quantipy logic`` expression to restrict the test to a subset
of the data, i.e. to check if variables will be empty if the dataset is filtered
a certain way. A very simple example:

>>> dataset.add_meta('test_var', 'int', 'Variable is empty')
>>> dataset.empty('test_var')
True

>>> dataset[dataset.take({'gender': 1}), 'test_var'] = 1
>>> dataset.empty('test_var')
False

>>> dataset.empty('test_var', {'gender': 2})
True


The ``DataSet`` method ``hide_empty_items()`` uses the emptiness tests to
automatically apply a **hiding rule** on all empty items found in the dataset.
To restrict this to specific arrays only, their names can be provided via the
``arrays`` argument. ``Batch.hide_empty()`` takes into account the current
``Batch.filter`` setup and by drops/hides *all* relevant empty variables from the
``xks`` list and summary aggregations by default. Summaries that would end up without valid
items because of this are automatically removed from the ``summaries`` collection
and the user is warned.

""""

**New**: ``qp.set_option('fast_stack_filters', True)``

A new option to enable a more efficient test for already existing filters
inside the ``qp.Stack`` object has been added. Set the ``'fast_stack_filters'``
option to ``True`` to use it, the default is ``False`` to ensure compatibility
in different versions of production DP template workspaces.

""""

**Update**: ``Stack.add_stats(..., factor_labels=True, ...)``

The parameter ``factor_labels`` is now also able to take the string ``'()'``,
then factors are written in the normal brackets next to the label (instead
of ``[]``).

In the new version factor_labels are also just added if there are none included
before, except new scales are used.

""""

**Bugfix**: ``DataSet`` ``np.NaN`` insertion to ``delimited_set`` variables

``np.NaN`` was incorrectly transformed when inserted into ``delimited_set``
before, leading to either ``numpy`` type conflicts or type casting exceptions.
This is now fixed.

===================
Latest (27/02/2018)
===================

**New**: ``DataSet._dimensions_suffix``

``DataSet`` has a new attribute ``_dimensions_suffix``, which is used as mask
suffix while running ``DataSet.dimensionize()``. The default is ``_grid`` and
it can be modified with ``DataSet.set_dim_suffix()``.

""""

**Update**: ``Stack._get_chain()`` (old chain)

The method is speeded-up. If a filter is already included in the Stack, it is
not calculated from scratch anymore. Additionally the method has a new parameter
``described``, which takes a describing dataframe of the Stack, so it no longer
needs to be calculated in each loop.

""""
**Update**: ``Stack.add_nets()`` (recoded ``Views``)

Nets that are applied on array variables will now also create a new recoded
array that reflects the net definitions if ``recoded`` is used. The
method has been creating only the item versions before.

""""

**Update**: ``Stack.add_stats()``

The method will now create a new metadata property called ``'factor'`` for each
variable it is applied on. You can only have one factor assigned to one
categorical value, so for multiple statistic definitions (exclusions, etc.)
it will get overwritten.

""""

**Update**: ``DataSet.from_batch()`` (``additions`` parameter)

The ``additions`` parameter has been updated to also be able to create recoded
variables from existing "additional" Batches that are attached to a parent one.
Filter variables will get the new meta ``'properties'`` tag ``'recoded_filter'``
and only have one category (``1``, ``'active'``). They are named simply
``'filter_1'``, ``'filter_2'`` and so on. The new possible values of the
parameters are now:

	* ``None``: ``as_addition()``-Batches are not considered.
	* ``'variables'``: Only cross- and downbreak variables are considered.
	* ``'filters'``: Only filters are recoded.
	* ``'full'``: ``'variables'`` + ``'filters'``

""""

**Bugfix**: ``ViewManager._request_views()``

Cumulative sums are only requested if they are included in the belonging
``Stack``. Additionally the correct related sig-tests are now taken for
cumulative sums.
