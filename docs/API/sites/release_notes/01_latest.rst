.. toctree::
	:maxdepth: 5
	:includehidden:

================
Upcoming (March)
================

**Update**: ``Stack.add_stats(..., factor_labels=True, ...)``

The parameter ``factor_labels`` is now also able to take the string ``'()'``,
then factors are written in the normal brackets next to the label (instead
of ``[]``).

In the new version factor_labels are also just added if there are none included
before, except new scales are used.

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
