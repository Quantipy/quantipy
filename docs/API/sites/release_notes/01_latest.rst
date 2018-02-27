.. toctree::
	:maxdepth: 5
	:includehidden:

===================
Upcoming (March)
===================


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

**Bugfix**: ``ViewManager._request_views()``

Cumulative sums are only requested if they are included in the belonging
``Stack``. Additionally the correct related sig-tests are now taken for
cumulative sums.
