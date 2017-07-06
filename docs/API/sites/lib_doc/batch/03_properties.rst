.. toctree::
  :maxdepth: 5
  :includehidden:

================================
Set properties of a ``qp.Batch``
================================

The section before explained how the main construction plan (``batch.x_y_map``)
is built, that describes which x-keys and y-keys are used to add ``qp.Link``\s
to a ``qp.Stack``. Now you will get to know how the missing information for the
``Link``\s are defined and which specific views get extracted for the
``qp.Cluster`` by adding some property options the ``qp.Batch`` instance.

----------------------------------------
Filter, weights and significance testing
----------------------------------------

``qp.Link``\s can be added to a ``qp.Stack`` data_key-level by defining its x
and y-keys, which is already done in ``.x_y_map``, and setting a filter.
This property can be edited in a ``qp.Batch`` instance with the
following methods:

>>> batch.add_filter('men only', {'gender': 1})
>>> batch.extend_filter({'q1': {'age': [20, 21, 22, 23, 24, 25]}})

Filters can be added globally or for a selection of x-keys only. Out of the
global filter, ``.sample_size`` is automatically calculated for each ``qp.Batch``
defintion.

Now all information are collected in the ``qp.Batch`` instance and the ``Stack``
can be populated with ``Link``\s in form of ``stack[data_key][filter_key][x_key][y_key]``.

For each ``Link`` ``qp.View``\s can be added, these views depend on a weight
definition, which is also defined in the ``qp.Batch``:

>>> batch.set_weights(['weight_a'])

Significance tests are a special ``View``; the sig. levels which they are
calculated on can be added to the ``qp.Batch`` like this:

>>> batch.set_sigtests(levels=[0.05])

-----------------------
Cell items and language
-----------------------

As ``qp.Stack`` is a container for a large amount of aggregations, it will
accommodate various ``qp.View``\s. The ``qp.Batch`` property ``.cell_items`` is
used to define which specfic ``Views`` will be taken to create a ``qp.Cluster``:

>>> batch.set_cell_items(['c', 'p'])

The property ``.language`` allows the user to define which ``text`` labels from
the meta data should be used for the extracted ``Views`` by entering a valid
text key:

>>> batch.set_language('en-GB')
