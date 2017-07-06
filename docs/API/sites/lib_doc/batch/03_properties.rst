.. toctree::
  :maxdepth: 5
  :includehidden:

================================
Set properties of a ``qp.Batch``
================================

The section before explained how the main construction plan (``batch.x_y_map``
is built, that describes which x-keys and y-keys are used to add ``qp.Link``\s 
to a ``qp.Stack``. Now you will get to know how the missing keys for these
links get defined and which link specific views get extracted for the 
``qp.Cluster`` by giving the ``qp.Batch`` instance some properties.

----------------------------
Filter, weights and sigtests
----------------------------

Links can be added to a ``qp.Stack`` datakey by defining its x and y-keys, 
which is already done in ``.x_y_map`` and by setting a filter.
This property can be edited in a ``qp.Batch`` instance with the 
following methods:

>>> batch.add_filter('men only', {'gender': 1})
>>> batch.extend_filter({'q1': {'age': [20, 21, 22, 23, 24, 25]}})

Filters can be added globally on a ``qp.Batch`` instance or for a selection of
x-keys. Out of the global filter ``.sample_size`` is automatically calculated 
for the ``qp.Batch``.

Now all information are collected in the ``qp.Batch`` instance and the Stack 
can be populated with links in form of ``stack[datakey][filter][x-key][y-key]``.

For each link ``qp.View``\s can be added, these views depend on a weight 
definition, which is also defined in the ``qp.Batch``:

>>> batch.set_weights(['weight_a'])

Some special views are significance tests, the levels on which they are 
calculated can be added to the ``qp.Batch`` instance like this:

>>> batch.set_sigtests(levels=[0.05])

-----------------------
Cell items and language
-----------------------

As ``qp.Stack`` is a container, it is able to accommodate any ``qp.View``\s.
The ``qp.Batch`` property ``.cell_items`` is needed to define which views 
should be taken to fill a ``qp.Cluster``:

>>> batch.set_cell_items(['c', 'p'])

The property ``.language`` allows the user to define which texts in the meta
component should be combined with the extracted views by entering a textkey:

>>> batch.set_language('en-GB')
