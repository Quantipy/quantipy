
=========
``Batch``
=========

``qp.Batch`` is a subclass of ``qp.DataSet`` and is a container for
structuring a ``qp.Link`` collection's specifications.

``qp.Batch`` is not only a subclass of ``qp.DataSet``, it also takes a
DataSet instance as input argument, inheriting  a few of its attributes, e.g.
``_meta``, ``_data``, ``valid_tks`` and ``text_key``.
All other ``Batch`` attributes are used as construction plans for populating a
``qp.Stack``, these get stored in the belonging ``DataSet`` meta component in
``_meta['sets']['batches'][batchname]``.

In general, it does not matter in which order ``Batch`` attributes are set by
methods, the class ensures that all attributes (also connected) are kept consistent.

All next sections are working with the following ``qp.DataSet`` instance::

	import quantipy as qp

	dataset = qp.DataSet('Example Data (A)')
	dataset.read_quantipy('Example Data (A).json', 'Example Data (A).csv')

The json and csv files you can find in ``quantipy/tests``.

.. toctree::
  :maxdepth: 5
  :includehidden:

  01_create_load
  02_variables
  03_properties
  04_subclass
