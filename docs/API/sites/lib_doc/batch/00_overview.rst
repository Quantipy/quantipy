.. toctree::
  :maxdepth: 5
  :includehidden:

=========
``Batch``
=========

``qp.Batch`` is a subclass of ``qp.DataSet`` and is a container for 
structuring a Link collection's specifications. 

``qp.Batch`` is not only a subclass of ``qp.DataSet``, it also takes a 
DataSet instance as input argument and takes over a few attributes, e.g. 
``_meta``, ``_data``, ``valid_tks`` and ``text_key``.
All other ``Batch`` attributes are used as construction plan for populating a 
``qp.Stack``, these get stored in the belonging ``DataSet`` meta component in
``_meta['sets']['batches'][batchname]``.

In general it does not matter in which order ``Batch`` attributes are set by
methods, the class ensures that all attributes (also connected) keep consistent.

All next sections are working with the following ``qp.DataSet`` instance:

	import quantipy as qp

	dataset = qp.DataSet('Example Data (A)')
	dataset.read_quantipy('Example Data (A).json', 'Example Data (A).csv')

The json and csv files you can find in ``quantipy/tests``.

-----------------------------------------
Creating/ Loading a ``qp.Batch`` instance
-----------------------------------------

As mentioned a ``Batch`` instance has a close connection to its belonging
``DataSet`` instance, because of that it is common to create a new instance 
running::

	batch1 = dataset.add_batch(name='batch1')
	batch2 = dataset.add_batch(name='batch2', ci=['c'], weights='weight')

It is also possible to load an already existing instance out of the information
stored in ``dataset._meta['sets']['batches']``::

	batch = dataset.get_batch('batch1')

Both methods, ``.add_batch()`` and ``.get_batch()``, are an easier way to 
use the ``init()`` method of ``qp.Batch``.

-------------------------------------------
Adding variables to a ``qp.Batch`` instance
-------------------------------------------

The included variables in a ``Batch`` constitude the main structure for the
``qp.Stack`` construction plan. Variables can be added as x-keys or y-keys and
the ``qp.Stack`` gets populated with all cross-tabulations of these keys:

>>> batch.add_x(['q1', 'q2', 'q5.q5_grid'])
>>> batch.add_y(['gender', 'q2b'])
Array summaries setup: Creating ['q5.q5_grid'].

A special case exists if the added variables contain arrays. As default for all
arrays in x-keys array summaries are created (array as x-key and '@'/total as 
y-key), see the output (*Array summaries setup: Creating ['q5.q5_grid'].*). 
If array summaries are requested only for a selection of variables or for none, 
use ``.make_summaries()``:

>>> batch.make_summaries(None)
Array summaries setup: Creating no summaries!

Arrays can also be transposed ('@'/total as x-key and array as y-key). 