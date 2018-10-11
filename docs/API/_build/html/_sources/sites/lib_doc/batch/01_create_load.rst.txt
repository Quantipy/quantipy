.. toctree::
  :maxdepth: 5
  :includehidden:

=========================================
Creating/ Loading a ``qp.Batch`` instance
=========================================

As mentioned, a ``Batch`` instance has a close connection to its belonging
``DataSet`` instance and we can easily create a new ``Batch`` from a ``DataSet``
as per::

	batch1 = dataset.add_batch(name='batch1')
	batch2 = dataset.add_batch(name='batch2', ci=['c'], weights='weight')

It is also possible to load an already existing instance out of the meta
stored in ``dataset._meta['sets']['batches']``::

	batch = dataset.get_batch('batch1')

Both methods, ``.add_batch()`` and ``.get_batch()``, are an easier way to
use the ``__init__()`` method of ``qp.Batch``.

An other way to get a new ``qp.Batch`` instance is to copy an existing one, in
that case all added open ends are removed from the new instance::

	copy_batch = batch.copy('copy_of_batch1')
