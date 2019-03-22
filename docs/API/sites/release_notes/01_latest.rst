.. toctree::
	:maxdepth: 5
	:includehidden:

===================
Latest (xx/04/2019)
===================

**Update** Batch (transposed) summaries

As announced a while ago, ``Batch.make_summaries()`` is fully deprecated now
and gives a NotImplementedError. Per default, all arrays in the downbreak
list are added to the ``Batch.x_y_map``. The array ``exclusive`` functionality
(add array, but skip items) is now supported by the new method
``Batch.exclusive_arrays()``.

Additionally ``Batch.transpose_array()`` is deprecated. Instead
``Batch.transpose()`` is available, which does not support ``replace`` anymore,
because the "normal" arrays needs to be included always. If the summaries are
not requested in the deliverables, they can be hidden in the ChainManager.