.. toctree::
	:maxdepth: 5
	:includehidden:


===================
Latest (09/04/2019)
===================

**New** Nesting in ``Batch.add_crossbreak()``

Nested crossbreaks can be defined for Excel deliverables, the nesting can be
defined by ``"var1 > var2"``. Nesting in more than two levels is available
``"var1 > var2 > var3 > ..."``, but nesting a group of variables is NOT supported
*"var1 > (var2, var3)"*.

""""

**New** Leveling

Running ``Batch.level(array, levels={})`` gives the option to aggregate leveled
arrays. If no ``levels`` are provided, automatically the ``Batch.yks`` are taken.


""""

**New** ``DataSet.used_text_keys()``

This new method loops over text objects in ``DataSet._meta`` and returns
all found ``text_keys``.

""""

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
