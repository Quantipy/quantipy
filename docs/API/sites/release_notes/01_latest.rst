.. toctree::
	:maxdepth: 5
	:includehidden:

===================
Upcoming (February)
===================


===================
Latest (12/01/2018)
===================

**New**: ``Audit``

``Audit`` is a new class which takes ``DataSet`` instances, compares and aligns
them.

The class compares/ reports/ aligns the following aspects:

	*	datasets are valid (``DataSet.validate()``)
	*	mismatches (variables are not included in all datasets)
	*	different types (variables are in more than one dataset, but have different types)
	*	labels (variables are in more than one dataset, but have different labels for the same text_key)
	*	value codes (variables are in more than one dataset, but have different value codes)
	*	value texts (variables are in more than one dataset, but have different value texts)
	*	array items (arrays are in more than one dataset, but have different items)
	*	item labels (arrays are in more than one dataset, but their items have different labels)

This is the first draft of the class, so it will need some testing and probably
adjustments.

""""

**New**: ``DataSet.reorder_items(name, new_order)``

The new method reorders the items of the included array. The ints in the
``new_order`` list match up to the number of the items
(``DataSet.item_no('item_name')``), not to the position.

""""

**New**: ``DataSet.valid_tks``, Arabic

Arabic (``ar-AR``) is included as default valid text-key.

""""

**New**: ``DataSet.extend_items(name, ext_items, text_key=None)``

The new method extends the items of an existing array.

""""

**Update**: ``DataSet.set_missings()``

The method is now limited to ``DataSet``, ``Batch`` does not inherit it.

""""

**Update**: ``DataSet``

The whole class is reordered and cleaned up. Some new deprecation warnings
will appear.

""""

**Update**: ``DataSet.add_meta()`` / ``DataSet.derive()``

Both methods will now raise a ``ValueError: Duplicated codes provided. Value codes must be unique!``
if categorical ``values`` definitions try to apply duplicated codes.

""""