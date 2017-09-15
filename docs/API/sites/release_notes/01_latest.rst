.. toctree::
	:maxdepth: 5
	:includehidden:

===================
Latest (15/09/2017)
===================

**New**: ``DataSet.save()`` and ``DataSet.revert()``

These two new methods are useful in interactive sessions like **Ipython** or
**Jupyter** notebooks. ``save()`` will make a temporary (only im memory, not
written to disk) copy of the ``DataSet`` and store its current state. You can
then use ``revert()`` to rollback to that snapshot of the data at a later
stage (e.g. a complex recode operation went wrong, reloading from the physical files takes
too long...).

**New**: ``DataSet.by_type(types=None)``

The ``by_type()`` method is replacing the soon to be deprecated implementation
of ``variables()`` (see below). It provides the same functionality
(``pd.DataFrame`` summary of variable types) as the latter.

**Update**: ``DataSet.variables()`` absorbs ``list_variables()`` and ``variables_from_set()``

In conjunction with the addition of ``by_type()``, ``variables()`` is
replacing the related ``list_variables()`` and ``variables_from_set()`` methods in order to offer a unified solution for querying the ``DataSet``\'s (main) variable collection.

**Update**: ``Batch.as_addition()``

The possibility to add multiple cell item iterations of one ``Batch`` definition
via that method has been reintroduced (it was working by accident in previous
versions with subtle side effects and then removed). Have fun!

**Update**: ``Batch.add_open_ends()``

The method will now raise an ``Exception`` if called on a ``Batch`` that has
been added to a parent one via ``as_addition()`` to warn the user and prevent
errors at the build stage::

   NotImplementedError: Cannot add open end DataFrames to as_addition()-Batches!

