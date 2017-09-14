.. toctree::
	:maxdepth: 5
	:includehidden:

=========================
Upcoming (September 2017)
=========================

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

===================
Latest (31/08/2017)
===================

**New**: ``DataSet.code_from_label(..., exact=True)``

The new parameter ``exact`` is implemented. If ``exact=True`` codes are returned
whose belonging label is equal the included ``text_label``. Otherwise the
method checks if the labels contain the included ``text_label``.

""""

**New**: ``DataSet.order(new_order=None, reposition=None)``

This new method can be used to change the global order of the ``DataSet``
variables. You can either pass a complete ``new_order`` list of variable names to
set the order or provide a list of dictionaries to move (multiple) variables
before a reference variable name. The order is reflected in the case data
``pd.DataFrame.columns`` order and the meta ``'data file'`` ``set`` object's items.

""""

**New**: ``DataSet.dichotomize(name, value_texts=None, keep_variable_text=True, ignore=None, replace=False, text_key=None)``

Use this to convert a ``'delimited set'`` variable into a set of binary coded
``'single'`` variables. Variables will have the values 1/0 and by default use
``'Yes'`` / ``'No'`` as the corresponding labels. Use the ``value_texts``
parameter to apply custom labels.

""""

**New**: ``Batch.extend_x(ext_xks)``

The new method enables an easy extension of ``Batch.xks``. In ``ext_xks``
included ``str`` are added at the end of ``Batch.xks``. Values of included
``dict``\s are positioned in front of the related key.

""""

**Update**: ``Batch.extend_y(ext_yks, ...)``

The parameter ``ext_yks`` now also takes ``dict``\s, which define the position
of the additional ``yks``.

""""

**Update**: ``Batch.add_open_ends(... replacements)``

The new parameter ``replacements`` is implemented. The method loops over the
whole pd.DataFrame and replaces all keys of the included ``dict``
with the belonging value.

""""

**Update**: ``Stack.add_stats(..., other_source)``

Statistic views can now be added to delimited sets if ``other_source`` is used.
In this case ``other_source`` must be a single or numerical variable.

""""

**Update**: ``DataSet.validate(..., spss_limits=False)``

The new parameter ``spss_limits`` is implemented. If ``spss_limits=True``, the
validate output dataframe is extended by 3 columns which show if the SPSS label
limitations are satisfied.

""""

**Bugfix**: ``DataSet.convert()``

A bug that prevented conversions from ``single`` to numeric types has been fixed.

""""

**Bugfix**: ``DataSet.add_meta()``

A bug that prevented the creation of numerical arrays outside of ``to.array()``
has been fixed. It is now possible to create ``array`` metadata without providing
category references.

""""

**Bugfix**: ``Stack.add_stats()``

Checking the statistic views is skipped now if no single typed variables are
included even if a checking cluster is provided.

""""

**Bugfix**: ``Batch.copy()``

Instead of using a deepcopy of the ``Batch`` instance, a new instance is created
and filled with the attributes of the initial one. Then the copied instance can
be used as additional ``Batch``.

""""

**Bugfix**: ``qp.core.builds.powerpoint``

Access to bar-chart series and colour-filling is now working for
different Powerpoint versions. Also a bug is fixed which came up in
``PowerPointpainter()`` for variables which have fixed categories and whose
values are located in ``lib``.
