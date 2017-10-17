.. toctree::
	:maxdepth: 5
	:includehidden:

===================
Upcoming (November)
===================

===================
Latest (17/10/2017)
===================

**New**: ``del DataSet['var_name']`` and ``'var_name' in DataSet`` syntax support

It is now possible to test membership of a variable name simply using the ``in``
operator instead of ``DataSet.var_exists('var_name')`` and delete a variable definition
from ``DataSet`` using the ``del`` keyword inplace of the ``drop('var_name')``
method.

""""

**New**: ``DataSet.is_single(name)``, ``.is_delimited_set(name)``, ``.is_int(name)``, ``.is_float(name)``, ``.is_string(name)``, ``.is_date(name)``, ``.is_array(name)``

These new methods make testing a variable's type easy.

""""

**Update**: ``DataSet.singles(array_items=True)`` and all other non-``array`` type iterators

It is now possible to exclude ``array`` items from ``singles()``, ``delimited_sets()``,
``ints()`` and ``floats()`` variable lists by setting the new ``array_items``
parameter to ``False``.

""""

**Update**: ``quantipy.sandbox.sandbox.Chain.paint(..., totalize=True)``

If ``totalize`` is ``True``, ``@``-Total columns of a (x-oriented) ``Chain.dataframe``
will be painted as ``'Total'`` instead of showing the corresponsing ``x``-variables
question text.

""""

**Update**: ``quantipy.core.weights.Rim.Rake``

The weighting algorithm's ``generate_report()`` method can be caught up in a
``MemoryError`` for complex weight schemes run on very large sample sizes. This
is now prevented to ensure the weight factors are computed with priority and
the algorithm is able to terminate correctly. A warning is raised::

   UserWarning: OOM: Could not finish writing report...

""""

**Update**: ``Batch.replace_y()``

Conditional replacements of y-variables of a ``Batch`` will now always also
automatically add the ``@``-Total indicator if not provided.

""""

**Bugfix**: ``DataSet.force_texts(...,  overwrite=True)``

Forced overwriting of existing ``text_key`` meta data was failing for ``array``
``mask`` objects. This is now solved.

""""


