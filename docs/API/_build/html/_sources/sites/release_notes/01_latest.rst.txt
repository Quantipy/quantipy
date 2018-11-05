.. toctree::
	:maxdepth: 5
	:includehidden:

===================
Latest (26/10/2018)
===================

**New**: Summarizing and rearranging  ``qp.Chain`` elements via ``ChainManager``

* ``cut(values, ci=None, base=False, tests=False)``

* ``join(title='Summary')``

It is now possible to summarize ``View`` aggregation results from existing ``Chain``
items  by restructuring and editing them via their ``ChainManager`` methods. The 
general idea behind building a summary ``Chain`` is to unify a set of results into
items  by restructuring and editing them via their ``ChainManager`` methods. The
general idea behind building a summary Chain is to unify a set of results into
one cohesive representation to offer an easy way to look at certain key figures
of interest in comparison to each other. To achieve this, the ``ChainManager`` class
has gained the new ``cut()`` and ``join()`` methods. Summaries are built post-
aggregation and therefore rely on what has been defined (via the ``qp.Batch``
class) and computed (via the ``qp.Stack`` methods) at previous stages.

The intended way of working with this new feature can be outlined as

	1. ``reorder()``
	2. ``cut()``
	3. ``join()``
	4. ``insert()``

In more detail:

*A) Grouping the results for the summary*

Both methods will operate on the *entire set* of ``Chains`` collected in a
``ChainManager``, so building a summary ``Chain`` will normally start with
restricting a copy of an existing ``ChainManager`` to the question variables
that you’re interested in. This can be done via ``clone()`` with
``reorder(..., inplace=True)`` or by assigning back the new instance from
``reorder(..., inplace=False)``.

*B) Selecting View results via* ``cut()``

This method lets you target the kind of results (nets, means, NPS scores,
only the frequencies, etc.) from a given ``qp.Chain.dataframe``. Elements must
be targeted by their underlying regular index values, e.g. ``'net_1'``, ``'net_2'``,
``'mean'``, ``1``, ``'calc'``, ... . Use the ``base`` and ``tests`` parameters
to also carry over the matching base rows and/or significance testing results.
The ``ci`` parameter additionally allows targeting only the ``'counts'`` or
``'c%'`` results if cell items are grouped together.

*C) Unifying the individual results with* ``join()``

Merging all new results into one, the ``join()`` method concatenates vertically
and relabels the x-axis to separate all variable results by their matching 
metadata ``text`` that has also been applied while creating the regular set of
and relabels the x-axis to separate all variable results by their matching
metadata ``text`` that has has also been applied while creating the regular set of
``Chain`` items. The new summary can then also be inserted back into its
originating ``ChainManager`` with ``insert()`` if desired.

""""

**Update**: ``Batch.add_variables(varlist)``

A ``qp.Batch`` can now carry a collection of variables that is **explicitly not**
directed towards any table-like builds. Variables from ``varlist`` will solely
be used in non-aggregation based, data transformation and export oriented
applications. To make this distinction more visible in the API, ``add_x()`` and
``add_y()`` have been renamed to ``add_downbreak()`` and ``add_crossbreak()``.
Users are warned and advised to switch to the new method versions via a
``DeprecationWarning``. In a future version of the library ``add_x()`` and
``add_y()`` will be removed.

""""

**Update**: ``Batch.copy()`` -> ``Batch.clone()``

Since ``qp.Batch`` is a subclass of ``qp.DataSet``, the ``copy()`` method is renamed into
``Batch.clone()``.


