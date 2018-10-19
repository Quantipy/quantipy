.. toctree::
	:maxdepth: 5
	:includehidden:

===================
Latest (01/10/2018)
===================

**New**: "rewrite" of Rules module (affecting sorting):

**sorting "normal" columns**:

* ``sort_on`` always '@'
* ``fix`` any categories
* ``sort_by_weight`` default is unweighted (None), but each weight (included
in data) can be used

If sort_by_weight and the view-weight differ, a warning is shown.

**sorting "expanded net" columns**:

* ``sort_on`` always '@'
* ``fix`` any categories
* sorting ``within`` or ``between`` net groups is available
* ``sort_by_weight``: as default the weight of the first found
expanded-net-view is taken. Only weights of aggregated net-views are possible

**sorting "array summaries"**:

* ``sort_on`` can be any desc ('median', 'stddev', 'sem', 'max', 'min',
'mean', 'upper_q', 'lower_q') or nets ('net_1', 'net_2', .... enumerated
by the net_def)
* ``sort_by_weight``: as default the weight of the first found desc/net-view
is taken. Only weights of aggregated desc/net-views are possible
* ``sort_on`` can also be any category, here each weight can be used to sort_on

""""

**New**: ``DataSet.min_value_count()``

A new wrapper for ``DataSet.hiding()`` is included. All values are hidden,
that have less counts than the included number ``min``.
The used data can be weighted or filtered using the parameters ``weight`` and
``condition``.

Usage as Batch method:
``Batch.min_value_count()`` without the parameters ``weight`` and
``condition`` automatically grabs ``Batch.weights[0]`` and ``Batch.filter``
to calculate low value counts.

""""

**New**: Prevent weak duplicated in data

As Python is case sensitive it is possible to have two or more variables with
the same name, but in lower- and uppercases. Most other software do not support
that, so a warning is shown if a weak dupe is created. Additionally
``Dataset.write_dimensions()`` performs auto-renaming is weak dupes are detected.

""""

**New**: Prevent single-cat delimited sets

``DataSet.add_meta(..., qtype='delimited set', categories=[...], ...)``
automatically switches ``qtype`` to single if only one category is defined.
``DataSet.convert(name, 'single')`` allows conversion from ``delimited set`` to
``single`` if the variable has only one category.
``DataSet.repair()`` and ``DataSt.remove_values()`` convert delimited sets
automatically to singles if only one category is included.

""""

**Update**: merge warnings + merging delimites sets

Warnings in ``hmerge()`` and ``vmerge()`` are updated. If a column exists in
the left and the right dataset, the type is compared. Some type inconsistencies
are allowed, but return a warning, while others end up in a raise.

delimited sets in ``vmerge()``:

If a column is a delimited set in the left dataset, but a single, int or float
in the right dataset, the data of the right column is converted into a delimited
set.

delimited sets in ``hmerge(...merge_existing=None)``:

For the hmerge a new parameter ``merge_existing`` is included, which can be
``None``, a list of variable-names or ``'all'``.

If delimited sets are included in left and right dataset:

* ``merge_existing=None``: Only meta is adjusted. Data is untouched (left data
is taken).
* ``merge_existing='all'``: Meta and data are merged for all delimited sets,
that are included in both datasets.
* ``merge_existing=[variable-names]``: Meta and data are merged for all
delimited sets, that are listed and included in both datasets.

""""

**Update**: encoding in ``DataSet.get_batch(name)``

The method is not that encoding sensitive anymore. It returns the depending
``Batch``, no matter if ``'...'``, ``u'...'`` or ``'...'.decode('utf8')`` is
included as name.

""""

**Update**: warning in weight engine

Missing codes in the sample are only alerted, if the belonging target is not 0.

""""

**Update**: ``DataSet.to_array(..., variables, ...)``

Duplicated vars in ``variables`` are not allowed anymore, these were causing
problems in the ChainManager class.

""""

**Update**: ``Batch.add_open_ends()``

Method raises an error if no vars are included in ``oe`` and ``break_by``. The
empty dataframe was causing issues in the ChainManager class.

""""

**Update**: ``Batch.extend_x()``

The method automatically checks if the included variables are arrays and adds
them to ``Batch.summaries`` if they are included yet.

""""
