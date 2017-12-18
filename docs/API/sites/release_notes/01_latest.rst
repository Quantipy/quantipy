.. toctree::
	:maxdepth: 5
	:includehidden:

===================
Upcoming (January)
===================

===================
Latest (18/12/2017)
===================

""""

**New**: ``Batch.remove_filter()``

Removes all defined (global + extended) filters from a Batch instance.

""""

**Update**: ``Batch.add_filter()``

It's now possible to extend the global filter of a Batch instance. These options
are possible.

Add first filter::

	>>> batch.filter, batch.filter_names
	'no_filter', ['no_filter']
	>>> batch.add_filter('filter1', logic1)
	>>> batch.filter, batch.filter_names
	{'filter1': logic1}, ['filter1']

Extend filter::

	>>> batch.filter, batch.filter_names
	{'filter1': logic}, ['filter1']
	>>> batch.add_filter('filter2', logic2)
	>>> batch.filter, batch.filter_names
	{'filter1' + 'filter2': intersection([logic1, logic2])}, ['filter1' + 'filter2']

Replace filter::

	>>> batch.filter, batch.filter_names
	{'filter1': logic}, ['filter1']
	>>> batch.add_filter('filter1', logic2)
	>>> batch.filter, batch.filter_names
	{'filter1': logic2}, ['filter1']

""""

**Update**: ``Stack.add_stats(..., recode)``

The new parameter ``recode`` defines if a new numerical variable is created which
satisfies the stat definitions.

""""

**Update**: ``DataSet.populate()``

A progress tracker is added to this method.

""""

**Bugfix**: ``Batch.add_open_ends()``

``=`` is removed from all responsess in the included variables, as it causes
errors in the Excel-Painter.

""""

**Bugfix**: ``Batch.extend_x()`` and ``Batch.extend_y()``

Check if included variables exist and unroll included masks.

""""

**Bugfix**: ``Stack.add_nets(..., calc)``

If the operator in calc is ``div``/ ``/``, the calculation is now performed
correctly.

""""


