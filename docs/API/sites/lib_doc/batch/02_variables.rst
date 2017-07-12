.. toctree::
  :maxdepth: 5
  :includehidden:

===========================================
Adding variables to a ``qp.Batch`` instance
===========================================

-----------------
x-keys and y-keys
-----------------

The included variables in a ``Batch`` constitute the main structure for the
``qp.Stack`` construction plan. Variables can be added as x-keys or y-keys, for
arrays all belonging items are automatically added and the ``qp.Stack`` gets
populated with all cross-tabulations of these keys:

>>> batch.add_x(['q1', 'q2', 'q6'])
>>> batch.add_y(['gender', 'q1'])
Array summaries setup: Creating ['q6'].

x-specific y-keys can be produced by manipulating the main y-keys, this edit
can be extending or replacing the existing keys:

>>> batch.extend_y(['locality', 'ethnicity'], on=['q1'])
>>> batch.replace_y(['locality', 'ethnicity'], on=['q2'])

With these settings the construction plan looks like that:

>>> print batch.x_y_map
OrderedDict([('q1', ['@', 'gender', 'q1', 'locality', 'ethnicity']), 
             ('q2', ['locality', 'ethnicity']), 
             ('q6', ['@']), 
             (u'q6_1', ['@', 'gender', 'q1']), 
             (u'q6_2', ['@', 'gender', 'q1']), 
             (u'q6_3', ['@', 'gender', 'q1'])])

------
Arrays
------

A special case exists if the added variables contain arrays. As default for all
arrays in x-keys array summaries are created (array as x-key and ``'@'``-referenced total as
y-key), see the output below (``Array summaries setup: Creating ['q6'].``).
If array summaries are requested only for a selection of variables or for none,
use ``.make_summaries()``:

>>> batch.make_summaries(None)
Array summaries setup: Creating no summaries!

Arrays can also be transposed (``'@'``-referenced total as x-key and array name
as y-key). If they are not in the batch summary list before, they are
automatically added and depending on the ``replace`` parameter only the
transposed or both types of summaries are added to the ``qp.Stack``:

>>> batch.transpose_array('q6', replace=False)
Array summaries setup: Creating ['q6'].

The construction plan now shows that both summary types are included:

>>> print batch.x_y_map
OrderedDict([('q1', ['@', 'gender', 'q1', 'locality', 'ethnicity']),
             ('q2', ['locality', 'ethnicity']),
             ('q6', ['@']),
             ('@', ['q6']),
             (u'q6_1', ['@', 'gender', 'q1']), 
             (u'q6_2', ['@', 'gender', 'q1']), 
             (u'q6_3', ['@', 'gender', 'q1'])])

--------------------
Verbatims/ open ends
--------------------

Another special case are verbatims. They will not be aggregated in a ``qp.Stack``,
but they have to be defined in a ``qp.Batch`` to add them later to a ``qp.Cluster``.

There are two different ways to add verbatims: Either all to one ``qp.Cluster``
key or each gets its own key. But both options can be done with the same method.

For splitting the verbatims, set ``split=True`` and insert as many titles as
included verbatims/ open ends:

>>> batch.add_open_ends(['q8a', 'q9a'], break_by=['record_number', 'age'],
						split=True, title=['oe_q8', 'oe_q9'])

For collecting all verbatims in one Cluster key, set ``split=False`` and add
only one ``title`` or use the default parameters:

>>> batch.add_open_ends(['q8a', 'q9a'], break_by=['record_number', 'age'])

--------------------
Special aggregations
--------------------

It is possible to add some special aggregations to a ``qp.Batch``, that are
not stored in the main construction plan ``.x_y_map``. One option is to give a
name for a Cluster key in which all y-keys are cross-tabulated against each
other:

>>> batch.add_y_on_y('y-keys')

Another possibility is to add a ``qp.Batch`` instance to an other instance.
The added Batch loses all information about verbatims and ``.y_on_y``, that
means only the main construction plan in ``.x_y_map`` gets adopted. Each of
the two batches is aggregated discretely in the ``qp.Stack``, but the added
instance gets included into the ``qp.Cluster`` of the first ``qp.Batch`` in
a key named by its instance name.

>>> batch1 = dataset.get_batch('batch1')
>>> batch2 = dataset.get_batch('batch2')
>>> batch2.add_x('q2b')
Array summaries setup: Creating no summaries!
>>> batch2.add_y('gender')
>>> batch2.as_addition('batch1')
Batch 'batch2' specified as addition to Batch 'batch1'. Any open end summaries and 'y_on_y' agg. have been removed!

The connection between the two ``qp.Batch`` instances you can see in ``.additional``
for the added instance and in ``._meta['sets']['batches']['batchname']['additions']``
for the first instance.