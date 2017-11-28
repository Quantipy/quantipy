.. toctree::
	:maxdepth: 5
	:includehidden:

===================
Upcoming (December)
===================

===================
Latest (28/11/2017)
===================

**New** ``DataSet.from_batch()``

Creates a new ``DataSet`` instance out of ``Batch`` definitions (xks, yks,
filter, weight, language, additions, edits).

""""

**New**: ``Batch.add_total()``

Defines if total column ``@`` should be included in the downbreaks (yks).

""""

**New**: ``Batch.set_unwgt_counts()``

If cellitems are ``cp`` and a weight is provided, it is possible to request
unweighted count views (percentages are still weighted).

""""

**Update**: ``Batch.add_y_on_y(name, y_filter=None, main_filter='extend')``

Multiple ``y_on_y`` aggregations can now be added to a ``Batch`` instance
and each can have an own filter. The y_on_y-filter can ``extend`` or ``replace``
the main_filter of the ``Batch``.

""""

**Update**: ``Stack.add_nets(..., recode)``

The new parameter ``recode`` defines if a new variable is created which
satisfies the net definitions. Different options for ``recode`` are:

	* ``'extend_codes'``: The new variable contains all codes of the original
	  variable and all nets as new categories.
	* ``'drop_codes'``: The new variable contains only all nets as new categories.
	* ``'collect_codes'`` or ``'collect_codes@cat_name'``: The new variable contains
	  all nets as new categories and another new category which sums all cases that
	  are not in any net. The new category text can be defined by adding ``@cat_name``
	  to ``collect_codes``. If none is provided ``Other`` is used as default.

""""

**Update**: ``Stack.add_nets()``

If a variable in the ``Stack`` already has a net_view, it gets overwritten
if a new net is added.

""""

**Update**: ``DataSet.set_missings(..., missing_map)``

The parameter ``missing_map`` can also handle lists now. All included
codes are be flagged as ``'exclude'``.

""""

**Update**: ``request_views(..., sums='mid')`` (``ViewManager``/``query.py``)

Allow different positions for sums in the view-order. They can be placed in
the middle (``'mid'``) between the basics/ nets and the stats or at the
``'bottom'`` after the stats.

""""

**Update/ New**: ``write_dimensions()``

Converting qp data to mdd and ddf files by using ``write_dimensions()`` is
updated now. A bug regarding encoding texts is fixed and additionally all
included ``text_keys`` in the meta are transferred into the mdd. Therefore
two new classes are included: ``DimLabels`` and ``DimLabel``.