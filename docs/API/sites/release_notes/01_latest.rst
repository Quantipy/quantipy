.. toctree::
	:maxdepth: 5
	:includehidden:

===================
Upcoming (November)
===================

**New**: ``QuantipyViews(['ebase'])``

It is now possible to create effective base views using the ``'ebase'`` view
method from the collection of ``known_methods``. The effective bases is defined
as: **ebase = (Sum(weights))^2 / Sum(weights^2)**

""""

**Update**: ``DataSet.remove_html()``

The method will not longer simply remove the html ``'<br/>'`` tag but transform
it into a string line break (``\n``). This will leave the line break visible in
the label information.

""""

===================
Latest (xx/10/2017)
===================

**New** ``DataSet.to_delimited_set(name, label, variables,
								   from_dichotomous=True, codes_from_name=True)``

Creates a new delimited set variable out of other variables. If the input-
variables are dichotomous (``from_dichotomous``), the new value-codes can be
taken from the variable-names or from the order of the variables
(``codes_from_name``).

""""

**Update** ``Stack.aggregate(..., bases={})``

A dictionary in form of::

	bases = {
		'cbase': {
			'wgt': True,
			'unwgt': False},
		'cbase_gross': {
			'wgt': True,
			'unwgt': True},
		'ebase': {
			'wgt': False,
			'unwgt': False}
			}

defines what kind of bases will be aggregated. If ``bases`` is provided the
old parameter ``unweighted_base`` and any bases in the parameter ``views``
will be ignored. If bases is not provided and any base is included in ``views``,
a dictionary is automatically created out of ``views`` and ``unweighted_base``.