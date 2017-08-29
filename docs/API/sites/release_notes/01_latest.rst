.. toctree::
	:maxdepth: 5
	:includehidden:

===================
Latest (xx/xx/2017)
===================

**New**: ``DataSet.code_from_label(..., exact=True)``

The new parameter ``exact`` is implemented. If ``exact=True`` codes are returned
whose belonging label is equal the included ``text_label``. Otherwise the
method checks if the labels contain the included ``text_label``.

""""

**New**: ``DataSet.validate(..., spss_limits=False)``

The new parameter ``spss_limits`` is implemented. If ``spss_limits=True`` the
validate output dataframe is extended by 3 columns which show if the SPSS
limitations are satisfied.

""""

**New**: ``Batch.add_open_ends(... replacements)``

The new parameter ``replacements`` is implemented. The method loops over the
whole pd.DataFrame and replaces all keys of the included ``dict``
with the belonging value.

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

**Update**: ``Stack.add_stats(..., other_source)``

Statistic views can now be added to delimited sets if ``other_source`` is used.
In this case ``other_source`` must be a single or numerical variable.

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

