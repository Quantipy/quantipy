.. toctree::
	:maxdepth: 5
	:includehidden:

===================
Upcoming (January)
===================

===================
Latest (18/12/2017)
===================

**Bugfix**: ``Batch.add_open_ends()``

``=`` is removed from all responsess in the included variables, as it causes
errors in the Excel-Painter.

""""

**Bugfix**: ``Batch.extend_x()``

Check if included variables exist and unroll included masks.

""""

**Bugfix**: ``Stack.add_nets(..., calc)``

If the operator in calc is ``div``/ ``/``, the calculation is now performed
correctly.

""""


