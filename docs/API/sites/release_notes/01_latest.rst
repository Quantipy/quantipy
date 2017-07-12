.. toctree::
	:maxdepth: 5
	:includehidden:

======
Latest
======

**New**: ``qp.Batch()``

This is a new object aimed at defining and structuring aggregation and build
setups. Please see an :doc:`extensive overview here <../lib_doc/batch/00_overview>`.

""""

**New**: ``Stack.aggregate()`` / ``add_nets()`` / ``add_stats()`` / ``add_tests()`` / ...

Connected to the new ``Batch`` class, some new ``Stack`` methods to ease up
view creation have been added. You can :doc:`find the docs here <../lib_doc/engine/00_overview>`.

""""

**New**: ``DataSet.populate()``

Use this to create a ``qp.Stack`` from ``Batch`` definitions. This connects the
``Batch`` and ``Stack`` objects; check out the  :doc:`Batch <../lib_doc/batch/00_overview>`
and :doc:`Analysis & aggregation <../lib_doc/engine/00_overview>` docs.

""""

**New**: ``DataSet.write_dimensions(path_mdd=None, path_ddf=None, text_key=None, mdm_lang='ENG', run=True, clean_up=True)``

It is now possible to directly convert a ``DataSet`` into a Dimensions .ddf/.mdd
file pair (given SPSS Data Collection Base Professional is installed on your
machine). By default, files will be saved to the same location in that the
``DataSet`` resides and keep its ``text_key``.

""""

**New**: ``DataSet.repair()``

This new method can be used to try to fix common ``DataSet`` metadata problems
stemming from outdated versions, incorrect manual editing of the meta dictionary
or other inconsistencies. The method is checking and repairing following issues:

	* ``'name'`` is present for all variable metadata
	* ``'source'`` and ``'subtype'`` references for array variables
	* correct ``'lib'``-based ``'values'`` object for array variables
	* ``text key``-dependent ``'x edits'`` / ``'y edits'`` meta data
	* ``['data file']['items']`` set entries exist in ``'columns'`` / ``'masks'``

""""

**New**: ``DataSet.subset(variables=None, from_set=None, inplace=False)``

As a counterpart to ``filter()``, ``subset()`` can be used to create a new
``DataSet`` that contains only a selection of variables. The new variables
collection can be provided either as a list of names or by naming an already
existing set containing the desired variables.

""""

**New**: ``DataSet.variables_from_set(setname)``

Get the list of variables belonging to the passed set indicated by
``setname``.

""""

**New**: ``DataSet.is_like_numeric(name)``

A new method to test if all of a ``string`` variable's values can be converted
to a numerical (``int`` / ``float``) type. Returns a boolean ``True`` / ``False``.

""""

**Update**: ``DataSet.convert()``

It is now possible to convert inplace from ``string`` to ``int`` / ``float`` if
the respective internal ``is_like_numeric()`` check identifies numeric-like values.

""""

**Update**: ``DataSet.from_components(..., reset=True)``, ``DataSet.read_quantipy(..., reset=True)``

Loaded ``.json`` metadata dictionaries will get cleaned now by default from any
user-defined, non-native objects inside the ``'lib'`` and ``'sets'``
collections. Set ``reset=False`` to keep any extra entires (restoring the old
behaviour).

""""

**Update**: ``DataSet.from_components(data_df, meta_dict=None, ...)``

It is now possible to create a ``DataSet`` instance by providing a ``pd.DataFrame``
alone, without any accompanying meta data. While reading in the case data, the meta
component will be created by inferring the proper ``Quantipy`` variable types
from the ``pandas`` ``dtype`` information.

""""

**Update**: ``Quantity.swap(var, ..., update_axis_def=True)``

It is now possible to ``swap()`` the ``'x'`` variable of an array based ``Quantity``,
as long as the length oh the constructing ``'items'`` collection is identical.
In addition, the new parameter ``update_axis_def`` is now by default enforcing
an update of the axis defintions (``pd.DataFrame`` column names, etc) while
previously the method was keeping the original index and column names. The old
behaviour can be restored by setting the parameter to ``False``.

*Array example*:

>>> link = stack[name_data]['no_filter']['q5']['@']
>>> q = qp.Quantity(link)
>>> q.summarize()
Array                     q5
Questions               q5_1         q5_2         q5_3         q5_4         q5_5         q5_6
Question Values
q5       All     8255.000000  8255.000000  8255.000000  8255.000000  8255.000000  8255.000000
         mean      26.410297    22.260569    25.181466    39.842883    24.399758    28.972017
         stddev    40.415559    38.060583    40.018463    46.012205    40.537497    41.903322
         min        1.000000     1.000000     1.000000     1.000000     1.000000     1.000000
         25%        3.000000     3.000000     3.000000     3.000000     1.000000     3.000000
         median     5.000000     3.000000     3.000000     5.000000     3.000000     5.000000
         75%        5.000000     5.000000     5.000000    98.000000     5.000000    97.000000
         max       98.000000    98.000000    98.000000    98.000000    98.000000    98.000000

*Updated axis definiton*:

>>> q.swap('q7', update_axis_def=True)
>>> q.summarize()
Array                     q7
Questions               q7_1         q7_2         q7_3       q7_4       q7_5       q7_6
Question Values
q7       All     1195.000000  1413.000000  3378.000000  35.000000  43.000000  36.000000
         mean       5.782427     5.423213     5.795145   4.228571   4.558140   5.333333
         stddev     2.277894     2.157226     2.366247   2.073442   2.322789   2.552310
         min        1.000000     1.000000     1.000000   1.000000   1.000000   1.000000
         25%        4.000000     4.000000     4.000000   3.000000   3.000000   3.000000
         median     6.000000     6.000000     6.000000   4.000000   4.000000   6.000000
         75%        8.000000     7.000000     8.000000   6.000000   6.000000   7.750000
         max        9.000000     9.000000     9.000000   8.000000   9.000000   9.000000

*Original axis definiton*:

>>> q = qp.Quantity(link)
>>> q.swap('q7', update_axis_def=False)
>>> q.summarize()
Array                     q5
Questions               q5_1         q5_2         q5_3       q5_4       q5_5       q5_6
Question Values
q5       All     1195.000000  1413.000000  3378.000000  35.000000  43.000000  36.000000
         mean       5.782427     5.423213     5.795145   4.228571   4.558140   5.333333
         stddev     2.277894     2.157226     2.366247   2.073442   2.322789   2.552310
         min        1.000000     1.000000     1.000000   1.000000   1.000000   1.000000
         25%        4.000000     4.000000     4.000000   3.000000   3.000000   3.000000
         median     6.000000     6.000000     6.000000   4.000000   4.000000   6.000000
         75%        8.000000     7.000000     8.000000   6.000000   6.000000   7.750000
         max        9.000000     9.000000     9.000000   8.000000   9.000000   9.000000

""""

**Bugfix**: ``DataSet.band()``

``band(new_name=None)``\'s automatic name generation was incorrectly creating
new variables with the name ``None_banded``. This is now fixed.