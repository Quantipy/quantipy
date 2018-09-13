.. toctree::
	:maxdepth: 5
	:includehidden:

====================
Upcoming (September)
====================

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


===================
Latest (04/06/2018)
===================

**New**: Additional variable (names) "getter"-like and resolver methods

* ``DataSet.created()``
* ``DataSet.find(str_tags=None, suffixed=False)``
* ``DataSet.names()``
* ``DataSet.resolve_name()``

A bunch of new methods enhancing the options of finding and testing for variable
names have been added. ``created()`` will list all variables that have been added
to a dataset using core functions, i.e. ``add_meta()`` and ``derive()``, resp.
all helper methods that use them internally (as ``band()`` or ``categorize()`` do
for instance).

The ``find()`` method is returning all variable names that contain any of the
provided substrings in ``str_tags``. To only consider names that end with these
strings, set ``suffixed=True``. If no ``str_tags`` are passed, the method will
use a default list of tags including ``['_rc', '_net', ' (categories', ' (NET', '_rec']``.

Sometimes a dataset might contain "semi-duplicated" names, variables that differ
in respect to case sensitivity but have otherwise identical names. Calling
``names()`` will report such cases in a ``pd.DataFrame`` that lists all name
variants under the respective ``str.lower()`` version. If no semi-duplicates
are found, ``names()`` will simply return ``DataSet.variables()``.

Lastly, ``resolve_name()`` can be used to return the "proper", existing representation(s) of a given variable name's spelling.

""""

**New**: ``Batch.remove()``

Not needed batches can be removed from ``meta``, so they are not aggregated
anymore.

""""

**New**: ``Batch.rename(new_name)``

Sometimes standard batches have long/ complex names. They can now be changed
into a custom name. Please take into account, that for most hubs the name of
omnibus batches should look like 'client ~ topic'.

""""

**Update**: Handling verbatims in ``qp.Batch``

Instead of holding the well prepared open-end dataframe in ``batch.verbatims``,
the attribute is now filled by ``batch.add_open_ends()`` with instructions to
create the open-end dataframe. It is easier to to modify/ overwrite existing
verbatims. Therefore also a new parameter is included ``overwrite=True``.

""""

**Update**: ``Batch.copy(..., b_filter=None, as_addition=False)``

It is now possible to define an additional filter for a copied batch and also
to set it as addition to the master batch.

""""

**Update**: Regrouping the variable list using ``DataSet.order(..., regroup=True)``

A new parameter called ``regroup`` will instruct reordering all newly created
variables into their logical position of the dataset's main variable order, i.e.
attempting to place *derived* variables after the *originating* ones.

""""

**Bugfix**: ``add_meta()`` and duplicated categorical ``values`` codes

Providing duplicated numerical codes while attempting to create new metadata
using ``add_meta()`` will now correctly raise a ``ValueError`` to prevent
corrupting the ``DataSet``.

>>> cats = [(1, 'A'), (2, 'B'), (3, 'C'), (3, 'D'), (2, 'AA')]
>>> dataset.add_meta('test_var', 'single', 'test label', cats)
ValueError: Cannot resolve category definition due to code duplicates: [2, 3]
