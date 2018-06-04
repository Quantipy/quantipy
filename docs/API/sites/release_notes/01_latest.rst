.. toctree::
	:maxdepth: 5
	:includehidden:

==============
Upcoming (May)
==============

**New**: Additional variable (names) "getter"-like and resolver methods

* ``DataSet.created()``
*  ``DataSet.find(str_tags=None, suffixed=False)``
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

===================
Latest (04/04/2018)
===================

**New**: Emptiness handlers in ``DataSet`` and ``Batch`` classes

* ``DataSet.empty(name, condition=None)``
* ``DataSet.empty_items(name, condition=None, by_name=True)``
* ``DataSet.hide_empty_items(condition=None, arrays=None)``
* ``Batch.hide_empty(xks=True, summaries=True)``

``empty()`` is used to test if regular variables are completely empty,
``empty_items()`` checks the same for the items of an array mask definition.
Both can be run on lists of variables. If a single variable is tested, the former
returns simply boolean, the latter will list all empty items. If lists are checked,
``empty()`` returns the sublist of empty variables, ``empty_items()`` is mapping
the list of empty items per array name. The ``condition`` parameter of these
methods takes a ``Quantipy logic`` expression to restrict the test to a subset
of the data, i.e. to check if variables will be empty if the dataset is filtered
a certain way. A very simple example:

>>> dataset.add_meta('test_var', 'int', 'Variable is empty')
>>> dataset.empty('test_var')
True

>>> dataset[dataset.take({'gender': 1}), 'test_var'] = 1
>>> dataset.empty('test_var')
False

>>> dataset.empty('test_var', {'gender': 2})
True


The ``DataSet`` method ``hide_empty_items()`` uses the emptiness tests to
automatically apply a **hiding rule** on all empty items found in the dataset.
To restrict this to specific arrays only, their names can be provided via the
``arrays`` argument. ``Batch.hide_empty()`` takes into account the current
``Batch.filter`` setup and by drops/hides *all* relevant empty variables from the
``xks`` list and summary aggregations by default. Summaries that would end up without valid
items because of this are automatically removed from the ``summaries`` collection
and the user is warned.

""""

**New**: ``qp.set_option('fast_stack_filters', True)``

A new option to enable a more efficient test for already existing filters
inside the ``qp.Stack`` object has been added. Set the ``'fast_stack_filters'``
option to ``True`` to use it, the default is ``False`` to ensure compatibility
in different versions of production DP template workspaces.

""""

**Update**: ``Stack.add_stats(..., factor_labels=True, ...)``

The parameter ``factor_labels`` is now also able to take the string ``'()'``,
then factors are written in the normal brackets next to the label (instead
of ``[]``).

In the new version factor_labels are also just added if there are none included
before, except new scales are used.

""""

**Bugfix**: ``DataSet`` ``np.NaN`` insertion to ``delimited_set`` variables

``np.NaN`` was incorrectly transformed when inserted into ``delimited_set``
before, leading to either ``numpy`` type conflicts or type casting exceptions.
This is now fixed.

