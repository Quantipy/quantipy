.. toctree::
 	:maxdepth: 5
	:includehidden:

==================
DataSet management
==================

--------------------------
Setting the variable order
--------------------------
The global variable order of a ``DataSet`` is dictated by the content of the
``meta['sets']['data file']['items']`` list and reflected in the structure of
the case data component's ``pd.DataFrame.columns``. There are two ways to set
a new order using the ``order(new_order=None, reposition=None)`` method:

**Define a full order**

Using this apporach requires that all ``DataSet`` variable names are passed
via the ``new_order`` parameter. Providing only a subset of the variables will
raise a ``ValueError``:

>>> dataset.order(['q1', 'q8'])
ValueError: 'new_order' must contain all DataSet variables.

Text...

**Change positions relatively**

Often only a few changes to the natural order of the ``DataSet`` are necessary,
e.g. derived variables should be moved alongside their originating ones or specific
sets of variables (demographics, etc.) should be grouped together. We can achieve
this using the ``reposition`` parameter as follows:

Text...

---------------------------------
Cloning, filtering and subsetting
---------------------------------

Sometimes you want to cut the data into sections defined by either case/respondent conditions (e.g. a survey wave) or a collection of variables (e.g.
a specific part of the questionnaire). To not permanently change an existing
``DataSet`` by accident, draw a copy of it first:

>>> copy_ds = dataset.clone()

Then you can use ``filter()`` to restrict cases (rows) or ``subset()`` to keep
only a selected range of variables (columns). Both methods can be used inplace
but will return a new object by default.

>>> keep = {'Wave': [1]}
>>> copy_ds.filter(alias='first wave', condition=keep, inplace=True)
>>> copy_ds._data.shape
(1621, 76)

After the filter has been applied, the ``DataSet`` is only showing cases that contain the value 1 in the ``'Wave'`` variable. The filter alias (a short name
to describe the arbitrarily complex filter ``condition``) is attached to the
instance:

>>> copy_ds.filtered
only first wave

We are now further reducing the ``DataSet`` by dropping all variables except the three ``array`` variables ``'q5'``, ``'q6'``, and ``'q7'`` using ``subset()``.

>>> reduced_ds = copy_ds.subset(variables=['q5', 'q6', 'q7'])

We can see that only the requested variables (``masks`` defintitions and the
constructing ``array`` items) remain in ``reduced_ds``:

>>> reduced_ds.by_type()
size: 1621 single delimited set array int float string date time N/A
0            q5_1                  q5
1            q5_2                  q7
2            q5_3                  q6
3            q5_4
4            q5_5
5            q5_6
6            q6_1
7            q6_2
8            q6_3
9            q7_1
10           q7_2
11           q7_3
12           q7_4
13           q7_5
14           q7_6

-------
Merging
-------

Intro text... As opposed to reducing an existing file...

Vertical (cases/rows) merging
-----------------------------

Text

Horizontal (variables/columns) merging
--------------------------------------

Text

-----------------------------
Savepoints and state rollback
-----------------------------

When working with big ``DataSet``\s and needing to perform a lot of data
preparation (deriving large amounts of new variables, lots of meta editing,
complex cleaning, ...) it can be beneficial to quickly store a snapshot of a
clean and consistent state of the ``DataSet``. This is most useful when working
in interactive sessions like **IPython** or **Jupyter notebooks** and might
prevent you from reloading files from disk or waiting for previous processes
to finish.

Savepoints are stored via ``save()`` and can be restored via ``revert()``.

.. note::
    Savepoints only exists in memory and are not written to disk. Only one
    savepoint can exist, so repeated ``save()`` calls will overwrite any previous
    versions of the ``DataSet``. To permanently save your data, please use one
    of the ``write`` methods, e.g. ``write_quantipy()``.
