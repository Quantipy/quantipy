.. toctree::
 	:maxdepth: 5
	:includehidden:

===================
Weight scheme setup
===================

-----------------------
Using the ``Rim`` class
-----------------------

The ``Rim`` object's purpose is to define the required setup of the weighting process, i.e. the *weight scheme* that should be used to compute the actual factor results per case in the dataset. While its main purpose is to provide a simple interface to structure weight schemes of all complexities, it also offers advanced options that control the underlying weighting algorithm itself and thus might impact the results.

To start working with a ``Rim`` object, we only need to think of a name for our scheme:

>>> scheme = qp.Rim('my_first_scheme')


--------------------
Target distributions
--------------------

A major and (probably the most important) step in specifying a weight scheme
is mapping the desired target population proportions to the categories of the related variables inside the data. This is done via a ``dict`` mapping.

For example, to equally weight female and male respondents in our sample, we
simply define:

>>> gender_targets = {}
>>> gender_targets['gender'] = {1: 50.0, 2: 50.0}
>>> gender_targets
{'gender': {1: 50.0, 2: 50.0}}

Since we are normally dealing with multiple variables at once, we collect
them in a ``list``, adding other variables naturally in the same way:

>>> dataset.band('age', [(19, 25), (26-35), (36, 49)])
>>> age_targets = {'age_banded': {1: 45.0, 2: 29.78, 3: 25.22}}
>>> all_targets = [gender_targets, age_targets]

The ``set_targets()`` method can now use the ``all_targets`` list to apply the target distributions to the ``Rim`` weight scheme setup (we are also providing an optional name for our group of variables) .

>>> scheme.set_targets(targets=all_targets, group_name='basic weights')

The ``Rim`` instance also allows inspecting these targets from itself now (you can see ``group_name`` parameter reflected here, it would fall back to ``'_default_name_'`` if none was provided):

>>> scheme.groups['basic weights']['targets']
[{'gender': {1: 50.0, 2: 50.0}}, {'age_banded': {1: 45.0, 2: 29.78, 3: 25.22}}]


-------------------------
Weight groups and filters
-------------------------

For more elaborate weight schemes, we are instead using the ``add_group()`` method
which is effectively a generalized version of ``set_targets()`` that supports
addressing subsets of the data by filtering. For example, differing target distributions (or even the scheme defining variables of interest) might be
required across several market segments or between survey periods.

We can illustrate this using the variable ``'Wave'`` from the dataset:

>>> dataset.crosstab('Wave', text=True, pct=True)
Question          Wave. Wave
Values                     @
Question   Values
Wave. Wave All         100.0
           Wave 1       19.6
           Wave 2       20.2
           Wave 3       20.5
           Wave 4       19.8
           Wave 5       19.9

Let's assume we want to use the original targets for the first three waves but
the remaining two waves need to reflect some changes in both gender and the age
distributions. We first define a new set of targets that should apply only to
the waves 4 and 5::

    gender_targets_2 = {'gender': {1: 30.0, 2: 70.0}}
    age_targets_2 = {'age_banded': {1: 35.4, 2: 60.91, 3: 3.69}}
    all_targets_2 = [gender_targets_2, age_targets_2]

We then set the filter expressions for the respective subsets of the data, as per:

>>> filter_wave1 = 'Wave == 1'
>>> filter_wave2 = 'Wave == 2'
>>> filter_wave3 = 'Wave == 3'
>>> filter_wave4 = 'Wave == 4'
>>> filter_wave5 = 'Wave == 5'

And add our weight specifications accordingly:

>>> scheme = qp.Rim('my_complex_scheme')
>>> scheme.add_groups(name='wave 1', filter_def=filter_wave1, targets=all_targets)
>>> scheme.add_groups(name='wave 2', filter_def=filter_wave2, targets=all_targets)
>>> scheme.add_groups(name='wave 3', filter_def=filter_wave3, targets=all_targets)
>>> scheme.add_groups(name='wave 4', filter_def=filter_wave4, targets=all_targets_2)
>>> scheme.add_groups(name='wave 5', filter_def=filter_wave5, targets=all_targets_2)


.. note::

    For historical reasons, the :doc:`logic operators <../dataprocessing/06_logics>` currently **do not** work within the ``Rim`` module. This means that all filter definitions need to be valid
    string expressions suitable for the ``pandas.DataFrame.query()`` `method <https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.query.html>`_.
    We are planning to abandon this limitation as soon as possible to enable
    easier and more complex filters that are consistent with the rest of the library.

---------------------
Setting group targets
---------------------

At this stage it might also be needed to balance out the survey waves themselves
in a certain way, e.g. make each wave count exactly the same (as you can see above
each wave accounts for roughly 20% of the full sample but not quite exactly).

With ``Rim.group_targets()`` we can apply an **outer** weighting to the **between**
group distribution while keeping the already set **inner** target proportions **within** each of them. Again we are using a ``dict``, this time mapping the
group names from above to the desired outcome percentages:

>>> group_targets = {'wave 1': 20.0,
...                  'wave 2': 20.0,
...                  'wave 3': 20.0,
...                  'wave 4': 20.0,
...                  'wave 5': 20.0}

>>> scheme.group_targets(group_targets)

To sum it up: Our weight scheme consists of five groups based on ``'Wave'`` that
resp. need to match two different sets of target distributions on the ``'gender'``
and ``'age_banded'`` variables with each group coming out as 20% of the full sample.

==============================
Integration within ``DataSet``
==============================

The computational core of the weighting algorithm is the
``quantipy.core.weights.rim.Rake`` class which can be accessed by working
with ``qp.WeightEngine()``, but it is much easier to directly use the ``DataSet.weight()``
method. Its full signature looks as follows::

    DataSet.weight(weight_scheme,
                   weight_name='weight',
                   unique_key='identity',
                   subset=None,
                   report=True,
                   path_report=None,
                   inplace=True,
                   verbose=True)

-----------------------------------
Weighting and weighted aggregations
-----------------------------------

As can been seen, we can simply provide our weight scheme ``Rim`` instance to
the method. Since the dataset already contains a variable called ``'weight'``
(and we do not want to overwrite that one) we set ``weight_name`` to be
``'weight_new'``. We also need to set ``unqiue_key='unique_id'`` as that is our
identifying key variable (that is needed to map the weight factors back into our
dataset):

>>> dataset.weight(scheme, weight_name='weight_new', unqiue_key='unique_id')

Before we take a look at the report that is printed (because of ``report=True``),
we want to manually check our results. For that, we can simply analyze some cross-
tabulations, weighted by our new weights! For a start, we check if we arrived at
the desired proportions for ``'gender'`` and ``'age_banded'`` per ``'Wave'``:

>>> dataset.crosstab(x='gender', y='Wave', w='weights_new', pct=True)
Question                            Wave. Wave
Values                                     All Wave 1 Wave 2 Wave 3 Wave 4 Wave 5
Question                     Values
gender. What is your gender? All         100.0  100.0  100.0  100.0  100.0  100.0
                             Male         42.0   50.0   50.0   50.0   30.0   30.0
                             Female       58.0   50.0   50.0   50.0   70.0   70.0

>>> dataset.crosstab(x='age_banded', y='Wave', w='weights_new', pct=True,
...                  decimals=2)
Question               Wave. Wave
Values                        All  Wave 1  Wave 2  Wave 3  Wave 4  Wave 5
Question        Values
age_banded. Age All        100.00  100.00  100.00  100.00  100.00  100.00
                19-25       41.16   45.00   45.00   45.00   35.40   35.40
                26-35       42.23   29.78   29.78   29.78   60.91   60.91
                36-49       16.61   25.22   25.22   25.22    3.69    3.69

Both results accurately reflect the desired proportions from our scheme. And we
can also verify the weighted distribution of ``'Wave'``, now completely
balanced:

>>> dataset.crosstab(x='Wave', w='weights_new', pct=True)
Question          Wave. Wave
Values                     @
Question   Values
Wave. Wave All         100.0
           Wave 1       20.0
           Wave 2       20.0
           Wave 3       20.0
           Wave 4       20.0
           Wave 5       20.0

-----------------------------
The isolated weight dataframe
-----------------------------

By default, the weighting operates ``inplace``, i.e. the weight vector will
be placed into the ``DataSet`` instance as a regular ``columns`` element:

>>> dataset.meta('weights_new')
                                     float
weights_new: my_first_scheme weights   N/A

>>> dataset['weights_new'].head()
   unique_id  weights_new
0     402891     0.885593
1   27541022     1.941677
2     335506     0.984491
3   22885610     1.282057
4     229122     0.593834

It is also possible to return a new ``pd.DataFrame`` that contains all relevant ``Rim``
scheme variables incl. the factor vector for external use cases or further
analysis:

>>> wdf = dataset.weight(scheme, weight_name='weights_new', unqiue_key='unique_id',
                         inplace=False)
>>> wdf.head()
   unique_id  gender  age_banded  weights_new  Wave
0     402891       1         1.0     0.885593     4
1   27541022       2         1.0     1.941677     1
2     335506       1         2.0     0.984491     3
3   22885610       1         2.0     1.282057     5
4     229122       1         3.0     0.593834     1
