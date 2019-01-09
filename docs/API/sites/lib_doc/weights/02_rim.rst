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

>>> scheme.add_groups(name='Wave 1', filter_def=filter_wave1, targets=all_targets)
>>> scheme.add_groups(name='Wave 2', filter_def=filter_wave2, targets=all_targets)
>>> scheme.add_groups(name='Wave 3', filter_def=filter_wave3, targets=all_targets)
>>> scheme.add_groups(name='Wave 4', filter_def=filter_wave4, targets=all_targets_2)
>>> scheme.add_groups(name='Wave 5', filter_def=filter_wave5, targets=all_targets_2)


GOTCHA: wave in [1, 2, 3] etc. vs. Wave == 1, Wave == 2 --> subsets of data!

.. note::

    For historical reasons, the :doc:`logic operators <../dataprocessing/06_logics>` currently **do not** work within the ``Rim`` module. This means that all filter definitions need to be valid
    string expressions suitable for the ``pandas.DataFrame.query()`` `method <https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.query.html>`_.
    We are planning to abandon this limitation as soon as possible to enable
    easier and more complex filters that are consistent with the rest of the library.


Advanced options
----------------

text

==============================
Integration within ``DataSet``
==============================

