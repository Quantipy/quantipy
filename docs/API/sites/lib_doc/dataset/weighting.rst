.. toctree::
  :maxdepth: 5
  :includehidden:


=======================
Case weight computation
=======================

| :ref:`genindex`
| :ref:`modindex`

""""

Creating weight schemes
-----------------------
:class: ``quantipy.Rim(name, lists=[], max_iterations=1000, convcrit=0.01, cap=0,
        dropna=True, impute_method="mean", weight_column_name=None,
        total=0)``
:method: ``set_targets(targets, group_name=None)``
:method: ``add_group(name=None, filter=None, targets=None)``

To weight a case data file, Quantipy collects the weight information in an
instance of a ``Rim`` scheme. The name corresponds to the algorithm used in
the data weighting process which is discussed in the seminal paper by Deming/
Fredrick (1940) [1]_ (therein coined as the *Raking* approach). *Rim
weighting* is simply another name for this particular method of calculating
case data weights.

**Simple example**

We create a ``Rim`` scheme instance as per:

>>> simple_scheme = qp.Rim(name='simple')

A weight scheme is mainly based on the defintion of a target distribution for
one or multiple variables. We define these targets in the form of a dict
mapping variable names to weight targets lists (please note: the
order of the items in the target lists must match the ascending order of the
data's variable values):

>>> targets_gender = [45.6, 54.4]
>>> targets_locality = [10, 15, 20, 25, 30]

	>>> weight_targets = {'gender': targets_gender,
	...                   'locality': targets_locality}

Using the ``Rim`` method ``set_targets()`` we pass the weights defintion to
the scheme:

>>> simple_scheme.set_targets(targets=weight_targets)

**Filtered weights and total scaling**

It is possible to restrict the weight scheme to a portion of the data by
adding a filter expression. Also, we can control the size of the total number
of weighted cases by passing the desired total sample size in the
``total`` parameter of the instance creation:

>>> W1_n1000_scheme = qp.Rim(name='W1, n=1000', total=1000)

We now use the ``add_group()`` method to apply a filter and the targets:

	>>> W1_n1000_scheme.add_group(name='only_wave_1',
	...                           filter='Wave==1',
	...                           targets=weight_targets)

**Inter-group distributions**

Using the ``group_targets`` parameter the distribution between weight groups
can be controlled as well, meaning that we can influence the proportions
between multiple weight groups.

>>> waves_equal_scheme = qp.Rim(name='equal_waves')

We now use the ``add_group()`` method to apply a filters per group:

	>>> waves_equal_scheme.add_group(name='W1',
	...                              filter='Wave==1',
	...                              targets=weight_targets)

	>>> waves_equal_scheme.add_group(name='W2',
	...                              filter='Wave==2',
	...                              targets=weight_targets)

	>>> waves_equal_scheme.add_group(name='W3',
	...                              filter='Wave==3',
	...                              targets=weight_targets)

	>>> waves_equal_scheme.add_group(name='W4',
	...                              filter='Wave==4',
	...                              targets=weight_targets)

	>>> waves_equal_scheme.add_group(name='W5',
	...                              filter='Wave==5',
	...                              targets=weight_targets)

Since we are dealing with 5 waves in total and want to factor all of them in
equally (since each wave is likely to have a different number of collected
cases), we assign group targets of 20% per filtered group:
	
	>>> groups_w = {'W1': 20,
	...             'W2': 20,
	...             'W3': 20,
	...             'W4': 20,
	...             'W5': 20}

	>>> waves_equal_scheme.group_targets(groups_w)
                  

Running the ``WeightEngine``
----------------------------
:class: ``WeightEngine(data=None, dropna=True, meta=None)``
:method: ``add_scheme(scheme, key)``
:method: ``run(schemes=[])``

With our weighting schemes set up, we can now start computing the factor
values per case data entry. After loading the case data into a
``pandas.DataFrame`` via

>>> path_data = '/Quantipy Example Data/Example Data (A).csv'
>>> my_data = qp.dp.io.load_csv(path_data)

a ``WeightEngine`` object instance is constructed like this:

>>> weighting = qp.WeightEngine(data=my_data).

We can now add the weight schemes using ``add_scheme()`` and call the
``run()`` method with the name of a scheme to start the algorithm.

>>> weighting.add_scheme(scheme=simple_scheme, key='unique_id')
>>> weighting.add_scheme(scheme=W1_n1000_scheme, key='unique_id')
>>> weighting.add_scheme(scheme=waves_equal_scheme, key='unique_id')

The ``key`` parameter in ``add_scheme()`` is used to specify a unique key
variable in the source file in order to correctly merge the computed factors
back to the case data. We can now decide if we want to compute all schemes or
just a selection of them: If the ``scheme`` parameter is omitted in calling
``WeightEngine.run()`` all schemes will be computed. We can, however, pass
a list of scheme names or a single scheme name. For example:

>>> weighting.run()

is equal to:

>>> weighting.run(['simple', 'only_wave_1', 'equal_waves'])

After the weight factors have been computed we can inspect some diagnostic
information, such as weighting efficiency and the minimum and maximum weight 
factors, using the ``report()`` method and passing a scheme name:

>>> rep = weighting.report('equal_waves')
>>> rep['summary']
Weight variable       weights_wave groups, intergroups                                                    
Weight group                                        W1           W2           W3           W4           W5
Total: unweighted                          1577.000000  1631.000000  1652.000000  1607.000000  1611.000000
Total: weighted                            1615.600000  1615.600000  1615.600000  1615.600000  1615.600000
Weighting efficiency                         50.947862    49.502673    52.664037    51.109783    49.947061
Iterations required                          13.000000    14.000000    11.000000    11.000000    11.000000
Minimum weight factor                         0.258910     0.259802     0.262141     0.242685     0.254012
Maximum weight factor                         2.982859     3.222140     2.849204     3.103139     3.129204
Weight factor ratio                          11.520823    12.402273    10.868968    12.786699    12.319134

If explicit weight groups have been defined using ``add_group()``, the report
will split the summary information per group. In comparision the report for
the basic case looks like this:

>>> rep = weighting.report('simple')
>>> rep['summary']
Weight variable               weights_simple
Weight group          __default_group_name__
Total: unweighted                8078.000000
Total: weighted                  8078.000000
Weighting efficiency               50.993734
Iterations required                11.000000
Minimum weight factor               0.258480
Maximum weight factor               2.938652
Weight factor ratio                11.368981

Without a group name specified, the default ``'__default_group_name__'``
will be assigned. We can also see how the internal naming for the
different weights works: it will always start with ``'weights_'`` followed by
the respective scheme name given by the user.

We can check if the calcualted factors for ``'equal_waves_scheme'`` have
produced the desired weighted distribution by quickly constructing a
`pandas.pivot_table() <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.pivot_table.html>`_:

*Checking* ``'gender'`` *distribution*

	>>> wdf = weighting.dataframe('equal_waves')
	... counts = pd.pivot_table(data=wdf, index='gender', columns='Wave',
	...                         values='weights_equal_waves', aggfunc='sum')
	... counts
	Wave           1         2         3         4         5
	gender                                                  
	1       736.7136  736.7136  736.7136  736.7136  736.7136
	2       878.8864  878.8864  878.8864  878.8864  878.8864

>>> pct = counts/counts.sum()*100
>>> pct
Wave       1     2     3     4     5
gender                              
1       45.6  45.6  45.6  45.6  45.6
2       54.4  54.4  54.4  54.4  54.4


*Checking* ``'locality'`` *distribution*

	>>> counts = pd.pivot_table(data=wdf, index='locality', columns='Wave',
	...                         values='weights_equal_waves', aggfunc='sum')
	... counts
	Wave           1       2       3       4       5
	locality                                        
	1         161.56  161.56  161.56  161.56  161.56
	2         242.34  242.34  242.34  242.34  242.34
	3         323.12  323.12  323.12  323.12  323.12
	4         403.90  403.90  403.90  403.90  403.90
	5         484.68  484.68  484.68  484.68  484.68

>>> pct = counts/counts.sum()*100
>>> pct
Wave       1   2   3   4   5
locality                    
1         10  10  10  10  10
2         15  15  15  15  15
3         20  20  20  20  20
4         25  25  25  25  25
5         30  30  30  30  30

*Checking intergroup* ``'Wave'`` *distribution*

	>>> counts = pd.pivot_table(data=wdf, index='Wave'
	...                         values='weights_equal_waves', aggfunc='sum')
	... counts
	Wave
	1    1615.6
	2    1615.6
	3    1615.6
	4    1615.6
	5    1615.6

>>> pct = counts/counts.sum()*100
>>> pct
Wave
1    20
2    20
3    20
4    20
5    20

Happy with the results, we can simply use `pandas.DataFrame.merge()
<http://pandas.pydata.org/pandas-docs/dev/generated/pandas.DataFrame.merge.html>`_
to merge the accepted weight from the ``WeightEngine`` back to our source
data file:

	>>> weighted_data = my_data.merge(wdf[['weights_equal_waves', 'unique_id']],
	...                               on='unqiue_id')

.. note::
		Case data records that have missing data (``np.NaN``) in one of the
		target variables are excluded from the weighting algorithm. After a merge,
		these cases will have ``np.NaN`` in the resulting weight variable
		as well.
.. warning::
	The ``impute_method`` parameter in ``quantipy.Rim()`` is experimental
	and can impute the (rounded) mean or median for missing entries if
	``dropna`` is set to ``False``. This should only be used for
	diagnostic runs.

.. rubric:: Footnotes
.. [1] Deming, W. Edwards/Stephan, Frederick F., 1940:
       
       *On a Least Squares Adjustment of a Sampled Frequency Table When the
       Expected Marginal Totals are Known.*
       
       The Annals of Mathematical Statistics, 11, no. 4, 427--444.