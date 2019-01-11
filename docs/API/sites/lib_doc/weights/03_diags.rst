.. toctree::
 	:maxdepth: 5
	:includehidden:

==============================
Diagnostics & advanced options
==============================

We did not yet take a look at the default weight report that offers some
additional information on the weighting outcome results and the even the
algorithm process itself (the report lists the internal weight variable name
that is always just a suffix of the scheme name)::

    Weight variable       weights_my_complex_scheme
    Weight group                             wave 1       wave 2       wave 3       wave 4       wave 5
    Weight filter                         Wave == 1    Wave == 2    Wave == 3    Wave == 4    Wave == 5
    Total: unweighted                   1621.000000  1669.000000  1689.000000  1637.000000  1639.000000
    Total: weighted                     1651.000000  1651.000000  1651.000000  1651.000000  1651.000000
    Weighting efficiency                  74.549628    78.874120    77.595143    53.744060    50.019937
    Iterations required                   13.000000     8.000000    11.000000    12.000000    10.000000
    Mean weight factor                     1.018507     0.989215     0.977501     1.008552     1.007322
    Minimum weight factor                  0.513928     0.562148     0.518526     0.053652     0.050009
    Maximum weight factor                  2.243572     1.970389     1.975681     2.517704     2.642782
    Weight factor ratio                    4.365539     3.505106     3.810189    46.926649    52.846124

The weighting efficiency
------------------------

After all, getting the sample to match to the desired
population proportions **always** comes at a cost. This cost is captured in a
statistical measure called the **weighting efficiency** and is featured in the report
as well. It is a metric for evaluation of the sample vs. targets match, i.e.
the sample balance compared to the weight scheme, i.e. you can also inversely
view it as the amount of distortion that was needed to arrive at the weighted
figures, that is, how much the data is manipulated by the weighting. A **low**
efficiency indicates a **larger** bias introduced by the weights.

Let :math:`w` denote our weight vector containing the factor for each :math:`i`
respondent, then the mathematical definititon of the (total) weighting
efficiency :math:`we` is:

.. math:: we = \frac{\frac{[\sum{w_i}]^2}{\sum i}}{\sum{w_i^2}} * 100


Which is the quotient of the squared sum of weights and the number of cases
divided by the sum of squared weights (expressed as a percentage).

We can manually check the figure for group ``'wave 1'``. We first recreate the
filter that has been used, which we can also derive the number of cases ``n`` from:

>>> f = dataset.take({'Wave': [1]})
>>> n = len(f)
>>> n
1621

The sum of weights squared ``sws`` is then:

>>> sws = (dataset[f, 'weights_new'].sum()) ** 2
>>> sws
2725801.0

And the sum of squared weights ``ssw``:

>>> ssw = (dataset[f, 'weights_new']**2).sum()
>>> ssw
2255.61852968

Which enables us to calculate the weighting efficiency ``we`` as per:

>>> we = (sws / n) / ssw * 100
>>> we
74.5496275503

Generally, weighting efficiency results below the 80% mark indicate a high sample
vs. population mismatch. Dropping below 70% should be a reason to reexamine the
weight scheme specifications or analysis design.

To better understand why the weighting efficiency is good for judging the quality
of the weighting, we can look at its relation to the **effective sample size**
(the effective base). In our example, the effective base of the weight group would
be around 0.75 * 1621 = 1215.75. This means that we are dealing with an effective
sample of only 1216 cases for weighted statistical analysis and inference. In other
words, the weighting reduces the reliability of the sample as if we had sampled
roughly 400 (about 25%) respondents less.


Total base adjustment
---------------------

TODO.

Gotchas
-------

**[A] Group subsets and target application**

In the example we have defined five weight groups, one for each of the waves,  although we only had two differing sets of targets we wanted to match. One could be
tempted to only set two weight groups because of this, using the filters:

>>> f1 = 'Wave in [1, 2, 3]'

and

>>> f1 = 'Wave in [4, 5]'

It is crucial to remember that the algorithm applied on the weight group's
overall data base, i.e. the above definition would achieve the targets waves
inside the two groups (Waves 1/2/3 and Waves 4/5) and **not within** each of the
waves.
