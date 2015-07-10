.. toctree::
  :maxdepth: 5
  :includehidden:

======================
Extraction and Exports
======================

| :ref:`genindex`
| :ref:`modindex`

""""

Aggregations in a  ``Chain``
----------------------------

:method: ``quantipy.Stack.get_chain(name=None, data_keys=None, filters=None,
         x=None, y=None, views=None, post_process=True, orient_on=None,
         select=None)``
:class: ``quantipy.Chain(name='')``
:method: ``quantipy.Chain.concat()``

Links and their associated Views inside the Stack can easily form a
massive pile of data aggregations. While every single View can be queried from
the Stack individually, extracting structured batches is simplified by the
``Chain`` object. A ``Chain`` is a one-to-many relation of Links that sums up
Views in a specified order and has an orientation axis. *One-to-many* means
that either one x or y axis variable serves as the orientation axis and the
other axis picks up multiple variables. The example below demonstrates the
Chain structure, consider the following Stack:

:: 

	x                                                       birth_year            q5_1              q8         
	y                                                           gender locality gender locality gender locality
	view                                                                                                       
	x|frequency|x:y||weight_a|cbase                                  1        1      1        1      1        1
	x|frequency|x:y|||cbase                                          1        1      1        1      1        1
	x|frequency|x[(1,2,3,4,5)]:y|y|weight_a|net3                   NaN      NaN    NaN      NaN      1        1
	x|frequency|x[(1,2,3,4,5)]:y|y||net3                           NaN      NaN    NaN      NaN      1        1
	x|frequency|x[(1,2,3,4,5)]:y||weight_a|net3                    NaN      NaN    NaN      NaN      1        1
	x|frequency|x[(1,2,3,4,5)]:y|||net3                            NaN      NaN    NaN      NaN      1        1
	x|frequency||y|weight_a|c%                                       1        1      1        1      1        1
	x|frequency||y||c%                                               1        1      1        1      1        1
	x|frequency|||weight_a|counts                                    1        1      1        1      1        1
	x|frequency||||counts                                            1        1      1        1      1        1
	x|mean|x:y||weight_a|mean                                        1        1      1        1    NaN      NaN
	x|mean|x:y|||mean                                                1        1      1        1    NaN      NaN
	x|tests.means.Dim.05|x:y||weight_a|level05                       1        1      1        1    NaN      NaN
	x|tests.means.Dim.05|x:y|||level05                               1        1      1        1    NaN      NaN
	x|tests.means.Dim.10|x:y||weight_a|level10                       1        1      1        1    NaN      NaN
	x|tests.means.Dim.10|x:y|||level10                               1        1      1        1    NaN      NaN
	x|tests.props.Dim.05|x[(1,2,3,4,5)]:y||weight_a|level05        NaN      NaN    NaN      NaN      1        1
	x|tests.props.Dim.05|x[(1,2,3,4,5)]:y|||level05                NaN      NaN    NaN      NaN      1        1
	x|tests.props.Dim.05|||weight_a|level05                          1        1      1        1      1        1
	x|tests.props.Dim.05||||level05                                  1        1      1        1      1        1
	x|tests.props.Dim.10|x[(1,2,3,4,5)]:y||weight_a|level10        NaN      NaN    NaN      NaN      1        1
	x|tests.props.Dim.10|x[(1,2,3,4,5)]:y|||level10                NaN      NaN    NaN      NaN      1        1
	x|tests.props.Dim.10|||weight_a|level10                          1        1      1        1      1        1
	x|tests.props.Dim.10||||level10                                  1        1      1        1      1        1

There are count and percentage representations of cell frequencies, different
bases, nets, means and tests of signficance in both weighted and unweighted
terms. We are only interested in the weighted bases, cell and net percentages
as well as the proportions tests of level 0.10. To get a ``Chain`` object from
the Stack we use the ``get_chain()`` method:

>>> x = ['q5_1', 'q8', 'birth_year']
>>> y = ['gender']

::	

	>>> chain_views = ['x|frequency|x:y||weight_a|cbase',
	...                'x|frequency||y|weight_a|c%',
	...                'x|tests.props.Dim.10|x:y||weight_a|level10',
	...                'x|frequency|x[(1,2,3,4,5)]:y|y|weight_a|net3',
	...                'x|tests.props.Dim.10|x[(1,2,3,4,5)]:y||weight_a|level10']

>>> ychain = stack.get_chain(x=x, y=y, views=chain_views)

A look at the object instance shows us the definition of the Chain:

>>> ychain
<class 'quantipy.core.chain.Chain'>:
orientation-axis: y - gender,
content-axis: ['q5_1', 'q8', 'birth_year'], 
views: 5

Specifying multiple y variables but only one x variable switches around the
relation between the axes and is giving us a Chain with an ``'x'`` orientation:

>>> x = ['q8']
>>> y = ['gender', 'locality']
>>> xchain = stack.get_chain(x=x, y=y, views=chain_views)
>>> xchain
<class 'quantipy.core.chain.Chain'>:
orientation-axis: x - q8,
content-axis: ['gender', 'locality'], 
views: 5

As you may have noticed, not all the Views requested in the Chain are
available for all the Links. A Chain does not care about this: the list of
passed Views is determining the order in which the Views should appear for a
Link, *if they are found for a Link*. In our example this means, that all
Links will show the column base first, then the cell frequencies, then the
proportion test results. For the the q8 variable this is then followed by the
net and the significance test of the net.

The orientation-based structure of a Chain can also be visualized using the
object's ``concat()`` method. For the ``'y'`` orientation example this will
produce the following concatenated ``pandas.DataFrame``:

>>> ychain.concat()
Question             gender          
Values                    1         2
Question   Values                    
q5_1       cbase   3970.518  4284.482
           1       5.933293  4.693466
           2       8.321889  9.034933
           3       35.66731  25.61017
           4       2.018555  1.345397
           5       26.31569   34.4595
           97      3.032247  1.996687
           98      18.71101  22.85984
           1            [2]       NaN
           2            NaN       NaN
           3            [2]       NaN
           4            [2]       NaN
           5            NaN       [1]
           97           [2]       NaN
           98           NaN       [1]
q8         cbase   1156.804  1102.966
           1       38.89881  38.10631
           2       11.22726  9.559564
           3       23.41135  23.67545
           4       40.36939  39.74404
           5       53.54874  53.07162
           96      12.45491  10.70058
           98      2.152411   1.53434
           1            NaN       NaN
           2            NaN       NaN
           3            NaN       NaN
           4            NaN       NaN
           5            NaN       NaN
           96           NaN       NaN
           98           NaN       NaN
           net3    89.44168  91.53253
           net3         NaN       NaN
birth_year cbase   3970.518  4284.482
           1966    3.016231  3.528123
           1967    3.085269   3.62528
           1968    2.643831  2.906797
           1969     3.24588  3.171376
           1970    3.281252  2.738057
           1971     3.29454  3.017625
           1972    3.466121  3.054544
           1973    3.964906   2.78678
           1974    2.826167  3.134981
           1975     3.40827  3.378225
           1976    3.369664  2.984534
           1977    3.453286  3.343299
           1978    2.812025  3.528503
           1979    2.522868  3.172579
           1980    2.769081  3.471606
           1981    3.215933  3.563037
           1982    3.370884   2.72667
           1983    3.949617  3.497545
           1984    3.868383  2.839673
           1985    3.014046  3.307562
           1986    3.443785   3.62427
           1987    3.802384   3.29005
           1988    3.335784  3.072163
           1989    2.944069  3.262956
           1990    2.849785  2.702262
           1991    3.194374   3.01791
           1992    3.218254  3.388331
           1993    3.511948   3.94477
           1994    3.409377  3.116867
           1995    2.822723  3.728384
           1996    2.889264  3.075241
           1966         NaN       NaN
           1967         NaN       NaN
           1968         NaN       NaN
           1969         NaN       NaN
           1970         NaN       NaN
           1971         NaN       NaN
           1972         NaN       NaN
           1973         [2]       NaN
           1974         NaN       NaN
           1975         NaN       NaN
           1976         NaN       NaN
           1977         NaN       NaN
           1978         NaN       NaN
           1979         NaN       NaN
           1980         NaN       NaN
           1981         NaN       NaN
           1982         NaN       NaN
           1983         NaN       NaN
           1984         [2]       NaN
           1985         NaN       NaN
           1986         NaN       NaN
           1987         NaN       NaN
           1988         NaN       NaN
           1989         NaN       NaN
           1990         NaN       NaN
           1991         NaN       NaN
           1992         NaN       NaN
           1993         NaN       NaN
           1994         NaN       NaN
           1995         NaN       [1]
           1996         NaN       NaN

In contrast, ``concat()`` produces the following for the ``'x'``-oriented
Chain:

>>> xchain.concat()
Question              gender                locality                                          
Values                     1            2          1         2         3           4         5
Question Values                                                                               
q8       cbase   1156.803818  1102.965971   916.5352  620.5155  327.7373  162.460592   182.285
         1         38.898815    38.106305    41.6875  40.54213  33.43801   27.776232  33.40518
         2         11.227264     9.559564   10.39797  10.11604  9.023512   13.893351  11.14113
         3         23.411353    23.675454   23.85291  23.32881  24.48457   19.190888  25.47511
         4         40.369388    39.744040   40.57247  38.54357  40.00017   43.718150  43.83815
         5         53.548738    53.071618   51.28179  54.32438  52.34684   52.248305  62.40976
         96        12.454906    10.700579   13.11737  11.22274  9.069246    9.119538  12.49915
         98         2.152411     1.534340    1.38739  2.496877  2.772875    2.098482         0
         1               NaN          NaN  [3, 4, 5]    [3, 4]       NaN         NaN       NaN
         2               NaN          NaN        NaN       NaN       NaN         NaN       NaN
         3               NaN          NaN        NaN       NaN       NaN         NaN       NaN
         4               NaN          NaN        NaN       NaN       NaN         NaN       NaN
         5               NaN          NaN        NaN       NaN       NaN         NaN    [1, 3]
         96              NaN          NaN        NaN       NaN       NaN         NaN       NaN
         98              NaN          NaN        NaN       [5]       [5]         NaN       NaN
         net3      89.441681    91.532533   89.97974  89.99237  90.84682   92.701017  92.48084
         net3            NaN          NaN        NaN       NaN       NaN         NaN       NaN

Multiple Chains in ``Clusters`` 
-------------------------------

:class: ``quantipy.Cluster(name='')``
:method: ``quantipy.Cluster.merge()``

In the above we took care about the orientation of the Chains by following the
*one-to-many* concept through passing either a single x or y axis variable.
But ``get_chain()`` method also provides the ``orient_on`` parameter that can
be set to ``'x'`` or ``'y'`` while passing multiple variables for the Link axes.
The method will then return a *list* of ``Chain`` objects:

>>> x = ['q5_1', 'q8', 'birth_year']
>>> y = ['gender', 'locality']
>>> multi_ychain = stack.get_chain(x=x, y=y, views=chain_views, orient_on='y')
>>> multi_ychain
[<class 'quantipy.core.chain.Chain'>:
orientation-axis: y - gender,
content-axis: ['q8', 'q5_1', 'birth_year'], 
views: 5,
<class 'quantipy.core.chain.Chain'>:
orientation-axis: y - locality,
content-axis: ['q8', 'q5_1', 'birth_year'], 
views: 5]

Since we decided to orient the Chains on the y-axis we end up with two Chains,
one for each of the variables passed to the ``'y'`` parameter. For
``orient_on='x'`` three Chains are created likewise:

>>> multi_xchain = stack.get_chain(x=x, y=y, views=chain_views, orient_on='x')
>>> multi_xchain
[<class 'quantipy.core.chain.Chain'>:
orientation-axis: x - q8,
content-axis: ['gender', 'locality'], 
views: 5,
<class 'quantipy.core.chain.Chain'>:
orientation-axis: x - q5_1,
content-axis: ['gender', 'locality'], 
views: 3,
<class 'quantipy.core.chain.Chain'>:
orientation-axis: x - birth_year,
content-axis: ['gender', 'locality'], 
views: 3]

Chains, and in particular such multiple Chains in lists, are closely related
to Quantipy's ``Cluster`` container that collects and makes them ready for
exporting into a build, i.e. a MS Excel workbook. Having set up our Chains, we
now create a Cluster and give it a name:

>>> cluster = qp.Cluster('my_first_cluster')

Using the ``add_chain()`` method we feed our list of Chains into the Cluster:

>>> cluster.add_chain(multi_ychain)

Similar to ``Chain.concat()``, Clusters provide ``merge()`` to connect all the
Chains given to them, producing a ``pandas.DataFrame``:

>>> cluster.merge()
Question             gender                locality                                                 
Values                    1         2             1         2             3          4             5
Question   Values                                                                                   
q8         cbase   1156.804  1102.966      916.5352  620.5155      327.7373   162.4606       182.285
           1       38.89881  38.10631       41.6875  40.54213      33.43801   27.77623      33.40518
           2       11.22726  9.559564      10.39797  10.11604      9.023512   13.89335      11.14113
           3       23.41135  23.67545      23.85291  23.32881      24.48457   19.19089      25.47511
           4       40.36939  39.74404      40.57247  38.54357      40.00017   43.71815      43.83815
           5       53.54874  53.07162      51.28179  54.32438      52.34684   52.24831      62.40976
           96      12.45491  10.70058      13.11737  11.22274      9.069246   9.119538      12.49915
           98      2.152411   1.53434       1.38739  2.496877      2.772875   2.098482             0
           1            NaN       NaN     [3, 4, 5]    [3, 4]           NaN        NaN           NaN
           2            NaN       NaN           NaN       NaN           NaN        NaN           NaN
           3            NaN       NaN           NaN       NaN           NaN        NaN           NaN
           4            NaN       NaN           NaN       NaN           NaN        NaN           NaN
           5            NaN       NaN           NaN       NaN           NaN        NaN        [1, 3]
           96           NaN       NaN           NaN       NaN           NaN        NaN           NaN
           98           NaN       NaN           NaN       [5]           [5]        NaN           NaN
           net3    89.44168  91.53253      89.97974  89.99237      90.84682   92.70102      92.48084
           net3         NaN       NaN           NaN       NaN           NaN        NaN           NaN
q5_1       cbase   3970.518  4284.482      2908.111  2217.594      1294.096   810.7845      858.5273
           1       5.933293  4.693466      4.674371  4.704457      6.859898     6.2949      5.734177
           2       8.321889  9.034933       10.3253  8.340333      8.077129   7.216118      5.510142
           3       35.66731  25.61017      32.15515  30.61133       29.7208   27.34303      28.37664
           4       2.018555  1.345397      1.389991  2.005244      2.322905   1.113802      1.537438
           5       26.31569   34.4595      30.69806  31.71846      29.56198   30.71674      28.68149
           97      3.032247  1.996687      1.889509  2.172875      2.691972   3.022855      4.405704
           98      18.71101  22.85984      18.86763   20.4473      20.76532   24.29255      25.75441
           1            [2]       NaN           NaN       NaN        [1, 2]        NaN           NaN
           2            NaN       NaN  [2, 3, 4, 5]       [5]           [5]        NaN           NaN
           3            [2]       NaN        [4, 5]       NaN           NaN        NaN           NaN
           4            [2]       NaN           NaN       NaN           [1]        NaN           NaN
           5            NaN       [1]           NaN       NaN           NaN        NaN           NaN
           97           [2]       NaN           NaN       NaN           NaN        NaN     [1, 2, 3]
           98           NaN       [1]           NaN       NaN           NaN     [1, 2]     [1, 2, 3]
birth_year cbase   3970.518  4284.482      2908.111  2217.594      1294.096   810.7845      858.5273
           1966    3.016231  3.528123      3.714028  3.521153      2.426386   2.302449      3.452336
           1967    3.085269   3.62528      3.832932  2.714717      3.339849   3.333808      3.129163
           1968    2.643831  2.906797       2.09384  3.082386      3.421312   3.573883      2.391243
           1969     3.24588  3.171376      3.210962  3.159343      2.796123   3.430637      3.637081
           1970    3.281252  2.738057      2.495499  4.168871      2.172937   2.997884      2.704858
           1971     3.29454  3.017625      3.143637  3.265102      3.690142   2.441045      3.045997
           1972    3.466121  3.054544      3.153594   3.60891      2.578684   3.510555      3.030898
           1973    3.964906   2.78678      3.243344  3.053308      3.671436   3.496161       2.87451
           1974    2.826167  3.134981      2.906848  2.915858      3.322733   3.052669      2.816916
           1975     3.40827  3.378225      3.486123  3.562104      3.146178   4.391299      2.161965
           1976    3.369664  2.984534      2.949959  2.804173      2.602945   5.433613      3.822716
           1977    3.453286  3.343299        3.0441  3.425894      3.863804    4.77948      2.780835
           1978    2.812025  3.528503      3.148629  3.272397      3.349178   3.501084       2.38414
           1979    2.522868  3.172579      2.480044  2.607948      3.963321   3.295585      2.914436
           1980    2.769081  3.471606      3.128281  2.462294      4.777291   2.828234      2.706256
           1981    3.215933  3.563037      3.405635  4.129531      2.974902   3.192628      2.643825
           1982    3.370884   2.72667      3.101977  2.883923      2.607615   2.301522      4.655531
           1983    3.949617  3.497545      3.547004  4.339965      3.414661    2.89022      3.869462
           1984    3.868383  2.839673      3.585673  2.730983      5.007859   1.482894      3.472415
           1985    3.014046  3.307562      3.606378   3.45098      2.682728   2.482051      2.037462
           1986    3.443785   3.62427      3.412858  3.674139      2.741745   3.775043       5.05131
           1987    3.802384   3.29005       3.16438   4.02432      3.198525   4.477651      3.284129
           1988    3.335784  3.072163      3.101305  3.447212       3.54902   2.967887      2.765343
           1989    2.944069  3.262956      3.179805  2.313161      3.267301   3.605417      4.164009
           1990    2.849785  2.702262      3.077648  2.321085      2.861169   1.857519      3.604707
           1991    3.194374   3.01791      3.806815  3.028697      2.468181   1.960119      2.930967
           1992    3.218254  3.388331      3.288481  2.775259      3.908034   3.530577      3.875228
           1993    3.511948   3.94477      4.023867  3.296408      3.344111   3.719981      4.546781
           1994    3.409377  3.116867       3.40373  3.977798      2.476524      2.565      2.813958
           1995    2.822723  3.728384      3.197333   2.71255      3.756391   4.022402      3.743874
           1996    2.889264  3.075241      3.065291  3.269531      2.618915   2.800702       2.68765
           1966         NaN       NaN           [3]       NaN           NaN        NaN           NaN
           1967         NaN       NaN           [2]       NaN           NaN        NaN           NaN
           1968         NaN       NaN           NaN       [1]           [1]        [1]           NaN
           1969         NaN       NaN           NaN       NaN           NaN        NaN           NaN
           1970         NaN       NaN           NaN    [1, 3]           NaN        NaN           NaN
           1971         NaN       NaN           NaN       NaN           NaN        NaN           NaN
           1972         NaN       NaN           NaN       NaN           NaN        NaN           NaN
           1973         [2]       NaN           NaN       NaN           NaN        NaN           NaN
           1974         NaN       NaN           NaN       NaN           NaN        NaN           NaN
           1975         NaN       NaN           NaN       NaN           NaN        [5]           NaN
           1976         NaN       NaN           NaN       NaN           NaN  [1, 2, 3]           NaN
           1977         NaN       NaN           NaN       NaN           NaN     [1, 5]           NaN
           1978         NaN       NaN           NaN       NaN           NaN        NaN           NaN
           1979         NaN       NaN           NaN       NaN        [1, 2]        NaN           NaN
           1980         NaN       NaN           NaN       NaN  [1, 2, 4, 5]        NaN           NaN
           1981         NaN       NaN           NaN       NaN           NaN        NaN           NaN
           1982         NaN       NaN           NaN       NaN           NaN        NaN  [1, 2, 3, 4]
           1983         NaN       NaN           NaN       NaN           NaN        NaN           NaN
           1984         [2]       NaN           [4]       NaN     [1, 2, 4]        NaN           [4]
           1985         NaN       NaN           [5]       NaN           NaN        NaN           NaN
           1986         NaN       NaN           NaN       NaN           NaN        NaN        [1, 3]
           1987         NaN       NaN           NaN       NaN           NaN        NaN           NaN
           1988         NaN       NaN           NaN       NaN           NaN        NaN           NaN
           1989         NaN       NaN           NaN       NaN           NaN        NaN           [2]
           1990         NaN       NaN           NaN       NaN           NaN        NaN           [4]
           1991         NaN       NaN        [3, 4]       NaN           NaN        NaN           NaN
           1992         NaN       NaN           NaN       NaN           NaN        NaN           NaN
           1993         NaN       NaN           NaN       NaN           NaN        NaN           NaN
           1994         NaN       NaN           NaN       [3]           NaN        NaN           NaN
           1995         NaN       [1]           NaN       NaN           NaN        NaN           NaN
           1996         NaN       NaN           NaN       NaN           NaN        NaN           NaN

The DataFrames produced via ``Chain.concat()`` and ``Cluster.merge()`` can
quickly get huge and viewing the raw console output might not be very helpful.
Rememeber that ``pandas`` provides a good range of exporting features, for
example  `to_excel()
<http://pandas.pydata.org/pandas-docs/dev/generated/pandas.DataFrame.to_excel.html>`_.

.. note::

    * The lists of Chains that are fed into a Cluster instance must all share
      the same orientation.   
    
    * Currently, there is no support for Chains/Clusters that are containing the
      ``'default'`` preset View so it will get removed from a Chain specification
      if found in the ``views`` parameter. A future version of Quantipy will
      integrate ``'default'``.



``ExcelPainter``: 1st class tables
----------------------------------

:function: ``quantipy.ExcelPainter(path_excel, meta, cluster, grouped_views=None,
           text_key=None, annotations={}, display_names=None, create_toc=False)``

Quantipy's use of file meta makes it possible to generate properly formatted
and fully labeled exports to MS Excel via its ``ExcelPainter`` that consumes
Clusters. In combination with an associated meta file the ``ExcelPainter``
applies question and category text labels to the View dataframes or attaches
any specified aggregation texts from the View's meta. For each Cluster passed
one Excel workbook sheet is generated. This, in combination with other
structuring options e.g. annotations and the ability to group View aggregates
inside single cells, allows for flexible construction of reporting outputs.

Sticking with the quite simple Cluster from above, we can have a first quick
look on such an Excel build output. We need to specify the full path incl. the
file name of the resulting workbook, the meta file source and a Cluster:  

  >>>  qp.ExcelPainter(path_data+"my_excel_build",
  ...                  meta=my_meta, cluster=Cluster)

And this how the export looks in general:

|

.. image:: C:/Users/alt/AppData/Local/Continuum/Anaconda/Lib/site-packages/quantipy/docs/API/sites/xlsx_output_simple.jpg

We can attach some background information by specifying up to 3 annotations per
Cluster/sheet and adding them as a list to the ``annotations`` parameter as a
dict that maps them to the Cluster's name:

>>> ann1 = 'Quantipy Excel build'
>>> ann2 = 'from: Quantipy docs'
>>> ann3 = 'All data: weighted, tests on level 0.10'

  >>>  qp.ExcelPainter(path_data+"my_excel_build",
  ...                  meta=my_meta, cluster=Cluster,
  ...                  annotations={'my_first_cluster': [ann1, ann2, ann3]})

This is changing the header section of the table output to:

.. image:: C:/Users/alt/AppData/Local/Continuum/Anaconda/Lib/site-packages/quantipy/docs/API/sites/xlsx_output_annotations.jpg

We now use the ``grouped_views`` parameter to group the results of the tests of
significance with the cell frequencies to get a more compact analysis report.
The grouping is supplied via a list of lists that contain the views to be
dispayed together, again mapped to the Cluster name in form of a dict...

  >>> grouped = [
  ...            ['x|frequency||y|weight_a|c%',
  ...             'x|tests.props.Dim.10|||weight_a|level10'],
  ...            ['x|frequency|x[(1,2,3,4,5)]:y|y|weight_a|net3',
  ...             'x|tests.props.Dim.10|x[(1,2,3,4,5)]:y||weight_a|level10']
  ...           ]

  >>>  qp.ExcelPainter(path_data+"my_excel_build",
  ...                  meta=my_meta, cluster=Cluster,
  ...                  annotations={'my_first_cluster': [ann1, ann2, ann3]},
  ...                  grouped_views={'my_first_cluster': grouped})

\... which results in the following:

.. image:: C:/Users/alt/AppData/Local/Continuum/Anaconda/Lib/site-packages/quantipy/docs/API/sites/xlsx_output_groupedviews_simple.jpg

To provide better navigation when the amount of aggregations inside Clusters
is large, ``ExcelPainter`` can also generate a table of contents with hyperlinks
to the x-axis variables as an additional sheet by setting the ``create_toc``
parameter to ``True``. 
  
  >>>  qp.ExcelPainter(path_data+"my_excel_build",
  ...                  meta=my_meta, cluster=Cluster,
  ...                  annotations={'My first Cluster': [ann1, ann2, ann3]},
  ...                  grouped_views={'My first Cluster': grouped},
  ...                  create_toc=True)

The first sheet of the workbook now contains the TOC:

.. image:: C:/Users/alt/AppData/Local/Continuum/Anaconda/Lib/site-packages/quantipy/docs/API/sites/xlsx_toc_simple_table.jpg

.. note::
  
    * It is required that Cluster names do not contain any whitespaces for the
      hyperlink feature to work!

    * Excel exports generated by quantipy.ExcelPainter() must contain a base-
      type view (e.g. the preset ``'cbase'``) as the first View specified in
      the ``views`` parameter of ``stack.get_chain()``. 