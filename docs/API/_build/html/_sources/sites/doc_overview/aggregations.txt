.. toctree::
   :maxdepth: 5
   :includehidden:

``quantify`` aggregation engine
===============================

| :ref:`genindex`
| :ref:`modindex`

""""

Data concept
------------
:class: ``quantipy.Quantity(link, weight=None, xsect_filter=None)``

The ``quantipy.Quantity`` object consists of a matrix-like 1/0 representation
of the case data alongside simple, yet flexible manipulation and analysis
methods. Since ``Quantity`` effectively converts each variable into its dummy
equivalent, Quantipy handles e.g. multiple choice variables natively, i.e. as
if they were "regular" ones. Data aggregation is implemented using the ``numpy``
library's vectorization and broadcasting features to offer support for the
full range of statistical analyses using arbitray case record weights. The
data conversion can be illustrated as follows::
	
	----------------------------
	|   xsect   |  ysect  | wv |
	============================
	| 1 2 3 4 5 | 1  2  3 |    |  
	============================ 
	| 0 1 0 0 0 | 0  1  0 | 1  |   
	---------------------------- 
	| 0 0 1 0 0 | 1  1  1 | 1  |   
	---------------------------- 
	| 0 0 0 0 1 | 0  0  1 | 1  |   
	---------------------------- 
	| ... ... . | ... ... |..  |   
	----------------------------

*xsect* and *ysect* are the x- and y-variables dummy components of the matrix,
*wv* is a corresponding weight vector (or just a vector of 1-entries if no
weight is selected). Generating a ``Quantity`` object by is done by passing a
``Quantipy.Link`` (and an optional weight variable) to its constructor.
Inspecting a new ``Quantity`` instance will show:

>>> link = stack[name_project]['no_filter']['q8']['gender']
>>> q = qp.Quantity(link)
>>> q
<class 'quantipy.core.quantify.q.Quantity'>, 
x: q5_1, xdef: [1, 2, 3, 4, 5, 97, 98] y: gender, ydef: [1, 2], w:@1

The data's matrix representation is accessed via the ``matrix`` property. As
the matrix is of type ``numpy.ndarray``  a quick look at its shape helps to
demonstrate the basic data concept:

>>> q.matrix
[[ 0.  0.  0. ...,  1.  0.  1.]
 [ 0.  0.  0. ...,  0.  1.  1.]
 [ 1.  0.  0. ...,  1.  0.  1.]
 ..., 
 [ 0.  0.  0. ...,  0.  1.  1.]
 [ 1.  0.  0. ...,  0.  1.  1.]
 [ 0.  1.  1. ...,  0.  1.  1.]]

>>> q.matrix.shape
(2367, 10L)

We can see that the matrix shows data for 2367 row entries (the cases) and the
sectional definitions of the columns reflects the information from above:
``q.xdef`` has a length of 7, ``q.ydef`` one of 2 and the weight vector (in
this unweighted example only consisting of 1-entries) sums it up to 10.

Computation methods
-------------------

Cell frequencies
""""""""""""""""

:method: ``quantipy.Quantity.count(show='freq', margin=True, as_df=True)``

Using ``count()`` allows to quickly produce contigency table analyses on data
distributions:

>>> link = stack[name_project]['no_filter']['q8']['gender']
>>> q = qp.Quantity(link, 'weight_a')
>>> q.count()
Question              gender                          
Values                     1            2          All
Question Values                                       
q8       All     1156.803818  1102.965971  2259.769789
         1        449.982974   420.299580   870.282554
         2        129.877415   105.438740   235.316156
         3        270.823427   261.132200   531.955628
         4        466.994617   438.363240   905.357857
         5        619.453850   585.361892  1204.815742
         96       144.078833   118.023747   262.102580
         98        24.899174    16.923248    41.822422

By providing ``margin=False`` in the method call, the 'All'-margins that show
the column and row base sizes are omitted from the output:

>>> q.count(margin=False)
Question             gender            
Values                    1           2
Question Values                        
q8       1       449.982974  420.299580
         2       129.877415  105.438740
         3       270.823427  261.132200
         4       466.994617  438.363240
         5       619.453850  585.361892
         96      144.078833  118.023747
         98       24.899174   16.923248

It is also possible to exclusively show (effective) base sizes using the
``show`` parameter:

>>> q.count(show='cbase', margin=True)
Question              gender                          
Values                     1            2          All
Question Values                                       
q8       cbase   1156.803818  1102.965971  2259.769789

>>> q.count(show='rbase', margin=False)
Question              gender
Values                 rbase
Question Values             
q8       1        870.282554
         2        235.316156
         3        531.955628
         4        905.357857
         5       1204.815742
         96       262.102580
         98        41.822422

>>> q.count(show='ebase', margin=True)
Question             gender                         
Values                    1           2          All
Question Values                                     
q8       ebase   792.151179  795.996416  1587.054368

Numerical statistics
""""""""""""""""""""

:method: ``quantipy.Quantity.describe(show='summary', margin=True, as_df=True)``

Similar to ``count()``, the ``describe()`` method will by default generate a
summary table of distribution statistics. The ``margin`` parameter controls
showing and hiding the analysis for the total (bivariate) number of cases in
the Link definition:

>>> link = stack[name_project]['no_filter']['q5_1']['gender']
>>> q = qp.Quantity(link, 'weight_a')
>>> q.describe()
Question              gender                          
Values                     1            2          All
Question Values                                       
q5_1     All     3970.518490  4284.481510  8255.000000
         mean      23.970385    27.112158    25.601017
         stddev    38.969433    40.745416    39.929528
         min        1.000000     1.000000     1.000000
         25%        3.000000     3.000000     3.000000
         median     4.000000     5.000000     5.000000
         75%        5.000000     5.000000     5.000000
         max       98.000000    98.000000    98.000000

>>> q.describe(margin=False)
Question              gender             
Values                     1            2
Question Values                          
q5_1     All     3970.518490  4284.481510
         mean      23.970385    27.112158
         stddev    38.969433    40.745416
         min        1.000000     1.000000
         25%        3.000000     3.000000
         median     4.000000     5.000000
         75%        5.000000     5.000000
         max       98.000000    98.000000

It is also possible to only show isolated statistics by specifying the ``show``
parameter as one of the following values:

+---------------+---------------------------+------------------------+
| ``show``      | Calculation               | Notes                  |
+===============+===========================+========================+
| ``'mean'``    | Arithmetic mean           |                        |
+---------------+---------------------------+------------------------+
| ``'median'``  | Median, 0.5-percentile    | per SPSS Statistics/   |
|               |                           | Dimensions approach,   |
|               |                           | method 6 in [1]_       |
+---------------+---------------------------+------------------------+
| ``'var'``     | Unbiased (n-1) sample     |                        |
|               | variance                  |                        |
+---------------+---------------------------+------------------------+
| ``'stddev'``  | Standard deviation        | ``np.sqrt(var)``       |
+---------------+---------------------------+------------------------+
| ``'varcoeff'``| Coefficient of variation  |                        |
+---------------+---------------------------+------------------------+
| ``'sem'``     | Standard error of the     |                        |
|               | mean                      |                        |
+---------------+---------------------------+------------------------+
| ``'max'``     | Maximum                   | always unweighted      |
+---------------+---------------------------+------------------------+
| ``'min'``     | Minimum                   | always unweighted      |
+---------------+---------------------------+------------------------+ 
	
>>> q.describe(show='mean', margin=True)
Question            gender                      
Values                   1          2        All
Question Values                                 
q5_1     mean    23.970385  27.112158  25.601017

Code arithmetic
"""""""""""""""

:method: ``quantipy.Quantity.combine(codes=None, op=None, op_only_False, margin=True,
		 as_df=True)``

``combine()`` is a flexible method to summarize code values into groups
on-the-fly with the additional functionality to sum or substract the results
from one another. A simple example is to calculate a net category of values
found in the data:

>>> link = stack[name_project]['no_filter']['q8']['gender']
>>> q = qp.Quantity(link, 'weight_a')
>>> q.combine(group=[1, 2, 3])
Question            gender                       
Values                   1          2         All
Question Values                                  
q8       net     603.30352  548.74762  1152.05114

The ``group`` parameter can also be provided as a list of dict mappings grouping 
names to lists of codes in order to generate multiple nets at once:

>>> grps = [{'A': [1,2,3]}, {'B': [4, 5]}, {'C (residuals)': [96, 98]}]
>>> q.combine(group=grps)
Question                    gender                         
Values                           1           2          All
Question Values                                            
q8       A              603.303520  548.747620  1152.051140
         B              859.773667  814.457890  1674.231557
         C (residuals)  168.978007  134.946995   303.925002

To calculate a result from the new code groups the ``op`` parameter accepts a
dict in the form of::

	{calculation_name: (operator, [group_1, group_2])}

For example, to perform the Net Promoter Score metric frequently used in MR,
``combine()`` could be used as follows (please ignore the not completely
appropriate scaling of the question)::

	>>> link = stack[name_project]['no_filter']['q6_1']['gender']
	>>> q = qp.Quantity(link, 'weight_a')
	>>> grps = [{'Promoters': [9,10]},
	...         {'Passives': [7,8]},
	...         {'Detractors': [1,2,3,4,5,6]}]
	>>> calc = ['NPS': (sub, ['Promoters', 'Detractors'])]
	>>> q.combine(groups=grps, op=calc)

	Question                  gender                          
	Values                         1            2          All
	Question Values                                           
	q6_1     Promoters    441.785993   513.365920   955.151912
	         Passives      59.038108    62.389019   121.427126
	         Detractors  3469.694389  3708.726572  7178.420961
	         NPS        -3027.908396 -3195.360652 -6223.269049

Again, the ``margin`` parameter is supported and it is possible to only show the
calculation result exclusively setting ``op_only`` to ``True``:

>>> q.combine(groups=grps, op=calc, op_only=True, margin=False)
Question              gender             
Values                     1            2
Question Values                          
q6_1     NPS    -3027.908396 -3195.360652

Computation result handling
---------------------------

The ``result`` property
"""""""""""""""""""""""
``Quantity`` aggregations results are passed and stored in the object's
``result`` property, i.e. the ``__repr__`` method will return its content
instead of the general description of the object as long as ``result`` is not
``None``:

>>> link = stack[name_project]['no_filter']['q8']['gender']
>>> q = qp.Quantity(link, 'weight_a').count()
>>> q
Question              gender                          
Values                     1            2          All
Question Values                                       
q8       All     1156.803818  1102.965971  2259.769789
         1        449.982974   420.299580   870.282554
         2        129.877415   105.438740   235.316156
         3        270.823427   261.132200   531.955628
         4        466.994617   438.363240   905.357857
         5        619.453850   585.361892  1204.815742
         96       144.078833   118.023747   262.102580
         98        24.899174    16.923248    41.822422

In the method signatures of ``count()``, ``describe()`` and ``combine()`` the
``as_df`` parameter is set to ``True`` by default, which means that computation
results are returned as multi-indexed ``pandas.DataFrame`` before
they are passed to ``result``: On both the index and the column axis,
the question (=variable) name is presented on the first and the answer code
entries found in the data on the second level of the index.
It is, however, possible to store the underlying raw ``numpy.array`` by setting
``as_df=False``:

>>> q = qp.Quantity(link, 'weight_a').count(as_df=False)
>>> q
[[ 1156.80381804  1102.96597105  2259.7697891 ]
 [  449.98297371   420.29958038   870.28255409]
 [  129.87741546   105.43874045   235.31615591]
 [  270.8234274    261.13220015   531.95562755]
 [  466.99461685   438.36324012   905.35785697]
 [  619.45385041   585.36189159  1204.815742  ]
 [  144.0788332    118.02374662   262.10257982]
 [   24.89917382    16.92324804    41.82242186]]

That makes it possible to use both ``pandas`` and ``numpy`` library methods on
a ``Quantity`` result to further modify the data aggregate:

>>> q.result[1:,:].mean()
385.871708399

``to_df()`` export
""""""""""""""""""

:method: ``quantipy.Quantity.to_df(row_val=None, col_val=None)``

Any ``Quantity.result`` that has been stored as type ``numpy.array`` can be
transformed to a ``pandas.DataFrame`` and passed to the ``result`` property
using the ``to_df()`` method. Use cases for a manual export range from simply
changing the text displayed in the Values-level of the ``pandas.MultiIndex`` to
applying a well-formatted layout to customizing aggregations or View method
programming.

Consider the simple case first in which the object's default
appearance of a column base aggregation result is overwritten: By providing a
new text via the ``row_val`` parameter it changes to:
	
>>> new_text = 'Sample: women only'
>>> q.count(show='cbase', as_df=False).to_df(row_val=new_text)
Question                          gender                          
Values                                 1            2          All
Question Values                                                   
q8       Sample: women only  1156.803818  1102.965971  2259.769789

For more complex operations, however, it is important to be aware of the
shape of the aggregation result and provide the matching number of text values:
Picking up the already known result from the previous paragraph, ``to_df()``
would need to know the values for both the index and the column axis for the
``pandas.DataFrame`` since it cannot infer them:

>>> q.result = q.result[1:,:].mean()
>>> q.to_df(row_val='mean of all counts', col_val='just a demo')
>>> q
Question                         gender
Values                      just a demo
Question Values                        
q8       mean of all counts  385.871708

Percentages
"""""""""""

:method: ``quantipy.Quantity.normalize(on='col')``

``count()`` and ``combine()`` results are always expressed in terms of simple
cell counts. The ``normalize()`` method therefore takes the duty to easily
compute a percentage representation with regard to either the column
(by default) or row base sizes, i.e. changing the result to column or row
percentages. Normalizing works together with both ``numpy.array`` or
``pandas.DataFrame`` type of results:

>>> q = qp.Quantity(link, 'weight_a').count()
>>> q.normalize()
>>> q
Question             gender                        
Values                    1           2         All
Question Values                                    
q8       All     100.000000  100.000000  100.000000
         1        38.898815   38.106305   38.512001
         2        11.227264    9.559564   10.413280
         3        23.411353   23.675454   23.540258
         4        40.369388   39.744040   40.064163
         5        53.548738   53.071618   53.315862
         96       12.454906   10.700579   11.598641
         98        2.152411    1.534340    1.850738

>>> q = qp.Quantity(link, 'weight_a').combine(group=[1, 2, 3], as_df=False)
>>> q.normalize('row').to_df('net - row % of codes: 1, 2, 3')
>>> q
Question                                   gender                
Values                                          1          2  All
Question Values                                                  
q8       net - row % of codes: 1, 2, 3  52.367773  47.632227  100

Code value scaling and exclusion 
--------------------------------

:method: ``quantipy.Quantity.rescale(scaling)``
:method: ``quantipy.Quantity.missingfy(codes=None, keep_codes=False,
		 keep_base=True, inplace=True)``

Adding to the aggregation features, the ``Quantity`` object also provides
utilities to change the factor scaling of the associated case data and to ignore
certain codes from aggregation procedures. A common use case for the latter is
to ignore non-response or residual categories from the sample size when
calculating summary statistics or percentages: Values 96 and 98 from the
now familiar q8 question can be identified as "None of them" and "Don't know"
respectively, ``missingfy()`` helps to get rid of them:

>>> q.missingfy(codes=[96, 98], keep_base=False)
>>> q.count()
>>> q
Question             gender                         
Values                    1           2          All
Question Values                                     
q8       All     987.825811  968.018976  1955.844787
         1       449.982974  420.299580   870.282554
         2       129.877415  105.438740   235.316156
         3       270.823427  261.132200   531.955628
         4       466.994617  438.363240   905.357857
         5       619.453850  585.361892  1204.815742
         96        0.000000    0.000000     0.000000
         98        0.000000    0.000000     0.000000

Since codes 96 and 98 are no longer considered in the aggregation, the
'All'-margins now only take into account codes 1 to 5, which is reflected in the
percentage values:

>>> q.normalize()
>>> q
Question             gender                        
Values                    1           2         All
Question Values                                    
q8       All     100.000000  100.000000  100.000000
         1        45.552867   43.418527   44.496504
         2        13.147805   10.892218   12.031433
         3        27.416112   26.975938   27.198254
         4        47.274996   45.284571   46.289862
         5        62.708814   60.470084   61.600785
         96        0.000000    0.000000    0.000000
         98        0.000000    0.000000    0.000000


The ``rescale()`` method is especially useful in combination with ``describe()``
as it will transform a scale question on-the-fly. In the following example
the 5-point scale question q5_1 is first cleaned from the non-response options
97 and 98 and then rescaled to range from 0-100 instead from 1 to 5. Without any
modifcations, the ``describe()`` result was looking like this (see above)::

	Question              gender                          
	Values                     1            2          All
	Question Values                                       
	q5_1     All     3970.518490  4284.481510  8255.000000
	         mean      23.970385    27.112158    25.601017
	         stddev    38.969433    40.745416    39.929528
	         min        1.000000     1.000000     1.000000
	         25%        3.000000     3.000000     3.000000
	         median     4.000000     5.000000     5.000000
	         75%        5.000000     5.000000     5.000000
	         max       98.000000    98.000000    98.000000

**Missingfying codes 97 and 98** will correct the sample statictics - the base is
reduced and e.g. the mean is now inside the value range of the 1-to-5 scale:
	
>>> q.missingfy(codes=[96, 98], keep_base=False)
>>> q.describe()
>>> q
Question              gender                          
Values                     1            2          All
Question Values                                       
q5_1     All     3107.198494  3219.508253  6326.706747
         mean       3.440364     3.689914     3.567354
         stddev     1.259799     1.321693     1.297576
         min        1.000000     1.000000     1.000000
         25%        3.000000     3.000000     3.000000
         median     3.000000     3.000000     3.000000
         75%        5.000000     5.000000     5.000000
         max        5.000000     5.000000     5.000000

**Rescaling to 0-100** is done by simply passing a dict that maps old to new
codes and passing it to ``rescale()``:
	
>>> new_scaling = {1: 0, 2:25, 3:50, 4:75, 5:100}
>>> q.rescale(new_scaling)
>>> q.describe()
>>> q
Question              gender                          
Values                     1            2          All
Question Values                                       
q5_1     All     3107.198494  3219.508253  6326.706747
         mean      61.009105    67.247852    64.183853
         stddev    31.494975    33.042328    32.439408
         min        0.000000     0.000000     0.000000
         25%       50.000000    50.000000    50.000000
         median    50.000000    50.000000    50.000000
         75%      100.000000   100.000000   100.000000
         max      100.000000   100.000000   100.000000

As expected, the statistics now range between 0-100.

.. rubric:: Footnotes
.. [1] Hyndman, R. J. and Fan, Y. (1996):

       *Sample Quantiles in Statistical Packages*

       The American Statistician 50, no. 4, 361--365.