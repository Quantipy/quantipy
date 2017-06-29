.. toctree::
  :maxdepth: 5
  :includehidden:

========================
View method manipulation
========================

| :ref:`genindex`
| :ref:`modindex`

""""

Control via ``kwargs``
----------------------
Quantipy's main focus is to provide a flexible, yet easy-to-use interface to
generate important MR-centric data aggregations in a bulk. *View methods*
provide a means to group similar computations or calculation flows into
cohesive units that respond dynamically to their inputs.

The key element that controls view method behaviour is the specification of
``kwargs`` (`key word arguments <https://docs.python.org/2/tutorial/controlflow.html#keyword-arguments>`_)
that dictate the aggregation process. For example, ``kwargs`` control if an
aggregation of simple cell counts is turned into percentages and if so if the
latter will become column or row percentages. This can be seen by comparing
the defintions of the ``kwarg`` parameter between the three
``QuantipyViews.known_views`` ``'counts'``, ``'c%'`` and ``'r%'``::

        self.known_methods['counts'] = {
            'method': 'frequency',
            'kwargs': {
            	'text': ''
            }
        }        
        self.known_methods['c%'] = {
            'method': 'frequency',
            'kwargs': {
            	'text': '',
                'rel_to': 'y'
            }
        }
        self.known_methods['r%'] = {
            'method': 'frequency',
            'kwargs': {
            	'text': '',
                'rel_to': 'x'
            }
        }

All three are using the ``frequency`` view method but the ``'rel_to'``
parameter passed in the ``kwargs`` dictionary differs between them.
Internally, it instructs the execution of ``Quantity.normalize()``, yielding
the required representation of the result. Keeping in mind that the decision
to show absolute or relative frequency applies to other count-based
aggregations like nets as well, it seems rather natural to unite similar
calculations and control their differences via such ``kwargs``.

There is a set of *common* ``'kwargs'`` that are used by all view methods
since they are needed for the general process flow. Inside a view method
they are provided by using ``View.std_params()`` which is returning a tuple
of the following items:

+---------------+---------------------------+-------------+----------------------+
| ``kwarg``     | controls                  | defaults to | accepts [1]_         |
+===============+===========================+=============+======================+
| ``'weights'`` | weighted aggregation      | ``None``    | var. name as ``str`` |
+---------------+---------------------------+-------------+----------------------+
| ``'pos'``     | orientation of the result | ``'x'``     | ``'x'``              | 
+---------------+---------------------------+-------------+----------------------+
| ``'relation'``| tbd.                      | ``None``    | tbd.                 | 
+---------------+---------------------------+-------------+----------------------+
| ``'rel_to'``  | normalization of results  | ``None``    | ``'x'``, ``'y'``,    | 
|               |                           |             | ``None``             | 
+---------------+---------------------------+-------------+----------------------+
| ``'text'``    | label for aggregations    | ``''``      | any ``'str'``        | 
+---------------+---------------------------+-------------+----------------------+

Except for ``'text'``, the values of these common ``kwargs`` are reflected
in the :doc:`view notation <views_concept_notation>`. 

Using ``ViewMapper`` templates
-------------------------------
To speed up the manipulation of view methods and allow for flexible batch
processing of aggregations, the ``ViewMapper`` class offers three tools:

1. ``make_template()`` will cycle through a selection of kwargs parameter
   values (given as lists) that should apply to a view method. 

2. ``add_method()`` is building on the template created, specifying the
   remaining ``kwargs`` to create different versions of an aggregation
   procedure.

3. ``subset()`` can extract a selection of view methods defined via
   ``add_method()``, to allow reusing template definitions while discretely
   applying the method specifications to portions of the Stack.

For example, you can create a template for calculating net code aggregations
by specifying ``iterators`` for ``'rel_to'`` becoming [``None``, ``'y'``] and
``'weights'`` [``None``, ``'weight_a'``] to generate weighted and unweighted
versions of count and column percentages of the net. Then, by adding the
method details, you define the codes that should feed into Top- and Bottom-
Boxes alongside their appropriate labels and apply different versions to
groups of variables with different scale point ranges.

Please see the following use cases on how to generate custom Views:

``frequency``
"""""""""""""
**specific** ``kwargs``:

+----------------+---------------------------+-------------+---------------------+
| ``kwarg``      | controls                  | defaults to | accepts             |
+================+===========================+=============+=====================+
| ``'logic'``    | new code group(s)         | ``None``    | list, list of dicts,|
|                | construction              |             | quantipy logic exp. |
+----------------+---------------------------+-------------+---------------------+
| ``'calc'``     | sum or difference         | ``None``    | dict defining a     |
|                | calculation of groups     |             | calculation         |
+----------------+---------------------------+-------------+---------------------+
| ``'calc_only'``| isolated appearance of    | ``False``   | ``boolean``         |
|                | ``'calc'`` result         |             |                     |
+----------------+---------------------------+-------------+---------------------+

**1: Generating a simple net code, only counts version**

*Generating a new template without any iterators*

>>> nets = qp.ViewMapper()
>>> nets.make_template('frequency')

*Using the* ``'logic'`` ``kwarg`` *to set up the codes to group and passing
it into* ``add_method()``

>>> net_spec = {'logic': [1, 2]}
>>> nets.add_method(name='net1', kwargs=net_spec)

*Adding the View*

>>> stack.add_link(x='q8', y=['@', 'gender'], views=nets)

*Results*

>>> stack.describe(index='view', columns=['x', 'y'])
x                             q8       
y                              @ gender
view                                   
x|frequency|x[(1,2)]:y|||net1  1      1

>>> stack[my_data_key]['no_filter']['q8']['@']['x|frequency|x[(1,2)]:y|||net1']
Question           q8
Values              @
Question Values      
q8       net1    1040

**2: Simple net, counts and column percentages, weighted and unweighted**

*Defining a template with iterators for counts and percentages, both weighted
and unweighted*

>>> nets = qp.ViewMapper()
>>> iterator_spec = {'rel_to': [None, 'y'], 'weights': [None, 'weight_a']}
>>> nets.make_template('frequency', iterators=iterator_spec)                                                  

*Specifiyng the net code and adding the View as done before*

>>> net_spec = {'logic': [1, 2]}
>>> nets.add_method(name='net1', kwargs=net_spec)
>>> stack.add_link(x='q8', y=['@', 'gender'], views=nets)

*Results*

>>> stack.describe(index='view', columns=['x', 'y'])
x                                      q8       
y                                       @ gender
view                                            
x|frequency|x[(1,2)]:y|y|weight_a|net1  1      1
x|frequency|x[(1,2)]:y|y||net1          1      1
x|frequency|x[(1,2)]:y||weight_a|net1   1      1
x|frequency|x[(1,2)]:y|||net1           1      1

>>> stack[my_data_key]['no_filter']['q8']['gender']['x|frequency|x[(1,2)]:y|y|weight_a|net1']
Question            gender           
Values                   1          2
Question Values                      
q8       net1    44.263136  42.433038

**3: Setting up a template, adding different methods, applying subsets**

*Defining the iterators for the template*

>>> nets = qp.ViewMapper()
>>> iterator_spec = {'rel_to': [None, 'y']}
>>> nets.make_template('frequency', iterators=iterator_spec)

*First net: codes 1&2*

>>> nets.add_method(name='net1', kwargs={'logic': [1, 2]})

*Second net: codes 3&4*

>>> nets.add_method(name='net2', kwargs={'logic': [4, 5]})

*Third net: codes 1, 2, 3, 4, 5, 6*

>>> nets.add_method(name='net3', kwargs={'logic': [1, 2, 3, 4, 5, 6]})

*Using* ``subset()`` *to apply only specific nets where appropriate*

>>> stack.add_link(x=['q5_1', 'q5_2'], y=['@', 'gender'], views=nets.subset(['net1', 'net2']))
>>> stack.add_link(x=['q2'], y=['@', 'gender'], views=nets.subset(['net3']))

*Results*

>>> stack.describe(index='view', columns=['x', 'y'])
x                                       q2        q5_1        q5_2       
y                                        @ gender    @ gender    @ gender
view                                                                     
x|frequency|x[(1,2)]:y|y||net1         NaN    NaN    1      1    1      1
x|frequency|x[(1,2)]:y|||net1          NaN    NaN    1      1    1      1
x|frequency|x[(1,2,3,4,5,6)]:y|y||net3   1      1  NaN    NaN  NaN    NaN
x|frequency|x[(1,2,3,4,5,6)]:y|||net3    1      1  NaN    NaN  NaN    NaN
x|frequency|x[(4,5)]:y|y||net2         NaN    NaN    1      1    1      1
x|frequency|x[(4,5)]:y|||net2          NaN    NaN    1      1    1      1

>>> stack[my_data_key]['no_filter']['q2']['gender']['x|frequency|x[(1,2,3,4,5,6)]:y|y||net3']
Question            gender           
Values                   1          2
Question Values                      
q2       net3    84.561892  79.308136

**4: Multiple code groups, weighted and unweighted counts, column %, row %**

*Defining the iterator template*

>>> groups = qp.ViewMapper()
>>> iterator_spec = {'rel_to': [None, 'y', 'x'], 'weights': [None, 'weight_a']}
>>> groups.make_template('frequency', iterators=iterator_spec)

*Setup of code groups, definition of method, adding the Views*

    >>> multi_groups = [
    ...                 {'Unlikely': [1,  2, 3]},
    ...                 {'Likely': [4, 5]},
    ...                 {'Does not apply/Not answered': [97, 98]}
    ...                ]

>>> groups.add_method(name='likeliness', kwargs={'logic': groups})

>>> x_vars = ['q5_' + str(subquestion) for subquestion in xrange(1,7)]
>>> x_vars
['q5_1', 'q5_2', 'q5_3', 'q5_4', 'q5_5', 'q5_6']

>>> stack.add_link(x=x_vars, y=['gender'], views=groups)

*Results*

>>> stack.describe(index='views')
view
x|frequency|x[(1,2,3),(4,5),(97,98)]:y|x|weight_a|likeliness    6
x|frequency|x[(1,2,3),(4,5),(97,98)]:y|x||likeliness            6
x|frequency|x[(1,2,3),(4,5),(97,98)]:y|y|weight_a|likeliness    6
x|frequency|x[(1,2,3),(4,5),(97,98)]:y|y||likeliness            6
x|frequency|x[(1,2,3),(4,5),(97,98)]:y||weight_a|likeliness     6
x|frequency|x[(1,2,3),(4,5),(97,98)]:y|||likeliness             6

>>> link = stack[my_data_key]['no_filter']['q5_1']['gender']

>>> link['x|frequency|x[(1,2,3),(4,5),(97,98)]:y|x||likeliness']
Question                                 gender           
Values                                        1          2
Question Values                                           
q5_1     Unlikely                     53.318765  46.681235
         Likely                       42.334361  57.665639
         Does not apply/Not answered  45.095095  54.904905

>>> link['x|frequency|x[(1,2,3),(4,5),(97,98)]:y|y||likeliness']
Question                                 gender           
Values                                        1          2
Question Values                                           
q5_1     Unlikely                     49.392713  39.716477
         Likely                       27.808704  34.789682
         Does not apply/Not answered  22.798583  25.493842

**5: Multiple code groups with additional calculation on aggregates**

*Iterator template*

>>> nps = qp.ViewMapper()
>>> iterator_spec = {'rel_to': [None, 'y']}
>>> groups.make_template('frequency', iterators=iterator_spec)

*Code groups and calculation method definition*

>>> from operator import sub

    >>> nps_groups = [
    ...               {'Promoters': [9, 10]},
    ...               {'Passives': [6, 7, 8]},
    ...               {'Detractors': [1, 2, 3, 4, 5]}
    ...              ]
    ... nps_score = {'NPS': (sub, ['Promoters', 'Detractors'])}

>>> method_spec_full = {'logic': nps_groups,  'calc': nps_score} 
>>> method_spec_score = {'logic': nps_groups,  'calc': nps_score, 'calc_only': True}

>>> nps.add_method(name='nps_full', kwargs=method_spec_full)
>>> nps.add_method(name='nps_score', kwargs=method_spec_score)

>>> stack.add_link(x='q6_1', y='gender', views=nps)

*Results*

>>> stack.describe(index='view')
view
x|frequency|x[(9,10),(6,7,8),(1,2,3,4,5)]:y|y||nps_full     1
x|frequency|x[(9,10),(6,7,8),(1,2,3,4,5)]:y|y||nps_score    1
x|frequency|x[(9,10),(6,7,8),(1,2,3,4,5)]:y|||nps_full      1
x|frequency|x[(9,10),(6,7,8),(1,2,3,4,5)]:y|||nps_score     1

>>> link = stack[my_data_key]['no_filter']['q6_1']['gender']

>>> link['x|frequency|x[(9,10),(6,7,8),(1,2,3,4,5)]:y|y||nps_full']
Question                gender           
Values                       1          2
Question Values                          
q6_1     Promoters   10.298583  10.969091
         Passives     3.441296   3.137346
         Detractors  86.260121  85.893563
         NPS        -75.961538 -74.924471

>>> link['x|frequency|x[(9,10),(6,7,8),(1,2,3,4,5)]:y|y||nps_score']
Question            gender           
Values                   1          2
Question Values                      
q6_1     NPS    -75.961538 -74.924471

``descriptives``
""""""""""""""""
**specific** ``kwargs``:

+----------------+---------------------------+-------------+---------------------+
| ``kwarg``      | controls                  | defaults to | accepts             |
+================+===========================+=============+=====================+
| ``'stats'``    | the measure to compute    | ``'mean'``  | valid ``str`` [2]_  |
|                |                           |             |                     |
+----------------+---------------------------+-------------+---------------------+
| ``'exclude'``  | exclusion of codes/values | ``None``    | list of ``float`` or|
|                |                           |             | ``int``             |
+----------------+---------------------------+-------------+---------------------+
| ``'rescale'``  | replacement of scale point| ``None``    | dict mapping old to |
|                | factors                   |             | new codes           |
+----------------+---------------------------+-------------+---------------------+

**1: Generating means, variance and standard deviation using iterators**

*Setting up the template iterators*

>>> stats = qp.ViewMapper()
>>> iterator_spec = {'stats': ['mean', 'var', 'stddev']}
>>> stats.make_template('descriptives', iterators=iterator_spec)

*Adding a method and adding the Views to the Stack (numeric x-axis)*

>>> stats.add_method(name='stats')
>>> stack.add_link(x='age', y='gender', views=stats)

*Results*

>>> stack.describe(index='view', columns=['x', 'y'])
x                       age
y                    gender
view                       
x|mean|x:y|||stats        1
x|stddev|x:y|||stats      1
x|var|x:y|||stats         1

>>> stack[my_data_key]['no_filter']['age']['gender']['x|mean|x:y|||stats']
Question           gender           
Values                  1          2
Question Values                     
age      mean    33.97748  33.840576

>>> stack[my_data_key]['no_filter']['age']['gender']['x|var|x:y|||stats']
Question            gender          
Values                   1         2
Question Values                     
age      var     78.745886  81.08104

>>> stack[my_data_key]['no_filter']['age']['gender']['x|stddev|x:y|||stats']
Question           gender          
Values                  1         2
Question Values                    
age      stddev  8.873888  9.004501

**2: Exluding cases with age above 20**

*Setting up the template iterators as done before*

>>> stats = qp.ViewMapper()
>>> iterator_spec = {'stats': ['mean', 'var', 'stddev']}
>>> stats.make_template('descriptives', iterators=iterator_spec)

*Preparing the list of code values to exclude and passing the* ``'exclude'``
``kwarg`` *to* ``add_method())``

>>> exclude_age = list(xrange(21, 110))
>>> stats_spec = {'exclude': exclude_age}
>>> stats.add_method(name='up_to_20', kwargs=stats_spec)

*Results*

>>> stack.describe(index='view', columns=['x', 'y'])
x                                       age
y                                    gender
view                                       
x|mean|x:y|||up_to_20        1
x|stddev|x:y|||up_to_20      1
x|var|x:y|||up_to_20         1

>>> stack[my_data_key]['no_filter']['age']['gender']['x|mean|x:y|||up_to_20']
Question            gender           
Values                   1          2
Question Values                      
age      mean    19.504274  19.552083

>>> stack[my_data_key]['no_filter']['age']['gender']['x|var|x:y|||up_to_20']
Question           gender          
Values                  1         2
Question Values                    
age      var     0.251055  0.248149

>>> stack[my_data_key]['no_filter']['age']['gender']['x|stddev|x:y|||up_to_20']
Question           gender          
Values                  1         2
Question Values                    
age      stddev  0.501054  0.498146

**3: Generating means and median, weighted and unweighted**

*Iterator definition*

>>> stats = qp.ViewMapper()
>>> iterator_spec = {'stats': ['mean', 'median'], 'weights': [None, weight_a]}
>>> stats.make_template('descriptives', iterators=iterator_spec)

*Method creation and adding View: code exclusion and rescaling 1-5 to 0-100
(categorical x-axis, metadata file assumed)*

    >>> met_def = {
    ...            'text': 'excl. 97, 98'
    ...            'exclude': [97, 98],    
    ...            'rescale': {1: 0, 2: 25, 3: 50, 4:75, 5:100}
    ...           }

>>> stats.add_method('means_medians', kwargs=met_def)

>>> x_vars = ['q5_' + str(subquestion) for subquestion in xrange(1,7)]
>>> x_vars
['q5_1', 'q5_2', 'q5_3', 'q5_4', 'q5_5', 'q5_6']

>>> stack.add_link(x=x_vars, y='locality', views=stats)

*Results*

>>> stack.describe(index='views')
view
x|mean|x[0,25,50,75,100]:y||weight_a|means_medians      6
x|mean|x[0,25,50,75,100]:y|||means_medians              6
x|median|x[0,25,50,75,100]:y||weight_a|means_medians    6
x|median|x[0,25,50,75,100]:y|||means_medians            6

>>> link = stack[my_data_key]['no_filter']['q5_1']['locality']

>>> link['x|mean|x[0,25,50,75,100]:y||weight_a|means_medians']
Question          locality                                            
Values                   1          2          3          4          5
Question Values                                                       
q5_1     mean    63.601247  65.408707  62.950265  64.700972  65.006439

>>> link['x|mean|x[0,25,50,75,100]:y||weight_a|means_medians'].meta()['agg']['text']
Mean excl. 97, 98

>>> link['x|median|x[0,25,50,75,100]:y||weight_a|means_medians']
Question        locality                
Values                 1   2   3   4   5
Question Values                         
q5_1     median       50  50  50  50  50

>>> link['x|median|x[0,25,50,75,100]:y||weight_a|means_medians'].meta()['agg']['text']
Median excl. 97, 98

.. rubric:: Footnotes
.. [1] **Please note**: Currently, Quantipy can compute aggregations which are
       applied to the x-axis of Link (``xpos`` being ``'x'``, except for the
       preset``rbase``). The ``rel_to`` parameter only applies to the
       ``frequency()`` method for now. A future goal is to implement a full
       *broadcasting*-like concept that is able to apply aggregations on the
       y-axis of a Link as well, e.g. net codes that summarize column codes. 
.. [2] Supported: ``'mean'``, ``'median'``, ``'var'``, ``'stddev'``,
       ``'varcoeff'``, ``'sem'``, ``'max'``, ``'min'``
