.. toctree::
  :maxdepth: 5
  :includehidden:

Signficance testing
===================

| :ref:`genindex`
| :ref:`modindex`

""""

The ``Test`` class
------------------

:class: ``quantipy.Test(link, view_name_notation)``

Quantipy offers tests of statistical significance for means and proportions,
including nets and other user-created code combinations, via its ``Test``
class. Tests are based on comparisons between all unqiue column code pairs of
a ``Link``. Similiar to the ``Quantity`` object, testing can be performed
ad-hoc on appropriate views kept in a Stack or in an automated fashion
using the ``View`` object's features.

To create ``Test`` instance, a Link and the view notation string of a **count**
-based (i.e. **not** column a row percentages) or a **mean** View must be
provided. For example, let's consider a small Stack that contains some
frequency, net code and mean aggregations:

>>> stack.describe(index='view', columns=['x', 'y'])
x                                   birth_year            q5_1              q8         
y                                       gender locality gender locality gender locality
view                                                                                   
x|frequency|x[(1,2)]:y|y||net1             NaN      NaN    NaN      NaN      1        1
x|frequency|x[(1,2)]:y|||net1              NaN      NaN    NaN      NaN      1        1
x|frequency||y||c%                           1        1      1        1      1        1
x|frequency||||counts                        1        1      1        1      1        1
x|mean|x:y|||mean                            1        1      1        1    NaN      NaN
x|mean|x[1,2,3,4,5]:y|||excl. 97/98        NaN      NaN      1        1    NaN      NaN

First we will test the proportions and the simple preset mean on the following
Link:

>>> link = stack[my_data_key]['no_filter']['q5_1']['gender']

To create a ``Test`` object for our proportions test, passing the view
notation results in:

	>>> props_test = qp.Test(link, 'x|frequency||||counts')
	>>> props_test
	<class 'quantipy.core.quantify.engine.Test'>, test metric: proportions,
	parameters: None, mimicked: None, level: None

While doing the same for the simple mean shows the following:

	>>> means_test = qp.Test(link, 'x|mean|x:y|||mean')
	>>> means_test
	<class quantipy.core.quantify.engine.Test'>, test metric: means,
	parameters: None, mimicked: None, level: None

This tells us that the ``Test`` object instances correctly infered the metric
to be tested. However, the tests do not have any options set yet, their
``parameters``, ``mimicked`` and ``level`` information are all ``None``. To set
Quantipy's default test options, calling the ``set_params()`` method will
assign the parameter values, most noteworthy the level of signficance.

>>> props_test.set_params()
>>> props_test.level
0.05

>>> means_test.set_params()
>>> means_test.level
0.05

To start the test algorithm and show its results, we simply use ``run()``:

>>> props_test.run()
Question        gender     
Values               1    2
Question Values            
q5_1     1         NaN  NaN
         2         NaN  NaN
         3         [2]  NaN
         4         NaN  NaN
         5         NaN  [1]
         97        [2]  NaN
         98        NaN  [1]

>>> means_test.run()
Question        gender     
Values               1    2
Question Values            
q5_1     mean      NaN  [1]

Test results are presented by showing a list of column codes to that the resp.
relevant column mean or count results are significant larger to. Therefore,
in the above means test example the mean for the column group ``'2'`` is
larger than the one in the group of column code ``'1'``. To demonstrate this
on a case with more column codes, let's look at a proportions test for the
following Link:

>>> link = stack[my_data_key]['no_filter']['q8']['locality']
>>> props_test = qp.Test(link, 'x|frequency||||counts')
>>> props_test.set_params().run()
Question          locality                        
Values                   1          2    3   4   5
Question Values                                   
q8       1       [3, 4, 5]  [3, 4, 5]  NaN NaN NaN
         2             NaN        NaN  NaN NaN NaN
         3             NaN        NaN  NaN NaN NaN
         4             NaN        NaN  NaN NaN NaN
         5             NaN        NaN  NaN NaN NaN
         96            NaN        NaN  NaN NaN NaN
         98            NaN        [5]  [5] NaN NaN

Test options & mimicking
------------------------

:method: ``quantipy.Test.set_params(level='mid', mimic='Dim', testtype='pooled',
                   				    use_ebase=True, ovlp_correc=True,
                   				    cwi_filter=False)``

The ``set_params()`` method does a lot more than just setting the level of
signficiance. As a default it will perform a pooled variances test that uses
effective sample sizes and an overlap control formula to balance the
effects of possible data weighting and multi-coded column groups, i.e. when a
Link's y-axis consists of a multiple response question. This test setup mimics
the test algorithm of *SPSS Data Collection Professional*/*Dimensions*
(illustrated by the value of the ``mimic`` parameter being ``'Dim'`` ).

To mimic one of the algorithms used by the *Askia* survey software (a z-score
based, unpooled variances approach), ``mimic`` can be set to ``'askia'``. In
comparison, the two ``mimic`` options correspond to the following test
settings:

+----------------+-------------------------------------+-------------------------------------+
|                |``mimic='Dim'``                      | ``mimic='askia'``                   |
+================+=====================================+=====================================+
| ``testtype``   | ``pooled``:                         | ``unpooled``:                       |
|                | pooled variances,                   | unpooled variances,                 |
|                | assumes equal variances             | assumes unequal variances           |      
+----------------+-------------------------------------+-------------------------------------+
| ``use_ebase``  | ``True``:                           | ``False``:                          |
|                | effective sample sizes instead of   | weighted sample sizes used for      |
|                | weighted ones for weighted data     | weighted data                       |
+----------------+-------------------------------------+-------------------------------------+
| ``ovlp_correc``| ``True``:                           | ``False``:                          |
|                | case data overlap between column    | overlap is ignored and might bias   |
|                | codes is controlled for and adjusted| mean tests when y-axis is multi-code|
+----------------+-------------------------------------+-------------------------------------+
| ``cwi_filter`` | ``False``:                          | ``True``:                           |
|                | TODO                                | TODO                                |
|                |                                     |                                     |
+----------------+-------------------------------------+-------------------------------------+


.. note::
	Currently, direct changes to the parameter settings are ignored by Quantipy.
	It is therefore not possible to combine e.g. a test that uses unpooled
	variances in combination with overlap correction.  In a future version of
	Quantipy this limitation will be overcome and the test settings will become
	fully exposed to the user.


**Level of significance**: Due to the different (mimicked) test approaches,
levels of signficance generally are choosen by specifying ``level`` to either
be ``'low'`` (=0.1), ``'mid'`` (=0.05) or ``'high'`` (=0.01), which covers the
regularly used test levels in MR environments. When mimicking the *Dimensions*
package it is also possible to pass the level of significance as a ``float``
value between 0 and 1. If this is done mimickig ``'askia'``, the level will
fall back to ``'low'``.  

``coltests`` view method
------------------------
Just as the ``frequency`` and ``descriptives`` view methods wrap the
``Quantity`` aggregation features, ``coltests`` is doing the same with
Quantipy's significance testing functionality. Applying batch-style
significance testing to a Stack therefore is made easy by using
``ViewMapper``'s ``make_template()`` and ``add_method()`` methods. The
``coltests`` **specific** ``kwargs`` are as follows:

+----------------+---------------------------+-------------+---------------------+
| ``kwarg``      | controls                  | defaults to | accepts             |
+================+===========================+=============+=====================+
| ``'metric'``   | props or means test       | ``'props'`` | ``'props'`` or      |
|                | calculation               |             | ``'means'``         |
+----------------+---------------------------+-------------+---------------------+
| ``'mimic'``    | test setup to be used     | ``'Dim'``   | ``'Dim'`` or        |
|                | via mimicking             |             | ``'askia'``         |
+----------------+---------------------------+-------------+---------------------+
| ``'level'``    | level of significance     | ``'mid'``   | ``'low'``, ``'mid'``|
|                |                           |             | or ``'high'``;      |
|                |                           |             | ``float`` value     | 
+----------------+---------------------------+-------------+---------------------+

Using the example Stack from above, here are some usage examples:

**1. Running the default**

>>> default_sigtest = qp.ViewMapper()
>>> default_sigtest.make_template('coltests')

>>> default_sigtest.add_method(name='props_Dim_0.05')
>>> stack.add_link(views=default_sigtest)

>>> stack.describe(index='view', columns=['x', 'y'])
x                                                birth_year            q5_1              q8         
y                                                    gender locality gender locality gender locality
view                                                                                                
x|frequency|x:y|||cbase                                   1        1      1        1      1        1
x|frequency|x[(1,2)]:y|y||net1                          NaN      NaN    NaN      NaN      1        1
x|frequency|x[(1,2)]:y|||net1                           NaN      NaN    NaN      NaN      1        1
x|frequency||y||c%                                        1        1      1        1      1        1
x|frequency||||counts                                     1        1      1        1      1        1
x|mean|x:y|||mean                                         1        1      1        1    NaN      NaN
x|mean|x[1,2,3,4,5]:y|||excl. 97/98                     NaN      NaN      1        1    NaN      NaN
x|tests.props.Dim.10|x[(1,2)]:y|||props_Dim_0.05        NaN      NaN    NaN      NaN      1        1
x|tests.props.Dim.10||||props_Dim_0.05                    1        1      1        1      1        1

**2. Testing with different levels of significance**

*Iterator template for levels*

>>> multiple_tests = qp.ViewMapper()
>>> iterator_spec = {'level': ['high', 'mid', 'low']}
>>> multiple_tests.make_template('coltests', iterators=iterator_spec)

*Specifying the view method and running the tests on the whole Stack*

>>> multiple_tests.add_method(name='props_tests_Dim')
>>> stack.add_link(views=multiple_tests)

*Results*

>>> stack.describe(index='view', columns=['x', 'y'])
x                                                 birth_year            q5_1              q8         
y                                                     gender locality gender locality gender locality
view                                                                                                 
x|frequency|x[(1,2)]:y|y||net1                           NaN      NaN    NaN      NaN      1        1
x|frequency|x[(1,2)]:y|||net1                            NaN      NaN    NaN      NaN      1        1
x|frequency||y||c%                                         1        1      1        1      1        1
x|frequency||||counts                                      1        1      1        1      1        1
x|mean|x:y|||mean                                          1        1      1        1    NaN      NaN
x|mean|x[1,2,3,4,5]:y|||excl. 98/98                      NaN      NaN      1        1    NaN      NaN
x|tests.props.Dim.01|x[(1,2)]:y|||props_tests_Dim        NaN      NaN    NaN      NaN      1        1
x|tests.props.Dim.01||||props_tests_Dim                    1        1      1        1      1        1
x|tests.props.Dim.05|x[(1,2)]:y|||props_tests_Dim        NaN      NaN    NaN      NaN      1        1
x|tests.props.Dim.05||||props_tests_Dim                    1        1      1        1      1        1
x|tests.props.Dim.10|x[(1,2)]:y|||props_tests_Dim        NaN      NaN    NaN      NaN      1        1
x|tests.props.Dim.10||||props_tests_Dim                    1        1      1        1      1        1

>>> link = stack[my_data_key]['no_filter']['q8']['gender']

>>> link['x|tests.props.Dim.01|x[(1,2)]:y|||props_tests_Dim']
Question        locality                 
Values                 1    2   3   4   5
Question Values                          
q8       net1        [3]  [3] NaN NaN NaN

>>> link['x|tests.props.Dim.05|x[(1,2)]:y|||props_tests_Dim']
Question          locality                       
Values                   1          2   3   4   5
Question Values                                  
q8       net1    [3, 4, 5]  [3, 4, 5] NaN NaN NaN

>>> link['x|tests.props.Dim.10|x[(1,2)]:y|||props_tests_Dim']
Question          locality                       
Values                   1          2   3   4   5
Question Values                                  
q8       net1    [3, 4, 5]  [3, 4, 5] NaN NaN NaN

**3. Different levels of significance, tests for means**

*Iterator template for levels and metrics*

>>> all_tested = qp.ViewMapper()

  >>> iterator_spec = {
  ...                  'level': ['high', 'mid', 'low'],
  ...                  'metric': ['props', 'means']
  ...                  }

>>> all_tested.make_template('coltests', iterators=iterator_spec)

*Creating and adding the view methods*

>>> all_tested.add_method(name='column_tests_Dim')
>>> stack.add_link(views=all_tested)

*Results*

>>> stack.describe(index='view', columns=['x', 'y'])
x                                           birth_year            q5_1              q8         
y                                               gender locality gender locality gender locality
view                                                                                           
x|frequency|x[(1,2)]:y|y||net1                     NaN      NaN    NaN      NaN      1        1
x|frequency|x[(1,2)]:y|||net1                      NaN      NaN    NaN      NaN      1        1
x|frequency||y||c%                                   1        1      1        1      1        1
x|frequency||||counts                                1        1      1        1      1        1
x|mean|x:y|||mean                                    1        1      1        1    NaN      NaN
x|mean|x[1,2,3,4,5]:y|||excl. 98/98                NaN      NaN      1        1    NaN      NaN
x|tests.means.Dim.01|x:y|||tests                     1        1      1        1    NaN      NaN
x|tests.means.Dim.01|x[1,2,3,4,5]:y|||tests        NaN      NaN      1        1    NaN      NaN
x|tests.means.Dim.05|x:y|||tests                     1        1      1        1    NaN      NaN
x|tests.means.Dim.05|x[1,2,3,4,5]:y|||tests        NaN      NaN      1        1    NaN      NaN
x|tests.means.Dim.10|x:y|||tests                     1        1      1        1    NaN      NaN
x|tests.means.Dim.10|x[1,2,3,4,5]:y|||tests        NaN      NaN      1        1    NaN      NaN
x|tests.props.Dim.01|x[(1,2)]:y|||tests            NaN      NaN    NaN      NaN      1        1
x|tests.props.Dim.01||||tests                        1        1      1        1      1        1
x|tests.props.Dim.05|x[(1,2)]:y|||tests            NaN      NaN    NaN      NaN      1        1
x|tests.props.Dim.05||||tests                        1        1      1        1      1        1
x|tests.props.Dim.10|x[(1,2)]:y|||tests            NaN      NaN    NaN      NaN      1        1
x|tests.props.Dim.10||||tests                        1        1      1        1      1        1

>>> link = stack[my_data_key]['no_filter']['birthyear']['gender']
>>> link['x|tests.means.Dim.10|x:y|||tests']
Question          gender    
Values                 1   2
Question   Values           
birth_year mean      NaN NaN

>>> link = stack[my_data_key]['no_filter']['q5_1']['gender']
>>> link['x|tests.means.Dim.10|x[1,2,3,4,5]:y|||tests']
Question        gender     
Values               1    2
Question Values            
q5_1     mean      NaN  [1]

**4. Testing only a portion of a Stack**

*Setting up iterators and template*

>>> q8_only = qp.ViewMapper()
>>> iterator_spec = {'level': ['high', 'mid', 'low']}
>>> q8_only.make_template('coltests', iterators=iterator_spec)

*Creating and adding the methods to a selection of Links*

>>> q8_only.add_method(name='tests_on_q8')
>>> stack.add_link(x='q8', views=q8_only)

*Results*

>>> stack.describe(index='view', columns=['x', 'y'])
x                                             birth_year            q5_1              q8         
y                                                 gender locality gender locality gender locality
view                                                                                             
x|frequency|x[(1,2)]:y|y||net1                       NaN      NaN    NaN      NaN      1        1
x|frequency|x[(1,2)]:y|||net1                        NaN      NaN    NaN      NaN      1        1
x|frequency||y||c%                                     1        1      1        1      1        1
x|frequency||||counts                                  1        1      1        1      1        1
x|mean|x:y|||mean                                      1        1      1        1    NaN      NaN
x|mean|x[1,2,3,4,5]:y|||excl. 98/98                  NaN      NaN      1        1    NaN      NaN
x|tests.props.Dim.01|x[(1,2)]:y|||tests_on_q8        NaN      NaN    NaN      NaN      1        1
x|tests.props.Dim.01||||tests_on_q8                  NaN      NaN    NaN      NaN      1        1
x|tests.props.Dim.05|x[(1,2)]:y|||tests_on_q8        NaN      NaN    NaN      NaN      1        1
x|tests.props.Dim.05||||tests_on_q8                  NaN      NaN    NaN      NaN      1        1
x|tests.props.Dim.10|x[(1,2)]:y|||tests_on_q8        NaN      NaN    NaN      NaN      1        1
x|tests.props.Dim.10||||tests_on_q8                  NaN      NaN    NaN      NaN      1        1

**4. Using different test levels for different portions of the Stack**

*Setting up the template*

>>> specific_levels = qp.ViewMapper()
>>> specific_levels.make_template('coltests')

*Adding methods wih different levels and using* ``subset()``

>>> specific_levels.add_method(name='0.1', kwargs={'level': 'high'})
>>> specific_levels.add_method(name='0.5', kwargs={'level': 'mid'})
>>> specific_levels.add_method(name='very low level', kwargs={'level': 0.2})

>>> stack.add_link(x='q8', views=specific_levels.subset(['0.1', '0.5']))
>>> stack.add_link(x='birth_year', views=specific_levels.subset('very low level'))
>>> stack.add_link(x='q5_1', views=specific_levels.subset('0.5'))

*Results*

>>> stack.describe(index='view', columns=['x', 'y'])
x                                      birth_year            q5_1              q8         
y                                          gender locality gender locality gender locality
view                                                                                      
x|frequency|x[(1,2)]:y|y||net1                NaN      NaN    NaN      NaN      1        1
x|frequency|x[(1,2)]:y|||net1                 NaN      NaN    NaN      NaN      1        1
x|frequency||y||c%                              1        1      1        1      1        1
x|frequency||||counts                           1        1      1        1      1        1
x|mean|x:y|||mean                               1        1      1        1    NaN      NaN
x|mean|x[1,2,3,4,5]:y|||excl. 98/98           NaN      NaN      1        1    NaN      NaN
x|tests.props.Dim.01|x[(1,2)]:y|||0.1         NaN      NaN    NaN      NaN      1        1
x|tests.props.Dim.01||||0.1                   NaN      NaN    NaN      NaN      1        1
x|tests.props.Dim.05|x[(1,2)]:y|||0.5         NaN      NaN    NaN      NaN      1        1
x|tests.props.Dim.05||||0.5                   NaN      NaN      1        1      1        1
x|tests.props.Dim.20||||very low level          1        1    NaN      NaN    NaN      NaN