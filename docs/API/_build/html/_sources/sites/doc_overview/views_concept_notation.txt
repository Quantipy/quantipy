.. toctree::
  :maxdepth: 5
  :includehidden:

=========================
View concept introduction
=========================

| :ref:`genindex`
| :ref:`modindex`

""""

What is a View?
---------------
View objects are created by the ``quantipy.ViewMapper`` class that is using
*view methods* defined in the ``quantipy.core.view_generators.view_maps``
module. ``ViewMapper`` handles the coordination and structuring duties of
creating blueprints for the aggregation instructions specified in the view
methods. 

Broadly speaking, the Quantipy ``View`` object is a framework to control,
automate and extend data aggregation routines based on the ``quantify``
computation methods. In addition, Views offer self-inspection methods that
can feedback meta information on their creation details, data modifications
and export specifications. Also, the class offers convience functions to
ease up writing your own view methods and is easily extendable.


Using preset ``QuantipyViews``
------------------------------

Adding views: One-line example
""""""""""""""""""""""""""""""
The easiest way to demonstrate how Views work is to turn to the *preset*
views organized in ``known_methods`` property of the
``view_maps.QuantipyViews`` class first. The following 8 views are ready to
simply get passed to the ``views`` parameter of the
``quantipy.Stack.add_link()`` method and provide quick access to very simple
baseline aggregations that are regularly used in MR analysis job:

+---------------+---------------------------+------------------------+
| Preset        | Calculation               | view method            |
+===============+===========================+========================+
| ``'default'`` | meta dependent summary    | ``default``            | 
+---------------+---------------------------+------------------------+
| ``'cbase'``   | column base sizes         | ``frequency``          |
+---------------+---------------------------+------------------------+
| ``'rbase'``   | row base sizes            | ``frequency``          |
+---------------+---------------------------+------------------------+
| ``'ebase'``   | effective base sizes      | ``frequency``          |
+---------------+---------------------------+------------------------+
| ``'counts'``  | counts cross-tabulated    | ``frequency``          |
+---------------+---------------------------+------------------------+
| ``'c%'``      | col. perc. cross-tabulated| ``frequency``          |
+---------------+---------------------------+------------------------+
| ``'r%'``      | row perc. cross-tabulated | ``fequency``           |
+---------------+---------------------------+------------------------+
| ``'mean'``    | mean                      | ``decriptives``        |
+---------------+---------------------------+------------------------+ 

It is instructive to see how the different presets behave when running a bulk
analysis defined by a Stack setup. In the following we use ``'default'``
(from the view method with the same name), ``'c%'`` (stemming the ``frequency``
view method) and ``'mean'`` (computed via the ``descriptives`` method).
Constructing a Stack including views is done per:

>>> xs = ['q5_1', 'q8', 'birth_year']
>>> ys = ['gender', 'Wave', 'birth_month']
>>> my_first_views = ['default', 'c%', 'mean']
>>> stack.add_link(x=xs, y=ys, views=my_first_views)

To boost the Stack up even furter, specifiying the ``add_link()`` parameter
``weights`` allows to generate the aggregations weighted and unweighted in a
single run:

>>> stack.add_link(x=xs, y=ys, views=my_first_views, weights=['weight_a', None])

``None`` in the list for ``weights`` indicates unweighted processing,
``'weight_a'`` is a variable of type ``float``. For further reference, we call
this Stack ``simple_stack``:

>>> simple_stack = stack

.. note::
  Variables used for weighted aggregations always have to be of type ``float``
  in order to yield correct results.

Adding views: Complex example
"""""""""""""""""""""""""""""
Generating the Views alongside the Link definitions is not required by any
means: It is perfectly fine to set up the Stack in terms of
data keys, filters and x/y specifications first and then add the VSiews in
another call of ``add_link()`` passing in a list of views, optionally
controlling to which data keys, filter definitions or x/y variable selections
the views are getting applied to.

Similar to the example from the :doc:`Links and Stack <link_and_stack>` part
of this documentation, let's set up a second Stack that contains filters
in addition to the x and y variables from above:

  >>> complex_stack.add_link(filters=['no_filter', 'gender==1', 'gender==2'],
  ...                        x=['q5_1', 'q8', 'birth_year'],
  ...                        y=['gender', 'Wave', 'birth_month'])

Now, the weighted and unweighted ``'default'`` view is added to everything
contained in the ``complex_stack``:

  >>> complex_stack.add_link(filters=['no_filter', 'gender==1', 'gender==2'],
  ...                        views=['default'], weights=[None, 'weight_a'])

We add the unweighted ``'mean'`` only to the unfiltered version of the data:

>>> complex_stack.add_link(filters=['no_filter'], views=['mean'])

\... and restrict the ``'c%'`` view to the the data that is filtered for
``'gender==1'`` and the y variables ``'birth_month'`` and ``'Wave'``,
generating only the weighted version:

  >>> complex_stack.add_link(filters=['gender==1'],
  ...                        y=['birth_month', 'Wave']
  ...                        weights='weight_a')

What's inside the Stack?
""""""""""""""""""""""""
To see what we have produced, the ``describe()`` method it used again,
now giving additional information on the freshly added Views. Let's start with
the simple example, producing an overview of views by x/y variables:

>>> simple_stack.describe(index='view', columns=['x', 'y'])
x                               birth_year                    q5_1                      q8                   
y                                     Wave birth_month gender Wave birth_month gender Wave birth_month gender
view                                                                                                         
x|default|x:y||weight_a|default          1           1      1    1           1      1    1           1      1
x|default|x:y|||default                  1           1      1    1           1      1    1           1      1
x|frequency||y|weight_a|c%               1           1      1    1           1      1    1           1      1
x|frequency||y||c%                       1           1      1    1           1      1    1           1      1
x|mean|x:y||weight_a|mean                1           1      1    1           1      1  NaN         NaN    NaN
x|mean|x:y|||mean                        1           1      1    1           1      1  NaN         NaN    NaN

Two things immediately strike the eye:

1. The names of the views are not identical to the short names passed into
   the ``add_link()`` method.
2. The ``'mean'`` is missing for Links that have question q8 on the x-axis,
   indicated by ``NaN`` in ``describe()``'s output.

The reason for the somewhat odd view names is that Quantipy is using a special
*view notation* to generate the final key of the Stack dictionary. Since
there might be different versions of a ``'c%'`` or ``'mean'`` View, due e.g. to
multiple weights or excluded codes, the view notation is making sure that each
aggregation can be placed into the Stack without overwriting a related already
existing one. We will explore the notation further in a bit <ANCHOR>.

But why is the mean missing for variable ``'q8'``? It is a multi-coded
question, calculating summary statistics like the mean on this data type does
not make much sense, so the ``descriptives`` view method checks the variable's
type against the file meta and only proceeds if it can confirm a single-coded
categorical or numeric (``int`` or ``float``) variable. Without any file meta,
however, such a check will will get skipped and the mean would be created
regardless.

Turning to the ``complex_stack``, we add the ``'filter'`` level to
``describe()`` which reveals that we ended up with the aggregations that we
requested, not specified Link/View combinations are correctly marked ``NaN``:

>>> complex_stack.describe(columns=['x', 'y', 'filter'], columns='view')
view                              x|default|x:y||weight_a|default  x|default|x:y|||default  x|frequency||y|weight_a|c%  x|mean|x:y|||mean
x          y           filter                                                                                                            
birth_year Wave        gender==1                                1                        1                           1                NaN
                       gender==2                                1                        1                         NaN                NaN
                       no_filter                                1                        1                         NaN                  1
           birth_month gender==1                                1                        1                           1                NaN
                       gender==2                                1                        1                         NaN                NaN
                       no_filter                                1                        1                         NaN                  1
           gender      gender==1                                1                        1                         NaN                NaN
                       gender==2                                1                        1                         NaN                NaN
                       no_filter                                1                        1                         NaN                  1
q5_1       Wave        gender==1                                1                        1                           1                NaN
                       gender==2                                1                        1                         NaN                NaN
                       no_filter                                1                        1                         NaN                  1
           birth_month gender==1                                1                        1                           1                NaN
                       gender==2                                1                        1                         NaN                NaN
                       no_filter                                1                        1                         NaN                  1
           gender      gender==1                                1                        1                         NaN                NaN
                       gender==2                                1                        1                         NaN                NaN
                       no_filter                                1                        1                         NaN                  1
q8         Wave        gender==1                                1                        1                           1                NaN
                       gender==2                                1                        1                         NaN                NaN
                       no_filter                                1                        1                         NaN                NaN
           birth_month gender==1                                1                        1                           1                NaN
                       gender==2                                1                        1                         NaN                NaN
                       no_filter                                1                        1                         NaN                NaN
           gender      gender==1                                1                        1                         NaN                NaN
                       gender==2                                1                        1                         NaN                NaN
                       no_filter                                1                        1                         NaN                NaN 

To conclude this section, we have a quick look at the ``'default'`` view to
complete our understanding of the Stack object and to become familiar with
this View's special behaviour: ``'default'`` requires file metadata passed
into the Stack, since it will create either a categorical or numerical summary
tabulation depending on the type of the x-axis variable. Recalling that
``'birth_year'`` is of type int and ``'q8'`` a multi-coded categorical question,
we pass the view notation key as the final level of the Stack, returning the
View object that stores the aggregation:

*numerical* ``'default'``

>>> link = complex_stack[my_data_key]['no_filter']['birth_year']['gender']
>>> link['x|default|x:y|||default']
Question                gender                          
Values                       1            2          All
Question   Values                                       
birth_year All     3952.000000  4303.000000  8255.000000
           mean    1981.022520  1981.159424  1981.093882
           stddev     8.873888     9.004501     8.941930
           min     1966.000000  1966.000000  1966.000000
           25%     1973.000000  1973.000000  1973.000000
           median  1981.000000  1981.000000  1981.000000
           75%     1989.000000  1989.000000  1989.000000
           max     1996.000000  1996.000000  1996.000000

*categorical* ``'default'``

>>> link = complex_stack[my_data_key]['no_filter']['q8']['gender']
>>> link['x|default|x:y|||default']
Question        gender            
Values               1     2   All
Question Values                   
q8       All      1211  1156  2367
         1         473   476   949
         2         112   104   216
         3         293   302   595
         4         507   463   970
         5         636   599  1235
         96        165   118   283
         98         26    23    49

Using the categorical example and comparing it with the filtered versions,
shows that the filtering worked as expected as well:

>>> complex_stack[my_data_key]['gender==1']['q8']['gender']['x|default|x:y|||default']
Question        gender      
Values               1   All
Question Values             
q8       All      1211  1211
         1         473   473
         2         112   112
         3         293   293
         4         507   507
         5         636   636
         96        165   165
         98         26    26

>>> complex_stack[my_data_key]['gender==2']['q8']['gender']['x|default|x:y|||default']
Question        gender      
Values               2   All
Question Values             
q8       All      1156  1156
         1         476   476
         2         104   104
         3         302   302
         4         463   463
         5         599   599
         96        118   118
         98         23    23

View notation, meta and self-inspection
---------------------------------------
The view notation is composed of different parts describing the defining
characteristics of the internal view processing. Working with Views and seeing
different examples of view notations, this *mini language* will become more
and more obvious. Luckily, ``describe()`` offers a helpful feature to break up
the View names into their parts via the ``split_view_names`` parameter that we
now set to ``True``:

>>> simple_stack.describe(split_view_names=True)
                              view xpos        agg relation rel_to   weights shortname
0  x|default|x:y||weight_a|default    x    default      x:y         weight_a   default
1          x|default|x:y|||default    x    default      x:y                    default
2       x|frequency||y|weight_a|c%    x  frequency               y  weight_a        c%
3                x|mean|x:y|||mean    x       mean      x:y                       mean

:TODO: Write short example explanation

Once a view has been successfully calculated, there are different ways to
get information on its construction details without the need to "translate"
the view notation. Calling ``meta()`` on a View will return aggregation
characteristics in the following dict structure

>>> link = simple_stack[my_data_key]['no_filter']['q8']['gender']
>>> my_view = link['x|frequency||y|weight_a|c%']
>>> my_view.meta()
{
 'agg':
      {'name': 'c%', 'text': '', 'is_weighted': True, 'weights': 'weight_a',
       'fullname': 'x|frequency||y|weight_a|c%', 'method': 'frequency'},
  'x':
      {'name': 'q8', 'is_multi': True, 'is_nested': False},
  'y':
      {'name': 'gender', 'is_multi': False, 'is_nested': False},
  'shape':
      (7, 2)
}

that summarizes some global information. There is also a range of methods to
extract specific properties of the aggregation, for example:

>>> my_view.is_pct()
True
>>> my_view.is_weighted()
True

**Full list of self-inspection methods**

* missing()
* rescaling()
* weights()
* is_weighted()
* is_pct()
* is_base()
* is_net()
* is_counts()
* is_stat()
* is_meanstest()
* is_propstest()
