.. toctree::
 	:maxdepth: 5
	:includehidden:

====================
Inspecting variables
====================

------------------------------
Querying and slicing case data
------------------------------
A ``qp.DataSet`` is mimicking ``pandas``-like item access, i.e. passing a variable
name into the ``[]``-accessor will return a ``pd.DataFrame`` view of the
case data component. That means that we can chain any ``pd.DataFrame`` method to
the query:

>>> ds['q9'].head()
     q9
0   99;
1  1;4;
2   98;
3  1;4;
4   99;

There is the same support for selecting multiple variables at once:

>>> ds[['q9', 'gender']].head()
     q9  gender
0   99;       1
1  1;4;       2
2   98;       1
3  1;4;       1
4   99;       1

To integrate ``array`` (``masks``) variables into this behaviour, passing an
``array`` name will automatically call its item list:

>>> ds['q6'].head()
   q6_1  q6_2  q6_3
0     1     1     1
1     1   NaN     1
2     1   NaN     2
3     2   NaN     2
4     2    10    10

This can be combined with the ``list``-based selection as well:

>>> ds[['q6', 'q9', 'gender']].head()
   q6_1  q6_2  q6_3    q9  gender
0     1     1     1   99;       1
1     1   NaN     1  1;4;       2
2     1   NaN     2   98;       1
3     2   NaN     2  1;4;       1
4     2    10    10   99;       1


``DataSet`` case data supports row-slicing based on complex logical conditions
to inspect subsets of the data. We can use the ``take()`` with a ``Quantipy``
logic operation naturally for this:

>>> condition = intersection(
...    [{'gender': [1]},
...     {'religion': [3]},
...     {'q9': [1, 4]}])
>>> take = ds.take(condition)

>>> ds[take, ['gender', 'religion', 'q9']].head()
     gender  religion      q9
52        1         3  1;2;4;
357       1         3  1;3;4;
671       1         3  1;3;4;
783       1         3  2;3;4;
802       1         3      4;

.. seealso::
	Please find an overview of ``Quantipy`` logical operators and data slicing
	and masking in the :doc:`docs about complex logical conditions <06_logics>`!

--------------
Variable types
--------------
To get a summary of the all variables grouped by type, call ``variables()`` on
the ``DataSet``:

>>> ds.variables()
size: 8255     single delimited set array            int     float string        date      time N/A
0              gender            q2    q5  record_number    weight    q8a  start_time  duration
1            locality            q3    q7      unique_id  weight_a    q9a    end_time
2           ethnicity            q8    q6            age  weight_b
3            religion            q9            birth_day
4                  q1                        birth_month
5                 q2b                         birth_year
6                  q4
7                q5_1
8                q5_2
9                q5_3
10               q5_4
11               q5_5
12               q5_6
13               q6_1
14               q6_2
15               q6_3
16               q7_1
17               q7_2
18               q7_3
19               q7_4
20               q7_5
21               q7_6

We can restrict the output to certain types by providing the desired ones in
the ``only_type`` parameter:

>>> ds.variables(only_type='delimited set')
size: 8255 delimited set
0                     q2
1                     q3
2                     q8
3                     q9

>>> ds.variables(only_type=['delimited set', 'float'])
size: 8255 delimited set     float
0                     q2    weight
1                     q3  weight_a
2                     q8  weight_b
3                     q9       NaN

In addition to that, ``DataSet`` implements the following instance attributes
that return the corresponding variables as a ``list`` for easy iteration::

	DataSet.singles
	       .delimied_sets
	       .ints
	       .floats
	       .dates
	       .strings
	       .masks
	       .columns
	       .sets

>>> ds.delimited_sets
[u'q3', u'q2', u'q9', u'q8']

>>> for delimited_set in ds.delimited_sets:
...     print delimited_set
q3
q2
q9
q8

----------------------------------
Slicing & dicing  metadata objects
----------------------------------

Although it is possible to access a ``DataSet`` meta component via its ``_meta``
attribute directly, the prefered way to inspect and interact with with the metadata
is to use ``DataSet`` methods. For instance, the easiest way to view the most
important meta on a variable is to use the ``meta()`` method:

>>> ds.meta('q8')
delimited set                                      codes                         texts missing
q8: Which of the following do you regularly skip?
1                                                      1                     Breakfast    None
2                                                      2          Mid-morning snacking    None
3                                                      3                         Lunch    None
4                                                      4        Mid-afternoon snacking    None
5                                                      5                        Dinner    None
6                                                     96                  None of them    None
7                                                     98  Don't know (it varies a lot)    None

This output is extended with the ``item`` metadata if an ``array`` is passed:

>>> ds.meta('q6')
single                                             items                   item texts  codes                        texts missing
q6: How often do you take part in any of the fo...
1                                                   q6_1               Exercise alone      1     Once a day or more often    None
2                                                   q6_2       Join an exercise class      2               Every few days    None
3                                                   q6_3  Play any kind of team sport      3                  Once a week    None
4                                                                                          4             Once a fortnight    None
5                                                                                          5                 Once a month    None
6                                                                                          6        Once every few months    None
7                                                                                          7        Once every six months    None
8                                                                                          8                  Once a year    None
9                                                                                          9  Less often than once a year    None
10                                                                                        10                        Never    None

If the variable is not categorical, ``meta()`` returns simply:

>>> ds.meta('weight_a')
                             float
weight_a: Weight (variant A)   N/A

``DataSet`` also provides a lot of methods to access and return the several
meta objects of a variable to make various data processing tasks easier:

**Variable labels**: :meth:`quantipy.core.dataset.DataSet.text`

>>> ds.text('q8', text_key=None)
Which of the following do you regularly skip?

``values`` **object**: :meth:`quantipy.core.dataset.DataSet.values`

>>> ds.values('gender', text_key=None)
[(1, u'Male'), (2, u'Female')]

**Category codes**: :meth:`quantipy.core.dataset.DataSet.codes`

>>> ds.codes('gender')
[1, 2]

**Category labels**: :meth:`quantipy.core.dataset.DataSet.value_texts`

>>> ds.value_texts('gender', text_key=None)
[u'Male', u'Female']

``items`` **object**: :meth:`quantipy.core.dataset.DataSet.items`

>>> ds.items('q6', text_key=None)
[(u'q6_1', u'How often do you exercise alone?'),
 (u'q6_2', u'How often do you take part in an exercise class?'),
 (u'q6_3', u'How often do you play any kind of team sport?')]

**Item** ``'columns'`` **sources**: :meth:`quantipy.core.dataset.DataSet.sources`

>>> ds.sources('q6')
[u'q6_1', u'q6_2', u'q6_3']

**Item labels**: :meth:`quantipy.core.dataset.DataSet.item_texts`

>>> ds.item_texts('q6', text_key=None)
[u'How often do you exercise alone?',
 u'How often do you take part in an exercise class?',
 u'How often do you play any kind of team sport?']