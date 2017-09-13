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
name into the ``[]``-accessor will return a ``pandas.DataFrame`` view of the
case data component. That means that we can chain any ``pandas.DataFrame`` method to
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

----------------------------
Variable and value existence
----------------------------

any, all, code_count, is_nan, var_exists, codes_in_data, is_like_numeric
variables

______________________________________________________________________________

We can use ``variables()`` and ``var_exists()`` to generally test the membership
of variables inside ``DataSet``. The former is showing the list of all variables
registered inside the ``'data file'`` ``set``, the latter is checking if a variable's
``name`` is found in either the ``'columns'`` or ``'masks'`` collection. For
our example data, the variables are:

>>> dataset.variables()

So a test for the ``array`` ``'q5'`` should be positive:

>>> dataset.var_exists('q5')
True

In addition to ``Quantipy``\'s complex logic operators, the ``DataSet`` class
offers some quick case data operations for code existence tests. To return a
``pandas.Series`` of all empty rows inside a variable use ``is_nan()`` as per:

>>> dataset.is_nan('q8').head()
0    True
1    True
2    True
3    True
4    True
Name: q8, dtype: bool

Which we can also use to quickly check the number of missing cases...

>>> dataset.is_nan('q8').value_counts()
True     5888
False    2367
Name: q8, dtype: int64

... as well as use the result as slicer for the ``DataSet`` case data component,
e.g. to show the non-empty rows:

>>> slicer = dataset.is_nan('q8')
>>> dataset[~slicer, 'q8'].head()
Name: q8, dtype: int64
7       5;
11      5;
13    1;4;
14    4;5;
23    1;4;
Name: q8, dtype: object

Especially useful for ``delimited set`` and ``array`` data, the ``code_count()``
method is creating the ``pandas.Series`` of response values found. If applied on
an ``array``, the result is expressed across all source item variables:

>>> dataset.code_count('q6').value_counts()
3    5100
2    3155
dtype: int64

... which means that not all cases contain answers in all three of the array's items.

With some basic ``pandas`` we can double-check this result:

>>> pd.concat([dataset['q6'], dataset.code_count('q6')], axis=1).head()
   q6_1  q6_2  q6_3  0
0     1   1.0     1  3
1     1   NaN     1  2
2     1   NaN     2  2
3     2   NaN     2  2
4     2  10.0    10  3

``code_count()`` can optionally ignore certain codes via the ``count_only`` and
``count_not`` parameters:

>>> q2_count = dataset.code_count('q2', count_only=[1, 2, 3])
>>> pd.concat([dataset['q2'], q2_count], axis=1).head()
         q2  0
0  1;2;3;5;  3
1      3;6;  1
2       NaN  0
3       NaN  0
4       NaN  0


Similarly, the ``any()`` and ``all()`` methods yield slicers for cases obeying
the condition that at least one / all of the provided codes are found in the
response. Again, for ``array`` variables the conditions are extended across all
the items:

>>> dataset[dataset.all('q6', 5), 'q6']
      q6_1  q6_2  q6_3
374      5   5.0     5
2363     5   5.0     5
2377     5   5.0     5
4217     5   5.0     5
5530     5   5.0     5
5779     5   5.0     5
5804     5   5.0     5
6328     5   5.0     5
6774     5   5.0     5
7269     5   5.0     5
8148     5   5.0     5

>>> dataset[dataset.all('q8', [1, 2, 3, 4, 96]), 'q8']
845     1;2;3;4;5;96;
6242      1;2;3;4;96;
7321      1;2;3;4;96;
Name: q8, dtype: object


>>> dataset[dataset.any('q8', [1, 2, 3, 4, 96]), 'q8'].head()
13      1;4;
14      4;5;
23      1;4;
24    1;3;4;
25      1;4;
Name: q8, dtype: object

--------------
Variable types
--------------

To get a summary of the all variables grouped by type, call ``by_type()`` on
the ``DataSet``:

>>> ds.by_type()
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
the ``types`` parameter:

>>> ds.by_type(types='delimited set')
size: 8255 delimited set
0                     q2
1                     q3
2                     q8
3                     q9

>>> ds.by_type(types=['delimited set', 'float'])
size: 8255 delimited set     float
0                     q2    weight
1                     q3  weight_a
2                     q8  weight_b
3                     q9       NaN

In addition to that, ``DataSet`` implements the following methods
that return the corresponding variables as a ``list`` for easy iteration::

	DataSet.singles
	       .delimied_sets()
	       .ints()
	       .floats()
	       .dates()
	       .strings()
	       .masks()
	       .columns()
	       .sets()

>>> ds.delimited_sets()
[u'q3', u'q2', u'q9', u'q8']

>>> for delimited_set in ds.delimited_sets():
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