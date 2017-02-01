.. toctree::
  :maxdepth: 5
  :includehidden:

================
Data preparation
================

| :ref:`genindex`
| :ref:`modindex`

""""

Quantipy provides a number of convenience functions for working with 
your data. Many of these take advantage of Quantipy variable metadata 
and as such can manage, for example, the technical differences between
single- and multiple-choice variables for you.

All of the functions detailed in this article can all be imported using 
the following statements:

>>> from quantipy.core.tools.dp.prep import(
...     frange,
...     recode,
...     crosstab,
...     frequency,
...     get_index_mapper
... ) 

>>> from quantipy.core.tools.view.logic import (
...     has_any, not_any, has_count,
...     has_all, not_all, not_count,
...     is_lt, is_eq, is_gt,
...     is_le, is_ne, is_ge,
...     union, intersection
... )

Data management
===============

``frange``
----------

This function takes a string of abbreviated ranges, possibly delimited
by a comma (or some other character) and extrapolates its full, 
unabbreviated list of ints.

Signature/Docstring
"""""""""""""""""""

>>> def frange(range_def, sep=','):
...     """
...     Return the full, unabbreviated list of ints suggested by range_def. 
...     
...     This function takes a string of abbreviated ranges, possibly
...     delimited by a comma (or some other character) and extrapolates
...     its full, unabbreviated list of ints.
...     
...     Parameters
...     ----------
...     range_def : str
...         The range string to be listed in full. 
...     sep : str, default=','
...         The character that should be used to delimit discrete entries in
...         range_def.
...         
...     Returns
...     -------
...     res : list
...         The exploded list of ints indicated by range_def.
...     """

Basic range
"""""""""""
>>> frange('1-5')
[1, 2, 3, 4, 5]

Range in reverse
""""""""""""""""
>>> frange('15-11')
[15, 14, 13, 12, 11]

Combination 
"""""""""""
>>> frange('1-5,7,9,15-11')
[1, 2, 3, 4, 5, 7, 9, 15, 14, 13, 12, 11]

May include spaces for clarity
""""""""""""""""""""""""""""""
>>> frange('1-5, 7, 9, 15-11')
[1, 2, 3, 4, 5, 7, 9, 15, 14, 13, 12, 11]


``recode``
----------

This function takes a mapper of ``{key: logic}`` entries and injects the
key into the target column where its paired logic is True. The logic
may be arbitrarily complex and may refer to any other variable or 
variables in data. Where a pre-existing column has been used to 
start the recode, the injected values can replace or be appended to 
any data found there to begin with. Note that this function does
not edit the target column, it returns a recoded copy of the target
column. The recoded data will always comply with the column type
indicated for the target column according to the meta.

Signature/Docstring
"""""""""""""""""""

>>> def recode(meta, data, target, mapper, default=None, append=False,
...            intersect=None, initialize=None, fillna=None):
...     """
...     Return a recoded copy of the target column using the given mapper.
...     
...     This function takes a mapper of {key: logic} entries and injects the
...     key into the target column where its paired logic is True. The logic
...     may be arbitrarily complex and may refer to any other variable or 
...     variables in data. Where a pre-existing column has been used to 
...     start the recode, the injected values can replace or be appended to 
...     any data found there to begin with. Note that this function does
...     not edit the target column, it returns a recoded copy of the target
...     column. The recoded data will always comply with the column type
...     indicated for the target column according to the meta.
...     
...     Parameters
...     ----------
...     meta : dict
...         Quantipy meta document.    
...     data : pandas.DataFrame
...         Data accompanying the given meta document. 
...     target : str
...         The column name that is the target of the recode. If target
...         is not found in meta['columns'] this will fail with an error.
...         If target is not found in data.columns the recode will start
...         from an empty series with the same index as data. If target
...         is found in data.columns the recode will start from a copy
...         of that column.
...     mapper : dict
...         A mapper of {key: logic} entries.
...     default : str
...         The column name to default to in cases where unattended lists
...         are given in your logic, where an auto-transformation of 
...         {key: list} to {key: {default: list}} is provided. Note that
...         lists in logical statements are themselves a form of shorthand
...         and this will ultimately be interpreted as:
...         {key: {default: has_any(list)}}.
...     append : bool, default=False
...         Should the new recodd data be appended to items already found
...         in series? If False, data from series (where found) will
...         overwrite whatever was found for that item in ds1 instead.
...     intersect : logical statement, default=None
...         If a logical statement is given here then it will be used as an
...         implied intersection of all logical conditions given in the
...         mapper.
...     initialize : str or np.NaN, default=None
...         If not None, a copy of the data named column will be used to
...         populate the target column before the recode is performed.
...         Alternatively, initialize can be used to populate the target
...         column with np.NaNs (overwriting whatever may be there) prior
...         to the recode.
...     fillna : int, default=None
...         If not None, the value passed to fillna will be used on the
...         recoded series as per pandas.Series.fillna().
...     
...     Returns
...     -------
...     series : pandas.Series
...         The series in which the recoded data will be stored and 
...         returned.
...     """


target
""""""

``target`` controls which column meta should be used to control the 
result of the recode operation. This is important because you cannot 
recode multiple responses into a 'single'-typed column.

The ``target`` column **must** already exist in meta.

The ``recode`` function is effectively a request to return a copy of
the ``target`` column, recoded as instructed. ``recode`` does not
edit the ``target`` column in place, it returns a recoded copy of it.

If the ``target`` column does not already exist in ``data`` then a new
series, named accordingly and initialized with ``np.NaN``, will begin
the recode. 

Return a recoded version of the column ``radio_stations_xb`` edited
based on the given mapper:

>>> recoded = recode(
...     meta, data, 
...     target='radio_stations_xb', 
...     mapper=mapper
... )

By dfault, recoded data resulting from the the mapper will replace any 
data already sitting in the target column (on a cell-by-cell basis).

mapper
""""""

A mapper is a dict of ``{value: logic}`` entries where value represents
the data that will be injected for cases where the logic is True.

Here's a simplified example of what a mapper looks like:

>>> mapper = {
...     1: logic_A,
...     2: logic_B,
...     3: logic_C,
... }

1 will be generated where ``logic_A`` is ``True``, 2 where ``logic_B`` is 
``True`` and 3 where ``logic_C`` is ``True``.

The recode function, by referencing the type indicated by the meta,
will manage the complications involved in single vs delimited set 
data.

>>> mapper = {
...     901: {'radio_stations': frange('1-13')},
...     902: {'radio_stations': frange('14-20')},
...     903: {'radio_stations': frange('21-25')}
... }

This means: inject 901 if the column ``radio_stations`` has any of the
values 1-13, 902 where ``radio_stations`` has any of the values 14-20
and 903 where ``radio_stations`` has any of the values 21-25.

default
"""""""

If you had lots of values to generate from the same reference column
(say most/all of them were based on ``radio_stations``) then we can
omit the wildcard logic format and use recode's default parameter.

>>> recoded = recode(
...     meta, data, 
...     target='radio_stations_xb', 
...     mapper={
...         901: frange('1-13'),
...         902: frange('14-20'),
...         903: frange('21-25')
...     },
...     default='radio_stations'
... )

This means, all unkeyed logic will default to be keyed to 
``radio_stations``. In this case the three codes 901, 902 and 903 will
be generated based on the data found in ``radio_stations``.

You can combine this with reference to other columns, but you can only
provide one default column.

>>> recoded = recode(
...     meta, data, 
...     target='radio_stations_xb', 
...     mapper={
...         901: frange('1-13'),
...         902: frange('14-20'),
...         903: frange('21-25'),
...         904: {'age': frange('18-34')}
...     },
...     default='radio_stations'
... )

Given that logic can be arbitrarily complicated, mappers can be as
well. You'll see an example of a mapper that recodes a segmentation
in **Example 4**, below.

append
""""""

If you want the recoded data to be appended to whatever may
already be in the target column (this is only applicable for 'delimited 
set'-typed columns), then you should use the append parameter.

>>> recoded = recode(
...     meta, data, 
...     target='radio_stations_xb', 
...     mapper=mapper,
...     append=True
... )

The precise behaviour of the append parameter can be seen in the 
following examples.

Given the following data:

>>> df['radio_stations_xb']
1    6;7;9;13;
2          97;
3          97;
4    13;16;18;
5         2;6;
Name: radio_stations_xb, dtype: object

We generate a recoded value of 901 if any of the values 1-13 are 
found. With the default ``append=False`` behaviour we will return the 
following:

>>> target = 'radio_stations_xb'
>>> recode(meta, data, target, mapper)
1    901;
2     97;
3     97;
4    901;
5    901;
Name: radio_stations_xb, dtype: object

However, if we instead use ``append=True``, we will return the following:

>>> target = 'radio_stations_xb'
>>> recode(meta, data, target, mapper, append=True)
1    6;7;9;13;901;
2              97;
3              97;
4    13;16;18;901;
5         2;6;901;
Name: radio_stations_xb, dtype: object

intersect
"""""""""

One way to help simplify complex logical conditions, especially when
they are in some way repetitive, is to use ``intersect``, which
accepts any logical statement and forces every condition in the mapper
to become the intersection of both it and the intersect condition.

For example, we could limit our recode to males by giving a logical
condition to that effect to ``intersect``:

>>> recoded = recode(
...     meta, data, 
...     target='radio_stations_xb', 
...     mapper={
...         901: frange('1-13'),
...         902: frange('14-20'),
...         903: frange('21-25'),
...         904: {'age': frange('18-34')}
...     },
...     default='radio_stations',
...     intersect={'gender': [1]}
... )

initialize
""""""""""

You may also ``initialize`` your copy of the target column as part of your 
recode operation. You can ``initalize`` with either np.NaN (to overwrite
anything that may already be there when your recode begins) or by naming
another column. When you name another column a copy of the data from that
column is used to initialize your recode. 

Initialization occurs **before** your recode.

>>> recoded = recode(
...     meta, data, 
...     target='radio_stations_xb', 
...     mapper={
...         901: frange('1-13'),
...         902: frange('14-20'),
...         903: frange('21-25'),
...         904: {'age': frange('18-34')}
...     },
...     default='radio_stations',
...     initialize=np.NaN
... )

>>> recoded = recode(
...     meta, data, 
...     target='radio_stations_xb', 
...     mapper={
...         901: frange('1-13'),
...         902: frange('14-20'),
...         903: frange('21-25'),
...         904: {'age': frange('18-34')}
...     },
...     default='radio_stations',
...     initialize='radio_stations'
... )

fillna
""""""

You may also provide a ``fillna`` value that will be used as per
``pd.Series.fillna()`` **after** the recode has been performed.

>>> recoded = recode(
...     meta, data, 
...     target='radio_stations_xb', 
...     mapper={
...         901: frange('1-13'),
...         902: frange('14-20'),
...         903: frange('21-25'),
...         904: {'age': frange('18-34')}
...     },
...     default='radio_stations',
...     initialize=np.NaN,
...     fillna=99
... )

Example 1
"""""""""

Here's an example of copying an existing question and recoding onto it a 
net code.

Create the new metadata:

>>> meta['columns']['radio_stations_xb'] = copy.copy(
...     meta['columns']['radio_stations']
... )
>>> meta['columns']['radio_stations_xb']['values'].append(
...     {
...         "value": 901, 
...         "text": {"en-GB": "NET: Listened to radio in past 30 days"}
...     }
... )

Initialize the new column. In this case we're starting with a copy of
the ``radio_stations`` column:

>>> data['radio_stations_xb'] = data['radio_stations'].copy()

Recode the new column by appending the code 901 to it as indicated 
by the mapper:

>>> data['radio_stations_xb'] = recode(
...     meta, data,
...     target='radio_stations_xb',
...     mapper={
...         901: {'radio_stations': frange('1-23, 92, 94, 141')}
...     }, 
...     append=True
... )

Check the result:

>>> data[['radio_stations', 'radio_stations_xb']].head(20)
   radio_stations radio_stations_cb
0              5;            5;901;
1             97;               97;
2             97;               97;
3             97;               97;
4             97;               97;
5              4;            4;901;
6             11;           11;901;
7              4;            4;901;
8             97;               97;
9             97;               97;
10            97;               97;
11            92;           92;901;
12            97;               97;
13       1;13;17;      1;13;17;901;
14             6;            6;901;
15      1;5;6;10;     1;5;6;10;901;
16             6;            6;901;
17        2;4;16;       2;4;16;901;
18          6;10;         6;10;901;
19             6;            6;901;

Example 2
"""""""""

Here's an example where the value 1 is generated based on some logic
and then all remaining cases are given the value 2 using the
pandas.Series.fillna() method.

Create the new metadata

>>> meta['columns']['age_xb'] = {
...     'type': 'single',
...     'text': {'en-GB': 'Age'},
...     'values': [
...         {'value': 1, 'text': {'en-GB': '16-25'}},
...         {'value': 2, 'text': {'en-GB': 'Others'}}
...     ]
...     }

Initialize the new column:

>>> data['age_xb'] = np.NaN

Recode the new column:

>>> data['age_xb'] = recode(
...     meta, data, 
...     target='age_xb', 
...     mapper={
...         1: {'age': frange('16-40')}
...     }
... )

Fill all cases that are still empty with the value 2:

>>> data['age_xb'].fillna(2, inplace=True)

Check the result:

>>> data[['age', 'age_xb']].head(20)
    age  age_grp_rc
0    22           1
1    68           2
2    32           1
3    44           2
4    33           1
5    52           2
6    54           2
7    44           2
8    62           2
9    49           2
10   64           2
11   73           2
12   43           2
13   28           1
14   66           2
15   39           1
16   51           2
17   50           2
18   77           2
19   42           2

Example 3
"""""""""

Here's a typical example of recoding age into custom bands.

In this case we're using list comprehension to generate the first ten
values objects and then concatenate that with a final '65+' value object 
which doesn't folow the same label format.

Create the new metadata:

>>> meta['columns']['age_xb_1'] = {
...     'type': 'single',
...     'text': {'en-GB': 'Age'},
...     'values': [
...         {
...             'value': i, 
...             'text': {'en-GB': '{}-{}'.format(r[0], r[1])}
...         }
...         for i, r in enumerate(
...             [
...                 [18, 20], 
...                 [21, 25], [26, 30], 
...                 [31, 35], [36, 40], 
...                 [41, 45], [46, 50], 
...                 [51, 55], [56, 60], 
...                 [61, 65] 
...             ], 
...             start=1
...         )
...     ] + [
...         {
...             'value': 11, 
...             'text': {'en-GB': '65+'}
...         }
...     ]
... }

Initialize the new column:

>>> data['age_xb_1'] = np.NaN

Recode the new column:

>>> data['age_xb_1'] = recode(
...     meta, data, 
...     target='age_xb_1', 
...     mapper={
...         1: frange('18-20'),
...         2: frange('21-25'),
...         3: frange('26-30'),
...         4: frange('31-35'),
...         5: frange('36-40'),
...         6: frange('41-45'),
...         7: frange('46-50'),
...         8: frange('51-55'),
...         9: frange('56-60'),
...         10: frange('61-65'),
...         11: frange('66-99')
...     }, 
...     default='age'
... )

Check the result:

>>> data[['age', 'age_xb_1']].head(20)
    age  age_cb
0    22       2
1    68      11
2    32       4
3    44       6
4    33       4
5    52       8
6    54       8
7    44       6
8    62      10
9    49       7
10   64      10
11   73      11
12   43       6
13   28       3
14   66      11
15   39       5
16   51       8
17   50       7
18   77      11
19   42       6

Example 4
"""""""""

Here's an example of using a complicated, nested series of logic
statements to recode an obscure segmentation.

The segemenation was given with the following definition:

**1 - Self-directed:**

- If q1_1 in [1,2] and q1_2 in [1,2] and q1_3 in [3,4,5]

**2 - Validators:**

- If q1_1 in [1,2] and q1_2 in [1,2] and q1_3 in [1,2]

**3 - Delegators:**

- If (q1_1 in [3,4,5] and q1_2 in [3,4,5] and q1_3 in [1,2]) 
- Or (q1_1 in [3,4,5] and q1_2 in [1,2] and q1_3 in [1,2]) 
- Or (q1_1 in [1,2] and q1_2 in [3,4,5] and q1_3 in [1,2])

**4 - Avoiders:**  

- If (q1_1 in [3,4,5] and q1_2 in [3,4,5] and q1_3 in [3,4,5]) 
- Or (q1_1 in [3,4,5] and q1_2 in [1,2] and q1_3 in [3,4,5]) 
- Or (q1_1 in [1,2] and q1_2 in [3,4,5] and q1_3 in [3,4,5])

**5 - Others:**  

- Everyone else.

Create the new metadata:

>>> meta['columns']['segments'] = {
...     'type': 'single',
...     'text': {'en-GB': 'Segments'},
...     'values': [
...         {'value': 1, 'text': {'en-GB': 'Self-directed'}},
...         {'value': 2, 'text': {'en-GB': 'Validators'}},
...         {'value': 3, 'text': {'en-GB': 'Delegators'}},
...         {'value': 4, 'text': {'en-GB': 'Avoiders'}},
...         {'value': 5, 'text': {'en-GB': 'Other'}},
...     ]
... }

Initialize the new column?

>>> data['segments'] = np.NaN

Create the mapper separately, since it's pretty massive!

See the **Complex logic** section for more information and examples
related to the use of ``union`` and ``intersection``.

>>> mapper = {
...     1: intersection([
...         {"q1_1": [1, 2]},
...         {"q1_2": [1, 2]},
...         {"q1_3": [3, 4, 5]}
...     ]),
...     2: intersection([
...         {"q1_1": [1, 2]},
...         {"q1_2": [1, 2]},
...         {"q1_3": [1, 2]}
...     ]),
...     3: union([
...         intersection([
...             {"q1_1": [3, 4, 5]},
...             {"q1_2": [3, 4, 5]},
...             {"q1_3": [1, 2]}
...         ]),
...         intersection([
...             {"q1_1": [3, 4, 5]},
...             {"q1_2": [1, 2]},
...             {"q1_3": [1, 2]}
...         ]),
...         intersection([
...             {"q1_1": [1, 2]},
...             {"q1_2": [3, 4, 5]},
...             {"q1_3": [1, 2]}
...         ]),
...     ]),
...     4: union([
...         intersection([
...             {"q1_1": [3, 4, 5]},
...             {"q1_2": [3, 4, 5]},
...             {"q1_3": [3, 4, 5]}
...         ]),
...         intersection([
...             {"q1_1": [3, 4, 5]},
...             {"q1_2": [1, 2]},
...             {"q1_3": [3, 4, 5]}
...         ]),
...         intersection([
...             {"q1_1": [1, 2]},
...             {"q1_2": [3, 4, 5]},
...             {"q1_3": [3, 4, 5]}
...         ])
...     ])
... }

Recode the new column:

>>> data['segments'] = recode(
...     meta, data, 
...     target='segments', 
...     mapper=mapper
... )

.. note::
  Anything not at the top level of the mapper will not benefit from using
  the ``default`` parameter of the recode function. In this case, for example,
  saying ``default='q1_1'`` would not have helped. Everything in a nested level
  of the mapper, including anything in a ``union`` or ``intersection`` list,
  must use the explicit dict form ``{"q1_1": [1, 2]}``.

Fill all cases that are still empty with the value 5:

>>> data['segments'].fillna(5, inplace=True)

Check the result:

>>> data[['q1_1', 'q1_2', 'q1_3', 'segments']].head(20)
    q1_1  q1_2  q1_3  segments
0      3     3     3         4
1      3     3     3         4
2      1     1     3         1
3      1     1     2         2
4      2     2     2         2
5      1     1     5         1
6      2     3     2         3
7      2     2     3         1
8      1     1     4         1
9      3     3     3         4
10     3     3     4         4
11     2     2     4         1
12     1     1     5         1
13     2     2     4         1
14     1     1     1         2
15     2     2     4         1
16     2     2     3         1
17     1     1     5         1
18     5     5     1         3
19     1     1     4         1

``get_index_mapper``
--------------------

This function converts a mapper in the form
``{key: logic}`` to a mapper in the form ``{key: index}``, where index is
the index needed to slice from data those cases for which logic
was True. This is very useful when checking complicated logic
statements, because it gives you access to the cases that result
from any logical statement.

Signature/Docstring
"""""""""""""""""""

>>> def get_index_mapper(meta, data, mapper, default=None):
...     """
...     Convert a {value: logic} map to a {value: index} map.
...     
...     This function takes a mapper of {key: logic} entries and resolves
...     the logic statements using the given meta/data to return a mapper
...     of {key: index}. The indexes returned can be used on data to isolate
...     the cases described by arbitrarily complex logical statements.
...     
...     Parameters
...     ----------
...     meta : dict
...         Quantipy meta document.
...     data : pandas.DataFrame
...         Data accompanying the given meta document.       
...     mapper : dict
...         A mapper of {key: logic}
...     default : str
...         The column name to default to in cases where unattended lists
...         are given as logic, where an auto-transformation of {key: list}
...         to {key: {default: list}} is provided.
...     
...     Returns
...     -------
...     index_mapper : dict
...         A mapper of {key: index}
...     """

Basic usage
"""""""""""

The segmentation mapper that was generated in **Example 4** of the article
explaining the ``recode`` function is an example of a set of conditions
that are too complicated to check with a simple crosstab. In these
cases there's no way around looking at the data selected by the
logical condition and interrogating it directly to confirm that the
statement has returned what you expect it to.

Assuming the segmentation mapper from that example, you can return
the index mapper like this:

>>> index_mapper = get_index_mapper(meta, data, mapper)

Which returns the following:

>>> index_mapper
... {
...   1: Int64Index([   1,    5,    7,   13,   14,   17,   18,   24,   25,   30, 
...          ...
...          2087, 2088, 2090, 2091, 2093, 2094, 2098, 2103, 2104, 2107],
...          dtype='int64', length=1167),
...   2: Int64Index([   6,   19,   22,   37,   38,   48,   64,   67,   69,   71, 
...          ...
...          2073, 2075, 2077, 2096, 2101, 2102, 2106, 2108, 2109, 2110],
...          dtype='int64', length=261),
...   3: Int64Index([  12,   40,   52,   75,   80,   87,   89,  118,  132,  133, 
...          ...
...          1888, 1919, 1924, 1940, 1943, 1962, 1973, 1983, 2025, 2052],
...          dtype='int64', length=119),
...   4: Int64Index([   2,    3,    4,    8,    9,   10,   11,   15,   16,   20, 
...          ...
...          2082, 2085, 2089, 2092, 2095, 2097, 2099, 2100, 2105, 2111],
...          dtype='int64', length=564)
... }

Inspecting the data
"""""""""""""""""""

This can then be used with the pandas.DataFrame.loc[...] indexer in the 
following way:

>>> data[[q1_1, q1_2, q1_3]].loc[index_mapper[1]]
      q1_1    q1_2    q1_3
1        2       2       3
5        2       1       3
7        2       2       4
13       1       2       3
14       2       2       4
17       2       1       4
18       1       2       4
24       2       1       3
25       1       1       3
30       1       1       3
32       2       2       3
36       1       1       3
39       1       1       5
41       2       2       3
42       1       1       4
45       2       2       4
46       1       1       5
47       2       2       4
49       2       2       4
50       2       2       3
51       1       1       5
53       1       1       4
55       2       1       3
57       1       1       4
59       1       1       4
60       1       1       4
62       2       2       3
65       1       1       3
66       2       2       3
73       1       1       3
...                              ...                             ...                             ...

Here we're able to verify that the index mapper is returning a slicer
compatible with the logical statement indicated earlier.

Complex logic
=============

We saw in ``recode`` **Example 4** how multiple conditions can be 
combined using ``union`` or ``intersection``. As demonstrated by that
example, recode mappers can be arbitrarily nested as long as they are 
well-formed.

``union``
---------

``union`` takes a list of logical conditions that will be treated with 
**or** logic.

Where **any** of logic_A, logic_B **or** logic_C are ``True``:

>>> union([logic_A, logic_B, logic_C])

``intersection``
----------------

``intersection`` takes a list of conditions that will be 
treated with  **and** logic.

Where **all** of logic_A, logic_B **and** logic_C are ``True``:

>>> intersection([logic_A, logic_B, logic_C])

"list logic"
------------

All of the value-conditions we've seen so far have used an implied *or* logic.

For example ``{"q1_1": [1, 2]}`` is an example of list-logic, where ``[1, 2]``
will be interpreted as ``has_any([1, 2])``, meaning if **q1_1** has any of the 
values **1** or **2**.

``q1_1`` has any of the responses 1, 2 or 3:

>>> {"q1_1": [1, 2, 3]}

``has_any``
-----------

``q1_1`` has any of the responses 1, 2 or 3:

>>> {"q1_1": has_any([1, 2, 3])}

``q1_1`` has any of the responses 1, 2 or 3 and no others:

>>> {"q1_1": has_any([1, 2, 3], exclusive=True)}


``not_any``
-----------

``q1_1`` doesn't have any of the responses 1, 2 or 3:

>>> {"q1_1": not_any([1, 2, 3])}

``q1_1`` doesn't have any of the responses 1, 2 or 3 but has some others:

>>> {"q1_1": not_any([1, 2, 3], exclusive=True)}

``has_all``
-----------

``q1_1`` has all of the responses 1, 2 and 3:

>>> {"q1_1": has_all([1, 2, 3])}

``q1_1`` has all of the responses 1, 2 and 3 and no others:

>>> {"q1_1": has_all([1, 2, 3], exclusive=True)}
 
``not_all``
-----------

``q1_1`` doesn't have all of the responses 1, 2 and 3:

>>> {"q1_1": not_all([1, 2, 3])}

``q1_1`` doesn't have all of the responses 1, 2 and 3 but has some others:

>>> {"q1_1": not_all([1, 2, 3], exclusive=True)}

``has_count``
-------------

``q1_1`` has exactly 2 responses:

>>> {"q1_1": has_count(2)}

``q1_1`` has 1, 2 or 3 responses:

>>> {"q1_1": has_count([1, 3])}

``q1_1`` has 1 or more responses:

>>> {"q1_1": has_count([is_ge(1)])}

``q1_1`` has 1, 2 or 3 responses from the response group 5, 6, 7, 8 or 9:

>>> {"q1_1": has_count([1, 3, [5, 6, 7, 8, 9]])}

``q1_1`` has 1 or more responses from the response group 5, 6, 7, 8 or 9:

>>> {"q1_1": has_count([is_ge(1), [5, 6, 7, 8, 9]])}

``not_count``
-------------

``q1_1`` doesn't have exactly 2 responses:

>>> {"q1_1": not_count(2)}

``q1_1`` doesn't have 1, 2 or 3 responses:

>>> {"q1_1": not_count([1, 3])}

``q1_1`` doesn't have 1 or more responses:

>>> {"q1_1": not_count([is_ge(1)])}

``q1_1`` doesn't have 1, 2 or 3 responses from the response group 5, 6, 7, 8 or 9: 

>>> {"q1_1": not_count([1, 3, [5, 6, 7, 8, 9]])}

``q1_1`` doesn't have 1 or more responses from the response group 5, 6, 7, 8 or 9:

>>> {"q1_1": not_count([is_ge(1), [5, 6, 7, 8, 9]])}

Instant aggregations
====================

``crosstab``
------------

This function uses the given meta and data to create a 
type-appropriate cross-tabulation (pivot table) of the named x and y
variables. The result may be either counts or column percentages, 
weighted or unweighted.

Signature/Docstring
"""""""""""""""""""

>>> def crosstab(meta, data, x, y, get='count', decimals=1):
...     """
...     Return a type-appropriate crosstab of x and y.
...     
...     This function uses the given meta and data to create a 
...     type-appropriate cross-tabulation (pivot table) of the named x and y
...     variables. The result may be either counts or column percentages, 
...     weighted or unweighted.
...     
...     Parameters
...     ----------
...     meta : dict
...         Quantipy meta document.    
...     data : pandas.DataFrame
...         Data accompanying the given meta document. 
...     x : str
...         The variable that should be placed into the x-position.
...     y : str
...         The variable that should be placed into the y-position.
...     get : str, default='count'
...         Control the type of data that is returned. 'count' will return
...         absolute counts and 'normalize' will return column percentages.
...     decimals : int, default=1
...         Control the number of decimals in the returned dataframe.
...     weight : str, default=None
...         The name of the weight variable that should be used on the data,
...         if any.
...     
...     Returns
...     -------
...     df : pandas.DataFrame
...         The crosstab as a pandas DataFrame.
...     """

Setup
"""""

Assuming the following recode:

>>> meta['columns']['qqincome_xb'] = copy.copy(meta['columns']['qincome'])
>>> meta['columns']['qincome_xb']['type'] = 'delimited set'
>>> meta['columns']['qincome_xb']['values'].extend([
...     {"value": 901, "text": {"en-GB": "Under £20k (Income group 1)"}},
...     {"value": 902, "text": {"en-GB": "£20,000 to £50,000 (Income group 2)"}},
...     {"value": 903, "text": {"en-GB": "£50,001 + (Income group 3)"}}
... ])
>>> data['qincome_xb'] = np.NaN
>>> data['qincome_xb'] = recode(
...     meta, data, 
...     target='qincome_xb', 
...     mapper={
...         901: [1],
...         902: frange('2-7'),
...         903: [8, 9, 10]
...     }, 
...     default='qincome', 
...     append=True
... )

Counts
""""""

You would normally generate a checking pivot between 'qincome' and
your newly recoded 'qincome_xb'.

>>> crosstab(
...     meta, data, 
...     x='qincome', 
...     y='qincome_xb'
... )
Question           qincome_xb                                                                        
Values                        All    1    2    3    4    5    6   7   8   9   10   11   12  901  902  903
Question    Values                                                                                       
qincome     All              2111  427  206  189  146  122  104  92  69  60  232  111  353  427  859  361
            1                 427  427    0    0    0    0    0   0   0   0    0    0    0  427    0    0
            2                 206    0  206    0    0    0    0   0   0   0    0    0    0    0  206    0
            3                 189    0    0  189    0    0    0   0   0   0    0    0    0    0  189    0
            4                 146    0    0    0  146    0    0   0   0   0    0    0    0    0  146    0
            5                 122    0    0    0    0  122    0   0   0   0    0    0    0    0  122    0
            6                 104    0    0    0    0    0  104   0   0   0    0    0    0    0  104    0
            7                  92    0    0    0    0    0    0  92   0   0    0    0    0    0   92    0
            8                  69    0    0    0    0    0    0   0  69   0    0    0    0    0    0   69
            9                  60    0    0    0    0    0    0   0   0  60    0    0    0    0    0   60
            10                232    0    0    0    0    0    0   0   0   0  232    0    0    0    0  232
            11                111    0    0    0    0    0    0   0   0   0    0  111    0    0    0    0
            12                353    0    0    0    0    0    0   0   0   0    0    0  353    0    0    0

Percentages
"""""""""""

You can also request a normalized pivot (using column percentages)
in the following way: 

>>> crosstab(
...     meta, data, 
...     x='qincome', 
...     y='qincome_xb', 
...     get='normalize'
... )
Question           qincome_xb                                                                               
Values                        All    1    2    3    4    5    6    7    8    9   10   11   12  901    902    903
Question    Values                                                                                              
qincome     All             100.0  100  100  100  100  100  100  100  100  100  100  100  100  100  100.0  100.0
            1                20.2  100    0    0    0    0    0    0    0    0    0    0    0  100    0.0    0.0
            2                 9.8    0  100    0    0    0    0    0    0    0    0    0    0    0   24.0    0.0
            3                 9.0    0    0  100    0    0    0    0    0    0    0    0    0    0   22.0    0.0
            4                 6.9    0    0    0  100    0    0    0    0    0    0    0    0    0   17.0    0.0
            5                 5.8    0    0    0    0  100    0    0    0    0    0    0    0    0   14.2    0.0
            6                 4.9    0    0    0    0    0  100    0    0    0    0    0    0    0   12.1    0.0
            7                 4.4    0    0    0    0    0    0  100    0    0    0    0    0    0   10.7    0.0
            8                 3.3    0    0    0    0    0    0    0  100    0    0    0    0    0    0.0   19.1
            9                 2.8    0    0    0    0    0    0    0    0  100    0    0    0    0    0.0   16.6
            10               11.0    0    0    0    0    0    0    0    0    0  100    0    0    0    0.0   64.3
            11                5.3    0    0    0    0    0    0    0    0    0    0  100    0    0    0.0    0.0
            12               16.7    0    0    0    0    0    0    0    0    0    0    0  100    0    0.0    0.0

Controlling decimals
""""""""""""""""""""

And you can control the number of decimals that return to you (the
default being one) using the decimals parameter:

>>> crosstab(
...     meta, data, 
...     x='qincome', 
...     y='qincome_xb', 
...     get='normalize', 
...     decimals=3
... )
Question           qincome_xb                                                                                   
Values                        All    1    2    3    4    5    6    7    8    9   10   11   12  901      902      903
Question    Values                                                                                                  
qincome     All           100.000  100  100  100  100  100  100  100  100  100  100  100  100  100  100.000  100.000
            1              20.227  100    0    0    0    0    0    0    0    0    0    0    0  100    0.000    0.000
            2               9.758    0  100    0    0    0    0    0    0    0    0    0    0    0   23.981    0.000
            3               8.953    0    0  100    0    0    0    0    0    0    0    0    0    0   22.002    0.000
            4               6.916    0    0    0  100    0    0    0    0    0    0    0    0    0   16.997    0.000
            5               5.779    0    0    0    0  100    0    0    0    0    0    0    0    0   14.203    0.000
            6               4.927    0    0    0    0    0  100    0    0    0    0    0    0    0   12.107    0.000
            7               4.358    0    0    0    0    0    0  100    0    0    0    0    0    0   10.710    0.000
            8               3.269    0    0    0    0    0    0    0  100    0    0    0    0    0    0.000   19.114
            9               2.842    0    0    0    0    0    0    0    0  100    0    0    0    0    0.000   16.620
            10             10.990    0    0    0    0    0    0    0    0    0  100    0    0    0    0.000   64.266
            11              5.258    0    0    0    0    0    0    0    0    0    0  100    0    0    0.000    0.000
            12             16.722    0    0    0    0    0    0    0    0    0    0    0  100    0    0.000    0.000

Weighted results
""""""""""""""""

You can also return any of these variations weighted by naming a 
weight variable with the weight parameter:

>>> crosstab(
...     meta, data, 
...     x='qincome', 
...     y='qincome_xb', 
...     get='count', 
...     decimals=2, 
...     weight='weights_UK18'
... )
Question           qincome_xb                                                                                                               
Values                        All      1       2    3       4       5       6      7      8      9      10      11     12    901     902     903
Question    Values                                                                                                                              
qincome     All           2078.00  448.3  214.24  189  147.55  121.93  107.93  85.48  56.81  53.31  209.15  104.41  339.9  448.3  866.13  319.27
            1              448.30  448.3    0.00    0    0.00    0.00    0.00   0.00   0.00   0.00    0.00    0.00    0.0  448.3    0.00    0.00
            2              214.24    0.0  214.24    0    0.00    0.00    0.00   0.00   0.00   0.00    0.00    0.00    0.0    0.0  214.24    0.00
            3              189.00    0.0    0.00  189    0.00    0.00    0.00   0.00   0.00   0.00    0.00    0.00    0.0    0.0  189.00    0.00
            4              147.55    0.0    0.00    0  147.55    0.00    0.00   0.00   0.00   0.00    0.00    0.00    0.0    0.0  147.55    0.00
            5              121.93    0.0    0.00    0    0.00  121.93    0.00   0.00   0.00   0.00    0.00    0.00    0.0    0.0  121.93    0.00
            6              107.93    0.0    0.00    0    0.00    0.00  107.93   0.00   0.00   0.00    0.00    0.00    0.0    0.0  107.93    0.00
            7               85.48    0.0    0.00    0    0.00    0.00    0.00  85.48   0.00   0.00    0.00    0.00    0.0    0.0   85.48    0.00
            8               56.81    0.0    0.00    0    0.00    0.00    0.00   0.00  56.81   0.00    0.00    0.00    0.0    0.0    0.00   56.81
            9               53.31    0.0    0.00    0    0.00    0.00    0.00   0.00   0.00  53.31    0.00    0.00    0.0    0.0    0.00   53.31
            10             209.15    0.0    0.00    0    0.00    0.00    0.00   0.00   0.00   0.00  209.15    0.00    0.0    0.0    0.00  209.15
            11             104.41    0.0    0.00    0    0.00    0.00    0.00   0.00   0.00   0.00    0.00  104.41    0.0    0.0    0.00    0.00
            12             339.90    0.0    0.00    0    0.00    0.00    0.00   0.00   0.00   0.00    0.00    0.00  339.9    0.0    0.00    0.00

``frequency``
-------------

This function uses the given meta and data to create a 
type-appropriate frequency table of the named x variable.
The result may be either counts or column percentages, weighted 
or unweighted.

Signature/Docstring
"""""""""""""""""""

>>> def frequency(meta, data, x, **kwargs):
...     """
...     Return a type-appropriate frequency of x.
...     
...     This function uses the given meta and data to create a 
...     type-appropriate frequency table of the named x variable.
...     The result may be either counts or column percentages, weighted 
...     or unweighted.
...     
...     Parameters
...     ----------
...     meta : dict
...         Quantipy meta document.    
...     data : pandas.DataFrame
...         Data accompanying the given meta document. 
...     x : str
...         The variable that should be placed into the x-position.
...     kwargs : kwargs
...         All remaining keyword arguments will be passed along to the
...         crosstab function.
...     
...     Returns
...     -------
...     f : pandas.DataFrame
...         The frequency as a pandas DataFrame.
...     """

Setup
"""""

Internally the frequency function automates the crosstab function,
so based on the same recode example given in the crosstab article.

Assuming the following recode:

>>> meta['columns']['qincome_xb'] = copy.copy(meta['columns']['qincome'])
>>> meta['columns']['qincome_xb']['type'] = 'delimited set'
>>> meta['columns']['qincome_xb']['values'].extend([
...     {"value": 901, "text": {"en-GB": "Under £20k (Income group 1)"}},
...     {"value": 902, "text": {"en-GB": "£20,000 to £50,000 (Income group 2)"}},
...     {"value": 903, "text": {"en-GB": "£50,001 + (Income group 3)"}}
... ])
>>> data['qincome_xb'] = np.NaN
>>> data['qincome_xb'] = recode(
...     meta, data, 
...     target='qincome_xb', 
...     mapper={
...         901: [1],
...         902: frange('2-7'),
...         903: [8, 9, 10]
...     }, 
...     default='qincome', 
...     append=True
... )

Counts
""""""

>>> frequency(meta, data, 'qincome_xb')
Question              qincome_xb
Values                             @
Question       Values               
qincome_xb     All              2111
               1                 427
               2                 206
               3                 189
               4                 146
               5                 122
               6                 104
               7                  92
               8                  69
               9                  60
               10                232
               11                111
               12                353
               901               427
               902               859
               903               361

Percentages
"""""""""""

>>> frequency(meta, data, 'qincome_xb', get='normalize')               
Question              qincome_xb
Values                             @
Question       Values               
qincome_xb     All             100.0
               1                20.2
               2                 9.8
               3                 9.0
               4                 6.9
               5                 5.8
               6                 4.9
               7                 4.4
               8                 3.3
               9                 2.8
               10               11.0
               11                5.3
               12               16.7
               901              20.2
               902              40.7
               903              17.1

Controlling decimals
""""""""""""""""""""

>>> frequency(meta, data, 'qincome_xb', get='normalize', decimals=3)
Question              qincome_xb
Values                             @
Question       Values               
qincome_xb     All           100.000
               1              20.227
               2               9.758
               3               8.953
               4               6.916
               5               5.779
               6               4.927
               7               4.358
               8               3.269
               9               2.842
               10             10.990
               11              5.258
               12             16.722
               901            20.227
               902            40.692
               903            17.101

Weighted results
""""""""""""""""

>>> frequency(meta, data, 'qincome_xb', decimals=2, weight='weights_UK18')
Question              qincome_xb
Values                             @
Question       Values               
qincome_xb     All           2078.00
               1              448.30
               2              214.24
               3              189.00
               4              147.55
               5              121.93
               6              107.93
               7               85.48
               8               56.81
               9               53.31
               10             209.15
               11             104.41
               12             339.90
               901            448.30
               902            866.13
               903            319.27

Rules
=====

Rules are ways by which you can control how an aggregation is returned
to you. There are three ways in which this can be done:

- Selecting values to be retained and in what order
- Sorting values along one of their vectors
- Selecting values to be dropped

These rules can be applied to either the ``x`` of a ``frequency`` or the 
``x`` and/or ``y`` of a crosstab.

Structure
---------

``rules`` is an optional object you can provide to any column object
in your meta. The ``rules`` object is a dict with ``x`` and ``y`` keys 
for axis-specific application. 

Below the ``x`` and ``y`` keys in the ``rules`` object can be 
instructions under the following keys:

- ``slicex``: Selecting values to be retained and in what order
- ``sortx``: Sorting values along one of their vectors
- ``dropx``: Selecting values to be dropped

Internally these three rule objects results in a call to functions of 
the same name. The contents of each becomes that function call's kwargs.

.. note::
  Despite the ``x`` in ``slicex``, ``sortx`` and ``dropx``, rules can
  be applied to either the ``x`` or ``y`` axes of an aggregation. The
  naming convention refers to the fact that these operations are only
  performed on the ``x`` axis of their receieved ``dataframe``. However,
  internally, when a ``y`` rule is being applied, the ``dataframe`` is
  simply transposed for the duration of the rule's application.

``slicex``
------------

Assuming a Quantipy-style view result this function takes an index
slice of df as indicated by values and returns the result.

Signature/Docstring
"""""""""""""""""""

>>> def slicex(df, values, keep_margins=True):
...     """
...     Return an index-wise slice of df, keeping margins if desired.
...     
...     Assuming a Quantipy-style view result this function takes an index
...     slice of df as indicated by values and returns the result.
...     
...     Parameters
...     ----------
...     df : pandas.DataFrame
...         The dataframe that should be sliced along the index.
...     values : list-like
...         A list of index values that should be sliced from df.
...     keep_margins : bool, default=True
...         If True and the margins index row exists, it will be kept. 
...     
...     Returns
...     -------
...     df : list
...         The sliced dataframe. 
...     """

Given the following ``crosstab``:

>>> crosstab(meta, data, 'qincome', 'qincome_cb')
Question               qincome_cb                                                                        
Values                        All    1    2    3    4    5    6   7   8   9   10   11   12  901  902  903
Question    Values                                                                                       
qincome     All              2078  425  204  187  146  119  103  89  68  59  226  101  351  425  848  353
            1                 425  425    0    0    0    0    0   0   0   0    0    0    0  425    0    0
            2                 204    0  204    0    0    0    0   0   0   0    0    0    0    0  204    0
            3                 187    0    0  187    0    0    0   0   0   0    0    0    0    0  187    0
            4                 146    0    0    0  146    0    0   0   0   0    0    0    0    0  146    0
            5                 119    0    0    0    0  119    0   0   0   0    0    0    0    0  119    0
            6                 103    0    0    0    0    0  103   0   0   0    0    0    0    0  103    0
            7                  89    0    0    0    0    0    0  89   0   0    0    0    0    0   89    0
            8                  68    0    0    0    0    0    0   0  68   0    0    0    0    0    0   68
            9                  59    0    0    0    0    0    0   0   0  59    0    0    0    0    0   59
            10                226    0    0    0    0    0    0   0   0   0  226    0    0    0    0  226
            11                101    0    0    0    0    0    0   0   0   0    0  101    0    0    0    0
            12                351    0    0    0    0    0    0   0   0   0    0    0  351    0    0    0

Set up the ``slicex`` ``rules``:

>>> meta['columns']['qincome']['rules'] = {
...     'x': {'slicex': {'values': [1, 3, 5, 7, 9, 11]}}
... }

>>> meta['columns']['qincome_cb']['rules'] = {
...     'y': {'slicex': {'values': [12, 10, 4, 6, 2, 901, 902, 903]}}
... }

Apply the ``rules`` using ``crosstab``:

>>> crosstab(meta, data, 'qincome', 'qincome_cb', rules=True)
Question               qincome_cb                                        
Values                        All   12   10    4    6    2  901  902  903
Question    Values                                                       
qincome     All              2078  351  226  146  103  204  425  848  353
            1                 425    0    0    0    0    0  425    0    0
            3                 187    0    0    0    0    0    0  187    0
            5                 119    0    0    0    0    0    0  119    0
            7                  89    0    0    0    0    0    0   89    0
            9                  59    0    0    0    0    0    0    0   59
            11                101    0    0    0    0    0    0    0    0

>>> crosstab(meta, data, 'qincome', 'qincome_cb', rules=['x'])
Question               qincome_cb                                                                        
Values                        All    1    2    3    4    5    6   7   8   9   10   11   12  901  902  903
Question    Values                                                                                       
qincome     All              2078  425  204  187  146  119  103  89  68  59  226  101  351  425  848  353
            1                 425  425    0    0    0    0    0   0   0   0    0    0    0  425    0    0
            3                 187    0    0  187    0    0    0   0   0   0    0    0    0    0  187    0
            5                 119    0    0    0    0  119    0   0   0   0    0    0    0    0  119    0
            7                  89    0    0    0    0    0    0  89   0   0    0    0    0    0   89    0
            9                  59    0    0    0    0    0    0   0   0  59    0    0    0    0    0   59
            11                101    0    0    0    0    0    0   0   0   0    0  101    0    0    0    0

>>> crosstab(meta, data, 'qincome', 'qincome_cb', rules=['y'])
Question               qincome_cb                                        
Values                        All   12   10    4    6    2  901  902  903
Question    Values                                                       
qincome     All              2078  351  226  146  103  204  425  848  353
            1                 425    0    0    0    0    0  425    0    0
            2                 204    0    0    0    0  204    0  204    0
            3                 187    0    0    0    0    0    0  187    0
            4                 146    0    0  146    0    0    0  146    0
            5                 119    0    0    0    0    0    0  119    0
            6                 103    0    0    0  103    0    0  103    0
            7                  89    0    0    0    0    0    0   89    0
            8                  68    0    0    0    0    0    0    0   68
            9                  59    0    0    0    0    0    0    0   59
            10                226    0  226    0    0    0    0    0  226
            11                101    0    0    0    0    0    0    0    0
            12                351  351    0    0    0    0    0    0    0

``sortx``
---------

This function sorts df, which is assumed to be a Quantipy-style
view result with appropriate index/column structure, using
a given column, while maintaining the position of margins if
they exist, and also optionally fixing certain values at the
bottom of the result without sorting them. Note that nested
variable view results are not yet supported.

Signature/Docstring
"""""""""""""""""""

>>> def sortx(df, sort_on='@', ascending=False, fixed=None, with_weight='auto'):
...     """
...     Sort the index of df on a column, keeping margins and fixing values.
...     
...     This function sorts df, which is assumed to be a Quantipy-style
...     view result with appropriate index/column structure, using
...     a given column, while maintaining the position of margins if
...     they exist, and also optionally fixing certain values at the
...     bottom of the result without sorting them. Note that nested
...     variable view results are not yet supported.
...     
...     Parameters
...     ----------
...     df : pandas.DataFrame
...         The Quantipy-style view result to be sorted
...     sort_on : str or int, default='@'
...         The column (on the innermost level of the column's
...         MultiIndex) on which to sort. By default sorting will be
...         based on the unfiltered frequency of the x variable. No
...         other sorting targets are currently supported.
...     ascending : bool, default=False
...         Sort ascending vs. descending. Default descending for
...         easier application to MR use cases.
...     fixed : list-like, default=None
...         A list of index values that should appear underneath
...         the sorted index values.
...     with_weight : None or str, default='auto'
...         If not 'auto' this is name of the weight that is being used for
...         the sort. 'auto' means that the same weight used in the original
...         computation is also used in the sort, but this argument provides
...         the ability to sort a computation done with one weight (or None)
...         on the results of another weight (or None).
...     
...     Returns
...     -------
...     df : pandas.DataFrame
...         The sorted df. 
...     """

Given the following ``crosstab``:

>>> crosstab(meta, data, 'qincome', 'qincome_cb')
Question               qincome_cb                                                                        
Values                        All    1    2    3    4    5    6   7   8   9   10   11   12  901  902  903
Question    Values                                                                                       
qincome     All              2078  425  204  187  146  119  103  89  68  59  226  101  351  425  848  353
            1                 425  425    0    0    0    0    0   0   0   0    0    0    0  425    0    0
            2                 204    0  204    0    0    0    0   0   0   0    0    0    0    0  204    0
            3                 187    0    0  187    0    0    0   0   0   0    0    0    0    0  187    0
            4                 146    0    0    0  146    0    0   0   0   0    0    0    0    0  146    0
            5                 119    0    0    0    0  119    0   0   0   0    0    0    0    0  119    0
            6                 103    0    0    0    0    0  103   0   0   0    0    0    0    0  103    0
            7                  89    0    0    0    0    0    0  89   0   0    0    0    0    0   89    0
            8                  68    0    0    0    0    0    0   0  68   0    0    0    0    0    0   68
            9                  59    0    0    0    0    0    0   0   0  59    0    0    0    0    0   59
            10                226    0    0    0    0    0    0   0   0   0  226    0    0    0    0  226
            11                101    0    0    0    0    0    0   0   0   0    0  101    0    0    0    0
            12                351    0    0    0    0    0    0   0   0   0    0    0  351    0    0    0

Set up the ``sortx`` ``rules``:

>>> meta['columns']['AMA_qqincome']['rules'] = {
...     'x': {'sortx': {'fixed': [1, 12, 10, 11]}}
... }

>>> meta['columns']['AMA_qqincome_cb']['rules'] = {
...     'y': {'sortx': {'fixed': [901, 902, 903], 'ascending': True}}
... }

Apply the ``rules`` using ``crosstab``:

>>> crosstab(meta, data, 'qincome', 'qincome_cb', rules=True)
Question               qincome_cb                                                                        
Values                        All   9   8   7   11    6    5    4    3    2   10   12    1  901  902  903
Question    Values                                                                                       
qincome     All              2078  59  68  89  101  103  119  146  187  204  226  351  425  425  848  353
            2                 204   0   0   0    0    0    0    0    0  204    0    0    0    0  204    0
            3                 187   0   0   0    0    0    0    0  187    0    0    0    0    0  187    0
            4                 146   0   0   0    0    0    0  146    0    0    0    0    0    0  146    0
            5                 119   0   0   0    0    0  119    0    0    0    0    0    0    0  119    0
            6                 103   0   0   0    0  103    0    0    0    0    0    0    0    0  103    0
            7                  89   0   0  89    0    0    0    0    0    0    0    0    0    0   89    0
            8                  68   0  68   0    0    0    0    0    0    0    0    0    0    0    0   68
            9                  59  59   0   0    0    0    0    0    0    0    0    0    0    0    0   59
            1                 425   0   0   0    0    0    0    0    0    0    0    0  425  425    0    0
            12                351   0   0   0    0    0    0    0    0    0    0  351    0    0    0    0
            10                226   0   0   0    0    0    0    0    0    0  226    0    0    0    0  226
            11                101   0   0   0  101    0    0    0    0    0    0    0    0    0    0    0

>>> crosstab(meta, data, 'qincome', 'qincome_cb', rules=['x'])
Question               qincome_cb                                                                        
Values                        All    1    2    3    4    5    6   7   8   9   10   11   12  901  902  903
Question    Values                                                                                       
qincome     All              2078  425  204  187  146  119  103  89  68  59  226  101  351  425  848  353
            2                 204    0  204    0    0    0    0   0   0   0    0    0    0    0  204    0
            3                 187    0    0  187    0    0    0   0   0   0    0    0    0    0  187    0
            4                 146    0    0    0  146    0    0   0   0   0    0    0    0    0  146    0
            5                 119    0    0    0    0  119    0   0   0   0    0    0    0    0  119    0
            6                 103    0    0    0    0    0  103   0   0   0    0    0    0    0  103    0
            7                  89    0    0    0    0    0    0  89   0   0    0    0    0    0   89    0
            8                  68    0    0    0    0    0    0   0  68   0    0    0    0    0    0   68
            9                  59    0    0    0    0    0    0   0   0  59    0    0    0    0    0   59
            1                 425  425    0    0    0    0    0   0   0   0    0    0    0  425    0    0
            12                351    0    0    0    0    0    0   0   0   0    0    0  351    0    0    0
            10                226    0    0    0    0    0    0   0   0   0  226    0    0    0    0  226
            11                101    0    0    0    0    0    0   0   0   0    0  101    0    0    0    0

>>> crosstab(meta, data, 'qincome', 'qincome_cb', rules=['y'])
Question               qincome_cb                                                                        
Values                        All   9   8   7   11    6    5    4    3    2   10   12    1  901  902  903
Question    Values                                                                                       
qincome     All              2078  59  68  89  101  103  119  146  187  204  226  351  425  425  848  353
            1                 425   0   0   0    0    0    0    0    0    0    0    0  425  425    0    0
            2                 204   0   0   0    0    0    0    0    0  204    0    0    0    0  204    0
            3                 187   0   0   0    0    0    0    0  187    0    0    0    0    0  187    0
            4                 146   0   0   0    0    0    0  146    0    0    0    0    0    0  146    0
            5                 119   0   0   0    0    0  119    0    0    0    0    0    0    0  119    0
            6                 103   0   0   0    0  103    0    0    0    0    0    0    0    0  103    0
            7                  89   0   0  89    0    0    0    0    0    0    0    0    0    0   89    0
            8                  68   0  68   0    0    0    0    0    0    0    0    0    0    0    0   68
            9                  59  59   0   0    0    0    0    0    0    0    0    0    0    0    0   59
            10                226   0   0   0    0    0    0    0    0    0  226    0    0    0    0  226
            11                101   0   0   0  101    0    0    0    0    0    0    0    0    0    0    0
            12                351   0   0   0    0    0    0    0    0    0    0  351    0    0    0    0

.. note::
  When generating chains it may be necessary to sometimes sort views computed
  using one weight variable (or none at all) by the results of the same column
  weighted with another weight variable (or none at all). An example of this
  would be when you're showing both weighted and unweighted column bases and
  you need to apply column-wise sorting to your chains. In this case you will
  need to make sure that you use the ``with_weight`` parameter when sorting
  to make sure that ALL sorted results will be sorted exactly the same way,
  regardless of their individual results based on their weight.

``dropx``
---------

Assuming a Quantipy-style view result this function drops index
rows indicated by values and returns the result.

Signature/Docstring
"""""""""""""""""""

>>> def dropx(df, values):
...     """
...     Return df after dropping values from the index.
...     
...     Assuming a Quantipy-style view result this function drops index
...     rows indicated by values and returns the result.
...     
...     Parameters
...     ----------
...     df : pandas.DataFrame
...         The dataframe that should have some index rows dropped.
...     values : list-like
...         A list of index values that should be dropped from the index.
...     
...     Returns
...     -------
...     df : list
...         The edited dataframe. 
...     """

Given the following ``crosstab``:

>>> crosstab(meta, data, 'qincome', 'qincome_cb')
Question               qincome_cb                                                                        
Values                        All    1    2    3    4    5    6   7   8   9   10   11   12  901  902  903
Question    Values                                                                                       
qincome     All              2078  425  204  187  146  119  103  89  68  59  226  101  351  425  848  353
            1                 425  425    0    0    0    0    0   0   0   0    0    0    0  425    0    0
            2                 204    0  204    0    0    0    0   0   0   0    0    0    0    0  204    0
            3                 187    0    0  187    0    0    0   0   0   0    0    0    0    0  187    0
            4                 146    0    0    0  146    0    0   0   0   0    0    0    0    0  146    0
            5                 119    0    0    0    0  119    0   0   0   0    0    0    0    0  119    0
            6                 103    0    0    0    0    0  103   0   0   0    0    0    0    0  103    0
            7                  89    0    0    0    0    0    0  89   0   0    0    0    0    0   89    0
            8                  68    0    0    0    0    0    0   0  68   0    0    0    0    0    0   68
            9                  59    0    0    0    0    0    0   0   0  59    0    0    0    0    0   59
            10                226    0    0    0    0    0    0   0   0   0  226    0    0    0    0  226
            11                101    0    0    0    0    0    0   0   0   0    0  101    0    0    0    0
            12                351    0    0    0    0    0    0   0   0   0    0    0  351    0    0    0

Set up the ``dropx`` ``rules``:

>>> meta['columns']['AMA_qqincome']['rules'] = {
...     'x': {'dropx': {'values': [1, 3, 5, 7, 9, 11]}}
... }

>>> meta['columns']['AMA_qqincome_cb']['rules'] = {
...     'y': {'dropx': {'values': [12, 10, 4, 6, 2, 901, 902, 903]}}
... }

Apply the ``rules`` using ``crosstab``:

>>> crosstab(meta, data, 'qincome', 'qincome_cb', rules=True)
Question               qincome_cb                                
Values                        All    1    3    5   7   8   9   11
Question    Values                                               
qincome     All              2078  425  187  119  89  68  59  101
            2                 204    0    0    0   0   0   0    0
            4                 146    0    0    0   0   0   0    0
            6                 103    0    0    0   0   0   0    0
            8                  68    0    0    0   0  68   0    0
            10                226    0    0    0   0   0   0    0
            12                351    0    0    0   0   0   0    0

>>> crosstab(meta, data, 'qincome', 'qincome_cb', rules=['x'])
Question               qincome_cb                                                                        
Values                        All    1    2    3    4    5    6   7   8   9   10   11   12  901  902  903
Question    Values                                                                                       
qincome     All              2078  425  204  187  146  119  103  89  68  59  226  101  351  425  848  353
            2                 204    0  204    0    0    0    0   0   0   0    0    0    0    0  204    0
            4                 146    0    0    0  146    0    0   0   0   0    0    0    0    0  146    0
            6                 103    0    0    0    0    0  103   0   0   0    0    0    0    0  103    0
            8                  68    0    0    0    0    0    0   0  68   0    0    0    0    0    0   68
            10                226    0    0    0    0    0    0   0   0   0  226    0    0    0    0  226
            12                351    0    0    0    0    0    0   0   0   0    0    0  351    0    0    0

>>> crosstab(meta, data, 'qincome', 'qincome_cb', rules=['y'])
Question               qincome_cb                                
Values                        All    1    3    5   7   8   9   11
Question    Values                                               
qincome     All              2078  425  187  119  89  68  59  101
            1                 425  425    0    0   0   0   0    0
            2                 204    0    0    0   0   0   0    0
            3                 187    0  187    0   0   0   0    0
            4                 146    0    0    0   0   0   0    0
            5                 119    0    0  119   0   0   0    0
            6                 103    0    0    0   0   0   0    0
            7                  89    0    0    0  89   0   0    0
            8                  68    0    0    0   0  68   0    0
            9                  59    0    0    0   0   0  59    0
            10                226    0    0    0   0   0   0    0
            11                101    0    0    0   0   0   0  101
            12                351    0    0    0   0   0   0    0


Combinations
------------

Any or all of these rules can be used in combination with each other.

Given the following ``crosstab``:

>>> crosstab(meta, data, 'qincome', 'qincome_cb')
Question               qincome_cb                                                                        
Values                        All    1    2    3    4    5    6   7   8   9   10   11   12  901  902  903
Question    Values                                                                                       
qincome     All              2078  425  204  187  146  119  103  89  68  59  226  101  351  425  848  353
            1                 425  425    0    0    0    0    0   0   0   0    0    0    0  425    0    0
            2                 204    0  204    0    0    0    0   0   0   0    0    0    0    0  204    0
            3                 187    0    0  187    0    0    0   0   0   0    0    0    0    0  187    0
            4                 146    0    0    0  146    0    0   0   0   0    0    0    0    0  146    0
            5                 119    0    0    0    0  119    0   0   0   0    0    0    0    0  119    0
            6                 103    0    0    0    0    0  103   0   0   0    0    0    0    0  103    0
            7                  89    0    0    0    0    0    0  89   0   0    0    0    0    0   89    0
            8                  68    0    0    0    0    0    0   0  68   0    0    0    0    0    0   68
            9                  59    0    0    0    0    0    0   0   0  59    0    0    0    0    0   59
            10                226    0    0    0    0    0    0   0   0   0  226    0    0    0    0  226
            11                101    0    0    0    0    0    0   0   0   0    0  101    0    0    0    0
            12                351    0    0    0    0    0    0   0   0   0    0    0  351    0    0    0

Set up the ``rules``:

>>> meta['columns']['AMA_qqincome']['rules'] = {
    'x': {
        'slicex': {'values': [1, 3, 5, 7, 9, 11]},
        'sortx': {'fixed': [7]},
        'dropx': {'values': [9, 11]}
    }
}

>>> meta['columns']['AMA_qqincome_cb']['rules'] = {
    'y': {
        'slicex': {'values': [12, 10, 4, 6, 2, 901, 902, 903]},
        'sortx': {'fixed': [901, 902, 903], 'ascending': True},
        'dropx': {'values': [10]}
    }
}

Apply the ``rules`` using ``crosstab``:

>>> crosstab(meta, data, 'qincome', 'qincome_cb', rules=True)
Question               qincome_cb                                   
Values                        All    6    4    2   12  901  902  903
Question    Values                                                  
qincome     All              2078  103  146  204  351  425  848  353
            1                 425    0    0    0    0  425    0    0
            3                 187    0    0    0    0    0  187    0
            5                 119    0    0    0    0    0  119    0
            7                  89    0    0    0    0    0   89    0

>>> crosstab(meta, data, 'qincome', 'qincome_cb', rules=['x'])
Question               qincome_cb                                                                        
Values                        All    1    2    3    4    5    6   7   8   9   10   11   12  901  902  903
Question    Values                                                                                       
qincome     All              2078  425  204  187  146  119  103  89  68  59  226  101  351  425  848  353
            1                 425  425    0    0    0    0    0   0   0   0    0    0    0  425    0    0
            3                 187    0    0  187    0    0    0   0   0   0    0    0    0    0  187    0
            5                 119    0    0    0    0  119    0   0   0   0    0    0    0    0  119    0
            7                  89    0    0    0    0    0    0  89   0   0    0    0    0    0   89    0

>>> crosstab(meta, data, 'qincome', 'qincome_cb', rules=['y'])
Question               qincome_cb                                   
Values                        All    6    4    2   12  901  902  903
Question    Values                                                  
qincome     All              2078  103  146  204  351  425  848  353
            1                 425    0    0    0    0  425    0    0
            2                 204    0    0  204    0    0  204    0
            3                 187    0    0    0    0    0  187    0
            4                 146    0  146    0    0    0  146    0
            5                 119    0    0    0    0    0  119    0
            6                 103  103    0    0    0    0  103    0
            7                  89    0    0    0    0    0   89    0
            8                  68    0    0    0    0    0    0   68
            9                  59    0    0    0    0    0    0   59
            10                226    0    0    0    0    0    0  226
            11                101    0    0    0    0    0    0    0
            12                351    0    0    0  351    0    0    0


