.. toctree::
 	:maxdepth: 5
	:includehidden:

====================
Custom data recoding
====================

---------------------------------
The ``recode()`` method in detail
---------------------------------
This function takes a mapper of ``{key: logic}`` entries and injects the
key into the target column where its paired logic is True. The logic
may be arbitrarily complex and may refer to any other variable or
variables in data. Where a pre-existing column has been used to
start the recode, the injected values can replace or be appended to
any data found there to begin with. Note that this function does
not edit the target column, it returns a recoded copy of the target
column. The recoded data will always comply with the column type
indicated for the target column according to the meta.


:method: ``recode(target, mapper, default=None, append=False,
                  intersect=None, initialize=None, fillna=None, inplace=True)``


``target``
----------

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

By default, recoded data resulting from the the mapper will replace any
data already sitting in the target column (on a cell-by-cell basis).

``mapper``
----------

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

``default``
-----------

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

``append``
---------

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

``intersect``
------------

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

``initialize``
--------------

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

``fillna``
----------

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

--------
Examples
--------

# 1
---
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

# 2
---
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

# 3
---
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

# 4
---
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
