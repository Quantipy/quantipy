.. toctree::
  :maxdepth: 5
  :includehidden:

===========================
Data preparation / recoding
===========================

| :ref:`genindex`
| :ref:`modindex`

""""

Tools for managing your data
----------------------------
Quantipy provides a number of convenience functions for working with 
your data. Many of these take advantage Quantipy variable metadata 
and as such can manage, for example, the technical differences between
single- and multiple-choice variables for you.

All these functions detailed here can be imported using the following
statement:

>>> from quantipy.core.tools.dp.prep import(
...     frange,
...     recode,
...     crosstab,
...     frequency,
...     get_index_mapper
... ) 

``frange``
----------

This function takes a string of abbreviated ranges, possibly delimited
by a comma (or some other character) and extrapolates its full, 
unabbreviated list of ints.

:import: ``from quantipy.core.tools.dp.prep import frange``

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

This function takes a mapper of {key: logic} entries and injects the
key into the target column where its paired logic is True. The logic
may be arbitrarily complex and may refer to any other variable or 
variables in data. Where a pre-existing column has been used to 
start the recode, the injected values can replace or be appended to 
any data found there to begin with. Note that this function does
not edit the target column, it returns a recoded copy of the target
column. The recoded data will always comply with the column type
indicated for the target column according to the meta.

:import: ``from quantipy.core.tools.dp.prep import recode``

Signature/Docstring
"""""""""""""""""""

>>> def recode(meta, data, target, mapper, append=True, default=None):
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
...     append : bool, default=False
...         Should the new recodd data be appended to items already found
...         in series? If False, data from series (where found) will
...         overwrite whatever was found for that item in ds1 instead.
...     default : str
...         The column name to default to in cases where unattended lists
...         are given in your logic, where an auto-transformation of 
...         {key: list} to {key: {default: list}} is provided. Note that
...         lists in logical statements are themselves a form of shorthand
...         and this will ultimately be interpreted as:
...         {key: {default: has_any(list)}}.
...     
...     Returns
...     -------
...     series : pandas.Series
...         The series in which the recoded data will be stored and 
...         returned.
...     """


Basic usage
"""""""""""

Some basic usage examples to get us started.

Return a recoded version of the column ``radio_stations_xb`` edited
based on the given mapper:

>>> recoded = recode(
...     meta, data, 
...     target='radio_stations_xb', 
...     mapper=mapper
... )

If the target column is a delimited set (as defined by the meta), then
the recoded data resulting from the the mapper will be appended to the
target column, rather than replace it. 

However, if you do not want the recoded data appended to whatever may
already be in the target column, then you should use the append 
parameter:

>>> recoded = recode(
...     meta, data, 
...     target='radio_stations_xb', 
...     mapper=mapper,
...     append=False
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
found. With the default append=True behaviour we will return the 
following:

>>> target = 'radio_stations_xb'
>>> recode(meta, data, target, mapper, append=True)
1    6;7;9;13;901;
2              97;
3              97;
4    13;16;18;901;
5         2;6;901;
Name: radio_stations_xb, dtype: object

However, if we instead use append=False, we will return the following:

>>> target = 'radio_stations_xb'
>>> recode(meta, data, target, mapper, append=False)
1    901;
2     97;
3     97;
4    901;
5    901;
Name: radio_stations_xb, dtype: object

Now that you have the basics, what does a mapper look like?

A mapper is a dict of {value: logic} entries where value represents
the data that will be generated for cases where the logic is True.

Here's a simplified example of what a mapper looks like:

>>> mapper = {
...     1: logic_A,
...     2: logic_B,
...     3: logic_C,
... }

1 will be generated where logic_A is True, 2 where logic_B is True and
3 where logic_C is True.

The recode function, by referencing the type indicated by the meta,
will manage the complications involved in single vs delimited set 
data. In fact, the recode function can be used to recode any type
of data including int, float, text, string or date!

Mapper logic is exactly the same as what you would normally use when
gnerating net views. Here's the mapper for the 901 recode mentioned
earlier.

>>> mapper = {
...     901: {'radio_stations': frange('1-13')}
... }

This means: generate 901 if the column 'radio_stations' has any of the
values 1-13.

If you had lots of values to generate from the same reference column
(say most/all of them were based on 'radio_stations') then we can
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
'radio_stations'. In this case the three codes 901, 902 and 903 will
be generated based on the data found in 'radio_stations'.

You can combine this with reference to other columns, but you can only
provide one default column.

recoded = recode(
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
further down.

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

Example 2
"""""""""

Here's an example where the value 1 is generated based on some logic
and then all remaining cases are given the value 2 using the
pandas.Series.fillna() method.

Create the new metadata

>>> meta['columns']['QIA_age_xb'] = {
...     'type': 'single',
...     'text': {'en-GB': 'Age'},
...     'values': [
...         {'value': 1, 'text': {'en-GB': '16-25'}},
...         {'value': 2, 'text': {'en-GB': 'Others'}}
...     ]
...     }

Initialize the new column:

>>> data['QIA_age_xb'] = np.NaN

Recode the new column:

>>> data['QIA_age_xb'] = recode(
...     meta, data, 
...     target='QIA_age_xb', 
...     mapper={
...         1: {'age': frange('16-25')}
...     }
... )

Fill all cases that are still empty with the value 2:

>>> data['QIA_age_xb'].fillna(2, inplace=True)

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

Example 3
"""""""""

Here's an example of using a complicated, nested series of logic
statements to recode an obscure segmentation.

The segemenation was given with the following definition from the
researcher:

**1 - Self-directed:**

If q1_1 in [1,2] and q1_2 in [1,2] and q1_3 in [3,4,5]

**2 - Validators:**

If q1_1 in [1,2] and q1_2 in [1,2] and q1_3 in [1,2]

**3 - Delegators:**

If (q1_1 in [3,4,5] and q1_2 in [3,4,5] and q1_3 in [1,2]) 
Or (q1_1 in [3,4,5] and q1_2 in [1,2] and q1_3 in [1,2]) 
Or (q1_1 in [1,2] and q1_2 in [3,4,5] and q1_3 in [1,2])

**4 - Avoiders:**  

If (q1_1 in [3,4,5] and q1_2 in [3,4,5] and q1_3 in [3,4,5]) 
Or (q1_1 in [3,4,5] and q1_2 in [1,2] and q1_3 in [3,4,5]) 
Or (q1_1 in [1,2] and q1_2 in [3,4,5] and q1_3 in [3,4,5])

**5 - Others:**  

Everyone else.

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

Initialize the new column?L

>>> data['segments'] = np.NaN

Create the mapper separately, since it's pretty massive!

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

Fill all cases that are still empty with the value 5:

>>> data['segments'].fillna(5, inplace=True)

