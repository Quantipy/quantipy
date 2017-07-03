-----------------------
Logic and set operaters
-----------------------

.. toctree::
	:maxdepth: 5
	:includehidden:

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