======================
Transforming variables
======================

.. toctree::
	:maxdepth: 5
	:includehidden:

-------
Copying
-------
It's often recommended to draw a clean copy of a variable before starting to
editing its meta or case data. With ``copy()`` you can add a copy to the
``DataSet`` that is identical to the original in all respects but its name. By
default, the copy's name will be suffixed with ``'_rec'``, but you can apply a
custom suffix by providing it via the ``suffix`` argument (leaving out the
``'_'`` which is added automatically):

>>> ds.copy('q3')
>>> ds.copy('q3', suffix='version2')

>>> ds.delimited_sets
[u'q3', u'q2', u'q9', u'q8', u'q3_rec', u'q3_version2']

Querying the ``DataSet``, we can see that all three version are looking identical:

>>> ds[['q3', 'q3_rec', 'q3_version2']].head()
       q3  q3_rec q3_version2
0  1;2;3;  1;2;3;      1;2;3;
1  1;2;3;  1;2;3;      1;2;3;
2  1;2;3;  1;2;3;      1;2;3;
3    1;3;    1;3;        1;3;
4      2;      2;          2;

We can, however, prevent copying the case data and simply add an "empty" copy
of the variable by passing ``copy_data=False``:

>>> ds.copy('q3', suffix='no_data', copy_data=False)

>>> ds[['q3', 'q3_rec', 'q3_version2', 'q3_no_data']].head()
       q3  q3_rec q3_version2  q3_no_data
0  1;2;3;  1;2;3;      1;2;3;         NaN
1  1;2;3;  1;2;3;      1;2;3;         NaN
2  1;2;3;  1;2;3;      1;2;3;         NaN
3    1;3;    1;3;        1;3;         NaN
4      2;      2;          2;         NaN

If we wanted to only copy a subset of the case data, we could also use a
:doc:`logical slicer <logic>` and supply it in the ``copy()`` operation's
``slicer`` parameter:

>>> slicer = {'gender': [1]}
>>> ds.copy('q3', suffix='only_men', copy_data=True, slicer=slicer)

>>> ds[['q3', 'gender', 'q3_only_men']].head()
       q3  gender q3_only_men
0  1;2;3;       1      1;2;3;
1  1;2;3;       2         NaN
2  1;2;3;       1      1;2;3;
3    1;3;       1        1;3;
4      2;       1          2;


-----------------------
Inplace type conversion
-----------------------
You can change the characteristics of existing ``DataSet`` variables by
converting from one ``type`` to another. Conversions happen ``inplace``, i.e.
no copy of the variable is taken prior to the operation. Therefore, you might
want to take a ``DataSet.copy()`` before using the ``convert(name, to)``
method.

Conversions need to modify both the ``meta`` and ``data`` component of the
``DataSet`` and are limited to transformations that keep the original and new
state of a variable consistent. The following conversions are currently
supported:

+--------------------------+-----------------+------------------------+--------------+----------------+-----------------+
| ``name`` (from-``type``) | ``to='single'`` | ``to='delimited set'`` | ``to='int'`` | ``to='float'`` | ``to='string'`` |
+--------------------------+-----------------+------------------------+--------------+----------------+-----------------+
| ``'single'``             |       [X]       |            X           |       X      |        X       |        X        |
+--------------------------+-----------------+------------------------+--------------+----------------+-----------------+
| ``'delimited set'``      |                 |           [X]          |              |                |                 |
+--------------------------+-----------------+------------------------+--------------+----------------+-----------------+
| ``'int'``                |        X        |                        |      [X]     |        X       |        X        |
+--------------------------+-----------------+------------------------+--------------+----------------+-----------------+
| ``'float'``              |                 |                        |              |       [X]      |        X        |
+--------------------------+-----------------+------------------------+--------------+----------------+-----------------+
| ``'string'``             |        X        |                        |       X*     |        X*      |       [X]       |
+--------------------------+-----------------+------------------------+--------------+----------------+-----------------+
| ``'date'``               |        X        |                        |              |                |        X        |
+--------------------------+-----------------+------------------------+--------------+----------------+-----------------+

\* *If all values of the variable are numerical, i.e.* ``DataSet.is_like_numeric()`` *returns* ``True``.

Each of these conversions will rebuild the variable meta data to match the ``to``
type. This means, that for instance a variable that is ``single`` will lose
its ``values`` object when transforming to ``int``, while the reverse operation
will create a ``values`` object that categorizes the unqiue numeric codes found in the
case data with their ``str`` representation as ``text`` meta. Consider the
variables ``q1`` (``single``) and ``age`` (``int``):

**From type** ``single`` **to** ``int``:

>>> ds.meta('q1')
single                                   codes                                   texts missing
q1: What is your main fitness activity?
1                                            1                                Swimming    None
2                                            2                         Running/jogging    None
3                                            3                         Lifting weights    None
4                                            4                                Aerobics    None
5                                            5                                    Yoga    None
6                                            6                                 Pilates    None
7                                            7                       Football (soccer)    None
8                                            8                              Basketball    None
9                                            9                                  Hockey    None
10                                          96                                   Other    None
11                                          98  I regularly change my fitness activity    None
12                                          99       Not applicable - I don't exercise    None

>>> ds.convert('q1', to='int')
>>> ds.meta('q1')
                                         int
q1: What is your main fitness activity?  N/A


**From type** ``int`` **to** ``single``:

>>> ds.meta('age')
          int
age: Age  N/A

>>> ds.convert('age', to='single')
>>> ds.meta('age')
single    codes texts missing
age: Age
1            19    19    None
2            20    20    None
3            21    21    None
4            22    22    None
5            23    23    None
6            24    24    None
7            25    25    None
8            26    26    None
9            27    27    None
10           28    28    None
11           29    29    None
12           30    30    None
13           31    31    None
14           32    32    None
15           33    33    None
16           34    34    None
17           35    35    None
18           36    36    None
19           37    37    None
20           38    38    None
21           39    39    None
22           40    40    None
23           41    41    None
24           42    42    None
25           43    43    None
26           44    44    None
27           45    45    None
28           46    46    None
29           47    47    None
30           48    48    None
31           49    49    None


--------------------------
Banding and categorization
--------------------------
In contrast to ``convert()``, the ``categorize()`` method creates a new
variable of type ``single``, acting as a short-hand for creating a renamed copy
and then type-transforming it. Therefore, it lets you quickly categorize
the unique values of a ``text``, ``int`` or ``date`` variable, storing
``values`` meta in the form of ``{'text': {'en-GB': str(1)}, 'value': 1}``.

>>>

Flexible banding of numeric data is provided thorugh ``DataSet.band()``: If a
variable is banded, it will standardly be added to the ``DataSet`` via the
original's name suffixed with ``'banded'``, e.g. ``'age_banded'``, keeping
the originating variables ``text`` label. The ``new_name`` and ``label``
parameters can be used to create custom variable names and labels. The banding
of the incoming data is controlled with the ``bands`` argument that expects a
list containing ``int``, ``tuples`` or ``dict``, where each type is used for a
different kind of group definition.

**Banding with** ``int`` **and** ``tuple``:

- Use an ``int`` to make a band of only one value
- Use a ``tuple`` to indicate (inclusive) group limits
- ``values`` ``text`` meta is infered
- Example: ``[0, (1, 10), (11, 14), 15, (16, 25)]``

**Banding with** ``dict``:

- The dict ``key`` will dicate the group's ``text`` label meta
- The dict ``value`` can pick up an ``int`` / ``tuple`` (see above)
- Example: ``[{'A': 0}, {'B': (1, 10)}, {'C': (11, 14)}, {'D': 15}, {'E': (16, 25)}]``
- Mixing allowed: ``[0, {'A': (1, 10)}, (11, 14), 15, {'B': (16, 25)}]``

For instance, we could band ``'age'`` into a new variable called ``'grouped_age'``
with ``bands`` being:

>>> bands = [{'Younger than 35': (19, 34)},
...          (35, 39),
...          {'Exactly 40': 40},
...          41,
...          (42, 60)]

>>> ds.band(name='age', bands=bands, new_name='grouped_age', label=None)

>>> ds.meta('grouped_age')
single            codes            texts missing
grouped_age: Age
1                     1  Younger than 35    None
2                     2            35-39    None
3                     3       Exactly 40    None
4                     4               41    None
5                     5            42-60    None

>>> ds.crosstab('age', 'grouped_age')
Question        grouped_age. Age
Values                       All Younger than 35 35-39 Exactly 40   41 42-60
Question Values
age. Age All                8255            4308  1295        281  261  2110
         19                  245             245     0          0    0     0
         20                  277             277     0          0    0     0
         21                  270             270     0          0    0     0
         22                  323             323     0          0    0     0
         23                  272             272     0          0    0     0
         24                  263             263     0          0    0     0
         25                  246             246     0          0    0     0
         26                  252             252     0          0    0     0
         27                  260             260     0          0    0     0
         28                  287             287     0          0    0     0
         29                  270             270     0          0    0     0
         30                  271             271     0          0    0     0
         31                  264             264     0          0    0     0
         32                  287             287     0          0    0     0
         33                  246             246     0          0    0     0
         34                  275             275     0          0    0     0
         35                  258               0   258          0    0     0
         36                  236               0   236          0    0     0
         37                  252               0   252          0    0     0
         38                  291               0   291          0    0     0
         39                  258               0   258          0    0     0
         40                  281               0     0        281    0     0
         41                  261               0     0          0  261     0
         42                  290               0     0          0    0   290
         43                  267               0     0          0    0   267
         44                  261               0     0          0    0   261
         45                  257               0     0          0    0   257
         46                  259               0     0          0    0   259
         47                  243               0     0          0    0   243
         48                  271               0     0          0    0   271
         49                  262               0     0          0    0   262

---------------------
Array transformations
---------------------

**Transposing arrays**

``DataSet`` offers tools to simplify common ``array`` variable operations.
You can switch the structure of ``items`` vs. ``values`` by producing the one
from the other using ``transpose()``. The transposition of an array will always
result in ``items`` that have the ``delimited set`` type in the corresponding
``'columns'`` metadata. That is because the transposed array is collecting
what former items have been assignd per former value:

>>> ds.transpose('q5')

*Original*

>>> ds['q5'].head()
   q5_1  q5_2  q5_3  q5_4  q5_5  q5_6
0     2     2     2     2     1     2
1     5     5     3     3     3     5
2     5    98     5     5     1     5
3     5     5     1     5     3     5
4    98    98    98    98    98    98


>>> ds.meta('q5')
single                                             items     item texts  codes                    texts missing
q5: How likely are you to do each of the follow...
1                                                   q5_1        Surfing      1  I would refuse if asked    None
2                                                   q5_2   Snowboarding      2            Very unlikely    None
3                                                   q5_3  Kite boarding      3        Probably wouldn't    None
4                                                   q5_4    Parachuting      4  Probably would if asked    None
5                                                   q5_5    Cave diving      5              Very likely    None
6                                                   q5_6    Windsurfing     97  I'm already planning to    None
7                                                                           98               Don't know    None

*Transposition*

>>> ds['q5_trans'].head()
  q5_trans_1  q5_trans_2 q5_trans_3 q5_trans_4 q5_trans_5 q5_trans_97   q5_trans_98
0         5;  1;2;3;4;6;        NaN        NaN        NaN         NaN           NaN
1        NaN         NaN     3;4;5;        NaN     1;2;6;         NaN           NaN
2         5;         NaN        NaN        NaN   1;3;4;6;         NaN            2;
3         3;         NaN         5;        NaN   1;2;4;6;         NaN           NaN
4        NaN         NaN        NaN        NaN        NaN         NaN  1;2;3;4;5;6;

>>> ds.meta('q5_trans')
delimited set                                             items               item texts codes          texts missing
q5_trans: How likely are you to do each of the ...
1                                                    q5_trans_1  I would refuse if asked     1        Surfing    None
2                                                    q5_trans_2            Very unlikely     2   Snowboarding    None
3                                                    q5_trans_3        Probably wouldn't     3  Kite boarding    None
4                                                    q5_trans_4  Probably would if asked     4    Parachuting    None
5                                                    q5_trans_5              Very likely     5    Cave diving    None
6                                                   q5_trans_97  I'm already planning to     6    Windsurfing    None
7                                                   q5_trans_98               Don't know

The method's ``ignore_items`` and ``ignore_values`` arguments can pick up
``items`` (indicated by their order number) and ``values`` to leave aside
during the transposition.

*Ignoring items*

The new ``values`` meta's numerical codes will always be enumerated from 1 to
the number of valid items for the transposition, so ignoring items 2, 3 and 4
will lead to:

>>> ds.transpose('q5', ignore_items=[2, 3, 4])

>>> ds['q5_trans'].head(1)
  q5_trans_1 q5_trans_2 q5_trans_3 q5_trans_4 q5_trans_5 q5_trans_97 q5_trans_98
0         2;       1;3;        NaN        NaN        NaN         NaN         NaN

>>> ds.values('q5_trans')
[(1, 'Surfing'), (2, 'Cave diving'), (3, 'Windsurfing')]

*Ignoring values*

>>> ds.transpose('q5', ignore_values=[1, 97])

>>> ds['q5_trans'].head(1)
   q5_trans_2 q5_trans_3 q5_trans_4 q5_trans_5 q5_trans_98
0  1;2;3;4;6;        NaN        NaN        NaN         NaN

>>> ds.items('q5_trans')
[('q5_trans_2', u'Very unlikely'),
 ('q5_trans_3', u"Probably wouldn't"),
 ('q5_trans_4', u'Probably would if asked'),
 ('q5_trans_5', u'Very likely'),
 ('q5_trans_98', u"Don't know")]

*Ignoring both items and values*

>>> ds.transpose('q5', ignore_items=[2, 3, 4], ignore_values=[1, 97])

>>> ds['q5_trans'].head(1)
  q5_trans_2 q5_trans_3 q5_trans_4 q5_trans_5 q5_trans_98
0       1;3;        NaN        NaN        NaN         NaN

>>> ds.meta('q5_trans')
delimited set                                             items               item texts codes        texts missing
q5_trans: How likely are you to do each of the ...
1                                                    q5_trans_2            Very unlikely     1      Surfing    None
2                                                    q5_trans_3        Probably wouldn't     2  Cave diving    None
3                                                    q5_trans_4  Probably would if asked     3  Windsurfing    None
4                                                    q5_trans_5              Very likely
5                                                   q5_trans_98               Don't know

**Flatten item answers**

- ``flatten()``