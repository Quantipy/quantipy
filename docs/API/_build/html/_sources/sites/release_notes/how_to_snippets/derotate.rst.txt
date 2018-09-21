.. toctree::
    :maxdepth: 5
    :hidden:

==========
Derotation
==========

------------------
What is derotation
------------------

Derotation of ``data`` is necessary if brands, products or something similar
(**levels**) are assessed and each respondent (case) rates a different
selection of that levels. So each **case** has several **responses**.
Derotation now means, that the ``data`` is switched from case-level to
responses-level.

**Example**: ``q1_1/q1_2``: On a scale from 1 to 10, how much do you like the
following drinks?

|   1:  water
|   2:  cola
|   3:  lemonade
|   4:  beer
|

**``data``**

+-------+---------+---------+------+------+--------+
| id    | drink_1 | drink_2 | q1_1 | q1_2 | gender |
+-------+---------+---------+------+------+--------+
| case1 | 1       | 3       | 2    | 8    | 1      |
+-------+---------+---------+------+------+--------+
| case2 | 1       | 4       | 9    | 5    | 2      |
+-------+---------+---------+------+------+--------+
| case3 | 2       | 4       | 6    | 10   | 1      |
+-------+---------+---------+------+------+--------+

**derotated ``data``**

+-------+-------+----------------+----+--------+
|       | drink | drink_levelled | q1 | gender |
+-------+-------+----------------+----+--------+
| case1 | 1     | 1              | 2  | 1      |
+-------+-------+----------------+----+--------+
| case1 | 2     | 3              | 8  | 1      |
+-------+-------+----------------+----+--------+
| case2 | 1     | 1              | 9  | 2      |
+-------+-------+----------------+----+--------+
| case2 | 2     | 4              | 5  | 2      |
+-------+-------+----------------+----+--------+
| case3 | 1     | 2              | 6  | 1      |
+-------+-------+----------------+----+--------+
| case3 | 2     | 4              | 10 | 1      |
+-------+-------+----------------+----+--------+

To identify which case rates which levels, some key-/level-variables are
included in the ``data``, in this example ``drink_1`` and ``drink_2``.
Variables (for example ``gender``) that are not included to this loop can also
be added.

---------------------------------
How to use ``DataSet.derotate()``
---------------------------------

The ``DataSet`` method takes a few parameters:

*   ``levels``: ``dict`` of ``list``

    Contains all key-/level-variables and the name for the new levelled variable.
    All key-/level-variables must have the same ``value_map``.

    >>> levels = {'drink': ['drink_1', 'drink_2']}


|

*   ``mapper``: ``list`` of ``dicts`` of ``list``

    Contains the looped questions and the new ``column`` name to which the
    looped questions will be combinded.

    >>> mapper = [{'q1': ['q1_1', 'q1_2']}]

|

*   ``other``: ``str`` or ``list`` of ``str``

    Contains all variables that should be assumed to the derotated ``data``, but
    which are not included in the loop.

    >>> other = 'gender'

|

*   ``unique_key``: ``str``

    Name of varibale that identifies cases in the initial ``data``.

    >>> unique_key = 'id'

|

*   ``dropna``: ``bool``, default ``True``

    If a case rates less then the possible counts of levels, these responses
    will be droped.

>>> ds = dataset.derotate(levels = {'drink': ['drink_1', 'drink_2']},
...                       mapper = [{'q1': ['q1_1', 'q1_2']}],
...                       other = 'gender',
...                       unique_key = 'id',
...                       dropna = True)

----------------------
What about ``arrays``?
----------------------

It is possible that also ``arrays`` are looped. In this case a mapper can look
like this:

>>> mapper = [{'q12_1': ['q12a[{q12a_1}].q12a_grid', 'q12b[{q12b_1}].q12b_grid',
...                      'q12c[{q12c_1}].q12c_grid', 'q12d[{q12d_1}].q12d_grid']},
...           {'q12_2': ['q12a[{q12a_2}].q12a_grid', 'q12b[{q12b_2}].q12b_grid',
...                      'q12c[{q12c_2}].q12c_grid', 'q12d[{q12d_2}].q12d_grid']},
...           {'q12_3': ['q12a[{q12a_3}].q12a_grid', 'q12b[{q12b_3}].q12b_grid',
...                      'q12c[{q12c_3}].q12c_grid', 'q12d[{q12d_3}].q12d_grid']},
...           {'q12_4': ['q12a[{q12a_4}].q12a_grid', 'q12b[{q12b_4}].q12b_grid',
...                      'q12c[{q12c_4}].q12c_grid', 'q12d[{q12d_4}].q12d_grid']},
...           {'q12_5': ['q12a[{q12a_5}].q12a_grid', 'q12b[{q12b_5}].q12b_grid',
...                      'q12c[{q12c_5}].q12c_grid', 'q12d[{q12d_5}].q12d_grid']},
...           {'q12_6': ['q12a[{q12a_6}].q12a_grid', 'q12b[{q12b_6}].q12b_grid',
...                      'q12c[{q12c_6}].q12c_grid', 'q12d[{q12d_6}].q12d_grid']},
...           {'q12_7': ['q12a[{q12a_7}].q12a_grid', 'q12b[{q12b_7}].q12b_grid',
...                      'q12c[{q12c_7}].q12c_grid', 'q12d[{q12d_7}].q12d_grid']},
...           {'q12_8': ['q12a[{q12a_8}].q12a_grid', 'q12b[{q12b_8}].q12b_grid',
...                      'q12c[{q12c_8}].q12c_grid', 'q12d[{q12d_8}].q12d_grid']},
...           {'q12_9': ['q12a[{q12a_9}].q12a_grid', 'q12b[{q12b_9}].q12b_grid',
...                      'q12c[{q12c_9}].q12c_grid', 'q12d[{q12d_9}].q12d_grid']},
...           {'q12_10': ['q12a[{q12a_10}].q12a_grid', 'q12b[{q12b_10}].q12b_grid',
...                       'q12c[{q12c_10}].q12c_grid', 'q12d[{q12d_10}].q12d_grid']},
...           {'q12_11': ['q12a[{q12a_11}].q12a_grid', 'q12b[{q12b_11}].q12b_grid',
...                       'q12c[{q12c_11}].q12c_grid', 'q12d[{q12d_11}].q12d_grid']},
...           {'q12_12': ['q12a[{q12a_12}].q12a_grid', 'q12b[{q12b_12}].q12b_grid',
...                       'q12c[{q12c_12}].q12c_grid', 'q12d[{q12d_12}].q12d_grid']},
...           {'q12_13': ['q12a[{q12a_13}].q12a_grid', 'q12b[{q12b_13}].q12b_grid',
...                       'q12c[{q12c_13}].q12c_grid', 'q12d[{q12d_13}].q12d_grid']}]]

Can be also writen like this:

>>> for y in frange('1-13'):
...     q_group = []
...     for x in  ['a', 'b', 'c', 'd']:
...         var = 'q12{}'.format(x)
...         var_grid = var + '[{' +  var + '_{}'.format(y) + '}].' + var + '_grid'
...         q_group.append(var_grid)
...     mapper.append({'q12_{}'.format(y): q_group})

So the derotated ``dataset`` will lose its ``meta`` information about the
``mask`` and only the ``columns`` ``q12_1`` to ``q12_13`` will be added. To
receive back the ``mask`` structure, use the method ``dataset.to_array()``:

>>> variables = [{'q12_1': u'label 1'},
...              {'q12_2': u'label 2'},
...              {'q12_3': u'label 3'},
...              {'q12_4': u'label 4'},
...              {'q12_5': u'label 5'},
...              {'q12_6': u'label 6'},
...              {'q12_7': u'label 7'},
...              {'q12_8': u'label 8'},
...              {'q12_9': u'label 9'},
...              {'q12_10': u'label 10'},
...              {'q12_11': u'label 11'},
...              {'q12_12': u'label 12'},
...              {'q12_13': u'label 13'}]
>>> ds.to_array('qTP', variables, 'Var_name')

``variables`` can also be a list of variable-names, then the ``mask-items``
will be named by its belonging ``columns``.

``arrays`` included in ``other`` will keep their ``meta`` structure.