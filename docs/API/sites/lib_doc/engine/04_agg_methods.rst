.. toctree::
 	:maxdepth: 5
	:includehidden:

================
View aggregation
================

All following examples are working with a ``qp.Stack`` that was populated out
of a ``qp.DataSet`` that includes the ensuing ``qp.Batch`` definitions:

>>> batch1 = dataset.get_batch('batch1')
>>> batch1.add_y_on_y('y_keys')
>>> print batch1.x_y_map
OrderedDict([('q1', ['@', 'gender', 'q1', 'locality', 'ethnicity']),
             ('q2', ['locality', 'ethnicity']),
             ('q6', ['@']),
             ('@', ['q6']),
             (u'q6_1', ['@', 'gender', 'q1']),
             (u'q6_2', ['@', 'gender', 'q1']),
             (u'q6_3', ['@', 'gender', 'q1'])])
>>> print batch1.x_filter_map
OrderedDict([('q1', {'(men only)+(q1)': (<function _intersection at 0x0000000019AE06D8>, [{'gender': 1}, {'age': [20, 21, 22, 23, 24, 25]}])}),
             ('q2', {'men only': {'gender': 1}}),
             ('q6', {'men only': {'gender': 1}}),
             ('q6_1', {'men only': {'gender': 1}}),
             ('q6_2', {'men only': {'gender': 1}}),
             ('q6_3', {'men only': {'gender': 1}})])
>>> print batch1.weights
['weight_a']

>>> batch2 = dataset.get_batch('batch2')
>>> print batch2.x_y_map
OrderedDict([('q2b', ['@', 'gender'])])
>>> print batch2.x_filter_map
OrderedDict([('q2b', 'no_filter')])
>>> print batch2.weights
['weight']

-----------
Basic views
-----------

Assuming a ``qp.Stack`` is populated with ``qp.Links``, then it is possible to
add various ``qp.View``\s to each ``Link``. This can be performed by running
``Stack.add_link()`` while its parameter ``views`` is not ``None``. Alternatively
the ``qp.Batch`` definitions, stored in the meta object of the ``Stack``, can
help to add basic ``View``\s, e.g. counts, percentages, bases and sums. Using
this option by running ``Stack.aggregate()`` makes it easier to add a large
amount of ``View``\s, regarding all special cases, in one step.

Adding ``c%`` and unweighted ``base`` ``View``\s to all ``Link``\s that are
defined by ``batch2``, can be performed like this:

>>> stack.aggregate(views=['c%'], unweighted_base=True, batches='batch2', verbose=False)
>>> stack.describe()
                data           filter       x          y               view  #
0   Example Data (A)         men only      q1         q1                NaN  1
1   Example Data (A)         men only      q1          @                NaN  1
2   Example Data (A)         men only      q1     gender                NaN  1
3   Example Data (A)         men only       @         q6                NaN  1
4   Example Data (A)         men only      q2  ethnicity                NaN  1
5   Example Data (A)         men only      q2   locality                NaN  1
6   Example Data (A)         men only    q6_1         q1                NaN  1
7   Example Data (A)         men only    q6_1          @                NaN  1
8   Example Data (A)         men only    q6_1     gender                NaN  1
9   Example Data (A)         men only    q6_2         q1                NaN  1
10  Example Data (A)         men only    q6_2          @                NaN  1
11  Example Data (A)         men only    q6_2     gender                NaN  1
12  Example Data (A)         men only    q6_3         q1                NaN  1
13  Example Data (A)         men only    q6_3          @                NaN  1
14  Example Data (A)         men only    q6_3     gender                NaN  1
15  Example Data (A)         men only  gender         q1                NaN  1
16  Example Data (A)         men only  gender          @                NaN  1
17  Example Data (A)         men only  gender     gender                NaN  1
18  Example Data (A)         men only      q6          @                NaN  1
19  Example Data (A)  (men only)+(q1)      q1         q1                NaN  1
20  Example Data (A)  (men only)+(q1)      q1          @                NaN  1
21  Example Data (A)  (men only)+(q1)      q1   locality                NaN  1
22  Example Data (A)  (men only)+(q1)      q1  ethnicity                NaN  1
23  Example Data (A)  (men only)+(q1)      q1     gender                NaN  1
24  Example Data (A)        no_filter     q2b          @  x|f|:|y|weight|c%  1
25  Example Data (A)        no_filter     q2b          @     x|f|x:|||cbase  1
26  Example Data (A)        no_filter     q2b     gender  x|f|:|y|weight|c%  1
27  Example Data (A)        no_filter     q2b     gender     x|f|x:|||cbase  1

Obviously ``View``\s are only added to ``Link``\s defined by ``batch2`` and
automatically weighted ``View``\s take the weight definition of ``batch2``,
which is evident from the view keys (``x|f|:|y|weight|c%``). Each view key
is an explicit definition for a ``View``. That means combining the information
of the four ``Link`` attributes with a view key, leads to a definite dataframe
and its belonging meta information:

>>> stack['Example Data (A)']['no_filter']['q2b']['gender']['x|f|:|y|weight|c%']
Question               q2b
Values                   @
Question Values
q2b      1       11.992144
         2       80.802580
         3        7.205276
>>> stack['Example Data (A)']['no_filter']['q2b']['@']['x|f|:|y|weight|c%'].meta()
{
    "agg": {
        "weights": "weight",
        "name": "c%",
        "grp_text_map": null,
        "text": "",
        "fullname": "x|f|:|y|weight|c%",
        "is_weighted": true,
        "method": "frequency",
        "is_block": false
    },
    "x": {
        "is_array": false,
        "name": "q2b",
        "is_multi": false,
        "is_nested": false
    },
    "shape": [
        3,
        1
    ],
    "y": {
        "is_array": false,
        "name": "@",
        "is_multi": false,
        "is_nested": false
    }
}

Now we are adding ``View``\s to all ``batch1``-defined ``Link``\s to see what
is meant by special cases:

>>> stack.aggregate(views=['c%', 'counts'], unweighted_base=True, batches='batch1', verbose=False)
>>> stack.describe(['x', 'view'], 'y').loc[['@', 'q6'], ['@', 'q6']]
y                            @   q6
x  view
@  x|f|:|y|weight_a|c%     NaN  1.0
   x|f|:||weight_a|counts  NaN  1.0
q6 x|f|:|y|weight_a|c%     1.0  NaN
   x|f|:||weight_a|counts  1.0  NaN

Even if unweighted bases are requested, they get skipped for array summaries
and transposed arrays.

And another outlier appears if ``y_on_y`` is requested; For a variable (in this
example ``q1``), used as cross- and downbreak, with an extended filter,
two ``Link``\s with ``View``\s are created:

>>> stack.describe(['y', 'filter', 'view'], 'x').loc['q1', 'q1']
filter           view
(men only)+(q1)  x|f|:|y|weight_a|c%       1.0
                 x|f|:||weight_a|counts    1.0
                 x|f|x:|||cbase            1.0
men only         x|f|:|y|weight_a|c%       1.0
                 x|f|:||weight_a|counts    1.0
                 x|f|x:|||cbase            1.0

The first one is the aggregation defined by the ``Batch`` construction plan,
the second one shows the ``y_on_y`` aggregation using the main ``Batch.filter``.

In general ``Stack.aggregate()`` is only able to add ``View``\s to already
existing ``Link``\s (see :doc:`DataSet.populate() <01_links_stacks>`).

-------------------------
Non-categorical variables
-------------------------

>>> batch3 = dataset.add_batch('batch3')
>>> batch3.add_x('age')
>>> stack = dataset.populate('batch3')
>>> stack.describe()
               data     filter    x  y  view  #
0  Example Data (A)  no_filter  age  @   NaN  1

Non-categorical variables (ints or floats) are handled in a special way.
There are two options:

	*	Treat them like categorical variables:
		Append them to the parameter ``categorize``, then counts, percentage
		and sum aggregations can be added alongside the ``cbase`` ``View``.

		>>> stack.aggregate(views=['c%', 'counts', 'counts_sum', 'c%_sum'],
		                    unweighted_base=True,
		                    categorize=['age'],
		                    batches='batch3',
		                    verbose=False)
		>>> stack.describe()
		               data     filter    x  y                     view  #
		0  Example Data (A)  no_filter  age  @           x|f|:|||counts  1
		1  Example Data (A)  no_filter  age  @     x|f.c:f|x:|y||c%_sum  1
		2  Example Data (A)  no_filter  age  @              x|f|:|y||c%  1
		3  Example Data (A)  no_filter  age  @           x|f|x:|||cbase  1
		4  Example Data (A)  no_filter  age  @  x|f.c:f|x:|||counts_sum  1

	*	Do not categorize the variable:
		Only ``cbase`` ``View`` is created and additional descriptive statics
		``View``\s must be added. The method will give you a warning:

		>>> stack.aggregate(views=['c%', 'counts', 'counts_sum', 'c%_sum'],
		                    unweighted_base=True,
		                    batches='batch3',
		                    verbose=True)
		Warning: Found 1 non-categorized numeric variable(s): ['age'].
		Descriptive statistics must be added!
		>>> stack.describe()
		               data     filter    x  y            view  #
		0  Example Data (A)  no_filter  age  @  x|f|x:|||cbase  1

----------------------
Descriptive statistics
----------------------

>>> b_name = 'batch4'
>>> batch4 = dataset.add_batch(b_name)
>>> batch4.add_x(['q2b', 'q6', 'age'])
>>> stack = dataset.populate(b_name)
>>> stack.aggregate(views='counts', unweighted_base=False, batches=b_name, verbose=False)
>>> stack.describe()
               data     filter     x  y            view  #
0  Example Data (A)  no_filter   q2b  @  x|f|:|||counts  1
1  Example Data (A)  no_filter  q6_1  @  x|f|:|||counts  1
2  Example Data (A)  no_filter  q6_2  @  x|f|:|||counts  1
3  Example Data (A)  no_filter  q6_3  @  x|f|:|||counts  1
4  Example Data (A)  no_filter   age  @  x|f|x:|||cbase  1
5  Example Data (A)  no_filter    q6  @  x|f|:|||counts  1

Adding descriptive statistics ``View``\s like mean, stddev, min, max, median
or sem can be added with the method ``stack.add_stats()``. With the parameters
``other_source``, ``rescale`` and ``exclude`` you can specify the calculation.
Again each combination of the parameters refers to a unique view key. Note that
in ``on_vars`` included arrays get unrolled, that means also all belonging
array items get equipped with the added ``View``:

>>> stack.add_stats(on_vars=['q2b', 'age'], stats='mean', _batches=b_name, verbose=False)
>>> stack.add_stats(on_vars=['q6'], stats='stddev', _batches=b_name, verbose=False)
>>> stack.add_stats(on_vars=['q2b'], stats='mean', rescale={1:100, 2:50, 3:0},
...                 custom_text='rescale mean', _batches=b_name, verbose=False)
>>> stack.describe('view', 'x')
x                               age  q2b   q6  q6_1  q6_2  q6_3
view
x|d.mean|x:|||stat              1.0  1.0  NaN   NaN   NaN   NaN
x|d.mean|x[{100,50,0}]:|||stat  NaN  1.0  NaN   NaN   NaN   NaN
x|d.stddev|x:|||stat            NaN  NaN  1.0   1.0   1.0   1.0
x|f|:|||counts                  NaN  1.0  1.0   1.0   1.0   1.0
x|f|x:|||cbase                  1.0  NaN  NaN   NaN   NaN   NaN

----
Nets
----

>>> b_name = 'batch5'
>>> batch5 = dataset.add_batch(b_name)
>>> batch5.add_x(['q2b', 'q6'])
>>> stack = dataset.populate(b_name)
>>> stack.aggregate(views=['counts', 'c%'], unweighted_base=True, batches=b_name, verbose=False)
>>> stack.describe('view', 'x')
x               q2b   q6  q6_1  q6_2  q6_3
view
x|f|:|y||c%     1.0  1.0   1.0   1.0   1.0
x|f|:|||counts  1.0  1.0   1.0   1.0   1.0
x|f|x:|||cbase  1.0  NaN   1.0   1.0   1.0

Net-like ``View``\s can be added with the method ``Stack.add_nets()`` by defining
``net_map``\s for selected variables. There is a distinction between two different
types of net ``View``\s:

    *   Expanded nets:
        The existing counts or percentage ``View``\s are replaced with the new
        net ``View``\s, which will the net-defining codes after or before the
        computed net groups (i.e. "overcode" nets).

        >>> stack.add_nets('q2b', [{'Top2': [1, 2]}], expand='after', _batches=b_name, verbose=False)
        >>> stack.describe('view', 'x')
        x                       q2b   q6  q6_1  q6_2  q6_3
        view
        x|f|:|y||c%             NaN  1.0   1.0   1.0   1.0
        x|f|:|||counts          NaN  1.0   1.0   1.0   1.0
        x|f|x:|||cbase          1.0  NaN   1.0   1.0   1.0
        x|f|x[{1,2}+]*:|y||net  1.0  NaN   NaN   NaN   NaN
        x|f|x[{1,2}+]*:|||net   1.0  NaN   NaN   NaN   NaN

    *   Not expanded nets:
        The new net ``View``\s are added to the stack, which contain only the
        computed net groups.

        >>> stack.add_nets('q2b', [{'Top2': [1, 2]}], _batches=b_name, verbose=False)
        >>> stack.describe('view', 'x')
        x                     q2b   q6  q6_1  q6_2  q6_3
        view
        x|f|:|y||c%           1.0  1.0   1.0   1.0   1.0
        x|f|:|||counts        1.0  1.0   1.0   1.0   1.0
        x|f|x:|||cbase        1.0  NaN   1.0   1.0   1.0
        x|f|x[{1,2}]:|y||net  1.0  NaN   NaN   NaN   NaN
        x|f|x[{1,2}]:|||net   1.0  NaN   NaN   NaN   NaN

The difference between the two net types are also visible in the view keys:
``x|f|x[{1,2}+]*:|||net`` versus ``x|f|x[{1,2}]:|||net``.

~~~~~~~~~~~~~~~
Net definitions
~~~~~~~~~~~~~~~

To create more complex net definitions the method ``quantipy.net()`` can be used,
which generates a well-formatted instruction dict and appends it to the ``net_map``.
It's a helper especially concerning including various texts with different
valid ``text_keys``. The next example shows how to prepare a net for 'q6'
(promoters, detractors):

>>> q6_net = qp.net([], [1, 2, 3, 4, 5, 6], 'Promotors', ['en-GB', 'sv-SE'])
>>> q6_net = qp.net(q6_net, [9, 10], {'en-GB': 'Detractors',
...                                   'sv_SE': 'Detractors',
...                                   'de-DE': 'Kritiker'})
>>> qp.net(q6_net[0], text='Promoter', text_key='de-DE')
>>> print q6_net
[
    {
        "1": [1, 2, 3, 4, 5, 6],
        "text": {
            "en-GB": "Promotors",
            "sv-SE": "Promotors",
            "de-DE": "Promoter"
        }
    },
    {
        "2": [9, 10],
        "text": {
            "en-GB": "Detractors",
            "sv_SE": "Detractors",
            "de-DE": "Kritiker"
        }
    }
]

~~~~~~~~~~~~
Calculations
~~~~~~~~~~~~

``Stack.add_nets()`` has the parameter ``calc``, which allows adding ``View``\s
that are calculated out of the defined nets. The method ``qp.calc()`` is a
helper to create a well-formatted instruction dict for the calculation.
For example calculate the NPS (promoters - detractors) for 'q6', see the example
above and create the following calculation:

>>> q6_calc = qp.calc((1, '-', 2), 'NPS', ['en-GB', 'sv-SE', 'de-DE'])
>>> print q6_calc
OrderedDict([('calc', ('net_1', <built-in function sub>, 'net_2')),
            ('calc_only', False),
            ('text', {'en-GB': 'NPS',
                      'sv-SE': 'NPS',
                      'de-DE': 'NPS'})])

>>> stack.add_nets('q6', q6_net, calc=q6_calc, _batches=b_name, verbose=False)
>>> stack.describe('view', 'x')
x                                                   q2b   q6  q6_1  q6_2  q6_3
view
x|f.c:f|x[{1,2,3,4,5,6}],x[{9,10}],x[{1,2,3,4,5...  NaN  1.0   1.0   1.0   1.0
x|f.c:f|x[{1,2,3,4,5,6}],x[{9,10}],x[{1,2,3,4,5...  NaN  1.0   1.0   1.0   1.0
x|f|:|y||c%                                         1.0  1.0   1.0   1.0   1.0
x|f|:|||counts                                      1.0  1.0   1.0   1.0   1.0
x|f|x:|||cbase                                      1.0  NaN   1.0   1.0   1.0

Also evident is, that nets added on arrays are also added for all array items.

---------------
Cumulative sums
---------------

Cumulative sum ``View``\s can be added to a specified collection of xks of the
``Stack`` using ``stack.cumulative_sum()``. These ``View``\s are always complete,
that means counts and percentage ``View``\s get replaced by them:

>>> b_name = 'batch6'
>>> batch6 = dataset.add_batch(b_name)
>>> batch6.add_x(['q2b', 'q6'])
>>> stack = dataset.populate(b_name)
>>> stack.aggregate(views=['counts', 'c%'], unweighted_base=True, batches=b_name, verbose=False)
>>> stack.cumulative_sum('q6', verbose=False)
>>> stack.describe('view', 'x')
x                                      q1  q2b   q6  q6_1  q6_2  q6_3
view
x|f.c:f|x++:|y|weight_a|c%_cumsum     NaN  NaN  1.0   3.0   3.0   3.0
x|f.c:f|x++:|y||c%_cumsum             NaN  NaN  1.0   1.0   1.0   1.0
x|f.c:f|x++:||weight_a|counts_cumsum  NaN  NaN  1.0   3.0   3.0   3.0
x|f.c:f|x++:|||counts_cumsum          NaN  NaN  1.0   1.0   1.0   1.0
x|f|:|y||c%                           1.0  1.0  NaN   NaN   NaN   NaN
x|f|:|||counts                        1.0  1.0  NaN   NaN   NaN   NaN
x|f|x:|||cbase                        1.0  1.0  NaN   1.0   1.0   1.0

------------------
Significance tests
------------------

>>> batch2 = dataset.get_batch('batch2')
>>> batch2.set_sigtests([0.05])
>>> batch5 = dataset.get_batch('batch5')
>>> batch5.set_sigtests([0.01, 0.05])
>>> stack = dataset.populate(['batch2', 'batch5'])
>>> stack.aggregate('counts', batches=['batch2', 'batch5'], verbose=False)
>>> stack.describe(['view', 'y'], 'x')
x                            q2b   q6  q6_1  q6_2  q6_3
view                 y
x|f|:||weight|counts @       1.0  NaN   NaN   NaN   NaN
                     gender  1.0  NaN   NaN   NaN   NaN
x|f|:|||counts       @       1.0  1.0   1.0   1.0   1.0
x|f|x:|||cbase       @       1.0  NaN   1.0   1.0   1.0
                     gender  1.0  NaN   NaN   NaN   NaN

Significance tests can only be added ``Batch``-wise, which indicates that
significance levels must be defined for each ``Batch`` before running
``stack.add_tests()``.

>>> stack.add_tests(['batch2', 'batch5'], verbose=False)
>>> stack.describe(['view', 'y'], 'x')
x                                               q2b   q6  q6_1  q6_2  q6_3
view                                    y
x|f|:||weight|counts                    @       1.0  NaN   NaN   NaN   NaN
                                        gender  1.0  NaN   NaN   NaN   NaN
x|f|:|||counts                          @       1.0  1.0   1.0   1.0   1.0
x|f|x:|||cbase                          @       1.0  NaN   1.0   1.0   1.0
                                        gender  1.0  NaN   NaN   NaN   NaN
x|t.props.Dim.01|:|||significance       @       1.0  NaN   1.0   1.0   1.0
x|t.props.Dim.05|:||weight|significance @       1.0  NaN   NaN   NaN   NaN
                                        gender  1.0  NaN   NaN   NaN   NaN
x|t.props.Dim.05|:|||significance       @       1.0  NaN   1.0   1.0   1.0
