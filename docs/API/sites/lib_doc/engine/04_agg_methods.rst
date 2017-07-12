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

>>> batch4 = dataset.add_batch('batch4')
>>> batch4.add_x(['q2b', 'q6', 'age'])
>>> stack = dataset.populate('batch4')
>>> stack.aggregate(views='counts', unweighted_base=False, batches='batch4', verbose=False)
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

