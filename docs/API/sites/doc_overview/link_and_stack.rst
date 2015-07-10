.. Quantipy documentation master file, created by
   sphinx-quickstart on Wed May 20 09:00:43 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
	:maxdepth: 5
	:includehidden:

===============
Links and Stack 
===============

| :ref:`genindex`
| :ref:`modindex`

""""

Relationships between variables
-------------------------------
Quantipy provides the ``quantipy.Stack`` object that serves as a container
for bulk analysis results. The Stack is a nested dictionary for which the 
different levels represent *requested* relationships between pairs of variables 
(``x`` and ``y``) in the source data, held in ``quantipy.Link`` objects.

A Stack dictionary's key hierarchy, with regard to its Link defintions, can 
be summarized as: 

	``[data_key][filter_key][x_key][y_key]``

and is the collection of all Links specified prior or during the data
aggregation process.

Setting up a Stack
""""""""""""""""""

:method: ``quantipy.Stack.add_data(data_key, meta=None, data=None)``

All data analysis performed in Quantipy relies on the ``Stack`` object which
must at least be provided with a case data source of type ``pandas.DataFrame``
that is associated with a unique data key. The data key is used as a reference
name for the underlying data file when working with the ``Stack`` instance and
constitutes the structure's first level. Adding a data source is done using
``add_data()``:

>>> my_data_key = 'Quantipy example'
>>> path_data = '/Quantipy Example Data/Example Data (A).csv'
>>> my_data = qp.dp.io.load_csv(path_data)
>>> stack = qp.Stack()
>>> stack.add_data(data_key=my_data_key, data=my_data)

The last two lines of the above example can also be written in the more 
compact form:

>>> stack = qp.Stack(add_data={'data_key': my_data_key, 'data': my_data})

Now is possible to access the Stack's ``data`` property which holds the 
``pandas.DataFrame`` representation of the case data file:

>>> type(stack[my_data_key].data)
<class 'pandas.core.frame.DataFrame'>

>>> stack[my_data_key].data[['q5_1', 'gender']].head()
       q5_1  gender
id_L1              
1         2       1
2         5       2
3         5       1
4         5       1
5        98       1

This basic setup implicitly adds the second level of the Stack structure
as well by indicating that the data is not filtered:

>>> stack[my_data_key].keys()
['no_filter']

Adding to the Stack
"""""""""""""""""""

:method: ``quantipy.Stack.add_link(data_keys=None, filters=['no_filter'], x=None,
         y=None, views=None, weights=None, variables=None)``

It is now possible to start growing the Stack using ``add_link()``, a method
that can be used in a few different ways. This section deals with its
``filters``, ``x`` and ``y`` parameters. 

``x`` and ``y`` refer to the defintions of ``quantipy.Link`` objects, a linking
of variables to each other. The most basic expression of these links is in the
form of a cross-tabulation or pivot table (since quantifying the relationship 
between variables involves aggregating it). x and y refer to the row and column
items of such a table, respectively. 



Defining relationships between variables that might be
analyzed later on is as easy as:

>>> stack.add_link(x=['q8', 'q5_1', 'q6_1'], y=['@', 'gender', 'age'])

Note the special ```@```-"variable" in the list passed for ``y`` parameter: It
instructs the creation of a *total* Link, i.e. a Link for each variable in the
list for ``x`` without a relationship to a ``y`` variable. By creating the
Links for between x and y, ``add_link()`` adds the third and fourth level to
the Stack:

>>> stack[my_data_key]['no_filter'].keys()
['q8', 'q5_1', 'q6_1']

>>> stack[my_data_key]['no_filter']['q5_1'].keys()
['@', 'age', 'gender']

Using the ``filters`` parameter, the Links can be created for  specific
portions of the case data individually. The filter expression must be a string
valid for the  `pandas DataFrame.query()
<http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.query.html>`_
method:

	>>> stack.add_link(filters=['gender==1', 'gender==2'],
	... x=['q8', 'q5_1', 'q6_1'], y=['@', 'gender', 'age'])

>>> stack[my_data_key].keys()
['gender==1', 'gender==2', 'no_filter']

Looking at the ``data`` property at the filter level of the Stack shows the
filtered case data accordingly now:

>>> stack[my_data_key]['gender==1'].data['gender'].value_counts()
1    3952
dtype: int64

>>> stack[my_data_key]['gender==2'].data['gender'].value_counts()
2    4303
dtype: int64

>>> stack[my_data_key]['no_filter'].data['gender'].value_counts()
2    4303
1    3952
dtype: int64

What's inside a Stack?
""""""""""""""""""""""

:method: ``quantipy.Stack.describe(index=None, columns=None, query=None,
								   split_view_names=False)``

Given its nested structure and the potentially vast amount Links
specified, inspecting a Stack by asking for its ``keys()`` and ``values()``
does not appear as a practical solution. Hence, the Stack's ``describe()``
method is the main tool to get quick access to summaries of its contents in
form of a ``pandas.DataFrame``, internally created using the `pivot_table()
method <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.pivot_table.html>`_.
Without any of its parameters specified, the method will show a simple
overview of a Stack's defining structural elements, e.g.

>>> stack.describe()
                data     filter     x       y  view  #
0   Quantipy example  gender==1    q8       @   NaN  1
1   Quantipy example  gender==1    q8     age   NaN  1
2   Quantipy example  gender==1    q8  gender   NaN  1
3   Quantipy example  gender==1  q5_1       @   NaN  1
4   Quantipy example  gender==1  q5_1     age   NaN  1
5   Quantipy example  gender==1  q5_1  gender   NaN  1
6   Quantipy example  gender==1  q6_1       @   NaN  1
7   Quantipy example  gender==1  q6_1     age   NaN  1
8   Quantipy example  gender==1  q6_1  gender   NaN  1
9   Quantipy example  gender==2    q8       @   NaN  1
10  Quantipy example  gender==2    q8     age   NaN  1
11  Quantipy example  gender==2    q8  gender   NaN  1
12  Quantipy example  gender==2  q5_1       @   NaN  1
13  Quantipy example  gender==2  q5_1     age   NaN  1
14  Quantipy example  gender==2  q5_1  gender   NaN  1
15  Quantipy example  gender==2  q6_1       @   NaN  1
16  Quantipy example  gender==2  q6_1     age   NaN  1
17  Quantipy example  gender==2  q6_1  gender   NaN  1
18  Quantipy example  no_filter    q8       @   NaN  1
19  Quantipy example  no_filter    q8     age   NaN  1
20  Quantipy example  no_filter    q8  gender   NaN  1
21  Quantipy example  no_filter  q5_1       @   NaN  1
22  Quantipy example  no_filter  q5_1     age   NaN  1
23  Quantipy example  no_filter  q5_1  gender   NaN  1
24  Quantipy example  no_filter  q6_1       @   NaN  1
25  Quantipy example  no_filter  q6_1     age   NaN  1
26  Quantipy example  no_filter  q6_1  gender   NaN  1


Defining the ``index`` and/or ``columns`` parameters allows to show (nested)
structures:

>>> stack.describe(index=['x'], columns=['y'])
y     @  age  gender
x                   
q5_1  3    3       3
q6_1  3    3       3
q8    3    3       3

>>> >>> stack.describe(index=['x', 'y'], columns=['filter'])
filter       gender==1  gender==2  no_filter
x    y                                      
q5_1 @               1          1          1
     age             1          1          1
     gender          1          1          1
q6_1 @               1          1          1
     age             1          1          1
     gender          1          1          1
q8   @               1          1          1
     age             1          1          1
     gender          1          1          1

or elements in isolation:

>>> stack.describe(index=['x'])
x
q5_1    9
q6_1    9
q8      9

Working with metadata
----------------------

:method: ``quantipy.Stack.add_data(data_key, meta=None, data=None)``
:method: ``quantipy.Stack.variable_types(data_key, only_type=None)``

One of Quantipy's key features is the inclusion of metadata that describes
the raw case data file's content and structure. Metadata can take many forms,
examples include:

* Global information on data sources
	* project names
	* project dates
* Variable and value labels
	* question texts
	* explanatory information on a variable
	* meaning of categorical codes in discrete variables
* Instructions on how data is interpreted
	* numeric: float, int
	* categorical: single- or multi-coded
	* how is a collection of values understood generally? categorical vs.
	  numeric interpretation
	* date, text, etc.
* Information on data structures of higher orders
	* item batteries, grid-like data
	* hierarchical data
* Information on special value codes
	* residual categories and *missing data* indicators

Considering that a main purpose of the Stack object is the automation of data
aggregation and reporting, file metadata acts an useful source of  gathering
instructions on how data is understood, processed and exported. The
``add_data()`` method therefore can pick up a Quantipy compatible metadata
file and store it alongside the related case data by specifying it in the
``meta`` parameter. Building on the example above, we can enrich our Stack
with metadata as per:

*Defining the data key and loading the case data*

>>> my_data_key = 'Quantipy example'
>>> path_data = '/Quantipy Example Data/Example Data (A).csv'
>>> my_data = qp.dp.io.load_csv(path_data)

*Loading the associated metadata file*

>>> path_meta = '/Quantipy Example Data/Example Data (A).json'
>>> my_meta = qp.dp.io.load_json(path_meta)

*Adding both case data and metadata to the Stack*

>>> stack = qp.Stack()
>>> stack.add_data(data_key=my_data_key, data=my_data, meta=my_meta)

With ``my_meta`` assigned to the ``meta`` property, the Stack now has access to
the complete collection of meta information on the source case data. This allows
to conveniently collect information on the variable types , using
``variable_types()`` method, e.g. in order to choose appropriate aggregation
methods:

>>> stack.variable_types(my_data_key)  
{'date': ['start_time', 'end_time'],
 'single': ['gender', 'locality', 'ethnicity', 'religion', 'q1', 'q2b', 'q4',
 'q5_1', 'q5_2', 'q5_3', 'q5_4', 'q5_5', 'q5_6', 'q6_1', 'q6_2', 'q6_3', 'q7_1',
 'q7_2', 'q7_3', 'q7_4', 'q7_5', 'q7_6', 'Wave'],
 'string': ['q8a', 'q9a'],
 'time': ['duration'],
 'int': ['record_number', 'unique_id', 'age', 'birth_day', 'birth_month',
 'birth_year'],
 'array': [u'q5', u'q7', u'q6'],
 'float': ['weight', 'weight_a', 'weight_b'],
 'delimited set': ['q2', 'q3', 'q8', 'q9']}

When only specific types of data are of interest, the ``only_type`` parameter
can be used to reduce the output to, for instance, show only multi-coded
variables (*delimited sets*) found in the file.

>>> stack.variable_types(my_data_key, only_type='delimited set')  
['q2', 'q3', 'q8', 'q9']


Working with multiple data sources
----------------------------------
It is possible to attach multiple file sources to a Stack by simply assigning
them to unique data keys. If we use our inital Stack, we can add another
source via ``add_data()``:

*Setting up the file paths & loading data*

>>> my_other_data_key = 'Other casedata'
>>> path_other_data = '/Quantipy Example Data/engine_B_data.csv'
>>> path_other_meta = '/Quantipy Example Data/engine_B_meta.json'
>>> my_other_data = qp.dp.io.load_csv(path_other_data)
>>> my_other_meta = qp.dp.io.load_json(path_other_meta)

*Adding the data to the Stack*

	>>> stack.add_data(data_key=my_other_data_key,
	...                data=my_other_data,
	...                meta=my_other_meta)

*Creating Links (taking the correct data key into account)*

	>>> stack.add_link(data_keys=my_other_data_key,
	...                x=['cost_dinner', 'q4'],
	...                y=['profile_gender', 'age_group'])

*Stack content*

>>> stack.describe(index=['data', 'filter'], columns=['x', 'y'])
x                          cost_dinner                       q4                q5_1            q6_1             q8           
y                            age_group profile_gender age_group profile_gender    @ age gender    @ age gender   @ age gender
data             filter                                                                                                      
Another file     no_filter           1              1         1              1  NaN NaN    NaN  NaN NaN    NaN NaN NaN    NaN
Quantipy example gender==1         NaN            NaN       NaN            NaN    1   1      1    1   1      1   1   1      1
                 gender==2         NaN            NaN       NaN            NaN    1   1      1    1   1      1   1   1      1
                 no_filter         NaN            NaN       NaN            NaN    1   1      1    1   1      1   1   1      1

A Stack instance can also be called directly with the ``add_data`` parameter
that can handle dicts of tuples or dicts structuring the data keys and input
file pairs. Instead of using two subsequent calls to the ``add_data()`` method,
the following two examples establish the same input data conections as before:

*dict type structure*

	>>> data_dict = {
	...              my_data_key: {'data': my_data, 'meta': my_meta},
	...              my_other_data_key: {'data': my_other_data, 'meta': my_other_meta}
	...             }

>>> stack = qp.Stack(add_data=data_dict)

*tuple type structure* 

	>>> data_tuple = {
	...               my_data_key: (my_data, my_meta),
	...               my_other_data_key: (my_other_data, my_other_meta)
	...              }

>>> stack = qp.Stack(add_data=data_tuple)

**To stress it again**, when adding Links to the Stack, we now need to
consider the appropriate data keys in order to target the matching input data
for our filters and x/y-axis variables. If we simply try to generate the
original Links on our new multiple sources holding Stack, we will run into an
error since not all variables are found in both files:

>>> stack.add_link(x=['q8', 'q5_1', 'q6_1'], y=['@', 'gender', 'age'])
ValueError: for data key: Another file
x=['q8', 'q5_1', 'q6_1'] not found, y=['gender'] not found.

Therefore, we define the ``data_key`` parameter explicitely...
	
	>>> stack.add_link(data_keys=my_data_key,
	...                filters=['no_filter', 'gender==1', 'gender==2']
	...                x=['q8', 'q5_1', 'q6_1'],
	...                y=['@', 'gender', 'age'])

	>>> stack.add_link(data_keys=my_other_data_key,
	...                x=['cost_dinner', 'q4'],
	...                y=['profile_gender', 'age_group'])

\... and end up with the correct Links:

>>> stack.describe(index=['data', 'filter'], columns=['x', 'y'])
x                          cost_dinner                       q4                q5_1            q6_1             q8           
y                            age_group profile_gender age_group profile_gender    @ age gender    @ age gender   @ age gender
data             filter                                                                                                      
Another file     no_filter           1              1         1              1  NaN NaN    NaN  NaN NaN    NaN NaN NaN    NaN
Quantipy example gender==1         NaN            NaN       NaN            NaN    1   1      1    1   1      1   1   1      1
                 gender==2         NaN            NaN       NaN            NaN    1   1      1    1   1      1   1   1      1
                 no_filter         NaN            NaN       NaN            NaN    1   1      1    1   1      1   1   1      1