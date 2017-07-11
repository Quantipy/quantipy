.. toctree::
 	:maxdepth: 5
	:includehidden:

=======================
Collecting aggregations
=======================

Talking about collected aggregations you will always get in touch with 
``qp.Stack``. A ``Stack`` can be handled as a container that can include a 
large amount of aggregations in form of ``qp.Link``\s. 

----------------------
What is a ``qp.Link?``
----------------------

A ``qp.Link`` has four attributes that make it unique and define how it can 
be stored in a ``qp.Stack``. These four attributes are ``data_key``, ``filter``,
``x`` (downbreak) and ``y`` (crossbreak), which are positioned in a ``Stack`` 
like in a tree diagram:
	*	Each ``Stack`` can have various ``data_key``\s.
	*	Each ``data_key`` can have various ``filter``\s.
	*	Each ``filter`` can have various ``x``\s.
	*	Each ``x`` can have various ``y``\s.
Consequently ``Stack[dk][filter][x][y]`` is one ``qp.Link`` and can be added 
using ``stack.add_link()``.

``qp.Link``\s are used to group different ``qp.View``\s (frequencies, 
calculations, ect.) that are applied on the same four attributes.

------------------------------------------
Populate a ``qp.Stack`` with ``qp.Link``\s
------------------------------------------

A ``qp.Stack`` is able to cope with a large amount of aggregations, so it is 
impractical to add ``Link``\s one by one with ``Stack.add_link()``. It is easier
to create a construction plan (``qp.Batch``) and use the stored settings in
``DataSet._meta['sets']['batches']`` to populate a ``qp.Stack``. Presumed 
``dataset`` is a ``qp.DataSet`` instance containing the definitions of two 
``qp.Batch``\es, then a ``qp.Stack`` can be created and filled running::

	stack = dataset.populate(batches='all')

For the ``Batch`` definitions from :doc:`here <../batch/00_overview>`, you 
will get the following construction plans:

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

>>> batch2 = dataset.get_batch('batch2')
>>> print batch2.x_y_map
OrderedDict([('q2b', ['@', 'gender'])])
>>> print batch2.x_filter_map
OrderedDict([('q2b', 'no_filter')])

As both ``Batch``\es refer to the same data file, the same ``data_key`` (in this
case the name of ``dataset``) is linked.

After populating the ``Stack`` content can be viewed using ``.describe()``:

>>> stack.describe()
                data           filter       x          y  view  #
0   Example Data (A)         men only      q1         q1   NaN  1
1   Example Data (A)         men only      q1          @   NaN  1
2   Example Data (A)         men only      q1     gender   NaN  1
3   Example Data (A)         men only       @         q6   NaN  1
4   Example Data (A)         men only      q2  ethnicity   NaN  1
5   Example Data (A)         men only      q2   locality   NaN  1
6   Example Data (A)         men only    q6_1         q1   NaN  1
7   Example Data (A)         men only    q6_1          @   NaN  1
8   Example Data (A)         men only    q6_1     gender   NaN  1
9   Example Data (A)         men only    q6_2         q1   NaN  1
10  Example Data (A)         men only    q6_2          @   NaN  1
11  Example Data (A)         men only    q6_2     gender   NaN  1
12  Example Data (A)         men only    q6_3         q1   NaN  1
13  Example Data (A)         men only    q6_3          @   NaN  1
14  Example Data (A)         men only    q6_3     gender   NaN  1
15  Example Data (A)         men only  gender         q1   NaN  1
16  Example Data (A)         men only  gender          @   NaN  1
17  Example Data (A)         men only  gender     gender   NaN  1
18  Example Data (A)         men only      q6          @   NaN  1
19  Example Data (A)  (men only)+(q1)      q1         q1   NaN  1
20  Example Data (A)  (men only)+(q1)      q1          @   NaN  1
21  Example Data (A)  (men only)+(q1)      q1   locality   NaN  1
22  Example Data (A)  (men only)+(q1)      q1  ethnicity   NaN  1
23  Example Data (A)  (men only)+(q1)      q1     gender   NaN  1
24  Example Data (A)        no_filter     q2b          @   NaN  1
25  Example Data (A)        no_filter     q2b     gender   NaN  1

You can find all combinations that are also defined in the ``x_y_map`` in the 
``Stack``, but also ``Link``\s like ``Stack['Example Data (A)']['men only']['gender']['gender']``
are included. These are special cases and arise from ``y_on_y`` settings.
Sometimes it is helpful to group a ``describe``-dataframe and create a 
cross-tabulation of the four ``Link`` attributes to get a better overview:

>>> stack.describe('x', 'filter')
filter  (men only)+(q1)  men only  no_filter
x                                           
@                   NaN       1.0        NaN
gender              NaN       3.0        NaN
q1                  5.0       3.0        NaN
q2                  NaN       2.0        NaN
q2b                 NaN       NaN        2.0
q6                  NaN       1.0        NaN
q6_1                NaN       3.0        NaN
q6_2                NaN       3.0        NaN
q6_3                NaN       3.0        NaN

This example shows how many ``Links`` are included for each x-filter combination.

