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
------------------------------

Docstring
"""""""""

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


