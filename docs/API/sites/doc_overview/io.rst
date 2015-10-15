.. toctree::
  :maxdepth: 5
  :includehidden:

============
Data sources
============

| :ref:`genindex`
| :ref:`modindex`

""""

Case and meta data
------------------

In Quantipy, case data is natively represented as a ``pandas.DataFrame``
object, while metadata is organized in form of a ``dict``.

In the most basic case, case data and associated meta files can be provided
as tab-delimited ``.csv`` and a ``.json`` string (that follows Quantipy's
metadata structure) respectively. Loading files into Quantipy is done using the
``quantipy.dp.io.load_csv()`` and ``quantipy.dp.io.load_json()`` methods.

Quantipy's metadata structure is managed in pure Python and supports a
mapping syntax to allow shared and referenced content. Currently there are 6
meta elements in the document:

============== ==============================================================
element        contains
============== ==============================================================
``'type'``	   case data type
``'info'``	   info on the source data
``'lib'``	   shared use references
``'columns'``  info on ``DataFrame`` columns (Quantipy types, labels, etc.)
``'sets'``	   ordered groups of variables pointing to other parts of the meta
``'masks'``    complex variable type definitions (arrays, dichotomous, etc.)
============== ==============================================================

.. seealso::
	Usage examples of loading and working with case and metadata files are
	provided throughout the :doc:`Stack <link_and_stack>`, :doc:`Aggregations
	<aggregations>` and :doc:`Views <views_concept_notation>` sections of the
	docs.

Supported conversions
---------------------

In adddition to providing plain ``.csv``/``.json`` data (pairs), source files
can be read into Quantipy using a number of I/O functions to deal with
standard file formats encountered in the market research industry:

+-------------+-------------+-------------+-------------+
| Software    | Format      | Read        | Write       |
+=============+=============+=============+=============+
| SPSS        | .sav        | Yes         | Yes         |
| Statistics  |             |             |             | 
+-------------+-------------+-------------+-------------+
| SPSS        | .dff/.mdd   | Yes         | No          |
| Dimensions  |             |             |             |
+-------------+-------------+-------------+-------------+
| Decipher    |tab-delimited| Yes         | No          |
|             |.json/ .txt  |             |             |
+-------------+-------------+-------------+-------------+
| Ascribe     |tab-delimited| Yes         | No          |
|             |.xml/ .txt   |             |             |
+-------------+-------------+-------------+-------------+

The following functions are designed to convert the different file formats'
structures into inputs understood by Quantipy.

SPSS Statistics
"""""""""""""""

**Reading:**

>>> from quantipy.core.tools.dp.io import read_spss
>>> meta, data = read_spss(path_sav)

.. note::
  On a Windows machine you MUST use ``ioLocale=None`` when reading
  from SPSS. This means if you are using a Windows machine your base
  example for reading from SPSS is 
  ``meta, data = read_spss(path_sav, ioLocale=None)``.

When reading from SPSS you have the opportunity to specify a custom
dichotomous values map, that will be used to convert all dichotomous
sets into Quantipy delimited sets, using the ``dichot`` argument. 

The entire read operation will use the same map on all dichotomous 
sets so they must be applied uniformly throughout the SAV file. The 
default map that will be used if none is provided will be 
``{'yes': 1, 'no': 0}``. 

>>> meta, data = read_spss(path_sav, dichot={'yes': 1, 'no': 2})

SPSS dates will be converted to pandas dates by default but
if this results in conversion issues or failures you can read
the dates in as Quantipy strings to deal with them later, using the
``dates_as_strings`` argument.

>>> meta, data = read_spss(path_sav, dates_as_strings=True)

**Writing:**

>>> from quantipy.core.tools.dp.io import write_spss
>>> write_spss(path_sav, meta, data)

By default SPSS files will be generated from the ``'data file'``
set found in ``meta['sets']``, but a custom set can be named instead
using the ``from_set`` argument. 

>>> write_spss(path_sav_analysis, meta, data, from_set='sav-export')

The custom set must be well-formed:

>>> "sets" : {
...     "sav-export": {
...         "items": [
...             "columns@Q1", 
...             "columns@Q2", 
...             "columns@Q3",
...             ...
...         ]
...     }
... }

Dimensions
""""""""""

**Reading:**

>>> from quantipy.core.tools.dp.io import read_dimensions
>>> meta, data = read_dimensions(path_mdd, path_ddf)

Decipher
""""""""

**Reading:**

>>> from quantipy.core.tools.dp.io import read_decipher
>>> meta, data = read_decipher(path_json, path_txt)

Ascribe
""""""""

**Reading:**

>>> from quantipy.core.tools.dp.io import read_ascribe
>>> meta, data = read_ascribe(path_xml, path_txt)

