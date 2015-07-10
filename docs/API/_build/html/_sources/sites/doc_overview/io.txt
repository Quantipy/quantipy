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
| SPSS        | .dff/.mdd   | Yes         | No          |
| Dimensions  |             |             |             |
+-------------+-------------+-------------+-------------+
| SPSS        | .sav        | Yes         | Yes         |
| Statistics  |             |             |             | 
+-------------+-------------+-------------+-------------+
| Decipher    |tab-delimited| Yes         | No          |
|             |.json/ .txt  |             |             |
+-------------+-------------+-------------+-------------+
| Ascribe     |tab-delimited| Yes         | No          |
|             |.xml/ .txt   |             |             |
+-------------+-------------+-------------+-------------+

The following functions are designed to convert the different file formats'
structures into inputs understood by Quantipy.

Reading from **Dimensions**:

>>> from quantipy.core.tools.dp.io import read_dimensions
>>> meta, data = read_dimensions(path_mdd, path_ddf)

Reading from **SPSS Statistics**:

>>> from quantipy.core.tools.dp.io import read_spss
>>> meta, data = read_spss(path_sav, meta, data)

Writing to **SPSS Statistics**:

>>> from quantipy.core.tools.dp.io import write_spss
>>> write_spss(path_sav, meta, data)

Reading from **Decipher**:

>>> from quantipy.core.tools.dp.io import read_decipher
>>> meta, data = read_decipher(path_json, path_txt)

Reading from **Ascribe**:

>>> from quantipy.core.tools.dp.io import read_ascribe
>>> meta, data = read_ascribe(path_xml, path_txt)

