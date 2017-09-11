.. toctree::
 	:maxdepth: 5
	:includehidden:

==================
DataSet components
==================

------------------
Case and meta data
------------------

``Quantipy`` builds upon the ``pandas`` library to feature the ``DataFrame``
and ``Series`` objects in the case data component of its ``DataSet`` object.
Additionally, each ``DataSet`` offers a metadata component to describe the
data columns and provide additional information on the characteristics of the
underlying structure. The metadata document is implemented as a nested ``dict``
and provides the following ``keys`` on its first level:

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

---------------------------------
``columns`` and ``masks`` objects
---------------------------------

There are two variable collections inside a ``Quantipy`` metadata document:
``'columns'`` is storing the meta for each accompanying ``pandas.DataFrame``
column object, while ``'masks'`` are building upon the regular ``'columns'``
metadata but additionally employ special meta instructions to define
complex data types. An example is the the ``'array'`` type that (in MR speak) maps
multiple "question" variables to one "answer" object.

"Simple"" data definitons that are supported by ``Quantipy`` can either be numeric
``'float'`` and ``'int'`` types, categorical ``'single'`` and ``'delimited set'``
variables or of type ``'string'``, ``'date'`` and ``'time'``.

---------------------------------------------
Languages: ``text`` and ``text_key`` mappings
---------------------------------------------
Throughout ``Quantipy`` metadata all label information, e.g. variable question
texts and category descriptions, are stored in ``text`` objects that are mapping
different language (or context) versions of a label to a specific ``text_key``.
That way the metadata can support multi-language and multi-purpose (for example
detailed/extensive vs. short question texts) label information in a digestable
format that is easy to query:

>>> meta['columns']['q1']['text']
{'en-GB': 'This is a long English label',
 'de-DE': 'Das ist ein langes deutsches Label',
 'x edits': 'Short label'}

Valid ``text_key`` settings are:

============== ==============================================================
``text_key``   Language / context
============== ==============================================================
``'en-GB'``	   English
``'de-DE'``	   German
``'fr-FR'``	   French
``'da-DK'``    Danish
``'sv-SV'``	   Swedish
``'nb-NO'``    Norwegian
``'fi-FI'``    Finnish
``'x edits'``  Build label edit for x-axis
``'y edits'``  Build label edit for y-axis
============== ==============================================================

-----------------------------
Categorical ``values`` object
-----------------------------
``single`` and ``delimited set`` variables restrict the possible case data
entries to a list of ``values`` that consist of numeric answer codes and their
``text`` labels, defining distinct categories:

>>> meta['columns']['q1']['values']
[{'value': 1,
  'text': {'en-GB': 'Dog'}
 },
 {'value': 2,
  'text': {'en-GB': 'Cat'}
 },
 {'value': 3,
  'text': {'en-GB': 'Bird'}
 },
 {'value': -9,
  'text': {'en-GB': 'Not an animal'}
 }]

------------------
The ``array`` type
------------------
Turning to the ``masks`` collection of the metadata, ``array`` variables
group together a collection of variables that share a common response options
scheme, i.e. different statements (usually referencing a broader topic) that
are answered using the same scale. In the ``Quantipy`` metadata document, an
``array`` variable has a ``subtype`` that describes the type of the
constructing source variables listed in the ``items`` object. In contrast to simple variable types, any
categorical ``values`` metadata is stored inside the shared information collection
``lib``, for access from both the ``columns`` and ``masks`` representation of
``array`` elements:

>>> meta['masks']['q5']
{u'items': [{u'source': u'columns@q5_1', u'text': {u'en-GB': u'Surfing'}},
  {u'source': u'columns@q5_2', u'text': {u'en-GB': u'Snowboarding'}},
  {u'source': u'columns@q5_3', u'text': {u'en-GB': u'Kite boarding'}},
  {u'source': u'columns@q5_4', u'text': {u'en-GB': u'Parachuting'}},
  {u'source': u'columns@q5_5', u'text': {u'en-GB': u'Cave diving'}},
  {u'source': u'columns@q5_6', u'text': {u'en-GB': u'Windsurfing'}}],
 u'name': u'q5',
 u'subtype': u'single',
 u'text': {u'en-GB': u'How likely are you to do each of the following in the next year?'},
 u'type': u'array',
 u'values': 'lib@values@q5'}

>>> meta['lib']['values']['q5']
[{u'text': {u'en-GB': u'I would refuse if asked'}, u'value': 1},
 {u'text': {u'en-GB': u'Very unlikely'}, u'value': 2},
 {u'text': {u'en-GB': u"Probably wouldn't"}, u'value': 3},
 {u'text': {u'en-GB': u'Probably would if asked'}, u'value': 4},
 {u'text': {u'en-GB': u'Very likely'}, u'value': 5},
 {u'text': {u'en-GB': u"I'm already planning to"}, u'value': 97},
 {u'text': {u'en-GB': u"Don't know"}, u'value': 98}]

Exploring the ``columns`` meta of an array item shows the same ``values`` reference pointer and informs about its ``parent`` meta structure, i.e. the
array's ``masks`` defintion:

>>> meta['columns']['q5_1']
{u'name': u'q5_1',
 u'parent': {u'masks@q5': {u'type': u'array'}},
 u'text': {u'en-GB': u'How likely are you to do each of the following in the next year? - Surfing'},
 u'type': u'single',
 u'values': u'lib@values@q5'}