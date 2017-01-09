.. toctree::
	:maxdepth: 5
	:includehidden:

===========
What's new?
===========

| :ref:`genindex`
| :ref:`modindex`

""""

*0.1.0*

Features
--------

First version of official documentation
"""""""""""""""""""""""""""""""""""""""
Version 0.1.0 of Quantipy is starting off the official documentation of the
library. The docs are based on workflow examples to demonstrate and explain
key features, capabilities and current limits.

SPSS Statictics .sav file format I/O support added
""""""""""""""""""""""""""""""""""""""""""""""""""
Quantipy is now able to convert SPSS .sav files into or from its native input
data sources (``pandas.DataFrame``/``dict``).


Rewrite of ``Quantity`` class
"""""""""""""""""""""""""""""
The aggregation engine has undergone some modifications to harmonize the main
computation methods and add more flexibilty. There are no substantial changes
to how ``Quantity`` interacts with the view methods, except for the instruction
of code group arithmetics (to e.g. produce a Net Promoter Score view). It is
recommended to read through the documentation of the
:doc:`aggregation engine <doc_overview/aggregations>` and to review the
:doc:`view method customization examples <doc_overview/views_freq_desc>`.

``Chain.concat()`` and ``Cluster.merge()`` methods
""""""""""""""""""""""""""""""""""""""""""""""""""
Chains and Clusters can now produce ``pandas.DataFrame`` outputs that visualize
their aggregation results. This enables the creation of raw reports that are
not requiring any metadata sources.
Please see examples :doc:`here <doc_overview/chain_cluster_build>`


Mimicked askia tests now support nets / code groups
"""""""""""""""""""""""""""""""""""""""""""""""""""
Tests of signficance that are mimicking the *askia* software package now
support testing of net-type Views just as their *Dimensions* counterparts
already did in the last release.

Improvements
------------

View notation split output in Stack.describe() reduced
""""""""""""""""""""""""""""""""""""""""""""""""""""""
When setting the ``split_view_names`` parameter to ``True``, the output of the
Stack's ``describe()`` method will now only show all unique notations and their
components to improve readability.

View object and View meta
"""""""""""""""""""""""""
A View's meta is now accessed via the ``meta()`` method. The ``meta``
attribute has been removed. A range of :doc:`self-inspection methods
<doc_overview/views_concept_notation>` have been added.

Bugfixes
--------

* Fixed a bug that was causing complex logics view to run into a error when
  performend on filtered Links.
* Fixed a bug in ``coltests()`` view method that was computing tests of
  signficance multiple times.
