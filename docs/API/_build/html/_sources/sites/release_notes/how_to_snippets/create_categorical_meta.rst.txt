.. toctree::
	:maxdepth: 5
	:hidden:

=============================================
Different ways of creating categorical values
=============================================

The ``DataSet`` methods ``add_meta()``, ``extend_values()`` and ``derive()``
offer three alternatives for specifying the categorical values of ``'single'``
and ``'delimited set'`` typed variables. The approaches differ with respect to
how the mapping of numerical value codes to value text labels is handled.

**(1) Providing a list of text labels**

By providing the category labels only as a list of ``str``, ``DataSet``
is going to create the numerical codes by simple enumeration:

>>> name, qtype, label = 'test_var', 'single', 'The test variable label'

>>> cats = ['test_cat_1', 'test_cat_2', 'test_cat_3']
>>> dataset.add_meta(name, qtype, label, cats)

>>> dataset.meta('test_var')
single                             codes       texts missing
test_var: The test variable label
1                                      1  test_cat_1    None
2                                      2  test_cat_2    None
3                                      3  test_cat_3    None

**(2) Providing a list of numerical codes**

If only the desired numerical codes are provided, the label information for all
categories consequently will appear blank. In such a case the user will, however,
get reminded to add the ``'text'`` meta in a separate step:

>>> cats = [1, 2, 98]
>>> dataset.add_meta(name, qtype, label, cats)
...\\quantipy\core\dataset.py:1287: UserWarning: 'text' label information missing,
only numerical codes created for the values object. Remember to add value 'text' metadata manually!

>>> dataset.meta('test_var')
single                             codes texts missing
test_var: The test variable label
1                                      1          None
2                                      2          None
3                                     98          None

**(3) Pairing numerical codes with text labels**

To explicitly assign codes to corresponding labels, categories can also be
defined as a list of tuples of codes and labels:

>>> cats = [(1, 'test_cat_1') (2, 'test_cat_2'), (98, 'Don\'t know')]
>>> dataset.add_meta(name, qtype, label, cats)

>>> dataset.meta('test_var')
single                             codes       texts missing
test_var: The test variable label
1                                      1  test_cat_1    None
2                                      2  test_cat_2    None
3                                     98  Don't know    None

.. note::
	All three approaches are also valid for defining the ``items`` object for
	``array``-typed ``masks``.