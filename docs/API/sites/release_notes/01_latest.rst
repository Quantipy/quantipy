.. toctree::
	:maxdepth: 5
	:includehidden:

===================
Latest (12/01/2019)
===================

**New**: ``Chain.export()`` / ``assign()`` and custom calculations

Expanding on the current ``Chain`` editing features provided via ``cut()``
and ``join()``, it is now possible to calculate additional row and column results using plain ``pandas.dataframe`` methods. Use ``Chain.export()`` to work on a simplified ``Chain.dataframe`` and ``assign()`` to rebuild it properly when
finished.

An intro to this feature can be viewed here: "LINK TO THE DOC"


**New**: ``Batch.as_main(keep=True)`` to change ``qp.Batch`` relations

It is now possible to promote an ``.additional`` Batch to a main/regular one. Optionally, the original parent Batch can be erased by setting ``keep=False``. Example:

*Starting from:*

>>> dataset.batches(main=True, add=False)
['batch 2', 'batch 5']

>>> dataset.batches(main=False, add=True)
['batch 4', 'batch 3', 'batch 1']

*We turn* ``batch 3`` *into a normal one:*

>>> b = dataset.get_batch('batch 3')
>>> b.as_main()

>>> dataset.batches(main=True, add=False)
['batch 2', 'batch 5', 'batch 3']

>>> dataset.batches(main=False, add=True)
['batch 4', 'batch 1']


**New**: On-the-fly rebasing via ``Quantity.normalize(on='y', per_cell=False)``

Quantipy's engine will now accept another variable's base for (column) percentage
computations. Furthermore, it is possible to rebase the cell counts to the *cell
frequencies of the other variable's cross-tabulation* by setting ``per_cell=True``,
i.e. rebase variables with identical categories to their respective per-category results. The following example shows how ``'A1'`` results are serving as cell bases
for the percentages of ``'A2'``:

>>> l = stack[stack.keys()[0]]['no_filter']['A1']['datasource']
>>> q = qp.Quantity(l)
>>> q.count()
Question        datasource
Values                 All      1       2       3      4       5       6
Question Values
A1       All        6984.0  767.0  1238.0  2126.0  836.0  1012.0  1005.0
         1          1141.0  503.0    78.0   109.0  102.0   155.0   194.0
         2          2716.0  615.0   406.0   499.0  499.0   394.0   303.0
         3          1732.0  603.0    89.0   128.0  101.0   404.0   407.0
         4          5391.0  644.0   798.0  1681.0  655.0   796.0   817.0
         5          4408.0  593.0   177.0  1649.0  321.0   818.0   850.0
         6          3584.0  615.0   834.0   834.0  327.0   507.0   467.0
         7          4250.0  588.0   724.0  1717.0  540.0    55.0   626.0
         8          3729.0  413.0  1014.0   788.0  311.0   539.0   664.0
         9          3575.0  496.0   975.0   270.0  699.0   230.0   905.0
         10         4074.0  582.0   910.0  1148.0  298.0   861.0   275.0
         11         2200.0  446.0   749.0   431.0  177.0   146.0   251.0
         12         5554.0  612.0   987.0  1653.0  551.0   860.0   891.0
         13          544.0   40.0   107.0   232.0   87.0    52.0    26.0

>>> l = stack[stack.keys()[0]]['no_filter']['A2']['datasource']
>>> q = qp.Quantity(l)
>>> q.count()
Question        datasource
Values                 All      1       2       3      4      5      6
Question Values
A2       All        6440.0  727.0  1131.0  1894.0  749.0  960.0  979.0
         1           568.0  306.0    34.0    32.0   48.0   63.0   85.0
         2          1135.0  417.0   107.0    88.0  213.0  175.0  135.0
         3           975.0  426.0    43.0    49.0   49.0  220.0  188.0
         4          2473.0  350.0   267.0   599.0  431.0  404.0  422.0
         5          2013.0  299.0    88.0   573.0  162.0  417.0  474.0
         6          1174.0  342.0   219.0   183.0  127.0  135.0  168.0
         7          1841.0  355.0   161.0   754.0  285.0   21.0  265.0
         8          1740.0  265.0   376.0   327.0  160.0  212.0  400.0
         9          1584.0  181.0   390.0    89.0  398.0   94.0  432.0
         10         1655.0  257.0   356.0   340.0  137.0  443.0  122.0
         11          766.0  201.0   241.0   101.0   76.0   53.0   94.0
         12         2438.0  217.0   528.0   497.0  247.0  459.0  490.0
         13         1532.0   72.0   286.0   685.0  118.0  183.0  188.0

>>> q.normalize(on='A1', per_cell=True)
Question         datasource
Values                  All           1           2           3           4           5           6
Question Values
A2       All      92.210767   94.784876   91.357027   89.087488   89.593301   94.861660   97.412935
         1        49.780894   60.834990   43.589744   29.357798   47.058824   40.645161   43.814433
         2        41.789396   67.804878   26.354680   17.635271   42.685371   44.416244   44.554455
         3        56.293303   70.646766   48.314607   38.281250   48.514851   54.455446   46.191646
         4        45.872751   54.347826   33.458647   35.633551   65.801527   50.753769   51.652387
         5        45.666969   50.421585   49.717514   34.748332   50.467290   50.977995   55.764706
         6        32.756696   55.609756   26.258993   21.942446   38.837920   26.627219   35.974304
         7        43.317647   60.374150   22.237569   43.913803   52.777778   38.181818   42.332268
         8        46.661303   64.164649   37.080868   41.497462   51.446945   39.332096   60.240964
         9        44.307692   36.491935   40.000000   32.962963   56.938484   40.869565   47.734807
         10       40.623466   44.158076   39.120879   29.616725   45.973154   51.451800   44.363636
         11       34.818182   45.067265   32.176235   23.433875   42.937853   36.301370   37.450199
         12       43.896291   35.457516   53.495441   30.066546   44.827586   53.372093   54.994388
         13      281.617647  180.000000  267.289720  295.258621  135.632184  351.923077  723.076923




**New**: ``DataSet.missings(name=None)``

This new method returns the missing data defintiton for the provided variable or
all missing defintitons found in the dataset (if ``name`` is omitted).

>>> dataset.missings()
{u'q10': {u'exclude': [6]},
 u'q11': {u'exclude': [977]},
 u'q17': {u'exclude': [977]},
 u'q23_1_new': {u'exclude': [8]},
 u'q25': {u'exclude': [977]},
 u'q32': {u'exclude': [977]},
 u'q38': {u'exclude': [977]},
 u'q39': {u'exclude': [977]},
 u'q48': {u'exclude': [977]},
 u'q5': {u'exclude': [977]},
 u'q9': {u'exclude': [977]}}

**Update**: ``DataSet.batches(main=True, add=False)``

The collection of ``Batch`` sets can be separated by ``main`` and ``add``\itional
ones (see above) to make analyzing Batch setups and relations easier. The default is still to return all Batch names.

**Bugfix**: Stack, ``other_source`` statistics failing for delimited sets

A bug that prevented ``other_source`` statistics being computed on delimited set type variables has been resolved by adjusting the underlying data type checking mechanic.