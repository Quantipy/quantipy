# Quantipy
### Python for people data
Quantipy is an open-source data processing, analysis and reporting software project that builds on the excellent pandas and numpy libraries. Aimed at people data, Quantipy offers support for native handling of special data types like multiple choice variables, statistical analysis using case or observation weights, DataFrame metadata and pretty data exports.

### Key features
  - Reads plain .csv, converts from Dimensions, SPSS, Decipher, or Ascribe
  - Open metadata format to describe and manage datasets
  - Powerful, metadata-driven cleaning, editing, recoding and transformation of datasets
  - Computation and assessment of data weights
  - Easy-to-use analysis interface
  - Automated data aggregation using ``Batch`` defintions
  - Structured analysis and reporting via Chain and Cluster containers
  - Exports to SPSS, Dimensions ddf/mdd, MS Excel and Powerpoint with flexible layouts and various options

#### Contributors
- Kerstin Müller, Alexander Buchhammer, Alasdair Eaglestone, James Griffiths: https://yougov.co.uk
- Datasmoothie’s Birgir Hrafn Sigurðsson and Geir Freysson: http://datasmoothie.io/

### Required libraries before installation
We recommend installing [Anaconda for Python 2.7](http://continuum.io/downloads)
which will provide most of the required libraries and an easy means of keeping
them up-to-date over time.
  - Python 2.7.8
  - Numpy 1.11.3
  - Pandas 0.19.2

## 5-minutes to Quantipy

**Get started**

Start a new folder called 'Quantipy-5' and add a subfolder called 'data'.

You can find an example dataset in quantipy/tests:

- Example Data (A).csv
- Example Data (A).json

Put these files into your ``'data'`` folder.

Start with some import statements:

```python
import pandas as pd
import quantipy as qp

from quantipy.core.tools.dp.prep import frange

# This is a handy bit of pandas code to let you display your dataframes
# without having them split to fit a vertical column.
pd.set_option('display.expand_frame_repr', False)
```

**Load, inspect and edit your data**

Load the input files in a ``qp.DataSet`` instance and inspect the metadata
with methods like ``.variables()``, ``.meta()`` or ``.crosstab()``:
```python
# Define the paths of your input files
path_json = './data/Example Data (A).json'
path_csv = './data/Example Data (A).csv'

dataset = qp.DataSet('Example Data (A)')
dataset.read_quantipy(path_json, path_csv)

dataset.crosstab('q2', text=True)
```

```
Question                                                           q2. Which, if any, of these other sports have you ever participated in?
Values                                                                                                                                   @
Question                                           Values
q2. Which, if any, of these other sports have y... All                                                         2999.0
                                                   Sky diving                                                  1127.0
                                                   Base jumping                                                1366.0
                                                   Mountain biking                                             1721.0
                                                   Kite boarding                                                649.0
                                                   Snowboarding                                                 458.0
                                                   Parachuting                                                  428.0
                                                   Other                                                        492.0
                                                   None of these                                                 53.0
```

Variables can be created, recoded or edited with DataSet methods, e.g. ``derive()``:
```python
mapper = [(1,  'Any sports', {'q2': frange('1-6, 97')}),
          (98, 'None of these', {'q2': 98})]

dataset.derive('q2_rc', 'single', dataset.text('q2'), mapper)
dataset.meta('q2_rc')
```

```
single                                              codes          texts missing
q2_rc: Which, if any, of these other sports hav...
1                                                       1     Any sports    None
2                                                      98  None of these    None
```

The  ``DataSet`` case data component can be inspected with the []-indexer, as known from a ``pd.DataFrame``:
```python

dataset[['q2', 'q2_rc']].head(5)
```

```
        q2  q2_rc
0  1;2;3;5;    1.0
1      3;6;    1.0
2       NaN    NaN
3       NaN    NaN
4       NaN    NaN
```

**Analyse and create aggregations batchwise**

A ``qp.Batch`` as a subclass of ``qp.DataSet`` is a container for structuring a
data analysis and aggregation specifications:
```python
batch = dataset.add_batch('batch1')
batch.add_x(['q2', 'q2b', 'q5'])
batch.add_y(['gender', 'q2_rc'])
```

The batch definitions are stored in ``dataset._meta['sets']['batches']['batch1']``.
A ``qp.Stack`` can be created and populated based on the available ``qp.Batch``
definitions stored in the ``qp.DataSet``:
```python
stack = dataset.populate()
stack.describe()
```

```
                data     filter     x       y  view  #
0   Example Data (A)  no_filter   q2b       @   NaN  1
1   Example Data (A)  no_filter   q2b   q2_rc   NaN  1
2   Example Data (A)  no_filter   q2b  gender   NaN  1
3   Example Data (A)  no_filter    q2       @   NaN  1
4   Example Data (A)  no_filter    q2   q2_rc   NaN  1
5   Example Data (A)  no_filter    q2  gender   NaN  1
6   Example Data (A)  no_filter    q5       @   NaN  1
7   Example Data (A)  no_filter  q5_3       @   NaN  1
8   Example Data (A)  no_filter  q5_3   q2_rc   NaN  1
9   Example Data (A)  no_filter  q5_3  gender   NaN  1
10  Example Data (A)  no_filter  q5_2       @   NaN  1
11  Example Data (A)  no_filter  q5_2   q2_rc   NaN  1
12  Example Data (A)  no_filter  q5_2  gender   NaN  1
13  Example Data (A)  no_filter  q5_1       @   NaN  1
14  Example Data (A)  no_filter  q5_1   q2_rc   NaN  1
15  Example Data (A)  no_filter  q5_1  gender   NaN  1
16  Example Data (A)  no_filter  q5_6       @   NaN  1
17  Example Data (A)  no_filter  q5_6   q2_rc   NaN  1
18  Example Data (A)  no_filter  q5_6  gender   NaN  1
19  Example Data (A)  no_filter  q5_5       @   NaN  1
20  Example Data (A)  no_filter  q5_5   q2_rc   NaN  1
21  Example Data (A)  no_filter  q5_5  gender   NaN  1
22  Example Data (A)  no_filter  q5_4       @   NaN  1
23  Example Data (A)  no_filter  q5_4   q2_rc   NaN  1
24  Example Data (A)  no_filter  q5_4  gender   NaN  1
```

Each of the defintions is a ``qp.Link``. These can be e.g. analyzed in various ways,
e.g. grouped catgeories can be calculated using the engine ``qp.Quantity``:
```python
link = stack[dataset.name]['no_filter']['q2']['q2_rc']
q = qp.Quantity(link)
q.group(frange('1-6, 97'), axis='x', expand='after')
q.count()
```

```
Question          q2_rc
Values              All       1    98
Question Values
q2       All     2999.0  2946.0  53.0
         net     2946.0  2946.0   0.0
         1       1127.0  1127.0   0.0
         2       1366.0  1366.0   0.0
         3       1721.0  1721.0   0.0
         4        649.0   649.0   0.0
         5        458.0   458.0   0.0
         6        428.0   428.0   0.0
         97       492.0   492.0   0.0
```

We can also simply add so called ``qp.view``s to the whole of the ``qp.Stack``:
```python
stack.aggregate(['counts', 'c%'], False, verbose=False)
stack.add_stats('q2b', stats=['mean'], rescale={1: 100, 2:50, 3:0}, verbose=False)

stack.describe('view', 'x')
```

```
x                                q2  q2b   q5  q5_1  q5_2  q5_3  q5_4  q5_5  q5_6
view
x|d.mean|x[{100,50,0}]:|||stat  NaN  3.0  NaN   NaN   NaN   NaN   NaN   NaN   NaN
x|f|:|y||c%                     3.0  3.0  1.0   3.0   3.0   3.0   3.0   3.0   3.0
x|f|:|||counts                  3.0  3.0  1.0   3.0   3.0   3.0   3.0   3.0   3.0
```

```python
link = stack[dataset.name]['no_filter']['q2b']['q2_rc']
link['x|d.mean|x[{100,50,0}]:|||stat']
```

```
Question             q2_rc
Values                  1          98
Question Values
q2b      mean    52.354167  43.421053
```

## More examples
There is so much more you can do with Quantipy... why don't you explore the docs to find out!
