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


Start a new folder called 'Quantipy-5' and add a subfolder called 'data'.

You can find an example dataset in quantipy/tests:

- Example Data (A).csv
- Example Data (A).json

Put these files into your 'data' folder.

Start with some import statements:

```python
import pandas as pd
import quantipy as qp

from quantipy.core.tools.dp.prep import frange

# This is a handy bit of pandas code to let you display your dataframes 
# without having them split to fit a vertical column.
pd.set_option('display.expand_frame_repr', False)
```

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

Variables can be created, recoded or edited with DataSet methods:
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

DataSet case component can be inspected with []-indexer:
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

``qp.Batch`` is a subclass of ``qp.DataSet`` and is a container for structuring a 
Link collection's specifications.

The batch definitions are stored in ``dataset._meta['sets']['batches']['batch1']``:
```python
batch = dataset.add_batch('batch1')
batch.add_x(['q1', 'q2', 'q5'])
batch.add_y(['gender', 'q2_rc'])
```

A ``qp.Stack`` can be created and populated based on all available ``qp.Batch``
definitions, that are stored in ``qp.DataSet``:
```python
stack = dataset.populate()
stack.describe()
```

```
                data     filter     x       y  view  #
0   Example Data (A)  no_filter    q1       @   NaN  1
1   Example Data (A)  no_filter    q1   q2_rc   NaN  1
2   Example Data (A)  no_filter    q1  gender   NaN  1
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









```python
import pandas as pd
import quantipy as qp

from quantipy.core.tools.dp.io import load_json
from quantipy.core.helpers.functions import paint_dataframe

# This is a handy bit of pandas code to let you display your
# dataframes without having them split to fit a vertical column.
pd.set_option('display.expand_frame_repr', False)

# Set up the required path variables
path_data = './data/'
name_data = 'Example Data (A)'

# Paths to the input files
path_json = '{}{}.json'.format(path_data, name_data)
path_csv = '{}{}.csv'.format(path_data, name_data)

# Paths to expected Quantipy files we will want to save
path_stack = '{}{}.stack'.format(path_data, name_data)
path_cluster = '{}{}'.format(path_data, name_data)
path_excel = '{}{}.xlsx'.format(path_data, name_data)

# Load the case metadata and the case data
meta = load_json(path_json)
data = pd.DataFrame.from_csv(path_csv)

# Create a stack (container for aggregations) and add the
# source data to it
stack = qp.Stack(add_data={'Example': {'data': data, 'meta': meta}})

# If you want to list your variables by type you can use
# something like this.
cols_by_type = {
    t: [
        col
        for col in meta['columns']
        if meta['columns'][col]['type']==t
    ]
    for t in ['single', 'delimited set', 'int', 'float', 'string']
}
singles = cols_by_type['single']
multiples = cols_by_type['delimited set']
ints = cols_by_type['int']

# Quantipy cares about the links between variables, so set up x and y lists
x_vars = ['q1', 'q2']
y_vars = ['gender', 'ethnicity']

# Add variable links and views (aggregations) on those links
stack.add_link(x=x_vars, y=y_vars, views=['cbase', 'c%'])

# Save the stack
stack.save(path_stack)

# See what's in the stack (what aggregations exist already?)
print stack.describe()

#       data     filter   x          y                     view  #
# 0  Example  no_filter  q1     gender       x|frequency||y||c%  1
# 1  Example  no_filter  q1     gender  x|frequency|x:y|||cbase  1
# 2  Example  no_filter  q1  ethnicity       x|frequency||y||c%  1
# 3  Example  no_filter  q1  ethnicity  x|frequency|x:y|||cbase  1
# 4  Example  no_filter  q2     gender       x|frequency||y||c%  1
# 5  Example  no_filter  q2     gender  x|frequency|x:y|||cbase  1
# 6  Example  no_filter  q2  ethnicity       x|frequency||y||c%  1
# 7  Example  no_filter  q2  ethnicity  x|frequency|x:y|||cbase  1

# These are the keys under which our base and column percentages
# are saved, we'll use them to get them out of the stack.
view_keys = [
    'x|frequency|x:y|||cbase',
    'x|frequency||y||c%'
]

# Isolate a single aggregation in the stack and take a look at it
data_key = 'Example'
filter_key = 'no_filter'
x_key = x_vars[0]
y_key = y_vars[0]
view_key = 'x|frequency||y||c%'
# Look at the raw dataframe
df = stack[data_key][filter_key][x_key][y_key][view_key].dataframe
print df

# Question            gender
# Values                   1          2
# Question Values
# q1       1        3.669028   3.532419
#          2        5.187247   4.462003
#          3       27.682186  27.980479
#          4       36.386640  36.277016
#          5        2.403846   2.300720
#          6        5.035425   6.460609
#          7       11.310729  10.388101
#          8        1.644737   1.533814
#          9        0.075911   0.023240
#          96       1.037449   1.161980
#          98       0.986842   1.510574
#          99       4.579960   4.369045

# Paint the labels onto the raw dataframe
print paint_dataframe(df, meta)

# Question                                                     gender. What is your gender?
# Values                                                                    Male     Female
# Question                                Values
# q1. Min fitness activity? Swimming                                    3.669028   3.532419
#                           Running/jogging                             5.187247   4.462003
#                           Lifting weights                            27.682186  27.980479
#                           Aerobics                                   36.386640  36.277016
#                           Yoga                                        2.403846   2.300720
#                           Pilates                                     5.035425   6.460609
#                           Football (soccer)                          11.310729  10.388101
#                           Basketball                                  1.644737   1.533814
#                           Hockey                                      0.075911   0.023240
#                           Other                                       1.037449   1.161980
#                           I regularly change my fitness activity      0.986842   1.510574
#                           Not applicable - I don't exercise           4.579960   4.369045

# Extract chains of links from the stack in preparation for the build
# Chains are a subset of the stack drawn out in a special shape that
# represents a one-to-many set of relationship.
chains = stack.get_chain(x=x_vars, y=y_vars, views=view_keys, orient_on='x')

# The first chain is 'q1' to 'gender' and 'ethnicity'
print chains[0].describe()

#       data     filter   x          y                     view  #
# 0  Example  no_filter  q1     gender       x|frequency||y||c%  1
# 1  Example  no_filter  q1     gender  x|frequency|x:y|||cbase  1
# 2  Example  no_filter  q1  ethnicity       x|frequency||y||c%  1
# 3  Example  no_filter  q1  ethnicity  x|frequency|x:y|||cbase  1

# The second chain is 'q2' to 'gender' and 'ethnicity'
print chains[1].describe()

#       data     filter   x          y                     view  #
# 0  Example  no_filter  q2     gender       x|frequency||y||c%  1
# 1  Example  no_filter  q2     gender  x|frequency|x:y|||cbase  1
# 2  Example  no_filter  q2  ethnicity       x|frequency||y||c%  1
# 3  Example  no_filter  q2  ethnicity  x|frequency|x:y|||cbase  1

# Create a cluster and fill it with the chains
# The cluster is consumed by a build
cluster = qp.Cluster('Percentages')
cluster.add_chain(chains)
cluster.save(path_cluster)

# Use the cluster to build an XLSX
qp.ExcelPainter(
    path_excel=path_excel,
    meta=meta,
    cluster=[cluster],
    create_toc=False,
    display_names=['x', 'y']
)

print 'Finished!'
```

## More examples
There is so much more you can do with Quantipy... why don't you explore the docs to find out!
