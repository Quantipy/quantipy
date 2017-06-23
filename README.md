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
We recommend installing [Anaconda for Python 2.7](http://continuum.io/downloads) which will provide most of the required libraries and an easy means of keeping them up-to-date over time.
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
