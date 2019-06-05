# encoding: utf-8
import re
import warnings
import numpy as np
import pandas as pd
import sys
import importlib

default_stdout = sys.stdout
default_stderr = sys.stderr
importlib.reload(sys)
sys.setdefaultencoding('utf-8')
sys.stdout = default_stdout
sys.stderr = default_stderr

BASE_COL = '@'
BASE_ROW = ['is_counts', 'is_c_base']
PCT_TYPES = ['is_c_pct', 'is_r_pct']
NOT_PCT_TYPES = ['is_stat']
CONTINUATION_STR = "(continued {})"
MAX_PIE_ELMS = 4


def float2String(input, ndigits=0):
    """
    Round and converts the input, if int/float or list of, to a string.

    Parameters
    ----------
    input: int/float or list of int/float
    ndigits: int
        number of decimals to round to

    Returns
    -------
    output: string or list of strings
         depending on the input
    """

    output = input
    if not isinstance(input, list):
        output = [output]
    output = [round(x, ndigits) for x in output]
    output = list(map(int, output))
    output = list(map(str, output))
    if not isinstance(input, list):
        output = output[0]
    return output

def uniquify(l):
    """
    Return the given list without duplicates, retaining order.

    See Dave Kirby's order preserving uniqueifying list function
    http://www.peterbe.com/plog/uniqifiers-benchmark
    """

    seen = set()
    seen_add = seen.add
    uniques = [x for x in l if x not in seen and not seen_add(x)]

    return uniques


def strip_levels(df, rows=None, columns=None):
    """
    Function that strips a MultiIndex DataFrame for specified row and column index

    Parameters
    ----------
    df: pandas.DataFrame
    rows: int
        Row index to remove, default None
    columns: int
        Column index to remove, default None

    Returns
    -------
    df_strip: pandas.DataFrame
        The input dataframe stripped for specified levels

    """
    df_strip = df.copy()
    if rows is not None:
        if df_strip.index.nlevels > 1:
            df_strip.index = df_strip.index.droplevel(rows)
    if columns is not None:
        if df_strip.columns.nlevels > 1:
            df_strip.columns = df_strip.columns.droplevel(columns)
    return df_strip


def as_numeric(df):
    """
    Runs through all values in input DataFrame and replaces
    ',' to '.'
    '%' to ''
    '-' to '0'
    '*' to '0'

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
        with values as float

    """

    if not df.values.dtype in ['float64', 'int64']:
        data = [[float(str(value).replace(',','.').replace('%','').replace('-','0').replace('*','0')) for value in values] for values in df.values]
        df = pd.DataFrame(data, index=df.index, columns=df.columns)
    return df.copy()


def is_grid_slice(chain):
    """
    Returns True if chain is a grid slice

    Parameters
    ----------
    chain: quantipy.Chain

    Returns
    -------
    bool
        True if grid slice

    """
    pattern = '\[\{.*?\}\].'
    found = re.findall(pattern, chain.name)
    if len(found) > 0 and chain._array_style == -1:
        return True


def get_indexes_from_list(lst, find, exact=True):
    """
    Helper function that search for element in a list and
    returns a list of indexes for element match
    E.g.
    get_indexes_from_list([1,2,3,1,5,1], 1) returns [0,3,5]
    get_indexes_from_list(['apple','banana','orange','lemon'], 'orange') -> returns [2]
    get_indexes_from_list(['apple','banana','lemon',['orange', 'peach']], 'orange') -> returns []
    get_indexes_from_list(['apple','banana','lemon',['orange', 'peach']], ['orange'], False) -> returns [3]

    Parameters
    ----------
    lst: list
        The list to look in
    find: any
        the element to find, can be a list
    exact: bool
        If False then index are returned if find in lst-item otherwise
        only if find = lst-item

    Returns
    -------
    list of int

    """
    if exact == True:
        return [index for index, value in enumerate(lst) if value == find]
    else:
        if isinstance(find,list):
            return [index for index, value in enumerate(lst) if set(find).intersection(set(value))]
        else:
            return [index for index, value in enumerate(lst) if find in value]


def auto_charttype(df, array_style, max_pie_elms=MAX_PIE_ELMS):
    """
    Auto suggest chart type based on dataframe analysis
    TODO Move this to Class PptxDataFrame()

    Parameters
    ----------
    df: pandas.DataFrame
        Not multiindex
    array_style: int
        array_style as returned from Chain Class
    max_pie_elms: int
        Max number of elements in Pie chart

    Returns
    -------
    str
        One of charttypes ('bar_clustered', 'bar_stacked_100', 'pie')

    """
    if array_style == -1: # Not array summary
        chart_type = 'bar_clustered'
        if len(df.index.get_level_values(-1)) <= max_pie_elms:
            if len(df.columns.get_level_values(-1)) == 1:
                chart_type = 'pie'
    elif array_style == 0:
        chart_type = 'bar_stacked_100'
        # TODO _auto_charttype - return 'bar_stacked' if rows not sum to 100
    else:
        chart_type = 'bar_clustered'

    return chart_type


def fill_gaps(l):
    """
    Return l replacing empty strings with the value from the previous position.

    Parameters
    ----------
    l: list

    Returns
    -------
    list
    """

    lnew = []
    for i in l:
        if i == '':
            lnew.append(lnew[-1])
        else:
            lnew.append(i)
    return lnew


def fill_index_labels(df):
    """
    Fills in blank labels in the second level of df's multi-level index.

    Parameters
    ----------
    df: pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
    """

    _0, _1 = list(zip(*df.index.values.tolist()))
    _1new = fill_gaps(_1)
    dfnew = df.copy()
    dfnew.index = pd.MultiIndex.from_tuples(list(zip(_0, _1new)), names=df.index.names)
    return dfnew


def fill_column_values(df, icol=0):
    """
    Fills empty values in the targeted column with the value above it.

    Parameters
    ----------
    df: pandas.DataFrame
    icol: int

    Returns
    -------
    pandas.DataFrame
    """

    v = df.iloc[:,icol].fillna('').values.tolist()
    vnew = fill_gaps(v)
    dfnew = df.copy() # type: pd.DataFrame
    dfnew.iloc[:,icol] = vnew
    return dfnew


class PptxDataFrame(object):
    """
    Class for handling the dataframe to be charted.
    The class is instantiated from the class PptxChain and holds
    the chains dataframe, flattened and ready for charting.
    A series of get cell-types methods can be used to select specific cell-types.

    Parameters:
    ----------
    df: pandas.DataFrame
        The actual dataframe ready to use with PptxPainter
    array_style: int
        Array style as given by quantipy.chain.array_style
    cell_types: list
        The dataframes cell types as given by quantipy.chain.contents

    """

    def __init__(self, dataframe, cell_types, array_style):

        self.array_style = array_style
        self.cell_items = cell_types
        self.df = dataframe # type: pd.DataFrame
        self.__frames = []

    def __call__(self):
        return self.df

    def to_table(self, decimals=2, pct_decimals=2, decimal_separator='.'):
        """
        Returns self.df formatted to be added to a table in a slide.
        Basically just rounds values and if cell type = % then multiply values with 100
        # todo : should'nt be here, move to PptxPainter

        Parameters
        ----------
        decimals:  int
            Number of decimals for not percentage cell_types
        pct_decimals: int
            Number of decimals for percentage cell_types
        decimal_separator: str

        Returns
        -------
        self

        """

        df = self.df
        if df.empty:
            return self
        if self.array_style == -1:
            df = df.T

        df = df.fillna('')

        # Percent type cells
        pct_indexes = get_indexes_from_list(self.cell_items, PCT_TYPES, exact=False)
        df.iloc[:, pct_indexes] *= 100
        df.iloc[:, pct_indexes] = df.iloc[:, pct_indexes].round(decimals=pct_decimals)

        # Not percent type cells
        not_pct_indexes = get_indexes_from_list(self.cell_items, NOT_PCT_TYPES, exact=False)
        df.iloc[:, not_pct_indexes] = df.iloc[:, not_pct_indexes].round(decimals=decimals)

        df = df.astype('str')

        if pct_decimals == 0 or decimals == 0:
            pct_columns = df.columns[pct_indexes].tolist() if pct_decimals == 0 else []
            not_pct_columns = df.columns[not_pct_indexes].tolist() if decimals == 0 else []
            columns = pct_columns + not_pct_columns
            df[columns] = df[columns].replace('\.0', '', regex=True)

        if not decimal_separator == '.':
            df = df.replace('\.', ',', regex=True)

        if self.array_style == -1:
            df = df.T

        self.df = df

        return self

    def _select_categories(self, categories):
        """
        Returns a copy of self.df having only the categories requested

        Parameters
        ----------
        categories : list
            A list of ints specifying the categories from self.df to return

        Returns
        -------
        pptx_df_copy : PptxDataFrame

        """

        if self.array_style == -1:
            df_copy=self.df.iloc[categories]
        else:
            df_copy = self.df.iloc[:,categories]

        pptx_df_copy = PptxDataFrame(df_copy,self.cell_items,self.array_style)
        pptx_df_copy.cell_items = [self.cell_items[i] for i in categories]

        return pptx_df_copy

    def get_means(self):
        """
        Return a copy of the PptxDataFrame containing only mean type categories

        Returns
        -------
        PptxDataFrame

        """
        return self.get('means')

    def get_nets(self):
        """
        Return a copy of the PptxDataFrame only containing net type categories

        Returns
        -------
        PptxDataFrame

        """
        return self.get('net')


    def get_cpct(self):
        """
        Return a copy of the PptxDataFrame only containing column percentage categories

        Returns
        -------
        PptxDataFrame

        """
        return self.get('c_pct')

    def get_propstest(self):
        """
        Return a copy of the PptxDataFrame only containing sig testing type categories

        Returns
        -------
        PptxDataFrame

        """
        return self.get('tests')

    def get_stats(self):
        """
        Return a copy of the PptxDataFrame only containing stat type categories

        Returns
        -------
        PptxDataFrame

        """
        return self.get('stats')

    def _get_propstest_index(self):
        """
        Return a list of index numbers from self.cell_items of type 'is_propstest'

        Returns
        -------
        row_list : list

        """
        row_list = get_indexes_from_list(self.cell_items, 'is_propstest', exact=False)
        return row_list

    def _get_stats_index(self):
        """
        Return a list of index numbers from self.cell_items of type 'is_stat'

        Returns
        -------
        row_list : list

        """
        row_list = get_indexes_from_list(self.cell_items, 'is_stat', exact=False)
        return row_list

    def _get_cpct_index(self):
        """
        Return a list of index numbers from self.cell_items of type 'is_c_pct' and not types
        'is_net', 'net', 'is_c_pct_sum'

        Returns
        -------
        row_list : list

        """

        row_list = get_indexes_from_list(self.cell_items, 'is_c_pct', exact=False)
        dont_want = get_indexes_from_list(self.cell_items, ['is_net', 'net', 'is_c_pct_sum'], exact=False)
        not_net = get_indexes_from_list(self.cell_items, ['normal', 'expanded'], exact=False)

        for x in dont_want:
            if x in row_list and x not in not_net:
                row_list.remove(x)

        return row_list

    def _get_nets_index(self):
        """
        Return a list of index numbers from self.cell_items of types 'is_net' or 'net' and not types
        'is_propstest', 'calc', 'normal', 'is_c_pct_sum', 'is_counts', 'expanded'

        Returns
        -------
        row_list : list

        """

        row_list = get_indexes_from_list(self.cell_items, ['is_net', 'net'], exact=False)
        dont_want = get_indexes_from_list(self.cell_items,
                                          ['is_propstest', 'calc', 'normal', 'is_c_pct_sum', 'is_counts', 'expanded'],
                                          exact=False)

        for x in dont_want:
            if x in row_list:
                row_list.remove(x)

        return row_list

    def _get_means_index(self):
        """
        Return a list of index numbers from self.cell_items of type 'is_mean' and not type
        'is_meanstest'

        Returns
        -------
        row_list : list

        """

        row_list = get_indexes_from_list(self.cell_items, ['is_mean'], exact=False)
        dont_want = get_indexes_from_list(self.cell_items, ['is_meanstest'], exact=False)

        for x in dont_want:
            if x in row_list:
                row_list.remove(x)

        return row_list

    def get(self, cell_types, original_order=True):
        """
        Method to get specific cell types from chains dataframe.
        Will return a copy of the PptxDataFrame instance containing only
        the requested cell types.
        Available types are 'c_pct, net, mean, test, stat'

        Parameters
        ----------
        cell_types : str
            A string of comma separated cell types to return.
        original_order: Bool
            Only relevant if more than one cell type is requested.
            If True, cell types are returned in the same order as input dataframe.
            If False, cell types will be returned in the order they are requested.

        Returns
        -------
        PptxDataFrame

        """
        method_map = {'c_pct': self._get_cpct_index,
                      'pct': self._get_cpct_index,
                      'net': self._get_nets_index,
                      'nets': self._get_nets_index,
                      'mean': self._get_means_index,
                      'means': self._get_means_index,
                      'test': self._get_propstest_index,
                      'tests': self._get_propstest_index,
                      'stats': self._get_stats_index,
                      'stat': self._get_stats_index}
        # TODO Add methods for 'stddev', 'min', 'max', 'median', 't_means'
        available_cell_types = set(method_map.keys())
        if isinstance(cell_types, str):
            cell_types = re.sub(' +', '', cell_types)
            cell_types = cell_types.split(',')
        value_test = set(cell_types).difference(available_cell_types)
        if value_test:
            raise ValueError("Cell type: {} is not an available cell type. \n Available cell types are {}".format(cell_types, available_cell_types))

        cell_types_list = []

        for cell_type in cell_types:
            cell_types_list.extend(method_map[cell_type]())

        if original_order:  cell_types_list.sort()

        new_pptx_df = self._select_categories(cell_types_list)

        return new_pptx_df


class PptxChain(object):
    """
    This class is a wrapper around Chain class to prepare for PPTX charting.

    Parameters
    ----------
    chain: quantipy.sandbox.sandbox.Chain
    is_varname_in_qtext: Bool
        Is question name is included in question text?
            False:  No question name included in question text
            True:   Question name included in question text, mask items has short question name included.
            'Full': Question name included in question text, mask items has full question name included.
    crossbreak: str
        Select a crossbreak to include in charts. Default is None
    base_type: str
        Select the base type to show in base descriptions: 'weighted' or 'unweighted'
    decimals: int
        Select the number of decimals to include from Chain.dataframe
    verbose: Bool

    """

    def __init__(self, chain, is_varname_in_qtext=True, crossbreak=None, base_type='weighted', decimals=2, verbose=True):

        self._chart_type = None
        self._sig_test = None # type: list # is updated by ._select_crossbreak()
        self.crossbreak_qtext = None # type: str # is updated by ._select_crossbreak()
        self.verbose = verbose
        self._decimals = decimals
        self._chain = chain
        self.name = chain.name
        self.xkey_levels = chain.dataframe.index.nlevels
        self.ykey_levels = chain.dataframe.columns.nlevels
        self.index_map = self._index_map()
        self.is_mask_item = chain._is_mask_item
        self.x_key_name = chain._x_keys[0]
        self.source = chain.source
        self._var_name_in_qtext = is_varname_in_qtext
        self.array_style = chain.array_style
        self.is_grid_summary = True if chain.array_style in [0,1] else False
        self.crossbreak = self._check_crossbreaks(crossbreak) if crossbreak else [BASE_COL]
        self.x_key_short_name = self._get_short_question_name()
        self.chain_df = self._select_crossbreak() # type: pd.DataFrame
        self.xbase_indexes = self._base_indexes()
        self.xbase_labels = ["Base"] if self.xbase_indexes is None else [x[0] for x in self.xbase_indexes]
        self.xbase_count = ""
        self.xbase_label = ""
        self.xbase_index = 0
        self.ybases = None
        self.select_base(base_type=base_type)
        self.base_description = "" if chain.base_descriptions is None else chain.base_descriptions
        if self.base_description[0:6].lower() == "base: ": self.base_description = self.base_description[6:]
        self._base_text = None
        self.question_text = self.get_question_text(include_varname=False)
        self.chart_df = self.prepare_dataframe()
        self.continuation_str = CONTINUATION_STR
        self.vals_in_labels = False

    def __str__(self):
        str_format = ('Table name: {}'
                      '\nX key name: {}'
                      '\nShort x key name: {}'
                      '\nGrid summary: {}'
                      '\nQuestion text: {}'
                      '\nBase description: {}'
                      '\nBase label: {}'
                      '\nBase size: {}'
                      '\nRequested crossbreak: {}'
                      '\n')
        return str_format.format(getattr(self, 'name', 'None'),
                                 getattr(self, 'x_key_name', 'None'),
                                 getattr(self, 'x_key_short_name', 'None'),
                                 getattr(self, 'is_grid_summary', 'None'),
                                 getattr(self, 'question_text', 'None'),
                                 getattr(self, 'base_description', 'None'),
                                 getattr(self, 'xbase_labels', 'None'),
                                 getattr(self, 'ybases', 'None'),
                                 getattr(self, 'crossbreak', 'None'))

    def __repr__(self):
        return self.__str__()

    @property
    def sig_test(self):

        # Get the sig testing
        sig_df = self.prepare_dataframe()
        sig_df = sig_df.get_propstest()
        _sig_test = sig_df.df.values.tolist()

        # Assume that all items in the list of sig tests has same length
        check_list = [len(x) for x in _sig_test]
        assert check_list.count(check_list[0]) == len(check_list), 'List of sig test results is not uniform'

        self._sig_test = [list(zip(*_sig_test))[i] for i in range(len(_sig_test[0]))]
        return self._sig_test

    @property
    def chart_type(self):
        if self._chart_type is None:
            self._chart_type = auto_charttype(self.chart_df.get('pct,net').df, self.array_style)

        return self._chart_type

    @chart_type.setter
    def chart_type(self, chart_type):
        self._chart_type = chart_type

    def _base_indexes(self):
        """
        Finds all categories of type 'is_counts' and 'is_c_base' and then returns
        a list of tuples holding (label, index, cell_content, value) for each base.
        Method only used when instantiating Class.
        Poppulates self.xbase_indexes

        Eg. [(u'Unweighted base', 0, ['is_counts', 'is_c_base'], 1003.0),
             (u'Base', 1, ['weight_1', 'is_weighted', 'is_counts', 'is_c_base'], 1002.9999999398246)]

        Returns
        -------
        list

        """
        cell_contents = self._chain.describe()
        if self.array_style == 0:
            row = min([k for k, va in list(cell_contents.items())
                              if any(pct in v for v in va for pct in PCT_TYPES)])
            cell_contents = cell_contents[row]

        # Find base rows
        bases = get_indexes_from_list(cell_contents, BASE_ROW, exact=False)
        skip = get_indexes_from_list(cell_contents, ['is_c_base_gross'], exact=False)
        base_indexes = [idx for idx in bases if not idx in skip] or bases

        # Show error if no base elements found
        if not base_indexes:
            #msg = "No 'Base' element found, base size will be set to None"
            #warnings.warn(msg)
            return None

        cell_contents = [cell_contents[x] for x in base_indexes]

        if self.array_style == -1 or self.array_style == 1:

            xlabels = self._chain.dataframe.index.get_level_values(-1)[base_indexes].tolist()
            base_counts = self._chain.dataframe.iloc[base_indexes, 0]

        else:

            xlabels = self._chain.dataframe.columns.get_level_values(-1)[base_indexes].tolist()
            base_counts = self._chain.dataframe.iloc[0, base_indexes]

        return list(zip(xlabels, base_indexes, cell_contents, base_counts))

    def select_base(self,base_type='weighted'):
        """
        Uses self.xbase_indexes to set
        self.xbase_label,
        self.xbase_count,
        self.xbase_index
        self.ybases

        Parameters
        ----------
        base_type: str
            String to define which base type to use: 'weighted' or 'unweighted'

        Returns
        -------
        None, sets self

        """

        if not self.xbase_indexes:
            msg = "No 'Base' element found"
            warnings.warn(msg)
            return None

        if base_type: base_type = base_type.lower()
        if not base_type in ['unweighted','weighted']:
            raise TypeError('base_type misspelled, choose weighted or unweighted')

        cell_contents = [x[2] for x in self.xbase_indexes]
        if base_type == 'weighted':
            index = [x for x, items in enumerate(cell_contents) if 'is_weighted' in items]
        else:
            index = [x for x, items in enumerate(cell_contents) if not 'is_weighted' in items]

        if not index: index=[0]

        # print "self.xbase_indexes: ", self.xbase_indexes
        total_base = self.xbase_indexes[index[0]][3]
        total_base = np.around(total_base, decimals=self._decimals)
        self.xbase_count = float2String(total_base)
        self.xbase_label = self.xbase_labels[index[0]]
        self.xbase_index = self.xbase_indexes[index[0]][1]
        self.ybases = self._get_y_bases()

    def _get_y_bases(self):
        """
        Retrieves the y-keys base label and base size from the dataframe.
        If no crossbreak is requested the output is a list with one tuple, eg. [(u'Total', '1003')].
        If eg. 'gender' is selected as crossbreak the output is [(u'Female', '487'), (u'Male', '516')]

        Only used in method select_base to poppulate self.ybases.

        Returns
        -------
        list
            List of tuples [(base label, base size)]

        """
        base_index = self.xbase_index

        if not self.is_grid_summary:

            # Construct a list of tuples with (base label, base size, test letter)
            base_values = self.chain_df.iloc[base_index, :].values.tolist()
            base_values = np.around(base_values, decimals=self._decimals).tolist()
            base_values = float2String(base_values)
            base_labels = list(self.chain_df.columns.get_level_values('Values'))
            if self._chain.sig_levels:
                base_test   = list(self.chain_df.columns.get_level_values('Test-IDs'))
                bases = list(zip(base_labels, base_values, base_test))
            else:
                bases = list(zip(base_labels, base_values))

        else: # Array summary
            # Find base columns

            # Construct a list of tuples with (base label, base size)
            base_values = self.chain_df.T.iloc[base_index,:].values.tolist()
            base_values = np.around(base_values, decimals=self._decimals).tolist()
            base_values = float2String(base_values)
            base_labels = list(self.chain_df.index.get_level_values(-1))
            bases = list(zip(base_labels, base_values))

        #print ybases
        return bases

    def _index_map(self):
        """
        Map not painted self._chain.dataframe.index with painted index into
        a list of tuples (notpainted, painted).
        If grid summary, self._chain.dataframe.columns are map'ed instead.

        Example:
        [('All', u'Base'), (1, u'Yes'), ('', u''), (2, u'No'), ('', u''), (8, u'Dont know'),
        ('', u''), ('sum', u'Totalsum')]

        Only used to poppulate self.index_map in __init__

        Returns
        -------
        list

        """
        if self._chain.painted:  # UnPaint if painted
            self._chain.toggle_labels()
        if self._chain.array_style == -1:
            unpainted_index = self._chain.dataframe.index.get_level_values(-1).tolist()
        else:
            unpainted_index = self._chain.dataframe.columns.get_level_values(-1).tolist()
        if not self._chain.painted:  # Paint if not painted
            self._chain.toggle_labels()
        if self._chain.array_style == -1:
            painted_index = self._chain.dataframe.index.get_level_values(-1).tolist()
        else:
            painted_index = self._chain.dataframe.columns.get_level_values(-1).tolist()

        return list(zip(unpainted_index, painted_index))

    def _select_crossbreak(self):
        """
        Takes self._chain.dataframe and returns a copy with only the columns
        stated in self.crossbreak.

        Only used to poppulate self.chain_df in __init__.

        Returns
        -------
        pd.DataFrame

        """
        cell_items = self._chain.cell_items.split('_')

        if not self.is_grid_summary:
            # Keep only requested columns
            if self._chain.painted: # UnPaint if painted
                self._chain.toggle_labels()

            all_columns = self._chain.dataframe.columns.get_level_values(0).tolist() # retrieve a list of the not painted column values for outer level
            if self._chain.axes[1].index(BASE_COL) == 0:
                all_columns[0] = BASE_COL # Need '@' as the outer column label

            column_selection = []
            for cb in self.crossbreak:
                column_selection = column_selection + (get_indexes_from_list(all_columns, cb))

            if not self._chain.painted: # Paint if not painted
                self._chain.toggle_labels()

            all_columns = self._chain.dataframe.columns.get_level_values(0).tolist() # retrieve a list of painted column values for outer level

            col_qtexts = [all_columns[x] for x in column_selection] # determine painted column values for requested crossbreak
            self.crossbreak_qtext = uniquify(col_qtexts) # Save q text for crossbreak in class atribute

            # Slice the dataframes columns based on requested crossbreaks
            df = self._chain.dataframe.iloc[:, column_selection]

            if len(cell_items) > 1:
                df = fill_index_labels(df)

        else:
            if len(cell_items) > 1:
                cell_contents = self._chain.describe()
                rows = [k for k, va in list(cell_contents.items())
                        if any(pct in v for v in va for pct in PCT_TYPES)]
                df_filled = fill_index_labels(fill_column_values(self._chain.dataframe))
                df = df_filled.iloc[rows, :]
                #for index in base_indexes:
                #    base_values = self.chain.dataframe.iloc[rows_bad, index].values
                #    base_column = self.chain.dataframe.columns[index]
                #    df.loc[:,[base_column]] = base_values
            else:
                df = self._chain.dataframe

        df_rounded = np.around(df, decimals=self._decimals, out=None)
        return df_rounded

    @property
    def ybase_values(self):
        """
        Returns a list with y base values picked from self.ybases.

        Returns
        -------
        list

        """
        if not hasattr(self, "_ybase_values"):
            self._ybase_values=[x[1] for x in self.ybases]
        return self._ybase_values

    @property
    def ybase_value_labels(self):
        """
        Returns a list with y base labels picked from self.ybases.

        Returns
        -------
        list

        """
        if not hasattr(self, "_ybase_value_labels"):
            self._ybase_value_labels=[x[0] for x in self.ybases]
        return self._ybase_value_labels

    @property
    def ybase_test_labels(self):
        """
        Returns a list with y base test labels picked from self.ybases.
        Eg. ['A', 'B']
        Returns
        -------
        list

        """

        if not hasattr(self, "_ybase_test_labels"):
            if self.is_grid_summary:
                self._ybase_test_labels = None
                return None
            self._ybase_test_labels=[x[2] for x in self.ybases]
        return self._ybase_test_labels

    def add_test_letter_to_column_labels(self, sep=" ", prefix=None, circumfix='()'):
        """
        Adds test letter to dataframe column labels.

        Parameters
        ----------
        sep: str
            A string to separate the column label from the test letter, default is a single space.
        prefix: str
            An optional prefix.
        circumfix: str
            A two char string used to enclose the test letter.
            Default '()'

        Returns
        -------
        None
            changes self.chain_df

        """
        # Checking input
        if circumfix is None:
            circumfix = list(('',) * 2)
        else:
            if not isinstance(circumfix, str) or len(circumfix) != 2:
                raise TypeError("Parameter circumfix needs a string with length 2")
            circumfix = list(circumfix)

        str_parameters = ['sep', 'prefix']
        for i in str_parameters:
            if not isinstance(eval(i), (str, type(None))):
                raise TypeError("Parameter {} must be a string".format(i))

        if self.is_grid_summary:
            pass

        else:

            column_labels = self.chain_df.columns.get_level_values('Values')

            # Edit labels
            new_labels_list = {}
            for x, y in zip(column_labels, self.ybase_test_labels):
                new_labels_list.update({x: x + (sep or '') + circumfix[0] + (prefix or '') + y + circumfix[1]})

            self.chain_df = self.chain_df.rename(columns=new_labels_list)

    def place_vals_in_labels(self, base_position=0, orientation='side', values=None, sep=" ", prefix="n=", circumfix="()", setup='if_differs'):
        """
        Takes values from input list and adds them to self.chain_df's categories,
        Meaning rows if grid summary, otherwise columns.

        Can be used to insert base values in side labels for a grid summary.

        Parameters
        ----------
        base_position: for future usage
        orientation: for future usage
        values: list
            a list with same number of values as categories in self.chain_df
        sep: str
            A string to separate the categories from the insert, default is a single space.
        prefix: str
            A prefix to add to the insert. Default 'n='
        circumfix: str
            A two char string used to enclose the insert.
            Default '()'
        setup: str
            A string telling when to insert value ('always', 'if_differs', 'never')

        Returns
        -------
        None
            Changes self.chain_df

        """
        if setup=='never': return

        # Checking input
        if circumfix is None:
            circumfix = list(('',) * 2)
        else:
            if not isinstance(circumfix, str) or len(circumfix) != 2:
                raise TypeError("Parameter circumfix needs a string with length 2")
            circumfix = list(circumfix)

        str_parameters = ['sep', 'prefix', 'orientation', 'setup']
        for i in str_parameters:
            if not isinstance(eval(i), (str, type(None))):
                raise TypeError("Parameter {} must be a string".format(i))

        valid_orientation = ['side', 'column']
        if orientation not in valid_orientation:
            raise ValueError("Parameter orientation must be either of {}".format(valid_orientation))

        valid_setup  = ['always', 'if_differs', 'never']
        if setup not in valid_setup:
            raise ValueError("Parameter setup must be either of {}".format(valid_setup))

        if self.is_grid_summary:
            if (len(uniquify(self.ybase_values))>1 and setup=='if_differs') or setup=='always':

                # grab row labels
                index_labels = self.chain_df.index.get_level_values(-1)

                # Edit labels
                new_labels_list = {}
                for x, y in zip(index_labels, values):
                    new_labels_list.update({x: x + (sep or '') + circumfix[0]+ (prefix or '') + str(y) + circumfix[1]})

                self.chain_df = self.chain_df.rename(index=new_labels_list)
                self.vals_in_labels = True

        else:

            # grab row labels
            index_labels = self.chain_df.columns.get_level_values('Values')

            # Edit labels
            new_labels_list = {}
            for x, y in zip(index_labels, values):
                new_labels_list.update({x: x + (sep or '') + circumfix[0] + (prefix or '') + str(y) + circumfix[1]})

            # Saving column index for level 'Question' in case it accidentially gets renamed
            index_level_values = self.chain_df.columns.get_level_values('Question')

            self.chain_df = self.chain_df.rename(columns=new_labels_list)

            # Returning column index for level 'Question' in case it got renamed
            self.chain_df.columns.set_levels(index_level_values, level='Question', inplace=True)

            self.vals_in_labels = True

    @property
    def base_text(self):
        return self._base_text

    @base_text.setter
    def base_text(self, base_text):
        self._base_text = base_text

    def set_base_text(self, base_value_circumfix="()", base_label_suf=":", base_description_suf=" - ", base_value_label_sep=", ", base_label=None):
        """
        Returns the full base text made up of base_label, base_description and ybases, with some delimiters.
        Setup is "base_label + base_description + base_value"

        Parameters
        ----------
        base_value_circumfix: str
            Two chars to surround the base value
        base_label_suf: str
            A string to add after the base label
        base_description_suf: str
            A string to add after the base_description
        base_value_label_sep: str
            A string to separate base_values if more than one
        base_label: str
            An optional string to use instead of self.xbase_label

        Returns
        -------
        str
            Sets self._base_text

        """
        # Checking input
        if base_value_circumfix is None:
            base_value_circumfix = list(('',) * 2)
        else:
            if not isinstance(base_value_circumfix, str) or len(base_value_circumfix) != 2:
                raise TypeError("Parameter base_value_circumfix needs a string with length 2")
            base_value_circumfix = list(base_value_circumfix)

        str_parameters = ['base_label_suf', 'base_description_suf', 'base_value_label_sep', 'base_label']
        for i in str_parameters:
            if not isinstance(eval(i), (str, type(None))):
                raise TypeError("Parameter {} must be a string".format(i))

        # Base_label
        if base_label is None:
            base_label = self.xbase_label

        if self.base_description:
            base_label = "{}{}".format(base_label,base_label_suf or '')
        else:
            base_label = "{}".format(base_label)

        # Base_values
        if self.xbase_indexes:
            base_values = self.ybase_values[:]
            for index, base in enumerate(base_values):
                base_values[index] = "{}{}{}".format(base_value_circumfix[0], base, base_value_circumfix[1])
        else:
            base_values=[""]

        # Base_description
        base_description = ""
        if self.base_description:
            if len(self.ybases) > 1 and not self.vals_in_labels and self.array_style==-1:
                base_description = "{}{}".format(self.base_description, base_description_suf or '')
            else:
                base_description = "{} ".format(self.base_description)

        # ybase_value_labels
        base_value_labels = self.ybase_value_labels[:]

        # Include ybase_value_labels in base values if more than one base value
        base_value_text = ""
        if base_value_label_sep is None: base_value_label_sep = ''
        if len(base_values) > 1:
            if not self.vals_in_labels:
                if self.xbase_indexes:
                    for index, label in enumerate(zip(base_value_labels, base_values)):
                        base_value_text="{}{}{} {}".format(base_value_text, base_value_label_sep, label[0], label[1])
                    base_value_text = base_value_text[len(base_value_label_sep):]
                else:
                    for index, label in enumerate(base_value_labels):
                        base_value_text="{}{}{}".format(base_value_text, base_value_label_sep, label)
                    base_value_text = base_value_text[len(base_value_label_sep):]
            else:
                if not self.is_grid_summary:
                    base_value_text = "({})".format(self.xbase_count)

        # Final base text
        if not self.is_grid_summary:
            if len(self.ybases) == 1:
                if base_description:
                    base_text = "{} {}{}".format(base_label,base_description,base_values[0])
                else:
                    base_text = "{} {}".format(base_label, base_values[0])
            else:
                if base_description:
                    base_text = "{} {}{}".format(base_label,base_description,base_value_text)
                else:
                    base_text = "{} {}".format(base_label,base_value_text)
        else: # Grid summary
            if len(uniquify(self.ybase_values)) == 1:
                if base_description:
                    base_text = "{} {}{}".format(base_label,base_description,base_values[0])
                else:
                    base_text = "{} {}".format(base_label, base_values[0])
            else:
                if base_description:
                    base_text = "{} {}".format(base_label, base_description)
                else:
                    base_text = ""

        self._base_text = base_text

    def _check_crossbreaks(self, crossbreaks):
        """
        Checks the crossbreaks input for duplicates and that crossbreak exist in the chain.

        Parameters
        ----------
        crossbreaks: list
            List of strings

        Returns
        -------
        list
            The crossbreaks list stripped for duplicates and not existing crossbreaks

        """
        if not isinstance(crossbreaks, list):
            crossbreaks = [crossbreaks]

        if not self.is_grid_summary:
            for cb in crossbreaks[:]:
                if cb not in self._chain.axes[1]:
                    crossbreaks.remove(cb)
                    if self.verbose:
                        msg = 'Requested crossbreak: \'{}\' is not found for chain \'{}\' and will be ignored'.format(cb, chain.name)
                        warnings.warn(msg)
            if crossbreaks == []: crossbreaks = None
        else:
            pass # just ignore checking if Grid Summary
            #crossbreaks = None

        return uniquify(crossbreaks) if crossbreaks is not None else [BASE_COL]

    def _get_short_question_name(self):
        """
        Retrieves 'short' question name.
        Used in __init__ to poppulate self.x_key_short_name

        Returns
        -------
        str

        """
        if not self.is_grid_summary: # Not grid summary
            if self.is_mask_item: # Is grid slice
                pattern = '(?<=\[\{).*(?=\}\])'
                result_list = re.findall(pattern, self.x_key_name)
                if result_list:
                    return result_list[0] # TODO Hmm what if grid has more than one level
                else:
                    return self.x_key_name

            else: # Not grid slice
                return self.x_key_name

        else: # Is grid summary
            find_period = self.x_key_name.find('.')
            if find_period > -1:
                return self.x_key_name[:find_period]
            else:
                return self.x_key_name

    def get_question_text(self, include_varname=False):
        """
        Retrieves the question text from the dataframe.
        If include_varname=True then the question text will be prefixed the var name.

        Parameters
        ----------
        include_varname: Bool

        Returns
        -------
        str

        """
        # Get variable name
        var_name = self.x_key_name
        if self.is_mask_item:
            if self._var_name_in_qtext == True:
                var_name = self.x_key_short_name

        # Get question text, stripped for variable name
        question_text = self.chain_df.index[0][0]
        if self._var_name_in_qtext:
            question_text = question_text[len(var_name) + 2:]

        # Include the full question text for mask items if missing
        if self.is_mask_item:
            question_text = self._mask_question_text(question_text)

        # Add variable name to question text if requested
        if include_varname:
            question_text = '{}. {}'.format(self.x_key_short_name, question_text)

        # Remove consecutive line breaks and spaces
        question_text = re.sub('\n+', '\n', question_text)
        question_text = re.sub('\r+', '\r', question_text)
        question_text = re.sub(' +', ' ', question_text)

        return question_text.strip()

    def _mask_question_text(self, question_text):
        """
        If chain is a mask item (a grid slice), then the parent question text
        is added to question text unless already included.

        Final question text in the form "parent_question_text - mask_question_text"

        Only used in self.get_question_text().

        Parameters
        ----------
        question_text: str

        Returns
        -------
        str

        """
        if self.source == "native":
            if self.is_mask_item:
                meta = self._chain._meta
                cols = meta['columns']
                masks = meta['masks']
                parent = list(cols[self.x_key_name]['parent'].keys())[0].split('@')[-1]
                m_text = masks[parent]['text']
                text = m_text.get('x edit', m_text).get(meta['lib']['default text'])
                if not text.strip() in question_text:
                    question_text = '{} - {}'.format(text, question_text)

        return question_text

    def prepare_dataframe(self):
        """
        Prepares self.chain_df for charting, that is removes all outer levels
        and prepares the dataframe for PptxPainter.

        Returns
        -------
        pd.DataFrame
            An edited copy of self.chain_df

        """
        # Strip outer level
        df = strip_levels(self.chain_df, rows=0, columns=0)
        df = strip_levels(df, columns=1)

        # Strip HTML TODO Is 'Strip HTML' at all nessecary?

        # Check that the dataframe is numeric
        all_numeric = all(df.applymap(lambda x: isinstance(x, (int, float)))) == True
        if not all_numeric:
            df = as_numeric(df)

        # For rows that are type '%' divide by 100
        indexes = []
        cell_contents = self._chain.describe()
        if self.is_grid_summary:
            colpct_row = min([k for k, va in list(cell_contents.items())
                              if any(pct in v for v in va for pct in PCT_TYPES)])
            cell_contents = cell_contents[colpct_row]

        for i, row in enumerate(cell_contents):
            for type in row:
                for pct_type in PCT_TYPES:
                    if type == pct_type:
                        indexes.append(i)
        if not self.is_grid_summary:
            df.iloc[indexes] /= 100
        else:
            df.iloc[:, indexes] /= 100

        # Make a PptxDataFrame instance
        chart_df = PptxDataFrame(df, cell_contents, self.array_style)
        # Choose a basic Chart type that will fit dataframe TODO Move this to init of Class PptxDataFrame
        chart_df.chart_type = auto_charttype(df, self.array_style)

        return chart_df

