# encoding: utf-8
import re
import warnings
import numpy as np
import pandas as pd
from quantipy.sandbox.sandbox import Chain
import topy.core.ygpy.tools as t
t.use_encoding('utf-8')

BASE_COL = '@'
BASE_ROW = ['is_counts', 'is_c_base']
PCT_TYPES = ['is_c_pct', 'is_r_pct']
CONTINUATION_STR = "(continued {})"
MAX_PIE_ELMS = 4


def float2String(input, ndigits=0):
    """
    Round and converts the input, if int/float or list of, to a string.
    :param
        input: int/float or list of int/float
        ndigits: number of decimals to round to
    :return:
        output: string or list of strings depeneding on the input
    """
    output = input
    if not isinstance(input, list):
        output = [output]
    output = map(lambda x: round(x, ndigits), output)
    output = map(int, output)
    output = map(str, output)
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
    Can be used with all DP-systems
    :param
    rows: Int, default None
        Row index to remove
    columns: Int, default None
        Column index to remove
    :return:
    df_strip: A pandas.Dataframe
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
    Can be used with all DP-systems
    :param
    df: Pandas.Dataframe
        A dataframe of any structure, also multiindex
    :return:
    df: A pandas.Dataframe
        dataframe with values as float
    """

    if not df.values.dtype in ['float64', 'int64']:
        data = [[float(str(value).replace(',','.').replace('%','').replace('-','0').replace('*','0')) for value in values] for values in df.values]
        df = pd.DataFrame(data, index=df.index, columns=df.columns)
    return df.copy()


def is_grid_slice(chain):
    """
    Returns True if chain is a grid slice
    :param
        chain: the chain instance
    :return: True id grid slice
    """
    pattern = '\[\{.*?\}\].'
    found = re.findall(pattern, chain.name)
    if len(found) > 0 and chain._array_style == -1:
        return True


def paint(chains, text_key, *args, **kwargs):
    text_key = text_key['x'][0]
    chains.paint_all(text_key=text_key, *args, **kwargs)

    for chain in chains:
        df = chain.dataframe
        # question text as it is in the dataframe, but without question name
        xkey = chain._x_keys[0]
        question_text = chain.dataframe.index[0][0][len(xkey) + 2:]
        # need the xkey question text from ['masks'] if grid slice
        if is_grid_slice(chain):
            # search for the text before and after [{}]. in chain.name
            pattern = '.*(?=\[\{)|(?<=\}\])\..*'
            found = re.findall(pattern, chain.name)
            # Construct grid name from found
            grid_name = found[0] + found[-1]
            # Find the grid mask question text
            grid_text = chain._meta['masks'][grid_name]['text'][text_key]
            if grid_text in question_text: continue
            # Construct the new grid slice question text that includes grid mask question text
            question_text = '{}. {} - {}'.format(chain.name, grid_text, question_text)
            # Replace xkey index label with the new question text
            chain.dataframe.rename({df.index.values[0][0]: question_text}, inplace = True)

        # For grids I dont want the question text included in element labels
        if chain.array_style > -1:
            for index, row in enumerate(chain.dataframe.index):
                element_label = row[1]
                if question_text in element_label:
                    element_label = element_label.replace(question_text, "").strip()
                    if element_label[0] == "-":
                        element_label = element_label[1:].strip()

                chain.dataframe.rename({df.index.values[index][1]: element_label}, inplace=True)


def get_indexes_from_list(lst, find, exact=True):
    """
    Helper function that search for element in a list and
    returns a list of indexes for element match
    E.g. get_indexes_from_list([1,2,3,1,5,1], 1) returns [0,3,5]
    :param
        lst: a list
        find: the element to find. Can be a list
        exact: If False the index are returned if find in value
    :return: a list of ints
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
    :param
        df: a Pandas Dataframe, not multiindex
        array_style: array_style as returned from Chain Class
    :return: charttype ('bar_clustered', 'bar_stacked', 'bar_stacked_100', 'pie')
    """
    if array_style == -1: # Not array summary
        chart_type = 'bar_clustered'
        if len(df.index.get_level_values(-1)) <= max_pie_elms:
            if len(df.columns.get_level_values(-1)) == 1:
                chart_type = 'pie'
    else: # Array Sum
        chart_type = 'bar_stacked_100'
        # TODO _auto_charttype - return 'bar_stacked' if rows not sum to 100

    return chart_type

class PptxDataFrame(pd.DataFrame):
    """
    Adds some PPTX friendly methods to the standard pd.DataFrame class
    """

    # TODO class PptxDataFrame - Add more methods

    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False):
        super(PptxDataFrame, self).__init__(data, index, columns, dtype, copy)
        self.array_style = None
        self.cell_contents = None
        self.__frames = []
        self.chart_type = None # TODO PptxDataFrame - How to do: if a user sets chart_type it is auto checked for correctnes

    def make_copy(self, data=None, index=None, columns=None):

        val = data if isinstance(data, (np.ndarray)) else self.values
        idx = index if isinstance(index, (pd.Index, pd.Int64Index, pd.MultiIndex)) else self.index
        col = columns if isinstance(columns, (pd.Index, pd.Int64Index, pd.MultiIndex)) else self.columns
        df_copy = PptxDataFrame(data=val, index=idx, columns=col)
        df_copy.cell_contents = self.cell_contents
        df_copy.array_style = self.array_style
        df_copy.chart_type = self.chart_type

        return df_copy

    def get_cpct(self):
        row_list = get_indexes_from_list(self.cell_contents, 'is_c_pct', exact=False)
        dont_want = get_indexes_from_list(self.cell_contents, ['is_net','net','is_c_pct_sum'], exact=False)

        for x in dont_want:
            if x in row_list:
                row_list.remove(x)

        if self.array_style == -1:
            df_copy=self.iloc[row_list]
        else:
            df_copy = self.iloc[:,row_list]

        pptx_df_copy = self.make_copy(data=df_copy.values, index=df_copy.index, columns=df_copy.columns)
        pptx_df_copy.chart_type = auto_charttype(pptx_df_copy, pptx_df_copy.array_style)
        cell_contents = pptx_df_copy.cell_contents
        pptx_df_copy.cell_contents = [cell_contents[i] for i in row_list]

        return pptx_df_copy

    def get_nets(self):
        row_list = get_indexes_from_list(self.cell_contents, ['is_net','net'], exact=False)
        dont_want = get_indexes_from_list(self.cell_contents, ['is_propstest'], exact=False)

        for x in dont_want:
            if x in row_list:
                row_list.remove(x)

        if self.array_style == -1:
            df_copy = self.iloc[row_list]
        else:
            df_copy = self.iloc[:, row_list]

        pptx_df_copy = self.make_copy(data=df_copy.values, index=df_copy.index, columns=df_copy.columns)
        pptx_df_copy.chart_type = auto_charttype(pptx_df_copy, pptx_df_copy.array_style)
        cell_contents = pptx_df_copy.cell_contents
        pptx_df_copy.cell_contents = [cell_contents[i] for i in row_list]

        return pptx_df_copy

    def get(self, cell_types, sort=False):
        """
        Method to get specific elements from chains dataframe
        :param
            cel_types: A comma separated list of cell types to return. Available types are 'c_pct,net'
            sort: Sort the elements ascending or decending. Str 'asc', 'dsc' or False
        :return: df_copy, a Pandas dataframe. Element types will be returned in the order they are requested
        """
        method_map = {'c_pct': self.get_cpct,
                      'net': self.get_nets}
        # TODO Add methods for 'mean', 'stddev', 'min', 'max', 'median', 't_props', 't_means'
        available_celltypes = set(method_map.keys())
        if isinstance(cell_types, basestring):
            cell_types = re.sub(' +', '', cell_types)
            cell_types = cell_types.split(',')
        value_test = set(cell_types).difference(available_celltypes)
        if value_test:
            raise ValueError("Cell type: {} is not an available cell type. \n Available cell types are {}".format(cell_types, available_celltypes))

        frames = []
        cell_contents = []
        for cell_type in cell_types:
            frame = method_map[cell_type]()
            frames.append(frame)
            cell_contents += frame.cell_contents
        new_df=pd.concat(frames, axis=0 if self.array_style==-1 else 1)

        pptx_df = self.make_copy(data=new_df.values, index=new_df.index, columns=new_df.columns)
        pptx_df.chart_type = auto_charttype(pptx_df, pptx_df.array_style)
        pptx_df.cell_contents = cell_contents

        return pptx_df


class PptxChain(object):
    """
    This class is a wrapper around Chain() class to prepare for PPTX charting
    """

    def __init__(self, chain, is_varname_in_qtext=True, crossbreak=None, base_type='weighted', verbose=True):
        """
        :param
            chain: An instance of Chain class
            is_varname_in_qtext: Is var name included in the painted chain dataframe? (False, True, 'full', 'ignore')
            crossbreak:
        """
        self.verbose = verbose
        self.chain = chain
        self.index_map = self._index_map()
        self.is_mask_item = chain._is_mask_item
        self.x_key_name = chain._x_keys[0]
        self.source = chain.source
        self._var_name_in_qtext = is_varname_in_qtext
        self.array_style = chain.array_style
        self.is_grid_summary = True if chain.array_style in [0,1] else False
        if crossbreak:
            if not isinstance(crossbreak, list):
                crossbreak = [crossbreak]
            crossbreak = self._check_crossbreaks(chain, crossbreak)
        else:
            crossbreak = [BASE_COL]
        self.name = chain.name
        self.x_key_short_name = self._get_short_question_name()
        self.crossbreak = crossbreak
        self.xbase_indexes = self._base_indexes()
        self.xbase_labels = ["Base"] if self.xbase_indexes == False else [x[0] for x in self.xbase_indexes]
        self.xbase_count = ""
        self.xbase_label = ""
        self.xbase_index = 0
        self.select_base(base_type=base_type)
        self.chain_df = chain.dataframe if self.is_grid_summary else self._select_crossbreak()
        self.base_description = "" if chain.base_descriptions == None else chain.base_descriptions
        if self.base_description[0:6] == "Base: ": self.base_description = self.base_description[6:]
        self.question_text = self.get_question_text(include_varname=False)
        self.crossbreaks_qtext = []
        self.ybases = self._get_bases()
        self.xkey_levels = chain.dataframe.index.nlevels
        self.ykey_levels = chain.dataframe.columns.nlevels
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

    def select_base(self,base_type='weighted'):
        """
        Sets self.xbase_label and self.xbase_count
        :param base_type: str
        :return: None set self
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
        self.xbase_count = float2String(total_base)
        self.xbase_label = self.xbase_labels[index[0]]
        self.xbase_index = self.xbase_indexes[index[0]][1]

    def _base_indexes(self):
        """
        Returns label and index of bases found in x keys as list
        :return: list - self.xbase_indexes
        """

        cell_contents = self.chain.describe()
        if self.array_style == 0: cell_contents = cell_contents[0]

        # Find base rows
        base_indexes = get_indexes_from_list(cell_contents, BASE_ROW, exact=False)
        # Show error if no base elements found
        if not base_indexes:
            #msg = "No 'Base' element found, base size will be set to None"
            #warnings.warn(msg)
            self.xbase_indexes = False
            return None

        cell_contents = [cell_contents[x] for x in base_indexes]

        if self.array_style == -1 or self.array_style == 1:

            xlabels = self.chain.dataframe.index.get_level_values(-1)[base_indexes].tolist()
            base_counts = self.chain.dataframe.iloc[base_indexes, 0]

        else:

            xlabels = self.chain.dataframe.columns.get_level_values(-1)[base_indexes].tolist()
            base_counts = self.chain.dataframe.iloc[0,base_indexes]

        return zip(xlabels, base_indexes, cell_contents, base_counts)

    def _index_map(self):
        """
        Map not painted index with painted index into a list of tuples (notpainted, painted)
        :return:
        """
        if self.chain.painted:  # UnPaint if painted
            self.chain.toggle_labels()
        if self.chain.array_style == -1:
            unpainted_index = self.chain.dataframe.index.get_level_values(-1).tolist()
        else:
            unpainted_index = self.chain.dataframe.columns.get_level_values(-1).tolist()
        if not self.chain.painted:  # Paint if not painted
            self.chain.toggle_labels()
        if self.chain.array_style == -1:
            painted_index = self.chain.dataframe.index.get_level_values(-1).tolist()
        else:
            painted_index = self.chain.dataframe.columns.get_level_values(-1).tolist()

        return zip(unpainted_index, painted_index)

    def _select_crossbreak(self):
        """
        Takes self.chain.dataframe and returns only the columns stated in self.crossbreak
        :return:
        """

        if not self.is_grid_summary:
            # Keep only requested columns
            if self.chain.painted: # UnPaint if painted
                self.chain.toggle_labels()

            all_columns = self.chain.dataframe.columns.get_level_values(0).tolist() # retrieve a list of the not painted column values for outer level
            if self.chain.axes[1].index(BASE_COL) == 0:
                all_columns[0] = BASE_COL # Need '@' as the outer column label

            column_selection = []
            for cb in self.crossbreak:
                column_selection = column_selection + (get_indexes_from_list(all_columns, cb))

            if not self.chain.painted: # Paint if not painted
                self.chain.toggle_labels()

            all_columns = self.chain.dataframe.columns.get_level_values(0).tolist() # retrieve a list of painted column values for outer level

            col_qtexts = [all_columns[x] for x in column_selection] # determine painted column values for requested crossbreak
            self.crossbreaks_qtext = uniquify(col_qtexts) # Save q text for crossbreak in class atribute

            # Slice the dataframes columns based on requested crossbreaks
            df = self.chain.dataframe.iloc[:, column_selection]

        return df

    @property
    def ybase_values(self):
        if not hasattr(self, "_base_values"):
            self._base_values=[x[1] for x in self.ybases]
        return self._base_values

    @property
    def ybase_value_labels(self):
        if not hasattr(self, "_base_value_labels"):
            self._base_value_labels=[x[0] for x in self.ybases]
        return self._base_value_labels

    def place_vals_in_labels(self, base_position=0, orientation='side', drop_base=False, values=None, sep=" ", prefix="n=", circumfix="()", setup='if_differs'):
        """
        Takes values from a given column or row and inserts it to the df's row or column labels.
        Can be used to insert base values in side labels for a grid summary
        :param
        base_position: Int, Default 0
            Which row/column to pick the base element from
        orientation: Str: 'side' or 'column', Default 'side'
            Add base to row or column labels.
        drop_base: Bool, Default True
            Removes the base column/row just added to labels
        values: list
            the list of values to insert
        sep: str
            the string to use to separate the value from the label
        prefix: str
            A string to insert as a prefix to the label
        circumfix: str
            A character couple to surround the value
        setup: str
            A string telling when to insert value ('always', 'if_differs', 'never')
        """
        if setup=='never': return

        circumfix = list(circumfix)

        if self.is_grid_summary:
            if (len(uniquify(self.ybase_values))>1 and setup=='if_differs') or setup=='always':

                # grab row labels
                index_labels = self.chain_df.index.get_level_values(-1)

                # Edit labels
                new_labels_list = {}
                for x, y in zip(index_labels, values):
                    new_labels_list.update({x: x + sep + circumfix[0]+ prefix + str(y) + circumfix[1]})

                self.chain_df = self.chain_df.rename(index=new_labels_list)
                self.vals_in_labels = True

        else:

            # grab row labels
            index_labels = self.chain_df.columns.get_level_values('Values')

            # Edit labels
            new_labels_list = {}
            for x, y in zip(index_labels, values):
                new_labels_list.update({x: x + sep + circumfix[0] + prefix + str(y) + circumfix[1]})

            self.chain_df = self.chain_df.rename(columns=new_labels_list)
            self.vals_in_labels = True

    def base_text(self, base_value_circumfix="()", base_label_suf=":", base_description_suf=" - ", base_value_label_sep=", "):
        """
        Returns the full base text made up of base_label, base_description and ybases, with some delimiters
        :param
            base_value_circumfix: chars to surround the base value
            base_label_suf: char to put after base_label
            base_description_suf: When more than one column, use this char after base_description
            base_value_label_sep: char to separate column label, if more than one
        :return:
        """
        # Base_label
        base_label = self.xbase_label

        if self.base_description:
            base_label = u"{}{}".format(base_label,base_label_suf)
        else:
            base_label = u"{}".format(base_label)

        # Base_values
        if self.xbase_indexes:
            base_value_circumfix = list(base_value_circumfix)
            base_values = self.ybase_values[:]
            for index, base in enumerate(base_values):
                base_values[index] = u"{}{}{}".format(base_value_circumfix[0], base, base_value_circumfix[1])
        else:
            base_values=[""]

        # Base_description
        base_description = ""
        if self.base_description:
            if len(self.ybases) > 1 and not self.vals_in_labels and self.array_style==-1:
                base_description = u"{}{}".format(self.base_description, base_description_suf)
            else:
                base_description = u"{} ".format(self.base_description)

        # ybase_value_labels
        base_value_labels = self.ybase_value_labels[:]

        # Include ybase_value_labels in base values if more than one base value
        base_value_text = ""
        if len(base_values) > 1:
            if not self.vals_in_labels:
                if self.xbase_indexes:
                    for index, label in enumerate(zip(base_value_labels, base_values)):
                        base_value_text=u"{}{}{} {}".format(base_value_text, base_value_label_sep, label[0], label[1])
                    base_value_text = base_value_text[len(base_value_label_sep):]
                else:
                    for index, label in enumerate(base_value_labels):
                        base_value_text=u"{}{}{}".format(base_value_text, base_value_label_sep, label)
                    base_value_text = base_value_text[len(base_value_label_sep):]
            else:
                if not self.is_grid_summary:
                    base_value_text = u"({})".format(self.xbase_count)

        # Final base text
        if not self.is_grid_summary:
            if len(self.ybases) == 1:
                if base_description:
                    base_text = u"{} {}{}".format(base_label,base_description,base_values[0])
                else:
                    base_text = u"{} {}".format(base_label, base_values[0])
            else:
                if base_description:
                    base_text = u"{} {}{}".format(base_label,base_description,base_value_text)
                else:
                    base_text = u"{} {}".format(base_label,base_value_text)
        else: # Grid summary
            if len(uniquify(self.ybase_values)) == 1:
                if base_description:
                    base_text = u"{} {}{}".format(base_label,base_description,base_values[0])
                else:
                    base_text = u"{} {}".format(base_label, base_values[0])
            else:
                if base_description:
                    base_text = u"{} {}".format(base_label,base_description)
                else:
                    base_text = ""


#        # Final base text
#        if not self.is_grid_summary:
#            if len(self.ybases) == 1:
#                if base_description:
#                    base_text = "{} {} {}".format(base_label,base_description,ybase_values[0])
#                else:
#                    base_text = "{} {}".format(base_label, ybase_values[0])
#            else:
#                if base_description:
#                    base_text = "{} {}{}".format(base_label,base_description,base_value_text)
#                else:
#                    base_text = "{} {}".format(base_label,base_value_text)
#        else: # Grid Summary
#            if len(uniquify(ybase_values))== 1:
#                if base_description:
#                    base_text = "{} {} {}".format(base_label, base_description, ybase_values[0])
#                else:
#                    base_text = "{} {}".format(base_label, ybase_values[0])
#            else:
#                if base_description:
#                    base_text = "{} {}".format(base_label, base_description)
#                else:
#                    base_text = ""

        #print (base_text)
        return base_text

    def _check_crossbreaks(self, chain, crossbreaks):
        """
        Checks the existence of the requested crossbreaks
        :param chain:
        :param crossbreaks:
        :return:
        """
        if not self.is_grid_summary:
            for cb in crossbreaks[:]:
                if cb not in chain.axes[1]:
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
        Retrieves question name
        :param
            chain: the chain instance
        :return: question_name (as string)
        """
        if not self.is_grid_summary: # Not grid summary
            if self.is_mask_item: # Is grid slice
                pattern = '(?<=\[\{).*(?=\}\])'
                result_list = re.findall(pattern, self.x_key_name)
                if result_list:
                    return result_list[0] # TODO Hmm what if more than one level grid
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
        Assumes that self.chain_df has one of the following setups  regarding question name self._var_name_in_qtext:
            False:  No question name included in question text
            True:   Question text included in question text, mask items has short question name included.
            'Full': Question text included in question text, mask items has full question name included.

        :param
            include_varname: Include question name in question text (bool)
        :return: question_txt (as string)
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
            question_text = u'{}. {}'.format(self.x_key_short_name, question_text)

        # Remove consecutive line breaks and spaces
        question_text = re.sub('\n+', '\n', question_text)
        question_text = re.sub('\r+', '\r', question_text)
        question_text = re.sub(' +', ' ', question_text)

        return question_text.strip()

    def _mask_question_text(self, question_text):
        """
        Adds to a mask items question text the array question text
        if not allready there
        :param
            question_text: the question text from the mask item
        :return:
            question_text appended the array question text
        """
        if self.source == "native":
            meta=self.chain._meta
            cols = meta['columns']
            masks = meta['masks']
            if self.is_mask_item:
                parent = cols[self.x_key_name]['parent'].keys()[0].split('@')[-1]
                m_text = masks[parent]['text']
                text = m_text.get('x edit', m_text).get(meta['lib']['default text'])
                if not text.strip() in question_text:
                    question_text = u'{} - {}'.format(text, question_text)

        return question_text

    def _is_base_row(self, row):
        """
        Return True if Row is a Base row
        :param
            row: The row to check (list)
        :return:
            True or False
        """
        for item in BASE_ROW:
            if item not in row:
                return False
        return True

    def _get_bases(self):
        """
        Retrieves the base label and base size from the dataframe
        :param chain: the chain instance
        :return: ybases - list of tuples [(base label, base size)]
        """

        base_index = self.xbase_index

        if not self.is_grid_summary:

            # Construct a list of tuples with (base label, base size)
            base_values = self.chain_df.iloc[base_index, :].values.tolist()
            base_values = float2String(base_values)
            base_labels = list(self.chain_df.columns.get_level_values('Values'))
            bases = zip(base_labels, base_values)

        else: # Array summary
            # Find base columns

            # Construct a list of tuples with (base label, base size)
            base_values = self.chain_df.T.iloc[base_index,:].values.tolist()
            base_values = float2String(base_values)
            base_labels = list(self.chain_df.index.get_level_values(-1))
            bases = zip(base_labels, base_values)

        #print ybases
        return bases

    # def _get_rowbase_label(self):
    #     """
    #     Retrieves the base label from the dataframe
    #     :param
    #         chain: the chain instance
    #
    #     :return: base_label - Str
    #     """
    #
    #     # Paint if not painted
    #     if not self.chain.painted:
    #         self.chain.toggle_labels()
    #         self.chain_df = self.chain.dataframe
    #
    #     if self.xbase_indexes:
    #         if not self.is_grid_summary:
    #             cell_contents = self.chain.describe()
    #             row = get_indexes_from_list(cell_contents, BASE_ROW, exact=False) #TODO Cater for more that one base row
    #             base_label = self.chain_df.index.get_level_values(-1)[row[0]]
    #         else:
    #             cell_contents = self.chain.describe()[0]
    #             row = get_indexes_from_list(cell_contents, BASE_ROW, exact=False) #TODO Cater for more that one base row
    #             base_label = self.chain_df.T.index.get_level_values(-1)[row[0]]
    #     else:
    #         base_label = 'Base'
    #     return base_label

    def prepare_dataframe(self):
        """
        Prepares the dataframe for charting
        :param
            crossbreak
        :return: copy of chain.dataframe containing only rows and cols that are to be charted
        """

        #if not self.is_grid_summary:
            # Keep only requested columns
            #if self.chain.painted: # UnPaint if painted
            #    self.chain.toggle_labels()
            #    self.chain_df = self.chain.dataframe


            # all_columns = self.chain_df.columns.get_level_values(0).tolist() # retrieve a list of the not painted column values for outer level
            # if self.chain.axes[1].index(BASE_COL) == 0: # Need '@' instead of the outer row value
            #     all_columns[0] = BASE_COL
            #
            # if crossbreak == None:
            #     crossbreak = [BASE_COL]
            #
            # column_selection = []
            # for cb in crossbreak:
            #     column_selection = column_selection + (get_indexes_from_list(all_columns, cb))
            #
            # if not self.chain.painted: # Paint if not painted
            #     self.chain.toggle_labels()
            #     self.chain_df = self.chain.dataframe
            #
            # all_columns = self.chain_df.columns.get_level_values(0).tolist() # retrieve a list of painted column values for outer level
            #
            # col_qtexts = [all_columns[x] for x in column_selection] # determine painted column values for requested crossbreak
            # self.crossbreaks_qtext = uniquify(col_qtexts) # Save q text for crossbreak in class atribute

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
        if not self.is_grid_summary:
            cell_contents = self.chain.describe()
        else:
            cell_contents = self.chain.describe()[0]

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
        chart_df = PptxDataFrame(data=df.values, index=df.index, columns=df.columns)
        if not self.is_grid_summary:
            chart_df.cell_contents = self.chain.describe() # TODO Is this okay? to initialize a class attribute outside of the class
        else:
            chart_df.cell_contents = self.chain.describe()[0]
        chart_df.array_style = self.chain.array_style

        # Choose a basic Chart type that will fit dataframe
        chart_df.chart_type = auto_charttype(chart_df, chart_df.array_style)

        return chart_df

