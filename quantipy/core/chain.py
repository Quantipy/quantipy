import pickle
from collections import defaultdict
from .helpers import functions as helpers
from .view import View
import pandas as pd
import copy

class Chain(defaultdict):
    """
    Container class that holds ordered Link defintions and associated Views.

    The Chain object is a subclassed dict of list where each list contains one
    or more View aggregations of a Stack. It is an internal class included and
    used inside the Stack object. Users can interact with the data directly
    through the Chain or through the related Cluster object.
    """

    def __init__(self, name=None):
        super(Chain, self).__init__(Chain)
        self.name = name
        self.orientation = None
        self.source_name = None
        self.source_type = None
        self.source_length = None
        self.len_of_axis = None
        self.content_of_axis = None
        self.data_key = None
        self.filter = None
        self.views = None
        # self.view_sizes = None
        # self.view_lengths = None
        self.has_weighted_views = False
        self.x_hidden_codes = None
        self.y_hidden_codes = None
        self.x_new_order = None
        self.y_new_order = None
        self.props_tests = list()
        self.props_tests_levels = list()
        self.means_tests = list()
        self.means_tests_levels = list()
        self.has_props_tests = False
        self.has_means_tests = False
        self.is_banked = False
        self.banked_spec = None
        self.banked_view_key = None
        self.banked_meta = None
        self.base_text = None
        self.annotations = None

    def __repr__(self):
        return ('%s:\norientation-axis: %s - %s,\ncontent-axis: %s, \nviews: %s'
                %(Chain, self.orientation, self.source_name,
                  self.content_of_axis, len(self.views)))

    def __setstate__(self, attr_dict):
        self.__dict__.update(attr_dict)

    def __reduce__(self):
        return self.__class__, (self.name, ), self.__dict__, None, iter(list(self.items()))

    def save(self, path=None):
        """
        This method saves the current chain instance (self) to file (.chain) using cPickle.

        Attributes :
            path (string)
              Specifies the location of the saved file, NOTE: has to end with '/'
              Example: './tests/'
        """
        if path is None:
            path_chain = "./{}.chain".format(self.name)
        else:
            path_chain = path
        f = open(path_chain, 'wb')
        pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        f.close()

    def copy(self):
        """
        Create a copy of self by serializing to/from a bytestring using
        cPickle.
        """
        new_chain = pickle.loads(
            pickle.dumps(self, pickle.HIGHEST_PROTOCOL))
        return new_chain

    def _lazy_name(self):
        """
        Apply lazy-name logic to chains created without an explicit name.
         - This method does not take any responsibilty for uniquley naming chains
        """
        self.name = '%s.%s.%s.%s' % (self.orientation, self.source_name, '.'.join(self.content_of_axis), '.'.join(self.views).replace(' ', '_'))

    def _derive_attributes(self, data_key, filter, x_def, y_def, views, source_type=None, orientation=None):
        """
        A simple method that is deriving attributes of the chain from its specification:
        (some attributes are only updated when chains get post-processed,
        i.e. there is meta data available for the dataframe)
        -- examples:
            - orientation:        directional alignment of the link
            - source_name:        name of the orientation defining variable
            - source_type:        dtype of the source variable
            - len_of_axis:        number of variables in the non-orientation axis
            - views:              the list of views specified in the chain
            - view_sizes:         a list of lists of dataframe index and column lenght tuples,
                                  matched on x/view index (only when post-processed)

        """
        if x_def is not None or y_def is not None:
            self.orientation = orientation
            if self.orientation=='x':
                self.source_name = ''.join(x_def)
                self.len_of_axis = len(y_def)
                self.content_of_axis = y_def
            else:
                self.source_name = ''.join(y_def)
                self.len_of_axis = len(x_def)
                self.content_of_axis = x_def

            self.views = views
            self.data_key = data_key
            self.filter = filter
            self.source_type = source_type

    def concat(self):
        """
        Concatenates all Views found for the Chain definition along its
        orientations axis.
        """
        views_on_var = []
        contents = []
        full_chain = []
        all_chains = []
        chain_query = self[self.data_key][self.filter]
        if self.orientation == 'y':
            for var in self.content_of_axis:
                contents = []
                for view in self.views:
                    try:
                        res = (chain_query[var][self.source_name]
                               [view].dataframe.copy())
                        if self.source_name == '@':
                            res.columns = pd.MultiIndex.from_product(
                                ['@', '-'], names=['Question', 'Values'])
                        views_on_var.append(res)
                    except:
                        pass
                contents.append(views_on_var)
            for c in contents:
                full_chain.append(pd.concat(c, axis=0))
            concat_chain =  pd.concat(full_chain, axis=0)
        else:
            for var in self.content_of_axis:
                views_on_var = []
                for view in self.views:
                    try:
                        res = (chain_query[self.source_name][var]
                               [view].dataframe.copy())
                        if var == '@':
                            res.columns = pd.MultiIndex.from_product(
                                ['@', '-'], names=['Question', 'Values'])
                        views_on_var.append(res)
                    except:
                        pass
                contents.append(pd.concat(views_on_var, axis=0))
            concat_chain = pd.concat(contents, axis=1)
        return concat_chain

    def view_sizes(self):

        dk = self.data_key
        fk = self.filter
        xk = self.source_name
        sizes = []
        for yk in self.content_of_axis:
            vk_sizes = []
            for vk in self.views:
                vk_sizes.append(self[dk][fk][xk][yk][vk].dataframe.shape)
            sizes.append(vk_sizes)

        return sizes

    def view_lengths(self):

        lengths = [
            next(zip(*view_size))
            for view_size in [y_size for y_size in self.view_sizes()]]

        return lengths

    def describe(self, index=None, columns=None, query=None):
        """ Generates a list of all link defining stack keys.
        """
        stack_tree = []
        for dk in list(self.keys()):
            path_dk = [dk]
            filters = self[dk]

            for fk in list(filters.keys()):
                path_fk = path_dk + [fk]
                xs = self[dk][fk]

                for sk in list(xs.keys()):
                    path_sk = path_fk + [sk]
                    ys = self[dk][fk][sk]

                    for tk in list(ys.keys()):
                        path_tk = path_sk + [tk]
                        views = self[dk][fk][sk][tk]

                        for vk in list(views.keys()):
                            path_vk = path_tk + [vk, 1]
                            stack_tree.append(tuple(path_vk))

        column_names = ['data', 'filter', 'x', 'y', 'view', '#']
        df = pd.DataFrame.from_records(stack_tree, columns=column_names)

        if not query is None:
            df = df.query(query)
        if not index is None or not columns is None:
            df = df.pivot_table(values='#', index=index, columns=columns, aggfunc='count')
        return df

    # STATIC METHODS

    @staticmethod
    def load(filename):
        """
        This method loads the pickled object that is made using method: save()

        Attributes:
            filename ( string )
              Specifies the name of the file to be loaded.
              Example of use: new_stack = Chain.load("./tests/ChainName.chain")
        """
        f = open(filename, 'rb')
        new_stack = pickle.load(f)
        f.close()
        return new_stack
