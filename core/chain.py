import cPickle
from collections import defaultdict
from helpers import functions as helpers
from view import View
import pandas as pd
from quantipy.core.tools.dp.prep import show_df
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
        self.view_sizes = None
        self.view_lengths = None
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

    def __repr__(self):
        return ('%s:\norientation-axis: %s - %s,\ncontent-axis: %s, \nviews: %s' 
                %(Chain, self.orientation, self.source_name,
                  self.content_of_axis, len(self.views)))

    def __setstate__(self, attr_dict):
        self.__dict__.update(attr_dict)

    def __reduce__(self):
        return self.__class__, (self.name, ), self.__dict__, None, self.iteritems()

    def save(self, path="./"):
        """
        This method saves the current chain instance (self) to file (.chain) using cPickle.

        Attributes :
            path (string)
              Specifies the location of the saved file, NOTE: has to end with '/'
              Example: './tests/'
        """
        f = open(path+self.name+'.chain', 'wb')
        cPickle.dump(self, f, cPickle.HIGHEST_PROTOCOL)
        f.close()

    def _validate_x_y_combination(self, x_keys, y_keys, orient_on):
        """
        Make sure that the x and y variables for the chain obey the following rules:
         - Either x or y must have only one variable (which defines its orientation)
         - The part that doesn't have one variable must have more than one variable (e.g. x=['a'] y=['b', 'c']
        """
        if len(x_keys) > 1 and len(y_keys) > 1 and not orient_on:
            raise ValueError("If the number of keys for both x and y are greater than 1, whether or not you have specified the x and y values, orient_on must be either 'x' or 'y'.")

    def _lazy_name(self):
        """
        Apply lazy-name logic to chains created without an explicit name.
         - This method does not take any responsibilty for uniquley naming chains
        """
        self.name = '%s.%s.%s.%s' % (self.orientation, self.source_name, '.'.join(self.content_of_axis), '.'.join(self.views).replace(' ', '_'))

    def _derive_attributes(self, data_key, filter, x_def, y_def, views, source_type=None):
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
            self.orientation = 'x' if len(x_def) == 1 and not len(y_def) == 1 else 'y'
            self.source_name = ''.join(x_def) if len(x_def) == 1 and not len(y_def) == 1 else ''.join(y_def)
            self.len_of_axis = len(y_def) if len(x_def) == 1 and not len(y_def) == 1 else len(x_def)
            self.content_of_axis = y_def if len(x_def) == 1 and not len(y_def) == 1 else x_def

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

    def _post_process_shapes(self, meta, rules=False):
        """
        The method is used while extracting shape sub-structures from the Stack using .get_chain().
        If metadata is available for the input data file, post-processing will update...
        ... the view.dataframes/.meta of the shape:
            - fully-indexed axes (numerical codes)
            - ...
        ... the general and specific attributes of the given shape
        """
        links = helpers.get_links(self)
#         views = helpers.get_views(self)

        if meta:

            self.x_hidden_codes = [
                [] for _ in xrange(len(self.content_of_axis) * len(self.views))
            ]
            self.x_new_order = [
                [] for _ in xrange(len(self.content_of_axis) * len(self.views))
            ]

            for link in links:
#                 views = helpers.get_views(link)
                for key, raw_view in link.iteritems():
                    
                    raw_view_meta = raw_view.meta()

                    if raw_view_meta['agg']['is_weighted']:
                        self.has_weighted_views = True
                                                    
                    idx = (
                        len(self.views) * self.content_of_axis.index(
                            raw_view_meta['xy'.replace(self.orientation, '')]['name']
                        )
                    ) + self.views.index(raw_view_meta['agg']['fullname'])
    
                    if 'x_hidden_in_views' in raw_view_meta:
                        if raw_view_meta['agg']['name'] in \
                            raw_view_meta['x_hidden_in_views']:
                            self.x_hidden_codes[idx] = raw_view_meta['x_hidden_codes']
    
                    if 'x_new_order_in_views' in raw_view_meta:
                        if raw_view_meta['agg']['name'] in raw_view_meta['x_new_order_in_views']:
                            self.x_new_order[idx] = raw_view_meta['x_new_order']
    
    #                 self.y_hidden_codes = [] # - helpers function does not exist for hiding codes in y vars.
                    
                    pp_view = copy.copy(raw_view)
                    # pp_view.dataframe = helpers.create_full_index_dataframe(
                    #     df=raw_view.dataframe.copy(), 
                    #     meta=meta, 
                    #     view_meta=raw_view_meta,
                    #     rules=rules
                    # )
    
                    vk = raw_view_meta['agg']['fullname']

                    if raw_view.dataframe.shape==(1, 1):
                        rule_override = False
                    else:
                        rule_override = rules

                    # rule_override = False
                    # if rules and '|frequency||' in vk:
                    #     rule_override = rules

                    pp_view.dataframe = show_df(
                        raw_view.dataframe.copy(), 
                        meta, 
                        show='values', 
                        rules=rule_override,
                        full=True,
                        link=link,
                        vk=vk
                    )
                    
                    # Add rules to view meta if found
                    if rules:
                        
                        rx = None
                        ry = None
    
                        if link.x!='@':
                            if 'rules' in meta['columns'][link.x]:
                                rx = meta['columns'][link.x]['rules'].get('x', None)
    
                        if link.y!='@':
                            if 'rules' in meta['columns'][link.y]:
                                ry = meta['columns'][link.y]['rules'].get('y', None)
                                
                        pp_view.meta()['x']['rules'] = rx
                        pp_view.meta()['y']['rules'] = ry
                            
                    pp_view.meta()['x']['size'] = pp_view.dataframe.shape[0]-\
                                                len(self.x_hidden_codes[idx])
                    pp_view.meta()['y']['size'] = pp_view.dataframe.shape[1]
                    pp_view.meta()['shape'] = (
                        pp_view.meta()['x']['size'], 
                        pp_view.meta()['y']['size']
                    )
    
                    y_name = (raw_view_meta['y']['name'][1:] \
                        if len(raw_view_meta['y']['name']) > 1 and \
                        raw_view_meta['y']['name'][0] == '@' else \
                        raw_view_meta['y']['name'])
                    
                    self[self.data_key][self.filter][raw_view_meta['x']
                        ['name']][y_name][raw_view_meta['agg']['fullname']] = pp_view
    
                    if raw_view_meta['agg']['method'] == 'coltests': 
    
                        fullname = raw_view_meta['agg']['fullname']
                        relation = fullname.split('|')[2]
    
                        if len(relation) == 0 and fullname not in self.props_tests:
                            self.props_tests.append(fullname)
                            self.props_tests_levels.append(raw_view.is_propstest())
                            self.has_props_tests = True
    
                        elif len(relation) > 0 and fullname not in self.means_tests:
                            self.means_tests.append(fullname)
                            self.means_tests_levels.append(raw_view.is_meanstest())
                            self.has_means_tests = True

        self.view_sizes = []
        for xyi in xrange(len(self.content_of_axis)):
            self.view_sizes.append([])
            if self.orientation == 'y':
                x, y = self.content_of_axis[xyi], self.source_name
            elif self.orientation == 'x':
                y, x = self.content_of_axis[xyi], self.source_name
            for view in self.views:
                if (self[self.data_key][self.filter]
                       [x][y][view].__class__.__name__) == "View":
                    self.view_sizes[xyi].append(
                        (self[self.data_key][self.filter]
                            [x][y][view].meta()['shape'])
                    )
                else:
                     self.view_sizes[xyi].append((0, 0))

        self.view_lengths = [[view[0] for view in var] for var in self.view_sizes]
        
        if self.source_name == '@':
            self.source_length = 1
        else:
            self.source_length = max(
                [vsize[1] for vsizes in self.view_sizes for vsize in vsizes]
            )
 
    def describe(self, index=None, columns=None, query=None):
        """ Generates a list of all link defining stack keys.
        """
        stack_tree = []
        for dk in self.keys():
            path_dk = [dk]
            filters = self[dk]

            for fk in filters.keys():
                path_fk = path_dk + [fk]
                xs = self[dk][fk]

                for sk in xs.keys():
                    path_sk = path_fk + [sk]
                    ys = self[dk][fk][sk]

                    for tk in ys.keys():
                        path_tk = path_sk + [tk]
                        views = self[dk][fk][sk][tk]

                        for vk in views.keys():
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
        new_stack = cPickle.load(f)
        f.close()
        return new_stack
