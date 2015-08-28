#-*- coding: utf-8 -*-
import io
import itertools
import json
import pandas as pd
import numpy as np
import copy

from link import Link
from chain import Chain
from helpers import functions
from view_generators.view_mapper import ViewMapper
from view_generators.view_maps import QuantipyViews
from quantipy.core.tools.dp.spss.reader import parse_sav_file
from quantipy.core.tools.dp.io import unicoder
from cache import Cache

import itertools
from collections import defaultdict, OrderedDict

# Pickle modules
import cPickle

# Compression methods
import gzip
try:
    import pylzma
except:
    print 'Compression library pylzma not found. When saving a Quantipy Stack instance, please do not change the default compression type.'
    pass

class Stack(defaultdict):
    """
    Container of quantipy.Link objects holding View objects.

    A Stack is nested dictionary that structures the data and variable
    relationships storing all View aggregations performed.
    """

    def __init__(self,
                 name="",
                 add_data=None):

        super(Stack, self).__init__(Stack)

        self.name = name
        self.key = None
        self.parent = None
    
        # This is the root of the stack
        # It is used by the get/set methods to determine
        # WHERE in the stack those methods are.
        self.stack_pos = "stack_root"

        self.x_variables = None
        self.y_variables = None

        self.__view_keys = []

        if add_data:
            for key in add_data:
                if isinstance(add_data[key], dict):
                    self.add_data(
                        data_key=key,
                        data=add_data[key].get('data', None),
                        meta=add_data[key].get('meta', None)
                    )
                elif isinstance(add_data[key], tuple):
                    self.add_data(
                        data_key=key,
                        data=add_data[key][0],
                        meta=add_data[key][1]
                    )
                else:
                    raise TypeError(
                        "All data_key values must be one of the following types: "
                        "<dict> or <tuple>. "
                        "Given: %s" % (type(add_data[key]))
                    )

    def __setstate__(self, attr_dict):
        self.__dict__.update(attr_dict)

    def __reduce__(self):
        arguments = (self.name, )
        state = self.__dict__.copy()
        if 'cache' in state:
            state.pop('cache')
            state['cache'] = Cache() # Empty the cache for storage
        return self.__class__, arguments, state, None, self.iteritems()

    def __setitem__(self, key, val):
        """ The 'set' method for the Stack(dict)

            It 'sets' the value in it's correct place in the Stack
            AND applies a 'stack_pos' value depending on WHERE in
            the stack the value is being placed.
        """
        super(Stack, self).__setitem__(key, val)

        # The 'meta' portion of the stack is a standar dict (not Stack)
        try:
            if isinstance(val, Stack) and val.stack_pos is "stack_root":
                val.parent = self
                val.key = key

                # This needs to be compacted and simplified.
                if self.stack_pos is "stack_root":
                    val.stack_pos = "data_root"
                elif self.stack_pos is "data_root":
                    val.stack_pos = "filter"
                elif self.stack_pos is "filter":
                    val.stack_pos = "x"

        except AttributeError:
            pass

    def __getitem__(self, key):
        """ The 'get' method for the Stack(dict)

            The method 'gets' a value from the stack. If 'stack_pos' is 'y'
            AND the value isn't a Link instance THEN it tries to query the
            stack again with the x/y variables swapped and IF that yelds
            a result that is a Link object THEN it sets a 'transpose' variable
            as True in the result and the result is transposed.
        """
        val = defaultdict.__getitem__(self, key)
        return val

    def add_data(self, data_key, data=None, meta=None, ):
        """
        Sets the data_key into the stack, optionally mapping data sources it.
        
        It is possible to handle the mapping of data sources in different ways:

        * no meta or data (for proxy links not connected to source data)
        * meta only (for proxy links with supporintg meta)
        * data only (meta will be inferred if possible)
        * data and meta

        Parameters
        ----------
        data_key : str
            The reference name for a data source connected to the Stack.
        data : pandas.DataFrame
            The input (case) data source.
        meta : dict or OrderedDict
            A quantipy compatible metadata source that describes the case data.

        Returns
        -------
        None
        """
        self._verify_key_types(name='data', keys=data_key)

        if data_key in self.keys():
            raise UserWarning("You have chosen to overwrite the source data and meta for Stack['%s']")

        if data is not None:
            if isinstance(data, pd.DataFrame):
                if meta is None:
                    # To do: infer meta from DataFrame
                    meta = {'info': None, 'lib': None, 'sets': None,
                            'columns': None, 'masks': None}
                # Add a special column of 1s
                data['@1'] = np.ones(len(data.index))
                data.index = list(xrange(0, len(data.index)))
            else:
                raise TypeError(
                    "The 'data' given to Stack.add_data() must be one of the following types: "
                    "<pandas.DataFrame>"
                )

        if not meta is None:
            if isinstance(meta, (dict, OrderedDict)):
                # To do: verify incoming meta
                pass
            else:
                raise TypeError(
                    "The 'meta' given to Stack.add_data() must be one of the following types: "
                    "<dict>, <collections.OrderedDict>."
                )

        # Add the data key to the stack
        # self[data_key] = {}

        # Add the meta and data to the data_key position in the stack
        self[data_key].meta = meta
        self[data_key].data = data
        self[data_key].cache = Cache()
        self[data_key]['no_filter'].data = self[data_key].data

    def remove_data(self, data_keys):
        """
        Deletes the data_key(s) and associated data specified in the Stack.

        Parameters
        ----------
        data_keys : str or list of str
            The data keys to remove.
        
        Returns
        -------
        None
        """
        self._verify_key_types(name='data', keys=data_keys)
        if isinstance(data_keys, (str, unicode)):
            data_keys = [data_keys]
        for data_key in data_keys:
            del self[data_key]

    def variable_types(self, data_key, only_type=None):
        """
        Group variables by data types found in the meta.

        Parameters
        ----------
        data_key : str
            The reference name of a case data source hold by the Stack instance.
        only_type : {'int', 'float', 'single', 'delimited set', 'string',
                     'date', time', 'array'}, optional
            Will restrict the output to the given data type.

        Returns
        -------
        types : dict or list of str
            A summary of variable names mapped to their data types, in form of
            {type_name: [variable names]} or a list of variable names
            confirming only_type.
        """
        if self[data_key].meta['columns'] is None:
            return 'No meta attached to data_key: %s' %(data_key)
        else:
            types = {
                'int': [],
                'float': [],
                'single': [],
                'delimited set': [],
                'string': [],
                'date': [],
                'time': [],
                'array': []
            }
            not_found = []
            for col in self[data_key].data.columns:
                if not col in ['@1', 'id_L1', 'id_L1.1']: 
                    try:
                        types[
                              self[data_key].meta['columns'][col]['type']
                             ].append(col)
                    except:
                        not_found.append(col)                
            for mask in self[data_key].meta['masks'].keys():
                types[self[data_key].meta['masks'][mask]['type']].append(mask)
            if not_found:
                print '%s not found in meta file. Ignored.' %(not_found)
            if only_type:
                return types[only_type]
            else:
                return types


    def get_chain(self, name=None, data_keys=None, filters=None, x=None, y=None,
                  views=None, post_process=True, orient_on=None, select=None,
                  rules=False):
        """
        Construct a "chain" shaped subset of Links and their Views from the Stack.
        
        A chain is a one-to-one or one-to-many relation with an orientation that
        defines from which axis (x or y) it is build. 

        Parameters
        ----------
        name : str, optional
            If not provided the name of the chain is generated automatically.
        data_keys, filters, x, y, views : str or list of str
            Views will be added reflecting the order in ``views`` parameter. If
            both ``x`` and ``y`` have multiple items, you must specify the
            ``orient_on`` parameter.
        post_process : bool, default True
            If file meta is found, views inside the chain will get their Values-
            axes codes checked against the category lists and missing codes will
            be added.
        orient_on : {'x', 'y'}, optional
             Must be specified if both ``x`` and ``y`` are lists of multiple
             items.
        select : tbc.
            :TODO: document this!

        Returns
        -------
        chain : Chain object instance
        """

        #Make sure all the given keys are in lists
        data_keys = self._force_key_as_list(data_keys)
        # filters = self._force_key_as_list(filters)
        views = self._force_key_as_list(views)

        if orient_on:
            if x is None:
                x = self.describe()['x'].drop_duplicates().values.tolist()
            if y is None:            
                y = self.describe()['y'].drop_duplicates().values.tolist()
            if views is None:
                views = self._Stack__view_keys
                views = [v for v in views if '|default|' not in v]
            return self.__get_chains(name=name, data_keys=data_keys,
                                     filters=filters, x=x, y=y, views=views,
                                     post_process=post_process,
                                     orientation=orient_on, select=select)
        else:
            chain = Chain(name)
            found_views = []
            missed_views = []

            #Make sure all the given keys are in lists
            x = self._force_key_as_list(x)
            y = self._force_key_as_list(y)

            if data_keys is None:
                # Apply lazy data_keys if none given
                data_keys = self.keys()

            the_filter = "no_filter" if filters is None else filters

            if self.__has_list(data_keys):
                for key in data_keys:

                    # Use describe method to get x keys if not supplied.
                    if x is None:
                        x_keys = self.describe()['x'].drop_duplicates().values.tolist()
                    else:
                        x_keys = x

                    # Use describe method to get y keys if not supplied.
                    if y is None:
                        y_keys = self.describe()['y'].drop_duplicates().values.tolist()
                    else:
                        y_keys = y

                     # Use describe method to get view keys if not supplied.
                    if views is None:
                        v_keys = self.describe()['view'].drop_duplicates().values.tolist()
                        v_keys = [v_key for v_key in v_keys if '|default|'
                                  not in v_key]
                    else:
                        v_keys = views

                    chain._validate_x_y_combination(x_keys, y_keys, orient_on)
                    chain._derive_attributes(key,the_filter,x_keys,y_keys,views)

                    # Apply lazy name if none given
                    if name is None:
                        chain._lazy_name()

                    for x_key in x_keys:
                        for y_key in y_keys:

                            if views is None:
                                chain[key][the_filter][x_key][y_key] = self[key][the_filter][x_key][y_key]
                            else:
                                for view in views:
                                    try:
                                        stack_view = self[key][the_filter][x_key][y_key][view]
                                        chain[key][the_filter][x_key][y_key][view] = stack_view

                                        if view not in found_views:
                                            found_views.append(view)
                                    except KeyError:
                                        if view not in missed_views:
                                            missed_views.append(view)
            else:
                raise ValueError('One or more of your data_keys ({data_keys}) is not in the stack ({stack_keys})'.format(data_keys=data_keys, stack_keys=self.keys()))
            if found_views:
                chain.views = [view for view in chain.views
                               if view in found_views]

            for view in missed_views:
                if view in found_views:
                    missed_views.remove(view)

        if post_process:
            chain._post_process_shapes(self[chain.data_key].meta, rules)

        if select is not None:
            for view in chain[key][the_filter][x_key][y_key]:
                df = chain[key][the_filter][x_key][y_key][view].dataframe
                levels = df.index.levels
                selection = {}
                for var in select:
                    level = functions.find_variable_level(levels, var)
                    if level is not None:
                        selection[var] = level

                #Don't do anything if the selection doesnt produce a result
                if selection:
                    # selection = {var: functions.find_variable_level(levels, var) for var in select}
                    list_of_dfs = [df.xs(var, level=selection[var]) for var in selection.keys()]
                    new_df = pd.concat(list_of_dfs)
                    # Reconstruct the index
                    new_df.index= pd.MultiIndex.from_product([levels[0],selection.keys()], names=df.index.names)
                    chain[key][the_filter][x_key][y_key][view].dataframe = new_df

        return chain

    def reduce(self, data_keys=None, filters=None, x=None, y=None, variables=None, views=None):
        '''
        Remove keys from the matching levels, erasing discrete Stack portions.

        Parameters
        ----------
        data_keys, filters, x, y, views : str or list of str

        Returns
        -------
        None
        '''

        # Ensure given keys are all valid types
        self._verify_multiple_key_types(
            data_keys=data_keys,
            filters=filters,
            x=x,
            y=y,
            variables=variables,
            views=views
        )

        # Make sure all the given keys are in lists
        data_keys = self._force_key_as_list(data_keys)
        filters = self._force_key_as_list(filters)
        views = self._force_key_as_list(views)
        if not variables is None:
            variables = self._force_key_as_list(variables)
            x = variables
            y = variables
        else:
            x = self._force_key_as_list(x)
            y = self._force_key_as_list(y)

        # Make sure no keys that don't exist anywhere were passed
        key_check = {
            'data': data_keys,
            'filter': filters,
            'x': x,
            'y': y,
            'view': views
        }
        contents = self.describe()
        for key_type, keys in key_check.iteritems():
            if not keys is None:
                uk = contents[key_type].unique()
                if not any([tk in uk for tk in keys]):
                    raise ValueError(
                        "Some of the %s keys passed to stack.reduce() "
                        "weren't found. Found: %s. "
                        "Given: %s" % (key_type, uk, keys)
                    )

        if not data_keys is None:
            for dk in data_keys:
                try:
                    del self[dk]
                except:
                    pass

        for dk in self.keys():
            if not filters is None:
                for fk in filters:
                    try:
                        del self[dk][fk]
                    except:
                        pass

            for fk in self[dk].keys():
                if not x is None:
                    for xk in x:
                        try:
                            del self[dk][fk][xk]
                        except:
                            pass

                for xk in self[dk][fk].keys():
                    if not y is None:
                        for yk in y:
                            try:
                                del self[dk][fk][xk][yk]
                            except:
                                pass

                    for yk in self[dk][fk][xk].keys():
                        if not views is None:
                            for vk in views:
                                try:
                                    del self[dk][fk][xk][yk][vk]
                                except:
                                    pass

    def add_link(self, data_keys=None, filters=['no_filter'], x=None, y=None,
                 views=None, weights=None, variables=None):
        """
        Add Link and View defintions to the Stack.

        The method can be used flexibly: It is possible to pass only Link
        defintions that might be composed of filter, x and y specifications,
        only views incl. weight variable selections or arbitrary combinations of
        the former.

        :TODO: Remove ``variables`` from parameter list and method calls.

        Parameters
        ----------
        data_keys : str, optional
            The data_key to be added to. If none is given, the method will try
            to add to all data_keys found in the Stack.
        filters : list of str describing filter defintions, default ['no_filter']
            The string must be a valid input for the
            pandas.DataFrame.query() method.
        x, y : str or list of str
            The x and y variables to constrcut Links from.
        views : list of view method names.
            Can be any of Quantipy's preset Views or the names of created
            view method specifications.
        weights : list, optional
            The names of weight variables to consider in the data aggregation
            process. Weight variables must be of type ``float``.

        Returns
        -------
        None
        """
        if data_keys is None:
            data_keys = self.keys()
        else:
            self._verify_key_types(name='data', keys=data_keys)
            data_keys = self._force_key_as_list(data_keys)

        if not isinstance(views, ViewMapper):
            # Use DefaultViews if no view were given
            if views is None:
                pass
            #    views = DefaultViews()
            elif isinstance(views, (list, tuple)):
                views = QuantipyViews(views=views)
            else:
                raise TypeError(
                    "The views past to stack.add_link() must be type <quantipy.view_generators.ViewMapper>, "
                    "or they must be a list of method names known to <quantipy.view_generators.QuantipyViews>."
                )

        self._verify_key_types(name='filter', keys=filters)
        filters = self._force_key_as_list(filters)

        if not variables is None:
            if not x is None or not y is None:
                raise ValueError(
                    "You cannot pass both 'variables' and 'x' and/or 'y' to stack.add_link() "
                    "at the same time."
                )

        x = self._force_key_as_list(x)
        y = self._force_key_as_list(y)

        # Get the lazy y keys none were given and there is only 1 x key
        if not x is None:
            if len(x)==1 and y is None:
                y = self.describe(
                    index=['y'], 
                    query="x=='%s'" % (x[0])
                ).index.tolist()
        
        # Get the lazy x keys none were given and there is only 1 y key
        if not y is None:
            if len(y)==1 and x is None:      
                x = self.describe(
                    index=['x'], 
                    query="y=='%s'" % (y[0])
                ).index.tolist()

        for dk in data_keys:
            self._verify_key_exists(dk)
            for filter_def in filters:
                # if not filter_def in self[dk].keys():
                if filter_def=='no_filter':
                    self[dk][filter_def].data = self[dk].data
                    self[dk][filter_def].meta = self[dk].meta
                else:
                    try:
                        self[dk][filter_def].data = self[dk].data.query(filter_def)
                        self[dk][filter_def].meta = self[dk].meta
                    except Exception, ex:
                        raise UserWarning('A filter definition is invalid and will be skipped: {filter_def}'.format(filter_def=filter_def))
                        continue
                fdata = self[dk][filter_def].data
                if len(fdata) == 0:
                    raise UserWarning('A filter definition resulted in no cases and will be skipped: {filter_def}'.format(filter_def=filter_def))
                    continue
                self.__create_links(data=fdata, data_key=dk, the_filter=filter_def, x=x, y=y, views=views, weights=weights, variables=variables)

    def describe(self, index=None, columns=None, query=None, split_view_names=False):
        """
        Generates a structured overview of all Link defining Stack elements.
        
        Parameters
        ----------
        index, columns : str of or list of {'data', 'filter', 'x', 'y', 'view'},
                         optional
            Controls the output representation by structuring a pivot-style 
            table according to the index and column values.
        query : str
            A query string that is valid for the pandas.DataFrame.query() method.
        split_view_names : bool, default False
            If True, will create an output of unique view name notations split
            up into their components.

        Returns
        -------
        description : pandas.DataFrame
            DataFrame summing the Stack's structure in terms of Links and Views.
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

                        if views.keys():
                            for vk in views.keys():
                                path_vk = path_tk + [vk, 1]
                                stack_tree.append(tuple(path_vk))
                        else:
                            path_vk = path_tk + ['|||||', 1]
                            stack_tree.append(tuple(path_vk))
                            
        column_names = ['data', 'filter', 'x', 'y', 'view', '#']
        description = pd.DataFrame.from_records(stack_tree, columns=column_names)
        if split_view_names:
            views_as_series = pd.DataFrame(
                description.pivot_table(values='#', columns='view', aggfunc='count')
                ).reset_index()['view']
            parts = ['xpos', 'agg', 'relation', 'rel_to', 'weights', 
                     'shortname']
            description = pd.concat(
                (views_as_series,
                 pd.DataFrame(views_as_series.str.split('|').tolist(),
                              columns=parts)), axis=1)
        
        description.replace('|||||', np.NaN, inplace=True)
        if query is not None:
            description = description.query(query)
        if not index is None or not columns is None:
            description = description.pivot_table(values='#', index=index, columns=columns,
                                aggfunc='count')
        return description

    def save(self, path_stack, compression="gzip", store_cache=True, 
             decode_str=False):
        """
        Save Stack instance to .stack file.

        Parameters
        ----------
        path_stack : str
            The full path to the .stack file that should be created, including
            the extension.
        compression : {'gzip', 'lzma'}, default 'gzip'
            The intended compression type. 'lzma' offers high compression but
            can be very slow.
        store_cache : bool, default True
            Stores the MatrixCache in a file in the same location.
        decode_str : bool, default=True
            If True the unicoder function will be used to decode all str
            objects found anywhere in the meta document/s.
        
        Returns
        -------
        None
        """
        protocol = cPickle.HIGHEST_PROTOCOL
        if not path_stack.endswith('.stack'):
            raise ValueError(
                "To avoid ambiguity, when using Stack.save() you must provide the full path to "
                "the stack file you want to create, including the file extension. For example: "
                "stack.save(path_stack='./output/MyStack.stack'). Your call looks like this: "
                "stack.save(path_stack='%s', ...)" % (path_stack)
            )

        # Make sure there are no str objects in any meta documents. If
        # there are any non-ASCII characters will be encoded 
        # incorrectly and lead to UnicodeDecodeErrors in Jupyter.
        if decode_str:
            for dk in self.keys():
                self[dk].meta = unicoder(self[dk].meta)

        if compression is None:
            f = open(path_stack, 'wb')
            cPickle.dump(self, f, protocol)
        elif compression.lower() == "lzma":
            f = open(path_stack, 'wb')
            cPickle.dump(pylzma.compress(bytes(self)), f, protocol)
        else:
            f = gzip.open(path_stack, 'wb')
            cPickle.dump(self, f, protocol)

        if store_cache:
            caches = {}
            for key in self.keys():
                caches[key] = self[key].cache

            path_cache = path_stack.replace('.stack', '.cache')
            if compression is None:
                f1 = open(path_cache, 'wb')
                cPickle.dump(caches, f1, protocol)
            elif compression.lower() == "lzma":
                f1 = open(path_cache, 'wb')
                cPickle.dump(pylzma.compress(bytes(caches)), f1, protocol)
            else:
                f1 = gzip.open(path_cache, 'wb')
                cPickle.dump(caches, f1, protocol)

            f1.close()

        f.close()

    # def get_slice(data_key=None, x=None, y=None, filters=None, views=None):
    #     """  """
    #     pass

    # STATIC METHODS

    @staticmethod
    def from_sav(data_key, filename, name=None, path=None, ioLocale="en_US.UTF-8", ioUtf8=True):
        """
        Creates a new stack instance from a .sav file.

        Parameters
        ----------
        data_key : str
            The data_key for the data and meta in the sav file.
        filename : str
            The name to the sav file.
        name : str
            A name for the sav (stored in the meta).
        path : str
            The path to the sav file.
        ioLocale : str
            The locale used in during the sav processing.
        ioUtf8 : bool
            Boolean that indicates the mode in which text communicated to or
            from the I/O module will be.
        
        Returns
        -------
        stack : stack object instance
            A stack instance that has a data_key with data and metadata
            to run aggregations.
        """
        if name is None:
            name = data_key

        meta, data = parse_sav_file(filename=filename, path=path, name=name, ioLocale=ioLocale, ioUtf8=ioUtf8)
        return Stack(add_data={name: {'meta': meta, 'data':data}})

    @staticmethod
    def load(path_stack, compression="gzip", load_cache=False):
        """
        Load Stack instance from .stack file.

        Parameters
        ----------
        path_stack : str
            The full path to the .stack file that should be created, including
            the extension.
        compression : {'gzip', 'lzma'}, default 'gzip'
            The compression type that has been used saving the file.
        load_cache : bool, default False
            Loads MatrixCache into the Stack a .cache file is found.
        
        Returns
        -------
        None
        """


        if not path_stack.endswith('.stack'):
            raise ValueError(
                "To avoid ambiguity, when using Stack.load() you must provide the full path to "
                "the stack file you want to create, including the file extension. For example: "
                "stack.load(path_stack='./output/MyStack.stack'). Your call looks like this: "
                "stack.load(path_stack='%s', ...)" % (path_stack)
            )

        if compression is None:
            f = open(path_stack, 'rb')
        elif compression.lower() == "lzma":
            f = pylzma.decompress(open(path_stack, 'rb'))  # there seems to be a problem here!
        else:
            f = gzip.open(path_stack, 'rb')

        new_stack = cPickle.load(f)
        f.close()

        if load_cache:
            path_cache = path_stack.replace('.stack', '.cache')
            if compression is None:
                f = open(path_cache, 'rb')
            elif compression.lower() == "lzma":
                f = pylzma.decompress(open(path_cache, 'rb'))  # there seems to be a problem here!
            else:
                f = gzip.open(path_cache, 'rb')

            caches = cPickle.load(f)
            for key in caches.keys():
                if key in new_stack.keys():
                    new_stack[key].cache = caches[key]
                else:
                    raise ValueError(
                        "Tried to insert a loaded MatrixCache in to a data_key in the stack that"
                        "is not in the stack. The data_key is '{}', available keys are {}"
                        .format(key, caches.keys())
                    )
            f.close()

        return new_stack


    # PRIVATE METHODS

    def __get_all_y_keys(self, data_key, the_filter="no_filter"):
        if(self.stack_pos == 'stack_root'):
            return self[data_key].y_variables
        else:
            raise KeyError("get_all_y_keys can only be called from a stack at root level. Current level is '{0}'".format(self.stack_pos))

    def __get_all_x_keys(self, data_key, the_filter="no_filter"):
        if(self.stack_pos == 'stack_root'):
            return self[data_key].x_variables
        else:
            raise KeyError("get_all_x_keys can only be called from a stack at root level. Current level is '{0}'".format(self.stack_pos))

    def __get_all_x_keys_except(self, data_key, exception):
        keys = self.__get_all_x_keys(data_key)
        return [i for i in keys if i != exception[0]]

    def __get_all_y_keys_except(self, data_key, exception):
        keys = self.__get_all_y_keys(data_key)
        return [i for i in keys if i != exception[0]]

    def __set_x_key(self, key):
        if self.x_variables is None:
            self.x_variables = set(key)
        else:
            self.x_variables.update(key)

    def __set_y_key(self, key):
        if self.y_variables is None:
            self.y_variables = set(key)
        else:
            self.y_variables.update(key)

    def _set_x_and_y_keys(self, data_key, x, y):
        """
        Sets the x_variables and y_variables in the data part of the stack for this data_key, e.g. stack['Jan'].
        This method can also be used to add to the current lists and it makes sure the list stays unique.
        """
        if self.stack_pos == 'stack_root':
            self[data_key].__set_x_key(x)
            self[data_key].__set_y_key(y)
        else:
            raise KeyError("set_x_keys can only be called from a stack at root level. Current level is '{0}'".format(self.stack_pos))

    def __create_combinations(self, data, data_key, x=None, y=None, weight=None, variables=None):
        if isinstance(y, str):
            y = [y]
        if isinstance(x, str):
            x = [x]

        has_metadata = self[data_key].meta is not None and not isinstance(self[data_key].meta, Stack)

        # any(...) returns true if ANY of the vars are not None
        if any([x, y]) and variables is not None:
            # Raise an error if variables AND x/y are BOTH supplied
            raise ValueError("Either use the 'variables' OR 'x', 'y' NOT both.")

        if not any([x, y]):
            if variables is None:
                if not has_metadata:
                    # "fully-lazy" method. (variables, x and y are all None)
                    variables = data.columns.tolist()

        if variables is not None:
            x = variables
            y = variables
            variables = None

        # Ensure that we actually have metadata
        if has_metadata:
            # THEN we try to create the combinations with metadata
            combinations = self.__create_combinations_with_meta(data=data, data_key=data_key, x=x, y=y, weight=weight)
        else:
            # Either variables or both x AND y are supplied. Then create the combinations from that.
            combinations = self.__create_combinations_no_meta(data=data, data_key=data_key, x=x, y=y, weight=weight)

        unique_list = set([item for comb in combinations for item in comb])

        return combinations, unique_list

    def __create_combinations_with_meta(self, data, data_key, x=None, y=None, weight=None):
        # TODO: These meta functions should possibly be in the helpers functions
        metadata_columns = self[data_key].meta['columns'].keys()
        for mask, mask_data in self[data_key].meta['masks'].iteritems():
            # TODO :: Get the static list from somewhere. not hardcoded.
            if mask_data['type'].lower() in ['array', 'dichotomous set',
                                             "categorical set"]:
                metadata_columns.append(mask)
                for item in mask_data['items']:
                    if "source" in item:
                        column = item["source"].split('@')[1]
                        metadata_columns.remove(column)
            elif mask_data['type'].lower() in ["overlay"]:
                pass
        # Use all from the metadata, if nothing is specified (fully-lazy)
        if x is None and y is None:
            x = metadata_columns
            y = metadata_columns
        if all([x, y]):
            metadata_columns = list(set(metadata_columns + x + y))
        elif x is not None:
            metadata_columns = list(set(metadata_columns + x))
        elif y is not None:
            metadata_columns = list(set(metadata_columns + y))
        combinations = functions.create_combinations_from_array(sorted(metadata_columns))

        for var in [x, y]:
            if var is not None:
                if weight in var:
                    var.remove(weight)
        if all([x, y]):
            combinations = [(x_item, y_item) for x_item, y_item in combinations
                            if x_item in x and y_item in y]
        elif x is not None:
            combinations = [(x_item, y_item) for x_item, y_item in combinations
                            if x_item in x]
        elif y is not None:
            combinations = [(x_item, y_item) for x_item, y_item in combinations
                            if y_item in y]

        return combinations

    def __create_combinations_no_meta(self, data, data_key, x=None, y=None, weight=None):
        if x is None:
            x = data.columns.tolist()
        if y is None:
            y = data.columns.tolist()
        for var in [x, y]:
            if weight in var:
                var.remove(weight)
        combinations = [(x_item, y_item) for x_item in x for y_item
                        in y if x_item != y_item]
        self._set_x_and_y_keys(data_key, x, y)

        return combinations

    def __create_links(self, data, data_key, views, variables=None, x=None, y=None,
                       the_filter=None, store_view_in_link=False, weights=None):
        if views is not None:
            has_links = True if self[data_key][the_filter].keys() else False
            if has_links:
                xs = self[data_key][the_filter].keys()
                if x is not None:
                    valid_x = [xk for xk in xs if xk in x]
                    valid_x.extend(x)
                    x = set(valid_x)
                else:
                    x = xs
                ys = list(set(itertools.chain.from_iterable(
                    [self[data_key][the_filter][xk].keys()
                     for xk in xs])))
                if y is not None:
                    valid_y = [yk for yk in ys if yk in y]
                    valid_y.extend(y)
                    y = set(valid_y)
                else:
                    y = ys
        if self._x_and_y_keys_in_file(data_key, data, x, y):
            for x_key, y_key in itertools.product(x, y):
                    if y_key == '@':
                        if not isinstance(self[data_key][the_filter][x_key][y_key], Link):
                            link = Link(
                                        the_filter=the_filter,
                                        x=x_key,
                                        y='@',
                                        data_key=data_key,
                                        stack=self,
                                        store_view=store_view_in_link,
                                        create_views=False
                                        )
                            self[data_key][the_filter][x_key]['@'] = link
                        else:
                            link = self[data_key][the_filter][x_key]['@']
                    elif x_key == '@':
                        if not isinstance(self[data_key][the_filter][x_key][y_key], Link):
                            link = Link(
                                        the_filter=the_filter,
                                        x='@',
                                        y=y_key,
                                        data_key=data_key,
                                        stack=self,
                                        store_view=store_view_in_link,
                                        create_views=False
                                        )
                            self[data_key][the_filter]['@'][y_key] = link
                        else:
                            link = self[data_key][the_filter]['@'][y_key]
                    else:
                        if not isinstance(self[data_key][the_filter][x_key][y_key], Link):
                            link = Link(
                                        the_filter=the_filter,
                                        x=x_key,
                                        y=y_key,
                                        data_key=data_key,
                                        stack=self,
                                        store_view=store_view_in_link,
                                        create_views=False
                                        )
                            self[data_key][the_filter][x_key][y_key] = link
                        else:
                            link = self[data_key][the_filter][x_key][y_key]
                    if views is not None:
                        views._apply_to(link, weights)

    def _x_and_y_keys_in_file(self, data_key, data, x, y):
        data_columns = data.columns.tolist()
        x_not_found = [var for var in x if not var in data_columns
                       and not var == '@']
        y_not_found = [var for var in y if not var in data_columns
                       and not var == '@']
        if not x_not_found and not y_not_found:
            return True
        elif x_not_found and y_not_found:
            raise ValueError('for data key: %s\nx=%s not found, y=%s not found.'
                             % (data_key, x_not_found, y_not_found))
        elif x_not_found:
            raise ValueError('for data key: %s\nx=%s not found.'
                             % (data_key, x_not_found))
        elif y_not_found and y_not_found:
            raise ValueError('for data key: %s\ny=%s not in found.'
                             % (data_key, y_not_found))

    def __clean_column_names(self, columns):
        """
        Remove extra doublequotes if there are any
        """
        cols = []
        for column in columns:
            cols.append(column.replace('"', ''))
        return cols

    def __generate_key_from_list_of(self, list_of_keys):
        """
        Generate keys from a list (or tuple).
        """
        list_of_keys = list(list_of_keys)
        list_of_keys.sort()
        return ",".join(list_of_keys)

    def __has_list(self, small):
        """
        Check if object contains a list of strings.
        """
        keys = self.keys()
        for i in xrange(len(keys)-len(small)+1):
            for j in xrange(len(small)):
                if keys[i+j] != small[j]:
                    break
            else:
                return i, i+len(small)
        return False

    def __get_all_combinations(self, list_of_items):
        """Generates all combinations of items from a list """
        return [itertools.combinations(list_of_items, index+1)
                for index in range(len(list_of_items))]

    def __get_stack_pointer(self, stack_pos):
        """Takes a stack_pos and returns the stack with that location
            raises an exception IF the stack pointer is not found
        """
        if self.parent.stack_pos == stack_pos:
            return self.parent
        else:
            return self.parent.__get_stack_pointer(stack_pos)

    def __get_chains(self, name, data_keys, filters, x, y, views, orientation,
                     post_process, select):
        """
        Wrapper around .get_chain() to pull multiple chains from the stack.
        """
        if orientation == 'y':
            return [self.get_chain(name, data_keys, filters, x, y_var, views,
                                   post_process, select)
                    for y_var in y]
        elif orientation == 'x':
            return [self.get_chain(name, data_keys, filters, x_var, y, views,
                                   post_process, select)
                    for x_var in x]
        else:
            raise ValueError("Unknown orientation type. Please use 'x' or 'y'.")

    def _verify_multiple_key_types(self, data_keys=None, filters=None, x=None,
                                   y=None, variables=None, views=None):
        """
        Verify that the given keys str or unicode or a list or tuple of those.
        """
        if data_keys is not None:
            self._verify_key_types(name='data', keys=data_keys)

        if filters is not None:
            self._verify_key_types(name='filter', keys=filters)

        if x is not None:
            self._verify_key_types(name='x', keys=x)

        if y is not None:
            self._verify_key_types(name='y', keys=y)

        if variables is not None:
            self._verify_key_types(name='variables', keys=variables)

        if views is not None:
            self._verify_key_types(name='view', keys=views)

    def _verify_key_exists(self, key, stack_path=[]):
        """
        Verify that the given key exists in the stack at the path targeted.
        """
        error_msg = (
            "Could not find the {key_type} key '{key}' in: {stack_path}. "
            "Found {keys_found} instead."
        )
        try:
            dk = stack_path[0]
            fk = stack_path[1]
            xk = stack_path[2]
            yk = stack_path[3]
            vk = stack_path[4]
        except:
            pass
        try:
            if len(stack_path) == 0:
                if key not in self:
                    key_type, keys_found = 'data', self.keys()
                    stack_path = 'stack'
                    raise ValueError
            elif len(stack_path) == 1:
                if key not in self[dk]:
                    key_type, keys_found = 'filter', self[dk].keys()
                    stack_path = 'stack[{dk}]'.format(
                        dk=dk)
                    raise ValueError
            elif len(stack_path) == 2:
                if key not in self[dk][fk]:
                    key_type, keys_found = 'x', self[dk][fk].keys()
                    stack_path = 'stack[{dk}][{fk}]'.format(
                        dk=dk, fk=fk)
                    raise ValueError
            elif len(stack_path) == 3:
                if key not in self[dk][fk][xk]:
                    key_type, keys_found = 'y', self[dk][fk][xk].keys()
                    stack_path = 'stack[{dk}][{fk}][{xk}]'.format(
                        dk=dk, fk=fk, xk=xk)
                    raise ValueError
            elif len(stack_path) == 4:
                if key not in self[dk][fk][xk][yk]:
                    key_type, keys_found = 'view', self[dk][fk][xk][yk].keys()
                    stack_path = 'stack[{dk}][{fk}][{xk}][{yk}]'.format(
                        dk=dk, fk=fk, xk=xk, yk=yk)
                    raise ValueError
        except ValueError:
            print error_msg.format(
                key_type=key_type,
                key=key,
                stack_path=stack_path,
                keys_found=keys_found
            )

    def _force_key_as_list(self, key):
        """Returns key as [key] if it is str or unicode"""
        return [key] if isinstance(key, (str, unicode)) else key

    def _verify_key_types(self, name, keys):
        """
        Verify that the given keys str or unicode or a list or tuple of those.
        """
        if isinstance(keys, (list, tuple)):
            for key in keys:
                self._verify_key_types(name, key)
        elif isinstance(keys, (str, unicode)):
            pass
        else:
            raise TypeError(
                "All %s keys must be one of the following types: "
                "<str> or <unicode>, "
                "<list> of <str> or <unicode>, "
                "<tuple> of <str> or <unicode>. "
                "Given: %s" % (name, keys)
            )
