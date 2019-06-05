#-*- coding: utf-8 -*-
import marshal
import copy
from types import FunctionType
from collections import OrderedDict
from itertools import product
import quantipy as qp

class ViewMapper(OrderedDict):
    """
    Applies View computation results to Links based on the view method's
    kwargs, handling the coordination and structuring side of the aggregation
    process.
    """
    def __init__(self, views=None, template=None):
        super(ViewMapper, self).__init__() # Initiate the ordered dict

        self.known_methods = OrderedDict()
        self.__init_known_methods__()
        self.__init_custom_methods__()

        # Set the template into the instance if one is provided
        self.template = template

        # If no view method is views the choose all known_methods
        if views is None:
            views = list(self.known_methods.keys())

        # populate the view instance with the views methods from the known_methods.
        for method in views:
            if method in self.known_methods:
                self[method] = self.known_methods[method]
            else:
                self[method] = views[method]

    def __setstate__(self, view_dict):
        # Reconstruct the View object after serialization
        for view_name in view_dict:
            kwargs = view_dict[view_name]['kwargs']
            method_as_bytes = view_dict[view_name]['method'] # method stored as bytes

            # Set the kwargs into the deserialized view
            self[view_name] = {'kwargs': kwargs}

            # Set the method into the deserialized view
            method_as_string = marshal.loads(method_as_bytes) # Returns method as string
            self[view_name]['method'] = FunctionType(code=method_as_string, globals=globals(), name=view_name)

    def __reduce__(self):
        class_type = self.__class__
        arguments = list(self.keys())
        setitem_dict = {}

        # Must use .keys() because the loop removes callable methods from self
        for view_name in list(self.keys()):
            if callable(self[view_name]['method']):
                view = self.pop(view_name)
                kwargs = view['kwargs']
                method = view['method']
                setitem_dict[view_name] = {'method':marshal.dumps(method.__code__), 'kwargs':kwargs}

        return (class_type, tuple(arguments), setitem_dict, None, None)

    def make_template(self, method, iterators=None):
        """
        Generate a view method template that cycles through kwargs values.

        Parameters
        ----------
        method : {'frequency', 'descriptives', 'coltests'}
            The baseline view method to be used.
        iterators : dict
            A dictionary mapping of view method kwargs to lists of values.

        Returns
        -------
        None
            Sets the template inside ViewMapper instance.
        """
        view_method = eval('qp.QuantipyViews().' + method)
        if iterators is not None:
            template = {'method': view_method,
                        'kwargs': {'iterators': {k: v for k, v in list(iterators.items())}}}
        else:
            template = {'method': view_method, 'kwargs': {}}
        self.template = template
        return self

    def add_method(self, name=None, method=None, kwargs={}, template=None):
        """
        Add a method to the instance of the ViewMapper.

        Parameters
        ----------
        name : str
            The short name of the View.
        method : view method
            The view method that will be used to derivce the result
        kwargs : dict
            The keyword arguments needed by the view method.
        template : dict
            A ViewMapper template that contains information on view method and
            kwargs values to iterate over.

        Returns
        -------
        None
            Updates the ViewMapper instance with a new method definiton.
        """
        if template is None and not self.template is None:
            template = self.template

        if not template is None:
            name = template.get('name', name)
            method = template.get('method', method)
            kwargs = dict(template.get('kwargs', {}), **kwargs)

        if None in [name, method]:
            raise TypeError(
                "You must provide a 'name' and 'method' to add_method(), \n"
                "either directly in the method call or through a ViewMapper template. \n"
                "You gave: \n"
                "name: {name} \n"
                "method: {method} \n".format(
                    name=name,
                    method=method
                )
            )

        self[name] = {'method': method, 'kwargs': kwargs}

    def subset(self, views, strict_selection=True):
        """
        Copy ViewMapper instance retaining only the View names provided.

        Parameters
        ----------
        views : list of str
            The selection of View names to keep.
        strict_selection : bool, default True
            TODO

        Returns
        -------
        subset : ViewMapper instance
        """
        if isinstance(views, str):
            views = [views]
        self_keys = set(self.keys())
        requested_keys = set(views)
        valid_keys = self_keys.intersection(requested_keys)
        if not valid_keys:
            raise KeyError(
                "None of the view keys you attempted to extract using 'subset' "
                "were found in this ViewMapper instance. "
                "You requested: %s, found: %s" % (views, list(self.keys()))
            )
        if strict_selection:
            invalid_keys = requested_keys - self_keys
            if invalid_keys:
                raise KeyError(
                    "Some of the view keys you attempted to extract using 'subset' "
                    "were not found in this ViewMapper instance. "
                    "You requested: %s, found: %s" % (views, list(self.keys()))
                )
        subset = self.copy()
        for view in list(subset.keys()):
            if not view in views:
                del subset[view]
        return subset

    def _custom_methods(self):
        """
        Returns a list of methods not found in the known_methods dict.
        """
        return [method for method in self if method not in self.known_methods]

    def _get_method_types(self, link):
        """
        Returns a string that is used to determine how to generate the View.
        """
        x = link.x if not link.x == '@' else link.y
        y = link.y if not link.y == '@' else link.x

        transpose = False
        meta = link.get_meta()

        if meta is not None:
            if '>' in y:
                y = y.split('>')[0]
            if '>' in x:
                x = x.split('>')[0]
            # Get the type from the metadata
            for pos in ['columns', 'masks']:
                if x in meta[pos]:
                    x_type = meta[pos][x]['type']

                if y.replace('@','') in meta[pos]:
                    y_type = meta[pos][y.replace('@','')]['type']

            try:
                if y_type in ["categorical set", "dichotomous set",
                              "delimited set", "array"]:
                    transpose = False
            except:
                print("Can't find a y called %s" % (link.y))

        else:
            # Infer the type from the pandas types
            # "columns"/"masks"
            # "type"              Pandas dtypes           Aggregation method
            # ----------------------------------------------------------------
            # "int"               [int64*]                .describe()
            # "float"             [float64*]              .describe()
            # "text"              [object]                .value_counts()
            # "date"              [datetime64*]           qp.date.describe()
            # "time"              [timedelta64*]          qp.time.describe()
            data = link.get_data()
            x_dtype = data.dtypes[x].name
            y_dtype = data.dtypes[y].name
            types = []
            for index, dtype in enumerate([x_dtype, y_dtype]):
                if 'int' in dtype:
                    types.append('int')
                elif 'float' in dtype:
                    types.append('float')
                elif 'object' in dtype:
                    types.append('single') # 'single' uses value_counts
                elif 'date' in dtype:
                    types.append('date')
                elif 'time' in dtype:
                    types.append('time')

            x_type, y_type = types[0], types[1]

        return (x_type, y_type, transpose)

    def _apply_to(self, link, weights=None):
        """
        Loop through view methods applying them to the Link.

        Parameters
        ----------
        link : Link
        weights : Weight variable as str or list of str

        """

        # Keep a clean cope of the weights given in args
        arg_weights = weights

        for name, values in list(self.items()):

            # Take a copy of the clean arg_weights value
            weights = copy.copy(arg_weights)

            method = values['method']
            kwargs = values.get('kwargs', '')

            # Get weight instructions from both kwargs and apply_to(weights)
            # Ensure both are end up as lists
            # While apply_to() still has a weights arg, it will need to
            # trump weights given in kwargs
            if weights is None:
                override_weights = False
                weights = kwargs.get('weights', None)
                if not isinstance(weights, (list, tuple)):
                    weights = [weights]
            else:
                override_weights = True
                if not isinstance(weights, (list, tuple)):
                    weights = [weights]

            # Get rel_to from kwargs and ensure it is a list
            rel_to = kwargs.get('rel_to', None)
            if not isinstance(rel_to, (list, tuple)):
                rel_to = [rel_to]

            # Get iterators from kwargs, or create default if none is found
            iterators = kwargs.get('iterators', {}).copy()

            # Make sure iterators has a 'weights' key
            # Where weights have been given via apply_to(weights), allow
            # them to override anything that may have been picked up via
            # an iterator object
            if override_weights or not 'weights' in iterators:
                iterators['weights'] = weights

            # Make sure the given weights provided are in the iterator weights keys
            # This catches when iterators are given in kwargs but additional weights
            # have also been provided somehow (such as by using add_link(.., weights))
            # In these situations the iterated weights should be the combination of
            # both instructions
            for weight in weights:
                if not weight in iterators['weights']:
                    iterators['weights'].append(weight)

            # Make sure the given rel_tos provided are in the iterator rel_to keys
            # This catches when iterators are given in kwargs but additional rel_tos
            # have also been provided somehow
            # In these situations the iterated weights should be the combination of
            # both instructions
            if not 'rel_to' in iterators:
                iterators['rel_to'] = rel_to
            for rel in rel_to:
                if not rel in iterators['rel_to']:
                    iterators['rel_to'].append(rel_to)

            # Get the product of all the targeted iterators
            view_iterations = self.__get_view_iterations__(iterators)

            # Run the view method for all the requested iterations
            for view_iter in view_iterations:
                kwargs.update(view_iter)
                if isinstance(method, str):
                    getattr(self, method)(link, name, kwargs)
                else:
                    method(link, name, kwargs)

    # Private
    def __print_exception_message__(self,message, link, name):
        print("Error generating View: '{name}', x: '{x}', y: '{y}'. Error : '{message}'.\n".format(name=name, x=link.x, y=link.y, message=message))

    # "proxy" methods. The core class doesn't know any view methods but this method can be extended.
    def __init_custom_methods__(self):
        pass

    def __init_known_methods__(self):
        pass

    def __get_view_iterations__(self, iterators):
        ''' Returns a list of dicts needed for multiple-view generation
        by a view method.

        Parameters
        ----------
        iterators : dict
            Dict of lists, where the product of the lists needs to
            be yielded one at a time using the same keys as the
            incoming dict.

        meta : Quantipy meta object pared to data

        Returns
        ----------
        iterations : list of dicts
        '''
        keys, items = list(zip(*list(iterators.items())))
        iterations = [dict(list(zip(keys, x))) for x in product(*items)]

        return iterations
