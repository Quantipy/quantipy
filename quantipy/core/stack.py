#!/usr/bin/python
# -*- coding: utf-8 -*-
from ..__imports__ import *  # noqa

from .meta import Meta

# from .chainmanager import ChainManager
from .views.view_mapper import ViewMapper

logger = get_logger(__name__)


class Stack(defaultdict):
    """
    Container of quantipy.Link objects holding View objects.

    A Stack is nested dictionary that structures the data and variable
    relationships storing all View aggregations performed.
    """
    def __init__(self, name="", add_data=None):
        """
        Initialize a Stack instance with stack_position 'stack_root'.

        Parameters
        ----------
        name: str
            Name of the stack, which is used as key for deeper stack positions.
        add_data: dict of dict
            A dictionary in form of
            ```
            add_data = {
                'data_key' : {
                    'data': pd.DataFrame, 'meta': qp.Meta,
                    # or
                    'dataset': qp.DataSet
                }
            }
        """
        # inherit default dict behaviour
        super(Stack, self).__init__(Stack)

        # The position is used by the get and set methods
        self.name = name  # == key in parnt stack
        self.parent = None
        self.stack_pos = "stack_root"

        # included variables
        self.x_variables = []
        self.y_variables = []

        if add_data:
            for key, value in add_data.items():
                self.add_data(key, **value)

    def __setstate__(self, attr_dict):
        """
        Is used while unpickling an instance.
        """
        self.__dict__.update(attr_dict)

    def __reduce__(self):
        """
        Is used while pickling an instance.
        """
        arguments = (self.name, )
        state = self.__dict__.copy()
        if 'cache' in state:
            state.pop('cache')
            state['cache'] = Cache()  # Empty the cache for storage
        return self.__class__, arguments, state, None, self.iteritems()

    def __setitem__(self, key, val):
        """
        Inherit defaultdict's "set" and defines the position for the new value.
        """
        parents = {
            "stack_root": "data_root",
            "data_root": "filter_root",
            "filter": "x_root"}
        position = parents.get(self.stack_pos)
        if position and not isinstance(val, Stack):
            err = "Can only set a new Stack in position '{}'.".format(position)
            logger.error(err); raise TypeError(err)
        if key and not isinstance(key, str):
            err = "'key' must be of type 'str'."
            logger.error(err); raise TypeError(err)
        if key in self:
            msg = "Overwrite key '{}' in position '{}'".format(key, position)
        else:
            msg = "Set new key '{}' is position '{}'".format(key, position)
        logger.debug(msg)

        if isinstance(val, Stack):
            val.name = key
            val.parent = self
            val.stack_pos = parents.get(self.stack_pos)
        super(Stack, self).__setitem__(key, val)

    # -------------------------------------------------------------------------
    # io
    # -------------------------------------------------------------------------
    def save(self, name, path=".", compression="gzip", store_cache=True,
             dataset=False, describe=False):
        """
        Save Stack instance to .stack file.

        Parameters
        ----------
        path_stack : str
            The full path to the .stack file that should be created, including
            the extension.
        compression : {'gzip'}, default 'gzip'
            The intended compression type.
        store_cache : bool, default True
            Stores the MatrixCache in a file in the same location.
        decode_str : bool, default=True
            If True the unicoder function will be used to decode all str
            objects found anywhere in the meta document/s.
        dataset : bool, default=False
            If True a json/csv will be saved parallel to the saved stack
            for each data key in the stack.
        describe : bool, default=False
            If True the result of stack.describe().to_excel() will be
            saved parallel to the saved stack.
        """
        path_stack = os.path.join(path, name or "{}.stack".format(self.name))
        if not path_stack.endswith('.stack'):
            err = "Can only save with file extension '.stack'."
            logger.errror(err); raise ValueError(err)

        pickle_dump(self, path_stack, compression)

        if store_cache:
            caches = {key: self[key].cache for key in self.keys()}
            path_cache = path_stack.replace('.stack', '.cache')
            pickle_dump(caches, path_cache, compression)

        if dataset:
            from .dataset import DataSet
            for key in self.keys():
                ds = DataSet.from_stack(self, key)
                ds.name = "{} [{}]".format(name or self.name, key)
                ds.to_quantipy(path=path)

        if describe:
            path_describe = path_stack.replace('.stack', '.xlsx')
            self.describe().to_excel(path_describe)

    @classmethod
    def load(cls, path_stack, compression="gzip", load_cache=False):
        """
        Load Stack instance from .stack file.

        Parameters
        ----------
        path_stack : str
            The full path to the .stack file that should be created, including
            the extension.
        compression : {'gzip'}, default 'gzip'
            The compression type that has been used saving the file.
        load_cache : bool, default False
            Loads MatrixCache into the Stack a .cache file is found.
        """
        if not path_stack.endswith('.stack'):
            err = "Can only load with file extension '.stack'."
            logger.errror(err); raise ValueError(err)

        stack = pickle_load(path_stack, compression)

        if load_cache:
            path_cache = path_stack.replace('.stack', '.cache')
            caches = pickle_load(path_cache, compression)
            for key in caches.keys():
                if key in stack.keys():
                    stack[key].cache = caches[key]
        return stack

    # -------------------------------------------------------------------------
    # meta / data
    # -------------------------------------------------------------------------
    def add_data(self, data_key, **kwargs):
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
        """
        from .dataset import DataSet
        data = kwargs.get("data")
        meta = kwargs.get("meta")
        dataset = kwargs.get("dataset")
        if dataset:
            if isinstance(dataset, DataSet):
                meta = dataset._meta.clone()
                data = dataset._data.copy()
            else:
                err = "kwarg 'dataset' must be of type qp.DataSet!"
                logger.error(err); raise TypeError(err)
        if meta is None or data is None:
            err = "kwargs must either contain 'meta' and 'data' or 'dataset'."
            logger.error(err); raise ValueError(err)
        elif not isinstance(meta, Meta):
            err = "'meta' must be of type qp.Meta!"
            logger.error(err); raise TypeError(err)
        elif not isinstance(data, pd.DataFrame):
            err = "'data' must be of type pd.DataFrame!"
            logger.error(err); raise TypeError(err)

        self[data_key].meta = meta
        self[data_key].data = data
        self[data_key].cache = Cache()

    def apply_meta_edits(self, batch, data_key, freeze=False):
        """
        Take over meta of a Batch in self[data_key].meta["sets"]["batches"].

        Parameters
        ----------
        batch: str
            Name of the Batch whose meta are taken.
        freeze: bool, default False
            *  True: Save current meta in property 'master_meta'.
        """
        if freeze:
            self.freeze_master_meta(data_key)
        meta = Meta(self[data_key].meta["sets"]["batches"][batch]["meta"])
        self[data_key].meta = meta.clone()

    def freeze_master_meta(self, data_key):
        """
        Save ``.meta`` in ``.master_meta`` for a defined data_key.
        """
        self[data_key].master_meta = self[data_key].meta

    def restore_meta(self, data_key):
        """
        Restore the ``.master_meta`` for a defined data_key if it exists.

        Undo self.apply_meta_edits()
        """
        if hasattr(self[data_key], "master_meta"):
            self[data_key].meta = self[data_key].master_meta

    # -------------------------------------------------------------------------
    # links
    # -------------------------------------------------------------------------
    @params(to_list=["filters", "x", "y", "views", "weights"])
    def add_link(self, data_key, filters=[], x=[], y=[], views=[],
                 weights=[]):
        """
        Add Link and View defintions to the Stack.

        The method can be used flexibly: It is possible to pass only Link
        defintions that might be composed of filter, x and y specifications,
        only views incl. weight variable selections or arbitrary combinations
        of the former.

        Parameters
        ----------
        filters : (list of )str or None
            Names of filter variables in metadata or None for unfiltered data
        x, y : str or list of str
            The x and y variables to constrcut Links from.
        views : list of view method names.
            Can be any of Quantipy's preset Views or the names of created
            view method specifications.
        weights : list, optional
            The names of weight variables to consider in the data aggregation
            process. Weight variables must be of type ``float``.
        """
        from .dataset import DataSet
        if data_key not in self:
            err = "'{}' is not a valid datakey.".format(data_key)
            logger.error(err); raise KeyError(err)
        vmapper = []
        if views:
            vnames = []
            for view in views:
                if isinstance(view, ViewMapper):
                    vmapper.append(view)
                else:
                    vnames.append(view)
            if vnames:
                vmapper.extend(QuantipyViews(views=vnames))

        invalid = []
        if not filters:
            filters = [None]
        for fk in filters:
            if fk:
                if not self[data_key].meta.is_filter(fk):
                    invalid.add(fk)
                    continue
                if fk not in self[data_key]:
                    ds = DataSet.from_stack(self, data_key)
                    f_ds = ds.isfilter(fk)
                    self[data_key][fk].data = f_ds._data
            self.__create_links(data_key, fk, x, y, vmapper, weights)
        if invalid:
            warn = "skipped '{}': invalid filters.".format(invalid)
            logger.warning(warn)

    def __create_links(self, dk, fk, xs, ys, vmapper, weights):
        if not xs:
            xs = self[dk][fk].keys()
        xs = [x for x in xs if x == "@" or x in self[dk].meta]
        if not xs:
            err = "Please define 'x'"
            logger.error(err); raise ValueError(err)
        if not ys:
            ys = list(set([y for x in xs for y in self[dk][fk][x]]))
        ys = [y for y in ys if y == "@" or y in self[dk].meta]
        if not ys:
            err = "Please define 'y'"
            logger.error(err); raise ValueError(err)

        for xk, yk in product(xs, ys):
            if xk == yk == "@":
                continue
            link = self[dk][fk][xk][yk]
            if not isinstance(link, Link):
                link = Link(self, dk, fk, xk, yk)
                self[dk][fk][xk][yk] = link
            for vm in vmapper:
                vm._apply_to(link, weights)

    def describe(self, index=None, columns=None, query=None,
                 split_view_names=False):
        """
        Generates a structured overview of all Link defining Stack elements.

        Parameters
        ----------
        index, columns: str of or list {'data', 'filter', 'x', 'y', 'view'}
            Controls the output representation by structuring a pivot-style
            table according to the index and column values.
        query : str
            A query string that is valid for the pandas.DataFrame.query()
            method.
        split_view_names : bool, default False
            If True, will create an output of unique view name notations split
            up into their components.

        Returns
        -------
        pandas.DataFrame
            DataFrame summing the Stack's structure in terms of Links and
            Views.
        """
        stack_tree = []
        for dk in self.keys():
            for fk in self[dk].keys():
                for xk in self[dk][fk].keys():
                    for yk in self[dk][fk][xk].keys():
                        link = self[dk][fk][xk][yk]
                        for vk in list(link) or ["|||||"]:
                            stack_tree.append(link._path + [vk, 1])
        column_names = ['data', 'filter', 'x', 'y', 'view', '#']
        desc = pd.DataFrame.from_records(stack_tree, columns=column_names)

        if split_view_names:
            pt = desc.pivot_table(values='#', columns='view', aggfunc='count')
            views = pd.DataFrame(pt).reset_index()['view']
            parts = [
                'xpos', 'agg', 'condition', 'rel_to', 'weights', 'shortname']
            views_split = pd.DataFrame(
                views_as_series.str.split('|').tolist(), columns=parts)
            desc = pd.concat([views, views_split], axis=1)

        desc.replace('|||||', np.NaN, inplace=True)
        if query:
            desc = desc.query(query)
        if index or columns:
            desc = desc.pivot_table(
                values='#', index=index, columns=columns, aggfunc='count')
        return desc

    # -------------------------------------------------------------------------
    # frequency & crosstab
    # -------------------------------------------------------------------------
    def frequency(self, dk, fk=None, xk=None, yks=None, view='counts',
                  decimals=1, weight=None, text=False, rules=False,
                  xtotal=False):
        """
        Return a type-appropriate frequency of x.

        Parameters
        ----------
        x : str, default=None
            The column of data for which a frequency should be generated
            on the x-axis.
        y : str, default=None
            The column of data for which a frequency should be generated
            on the y-axis.
        view : str {"counts", "c%"}, default='counts'
            Control the type of data that is returned.
        decimals : int, default=1
            Control the number of decimals in the returned dataframe.
        weight : str, default=None
            The name of the weight variable that should be used on the data,
            if any.
        text : bool, default=False
            How the index and columns should be displayed.
            *  False : 'values' are used
            *  True : 'texts' are used
        rules : bool or list-like, default=False
            If True then all rules that are found will be applied. If
            list-like then rules with those keys will be applied.
        xtotal : bool, default=False
            If True, the first column of the returned dataframe will be the
            regular frequency of the x column.
        """
        # TBD!
        pass

    # -------------------------------------------------------------------------
    # aggregations
    # -------------------------------------------------------------------------
    @params(to_list=['categorize', 'xs'])
    def aggregate(self, views, bases={}, categorize=[], batches='all', xks=[]):
        """
        Add views to all defined ``qp.Link`` in ``qp.Stack``.

        Parameters
        ----------
        views: (list of) str or qp.ViewMapper
            ``views`` that are added.
        bases: dict
            Defines which bases should be aggregated, weighted or unweighted.
            ```
            bases = {
                "wgt"
            }
        categorize: (list of) str
            Determines how numerical data is handled:
            If provided, the variables will get counts and percentage
            aggregations ('counts', 'c%') alongside the base views.
            If False, only base views are generated for non-categorical types.
        batches: str/ list of str, default 'all'
            Name(s) of ``qp.Batch`` instance(s) that are used to aggregate the
            ``qp.Stack``.
        """
        from .dataset import DataSet
        views, bases, complete, c_n_views = self._dissect_views(views, bases)
        for dk in self.keys():
            ds = DataSet.from_stack(dk)
            use_batches = self._check_batches(dk, batches)
            numerics = self[dk].meta.ints + self[dk].meta.floats
            for batch in use_batches:
                logger.info("aggregating for batch '{}'".format(batch))
                b = ds.get_batch(batch)
                b_weights = [b_w for b_w in b.weights if b_w]
                unwgt_c = all(
                    [b.unwgt_counts, c_n_views, None not in b.weights])
                mapping = []
                # reduce x_y_map if on_vars are defined
                if xks:
                    for xk, yks in b.x_y_map:
                        if xk not in xks:
                            logger.debug(
                                "'{}' is not in '{}'".format(xk, batch))
                            continue
                        elif xk in b.transposed:
                            mapping.append(("@", xk))
                    mapping.append(xk, yks)
                else:
                    mapping = b.x_y_map
                for xk, yks in mapping:
                    # get filter_key
                    if xk == "@":
                        fk = b.x_filter_map[yks[0]]
                    else:
                        fk = b.x_filter_map[xk]
                    # add bases
                    self._add_base_views(dk, fk, xk, yks, bases, b_weights)
                    # remove existing nets for link if new view is a net
                    if isinstance(views, ViewMapper) and views.get("net"):
                        for yk in yks:
                            link = self[dk][fk][xk][yk]
                            link._remove_nets(b.weights)
                    # add unweighted views for counts/ nets
                    if unwgt_c:
                        self.add_link(dk, fk, xk, yks, views=c_n_views)
                    # add common views
                    if not(x in numerics and x not in categorize):
                        self.add_link(dk, fk, xk, yks, views, b.weights)
                    else:
                        msg = (
                            "Warning: '{}' is non-categorized numeric variable"
                            ". Descriptive statistics must be added!")
                        logger.warning(msg.format(x))
                    # remove views if complete (cumsum/ nets)
                    if complete:
                        for yk in yks:
                            if all([b.y_on_y,
                                    xk in b.yks,
                                    fk in list(b.y_filter_map.values())]):
                                continue
                            link = self[dk][fk][xk][yk]
                            link._remove_completes(b.weights)
                for yy in b.y_on_y:
                    # get filter_key
                    fk = b.y_filter_map[yy]
                    for yk in b.yks:
                        if xks and yk not in xks:
                            warn = "'{}' is not in '{}'".format(yk, batch)
                            logger.warning(warn)
                            continue
                        # add bases
                        self._add_base_views(dk, fk, yk, yks, bases, b_weights)
                        # remove existing nets for link if new view is a net
                        if isinstance(views, ViewMapper) and views.get("net"):
                            for yk2 in yks:
                                link = self[dk][fk][yk][yk2]
                                link._remove_nets(b.weights)
                        # add unweighted views for counts/ nets
                        if unwgt_c:
                            self.add_link(dk, fk, yk, yks, views=c_n_views)
                        # add common views
                        self.add_link(dk, fk, yk, yks, views, b.weights)

    def _dissect_views(self, views, bases):
        complete = False
        count_net_views = ['counts', 'counts_sum', 'counts_cumsum']
        if isinstance(views, ViewMapper):
            complete = views[views.keys()[0]]['kwargs'].get('complete', False)
            if "net" in views:
                counts_nets = ViewMapper()
                counts_nets.make_template('frequency', {'rel_to': [None, 'y']})
                options = {
                    'logic': views['net']['kwargs']['logic'],
                    'axis': 'x',
                    'expand': views['net']['kwargs']['expand'],
                    'complete': views['net']['kwargs']['complete'],
                    'calc': views['net']['kwargs']['calc']}
                counts_nets.add_method('net', kwargs=options)
        else:
            views_listed = ensure_list(views)
            views = []
            for view in views_listed:
                if "base" in view:
                    bases[view].update({"wgt": True, "unwgt": True})
                else:
                    if "cumsum" in view:
                        complete = True
                    views.append(view)
            counts_nets = [v for v in count_net_views if v in views]
        return views, bases, complete, counts_nets

    def _check_batches(self, dk, batches="all"):
        valid = self[dk].meta.batches
        if batches == "all":
            return valid
        elif not batches:
            return []
        else:
            return [batch for batch in batches if batch in valid]

    def _add_base_views(self, dk, fk, xk, yks, bases, weights=[]):
        for ba, weighted in bases.items():
            wgt = weighted.get('wgt')
            unwgt = weighted.get('unwgt')
            if wgt and weights:
                self.add_link(dk, fk, xk, yks, [ba], weights)
            if unwgt or (wgt and not weights):
                self.add_link(df, fk, xk, yks, [ba], None)

    # -------------------------------------------------------------------------
    # nets, stats, cumulative sums, sig tests
    # -------------------------------------------------------------------------
    def _add_checking_chain(self, dk, chainmanager, name, x, y, views):
        key, view, c_view = views
        c_stack = chainmanager.stack
        c_stack.add_link(x=x, y=y, views=view, weights=None)
        c_stack.add_link(x=x, y=y, views=c_view, weights=None)
        c_views = c_stack.describe('view').index.tolist()
        len_v_keys = len(view)
        view_keys = ['x|f|x:|||cbase', 'x|f|:|||counts'][0:len_v_keys]
        c_views = view_keys + [
            v for v in c_views if v.endswith('{}_check'.format(key))]
        chainmanager.get(
            'checks', 'no_filter', x, y, c_views, folder=name, rules=False)

    @params(to_list=['on_vars', '_batches'])
    def add_nets(self, on_vars, net_map, expand=None, calc=None, rebase=None,
                 text_prefix='Net:', checking_cm=None, batches='all',
                 recode='auto', mis_in_rec=False):
        """
        Add a net-like view to a specified collection of x keys of the stack.

        Parameters
        ----------
        on_vars : list
            The list of x variables to add the view to.
        net_map : list of dicts
            The listed dicts must map the net/band text label to lists of
            categorical answer codes to group together, e.g.:
            ```
            net_map = [
                {'Top3': [1, 2, 3]},
                {'Bottom3': [4, 5, 6]}]
            ```
            It is also possible to provide enumerated net definition
            dictionaries that are explicitly setting ``text`` metadata per
            ``text_key`` entries:
            ```
            net_map = [
                {
                    1: [1, 2],
                    'text': {
                        'en-GB': 'UK NET TEXT',
                        da-DK': 'DK NET TEXT',
                        'de-DE': 'DE NET TEXT'}
                }]
            ```
        expand : {'before', 'after'}, default None
            If provided, the view will list the net-defining codes after or
            before the computed net groups (i.e. "overcode" nets).
        calc : dict, default None
            A dictionary that is attaching a text label to a calculation
            expression using the the net definitions. The nets are referenced
            as per 'net_1', 'net_2', 'net_3', ... .
            Supported calculation expressions are add, sub, div, mul. Example:
            ```
            calc = {
                'calc': ('net_1', add, 'net_2'), '
                text': {
                    'en-GB': 'UK CALC LAB',
                    'da-DK': 'DA CALC LAB',
                    'de-DE': 'DE CALC LAB'}}
            ```
        rebase : str, default None
            Use another variables margin's value vector for column percentage
            computation.
        text_prefix : str, default 'Net:'
            By default each code grouping/net will have its ``text`` label
            prefixed with 'Net: '. Toggle by passing None (or an empty str, '')
        checking_cm : quantipy.ChainManager, default None
            When provided, an automated checking aggregation will be added to
            the ``ChainManager`` instance.
        batches: str or list of str
            Only for ``qp.Links`` that are defined in this ``qp.Batch``
            instances views are added.
        recode: {'extend_codes', 'drop_codes', 'collect_codes',
            'collect_codes@cat_name'}, default 'auto'
            Adds variable with nets as codes to DataSet/Stack. If
            'extend_codes', codes are extended with nets. If 'drop_codes', new
            variable only contains nets as codes. If 'collect_codes' or
            'collect_codes@cat_name' the variable contains nets and another
            category that summarises all codes which are not included in any
            net. If no cat_name is provided, 'Other' is taken as default
        mis_in_rec: bool, default False
            Skip or include codes that are defined as missing when recoding
            from net definition.
        """
        for dk in self.keys():
            meta = self[dk].meta
            on_vars = meta.unroll(on_vars, both="all")
            check_on = [
                meta.get_sources(var)[0] if meta.is_array(var) else var
                for var in on_vars if not meta.is_array_item(var)]

            tks = meta.used_text_keys()
            netdef = self._netdef_from_map(net_map, expand, text_prefix, tks)
            calc_only = False
            if calc:
                calc = self._check_and_update_calc(calc, tks)
                calc_only = calc.pop('calc_only', False)

            view = ViewMapper()
            rel_to = "y" if not rebase else "{}.base".format(rebase)
            view.make_template('frequency', {'rel_to': [None, rel_to]})
            options = {
                'logic': netdef,
                'axis': 'x',
                'expand': expand if expand in ['after', 'before'] else None,
                'complete': True if expand else False,
                'calc': calc,
                'calc_only': calc_only}
            view.add_method('net', kwargs=options)
            self.aggregate(view, batches=batches, xks=on_vars)

            if recode:
                from .dataset import DataSet
                ds = DataSet.from_stack(dk)
                ds.recode_from_net_def(
                    on_vars, net_map, expand, recode, text_prefix, mis_in_rec)

            if isinstance(checking_cm, Chainmanager) and check_on:
                view['net_check'] = view.pop('net')
                view['net_check']['kwargs']['iterators'].pop('rel_to')
                for v in check_on:
                    v_net = '{}_net'.format(v)
                    v_net = v_net.split('.')[-1]
                    if v_net not in checking_cm.folder_names:
                        self._add_checking_chain(
                            dk, checking_cm, v_net, v, ['@', v],
                            ('net', ['cbase'], view))
            else:
                logger.info("Skipping checks.")

    @staticmethod
    def _netdef_from_map(net_map, expand, prefix, text_key):
        netdef = []
        for no, net in enumerate(net_map, start=1):
            if 'text' in net:
                logic = net[no]
                text = net['text']
            else:
                logic = net.values()[0]
                text = {t: net.keys()[0] for t in text_key}
            logic = ensure_list(logic)
            for k, v in text.items():
                if expand:
                    text[k] = "{} (NET)".format(v)
                elif prefix:
                    text[k] = "{} {}".format(prefix, v)
            netdef.append({'net_{}'.format(no): logic, 'text': text})
        return netdef

    @staticmethod
    def _check_and_update_calc(calc_expression, text_key):
        if not isinstance(calc_expression, dict):
            err = (
                "'calc' must be a dict in form of\n"
                "{'calculation label': (net # 1, operator, net # 2)}")
            logger.error(err); raise TypeError(err_msg)
        for k, v in calc_expression.items():
            if k not in ['text', 'calc_only']:
                exp = v
            if not k == 'calc_only':
                text = v
        if 'text' not in calc_expression:
            text = {tk: text for tk in text_key}
            calc_expression['text'] = text
        if not isinstance(exp, (tuple, list)) or len(exp) != 3:
            err_msg = (
                "Not properly formed expression found in 'calc':\n"
                "{}\nMust be provided as (net # 1, operator, net # 2)")
            err_msg = err_msg.format(exp)
            logger.error(err_msg); raise TypeError(err_msg)
        return calc_expression

    @params(to_list=['on_vars', 'stats', 'exclude', '_batches'])
    def add_stats(self, on_vars, stats=['mean'], other_source=None,
                  rescale=None, drop=True, exclude=None, factor_labels=True,
                  custom_text=None, batches='all', checking_cm=None):
        """
        Add a descriptives view to a specified collection of xks of the stack.

        Valid descriptives views:
            {'mean', 'stddev', 'min', 'max', 'median', 'sem'}

        Parameters
        ----------
        on_vars : list
            The list of x variables to add the view to.
        stats : list of str, default ``['mean']``
            The metrics to compute and add as a view.
        other_source : str
            If provided the Link's x-axis variable will be swapped with the
            (numerical) variable provided. This can be used to attach
            statistics of a different variable to a Link definition.
        rescale : dict
            A dict that maps old to new codes,
            e.g. {1: 5, 2: 4, 3: 3, 4: 2, 5: 1}
        drop : bool, default True
            If ``rescale`` is provided all codes that are not mapped will be
            ignored in the computation.
        exclude : list
            Codes/values to ignore in the computation.
        factor_labels : bool / str, default True
            Writes the (rescaled) factor values next to the category text label
            If True, square-brackets are used.
            If '()', normal brackets are used.
        custom_text : str, default None
            A custom string affix to put at the end of the requested
            statistics' names.
        checking_cm : quantipy.Chainmanager, default None
            When provided, an automated checking aggregation will be added to
            the ``Chainmanager`` instance.
        batches: str or list of str
            Only for ``qp.Links`` that are defined in this ``qp.Batch``
            instances views are added.
        """
        from .dataset import DataSet

        if other_source and not isinstance(other_source, str):
            err = "'other_source' must be a str!"
            logger.error(err); raise ValueError(err)

        drop = False if not rescale else drop

        warn = "Cannot add stats on '{}'."
        options = {
            'stats': '',
            'source': other_source,
            'rescale': rescale,
            'drop': drop,
            'exclude': exclude,
            'axis': 'x',
            'text': '' if not custom_text else custom_text}

        for dk in self.keys():
            ds = DataSet.from_stack(dk)
            apply_to = ds.unroll(on_vars, both="all")
            check_on = [
                ds.get_sources(var)[0] if ds.is_array(var) else var
                for var in apply_to if not ds.is_array_item(var)]
            no_os = not other_source
            for v in apply_to:
                w = None
                if ds.is_delimited_set(v) and no_os:
                    w = warn + 'Stats are not valid on delimited sets!'
                elif not ds.is_categorical(v) and no_os:
                    w = warn + 'No values defined!'
                if w:
                    logger.warning(w.format(v))
                    try:
                        apply_to.remove(v)
                        check_on.remove(v)
                    except ValueError:
                        pass
                    continue
                if no_os:
                    ds.meta._add_factor_meta(v, rescale, drop, exclude)

            view = ViewMapper()
            view.make_template('descriptives')
            for stat in stats:
                options['stats'] = stat
                view.add_method('stat', kwargs=options)
                self.aggregate(view, batches=batches, xks=apply_to)

            if factor_labels:
                args = [rescale, drop, exclude, factor_labels]
                for batch in self._check_batches(batches):
                    b = ds.get_batch(batch)
                    b.meta._add_factor_labs(check_on, *args)
            if checking_cm and 'mean' in stats and check_on:
                cm_meta = checking_cm.stack['checks'].meta
                cm_meta._add_factor_labs(check_on, *args)
                options['stats'] = 'mean'
                c_view = ViewMapper().make_template('descriptives')
                c_view.add_method('stat_check', kwargs=options)

                views = ('stat', ['cbase', 'counts'], c_view)
                self._add_checking_chain(
                    dk, checking_cm, 'stat_check', check_on, ['@'], views)

    @params(to_list=['on_vars', '_batches'])
    def cumulative_sum(self, on_vars, _batches='all', verbose=True):
        """
        Add cumulative sum view to a specified collection of xks of the stack.

        Parameters
        ----------
        on_vars : list
            The list of x variables to add the view to.
        _batches: str or list of str
            Only for ``qp.Links`` that are defined in this ``qp.Batch``
            instances views are added.

        Returns
        -------
        None
            The stack instance is modified inplace.
        """
        views = ['counts_cumsum', 'c%_cumsum']
        for dk in self.keys():
            xks = self[dk].meta.unroll(on_vars, both="all")
            self.aggregate(views, batches=batches, xks=xks)

    @params(to_list=['batches'])
    def add_tests(self, batches='all', verbose=True):
        """
        Apply coltests for selected batches.

        Sig. Levels are taken from ``qp.Batch`` definitions.

        Parameters
        ----------
        _batches: str or list of str
            Only for ``qp.Links`` that are defined in this ``qp.Batch``
            instances views are added.

        Returns
        -------
        None
        """
        # self._remove_coltests()
        for dk in self.keys():
            ds = DataSet.from_stack(dk)
            use_batches = self._check_batches(dk, batches)
            for batch in use_batches:
                b = ds.get_batch(batch)
                sigpro = b.sigproperties
                levels = sigpro.get('siglevels', [])
                mimic = sigpro.get('mimic', ['Dim'])
                flag = sigpro.get('flag_bases', [30, 100])
                total = sigpro.get('test_total', None)

                vm_tests = ViewMapper().make_template(
                    method='coltests',
                    iterators={
                        'metric': ['props', 'means'],
                        'mimic': mimic,
                        'level': levels})
                vm_tests.add_method(
                    'significance',
                    kwargs={
                        'flag_bases': flag,
                        'test_total': total,
                        'groups': 'Tests'})
                for xk, yks in b.x_y_map:
                    if xk == "@":
                        continue
                    fk = b.x_filter_map[xk]
                    for yk in yks:
                        link = self[dk][fk][xk][yk]
                        link._remove_coltests()
                    if levels:
                        self.add_link(dk, fk, xk, yks, vm_tests, b.weights)
                for yy in b.y_on_y:
                    fk = b.y_filter_map[yy]
                    xks = b.yks[1:] if b.total else b.yks[:]
                    for xk in xks:
                        for yk in b.yks:
                            link = self[dk][fk][xk][yk]
                            link._remove_coltests()
                    if levels:
                        self.add_link(dk, fk, xks, b.yks, vm_tests, b.weights)

    def get_chain(self, *args, **kwargs):
        from .chains.chainmanager import ChainManager
        chain = ChainManager(self)
        chain = chain.get(*args, **kwargs)
        return chain


class Link(dict):
    """
    The Link object is a subclassed dictionary that generates an instance of
    Pandas.DataFrame for every view method applied
    """
    def __init__(self, stack, data_key, filter_key, x_key, y_key):

        self.fk = filter_key
        self.yk = y_key
        self.xk = x_key
        self.dk = data_key
        self.stack = stack
        self.transpose = False

        self._path = [self.dk, self.fk, self.xk, self.yk]

    def __getitem__(self, key):
        val = dict.__getitem__(self, key)
        if self.transpose and "T" in dir(val):
            return val.T
        else:
            return val

    @property
    def meta(self):
        return self.stack[self.dk].meta

    @property
    def data(self):
        if self.fk:
            return self.stack[self.dk][self.fk].data
        else:
            return self.stack[self.dk].data

    @property
    def cache(self):
        return self.stack[self.dk].cache

    def _remove_nets(self, weights=[]):
        for vk in self.keys()[:]:
            if vk.split('|')[-1] == 'net':
                w = vk.split('|')[-2] or None
                if w in weight:
                    del self[vk]

    def _remove_completes(self, weights=[]):
        for vk in self.keys()[:]:
            vks = vk.split("|")
            if vks[-2] in weights and vks[-1] == "c%" and vks[-3] == "y":
                del self[vk]
            elif vks[-2] in weights and vks[-1] == "counts" and vks[-3] == "":
                del self[vk]

    def _remove_coltests(self, props=True, means=True):
        for vk in self:
            del_prop = props and 't.props' in vk
            del_mean = means and 't.means' in vk
            if del_prop or del_mean:
                del self[vk]

    @params(to_list="names")
    def add_views(self, names, method=None, iterators={}, kwargs={}):
        vm = ViewMapper(names)
        if method is not None:
            vm.make_template(method, iterators)
            for name in names:
                vm.add_method(name, kwargs)
        vm._apply_to(self)


class Cache(defaultdict):

    def __init__(self):
        super(Cache, self).__init__(Cache)

    def __reduce__(self):
        return self.__class__, tuple(), None, None, self.iteritems()

    def set_obj(self, collection, key, obj):
        """
        Save a Quantipy resource inside the cache.

        Parameters
        ----------
        collection : {
            "matrices", "weight_vectors", "quantities", "mean_view_names",
            "count_view_names"}
            The key of the collection the object should be placed in.
        key : str
            The reference key for the object.
        obj : Specific Quantipy or arbitrary Python object.
            The object to store inside the cache.
        """
        self[collection][key] = obj

    def get_obj(self, collection, key):
        """
        Look up if an object exists in the cache and return it.

        Parameters
        ----------
        collection : {
            "matrices", "weight_vectors", "quantities", "mean_view_names",
            "count_view_names"}
            The key of the collection to look into.
        key : str
            The reference key for the object.

        Returns
        -------
        obj : Specific Quantipy or arbitrary Python object.
            The cached object mapped to the passed key.
        """
        if key in self[collection]:
            return self[collection][key]
        elif collection == "matrices":
            return (None, None)
        elif collection == "squeezed":
            return (None, None, None, None, None, None, None)
        else:
            return None
