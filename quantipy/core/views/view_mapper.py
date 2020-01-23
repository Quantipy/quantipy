#!/usr/bin/python
# -*- coding: utf-8 -*-
from ...__imports__ import *  # noqa

from .view import View
from ..engine import (
    Quantity,
    Test)


class ViewMapper(OrderedDict):
    """
    Applies View computation results to Links based on the view method's
    kwargs, handling the coordination and structuring side of the aggregation
    process.
    """
    def __init__(self, views=[], template=None):
        super(ViewMapper, self).__init__()
        self.template = template
        for view in views:
            self[view] = copy.deepcopy(KNOWN_METHODS.get(view, {}))

    def __setstate__(self, view_dict):
        # Reconstruct the View object after serialization
        for view in view_dict:
            kwargs = view_dict[view]['kwargs']
            method_as_bytes = view_dict[view]['method']
            method_as_string = marshal.loads(method_as_bytes)
            self[view] = {
                'kwargs': kwargs,
                'method': FunctionType(
                    code=method_as_string, globals=globals(), name=view)}

    def __reduce__(self):
        class_type = self.__class__
        arguments = self.keys()
        setitem_dict = {}

        # Must use .keys() because the loop removes callable methods from self
        for vk in self.keys():
            if callable(self[vk]['method']):
                view = self.pop(vk)
                kwargs = view['kwargs']
                method = view['method']
                setitem_dict[vk] = {
                    'method':
                        marshal.dumps(method.__code__), 'kwargs': kwargs}

        return (class_type, tuple(arguments), setitem_dict, None, None)

    def make_template(self, method, iterators={}):
        """
        Generate a view method template that cycles through kwargs values.

        Parameters
        ----------
        method : {'frequency', 'descriptives', 'coltests'}
            The baseline view method to be used.
        iterators : dict
            A dictionary mapping of view method kwargs to lists of values.
        """
        if not (method in ["default", "frequency", "descriptives"] or
                callable(method)):
            raise TypeError("Given 'method' needs to be callable.")

        self.template = {
            "method": method,
            "kwargs": {"iterators": iterators}
        }

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
        """
        template = template or self.template
        if template:
            name = template.get('name', name)
            method = template.get('method', method)
            kwargs = dict(template.get('kwargs', {}), **kwargs)

        if None in [name, method]:
            err = (
                "You must provide a 'name' and 'method' to add_method()"
                "either directly in the method call or through a ViewMapper"
                " template.")
            raise ValueError(err)

        self[name] = {'method': method, 'kwargs': kwargs}

    @params(to_list="views")
    def subset(self, views, inplace=False):
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
        valids = [view for view in views if view in self]
        if not valids:
            err = "No valid view given!"
            raise KeyError(err)
        if inplace:
            subset = self
        else:
            subset = self.copy()
        keys = list(subset.keys())
        for view in keys:
            if view not in views:
                del subset[view]
        if not inplace:
            return subset

    def _apply_to(self, link, weights=[]):
        """
        Loop through view methods applying them to the Link.

        Parameters
        ----------
        link : Link
        weights : Weight variable as str or list of str
        """
        print('*'*60)
        print(self.items())
        for view, defs in list(self.items())[:]:
            print(view)

            method = defs["method"]
            kwargs = copy.deepcopy(defs.get("kwargs", {}))
            iterators = kwargs.pop('iterators', {})

            # weights parameter > weights kwargs + weights iterators
            if not weights:
                view_weights = ensure_list(kwargs.get("weights", [None]))
                if "weights" not in iterators:
                    iterators['weights'] = []
                for weight in view_weights:
                    if weight not in iterators["weights"]:
                        iterators["weights"].append(weight)
            else:
                iterators['weights'] = weights[:]

            # extend iterators rel_to by kwargs rel_to
            rel_to = ensure_list(kwargs.get('rel_to') or [""])
            if not 'rel_to' in iterators:
                iterators['rel_to'] = []
            iterators['rel_to'].extend(rel_to)
            iterators['rel_to'] = list(set([
                rel if rel else "" for rel in iterators['rel_to']]))

            # Get the product of all the targeted iterators
            keys, items = zip(*iterators.items())
            view_iterations = [dict(zip(keys, x)) for x in product(*items)]

            # Run the view method for all the requested iterations
            for view_iter in view_iterations:
                kwargs.update(view_iter)
                if isinstance(method, str):
                    getattr(self, method)(link, view, kwargs)
                else:
                    method(link, view, kwargs)
                print(link)

    def default(self, link, name, kwargs):
        """
        Adds a file meta dependent aggregation to a Stack.

        Checks the Link definition against the file meta and produces
        either a numerical or categorical summary tabulation including
        marginal the results.

        Parameters
        ----------
        link : Quantipy Link object.
        name : str
            The shortname applied to the view.
        kwargs : dict
        """
        view = View(link, name, "default", kwargs)
        meta = link.meta
        q = Quantity(link, weight=view.weight)
        if not(q.type == 'array' and not q.yk == '@'):
            x_is_categorical = meta.is_categorical(link.xk)
            x_is_numeric = meta.is_numeric(link.xk)
            x_is_array = meta.is_array(link.xk)
            y_is_categorical = meta.is_categorical(link.yk)
            y_is_numeric = meta.is_numeric(link.yk)
            if link.yk == '@':
                if x_is_categorical or x_is_array:
                    view_df = q.count().result
                elif x_is_numeric:
                    view_df = q.summarize().result
                    view_df.drop((link.xk, 'All'), axis=0, inplace=True)
            elif link.xk == '@':
                if y_is_categorical:
                    view_df = q.count().result
                elif y_is_numeric:
                    view_df = q.summarize().result
                    view_df.drop((link.yk, 'All'), axis=1, inplace=True)
            else:
                if x_is_categorical and (y_is_categorical or y_is_numeric):
                    view_df = q.count().result
                elif x_is_numeric and (y_is_categorical or y_is_numeric):
                    view_df = q.summarize().result
                    view_df.drop((link.xk, 'All'), axis=0, inplace=True)
                    view_df.drop((link.yk, 'All'), axis=1, inplace=True)
            view.dataframe = view_df
            link[view.notation] = view

    def frequency(self, link, name, kwargs):
        """
        Adds count-based views on a Link defintion to the Stack object.

        ``frequency`` is able to compute several aggregates that are based on
        the count of code values in uni- or bivariate Links. This includes
        bases / samples sizes, raw or normalized cell frequencies and code
        summaries like simple and complex nets.

        Parameters
        ----------
        link : Quantipy Link object.
        name : str
            The shortname applied to the view.
        kwargs : dict
        Keyword arguments (specific)
        text : str, optional, default None
            Sets an optional label in the meta component of the view that is
            used when the view is passed into a Quantipy build (e.g. Excel,
            Powerpoint).
        logic : list of int, list of dicts or core.tools.view.logic operation
            If a list is passed this instructs a simple net of the codes given
            as int. Multiple nets can be generated via a list of dicts that
            map names to lists of ints. For complex logical statements,
            expression are parsed to identify the qualifying rows in the data.
            For example::

                # simple net
                'logic': [1, 2, 3]

                # multiple nets/code groups
                'logic': [{'A': [1, 2]}, {'B': [3, 4]}, {'C', [5, 6]}]

                # code logic
                'logic': has_all([1, 2, 3])
        """
        view = View(link, name, "f", kwargs)
        meta = link.meta
        q = Quantity(
            link,
            weight=view.weight,
            ignore_flags=view.kwargs.get("ignore_flags", False))
        if not(q.type == 'array' and not q.yk == '@'):
            if q.leveled:
                # leveled
                leveled = Level(q)
                if rel_to:
                    leveled.percent()
                elif kwargs.get("axis") == 'x':
                    leveled.base()
                else:
                    leveled.count()
                view.dataframe = leveled.lvldf
            elif view._logic:
                # nets
                q.group(
                    groups=view._logic,
                    axis=view.axis,
                    expand=view._expand,
                    complete=view._complete)
                q.count(axis=None, as_df=False, margin=False)
                view.spec_condition()
            else:
                # others
                q.count(
                    axis=view.axis,
                    raw_sum=view.kwargs.get("raw_sum", False),
                    cum_sum=view.kwargs.get("cum_sum", False),
                    effective=view.kwargs.get("effective", False),
                    margin=False, as_df=False)
            per_cell = False
            rel_to = view.rel_to
            if len(rel_to.split(".")) == 2:
                rel_to, rel_to_kind = rel_to.split(".")
                if rel_to_kind == "cells":
                    per_cell = True
                if name == "counts":
                    view.kwargs["rebased"] = True
            if rel_to:
                if q.type == 'array':
                    rel_to = 'y'
                q.normalize(rel_to, per_cell=per_cell)
            q.to_df()
            view._cbases = q.cbase
            view._rbases = q.rbase
            if view._calc:
                q.calc(view._calc, view.axis, result_only=view._calc_only)
                view._method = "f.c:f"
            elif name in ['counts_sum', 'c%_sum', 'counts_cumsum',
                          'c%_cumsum']:
                view._method = "f.c:f"
            else:
                view._method = "f"
            if not q.leveled:
                if q.type == 'array' and link.yk == "@":
                    view.dataframe = q.result.T
                else:
                    view.dataframe = q.result
            view.kwargs["exclude"] = q.miss_x
            view.spec_condition()
            link[view.notation] = view

    def descriptives(self, link, name, kwargs):
        """
        Adds num. distribution statistics of a Link defintion to the Stack.

        ``descriptives`` views can apply a range of summary statistics.
        Measures include statistics of centrality, dispersion and mass.

        Parameters
        ----------
        link : Quantipy Link object.
        name : str
            The shortname applied to the view.
        kwargs : dict
        Keyword arguments (specific)
        text : str, optional, default None
            Sets an optional label suffix for the meta component of the view
            which will be appended to the statistic name and used when the
            view is passed into a Quantipy build (e.g. Excel, Powerpoint).
        stats : str, default 'mean'
            The measure to compute.
        exclude : list of int
             Codes that will not be considered calculating the result.
        rescale : dict
            A mapping of {old code: new code}, e.g.::

                {
                 1: 0,
                 2: 25,
                 3: 50,
                 4: 75,
                 5: 100
                }
        drop : bool
            If ``rescale`` provides a new scale defintion, ``drop`` will remove
            all codes that are not transformed. Acts as a shorthand for manually
            passing any remaining codes in ``exclude``.
        """
        print('la')
        view = View(link, name, "descriptives", kwargs)
        if not view._xk['is_multi'] or view._source:

            view.kwargs['calc_only'] = True

            q = Quantity(link, view.weight)
            if view._source:
                q = self._swap_and_rebase(q, view._source)
            if not(q.type == 'array' and q.yk == '@'):
                if not view._stats:
                    view.kwargs["stats"] = "mean"
                if view._exclude:
                    q.exclude(view._exclude, axis=view.axis)
                if view._rescale:
                    q.rescale(view._rescale, view._drop)

                q.summarize(stat=view._stats, margin=False, as_df=True)

                if view._calc:
                    q.calc(view._calc, result_only=True)
                    method_nota = 'd.' + view._stats + '.c:f'
                else:
                    method_nota = 'd.' + view._stats
                view._method = method_nota
                view._cbases = q.cbase
                view._rbases = q.rbase
                if q.type == 'array':
                    view.dataframe = q.result.T
                else:
                    view.dataframe = q.result
                view.kwargs["exclude"] = q.miss_x
                view.translate_metric()
                view.spec_condition()
                print(view.notation)
                link[view.notation] = view

    @staticmethod
    def _swap_and_rebase(quantity, variable, axis='x'):
        rebase_on = {quantity.xk: not_count(0)}
        quantity.swap(var=variable, axis=axis, update_axis_def=False)
        try:
            quantity.filter(rebase_on, keep_base=False, inplace=True)
        except KeyError:
            warn = "Couldn't rebase 'source'-swapped array-type Quantity: {} "
            warn += "on {}\nPlease check descriptive stats results for correct"
            warn += " base sizes!"
            warnings.warn(warn.format(quantity.xk, variable))
        return quantity

    def coltests(self, link, name, kwargs):
        """
        Will test appropriate views from a Stack for stat. sig. differences.

        Tests can be performed on frequency aggregations (generated by
        ``frequency``) and means (from ``summarize``) and will compare all
        unique column pair combinations.

        Parameters
        ----------
        link : Quantipy Link object.
        name : str
            The shortname applied to the view.
        kwargs : dict
        Keyword arguments (specific):
        text : str, optional, default None
            Sets an optional label in the meta component of the view that is
            used when the view is passed into a Quantipy build (e.g. Excel,
            Powerpoint).
        metric : {'props', 'means'}, default 'props'
            Determines whether a proportion or means test algorithm is
            performed.
        test_total : bool, deafult False
            If True, the each View's y-axis column will be tested against the
            uncoditional total of its x-axis.
        mimic : {'Dim', 'askia'}, default 'Dim'
            It is possible to mimic the test logics used in other statistical
            software packages by passing them as instructions. The method will
            then choose the appropriate test parameters.
        level: {'high', 'mid', 'low'} or float
            Sets the level of significance to which the test is carried out.
            Given as str the levels correspond to ``'high'`` = 0.01, ``'mid'``
            = 0.05 and ``'low'`` = 0.1. If a float is passed the specified
            level will be used.
        flags : list of two int, default None
            Base thresholds for Dimensions-like tests, e.g. [30, 100]. First
            int is minimum base for reported results, second int controls small
            base indication.

        Returns
        -------
        None
            Adds requested View to the Stack, storing it under the full
            view name notation key.

        .. note::

            Mimicking the askia software (``mimic`` = ``'askia'``)
            restricts the values to be one of ``'high'``, ``'low'``,
            ``'mid'``. Any other value passed will make the algorithm fall
            back to ``'low'``. Mimicking Dimensions (``mimic`` =
            ``'Dim'``) can use either the str or float version.
        """
        if not kwargs.get("metric"):
            kwargs["metric"] = "props"
        get = 'count' if kwargs["metric"] == 'props' else 'mean'
        views = self._get_view_names(link, kwargs.get("weights"), get=get)
        for vk in views:
            dep_view = link[vk]
            view = View(link, name, "coltests", kwargs)
            test = Test(link, vk, view._test_total or False)
            if not view._level:
                view.kwargs["level"] = "low"
            if not view._mimic:
                view.kwargs["mimic"] = "Dim"
            if view._mimic == "Dim":
                test.set_params(level=view._level, flag_bases=view._flag_bases)
            elif view._mimic == 'askia':
                test.set_params(
                    testtype='unpooled',
                    level=view._level,
                    mimic=view._mimic,
                    use_ebase=False,
                    ovlp_correc=False,
                    cwi_filter=True)
            view.kwargs["level"] = test.level
            view.dataframe = test.run()

            view.condition = dep_view.condition
            view._method = "t.{}.{}.{}{}".format(
                view._metric,
                view._mimic,
                "{:.2f}".format(view._level)[2:],
                "+@" if view._test_total else "")
            link[view.notation] = view

    @staticmethod
    def _get_view_names(link, weight, get='count'):
        """
        Filter the views contained in a Stack by specific names.
        """
        if not weight:
            weight = ""
        collection = "{}_view_names".format(get)
        key = "{}_names".format(weight or "")
        view_name_list = link.cache.get_obj(collection, key)
        if not view_name_list:
            allviews = link.keys()
            if get == 'count':
                view_name_list = [
                    vk for vk in link.keys()
                    if all([
                        link[vk].is_counts,
                        link[vk].weight == weight,
                        not link[vk].is_base,
                        not link[vk].is_sum])]
            else:
                view_name_list = [
                    vk for vk in link.keys()

                    if all([
                        link[vk]._method == "d.mean",
                        link[vk].weight == weight
                        ])]
            link.cache.set_obj(collection, key, view_name_list)
        return view_name_list
