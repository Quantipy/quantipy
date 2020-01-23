#!/usr/bin/python
# -*- coding: utf-8 -*-
from ..__imports__ import *  # noqa

logger = get_logger(__name__)


class Rules(object):

    # ------------------------------------------------------------------------
    # init
    # ------------------------------------------------------------------------
    def __init__(self, link, vk, rweight=None, axes=['x', 'y']):
        self.vk = vk
        self.link = link
        self.meta = link.meta
        self.axes = axes or []
        if link.xk == "@":
            self.transposed = True
            self.xk = link.yk
            self.yk = link.xk
            self._df = link[vk].dataframe.T.copy()
        else:
            self.transposed = False
            self.xk = link.xk
            self.yk = link.yk
            self._df = link[vk].dataframe.copy()

        self._x_rules = None
        self._y_rules = None
        self._sort_weight = self._get_sort_weight(rweight)
        self.x_slicer = None
        self.y_slicer = None

    @property
    def df(self):
        if self.transposed:
            return self._df.T
        else:
            return self._df

    @property
    def x_rules(self):
        # lazy property
        if self._x_rules is None:
            if "x" in self.axes:
                self._x_rules = self.meta.get_rules(self.xk, "x")
            else:
                self._x_rules = {}
        return self._x_rules

    @property
    def y_rules(self):
        # lazy property
        if self._y_rules is None:
            if "y" in self.axes:
                self._y_rules = self.meta.get_rules(self.yk, "y")
            else:
                self._y_rules = {}
        return self._y_rules

    @property
    def is_array_summary(self):
        return self.meta.is_array(self.xk)

    def _get_sort_weight(self, use_weight):
        if 'sortx' in self.x_rules:
            sort_on = self.x_rules['sortx'].get('sort_on', '@')
            sort_weight = self.x_rules['sortx']['with_weight'] or ''
            if sort_weight == 'auto' and use_weight is None:
                sort_weight = self.link[self.vk].weight
            else:
                sort_weight = use_weight
            return sort_weight
        else:
            return ""

    def _other_link(self, col):
        return self.link.stack[self.link.dk][self.link.fk][col]["@"]

    # -------------------------------------------------------------------------
    # apply rules
    # -------------------------------------------------------------------------
    def apply(self):
        self._get_slicer()

        viable_axes = self.rule_viable_axes()
        if not viable_axes:
            return None
        else:
            df = self._df

        if 'x' in viable_axes and self.x_slicer:
            rule_codes = set(self.x_slicer)
            view_codes = set(df.index.tolist())
            if not rule_codes - view_codes:
                df = df.loc[self.x_slicer]

        if 'y' in viable_axes and self.y_slicer:
            df = df[self.y_slicer]
            if self.link[self.vk].method == "coltests":
                df = self.verify_test_results(df)
        self._df = df

    def rule_viable_axes(self):
        view = self.link[self.vk]
        condition = view.condition
        viable_axes = ['x', 'y']
        # x conditions to remove
        cond1 = all([
            any([
                condition.startswith("x") and not view.is_expanded_net,
                view.is_stat]),
            not self.is_array_summary])
        cond2 = not re.search('x\[.+:y$', condition) == None
        cond3 = not re.search('y:x\[.+', condition) == None
        if any([cond1, cond2, cond3]):
            viable_axes.remove('x')
        # y conditions to remove
        cond1 = ":" in condition and condition.split(":")[1].startswith("y")
        cond2 = not re.search('x:y\[.+', condition) == None
        cond3 = not re.search('y\[.+:x$', condition) == None
        cond4 = all([
            self.yk == "@",
            self.is_array_summary,
            view.name in ['counts', 'c%', 'r%']])
        if any([cond1, cond2, cond3, cond4]):
            viable_axes.remove('y')
        return viable_axes

    @staticmethod
    def verify_test_results(df):
        """
        Verify tests results in df are consistent with existing columns.

        This function verifies that all of the test results present in df
        only refer to column headings that actually exist in df. This is
        needed after rules have been applied at which time some columns
        may have been dropped.

        Parameters
        ----------
        df : pandas.DataFrame
            The view dataframe showing column tests results.

        Returns
        -------
        df : pandas.DataFrame
            The view dataframe showing edited column tests results.
        """
        def verify_test_value(value):
            """
            Verify a specific test value.
            """
            if isinstance(value, str):
                is_minimum = False
                is_small = False
                if value.endswith('*'):
                    if value.endswith('**'):
                        is_minimum = True
                        value = value[:-2]
                    else:
                        is_small = True
                        value = value[:-1]
                if len(value)>0:
                    if len(value)==1:
                        value = set(value)
                    else:
                        value = set([int(i) if i.isdigit() else i
                                     for i in list(value[1:-1].split(', '))])
                    value = cols.intersection(value)
                    if not value:
                        value = ''
                    elif len(value)==1:
                        value = str(list(value))
                    else:
                        value = str(sorted(list(value)))
                if is_minimum:
                    value = value + '**'
                elif is_small:
                    value = value + '*'
                elif len(value) == 0:
                    value = np.NaN

                return value
            else:
                return value

        cols = set([int(v) for v in zip(*[c for c in df.columns])[1]])
        df = df.applymap(verify_test_value)

        return df

    # -------------------------------------------------------------------------
    # prepare slicer
    # -------------------------------------------------------------------------
    def _get_slicer(self):
        if not self.xk == "@":
            self._get_x_slicer()
        if not self.yk == "@":
            self._get_y_slicer()

    def _get_x_slicer(self):
        if "sortx" in self.x_rules:
            sort_on = self.x_rules['sortx'].get('sort_on', '@')
            expanded_net = [
                vk for vk, view in self.link.items()
                if view.is_expanded_net and view.weight == self._sort_weight]
            if expanded_net and not self.is_array_summary:
                if not sort_on == '@':
                    err = 'Cannot sort expanded nets on {}.'.format(sort_on)
                    raise AttributeError(err)
                view = self._other_link(self.xk)[expanded_net[0]]
                f = self.sort_expanded_nets(view, self.x_rules['sortx'])
                self.x_slicer = self._get_rules_slicer(
                    f, self.x_rules, ['slicex', 'dropx'])
            # get df-desc-slice to sort on
            elif sort_on in STAT_VIEWS:
                f = self._get_descriptive_via_stack(sort_on)
            # get df-net-slice to sort on
            elif isinstance(sort_on, str) and sort_on.startswith('net'):
                f = self._get_net_via_stack(sort_on)
            # get df-freq-slice to sort on
            else:
                f = self._get_frequency_via_stack(self.xk, self._sort_weight)
        else:
            # get df for hiding + slicing
            f = self._get_frequency_via_stack(self.xk, None)
        self.x_slicer = self._get_rules_slicer(f, self.x_rules)

    def _get_y_slicer(self):
        f = self._get_frequency_via_stack(self.yk, None)
        self.y_slicer = self._get_rules_slicer(f, self.y_rules)

    def sort_expanded_nets(self, view, sortx):
        within = sortx.get('within', True)
        between = sortx.get('between', True)
        ascending = sortx.get('ascending', False)
        df = view.dataframe.copy()
        fix = [
            x for x in ensure_list(sortx.get('fixed', []))
            if x in df.index.get_level_values(1).tolist()]
        if not within and not between:
            return view.dataframe
        name = df.index.levels[0][0]
        col = (df.columns.levels[0][0], '@')
        # determine net groups + expanded codes vs. regular codes
        groups = view._expanded_net_groups
        sort = [
            (name, v) for v in df.index.get_level_values(1)
            if v not in fix and (v in groups['normal'] or
                                 v in groups["net"].keys())]
        temp_df = df.loc[sort]
        # sort between groups
        if between:
            temp_df = temp_df.sort_values(col, 0, ascending=ascending)
        # sort within the net groups
        final_index = []
        for idx in temp_df.index.get_level_values(1).tolist():
            if idx in groups["normal"]:
                final_indexa.append(idx)
            elif idx in groups["net"]:
                fixed_in_group = [x for x in groups["net"][idx] if x in fix]
                sort_in_group = [
                    (name, x) for x in groups["net"][idx] if x not in fix]
                temp_df = df.loc[sort_in_group]
                if within:
                    temp_df = temp_df.sort_values(col, 0, ascending=ascending)
                final_index.extend([idx])
                final_index.extend(temp_df.index.get_level_values(1).tolist())
                final_index.extend(fixed_in_group)
                fix = [c for c in fix if c not in fixed_in_group]
        # build final index including any fixed codes
        final_index = [(name, i) for i in final_index]
        if fix:
            final_index.extend([(name, f) for f in fix])
        df = df.reindex(final_index)
        return df

    def _get_frequency_via_stack(self, col, weight):
        if not weight:
            vk = 'x|f|:|||counts'
        else:
            vk = 'x|f|:||{}|counts'.format(weight)
            view_weight = self.link[self.vk].weight
            link_weights = [
                view.weight for view in self.link.values()
                if "base" not in view.name]
            if not (weight == view_weight or weight in link_weights):
                warn = ("{}: view-weight and weight to sort on differ ('{}' vs"
                        " '{}')")
                logger.warning(warn)
        link = self._other_link(col)
        if vk not in link:
            from .stack import Stack
            root = self.link.stack[self.link.dk]
            stack = Stack(
                "ct", add_data={"ct": {"meta": root.meta,
                                       "data": root.data}})
            stack.add_link(
                "ct", self.link.fk, col, "@", ["counts"], weight)
            link = stack["ct"][self.link.fk][col]["@"]
        return link[vk].dataframe

    def _get_descriptive_via_stack(self, desc='mean'):
        link = self._other_link(self.xk)
        desc_key = link.get_vks_by_method(
            "d.{}".format(desc), self._sort_weight, False)
        if not desc_key:
            err = "No {} view to sort '{}' on found!"
            raise RuntimeError(err.format(desc, self.xk))
        elif len(desc_key) > 1:
            err = "Multiple {} views found for '{}'. Unable to sort!"
            raise RuntimeError(err.format(desc, self.xk))
        return link[desc_key[0]].dataframe

    def _get_net_via_stack(self, net='net_1'):
        link = self._other_link(self.xk)
        net_no = int(net.split('_')[-1])
        net_keys = link.get_vks_by_name("net", self._sort_weight)
        net_key = [
            vk for vk in net_keys
            if len(link[vk].condition.split(",x")) >= net_no]
        if not net_key:
            err = "No net view to sort '{}' on found!".format(self.xk)
            raise RuntimeError(err)
        return link[net_key[0]].dataframe

    def _get_rules_slicer(self, f, rules, apply_rules=None):
        f = f.copy()
        rulesx = OrderedDict([
            ('slicex', self.slicex),
            ('sortx', self.sortx),
            ('dropx', self.dropx)])

        if not apply_rules:
            apply_rules = list(rulesx.keys())
        for r, method in rulesx.items():
            if apply_rules and r in apply_rules:
                if r in rules:
                    f = method(f, **rules[r])
        rules_slicer = f.index.values.tolist()
        col_key = f.index.levels[0].tolist()[0]
        if (col_key, 'All') in rules_slicer:
            rules_slicer.remove((col_key, 'All'))
        return rules_slicer

    def sortx(self, df, sort_on='@', within=True, between=True, ascending=False,
              fixed=None, with_weight='auto'):
        """
        Sort the index of df on a column, keeping margins and fixing values.

        This function sorts df, which is assumed to be a Quantipy-style
        view result with appropriate index/column structure, using
        a given column, while maintaining the position of margins if
        they exist, and also optionally fixing certain values at the
        bottom of the result without sorting them. Note that nested
        variable view results are not yet supported.

        Parameters
        ----------
        df : pandas.DataFrame
            The Quantipy-style view result to be sorted
        sort_on : str or int, default='@'
            The column (on the innermost level of the column's
            MultiIndex) on which to sort. By default sorting will be
            based on the unfiltered frequency of the x variable. No
            other sorting targets are currently supported.
        ascending : bool, default=False
            Sort ascending vs. descending. Default descending for
            easier application to MR use cases.
        within : bool, default True
            Applies only to variables that have been aggregated by creating a
            an ``expand`` grouping / overcode-style ``View``:
            If True, will sort frequencies inside each group.
        between : bool, default True
            Applies only to variables that have been aggregated by creating a
            an ``expand`` grouping / overcode-style ``View``:
            If True, will sort group and regular code frequencies with regard
            to each other.
        fixed : list-like, default=None
            A list of index values that should appear underneath
            the sorted index values.
        with_weight : None or str, default='auto'
            If not 'auto' this is name of the weight that is being used for
            the sort. 'auto' means that the same weight used in the original
            computation is also used in the sort, but this argument provides
            the ability to sort a computation done with one weight (or None)
            on the results of another weight (or None).

        Returns
        -------
        df : pandas.DataFrame
            The sorted df.
        """
        # If the index is from a frequency then the rule
        # should be skipped
        try:
            if df.index.levels[1][0]=='@':
                return df
            # Get question names for index and columns from the
            # index/column level 0 values
            name_x = df.index.levels[0][0]
            name_y = df.columns.levels[0][0]

            if (name_x, 'All') in df.index:
                # Get the margin slicer
                s_all = [(name_x, 'All')]
                # Get non-margin index slicer for the sort
                # (if fixed has been used it will be edited)
                s_sort = df.drop((name_x, 'All')).index.tolist()
            else:
                s_all = []
                s_sort = df.index.tolist()

            # Get fixed slicer
            if fixed is None:
                s_fixed = []
            else:
                s_fixed = [(name_x, value) for value in fixed
                           if (name_x, value) in s_sort]
                # Drop fixed tuples from the sort slicer
                s_sort = [t for t in s_sort if not t in s_fixed]

            # Get sorted slicer
            try:
                sort_on = int(sort_on)
            except:
                sort_on = str(sort_on)
            sort_col = (name_y, sort_on)
            df_sorted = df.loc[s_sort].sort_values(sort_col, 0, ascending)
            s_sort = df_sorted.index.tolist()
            df = df.loc[s_all + s_sort + s_fixed]
            return df
        except UnboundLocalError:
            logger.warning('Could not sort on {}'.format(sort_on))
            return df

    def slicex(self, df, values, keep_margins=True):
        """
        Return an index-wise slice of df, keeping margins if desired.

        Assuming a Quantipy-style view result this function takes an index
        slice of df as indicated by values and returns the result.

        Parameters
        ----------
        df : pandas.DataFrame
            The dataframe that should be sliced along the index.
        values : list-like
            A list of index values that should be sliced from df.
        keep_margins : bool, default=True
            If True and the margins index row exists, it will be kept.

        Returns
        -------
        df : list
            The sliced dataframe.
        """

        # If the index is from a frequency then the rule
        # should be skipped
        if df.index.levels[1][0]=='@':
            return df
        name_x = df.index.levels[0][0]
        slicer = [(name_x, value) for value in values]
        if keep_margins and (name_x, 'All') in df.index:
            slicer = [(name_x, 'All')] + slicer
        df = df.loc[slicer]
        return df

    def dropx(self, df, values):
        """
        Return df after dropping values from the index.

        Assuming a Quantipy-style view result this function drops index
        rows indicated by values and returns the result.

        Parameters
        ----------
        df : pandas.DataFrame
            The dataframe that should have some index rows dropped.
        values : list-like
            A list of index values that should be dropped from the index.

        Returns
        -------
        df : list
            The edited dataframe.
        """

        # If the index is from a frequency then the rule
        # should be skipped
        if df.index.levels[1][0]=='@':
            return df
        name_x = df.index.levels[0][0]
        slicer = [(name_x, value) for value in values
                  if (name_x, value) in df.index]
        if slicer:
            df = df.drop(slicer)
        return df
