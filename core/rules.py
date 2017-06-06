import re
import pandas as pd
from collections import OrderedDict
import copy
import quantipy as qp

class Rules(object):

    def __init__(self, link, view_name, axes=['x', 'y']):

        RULES_WEIGHT = None
        self.link = link
        self.view_name = view_name
        self.view_df = link[view_name].dataframe
        self.stack_base = link.stack[link.data_key]
        self.link_base = self.stack_base[link.filter]
        self.meta = self.stack_base.meta
        self.array_summary = self._is_array_summary()
        self.transposed_summary = self._is_transposed_summary()
        self.x_rules = self._set_rules_params(axes, 'x', RULES_WEIGHT)
        self.y_rules = self._set_rules_params(axes, 'y', RULES_WEIGHT)
        self.x_slicer = None
        self.y_slicer = None
        self.rules_weight = RULES_WEIGHT
        self.rules_view_df = None

    def rules_df(self):
        return self.rules_view_df

    def show_rules(self, axis=None):
        """
        """
        if not axis:
            return {'X': self.x_rules}, {'y': self.y_rules}
        elif axis == 'x':
            return {'x': self.x_rules}
        elif axis == 'y':
            return {'y': self.y_rules}
        else:
            err = "If provided, 'axis' must be one of {'x', 'y'}"
            raise valueError(err)

    def show_slicers(self, axis=None):
        if not axis:
            return {'x': self.x_slicer}, {'y': self.y_slicer}
        elif axis == 'x':
            return {'x': self.x_slicer}
        elif axis == 'y':
            return {'y': self.y_slicer}
        else:
            err = "If provided, 'axis' must be one of {'x', 'y'}"
            raise valueError(err)

    def apply(self):
        self.get_slicer()
        viable_axes = self.rule_viable_axes()
        if not viable_axes:
            df = self.view_df
        else:
            df = self.view_df.copy()
        if 'x' in viable_axes and not self.x_slicer is None:
            rule_codes = set(self.x_slicer)
            view_codes = set(df.index.tolist())
            if not rule_codes - view_codes:
                df = df.loc[self.x_slicer]

        if 'x' in viable_axes and self.transposed_summary and self.y_slicer:
            df = df.loc[self.rules_y_slicer]

        if 'y' in viable_axes and not self.y_slicer is None:
            df = df[self.y_slicer]
            if self.view_name.split('|')[1].startswith('t.'):
                df = self.verify_test_results(df)
        self.rules_view_df = df

    def _find_expanded_net_groups(self, exp_net_view):
        groups = OrderedDict()
        view = exp_net_view
        logic = view._kwargs.get('logic')
        description = view.describe_block()
        groups['codes'] = [c for c, d in description.items() if d == 'normal']
        net_names = [v for v, d in description.items() if d == 'net']
        for l in logic:
            new_l = copy.deepcopy(l)
            for k in l:
                if k not in net_names:
                    del new_l[k]
            groups[new_l.keys()[0]] = new_l.values()[0]
        groups['codes'] = [c for c, d in description.items() if d == 'normal']
        return groups

    def _find_expanded_nets(self, all_views, rule_axis):
        w = '' if not self.rules_weight else self.rules_weight
        expanded_net = [v for v in all_views if '}+]' in v
                        and v.split('|')[-2] == w
                        and v.split('|')[1] == 'f' and
                        not v.split('|')[3] == 'x']
        if expanded_net:
            if len(expanded_net) > 1:
                if len(expanded_net) == 2:
                    if expanded_net[0].split('|')[2] == expanded_net[1].split('|')[2]:
                        expanded_net = expanded_net[0]
                else:
                    msg = ("Multiple 'expand' using views found for '{}'. "
                           "Unable to sort!")
                    raise RuntimeError(msg.format(col_key))
            else:
                expanded_net = expanded_net[0]

            cond_expand = expanded_net.split('|')[2]
            cond_view = self.view_name.split('|')[2]
            if not cond_expand == cond_view or rule_axis == self.y_rules:
                expanded_net = []

            return expanded_net

    def get_slicer(self):
        """
        """
        for rule_axis in [self.x_rules, self.y_rules]:
            if not rule_axis: continue
            if rule_axis == self.x_rules:
                col_key = self._xrule_col
            else:
                col_key = self._yrule_col
            rules_slicer = None
            views = self.link_base[col_key]['@'].keys()

            w = '' if self.rules_weight is None else self.rules_weight
            weight = self.rules_weight

            expanded_net = self._find_expanded_nets(views, rule_axis)

            if 'sortx' in rule_axis:
                on_mean = self.x_rules['sortx'].get('sort_on', '@') == 'mean'
            else:
                on_mean = False
            if 'sortx' in rule_axis and on_mean:
                f = self._get_descriptive_via_stack(col_key)

            elif 'sortx' in rule_axis and expanded_net:
                within = rule_axis['sortx'].get('within', False)
                between = rule_axis['sortx'].get('between', False)
                fix = rule_axis['sortx'].get('fixed', False)
                ascending = rule_axis['sortx'].get('ascending', False)
                view = self.link_base[col_key]['@'][expanded_net]
                f = self.sort_expanded_nets(view, between=between, within=within,
                                            ascending=ascending, fix=fix)
            else:
                f = self._get_frequency_via_stack(col_key)

            if rule_axis == self.x_rules and self.array_summary:
                slice_array_items = True
            else:
                slice_array_items = False

            if self.transposed_summary or (not slice_array_items and self.array_summary):
                rules_slicer = self._get_rules_slicer(f.T, rule_axis)
            else:
                if not expanded_net or ('sortx' in rule_axis and on_mean):
                    rules_slicer = self._get_rules_slicer(f, rule_axis)
                else:
                    rules_slicer = f.index.values.tolist()
            try:
                rules_slicer.remove((col_key, 'All'))
            except:
                pass

            if rule_axis == self.x_rules:
                self.x_slicer = rules_slicer
            else:
                self.y_slicer = rules_slicer
        return None


    def _set_rules_params(self, all_rules_axes, rules_axis, rules_weight):
        if rules_axis == 'x' and 'x' not in all_rules_axes:
            return None
        elif rules_axis == 'y' and 'y' not in all_rules_axes:
            return None
        k, f, x, y = self.link.data_key, self.link.filter, self.link.x, self.link.y

        rules = None
        if rules_axis == 'x':
            if not self.array_summary and not self.transposed_summary:
                xcol = x
                ycol = None
                try:
                    rules = self.meta['columns'][x]['rules']['x']
                    self._xrule_col = x
                except:
                    pass
            elif self.array_summary:
                xcol = x
                ycol = '@'
                try:
                    rules = self.meta['masks'][x]['rules']['x']
                    self._xrule_col = x
                except:
                    pass
            elif self.transposed_summary:
                xcol = '@'
                ycol = y
                try:
                    rules = self.meta['masks'][y]['rules']['x']
                    self._xrule_col = y
                except:
                    pass
        elif rules_axis == 'y':
            if not self.array_summary and not self.transposed_summary:
                xcol = None
                ycol = x
                try:
                    rules = self.meta['columns'][y]['rules']['y']
                    self._yrule_col = y
                except:
                    pass
            elif self.array_summary:
                xcol = x
                ycol = '@'
                try:
                    rules = self.meta['masks'][x]['rules']['y']
                    self._yrule_col = x
                except:
                    pass
            elif self.transposed_summary:
                xcol = '@'
                ycol = y
                try:
                    rules = self.meta['masks'][y]['rules']['x']
                    self._yrule_col = y
                except:
                    pass
        return rules

    def _is_array_summary(self):
        return self.link.x in self.meta['masks']

    def _is_transposed_summary(self):
        return self.link.x == '@' and self.link.y in self.meta['masks']

    def _get_frequency_via_stack(self, col):
        weight_notation = '' if self.rules_weight is None else self.rules_weight
        vk = 'x|f|:||{}|counts'.format(weight_notation)
        try:
            f = self.link_base[col]['@'][vk].dataframe
        except (KeyError, AttributeError) as e:
            try:
                f = self.link_base['@'][col][vk].dataframe.T
            except (KeyError, AttributeError) as e:
                freq = qp.core.tools.dp.prep.frequency
                f = freq(self.stack_base.meta, self.stack_base.data,
                         x=col, weight=self.rules_weight)
        return f

    def _get_descriptive_via_stack(self, col):
        l = self.link_base[col]['@']
        w = '' if self.rules_weight is None else self.rules_weight
        mean_key = [k for k in l.keys() if 'd.mean' in k.split('|')[1] and
                    k.split('|')[-2] == w]
        if not mean_key:
            msg = "No mean view to sort '{}' on found!"
            raise RuntimeError(msg.format(col))
        elif len(mean_key) > 1:
            msg = "Multiple mean views found for '{}'. Unable to sort!"
            raise RuntimeError(msg.format(col))
        else:
            mean_key = mean_key[0]
        vk = mean_key
        d = l[mean_key].dataframe
        return d

    def _get_rules_slicer(self, f, rules, copy=True):

        if copy:
            f = f.copy()

        if 'slicex' in rules:
            kwargs = rules['slicex']
            values = kwargs.get('values', None)
    #         if not values is None:
    #             kwargs['values'] = [val for val in values]
            f = self.slicex(f, **kwargs)

        if 'sortx' in rules:
            kwargs = rules['sortx']
            fixed = kwargs.get('fixed', None)
            sort_on = kwargs.get('sort_on', '@')
    #         if not fixed is None:
    #             kwargs['fixed'] = [fix for fix in fixed]
            f = self.sortx(f, **kwargs)

        if 'dropx' in rules:
            kwargs = rules['dropx']
            values = kwargs.get('values', None)
    #         if not values is None:
    #             kwargs['values'] = [v for v in values]
            f = self.dropx(f, **kwargs)

        return f.index.values.tolist()

    def sort_expanded_nets(self, view, within=True, between=True, ascending=False,
                           fix=None):
        if not within and not between:
            return view.dataframe
        df = view.dataframe

        name = df.index.levels[0][0]
        sort_col = (df.columns.levels[0][0], '@')
        # get valid fixed codes
        if not fix:
            fix_codes = []
        else:
            if not isinstance(fix, list):
                fix_codes = [fix]
            else:
                fix_codes = fix
            fix_codes = [c for c in fix_codes if c in
                         df.index.get_level_values(1).tolist()]
        # determine net groups + expanded codes vs. regular codes
        net_groups = self._find_expanded_net_groups(view)
        sort = [(name, v) for v in df.index.get_level_values(1)
                if (v in net_groups['codes'] or
                v in net_groups.keys()) and not v in fix_codes]
        # sort between groups
        if between:
            if pd.__version__ == '0.19.2':
                temp_df = df.loc[sort].sort_values(sort_col, 0, ascending=ascending)
            else:
                temp_df = df.loc[sort].sort_index(0, sort_col, ascending=ascending)
        else:
            temp_df = df.loc[sort]
        between_order = temp_df.index.get_level_values(1).tolist()
        code_group_list = []
        for g in between_order:
            if g in net_groups:
                code_group_list.append([g] + net_groups[g])
            elif g in net_groups['codes']:
                code_group_list.append([g])
        final_index = []
        # sort within the net groups
        for g in code_group_list:
            is_code = len(g) == 1
            if not is_code:
                fixed_net_name = g[0]
                sort = [(name, v) for v in g[1:]]
                if within:
                    if pd.__version__ == '0.19.2':
                        temp_df = df.loc[sort].sort_values(sort_col, 0, ascending=ascending)
                    else:
                        temp_df = df.loc[sort].sort_index(0, sort_col, ascending=ascending)
                else:
                    temp_df = df.loc[sort]
                new_idx = [fixed_net_name] + temp_df.index.get_level_values(1).tolist()
                final_index.extend(new_idx)
            else:
                final_index.extend(g)
        # build final index including any fixed codes
        final_index = [(name, i) for i in final_index]
        if fix_codes:
            fix_codes = [(name, f) for f in fix_codes]
            final_index.extend(fix_codes)
        df = df.reindex(final_index)
        return df

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
                s_fixed = [(name_x, value) for value in fixed]
                # Drop fixed tuples from the sort slicer
                s_sort = [t for t in s_sort if not t in s_fixed]

            # Get sorted slicer
            if (name_y, sort_on) in df.columns:
                sort_col = (name_y, sort_on)
            elif (name_y, str(sort_on)) in df.columns:
                sort_col = (name_y, str(sort_on))
            if pd.__version__ == '0.19.2':
                df_sorted = df.loc[s_sort].sort_values(sort_col, 0, ascending)
            else:
                df_sorted = df.loc[s_sort].sort_index(0, sort_col, ascending)
            s_sort = df_sorted.index.tolist()
            df = df.loc[s_all+s_sort+s_fixed]
            return df
        except UnboundLocalError:
            print 'Could not sort on {}'.format(sort_on)
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
        slicer = [(name_x, value) for value in values]

        if not all([s in df.index for s in slicer]):
            raise KeyError (
                "Some of of the values from the list %s cannot be dropped"
                " from the dataframe because they were not found in %s."
                " Be careful that you are not both slicing and/or sorting"
                " any values that you are also trying to drop." % (
                    values,
                    df.index.tolist()
                )
            )
        df = df.drop(slicer)
        return df


    def rule_viable_axes(self):
        viable_axes = ['x', 'y']
        condensed_x = False
        condensed_y = False

        meta = self.meta
        x, y = self.link.x, self.link.y
        vk = self.view_name
        array_summary = (x in meta['masks'] and y == '@')
        transposed_summary = (y in meta['masks'] and x == '@')
        v_method = vk.split('|')[1]
        relation = vk.split('|')[2]
        s_name = vk.split('|')[-1]
        descriptive = v_method.startswith('.d')
        exp_net = '}+]' in relation
        array_sum_freqs = array_summary and s_name in ['counts', 'c%', 'r%']


        if transposed_summary:
            x, y = y, x

        if (relation.split(":")[0].startswith('x') and not exp_net) or descriptive:
            if not array_summary:
                condensed_x = True
        elif relation.split(":")[1].startswith('y'):
            condensed_y = True
        else:
            if re.search('x\[.+:y$', relation) != None:
                condensed_x = True
            elif re.search('x:y\[.+', relation) != None:
                condensed_y = True
            if re.search('y\[.+:x$', relation) != None:
                condensed_y = True
            elif re.search('y:x\[.+', relation) != None:
                condensed_x = True

        if condensed_x or x=='@': viable_axes.remove('x')
        if condensed_y or (y=='@' and not array_sum_freqs): viable_axes.remove('y')

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
                        value = set([int(i) for i in list(value[1:-1].split(','))])
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
                elif len(value)==0:
                    value = np.NaN

                return value
            else:
                return value


        cols = set([int(v) for v in zip(*[c for c in df.columns])[1]])
        df = df.applymap(verify_test_value)

        return df