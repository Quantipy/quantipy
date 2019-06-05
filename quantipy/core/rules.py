import re
import pandas as pd
from collections import OrderedDict
import copy
import quantipy as qp
import numpy as np
import warnings

class Rules(object):

    # ------------------------------------------------------------------------
    # init
    # ------------------------------------------------------------------------
    def __init__(self, link, view_name, axes=['x', 'y'], rweight=None):
        self.link = link
        self.view_name = view_name
        self.stack_base = link.stack[link.data_key]
        self.link_base = self.stack_base[link.filter]
        self.meta = self.stack_base.meta
        self.array_summary = self._is_array_summary()
        self.transposed_summary = self._is_transposed_summary()
        if self.transposed_summary:
            self.view_df = link[view_name].dataframe.T
            self.array_summary = True
        else:
            self.view_df = link[view_name].dataframe
        self._xrule_col = None
        self._yrule_col = None
        self._sort_weight = self._get_sort_weight(rweight)
        self.x_rules = self._set_rules_params(axes, 'x')
        self.y_rules = self._set_rules_params(axes, 'y')
        self.x_slicer = None
        self.y_slicer = None
        self.rules_view_df = None

    def _is_array_summary(self):
        return self.link.x in self.meta['masks']

    def _is_transposed_summary(self):
        return self.link.x == '@' and self.link.y in self.meta['masks']

    def _set_rules_params(self, all_rules_axes, rules_axis):
        if rules_axis == 'x' and 'x' not in all_rules_axes:
            return None
        elif rules_axis == 'y' and 'y' not in all_rules_axes:
            return None
        k, f, x, y = self.link.data_key, self.link.filter, self.link.x, self.link.y
        if self.transposed_summary:
            x, y = y, x
        rules = None
        if rules_axis == 'x':
            if not self.array_summary:
                try:
                    rules = self.meta['columns'][x]['rules']['x']
                    self._xrule_col = x
                except:
                    pass
            else:
                try:
                    rules = self.meta['masks'][x]['rules']['x']
                    self._xrule_col = x
                except:
                    pass
        elif rules_axis == 'y':
            if not self.array_summary:
                try:
                    rules = self.meta['columns'][y]['rules']['y']
                    self._yrule_col = y
                except:
                    pass
            else:
                try:
                    rules = self.meta['masks'][x]['rules']['y']
                    self._yrule_col = x
                except:
                    pass
        return rules

    def _get_sort_weight(self, use_weight):
        var = self.link.y if self.link.x=='@' else self.link.x
        try:
            collection = self.meta['columns'][var]
        except:
            collection = self.meta['masks'][var]
        rules = collection.get('rules', {}).get('x', {})
        if 'sortx' in rules:
            sort_on = rules['sortx'].get('sort_on', '@')
            sort_weight = rules['sortx']['with_weight'] or ''
            if sort_weight == 'auto':
                if use_weight is None:
                    sort_weight = self.view_name.split('|')[-2]
                else:
                    sort_weight = use_weight
            return sort_weight
        else:
            return None

    # ------------------------------------------------------------------------
    # display
    # ------------------------------------------------------------------------

    def rules_df(self):
        if self.transposed_summary:
            return self.rules_view_df.T
        else:
            return self.rules_view_df

    def show_rules(self, axis=None):
        """
        """
        if not axis:
            return {'x': self.x_rules}, {'y': self.y_rules}
        elif axis == 'x':
            return {'x': self.x_rules}
        elif axis == 'y':
            return {'y': self.y_rules}
        else:
            err = "If provided, 'axis' must be one of {'x', 'y'}"
            raise ValueError(err)

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

    # ------------------------------------------------------------------------
    # apply rules
    # ------------------------------------------------------------------------

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

        if 'y' in viable_axes and not self.y_slicer is None:
            df = df[self.y_slicer]
            if self.view_name.split('|')[1].startswith('t.'):
                df = self.verify_test_results(df)
        self.rules_view_df = df
        return None

    def get_slicer(self):
        """
        """
        for axis, rule_axis in enumerate([self.x_rules, self.y_rules]):
            if not rule_axis: continue

            # get all views of the link, depending on axis
            col_key = self._xrule_col if axis == 0 else self._yrule_col
            views = list(self.link_base[col_key]['@'].keys())

            # get df (-slice) to apply rule on
            if 'sortx' in rule_axis:
                sort_on = rule_axis['sortx'].get('sort_on', '@')

                sort_on_stat = False
                sort_on_net = False

                if isinstance(sort_on, str):
                    sort_on_stat = sort_on in [
                        'median', 'stddev', 'sem', 'max', 'min', 'mean',
                        'upper_q', 'lower_q']
                    sort_on_net = sort_on.startswith('net')

                expanded_net = self._find_expanded_nets(views, rule_axis)

                # sort expanded nets
                if expanded_net and not self.array_summary:
                    if not sort_on == '@':
                        msg = 'Cannot sort expanded nets on {}.'
                        raise AttributeError(msg .format(sort_on))
                    view = self.link_base[col_key]['@'][expanded_net]
                    f = self.sort_expanded_nets(view, rule_axis['sortx'])
                    r_slicer = self._get_rules_slicer(f, rule_axis,
                                                      ['slicex', 'dropx'])
                    if axis == 0:
                        self.x_slicer = r_slicer
                    else:
                        self.y_slicer = r_slicer
                    return None
                # get df-desc-slice to sort on
                elif sort_on_stat:
                    f = self._get_descriptive_via_stack(col_key, sort_on)
                # get df-net-slice to sort on
                elif sort_on_net:
                    f = self._get_net_via_stack(col_key, sort_on)
                # get df-freq-slice to sort on
                else:
                    f = self._get_frequency_via_stack(col_key, axis, self._sort_weight)
            # get df for hiding + slicing
            else:
                f = self._get_frequency_via_stack(col_key, axis, None)

            # get rules slicer
            f = f.T if self.array_summary and axis == 1 else f
            r_slicer = self._get_rules_slicer(f, rule_axis)
            if axis == 0:
                self.x_slicer = r_slicer
            else:
                self.y_slicer = r_slicer
        return None

    def _get_frequency_via_stack(self, col, axis, weight):
        if weight is None:
            vk = 'x|f|:|||counts'
        else:
            vk = 'x|f|:||{}|counts'.format(weight)
            view_weight = self.view_name.split('|')[-2]
            link_weights = [k.split('|')[-2] for k in list(self.link.keys())
                            if not 'base' in k.split('|')[-1]]
            if not (weight == view_weight or weight in link_weights):
                msg = "\n{}: view-weight and weight to sort on differ ('{}' vs '{}')\n"
                warnings.warn(msg.format(col, view_weight, weight or None))
        try:
            if self.transposed_summary:
                f = self.link_base['@'][col][vk].dataframe.T
            else:
                f = self.link_base[col]['@'][vk].dataframe
        except (KeyError, AttributeError) as e:
            freq = qp.core.tools.dp.prep.frequency
            f = freq(self.stack_base.meta, self.stack_base.data, x=col,
                     weight=weight or None)
        return f

    def _get_descriptive_via_stack(self, col, desc='mean'):
        l = self.link_base[col]['@']
        w = self._sort_weight
        desc_key = [k for k in list(l.keys()) if 'd.{}'.format(desc) in k.split('|')[1]
                    and k.split('|')[-2] == w]
        if not desc_key:
            msg = "No {} view to sort '{}' on found!"
            raise RuntimeError(msg.format(desc, col))
        elif len(desc_key) > 1:
            msg = "Multiple {} views found for '{}'. Unable to sort!"
            raise RuntimeError(msg.format(desc, col))
        else:
            desc_key = desc_key[0]
        d = l[desc_key].dataframe
        return d

    def _get_net_via_stack(self, col, net='net_1'):
        l = self.link_base[col]['@']
        w = self._sort_weight
        net_no = int(net.split('_')[-1])
        net_key = [k for k in list(l.keys()) if k.split('|')[-1] == 'net'
                    and len(k.split('|')[2].split(',x')) >= net_no
                    and k.split('|')[-2] == w]
        if not net_key:
            msg = "No net view to sort '{}' on found!"
            raise RuntimeError(msg.format(col))
        else:
            net_key = net_key[0]
        d = l[net_key].dataframe
        return d

    def _get_rules_slicer(self, f, rules, apply_rules=None):
        f = f.copy()
        rulesx = OrderedDict([
            ('slicex', self.slicex),
            ('sortx', self.sortx),
            ('dropx', self.dropx)])

        if not apply_rules: apply_rules = list(rulesx.keys())
        for r, method in list(rulesx.items()):
            if apply_rules and r in apply_rules:
                if r in rules:
                    f = method(f, **rules[r])
        rules_slicer = f.index.values.tolist()
        col_key = f.index.levels[0].tolist()[0]
        if (col_key, 'All') in rules_slicer:
            rules_slicer.remove((col_key, 'All'))

        return rules_slicer

    def _find_expanded_nets(self, all_views, rule_axis):
        expanded_net = [v for v in all_views if '}+]' in v
                        and v.split('|')[-2] == self._sort_weight
                        and v.split('|')[1] == 'f' and
                        not v.split('|')[3] == 'x']

        return expanded_net[0] if expanded_net else None

    def _find_expanded_net_groups(self, exp_net_view):
        groups = OrderedDict()
        view = exp_net_view
        logic = view._kwargs.get('logic')
        description = view.describe_block()
        groups['codes'] = [c for c, d in list(description.items()) if d == 'normal']
        net_names = [v for v, d in list(description.items()) if d == 'net']
        for l in logic:
            new_l = copy.deepcopy(l)
            for k in l:
                if k not in net_names:
                    del new_l[k]
            groups[list(new_l.keys())[0]] = list(new_l.values())[0]
        groups['codes'] = [c for c, d in list(description.items()) if d == 'normal']
        return groups

    def sort_expanded_nets(self, view, sortx):
        within = sortx.get('within', True)
        between = sortx.get('between', True)
        ascending = sortx.get('ascending', False)
        fix = sortx.get('fixed', None)
        if not within and not between: return view.dataframe
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
                v in list(net_groups.keys())) and not v in fix_codes]
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
                fixed_in_g = [v for v in g[1:] if v in fix_codes]
                sort = [(name, v) for v in g[1:] if not v in fixed_in_g]
                if within:
                    if pd.__version__ == '0.19.2':
                        temp_df = df.loc[sort].sort_values(sort_col, 0, ascending=ascending)
                    else:
                        temp_df = df.loc[sort].sort_index(0, sort_col, ascending=ascending)
                else:
                    temp_df = df.loc[sort]
                new_idx = [fixed_net_name] + temp_df.index.get_level_values(1).tolist() + fixed_in_g
                final_index.extend(new_idx)
                fix_codes = [c for c in fix_codes if not c in fixed_in_g]
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
            if pd.__version__ == '0.19.2':
                df_sorted = df.loc[s_sort].sort_values(sort_col, 0, ascending)
            else:
                df_sorted = df.loc[s_sort].sort_index(0, sort_col, ascending)
            s_sort = df_sorted.index.tolist()
            df = df.loc[s_all+s_sort+s_fixed]
            return df
        except UnboundLocalError:
            print(('Could not sort on {}'.format(sort_on)))
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

    # ------------------------------------------------------------------------

    def rule_viable_axes(self):
        viable_axes = ['x', 'y']
        condensed_x = False
        condensed_y = False

        meta = self.meta
        if self.transposed_summary:
            y, x = self.link.x, self.link.y
        else:
            x, y = self.link.x, self.link.y
        vk = self.view_name
        array_summary = (x in meta['masks'] and y == '@')
        v_method = vk.split('|')[1]
        relation = vk.split('|')[2]
        s_name = vk.split('|')[-1]
        descriptive = v_method.startswith('.d')
        exp_net = '}+]' in relation
        array_sum_freqs = array_summary and s_name in ['counts', 'c%', 'r%']

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
                        value = set([int(i) if i.isdigit() else i
                                     for i in list(value[1:-1].split(','))])
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