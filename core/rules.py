import re


class Rules(object):
    def __init__(self, link, view_name):

        ALL_RULES_AXES = ['x', 'y']
        RULES_WEIGHT = None

        self.link = link
        self.view_name = view_name
        self.view_df = link[view_name]
        self.stack_base = link.stack[link.data_key]
        self.link_base = self.stack_base[link.filter]
        self.meta = self.stack_base.meta
        self.array_summary = self._is_array_summary()
        self.transposed_summary = self._is_transposed_summary()

        self.x_rules = self._set_rules_params(ALL_RULES_AXES, 'x', RULES_WEIGHT)
        self.y_rules = self._set_rules_params(ALL_RULES_AXES, 'y', RULES_WEIGHT)

        self.x_slicer = None
        self.y_slicer = None

        self.rules_weight = RULES_WEIGHT

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


    def get_slicer(self):
        """
        """
        for rule_axis in [self.x_rules, self.y_rules]:
            if rule_axis == self.x_rules:
                col_key = self._xrule_col
            else:
                col_key = self._yrule_col
            rules_slicer = None
            views = self.link_base[col_key]['@'].keys()

            w = '' if self.rules_weight is None else self.rules_weight
            weight = self.rules_weight
            expanded_net = [v for v in views if '}+]' in v
                            and v.split('|')[-2] == w
                            and v.split('|')[1] == 'f' and
                            not v.split('|')[3] == 'x']

            if expanded_net:
                if len(expanded_net) > 1:
                    if len(expanded_net) == 2:
                        if expanded_net[0].split('|')[2] == expanded_net[1].split('|')[2]:
                            expanded_net = expanded_net[0]
                    else:
                        msg = "Multiple 'expand' using views found for '{}'. Unable to sort!"
                        raise RuntimeError(msg.format(col_key))
                else:
                    expanded_net = expanded_net[0]
            if 'sortx' in rule_axis:
                on_mean = rules['sortx'].get('sort_on', '@') == 'mean'
            else:
                on_mean = False
            if 'sortx' in rule_axis and on_mean:
                f = self.get_descriptive_via_stack(
                    data_key, the_filter, col_key, weight=weight)
            elif 'sortx' in rule_axis and expanded_net:
                within = rule_axis['sortx'].get('within', False)
                between = rule_axis['sortx'].get('between', False)
                fix = rule_axis['sortx'].get('fixed', False)
                ascending = rule_axis['sortx'].get('ascending', False)
                view = self[data_key][the_filter][col_key]['@'][expanded_net]
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
                ycol = y
                try:
                    rules = self.meta['columns'][x]['rules']['x']
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



    # def _get_axis_rules(self, all_rules_axes, rules_axis, rules_weight):
    #         if rules_axis == 'x' and 'x' not in all_rules_axes:
    #             return None
    #         elif rules_axis == 'y' and 'y' not in all_rules_axes:
    #             return None
    #         k, f, x, y = self.link.data_key, self.link.filter, self.link.x, self.link.y
    #         array_summary = self._is_array_summary()
    #         transposed_summary = self._is_transposed_summary()
    #         axis_slicer = None
    #         if rules_axis == 'x':
    #             if not array_summary and not transposed_summary:
    #                 axis_slicer = self._compute_slicer(
    #                     x=x, weight=rules_weight)
    #             elif array_summary:
    #                 axis_slicer = self._compute_slicer(
    #                     x=x, y='@', weight=rules_weight, slice_array_items=True)
    #             elif transposed_summary:
    #                 axis_slicer = self._compute_slicer(
    #                     dk, the_filter, x='@', y=y, weight=rules_weight)
    #         elif rules_axis == 'y':
    #             if not array_summary and not transposed_summary:
    #                 axis_slicer = self._compute_slicer(
    #                     dk, the_filter, y=y, weight=rules_weight)
    #             elif array_summary:
    #                 axis_slicer = self._compute_slicer(
    #                  x=x, y='@', weight=rules_weight, slice_array_items=False)
    #             elif transposed_summary:
    #                 axis_slicer = self._compute_slicer(
    #                     dk, the_filter, x='@', y=y, weight=rules_weight)
    #         return axis_slicer

    # def _compute_slicer(self, x=None, y=None, weight=None, slice_array_items=False):
    #     k, f, x, y = self.link.data_key, self.link.filter, self.link.x, self.link.y
    #     array_summary = self._is_array_summary()
    #     transposed_summary = self._is_transposed_summary()
    #     rules = None
    #     if not array_summary and not transposed_summary:
    #         if not x is None:
    #             try:
    #                 rules = self.meta['columns'][x]['rules']['x']
    #                 self._rule_col_key = x
    #             except:
    #                 pass
    #         elif not y is None:
    #             try:
    #                 rules = self.meta['columns'][y]['rules']['y']
    #                 self._rule_col_key = y
    #             except:
    #                 pass

    #     elif array_summary:
    #         if slice_array_items:
    #             try:
    #                 rules = self.meta['masks'][x]['rules']['x']
    #                 self._rule_col_key = x
    #             except:
    #                 pass
    #         else:
    #             try:
    #                 rules = self.meta['masks'][x]['rules']['y']
    #                 self._rule_col_key = x
    #             except:
    #                 pass

    #     elif transposed_summary:
    #             try:
    #                 rules = self.meta['masks'][y]['rules']['x']
    #                 self._rule_col_key = y
    #             except:
    #                 pass

    #     return rules

    # def _get_axis_rules(self, all_rules_axes, rules_axis, rules_weight):
    #         if rules_axis == 'x' and 'x' not in all_rules_axes:
    #             return None
    #         elif rules_axis == 'y' and 'y' not in all_rules_axes:
    #             return None

    #         k, f, x, y = self.link.data_key, self.link.filter, self.link.x, self.link.y
    #         array_summary = self._is_array_summary()
    #         transposed_summary = self._is_transposed_summary()

    #         axis_slicer = None

    #         if rules_axis == 'x':
    #             if not array_summary and not transposed_summary:
    #                 axis_slicer = self._compute_slicer(
    #                     x=x, weight=rules_weight)
    #             elif array_summary:
    #                 axis_slicer = self._compute_slicer(
    #                     x=x, y='@', weight=rules_weight, slice_array_items=True)
    #             elif transposed_summary:
    #                 axis_slicer = self._compute_slicer(
    #                     dk, the_filter, x='@', y=y, weight=rules_weight)

    #         elif rules_axis == 'y':
    #             if not array_summary and not transposed_summary:
    #                 axis_slicer = self._compute_slicer(
    #                     dk, the_filter, y=y, weight=rules_weight)
    #             elif array_summary:
    #                 axis_slicer = self._compute_slicer(
    #                  x=x, y='@', weight=rules_weight, slice_array_items=False)
    #             elif transposed_summary:
    #                 axis_slicer = self._compute_slicer(
    #                     dk, the_filter, x='@', y=y, weight=rules_weight)

    #         return axis_slicer


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
                print 'THIS IS UNSUPPORTED RIGHT NOW!'
                print 'FREQ/CROSSBREAK FUNCTION MUST WORK HERE!!!!!!!!!!!!'
                f = frequency(self[data_key].meta, self[data_key].data, x=col, weight=self.rules_weight)
        return f







    def _get_rules_slicer(self, f, rules, copy=True):

        if copy:
            f = f.copy()

        if 'slicex' in rules:
            kwargs = rules['slicex']
            values = kwargs.get('values', None)
    #         if not values is None:
    #             kwargs['values'] = [val for val in values]
            f = qp.core.tools.view.query.slicex(f, **kwargs)

        if 'sortx' in rules:
            kwargs = rules['sortx']
            fixed = kwargs.get('fixed', None)
            sort_on = kwargs.get('sort_on', '@')
    #         if not fixed is None:
    #             kwargs['fixed'] = [fix for fix in fixed]
            f = qp.core.tools.view.query.sortx(f, **kwargs)

        if 'dropx' in rules:
            kwargs = rules['dropx']
            values = kwargs.get('values', None)
    #         if not values is None:
    #             kwargs['values'] = [v for v in values]
            f = self.dropx(f, **kwargs)
        return f.index.values.tolist()

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