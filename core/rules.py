def get_axis_slicer(link, all_rules_axes, rules_axis, rules_weight):
        if rules_axis == 'x' and 'x' not in all_rules_axes:
            return None
        elif rules_axis == 'y' and 'y' not in all_rules_axes:
            return None

        k, f, x, y = link.data_key, link.filter, link.x, link.y
        meta = link.stack[k].meta

        array_summary = _is_array_summary(meta, x, y)
        transposed_summary = _is_transposed_summary(meta, x, y)

        axis_slicer = None

        if rules_axis == 'x':
            if not array_summary and not transposed_summary:
                axis_slicer = _compute_slicer(link, x=x, weight=rules_weight)
            elif array_summary:
                axis_slicer = _compute_slicer(
                    dk, the_filter, x=x, y='@', weight=rules_weight,
                    slice_array_items=True)
            elif transposed_summary:
                axis_slicer = _compute_slicer(
                    dk, the_filter, x='@', y=y, weight=rules_weight)
            print axis_slicer

            quit()

        elif rules_axis == 'y':
            if not array_summary and not transposed_summary:
                axis_slicer = _compute_slicer(
                    dk, the_filter, y=y, weight=rules_weight)
            elif array_summary:
                axis_slicer = _compute_slicer(
                    dk, the_filter, x=x, y='@', weight=rules_weight,
                    slice_array_items=False)
            elif transposed_summary:
                axis_slicer = _compute_slicer(
                    dk, the_filter, x='@', y=y, weight=rules_weight)

        return axis_slicer


def _is_array_summary(meta, x, y):
    return x in meta['masks']

def _is_transposed_summary( meta, x, y):
    return x == '@' and y in meta['masks']

def _compute_slicer(link, x=None, y=None, weight=None, slice_array_items=False):

        k, f, x, y = link.data_key, link.filter, link.x, link.y
        meta = link.stack[k].meta
        array_summary = _is_array_summary(meta, x, y)
        transposed_summary = _is_transposed_summary(meta, x, y)

        rules = None

        if not array_summary and not transposed_summary:
            if not x is None:
                try:
                    rules = meta['columns'][x]['rules']['x']
                    col = x
                except:
                    pass
            elif not y is None:
                try:
                    rules = meta['columns'][y]['rules']['y']
                    col = y
                except:
                    pass

        elif array_summary:
            if slice_array_items:
                try:
                    rules = meta['masks'][x]['rules']['x']
                    col = x
                except:
                    pass
            else:
                try:
                    rules = meta['masks'][x]['rules']['y']
                    col = x
                except:
                    pass

        elif transposed_summary:
                try:
                    rules = meta['masks'][y]['rules']['x']
                    col = y
                except:
                    pass

        if not rules: return None

        # views = self[data_key][the_filter][col]['@'].keys()
        views = link.stack[k][f][col]['@'].keys()

        quit()

        w = '' if weight is None else weight
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
                    raise RuntimeError(msg.format(col))
            else:
                expanded_net = expanded_net[0]
        if 'sortx' in rules:
            on_mean = rules['sortx'].get('sort_on', '@') == 'mean'
        else:
            on_mean = False
        if 'sortx' in rules and on_mean:
            f = self.get_descriptive_via_stack(
                data_key, the_filter, col, weight=weight)
        elif 'sortx' in rules and expanded_net:
            within = rules['sortx'].get('within', False)
            between = rules['sortx'].get('between', False)
            fix = rules['sortx'].get('fixed', False)
            ascending = rules['sortx'].get('ascending', False)
            view = self[data_key][the_filter][col]['@'][expanded_net]
            f = self.sort_expanded_nets(view, between=between, within=within,
                                        ascending=ascending, fix=fix)
        else:
            f = self.get_frequency_via_stack(
                data_key, the_filter, col, weight=weight)

        if transposed_summary or (not slice_array_items and array_summary):
            rules_slicer = functions.get_rules_slicer(f.T, rules)
        else:
            if not expanded_net or ('sortx' in rules and on_mean):
                rules_slicer = functions.get_rules_slicer(f, rules)
            else:
                rules_slicer = f.index.values.tolist()
        try:
            rules_slicer.remove((col, 'All'))
        except:
            pass
        return rules_slicer

