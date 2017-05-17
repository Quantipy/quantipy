def axis_slicer_from_vartype(link, all_rules_axes, rules_axis, rules_weight):
        if rules_axis == 'x' and 'x' not in all_rules_axes:
            return None
        elif rules_axis == 'y' and 'y' not in all_rules_axes:
            return None

        meta = link.stack[link.data_key].meta
        f, x, y = link.filter, link.x, link.y


        quit()

        array_summary = self._is_array_summary(meta, x, y)
        transposed_summary = self._is_transposed_summary(meta, x, y)

        axis_slicer = None

        if rules_axis == 'x':
            if not array_summary and not transposed_summary:
                axis_slicer = self.get_rules_slicer_via_stack(
                    dk, the_filter, x=x, weight=rules_weight)
            elif array_summary:
                axis_slicer = self.get_rules_slicer_via_stack(
                    dk, the_filter, x=x, y='@', weight=rules_weight,
                    slice_array_items=True)
            elif transposed_summary:
                axis_slicer = self.get_rules_slicer_via_stack(
                    dk, the_filter, x='@', y=y, weight=rules_weight)
        elif rules_axis == 'y':
            if not array_summary and not transposed_summary:
                axis_slicer = self.get_rules_slicer_via_stack(
                    dk, the_filter, y=y, weight=rules_weight)
            elif array_summary:
                axis_slicer = self.get_rules_slicer_via_stack(
                    dk, the_filter, x=x, y='@', weight=rules_weight,
                    slice_array_items=False)
            elif transposed_summary:
                axis_slicer = self.get_rules_slicer_via_stack(
                    dk, the_filter, x='@', y=y, weight=rules_weight)

        return axis_slicer