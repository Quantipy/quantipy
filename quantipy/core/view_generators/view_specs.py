import pandas as pd
from quantipy.core.tools.qp_decorators import modify
from collections import OrderedDict
from itertools import chain
from operator import add, sub, mul
from operator import truediv as div

import re
import warnings

class ViewManager(object):
    def __init__(self, stack):
        self.stack = stack
        self.basics = None
        self.nets = None
        self.stats = None
        self.tests = None
        self.views = None
        self.grouping = None
        self.base_spec = None
        self.weighted = None
        self.sums_pos = None
        self._base_views = None
        self._grouped_views = None
        return None


    def _base_len(self):
        """
        """
        return len(self._base_views) if self._base_views else None

    def get_views(self, data_key=None, filter_key=None, weight=None,
                  freqs=True, nets=True, stats=['mean', 'stddev'],
                  sums='bottom', tests=None, cell_items='colpct',
                  ci_order='normal', bases='auto'):
        """
        Query the ``qp.Stack`` for the desired set of ``Views``.

        Parameters
        ----------
        data_key : str, default None
            The data_key name of the ``qp.Stack`` path to be queried.
        filter_key : str, default None
            The filter_key name of the ``qp.Stack`` path to be queried.
        weight : str, default None
            The name of the weight variable to look for searching aggregations.
        freqs : bool, default True
            Determines if all regular frequency-types are being pulled
            from the ``qp.Stack``. This includes counts, percentage and
            cumulative percentages.
        nets : bool, default True
            Determines if net-types are being pulled from the ``qp.Stack``.
            This includes simple, block-like and expanded nets as well as
            attached calculations.
        stats : (list of) {'mean', 'stddev', 'min', 'max', ...},
                default ['mean', 'stddev']
            The descriptive statistics (if any) to get from the ``qp.Stack``.
        sums: str {'bottom', 'mid'} or None, default 'bottom'
            The position of the sums (if any) in the view-list.
        tests : (list of) float, default None
            Text...
        cell_items: {'counts', 'colpct', 'rowpct', 'counts_colpct',
                     'counts_rowpct', 'colpct_rowpct', 'counts_colpct_rowpct'},
                    default 'colpct'
            The kind of frequency aggregations that should be returned: raw
            counts, column or row percentages or grouped versions of the
            former, e.g. 'counts_colpct' will show both counts and column
            percentages as a set of cell items.
        ci_order: {'normal', 'switched'}
            If more than one cell_item is requested, 'normal' returns pcts in
            the first place, 'switched' returns counts in the first place.
        bases : {'auto', 'both', 'weighted', 'unweighted'}
            The base view(s) to include. 'auto' will match the base to the
            ``weights`` parameter. If ``weights`` is provided (i.e. the
            parameter is not ``None``), 'both' will try to get both the 'unweighted'
            and the 'weighted' base, 'weighted' / 'unweighted' will try to get
            the respective version of the base view. The latter three will
            automatically fall back to the 'auto' behaviour if the passed value
            would lead to a failure or inconsistencies.

        Returns
        -------
        self
        """
        self.basics = freqs
        self.nets = nets
        self.stats = stats
        self.tests = tests
        self.weighted = weight
        self.sums_pos = sums
        cimap = {'c': 'counts', 'p': 'colpct', 'cp': 'counts_colpct'}
        for old, new in list(cimap.items()):
            if cell_items == old:
                cell_items = new
                msg = "'{}' is an old cell item reference, please use '{}' instead."
                warnings.warn(msg.format(old, new))
        valid_ci = ['counts', 'colpct', 'rowpct',
                    'counts_colpct', 'counts_rowpct', 'colpct_rowpct',
                    'counts_colpct_rowpct']
        valid_bases = ['auto', 'both', 'weighted', 'unweighted']
        if bases not in valid_bases:
            err = "'bases must be one of {}, not '{}'!".format(valid_bases, bases)
            raise ValueError(err)
        self.base_spec = bases
        if cell_items not in valid_ci:
            err = "'cell_items' must be one of {}, not '{}'!"
            raise ValueError(err.format(valid_ci, cell_items))
        stack = self.stack
        if not data_key:
            if len(list(stack.keys())) > 1:
                err = ("Must provide 'data_key' if more than one datasets are "
                       "connected to the Stack!")
                raise ValueError(err)
            else:
                data_key = list(stack.keys())[0]
        if not filter_key:
            no_filter_ph = 'no_filter' in stack[data_key] and list(stack[data_key].keys())
            if len(list(stack[data_key].keys())) > 1 and not no_filter_ph:
                err = ("Must provide 'filter_key' if more than one filter is "
                       "applied to the Stack!")
                raise ValueError(err)
            else:
                filter_key = list(stack[data_key].keys())[0]

        views = self._request_views(
            data_key=data_key, filter_key=filter_key, weight=self.weighted,
            frequencies=self.basics, nets=self.nets, descriptives=self.stats,
            sums=self.sums_pos, coltests=True if self.tests else False,
            sig_levels=self.tests if self.tests else [])
        self._grouped_views = views['grouped_views'][cell_items]
        self.views = views['get_chain'][cell_items]
        # Defining the base view layout vs. all collected views...
        # See also: .set_bases()
        views = self.views[:]
        if self.weighted:
            base_views = views[:6]
            other_views = views[6:]
        else:
            base_views = views[:3]
            other_views = views[3:]
        if bases == 'auto':
            wparam = 'w' if self.weighted else 'uw'
        elif bases == 'both':
            wparam = 'both'
        elif bases == 'weighted':
            wparam = 'w'
        elif bases == 'unweighted':
            wparam = 'uw'
        self.set_bases(base=wparam)
        self.views = self._base_views + other_views
        if ci_order == 'normal':
            self.group(switch=False)
        elif ci_order == 'switched':
            self.group(switch=True)
        return self

    def set_bases(self, base='w', gross=False, effective=False,
                  order=['base', 'gross', 'effective'], uw_pos='before',
                  sticky_gross=False):
        """
        Set the base (sample size) view presentation.

        Parameters
        ----------
        base : {'w', 'uw', 'both'}, default 'w'
            Show the *weighted* or *unweighted* version of the regular base or
            *both*.
        gross : {'w', 'uw', 'both'}, default False
            Show the *weighted* or *unweighted* version of the gross base or
            *both*.
        effective : {'w', 'uw', 'both'}, default False
            Show the *weighted* or *unweighted* version of the effective base
            or *both*.
        order : list of elements 'base', 'gross', 'effective',
                default ['base', 'gross', 'effective']
            Set the order in that regular, gross and effective bases should
            appear.
        uw_pos : {'after', 'before'}, default 'after'
            Define if unweighted bases appear before or after their weighted
            versions.
        sticky_gross : bool, default False
            If there are weighted and unweighted versions of both regular and
            gross bases, this option will alternate between them, taking into
            account the ``order`` and ``uw_pos`` parameter values. With this
            option set to ``True``, any *effective* bases, however,  will be
            placed at the end.

        Returns
        -------
        None
        """
        if not self.views:
            err = 'Cannot set base views, please run .get_views() before!'
            raise RuntimeError(err)
        bases = []
        # test for reasonable setup
        if not self.weighted:
            if base: base = 'uw'
            if gross: gross = 'uw'
            if effective: effective = 'uw'
        valid_order_items = ['base', 'gross', 'effective']
        if not all(b in valid_order_items for b in order):
            err = "Items in 'order' must be one of: {}!".format(valid_order_items)
            raise ValueError(err)
        # view key definitions
        base_vk = 'x|f|x:||{}|cbase'
        gross_vk = 'x|f|x:||{}|cbase_gross'
        effective_vk = 'x|f|x:||{}|ebase'
        uw_base_vk = base_vk.format('')
        uw_gross_vk = gross_vk.format('')
        uw_effective_vk = effective_vk.format('')
        if self.weighted:
            w_base_vk = base_vk.format(self.weighted)
            w_gross_vk = gross_vk.format(self.weighted)
            w_effective_vk = effective_vk.format(self.weighted)
        else:
            w_base_vk = w_gross_vk = w_effective_vk = None
        base_dict = {'base': [base, uw_base_vk, w_base_vk],
                     'gross': [gross, uw_gross_vk, w_gross_vk],
                     'effective': [effective, uw_effective_vk, w_effective_vk]}
        # assembling all base types...
        for base_type in order:
            btype = base_dict[base_type][0]
            uw_vk = base_dict[base_type][1]
            w_vk = base_dict[base_type][2]
            if btype:
                if btype == 'w':
                    bases.append(w_vk)
                elif btype == 'uw':
                    bases.append(uw_vk)
                else:
                    if uw_pos == 'after':
                        bases.extend([w_vk, uw_vk])
                    else:
                        bases.extend([uw_vk, w_vk])
        # rearrange reg. and gross bases if requested (alternate between them):
        if sticky_gross and gross == 'both' and base == 'both':
            sticky_bases = []
            other_bases = []
            for base in bases:
                btype = base.split('|')[-1]
                if btype in ['cbase_gross', 'cbase']:
                    sticky_bases.append(base)
                else:
                    other_bases.append(base)
            sticky_bases = sticky_bases[::2] + sticky_bases[1::2]
            bases = sticky_bases + other_bases
        # final list of base views:
        self.views = bases + self.views[self._base_len():]
        self._base_views = bases
        return None

    def group(self, style='reduce', switch=False):
        """
        Reorder the ``.views`` list to group belonging aggregations together.

        Parameters
        ----------
        style : {'reduce', 'repeat'}, default 'reduce'
            Defines how the grouping will instruct the indexing of concatenated
            ``View`` dataframes in a ``qp.Chain`` object. ``'reduced'`` will
            show each index code a single time, while ``'repeat'`` will show
            each code as many times as there are ``View`` objects referencing
            them.

        Returns
        -------
        self
        """
        self.grouping = style
        grouped_views = self._grouped_views

        if grouped_views is None:
            msg = 'Grouped views are not defined. Run ``.get_views()`` first.'
            raise ValueError(msg)
        if len(grouped_views) == 1 and len(grouped_views[0]) == 1:
            grouped_views = []
        full_grouped_views = []
        flat_gv = list(chain.from_iterable(grouped_views))

        non_grouped = [v for v in self.views[self._base_len():] if v not in flat_gv]

        regs, nets, comps, stats, sums = self._get_view_types(grouped_views)
        if switch:
            regs = self._switch(regs)
            comps = self._switch(comps)
            sums = self._switch(sums)
            nets = self._switch(nets)
        regs2, nets2, comps2, stats2, sums2 = self._get_view_types(non_grouped)
        regs.extend(regs2)
        nets.extend(nets2)
        comps.extend(comps2)
        stats.extend(stats2)
        sums.extend(sums2)
        if self.sums_pos == 'bottom':
            view_collection = regs + comps + nets  + stats + sums
        elif self.sums_pos == 'mid':
            view_collection = regs + comps + sums + nets + stats

        view_collection = self._base_views + view_collection

        for view_sect in view_collection:
            if view_sect:
                if isinstance(view_sect, list) and style == 'reduce':
                    full_grouped_views.append(tuple(view_sect))
                else:
                    full_grouped_views.append(view_sect)

        self.views = full_grouped_views
        return self

    @staticmethod
    def _get_view_types(views):
        regulars = []
        nets = []
        completes = []
        stats = []
        sums = []
        for v in views:
            if isinstance(v, list):
                split = v[0].split('|')
            else:
                split = v.split('|')
            if split[-1].endswith('_sum'):
                sums.append(v)
            elif split[1].startswith('d.'):
                stats.append(v)
            elif split[2].endswith(']*:'):
                completes.append(v)
            elif split[-1].startswith('net'):
                nets.append(v)
            else:
                regulars.append(v)
        return regulars, nets, completes, stats, sums

    @staticmethod
    def _switch(views):
        n_views = []
        for v in views:
            if not isinstance(v, list):
                n_views.append(v)
            elif v[0].split('|')[1].startswith('d.'):
                n_views.append(v)
            elif len(v) < 3 and any(view.split('|')[1].startswith('t.') for view in v):
                n_views.append(v)
            else:
                n_views.append(v[1::-1] + v[2:])
        return n_views

    def _request_views(self, data_key=None, filter_key=None, weight=None,
                      frequencies=True, nets=True, descriptives=["mean"],
                      sums=None, coltests=True, mimic='Dim',
                      sig_levels=[".05"], x=None, y=None):
        """
        Get structured, request-ready views from ``self.stack``.

        This function uses the given parameters to inspect the view keys in
        a stack and return them in a standard, structured form suitable for
        use with ``stack.get_chain()`` and
        ``qp.ExcelPainter(..., grouped_views)``. Configurations for
        counts-only (``'c'``), percentages-only (``'p'``) and combination
        counts+percentages (``'cp'``) are returned.

        Parameters
        ----------
        weight : str, default=None
            The name of the weight variable to be used when requesting
            weighted views.
        nets : bool, default=True
            If True, net views (frequency with logic) will be included.
        descriptives : list-like or None, default=["mean"]
            If list-like, the given descriptive statistics will be included,
            (e.g. ["mean", "stddev", "stderr"]. If the list is empty or None
            is given instead, no descriptive statistics will be included.
        sums : {'mid', 'bottom'}, deafult None
            Get any frequency summing views and place them at the bottom or
            in the middle between nets and descriptives.
        coltests : bool, default=True
            If True, column tests (proportions and means) will be included.
        mimic : str
            The mimic type to be targeted when finding coltests.
        sig_levels : list-like, default=[".05"]
            The level/s of significance being requested, e.g. [".05", ".1"]
            or any of ["low", "mid", "high"] for [".10", ".05", ".01"]
            respectively.
        x : str, default=None
            The x-keys to which the results should be restricted.
        y : str, default=None
            The y-keys to which the results should be restricted.

        Returns
        -------
        requested_views : dict
            The returned object is a nested dict in the form :

        ::

            requested_views = {
                'get_chain': {
                    'c': [...],
                    'p': [...],
                    'cp': [...]
                },
                'grouped_views': {
                    'c': [
                        [...],
                        [...],
                        ...
                    ],
                    'p': [
                        [...],
                        [...],
                        ...
                    ],
                    'cp': [
                        [...],
                        [...],
                        ...
                    ]
                }
            }

        """
        stack = self.stack
        described = stack.describe()

        if not data_key is None:
            if not isinstance(data_key, (list, tuple)):
                data_key = [data_key]
            described = described.loc[described['data'].isin(data_key)]

        if not filter_key is None:
            if not isinstance(filter_key, (list, tuple)):
                filter_key = [filter_key]
            described = described.loc[described['filter'].isin(filter_key)]

        if not x is None:
            if not isinstance(x, (list, tuple)):
                x = [x]
            described = described.loc[described['x'].isin(x)]

        if not y is None:
            if not isinstance(y, (list, tuple)):
                y = [y]
            described = described.loc[described['y'].isin(y)]

        all_views = sorted(described['view'].dropna().unique().tolist())

        ci = {'counts': [], 'colpct': [], 'rowpct': [],
              'counts_colpct': [], 'counts_rowpct': [], 'colpct_rowpct': [],
              'counts_colpct_rowpct': []
             }
        requested_views = {'get_chain': ci.copy(), 'grouped_views': ci.copy()}

        # Base views
        # bases = ['x|f|x:|||cbase']
        bases = ['x|f|x:|||cbase_gross', 'x|f|x:|||cbase', 'x|f|x:|||ebase']
        if weight is None:
            weight = ''
        else:
            bases.append('x|f|x:||%s|cbase_gross' % (weight))
            bases.append('x|f|x:||%s|cbase' % (weight))
            bases.append('x|f|x:||%s|ebase' % (weight))

        # Main views
        if frequencies:
            cs = ['x|f|:||%s|counts' % (weight)]
            ps = ['x|f|:|y|%s|c%%' % (weight)]
            rps = ['x|f|:|x|%s|r%%' % (weight)]
            cps = cs[:] + ps [:]
            crps = cs[:] + rps[:]
            psrps = ps[:] + rps[:]
            cpsrps = cs[:] + ps [:] + rps[:]
            csc = ['x|f.c:f|x++:||%s|counts_cumsum' % (weight)]
            csc = csc if csc[0] in all_views else []
            psc = ['x|f.c:f|x++:|y|%s|c%%_cumsum' % (weight)]
            psc = psc if psc[0] in all_views else []
            cpsc = csc[:] + psc[:]
        else:
            cs = []
            ps = []
            rps = []
            cps = []
            crps = []
            psrps = []
            cpsrps = []
            csc = []
            psc = []
            cpsc = []

        levels_ref = {
            "low": ".10",
            "mid": ".05",
            "high": ".01"
        }

        if not isinstance(sig_levels, (list, tuple)):
            sig_levels = [sig_levels]
        lvls = []
        for level in sig_levels:
            # Remove leading 0
            if not isinstance(level, str):
                level = str(level)
            if level[0]=='0': level = level[1:]
            if level in list(levels_ref.keys()):
                lvls.append(levels_ref[level])
            elif not re.match('\.[0-9]$', level) is None:
                lvls.append('{}0'.format(level))
            else:
                lvls.append(level)
        sig_levels = [str(i)[-3:] for i in sorted([float(s) for s in lvls])]
        sig_levels = [
            s if s.startswith('.') else '{}{}'.format(s[1:], 0)
            for s in sig_levels]

        # Column tests for main views
        if coltests:
            for level in sig_levels:
                # Main regular test views
                props_test_views = [
                    v for v in all_views
                    if 't.props.{}{}'.format(
                        mimic,
                        level
                    ) in v
                    and v.split('|')[2]==':'
                    and v.split('|')[4]==weight
                ]
                cs.extend(props_test_views)
                ps.extend(props_test_views)
                rps.extend(props_test_views)
                cps.extend(props_test_views)
                crps.extend(props_test_views)
                psrps.extend(props_test_views)
                cpsrps.extend(props_test_views)

                props_test_views_cumsum = [
                    v for v in all_views
                    if 't.props.{}{}'.format(
                        mimic,
                        level
                    ) in v
                    and v.split('|')[2]=='x++:'
                    and v.split('|')[4]==weight
                ]
                csc.extend(props_test_views_cumsum)
                psc.extend(props_test_views_cumsum)
                cpsc.extend(props_test_views_cumsum)

        # Net views
        if nets:
            net_cs = [
                [v] for v in all_views
                if v.split('|')[1].startswith('f')
                and v.split('|')[2].startswith('x[')
                and v.split('|')[3]==''
                and v.split('|')[4]==weight
            ]
            net_ps = [
                [v] for v in all_views
                if v.split('|')[1].startswith('f')
                and v.split('|')[2].startswith('x[')
                and v.split('|')[3]=='y'
                and v.split('|')[4]==weight
            ]
            net_rps = [
                [v] for v in all_views
                if v.split('|')[1].startswith('f')
                and v.split('|')[2].startswith('x[')
                and v.split('|')[3]=='x'
                and v.split('|')[4]==weight
            ]


            net_cps = []
            for vc in net_cs:
                for vp in net_ps:
                    if  vc[0] == vp[0].replace('|y|', '||'):
                        net_cps.append([vc[0], vp[0]])
                        break

            net_crps = []
            for vc in net_cs:
                for vp in net_rps:
                    if  vc[0] == vp[0].replace('|x|', '||'):
                        net_crps.append([vc[0], vp[0]])
                        break

            net_psrps = []
            for vc in net_ps:
                for vp in net_rps:
                    if  vc[0] == vp[0].replace('|x|', '|y|'):
                        net_psrps.append([vc[0], vp[0]])
                        break

            net_cpsrps = []
            for vc in net_cs:
                for vp in net_ps:
                    for vrp in net_rps:
                        if  vc[0] == vp[0].replace('|y|', '||') and vc[0] == vrp[0].replace('|x|', '||'):
                            net_cpsrps.append([vc[0], vp[0], vrp[0]])
                            break

            # Column tests
            if coltests:
                net_test_views = []
                for level in sig_levels:

                    if nets:
                        # Net test views
                        net_test_views.extend([
                            v for v in all_views
                            if 't.props.{}{}'.format(mimic, level) in v.split('|')[1]
                            and v.split('|')[2].startswith('x[')
                            and v.split('|')[4]==weight
                        ])
                for i, vc in enumerate(net_cs):
                    for vt in net_test_views:
                        eq_relation = vc[0].split('|')[2]  == vt.split('|')[2]
                        eq_weight = vc[0].split('|')[4] == vt.split('|')[4]
                        if eq_relation and eq_weight:
                            net_cs[i].append(vt)
                            net_ps[i].append(vt)
                            net_cps[i].append(vt)
                            if net_rps: net_rps[i].append(vt)
                            if net_crps: net_crps[i].append(vt)
                            if net_psrps: net_psrps[i].append(vt)
                            if net_cpsrps: net_cpsrps[i].append(vt)
        else:
            net_cs = False
            net_ps = False
            net_rps = False
            net_cps = False
            net_crps = False
            net_psrps = False
            net_cpsrps = False

        # Sum views
        if sums:
            sums_cs = [
                [v] for v in all_views
                if v.split('|')[3] == ''
                and v.split('|')[4] == weight
                and v.split('|')[-1].endswith('_sum')
            ]
            sums_ps = [
                [v] for v in all_views
                if v.split('|')[3] == 'y'
                and v.split('|')[4] == weight
                and v.split('|')[-1].endswith('_sum')
            ]

            if sums_cs:
                sums_cps = [[sums_cs[0][0], sums_ps[0][0]]]
                sums_cs_flat = sums_cs[0] if sums_cs else []
                sums_ps_flat = sums_ps[0] if sums_ps else []
                sums_cps_flat = sums_cs_flat[:] + sums_ps_flat[:]
            else:
                sums_cps = []
                sums_cs_flat = []
                sums_ps_flat = []
                sums_cps_flat = []

            sum_chains = [sums_cs_flat, sums_ps_flat, sums_cps_flat]
            sum_gvs = [sums_cs, sums_ps, sums_cps]
        # Descriptive statistics views
        if descriptives:
            views = {}
            rel_w = []
            for descriptive in descriptives:
                views[descriptive] = [
                    [v] for v in all_views
                    if v.split('|')[1].startswith('d.{}'.format(descriptive))
                    and v.split('|')[4] == weight
                ]
                for desv in views[descriptive]:
                    rel = desv[0].split('|')[2]
                    w = desv[0].split('|')[4]
                    if not tuple([rel, w]) in rel_w:
                        rel_w.append(tuple([rel, w]))

                # Column tests
                if descriptive=='mean' and coltests:
                    means_test_views = []
                    for level in sig_levels:
                        # Means test views
                        means_test_views.extend([
                            v for v in all_views
                            if v.split('|')[1].startswith('t.means.{}{}'.format(
                                mimic,
                                level
                            ))
                            and v.split('|')[4]==weight
                        ])

            if 'mean' in descriptives and coltests:
                for i, vbd in enumerate(views['mean']):
                    for vt in means_test_views:
                        eq_relation = vbd[0].split('|')[2] == vt.split('|')[2]
                        eq_weight = vbd[0].split('|')[4] == vt.split('|')[4]
                        if eq_relation and eq_weight:
                            views['mean'][i].append(vt)

            desc = []
            for rel, w in rel_w:
                rel_w_v = []
                for des in descriptives:
                    for desv in views[des]:
                        if desv[0].split('|')[2] == rel and desv[0].split('|')[4] == w:
                            rel_w_v.extend(desv)
                if rel_w_v:
                    desc.append(rel_w_v)

        else:
            desc = False

        # Construct request object

        requested_views['get_chain']['counts'] = bases + cs + csc
        requested_views['get_chain']['colpct'] = bases + ps + psc
        requested_views['get_chain']['rowpct'] = bases + rps + psc
        requested_views['get_chain']['counts_colpct'] = bases + cps + cpsc
        requested_views['get_chain']['counts_rowpct'] = bases + crps + psc
        requested_views['get_chain']['colpct_rowpct'] = bases + psrps + psc
        requested_views['get_chain']['counts_colpct_rowpct'] = bases + cpsrps + psc

        requested_views['grouped_views']['counts'] = [bases, cs, csc]
        requested_views['grouped_views']['colpct'] = [bases, ps, psc]
        requested_views['grouped_views']['rowpct'] = [bases, rps, psc]
        requested_views['grouped_views']['counts_colpct'] = [bases, cps, cpsc]
        requested_views['grouped_views']['counts_rowpct'] = [bases, crps, psc]
        requested_views['grouped_views']['colpct_rowpct'] = [bases, psrps, psc]
        requested_views['grouped_views']['counts_colpct_rowpct'] = [bases, cpsrps, psc]

        if sums == 'mid':
            requested_views['get_chain']['counts'].extend(sums_cs_flat)
            requested_views['get_chain']['colpct'].extend(sums_ps_flat)
            requested_views['get_chain']['rowpct'].extend(sums_ps_flat)
            requested_views['get_chain']['counts_colpct'].extend(sums_cps_flat)
            requested_views['get_chain']['counts_rowpct'].extend(sums_ps_flat)
            requested_views['get_chain']['counts_colpct_rowpct'].extend(sums_cps_flat)

            requested_views['grouped_views']['counts'].extend(sums_cs)
            requested_views['grouped_views']['colpct'].extend(sums_ps)
            requested_views['grouped_views']['rowpct'].extend(sums_ps)
            requested_views['grouped_views']['counts_colpct'].extend(sums_cps)
            requested_views['grouped_views']['counts_rowpct'].extend(sums_ps)
            requested_views['grouped_views']['counts_colpct_rowpct'].extend(sums_cps)

        if nets and net_cs and net_ps and net_cps:
            net_cs_flat = self._shake_nets([v for item in net_cs for v in item])
            net_ps_flat = self._shake_nets([v for item in net_ps for v in item])

            if net_rps:
                net_rps_flat = self._shake_nets([v for item in net_rps for v in item])
            else:
                net_rps_flat = []
            net_cps_flat = self._shake_nets([v for item in net_cps for v in item])

            if net_crps:
                net_crps_flat = self._shake_nets([v for item in net_crps for v in item])
            else:
                net_crps_flat = []

            if net_psrps:
                net_psrps_flat = self._shake_nets([v for item in net_psrps for v in item])
            else:
                net_psrps_flat = []

            if net_cpsrps:
                net_cpsrps_flat = self._shake_nets([v for item in net_cpsrps for v in item])
            else:
                net_cpsrps_flat = []

            requested_views['get_chain']['counts'].extend(net_cs_flat)
            requested_views['get_chain']['colpct'].extend(net_ps_flat)
            requested_views['get_chain']['rowpct'].extend(net_rps_flat)
            requested_views['get_chain']['counts_colpct'].extend(net_cs_flat)
            requested_views['get_chain']['counts_rowpct'].extend(net_crps_flat)
            requested_views['get_chain']['colpct_rowpct'].extend(net_psrps_flat)
            requested_views['get_chain']['counts_colpct_rowpct'].extend(net_cpsrps_flat)

            requested_views['grouped_views']['counts'].extend(net_cs)
            requested_views['grouped_views']['colpct'].extend(net_ps)
            requested_views['grouped_views']['rowpct'].extend(net_rps)
            requested_views['grouped_views']['counts_colpct'].extend(net_cps)
            requested_views['grouped_views']['counts_rowpct'].extend(net_crps)
            requested_views['grouped_views']['colpct_rowpct'].extend(net_psrps)
            requested_views['grouped_views']['counts_colpct_rowpct'].extend(net_cpsrps)

        if descriptives and desc:

            desc_flat = self._shake_descriptives(
                [v for item in desc for v in item],
                descriptives)

            requested_views['get_chain']['counts'].extend(desc_flat)
            requested_views['get_chain']['colpct'].extend(desc_flat)
            requested_views['get_chain']['rowpct'].extend(desc_flat)
            requested_views['get_chain']['counts_colpct'].extend(desc_flat)
            requested_views['get_chain']['counts_rowpct'].extend(desc_flat)
            requested_views['get_chain']['colpct_rowpct'].extend(desc_flat)
            requested_views['get_chain']['counts_colpct_rowpct'].extend(desc_flat)

            requested_views['grouped_views']['counts'].extend(desc)
            requested_views['grouped_views']['colpct'].extend(desc)
            requested_views['grouped_views']['rowpct'].extend(desc)
            requested_views['grouped_views']['counts_colpct'].extend(desc)
            requested_views['grouped_views']['counts_rowpct'].extend(desc)
            requested_views['grouped_views']['colpct_rowpct'].extend(desc)
            requested_views['grouped_views']['counts_colpct_rowpct'].extend(desc)

        if sums == 'bottom':
            requested_views['get_chain']['counts'].extend(sums_cs_flat)
            requested_views['get_chain']['colpct'].extend(sums_ps_flat)
            requested_views['get_chain']['rowpct'].extend(sums_ps_flat)
            requested_views['get_chain']['counts_colpct'].extend(sums_cps_flat)
            requested_views['get_chain']['counts_rowpct'].extend(sums_ps_flat)
            requested_views['get_chain']['counts_colpct_rowpct'].extend(sums_cps_flat)

            requested_views['grouped_views']['counts'].extend(sums_cs)
            requested_views['grouped_views']['colpct'].extend(sums_ps)
            requested_views['grouped_views']['rowpct'].extend(sums_ps)
            requested_views['grouped_views']['counts_colpct'].extend(sums_cps)
            requested_views['grouped_views']['counts_rowpct'].extend(sums_ps)
            requested_views['grouped_views']['counts_colpct_rowpct'].extend(sums_cps)

        # Remove bases and lists with one element
        for key in ['counts', 'colpct', 'rowpct', 'counts_colpct', 'counts_rowpct', 'colpct_rowpct', 'counts_colpct_rowpct']:

            requested_views['grouped_views'][key].pop(0)
            requested_views['grouped_views'][key] = [
                item
                for item in requested_views['grouped_views'][key]
                if len(item) > 1
            ]

            for i, item in enumerate(requested_views['grouped_views'][key]):
                requested_views['grouped_views'][key][i] = [
                    vk
                    for vk in item
                    if vk.split('|')[1] not in ['d.median', 'd.stddev',
                                                'd.sem', 'd.max', 'd.min', 'd.mean',
                                                'd.upper_q', 'd.lower_q'] or
                    vk.split('|')[1] == 'd.mean' and coltests
                ]

            requested_views['grouped_views'][key] = [
                item
                for item in requested_views['grouped_views'][key]
                if len(item) > 1
            ]

            if all(not rg for rg in requested_views['grouped_views'][key]):
                requested_views['grouped_views'][key] = []
        return requested_views


    @staticmethod
    def _uniquify_list(l):
        # De-dupe keys so far:
        # Credit: Dave Kirby's order preserving uniqueifying list function
        # http://www.peterbe.com/plog/uniqifiers-benchmark
        seen = set()
        seen_add = seen.add
        l = [x for x in l if x not in seen and not seen_add(x)]
        return l

    @staticmethod
    def _get_tests_slicer(s, reverse=False):
        """
        Returns the slicer needed to get tests in order from high to low.
        """
        tests_mapper = {}
        for idx_test in s.index:
            if s[idx_test].startswith('t.'):
                # old version backuped:
                # tests_mapper[float(s[idx_test][-3:])] = idx_test
                sigid = float(s[idx_test].split('.')[3].replace('+@', ''))
                tests_mapper[sigid] = idx_test
        tests_slicer = [
            tests_mapper[level]
            for level in sorted(tests_mapper.keys())
        ]
        return tests_slicer

    def _shake(self, l):
        """
        De-dupe and reorder view keys in l for _request_views.
        """

        s = pd.Series(self._uniquify_list(l))
        df = pd.DataFrame(s.str.split('|').tolist())
        df.insert(0, 'view', s)
        if pd.__version__ == '0.19.2':
            df.sort_values(by=[2, 1], inplace=True)
        else:
            df.sort_index(by=[2, 1], inplace=True)
        return df

    def _shake_nets(self, l):
        """
        De-dupe and reorder net view keys in l for _request_views.
        """
        l = self._shake(l)['view'].values.tolist()
        return l

    def _shake_descriptives(self, l, descriptives):
        """
        De-dupe and reorder descriptives view keys in l for _request_views.
        """

        df = self._shake(l)

        grouped = df.groupby(2)

        slicer = []
        for name, group in grouped:
            s = group[1]

            for i, desc in enumerate(descriptives):
                mean_found = False
                tests_done = False
                for idx in s.index:
                    if s[idx].startswith('d.{}'.format(desc)):
                        slicer.append(idx)
                        if desc=='mean':
                            mean_found = True
                    if desc=='mean' and mean_found and not tests_done:
                        tests_slicer = self._get_tests_slicer(s)
                        slicer.extend(tests_slicer)
                        tests_done = True

        s = df.loc[slicer]['view']
        l = s.values.tolist()
        return l


@modify(to_list='text_key')
def net(append_to=[], condition=None, text='', text_key=None):
    """
    Add a well-formed instruction dict for net to a net_map.

    Parameters
    ----------
    append_to: list or dict (list item)
        If a list is provided, the defined net is appended. If a list item is
        provided, the new text is added to the existing net.
    condition: list / dict (complex logic)
        List codes to band categorical answers in a net group. Use complex
        logic if external variables are involved.
    text: str or dict
        Text for the net. If a str is provided, a text_key is required.
        In a dict more than one text_key can be specified, for example:
        text = {'en-GB': 'the first net', 'de-DE': 'das erste net'}
    text_key: str, list of str
        If text is a str, it will be added for all defined text_keys.
    """
    if not (isinstance(text, dict) or text_key):
        raise ValueError("'text' must be a dict or a text_key must be provided.")
    elif not isinstance(text, dict):
        text = {tk: text for tk in text_key}
    if isinstance(append_to, dict):
        append_to['text'].update(text)
    else:
        net = {len(append_to) + 1: condition, 'text': text}
        append_to.append(net)
        return append_to

def calc(expression, text, text_key=None, exclusive=False):
    """
    Produce a well-formed instruction dict for a calculated net.

    At least two net-like groups get connected via a mathematical operator
    ('+', '-', '*', '/').

    Parameters
    ----------
    expression: tuple
        At least two net-like groups get connected via a mathematical operator
        ('+', '-', '*', '/'). The groups are called by position, for example:
        expression = (3, '-', 1)
    text: str or dict
        Text for the calculated net. If a str is provided, a text_key is required.
        In a dict more than one text_key can be specified, for example:
        text = {'en-GB': 'NPS', 'de-DE': 'Net Promotor Score'}
    text_key: str, list of str
        If text is a str, it will be added for all defined text_keys.
    exclusive: bool, default False
        If True the groups are suppressed and only the calculation result is kept.
    """
    if not (isinstance(text, dict) or text_key):
        raise ValueError("'text' must be a dict or a text_key must be provided.")
    elif not isinstance(text, dict):
        text = {tk: text for tk in text_key}
    operator = {'+': add, '-': sub, '*': mul, '/': div}
    instruction = OrderedDict([('calc', tuple(operator.get(e, 'net_{}'.format(e))
                                                           for e in expression)),
                   ('calc_only', exclusive),
                   ('text', text)])
    return instruction
