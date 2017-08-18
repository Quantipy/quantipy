import pandas as pd
from quantipy.core.tools.qp_decorators import modify
from collections import OrderedDict
from itertools import chain
from operator import add, sub, mul, div
import re

class ViewManager(object):
    def __init__(self, stack, basics=True, nets=True, stats=['mean'], tests=None):
        self.stack = stack
        self.basics = basics
        self.nets = nets
        self.stats = stats
        self.tests = tests
        self.views = None
        self.grouping = None
        self.base_spec = None
        self.weighted = False
        self._base_views = None
        self._grouped_views = None
        return None

    def get_views(self, data_key=None, filter_key=None, cell_items='p',
                  weights=None, bases='auto'):
        """
        Query the ``qp.Stack`` for the desired set of ``Views``.

        Parameters
        ----------
        data_key : str, default None
            The data_key name of the ``qp.Stack`` path to be queried.
        filter_key : str, default None
            The filter_key name of the ``qp.Stack`` path to be queried.
        cell_items: {'c', 'p', 'cp'}, default 'p'
            The kind of frequency aggregations that should be returned;
            'c'(ounts), 'p'(ercentages) or both ('cp').
        weights : str, default None
            The name of a weight variable that has been used in the aggregation
            and should now be queried from the ``qp.Stack``.
        bases : {'auto', 'both', 'weighted', 'unweighted'}
            The base view(s) to include. 'auto' will match the base to the
            ``weights`` parameter. If ``weights`` is provided (i.e. the
            parameter is not ``None``), 'both' will try to get both the 'unweighted'
            and the 'weighted' base, 'weighted' / 'unweighted' will try to get
            the respective version of the base view. The latter three will
            automatically fall back to the 'auto' behaviour if the passed value
            would lead to a failure.

        Returns
        -------
        self
        """
        valid_ci = ['c', 'p', 'cp']
        valid_bases = ['auto', 'both', 'weighted', 'unweighted']
        if bases not in valid_bases:
            err = "'bases must be one of {}, not {}!".format(valid_bases, bases)
            raise ValueError(err)
        self.base_spec = bases
        self.weighted = True if weights else False
        if cell_items not in valid_ci:
            err = "'cell_items' must be one of {}, not {}!"
            raise ValueError(err.format(valid_ci, cell_items))
        stack = self.stack
        if not data_key:
            if len(stack.keys()) > 1:
                err = ("Must provide 'data_key' if more than one datasets are "
                       "connected to the Stack!")
                raise ValueError(err)
            else:
                data_key = stack.keys()[0]
        if not filter_key:
            no_filter_ph = 'no_filter' in stack[data_key] and stack[data_key].keys()
            if len(stack[data_key].keys()) > 1 and not no_filter_ph:
                err = ("Must provide 'filter_key' if more than one filter is "
                       "applied to the Stack!")
                raise ValueError(err)
            else:
                filter_key = stack[data_key].keys()[0]

        views = self._request_views(
            data_key=data_key, filter_key=filter_key, weight=weights,
            frequencies=self.basics, nets=self.nets, descriptives=self.stats,
            sums='bottom', coltests=True if self.tests else False,
            sig_levels=self.tests if self.tests else [])

        self._grouped_views = views['grouped_views'][cell_items]
        self.views = views['get_chain'][cell_items]

        self._fixate_base_views()

        return self


    def _fixate_base_views(self):
        views = self.views[:]
        if views[1].split('|')[-1] != 'cbase':
            bases = [views[0]]
            other_views = views[1:]
        else:
            bases = views[:2]
            other_views = views[2:]
        has_both_bases = len(bases) == 2
        if self.base_spec == 'auto':
            if has_both_bases:
                if self.weighted:
                    bases = [bases[1]]
                else:
                    bases = [bases[0]]
            else:
                pass
        elif self.base_spec == 'both':
            pass
        elif self.base_spec == 'weighted':
            if has_both_bases:
                bases = [bases[1]]
            else:
                pass
        elif self.base_spec == 'unweighted':
            if has_both_bases:
                 bases = [bases[0]]
            else:
                pass
        self._base_views = bases
        self.views = other_views
        return None


    def group(self, style='reduce'):
        """
        Reorder the ``.views`` list to group belonging aggregations together.

        Parameters
        ----------
        style : {'reduce', 'repeat'}, default `reduce`
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

        non_grouped = [v for v in self.views if v not in flat_gv]


        if non_grouped:
            if not grouped_views:
                view_collection = non_grouped
            else:
                if grouped_views[-1][0].split('|')[1].startswith('f.c'):
                    regulars = grouped_views[:-1]
                else:
                    regulars = grouped_views

                # We need to grab all isolated stats (all that are not means with
                # tests if tests are requested)

                stats = [v for v in non_grouped if v.split('|')[1].startswith('d.')]

                # if not stats and not non_grouped[1].split('|')[1].startswith('f.c') :
                #     stats =  non_grouped[1:]

                sums = [v for v in non_grouped if v.split('|')[1].startswith('f.c')]

                if not sums and grouped_views[-1][0].split('|')[1].startswith('f.c'):
                    sums = [grouped_views[-1]]

                view_collection = regulars + stats + sums
        else:
            view_collection = grouped_views

        view_collection = self._base_views + view_collection

        for view_sect in view_collection:
            if isinstance(view_sect, list) and style == 'reduce':
                full_grouped_views.append(tuple(view_sect))
            else:
                full_grouped_views.append(view_sect)

        self.views = full_grouped_views
        return self



    def _request_views(self, data_key=None, filter_key=None, weight=None,
                      frequencies=True, nets=True, descriptives=["mean"],
                      sums=None, coltests=True, mimic='Dim',
                      sig_levels=[".05"], x=None, y=None, by_x=False):
        """
        Get structured, request-ready views from the stack.

        This function uses the given parameters to inspect the view keys in
        a stack and return them in a standard, structured form suitable for
        use with ``stack.get_chain()`` and
        ``qp.ExcelPainter(..., grouped_views)``. Configurations for
        counts-only (``'c'``), percentages-only (``'p'``) and combination
        counts+percentages (``'cp'``) are returned.

        Parameters
        ----------
        stack : quantipy.Stack
            The stack instance view keys will be drawn from.
        weight : str, default=None
            The name of the weight variable to be used when requesting
            weighted views.
        nets : bool, default=True
            If True, net views (frequency with logic) will be included.
        descriptives : list-like or None, default=["mean"]
            If list-like, the given descriptive statistics will be included,
            (e.g. ["mean", "stddev", "stderr"]. If the list is empty or None
            is given instead, no descriptive statistics will be included.
        sums : {'bottom'}, default None
            Get any frequency summing views and place them at the bottom.
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
        by_x : bool, default=False
            If True, the get_chain object in the returned dict will be
            structured as a dict of x-keys.

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

        if by_x:
            xks = described['x'].unique().tolist()
            requested_views = {
                'get_chain': {
                    xk: {'c': [], 'p': [], 'cp': []}
                    for xk in xks
                },
                'grouped_views': {'c': [], 'p': [], 'cp': []}
            }
            xks_views = {
                xk: [vk for vk in described.loc[described['x']==xk]['view']]
                for xk in xks
            }
        else:
            requested_views = {
                'get_chain': {'c': [], 'p': [], 'cp': []},
                'grouped_views': {'c': [], 'p': [], 'cp': []}
            }

        # Base views
        bases = ['x|f|x:|||cbase']
        if weight is None:
            weight = ''
        else:
            bases.append('x|f|x:||%s|cbase' % (weight))

        # Main views
        if frequencies:
            cs = ['x|f|:||%s|counts' % (weight)]
            ps = ['x|f|:|y|%s|c%%' % (weight)]
            cps = cs[:] + ps [:]
            csc = ['x|f.c:f|x++:||%s|counts_cumsum' % (weight)]
            psc = ['x|f.c:f|x++:|y|%s|c%%_cumsum' % (weight)]
            cpsc = csc[:] + psc[:]
        else:
            cs = []
            ps = []
            cps = []
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
            if not isinstance(level, (str, unicode)):
                level = str(level)
            if level[0]=='0': level = level[1:]
            if level in levels_ref.keys():
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
                cps.extend(props_test_views)

            for level in sig_levels:
                # Main cumulative test views
                props_test_views = [
                    v for v in all_views
                    if 't.props.{}{}'.format(
                        mimic,
                        level
                    ) in v
                    and v.split('|')[2]=='x++:'
                    and v.split('|')[4]==weight
                ]
                csc.extend(props_test_views)
                psc.extend(props_test_views)
                cpsc.extend(props_test_views)

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
            net_cps = []
            for vc in net_cs:
                for vp in net_ps:
                    if  vc[0] == vp[0].replace('|y|', '||'):
                        net_cps.append([vc[0], vp[0]])
                        break

            # Column tests
            if coltests:
                net_test_views = []
                for level in sig_levels:

                    if nets:
                        # Net test views
                        net_test_views.extend([
                            v for v in all_views
                            if v.split('|')[1]=='t.props.{}{}'.format(
                                mimic,
                                level
                            )
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
        else:
            net_cs = False
            net_ps = False
            net_cps = False

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


        # Descriptive statistics views
        if descriptives:
            views = {}
            for descriptive in descriptives:
                views[descriptive] = [
                    [v] for v in all_views
                    if v.split('|')[1] == 'd.{}'.format(descriptive)
                    and v.split('|')[4] == weight
                ]

                # Column tests
                if descriptive=='mean' and coltests:
                    means_test_views = []
                    for level in sig_levels:

                        # Means test views
                        means_test_views.extend([
                            v for v in all_views
                            if v.split('|')[1]=='t.means.{}{}'.format(
                                mimic,
                                level
                            )
                            and v.split('|')[4]==weight
                        ])

            base_desc =  descriptives[0]
            if 'mean' in descriptives and coltests:
                for i, vbd in enumerate(views['mean']):
                    for vt in means_test_views:
                        eq_relation = vbd[0].split('|')[2] == vt.split('|')[2]
                        eq_weight = vbd[0].split('|')[4] == vt.split('|')[4]
                        if eq_relation and eq_weight:
                            views['mean'][i].append(vt)

            if len(descriptives) > 1:
                for i, vbd in enumerate(views[base_desc]):
                    for rem_desc in descriptives[1:]:
                        for vrd in views[rem_desc]:
                            eq_relation = vbd[0].split('|')[2]  == vrd[0].split('|')[2]
                            eq_weight = vbd[0].split('|')[4] == vrd[0].split('|')[4]
                            if eq_relation and eq_weight:
                                views[base_desc][i].extend(vrd)

            desc = views[base_desc]
            # ----------------------------------------------------------------
            # This section reorders the all descriptives so that all means +
            # their tests are leading and other stats follow...
            # ----------------------------------------------------------------
            # mean_dom_desc = []
            # stats = []
            # for d in desc:
            #     valid_d = ('t.means', 'd.mean')
            #     means = [v for v in d if v.split('|')[1].startswith(valid_d)]
            #     d_stats = [v for v in d if not v.split('|')[1].startswith(valid_d)]
            #     stats.extend(d_stats)
            #     mean_dom_desc.append(means)
            # mean_dom_desc[-1].extend(stats)
            # desc = mean_dom_desc
            # ----------------------------------------------------------------
        else:
            desc = False

        # Construct request object
        if by_x:
            xks = described['x'].unique().tolist()
            all_views = {
                xk: described.loc[described['x']==xk]['view'].unique().tolist()
                for xk in xks
            }

        if by_x:
            for xk in xks:
                requested_views['get_chain'][xk]['c'] = bases + cs + csc
                requested_views['get_chain'][xk]['p'] = bases + ps + psc
                requested_views['get_chain'][xk]['cp'] = bases + cps + cpsc
        else:
            requested_views['get_chain']['c'] = bases + cs + csc
            requested_views['get_chain']['p'] = bases + ps + psc
            requested_views['get_chain']['cp'] = bases + cps + cpsc

        requested_views['grouped_views']['c'] = [bases, cs, csc]
        requested_views['grouped_views']['p'] = [bases, ps, psc]
        requested_views['grouped_views']['cp'] = [bases, cps, cpsc]

        if nets and net_cs and net_ps and net_cps:

            net_cs_flat = self._shake_nets([v for item in net_cs for v in item])
            net_ps_flat = self._shake_nets([v for item in net_ps for v in item])
            net_cps_flat = self._shake_nets([v for item in net_cps for v in item])

            if by_x:
                for xk in xks:
                    requested_views['get_chain'][xk]['c'].extend([
                        v for v in net_cs_flat
                        if v in xks_views[xk]])
                    requested_views['get_chain'][xk]['p'].extend([
                        v for v in net_ps_flat
                        if v in xks_views[xk]])
                    requested_views['get_chain'][xk]['cp'].extend([
                        v for v in net_cps_flat
                        if v in xks_views[xk]])
            else:
                requested_views['get_chain']['c'].extend(net_cs_flat)
                requested_views['get_chain']['p'].extend(net_ps_flat)
                requested_views['get_chain']['cp'].extend(net_cps_flat)


            requested_views['grouped_views']['c'].extend(net_cs)
            requested_views['grouped_views']['p'].extend(net_ps)
            requested_views['grouped_views']['cp'].extend(net_cps)


        if descriptives and desc:

            desc_flat = self._shake_descriptives(
                [v for item in desc for v in item],
                descriptives)

            if by_x:
                for xk in xks:
                    requested_views['get_chain'][xk]['c'].extend([
                        v for v in desc_flat
                        if v in xks_views[xk]])
                    requested_views['get_chain'][xk]['p'].extend([
                        v for v in desc_flat
                        if v in xks_views[xk]])
                    requested_views['get_chain'][xk]['cp'].extend([
                        v for v in desc_flat
                        if v in xks_views[xk]])
            else:
                requested_views['get_chain']['c'].extend(desc_flat)
                requested_views['get_chain']['p'].extend(desc_flat)
                requested_views['get_chain']['cp'].extend(desc_flat)

            requested_views['grouped_views']['c'].extend(desc)
            requested_views['grouped_views']['p'].extend(desc)
            requested_views['grouped_views']['cp'].extend(desc)

        if sums:
            requested_views['get_chain']['c'].extend(sums_cs_flat)
            requested_views['get_chain']['p'].extend(sums_ps_flat)
            requested_views['get_chain']['cp'].extend(sums_cps_flat)

            requested_views['grouped_views']['c'].extend(sums_cs)
            requested_views['grouped_views']['p'].extend(sums_ps)
            requested_views['grouped_views']['cp'].extend(sums_cps)

        # Remove bases and lists with one element
        for key in ['c', 'p', 'cp']:
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
                                                'd.sem', 'd.max', 'd.min']
                ]
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
                tests_mapper[float(s[idx_test][-3:])] = idx_test
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
                    if s[idx]=='d.{}'.format(desc):
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
