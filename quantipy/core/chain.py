#!/usr/bin/python
# -*- coding: utf-8 -*-

import copy

from quantipy.core.chainannotations import ChainAnnotations
from quantipy.core.view import View
from quantipy.core.tools.logger import get_logger
logger = get_logger(__name__)


class _TransformedChainDF(object):

    def __init__(self, chain):
        c = chain.clone()
        self.org_views = c.views
        self.df = c._frame
        self._org_idx = self.df.index
        self._edit_idx = range(0, len(self._org_idx))
        self._idx_valmap = {
            n: o
            for n, o in zip(self._edit_idx, self._org_idx.get_level_values(1))}
        self.df.index = self._edit_idx
        self._org_col = self.df.columns
        self._edit_col = range(0, len(self._org_col))
        self._col_valmap = {
            n: o
            for n, o in zip(self._edit_col, self._org_col.get_level_values(1))}
        self.df.columns = self._edit_col
        self.array_mi = c._array_style == 0
        self.nested_y = c._nested_y
        self._nest_mul = self._nesting_multiplier()

    def _nesting_multiplier(self):
        """
        """
        levels = self._org_col.nlevels
        if levels == 2:
            return 1
        else:
            return (levels / 2) + 1

    def _insert_viewlikes(self, new_index_flat, org_index_mapped):
        inserts = [new_index_flat.index(val) for val in new_index_flat
                   if not val in org_index_mapped.values()]
        flatviews = []
        for name, no in self.org_views.items():
            e = [name] * no
            flatviews.extend(e)
        for vno, i in enumerate(inserts):
            flatviews.insert(i, '__viewlike__{}'.format(vno))
        new_views = OrderedDict()
        no_of_views = Counter(flatviews)
        for fv in flatviews:
            if not fv in new_views: new_views[fv] = no_of_views[fv]
        return new_views

    def _updated_index_tuples(self, axis):
        """
        """
        if axis == 1:
            current = self.df.columns.values.tolist()
            mapped = self._col_valmap
            org_tuples = self._org_col.tolist()
        else:
            current = self.df.index.values.tolist()
            mapped = self._idx_valmap
            org_tuples = self._org_idx.tolist()
        merged = [mapped[val] if val in mapped else val for val in current]
        # ================================================================
        if (self.array_mi and axis == 1) or axis == 0:
            self._transf_views = self._insert_viewlikes(merged, mapped)
        else:
            self._transf_views = self.org_views
        # ================================================================
        i = d = 0
        new_tuples = []
        for merged_val in merged:
            idx = i-d if i-d != len(org_tuples) else i-d-1
            if org_tuples[idx][1] == merged_val:
                new_tuples.append(org_tuples[idx])
            else:
                empties = ['*'] * self._nest_mul
                new_tuple = tuple(empties + [merged_val])
                new_tuples.append(new_tuple)
                d += 1
            i += 1
        return new_tuples

    def _reindex(self):
        """
        """
        y_names = ['Question', 'Values']
        if not self.array_mi:
            x_names = y_names
        else:
            x_names = ['Array', 'Questions']
        if self.nested_y: y_names = y_names * (self._nest_mul - 1)
        tuples = self._updated_index_tuples(axis=1)
        self.df.columns = pd.MultiIndex.from_tuples(tuples, names=y_names)
        tuples = self._updated_index_tuples(axis=0)
        self.df.index = pd.MultiIndex.from_tuples(tuples, names=x_names)
        return None


class Chain(object):

    def __init__(self, stack, name, structure=None):
        self.stack = stack
        self.name = name
        self.structure = structure
        self.source = 'native'
        self.edited = False
        self._custom_views = None
        self.double_base = False
        self.grouping = None
        self.sig_test_letters = None
        self.totalize = False
        self.base_descriptions = None
        self.painted = False
        self.hidden = False
        self.annotations = ChainAnnotations()
        self._array_style = None
        self._group_style = None
        self._meta = None
        self._x_keys = None
        self._y_keys = None
        self._given_views = None
        self._grp_text_map = []
        self._text_map = None
        self._custom_texts = {}
        self._transl = View._metric_name_map()
        self._pad_id = None
        self._has_rules = None
        self._flag_bases = None
        self._is_mask_item = False
        self._shapes = None

        # properties
        self._frame = None  # self.dataframe

        # lazy properties
        self._default_text = None
        self._orientation = None

    def export(self):
        return _TransformedChainDF(self)

    def assign(self, transformed_chain_df):
        if not isinstance(transformed_chain_df, _TransformedChainDF):
            raise ValueError("Must pass an exported ``Chain`` instance!")
        transformed_chain_df._reindex()
        self._frame = transformed_chain_df.df
        self.views = transformed_chain_df._transf_views
        return None

    def __str__(self):
        if self.structure is not None:
            return '{}...\n{}'.format(
                self.__class__.__name__, str(self.structure.head()))

        str_format = (
            '{}...'
            '\nSource:          {}'
            '\nName:            {}'
            '\nOrientation:     {}'
            '\nX:               {}'
            '\nY:               {}'
            '\nNumber of views: {}')

        return str_format.format(
            self.__class__.__name__,
            getattr(self, 'source', 'native'),
            getattr(self, 'name', 'None'),
            getattr(self, 'orientation', 'None'),
            getattr(self, '_x_keys', 'None'),
            getattr(self, '_y_keys', 'None'),
            getattr(self, 'views', 'None'))

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        """Returns the total number of cells in the Chain.dataframe"""
        x = len(getattr(self, 'index', []))
        y = len(getattr(self, 'columns', []))
        return x * y

    def clone(self):
        return copy.deepcopy(self)

    @property
    def default_text(self):
        if not self._default_text:
            tk = self._meta['lib']['default text']
            if tk not in self._transl:
                self._transl[tk] = self._transl['en-GB']
            self._default_text = tk
        return self._default_text

    @property
    def orientation(self):
        if not self._orientation:
            if len(self._x_keys) == 1:
                self._orientation = "x"
            elif len(self._y_keys) == 1:
                self._orientation = "y"
            else:
                self._orientation = None
        return self._orientation

    @property
    def axis(self):
        return int(self.orientation == "x")

    @property
    def axes(self):
        if self.axis:
            return self._x_keys, self._y_keys
        else:
            return self._y_keys, self._x_keys

    @property
    def dataframe(self):
        return self._frame

    @property
    def index(self):
        if self.dataframe:
            return self._frame.index

    @property
    def columns(self):
        if self.dataframe:
            return self._frame.columns

    @property
    def frame_values(self):
        if self.dataframe:
            return self._frame.values

    @frame_values.setter
    def frame_values(self, frame_values):
        self._frame_values = frame_values

    @property
    def views(self):
        return self._views

    @views.setter
    def views(self, views):
        self._views = views

    @property
    def array_style(self):
        return self._array_style

    @property
    def shapes(self):
        if self._shapes is None:
            self._shapes = []
        return self._shapes

    @array_style.setter
    def array_style(self, link):
        array_style = -1
        for view in link.keys():
            if link[view].meta()['x']['is_array']:
                array_style = 0
            if link[view].meta()['y']['is_array']:
                array_style = 1
        self._array_style = array_style

    @property
    def pad_id(self):
        if self._pad_id is None:
            self._pad_id = 0
        else:
            self._pad_id += 1
        return self._pad_id

    @property
    def sig_levels(self):
        sigs = set([v for v in self._valid_views(True)
                    if v.split('|')[1].startswith('t.')])
        tests = [t.split('|')[1].split('.')[1] for t in sigs]
        levels = [t.split('|')[1].split('.')[3] for t in sigs]
        sig_levels = {}
        for m in zip(tests, levels):
            l = '.{}'.format(m[1])
            t = m[0]
            if t in sig_levels:
                sig_levels[t].append(l)
            else:
                sig_levels[t] = [l]
        return sig_levels

    @property
    def cell_items(self):
        if self.views:
            compl_views = [v for v in self.views if ']*:' in v]
            check_views = compl_views[:] or self.views.copy()
            for v in check_views:
                if v.startswith('__viewlike__'):
                    if compl_views:
                        check_views.remove(v)
                    else:
                        del check_views[v]

            non_freqs = ('d.', 't.')
            c = any(v.split('|')[3] == '' and
                    not v.split('|')[1].startswith(non_freqs) and
                    not v.split('|')[-1].startswith('cbase')
                    for v in check_views)
            col_pct = any(v.split('|')[3] == 'y' and
                          not v.split('|')[1].startswith(non_freqs) and
                          not v.split('|')[-1].startswith('cbase')
                          for v in check_views)
            row_pct = any(v.split('|')[3] == 'x' and
                          not v.split('|')[1].startswith(non_freqs) and
                          not v.split('|')[-1].startswith('cbase')
                          for v in check_views)
            c_colpct = c and col_pct
            c_rowpct = c and row_pct
            c_colrow_pct = c_colpct and c_rowpct
            single_ci = not (c_colrow_pct or c_colpct or c_rowpct)
            if single_ci:
                if c:
                    return 'counts'
                elif col_pct:
                    return 'colpct'
                else:
                    return 'rowpct'
            else:
                if c_colrow_pct:
                    return 'counts_colpct_rowpct'
                elif c_colpct:
                    if self._counts_first():
                        return 'counts_colpct'
                    else:
                        return 'colpct_counts'
                else:
                    return 'counts_rowpct'

    @property
    def _ci_simple(self):
        ci = []
        if self.views:
            for v in self.views:
                if 'significance' in v:
                    continue
                if ']*:' in v:
                    if v.split('|')[3] == '':
                        if 'N' not in ci:
                            ci.append('N')
                    if v.split('|')[3] == 'y':
                        if 'c%' not in ci:
                            ci.append('c%')
                    if v.split('|')[3] == 'x':
                        if 'r%' not in ci:
                            ci.append('r%')
                else:
                    if v.split('|')[-1] == 'counts':
                        if 'N' not in ci:
                            ci.append('N')
                    elif v.split('|')[-1] == 'c%':
                        if 'c%' not in ci:
                            ci.append('c%')
                    elif v.split('|')[-1] == 'r%':
                        if 'r%' not in ci:
                            ci.append('r%')

            return ci

    @property
    def ci_count(self):
        return len(self.cell_items.split('_'))

    @property
    def contents(self):
        if self.structure:
            return
        nested = self._array_style == 0
        if nested:
            dims = self._frame.shape
            contents = {row: {col: {} for col in range(0, dims[1])}
                        for row in range(0, dims[0])}
        else:
            contents = dict()
        for row, idx in enumerate(self._views_per_rows()):
            if nested:
                for i, v in idx.items():
                    contents[row][i] = self._add_contents(v)
            else:
                contents[row] = self._add_contents(idx)
        return contents

    @property
    def cell_details(self):
        lang = self._default_text if self._default_text == 'fr-FR' else 'en-GB'
        cd = CELL_DETAILS[lang]
        ci = self.cell_items
        cd_str = '%s (%s)' % (cd['cc'], ', '.join([cd[_] for _ in self._ci_simple]))
        against_total = False
        if self.sig_test_letters:
            mapped = ''
            group = None
            i =  0 if (self._frame.columns.nlevels in [2, 3]) else 4

            for letter, lab in zip(self.sig_test_letters, self._frame.columns.labels[-i]):
                if letter == '@':
                    continue
                if group is not None:
                    if lab == group:
                        mapped += '/' + letter
                    else:
                        group = lab
                        mapped += ', ' + letter
                else:
                    group = lab
                    mapped += letter
            test_types = cd['cp']
            if self.sig_levels.get('means'):
                test_types += ', ' + cd['cm']

            levels = []
            for key in ('props', 'means'):
                for level in self.sig_levels.get(key, iter(())):
                    l = '%s%%' % int(100. - float(level.split('+@')[0].split('.')[1]))
                    if l not in levels:
                        levels.append(l)
                    if '+@' in level:
                        against_total = True

            cd_str = cd_str[:-1] + ', ' + cd['str'] +'), '
            cd_str += '%s (%s, (%s): %s' % (cd['stats'], test_types, ', '.join(levels), mapped)
            if self._flag_bases:
                flags = ([], [])
                [(flags[0].append(min), flags[1].append(small)) for min, small in self._flag_bases]
                cd_str += ', %s: %s (**), %s: %s (*)' % (cd['mb'], ', '.join(map(str, flags[0])),
                                                         cd['sb'], ', '.join(map(str, flags[1])))
            cd_str += ')'

        cd_str = [cd_str]

        if against_total:
            cd_str.extend([cd['up'], cd['down']])
        return cd_str

    def describe(self):
        def _describe(cell_defs, row_id):
            descr = []
            for r, m in cell_defs.items():
                descr.append(
                    [k if isinstance(v, bool) else v for k, v in m.items() if v])
            if any('is_block' in d for d in descr):
                blocks = self._describe_block(descr, row_id)
                calc = 'calc' in blocks
                for d, b in zip(descr, blocks):
                    if b:
                        d.append(b) if not calc else d.extend([b, 'has_calc'])
            return descr
        if self._array_style == 0:
            description = {k: _describe(v, k) for k, v in self.contents.items()}
        else:
            description = _describe(self.contents, None)
        return description

    def _fill_cells(self):
        """
        """
        self._frame = self._frame.fillna(method='ffill')
        return None


    # @lazy_property
    def _counts_first(self):
        for v in self.views:
            sname = v.split('|')[-1]
            if sname in ['counts', 'c%']:
                if sname == 'counts':
                    return True
                else:
                    return False

    #@property
    def _views_per_rows(self):
        """
        """
        base_vk = 'x|f|x:||{}|cbase'
        counts_vk = 'x|f|:||{}|counts'
        pct_vk = 'x|f|:|y|{}|c%'
        mean_vk = 'x|d.mean|:|y|{}|mean'
        stddev_vk = 'x|d.stddev|:|y|{}|stddev'
        variance_vk = 'x|d.var|:|y|{}|var'
        sem_vk = 'x|d.sem|:|y|{}|sem'

        if self.source == 'Crunch multitable':
            ci = self._meta['display_settings']['countsOrPercents']
            w = self._meta['weight']
            if ci == 'counts':
                main_vk = counts_vk.format(w if w else '')
            else:
                main_vk = pct_vk.format(w if w else '')
            base_vk = base_vk.format(w if w else '')
            metrics = [base_vk] + (len(self.dataframe.index)-1) * [main_vk]
        elif self.source == 'Dimensions MTD':
            ci = self._meta['cell_items']
            w = None
            axis_vals = [axv['Type'] for axv in self._meta['index-emetas']]
            metrics = []
            for axis_val in axis_vals:
                if axis_val == 'Base':
                    metrics.append(base_vk.format(w if w else ''))
                if axis_val == 'UnweightedBase':
                    metrics.append(base_vk.format(w if w else ''))
                elif axis_val == 'Category':
                    metrics.append(counts_vk.format(w if w else ''))
                elif axis_val == 'Mean':
                    metrics.append(mean_vk.format(w if w else ''))
                elif axis_val == 'StdDev':
                    metrics.append(stddev_vk.format(w if w else ''))
                elif axis_val == 'StdErr':
                    metrics.append(sem_vk.format(w if w else ''))
                elif axis_val == 'SampleVar':
                    metrics.append(variance_vk.format(w if w else ''))
            return metrics
        else:
            #  Native Chain views
            # ----------------------------------------------------------------
            if self.edited and (self._custom_views and not self.array_style == 0):
                return self._custom_views
            else:
                if self._array_style != 0:
                    metrics = []
                    if self.orientation == 'x':
                        for view in self._valid_views():
                            view = self._force_list(view)
                            initial = view[0]
                            size = self.views[initial]
                            metrics.extend(view * size)
                    else:
                        for view_part in self.views:
                            for view in self._valid_views():
                                view = self._force_list(view)
                                initial = view[0]
                                size = view_part[initial]
                                metrics.extend(view * size)
                else:
                    counts = []
                    colpcts =  []
                    rowpcts = []
                    metrics = []
                    ci = self.cell_items
                    for v in self.views.keys():
                        if not v.startswith('__viewlike__'):
                            parts = v.split('|')
                            is_completed = ']*:' in v
                            if not self._is_c_pct(parts):
                                counts.extend([v]*self.views[v])
                            if self._is_r_pct(parts):
                                rowpcts.extend([v]*self.views[v])
                            if (self._is_c_pct(parts) or self._is_base(parts) or
                                self._is_stat(parts)):
                                colpcts.extend([v]*self.views[v])
                        else:
                            counts = counts + ['__viewlike__']
                            colpcts = colpcts + ['__viewlike__']
                            rowpcts = rowpcts + ['__viewlike__']
                    dims = self._frame.shape
                    for row in range(0, dims[0]):
                        if ci in ['counts_colpct', 'colpct_counts'] and self.grouping:
                            if row % 2 == 0:
                                if self._counts_first():
                                    vc = counts
                                else:
                                    vc = colpcts
                            else:
                                if not self._counts_first():
                                    vc = counts
                                else:
                                    vc = colpcts
                        else:
                            vc = counts if ci == 'counts' else colpcts
                        metrics.append({col: vc[col] for col in range(0, dims[1])})
        return metrics

    def _valid_views(self, flat=False):
        clean_view_list = []
        valid = self.views.keys()
        org_vc = self._given_views
        v_likes = [v for v in valid if v.startswith('__viewlike__')]
        if isinstance(org_vc, tuple):
            v_likes = tuple(v_likes)
        view_coll = org_vc + v_likes
        for v in view_coll:
            if isinstance(v, (str, unicode)):
                if v in valid:
                    clean_view_list.append(v)
            else:
                new_v = []
                for sub_v in v:
                    if sub_v in valid:
                        new_v.append(sub_v)
                if isinstance(v, tuple):
                    new_v = list(new_v)
                if new_v:
                    if len(new_v) == 1: new_v = new_v[0]
                    if not flat:
                        clean_view_list.append(new_v)
                    else:
                        if isinstance(new_v, list):
                            clean_view_list.extend(new_v)
                        else:
                            clean_view_list.append(new_v)
        return clean_view_list


    def _add_contents(self, viewelement):
        """
        """
        if viewelement.startswith('__viewlike__'):
            parts = '|||||'
            viewlike = True
        else:
            parts = viewelement.split('|')
            viewlike = False
        return dict(is_default=self._is_default(parts),
                    is_c_base=self._is_c_base(parts),
                    is_r_base=self._is_r_base(parts),
                    is_e_base=self._is_e_base(parts),
                    is_c_base_gross=self._is_c_base_gross(parts),
                    is_counts=self._is_counts(parts),
                    is_c_pct=self._is_c_pct(parts),
                    is_r_pct=self._is_r_pct(parts),
                    is_res_c_pct=self._is_res_c_pct(parts),
                    is_counts_sum=self._is_counts_sum(parts),
                    is_c_pct_sum=self._is_c_pct_sum(parts),
                    is_counts_cumsum=self._is_counts_cumsum(parts),
                    is_c_pct_cumsum=self._is_c_pct_cumsum(parts),
                    is_net=self._is_net(parts),
                    is_block=self._is_block(parts),
                    is_calc_only = self._is_calc_only(parts),
                    is_mean=self._is_mean(parts),
                    is_stddev=self._is_stddev(parts),
                    is_min=self._is_min(parts),
                    is_max=self._is_max(parts),
                    is_median=self._is_median(parts),
                    is_variance=self._is_variance(parts),
                    is_sem=self._is_sem(parts),
                    is_varcoeff=self._is_varcoeff(parts),
                    is_percentile=self._is_percentile(parts),
                    is_propstest=self._is_propstest(parts),
                    is_meanstest=self._is_meanstest(parts),
                    is_weighted=self._is_weighted(parts),
                    weight=self._weight(parts),
                    is_stat=self._is_stat(parts),
                    stat=self._stat(parts),
                    siglevel=self._siglevel(parts),
                    is_viewlike=viewlike)

    def _row_pattern(self, target_ci):
        """
        """
        cisplit = self.cell_items.split('_')
        if target_ci == 'c%':
            start = cisplit.index('colpct')
        elif target_ci == 'counts':
            start = cisplit.index('counts')
        repeat = self.ci_count
        return (start, repeat)

    def _view_idxs(self, view_tags, keep_tests=True, keep_bases=True, names=False, ci=None):
        """
        """
        if not isinstance(view_tags, list): view_tags = [view_tags]
        rowmeta = self.named_rowmeta
        nested = self.array_style == 0
        if nested:
            if self.ci_count > 1:
                rp_idx = self._row_pattern(ci)[0]
                rowmeta = rowmeta[rp_idx]
            else:
                rp_idx = 0
                rowmeta = rowmeta[0]
        rows = []
        for r in rowmeta:
            is_code = str(r[0]).isdigit()
            if 'is_counts' in r[1] and is_code:
                rows.append(('counts', r[1]))
            elif 'is_c_pct' in r[1] and is_code:
                rows.append(('c%', r[1]))
            elif 'is_propstest' in r[1]:
                rows.append((r[0], r[1]))
            elif 'is_meanstest' in r[1]:
                rows.append((r[0], r[1]))
            else:
                rows.append(r)
        invalids = []
        if not keep_tests:
            invalids.extend(['is_propstest', 'is_meanstest'])
        if ci == 'counts':
            invalids.append('is_c_pct')
        elif ci == 'c%':
            invalids.append('is_counts')
        idxs = []
        names = []
        order = []
        for i, row in enumerate(rows):
            if any([invalid in row[1] for invalid in invalids]):
                if not (row[0] == 'All' and keep_bases): continue
            if row[0] in view_tags:
                order.append(view_tags.index(row[0]))
                idxs.append(i)
                if nested:
                    names.append(self._views_per_rows()[rp_idx][i])
                else:
                    names.append(self._views_per_rows()[i])
        return (idxs, order) if not names else (idxs, names, order)


    @staticmethod
    def _remove_grouped_blanks(viewindex_labs):
        """
        """
        full = []
        for v in viewindex_labs:
            if v == '':
                full.append(last)
            else:
                last = v
                full.append(last)
        return full

    def _slice_edited_index(self, axis, positions):
        """
        """
        l_zero = axis.get_level_values(0).values.tolist()[0]
        l_one = axis.get_level_values(1).values.tolist()
        l_one = [l_one[p] for p in positions]
        axis_tuples = [(l_zero, lab) for lab in l_one]
        if self.array_style == 0:
            names = ['Array', 'Questions']
        else:
            names = ['Question', 'Values']
        return pd.MultiIndex.from_tuples(axis_tuples, names=names)


    def _non_grouped_axis(self):
        """
        """
        axis = self._frame.index
        l_zero = axis.get_level_values(0).values.tolist()[0]
        l_one = axis.get_level_values(1).values.tolist()
        l_one = self._remove_grouped_blanks(l_one)
        axis_tuples = [(l_zero, lab) for lab in l_one]
        if self.array_style == 0:
            names = ['Array', 'Questions']
        else:
            names = ['Question', 'Values']
        self._frame.index = pd.MultiIndex.from_tuples(axis_tuples, names=names)
        return None

    @property
    def named_rowmeta(self):
        if self.painted:
            self.toggle_labels()
        d = self.describe()
        if self.array_style == 0:
            n = self._frame.columns.get_level_values(1).values.tolist()
            n = self._remove_grouped_blanks(n)
            mapped = {rowid: zip(n, rowmeta) for rowid, rowmeta in d.items()}
        else:
            n = self._frame.index.get_level_values(1).values.tolist()
            n = self._remove_grouped_blanks(n)
            mapped = zip(n, d)
        if not self.painted: self.toggle_labels()
        return mapped

    @lazy_property
    def _nested_y(self):
        return any('>' in v for v in self._y_keys)

    def _is_default(self, parts):
        return parts[-1] == 'default'

    def _is_c_base(self, parts):
        return parts[-1] == 'cbase'

    def _is_r_base(self, parts):
        return parts[-1] == 'rbase'

    def _is_e_base(self, parts):
        return parts[-1] == 'ebase'

    def _is_c_base_gross(self, parts):
        return parts[-1] == 'cbase_gross'

    def _is_base(self, parts):
        return (self._is_c_base(parts) or
                self._is_c_base_gross(parts) or
                self._is_e_base(parts) or
                self._is_r_base(parts))

    def _is_counts(self, parts):
        return parts[1].startswith('f') and parts[3] == ''

    def _is_c_pct(self, parts):
        return parts[1].startswith('f') and parts[3] == 'y'

    def _is_r_pct(self, parts):
        return parts[1].startswith('f') and parts[3] == 'x'

    def _is_res_c_pct(self, parts):
        return parts[-1] == 'res_c%'

    def _is_net(self, parts):
        return parts[1].startswith(('f', 'f.c:f', 't.props')) and \
               len(parts[2]) > 3 and not parts[2] == 'x++'

    def _is_calc_only(self, parts):
        if self._is_net(parts) and not self._is_block(parts):
            return ((self.__has_freq_calc(parts) or
                     self.__is_calc_only_propstest(parts)) and not
                    (self._is_counts_sum(parts) or self._is_c_pct_sum(parts)))
        else:
            return False

    def _is_block(self, parts):
        if self._is_net(parts):
            conditions = parts[2].split('[')
            multiple_conditions = len(conditions) > 2
            expand = '+{' in parts[2] or '}+' in parts[2]
            complete = '*:' in parts[2]
            if expand or complete:
                return True
            if multiple_conditions:
                if self.__has_operator_expr(parts):
                    return True
                return False
            return False
        return False

    def _stat(self, parts):
        if parts[1].startswith('d.'):
            return parts[1].split('.')[-1]
        else:
            return None

    # non-meta relevant helpers
    def __has_operator_expr(self, parts):
        e = parts[2]
        for syntax in [']*:', '[+{', '}+']:
            if syntax in e: e = e.replace(syntax, '')
        ops = ['+', '-', '*', '/']
        return any(len(e.split(op)) > 1 for op in ops)

    def __has_freq_calc(self, parts):
        return parts[1].startswith('f.c:f')

    def __is_calc_only_propstest(self, parts):
        return self._is_propstest(parts) and self.__has_operator_expr(parts)

    @staticmethod
    def _statname(parts):
        split = parts[1].split('.')
        if len(split) > 1:
            return split[1]
        return split[-1]

    def _is_mean(self, parts):
        return self._statname(parts) == 'mean'

    def _is_stddev(self, parts):
        return self._statname(parts) == 'stddev'

    def _is_min(self, parts):
        return self._statname(parts) == 'min'

    def _is_max(self, parts):
        return self._statname(parts) == 'max'

    def _is_median(self, parts):
        return self._statname(parts) == 'median'

    def _is_variance(self, parts):
        return self._statname(parts) == 'var'

    def _is_sem(self, parts):
        return self._statname(parts) == 'sem'

    def _is_varcoeff(self, parts):
        return self._statname(parts) == 'varcoeff'

    def _is_percentile(self, parts):
        return self._statname(parts) in ['upper_q', 'lower_q', 'median']

    def _is_counts_sum(self, parts):
        return parts[-1].endswith('counts_sum')

    def _is_c_pct_sum(self, parts):
        return parts[-1].endswith('c%_sum')

    def _is_counts_cumsum(self, parts):
        return parts[-1].endswith('counts_cumsum')

    def _is_c_pct_cumsum(self, parts):
        return parts[-1].endswith('c%_cumsum')

    def _is_weighted(self, parts):
        return parts[4] != ''

    def _weight(self, parts):
        if parts[4] != '':
            return parts[4]
        else:
            return None

    def _is_stat(self, parts):
        return parts[1].startswith('d.')

    def _is_propstest(self, parts):
        return parts[1].startswith('t.props')

    def _is_meanstest(self, parts):
        return parts[1].startswith('t.means')

    def _siglevel(self, parts):
        if self._is_meanstest(parts) or self._is_propstest(parts):
            return parts[1].split('.')[-1]
        else:
            return None

    def _describe_block(self, description, row_id):
        if self.painted:
            repaint = True
            self.toggle_labels()
        else:
            repaint = False
        vpr = self._views_per_rows()
        if row_id is not None:
            vpr = [v[1] for v in vpr[row_id].items()]
            idx = self.dataframe.columns.get_level_values(1).tolist()
        else:
            idx = self.dataframe.index.get_level_values(1).tolist()
        idx_view_map = zip(idx, vpr)
        block_net_vk = [v for v in vpr if len(v.split('|')[2].split('['))>2 or
                        '[+{' in v.split('|')[2] or '}+]' in v.split('|')[2]]
        has_calc = any([v.split('|')[1].startswith('f.c') for v in block_net_vk])
        is_tested = any(v.split('|')[1].startswith('t.props') for v in vpr)
        if block_net_vk:
            expr = block_net_vk[0].split('|')[2]
            expanded_codes = set(map(int, re.findall(r'\d+', expr)))
        else:
            expanded_codes = []
        for idx, m in enumerate(idx_view_map):
            if idx_view_map[idx][0] == '':
                idx_view_map[idx] = (idx_view_map[idx-1][0], idx_view_map[idx][1])
        for idx, row in enumerate(description):
            if not 'is_block' in row:
                idx_view_map[idx] = None
        blocks_len = len(expr.split('],')) * (self.ci_count + is_tested)
        if has_calc: blocks_len -= (self.ci_count + is_tested)
        block_net_def = []
        described_nets = 0
        for e in idx_view_map:
            if e:
                if isinstance(e[0], (str, unicode)):
                    if has_calc and described_nets == blocks_len:
                        block_net_def.append('calc')
                    else:
                        block_net_def.append('net')
                        described_nets += 1
                else:
                    code = int(e[0])
                    if code in expanded_codes:
                        block_net_def.append('expanded')
                    else:
                        block_net_def.append('normal')
            else:
                block_net_def.append(e)
        if repaint: self.toggle_labels()
        return block_net_def

    def get(self, data_key, filter_key, x_keys, y_keys, views, rules=False,
            rules_weight=None, orient='x', prioritize=True):
        """ Get the concatenated Chain.DataFrame
        """
        self._meta = self.stack[data_key].meta
        self._given_views = views
        self._x_keys = x_keys
        self._y_keys = y_keys

        concat_axis = 0

        if rules:
            if not isinstance(rules, list):
                self._has_rules = ['x', 'y']
            else:
                self._has_rules = rules

        # use_views = views[:]
        # for first in self.axes[0]:
        #     for second in self.axes[1]:
        #         link = self._get_link(data_key, filter_key, first, second)

        #         for v in use_views:
        #             if v not in link:
        #                 use_views.remove(v)

        for first in self.axes[0]:
            found = []
            x_frames = []

            for second in self.axes[1]:
                if self.axis == 1:
                    link = self._get_link(data_key, filter_key, first, second)
                else:
                    link = self._get_link(data_key, filter_key, second, first)

                if link is None:
                    continue
                if prioritize: link = self._drop_substituted_views(link)
                found_views, y_frames = self._concat_views(
                    link, views, rules_weight)
                found.append(found_views)

                try:
                    if self._meta['columns'][link.x].get('parent'):
                        self._is_mask_item = True
                except KeyError:
                    pass

                # TODO: contains arrary summ. attr.
                # TODO: make this work y_frames = self._pad_frames(y_frames)

                self.array_style = link
                if self.array_style > -1:
                    concat_axis = 1 if self.array_style == 0 else 0
                    y_frames = self._pad_frames(y_frames)

                x_frames.append(pd.concat(y_frames, axis=concat_axis))

                self.shapes.append(x_frames[-1].shape)

            self._frame = pd.concat(self._pad(x_frames), axis=self.axis)


            if self._group_style == 'reduced' and self.array_style >- 1:
                scan_views = [v if isinstance(v, (tuple, list)) else [v]
                              for v in self._given_views]
                scan_views = [v for v in scan_views if len(v) > 1]
                no_tests = []
                for scan_view in scan_views:
                    new_views = []
                    for view in scan_view:
                        if not view.split('|')[1].startswith('t.'):
                            new_views.append(view)
                    no_tests.append(new_views)
                cond = any(len(v) >= 2 for v in no_tests)
                if cond:
                    self._frame = self._reduce_grouped_index(self._frame, 2, self._array_style)
            if self.axis == 1:
                self.views = found[-1]
            else:
                self.views = found

            self.double_base = len([v for v in self.views
                                    if v.split('|')[-1] == 'cbase']) > 1

            self._index = self._frame.index
            self._columns = self._frame.columns
            self._extract_base_descriptions()

        del self.stack

        return self

    def _toggle_bases(self, keep_weighted=True):
        df = self._frame
        is_array = self._array_style == 0
        contents = self.contents[0] if is_array else self.contents
        has_wgt_b = [k for k, v in contents.items()
                     if v['is_c_base'] and v['is_weighted']]
        has_unwgt_b = [k for k, v in contents.items()
                       if v['is_c_base'] and not v['is_weighted']]
        if not (has_wgt_b and has_unwgt_b):
            return None

        if keep_weighted:
            drop_rows = has_unwgt_b
            names = ['x|f|x:|||cbase']
        else:
            drop_rows = has_wgt_b
            names = ['x|f|x:||{}|cbase'.format(contents.values()[0]['weight'])]

        for v in self.views.copy():
            if v in names:
                del self._views[v]

        df = self._frame

        if is_array:
            cols = [col for x, col in enumerate(df.columns.tolist())
                    if not x in drop_rows]
            df = df.loc[:, cols]
        else:
            rows = [row for x, row in enumerate(df.index.tolist())
                    if not x in drop_rows]
            df = df.loc[rows, :]

        self._frame = df
        self._index = df.index
        self._columns = df.columns
        return None

    def _slice_edited_index(self, axis, positions):
        """
        """
        l_zero = axis.get_level_values(0).values.tolist()[0]
        l_one = axis.get_level_values(1).values.tolist()
        l_one = [l_one[p] for p in positions]
        axis_tuples = [(l_zero, lab) for lab in l_one]
        if self.array_style == 0:
            names = ['Array', 'Questions']
        else:
            names = ['Question', 'Values']
        return pd.MultiIndex.from_tuples(axis_tuples, names=names)

    def _drop_substituted_views(self, link):
        if any(isinstance(sect, (list, tuple)) for sect in self._given_views):
            chain_views = list(chain.from_iterable(self._given_views))
        else:
            chain_views = self._given_views
        has_compl = any(']*:' in vk for vk in link)
        req_compl = any(']*:' in vk for vk in chain_views)
        has_cumsum = any('++' in vk for vk in link)
        req_cumsum = any('++' in vk for vk in chain_views)
        if (has_compl and req_compl) or (has_cumsum and req_cumsum):
            new_link = copy.copy(link)
            views = []
            for vk in link:
                vksplit = vk.split('|')
                method, cond, name = vksplit[1], vksplit[2], vksplit[-1]
                full_frame = name in ['counts', 'c%']
                basic_sigtest = method.startswith('t.') and cond == ':'
                if not full_frame and not basic_sigtest: views.append(vk)
            for vk in link:
                if vk not in views: del new_link[vk]
            return new_link
        else:
            return link

    def _pad_frames(self, frames):
        """ TODO: doc string
        """
        empty_frame = lambda f: pd.DataFrame(index=f.index, columns=f.columns)

        max_lab = max(f.axes[self.array_style].size for f in frames)

        for e, f in enumerate(frames):
            size = f.axes[self.array_style].size
            if size < max_lab:
                f = pd.concat([f, empty_frame(f)], axis=self.array_style)
                order = [None] * (size * 2)
                order[::2] = list(xrange(size))
                order[1::2] = list(xrange(size, size * 2))
                if self.array_style == 0:
                    frames[e] = f.iloc[order, :]
                else:
                    frames[e] = f.iloc[:, order]

        return frames

    def _get_link(self, data_key, filter_key, x_key, y_key):
        """
        """
        base = self.stack[data_key][filter_key]
        if x_key in base:
            base = base[x_key]
            if y_key in base:
                return base[y_key]
            else:
                if self._array_style == -1:
                    self._y_keys.remove(y_key)
        else:
            self._x_keys.remove(x_key)
        return None

    def _index_switch(self, axis):
        """ Returns self.dataframe/frame index/ columns based on given x/ y
        """
        return dict(x=self._frame.index, y=self._frame.columns).get(axis)

    def _pad(self, frames):
        """ Pad index/ columns when nlevels is less than the max nlevels
        in list of dataframes.
        """
        indexes = []

        max_nlevels = [max(f.axes[i].nlevels for f in frames) for i in (0, 1)]

        for e, f in enumerate(frames):
            indexes = []
            for i in (0, 1):
                if f.axes[i].nlevels < max_nlevels[i]:
                    indexes.append(self._pad_index(f.axes[i], max_nlevels[i]))
                else:
                    indexes.append(f.axes[i])
            frames[e].index, frames[e].columns = indexes

        return frames

    def _pad_index(self, index, size):
        """ Add levels to columns MultiIndex so the nlevels matches
        the biggest columns MultiIndex in DataFrames to be concatenated.
        """
        pid = self.pad_id

        pad = ((size - index.nlevels) // 2)
        fill = int((pad % 2) == 1)

        names = list(index.names)
        names[0:0] = names[:2] * pad

        arrays = self._lzip(index.values)
        arrays[0:0] = [tuple('#pad-%s' % pid for _ in arrays[i])
                       for i in xrange(pad + fill)] * pad

        return pd.MultiIndex.from_arrays(arrays, names=names)

    @staticmethod
    def _reindx_source(df, varname, total):
        """
        """
        df.index = df.index.set_levels([varname], level=0, inplace=False)
        if df.columns.get_level_values(0).tolist()[0] != varname and total:
            df.columns = df.columns.set_levels([varname], level=0, inplace=False)
        return df

    def _concat_views(self, link, views, rules_weight, found=None):
        """ Concatenates the Views of a Chain.
        """
        frames = []

        totals = [[_TOTAL]] * 2

        if found is None:
            found = OrderedDict()

        if self._text_map is None:
            self._text_map = dict()

        for view in views:
            try:
                self.array_style = link

                if isinstance(view, (list, tuple)):

                    if not self.grouping:
                        self.grouping = True
                        if isinstance(view, tuple):
                            self._group_style = 'reduced'
                        else:
                            self._group_style = 'normal'

                    if self.array_style > -1:
                        use_grp_type = 'normal'
                    else:
                        use_grp_type = self._group_style

                    found, grouped = self._concat_views(link, view, rules_weight, found=found)
                    if grouped:
                        frames.append(self._group_views(grouped, use_grp_type))
                else:
                    agg = link[view].meta()['agg']
                    is_descriptive = agg['method'] == 'descriptives'
                    is_base = agg['name'] in ['cbase', 'rbase', 'ebase', 'cbase_gross']
                    is_sum = agg['name'] in ['counts_sum', 'c%_sum']
                    is_net = link[view].is_net()
                    oth_src = link[view].has_other_source()
                    no_total_sign = is_descriptive or is_base or is_sum or is_net

                    if link[view]._custom_txt and is_descriptive:
                        statname = agg['fullname'].split('|')[1].split('.')[1]
                        if not statname in self._custom_texts:
                            self._custom_texts[statname] = []
                        self._custom_texts[statname].append(link[view]._custom_txt)

                    if is_descriptive:
                        text = agg['name']
                        try:
                            self._text_map.update({agg['name']: text})
                        except AttributeError:
                            self._text_map = {agg['name']: text}
                    if agg['text']:
                        name = dict(cbase='All').get(agg['name'], agg['name'])
                        try:
                            self._text_map.update({name: agg['text']})
                        except AttributeError:
                            self._text_map = {name: agg['text'],
                                              _TOTAL: 'Total'}
                    if agg['grp_text_map']:
                        # try:
                        if not agg['grp_text_map'] in self._grp_text_map:
                            self._grp_text_map.append(agg['grp_text_map'])
                        # except AttributeError:
                        #     self._grp_text_map = [agg['grp_text_map']]

                    frame = link[view].dataframe
                    if oth_src:
                        frame = self._reindx_source(frame, link.x, link.y == _TOTAL)

                    # RULES SECTION
                    # ========================================================
                    # TODO: DYNAMIC RULES:
                    #   - all_rules_axes, rules_weight must be provided not hardcoded
                    #   - Review copy/pickle in original version!!!

                    rules_weight = None
                    if self._has_rules:
                        rules = Rules(link, view, self._has_rules, rules_weight)
                        # print rules.show_rules()
                        # rules.get_slicer()
                        # print rules.show_slicers()
                        rules.apply()
                        frame = rules.rules_df()
                    # ========================================================
                    if not no_total_sign and (link.x == _TOTAL or link.y == _TOTAL):
                        if link.x == _TOTAL:
                            level_names = [[link.y], ['@']]
                        elif link.y == _TOTAL:
                            level_names = [[link.x], ['@']]
                        try:
                            frame.columns.set_levels(level_names, level=[0, 1],
                                                     inplace=True)
                        except ValueError:
                            pass
                    frames.append(frame)
                    if view not in found:
                        if self._array_style != 0:
                            found[view] = len(frame.index)
                        else:
                            found[view] = len(frame.columns)

                if link[view]._kwargs.get('flag_bases'):
                    flag_bases = link[view]._kwargs['flag_bases']
                    try:
                        if flag_bases not in self._flag_bases:
                            self._flag_bases.append(flag_bases)
                    except TypeError:
                        self._flag_bases = [flag_bases]

            except KeyError:
                pass
        return found, frames

    @staticmethod
    def _temp_nest_index(df):
        """
        Flatten the nested MultiIndex for easier handling.
        """
        # Build flat column labels
        flat_cols = []
        order_idx = []
        i = -1
        for col in df.columns.values:
            flat_col_lab = ''.join(str(col[:-1])).strip()
            if not flat_col_lab in flat_cols:
                i += 1
                order_idx.append(i)
                flat_cols.append(flat_col_lab)
            else:
                order_idx.append(i)
        # Drop unwanted levels (keep last Values Index-level in that process)
        levels = list(range(0, df.columns.nlevels-1))
        drop_levels = levels[:-2]+ [levels[-1]]
        df.columns = df.columns.droplevel(drop_levels)
        # Apply the new flat labels and resort the columns
        df.columns.set_levels(levels=flat_cols, level=0, inplace=True)
        df.columns.set_labels(order_idx, level=0, inplace=True)
        return df, flat_cols

    @staticmethod
    def _replace_test_results(df, replacement_map, char_repr):
        """
        Swap all digit-based results with letters referencing the column header.

        .. note:: The modified df will be stripped of all indexing on both rows
        and columns.
        """
        all_dfs  = []
        ignore = False
        for col in replacement_map.keys():
            target_col = df.columns[0] if col == '@' else col
            value_df = df[[target_col]].copy()
            if not col == '@':
                value_df.drop('@', axis=1, level=1, inplace=True)
            values = value_df.replace(np.NaN, '-').values.tolist()
            r = replacement_map[col]
            new_values = []
            case = None
            for v in values:
                if isinstance(v[0], (str, unicode)):
                    if char_repr == 'upper':
                        case = 'up'
                    elif char_repr == 'lower':
                        case = 'low'
                    elif char_repr == 'alternate':
                        if case == 'up':
                            case = 'low'
                        else:
                            case = 'up'
                    for no, l in sorted(r.items(), reverse=True):
                        v = [char.replace(str(no), l if case == 'up' else l.lower())
                             if isinstance(char, (str, unicode))
                             else char for char in v]

                    new_values.append(v)
                else:
                    new_values.append(v)
            part_df = pd.DataFrame(new_values)
            all_dfs.append(part_df)
        letter_df = pd.concat(all_dfs, axis=1)
        # Clean it up
        letter_df.replace('-', np.NaN, inplace=True)
        for signs in [('[', ''), (']', ''), (', ', '.')]:
            letter_df = letter_df.applymap(lambda x: x.replace(signs[0], signs[1])
                                           if isinstance(x, (str, unicode)) else x)
        return letter_df

    @staticmethod
    def _get_abc_letters(no_of_cols, incl_total):
        """
        Get the list of letter replacements depending on the y-axis length.
        """
        repeat_alphabet = int(no_of_cols / 26)
        abc = list(string.ascii_uppercase)
        letters = list(string.ascii_uppercase)
        if repeat_alphabet:
            for r in range(0, repeat_alphabet):
                letter = abc[r]
                extend_abc = ['{}{}'.format(letter, l) for l in abc]
                letters.extend(extend_abc)
        if incl_total:
            letters = ['@'] + letters[:no_of_cols-1]
        else:
            letters = letters[:no_of_cols]
        return letters

    def _any_tests(self):
        vms = [v.split('|')[1] for v in self._views.keys()]
        return any('t.' in v for v in vms)

    def _no_of_tests(self):
        tests = [v for v in self._views.keys()
                 if v.split('|')[1].startswith('t.')]
        levels = [v.split('|')[1].split('.')[-1] for v in tests]
        return len(set(levels))

    def _siglevel_on_row(self):
        """
        """
        vpr = self._views_per_rows()
        tests = [(no, v) for no, v in enumerate(vpr)
                 if v.split('|')[1].startswith('t.')]
        s = [(t[0],
              float(int(t[1].split('|')[1].split('.')[3].split('+')[0]))/100.0)
             for t in tests]
        return s

    def transform_tests(self, char_repr='upper', display_level=True):
        """
        Transform column-wise digit-based test representation to letters.

        Adds a new row that is applying uppercase letters to all columns (A,
        B, C, ...) and maps any significance test's result cells to these column
        indicators.

        """
        if not self._any_tests(): return None
        # Preparation of input dataframe and dimensions of y-axis header
        df = self.dataframe.copy()
        number_codes = df.columns.get_level_values(-1).tolist()
        number_header_row = copy.copy(df.columns)
        if self._no_of_tests() != 2 and char_repr == 'alternate':
            char_repr = 'upper'
        has_total = '@' in self._y_keys
        if self._nested_y:
            df, questions = self._temp_nest_index(df)
        else:
            questions = self._y_keys
        all_num = number_codes if not has_total else [0] + number_codes[1:]
        # Set the new column header (ABC, ...)

        column_letters = self._get_abc_letters(len(number_codes), has_total)
        vals = df.columns.get_level_values(0).tolist()
        mi = pd.MultiIndex.from_arrays(
            (vals,
             column_letters))
        df.columns = mi
        self.sig_test_letters = df.columns.get_level_values(1).tolist()
        # Build the replacements dict and build list of unique column indices
        test_dict = OrderedDict()
        for num_idx, col in enumerate(df.columns):
            if col[1] == '@':
                question = col[1]
            else:
                question = col[0]
            if not question in test_dict: test_dict[question] = {}
            number = all_num[num_idx]
            letter = col[1]
            test_dict[question][number] = letter
        letter_df = self._replace_test_results(df, test_dict, char_repr)
        # Re-apply indexing & finalize the new crossbreak column header
        if display_level:
            levels = self._siglevel_on_row()
            index = df.index.get_level_values(1).tolist()
            for i, l in levels:
                index[i] = '#Level: {}'.format(l)
            l0 = df.index.get_level_values(0).tolist()[0]
            tuples = [(l0, i) for i in index]
            index = pd.MultiIndex.from_tuples(
                tuples, names=['Question', 'Values'])
            letter_df.index = index
        else:
            letter_df.index = df.index
        letter_df.columns = number_header_row
        letter_df = self._apply_letter_header(letter_df)
        self._frame = letter_df
        return self

    def _remove_letter_header(self):
        self._frame.columns = self._frame.columns.droplevel(level=-1)
        return None

    def _apply_letter_header(self, df):
        """
        """
        new_tuples = []
        org_names = [n for n in df.columns.names]
        idx = df.columns
        for i, l in zip(idx, self.sig_test_letters):
            new_tuples.append(i + (l, ))
        if not 'Test-IDs' in org_names:
            org_names.append('Test-IDs')
        mi = pd.MultiIndex.from_tuples(new_tuples, names=org_names)
        df.columns = mi
        return df

    def _extract_base_descriptions(self):
        """
        """
        if self.source == 'Crunch multitable':
            self.base_descriptions = self._meta['var_meta'].get('notes', None)
        else:
            base_texts = OrderedDict()
            arr_style = self.array_style
            if arr_style != -1:
                var = self._x_keys[0] if arr_style == 0 else self._y_keys[0]
                masks = self._meta['masks']
                columns = self._meta['columns']
                item = masks[var]['items'][0]['source'].split('@')[-1]
                test_item = columns[item]
                test_mask = masks[var]
                if 'properties' in test_mask:
                    base_text = test_mask['properties'].get('base_text', None)
                elif 'properties' in test_item:
                        base_text = test_item['properties'].get('base_text', None)
                else:
                    base_text = None
                self.base_descriptions = base_text
            else:
                for x in self._x_keys:
                    if 'properties' in self._meta['columns'][x]:
                        bt = self._meta['columns'][x]['properties'].get('base_text', None)
                        if bt:
                            base_texts[x] = bt
                if base_texts:
                    if self.orientation == 'x':
                        self.base_descriptions = base_texts.values()[0]
                    else:
                        self.base_descriptions = base_texts.values()

        return None

    def _ensure_indexes(self):
        if self.painted:
            self._frame.index, self._frame.columns = self.index, self.columns
            if self.structure is not None:
                self._frame.loc[:, :] = self.frame_values
        else:
            self.index, self.columns = self._frame.index, self._frame.columns
            if self.structure is not None:
                self.frame_values = self._frame.values

    def _finish_text_key(self, text_key, text_loc_x, text_loc_y):
        text_keys = dict()
        text_key = text_key or self._default_text

        if text_loc_x:
            text_keys['x'] = (text_loc_x, text_key)
        else:
            text_keys['x'] = text_key

        if text_loc_y:
            text_keys['y'] = (text_loc_y, text_key)
        else:
            text_keys['y'] = text_key

        return text_keys

    def paint(self, text_key=None, text_loc_x=None, text_loc_y=None, display=None,
              axes=None, view_level=False, transform_tests='upper', display_level=True,
              add_test_ids=True, add_base_texts='simple', totalize=False,
              sep=None, na_rep=None, transform_column_names=None,
              exclude_mask_text=False):
        """
        Apply labels, sig. testing conversion and other post-processing to the
        ``Chain.dataframe`` property.

        Use this to prepare a ``Chain`` for further usage in an Excel or Power-
        point Build.

        Parameters
        ----------
        text_keys : str, default None
            Text
        text_loc_x : str, default None
            The key in the 'text' to locate the text_key for the x-axis
        text_loc_y : str, default None
            The key in the 'text' to locate the text_key for the y-axis
        display : {'x', 'y', ['x', 'y']}, default None
            Text
        axes : {'x', 'y', ['x', 'y']}, default None
            Text
        view_level : bool, default False
            Text
        transform_tests : {False, 'upper', 'lower', 'alternate'}, default 'upper'
            Text
        add_test_ids : bool, default True
            Text
        add_base_texts : {False, 'all', 'simple', 'simple-no-items'}, default 'simple'
            Whether or not to include existing ``.base_descriptions`` str
            to the label of the appropriate base view. Selecting ``'simple'``
            will inject the base texts to non-array type Chains only.
        totalize : bool, default True
            Text
        sep : str, default None
            The seperator used for painting ``pandas.DataFrame`` columns
        na_rep : str, default None
            numpy.NaN will be replaced with na_rep if passed
        transform_column_names : dict, default None
            Transformed column_names are added to the labeltexts.
        exclude_mask_text : bool, default False
            Exclude mask text from mask-item texts.
        Returns
        -------
        None
            The ``.dataframe`` is modified inplace.
        """
        self._ensure_indexes()
        text_keys = self._finish_text_key(text_key, text_loc_x, text_loc_y)
        if self.structure is not None:
            self._paint_structure(text_key, sep=sep, na_rep=na_rep)
        else:
            self.totalize = totalize
            if transform_tests: self.transform_tests(transform_tests, display_level)
            # Remove any letter header row from transformed tests...
            if self.sig_test_letters:
                self._remove_letter_header()
            if display is None:
                display = _AXES
            if axes is None:
                axes = _AXES
            self._paint(text_keys, display, axes, add_base_texts,
                        transform_column_names, exclude_mask_text)
            # Re-build the full column index (labels + letter row)
            if self.sig_test_letters and add_test_ids:
                self._frame = self._apply_letter_header(self._frame)
            if view_level:
                self._add_view_level()
        self.painted = True
        return None

    def _paint_structure(self, text_key=None, sep=None, na_rep=None):
        """ Paint the dataframe-type Chain.
        """
        if not text_key:
            text_key = self._meta['lib']['default text']
        str_format = '%%s%s%%s' % sep

        column_mapper = dict()

        na_rep = na_rep or ''

        pattern = r'\, (?=\W|$)'

        for column in self.structure.columns:
            if not column in self._meta['columns']: continue

            meta = self._meta['columns'][column]

            if sep:
                column_mapper[column] = str_format % (column, meta['text'][text_key])
            else:
                column_mapper[column] = meta['text'][text_key]

            if meta.get('values'):
                values = meta['values']
                if isinstance(values, basestring):
                    pointers = values.split('@')
                    values = self._meta[pointers.pop(0)]
                    while pointers:
                        values = values[pointers.pop(0)]
                if meta['type'] == 'delimited set':
                    value_mapper = {
                        str(item['value']): item['text'][text_key]
                        for item in values
                    }
                    series = self.structure[column]
                    try:
                        series = (series.str.split(';')
                                        .apply(pd.Series, 1)
                                        .stack(dropna=False)
                                        .map(value_mapper.get) #, na_action='ignore')
                                        .unstack())
                        first = series[series.columns[0]]
                        rest = [series[c] for c in series.columns[1:]]
                        self.structure[column] = (
                            first
                                .str.cat(rest, sep=', ', na_rep='')
                                .str.slice(0, -2)
                                .replace(to_replace=pattern, value='', regex=True)
                                .replace(to_replace='', value=na_rep)
                        )
                    except AttributeError:
                        continue
                else:
                    value_mapper = {
                        item['value']: item['text'][text_key]
                        for item in values
                    }
                    self.structure[column] = (self.structure[column]
                                                  .map(value_mapper.get,
                                                       na_action='ignore')
                                             )

            self.structure[column].fillna(na_rep, inplace=True)

        self.structure.rename(columns=column_mapper, inplace=True)

    def _paint(self, text_keys, display, axes, bases, transform_column_names,
               exclude_mask_text):
        """ Paint the Chain.dataframe
        """
        indexes = []

        for axis in _AXES:
            index = self._index_switch(axis)
            if axis in axes:
                index = self._paint_index(index, text_keys, display, axis,
                                          bases, transform_column_names,
                                          exclude_mask_text)
            indexes.append(index)

        self._frame.index, self._frame.columns = indexes

    def _paint_index(self, index, text_keys, display, axis, bases,
                     transform_column_names, exclude_mask_text):
        """ Paint the Chain.dataframe.index1        """
        error = "No text keys from {} found in {}"
        level_0_text, level_1_text = [], []
        nlevels = index.nlevels

        if nlevels > 2:
            arrays = []

            for i in xrange(0, nlevels, 2):
                index_0 = index.get_level_values(i)
                index_1 = index.get_level_values(i+1)
                tuples = zip(index_0.values, index_1.values)
                names = (index_0.name, index_1.name)
                sub = pd.MultiIndex.from_tuples(tuples, names=names)
                sub = self._paint_index(sub, text_keys, display, axis, bases,
                                        transform_column_names, exclude_mask_text)
                arrays.extend(self._lzip(sub.ravel()))

            tuples = self._lzip(arrays)
            return pd.MultiIndex.from_tuples(tuples, names=index.names)

        levels = self._lzip(index.values)

        arrays = (self._get_level_0(levels[0], text_keys, display, axis,
                                    transform_column_names, exclude_mask_text),
                  self._get_level_1(levels, text_keys, display, axis, bases))

        new_index = pd.MultiIndex.from_arrays(arrays, names=index.names)

        return new_index

    def _get_level_0(self, level, text_keys, display, axis,
                     transform_column_names, exclude_mask_text):
        """
        """
        level_0_text = []

        for value in level:
            if str(value).startswith('#pad'):
                pass
            elif pd.notnull(value):
                if value in self._text_map.keys():
                    value = self._text_map[value]
                else:
                    text = self._get_text(value, text_keys[axis], exclude_mask_text)
                    if axis in display:
                        if transform_column_names:
                            value = transform_column_names.get(value, value)
                        value = '{}. {}'.format(value, text)
                    else:
                        value = text
            level_0_text.append(value)
        if '@' in self._y_keys and self.totalize and axis == 'y':
            level_0_text = ['Total'] + level_0_text[1:]
        return map(unicode, level_0_text)

    def _get_level_1(self, levels, text_keys, display, axis, bases):
        """
        """
        level_1_text = []
        if text_keys[axis] in self._transl:
            tk_transl = text_keys[axis]
        else:
            tk_transl = self._default_text
        c_text = copy.deepcopy(self._custom_texts) if self._custom_texts else {}
        for i, value in enumerate(levels[1]):
            if str(value).startswith('#pad'):
                level_1_text.append(value)
            elif pd.isnull(value):
                level_1_text.append(value)
            elif str(value) == '':
                level_1_text.append(value)
            elif str(value).startswith('#Level: '):
                level_1_text.append(value.replace('#Level: ', ''))
            else:
                translate = self._transl[self._transl.keys()[0]].keys()
                if value in self._text_map.keys() and value not in translate:
                    level_1_text.append(self._text_map[value])
                elif value in translate:
                    if value == 'All':
                        text = self._specify_base(i, text_keys[axis], bases)
                    else:
                        text = self._transl[tk_transl][value]
                        if value in c_text:
                            add_text = c_text[value].pop(0)
                            text = '{} {}'.format(text, add_text)
                    level_1_text.append(text)
                elif value == 'All (eff.)':
                    text = self._specify_base(i, text_keys[axis], bases)
                    level_1_text.append(text)
                else:
                    if any(self.array_style == a and axis == x for a, x in ((0, 'x'), (1, 'y'))):
                        text = self._get_text(value, text_keys[axis], True)
                        level_1_text.append(text)
                    else:
                        try:
                            values = self._get_values(levels[0][i])
                            if not values:
                                level_1_text.append(value)
                            else:
                                for item in self._get_values(levels[0][i]):
                                    if int(value) == item['value']:
                                        text = self._get_text(item, text_keys[axis])
                                        level_1_text.append(text)
                        except (ValueError, UnboundLocalError):
                            if self._grp_text_map:
                                for gtm in self._grp_text_map:
                                    if value in gtm.keys():
                                        text = self._get_text(gtm[value], text_keys[axis])
                                        level_1_text.append(text)
        return map(unicode, level_1_text)

    @staticmethod
    def _unwgt_label(views, base_vk):
        valid = ['cbase', 'cbase_gross', 'rbase', 'ebase']
        basetype = base_vk.split('|')[-1]
        views_split = [v.split('|') for v in views]
        multibase = len([v for v in views_split if v[-1] == basetype]) > 1
        weighted = base_vk.split('|')[-2]
        w_diff = len([v for v in views_split
                      if not v[-1] in valid and not v[-2] == weighted]) > 0
        if weighted:
            return False
        elif multibase or w_diff:
            return True
        else:
            return False

    def _add_base_text(self, base_val, tk, bases):
        if self._array_style == 0 and bases != 'all':
            return base_val
        else:
            bt = self.base_descriptions
            if isinstance(bt, dict):
                bt_by_key = bt[tk]
            else:
                bt_by_key = bt
            if bt_by_key:
                if bt_by_key.startswith('%s:' % base_val):
                    bt_by_key = bt_by_key.replace('%s:' % base_val, '')
                return '{}: {}'.format(base_val, bt_by_key)
            else:
                return base_val

    def _specify_base(self, view_idx, tk, bases):
        tk_transl = tk if tk in self._transl else self._default_text
        base_vk = self._valid_views()[view_idx]
        basetype = base_vk.split('|')[-1]
        unwgt_label = self._unwgt_label(self._views.keys(), base_vk)

        if unwgt_label:
            if basetype == 'cbase_gross':
                base_value = self._transl[tk_transl]['no_w_gross_All']
            elif basetype == 'ebase':
                base_value = 'Unweighted effective base'
            else:
                base_value = self._transl[tk_transl]['no_w_All']
        else:
            if basetype == 'cbase_gross':
                base_value = self._transl[tk_transl]['gross All']
            elif basetype == 'ebase':
                base_value = 'Effective base'
            elif not bases or (bases == 'simple-no-items' and self._is_mask_item):
                base_value = self._transl[tk_transl]['All']
            else:
                key = tk
                if isinstance(tk, tuple):
                    _, key = tk
                base_value = self._add_base_text(self._transl[tk_transl]['All'],
                                                 key, bases)
        return base_value

    def _get_text(self, value, text_key, item_text=False):
        """
        """
        if value in self._meta['columns'].keys():
            col = self._meta['columns'][value]
            if item_text and col.get('parent'):
                parent = col['parent'].keys()[0].split('@')[-1]
                items = self._meta['masks'][parent]['items']
                for i in items:
                    if i['source'].split('@')[-1] == value:
                        obj = i['text']
                        break
            else:
                obj = col['text']
        elif value in self._meta['masks'].keys():
            obj = self._meta['masks'][value]['text']
        elif 'text' in value:
            obj = value['text']
        else:
            obj = value
        return self._get_text_from_key(obj, text_key)

    def _get_text_from_key(self, text, text_key):
        """ Find the first value in a meta object's "text" key that matches a
        text_key for it's axis.
        """
        if isinstance(text_key, tuple):
            loc, key = text_key
            if loc in text:
                if key in text[loc]:
                    return text[loc][key]
                elif self._default_text in text[loc]:
                    return text[loc][self._default_text]
            if key in text:
                return text[key]
        for key in (text_key, self._default_text):
            if key in text:
                return text[key]
        return '<label>'

    def _get_values(self, column):
        """ Returns values from self._meta["columns"] or
        self._meta["lib"]["values"][<mask name>] if parent is "array"
        """
        if column in self._meta['columns']:
            values = self._meta['columns'][column].get('values', [])
        elif column in self._meta['masks']:
            values = self._meta['lib']['values'].get(column, [])
        if isinstance(values, (str, unicode)):
            keys = values.split('@')
            values = self._meta[keys.pop(0)]
            while keys:
                values = values[keys.pop(0)]
        return values

    def _add_view_level(self, shorten=False):
        """ Insert a third Index level containing View keys into the DataFrame.
        """
        vnames = self._views_per_rows()
        if shorten:
            vnames = [v.split('|')[-1] for v in vnames]
        self._frame['View'] = pd.Series(vnames, index=self._frame.index)
        self._frame.set_index('View', append=True, inplace=True)

    def toggle_labels(self):
        """ Restore the unpainted/ painted Index, Columns appearance.
        """
        if self.painted:
            self.painted = False
        else:
            self.painted = True

        attrs = ['index', 'columns']
        if self.structure is not None:
            attrs.append('_frame_values')
        for attr in attrs:
            vals = attr[6:] if attr.startswith('_frame') else attr
            frame_val = getattr(self._frame, vals)
            setattr(self._frame, attr, getattr(self, attr))
            setattr(self, attr, frame_val)

        if self.structure is not None:
            values = self._frame.values
            self._frame.loc[:, :] = self.frame_values
            self.frame_values = values

        return self

    @staticmethod
    def _single_column(*levels):
        """ Returns True if multiindex level 0 has one unique value
        """
        return all(len(level) == 1 for level in levels)

    def _group_views(self, frame, group_type):
        """ Re-sort rows so that they appear as being grouped inside the
        Chain.dataframe.
        """
        grouped_frame = []
        len_of_frame = len(frame)
        frame = pd.concat(frame, axis=0)
        index_order = frame.index.get_level_values(1).tolist()
        index_order =  index_order[:(len(index_order) / len_of_frame)]
        gb_df = frame.groupby(level=1, sort=False)
        for i in index_order:
            grouped_df = gb_df.get_group(i)
            if group_type == 'reduced':
                grouped_df = self._reduce_grouped_index(grouped_df, len_of_frame-1)
            grouped_frame.append(grouped_df)
        grouped_frame = pd.concat(grouped_frame, verify_integrity=False)
        return grouped_frame

    @staticmethod
    def _reduce_grouped_index(grouped_df, view_padding, array_summary=-1):
        idx = grouped_df.index
        q = idx.get_level_values(0).tolist()[0]
        if array_summary == 0:
            val = idx.get_level_values(1).tolist()
            for index in range(1, len(val), 2):
                val[index] = ''
            grp_vals = val
        elif array_summary == 1:
            grp_vals = []
            indexed = []
            val = idx.get_level_values(1).tolist()
            for v in val:
                if not v in indexed or v == 'All':
                    grp_vals.append(v)
                    indexed.append(v)
                else:
                    grp_vals.append('')
        else:
            val = idx.get_level_values(1).tolist()[0]
            grp_vals = [val] + [''] * view_padding
        mi = pd.MultiIndex.from_product([[q], grp_vals], names=idx.names)
        grouped_df.index = mi
        return grouped_df


    @staticmethod
    def _lzip(arr):
        """
        """
        return list(zip(*arr))

    @staticmethod
    def _force_list(obj):
        if isinstance(obj, (list, tuple)):
            return obj
        return [obj]

    @classmethod
    def __pad_id(cls):
        cls._pad_id += 1
        return cls._pad_id
