
import numpy as np
import pandas as pd
import quantipy as qp


# from matplotlib import pyplot as plt
# import matplotlib.image as mpimg

import string
import cPickle
import warnings

try:
    import seaborn as sns
    from PIL import Image
except:
    pass

from quantipy.core.cache import Cache
from quantipy.core.view import View
from quantipy.core.view_generators.view_mapper import ViewMapper
from quantipy.core.view_generators.view_maps import QuantipyViews
from quantipy.core.helpers.functions import emulate_meta
from quantipy.core.tools.view.logic import (
    has_any, has_all, has_count,
    not_any, not_all, not_count,
    is_lt, is_ne, is_gt,
    is_le, is_eq, is_ge,
    union, intersection, get_logic_index)
from quantipy.core.helpers.functions import (paint_dataframe,
                                             emulate_meta,
                                             get_text,
                                             finish_text_key)
from quantipy.core.tools.dp.prep import recode

from operator import add, sub, mul, div
from scipy.stats.stats import _ttest_finish as get_pval
from scipy.stats import chi2 as chi2dist
from scipy.stats import f as fdist
from itertools import combinations, chain, product
from collections import defaultdict, OrderedDict
import gzip

try:
    import dill
except:
    pass

import json
import copy
import time
import sys



from quantipy.core.rules import Rules


_TOTAL = '@'
_AXES = ['x', 'y']


def lazy_property(func):
    """ Decorator that makes a property lazy-evaluated, i.e. only set
    on first access.
    """
    attr_name = '_%s' % func.__name__
    docstring = func.__doc__

    @property
    def _lazy_property(self):
        try:
            return getattr(self, attr_name)
        except AttributeError:
            value = func(self)
            setattr(self, attr_name, value)
            return value

    return _lazy_property

class Chain(object):

    def __init__(self, stack, name=None):
        self.stack = stack
        self.name = name

        self._given_views = None
        self._grp_text_map = []
        self._text_map = None
        self._transl = qp.core.view.View._metric_name_map()
        self._pad_id = None
        self._frame = None
        self._meta = None
        self._has_rules = None
        self.grouping = None
        self._group_style = None


    def __str__(self):
        # TODO: Add checks on x/ y/ view/ orientation
        str_format = ('%s...'
                      '\nName:            %s'
                      '\nOrientation:     %s'
                      '\nX:               %s'
                      '\nY:               %s'
                      '\nNumber of views: %s')

        return str_format % (self.__class__.__name__,
                             getattr(self, 'name', 'None'),
                             getattr(self, 'orientation', 'None'),
                             getattr(self, 'x_keys', 'None'),
                             getattr(self, 'y_keys', 'None'),
                             getattr(self, 'views', 'None'))

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        """Returns the total number of cells in the Chain.dataframe"""
        return (len(getattr(self, 'index', [])) *
                len(getattr(self, 'columns', [])))

    @lazy_property
    def orientation(self):
        """ TODO: doc string
        """
        if len(self.x_keys) == 1 and len(self.y_keys) == 1:
            return 'x'
        elif len(self.x_keys) == 1:
            return 'x'
        elif len(self.y_keys) == 1:
            return 'y'
        if len(self.x_keys) > 1 and len(self.y_keys) > 1:
            return None

    @lazy_property
    def axis(self):
        # TODO: name appropriate?
        return int(self.orientation=='x')

    @lazy_property
    def axes(self):
        # TODO: name appropriate?
        if self.axis == 1:
            return self.x_keys, self.y_keys
        return self.y_keys, self.x_keys

    @property
    def dataframe(self):
        return self._frame

    @property
    def x_keys(self):
        return self._x_keys

    @x_keys.setter
    def x_keys(self, x_keys):
        self._x_keys = self._check_keys(x_keys)

    @property
    def y_keys(self):
        return self._y_keys

    @y_keys.setter
    def y_keys(self, y_keys):
        self._y_keys = self._check_keys(y_keys)

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, index):
        self._index = index

    @property
    def columns(self):
        return self._columns

    @columns.setter
    def columns(self, columns):
        self._columns = columns

    @property
    def views(self):
        return self._views

    @views.setter
    def views(self, views):
        self._views = views

    @property
    def array_style(self):
        return self._array_style

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
    def contents(self):
        contents = dict(views=dict(), rows=dict())

        for view in self._views_per_rows:
            parts = view.split('|')
            contents['views'][view] = dict(is_counts=self._is_counts(parts),
                                           is_c_base=self._is_c_base(parts),
                                           is_r_base=self._is_r_base(parts),
                                           is_c_pct=self._is_c_pct(parts),
                                           is_r_pct=self._is_r_pct(parts),
                                           is_weighted=self._is_weighted(parts),
                                           weight=self._weight(parts),
                                           is_stat=self._is_stat(parts),
                                           stat=self._stat(parts)
                                           )

        for row, idx in enumerate(self._views_per_rows):
            parts = idx.split('|')
            contents['rows'][row] = dict(is_counts=self._is_counts(parts),
                                         is_c_base=self._is_c_base(parts),
                                         is_r_base=self._is_r_base(parts),
                                         is_c_pct=self._is_c_pct(parts),
                                         is_r_pct=self._is_r_pct(parts),
                                         is_weighted=self._is_weighted(parts),
                                         weight=self._weight(parts),
                                         is_stat=self._is_stat(parts),
                                         stat=self._stat(parts)
                                         )

        return contents

    @lazy_property
    def _views_per_rows(self):
        """
        """
        metrics = []
        if self.orientation == 'x':
            for view in self._given_views:
                view = self._force_list(view)
                initial = view[0]
                if initial in self.views:
                    size = self.views[initial]
                    metrics.extend(view * size)
        else:
            for view_part in self.views:
                for view in self._given_views:
                    view = self._force_list(view)
                    initial = view[0]
                    if initial in view_part:
                        size = view_part[initial]
                        metrics.extend(view * size)
        return metrics

    def _is_counts(self, parts):
        return parts[1].startswith('f') and parts[3] == ''

    def _is_c_base(self, parts):
        return parts[-1] == 'cbase'

    def _is_r_base(self, parts):
        return parts[-1] == 'rbase'

    def _is_c_pct(self, parts):
        return parts[1].startswith('f') and parts[3] == 'y'

    def _is_r_pct(self, parts):
        return parts[1].startswith('f') and parts[3] == 'x'

    def _is_weighted(self, parts):
        return parts[4] != ''

    def _weight(self, parts):
        if parts[4] != '':
            return parts[4]
        else:
            return None

    def _is_stat(self, parts):
        return parts[1].startswith('d.')

    def _stat(self, parts):
        if parts[1].startswith('d.'):
            return parts[1].split('.')[-1]
        else:
            return None

    def _check_keys(self, keys):
        """ Checks given keys exist in meta['columns']
        """
        keys = self._force_list(keys)

        valid = self._meta['columns'].keys() + self._meta['masks'].keys()

        invalid = ['"%s"' % _ for _ in keys if _ not in valid and _ != _TOTAL]

        if invalid:
            raise ValueError("Keys %s do not exist in meta['columns'] or "
                              "meta['masks']." % ", ".join(invalid))

        return keys

    def _clone(self, name=None):
        return Chain(self.stack, name=name)

    def get(self, data_key, filter_key, x_keys, y_keys, views, orient='x',
            rules=True, rules_weight=None, prioritize=True):
        """
        TODO: Full doc string
        Get a (list of) Chain instance(s) in either 'x' or 'y' orientation.
        Chain.dfs will be concatenated along the provided 'orient'-axis.
        """
        # TODO: VERIFY data_key
        # TODO: VERIFY filter_key
        # TODO: Add verbose arg to get()
        if self._meta is None:
            self._meta = self.stack[data_key].meta

        self._given_views = views
        self.x_keys = x_keys
        self.y_keys = y_keys
        if rules:
            if not isinstance(rules, list):
                self._has_rules = ['x', 'y']
            else:
                self._has_rules = rules

        if len(self.x_keys) > 1 and len(self.y_keys) > 1:
            chains = []
            if orient == 'x':
                it, keys = self.x_keys, self.y_keys
            else:
                it, keys = self.y_keys, self.x_keys

            for key in it:
                x_key, y_key = (key, keys) if orient == 'x' else (keys, key)
                chain = self._clone(name=key)
                chain = chain.get(data_key, filter_key, x_key, y_key, views,
                                  rules=rules, prioritize=prioritize)
                chains.append(chain)

            del self.stack
            return chains

        return self._get(data_key, filter_key, views, rules=rules,
                         prio=prioritize)

    def _get(self, data_key, filter_key, views, rules=False, prio=True):
        """ Get the concatenated Chain.DataFrame
        """
        concat_axis = 0

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

                if prio: link = self._drop_substituted_views(link)
                found_views, y_frames = self._concat_views(link, views)
                found.append(found_views)

                # TODO: contains arrary summ. attr.
                # TODO: make this work y_frames = self._pad_frames(y_frames)

                self.array_style = link
                if self.array_style > -1:
                    concat_axis = 1
                    y_frames = self._pad_frames(y_frames)

                x_frames.append(pd.concat(y_frames, axis=concat_axis))


            self._frame = pd.concat(self._pad(x_frames), axis=self.axis)
            if self._group_style == 'reduced' and self.array_style >- 1:
                self._frame = self._reduce_grouped_index(self._frame, 2, True)


            if self.axis == 1:
                self.views = found[-1]
            else:
                self.views = found

            self._index = self._frame.index
            self._columns = self._frame.columns

        del self.stack

        return self

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
                self.y_keys.remove(y_key)
        else:
            self.x_keys.remove(x_key)
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


    def _concat_views(self, link, views, found=None):
        """ Concatenates the Views of a Chain.
        """

        totals = [[_TOTAL]] * 2

        if found is None:
            found = OrderedDict()

        frames = []

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

                    found, grouped = self._concat_views(link, view, found=found)
                    if grouped:
                        frames.append(self._group_views(grouped, use_grp_type))
                else:
                    agg = link[view].meta()['agg']
                    is_descriptive = agg['method'] == 'descriptives'
                    is_base = agg['name'] in ['cbase', 'rbase']
                    is_sum = agg['name'] in ['counts_sum', 'c%_sum']

                    no_total_sign = is_descriptive or is_base or is_sum

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

                    # RULES SECTION
                    # ========================================================
                    # TODO: DYNAMIC RULES:
                    #   - all_rules_axes, rules_weight must be provided not hardcoded
                    #   - Review copy/pickle in original version!!!

                    rules_weight = None
                    if self._has_rules:
                        rules = Rules(link, view, axes=self._has_rules)
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
                        found[view] = len(frame.index)
            except KeyError:
                pass
        return found, frames

    def transform_tests(self, keep_code_index=True):
        """
        Transform column-wise digit-based test representation to letters.

        Adds a new row that is applying uppercase letters to all columns (A,
        B, C, ...) and maps any significance test's result cells to these column
        indicators.

        Parameters
        ----------
        keep_code_index : bool, default False
            The original column MultiIndex might be kept with the letter
            identificators added as a third (innermost) level. Alternatively,
            the letter representation can replace the definition of the second
            column level.

        Returns
        -------
        None
        """
        df = self.dataframe.copy()

        number_header_row = copy.copy(df.columns)
        all_numbers = [0] + df.columns.get_level_values(1).tolist()[1:]
        all_letters = ['@'] + list(string.ascii_uppercase)
        break_len = len(all_numbers)

        # Set the new column header (ABC, ...)
        questions = self.y_keys
        column_letters = ['@'] + list(string.ascii_uppercase)[:break_len-1]
        df.columns.set_levels(levels=column_letters, level=1, inplace=True)
        df.columns.set_labels(labels=xrange(0, break_len), level=1, inplace=True)
        letter_header_row = df.columns

        # Build the replacements dict
        test_dict = OrderedDict()
        for q in questions:
            test_dict[q] = {}
        for num_idx, col in enumerate(df.columns):
            if col[1] == '@':
                question = col[1]
            else:
                question = col[0]
            number = all_numbers[num_idx]
            letter = col[1]
            test_dict[question][number] = letter

        # Do the replacements...
        all_dfs  = []
        for col in questions:
            replacer = test_dict[col]
            try:
                value_df = df[col].copy()
            except KeyError:
                value_df = df[[df.columns[0]]].copy()
            values = value_df.replace(np.NaN, '-').values.tolist()
            new_values = []
            for v in values:
                if isinstance(v[0], (str, unicode)):
                    for number, letter in replacer.items():
                        v = [digit.replace(str(number), letter) for digit in v]
                    new_values.append(v)
                else:
                    new_values.append(v)
            part_df = pd.DataFrame(new_values)
            all_dfs.append(part_df)

        # Build new df
        letter_df = pd.concat(all_dfs, axis=1)
        letter_df = letter_df.T.drop_duplicates().T
        letter_df.index = df.index
        if keep_code_index:
            letter_df.columns = number_header_row
            new_letter_df = letter_df.T
            id_s =  pd.Series(letter_header_row.get_level_values(1).tolist(),
                              index=new_letter_df.index)
            new_letter_df['Test-IDs'] = id_s
            new_letter_df.set_index('Test-IDs', append=True, inplace=True)
            new_letter_df = new_letter_df.T
            letter_df = new_letter_df
        else:
            letter_df.columns = letter_header_row

        # Clean it up
        letter_df.replace('-', np.NaN, inplace=True)
        for signs in [('[', ''), (']', ''), (', ', '.')]:
            letter_df = letter_df.applymap(lambda x: x.replace(signs[0], signs[1])
                                           if isinstance(x, (str, unicode)) else x)
        self._frame = letter_df
        return None

    def paint(self, text_keys=None, display=None, axes=None, view_level=False):
        """ TODO: Doc
        """
        if text_keys is None:
            text_keys = finish_text_key(self._meta, {})

        if display is None:
            display = _AXES

        if axes is None:
            axes = _AXES

        self._paint(text_keys, display, axes)

        if view_level:
            self._add_view_level()

        return self

    def _paint(self, text_keys, display, axes):
        """ Paint the Chain.dataframe
        """
        indexes = []

        for axis in _AXES:
            index = self._index_switch(axis)
            if axis in axes:
                index = self._paint_index(index, text_keys, display, axis)
            indexes.append(index)

        self._frame.index, self._frame.columns = indexes

    def _paint_index(self, index, text_keys, display, axis):
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
                sub = self._paint_index(sub, text_keys, display,axis)
                arrays.extend(self._lzip(sub.ravel()))

            tuples = self._lzip(arrays)
            return pd.MultiIndex.from_tuples(tuples, names=index.names)

        levels = self._lzip(index.values)

        arrays = (self._get_level_0(levels[0], text_keys, display, axis),
                  self._get_level_1(levels, text_keys, display, axis))
        new_index = pd.MultiIndex.from_arrays(arrays, names=index.names)
        if self.array_style > -1 and axis == 'y':
            return new_index.droplevel(0)
        return new_index

    def _get_level_0(self, level, text_keys, display, axis):
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
                    text = self._get_text(value, text_keys[axis])
                    if axis in display:
                        value = '{}. {}'.format(value, text)
                    else:
                        value = text
            level_0_text.append(value)
        return map(unicode, level_0_text)

    def _get_level_1(self, levels, text_keys, display, axis):
        """
        """
        level_1_text = []
        for i, value in enumerate(levels[1]):
            if str(value).startswith('#pad'):
                level_1_text.append(value)
            elif pd.isnull(value):
                level_1_text.append(value)
            elif str(value) == '':
                level_1_text.append(value)
            else:
                translate = self._transl[self._transl.keys()[0]].keys()
                if value in self._text_map.keys() and value not in translate:
                    level_1_text.append(self._text_map[value])
                elif value in translate:
                    text = self._transl[text_keys[axis][0]][value]
                    level_1_text.append(text)
                else:
                    if self.array_style == 0 and axis == 'x':
                        text = self._get_text(value, text_keys[axis])
                        level_1_text.append(text)
                    else:
                        try:
                            for item in self._get_values(levels[0][i]):
                                if int(value) == item['value']:
                                    text = self._get_text(item, text_keys[axis])
                                    level_1_text.append(text)
                        except ValueError:
                            if self._grp_text_map:
                                for gtm in self._grp_text_map:
                                    if value in gtm.keys():
                                        text = gtm[value][text_keys[axis][0]]
                                        level_1_text.append(text)
        return map(unicode, level_1_text)

    def _get_text(self, value, text_keys):
        """
        """
        if value in self._meta['columns'].keys():
            return self._get_text_from_keys(self._meta['columns'][value],
                                            text_keys
                                           )
        elif value in self._meta['masks'].keys():
            return self._get_text_from_keys(self._meta['masks'][value], text_keys)
        else:
            return self._get_text_from_keys(value, text_keys)

    def _get_text_from_keys(self, meta_obj, text_keys):
        """ Find the first value in a meta object's "text" key that matches a
        text_key for it's axis.
        """
        error = "No text keys from {} found in {}"

        for k in text_keys:
            if k in meta_obj['text']:
                return meta_obj['text'][k]

        return None

    def _get_values(self, column):
        """ Returns values from self._meta["columns"] or
        self._meta["lib"]["values"][<mask name>] if parent is "array"
        """
        try:
            values = self._meta['columns'][column]['values']
        except KeyError:
            values = self._meta['lib']['values'][column]

        if isinstance(values, (str, unicode)):
            keys = values.split('@')
            values = self._meta[keys.pop(0)]
            while keys:
                values = values[keys.pop(0)]

        return values

    def _add_view_level(self):
        """ Insert a third Index level containing View keys into the DataFrame.
        """
        self._frame['View'] = pd.Series(self._views_per_rows,
                                        index=self._frame.index)
        self._frame.set_index('View', append=True, inplace=True)

    def bank(self, to_bank):
        """ Extract rows per View key and generate new DataFrame containing
        only these.
        """
        raise NotImplementedError("Chain.bank() under construction")
        # if not isinstance(to_bank, list):
        #     to_bank = [to_bank]
        # self._add_view_level()
        # not_banked = [v for v in self._frame.index.get_level_values(1).tolist()
        #               if v not in to_bank]
        # self._frame = self._frame.drop(not_banked, axis=0, level=1)
        # self._frame.index = self._frame.index.droplevel(2)
        # idx_names = self._frame.index.names
        # self._frame = self._frame.reset_index(drop=False)
        # self._frame = self._frame.set_index(idx_names)
        # self._basic_index = self._frame.index
        # self._frame.columns = self._basic_columns
        # return self

    def toggle_labels(self):
        """ Restore the unpainted/ painted Index, Columns appearance.
        """
        index, columns = self._frame.index, self._frame.columns
        self._frame.index, self._frame.columns = self.index, self.columns
        self.index, self.columns = index, columns
        return self

    @staticmethod
    def _single_column(*levels):
        """ Returns True if multiindex level 0 has one unique value
        """
        return all(len(level) == 1 for level in levels)

    @staticmethod
    def _force_list(obj):
        if isinstance(obj, (list, tuple)):
            return obj
        return [obj]

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
    def _reduce_grouped_index(grouped_df, view_padding, array_summary=False):
        idx = grouped_df.index
        q = idx.get_level_values(0).tolist()[0]
        if array_summary:
            val = idx.get_level_values(1).tolist()
            for index in range(1, len(val), 2):
                val[index] = ''
            grp_vals = val
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

    @classmethod
    def __pad_id(cls):
        cls._pad_id += 1
        return cls._pad_id









##############################################################################

class Quantity(object):
    """
    The Quantity object is the main Quantipy aggregation engine.

    Consists of a link's data matrix representation and sectional defintion
    of weight vector (wv), x-codes section (xsect) and y-codes section
    (ysect). The instance methods handle creation, retrieval and manipulation
    of the data input matrices and section definitions as well as the majority
    of statistical calculations.
    """
    # -------------------------------------------------
    # Instance initialization
    # -------------------------------------------------
    def __init__(self, link, weight=None, use_meta=False, base_all=False):
        # Collect information on wv, x- and y-section
        self._uses_meta = use_meta
        self.ds = self._convert_to_dataset(link)
        self.d = self._data
        self.base_all = base_all
        self._dataidx = link.get_data().index
        if self._uses_meta:
            self.meta = self._meta
            if self.meta().values() == [None] * len(self.meta().values()):
                self._uses_meta = False
                self.meta = None
        else:
            self.meta = None
        self._cache = link.get_cache()
        self.f = link.filter
        self.x = link.x
        self.y = link.y
        self.w = weight if weight is not None else '@1'
        self.is_weighted = False
        self.type = self._get_type()
        if self.type == 'nested':
            self.nest_def = Nest(self.y, self.d(), self.meta()).nest()
        self._squeezed = False
        self.idx_map = None
        self.xdef = self.ydef = None
        self.matrix = self._get_matrix()
        self.is_empty = self.matrix.sum() == 0
        self.switched = False
        self.factorized = None
        self.result = None
        self.logical_conditions = []
        self.cbase = self.rbase = None
        self.comb_x = self.comb_y = None
        self.miss_x = self.miss_y = None
        self.calc_x = self.calc_y = None
        self._has_x_margin = self._has_y_margin = False

    def __repr__(self):
        if self.result is not None:
            return '%s' % (self.result)
        else:
            return 'Quantity - x: {}, xdef: {} y: {}, ydef: {}, w: {}'.format(
                self.x, self.xdef, self.y, self.ydef, self.w)

    # -------------------------------------------------
    # Matrix creation and retrievel
    # -------------------------------------------------
    def _convert_to_dataset(self, link):
        ds = qp.DataSet('')
        ds._data = link.stack[link.data_key].data
        ds._meta = link.get_meta()
        return ds

    def _data(self):
        return self.ds._data

    def _meta(self):
        return self.ds._meta

    def _get_type(self):
        """
        Test variable type that can be "simple", "nested" or "array".
        """
        if self._uses_meta:
            masks = [self.x, self.y]
            if any(mask in self.meta()['masks'].keys() for mask in masks):
                mask = {
                    True: self.x,
                    False: self.y}.get(self.x in self.meta()['masks'].keys())
                if self.meta()['masks'][mask]['type'] == 'array':
                    if self.x == '@':
                        self.x, self.y = self.y, self.x
                    return 'array'
            elif '>' in self.y:
                return 'nested'
            else:
                return 'simple'
        else:
            return 'simple'

    def _is_multicode_array(self, mask_element):
        return self.d()[mask_element].dtype == 'object'

    def _get_wv(self):
        """
        Returns the weight vector of the matrix.
        """
        return self.d()[[self.w]].values

    def weight(self):
        """
        Weight by multiplying the indicator entries with the weight vector.
        """
        self.matrix *= np.atleast_3d(self.wv)
        # if self.is_weighted:
        #     self.matrix[:, 1:, 1:] *=  np.atleast_3d(self.wv)
        # else:
        #     self.matrix *= np.atleast_3d(self.wv)
        # self.is_weighted = True
        return None

    def unweight(self):
        """
        Remove any weighting by dividing the matrix by itself.
        """
        self.matrix /= self.matrix
        # self.matrix[:, 1:, 1:] /= self.matrix[:, 1:, 1:]
        # self.is_weighted = False
        return None

    def _get_total(self):
        """
        Return a vector of 1s for the matrix.
        """
        return self.d()[['@1']].values

    def _copy(self):
        """
        Copy the Quantity instance, i.e. its data matrix, into a new object.
        """
        m_copy = np.empty_like(self.matrix)
        m_copy[:] = self.matrix
        c = copy.copy(self)
        c.matrix = m_copy
        return c

    def _get_response_codes(self, var):
        """
        Query the meta specified codes values for a meta-using Quantity.
        """
        if self.type == 'array':
            rescodes = [v['value'] for v in self.meta()['lib']['values'][var]]
        else:
            values = emulate_meta(
                self.meta(), self.meta()['columns'][var].get('values', None))
            rescodes = [v['value'] for v in values]
        return rescodes

    def _get_response_texts(self, var, text_key=None):
        """
        Query the meta specified text values for a meta-using Quantity.
        """
        if text_key is None: text_key = 'main'
        if self.type == 'array':
            restexts = [v[text_key] for v in self.meta()['lib']['values'][var]]
        else:
            values = emulate_meta(
                self.meta(), self.meta()['columns'][var].get('values', None))
            restexts = [v['text'][text_key] for v in values]
        return restexts

    def _switch_axes(self):
        """
        """
        if self.switched:
            self.switched = False
            self.matrix = self.matrix.swapaxes(1, 2)
        else:
            self.switched = True
            self.matrix = self.matrix.swapaxes(2, 1)
        self.xdef, self.ydef = self.ydef, self.xdef
        self._x_indexers, self._y_indexers = self._y_indexers, self._x_indexers
        self.comb_x, self.comb_y = self.comb_y, self.comb_x
        self.miss_x, self.miss_y = self.miss_y, self.miss_x
        return self

    def _reset(self):
        for prop in self.__dict__.keys():
            if prop in ['_uses_meta', 'base_all', '_dataidx', 'meta', '_cache',
                        'd', 'idx_map']:
                pass
            elif prop in ['_squeezed', 'switched']:
                self.__dict__[prop] = False
            else:
                self.__dict__[prop] = None
            self.result = None
        return None

    def swap(self, var, axis='x', inplace=True):
        """
        Change the Quantity's x- or y-axis keeping filter and weight setup.

        All edits and aggregation results will be removed during the swap.

        Parameters
        ----------
        var : str
            New variable's name used in axis swap.
        axis : {'x', 'y'}, default ``'x'``
            The axis to swap.
        inplace : bool, default ``True``
            Whether to modify the Quantity inplace or return a new instance.

        Returns
        -------
        swapped : New Quantity instance with exchanged x- or y-axis.
        """
        if axis == 'x':
            x = var
            y = self.y
        else:
            x = self.x
            y = var
        f, w = self.f, self.w
        if inplace:
            swapped = self
        else:
            swapped = self._copy()
        swapped._reset()
        swapped.x, swapped.y = x, y
        swapped.f, swapped.w = f, w
        swapped.type = swapped._get_type()
        swapped._get_matrix()
        if not inplace:
            return swapped

    def rescale(self, scaling, drop=False):
        """
        Modify the object's ``xdef`` property reflecting new value defintions.

        Parameters
        ----------
        scaling : dict
            Mapping of old_code: new_code, given as of type int or float.
        drop : bool, default False
            If True, codes not included in the scaling dict will be excluded.

        Returns
        -------
        self
        """
        proper_scaling = {old_code: new_code for old_code, new_code
                         in scaling.items() if old_code in self.xdef}
        xdef_ref = [proper_scaling[code] if code in proper_scaling.keys()
                    else code for code in self.xdef]
        if drop:
            to_drop = [code for code in self.xdef if code not in
                       proper_scaling.keys()]
            self.exclude(to_drop, axis='x')
        self.xdef = xdef_ref
        return self

    def exclude(self, codes, axis='x'):
        """
        Wrapper for _missingfy(...keep_codes=False, ..., keep_base=False, ...)
        Excludes specified codes from aggregation.
        """
        self._missingfy(codes, axis=axis, keep_base=False, inplace=True)
        return self

    def limit(self, codes, axis='x'):
        """
        Wrapper for _missingfy(...keep_codes=True, ..., keep_base=True, ...)
        Restrict the data matrix entires to contain the specified codes only.
        """
        self._missingfy(codes, axis=axis, keep_codes=True, keep_base=True,
                        inplace=True)
        return self

    def filter(self, condition, keep_base=True, inplace=False):
        """
        Use a Quantipy conditional expression to filter the data matrix entires.
        """
        if inplace:
            filtered = self
        else:
            filtered = self._copy()
        qualified_rows = self._get_logic_qualifiers(condition)
        valid_rows = self.idx_map[self.idx_map[:, 0] == 1][:, 1]
        filter_idx = np.in1d(valid_rows, qualified_rows)
        if keep_base:
            filtered.matrix[~filter_idx, 1:, :] = np.NaN
        else:
            filtered.matrix[~filter_idx, :, :] = np.NaN
        if not inplace:
            return filtered

    def _get_logic_qualifiers(self, condition):
        if not isinstance(condition, dict):
            column = self.x
            logic = condition
        else:
            column = condition.keys()[0]
            logic = condition.values()[0]
        idx, logical_expression = get_logic_index(self.d()[column], logic, self.d())
        logical_expression = logical_expression.split(':')[0]
        if not column == self.x:
            logical_expression = logical_expression.replace('x[', column+'[')
        self.logical_conditions.append(logical_expression)
        return idx

    def _missingfy(self, codes, axis='x', keep_codes=False, keep_base=True,
                   indices=False, inplace=True):
        """
        Clean matrix from entries preserving or modifying the weight vector.

        Parameters
        ----------
        codes : list
            A list of codes to be considered in cleaning.
        axis : {'x', 'y'}, default 'x'
            The axis to clean codes on. Refers to the Link object's x- and y-
            axes.
        keep_codes : bool, default False
            Controls whether the passed codes are kept or erased from the
            Quantity matrix data entries.
        keep_base: bool, default True
            Controls whether the weight vector is set to np.NaN alongside
            the x-section rows or remains unmodified.
        indices: bool, default False
            If ``True``, the data matrix indicies of the corresponding codes
            will be returned as well.
        inplace : bool, default True
            Will overwrite self.matrix with the missingfied matrix by default.
            If ``False``, the method will return a new np.array with the
            modified entries.

        Returns
        -------
        self or numpy.array (and optionally a list of int when ``indices=True``)
            Either a new matrix is returned as numpy.array or the ``matrix``
            property is modified inplace.
        """
        if inplace:
            missingfied = self
        else:
            missingfied = self._copy()
        if axis == 'y' and self.y == '@' and not self.type == 'array':
            return self
        elif axis == 'y' and self.type == 'array':
            ni_err = 'Cannot missingfy array mask element sections!'
            raise NotImplementedError(ni_err)
        else:
            if axis == 'y':
                missingfied._switch_axes()
            mis_ix = missingfied._get_drop_idx(codes, keep_codes)
            mis_ix = [code + 1 for code in mis_ix]
            if mis_ix is not None:
                for ix in mis_ix:
                    np.place(missingfied.matrix[:, ix],
                             missingfied.matrix[:, ix] > 0, np.NaN)
                if not keep_base:
                    if axis == 'x':
                        self.miss_x = codes
                    else:
                        self.miss_y = codes
                    if self.type == 'array':
                        mask = np.nansum(missingfied.matrix[:, missingfied._x_indexers],
                                         axis=1, keepdims=True)
                        mask /= mask
                        mask = mask > 0
                    else:
                        mask = np.nansum(np.sum(missingfied.matrix,
                                                axis=1, keepdims=False),
                                         axis=1, keepdims=True) > 0
                    missingfied.matrix[~mask] = np.NaN
                if axis == 'y':
                    missingfied._switch_axes()
            if inplace:
                self.matrix = missingfied.matrix
                if indices:
                    return mis_ix
            else:
                if indices:
                    return missingfied, mis_ix
                else:
                    return missingfied

    def _organize_global_missings(self, missings):
        hidden = [c for c in missings.keys() if missings[c] == 'hidden']
        excluded = [c for c in missings.keys() if missings[c] == 'excluded']
        shown = [c for c in missings.keys() if missings[c] == 'shown']
        return hidden, excluded, shown

    def _organize_stats_missings(self, missings):
        excluded = [c for c in missings.keys()
                    if missings[c] in ['d.excluded', 'excluded']]
        return excluded

    def _autodrop_stats_missings(self):
        if self.x == '@':
            pass
        elif self.ds._has_missings(self.x):
            missings = self.ds._get_missings(self.x)
            to_drop = self._organize_stats_missings(missings)
            self.exclude(to_drop)
        else:
            pass
        return None

    def _clean_from_global_missings(self):
        if self.x == '@':
            pass
        elif self.ds._has_missings(self.x):
            missings = self.ds._get_missings(self.x)
            hidden, excluded, shown = self._organize_global_missings(missings)
            if excluded:
                excluded_codes = excluded
                excluded_idxer = self._missingfy(excluded, keep_base=False,
                                                 indices=True)
            else:
                excluded_codes, excluded_idxer = [], []
            if hidden:
                hidden_codes = hidden
                hidden_idxer = self._get_drop_idx(hidden, keep=False)
                hidden_idxer = [code + 1 for code in hidden_idxer]
            else:
                hidden_codes, hidden_idxer = [], []
            dropped_codes = excluded_codes + hidden_codes
            dropped_codes_idxer = excluded_idxer + hidden_idxer
            self._x_indexers = [x_idx for x_idx in self._x_indexers
                                if x_idx not in dropped_codes_idxer]
            self.matrix = self.matrix[:, [0] + self._x_indexers]
            self.xdef = [x_c for x_c in self.xdef if x_c not in dropped_codes]
        else:
            pass
        return None

    def _get_drop_idx(self, codes, keep):
        """
        Produces a list of indices referring to the given input matrix's axes
        sections in order to erase data entries.

        Parameters
        ----------
        codes : list
            Data codes that should be dropped from or kept in the matrix.
        keep : boolean
            Controls if the the passed code defintion is interpreted as
            "codes to keep" or "codes to drop".

        Returns
        -------
        drop_idx : list
            List of x section matrix indices.
        """
        if codes is None:
            return None
        else:
            if keep:
                return [self.xdef.index(code) for code in self.xdef
                        if code not in codes]
            else:
                return [self.xdef.index(code) for code in codes
                        if code in self.xdef]

    def group(self, groups, axis='x', expand=None, complete=False):
        """
        Build simple or logical net vectors, optionally keeping orginating codes.

        Parameters
        ----------
        groups : list, dict of lists or logic expression
            The group/net code defintion(s) in form of...

            * a simple list: ``[1, 2, 3]``
            * a dict of list: ``{'grp A': [1, 2, 3], 'grp B': [4, 5, 6]}``
            * a logical expression: ``not_any([1, 2])``

        axis : {``'x'``, ``'y'``}, default ``'x'``
            The axis to group codes on.
        expand : {None, ``'before'``, ``'after'``}, default ``None``
            If ``'before'``, the codes that are grouped will be kept and placed
            before the grouped aggregation; vice versa for ``'after'``. Ignored
            on logical expressions found in ``groups``.
        complete : bool, default False
            If True, codes that define the Link on the given ``axis`` but are
            not present in the ``groups`` defintion(s) will be placed in their
            natural position within the aggregation, respecting the value of
            ``expand``.

        Returns
        -------
        None
        """
        # check validity and clean combine instructions
        if axis == 'y' and self.type == 'array':
            ni_err_array = 'Array mask element sections cannot be combined.'
            raise NotImplementedError(ni_err_array)
        elif axis == 'y' and self.y == '@':
            val_err = 'Total link has no y-axis codes to combine.'
            raise ValueError(val_err)
        grp_def = self._organize_grp_def(groups, expand, complete, axis)
        combines = []
        names = []
        # generate the net vectors (+ possible expanded originating codes)
        for grp in grp_def:
            name, group, exp, logical = grp[0], grp[1], grp[2], grp[3]
            one_code = len(group) == 1
            if one_code and not logical:
                vec = self._slice_vec(group[0], axis=axis)
            elif not logical and not one_code:
                vec, idx = self._grp_vec(group, axis=axis)
            else:
                vec = self._logic_vec(group)
            if axis == 'y':
                self._switch_axes()
            if exp is not None:
                m_idx = [ix for ix in self._x_indexers if ix not in idx]
                m_idx = self._sort_indexer_as_codes(m_idx, group)
                if exp == 'after':
                    names.extend(name)
                    names.extend([c for c in group])
                    combines.append(
                        np.concatenate([vec, self.matrix[:, m_idx]], axis=1))
                else:
                    names.extend([c for c in group])
                    names.extend(name)
                    combines.append(
                        np.concatenate([self.matrix[:, m_idx], vec], axis=1))
            else:
                names.extend(name)
                combines.append(vec)
            if axis == 'y':
                self._switch_axes()
        # re-construct the combined data matrix
        combines = np.concatenate(combines, axis=1)
        if axis == 'y':
            self._switch_axes()
        combined_matrix = np.concatenate([self.matrix[:, [0]],
                                          combines], axis=1)
        if axis == 'y':
            combined_matrix = combined_matrix.swapaxes(1, 2)
            self._switch_axes()
        # update the sectional information
        new_sect_def = range(0, combined_matrix.shape[1] - 1)
        if axis == 'x':
            self.xdef = new_sect_def
            self._x_indexers = self._get_x_indexers()
            self.comb_x = names
        else:
            self.ydef = new_sect_def
            self._y_indexers = self._get_y_indexers()
            self.comb_y = names
        self.matrix = combined_matrix

    def _slice_vec(self, code, axis='x'):
        '''
        '''
        if axis == 'x':
            code_idx = self.xdef.index(code) + 1
        else:
            code_idx = self.ydef.index(code) + 1
        if axis == 'x':
            m_slice = self.matrix[:, [code_idx]]
        else:
            self._switch_axes()
            m_slice = self.matrix[:, [code_idx]]
            self._switch_axes()
        return m_slice

    def _grp_vec(self, codes, axis='x'):
        netted, idx = self._missingfy(codes=codes, axis=axis,
                                      keep_codes=True, keep_base=True,
                                      indices=True, inplace=False)
        if axis == 'y':
            netted._switch_axes()
        net_vec = np.nansum(netted.matrix[:, netted._x_indexers],
                            axis=1, keepdims=True)
        net_vec /= net_vec
        return net_vec, idx

    def _logic_vec(self, condition):
        """
        Create net vector of qualified rows based on passed condition.
        """
        filtered = self.filter(condition=condition, inplace=False)
        net_vec = np.nansum(filtered.matrix[:, self._x_indexers], axis=1,
                            keepdims=True)
        net_vec /= net_vec
        return net_vec

    def _grp_type(self, grp_def):
        if isinstance(grp_def, list):
            if not isinstance(grp_def[0], (int, float)):
                return 'block'
            else:
                return 'list'
        elif isinstance(grp_def, tuple):
            return 'logical'
        elif isinstance(grp_def, dict):
            return 'wildcard'

    def _add_unused_codes(self, grp_def_list, axis):
        '''
        '''
        query_codes = self.xdef if axis == 'x' else self.ydef
        frame_lookup = {c: [[c], [c], None, False] for c in query_codes}
        frame = [[code] for code in query_codes]
        for grpdef_idx, grpdef in enumerate(grp_def_list):
            for code in grpdef[1]:
                if [code] in frame:
                    if grpdef not in frame:
                        frame[frame.index([code])] = grpdef
                    else:
                        frame[frame.index([code])] = '-'
        frame = [code for code in frame if not code == '-']
        for code in frame:
            if code[0] in frame_lookup.keys():
               frame[frame.index([code[0]])] = frame_lookup[code[0]]
        return frame

    def _organize_grp_def(self, grp_def, method_expand, complete, axis):
        """
        Sanitize a combine instruction list (of dicts): names, codes, expands.
        """
        organized_def = []
        codes_used = []
        any_extensions = complete
        any_logical = False
        if method_expand is None and complete:
            method_expand = 'before'
        if not self._grp_type(grp_def) == 'block':
            grp_def = [{'net': grp_def, 'expand': method_expand}]
        for grp in grp_def:
            if any(isinstance(val, (tuple, dict)) for val in grp.values()):
                if complete:
                    ni_err = ('Logical expr. unsupported when complete=True. '
                              'Only list-type nets/groups can be completed.')
                    raise NotImplementedError(ni_err)
                if 'expand' in grp.keys():
                    del grp['expand']
                expand = None
                logical = True
            else:
                if 'expand' in grp.keys():
                    grp = copy.deepcopy(grp)
                    expand = grp['expand']
                    if expand is None and complete:
                        expand = 'before'
                    del grp['expand']
                else:
                    expand = method_expand
                logical = False
            organized_def.append([grp.keys(), grp.values()[0], expand, logical])
            if expand:
                any_extensions = True
            if logical:
                any_logical = True
            codes_used.extend(grp.values()[0])
        if not any_logical:
            if len(set(codes_used)) != len(codes_used) and any_extensions:
                ni_err_extensions = ('Same codes in multiple groups unsupported '
                                     'with expand and/or complete =True.')
                raise NotImplementedError(ni_err_extensions)
        if complete:
            return self._add_unused_codes(organized_def, axis)
        else:
            return organized_def

    def _force_to_nparray(self):
        """
        Convert the aggregation result into its numpy array equivalent.
        """
        if isinstance(self.result, pd.DataFrame):
            self.result = self.result.values
            return True
        else:
            return False

    def _attach_margins(self):
        """
        Force margins back into the current Quantity.result if none are found.
        """
        if not self._res_is_stat():
            values = self.result
            if not self._has_y_margin and not self.y == '@':
                margins = False
                values = np.concatenate([self.rbase[1:, :], values], 1)
            else:
                margins = True
            if not self._has_x_margin:
                margins = False
                values = np.concatenate([self.cbase, values], 0)
            else:
                margins = True
            self.result = values
            return margins
        else:
            return False

    def _organize_expr_def(self, expression, axis):
        """
        """
        # Prepare expression parts and lookups for indexing the agg. result
        val1, op, val2 = expression[0], expression[1], expression[2]
        if self._res_is_stat():
            idx_c = [self.current_agg]
            offset = 0
        else:
            if axis == 'x':
                idx_c = self.xdef if not self.comb_x else self.comb_x
            else:
                idx_c = self.ydef if not self.comb_y else self.comb_y
            offset = 1
        # Test expression validity and find np.array indices / prepare scalar
        # values of the expression
        idx_err = '"{}" not found in {}-axis.'
        # [1] input is 1. scalar, 2. vector from the agg. result
        if isinstance(val1, list):
            if not val2 in idx_c:
                raise IndexError(idx_err.format(val2, axis))
            val1 = val1[0]
            val2 = idx_c.index(val2) + offset
            expr_type = 'scalar_1'
        # [2] input is 1. vector from the agg. result, 2. scalar
        elif isinstance(val2, list):
            if not val1 in idx_c:
                raise IndexError(idx_err.format(val1, axis))
            val1 = idx_c.index(val1) + offset
            val2 = val2[0]
            expr_type = 'scalar_2'
        # [3] input is two vectors from the agg. result
        elif not any(isinstance(val, list) for val in [val1, val2]):
            if not val1 in idx_c:
                raise IndexError(idx_err.format(val1, axis))
            if not val2 in idx_c:
                raise IndexError(idx_err.format(val2, axis))
            val1 = idx_c.index(val1) + offset
            val2 = idx_c.index(val2) + offset
            expr_type = 'vectors'
        return val1, op, val2, expr_type, idx_c

    @staticmethod
    def constant(num):
        return [num]

    def calc(self, expression, axis='x', result_only=False):
        """
        Compute (simple) aggregation level arithmetics.
        """
        unsupported = ['cbase', 'rbase', 'summary', 'x_sum', 'y_sum']
        if self.result is None:
            raise ValueError('No aggregation to base calculation on.')
        elif self.current_agg in unsupported:
            ni_err = 'Aggregation type "{}" not supported.'
            raise NotImplementedError(ni_err.format(self.current_agg))
        elif axis not in ['x', 'y']:
            raise ValueError('Invalid axis parameter: {}'.format(axis))
        is_df = self._force_to_nparray()
        has_margin = self._attach_margins()
        values = self.result
        expr_name = expression.keys()[0]
        if axis == 'x':
            self.calc_x = expr_name
        else:
            self.calc_y = expr_name
            values = values.T
        expr = expression.values()[0]
        v1, op, v2, exp_type, index_codes = self._organize_expr_def(expr, axis)
        # ====================================================================
        # TODO: generalize this calculation part so that it can "parse"
        # arbitrary calculation rules given as nested or concatenated
        # operators/codes sequences.
        if exp_type == 'scalar_1':
            val1, val2 = v1, values[[v2], :]
        elif exp_type == 'scalar_2':
            val1, val2 = values[[v1], :], v2
        elif exp_type == 'vectors':
            val1, val2 = values[[v1], :], values[[v2], :]
        calc_res = op(val1, val2)
        # ====================================================================
        if axis == 'y':
            calc_res = calc_res.T
        ap_axis = 0 if axis == 'x' else 1
        if result_only:
            if not self._res_is_stat():
                self.result = np.concatenate([self.result[[0], :], calc_res],
                                             ap_axis)
            else:
                self.result = calc_res
        else:
            self.result = np.concatenate([self.result, calc_res], ap_axis)
            if axis == 'x':
                self.calc_x = index_codes + [self.calc_x]
            else:
                self.calc_y = index_codes + [self.calc_y]
        self.cbase = self.result[[0], :]
        if self.type in ['simple', 'nested']:
            self.rbase = self.result[:, [0]]
        else:
            self.rbase = None
        if not self._res_is_stat():
            self.current_agg = 'calc'
            self._organize_margins(has_margin)
        else:
            self.current_agg = 'calc'
        if is_df:
            self.to_df()
        return self

    def count(self, axis=None, raw_sum=False, margin=True, as_df=True):
        """
        Count entries over all cells or per axis margin.

        Parameters
        ----------
        axis : {None, 'x', 'y'}, deafult None
            When axis is None, the frequency of all cells from the uni- or
            multivariate distribution is presented. If the axis is specified
            to be either 'x' or 'y' the margin per axis becomes the resulting
            aggregation.
        raw_sum : bool, default False
            If True will perform a simple summation over the cells given the
            axis parameter. This ignores net counting of qualifying answers in
            favour of summing over all answers given when considering margins.
        margin : bool, deafult True
            Controls whether the margins of the aggregation result are shown.
            This also applies to margin aggregations themselves, since they
            contain a margin in (form of the total number of cases) as well.
        as_df : bool, default True
            Controls whether the aggregation is transformed into a Quantipy-
            multiindexed (following the Question/Values convention)
            pandas.DataFrame or will be left in its numpy.array format.

        Returns
        -------
        self
            Passes a pandas.DataFrame or numpy.array of cell or margin counts
            to the ``result`` property.
        """
        if axis is None and raw_sum:
            raise ValueError('Cannot calculate raw sum without axis.')
        if axis is None:
            self.current_agg = 'freq'
        elif axis == 'x':
            self.current_agg = 'cbase' if not raw_sum else 'x_sum'
        elif axis == 'y':
            self.current_agg = 'rbase' if not raw_sum else 'y_sum'
        if not self.w == '@1':
            self.weight()
        if not self.is_empty or self._uses_meta:
            counts = np.nansum(self.matrix, axis=0)
        else:
            counts = self._empty_result()
        self.cbase = counts[[0], :]
        if self.type in ['simple', 'nested']:
            self.rbase = counts[:, [0]]
        else:
            self.rbase = None
        if axis is None:
            self.result = counts
        elif axis == 'x':
            if not raw_sum:
                self.result = counts[[0], :]
            else:
                self.result = np.nansum(counts[1:, :], axis=0, keepdims=True)
        elif axis == 'y':
            if not raw_sum:
                self.result = counts[:, [0]]
            else:
                if self.x == '@' or self.y == '@':
                    self.result = counts[:, [0]]
                else:
                    self.result = np.nansum(counts[:, 1:], axis=1, keepdims=True)
        self._organize_margins(margin)
        if as_df:
            self.to_df()
        self.unweight()
        return self

    def _empty_result(self):
        if self._res_is_stat() or self.current_agg == 'summary':
            self.factorized = 'x'
            xdim = 1 if self._res_is_stat() else 8
            if self.ydef is None:
                ydim = 1
            elif self.ydef is not None and len(self.ydef) == 0:
                ydim = 2
            else:
                ydim = len(self.ydef) + 1
        else:
            if self.xdef is not None:
                if len(self.xdef) == 0:
                    xdim = 2
                else:
                    xdim = len(self.xdef) + 1
                if self.ydef is None:
                    ydim = 1
                elif self.ydef is not None and len(self.ydef) == 0:
                    ydim = 2
                else:
                    ydim = len(self.ydef) + 1
            elif self.xdef is None:
                xdim = 2
                if self.ydef is None:
                    ydim = 1
                elif self.ydef is not None and len(self.ydef) == 0:
                    ydim = 2
                else:
                    ydim = len(self.ydef) + 1
        return np.zeros((xdim, ydim))

    def _effective_n(self, axis=None, margin=True):
        self.weight()
        effective = (np.nansum(self.matrix, axis=0)**2 /
                     np.nansum(self.matrix**2, axis=0))
        self.unweight()
        start_on = 0 if margin else 1
        if axis is None:
            return effective[start_on:, start_on:]
        elif axis == 'x':
            return effective[[0], start_on:]
        else:
            return effective[start_on:, [0]]

    def summarize(self, stat='summary', axis='x', margin=True, as_df=True):
        """
        Calculate distribution statistics across the given axis.

        Parameters
        ----------
        stat : {'summary', 'mean', 'median', 'var', 'stddev', 'sem', varcoeff',
                'min', 'lower_q', 'upper_q', 'max'}, default 'summary'
            The measure to calculate. Defaults to a summary output of the most
            important sample statistics.
        axis : {'x', 'y'}, default 'x'
            The axis which is reduced in the aggregation, e.g. column vs. row
            means.
        margin : bool, default True
            Controls whether statistic(s) of the marginal distribution are
            shown.
        as_df : bool, default True
            Controls whether the aggregation is transformed into a Quantipy-
            multiindexed (following the Question/Values convention)
            pandas.DataFrame or will be left in its numpy.array format.

        Returns
        -------
        self
            Passes a pandas.DataFrame or numpy.array of the descriptive (summary)
            statistic(s) to the ``result`` property.
        """
        self.current_agg = stat
        if self.is_empty:
            self.result = self._empty_result()
        else:
            self._autodrop_stats_missings()
            if stat == 'summary':
                stddev, mean, base = self._dispersion(axis, measure='sd',
                                                      _return_mean=True,
                                                      _return_base=True)
                self.result = np.concatenate([
                    base, mean, stddev,
                    self._min(axis),
                    self._percentile(perc=0.25),
                    self._percentile(perc=0.50),
                    self._percentile(perc=0.75),
                    self._max(axis)
                    ], axis=0)
            elif stat == 'mean':
                self.result = self._means(axis)
            elif stat == 'var':
                self.result = self._dispersion(axis, measure='var')
            elif stat == 'stddev':
                self.result = self._dispersion(axis, measure='sd')
            elif stat == 'sem':
                self.result = self._dispersion(axis, measure='sem')
            elif stat == 'varcoeff':
                self.result = self._dispersion(axis, measure='varcoeff')
            elif stat == 'min':
                self.result = self._min(axis)
            elif stat == 'lower_q':
                self.result = self._percentile(perc=0.25)
            elif stat == 'median':
                self.result = self._percentile(perc=0.5)
            elif stat == 'upper_q':
                self.result = self._percentile(perc=0.75)
            elif stat == 'max':
                self.result = self._max(axis)
        self._organize_margins(margin)
        if as_df:
            self.to_df()
        return self

    def _factorize(self, axis='x', inplace=True):
        self.factorized = axis
        if inplace:
            factorized = self
        else:
            factorized = self._copy()
        if axis == 'y':
            factorized._switch_axes()
        np.copyto(factorized.matrix[:, 1:, :],
                  np.atleast_3d(factorized.xdef),
                  where=factorized.matrix[:, 1:, :]>0)
        if not inplace:
            return factorized

    def _means(self, axis, _return_base=False):
        fact = self._factorize(axis=axis, inplace=False)
        if not self.w == '@1':
            fact.weight()
        fact_prod = np.nansum(fact.matrix, axis=0)
        fact_prod_sum = np.nansum(fact_prod[1:, :], axis=0, keepdims=True)
        bases = fact_prod[[0], :]
        means = fact_prod_sum/bases
        if axis == 'y':
            self._switch_axes()
            means = means.T
            bases = bases.T
        if _return_base:
            return means, bases
        else:
            return means

    def _dispersion(self, axis='x', measure='sd', _return_mean=False,
                    _return_base=False):
        """
        Extracts measures of dispersion from the incoming distribution of
        X vs. Y. Can return the arithm. mean by request as well. Dispersion
        measure supported are standard deviation, variance, coeffiecient of
        variation and standard error of the mean.
        """
        means, bases = self._means(axis, _return_base=True)
        unbiased_n = bases - 1
        self.unweight()
        factorized = self._factorize(axis, inplace=False)
        factorized.matrix[:, 1:] -= means
        factorized.matrix[:, 1:] *= factorized.matrix[:, 1:, :]
        if not self.w == '@1':
            factorized.weight()
        diff_sqrt = np.nansum(factorized.matrix[:, 1:], axis=1)
        disp = np.nansum(diff_sqrt/unbiased_n, axis=0, keepdims=True)
        disp[disp <= 0] = np.NaN
        disp[np.isinf(disp)] = np.NaN
        if measure == 'sd':
            disp = np.sqrt(disp)
        elif measure == 'sem':
            disp = np.sqrt(disp) / np.sqrt((unbiased_n + 1))
        elif measure == 'varcoeff':
            disp = np.sqrt(disp) / means
        self.unweight()
        if _return_mean and _return_base:
            return disp, means, bases
        elif _return_mean:
            return disp, means
        elif _return_base:
            return disp, bases
        else:
            return disp

    def _max(self, axis='x'):
        factorized = self._factorize(axis, inplace=False)
        vals = np.nansum(factorized.matrix[:, 1:, :], axis=1)
        return np.nanmax(vals, axis=0, keepdims=True)

    def _min(self, axis='x'):
        factorized = self._factorize(axis, inplace=False)
        vals = np.nansum(factorized.matrix[:, 1:, :], axis=1)
        if 0 not in factorized.xdef: np.place(vals, vals == 0, np.inf)
        return np.nanmin(vals, axis=0, keepdims=True)

    def _percentile(self, axis='x', perc=0.5):
        """
        Computes percentiles from the incoming distribution of X vs.Y and the
        requested percentile value. The implementation mirrors the algorithm
        used in SPSS Dimensions and the EXAMINE procedure in SPSS Statistics.
        It based on the percentile defintion #6 (adjusted for survey weights)
        in:
        Hyndman, Rob J. and Fan, Yanan (1996) -
        "Sample Quantiles in Statistical Packages",
        The American Statistician, 50, No. 4, 361-365.

        Parameters
        ----------
        axis : {'x', 'y'}, default 'x'
            The axis which is reduced in the aggregation, i.e. column vs. row
            medians.
        perc : float, default 0.5
            Defines the percentile to be computed. Defaults to 0.5,
            the sample median.

        Returns
        -------
        percs : np.array
            Numpy array storing percentile values.
        """
        percs = []
        factorized = self._factorize(axis, inplace=False)
        vals = np.nansum(np.nansum(factorized.matrix[:, 1:, :], axis=1,
                                   keepdims=True), axis=1)
        weights = (vals/vals)*self.wv
        for shape_i in range(0, vals.shape[1]):
            iter_weights = weights[:, shape_i]
            iter_vals = vals[:, shape_i]
            mask = ~np.isnan(iter_weights)
            iter_weights = iter_weights[mask]
            iter_vals = iter_vals[mask]
            sorter = np.argsort(iter_vals)
            iter_vals = np.take(iter_vals, sorter)
            iter_weights = np.take(iter_weights, sorter)
            iter_wsum = np.nansum(iter_weights, axis=0)
            iter_wcsum = np.cumsum(iter_weights, axis=0)
            k = (iter_wsum + 1.0) * perc
            if iter_vals.shape[0] == 0:
                percs.append(0.00)
            elif iter_vals.shape[0] == 1:
                percs.append(iter_vals[0])
            elif iter_wcsum[0] > k:
                wcsum_k = iter_wcsum[0]
                percs.append(iter_vals[0])
            elif iter_wcsum[-1] <= k:
                percs.append(iter_vals[-1])
            else:
                wcsum_k = iter_wcsum[iter_wcsum <= k][-1]
                p_k_idx = np.searchsorted(np.ndarray.flatten(iter_wcsum), wcsum_k)
                p_k = iter_vals[p_k_idx]
                p_k1 = iter_vals[p_k_idx+1]
                w_k1 = iter_weights[p_k_idx+1]
                excess = k - wcsum_k
                if excess >= 1.0:
                    percs.append(p_k1)
                else:
                    if w_k1 >= 1.0:
                        percs.append((1.0-excess)*p_k + excess*p_k1)
                    else:
                        percs.append((1.0-(excess/w_k1))*p_k +
                                     (excess/w_k1)*p_k1)
        return np.array(percs)[None, :]

    def _organize_margins(self, margin):
        if self._res_is_stat():
            if self.type == 'array' or self.y == '@' or self.x == '@':
                self._has_y_margin = self._has_x_margin = False
            else:
                if self.factorized == 'x':
                    if not margin:
                        self._has_x_margin = False
                        self._has_y_margin = False
                        self.result = self.result[:, 1:]
                    else:
                        self._has_x_margin = False
                        self._has_y_margin = True
                else:
                    if not margin:
                        self._has_x_margin = False
                        self._has_y_margin = False
                        self.result = self.result[1:, :]
                    else:
                        self._has_x_margin = True
                        self._has_y_margin = False
        if self._res_is_margin():
            if self.y == '@' or self.x == '@':
                if self.current_agg in ['cbase', 'x_sum']:
                    self._has_y_margin = self._has_x_margin = False
                if self.current_agg in ['rbase', 'y_sum']:
                    if not margin:
                        self._has_y_margin = self._has_x_margin = False
                        self.result = self.result[1:, :]
                    else:
                        self._has_x_margin = True
                        self._has_y_margin = False
            else:
                if self.current_agg in ['cbase', 'x_sum']:
                    if not margin:
                        self._has_y_margin = self._has_x_margin = False
                        self.result = self.result[:, 1:]
                    else:
                        self._has_x_margin = False
                        self._has_y_margin = True
                if self.current_agg in ['rbase', 'y_sum']:
                    if not margin:
                        self._has_y_margin = self._has_x_margin = False
                        self.result = self.result[1:, :]
                    else:
                        self._has_x_margin = True
                        self._has_y_margin = False
        elif self.current_agg in ['freq', 'summary', 'calc']:
            if self.type == 'array' or self.y == '@' or self.x == '@':
                if not margin:
                    self.result = self.result[1:, :]
                    self._has_x_margin = False
                    self._has_y_margin = False
                else:
                    self._has_x_margin = True
                    self._has_y_margin = False
            else:
                if not margin:
                    self.result = self.result[1:, 1:]
                    self._has_x_margin = False
                    self._has_y_margin = False
                else:
                    self._has_x_margin = True
                    self._has_y_margin = True
        else:
            pass

    def _sort_indexer_as_codes(self, indexer, codes):
        mapping = sorted(zip(indexer, codes), key=lambda l: l[1])
        return [i[0] for i in mapping]

    def _get_y_indexers(self):
        if self._squeezed or self.type in ['simple', 'nested']:
            if self.ydef is not None:
                idxs = range(1, len(self.ydef)+1)
                return self._sort_indexer_as_codes(idxs, self.ydef)
            else:
                return [1]
        else:
            y_indexers = []
            xdef_len = len(self.xdef)
            zero_based_ys = [idx for idx in xrange(0, xdef_len)]
            for y_no in xrange(0, len(self.ydef)):
                if y_no == 0:
                    y_indexers.append(zero_based_ys)
                else:
                    y_indexers.append([idx + y_no * xdef_len
                                       for idx in zero_based_ys])
        return y_indexers

    def _get_x_indexers(self):
        if self._squeezed or self.type in ['simple', 'nested']:
            idxs = range(1, len(self.xdef)+1)
            return self._sort_indexer_as_codes(idxs, self.xdef)
        else:
            x_indexers = []
            upper_x_idx = len(self.ydef)
            start_x_idx = [len(self.xdef) * offset
                           for offset in range(0, upper_x_idx)]
            for x_no in range(0, len(self.xdef)):
                x_indexers.append([idx + x_no for idx in start_x_idx])
            return x_indexers

    def _squeeze_dummies(self):
        """
        Reshape and replace initial 2D dummy matrix into its 3D equivalent.
        """
        self.wv = self.matrix[:, [-1]]
        sects = []
        if self.type == 'array':
            x_sections = self._get_x_indexers()
            y_sections = self._get_y_indexers()
            y_total = np.nansum(self.matrix[:, x_sections], axis=1)
            y_total /= y_total
            y_total = y_total[:, None, :]
            for sect in y_sections:
                sect = self.matrix[:, sect]
                sects.append(sect)
            sects = np.dstack(sects)
            self._squeezed = True
            sects = np.concatenate([y_total, sects], axis=1)
            self.matrix = sects
            self._x_indexers = self._get_x_indexers()
            self._y_indexers = []
        elif self.type in ['simple', 'nested']:
            x = self.matrix[:, :len(self.xdef)+1]
            y = self.matrix[:, len(self.xdef)+1:-1]
            for i in range(0, y.shape[1]):
                sects.append(x * y[:, [i]])
            sects = np.dstack(sects)
            self._squeezed = True
            self.matrix = sects
            self._x_indexers = self._get_x_indexers()
            self._y_indexers = self._get_y_indexers()
        #=====================================================================
        #THIS CAN SPEED UP PERFOMANCE BY A GOOD AMOUNT BUT STACK-SAVING
        #TIME & SIZE WILL SUFFER. WE CAN DEL THE "SQUEEZED" COLLECTION AT
        #SAVE STAGE.
        #=====================================================================
        # self._cache.set_obj(collection='squeezed',
        #                     key=self.f+self.w+self.x+self.y,
        #                     obj=(self.xdef, self.ydef,
        #                          self._x_indexers, self._y_indexers,
        #                          self.wv, self.matrix, self.idx_map))

    def _get_matrix(self):
        wv = self._cache.get_obj('weight_vectors', self.w)
        if wv is None:
            wv = self._get_wv()
            self._cache.set_obj('weight_vectors', self.w, wv)
        total = self._cache.get_obj('weight_vectors', '@1')
        if total is None:
            total = self._get_total()
            self._cache.set_obj('weight_vectors', '@1', total)
        if self.type == 'array':
            xm, self.xdef, self.ydef = self._dummyfy()
            self.matrix = np.concatenate((xm, wv), 1)
        else:
            if self.y == '@' or self.x == '@':
                section = self.x if self.y == '@' else self.y
                xm, self.xdef = self._cache.get_obj('matrices', section)
                if xm is None:
                    xm, self.xdef = self._dummyfy(section)
                    self._cache.set_obj('matrices', section, (xm, self.xdef))
                self.ydef = None
                self.matrix = np.concatenate((total, xm, total, wv), 1)
            else:
                xm, self.xdef = self._cache.get_obj('matrices', self.x)
                if xm is None:
                    xm, self.xdef = self._dummyfy(self.x)
                    self._cache.set_obj('matrices', self.x, (xm, self.xdef))
                ym, self.ydef = self._cache.get_obj('matrices', self.y)
                if ym is None:
                    ym, self.ydef = self._dummyfy(self.y)
                    self._cache.set_obj('matrices', self.y, (ym, self.ydef))
                self.matrix = np.concatenate((total, xm, total, ym, wv), 1)
        self.matrix = self.matrix[self._dataidx]
        self.matrix = self._clean()
        self._squeeze_dummies()
        self._clean_from_global_missings()
        return self.matrix

    def _dummyfy(self, section=None):
        if section is not None:
            # i.e. Quantipy multicode data
            if self.d()[section].dtype == 'object':
                section_data = self.d()[section].str.get_dummies(';')
                if self._uses_meta:
                    res_codes = self._get_response_codes(section)
                    section_data.columns = [int(col) for col in section_data.columns]
                    section_data = section_data.reindex(columns=res_codes)
                    section_data.replace(np.NaN, 0, inplace=True)
                if not self._uses_meta:
                    section_data.sort_index(axis=1, inplace=True)
            # i.e. Quantipy single-coded/numerical data
            else:
                section_data = pd.get_dummies(self.d()[section])
                if self._uses_meta and not self._is_raw_numeric(section):
                    res_codes = self._get_response_codes(section)
                    section_data = section_data.reindex(columns=res_codes)
                    section_data.replace(np.NaN, 0, inplace=True)
                section_data.rename(
                    columns={
                        col: int(col)
                        if float(col).is_integer()
                        else col
                        for col in section_data.columns
                    },
                    inplace=True)
            return section_data.values, section_data.columns.tolist()
        elif section is None and self.type == 'array':
            a_i = [i['source'].split('@')[-1] for i in
                   self.meta()['masks'][self.x]['items']]
            a_res = self._get_response_codes(self.x)
            dummies = []
            if self._is_multicode_array(a_i[0]):
                for i in a_i:
                    i_dummy = self.d()[i].str.get_dummies(';')
                    i_dummy.columns = [int(col) for col in i_dummy.columns]
                    dummies.append(i_dummy.reindex(columns=a_res))
            else:
                for i in a_i:
                    dummies.append(pd.get_dummies(self.d()[i]).reindex(columns=a_res))
            a_data = pd.concat(dummies, axis=1)
            return a_data.values, a_res, a_i

    def _clean(self):
        """
        Drop empty sectional rows from the matrix.
        """
        mat = self.matrix.copy()
        mat_indexer = np.expand_dims(self._dataidx, 1)
        if not self.type == 'array':
            xmask = (np.nansum(mat[:, 1:len(self.xdef)+1], axis=1) > 0)
            if self.ydef is not None:
                if self.base_all:
                    ymask = (np.nansum(mat[:, len(self.xdef)+1:-1], axis=1) > 0)
                else:
                    ymask = (np.nansum(mat[:, len(self.xdef)+2:-1], axis=1) > 0)
                self.idx_map = np.concatenate(
                    [np.expand_dims(xmask & ymask, 1), mat_indexer], axis=1)
                return mat[xmask & ymask]
            else:
                self.idx_map = np.concatenate(
                    [np.expand_dims(xmask, 1), mat_indexer], axis=1)
                return mat[xmask]
        else:
            mask = (np.nansum(mat[:, :-1], axis=1) > 0)
            self.idx_map = np.concatenate(
                [np.expand_dims(mask, 1), mat_indexer], axis=1)
            return mat[mask]

    def _is_raw_numeric(self, var):
        return self.meta()['columns'][var]['type'] in ['int', 'float']

    def _res_from_count(self):
        return self._res_is_margin() or self.current_agg == 'freq'

    def _res_from_summarize(self):
        return self._res_is_stat() or self.current_agg == 'summary'

    def _res_is_margin(self):
        return self.current_agg in ['tbase', 'cbase', 'rbase', 'x_sum', 'y_sum']

    def _res_is_stat(self):
        return self.current_agg in ['mean', 'min', 'max', 'varcoeff', 'sem',
                                    'stddev', 'var', 'median', 'upper_q',
                                    'lower_q']
    def to_df(self):
        if self.current_agg == 'freq':
            if not self.comb_x:
                self.x_agg_vals = self.xdef
            else:
                self.x_agg_vals = self.comb_x
            if not self.comb_y:
                self.y_agg_vals = self.ydef
            else:
                self.y_agg_vals = self.comb_y
        elif self.current_agg == 'calc':
            if self.calc_x:
                self.x_agg_vals = self.calc_x
                self.y_agg_vals = self.ydef if not self.comb_y else self.comb_y
            else:
                self.x_agg_vals = self.xdef if not self.comb_x else self.comb_x
                self.y_agg_vals = self.calc_y
        elif self.current_agg == 'summary':
            summary_vals = ['mean', 'stddev', 'min', '25%',
                            'median', '75%', 'max']
            self.x_agg_vals = summary_vals
            self.y_agg_vals = self.ydef
        elif self.current_agg in ['x_sum', 'cbase']:
            self.x_agg_vals = 'All' if self.current_agg == 'cbase' else 'sum'
            self.y_agg_vals = self.ydef
        elif self.current_agg in ['y_sum', 'rbase']:
            self.x_agg_vals = self.xdef
            self.y_agg_vals = 'All' if self.current_agg == 'rbase' else 'sum'
        elif self._res_is_stat():
            if self.factorized == 'x':
                self.x_agg_vals = self.current_agg
                self.y_agg_vals = self.ydef if not self.comb_y else self.comb_y
            else:
                self.x_agg_vals = self.xdef if not self.comb_x else self.comb_x
                self.y_agg_vals = self.current_agg
        # can this made smarter WITHOUT 1000000 IF-ELSEs above?:
        if ((self.current_agg in ['freq', 'cbase', 'x_sum', 'summary', 'calc'] or
                self._res_is_stat()) and not self.type == 'array'):
            if self.y == '@' or self.x == '@':
                self.y_agg_vals = '@'
        df = pd.DataFrame(self.result)
        idx, cols = self._make_multiindex()
        df.index = idx
        df.columns = cols
        self.result = df if not self.x == '@' else df.T
        if self.type == 'nested':
            self._format_nested_axis()
        return self

    def _make_multiindex(self):
        x_grps = self.x_agg_vals
        y_grps = self.y_agg_vals
        if not isinstance(x_grps, list):
            x_grps = [x_grps]
        if not isinstance(y_grps, list):
            y_grps = [y_grps]
        if not x_grps: x_grps = [None]
        if not y_grps: y_grps = [None]
        if self._has_x_margin:
            x_grps = ['All'] + x_grps
        if self._has_y_margin:
            y_grps = ['All'] + y_grps
        if self.type == 'array':
            x_unit = y_unit = self.x
            x_names = ['Question', 'Values']
            y_names = ['Array', 'Questions']
        else:
            x_unit = self.x if not self.x == '@' else self.y
            y_unit = self.y if not self.y == '@' else self.x
            x_names = y_names = ['Question', 'Values']
        x = [x_unit, x_grps]
        y = [y_unit, y_grps]
        index = pd.MultiIndex.from_product(x, names=x_names)
        columns = pd.MultiIndex.from_product(y, names=y_names)
        return index, columns

    def _format_nested_axis(self):
        nest_mi = self._make_nest_multiindex()
        if not len(self.result.columns) > len(nest_mi.values):
            self.result.columns = nest_mi
        else:
            total_mi_values = []
            for var in self.nest_def['variables']:
                total_mi_values += [var, -1]
            total_mi = pd.MultiIndex.from_product(total_mi_values,
                                                  names=nest_mi.names)
            full_nest_mi = nest_mi.union(total_mi)
            for lvl, c in zip(range(1, len(full_nest_mi)+1, 2),
                              self.nest_def['level_codes']):
                full_nest_mi.set_levels(['All'] + c, level=lvl, inplace=True)
            self.result.columns = full_nest_mi
        return None

    def _make_nest_multiindex(self):
        values = []
        names = ['Question', 'Values'] * (self.nest_def['levels'])
        for lvl_var, lvl_c in zip(self.nest_def['variables'],
                                  self.nest_def['level_codes']):
            values.append(lvl_var)
            values.append(lvl_c)
        mi = pd.MultiIndex.from_product(values, names=names)
        return mi

    def normalize(self, on='y'):
        """
        Convert a raw cell count result to its percentage representation.

        Parameters
        ----------
        on : {'y', 'x'}, default 'y'
            Defines the base to normalize the result on. ``'y'`` will
            produce column percentages, ``'x'`` will produce row
            percentages.

        Returns
        -------
        self
            Updates an count-based aggregation in the ``result`` property.
        """
        if self.x == '@':
            on = 'y' if on == 'x' else 'x'
        if on == 'y':
            if self._has_y_margin or self.y == '@' or self.x == '@':
                base = self.cbase
            else:
                if self._get_type() == 'array':
                    base = self.cbase
                else:
                    base = self.cbase[:, 1:]
        else:
            if self._has_x_margin:
                base = self.rbase
            else:
                base = self.rbase[1:, :]
        if isinstance(self.result, pd.DataFrame):
            if self.x == '@':
                self.result = self.result.T
            if on == 'y':
                base = np.repeat(base, self.result.shape[0], axis=0)
            else:
                base = np.repeat(base, self.result.shape[1], axis=1)
        self.result = self.result / base * 100
        if self.x == '@':
            self.result = self.result.T
        return self

    def rebase(self, reference, on='counts', overwrite_margins=True):
        """
        """
        val_err = 'No frequency aggregation to rebase.'
        if self.result is None:
            raise ValueError(val_err)
        elif self.current_agg != 'freq':
            raise ValueError(val_err)
        is_df = self._force_to_nparray()
        has_margin = self._attach_margins()
        ref = self.swap(var=reference, inplace=False)
        if self._sects_identical(self.xdef, ref.xdef):
            pass
        elif self._sects_different_order(self.xdef, ref.xdef):
            ref.xdef = self.xdef
            ref._x_indexers = ref._get_x_indexers()
            ref.matrix = ref.matrix[:, ref._x_indexers + [0]]
        elif self._sect_is_subset(self.xdef, ref.xdef):
            ref.xdef = [code for code in ref.xdef if code in self.xdef]
            ref._x_indexers = ref._sort_indexer_as_codes(ref._x_indexers,
                                                         self.xdef)
            ref.matrix = ref.matrix[:, [0] + ref._x_indexers]
        else:
            idx_err = 'Axis defintion is not a subset of rebase reference.'
            raise IndexError(idx_err)
        ref_freq = ref.count(as_df=False)
        self.result = (self.result/ref_freq.result) * 100
        if overwrite_margins:
            self.rbase = ref_freq.rbase
            self.cbase = ref_freq.cbase
        self._organize_margins(has_margin)
        if is_df: self.to_df()
        return self

    @staticmethod
    def _sects_identical(axdef1, axdef2):
        return axdef1 == axdef2

    @staticmethod
    def _sects_different_order(axdef1, axdef2):
        if not len(axdef1) == len(axdef2):
            return False
        else:
            if (x for x in axdef1 if x in axdef2):
                return True
            else:
                return False

    @staticmethod
    def _sect_is_subset(axdef1, axdef2):
        return set(axdef1).intersection(set(axdef2)) > 0

class Test(object):
    """
    The Quantipy Test object is a defined by a Link and the view name notation
    string of a counts or means view. All auxiliary figures needed to arrive
    at the test results are computed inside the instance of the object.
    """
    def __init__(self, link, view_name_notation, test_total=False):
        super(Test, self).__init__()
        # Infer whether a mean or proportion test is being performed
        view = link[view_name_notation]
        if view.meta()['agg']['method'] == 'descriptives':
            self.metric = 'means'
        else:
            self.metric = 'proportions'
        self.invalid = None
        self.no_pairs = None
        self.no_diffs = None
        self.parameters = None
        self.test_total = test_total
        self.mimic = None
        self.level = None
        # Calculate the required baseline measures for the test using the
        # Quantity instance
        self.Quantity = qp.Quantity(link, view.weights(), use_meta=True,
                                    base_all=self.test_total)
        self._set_baseline_aggregates(view)
        # Set information about the incoming aggregation
        # to be able to route correctly through the algorithms
        # and re-construct a Quantipy-indexed pd.DataFrame
        self.is_weighted = view.meta()['agg']['is_weighted']
        self.has_calc = view.has_calc()
        self.x = view.meta()['x']['name']
        self.xdef = view.dataframe.index.get_level_values(1).tolist()
        self.y = view.meta()['y']['name']
        self.ydef = view.dataframe.columns.get_level_values(1).tolist()
        columns_to_pair = ['@'] + self.ydef if self.test_total else self.ydef
        self.ypairs = list(combinations(columns_to_pair, 2))
        self.y_is_multi = view.meta()['y']['is_multi']
        self.multiindex = (view.dataframe.index, view.dataframe.columns)

    def __repr__(self):
        return ('%s, total included: %s, test metric: %s, parameters: %s, '
                'mimicked: %s, level: %s ')\
                % (Test, self.test_total, self.metric, self.parameters,
                   self.mimic, self.level)

    def _set_baseline_aggregates(self, view):
        """
        Derive or recompute the basic values required by the ``Test`` instance.
        """
        grps, exp, compl, calc, exclude, rescale = view.get_edit_params()
        if exclude is not None:
            self.Quantity.exclude(exclude)
        if self.metric == 'proportions' and self.test_total and view._has_code_expr():
            self.Quantity.group(grps, expand=exp, complete=compl)
        if self.metric == 'means':
            aggs = self.Quantity._dispersion(_return_mean=True,
                                             _return_base=True)
            self.sd, self.values, self.cbases = aggs[0], aggs[1], aggs[2]
            if not self.test_total:
                self.sd = self.sd[:, 1:]
                self.values = self.values[:, 1:]
                self.cbases = self.cbases[:, 1:]
        elif self.metric == 'proportions':
            if not self.test_total:
                self.values = view.dataframe.values.copy()
                self.cbases = view.cbases[:, 1:]
                self.rbases = view.rbases[1:, :]
                self.tbase = view.cbases[0, 0]
            else:
                agg = self.Quantity.count(margin=True, as_df=False)
                if calc is not None:
                    calc_only = view._kwargs.get('calc_only', False)
                    self.Quantity.calc(calc, axis='x', result_only=calc_only)
                self.values = agg.result[1:, :]
                self.cbases = agg.cbase
                self.rbases = agg.rbase[1:, :]
                self.tbase = agg.cbase[0, 0]

    def set_params(self, test_total=False, level='mid', mimic='Dim', testtype='pooled',
                   use_ebase=True, ovlp_correc=True, cwi_filter=False,
                   flag_bases=None):
        """
        Sets the test algorithm parameters and defines the type of test.

        This method sets the test's global parameters and derives the
        necessary measures for the computation of the test statistic.
        The default values correspond to the SPSS Dimensions Column Tests
        algorithms that control for bias introduced by weighting and
        overlapping samples in the column pairs of multi-coded questions.

        .. note:: The Dimensions implementation uses variance pooling.

        Parameters
        ----------
        test_total : bool, default False
            If set to True, the test algorithms will also include an existent
            total (@-) version of the original link and test against the
            unconditial data distribution.
        level : str or float, default 'mid'
            The level of significance given either as per 'low' = 0.1,
            'mid' = 0.05, 'high' = 0.01 or as specific float, e.g. 0.15.
        mimic : {'askia', 'Dim'} default='Dim'
            Will instruct the mimicking of a software specific test.
        testtype : str, default 'pooled'
            Global definition of the tests.
        use_ebase : bool, default True
            If True, will use the effective sample sizes instead of the
            the simple weighted ones when testing a weighted aggregation.
        ovlp_correc : bool, default True
            If True, will consider and correct for respondent overlap when
            testing between multi-coded column pairs.
        cwi_filter : bool, default False
            If True, will check an incoming count aggregation for cells that
            fall below a treshhold comparison aggregation that assumes counts
            to be independent.
        flag_bases : list of two int, default None
            If provided, the output dataframe will replace results that have
            been calculated on (eff.) bases below the first int with ``'**'``
            and mark results in columns with bases below the second int with
            ``'*'``

        Returns
        -------
        self
        """
        # Check if the aggregation is non-empty
        # and that there are >1 populated columns
        if np.nansum(self.values) == 0 or len(self.ydef) == 1:
            self.invalid = True
            if np.nansum(self.values) == 0:
                self.no_diffs = True
            if len(self.ydef) == 1:
                self.no_pairs = True
            self.mimic = mimic
            self.comparevalue, self.level = self._convert_level(level)
        else:
            # Set global test algorithm parameters
            self.invalid = False
            self.no_diffs = False
            self.no_pairs = False
            valid_mimics = ['Dim', 'askia']
            if mimic not in valid_mimics:
                raise ValueError('Failed to mimic: "%s". Select from: %s\n'
                                 % (mimic, valid_mimics))
            else:
                self.mimic = mimic
            if self.mimic == 'askia':
                self.parameters = {'testtype': 'unpooled',
                                   'use_ebase': False,
                                   'ovlp_correc': False,
                                   'cwi_filter': True,
                                   'base_flags': None}
                self.test_total = False
            elif self.mimic == 'Dim':
                self.parameters = {'testtype': 'pooled',
                                   'use_ebase': True,
                                   'ovlp_correc': True,
                                   'cwi_filter': False,
                                   'base_flags': flag_bases}
            self.level = level
            self.comparevalue, self.level = self._convert_level(level)
            # Get value differences between column pairings
            if self.metric == 'means':
                self.valdiffs = np.array(
                    [m1 - m2 for m1, m2 in combinations(self.values[0], 2)])
            if self.metric == 'proportions':
                # special to askia testing: counts-when-independent filtering
                if cwi_filter:
                    self.values = self._cwi()
                props = (self.values / self.cbases).T
                self.valdiffs = np.array([p1 - p2 for p1, p2
                                          in combinations(props, 2)]).T
            # Set test specific measures for Dimensions-like testing:
            # [1] effective base usage
            if use_ebase and self.is_weighted:
                if not self.test_total:
                    self.ebases = self.Quantity._effective_n(axis='x', margin=False)
                else:
                    self.ebases = self.Quantity._effective_n(axis='x', margin=True)
            else:
                self.ebases = self.cbases
            # [2] overlap correction
            if self.y_is_multi and self.parameters['ovlp_correc']:
                self.overlap = self._overlap()
            else:
                self.overlap = np.zeros(self.valdiffs.shape)
            # [3] base flags
            if flag_bases:
                self.flags = {'min': flag_bases[0],
                              'small': flag_bases[1]}
                self.flags['flagged_bases'] = self._get_base_flags()
            else:
                self.flags = None
        return self

    # -------------------------------------------------
    # Main algorithm methods to compute test statistics
    # -------------------------------------------------
    def run(self):
        """
        Performs the testing algorithm and creates an output pd.DataFrame.

        The output is indexed according to Quantipy's Questions->Values
        convention. Significant results between columns are presented as
        lists of integer y-axis codes where the column with the higher value
        is holding the codes of the columns with the lower values. NaN is
        indicating that a cell is not holding any sig. higher values
        compared to the others.
        """
        if not self.invalid:
            sigs = self.get_sig()
            return self._output(sigs)
        else:
            return self._empty_output()

    def get_sig(self):
        """
        TODO: implement returning tstats only.
        """
        stat = self.get_statistic()
        stat = self._convert_statistic(stat)
        if self.metric == 'means':
            diffs = pd.DataFrame(self.valdiffs, index=self.ypairs, columns=self.xdef).T
        elif self.metric == 'proportions':
            stat = pd.DataFrame(stat, index=self.xdef, columns=self.ypairs)
            diffs = pd.DataFrame(self.valdiffs, index=self.xdef, columns=self.ypairs)
        if self.mimic == 'Dim':
            return diffs[(diffs != 0) & (stat < self.comparevalue)]
        elif self.mimic == 'askia':
            return diffs[(diffs != 0) & (stat > self.comparevalue)]

    def get_statistic(self):
        """
        Returns the test statistic of the algorithm.
        """
        return self.valdiffs / self.get_se()

    def get_se(self):
        """
        Compute the standard error (se) estimate of the tested metric.

        The calculation of the se is defined by the parameters of the setup.
        The main difference is the handling of variances. **unpooled**
        implicitly assumes variance inhomogenity between the column pairing's
        samples. **pooled** treats variances effectively as equal.
        """
        if self.metric == 'means':
            if self.parameters['testtype'] == 'unpooled':
                return self._se_mean_unpooled()
            elif self.parameters['testtype'] == 'pooled':
                return self._se_mean_pooled()
        elif self.metric == 'proportions':
            if self.parameters['testtype'] == 'unpooled':
                return self._se_prop_unpooled()
            if self.parameters['testtype'] == 'pooled':
                return self._se_prop_pooled()

    # -------------------------------------------------
    # Conversion methods for levels and statistics
    # -------------------------------------------------
    def _convert_statistic(self, teststat):
        """
        Convert test statistics to match the decision rule of the test logic.

        Either transforms to p-values or returns the absolute value of the
        statistic, depending on the decision rule of the test.
        This is used to mimic other software packages as some tests'
        decision rules check test-statistic against pre-defined treshholds
        while others check sig. level against p-value.
        """
        if self.mimic == 'Dim':
            ebases_pairs = [eb1 + eb2 for eb1, eb2
                            in combinations(self.ebases[0], 2)]
            dof = ebases_pairs - self.overlap - 2
            dof[dof <= 1] = np.NaN
            return get_pval(dof, teststat)[1]
        elif self.mimic == 'askia':
            return abs(teststat)

    def _convert_level(self, level):
        """
        Determines the comparison value for the test's decision rule.

        Checks whether the level of test is a string that defines low, medium,
        or high significance or an "actual" level of significance and
        converts it to a comparison level/significance level tuple.
        This is used to mimic other software packages as some test's
        decision rules check test-statistic against pre-defined treshholds
        while others check sig. level against p-value.
        """
        if isinstance(level, (str, unicode)):
            if level == 'low':
                if self.mimic == 'Dim':
                    comparevalue = siglevel = 0.10
                elif self.mimic == 'askia':
                    comparevalue = 1.65
                    siglevel = 0.10
            elif level == 'mid':
                if self.mimic == 'Dim':
                    comparevalue = siglevel = 0.05
                elif self.mimic == 'askia':
                    comparevalue = 1.96
                    siglevel = 0.05
            elif level == 'high':
                if self.mimic == 'Dim':
                    comparevalue = siglevel = 0.01
                elif self.mimic == 'askia':
                    comparevalue = 2.576
                    siglevel = 0.01
        else:
            if self.mimic == 'Dim':
                comparevalue = siglevel = level
            elif self.mimic == 'askia':
                comparevalue = 1.65
                siglevel = 0.10

        return comparevalue, siglevel

    # -------------------------------------------------
    # Standard error estimates calculation methods
    # -------------------------------------------------
    def _se_prop_unpooled(self):
        """
        Estimated standard errors of prop. diff. (unpool. var.) per col. pair.
        """
        props = self.values/self.cbases
        unp_sd = ((props*(1-props))/self.cbases).T
        return np.array([np.sqrt(cat1 + cat2)
                         for cat1, cat2 in combinations(unp_sd, 2)]).T

    def _se_mean_unpooled(self):
        """
        Estimated standard errors of mean diff. (unpool. var.) per col. pair.
        """
        sd_base_ratio = self.sd / self.cbases
        return np.array([np.sqrt(sd_b_r1 + sd_b_r2)
                         for sd_b_r1, sd_b_r2
                         in combinations(sd_base_ratio[0], 2)])[None, :]

    def _se_prop_pooled(self):
        """
        Estimated standard errors of prop. diff. (pooled var.) per col. pair.

        Controlling for effective base sizes and overlap responses is
        supported and applied as defined by the test's parameters setup.
        """
        ebases_correc_pairs = np.array([1 / x + 1 / y
                                        for x, y
                                        in combinations(self.ebases[0], 2)])

        if self.y_is_multi and self.parameters['ovlp_correc']:
            ovlp_correc_pairs = ((2 * self.overlap) /
                                 [x * y for x, y
                                  in combinations(self.ebases[0], 2)])
        else:
            ovlp_correc_pairs = self.overlap

        counts_sum_pairs = np.array(
            [c1 + c2 for c1, c2 in combinations(self.values.T, 2)])
        bases_sum_pairs = np.expand_dims(
            [b1 + b2 for b1, b2 in combinations(self.cbases[0], 2)], 1)
        pooled_props = (counts_sum_pairs/bases_sum_pairs).T
        return (np.sqrt(pooled_props * (1 - pooled_props) *
                (np.array(ebases_correc_pairs - ovlp_correc_pairs))))

    def _se_mean_pooled(self):
        """
        Estimated standard errors of mean diff. (pooled var.) per col. pair.

        Controlling for effective base sizes and overlap responses is
        supported and applied as defined by the test's parameters setup.
        """
        ssw_base_ratios = self._sum_sq_w(base_ratio=True)
        enum = np.nan_to_num((self.sd ** 2) * (self.cbases-1))
        denom = self.cbases-ssw_base_ratios

        enum_pairs = np.array([enum1 + enum2
                               for enum1, enum2
                               in combinations(enum[0], 2)])
        denom_pairs = np.array([denom1 + denom2
                                for denom1, denom2
                                in combinations(denom[0], 2)])

        ebases_correc_pairs = np.array([1/x + 1/y
                                        for x, y
                                        in combinations(self.ebases[0], 2)])

        if self.y_is_multi and self.parameters['ovlp_correc']:
            ovlp_correc_pairs = ((2*self.overlap) /
                                 [x * y for x, y
                                  in combinations(self.ebases[0], 2)])
        else:
            ovlp_correc_pairs = self.overlap[None, :]

        return (np.sqrt((enum_pairs/denom_pairs) *
                        (ebases_correc_pairs - ovlp_correc_pairs)))

    # -------------------------------------------------
    # Specific algorithm values & test option measures
    # -------------------------------------------------
    def _sum_sq_w(self, base_ratio=True):
        """
        """
        if not self.Quantity.w == '@1':
            self.Quantity.weight()
        if not self.test_total:
            ssw = np.nansum(self.Quantity.matrix ** 2, axis=0)[[0], 1:]
        else:
            ssw = np.nansum(self.Quantity.matrix ** 2, axis=0)[[0], :]
        if base_ratio:
            return ssw/self.cbases
        else:
            return ssw

    def _cwi(self, threshold=5, as_df=False):
        """
        Derives the count distribution assuming independence between columns.
        """
        c_col_n = self.cbases
        c_cell_n = self.values
        t_col_n = self.tbase
        if self.rbases.shape[1] > 1:
            t_cell_n = self.rbases[1:, :]
        else:
            t_cell_n = self.rbases[0]
        np.place(t_col_n, t_col_n == 0, np.NaN)
        np.place(t_cell_n, t_cell_n == 0, np.NaN)
        np.place(c_col_n, c_col_n == 0, np.NaN)
        np.place(c_cell_n, c_cell_n == 0, np.NaN)
        cwi = (t_cell_n * c_col_n) / t_col_n
        cwi[cwi < threshold] = np.NaN
        if as_df:
            return pd.DataFrame(c_cell_n + cwi - cwi,
                                index=self.xdef, columns=self.ydef)
        else:
            return c_cell_n + cwi - cwi

    def _overlap(self):
        if self.is_weighted:
            self.Quantity.weight()
        m = self.Quantity.matrix.copy()
        m = np.nansum(m, 1) if self.test_total else np.nansum(m[:, 1:, 1:], 1)
        if not self.is_weighted:
            m /= m
        m[m == 0] = np.NaN
        col_pairs = list(combinations(range(0, m.shape[1]), 2))
        if self.parameters['use_ebase'] and self.is_weighted:
            # Overlap computation when effective base is being used
            w_sum_sq = np.array([np.nansum(m[:, [c1]] + m[:, [c2]], axis=0)**2
                                 for c1, c2 in col_pairs])
            w_sq_sum = np.array([np.nansum(m[:, [c1]]**2 + m[:, [c2]]**2, axis=0)
                        for c1, c2 in col_pairs])
            return np.nan_to_num((w_sum_sq/w_sq_sum)/2).T
        else:
            # Overlap with simple weighted/unweighted base size
            ovlp = np.array([np.nansum(m[:, [c1]] + m[:, [c2]], axis=0)
                             for c1, c2 in col_pairs])
            return (np.nan_to_num(ovlp)/2).T

    def _get_base_flags(self):
        bases = self.ebases[0]
        small = self.flags['small']
        minimum = self.flags['min']
        flags = []
        for base in bases:
            if base >= small:
                flags.append('')
            elif base < small and base >= minimum:
                flags.append('*')
            else:
                flags.append('**')
        return flags

    # -------------------------------------------------
    # Output creation
    # -------------------------------------------------
    def _output(self, sigs):
        res = {y: {x: [] for x in self.xdef} for y in self.ydef}
        test_columns = ['@'] + self.ydef if self.test_total else self.ydef
        for col, val in sigs.iteritems():
            if self._flags_exist():
                b1ix, b2ix = test_columns.index(col[0]), test_columns.index(col[1])
                b1_ok = self.flags['flagged_bases'][b1ix] != '**'
                b2_ok = self.flags['flagged_bases'][b2ix] != '**'
            else:
                b1_ok, b2_ok = True, True
            for row, v in val.iteritems():
                if v > 0:
                    if b2_ok:
                        if col[0] == '@':
                            res[col[1]][row].append('@H')
                        else:
                            res[col[0]][row].append(col[1])
                if v < 0:
                    if b1_ok:
                        if col[0] == '@':
                            res[col[1]][row].append('@L')
                        else:
                            res[col[1]][row].append(col[0])
        test = pd.DataFrame(res).applymap(lambda x: str(x))
        test = test.reindex(index=self.xdef, columns=self.ydef)
        if self._flags_exist():
           test = self._apply_base_flags(test)
           test.replace('[]*', '*', inplace=True)
        test.replace('[]', np.NaN, inplace=True)
        # removing test results on post-aggregation rows [calc()]
        if self.has_calc:
            if len(test.index) > 1:
                test.iloc[-1:, :] = np.NaN
            else:
                test.iloc[:, :] = np.NaN
        test.index, test.columns = self.multiindex[0], self.multiindex[1]
        return test

    def _empty_output(self):
        """
        """
        values = self.values
        if self.metric == 'proportions':
            if self.no_pairs or self.no_diffs:
                values[:] = np.NaN
            if values.shape == (1, 1) or values.shape == (1, 0):
                values = [np.NaN]
        if self.metric == 'means':
            if self.no_pairs:
                values = [np.NaN]
            if self.no_diffs and not self.no_pairs:
                values[:] = np.NaN
        return  pd.DataFrame(values,
                             index=self.multiindex[0],
                             columns=self.multiindex[1])
    def _flags_exist(self):
        return (self.flags is not None and
                not all(self.flags['flagged_bases']) == '')

    def _apply_base_flags(self, sigres, replace=True):
        flags = self.flags['flagged_bases']
        if self.test_total: flags = flags[1:]
        for res_col, flag in zip(sigres.columns, flags):
                if flag == '**':
                    if replace:
                        sigres[res_col] = flag
                    else:
                        sigres[res_col] = sigres[res_col] + flag
                elif flag == '*':
                    sigres[res_col] = sigres[res_col] + flag
        return sigres

class Nest(object):
    """
    Description of class...
    """
    def __init__(self, nest, data, meta):
        self.data = data
        self.meta = meta
        self.name = nest
        self.variables = nest.split('>')
        self.levels = len(self.variables)
        self.level_codes = []
        self.code_maps = None
        self._needs_multi = self._any_multicoded()

    def nest(self):
        self._get_nested_meta()
        self._get_code_maps()
        interlocked = self._interlock_codes()
        if not self.name in self.data.columns:
            recode_map = {code: intersection(code_pair) for code, code_pair
                          in enumerate(interlocked, start=1)}
            self.data[self.name] = np.NaN
            self.data[self.name] = recode(self.meta, self.data,
                                          target=self.name, mapper=recode_map)
        nest_info = {'variables': self.variables,
                     'level_codes': self.level_codes,
                     'levels': self.levels}
        return nest_info

    def _any_multicoded(self):
        return any(self.data[self.variables].dtypes == 'object')

    def _get_code_maps(self):
        code_maps = []
        for level, var in enumerate(self.variables):
            mapping = [{var: [int(code)]} for code
                       in self.level_codes[level]]
            code_maps.append(mapping)
        self.code_maps = code_maps
        return None

    def _interlock_codes(self):
        return list(product(*self.code_maps))

    def _get_nested_meta(self):
        meta_dict = {}
        qtext, valtexts = self._interlock_texts()
        meta_dict['type'] = 'delimited set' if self._needs_multi else 'single'
        meta_dict['text'] = {'en-GB': '>'.join(qtext[0])}
        meta_dict['values'] = [{'text' : {'en-GB': '>'.join(valtext)},
                                'value': c}
                               for c, valtext
                               in enumerate(valtexts, start=1)]
        self.meta['columns'][self.name] = meta_dict
        return None

    def _interlock_texts(self):
        all_valtexts = []
        all_qtexts = []
        for var in self.variables:
            var_valtexts = []
            values = self.meta['columns'][var]['values']
            all_qtexts.append(self.meta['columns'][var]['text'].values())
            for value in values:
                var_valtexts.append(value['text'].values()[0])
            all_valtexts.append(var_valtexts)
            self.level_codes.append([code['value'] for code in values])
        interlocked_valtexts = list(product(*all_valtexts))
        interlocked_qtexts = list(product(*all_qtexts))
        return interlocked_qtexts, interlocked_valtexts

##############################################################################

class Multivariate(object):
    def __init__(self):
        pass

    def _select_variables(self, x, y=None, w=None, drop_listwise=False):
        x_vars, y_vars = [], []
        if not isinstance(x, list): x = [x]
        if not isinstance(y, list) and not y=='@': y = [y]
        if w is None: w = '@1'
        wrong_var_sel_1_on_1 = 'Can only analyze 1-to-1 relationships.'
        if self.analysis == 'Reduction' and (not (len(x) == 1 and len(y) == 1) or y=='@'):
            raise AttributeError(wrong_var_sel_1_on_1)
        for var in x:
            if self.ds._is_array(var):
                if self.analysis == 'Reduction': raise AttributeError(wrong_var_sel_1_on_1)
                x_a_items = self.ds._get_itemmap(var, non_mapped='items')
                x_vars += x_a_items
            else:
                x_vars.append(var)
        if y and not y == '@':
            for var in y:
                if self.ds._is_array(var):
                    if self.analysis == 'Reduction': raise AttributeError(wrong_var_sel_1_on_1)
                    y_a_items = self.ds._get_itemmap(var, non_mapped='items')
                    y_vars += y_a_items
                else:
                    y_vars.append(var)
        elif y == '@':
            y_vars = x_vars
        if x_vars == y_vars or y is None:
            data_slice = x_vars + [w]
        else:
            data_slice = x_vars + y_vars + [w]
        if self.analysis == 'Relations' and y != '@':
            self.x = self.y = x_vars + y_vars
            self._org_x, self._org_y = x_vars, y_vars
        else:
            self.x = self._org_x = x_vars
            self.y = self._org_y = y_vars
        self.w = w
        self._analysisdata = self.ds[data_slice]
        self._drop_missings()
        if drop_listwise:
            self._analysisdata.dropna(inplace=True)
            valid = self._analysisdata.index
            self.ds._data = self.ds._data.ix[valid, :]
        return None

    def _drop_missings(self):
        data = self._analysisdata.copy()
        for var in data.columns:
            if self.ds._has_missings(var):
                drop = self.ds._get_missing_list(var, globally=False)
                data[var].replace(drop, np.NaN, inplace=True)
        self._analysisdata = data
        return None

    def _has_analysis_data(self):
        if not hasattr(self, '_analysisdata'):
            raise AttributeError('No analysis variables assigned!')

    def _has_yvar(self):
        if self.y is None:
            raise AttributeError('Must select at least one y-variable or '
                                 '"@"-matrix indicator!')

    def _get_quantities(self, create='all'):
        crossed_quantities = []
        single_quantities = []
        helper_stack = qp.Stack()
        helper_stack.add_data(self.ds.name, self.ds._data, self.ds._meta)
        w = self.w if self.w != '@1' else None

        for x, y in product(self.x, self.y):
            helper_stack.add_link(x=x, y=y)
            l = helper_stack[self.ds.name]['no_filter'][x][y]
            crossed_quantities.append(qp.Quantity(l, weight=w))

        for x in self._org_x+self._org_y:
            helper_stack.add_link(x=x, y='@')
            l = helper_stack[self.ds.name]['no_filter'][x]['@']
            single_quantities.append(qp.Quantity(l, weight=w))

        self.single_quantities = single_quantities
        self.crossed_quantities = crossed_quantities
        return None

class Reductions(Multivariate):
    def __init__(self, dataset):
        self.ds = dataset
        self.single_quantities = None
        self.crossed_quantities = None
        self.analysis = 'Reduction'

    def plot(self, type, point_coords):
        plt.set_autoscale_on = False
        plt.figure(figsize=(5, 5))
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        #plt.axvline(x=0.0, c='grey', ls='solid', linewidth=0.9)
        #plt.axhline(y=0.0, c='grey', ls='solid', linewidth=0.9)
        x = plt.scatter(point_coords['x'][0], point_coords['x'][1],
                        edgecolor='w', marker='o', c='red', s=20)
        y = plt.scatter(point_coords['y'][0], point_coords['y'][1],
                        edgecolor='k', marker='^', c='lightgrey', s=20)

        fig = x.get_figure()
        # print fig.get_axes()[0].grid()
        fig.get_axes()[0].tick_params(labelsize=6)
        fig.get_axes()[0].patch.set_facecolor('w')

        fig.get_axes()[0].grid(which='major', linestyle='solid', color='grey',
                               linewidth=0.6)

        fig.get_axes()[0].xaxis.get_major_ticks()[0].label1.set_visible(False)
        x0 = fig.get_axes()[0].get_position().x0
        y0 = fig.get_axes()[0].get_position().y0
        x1 = fig.get_axes()[0].get_position().x1
        y1 = fig.get_axes()[0].get_position().y1

        text = 'Correspondence map'
        plt.figtext(x0+0.015, 1.09-y0, text, fontsize=12, color='w',
                    fontweight='bold', verticalalignment='top',
                    bbox={'facecolor':'red', 'alpha': 0.8, 'edgecolor': 'w',
                          'pad': 10})



        label_map = self._get_point_label_map('CA', point_coords)
        for axis in label_map.keys():
            for lab, coord in label_map[axis].items():
                plt.annotate(lab, coord, ha='left', va='bottom',
                    fontsize=6)
            plt.legend((x, y), (self.x[0], self.y[0]),
                       loc='best', bbox_to_anchor=(1.325, 1.07),
                       ncol=2, fontsize=6, title='                         ')

        x_codes, x_texts = self.ds._get_valuemap(self.x[0], non_mapped='lists')
        y_codes, y_texts = self.ds._get_valuemap(self.y[0], non_mapped='lists')

        text = ' '*80
        for var in zip(x_codes, x_texts):
            text += '\n{}: {}\n'.format(var[0], var[1])
        fig.text(1.06-x0, 0.85, text, fontsize=5, verticalalignment='top',
                          bbox={'facecolor':'red',
                          'edgecolor': 'w', 'pad': 10})
        x_len = len(x_codes)
        text = ' '*80
        for var in zip(y_codes, y_texts):
            text += '\n{}: {}\n'.format(var[0], var[1])
        test = fig.text(1.06-x0, 0.85-((x_len)*0.0155)-((x_len)*0.0155)-0.05, text, fontsize=5, verticalalignment='top',
                          bbox={'facecolor': 'lightgrey', 'alpha': 0.65,
                          'edgecolor': 'w', 'pad': 10})

        logo = Image.open('C:/Users/alt/Documents/IPython Notebooks/Designs/Multivariate class/__resources__/YG_logo.png')
        newax = fig.add_axes([x0+0.005, y0-0.25, 0.1, 0.1], anchor='NE', zorder=-1)
        newax.imshow(logo)
        newax.axis('off')

        fig.savefig(self.ds.path + 'correspond.png', bbox_inches='tight', dpi=300)

    def correspondence(self, x, y, w=None, norm='sym', diags=True, plot=True):
        """
        Perform a (multiple) correspondence analysis.

        Parameters
        ----------
        norm : {'sym', 'princ'}, default 'sym'
            <DESCP>
        summary : bool, default True
            If True, the output will contain a dataframe that summarizes core
            information about the Inertia decomposition.
        plot : bool, default False
            If set to True, a correspondence map plot will be saved in the
            Stack's data path location.
        Returns
        -------
        results: pd.DataFrame
            Summary of analysis results.
        """
        self._select_variables(x, y, w)
        self._get_quantities()
        # 1. Chi^2 analysis
        obs, exp = self.expected_counts(x=x, y=y, return_observed=True)
        chisq, sig = self.chi_sq(x=x, y=y, sig=True)
        inertia = chisq / np.nansum(obs)
        # 2. svd on standardized residuals
        std_residuals = ((obs - exp) / np.sqrt(exp)) / np.sqrt(np.nansum(obs))
        sv, row_eigen_mat, col_eigen_mat, ev = self._svd(std_residuals)
        # 3. row and column coordinates
        a = 0.5 if norm == 'sym' else 1.0
        row_mass = self.mass(x=x, y=y, margin='x')
        col_mass = self.mass(x=x, y=y, margin='y')
        dim = min(row_mass.shape[0]-1, col_mass.shape[0]-1)
        row_sc = (row_eigen_mat * sv[:, 0] ** a) / np.sqrt(row_mass)
        col_sc = (col_eigen_mat.T * sv[:, 0] ** a) / np.sqrt(col_mass)

        if plot:
            # prep coordinates for plot
            item_sep = len(self.single_quantities[0].xdef)
            dim1_c = [r_s[0] for r_s in row_sc] + [c_s[0] for c_s in col_sc]
            # dim2_c = [r_s[1]*(-1) for r_s in row_sc] + [c_s[1]*(-1) for c_s in col_sc]
            dim2_c = [r_s[1] for r_s in row_sc] + [c_s[1] for c_s in col_sc]
            dim1_xitem, dim2_xitem = dim1_c[:item_sep], dim2_c[:item_sep]
            dim1_yitem, dim2_yitem = dim1_c[item_sep:], dim2_c[item_sep:]
            coords = {'x': [dim1_xitem, dim2_xitem],
                      'y': [dim1_yitem, dim2_yitem]}
            self.plot('CA', coords)
            plt.show()

        if diags:
            _dim = xrange(1, dim+1)
            chisq_stats = [chisq, 'sig: {}'.format(sig),
                           'dof: {}'.format((obs.shape[0] - 1)*(obs.shape[1] - 1))]
            _chisq = ([np.NaN] * (dim-3)) + chisq_stats
            _sig = ([np.NaN] * (dim-2)) + [chisq]
            _sv, _ev = sv[:dim, 0], ev[:dim, 0]
            _expl_inertia = 100 * (ev[:dim, 0] / inertia)
            _cumul_expl_inertia = np.cumsum(_expl_inertia)
            _perc_chisq = _expl_inertia / 100 * chisq
            labels = ['Dimension', 'Singular values', 'Eigen values',
                     'explained % of Inertia', 'cumulative % explained',
                     'explained Chi^2', 'Total Chi^2']
            results = pd.DataFrame([_dim, _sv, _ev, _expl_inertia,
                                    _cumul_expl_inertia,_perc_chisq, _chisq]).T
            results.columns = labels
            results.set_index('Dimension', inplace=True)
            return results

    def _get_point_label_map(self, type, point_coords):
        if type == 'CA':
            xcoords = zip(point_coords['x'][0],point_coords['x'][1])
            xlabels = self.crossed_quantities[0].xdef
            x_point_map = {lab: coord for lab, coord in zip(xlabels, xcoords)}
            ycoords = zip(point_coords['y'][0], point_coords['y'][1])
            ylabels = self.crossed_quantities[0].ydef
            y_point_map = {lab: coord for lab, coord in zip(ylabels, ycoords)}
            return {'x': x_point_map, 'y': y_point_map}

    def mass(self, x, y, w=None, margin=None):
        """
        Compute rel. margins or total cell frequencies of a contigency table.
        """
        counts = self.crossed_quantities[0].count(margin=False)
        total = counts.cbase[0, 0]
        if margin is None:
            return counts.result.values / total
        elif margin == 'x':
            return  counts.rbase[1:, :] / total
        elif margin == 'y':
            return  (counts.cbase[:, 1:] / total).T

    def expected_counts(self, x, y, w=None, return_observed=False):
        """
        Compute expected cell distribution given observed absolute frequencies.
        """
        #self.single_quantities, self.crossed_quantities = self._get_quantities()
        counts = self.crossed_quantities[0].count(margin=False)
        total = counts.cbase[0, 0]
        row_m = counts.rbase[1:, :]
        col_m = counts.cbase[:, 1:]
        if not return_observed:
            return (row_m * col_m) / total
        else:
            return counts.result.values, (row_m * col_m) / total

    def chi_sq(self, x, y, w=None, sig=False, as_inertia=False):
        """
        Compute global Chi^2 statistic, optionally transformed into Inertia.
        """
        obs, exp = self.expected_counts(x=x, y=y, return_observed=True)
        diff_matrix = ((obs - exp)**2) / exp
        total_chi_sq = np.nansum(diff_matrix)
        if sig:
            dof = (obs.shape[0] - 1) * (obs.shape[1] - 1)
            sig_result = np.round(1 - chi2dist.cdf(total_chi_sq, dof), 3)
        if as_inertia: total_chi_sq /= np.nansum(obs)
        if sig:
            return total_chi_sq, sig_result
        else:
            return total_chi_sq

    def _svd(self, matrix, return_eigen_matrices=True, return_eigen=True):
        """
        Singular value decomposition wrapping np.linalg.svd().
        """
        u, s, v = np.linalg.svd(matrix, full_matrices=False)
        s = s[:, None]
        if not return_eigen:
            if return_eigen_matrices:
                return s, u, v
            else:
                return s
        else:
            if return_eigen_matrices:
                return s, u, v, (s ** 2)
            else:
                return s, (s ** 2)

class LinearModels(Multivariate):
    """
    OLS REGRESSION, ...
    """
    def __init__(self, dataset):
        self.ds = dataset.copy()
        self.single_quantities = None
        self.crossed_quantities = None
        self.analysis = 'LinearModels'

    def set_model(self, y, x, w=None, intercept=True):
        """
        """
        self._select_variables(x=x, y=y, w=w, drop_listwise=True)
        self._get_quantities()
        self._matrix = self.ds[self.y + self.x + [self.w]].dropna().values
        ymean = self.single_quantities[-1].summarize('mean', as_df=False)
        self._ymean = ymean.result[0, 0]
        self._use_intercept = intercept
        self.dofs = self._dofs()
        predictors = ' + '.join(self.x)
        if self._use_intercept: predictors = 'c + ' + predictors
        self.formula = '{} ~ {}'.format(y, predictors)
        return self

    def _dofs(self):
        """
        """
        correction = 1 if self._use_intercept else 0
        obs = self._matrix[:, -1].sum()
        tdof = obs - correction
        mdof = len(self.x)
        rdof = obs - mdof - correction
        return [tdof, mdof, rdof]

    def _vectors(self):
        """
        """
        w = self._matrix[:, [-1]]
        y = self._matrix[:, [0]]
        x = self._matrix[:, 1:-1]
        x = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
        return w, y, x

    def get_coefs(self, standardize=False):
        coefs = self._coefs() if not standardize else self._betas()
        coef_df = pd.DataFrame(coefs,
                               index = ['-c-'] + self.x
                               if self._use_intercept else self.x,
                               columns = ['b']
                               if not standardize else ['beta'])
        coef_df.replace(np.NaN, '', inplace=True)
        return coef_df

    def _betas(self):
        """
        """
        corr_mat = Relations(self.ds).corr(self.x, self.y, self.w, n=False, sig=None,
                            drop_listwise=True, matrixed=True)
        corr_mat = corr_mat.values
        predictors = corr_mat[:-1, :-1]
        y = corr_mat[:-1, [-1]]
        inv_predictors = np.linalg.inv(predictors)
        betas = inv_predictors.dot(y)
        if self._use_intercept:
            betas = np.vstack([[np.NaN], betas])
        return betas

    def _coefs(self):
        """
        """
        w, y, x = self._vectors()
        coefs = np.dot(np.linalg.inv(np.dot(x.T, x*w)), np.dot(x.T, y*w))
        return coefs

    def get_modelfit(self, r_sq=True):
        anova, fit_stats = self._sum_of_squares()
        dofs = np.round(np.array(self.dofs)[:, None], 0)
        anova_stats = np.hstack([anova, dofs, fit_stats])
        anova_df = pd.DataFrame(anova_stats,
                                index=['total', 'model', 'residual'],
                                columns=['sum of squares', 'dof', 'R', 'R^2'])
        anova_df.replace(np.NaN, '', inplace=True)
        return anova_df


    def _sum_of_squares(self):
        """
        """
        w, y, x = self._vectors()
        x_w = x*w
        hat = x_w.dot(np.dot(np.linalg.inv(np.dot(x.T, x_w)), x.T))
        tss  = (w*(y - self._ymean)**2).sum()[None]
        rss = y.T.dot(np.dot(np.eye(hat.shape[0])-hat, y*w))[0]
        ess = tss-rss
        all_ss = np.vstack([tss, ess, rss])
        rsq = np.vstack([[np.NaN], ess/tss, [np.NaN]])
        r = np.sqrt(rsq)
        all_rs = np.hstack([r, rsq])
        return all_ss, all_rs

    def estimate(self, estimator='ols', diags=True):
        """
        """
        # Wrap up the modularized computation methods
        coefs, betas = self.get_coefs(), self.get_coefs(True)
        modelfit = self.get_modelfit()
        # Compute diagnostics, i.e. standard errors and sig. of estimates/fit
        # prerequisites
        w, _, x = self._vectors()
        rss = modelfit.loc['residual', 'sum of squares']
        ess = modelfit.loc['model', 'sum of squares']
        # coefficients: std. errors, t-stats, sigs
        c_se = np.diagonal(np.sqrt(np.linalg.inv(np.dot(x.T,x*w)) *
                                   (rss/self.dofs[-1])))[None].T
        c_sigs = np.hstack(get_pval(self.dofs[-1], coefs/c_se))
        c_diags = np.round(np.hstack([c_se, c_sigs]), 6)
        c_diags_df = pd.DataFrame(c_diags, index=coefs.index,
                                  columns=['se', 't-stat', 'p'])
        # modelfit: se, F-stat, ...
        m_se = np.vstack([[np.NaN], np.sqrt(rss/self.dofs[-1]), [np.NaN]])
        m_fstat = np.vstack([[np.NaN],
                             (ess/self.dofs[1]) / (rss/self.dofs[-1]),
                             [np.NaN]])
        m_sigs = 1-fdist.cdf(m_fstat, self.dofs[1], self.dofs[-1])
        m_diags = np.round(np.hstack([m_se, m_fstat, m_sigs]), 6)
        m_diags_df = pd.DataFrame(m_diags, index=modelfit.index,
                                  columns=['se', 'F-stat', 'p'])
        # Put everything together
        parameter_results = pd.concat([coefs, betas, c_diags_df], axis=1)
        fit_summary = pd.concat([modelfit, m_diags_df], axis=1).replace(np.NaN, '')
        return parameter_results, fit_summary

    def _lmg_models_per_var(self):
        all_models = self._lmg_combs()
        models_by_var = {x: [] for x in self.x}
        for var in self.x:
            qualified_models = []
            for model in all_models:
                if var in model: qualified_models.append(model)
            for qualified_model in qualified_models:
                q_m = list(qualified_model)
                q_m.remove(var)
                models_by_var[var].append([qualified_model, q_m])
        return models_by_var

    def _lmg_combs(self):
        full = self.x
        lmg_combs = []
        for combine_no in xrange(1, len(full)):
            lmg_combs.extend([list(comb) for comb in
                              list(combinations(full, combine_no))])
        lmg_combs.append(full)
        return lmg_combs

    def _rsq_lmg_subset(self, subset):
        self.set_model(self.y, subset, self.w)
        anova = self.get_modelfit()
        return anova['R^2'].replace('', np.NaN).dropna().values[0]

    def lmg(self, norm=True, plot=False):
        known_rsq = {}
        x_results = {}
        full_len = len(self.x)
        cols = self.y + self.x
        self._analysisdata = self._analysisdata.copy().dropna(subset=cols)
        total_rsq = self._rsq_lmg_subset(self.x)
        all_models = self._lmg_models_per_var()
        model_max_no = len(all_models.keys()) * len(all_models.values()[0])
        # print 'LMG analysis on {} models started...'.format(model_max_no)
        for x, diff_models in all_models.items():
            group_results = {size: [] for size in xrange(1, full_len + 1)}
            for diff_model in diff_models:
                # sys.stdout.write('|')
                # sys.stdout.flush()
                if not diff_model[1]:
                    if tuple(diff_model[0]) in known_rsq.keys():
                        r1 = known_rsq[tuple(diff_model[0])]
                    else:
                        r1 = self._rsq_lmg_subset(diff_model[0])
                        known_rsq[tuple(diff_model[0])] = r1
                    group_results[len(diff_model[0])].append((r1))
                else:
                    if tuple(diff_model[0]) in known_rsq.keys():
                        r1 = known_rsq[tuple(diff_model[0])]
                    else:
                        r1 = self._rsq_lmg_subset(diff_model[0])
                        known_rsq[tuple(diff_model[0])] = r1
                    if tuple(diff_model[1]) in known_rsq.keys():
                        r2 = known_rsq[tuple(diff_model[1])]
                    else:
                        r2 = self._rsq_lmg_subset(diff_model[1])
                        known_rsq[tuple(diff_model[1])] = r2
                    group_results[len(diff_model[0])].append((r1-r2))
            x_results[x] = group_results
        lmgs = []
        for var, results in x_results.items():
            res = np.mean([np.mean(val) for val in results.values()])
            lmgs.append((var, res))
            labs = ['Variable',
                    'Importance {}'.format('(normalized)' if norm else '')]
            result = pd.DataFrame(lmgs, columns=labs)
            result.set_index('Variable', inplace=True)
            result.index.name = 'LMG analysis'
            result.sort(columns=labs[1], ascending=False, inplace=True)
            if norm:
                result = result / total_rsq * 100
        # print '\n'
        return result

class Relations(Multivariate):
    """
    COV, CORR, SCATTER
    """
    def __init__(self, dataset):
        self.ds = dataset
        self.single_quantities = None
        self.crossed_quantities = None
        self.analysis = 'Relations'

    def _has_matrix_structure(self):
        return self.x == self.y

    def _make_index_pairs(self):
        if self._has_matrix_structure():
            full_range = len(self.x) - 1
        else:
            full_range = len(self.x + self.y) - 1
        x_range = range(0, len(self.x))
        y_range = range(x_range[-1] + 1, full_range + 1)
        if self._has_matrix_structure():
            return list(product(range(0, full_range+1), repeat=2))
        else:
            return list(product(x_range, y_range))

    def _sort_as_paired_stats(self, stat_list, pair_list):
        pairs = {pair: stat for pair, stat in zip(pair_list, stat_list)}
        if self._has_matrix_structure():
            return [(pairs[p[0], p[1]], pairs[p[1], p[0]]) for p in pair_list]
        else:
            return [(pairs[p[0], p[1]], pairs[p[0], p[1]]) for p in pair_list]

    def action_matrix(self, perf, imp, w=None, measures={
            'method': 'simple', 'perf': 'mean', 'imp': 'mean'}):
        """
        ... DESP ...

        Parameters
        ----------
        perf : str or list of str
            DESCP
        imp : list of str
            DESCP
        measures : dict {'method': ..., 'perf_stat': ..., 'imp_stat': ...}
            DECP

        Returns
        -------

        """
        method = measures['method']
        perf_stat, imp_stat = measures['perf'], measures['imp']
        if method in ['corr', 'reg']:
            raise NotImplementedError("{}-method unsupported.".format(method))
        if perf_stat != 'mean' and not isinstance(perf_stat, list):
            raise ValueError("'perf' stat must be list of codes or 'mean'.")
        if imp_stat != 'mean' and not isinstance(imp_stat, list):
            raise ValueError("'imp' stat must be list of codes or 'mean'.")
        # Two simple item batteries of identical length for performance and
        # importance dimensions
        if method == 'simple':
            self._select_variables(perf, imp, w)
            self._get_quantities()
            il = len(self._org_x)
            if perf_stat == 'mean':
                perfs = [q.summarize('mean', as_df=False, margin=False).result[0, 0]
                         for q in self.single_quantities[:il]]
                perfs = np.array(perfs)
            else:
                perfs = [q.group(perf_stat) for q in self.single_quantities[:il]]
                perfs = [p.count(as_df=False, margin=False).normalize().result[0, 0]
                        for p in perfs]
                perfs = np.array(perfs)
            if imp_stat == 'mean':
                imps = [q.summarize('mean', as_df=False, margin=False).result[0, 0]
                        for q in self.single_quantities[il:]]
                imps = np.array(imps)
            else:
                imps = [q.group(imp_stat) for q in self.single_quantities[il:]]
                imps = [i.count(as_df=False, margin=False).normalize().result[0, 0]
                        for i in imps]
                imps = np.array(imps)
        # Centering of data - currently only valid for the 'simple' approach!
        perf_mean, perf_sd = perfs.mean(), perfs.std(ddof=1)
        imps_mean, imps_sd = imps.mean(), imps.std(ddof=1)
        perf_c = (perfs -  perfs.mean()) / perfs.std(ddof=1)
        imps_c = (imps - imps.mean()) / imps.std(ddof=1)

        plt.set_autoscale_on = False
        plt.figure(figsize=(5, 5))
        plt.axvline(x=0.0, c='grey', ls='solid', linewidth=0.9)
        plt.axhline(y=0.0, c='grey', ls='solid', linewidth=0.9)
        if method == 'simple':
            xlim = max(abs(perf_c.min()), perf_c.max()) + 1.0
            ylim = max(abs(imps_c.min()), imps_c.max()) + 1.0
            plt.xlim([-xlim, xlim])
            plt.ylim([-ylim, ylim])
            vals = np.vstack([imps_c, perf_c]).T
        else:
            plt.xlim([0, 6])
            plt.ylim([-1, 1])
            vals = result.values
        x = plt.scatter(vals[:, 1], vals[:, 0],  edgecolor='w', marker='o',
                        c='red', s=80)
        fig = x.get_figure()
        xlab = 'Performance\n({})'
        xlab = xlab.format('mean' if perf_stat == 'mean' else 'top box')
        fig.get_axes()[0].set_xlabel(xlab)
        ylab = 'Importance\n({})'
        ylab = ylab.format('mean' if imp_stat == 'mean' else 'top box')
        fig.get_axes()[0].set_ylabel(ylab)
        plt.tick_params(axis='both', labelbottom='off', labelleft='off')

        x0 = fig.get_axes()[0].get_position().x0
        y0 = fig.get_axes()[0].get_position().y0
        x1 = fig.get_axes()[0].get_position().x1
        y1 = fig.get_axes()[0].get_position().y1

        ax = fig.get_axes()[0]
        fig.text(0.3, 0.87, 'critical improvement', ha='center', va='center',
                 fontsize=7, transform=ax.transAxes)
        fig.text(0.3, 0.16, 'action not required', ha='center', va='center',
                 fontsize=7, transform=ax.transAxes)
        fig.text(0.73, 0.87, 'leverage strengths', ha='center', va='center',
                 fontsize=7, transform=ax.transAxes)
        fig.text(0.73, 0.16, 'resource transfer\nopportunity', ha='center',
                 va='center', fontsize=7, transform=ax.transAxes)

        text = 'Priority matrix'
        plt.figtext(x0+0.015, 1.09-y0, text, fontsize=12, color='w',
                    fontweight='bold', verticalalignment='top',
                    bbox={'facecolor':'red', 'alpha': 0.8, 'edgecolor': 'w',
                          'pad': 10})
        label_vars = self._org_x
        text = ''
        for no, var in enumerate(label_vars, start=1):
            text += '\n{}: {}\n'.format(no, self.ds._get_label(var))
        fig.text(1.06-x0, 1.011-y0, text, fontsize=6, verticalalignment='top',
                 bbox={'facecolor':'lightgrey', 'alpha': 0.65,
                       'edgecolor': 'w', 'pad': 10})

        for no, coord in enumerate(vals, start=1):
            coord = [coord[1], coord[0]]
            plt.annotate(no, coord, ha='center', va='center',
                fontsize=7)

        logo = Image.open('C:/Users/alt/Documents/IPython Notebooks/Designs/Multivariate class/__resources__/YG_logo.png')
        newax = fig.add_axes([x0+0.005, y0-0.15, 0.1, 0.1], anchor='NE', zorder=-1)
        newax.imshow(logo)
        newax.axis('off')
        fig.savefig(self.ds.path + 'action_matrix.png', bbox_inches='tight', dpi=300)

    def cov(self, x, y, w=None, n=False, drop_listwise=False):
        self._select_variables(x, y, w, drop_listwise)
        if self.single_quantities is None: self._get_quantities()
        pairs = self._make_index_pairs()
        means = [q._drop_pairwise().summarize('mean', as_df=False).result[0, 0]
                 for q in self.crossed_quantities]
        means_paired = self._sort_as_paired_stats(means, pairs)
        xprods, unbiased_ns = [], []
        for pair, means_pair in zip(pairs, means_paired):
            data = self._analysisdata.copy()
            data = data.ix[:, [pair[0], pair[1], -1]].dropna().values
            m_diff = data[:, :-1] - means_pair
            xprods.append(np.nansum(m_diff[:, 0] * m_diff[:, 1] * data[:, -1]))
            unbiased_ns.append(np.nansum(data[:, -1]) - 1)
        cov = np.array(xprods) / np.array(unbiased_ns)
        cov = pd.DataFrame(cov.reshape(len(self.x), len(self.y)),
                           index=self.x, columns=self.y)
        cov.index.name = 'Covariance'
        if n:
            paired_ns = [n + 1 for n in unbiased_ns]
            return cov, paired_ns
        else:
            return cov
        # if n:

        #     return cov, paired_ns
        # else:
        #     return cov

    def corr(self, x, y, w=None, n=False, sig='full', drop_listwise=False, matrixed=False, plot=False):
        self._select_variables(x, y, w, drop_listwise)
        self._has_analysis_data()
        self._has_yvar()
        cov, ns = self.cov(x, y, w, True, drop_listwise)
        pairs = self._make_index_pairs()
        stddev = [q.summarize('stddev', as_df=False).result[0, 0]
                  for q in self.crossed_quantities]
        stddev_paired = self._sort_as_paired_stats(stddev, pairs)
        normalizer = [stddev1 * stddev2 for stddev1, stddev2 in stddev_paired]
        corr = cov / np.array(normalizer).reshape(cov.shape)
        ns = pd.DataFrame(np.array(ns).reshape(corr.shape),
                         index=corr.index, columns=corr.columns)
        corr.index.name = None

        if not matrixed:
            ns = ns.loc[self._org_x, self._org_y]
            corr = corr.loc[self._org_x, self._org_y]
        if not n and not sig and not plot:
            corr.index.name = 'Correlations'
            return corr

        if sig and sig not in ['flag', 'full']:
                raise ValueError('"sig" must be one of None, "flag" or "full".')
        sigtest = np.sqrt((ns-2)/(1-corr**2))*corr
        sigtest = pd.DataFrame(get_pval(ns-2, sigtest)[1], index=corr.index, columns=corr.columns)
        sigtest.replace(np.NaN, 0, inplace=True)
        sigtest_flags = sigtest.copy()[sigtest<0.05]
        sigtest_flags[sigtest_flags < 0.01] = 1
        sigtest_flags[(sigtest_flags != 1) & (~np.isnan(sigtest_flags))] = 2
        sigtest_flags.replace(1, '**', inplace=True)
        sigtest_flags.replace(2, '*', inplace=True)
        sigtest_flags.replace(np.NaN, '', inplace=True)

        collect = []
        corr_iter = np.round(corr.copy(), 4)
        sigtest = np.round(sigtest, 3)
        ns = np.round(ns, 0)
        for r, flag, p, n_ in zip(corr_iter.iterrows(), sigtest_flags.iterrows(), sigtest.iterrows(), ns.iterrows()):
            row1 = pd.DataFrame(r[1]).T
            row2 = pd.DataFrame(flag[1]).T
            row3 = pd.DataFrame(p[1]).T
            row4 = pd.DataFrame(n_[1]).T
            collect.append(pd.concat([row1, row2, row3, row4], axis=0))
        final = pd.concat(collect, axis=0)

        var = self._org_x if not matrixed else self._org_x + self._org_y
        stats = ['r', 'sig.', 'p', 'n']
        mi = pd.MultiIndex.from_product([var, stats], names=['', 'Correlations'])
        final.index = mi
        if not n or sig != 'full':
            if n:
                if sig == 'flag':
                    select = ['r', 'sig.', 'n']
                elif sig == 'p':
                    select = ['r', 'p', 'n']
            else:
                if sig == 'full':
                    select = ['r', 'sig.', 'p']
                elif sig == 'flag':
                    select = ['r', 'sig.']
                elif sig == 'p':
                    select = ['r', 'p']
            final = final[[group2 in select for group1, group2 in final.index]]

        if plot:
            if plot not in ['sig', 'full']:
                raise ValueError('"plot" must be one of None, "sig" or "full".')
            if plot == 'sig':
                corr = corr[sigtest < 0.05]
                center = np.mean(corr.replace(np.NaN, 0.0).values)
            else:
                center = np.mean(corr.values)

            colors = sns.blend_palette(['lightgrey', 'red'], as_cmap=True,
                                       n_colors=1000)

            corr_res = sns.heatmap(corr, annot=True, cbar=None, fmt='.2f',
                                   square=True, robust=True, cmap=colors,
                                   center=center, linewidth=1.0,
                                   annot_kws={'size': 8})

            fig = corr_res.get_figure()
            x0 = fig.get_axes()[0].get_position().x0
            y0 = fig.get_axes()[0].get_position().y0
            x1 = fig.get_axes()[0].get_position().x1
            y1 = fig.get_axes()[0].get_position().y1

            text = 'Correlation matrix (Pearson)'
            plt.figtext(x0+0.017, 1.115-y0, text, fontsize=12, color='w',
                        fontweight='bold', verticalalignment='top',
                        bbox={'facecolor':'red', 'alpha': 0.8, 'edgecolor': 'w',
                              'pad': 10})
            if self._has_matrix_structure():
                label_vars = self.x
            else:
                label_vars = self.x + self.y
            text = ''
            for var in label_vars:
                text += '\n{}: {}\n'.format(var, self.ds._get_label(var))
            fig.text(1.06-x0, 1.0-y0, text, fontsize=6, verticalalignment='top',
                     bbox={'facecolor':'lightgrey', 'alpha': 0.65,
                           'edgecolor': 'w', 'pad': 10})
            logo = Image.open('C:/Users/alt/Documents/IPython Notebooks/Designs/Multivariate class/__resources__/YG_logo.png')
            newax = fig.add_axes([x0+0.005, y0-0.25, 0.1, 0.1], anchor='NE', zorder=-1)
            newax.imshow(logo)
            newax.axis('off')
            fig.savefig(self.ds.path + 'corr.png', bbox_inches='tight', dpi=300)

        final.index.name = 'Correlation'
        return final

##############################################################################

class Cache(defaultdict):


    def __init__(self):
        # The 'lock_cache' raises an exception in the
        super(Cache, self).__init__(Cache)

    def __reduce__(self):
        return self.__class__, tuple(), None, None, self.iteritems()


    def set_obj(self, collection, key, obj):
        '''
        Save a Quantipy resource inside the cache.

        Parameters
        ----------
        collection : {'matrices', 'weight_vectors', 'quantities',
                      'mean_view_names', 'count_view_names'}
            The key of the collection the object should be placed in.
        key : str
            The reference key for the object.
        obj : Specific Quantipy or arbitrary Python object.
            The object to store inside the cache.

        Returns
        -------
        None
        '''
        self[collection][key] = obj

    def get_obj(self, collection, key):
        '''
        Look up if an object exists in the cache and return it.

        Parameters
        ----------
        collection : {'matrices', 'weight_vectors', 'quantities',
                      'mean_view_names', 'count_view_names'}
            The key of the collection to look into.
        key : str
            The reference key for the object.

        Returns
        -------
        obj : Specific Quantipy or arbitrary Python object.
            The cached object mapped to the passed key.
        '''
        if collection == 'matrices':
            return self[collection].get(key, (None, None))
        elif collection == 'squeezed':
            return self[collection].get(key, (None, None, None, None, None, None, None))
        else:
            return self[collection].get(key, None)

##############################################################################

class Link(Quantity, dict):
    def __init__(self, ds, filters=None, x=None, y=None, views=None):
        self.ds_key = ds.name
        self.filters = filters
        self.x = x
        self.y = y
        self.id = '[{}][{}][{}][{}]'.format(self.ds_key, self.filters, self.x,
                                            self.y)
        self.stack_connection = False
        self.quantified = False
        self._quantify(ds)
        #---------------------------------------------------------------------

    def _clear(self):
        ds_key, filters, x, y = self.ds_key, self.filters, self.x, self.y
        _id, stack_connection = self.id, self.stack_connection
        dataset, data, meta, cache = self.dataset, self.data, self.meta, self.cache
        self.__dict__.clear()
        self.ds_key, self.filters, self.x, self.y = ds_key, filters, x, y
        self.id, self.stack_connection = _id, stack_connection
        return None

    def _quantify(self, ds):
        # Establish connection to source dataset components when in Stack-mode
        def dataset():
            """
            Ensure a Link is able to track back to its orignating dataset.
            """
            return ds
        def data():
            """
            Ensure a Link is able to track back to its orignating case data.
            """
            return ds.data()
        def meta():
            """
            Ensure a Link is able to track back to its orignating meta data.
            """
            return ds.meta()
        def cache():
            """
            Ensure a Link is able to track back to its cached data vectors.
            """
            return ds.cache()
        self.dataset = dataset
        self.data = data
        self.meta = meta
        self.cache = cache
        Quantity.__init__(self, self)
        return None

#     def __repr__(self):
#         info = 'Link - id: {}\nquantified: {} | stack connected: {} | views: {}'
#         return info.format(self.id, self.quantified, self.stack_connection,
#                            len(self.values()))

    def describe(self):
        described = pd.Series(self.keys(), name=self.id)
        described.index.name = 'views'
        return described

##############################################################################

class Stack(defaultdict):
    def __init__(self, name=''):
        super(Stack, self).__init__(Stack)
        self.name = name
        self.ds = None

    # ====================================================================
    # THESE NEED TO GET A REVIEW!
    # ====================================================================
    # def __reduce__(self):
    #     arguments = (self.name, )
    #     states = self.__dict__.copy()
    #     if states['ds'] is not None:
    #         states['ds'].__dict__['_cache'] = Cache()
    #     return self.__class__, arguments, states, None, self.iteritems()

    # ====================================================================
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # ====================================================================

    # ------------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------------
    def add_dataset(self, dataset):
        self.ds = dataset

    def save(self, path_stack, compressed=False):
        if compressed:
            f = gzip.open(path_stack, 'wb')
        else:
            f = open(path_stack, 'wb')
        dill.dump(self, f)
        f.close()
        return None

    def load(self, path_stack, compressed=False):
        if compressed:
            f = gzip.open(path_stack, 'rb')
        else:
            f = open(path_stack, 'rb')
        loaded_stack = dill.load(f)
        f.close()
        return loaded_stack

    # ------------------------------------------------------------------------
    # DATA/LINK POPULATION
    # ------------------------------------------------------------------------
    def refresh(self):
        pass

    def populate(self, filters=None, x=None, y=None, weights=None, views=None):
        """
        Populate the Stack instance with Links that (optionally) hold Views.
        """
        if filters is None: filters = ['no_filter']
        for _filter in filters:
            for _x in x:
                for _y in y:
                    if not isinstance(self[self.ds.name][_filter][_x][_y], Link):
                        l = self.ds.link(_filter, _x, _y)
                        l.stack_connection = True
                        self[self.ds.name][_filter][_x][_y] = l
                    else:
                        l = self.get(self.ds.name, _filter, _x, _y)
                        l.stack_connection = True
                    if views is not None:
                        if not isinstance(views, ViewMapper):
                            # Use DefaultViews if no view were given
                            if views is None:
                                pass
                            elif isinstance(views, (list, tuple)):
                                views = QuantipyViews(views=views)
                            else:
                                print 'ERROR - VIEWS CRASHED!'
                        views._apply_to(l, weights)
                        l._clear()

    # ------------------------------------------------------------------------
    # INSPECTION & QUERY
    # ------------------------------------------------------------------------
    def get(self, ds_key=None, filters=None, x=None, y=None):
        """
        Return Link from Stack.
        """
        if ds_key is None and len(self.keys()) > 1:
            key_err = 'Cannot select from multiple datasets when no key is provided.'
            raise KeyError(key_err)
        elif ds_key is None and len(self.keys()) == 1:
            ds_key = self.keys()[0]
        if filters is None: filters = 'no_filter'
        if not isinstance(self[ds_key][filters][x][y], Link):
            l = Link(self.ds, filters, x, y)
        else:
            l = self[ds_key][filters][x][y]
            l._quantify(self.ds)
        return l

    def describe(self, index=None, columns=None, query=None, split_view_names=False):
        """
        Generates a structured overview of all Link defining Stack elements.

        Parameters
        ----------
        index, columns : str of or list of {'data', 'filter', 'x', 'y', 'view'},
                         optional
            Controls the output representation by structuring a pivot-style
            table according to the index and column values.
        query : str
            A query string that is valid for the pandas.DataFrame.query() method.
        split_view_names : bool, default False
            If True, will create an output of unique view name notations split
            up into their components.

        Returns
        -------
        description : pandas.DataFrame
            DataFrame summing the Stack's structure in terms of Links and Views.
        """
        stack_tree = []
        for dk in self.keys():
            path_dk = [dk]
            filters = self[dk]

#             for fk in filters.keys():
#                 path_fk = path_dk + [fk]
#                 xs = self[dk][fk]

            for fk in filters.keys():
                path_fk = path_dk + [fk]
                xs = self[dk][fk]

                for sk in xs.keys():
                    path_sk = path_fk + [sk]
                    ys = self[dk][fk][sk]

                    for tk in ys.keys():
                        path_tk = path_sk + [tk]
                        views = self[dk][fk][sk][tk]

                        if views.keys():
                            for vk in views.keys():
                                path_vk = path_tk + [vk, 1]
                                stack_tree.append(tuple(path_vk))
                        else:
                            path_vk = path_tk + ['|||||', 1]
                            stack_tree.append(tuple(path_vk))

        column_names = ['data', 'filter', 'x', 'y', 'view', '#']
        description = pd.DataFrame.from_records(stack_tree, columns=column_names)
        if split_view_names:
            views_as_series = pd.DataFrame(
                description.pivot_table(values='#', columns='view', aggfunc='count')
                ).reset_index()['view']
            parts = ['xpos', 'agg', 'condition', 'rel_to', 'weights',
                     'shortname']
            description = pd.concat(
                (views_as_series,
                 pd.DataFrame(views_as_series.str.split('|').tolist(),
                              columns=parts)), axis=1)

        description.replace('|||||', np.NaN, inplace=True)
        if query is not None:
            description = description.query(query)
        if not index is None or not columns is None:
            description = description.pivot_table(values='#', index=index, columns=columns,
                                aggfunc='count')
        return description
