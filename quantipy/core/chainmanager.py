#!/usr/bin/python
# -*- coding: utf-8 -*-
import copy
import string
import cPickle

import pandas as pd

from collections import Counter

from quantipy.core.chain import Chain
from quantipy.core.tools.qp_decorators import modify
from quantipy.core.tools.logger import get_logger
logger = get_logger(__name__)


class ChainManager(object):

    def __init__(self, stack):
        self.stack = stack
        self.__chains = []
        self.source = 'native'
        self.build_info = {}
        self._hidden = []

    def __str__(self):
        return '\n'.join([chain.__str__() for chain in self])

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, value):
        if isinstance(value, (unicode, str)):
            element = self.__chains[self._names_to_idxs[value]]
            is_folder = isinstance(element, dict)
            if is_folder:
                return element.values()[0]
            else:
                return element
        else:
            return self.__chains[value]

    def __len__(self):
        return len(self.__chains)

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < self.__len__():
            obj = self[self.n]
            self.n += 1
            return obj
        else:
            raise StopIteration
    next = __next__

    # -------------------------------------------------------------------------
    # properties and helpers
    # -------------------------------------------------------------------------

    @property
    def folders(self):
        """
        Folder indices, names and number of stored ``Chain`` items (as tuples).
        """
        return [
            (x, f.keys()[0], len(f.values()[0]))
            for x, f in enumerate(self)
            if isinstance(f, dict)]

    @property
    def folder_idxs(self):
        """
        The folders' index positions in self.
        """
        return [f[0] for f in self.folders]

    @property
    def folder_names(self):
        """
        The folders' names from self.
        """
        return [f[1] for f in self.folders]

    def _folders_to_idx(self):
        return {f[1]: f[0] for f in self.folders}

    def _idx_to_fold(self):
        return {f[0]: f[1] for f in self.folders}

    def _is_folder_ref(self, ref):
        return any(ref in f[:2] for f in self.folders)

    @property
    def singles(self):
        """
        The list of all non-folder ``Chain`` indices and names (as tuples).
        """
        return [
            (x, s.name) for x, s in enumerate(self) if not isinstance(s, dict)]

    @property
    def single_idxs(self):
        """
        The ``Chain`` instances' index positions in self.
        """
        return [s[0] for s in self.singles]

    @property
    def single_names(self):
        """
        The ``Chain`` instances' names.
        """
        return [s[1] for s in self.singles]

    def _singles_to_idx(self):
        return {name: i for i, name in self.singles}

    def _idx_to_singles(self):
        return dict(self.singles)

    def _is_single_ref(self, ref):
        return any(ref in s for s in self.singles)

    @property
    def chains(self):
        """
        The flattened list of all ``Chain`` items of self.
        """
        all_chains = []
        for c in self:
            if isinstance(c, dict):
                all_chains.extend(c.values()[0])
            else:
                all_chains.append(c)
        return all_chains

    @property
    def hidden(self):
        """
        All ``Chain`` elements that are hidden.
        """
        return [c.name for c in self.chains if c.hidden]

    @property
    def hidden_folders(self):
        """
        All hidden folders.
        """
        return [n for n in self._hidden if n in self.folder_names]

    def _names(self, unroll=False):
        if not unroll:
            return self.folder_names + self.single_names
        else:
            return [c.name for c in self.chains]

    def _idxs_to_names(self):
        return {
            x: c.keys()[0] if isinstance(c, dict) else c.name
            for x, c in enumerate(self)}

    def _names_to_idxs(self):
        return {n: i for i, n in self._idxs_to_names().items()}

    @modify(to_list="folders")
    def _verify_folder_name(self, folders):
        for folder in folders:
            if folder not in self.folder_names:
                err = "A folder named '{}' does not exist!".format(folder)
                logger.error(err); raise KeyError(err)
        return None

    @staticmethod
    def _verify_dupes_in_ref(refs, txt=""):
        if len(set(refs)) != len(refs):
            err = "Cannot {} from duplicate qp.Chain references: {}"
            err = err.format(txt)
            logger.error(err); raise KeyError(err)
        return None

    def _set_to_folderitems(self, folder):
        """
        Will keep only the ``values()`` ``qp.Chain`` item list from the named
        folder. Use this for within-folder-operations...
        """
        self._verify_folder_name(folder)
        org_chains = self.__chains[:]
        org_index = self._names_to_idxs[folder]
        self.__chains = self[folder]
        return org_chains, org_index

    def _rebuild_org_folder(self, folder, items, index):
        """
        After a within-folder-operation this method is using the returns
        of ``_set_to_folderitems`` to rebuild the originating folder.
        """
        self.fold(folder)
        new_folder = self.__chains[:]
        self.__chains = items
        self.__chains[index] = new_folder[0]
        return None

    def describe(self, by_folder=False, show_hidden=False):
        """
        Get a structual summary of all ``qp.Chain`` instances found in self.

        Parameters
        ----------
        by_folder : bool, default False
            If True, only information on ``dict``-structured (folder-like)
            ``qp.Chain`` items is shown, multiindexed by folder names and item
            enumerations.
        show_hidden : bool, default False
            If True, the summary will also include elements that have been set
            hidden using ``self.hide()``.
        """
        folders = []
        folder_items = []
        variables = []
        names = []
        array_sum = []
        sources = []
        banner_ids = []
        item_pos = []
        hidden = []
        bannermap = self._get_ykey_mapping()
        for pos, chains in enumerate(self):
            is_folder = isinstance(chains, dict)
            if is_folder:
                folder_name = chains.keys()
                chains = chains.values()[0]
                folder_items.extend(list(xrange(0, len(chains))))
                item_pos.extend([pos] * len(chains))
            else:
                chains = [chains]
                folder_name = [None]
                folder_items.append(None)
                item_pos.append(pos)

            if chains[0].structure is None:
                variables.extend([c._x_keys[0] for c in chains])
                names.extend([c.name for c in chains])
                folders.extend(folder_name * len(chains))
                array_sum.extend([True if c.array_style > -1 else False
                                 for c in chains])
                sources.extend(c.source if not c.edited else 'edited'
                               for c in chains)
                for c in chains:
                    for m in bannermap:
                        if m[0] == c._y_keys:
                            banner_ids.append(m[1])
            else:
                variables.extend([chains[0].name])
                names.extend([chains[0].name])
                folders.extend(folder_name)
                array_sum.extend([False])
                sources.extend(c.source for c in chains)
                banner_ids.append(None)
            for c in chains:
                if c.hidden:
                    hidden.append(True)
                else:
                    hidden.append(False)
        df_data = [
            item_pos, names, folders, folder_items, variables, sources,
            banner_ids, array_sum, hidden]
        df_cols = [
            'Position', 'Name', 'Folder', 'Item', 'Variable', 'Source',
            'Banner id', 'Array', 'Hidden']
        df = pd.DataFrame(df_data).T
        df.columns = df_cols
        if by_folder:
            df = df.set_index(['Position', 'Folder', 'Item'])
        if not show_hidden:
            df = df[df['Hidden'] is False][df.columns[:-1]]
        return df

    def _get_ykey_mapping(self):
        ys = []
        letters = string.ascii_uppercase + string.ascii_lowercase
        for c in self.chains:
            if c._y_keys not in ys:
                ys.append(c._y_keys)
        return zip(ys, letters)

    # -------------------------------------------------------------------------
    # checks
    # -------------------------------------------------------------------------

    def equals(self, other):
        """
        Test equality of self to another ``ChainManager`` object instance.

        .. note::
            Only the flattened list of ``Chain`` objects stored are tested,
            i.e. any folder structure differences are ignored. Use
            ``compare()`` for a more detailed comparison.

        Parameters
        ----------
        other : ``qp.ChainManager``
            Another ``ChainManager`` object to compare.

        Returns
        -------
        equality : bool
        """
        return self._check_equality(other, False)

    def _check_equality(self, other, return_diffs=True):
        chains1 = self.chains
        chains2 = other.chains
        diffs = {}
        if not len(chains1) == len(chains2):
            return False
        else:
            paired = zip(chains1, chains2)
            for c1, c2 in paired:
                atts1 = c1.__dict__
                atts2 = c2.__dict__
                for att in atts1.keys():
                    if isinstance(atts1[att], (pd.DataFrame, pd.Index)):
                        if not atts1[att].equals(atts2[att]):
                            diffs[att] = [atts1[att], atts2[att]]
                    else:
                        if atts1[att] != atts2[att]:
                            diffs[att] = [atts1[att], atts2[att]]
            return diffs if return_diffs else not diffs

    def compare(self, other, strict=True, full_summary=True):
        """
        Compare structure and content of self to another instance.

        Parameters
        ----------
        other : ``qp.ChainManager``
            Another ``ChainManager`` object to compare.
        strict : bool, default True
            Test if the structure of folders vs. single Chain objects is the
            same in both ChainManager instances.
        full_summary : bool, default True
            ``False`` will disable the detailed comparison ``pd.DataFrame``
            that informs about differences between the objects.
        """
        diffs = []
        if strict:
            same_structure = self._test_same_structure(other)
            if not same_structure:
                diffs.append('s')
        check = self._check_equality(other)
        if isinstance(check, bool):
            diffs.append('l')
        elif check:
            diffs.append('c')
        diffs_in = ''
        if diffs:
            if 'l' in diffs:
                diffs_in += '\n  -Length (number of stored Chain objects)'
            if 's' in diffs:
                diffs_in += (
                    '\n  -Structure (folders and/or single Chain order)')
            if 'c' in diffs:
                diffs_in += (
                    '\n  -Chain elements (properties and content of Chain '
                    'objects)')
        if diffs_in:
            result = 'ChainManagers are not identical:\n'
            result += '--------------------------------' + diffs_in
        else:
            result = 'ChainManagers are identical.'
        logger.info(result)
        return None

    def _test_same_structure(self, other):
        folders1 = self.folders
        singles1 = self.singles
        folders2 = other.folders
        singles2 = other.singles
        if (folders1 != folders2 or singles1 != singles2):
            return False
        else:
            return True

    # -------------------------------------------------------------------------
    # I/O
    # -------------------------------------------------------------------------

    def save(self, path, keep_stack=False):
        """
        """
        if not keep_stack:
            del self.stack
            self.stack = None
        f = open(path, 'wb')
        cPickle.dump(self, f, cPickle.HIGHEST_PROTOCOL)
        f.close()
        return None

    @staticmethod
    def load(path):
        """
        """
        f = open(path, 'rb')
        obj = cPickle.load(f)
        f.close()
        return obj

    def clone(self):
        """
        Return a full (deep) copy of self.
        """
        return copy.deepcopy(self)

    # -------------------------------------------------------------------------
    # modifications
    # -------------------------------------------------------------------------

    def hide(self, chains):
        """
        Flag elements as being hidden.

        Parameters
        ----------
        chains : (list) of int and/or str or dict
            The ``qp.Chain`` item and/or folder names to hide. To hide *within*
            a folder use a dict to map the desired Chain names to the belonging
            folder name.
        """
        self._toggle_vis(chains, 'hide')
        return None

    def unhide(self, chains=None):
        """
        Unhide elements that have been set as ``hidden``.

        Parameters
        ----------
        chains : (list) of int and/or str or dict, default None
            The ``qp.Chain`` item and/or folder names to unhide. To unhide
            *within* a folder use a dict to map the desired Chain names to the
            belonging folder name. If not provided, all hidden elements will be
            unhidden.
        """
        if not chains:
            chains = self.folder_names + self.single_names
        self._toggle_vis(chains, 'unhide')
        return None

    @modify(to_list="chains")
    def _toggle_vis(self, chains, mode='hide'):
        for chain in chains:
            if isinstance(chain, dict):
                fname = chain.keys()[0]
                elements = chain.values()[0]
                fidx = self._names_to_idxs[fname]
                folder = self[fidx][fname]
                for c in folder:
                    if c.name in elements:
                        c.hidden = True if mode == 'hide' else False
                        if mode == 'hide' and c.name not in self._hidden:
                            self._hidden.append(c.name)
                        if mode == 'unhide' and c.name in self._hidden:
                            self._hidden.remove(c.name)
            else:
                if chain in self.folder_names:
                    for c in self[chain]:
                        c.hidden = True if mode == 'hide' else False
                else:
                    self[chain].hidden = True if mode == 'hide' else False
                if mode == 'hide':
                    if chain not in self._hidden:
                        self._hidden.append(chain)
                else:
                    if chain in self._hidden:
                        self._hidden.remove(chain)
        return None

    @modify(to_list="chains")
    def fold(self, folder_name=None, chains=None):
        """
        Arrange non-``dict`` structured ``qp.Chain`` items in folders.

        All separate ``qp.Chain`` items will be mapped to their ``name``
        property being the ``key`` in the transformed ``dict`` structure.

        Parameters
        ----------
        folder_name : str, default None
            Collect all items in a folder keyed by the provided name. If the
            key already exists, the items will be appended to the ``dict``
            values.
        chains : (list) of int and/or str, default None
            Select specific ``qp.Chain`` items by providing their positional
            indices or ``name`` property value for moving only a subset to the
            folder.

        Returns
        -------
        None
        """
        if chains:
            if any(self._is_folder_ref(c) for c in chains):
                err = 'Cannot build folder from other folders!'
                raise ValueError(err)
            all_chain_names = []
            singles = []
            for c in chains:
                if isinstance(c, (str, unicode)):
                    all_chain_names.append(c)
                elif isinstance(c, int) and c in self._idx_to_singles():
                    all_chain_names.append(self._idx_to_singles()[c])
            for c in all_chain_names:
                singles.append(self[self._singles_to_idx()[c]])
        else:
            singles = [s for s in self if isinstance(s, Chain)]
        self._verify_dupes_in_ref(singles, "build folder")
        for s in singles:
            if folder_name:
                if folder_name in self.folder_names:
                    self[folder_name].append(s)
                else:
                    self.__chains.append({folder_name: [s]})
                del self.__chains[self._singles_to_idx()[s.name]]
            else:
                self.__chains[self._singles_to_idx()[s.name]] = {s.name: [s]}
        return None

    @modify(to_list="folder")
    def unfold(self, folder=None):
        """
        Remove folder but keep the collected items.

        The items will be added starting at the old index position of the
        original folder.

        Parameters
        ----------
        folder : (list of) str, default None
            The name of the folder to drop and extract items from. If not
            provided all folders will be unfolded.

        Returns
        -------
        None
        """
        if not folder:
            folder = self.folder_names
        self._verify_folder_name(folder)
        for f in folder:
            old_pos = self._names_to_idxs[f]
            items = self[f]
            start = self.__chains[: old_pos]
            end = self.__chains[old_pos + 1:]
            self.__chains = start + items + end
        return None

    @modify(to_list="values")
    def cut(self, values, ci=None, base=False, tests=False):
        """
        Isolate selected axis values in the ``Chain.dataframe``.

        Parameters
        ----------
        values : (list of) str
            The string must indicate the raw (i.e. the unpainted) second level
            axis value, e.g. ``'mean'``, ``'net_1'``, etc.
        ci : {'counts', 'c%', None}, default None
            The cell item version to target if multiple frequency
            representations are present.
        base : bool, default False
            Controls keeping any existing base view aggregations.
        tests : bool, default False
            Controls keeping any existing significance test view aggregations.

        Returns
        -------
        None
        """
        if 'cbase' in values:
            values[values.index('cbase')] = 'All'
        if base and 'All' not in values:
            values = ['All'] + values
        for c in self.chains:
            # force ci parameter for proper targeting on array summaries...
            if c.array_style == 0 and ci is None:
                _ci = c.cell_items.split('_')[0]
                if not _ci.startswith('counts'):
                    ci = '%'
                else:
                    ci = 'counts'
            if c.sig_test_letters:
                c._remove_letter_header()
            idxs, names, order = c._view_idxs(
                values, keep_tests=tests, keep_bases=base, names=True, ci=ci)
            idxs = [i for _, i in sorted(zip(order, idxs))]
            names = [n for _, n in sorted(zip(order, names))]
            if c.ci_count > 1:
                c._non_grouped_axis()
            if c.array_style == 0:
                c._fill_cells()
                start, repeat = c._row_pattern(ci)
                c._frame = c._frame.iloc[start::repeat, idxs]
            else:
                c._frame = c._frame.iloc[idxs, :]
                c.index = c._slice_edited_index(c.index, idxs)
            for v in c.views.copy():
                if v not in names:
                    del c._views[v]
                else:
                    c._views[v] = names.count(v)
            if not c._array_style == 0:
                if not tests:
                    c.sig_test_letters = None
                else:
                    c._frame = c._apply_letter_header(c._frame)
            c.edited = True

        return None

    def join(self, title='Summary'):
        """
        Join all ``qp.Chain```elements, concatenating along the matching axis.

        Parameters
        ----------
        title : {str, 'auto'}, default 'Summary'
            The new title for the joined axis' index representation.

        Returns
        -------
        None
        """
        custom_views = []
        self.unfold()
        chains = self.chains
        totalmul = len(chains[0]._frame.columns.get_level_values(0).tolist())
        concat_dfs = []
        for c in chains:
            new_label = []
            if c.sig_test_letters:
                c._remove_letter_header()
                c._frame = c._apply_letter_header(c._frame)
            df = c.dataframe

            if not c.array_style == 0:
                new_label.append(
                    df.index.get_level_values(0).values.tolist()[0])
                new_label.extend((len(c.describe()) - 1) * [''])
            else:
                new_label.extend(df.index.get_level_values(1).values.tolist())
            names = ['Question', 'Values']
            join_idx = pd.MultiIndex.from_product([[title], new_label],
                                                  names=names)
            df.index = join_idx

            df.rename(columns={c._x_keys[0]: 'Total'}, inplace=True)

            if not c.array_style == 0:
                custom_views.extend(c._views_per_rows())
            else:
                df.columns.set_levels(
                    levels=[title] * totalmul, level=0, inplace=True)
            concat_dfs.append(df)

        new_df = pd.concat(concat_dfs, axis=0, join='inner')
        self.chains[0]._frame = new_df
        self.reorder([0])
        self.rename({self.single_names[0]: title})
        self.fold()
        self.chains[0]._custom_views = custom_views
        return None

    @modify(to_list="order")
    def reorder(self, order, folder=None, inplace=True):
        """
        Reorder (folders of) ``qp.Chain`` items by providing a list of new
        indices or names.

        Parameters
        ----------
        order : list of int and/or str
            The folder or ``qp.Chain`` references to determine the new order
            of items. Any items not referenced will be removed from the new
            order.
        folder : str, default None
            If a folder name is provided, items will be sorted within that
            folder instead of applying the sorting to the general items
            collection.
        inplace : bool, default True
            By default the new order is applied inplace, set to ``False`` to
            return a new object instead.
        """
        if inplace:
            cm = self
        else:
            cm = self.clone()
        if folder:
            org_chains, org_index = self._set_to_folderitems(folder)
        new_idx_order = []
        for o in order:
            new_idx_order.append(self._names_to_idxs.get(o, o))
        self._verify_dupes_in_ref(new_idx_order, "reorder")
        items = [self.__chains[idx] for idx in new_idx_order]
        cm.__chains = items
        if folder:
            cm._rebuild_org_folder(folder, org_chains, org_index)
        if not inplace:
            return cm

    def rename(self, names, folder=None):
        """
        Rename (folders of) ``qp.Chain`` items by providing a mapping of old
        to new keys.

        Parameters
        ----------
        names : dict
            Maps existing names to the desired new ones, i.e.
            {'old name': 'new names'} pairs need to be provided.
        folder : str, default None
            If a folder name is provided, new names will only be applied
            within that folder. This is without effect if all ``qp.Chain.name``
            properties across the items are unique.

        Returns
        -------
        None
        """
        if not isinstance(names, dict):
            err = "''names' must be a dict of old_name: new_name pairs."
            raise ValueError(err)
        self._verify_folder_name(folder)
        for old, new in names.items():
            no_folder_name = folder and old not in self._names(False)
            no_name_across = not folder and old not in self._names(True)
            if no_folder_name and no_name_across:
                err = "'{}' is not an existing folder or ``qp.Chain`` name!"
                raise KeyError(err.format(old))
            else:
                within_folder = old not in self._names(False)
            if not within_folder:
                idx = self._names_to_idxs[old]
                if not isinstance(self.__chains[idx], dict):
                    self.__chains[idx].name = new
                else:
                    self.__chains[idx] = {new: self[old][:]}
            else:
                iter_over = self[folder] if folder else self.chains
                for c in iter_over:
                    if c.name == old:
                        c.name = new
        return None

    def paint_all(self, *args, **kwargs):
        """
        Apply labels, sig. testing conversion and other post-processing to the
        ``Chain.dataframe`` property.

        Use this to prepare a ``Chain`` for further usage in an Excel or Power-
        point Build.

        Parameters
        ----------
        text_key : str, default meta['lib']['default text']
            The language version of any variable metadata applied.
        text_loc_x : str, default None
            The key in the 'text' to locate the text_key for the
            ``pandas.DataFrame.index`` labels
        text_loc_y : str, default None
            The key in the 'text' to locate the text_key for the
            ``pandas.DataFrame.columns`` labels
        display : {'x', 'y', ['x', 'y']}, default None
            Text
        axes : {'x', 'y', ['x', 'y']}, default None
            Text
        view_level : bool, default False
            Text
        transform_tests : {False, 'full', 'cells'}, default cells
            Text
        totalize : bool, default False
            Text

        Returns
        -------
        None
            The ``.dataframe`` is modified inplace.
        """
        for chain in self:
            if isinstance(chain, dict):
                for c in chain.values()[0]:
                    c.paint(*args, **kwargs)
            else:
                chain.paint(*args, **kwargs)
        return None

    # -------------------------------------------------------------------------
    # extend, reduce, combine
    # -------------------------------------------------------------------------

    def _add_chains(self, chains, folder=None, index=-1, safe_names=False):
        """
        Migrate a new chain in self.__chains.
        """
        if folder:
            if not isinstance(folder, basestring):
                err = "folder must be a string!"
                logger.error(err); raise ValueError(err)
            if folder in self.folder_names:
                self[folder].extend(chains)
                return None
            chains = {folder: chains}
        if not index == -1:
            before_c = self.__chains[:index + 1]
            after_c = self.__chains[index + 1:]
            new_chains = before_c + chains + after_c
            self.__chains = new_chains
        else:
            self.__chains.extend(chains)
        if safe_names:
            self._uniquify_names()
        return None

    def _uniquify_names(self):
        all_names = Counter(self.single_names + self.folder_names)
        single_name_occ = Counter(self.single_names)
        folder_name_occ = {folder: Counter([c.name for c in self[folder]])
                           for folder in self.folder_names}
        for struct_name in all_names:
            if struct_name in folder_name_occ:
                iter_over = folder_name_occ[struct_name]
                is_folder = struct_name
            else:
                iter_over = single_name_occ
                is_folder = False
            for name, occ in iter_over.items():
                if occ > 1:
                    new_names = [
                        '{}_{}'.format(name, i) for i in range(1, occ + 1)]
                    idx = [s[0] for s in self.singles if s[1] == name]
                    pairs = zip(idx, new_names)
                    if is_folder:
                        for idx, c in enumerate(self[is_folder]):
                            c.name = pairs[idx][1]
                    else:
                        for p in pairs:
                            self.__chains[p[0]].name = p[1]
        return None

    @modify(to_list="chain")
    def add(self, chain, folder=None, index=-1, safe_names=False):
        """
        Extend ``self.__chains`` with an additional chain.

        Parameters
        ----------
        chain: qp.Chain instance
            The chain which is added.
        folder: str, default None
            If a folder name is added, the chain is added into this folder.
        index: int, default -1
            Position in self.__chains, where the new chains are added. If a
            folder name is given, the chains are added to an existing folder
            or the new folder is positioned at the defined index.
        safe_names : bool, default False
            If True and any duplicated element names are found after the
            operation, names will be made unique (by appending '_1', '_2',
            '_3', etc.).
        """
        if not all(isinstance(c, Chain) for c in chain):
            err = "chain must be a quantipy.Chain instance."
            logger.error(err); raise ValueError(err)
        self._add_chains(chain, folder, index, safe_names)
        return None

    def add_df(self, structure, meta_from=None, meta=None, name=None,
               folder=None, index=-1, safe_names=False):
        """
        Add a pandas.DataFrame as a Chain.

        Parameters
        ----------
        structure : ``pandas.Dataframe``
            The dataframe to add to the ChainManger
        meta_from : list, list-like, str, default None
            The location of the meta in the stack. Either a list-like object
            with data key and filter key or a str as the data key.
        meta : quantipy meta (dict)
            External meta used to paint the frame
        name : ``str``, default None
            The name to give the resulting chain. If not passed, the name will
            become the concatenated column names, delimited by a period.
        folder: str, default None
            If a folder name is added, the chain is added into this folder.
        index: int, default -1
            Position in self.__chains, where the new chains are added. If a
            folder name is given, the chains are added to an existing folder
            or the new folder is positioned at the defined index.
        safe_names : bool, default False
            If True and any duplicated element names are found after the
            operation, names will be made unique (by appending '_1', '_2',
            '_3', etc.).
        """
        name = name or '.'.join(structure.columns.tolist())

        chain = Chain(self.stack, name, structure=structure)
        chain._frame = chain.structure
        chain._index = chain._frame.index
        chain._columns = chain._frame.columns
        chain._frame_values = chain._frame.values

        if meta_from:
            if isinstance(meta_from, (str, unicode)):
                chain._meta = self.stack[meta_from].meta
            else:
                data_key, filter_key = meta_from
                chain._meta = self.stack[data_key][filter_key].meta
        elif meta:
            chain._meta = meta
        self._add_chains([chain], folder, index, safe_names)
        return None

    @modify(to_list=["x_keys", "y_keys", "views"])
    def get(self, data_key, filter_key, x_keys, y_keys, views, orient='x',
            rules=True, prioritize=True, folder=None, index=-1,
            safe_names=False):
        """
        Get chains from stack aggregations and add them to ``self.__chains``.

        Note
        ----
        Get a (list of) Chain instance(s) in either 'x' or 'y' orientation.
        Chain.dfs will be concatenated along the provided 'orient'-axis

        Parameters
        ----------
        data_key: str
            Get aggregations from ``self.stack[data_key]``.
        filter_key: str
            Get aggregations from ``self.stack[data_key][filter_key]``.
        x_keys: (list of) str
        y_keys: (list of) str
        views: (list of) str
            Dependent aggregations to the view names are extracted from stack.
        orient: str, {"x", "y"}
        rules: bool, default True
            Apply rules on the aggregation.
        prioritize: bool, default True
        folder: str, default None
            If a folder name is added, the chain is added into this folder.
        index: int, default -1
            Position in self.__chains, where the new chains are added. If a
            folder name is given, the chains are added to an existing folder
            or the new folder is positioned at the defined index.
        safe_names : bool, default False
            If True and any duplicated element names are found after the
            operation, names will be made unique (by appending '_1', '_2',
            '_3', etc.).
        """
        self._check_keys(data_key, filter_key, x_keys + y_keys)

        if orient == 'x':
            it, keys = x_keys, y_keys
        else:
            it, keys = y_keys, x_keys

        chains = []
        for key in it:
            xks, yks = ([key], keys) if orient == 'x' else (keys, [key])
            chain = Chain(self.stack, key)
            chain = chain.get(
                data_key, filter_key, xks, yks, views, rules, orient,
                prioritize)
            chains.append(chain)
        self._add_chains(chains, folder, index, safe_names)
        return chains

    def _check_keys(self, data_key, filter_key, keys):
        """
        Checks given keys exist in meta['columns']
        """
        if data_key not in self.stack.keys():
            err = "'{}' is not a valid data_key!".format(data_key)
            logger.error(err); raise KeyError(err)
        if filter_key not in self.stack[data_key].keys():
            err = "'{}' is not a valid filter_key!".format(filter_key)
            logger.error(err); raise KeyError(err)

        meta = self.stack[data_key].meta
        valids = meta["columns"].keys() + meta["masks"].keys() + ["@"]
        if any(k not in valids for k in keys):
            err = "Keys do not exist in meta columns or masks: {}"
            err = err.format([k for k in keys if k not in valids])
            logger.error(err); raise KeyError(err)

    def merge(self, other_cm, index=-1, safe_names=False):
        """
        Add elements from another ``ChainManager`` instance to self.

        Parameters
        ----------
        other_cm : ``quantipy.ChainManager``
            A ChainManager instance to draw the elements from.
        index : int, default -1
            The positional index after which new elements will be added.
            Defaults to -1, i.e. elements are appended at the end.
        safe_names : bool, default False
            If True and any duplicated element names are found after the
            operation, names will be made unique (by appending '_1', '_2',
            '_3', etc.).
        """
        if not isinstance(other_cm, ChainManager):
            err = "other_cm must be a quantipy.ChainManager instance."
            logger.error(err); raise ValueError(err)
        self._add_chains(other_cm.__chains, None, index, safe_names)
        return None

    @modify(to_list="chains")
    def remove(self, chains, folder=None, inplace=True):
        """
        Remove (folders of) ``qp.Chain`` items by providing a list of  indices
        or names.

        Parameters
        ----------
        chains : (list) of int and/or str
            ``qp.Chain`` items or folders by provided by their positional
            indices or ``name`` property.
        folder : str, default None
            If a folder name is provided, items will be dropped within that
            folder only instead of removing all found instances.
        inplace : bool, default True
            By default the new order is applied inplace, set to ``False`` to
            return a new object instead.
        """
        if inplace:
            cm = self
        else:
            cm = self.clone()
        if folder:
            org_chains, org_index = cm._set_to_folderitems(folder)
        remove_idxs = [cm._names_to_idxs.get(c, c) for c in chains]
        self._verify_dupes_in_ref(remove_idxs, "remove")
        new_items = []
        for pos, c in enumerate(cm):
            if pos not in remove_idxs:
                new_items.append(c)
        cm.__chains = new_items
        if folder:
            cm._rebuild_org_folder(folder, org_chains, org_index)
        if not inplace:
            return cm

    @modify(to_list="folders")
    def unite_folders(self, folders, new_name=None, drop=True):
        """
        Unite the items of two or more folders, optionally providing a new name

        If duplicated ``qp.Chain`` items are found, the first instance will be
        kept. The merged folder will take the place of the first folder named
        in ``folders``.

        Parameters
        ----------
        folders : list of int and/or str
            The folders to merge refernced by their positional index or by name
        new_name : str, default None
            Use this as the merged folder's name. If not provided, the name
            of the first folder in ``folders`` will be used instead.
        drop : bool, default True
            If ``False``, the original folders will be kept alongside the
            new merged one.
        """
        if len(folders) == 1:
            err = "'folders' must contain at least two folder names!"
            logger.error(err); raise ValueError(err)
        if not all(self._is_folder_ref(f) for f in folders):
            err = "One or more folder names from 'folders' do not exist!"
            logger.error(err); ValueError(err)
        folders = [self._folders_to_idx.get(f, f) for f in folders]
        folder_idx = self._names_to_idxs[folders[0]]
        if not new_name:
            new_name, folders = folders[0], folders[1:]
            seen_names = [c.name for c in self[new_name]]
        else:
            seen_names = []
        merged_items = []
        for folder in folders:
            for chain in self[folder]:
                if chain.name not in seen_names:
                    merged_items.append(chain)
                    seen_names.append(chain.name)

        self._add_chains(merged_items, new_name, folder_idx)
        if drop:
            self.remove(folders)
