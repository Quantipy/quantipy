#!/usr/bin/python
# -*- coding: utf-8 -*-
from ..__imports__ import *  # noqa

from .dataset import DataSet
from .meta import Meta

logger = get_logger(__name__)


class Batch(object):
    """
    A Batch is a construction plan for any deliverable (SPSS, XLSX, PPTX, ...).
    """
    # -------------------------------------------------------------------------
    # i/o
    # -------------------------------------------------------------------------
    def __init__(self, dataset, name, ci=['c', 'p'], weights=None, tests=None):
        if '-' in name:
            err = "Batch 'name' must not contain '-'!"
            logger.error(err); raise ValueError(err)
        if "batches" not in dataset.sets:
            dataset._meta["sets"]["batches"] = OrderedDict()
        self.dataset = dataset
        self._name = name
        self._batches = dataset._meta["sets"]["batches"]
        if name in dataset.batches:
            logger.info("Load batch: {}".format(name))
            self._load_batch()
        else:
            self._start_batch()
            self.cell_items = ci or []
            self.weights = weights
            self.set_sigproperties(tests)
        self._inherit_meta_functions()

    def _load_batch(self):
        """
        transform old batches (v0.1.1) into current form.
        """
        batch = self._batches[self.name]
        if "_section_starts" in batch:
            batch["sections"] = batch.pop("_section_starts")
        if "_variables" in batch:
            batch["variables"] = batch.pop("_variables")
        if "additional" in batch:
            batch["mains"] = []
            add = batch.pop("additional")
            if add:
                for ba, bdef in self._batches.items():
                    if self.name in bdef["additions"]:
                        batch["mains"].append(ba)
        if "filter_names" in batch:
            batch.pop("filter_names")
        if "forced_names" in batch:
            batch.pop("forced_names")
        if "meta_edits" in batch:
            edits = batch.pop("meta_edits")
            values = edits.pop("lib")
            edits["lib"] = {"values": values}
            meta = Meta(edits)
            batch["meta"] = meta
        if "language" in batch:
            batch["meta"].text_key = batch.pop("language")
        if "name" in batch:
            self.name = batch.pop("name")
        if "y_on_y_filter" in batch:
            batch.pop("y_on_y_filter")

    def _start_batch(self):
        meta = self.dataset._meta.clone()
        meta._clean_custom_sets_and_libs(True)
        self._batches[self.name] = {
            "mains": [],
            "additions": [],
            "xks": [],
            "yks": ["@"],
            "exclusive_yks_per_x": {},
            "extended_yks_per_x": {},
            "x_y_map": [],
            "y_on_y": [],
            "y_filter_map": {},
            "filter": None,
            "extended_filters_per_x": {},
            "x_filter_map": {},
            "variables": [],
            "verbatims": [],
            "weights": [],
            "sections": {},
            "total": True,
            "transposed": [],
            "leveled": {},
            "skip_items": [],
            "cell_items": [],
            "sigproperties": {},
            "unweighted_counts": False,
            "meta": meta,
            "build_info": {},
            "sample_size": len(self.dataset._data.index)
        }

    def _inherit_meta_functions(self):
        self.meta = self._batches[self.name]["meta"]
        self.set_hiding = self.meta.set_hiding
        self.set_sorting = self.meta.set_sorting
        self.set_slicing = self.meta.set_slicing
        self.used_text_keys = self.meta.used_text_keys
        self.force_texts = self.meta.force_texts
        self.replace_texts = self.meta.replace_texts
        self.select_text_keys = self.meta.select_text_keys
        self.remove_html = self.meta.remove_html
        self.set_text = self.meta.set_text
        self.set_item_texts = self.meta.set_item_texts
        self.set_value_texts = self.meta.set_value_texts
        self.replace_texts = self.meta.replace_texts
        self.reorder_values = self.meta.reorder_values
        self.reorder_items = self.meta.reorder_items
        self.set_property = self.meta.set_property
        self.del_property = self.meta.del_property
        self.set_missings = self.meta.set_missings
        self.del_missings = self.meta.del_missings

    def codes_in_data(self, name):
        """
        Get a list of codes that exist in (batch filtered) data.
        """
        if not self.filter:
            return self.dataset.codes_in_data(name)
        else:
            slicer = self.manifest_filter(self.filter)
            data = self.dataset[slicer, name].copy()
            if self.dataset.is_delimited_set(name):
                if not data.dropna().empty:
                    data_codes = data.str.get_dummies(';').columns.tolist()
                    data_codes = [int(c) for c in data_codes]
                else:
                    data_codes = []
            else:
                data_codes = pd.get_dummies(data).columns.tolist()
            return data_codes

    def hide_empty(self, xks=True, summaries=True):
        """
        Drop empty variables and hide array items from summaries.

        Parametes
        ---------
        xks : bool, default True
            Controls dropping "regular" variables and array items due to being
            empty.
        summaries : bool, default True
            Controls whether or not empty array items are hidden (by applying
            rules) in summary aggregations. Summaries that would end up with
            no valid items are automatically dropped altogether.
        """
        dbrks = self.downbreaks[:]
        for x in dbrks[:]:
            if self.dataset.is_array(x):
                e_items = self.dataset.empty(x, self.filter)
                if not e_items:
                    continue
                sources = self.dataset.get_sources(x)
                if summaries:
                    if len(e_items) == len(sources):
                        dbrks.remove(x)
                    else:
                        self.set_hiding(
                            x, e_items, axis='x', hide_values=False)
                if xks:
                    for i in e_items:
                        dbrks.remove(sources[i-1])
            elif not self.dataset.is_array_item(x):
                s = self.dataset[self.dataset.take(self.filter), x]
                if s.count() == 0:
                    dbrks.remove(x)
        self.downbreaks = dbrks

    def clone(self, name, b_filter=None, as_addition=False):
        """
        Create a copy of Batch instance.

        Parameters
        ----------
        name: str
            Name of the Batch instance that is copied.
        b_filter: str or qp complex logic
            Name of an existing filter variable or qp complex logic
        as_addition: bool, default False
            *  True: the new batch is added as addition to the master batch.

        Returns
        -------
        New/ copied Batch instance.
        """
        self._batches[name] = copy.deepcopy(self._batches[self.name])
        batch_copy = Batch(self.dataset, name)
        if b_filter:
            batch_copy.filter = b_filter
        if batch_copy.verbatims and b_filter and not as_addition:
            for oe in batch_copy.verbatims:
                oe["filter"] = batch_copy.filter
        if as_addition:
            batch_copy.as_addition(self.name)
        return batch_copy

    def remove(self):
        """
        Remove instance from meta object.
        """
        for mains in self.mains:
            self._batches[mains]["additions"].remove(self.name)
        for adds in self.additions:
            self._batches[adds]["mains"].remove(self.name)
        del self._batches[self.name]
        self = None

    # -------------------------------------------------------------------------
    # properties (simple get and set connection to meta)
    # -------------------------------------------------------------------------
    @property
    def build_info(self):
        return self._batches[self.name]["build_info"]

    @build_info.setter
    def build_info(self, info):
        self._batches[self.name]["build_info"] = info

    @property
    def cell_items(self):
        return self._batches[self.name]["cell_items"]

    @cell_items.setter
    @params(to_list=["ci"])
    def cell_items(self, ci):
        if any(c not in ['c', 'p', 'cp'] for c in ci):
            err = "'ci' cell items must be either 'c', 'p' or 'cp'."
            logger.error(err); raise ValueError(err)
        self._batches[self.name]["cell_items"] = ci

    @property
    def text_key(self):
        return self.meta.text_key

    @text_key.setter
    def text_key(self, value):
        self.meta.text_key = value

    @property
    def unweighted_counts(self):
        return self._batches[self.name]["unweighted_counts"]

    @unweighted_counts.setter
    def unweighted_counts(self, unwgt):
        if not isinstance(unwgt, bool):
            err = "'unweighted_counts' must be of type 'bool'!"
            logger.error(err); raise TypeError(err)
        self._batches[self.name]["unweighted_counts"] = unwgt

    @property
    def variables(self):
        return self._batches[self.name]["variables"]

    @variables.setter
    @params(to_list=["varlist"])
    def variables(self, varlist):
        self._batches[self.name]["variables"] = self.dataset.unroll(
            varlist, both="all")

    @property
    def weights(self):
        return self._batches[self.name]["weights"]

    @weights.setter
    @params(to_list=["w"])
    def weights(self, w):
        if not w or any(we is None for we in w):
            w = [None] + [we for we in w if we]
        if any(we not in self.dataset.columns for we in w if we):
            err = "{} contains invalid variable name.".format(w)
            logger.error(err); raise ValueError(err)
        self._batches[self.name]["weights"] = w

    # -------------------------------------------------------------------------
    # properties (only set by update)
    # -------------------------------------------------------------------------
    @property
    def sample_size(self):
        return self._batches[self.name]["sample_size"]

    def _samplesize_from_batch_filter(self):
        if self.filter:
            idx = self.dataset.manifest_filter(self.filter)
        else:
            idx = self.dataset._data.index
        self._batches[self.name]["sample_size"] = len(idx)

    # -------------------------------------------------------------------------
    # name
    # -------------------------------------------------------------------------
    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name):
        if new_name in self.dataset.batches:
            err = "'{}' is already included!".format(new_name)
            logger.error(err); raise ValueError(err)
        for main in self.mains:
            adds = self._batches[main]["additions"]
            adds[adds.index(self.name)] = new_name
        for add in self.additions:
            mains = self._batches[add]["mains"]
            mains[mains.index(self.name)] = new_name
        batches[new_name] = batches.pop(self.name)
        self._name = new_name

    # -------------------------------------------------------------------------
    # additions and mains
    # -------------------------------------------------------------------------
    @property
    def additions(self):
        return self._batches[self.name]["additions"]

    @property
    def mains(self):
        return self._batches[self.name]["mains"]

    def as_addition(self, bname):
        """
        Treat the Batch as additional aggregations, independent from the
        global Batch & Build setup.

        Parameters
        ----------
        bname: str
            Name of the Batch instance where the current instance is added to.
        """
        self._batches[bname]['additions'].append(self.name)
        self._batches[self.name]["mains"].append(bname)
        self._batches[self.name]["verbatims"] = []
        self._batches[self.name]["y_on_y"] = []

    def as_main(self):
        """
        Treat the Batch no longer as additional.
        """
        for main in self.mains[:]:
            self._batches[main]['additions'].remove(self.name)
            self.mains.remove(main)

    # -------------------------------------------------------------------------
    # filter
    # -------------------------------------------------------------------------
    @property
    def filter(self):
        return self._batches[self.name]["filter"]

    @filter.setter
    def filter(self, filter_def):
        if not filter:
            self._batches[self.name]["filter"] = None
        elif isinstance(filter_def, str):
            if self.dataset.is_filter(filter_def):
                self._batches[self.name]["filter"] = filter_def
            else:
                err = "'{}' is not a valid filter variable.".format(filter_def)
                logger.error(err); raise ValueError(err)
        elif isinstance(filter_def, tuple):
            fname = self.dataset.valid_var_name(filter_def[0])
            self.dataset.add_filter_var(fname, filter_def[1])
            self._batches[self.name]["filter"] = fname
        else:
            err = (
                "'filter_def' must either be name of a filter variable or"
                "tuple in form  of (filter_name, filter_logic).")
            logger.error(err); raise TypeError(err)
        self._samplesize_from_batch_filter()

    @property
    def extended_filters_per_x(self):
        return self._batches[self.name]["extended_filters_per_x"]

    def extend_filter(self, ext_filter, on=None):
        """
        Apply additonal filtering to specific x (downbreak) variables.

        Parameters
        ----------
        ext_filter: qp complex logic
            dict with variable name(s) as key, str or tupel of str, and logic
            as value. For example:
            ext_filters = {'q1': {'gender': 1}, ('q2', 'q3'): {'gender': 2}}
        """
        not_valid = [o for o in on if o not in self.downbreaks]
        if not_valid:
            err = '{} not defined as xks.'.format(not_valid)
            logger.error(err); raise ValueError(err)
        on = self.unroll(on)
        for x in on:
            fname = self.dataset.valid_var_name("{}_{}".format(self.filter, x))
            self.dataset.extend_filter_var(self.filter, ext_filter, x)
            self.extended_filters_per_x.update({x: f_name})
        self._map_x_to_y_and_filter()

    # -------------------------------------------------------------------------
    # sigproperties
    # -------------------------------------------------------------------------
    @property
    def sigproperties(self):
        return self._batches[self.name]["sigproperties"]

    @params(to_list=["levels"])
    def set_sigproperties(self, levels=None, flags=[30, 100],
                          test_total=False):
        """
        Specify a significance test setup.

        Parameters
        ----------
        levels: float/ list of float
            Level(s) for significance calculation(s).
        flags: list
            Flag low bases.
        test_total: bool
            * True: Add test against total column.
        mimic:
            Currently not implemented.
        """
        if not self.total and levels:
            err = "Cannot add sigtests without total column."
            logger.warning(err)
            levels = None
        levels = sorted([float(level) for level in levels])
        self._batches[self.name]["sigproperties"] = {
            'siglevels': levels,
            'test_total': test_total,
            'flag_bases': flags,
            'mimic': ['Dim']
        }

    # -------------------------------------------------------------------------
    # sections
    # -------------------------------------------------------------------------
    @property
    def sections(self):
        starts = self._batches[self.name]["sections"]
        sections = OrderedDict()
        sect = None
        for x in self.downbreaks:
            sect = starts.get(x, sect)
            if sect:
                if sect not in sections:
                    sections[sect] = []
                sections[sect].append(x)
        return sections

    def set_section(self, x_anchor, section):
        """
        Define section for downbreaks

        Parameters
        ----------
        x_anchor: str
            First variable of the new section.
        section: str
            Name of the new section.
        """
        if x_anchor in self.downbreaks:
            self._batches[self.name]["sections"][x_anchor] = section

    def del_section(self, section):
        """
        Remove section.
        """
        for k, v in self._batches[self.name]["sections"].items():
            if v == section:
                del self._batches[self.name]["sections"][k]

    # -------------------------------------------------------------------------
    # breaks
    # -------------------------------------------------------------------------
    @property
    def total(self):
        return self._batches[self.name]["total"]

    @total.setter
    def total(self, value):
        if not isinstance(value, bool):
            err = "'total' must be of type 'bool'."
            logger.error(err); raise TypeError(err)
        self._batches[self.name]["total"] = value
        if not value:
            self.set_sigproperties(None)
        self._map_x_to_y_and_filter()

    @property
    def crossbreaks(self):
        return self._batches[self.name]["yks"]

    @crossbreaks.setter
    @params(to_list=["xbrk"])
    def crossbreaks(self, xbrk):
        self._batches[self.name]["yks"] = self.dataset.unroll(xbrk)
        self._map_x_to_y_and_filter()

    @property
    def extended_yks_per_x(self):
        return self._batches[self.name]["extended_yks_per_x"]

    @params(to_list=['ext_xbrk', 'on'])
    def extend_crossbreaks(self, ext_xbrk, on=None):
        """
        Extend crossbreak/banner variables on specific downbreak variables.

        Parameters
        ----------
        ext_xbrk: str/ list or dict
            *  str/ list: Name(s) of variable(s) that are added as downbreak.
            *  dict: New crossbreaks are added by anchor.
        on: (list of) str
            Name(s) of variable(s) in the xks (downbreaks) for which the
            crossbreak should be extended.
        """
        ext_xbrk = self.dataset.unroll(ext_xbrk)
        if not on:
            self.crossbreaks.extend(ext_xbrk)
        else:
            on = self.unroll(on)
            for x in on:
                self.extended_yks_per_x.update({x: ext_xbrk})
        self._map_x_to_y_and_filter()

    @property
    def exclusive_yks_per_x(self):
        return self._batches[self.name]["exclusive_yks_per_x"]

    @params(to_list=['new_xbrk', 'on'])
    def replace_crossbreaks(self, new_xbrk, on):
        """
        Replace crossbreak/banner variables on specific downbreak variables.

        Parameters
        ----------
        new_xbrk: (list of) str
            Name(s) of variable(s) that are used as crossbreak.
        on: (list of) str
            Name(s) of variable(s) in the xks (downbreaks) for which the
            crossbreak should be replaced.
        """
        new_xbrk = self.dataset.unroll(new_xbrk)
        on = self.unroll(on)
        for x in on:
            self.exclusive_yks_per_x.update({x: new_yks})
        self._map_x_to_y_and_filter()

    @property
    def downbreaks(self):
        return self._batches[self.name]["xks"]

    @downbreaks.setter
    @params(to_list=["dbrk"])
    def downbreaks(self, dbrk):
        self._batches[self.name]["xks"] = self.dataset.unroll(dbrk, both="all")
        self._map_x_to_y_and_filter()

    def extend_downbreaks(self, ext_dbrk):
        """
        Extend downbreak variables with additional variables.

        Parameters
        ----------
        ext_dbrk: str/ list or dict
            *  str/ list: Name(s) of variable(s) that are added as downbreak.
            *  dict: New downbreaks are added by anchor.
        """
        dbrk = self.downbreaks[:]
        if isinstance(ext_dbrk, (str, list)):
            dbrk.extend(self.dataset.unroll(ext_dbrk, both="all"))
        elif isinstance(ext_dbrk, dict):
            for k, v in ext_dbrk.items():
                ext_dbrk[k] = self.dataset.unroll(v, both="all")
            dbrk = insert_by_anchor(dbrk, ext_dbrk)
        self.downbreaks = dbrk

    @property
    def x_y_map(self):
        return self._batches[self.name]["x_y_map"]

    @property
    def x_filter_map(self):
        return self._batches[self.name]["x_filter_map"]

    def _map_x_to_y_and_filter(self):
        def _get_yks(x):
            if x in self.exclusive_yks_per_x:
                yks = self.exclusive_yks_per_x[x]
            else:
                yks = copy.deepcopy(self.crossbreaks)
                yks = insert_by_anchor(yks, self.extended_yks_per_x.get(x, []))
            if self.total and "@" not in yks:
                yks = ["@"] + yks
            return yks

        x_y_map = []
        x_f_map = {}
        for x in self.downbreaks:
            fname = self.extended_filters_per_x.get(x, self.filter)
            if self.dataset.is_array(x):
                x_y_map.append((x, ["@"]))
                x_f_map[x] = fname
                if x in self.leveled:
                    level = "{}_level".format(x)
                    x_y_map.append((level, _get_yks(x)))
                    x_f_map[level] = fname
                if x in self.transposed:
                    x_y_map.append(("@", [x]))
                if x not in self.skip_items:
                    rules = self.meta.get_rules(x)
                    hiding = rules.get('dropx', {}).get('values', [])
                    for x2 in self.dataset.get_sources(x):
                        if x2 in hiding:
                            continue
                        elif x2 in self.downbreaks:
                            x_y_map.append((x2, _get_yks(x2)))
                            x_f_map[x2] = fname
            elif not self.dataset.is_array_item(x):
                x_y_map.append((x, _get_yks(x)))
                x_f_map[x] = fname
        self._batches[self.name]["x_y_map"] = x_y_map
        self._batches[self.name]["x_filter_map"] = x_f_map

    # -------------------------------------------------------------------------
    # arrays in dbrks
    # -------------------------------------------------------------------------
    @property
    def transposed(self):
        return self._batches[self.name]["transposed"]

    @transposed.setter
    @params(to_list="arrays")
    def transposed(self, arrays):
        if not all(self.dataset.is_array(a) for a in arrays):
            err = "Can only transpose arrays!"
            logger.error(err); raise TypeError(err)
        self._batches[self.name]["transposed"] = arrays
        self._map_x_to_y_and_filter()

    @property
    def skip_items(self):
        return self._batches[self.name]["skip_items"]

    @skip_items.setter
    @params(to_list="arrays")
    def skip_items(self, arrays):
        if not all(self.dataset.is_array(a) for a in arrays):
            err = "Can only skip array items!"
            logger.error(err); raise TypeError(err)
        self._batches[self.name]["skip_items"] = arrays
        self._map_x_to_y_and_filter()

    @property
    def leveled(self):
        return self._batches[self.name]["leveled"]

    @leveled.setter
    @params(to_list="arrays")
    def leveled(self, arrays):
        if not all(self.dataset.is_array(a) for a in arrays):
            err = "Can only level arrays!"
            logger.error(err); raise TypeError(err)
        self._batches[self.name]["leveled"] = arrays
        self._map_x_to_y_and_filter()

    # -------------------------------------------------------------------------
    # verbatims
    # -------------------------------------------------------------------------
    @property
    def verbatims(self):
        return self._batches[self.name]["verbatims"]

    @params(to_list=['oe', 'break_by'])
    def set_verbatims(self, oe, break_by=None, drop_empty=True, incl_nan=False,
                      replacements=None, title='open ends', filter_by=None):
        """
        Create respondent level based listings of open-ended text data.

        Parameters
        ----------
        oe : str or list of str
            The open-ended questions / verbatims to be added to the stack.
        break_by : str or list of str, default None
            If provided, these variables will be presented alongside the ``oe``
            data.
        drop_empty : bool, default True
            Case data that is missing valid entries will be dropped from the
            output.
        incl_nan: bool, default False
            Show __NaN__ in the output.
        replacements: dict, default None
            Replace strings in data.
        title : str, default 'open ends'
            Specifies the the Excel sheet name for the output. Overwrites
            verbatims if title already exists.
        filter_by : str or tuple
            *  str: name of a filter variable
            *  tuple: (filter_name, logic)
            Note: if filter_by is not provided, the batch's filter is taken
            automatically.
        """
        if self.mains:
            err = "Cannot add verbatims to additional batches."
            logger.error(err); raise NotImplementedError(err)
        dupes = dupes_in_list(oe + break_by)
        if dupes:
            err = "duplicates '{}' included in oe and break_by.".format(dupes)
            logger.error(err); raise ValueError(err)
        if not isinstance(title, str):
            err = "'title' must be of type string."
            logger.error(err); raise TypeError(err)
        if not replacements:
            replacements = {}
        elif not isinstance(repl, dict):
            err = "'replacements' must be of type dict."
            logger.error(err); raise TypeError(err)
        elif not all(isinstance(v, dict) for v in list(repl.values())):
            replacements = {self.text_key: replacements}

        if not filter_by:
            slicer = self.filter
        elif isinstance(filter_by, str):
            if self.dataset.is_filter(filter_by):
                slicer = filter_by
            else:
                err = "'{}' is not a valid filter variable.".format(filter_by)
                logger.error(err); raise ValueError(err)
        elif isinstance(filter_by, tuple):
            fname = self.dataset.valid_var_name(filter_by[0])
            self.dataset.add_filter_var(fname, filter_by[1])
            slicer = fname

        if any(title == verbatim["title"] for verbatim in self.verbatims):
            for verbatim in self.verbatims:
                if verbatim["title"] == title:
                    verbatim.update({
                        'filter': slicer,
                        'columns': oe,
                        'break_by': break_by,
                        'incl_nan': incl_nan,
                        'drop_empty': drop_empty,
                        'replace': replacements
                    })
        else:
            self.verbatims.append({
                'title': title,
                'filter': slicer,
                'columns': oe,
                'break_by': break_by,
                'incl_nan': incl_nan,
                'drop_empty': drop_empty,
                'replace': replacements
            })

    @params(to_list="title")
    def del_verbatims(self, title):
        """
        Remove verbatim definitions by title.
        """
        for idx, verbatim in enumerate(self.verbatims[:]):
            if verbatim["title"] in title:
                del self.verbatims[idx]

    # -------------------------------------------------------------------------
    # y_on_y
    # -------------------------------------------------------------------------
    @property
    def y_on_y(self):
        return self._batches[self.name]["y_on_y"]

    @property
    def y_filter_map(self):
        return self._batches[self.name]["y_filter_map"]

    def set_y_on_y(self, name, filter_by=None):
        """
        Produce aggregations crossing the (main) y variables with each other.

        Parameters
        ----------
        name: str
            key name for the y on y aggregation.
        filter_by: str or tuple
            *  str: name of a filter variable
            *  tuple: (filter_name, logic)
            Note: if filter_by is not provided, the batch's filter is taken
            automatically.
        """
        if not isinstance(name, str):
            err = "'name' attribute for add_y_on_y must be a str!"
            logger.error(err); raise TypeError(err)
        if not filter_by:
            self.y_filter_map[name] = self.filter
        elif isinstance(filter_by, str):
            if self.dataset.is_filter(filter_by):
                self.y_filter_map[name] = filter_by
            else:
                err = "'{}' is not a valid filter variable.".format(filter_by)
                logger.error(err); raise ValueError(err)
        elif isinstance(filter_by, tuple):
            fname = self.dataset.valid_var_name(filter_by[0])
            self.dataset.add_filter_var(fname, filter_by[1])
            self.y_filter_map[name] = fname
        if name not in self.y_on_y:
            self.y_on_y.append(name)

    @params(to_list="name")
    def del_verbatims(self, name):
        """
        Remove verbatim definitions by name.
        """
        for idx, y_on_y in enumerate(self.y_on_y[:]):
            if y_on_y == name:
                del self.y_on_y[idx]
                del self.y_filter_map[y_on_y]

    # -------------------------------------------------------------------------
    # to_dataset
    # -------------------------------------------------------------------------
    @params(to_list=["mode", "misc"])
    def to_dataset(self, mode=None, from_set="data file",
                   additions="sort_within", apply_edits=False,
                   integrate_rc=(["_rc", "_rb"], True),
                   misc=["RecordNo", "caseid", "identity"]):
        """
        Create a qp.DataSet instance out of the batch settings.

        Parameters
        ----------
        mode: list of str {'x', 'y', 'v', 'oe', 'w', 'f'}
            Variables to keep.
        from_set: str or list of str, default None
            Set name or a list of variables to sort against.
        additions: str {'sort_within, sort_between', False}
            Add variables from additional batches.
        apply_edits: bool
            * true: Edits like hiding/slicing/... are applied as actual
                variable adjustments.
        """
        # prepare variable list
        if not mode:
            mode = ['x', 'y', 'v', 'oe', 'w', 'f']
        vlist = self._get_vlist(mode)

        if additions == "sort_between":
            for add in self.additions:
                abatch = self.dataset.get_batch(add)
                vlist += abatch._get_vlist(mode)
        if not from_set:
            from_set = vlist
        vlist = self.align_order(vlist, from_set, integrate_rc, fix=misc)
        if additions == "sort_within":
            for add in self.additions:
                abatch = self.dataset.get_batch(add)
                add_list = abatch._get_vlist(mode)
                add_list = self.dataset.align_order(
                    add_list, from_set, integrate_rc, fix=misc)
                vlist += add_list
        vlist = uniquify_list(vlist)
        vlist = self.dataset.roll_up(vlist)

        # handle filters
        add_f = [self._batches[add]["filter"] for add in self.additions]
        filters = uniquify_list([self.filter] + add_f)
        if add_f and self.dataset.compare_filter(self.filter, add_f):
            f = "merge_filter"
            merge_f = filters
        else:
            f = self.filter
            merge_f = False

        # create ds
        ds = DataSet.from_components(
            self.name, self.dataset._data, self.meta, self.text_key)
        if merge_f:
            ds.merge_filter(f, filters)
        if f:
            ds.filter(self.name, {f: 0}, True)
            if merge_f:
                ds.drop(f)

        ds.create_set(str(self.name), included=vlist, overwrite=True)
        ds.subset(from_set=self.name, inplace=True)
        ds.order(vlist)

        if apply_edits:
            for var in ds.variables():
                codes = ds.get_codes(col)
                drops = ds.get_rules(col).pop("dropx", [])
                slicer = ds.get_rules(col).pop("slicex", codes)
                if all(isinstance(dr, str) for dr in drops):
                    ds.remove_items(var, [ds.get_item_no(dr) for dr in drops])
                elif drops or not codes == slicer:
                    remove = [code for code in slicer if not code in drops]
                    ds.remove_values(var, remove)
        if "oe" in mode:
            self._apply_oe_replacements(ds)
        return ds

    def _get_vlist(self, mode):
        match = {
            "x": "xks",
            "y": "yks",
            "v": "variables",
            "w": "weights"
        }
        vlist = []
        for key in mode:
            if key == "oe":
                oes = []
                for oe in var[:]:
                    if 'f' in mode:
                        oes += oe["columns"] + [oe["filter"]]
                    else:
                        oes += oe['columns']
                var = oes
            elif key == "f":
                var = [self.filter] + list(self.y_filter_map.values())
            else:
                var = self._batches[self.name][match[key]]
            for v in ensure_list(var):
                if v and v in self.dataset and v not in vlist:
                    vlist.append(v)
        return vlist

    def _apply_oe_replacements(self, dataset):
        numerical = ["int", "single", "is_delimited_set"]
        for oe in self.verbatims:
            if oe['replace']:
                data = dataset._data[oe["columns"]].copy()
                replacements = oe['replace'].get(self.text_key, {})
                for target, repl in replacements.items():
                    if not repl:
                        repl = np.NaN
                    data.replace(target, repl, inplace=True)
                dataset._data[oe["columns"]] = data
            if not oe['incl_nan']:
                for col in oe['columns']:
                    if not (ds.is_categorical(col) or ds.is_numeric(col)):
                        dataset._data[col].replace(np.NaN, '', inplace=True)

    # -------------------------------------------------------------------------
    # Deprecations
    # -------------------------------------------------------------------------
    def rename(self, new_name):
        """
        Rename instance and all dependet references.
        """
        warn = "This method will be deprecated soon.\n"
        warn += "Please use property setter of '{}' instead."
        logger.warning(warn.format("batch.name"))
        self.name = new_name

    def set_language(self, text_key):
        """
        Set ``Batch.language`` indicated via the ``text_key`` for Build exports

        Parameters
        ----------
        text_key: str
            The text_key used as language for the Batch instance
        """
        warn = "This method will be deprecated soon.\n"
        warn += "Please use property setter of '{}' instead."
        logger.warning(warn.format("batch.text_key"))
        self.text_key = text_key

    def set_cell_items(self, ci):
        """
        Assign cell items ('c', 'p', 'cp').

        Parameters
        ----------
        ci: str/ list of str, {'c', 'p', 'cp'}
            Cell items used for this Batch instance.
        """
        warn = "This method will be deprecated soon.\n"
        warn += "Please use property setter of '{}' instead."
        logger.warning(warn.format("batch.cell_items"))
        self.cell_items = ci

    def set_unweighted_counts(self, unwgt):
        """
        Assign if counts (incl. nets) should be aggregated unweighted.
        """
        warn = "This method will be deprecated soon.\n"
        warn += "Please use property setter of '{}' instead."
        logger.warning(warn.format("batch.unweighted_counts"))
        self.unweighted_counts = unwgt

    def set_weights(self, w):
        """
        Assign a weight variable setup.

        Parameters
        ----------
        w: str/ list of str
            Name(s) of the weight variable(s).
        """
        warn = "This method will be deprecated soon.\n"
        warn += "Please use property setter of '{}' instead."
        logger.warning(warn.format("batch.weights"))
        self.weights = w

    def add_variables(self, varlist):
        """
        Add additional variables to batch setup.

        Parameters
        ----------
        varlist : list
            A list of variable names.

        Note
        ----
        These variables won't appear in xlsx or pptx deliverable, but in sav
        outputs.
        """
        warn = "This method will be deprecated soon.\n"
        warn += "Please use property setter of '{}' instead."
        logger.warning(warn.format("batch.variables"))
        self.variables = varlist

    def add_downbreak(self, dbrk):
        """
        Set the downbreak (x) variables of the Batch.

        Parameters
        ----------
        dbrk: str, list of str, dict, list of dict
            Names of variables that are used as downbreaks. Forced names for
            Excel outputs can be given in a dict, for example:
            xks = ['q1', {'q2': 'forced name for q2'}, 'q3', ....]
        """
        warn = "This method will be deprecated soon.\n"
        warn += "Please use property setter of '{}' instead."
        logger.warning(warn.format("batch.downbreaks"))
        self.downbreaks = dbrk

    def level(self, array):
        """
        Produce leveled (a flat view of all item reponses) array aggregations.

        Parameters
        ----------
        array: str/ list of str
            Names of the arrays to add the levels to.
        """
        warn = "This method will be deprecated soon.\n"
        warn += "Please use property setter of '{}' instead."
        logger.warning(warn.format("batch.leveled"))
        self.leveled = array

    def add_crossbreak(self, xbrk):
        """
        Set the y (crossbreak/banner) variables of the Batch.

        Parameters
        ----------
        xbrk: str, list of str
            Variables that are added as crossbreaks. '@'/ total is added
            automatically.
        """
        warn = "This method will be deprecated soon.\n"
        warn += "Please use property setter of '{}' instead."
        logger.warning(warn.format("batch.crossbreaks"))
        self.crossbreaks = xbrk

    def transpose(self, name):
        """
        Create transposed aggregations for the requested variables.

        Parameters
        ----------
        name: str/ list of str
            Name of variable(s) for which transposed aggregations will be
            created.
        """
        warn = "This method will be deprecated soon.\n"
        warn += "Please use property setter of '{}' instead."
        logger.warning(warn.format("batch.transposed"))
        self.transposed = name

    def add_total(self, total=True):
        """
        Define if '@' is added to y_keys.
        """
        warn = "This method will be deprecated soon.\n"
        warn += "Please use property setter of '{}' instead."
        logger.warning(warn.format("batch.total"))
        self.total = total

    def set_filter(self, filter_def):
        """
        Apply a (global) filter to all the variables found in the Batch.

        Parameters
        ----------
        filter_def: str or qp complex logic
            *  str: Name of an existing filter variable.
            *  tuple: (filter_name, filter_def)
        """
        warn = "This method will be deprecated soon.\n"
        warn += "Please use property setter of '{}' instead."
        logger.warning(warn.format("batch.filter"))
        self.filter = filter_def