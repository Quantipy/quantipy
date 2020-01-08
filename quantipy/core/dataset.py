#!/usr/bin/python
# -*- coding: utf-8 -*-

from ..__imports__ import *  # noqa

from .io import (
    quantipy_from_dimensions,
    DimensionsWriter,
    # quantipy_from_ascribe,
    # parse_sav_file,
    # dimensions_from_quantipy,
    # save_sav
)
from .meta import Meta

logger = get_logger(__name__)


class DataSet(object):
    """
    A set of casedata ``pandas.DataFrame`` and meta data ``qp.Meta``.
    """
    def __init__(self, name, meta=None, data=None):
        self.name = name
        self.path = "."
        if data is None:
            data = pd.DataFrame()
        self._data = data
        if meta is None:
            meta = Meta()
        self._meta = meta
        self._inherit_meta_functions()

        # default fixes
        self.repair()

    def __contains__(self, name):
        return self._meta.var_exists(name)

    def __delitem__(self, name):
        self.drop(name)

    def __getitem__(self, var):
        if isinstance(var, tuple):
            sliced_access = True
            slicer = var[0]
            var = var[1]
        else:
            sliced_access = False
        var = self.unroll(var)
        if len(var) == 1:
            var = var[0]
        if sliced_access:
            return self._data.ix[slicer, var]
        else:
            return self._data[var]

    def __setitem__(self, name, val):
        if isinstance(name, tuple):
            sliced_insert = True
            slicer = name[0]
            name = name[1]
        else:
            sliced_insert = False
        scalar_insert = isinstance(val, (int, float, str))
        if scalar_insert and not np.isnan(val):
            if self.is_categorical(name):
                valid_codes = self.get_codes(name)
                if val not in valid_codes:
                    err = "{} is undefined for '{}'! Valid: {}".format(
                        val, name, self.get_codes(name))
                    logger.error(err); raise ValueError(err)
            if self.get_type(name) == 'delimited set':
                val = '{};'.format(val)
        if sliced_insert:
            self._data.loc[slicer, name] = val
        else:
            self._data[name] = val

    @property
    def masks(self):
        return self._meta.masks

    @property
    def columns(self):
        return self._meta.columns

    @property
    def singles(self):
        return self._meta.singles

    @property
    def delimited_sets(self):
        return self._meta.delimited_sets

    @property
    def ints(self):
        return self._meta.ints

    @property
    def floats(self):
        return self._meta.floats

    @property
    def dates(self):
        return self._meta.dates

    @property
    def strings(self):
        return self._meta.strings

    @property
    def filters(self):
        return self._meta.filters

    @property
    def hidden_arrays(self):
        return self._meta.hidden_arrays

    @property
    def sets(self):
        return self._meta.sets

    @property
    def batches(self):
        return self._meta.batches

    @property
    def text_key(self):
        return self._meta.text_key

    @text_key.setter
    def text_key(self, value):
        self._meta.text_key = value

    @property
    def valid_tks(self):
        return self._meta.valid_tks

    @valid_tks.setter
    def valid_tks(self, value):
        self._meta.valid_tks = value

    @property
    def dimensions_comp(self):
        return self._meta.dimensions_comp

    @dimensions_comp.setter
    def dimensions_comp(self, value):
        self._meta.dimensions_comp = value

    @property
    def dimensions_suffix(self):
        return self._meta.dimensions_suffix

    @dimensions_suffix.setter
    def dimensions_suffix(self, value):
        self._meta.dimensions_suffix = value

    def _inherit_meta_functions(self):
        # list variables
        self.describe = self._meta.describe
        self.variables = self._meta.variables
        self.variables_from_set = self._meta.variables_from_set
        self.by_type = self._meta.by_type
        self.by_property = self._meta.by_property

        # variable names
        self.find = self._meta.find
        self.names = self._meta.names
        self.get_weak_dupes = self._meta.get_weak_dupes
        self.valid_var_name = self._meta.valid_var_name

        # inspect variable
        self.is_single = self._meta.is_single
        self.is_delimited_set = self._meta.is_delimited_set
        self.is_int = self._meta.is_int
        self.is_float = self._meta.is_float
        self.is_date = self._meta.is_date
        self.is_string = self._meta.is_string
        self.is_array = self._meta.is_array
        self.is_array_item = self._meta.is_array_item
        self.is_numeric = self._meta.is_numeric
        self.is_filter = self._meta.is_filter
        self.is_categorical = self._meta.is_categorical

        # get and set/ modify variable information
        # types
        self.get_type = self._meta.get_type
        self.get_subtype = self._meta.get_subtype
        # texts
        self.get_text = self._meta.get_text
        self.set_text = self._meta.set_text
        self.remove_html = self._meta.remove_html
        self.replace_texts = self._meta.replace_texts
        self.repair_text_edits = self._meta.repair_text_edits
        # values
        self.get_values = self._meta.get_values
        self.extend_values = self._meta.extend_values
        self.reorder_values = self._meta.reorder_values
        self.get_value_texts = self._meta.get_value_texts
        self.set_value_texts = self._meta.set_value_texts
        self.get_codes = self._meta.get_codes
        self.get_codes_from_label = self._meta.get_codes_from_label
        # items / arrays
        self.get_items = self._meta.get_items
        self.get_item_no = self._meta.get_item_no
        self.get_item_texts = self._meta.get_item_texts
        self.set_item_texts = self._meta.set_item_texts
        self.reorder_items = self._meta.reorder_items
        self.get_parent = self._meta.get_parent
        self.get_sources = self._meta.get_sources
        self.unbind = self._meta.unbind
        # factors
        self.get_factors = self._meta.get_factors
        self.set_factors = self._meta.set_factors
        self.del_factors = self._meta.del_factors
        # properties
        self.get_property = self._meta.get_property
        self.set_property = self._meta.set_property
        self.del_property = self._meta.del_property
        # missings
        self.get_missings = self._meta.get_missings
        self.set_missings = self._meta.set_missings
        self.del_missings = self._meta.del_missings
        # rules
        self.get_rules = self._meta.get_rules
        self.set_sorting = self._meta.set_sorting
        self.set_hiding = self._meta.set_hiding
        self.set_slicing = self._meta.set_slicing
        # text_keys
        self.used_text_keys = self._meta.used_text_keys
        self.force_texts = self._meta.force_texts
        self.select_text_keys = self._meta.select_text_keys
        # lists and sets
        self.get_set = self._meta.get_set
        self.create_set = self._meta.create_set
        self.extend_set = self._meta.extend_set
        self.roll_up = self._meta.roll_up
        self.unroll = self._meta.unroll
        # batches
        self.get_batches = self._meta.get_batches
        self.adds_per_mains = self._meta.adds_per_mains

    # -------------------------------------------------------------------------
    # file i/o / conversions
    # -------------------------------------------------------------------------
    @staticmethod
    def _verbose_io(name, path, software, load=True):
        if load:
            msg = "Load qp.DataSet from '{}' data: '{}' ('{}')"
        else:
            msg = "Save qp.DataSet as '{}' data: '{}' ('{}')"
        logger.info(msg.format(software, name, path))

    def clone(self):
        """
        Get a deep copy of the ``DataSet`` instance.
        """
        name = self.name
        meta = self._meta.clone()
        data = self._data.copy()
        return DataSet.from_components(name, data, meta, self.text_key)

    def filter(self, condition, inplace=False):
        """
        Filter the instance's data.

        Parameters
        ----------
        condition : str or qp.Logic
            Name of a filter variable or logic, which is used to slice the data
        inplace : bool, default False
            Return a new instance or modify the instance inplace.
        """
        ds = self if inplace else self.clone()
        slicer = ds.take(condition)
        ds._data = ds._data.iloc[slicer, :]
        if not inplace:
            return ds

    def subset(self, variables=None, from_set=None, inplace=False):
        """
        Create a version of self with a reduced collection of variables.

        Parameters
        ----------
        variables : str or list of str, default None
            A list of variable names to include in the new DataSet instance.
        from_set : str
            The name of an already existing set to base the new DataSet on.
        inplace : bool, default False
            Return a new instance or modify the instance inplace.
        """
        ds = self if inplace else self.clone()
        ds._meta.subset(variables, from_set, inplace=True)
        keep = ds.unroll(ds.variables_from_set("data file")) + ["@1"]
        drop = [col for col in ds._data.columns if col not in keep]
        ds._data.drop(drop, 1, inplace=True)
        if not inplace:
            return ds

    @classmethod
    def from_quantipy(cls, name, path=".", reset=True):
        """
        Load Quantipy .csv/.json files, connecting as data and meta components.

        Parameters
        ----------
        name : str
            The name (without suffix) of the csv and json files to load.
        path : str
            The path, where the csv and json files are located.
        reset : bool, default True
            Clean the `'lib'` and ``'sets'`` metadata collections from
            non-native entries.

        Example:
        ```
        dataset = qp.DataSet.from_quantipy(
            name="Example Data (A)",
            path="../tests/data/",
            reset=True)
        ```
        """
        path_json = os.path.join(path, u"{}.json".format(name))
        meta = Meta.from_json(path_json, reset)
        path_csv = os.path.join(path, u"{}.csv".format(name))
        data = load_csv(path_csv)
        dataset = cls(name, meta, data)
        dataset.path = path
        dataset._verbose_io(name, path, "quantipy")
        return dataset

    @classmethod
    def from_components(cls, name, data, meta=None, text_key=None):
        """
        Attach data and meta directly to the ``DataSet`` instance.

        Parameters
        ----------
        data : pandas.DataFrame
            A DataFrame that contains case data entries for the ``DataSet``.
        meta: qp.Meta instance, default None
            A qp.Meta instance that stores meta data describing the columns of
            the dataframe.
        text_key : str, default None
            The text_key to be used. If not provided, it will be attempted to
            use the 'default text' from the ``meta['lib']`` definition.
        """
        if not isinstance(data, pd.DataFrame):
            err = "'data' must be a pandas.DataFrame."
            logger.error(err); raise TypeError(err)
        if meta and not isinstance(meta, Meta):
            err = "'meta' must be qp.Meta."
            logger.error(err); raise TypeError(err)
        elif not meta:
            meta = Meta.inferred_from_df(data, text_key)
        dataset = cls(name, meta, data)
        return dataset

    @classmethod
    def from_stack(cls, stack, dk, fk=None):
        """
        Use ``qp.Stack`` data and meta to create a ``DataSet`` instance.

        Parameters
        ----------
        stack : qp.Stack
            The Stack instance to convert.
        fk: string
            Filter name if the stack contains more than one filters.
        """
        meta = stack[dk].meta
        if not fk:
            data = stack[dk].data
        else:
            data = stack[dk][fk].data
        return cls.from_components(dk, data, meta)

    @classmethod
    def from_dimensions(cls, name, path="."):
        """
        Load Dimensions .ddf/.mdd files, connecting as data and meta components

        Parameters
        ----------
        name : str
            The name (without suffix) of the ddf and mdd files to load.
        path : str
            The path, where the ddf and mdd files are located.

        Example
        ```
        dataset = qp.DataSet.from_dimensions(
            "Example Data (A)",
            "../tests/data/",
            reset=True)
        ```
        """
        path_mdd = os.path.join(path, u"{}.mdd".format(name))
        path_ddf = os.path.join(path, u"{}.ddf".format(name))
        meta, data = quantipy_from_dimensions(path_mdd, path_ddf)
        dataset = cls(name, meta, data)
        dataset.path = path
        dataset._verbose_io(name, path, "dimensions")
        return dataset

    @classmethod
    def from_ascribe(cls, name, path="."):
        """
        Load Ascribe .xml/.txt files, connecting as data and meta components.

        Parameters
        ----------
        name : str
            The name (without suffix) of the xml and txt files to load.
        path : str
            The path, where the xml and txt files are located.

        Note:
            The files expect the following naming convention:
            "{name}.codebooks.xml" and "{name}_tabdelimited.txt"
        """
        path_xml = os.path.join(path, u"{}.codebooks.xml".format(name))
        path_txt = os.path.join(path, u"{}_tabdelimited.txt".format(name))
        meta, data = quantipy_from_ascribe(path_xml, path_txt)
        dataset = cls(name, meta, data)
        dataset.path = path
        dataset._verbose_io(name, path, "ascribe")
        return dataset

    @classmethod
    def from_spss(cls, name, path="."):
        """
        Load SPSS Statistics .sav files, connecting as data and meta components

        Parameters
        ----------
        name : str
            The name (without suffix) of the sav file to load.
        path : str
            The path, where the sav file is located.
        """
        path_sav = os.path.join(path, u"{}.sav".format(name))
        meta, data = parse_sav_file(path_sav)
        dataset = cls(name, meta, data)
        dataset.path = path
        dataset._verbose_io(name, path, "SPSS")
        return dataset

    def to_quantipy(self, name=None, path="."):
        """
        Write the data and meta components to .csv/.json files.

        Parameters
        ----------
        name : str
            The name (without suffix) of the csv and json files to save.
            If None is provided, the instance's name is taken.
        path : str
            The path, where the csv and json files are saved.
            If None is provided, the instance's path is taken.
        """
        path = path or self.path
        name = name or self.name
        path_json = os.path.join(path, u"{}.json".format(name))
        path_csv = os.path.join(path, u"{}.csv".format(name))
        self._meta.to_json(path_json)
        self._data.to_csv(path_csv)
        self._verbose_io(name, path, "quantipy", False)

    def to_components(self, inplace=True):
        """
        Return the ``meta`` and ``data`` components of the DataSet instance.
        """
        if inplace:
            return self._meta, self._data
        else:
            return copy.deepcopy(self._meta), self._data.copy()

    def to_dimensions(self, name=None, path=".", **kwargs):
        """
        Build Dimensions/SPSS Base Professional .ddf/.mdd data pairs.

        .. note:: SPSS Data Collection Base Professional must be installed on
            the machine. The method is creating .mrs and .dms scripts which are
            executed through the software's API.

        Parameters
        ----------
        name : str
            The name (without suffix) of the mdd and ddf files to save.
            If None is provided, the instance's name is taken.
        path : str
            The path, where the mdd and ddf files are saved.
            If None is provided, the instance's path is taken.
        kwargs : dict
            See ``qp.core.io.dimensions.writer.dimensions_from_quantipy()``
        """
        path = path or self.path
        name = name or self.name
        ds = self.clone()
        if not ds.dimensions_comp:
            ds.dimensionize()
        writer = DimensionsWriter(
            ds._data, ds._meta, kwargs.pop("text_key", None))
        writer.run(name, path, **kwargs)
        self._verbose_io(name, path, "dimensions", False)

    @params(text_key='text_key')
    def to_spss(self, name=None, path=None, **kwargs):
        """
        Convert the Quantipy DataSet into a SPSS .sav data file.

        Parameters
        ----------
        name : str
            The name (without suffix) of the sav file to save.
            If None is provided, the instance's name is taken.
        path : str
            The path, where the sav file is saved.
            If None is provided, the instance's path is taken.
        kwargs : dict
            See ``qp.core.io.spss.writer.save_sav()``
        """
        set_encoding('cp1252')
        text_key = text_key or self.text_key
        path = path or self.path
        name = name or self.name
        path_sav = os.path.join(path, u"{}.sav".format(name))
        save_sav(path_sav, self._meta, self._data, text_key=text_key,
                 mrset_tag_style=mrset_tag_style)
        set_encoding('utf-8')

    # -------------------------------------------------------------------------
    # repair
    # -------------------------------------------------------------------------
    def repair(self):
        """
        Try to fix legacy meta data inconsistencies and badly shaped array /
        datafile items ``'sets'`` meta definitions.
        """
        self._repair_date_columns()
        self._repair_default_var()
        self._repair_secure_vars()
        self._rename_blacklist_vars()
        self._repair_one_cat_sets()
        self._rename_weak_dupes()
        self.reset_index()

    def _rename_blacklist_vars(self):
        blacklist_var = []
        for var in BLACKLIST_VARIABLES:
            if var in self:
                n_var = self.valid_var_name(var)
                if not var == n_var:
                    self.rename(var, n_var)
                    blacklist_var.append([var, n_var])
        if blacklist_var:
            msg = "Renamed blacklist variables:\n{}".format(
                "\n".join(["-->".join(renamed) for renamed in blacklist_var]))
            logger.info(msg)

    def _rename_weak_dupes(self):
        dupes = self.names(as_df=False)
        renamed_wd = []
        if isinstance(dupes, dict):
            for var in list(dupes.values())[1:]:
                n_var = self.valid_var_name(var)
                self.rename(var, n_var)
                renamed_wd.append([var, n_var])
        if renamed_wd:
            msg = "Renamed weak duplicates:\n{}".format(
                "\n".join(["-->".join(renamed) for renamed in renamed_wd]))
            logger.info(msg)

    def _repair_date_columns(self):
        for col in self.dates:
            self._data[col] = pd.to_datetime(self._data[col])

    def _repair_default_var(self):
        self.add_meta("@1", "int", "")
        self._data['@1'] = np.ones(len(self._data))
        for var in INVALID_VARS:
            if var in self._data.columns:
                self._data.drop(var, axis=1, inplace=True)
            if var in self:
                self._meta.drop(var)

    def _repair_secure_vars(self):
        """
        Add variables in the CSV missing from the data-file set
        """
        ignore = ['@1']
        actual = self.unroll(self.variables_from_set())
        expected = self._data.columns.values.tolist()
        missing = [col for col in expected if col not in actual + ignore]
        if missing:
            self.extend_set(missing, "data file")

    def _repair_one_cat_sets(self):
        for ds in self.delimited_sets:
            if len(self.get_codes(ds)) == 1:
                self.convert(n, "single")
                logger.info("Auto conversion of '{}' to single.".format(ds))

    def reset_index(self):
        self._data.reset_index(drop=True, inplace=True)
        self._data = self._data[["@1"] + self.unroll(self.variables())]

    # ------------------------------------------------------------------------
    # inspect
    # ------------------------------------------------------------------------
    @params(is_var=["name"])
    def is_nan(self, name):
        """
        Detect empty entries in the ``_data`` rows.

        Parameters
        ----------
        name : str
            The column variable name keyed in ``meta['columns']``.

        Returns
        -------
        count : pandas.Series
            A series with the results as bool.
        """
        return self._data[name].isnull()

    @params(is_var=["name"], to_list=["codes"])
    def any(self, name, codes):
        """
        Return a logical has_any() slicer for the passed codes.

        .. note:: When applied to an array mask, the has_any() logic is ex-
            tended to the item sources, i.e. the it must itself be true for
            *at least one of* the items.

        Parameters
        ----------
        name : str, default None
            The column variable name keyed in ``_meta['columns']`` or
            ``_meta['masks']``.
        codes : int or list of int
            The codes to build the logical slicer from.

        Returns
        -------
        slicer : pandas.Index
            The indices fulfilling has_any([codes]).
        """
        if self.is_array(name):
            logics = []
            for s in self.get_sources(name):
                logics.append({s: has_any(codes)})
            slicer = self.take(union(logics))
        else:
            slicer = self.take({name: has_any(codes)})
        return slicer

    @params(is_var=["name"], to_list=["codes"])
    def all(self, name, codes):
        """
        Return a logical has_all() slicer for the passed codes.

        .. note:: When applied to an array mask, the has_all() logic is ex-
            tended to the item sources, i.e. the it must itself be true for
            *all* the items.

        Parameters
        ----------
        name : str, default None
            The column variable name keyed in ``_meta['columns']`` or
            ``_meta['masks']``.
        codes : int or list of int
            The codes to build the logical slicer from.

        Returns
        -------
        slicer : pandas.Index
            The indices fulfilling has_all([codes]).
        """
        if self.is_array(name):
            logics = []
            for s in self.get_sources(name):
                logics.append({s: has_all(codes)})
            slicer = self.take(intersection(logics))
        else:
            slicer = self.take({name: has_all(codes)})
        return slicer

    @params(is_var=["name"])
    def empty(self, name, condition=None):
        """
        Get all empty items of an array or check emptiness of included column.

        Parameters
        ----------
        name : str
            The variable name.
        condition : Quantipy logic expression, default None
            A logical condition expressed as Quantipy logic that determines
            which subset of the case data rows to be considered.
        """
        empty = []
        columns = self.unroll(name)
        if condition:
            df = pd.DataFrame(self[self.take(condition), columns])
        else:
            df = self._data
        for col in columns:
            if df[col].count() == 0:
                empty.append(col)
        return bool(empty) if len(columns) == 1 else empty

    @params(is_mask=["name"], repeat=["name"])
    def hide_empty_items(self, name, condition=None):
        """
        Apply ``rules`` meta to automatically hide empty array items.

        Parameters
        ----------
        name : str
            The variable name (mask)
        condition : Quantipy logic expression
            A logical condition expressed as Quantipy logic that determines
            which subset of the case data rows to be considered.
        """
        empty_items = self.empty(name, condition)
        if empty_items:
            if len(empty_items) == len(self.sources(name)):
                self.set_property("_no_valid_items", True, True)
            else:
                self.set_hiding(arr, items, axis='x', hide_values=False)

    @params(is_column=["name"], repeat=["name"], axis=["axis"],
            to_list=["axis"])
    def min_value_count(self, name, min=50, weight=None, condition=None,
                        axis='y'):
        """
        Wrapper for self.hiding(), which is hiding low value_counts.

        Parameters
        ----------
        variables: str/ list of str
            Name(s) of the variable(s) whose values are checked against the
            defined border.
        min: int
            If the amount of counts for a value is below this number, the
            value is hidden.
        weight: str, default None
            Name of the weight, which is used to calculate the weigthed counts.
        condition: complex logic
            The data, which is used to calculate the counts, can be filtered
            by the included condition.
        axis: {'y', 'x', ['x', 'y']}, default None
            The axis on which the values are hidden.
        """
        ct = self.crosstab(name, w=weight, text=False, f=condition)
        df = ct[name]['@'][name]
        hide = []
        for i, c in zip(df.index, df.values):
            if c < min:
                hide.append(i)
        if hide:
            codes = self.get_codes(v)
            hide = [h for h in hide if not h == 'All' and h in codes]
            self.set_hiding(v, hide, axis)

    @params(is_var=["name"])
    def is_like_numeric(self, name):
        """
        Test if a ``string``-typed variable can be expressed numerically.
        """
        if not self.is_string(name):
            if self.is_array(name):
                qtype = self.get_subtype(name)
            else:
                qtype = self.get_type(name)
            err = "'{}' is not of type string (but {}).".format(name, qtype)
            logger.error(err); raise ValueError(err)
        try:
            self[self.unroll(name)].astype(float)
            return True
        except ValueError:
            return False

    @params(is_column=["name"], is_cat=["name"])
    def get_codes_in_data(self, name, condition=None):
        """
        Get a list of codes that exist in data.
        """
        slicer = self.take(condition)
        s = self[slicer, name].copy()
        if self.is_delimited_set(name):
            if not s.dropna().empty:
                data_codes = s.str.get_dummies(';').columns.tolist()
                data_codes = [int(c) for c in data_codes]
            else:
                data_codes = []
        else:
            data_codes = pd.get_dummies(s).columns.tolist()
        return data_codes

    @params(is_column=["name"])
    def get_duplicates(self, name='identity'):
        """
        Returns a list with duplicated values for the provided name.
        """
        if self.is_delimited_set(name):
            err = "Can not check duplicates for delimited sets."
            logger.error(err); raise TypeError(err)
        vals = self._data[name].value_counts()
        vals = vals.copy().dropna()
        if self.is_string(name):
            vals = vals.drop('__NA__')
        vals = vals[vals >= 2].index.tolist()
        if self.is_int(name):
            vals = [int(i) for i in vals]
        elif self.is_float(name):
            vals = [float(i) for i in vals]
        return vals

    @params(is_column=["unique_id", "sort_by"])
    def drop_duplicates(self, unique_id='identity', keep='first',
                        sort_by=None):
        """
        Drop duplicated cases from self._data.

        Parameters
        ----------
        unique_id : str
            Variable name that gets scanned for duplicates.
        keep : str, {'first', 'last'}
            Keep first or last of the duplicates.
        sort_by : str
            Name of a variable to sort the data by, for example "endtime".
            It is a helper to specify `keep`.
        """
        if sort_by:
            self._data.sort(sort_by, inplace=True)
            self.reset_index()
        if self.duplicates(unique_id):
            cases_before = self._data.shape[0]
            self._data.drop_duplicates(
                subset=unique_id, keep=keep, inplace=True)
            if self._verbose_infos:
                cases_after = self._data.shape[0]
                droped_cases = cases_before - cases_after
                msg = '{} duplicated case(s) dropped, {} cases remaining'
                logger.info(msg.format(droped_cases, cases_after))

    # ------------------------------------------------------------------------
    # DP
    # ------------------------------------------------------------------------
    @params(is_var=["name"], is_cat=["name"],
            to_list=["count_only", "count_not"])
    def code_count(self, name, count_only=None, count_not=None):
        """
        Get the total number of codes/ entries found per row.

        .. note:: Will be 0/1 for type ``single`` and range between 0 and the
            number of possible values for type ``delimited set``.

        Parameters
        ----------
        name : str
            The variable name keyed in ``meta['columns']`` or ``meta['masks']``
        count_only : int or list of int, default None
            Pass a list of codes to restrict counting to.
        count_not : int or list of int, default None
            Pass a list of codes that should no be counted.

        Returns
        -------
        count : pandas.Series
            A series with the results as ints.
        """
        if count_only and count_not:
            err = "Must pass either 'count_only' or 'count_not', not both!"
            logger.error(err); raise ValueError(err)
        dummy = self.make_dummy(name, partitioned=False)
        if count_not:
            count_only = list(set(
                [c for c in dummy.columns if c not in count_not]))
        if count_only:
            dummy = dummy[count_only]
        count = dummy.sum(axis=1)
        return count

    def take(self, condition):
        """
        Create an index slicer to select rows from the DataFrame component.

        Parameters
        ----------
        condition : str or qp logic expression
            Name of a filter variable or logic condition that determines
            which subset of the case data rows to be kept.

        Returns
        -------
        slicer : pandas.Index
            The indices fulfilling the passed logical condition.
        """
        if not condition:
            return self._data.index
        full_data = self._data.copy()
        series_data = self._data['@1'].copy()
        slicer, _ = get_logic_index(series_data, condition, full_data)
        return slicer

    def crosstab(self, x, y="@", f=None, **kwargs):
        """
        Return a type-appropriate crosstab of x and y.

        Parameters
        ----------
        x : str
            The variable that should be placed into the x-position.
        y : str, default "@" (Total)
            The variable that should be placed into the y-position.
        w : str, default None
            The name of the weight variable that should be used on the data.
        f : str or qp.Logic, default None
            The name of a filter variable or a qp.Logic, to slice the data.
        pct : bool, default False
            Control the type of data that is returned (pct or counts).
        decimals : int, default 1
            Control the number of decimals in the returned dataframe.
        text : bool, default True
            Control the index and columns (texts or codes)
        rules : bool, default False
            If True then all rules that are found will be applied. If
            list-like then rules with those keys will be applied.
        xtotal : bool, default False
            If True, the first column of the returned dataframe will be the
            regular frequency of the x column.

        Note:
            See ``qp.Stack.crosstab()``
        """
        ds = self.filter(f)
        stack = Stack(
            name="ct",
            add_data={"ct": {"meta": ds._meta, "data": ds._data}})
        return stack.crosstab(x, y, f, **kwargs)

    @params(repeat=["name"])
    def _add_data_column(self, name, replace=True):
        if replace or name not in self._data.columns:
            if self.is_delimited_set(name):
                self._data[name] = np.NaN
            else:
                self._data[name] = np.NaN

    @params(to_list=['categories', 'items'], repeat=["name"])
    def add_meta(self, name, qtype, label, categories=None, items=None,
                 text_key=None, replace=True, properties={}):
        """
        Create and insert a well-formed meta object.

        Parameters
        ----------
        name : str
            The variable name.
        qtype : {'int', 'float', 'single', 'delimited set', 'date', 'string'}
            The structural type of the data the meta describes.
        label : str, default ""
            The ``text`` label information.
        categories : list of str, int, or tuples in form of (int, str)
            Example:
            ``[(1, 'Elephant'), (2, 'Mouse'), (999, 'No animal')]``
        items : list of str or tuples in form of (int, str)
            Example:
            ``[(1 'The 1st item'), (2, 'The 2nd item'), (99, 'Last item')]``
        text_key : str, default None
            Text key for text-based label information.
        replace : bool, default True
            If True, an already existing corresponding ``pd.DataFrame``
            column in the case data component will be overwritten with a
            new (empty) one.
        """
        self._meta.add_meta(name, qtype, label, categories, items, text_key,
                            properties)
        name = self.unroll(name)
        for n in name:
            self._add_data_column(name, replace)
        variables = self.unroll(self.variables()) + ["@1"]
        if any(col not in variables for col in self._data.columns):
            for col in self._data.columns.tolist():
                if col not in variables:
                    self._data.drop(col, 1, inplace=True)

    @params(repeat=["name"])
    def categorize(self, name, categorized_name=None):
        """
        Categorize an ``int``/``string``/``text`` variable to ``single``.

        The ``values`` object of the categorized variable is populated with the
        unique values found in the originating variable (ignoring np.NaN /
        empty row entries).

        Parameters
        ----------
        name : str
            The variable name
        categorized_name : str
            If provided, the categorized variable's new name will be drawn
            from here, otherwise a default name in form of ``'name#'`` will be
            used.
        """
        if not categorized_name:
            categorized_name = "{}#".format(name)
        self.copy(name, categorized_name)
        self.convert(categorized_name, 'single')

    @params(repeat=["name"], is_column=["name"], is_cat=["name"],
            text_key=["text_key"], to_list=["ignore"])
    def dichotomize(self, name, keep_variable_text=True, ignore=[],
                    replace=False, text_key=None):
        """
        Dichotomize a categorical variable.

        Parameters
        ----------
        name : str
            The variable name (column)
        keep_variable_text : bool, default True
            *  True: The new label is variable label + the value label
            *  False: The new label is the value label
        ignore : (list of) int
            Codes to ignore while conversion.
        replace : bool, default False
            Replace the initial variable in the datafile set with the new
            dichotomised variables and remove it completely from the instance.
        """
        values = self.values(name, text_key)
        new_vars = []
        for value in values:
            code, text = value
            new_name = "{}_{}".format(name, code)
            if keep_variable_text:
                label = '{}: {}'.format(self.get_text(name, text_key), text)
            else:
                label = text
            mapper = [
                (1, "Yes", {name: has_any([code])}),
                (0, "No", {name: not_any([code])})]
            self.derive(new_name, "single", label, mapper)
            new_vars.append(new_name)
        if replace:
            self.order(reposition={name: new_vars})
            self.drop(name)

    @params(repeat=["name"], is_column=["name"])
    def first_responses(self, name, n=3, others='others', reduce_values=False):
        """
        Create n-first mentions from the set of responses of a delimited set.

        Parameters
        ----------
        name : str
            The variable name (column)
        n : int, default 3
            The number of mentions that will be turned into single-type
            variables, i.e. 1st mention, 2nd mention, 3rd mention, 4th mention,
            etc.
        others : None or str, default 'others'
            If provided, all remaining values will end up in a new delimited
            set variable reduced by the responses transferred to the single
            mention variables.
        reduce_values : bool, default False
            If True, each new variable will only list the categorical value
            metadata for the codes found in the respective data vector, i.e.
            not the initial full codeframe.
        """
        if not self.is_delimited_set(name) or self.empty(name):
            return None
        created = []
        values = self.values(name)
        for _n in frange('1-{}'.format(n)):
            n_name = '{}_{}'.format(name, _n)
            n_label = '{} ({})'.format(self.get_text(name), _n)
            self.add_meta(n_name, 'single', n_label, values)
            n_vector = self[name].str.split(';', n=_n, expand=True)[_n - 1]
            self[n_name] = n_vector.replace(('', None), np.NaN).astype(float)
            created.append(n_name)
        if others:
            o_name = '{}_{}'.format(name, others)
            o_label = '{} ({})'.format(self.get_text(name), others)
            self.add_meta(o_name, 'delimited set', o_label, values)
            o_string = self[name].str.split(';', n=n, expand=True)[n]
            self[o_name] = o_string.replace(('', None), np.NaN)
            created.append(o_name)
        if reduce_values:
            for v in created:
                reduce_codes = [
                    value[0] for value in values
                    if value[0] not in self.get_codes_in_data(v)]
                self.remove_values(v, reduce_codes)

    @params(repeat=["name"], is_mask=["name"], to_list=["codes"],
            text_key=["text_key"])
    def flatten(self, name, codes, new_name=None, text_key=None):
        """
        Create a variable that groups array mask item answers to categories.

        Parameters
        ----------
        name : str
            The array variable name keyed in ``meta['masks']`` that will
            be converted.
        codes : int, list of int
            The answers codes that determine the categorical grouping.
            Item labels will become the category labels.
        new_name : str, default None
            The name of the new delimited set variable. If None, ``name`` is
            suffixed with '_rec'.
        text_key : str, default None (``== self.text_key``)
            Text key for text-based label information.
        """
        if not new_name:
            new_name = '{}_rec'.format(Meta.dims_free_array_name(name))
        label = self.get_text(name)
        cats = self.get_item_texts(name)
        self.add_meta(new_name, 'delimited set', label, cats)
        for x, source in enumerate(self.get_sources(name), 1):
            self.recode(new_name, {x: {source: codes}}, append=True)

    @params(repeat='name')
    def drop(self, name, ignore_items=False):
        """
        Drops variables from meta and data components of the ``DataSet``.

        Parameters
        ----------
        name : str or list of str
            The column variable name keyed in ``_meta['columns']`` or
            ``_meta['masks']``.
        ignore_items: bool
            If False source variables for arrays in ``_meta['columns']``
            are dropped, otherwise kept.
        Returns
        -------
        None
            DataSet is modified inplace.
        """
        data_drop = self.unroll(name)
        is_array = self.is_array(name)
        self._meta.drop(name, ignore_items)
        if not is_array or not ignore_items:
            self._data.drop(data_drop, 1, inplace=True)

    @params(to_list=['copy_only', 'copy_not'])
    def copy(self, name, new_name=None, copy_data=True, slicer=None,
             copy_only=None, copy_not=None):
        """
        Copy meta and case data of the variable defintion given per ``name``.

        Parameters
        ----------
        name : str
            The variable name (mask or column)
        new_name : str, default None
            The new variable name, is None is provided, the initial name is
            extended by the suffix ``_rec``.
        copy_data : bool, default True
            The new variable assumes the ``data`` of the original variable.
        slicer : str or dict
            If the data is copied it is possible to filter the data with a
            complex logic or a filter variable.
            Example: ``slicer = {'q1': not_any([99])}``
        copy_only: int or list of int, default None
            If provided, the copied version of the variable will only contain
            (data and) meta for the specified codes.
        copy_not: int or list of int, default None
            If provided, the copied version of the variable will contain
            (data and) meta for the all codes, except of the indicated.
        """
        if not new_name:
            new_name = "{}_rec".format(name)
        self._meta.copy(name, new_name, copy_only, copy_not)
        old = self.unroll(name)
        new = self.unroll(new_name)
        self._add_data_column(new)
        if copy_data:
            for o, n in zip(old, new):
                self[self.take(slicer), n] = self[o].copy()
                if self.is_categorical(n):
                    remove = [
                        code for code in self.get_codes_in_data(n)
                        if code not in self.get_codes(n)]
                    if remove:
                        self[n] = self[n].apply(
                            lambda x: remove_codes(x, remove))

    def copy_array_data(self, source, target, source_items=None,
                        target_items=None, slicer=None):
        """
        Take over data of an array to another array
        """
        if not set(self.get_codes(source)) == set(self.get_codes(target)):
            err = "Expect equal codes for source and target."
            logger.error(err); raise ValueError(err)
        sources = [
            s for s in self.get_sources(source)
            if not source_items or self.get_item_no(s) not in source_items]
        targets = [
            t for t in self.get_sources(target)
            if not target_items or self.get_item_no(t) not in target_items]
        for s, t in zip(sources, targets):
            self[self.take(slicer), t] = self[s]

    @params(is_mask=["name"], to_list=["ignore_items", "ignore_values"],
            text_key=["text_key"])
    def transpose(self, name, new_name, ignore_items=[], ignore_values=[],
                  text_key=None):
        """
        Create a new array mask with transposed items / values structure.

        Parameters
        ----------
        name : str
            The variable name (mask).
        new_name : str
            The name of the new mask
        ignore_items : (list of) int
            If provided, the items listed by their order number in the
            ``self['masks'][name]['items']`` object will not be part of the
            transposed array. This means they will be ignored while creating
            the new value codes meta.
        ignore_codes : (list of) int
            If provided, the listed code values will not be part of the
            transposed array. This means they will not be part of the new
            item meta.
        text_key : str, default None == self.text_key
            Text key for text-based label information.
        """
        if not self._meta._verify_new_name(new_name):
            err = "Cannot create '{}'. Weak duplicates exist: {}"
            err = err.format(new_name, self.get_weak_dupes(new_name))
            logger.error(err); raise ValueError(err)

        label = self.get_text(name, text_key=text_key)
        items = [
            (self.get_item_no(i[0]), i[1]) for i in self.get_items(name)
            if self.get_item_no(i[0]) not in ignore_items]
        values = [v for v in self.get_values(name) if v[0] not in ignore_codes]
        self.add_meta(new_name, "delimited set", label, items=values,
                      categories=items, text_key=text_key)
        sources = self.get_sources(new_name)
        for code, source in zip(values, sources):
            mapper = {no: {item: code[0]} for no, item in items}
            self.recode(source, mapper, append=True)

    @params(is_column=["target"], repeat=["target"])
    def recode(self, target, mapper, intersect=None, initialize=None,
               fillna=None):
        """
        Add codes to a data series by using a recode mapper.

        Parameters
        ----------
        target : str
            The variable name (column).
        mapper : dict
            A mapper of structure: ``{code: qp-logic}``
            Code is added to the series if logic is satisfied.
        intersect : str or qp-logic, default None
            A filter variable name or quantipy logic which is added as
            intersection to all given conditions in mapper.
        initialize : str or np.NaN, default None
            If not None, a copy of the data named column will be used to
            populate the target column before the recode is performed.
            Alternatively, initialize can be used to populate the target
            column with np.NaNs (overwriting whatever may be there) prior
            to the recode.
        fillna : int, default=None
            If provided, empty cases will be filled with this value after the
            recode is performed.
        """
        if fillna and self.is_categorical(target):
            if fillna not in self.get_codes(target):
                err = "'{}' is not a valid code for {}".format(fillna, target)
                logger.error(err); raise ValueError(err)
        if initialize is not None:
            if isinstance(initialize, str):
                if not self.get_type(initialize) == self.get_type(target):
                    err = (
                        "Cannot initialize '{}' with '{}', types are "
                        "different")
                    err = err.format(target, initialize)
                    logger.error(err); raise ValueError(err)
                self._data[target] = self._data[initialize].copy()
            elif np.isnan(initialize):
                self._add_data_column(target)
        mapper = self._index_mapper(mapper, intersect)
        self._recode_from_index_mapper(target, mapper)
        if fillna:
            self[self.take({target: has_count(0)}), target] = fillna

    @params(is_column=["target"], repeat=["target"])
    def uncode(self, target, mapper, intersect=None):
        """
        Remove codes from a data series by using an uncode mapper.

        Parameters
        ----------
        target : str
            The variable name (column).
        mapper : dict
            A mapper of structure: ``{code: qp-logic}``
            Code is removed from the series if logic is satisfied.
        intersect : str or qp-logic, default None
            A filter variable name or quantipy logic which is added as
            intersection to all given conditions in mapper.
        """
        index_mapper = self._index_mapper(mapper, intersect)
        for code, index in index_mapper.items():
            col = self[target][index].copy()
            col = col.apply(lambda x: remove_codes(x, [code]))
            self._data.loc[index, target] = col

    def interlock(self, name, label, variables, val_text_sep='/'):
        """
        Build a new category-intersected variable from >=2 incoming variables.

        Parameters
        ----------
        name : str
            The new variable name.
        label : str
            The new text label for the created variable.
        variables : list of >= 2 str or dict (mapper)
            The column names of the variables that are feeding into the
            intersecting recode operation. Or dicts/mapper to create temporary
            variables for interlock. Can also be a mix of str and dict.
            >>> ['gender',
            ...  {'agegrp': [(1, '18-34', {'age': frange('18-34')}),
            ...              (2, '35-54', {'age': frange('35-54')}),
            ...              (3, '55+', {'age': is_ge(55)})]},
            ...  'region']
        val_text_sep : str, default '/'
            The passed character (or any other str value) wil be used to
            separate the incoming individual value texts to make up the inter-
            sected category value texts, e.g.: 'Female/18-30/London'.
        """
        if not isinstance(variables, list) or len(variables) < 2:
            err = "'variables' must be a list of at least two items!"
            logger.error(err); raise ValueError(err)

        i_variables = []
        new_variables = []
        for var in variables:
            if isinstance(var, dict):
                v = var.keys()[0]
                mapper = var.values()[0]
                if self._is_delimited_set_mapper(mapper):
                    qtype = 'delimited set'
                else:
                    qtype = 'single'
                self.derive('{}_temp'.format(v), qtype, v, mapper)
                i_variables.append('{}_temp'.format(v))
                new_variables.append('{}_temp'.format(v))
            else:
                i_variables.append(var)

        if any(self.is_array(v) for v in i_variables):
            raise TypeError('Cannot interlock within array-typed variables!')
        if any(self.is_delimited_set(v) for v in i_variables):
            qtype = 'delimited set'
        else:
            qtype = 'single'

        codes = [self.get_codes(iv) for iv in i_variables]
        texts = [self.get_value_texts(iv) for iv in i_variables]
        zipped = zip(list(product(*codes)), list(product(*texts)))
        categories = []
        cat_id = 0
        for codes, texts in zipped:
            cat_id += 1
            cat_label = val_text_sep.join(texts)
            rec = [{iv: [c]} for iv, c in zip(i_variables, codes)]
            rec = intersection(rec)
            categories.append((cat_id, cat_label, rec))
        self.derive(name, qtype, label, categories)
        for var in new_variables:
            self.drop(var)

    @params(is_mask=["name"])
    def _level(self, name):
        lvlname = '{}_level'.format(name)
        self.copy(name, lvlname)
        items = self.get_items(name)
        sources = enumerate(self.get_sources(lvlname), 1)
        codes = self.get_codes(lvlname)
        max_code = len(codes)
        replace_codes = {}
        mapped_codes = {c: [] for c in self.get_codes(name)}

        for no, source in sources:
            offset = (no - 1) * max_code
            new_codes = frange('{}-{}'.format(offset + 1, offset + max_code))
            replace_codes[source] = dict(zip(codes, new_codes))

        for source, codes in replace_codes.items():
            self[source].replace(codes, inplace=True)
            self[source].replace(np.NaN, '', inplace=True)
            for org, new in codes.items():
                mapped_codes[org].append(new)

        code_range = frange('1-{}'.format(max_code * len(items)))
        labels = []
        for item in items:
            for value in self.get_value_texts(name):
                labels.append("{}_{}".format(value, item[0]))
        cats = list(zip(code_range, labels))
        new_sources = self.get_sources(lvlname)
        self.unbind(lvlname)
        self.add_meta(lvlname, 'delimited set', self.get_text(name), cats)
        self[lvlname] = self[new_sources].astype('str').apply(
            lambda x: ';'.join(x).replace('.0', ''), axis=1)
        self.drop(new_sources)
        self.set_property(lvlname, "level", {
            'source': name,
            'level_codes': mapped_codes,
            'item_look': self.get_sources(name)[0]})

    @params(text_key=["text_key"])
    def derive(self, name, qtype, label, cond_map, text_key=None):
        """
        Create meta and recode case data by specifying derived category logics.

        Parameters
        ----------
        name : str
            The new variable name.
        qtype : [``int``, ``float``, ``single``, ``delimited set``]
            The structural type of the data the meta describes.
        label : str
            The ``text`` label information.
        cond_map : list of tuples
            Tuples of either two or three elements of following structures:

            2 elements, no labels provided:
            (code, <qp logic expression here>),

            2 elements, no codes provided:
            ('text label', <qp logic expression here>),

            3 elements, with codes + labels:
            (code, 'Label goes here', <qp logic expression here>),
        """
        err_msg = (
            "'cond_map' structure not understood. Must pass a list "
            "of 2 (code, logic) / (text, logic) or 3 (code, text label, "
            "logic) element tuples!")
        if all(len(cond) == 3 for cond in cond_map):
            categories = [(cond[0], cond[1]) for cond in cond_map]
            idx_mapper = {cond[0]: cond[-1] for cond in cond_map}
        elif all(len(cond) == 2 for cond in cond_map):
            all_int = all(isinstance(cond[0], int) for cond in cond_map)
            all_str = all(isinstance(cond[0], str) for cond in cond_map)
            if not (all_str or all_int):
                raise TypeError(err_msg)
            categories = [cond[0] for cond in cond_map]
            if all_int:
                idx_mapper = {cond[0]: cond[-1] for cond in cond_map}
            if all_str:
                idx_mapper = {
                    c: cond[-1] for c, cond in enumerate(cond_map, start=1)}
        else:
            raise TypeError(err_msg)
        self.add_meta(name, qtype, label, categories, text_key=text_key)
        self.recode(name, idx_mapper)

    @params(is_column=["name"], text_key=["text_key"])
    def band(self, name, bands, new_name=None, label=None, text_key=None):
        """
        Group numeric data with band definitions treated as group text labels.

        Wrapper around ``derive()`` for quick banding of numeric
        data.

        Parameters
        ----------
        name : str
            The variable name (column).
        bands : list of int/tuple *or* dict mapping the former to value texts
            The categorical bands to be used. Bands can be single numeric
            values or ranges, e.g.: [0, (1, 10), 11, 12, (13, 20)].
            Be default, each band will also make up the value text of the
            category created in the ``_meta`` component. To specify custom
            texts, map each band to a category name e.g.:
            [{'A': 0},
            {'B': (1, 10)},
            {'C': 11},
            {'D': 12},
            {'E': (13, 20)}]
        new_name : str, default None
            The created variable will be named ``'<name>_banded'``, unless a
            desired name is provided explicitly here.
        label : str, default None
            The created variable's text label will be identical to the origi-
            nating one's passed in ``name``, unless a desired label is provided
            explicitly here.
        """
        if not self.is_numeric(name):
            err = "Can only band numeric typed data! {} is {}."
            err = err.format(name, self.get_type(name))
            logger.error(err); raise ValueError(msg)
        if not new_name:
            new_name = '{}_banded'.format(name)
        if not label:
            label = self.get_text(name, False, text_key)
        franges = []
        for idx, band in enumerate(bands, start=1):
            lab = None
            if isinstance(band, dict):
                lab = band.keys()[0]
                band = band.values()[0]
            if isinstance(band, tuple):
                if band[0] < 0:
                    raise ValueError('Cannot band with lower bound < 0.')
                elif band[1] < 0:
                    raise ValueError('Cannot band with upper bound < 0.')
                r = '{}-{}'.format(band[0], band[1])
                franges.append([idx, lab or r, {name: frange(r)}])
            else:
                r = str(band)
                franges.append([idx, lab or r, {name: [band]}])
        self.derive(new_name, 'single', label, franges, text_key=text_key)

    @params(to_list=["variables"])
    def to_delimited_set(self, name, label, variables, from_dichotomous=True,
                         codes_from_name=True):
        """
        Combines multiple single variables to new delimited set variable.

        Parameters
        ----------
        name: str
            Name of new delimited set
        label: str
            Label text for the new delimited set.
        variables: (list of) str/ tuples
            variables that get combined into the new delimited set. If they are
            dichotomous (from_dichotomous=True), the labels of the variables
            are used as category texts or if tuples are included, the second
            items will be used for the category texts. If the variables are
            categorical (from_dichotomous=False) the values of the variables
            need to be eqaul and are taken for the delimited set.
        from_dichotomous: bool, default True
            Define if the input variables are dichotomous or categorical.
        codes_from_name: bool, default True
            If from_dichotomous=True, the codes can be taken from the Variable
            names, if they are in form of 'q01_1', 'q01_3', ...
            In this case the codes will be 1, 3, ....
        """
        columns = [v[0] if isinstance(v, tuple) else v for v in variables]
        if not all(self.is_categorical(col) for col in columns):
            err = "'variables' must be categorical."
            logger.error(err); raise ValueError(err)
        if from_dichotomous:
            if not all(set(self.get_codes(c)) == set([0, 1]) for c in columns):
                err = "'variables' must be dichotomous."
                logger.error(err); raise ValueError(err)
            mapper = []
            for x, col in enumerate(variables, 1):
                if codes_from_name:
                    x = int(col.split('_')[-1])
                if isinstance(col, tuple):
                    text = col[1]
                else:
                    text = self.get_text(col)
                mapper.append((x, text, {col: [1]}))
        else:
            values = self.values(columns[0])
            if not all(self.values(c) == values for c in columns):
                err = "'variables' must have same values."
            mapper = []
            for v in values:
                mapper.append((v[0], v[1], union([{c: v[0]} for c in cols])))
        self.derive(name, 'delimited set', label, mapper)

    def to_array(self, name, variables, label):
        """
        Combines column variables with same ``values`` meta into an array.

        Parameters
        ----------
        name : str
            Name of the new variable (mask).
        variables : list of str
            Variable names that become items of the array.
        label : str
            Text label for the mask itself.
        """
        old = uniquify_list(variables)
        self._meta.to_array(name, variables, label)
        if self.dimensions_comp:
            new = self.get_sources(name)
            mapper = dict(zip(old, new))
            self._data.rename(columns=mapper, inplace=True)

    @params(to_list=["levels", "other"])
    def derotate(self, level=None, levels=[], mapper={}, other=None,
                 unique_key='identity', dropna=True):
        """
        Derotate data and meta using the given mapper, and appending others.

        This function derotates data using the specification defined in
        mapper, which is a list of dicts of lists, describing how
        columns from data can be read as a heirarchical structure.

        Returns derotated DataSet instance and saves data and meta as json
        and csv.

        Parameters
        ----------
        level : str, default None
            Name of new column variable to identify cases.
        levels : list
            Column variables that are used to create the new level variable.
        mapper : list of dicts of lists
            A mapping of new column names to source columns:
            ```
            mapper = [
                {'q14_1': ['q14_1_1', 'q14_1_2', 'q14_1_3']},
                {'q14_2': ['q14_2_1', 'q14_2_2', 'q14_2_3']},
                {'q14_3': ['q14_3_1', 'q14_3_2', 'q14_3_3']}]
            ```
        unique_key: str
            Name of column variable that will be copied to new dataset.
        other: list (optional; default=None)
            A list of additional columns from the source data to be appended
            to the end of the resulting stacked dataframe.

        Returns
        -------
        new ``qp.DataSet`` instance
        """
        for mapping in mapper:
            col, sources = list(mapping.items())[0]
            sources = ensure_list(sources)
            if not len(sources) == len(levels):
                err = "Mismatch of length of sources and levels!"
                logger.error(err); raise ValueError(err)
            mapping[col] = sources

        variables = [unique_key] + other + levels
        columns = self.unroll(variables)
        # derotated data
        dfs = []
        for mapping in mapper:
            col, sources = list(mapping.items())[0]
            df = self._data[sources].copy()
            df = df.stack().reset_index([1])
            df.columns = [level, col]
            df[level] = df[level].map(
                {el: ind for ind, el in enumerate(sources, 1)})
            df.set_index([level], append=True, drop=True, inplace=True)
            dfs.append(df)
        new_df = pd.concat(dfs, axis=1)
        new_df = new_df.reset_index(1)
        new_df = new_df.join(self._data[columns])
        new_df.reset_index(drop=True, inplace=True)

        # new meta
        variables = self.roll_up(variables)
        new_meta = self.subset(variables)
        for mapping in mapper:
            col, sources = list(mapping.items())[0]
            if new_meta.is_array_item(sources[0]):
                new_meta.unbind(new_meta.get_parent(sources[0]))
            new_meta.drop(sources[1:])
            new_meta.rename(sources[0], col)

        # new ds instance
        name = '{}_derotated'.format(self.name)
        ds = self.from_components(name, new_df, new_meta, self.text_key)
        ds.path = self.path

        # some recodes/ edits
        ds._meta.add_meta(level, 'single', level, levels)

        levelled = '{}_levelled'.format(level)
        ds.add_meta(levelled, 'single', level, self.get_values(levels[0]))

        for x, lev in enumerate(levels, 1):
            rec = {y: {lev: y} for y in ds.get_codes(levelled)}
            ds.recode(levelled, rec, intersect={level: x})

        cols = ['@1', unique_key, level, levelled] + levels
        cols += [new_var.keys()[0] for new_var in mapper] + self.unroll(other)
        ds._data = ds._data[cols]

        # save ``DataSet`` instance as json and csv
        ds.to_quantipy()
        return ds

    @params(to_list="name")
    def _recode_from_net_def(self, name, net_map, expand=None, recode="auto",
                             prefix="Net: ", mis_in_rec=False):
        """
        Create variables from net definitions.

        Parameters
        ----------
        name : str
            Variable name to use as recode source.
        net_map : list of dicts
            Net definitions in form of
            ```
            net_def = [{
                "net_1": net_logic,  # e.g. list of codes
                "text": {tk: net_label}
            }, ...]
            ```
        expand : {'before', 'after'}, default None
            If provided, the net codes will list the net-defining codes after
            or before the computed net groups (i.e. "overcode" nets).
        recode: {'extend_codes', 'drop_codes', 'collect_codes',
                 'collect_codes@cat_name'}, default 'auto'
            Adds variable with nets as codes to DataSet.
            *  'extend_codes': codes are extended with nets.
            *  'drop_codes': new variable only contains nets as codes.
            *  'collect_codes' or 'collect_codes@cat_name': the variable
                    contains nets and another category that summarises all
                    codes which are not included in any net.
                    If no cat_name is provided, 'Other' is taken as default
        mis_in_rec: bool, default False
            Skip or include codes that are defined as missing when recoding
            from net definition.
        """
        name = self.unroll(name)
        forced_recode = False
        valid = ['extend_codes', 'drop_codes', 'collect_codes']
        if recode == 'auto':
            recode = 'collect_codes'
            forced_recode = True
        if not any(rec in recode for rec in valid):
            raise ValueError("'recode' must be one of {}".format(valid))

        masks = defaultdict(defaultdict)
        for col in name:
            new = self.valid_var_name("{}_rc".format(col))

            # collect array items
            if self.is_array_item(col):
                parent = self.get_parent(col)
                no = self.get_item_no(col)
                masks[parent][no] = new

            # create mapper to derive new variable
            mapper, q_type, labels, simple_nets = self._get_net_mapper(
                col, net_map, recode, prefix)
            self.derive(new, q_type, self.text(var), mapper)

            # meta edits for new variable
            for tk, labs in labels.items():
                self.set_value_texts(new, labs, tk)
                text = self.get_text(var, tk) or self.get_text(var, None)
                self.set_text(new, text, tk)

            # properties
            self.set_property(new, "recoded_net", var)
            if simple_nets:
                self.set_property(new, "simple_org_expr", simple_nets)
                if self.is_array_item(col):
                    masks[parent]["simple_org_expr"] = simple_nets

            props = self._meta['columns'][var].get("properties", {})
            for pname, prop in props.items():
                if not pname == 'survey':
                    self(new, pname, prop)
            logger.info('Created: {}'. format(new))
            if forced_recode:
                logger.warning("'{}' was a forced recode.".format(new))

            # order, remove codes
            if 'collect_codes' in recode:
                others = [{var: not_count(0)}, {new: has_count(0)}]
                missings = self.get_missings(var)
                if not mis_in_rec and missings:
                    others.append({var: not_any(missings["exclude"])})

                has_other_logic = self.take(intersection(others)).tolist()
                if self.is_array_item(var) or has_other_logic:
                    if '@' in recode:
                        cat_name = recode.split('@')[-1]
                    else:
                        cat_name = 'Other'
                    code = len(mapper) + 1
                    self.extend_values(new, [(code, str(code))])
                    for tk in labels.keys():
                        self.set_value_texts(new, {code: cat_name}, tk)
                    self.recode(new, {code: intersection(others)})
            if recode == 'extend_codes' and expand:
                codes = self.get_codes(var)
                new = [c for c in self.get_codes(new) if c not in codes]
                order = []
                remove = []
                for x, y, z in mapper[:]:
                    if x not in new:
                        order.append(x)
                    else:
                        vals = z.values()[0]
                        if not isinstance(vals, list):
                            remove.append(vals)
                            vals = [vals]
                        if expand == 'after':
                            idx = order.index(
                                codes[min([codes.index(v) for v in vals])])
                        elif expand == 'before':
                            idx = order.index(
                                codes[max([codes.index(v) for v in vals])]) + 1
                        order.insert(idx, x)
                self.reorder_values(rew, order)
                self.remove_values(rew, remove)

        # summarize masks
        for mask, items in masks.items():
            simple_net = items.pop("simple_org_expr", None)
            newm = self.valid_var_name("{}_rc".format(mask))
            self.to_array(newm, list(sorted(list(items.items()))), '')
            self.set_property(newm, "recoded_net", mask, True)
            if simple_net:
                self.set_property(newm, 'simple_org_expr', simple_net)
            props = self._meta['masks'][mask].get("properties", {})
            for pname, prop in props.items():
                if not pname == 'survey':
                    self.set_property(newm, pname, prop, True)
            msg = "Mask {} built from recoded view variables!"
            logger.info(msg.format(dims_name))

    def _get_net_mapper(self, name, net_map, recode, prefix):
        mapper = []
        if recode == 'extend_codes':
            mapper += [(x, y, {var: x}) for (x, y) in ds.values(var)]
            max_code = max(ds.codes(var))
        elif recode == 'drop_codes' or 'collect_codes' in recode:
            max_code = 0

        appends = []
        labels = {}
        simple_nets = []
        for x, net in enumerate(net_map[:], 1):
            labs = net.pop('text')
            code = max_code + x
            for tk, lab in labs.items():
                if tk not in labels:
                    labels[tk] = {}
                labels[tk].update({code: '{} {}'.format(text_prefix, lab)})
            appends.append((code, str(code), {var: net.values()[0]}))
            if isinstance(n.values()[0], list):
                simple_nets.append((
                    '{} {}'.format(text_prefix, labs[self.text_key]),
                    net.values()[0]))
        if not len(appends) == len(simple_nets):
            simple_nets = []
        mapper.extend(appends)
        if self._is_delimited_set_mapper(mapper):
            q_type = 'delimited set'
        else:
            q_type = 'single'
        return mapper, q_type, labels, simple_nets

    # ------------------------------------------------------------------------
    # Converting
    # ------------------------------------------------------------------------
    @params(is_var=["name"])
    def convert(self, name, to):
        """
        Convert meta and case data between compatible variable types.

        Parameters
        ----------
        name : str
            The column variable name keyed in ``meta['columns']`` that will
            be converted.
        to : str {"single", "delimited set", "int", "float", "string", "date"}
            The variable type to convert to, only valid depending on the
            initial variable type.
        """
        if self.is_array(name):
            qtype = self.get_subtype(name)
        else:
            qtype = self.get_type(name)
        if qtype == to:
            logger.info("'{}' is already if type '{}'".format(name, qtype))
            return None
        if to not in COMPATIBLE_TYPES.get(qtype, []):
            err = "Cannot convert '{}' into '{}'.".format(qtype, to)
            logger.error(err); raise ValueError(err)
        elif self.is_array_item(name):
            err = "Cannot convert a single array item."
        funcs = {
            "int": self._as_int,
            "float": self._as_float,
            "single": self._as_single,
            "delimited set": self._as_delimited_set,
            "string": self._as_string
        }
        funcs[to](name)

    def _as_float(self, name):
        """
        Change type to ``float``.
        """
        if self.is_array(name):
            for source in self.get_sources(name):
                self._as_float(source)
            self._meta._set_type(name, "float")
        else:
            if self.is_string(name) and not self.is_like_numeric(name):
                self._as_single(name)
            self[name] = self[name].apply(
                lambda x: float(x) if not np.isnan(x) else np.NaN)
            if not self.is_array_item(name):
                self._meta._set_type(name, "float")

    def _as_int(self, name):
        """
        Change type to ``int``.
        """
        is_categorical = self.is_categorical(name)
        if self.is_array(name):
            for source in self.get_sources(name):
                self._as_int(source)
            self._meta._set_type(name, "int")
            if is_categorical:
                self._meta._del_values(name)
        else:
            if self.is_string(name) and not self.is_like_numeric(name):
                self._as_single(name)
            self[name] = self[name].apply(
                lambda x: int(x) if not np.isnan(x) else np.NaN)
            if not self.is_array_item(name):
                self._meta._set_type(name, "int")
                if is_categorical:
                    self._meta._del_values(name)

    def _as_delimited_set(self, name):
        """
        Change type to ``delimited set``.
        """
        if self.is_array(name):
            for source in self.get_sources(name):
                self._as_delimited_set(source)
            self._meta._set_type(name, "delimited set")
        else:
            self[name] = self[name].apply(
                lambda x: str(int(x)) + ';' if not np.isnan(x) else np.NaN)
            if not self.is_array_item(name):
                self._meta._set_type(name, "delimited set")

    def _as_single(self, name):
        """
        Change type to ``single``.
        """
        if self.is_delimited_set(name) and len(self.get_codes(name)) > 1:
            err = "Cannot convert delimited set into single."
            logger.error(err); raise ValueError(err)
        if self.is_array(name):
            for source in self.get_sources(name):
                values = self._as_single(source)
            self._meta._set_type(name, "single")
            self._meta._set_values(name, values)
        else:
            series = self[name]
            if self.is_int(name):
                num_vals = sorted(series.dropna().astype(int).unique())
                values = self._meta.start_values(num_vals)
            elif self.is_date(name):
                str_vals = series.order().astype(str).unique()
                values = self._meta.start_values(str_vals)
                replace_map = {v: i for i, v in enumerate(str_vals, 1)}
                series.replace(replace_map, inplace=True)
            elif self.is_string(name):
                series.replace({"__NA__": np.NaN}, inplace=True)
                str_vald = series.dropna().unique()
                values = self._meta.start_values(str_vald)
                replace_map = {v: i for i, v in enumerate(str_vals, 1)}
                series.replace(replace_map, inplace=True)
            elif self.is_delimited_set(name):
                self[name] = series.apply(
                    lambda x:
                        np.NaN if np.isnan(x) else int(x.replace(';', '')))
            if not self.is_array_item(name):
                self._meta._set_type(name, "single")
                self._meta._set_values(name, values)
            else:
                return values

    def _as_string(self, name):
        """"
        Change type to ``string``.
        """
        if self.is_array(name):
            for source in self.get_sources(name):
                self._as_delimited_set(source)
            self._meta._set_type(name, "string")
            if is_categorical:
                self._meta._del_values(name)
        else:
            self[name] = self[name].astype(str)
            if not self.is_array_item(name):
                self._meta._set_type(name, "string")
                if is_categorical:
                    self._meta._del_values(name)

    # ------------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------------
    def _index_mapper(self, mapper, intersect=None):
        """
        Convert a {value: logic} map to a {value: index} map.
        """
        series = pd.Series(1, index=self._data.index)
        index_mapper = {}
        for k, v in mapper.items():
            if isinstance(v, str):
                if not self.is_filter(v):
                    err = "'{}' is not a valid filter variable.".format(v)
                    logger.error(err); raise ValueError(err)
                else:
                    v = {v: 0}
            elif not isinstance(v, (dict, tuple)):
                err = "'{}' is not a valid qp logic.".format(v)
                logger.error(err); raise TypeError(err)
            index_mapper[k] = get_logic_index(series, v, self._data)[0]
        return index_mapper

    def _recode_from_index_mapper(self, target, mapper):
        if self.is_delimited_set(target):
            cols = [str(key) for key in sorted(mapper.keys())]
            df = pd.DataFrame(0, index=self._data.index, columns=cols)
            for k, v in mapper.items():
                df[str(k)].loc[v] = 1
            s = condense_dichotomous_set(df)
            self[target] = self[target].astype("str")
            self[target] = self[target].combine(
                s, lambda x, y: merge_delimited_sets(x, y))
        else:
            for k, v in mapper.items():
                self._data[target].loc[v] = k

    def _is_delimited_set_mapper(self, mapper):
        if isinstance(mapper, list):
            logics = [val[-1] for val in mapper]
        elif isinstance(mapper, dict):
            logics = mapper.values()
        else:
            msg = ("mapper must have the form: {1: logic, 2: logic,...} or ",
                   "[(1, label, logic), (2, label, logic),...]")
            raise ValueError(msg)
        logic_series = []
        for log in logics:
            index = self.take(log)
            s = pd.Series(index=index, data=True)
            logic_series.append(s)
        df = pd.concat(logic_series, axis=1)
        df = df.sum(1)
        if len(df.value_counts()) > 1:
            return True
        else:
            return False

    def enumerator(self, name):
        x = 1
        n = name
        while n in self:
            x += 1
            n = '{}_{}'.format(name, x)
        return n

    def _logic_as_pd_expr(self, logic, prefix='default'):
        """
        """
        varname = '{}__logic_dummy__'.format(prefix).replace(' ', '_')
        category = [(1, 'select', logic)]
        meta = (varname, 'single', '', category)
        self.derive(*meta)
        return '{}==1'.format(varname)

    @params(is_var=["name"])
    def make_dummy(self, name, partitioned=False):
        if self.is_array(name):
            sources = self.get_sources(name)
            if self.is_categorical(name):
                codes = self.get_codes(name)
            else:
                codes = sorted(uniquify_list(flatten_list(
                    [self.get_codes_in_data(source) for source in sources])))
            dummy_data = []
            if any(self[source].dtype == 'object' for source in sources):
                for source in sources:
                    try:
                        i_dummy = self[sourcei].str.get_dummies(';')
                        i_dummy.columns = [int(col) for col in i_dummy.columns]
                    except:  # noqa
                        i_dummy = self._data[[source]]
                        i_dummy.columns = [0]
                    dummy_data.append(i_dummy.reindex(columns=codes))
            else:
                for source in sources:
                    if codes:
                        dummy_data.append(
                            pd.get_dummies(self[source]).reindex(columns=codes)
                        )
                    else:
                        dummy_data.append(pd.get_dummies(self[source]))
            dummy_data = pd.concat(dummy_data, axis=1)
            if not partitioned:
                return dummy_data
            else:
                return dummy_data.values, codes, sources
        else:
            if self.is_delimited_set(name):
                try:
                    dummy_data = self[name].str.get_dummies(';')
                except:  # noqa
                    dummy_data = self._data[[name]]
                    dummy_data.columns = [0]
                codes = self.get_codes(name)
                dummy_data.columns = [int(col) for col in dummy_data.columns]
                dummy_data = dummy_data.reindex(columns=codes)
                dummy_data.replace(np.NaN, 0, inplace=True)
            else:  # single, int, float data
                dummy_data = pd.get_dummies(self[name])
                if self.is_single(name):
                    codes = self.get_codes(name)
                    dummy_data = dummy_data.reindex(columns=codes)
                    dummy_data.replace(np.NaN, 0, inplace=True)
                dummy_data.rename(
                    columns={
                        col: int(col) if float(col).is_integer() else col
                        for col in dummy_data.columns
                    },
                    inplace=True)
            if not partitioned:
                return dummy_data
            else:
                return dummy_data.values, dummy_data.columns.tolist()

    # ------------------------------------------------------------------------
    # Weight
    # ------------------------------------------------------------------------
    def weight(self, weight_scheme, weight_name='weight',
               unique_key='identity', subset=None, report=True,
               path_report=None):
        """
        Weight the ``DataSet`` according to a well-defined weight scheme.

        Parameters
        ----------
        weight_scheme : quantipy.Rim instance
            A rim weights setup with defined targets. Can include multiple
            weight groups and/or filters.
        weight_name : str, default 'weight'
            A name for the float variable that is added to pick up the weight
            factors.
        unique_key : str, default 'identity'.
            A variable inside the ``DataSet`` instance that will be used to
            the map individual case weights to their matching rows.
        subset : Quantipy complex logic expression
            A logic to filter the DataSet, weighting only the remaining subset.
        report : bool, default True
            If True, will report a summary of the weight algorithm run
            and factor outcomes.
        path_report : str, default None
            A file path to save an .xlsx version of the weight report to.
        """
        if subset:
            ds = self.filter('subset', subset, False)
            meta, data = ds.split()
        else:
            meta, data = self.split()
        engine = qp.WeightEngine(data, meta=meta)
        engine.add_scheme(weight_scheme, key=unique_key, verbose=verbose)
        engine.run()

        org_wname = weight_name
        if report:
            print(engine.get_report())
        if path_report:
            df = engine.get_report()
            full_file_path = '{} ({}).xlsx'.format(path_report, weight_name)
            df.to_excel(full_file_path)
            print('Weight report saved to:\n{}'.format(full_file_path))
        s_name = weight_scheme.name
        s_w_name = 'weights_{}'.format(s_name)
        if inplace:
            weight_description = '{} weights'.format(s_name)
            data_wgt = engine.dataframe(s_name)[[unique_key, s_w_name]]
            data_wgt.rename(columns={s_w_name: org_wname}, inplace=True)
            if org_wname not in self._meta['columns']:
                self.add_meta(org_wname, 'float', weight_description)
            self.update(data_wgt, on=unique_key)
        else:
            wdf = engine.dataframe(weight_scheme.name)
            return wdf.rename(columns={s_w_name: org_wname})

    # ------------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------------
    @params(to_list=['logic'])
    def add_filter_var(self, name, logic, overwrite=False):
        """
        Create filter-var, that allows index slicing using ``manifest_filter``

        Parameters
        ----------
        name: str
            Name and label of the new filter-variable, which gets also listed
            in DataSet.filters
        logic: complex logic/ str, list of complex logic/ str
            Logic to keep cases.
            Complex logic should be provided in form of:
            ```
            {
            'label': 'any text',
            'logic': {var: keys} / intersection/ ....
            }
            ```
            If a str (column-name) is provided, automatically a logic is
            created that keeps all cases which are not empty for this column.
            If logic is a list, each included list-item becomes a category of
            the new filter-variable and all cases are kept that satisfy all
            conditions (intersection)
        overwrite: bool, default False
            Overwrite an already existing filter-variable.
        """
        if name in self and overwrite:
            self.drop(name)
        if not self._meta._verify_new_name(name):
            err = "Cannot create '{}'. Weak duplicates exist: {}"
            err = err.format(name, self.get_weak_dupes(name))
            logger.error(err); raise ValueError(err)
        values = [(0, 'keep', None)]
        values += self._transform_filter_logics(logic, 1)
        self.add_meta(
            name, 'delimited set', name, [(x, y) for x, y, z in values],
            properties={'recoded_filter': True})
        self.recode(name, {x: z for x, y, z in values[1:]})
        self.recode(name, {0: {name: has_count(len(values) - 1)}})

    @params(to_list=['logic'])
    def extend_filter_var(self, name, logic, suffix=None):
        """
        Extend logic of an existing filter-variable.

        Parameters
        ----------
        name: str
            Name of the existing filter variable.
        logic: (list of) complex logic/ str
            Additional logic to keep cases (intersection with existing logic).
            Complex logic should be provided in form of:
            ```
            {
            'label': 'any text',
            'logic': {var: keys} / intersection/ ....
            }
            ```
        suffix: str, default None
            Addition to the filter-name to create a new filter. If it is None
            the existing filter-variable is overwritten.
        """
        if not self.is_filter(name):
            err = '{} is no valid filter-variable.'.format(name)
            logger.error(err); raise ValueError(err)
        if suffix:
            f_name = '{}_{}'.format(name, suffix)
            if not self._meta._verify_new_name(f_name):
                err = "Cannot create '{}'. Weak duplicates exist: {}"
                err = err.format(f_name, self.get_weak_dupes(f_name))
                logger.error(err); raise ValueError(err)
            self.copy(name, f_name)
        else:
            f_name = name
        self.uncode(f_name, {0: {f_name: 0}})
        values = self._transform_filter_logics(
            logic, max(self.get_codes(f_name)) + 1)
        self.extend_values(f_name, [(x, y) for x, y, z in values])
        self.recode(f_name, {x: z for x, y, z in values})
        self.recode(
            f_name, {0: {f_name: has_count(len(self.get_codes(f_name)) - 1)}})
        text = '{} _ {}'.format(self.get_text(f_name), suffix)
        self.set_text(f_name, text)

    def _transform_filter_logics(self, logic, start):
        if not logic:
            logic = ['@1']
        values = []
        for x, log in enumerate(logic, start):
            if isinstance(log, str):
                if log not in self:
                    raise KeyError("{} is not included in Dataset".format(log))
                val = (x, '{} not empty'.format(log), {log: not_count(0)})
            elif isinstance(log, dict):
                if not ('label' in log and 'logic' in log):
                    log = {'label': str(x), 'logic': log}
                val = (x, log['label'], log['logic'])
            else:
                try:
                    log[0].__name__ in ['_intersection', '_union']
                    val = (x, str(x), log)
                except IndexError:
                    msg = 'Invalid logic'
                    raise TypeError(msg)
            values.append(val)
        return values

    @params(to_list=['values'])
    def reduce_filter_var(self, name, values):
        """
        Remove values from filter-variables and recalculate the filter.
        """
        if not self.is_filter(name):
            raise KeyError('{} is no valid filter-variable.'.format(name))
        if 0 in values:
            raise ValueError('Cannot remove the 0-keep value from filter var')
        elif len([x for x in self.get_codes(name) if x not in values]) <= 1:
            raise ValueError('Cannot remove all values from filter var.')
        self.uncode(name, {0: {name: 0}})
        self.remove_values(name, values)
        self.recode(
            name, {0: {name: has_count(len(self.get_codes(name)) - 1)}})

    def manifest_filter(self, name):
        """
        Get index slicer from filter-variables.

        Parameters
        ----------
        name: str
            Name of the filter_variable.
        """
        if not self.is_filter(name):
            raise KeyError('{} is no valid filter-variable.'.format(name))
        return self.take(name)

    @params(to_list="filters")
    def merge_filter(self, name, filters):
        if not all(f in self.filters for f in filters):
            raise KeyError("Not all included names are valid filters.")
        logic = {
            'label': 'merged filter logics',
            'logic': union([{f: 0} for f in filters])}
        self.add_filter_var(name, logic, True)

    @params(is_column=["name1", "name2"], to_list=['name2'])
    def compare_filter(self, name1, name2):
        """
        Show if filters result in the same index.

        Parameters
        ----------
        name1: str
            Name of the first filter variable
        name2: str/ list of st
            Name(s) of the filter variable(s) to compare with.
        """
        if not all(self.is_filter(f) for f in [name1] + name2):
            raise ValueError('Can only compare filter variables')
        equal = True
        f0 = self.manifest_filter(name1).tolist()
        for f in name2:
            if not f0 == self.manifest_filter(f).tolist():
                equal = False
        return equal

    @params(to_list=["name2"])
    def is_subfilter(self, name1, name2):
        """
        Verify if index of name2 is part of the index of name1.
        """
        idx = self.manifest_filter(name1).tolist()
        included = True
        for n in name2:
            if [i for i in self.manifest_filter(n).tolist() if i not in idx]:
                included = False
        return included

    # ------------------------------------------------------------------------
    # lists/ sets of variables/ data file items
    # ------------------------------------------------------------------------
    @params(to_list=['vlist', 'fix'])
    def align_order(self, vlist, align_against=None,
                    integrate_rc=(["_rc", "_rb"], True), fix=[]):
        """
        Align list to existing order.

        Parameters
        ----------
        vlist: list of str
            The list which should be reordered.
        align_against: str or list of str, default None
            The list of variables to align against. If a string is provided,
            the depending set list is taken. If None, "data file" set is taken.
        integrate_rc: tuple (list, bool)
            The provided list are the suffixes for recodes, the bool decides
            whether parent variables should be replaced by their recodes if
            the parent variable is not in vlist.
        fix: list of str
            Variables which are fixed at the beginning of the reordered list.
        """
        # get list to align against
        if not align_against:
            align_against = self.variables_from_set("data file")
        elif isinstance(align_against, str):
            align_against = self.variables_from_set(align_against)

        # recode suffixes and replace parent
        if not integrate_rc:
            integrate_rc = ([], False)
        rec_suf, repl_parent = integrate_rc

        # create aligned order
        new_vlist = fix[:]
        for v in align_against:
            recodes = ["{}{}".format(v, suf) for suf in rec_suf]
            if v in vlist:
                if v not in new_vlist:
                    new_vlist.append(v)
                for rec in recodes:
                    if rec in vlist and rec not in new_vlist:
                        new_vlist.append(rec)
            elif repl_parent:
                for rec in recodes:
                    if rec in vlist and rec not in new_vlist:
                        new_vlist.append(rec)

        # add missing vars
        miss = [v for v in vlist if v not in new_vlist]
        new_vlist += miss
        return new_vlist

    @params(to_list='new_order')
    def order(self, new_order=None, reposition=None):
        """
        Set the global order of the DataSet variables collection.

        Parameters
        ----------
        new_order : list
            A list of all DataSet variables in the desired order.
        reposition : dict
            mapping of anchor and items to include. Items are added before the
            anchor.
        """
        if new_order and reposition:
            err = "Can only either apply ``new_order`` or ``reposition``"
            logger.error(err); raise ValueError(err)
        if new_order:
            if not sorted(self.variables()) == sorted(new_order):
                err = "'new_order' must contain all DataSet variables."
                logger.error(err); raise ValueError(err)
            self.create_set("data file", include=new_order, overwrite=True)
        elif reposition:
            new_order = insert_by_anchor(self.variables(), reposition)
            new_order = uniquify_list(new_order)
            self.create_set("data file", include=new_order, overwrite=True)
        new_order = ["@1"] + self.unroll(new_order)
        self._data = self._data[new_order]

    # ------------------------------------------------------------------------
    # merging
    # ------------------------------------------------------------------------
    @params(repeat=["dataset"])
    def merge_texts(self, dataset):
        """
        Merge meta only.

        dataset : qp.DataSet
            The instance which is merged into self.
        """
        ds = dataset.clone()
        self._compare_types(ds)
        self._meta.merge(ds._meta, True)

    def _compare_types(self, merge_ds):
        skip = []
        for v in ds.variables():
            if v in self:
                if ds.is_array(v):
                    if not self.is_array(v):
                        skip.append(v)
                        continue
                    ltype = self.get_subtype(v)
                    rtype = ds.get_subtype(v)
                else:
                    ltype = self.get_type(v)
                    rtype = ds.get_type(v)
                if ltype not in COMPATIBLE_TYPES[rtype]:
                    skip.append(v)
                    continue
                ds.convert(v, ltype)

    @params(repeat=["dataset"], is_column=["on", "left_on"])
    def hmerge(self, dataset, on=None, left_on=None, right_on=None,
               overwrite=False, from_set=None):
        """
        Merge other QP dataset instances to extend the variables.

        Parameters
        ----------
        dataset : qp.DataSet
            The instance which is merged into self.
        on : str
            The variable name to identify cases in both datasets.
        left_on : str
            The variable name to identify cases in the left dataset.
        right_on : str
            The variable name to identify cases in the right dataset.
        overwrite : bool
            Define priorization for meta information (e.g. texts):
            *  True: The right dataset is prioritized
            *  False: The left dataset is prioritized
        from_set : str
            A set name from which variables are taken from.
        """
        if on:
            left_on = on
            right_on = on
        else:
            if not (left_on and right_on):
                err = "Please define variables to identify the cases."
                logger.error(err); raise ValueError(err)
        if from_set:
            ds = dataset.subset(from_set=from_set)
        else:
            ds = dataset.clone()
        slicer = ~data[right_on].isin(self[left_on].values.tolist()).tolist()
        ds.filter({right_on: slicer}, inplace=True)
        self._compare_types(ds)
        skip = self._meta.merge(ds._meta, overwrite)
        skip = self.unroll(skip)
        ds._data.set_index(right_on, drop=False)
        self._data.set_index(left_on, drop=False)
        new = []
        update = []
        for col in ds._data.columns.tolist():
            if col in skip or col == right_on:
                continue
            if col in self._data.columns:
                if self.is_delimited_set(col):
                    self[col] = self[col].astype("str")
                    self[col] = self[col].combine(
                        ds[col],
                        lambda x, y: merge_delimited_sets(x, y))
                else:
                    update.append(col)
            else:
                new.append(col)
        if update:
            self._data.update(ds[update])
        if new:
            kwargs = {'left_on': left_on, 'right_on': right_on, 'how': 'left'}
            self._data.merge(ds._data[new], **kwargs)
        self.reset_index()

    @params(repeat=["dataset"], is_column=["on", "left_on"])
    def vmerge(self, dataset, on=None, left_on=None, right_on=None,
               row_id_name="datasource", left_id=1, right_id=2,
               overwrite=False, from_set=None):
        """
        Merge other QP dataset instances to extend the cases.

        Parameters
        ----------
        dataset : qp.DataSet
            The instance which is merged into self.
        on : str
            The variable name to identify cases in both datasets.
        left_on : str
            The variable name to identify cases in the left dataset.
        right_on : str
            The variable name to identify cases in the right dataset.
        row_id_name : str, default "datasource"
            The named column will be filled with the ids indicated for each
            dataset, as per left_id and right_id
        left_id : int, default 1
            Where the row_id_name column is not already populated for the
            dataset_left, this value will be populated.
        right_id : int, default 2
            Where the row_id_name column is not already populated for the
            dataset_right, this value will be populated.
        overwrite : bool
            Define priorization for meta information (e.g. texts):
            *  True: The right dataset is prioritized
            *  False: The left dataset is prioritized
        from_set : str
            A set name from which variables are taken from.
        """
        if on:
            left_on = on
            right_on = on
        else:
            if not (left_on and right_on):
                err = "Please define variables to identify the cases."
                logger.error(err); raise ValueError(err)
        if from_set:
            ds = dataset.subset(from_set=from_set)
        else:
            ds = dataset.clone()
        ds.subset(variables=self.variables() + [right_on], inplace=True)
        slicer = ~data[right_on].isin(self[left_on].values.tolist()).tolist()
        ds.filter({right_on: slicer}, inplace=True)
        self._compare_types(ds)
        skip = self._meta.merge(ds._meta, overwrite)
        ds.drop(skip)
        ds._data.set_index(right_on, drop=False)
        self._data.set_index(left_on, drop=False)
        self._data = pd.concat([self._data, ds._data])
        self.reset_index()
        if row_id_name not in self:
            self.add_meta(row_id_name, "int", row_id_name)
        lid = self.take(intersection([
            {left_on: not_any(slicer)}, {row_id_name: has_count(0)}]))
        self[lid, row_id_name] = left_id
        rid = self.take(intersection([
            {left_on: has_any(slicer)}, {row_id_name: has_count(0)}]))
        self[rid, row_id_name] = right_id

    # renaming
    # ------------------------------------------------------------------------
    @params(is_var=["name"])
    def rename(self, name, new_name, ignore_items=False):
        """
        Change meta and data name references of the variable defintion.

        Parameters
        ----------
        name : str
            The variable name.
        new_name : str
            The new variable name.
        ignore_items : bool, default False
            *  False : Array items inherit mask's name. Suffix is either the
            initial suffix digit (``_x``) or the item_no.
        """
        old = self.unroll(name)
        self._meta.rename(name, new_name, ignore_items)
        new = self.unroll(new_name)
        mapper = dict(zip(old, new))
        self._data.rename(columns=mapper, inplace=True)

    def dimensionize(self):
        """
        Rename the dataset columns for Dimensions compatibility.
        """
        if self.dimensions_comp:
            err = "Instance is already dimensionized."
            logger.error(err); raise ValueError(err)
        old = self.unroll(self.variables())
        self._meta.dimensionize()
        new = self.unroll(self.variables())
        mapper = dict(zip(old, new))
        self._data.rename(columns=mapper, inplace=True)

    def undimensionize(self, names=None, mapper_to_meta=False):
        """
        Rename the dataset columns to remove Dimensions compatibility.
        """
        if not self.dimensions_comp:
            err = "Instance is already undimensionized."
            logger.error(err); raise ValueError(err)
        old = self.unroll(self.variables())
        self._meta.undimensionize()
        new = self.unroll(self.variables())
        mapper = dict(zip(old, new))
        self._data.rename(columns=mapper, inplace=True)

    # values and codes
    # ------------------------------------------------------------------------
    @params(to_list=["remove"], repeat=["name"])
    def remove_values(self, name, remove):
        """
        Erase value codes safely from both meta and case data components.

        Parameters
        ----------
        name : str
            The variable name (column or mask).
        remove : (list of) int
            The codes to be removed from the ``DataSet`` variable.
        """
        self._meta.remove_values(name, remove)
        for n in self.unroll(name):
            self[n] = self[n].apply(lambda x: remove_codes(x, remove))

    # array items
    # ------------------------------------------------------------------------
    @params(to_list=["remove"])
    def remove_items(self, name, remove):
        """
        Erase array mask items.

        Parameters
        ----------
        name : str
            The variable name (mask).
        remove : (list of) int
            The items to remove. The included ints match up to
            the number of the items (``self.get_item_no('item_name')``).
        """
        sources = self.get_sources(name)
        self._meta.remove_items(name, remove)
        for no, source in enumerate(sources, 1):
            if no in remove:
                self._data.drop(source, axis=1, inplace=True)

    @params(to_list=["new_items"])
    def extend_items(self, name, new_items, text_key=None):
        """
        Extend mask items of an existing array.

        Parameters
        ----------
        name: str
            The variable name (mask).
        new_items : (list of str) or tuples (int, str)
            A list with items of the same type:
            *  all strings -> used as labels, number are created by enumeration
            *  all tuples -> ``[(number, label), ...]``
        text_key : str, default None == self.text_key
            Text key for text-based label information.
        """
        self._meta.extend_items(name, new_items, text_key)
        for source in self.get_sources(name):
            self._add_data_column(name, replace=False)

    # ------------------------------------------------------------------------
    # batch
    # ------------------------------------------------------------------------
    @params(to_list=['ci', 'weights', 'tests'])
    def add_batch(self, name, ci=['c', 'p'], weights=[], tests=[]):
        from .batch import Batch
        return Batch(self, name, ci, weights, tests)

    def get_batch(self, name=None):
        """
        Get existing Batch instance from DataSet meta information.

        Parameters
        ----------
        name: str
            Name of existing Batch instance.
        """
        from .batch import Batch
        if not name:
            return [Batch(self, b) for b in self.batches]
        elif name in self.batches:
            return Batch(self, name)
        else:
            err = "'{}' is not a valid batch.".format(name)
            logger.error(err); raise KeyError(name)

    # @params(to_list='batches')
    # def populate(self, batches='all', verbose=True):
    #     """
    #     Create a ``qp.Stack`` based on all available ``qp.Batch`` definitions

    #     Parameters
    #     ----------
    #     batches: str/ list of str
    #         Name(s) of ``qp.Batch`` instances that are used to populate the
    #         ``qp.Stack``.

    #     Returns
    #     -------
    #     qp.Stack
    #     """
    #     dk = self.name
    #     meta = self._meta
    #     data = self._data
    #     stack = Stack(name='aggregations', add_data={dk: (data, meta)})
    #     batches = stack._check_batches(dk, batches)
    #     for name in batches:
    #         batch = meta['sets']['batches'][name]
    #         xys = batch['x_y_map']
    #         fs = batch['x_filter_map']
    #         fy = batch['y_filter_map']
    #         my = batch['yks']
    #         total_len = len(xys) + len(batch['y_on_y'])
    #         for idx, xy in enumerate(xys, start=1):
    #             x, y = xy
    #             if x == '@':
    #                 if fs[y[0]] is None:
    #                     fi = 'no_filter'
    #                 else:
    #                     fi = {fs[y[0]]: {fs[y[0]]: 0}}
    #                 stack.add_link(dk, fi, x='@', y=y)
    #             else:
    #                 if fs[x] is None:
    #                     fi = 'no_filter'
    #                 else:
    #                     fi = {fs[x]: {fs[x]: 0}}
    #                 stack.add_link(dk, fi, x=x, y=y)
    #             if verbose:
    #                 done = float(idx) / float(total_len) * 100
    #                 print '\r',
    #                 time.sleep(0.01)
    #                 print 'Batch [{}]: {} %'.format(name, round(done, 1)),
    #                 sys.stdout.flush()
    #         for idx, y_on_y in enumerate(batch['y_on_y'], len(xys) + 1):
    #             if fy[y_on_y] is None:
    #                 fi = 'no_filter'
    #             else:
    #                 fi = {fy[y_on_y]: {fy[y_on_y]: 1}}
    #             stack.add_link(dk, fi, x=my[1:], y=my)
    #             if verbose:
    #                 done = float(idx) / float(total_len) * 100
    #                 print '\r',
    #                 time.sleep(0.01)
    #                 print 'Batch [{}]: {} %'.format(name, round(done, 1)),
    #                 sys.stdout.flush()
    #         if verbose:
    #             print '\n'
    #     return stack
