#!/usr/bin/python
# -*- coding: utf-8 -*-

from ..__imports__ import *  # noqa

from .io import (
    quantipy_from_dimensions,
    quantipy_from_ascribe,
    parse_sav_file,
    dimensions_from_quantipy,
    save_sav
)
from .meta import Meta


class DataSet(object):
    """
    A set of casedata ``pandas.DataFrame`` and meta data ``qp.Meta``.
    """
    def __init__(self, name, meta, data):
        self.name = name
        self.path = "."
        if not data:
            data = pd.DataFrame()
        self._data = data
        if not meta:
            meta = Meta()
        self._meta = meta
        self._inherit_meta_properties()
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
        scalar_insert = isinstance(val, (int, float, str, unicode))
        if scalar_insert and not np.isnan(val):
            valid_codes = self.codes(name)
            if self.is_categorical(name) and val not in valid_codes:
                err = "{} is undefined for '{}'! Valid: {}".format(
                    val, name, self.codes(name))
                logger.error(err); raise ValueError(err)
            if self.get_type(name) == 'delimited set':
                val = '{};'.format(val)
        if sliced_insert:
            self._data.loc[slicer, name] = val
        else:
            self._data[name] = val

    def _inherit_meta_properties(self):
        self.text_key = self._meta.text_key
        self.valid_tks = self._meta.valid_tks
        self.dimensions_comp = self._meta.dimensions_comp
        self.dimensions_suffix = self._meta.dimensions_suffix

        self.masks = self._meta.masks
        self.columns = self._meta.columns

        self.singles = self._meta.singles
        self.delimited_sets = self._meta.delimited_sets
        self.ints = self._meta.ints
        self.floats = self._meta.floats
        self.dates = self._meta.dates
        self.strings = self._meta.strings

        self.filters = self._meta.filters

        self.sets = self._meta.sets
        self.batches = self._meta.batches

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
        self.is_array = self._meta.is_array
        self.is_array_item = self._meta.is_array_item
        self.is_numeric = self._meta.is_numeric
        self.is_filter = self._meta.is_filter
        self.is_categorical = self._meta.is_categorical

        # get and set variable information
        # types
        self.get_type = self._meta.get_type
        self.get_subtype = self._meta.get_subtype
        # texts
        self.get_text = self._meta.get_text
        self.set_text = self._meta.set_text
        self.remove_html = self._meta.remove_html
        self.replace_texts = self._meta.replace_texts
        # values
        self.get_values = self._meta.get_values
        self.get_value_texts = self._meta.get_value_texts
        self.set_value_texts = self._meta.set_value_texts
        self.get_codes = self._meta.get_codes
        self.get_codes_from_label = self._meta.get_codes_from_label
        # items / arrays
        self.get_items = self._meta.get_items
        self.get_item_no = self._meta.get_item_no
        self.get_item_texts = self._meta.get_item_texts
        self.set_item_texts = self._meta.set_item_texts
        self.get_parent = self._meta.get_parent
        self.get_sources = self._meta.get_sources
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

        # text_keys
        self.used_text_keys = self._meta.used_text_keys
        self.force_texts = self._meta.force_texts
        self.select_text_keys = self._meta.select_text_keys

        # lists and sets
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
        return copy.deepcopy(self)

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
        keep = ds.variables_from_set("data file")
        drop = ds.unroll([col for col in ds._data.columns if col not in keep])
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
        self._verbose_io(name, path, "quantipy")
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
    def from_stack(cls, stack, fk="no_filter"):
        """
        Use ``qp.Stack`` data and meta to create a ``DataSet`` instance.

        Parameters
        ----------
        stack : qp.Stack
            The Stack instance to convert.
        fk: string
            Filter name if the stack contains more than one filters.
        """
        name = stack.name
        meta = stack[name].meta
        data = stack[name][fk].data
        return cls.from_components(name, data, meta)

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
        self._verbose_io(name, path, "dimensions")
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
        self._verbose_io(name, path, "ascribe")
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
        self._verbose_io(name, path, "SPSS")
        return dataset

    def to_quantipy(self, name=None, path=None):
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

    def to_components(self):
        """
        Return the ``meta`` and ``data`` components of the DataSet instance.
        """
        return self._meta, self._data

    @verify(text_keys='text_key')
    def to_dimensions(self, name=None, path=None, **kwargs):
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
        path_mdd = os.path.join(path, u"{}.mdd".format(name))
        path_ddf = os.path.join(path, u"{}.ddf".format(name))
        ds = self.clone()
        if not ds.dimensions_comp:
            ds.dimensionize()
        dimensions_from_quantipy(
            ds._meta, ds._data, path_mdd, path_ddf, text_key)
        self._verbose_io(name, path, "dimensions", False)

    @verify(text_keys='text_key')
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
            self.extend_set("data file", missing)

    def _repair_one_cat_sets(self):
        for ds in self.delimited_sets:
            if len(self.codes(ds)) == 1:
                self.convert(n, "single")
                logger.info("Auto conversion of '{}' to single.".format(ds))

    def reset_index(self):
        self._data.reset_index(drop=True, inplace=True)

    # ------------------------------------------------------------------------
    # inspect
    # ------------------------------------------------------------------------
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

    def get_codes_in_data(self, name):
        """
        Get a list of codes that exist in data.
        """
        if not self.is_categorical(name):
            err = "Can only find codes in data for categorical variables."
            logger.error(err); raise ValueError(err)
        s = self._data[name]
        if self.is_delimited_set(name):
            if not s.dropna().empty:
                data_codes = s.str.get_dummies(';').columns.tolist()
                data_codes = [int(c) for c in data_codes]
            else:
                data_codes = []
        else:
            data_codes = pd.get_dummies(s).columns.tolist()
        return data_codes

    def get_duplicates(self, name='identity'):
        """
        Returns a list with duplicated values for the provided name.

        Parameters
        ----------
        name : str, default 'identity'
            The column variable name keyed in ``meta['columns']``.
        """
        qtype = self.get_type(name)
        if qtype in ['array', 'delimited set', 'float']:
            err = "Can not check duplicates for type '{}'.".format(qtype)
            logger.error(err); raise TypeError(err)
        vals = self._data[name].value_counts()
        vals = vals.copy().dropna()
        if qtype == 'string':
            vals = vals.drop('__NA__')
        vals = vals[vals >= 2].index.tolist()
        if not qtype == 'string':
            vals = [int(i) for i in vals]
        return vals

    @verify(variables={'sort_by': 'columns'})
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
    @modify(to_list=['count_only', 'count_not'])
    @verify(variables={'name': 'both'}, categorical='name')
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
        full_data = self._data.copy()
        series_data = self._data['@1'].copy()
        slicer, _ = get_logic_index(series_data, condition, full_data)
        return slicer

    @verify(variables={'name': 'columns'})
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

    @modify(to_list='codes')
    @verify(variables={'name': 'both'})
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
            for s in self.sources(name):
                logics.append({s: has_any(codes)})
            slicer = self.take(union(logics))
        else:
            slicer = self.take({name: has_any(codes)})
        return slicer

    @modify(to_list='codes')
    @verify(variables={'name': 'both'})
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
            for s in self.sources(name):
                logics.append({s: has_all(codes)})
            slicer = self.take(intersection(logics))
        else:
            slicer = self.take({name: has_all(codes)})
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

    # ------------------------------------------------------------------------
    # Converting
    # ------------------------------------------------------------------------
    @verify(variables={'name': 'columns'})
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
            for source in self.sources(name):
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
            for source in self.sources(name):
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
            for source in self.sources(name):
                self._as_delimited_set(source)
            self._meta._set_type(name, "delimited set")
            if is_categorical:
                self._meta._del_values(name)
        else:
            self[name] = self[name].apply(
                lambda x: str(int(x)) + ';' if not np.isnan(x) else np.NaN)
            if not self.is_array_item(name):
                self._meta._set_type(name, "delimited set")
                if is_categorical:
                    self._meta._del_values(name)

    def _as_single(self, name):
        """
        Change type to ``single``.
        """
        if self.is_delimited_set(name) and len(self.codes(name)) > 1:
            err = "Cannot convert delimited set into single."
            logger.error(err); raise ValueError(err)
        if self.is_array(name):
            for source in self.sources(name):
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
            for source in self.sources(name):
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
    def _check_and_update_element_def(self, element_def):
        all_int = all(isinstance(v, int) for v in element_def)
        all_str = all(isinstance(v, (str, unicode)) for v in element_def)
        all_tuple = all(isinstance(v, tuple) for v in element_def)
        if not (all_int or all_str or all_tuple):
            err = ("The provided value or item element defintion is invalid:\n{}\n"
                   "Please provide either a list of int, a list of str or a "
                   "list of tuple!")
            raise TypeError(err.format(element_def))
        if all_int:
            if self._verbose_infos:
                warn_msg = ("'text' label information missing, only numerical "
                            "codes created for the element object. Remember to "
                            "add value 'text' metadata manually!")
                warnings.warn(warn_msg)
            element_def = [(c, '') for c in element_def]
        return element_def

    def _make_values_list(self, categories, text_key, start_at=None):
        categories = self._check_and_update_element_def(categories)
        if not start_at:
            start_at = 1
        if not all([isinstance(cat, tuple) for cat in categories]):
            vals = [self._value(no, text_key, lab) for no, lab in
                    enumerate(categories, start_at)]
        else:
            vals = [self._value(cat[0], text_key, cat[1]) for cat in categories]
        codes = [v['value'] for v in vals]
        dupes = [c for c, count in Counter(codes).items() if count > 1]
        if dupes:
            err = "Cannot resolve category definition due to code duplicates: {}"
            raise ValueError(err.format(dupes))
        return vals

    def _add_to_datafile_items_set(self, name):
        datafile_items = self._meta['sets']['data file']['items']
        if self.is_array(name):
            append_name = 'masks@{}'.format(name)
        else:
            append_name = 'columns@{}'.format(name)
        if not append_name in datafile_items and not self._is_array_item(name):
            datafile_items.append(append_name)
        return None

    def _add_all_renames_to_mapper(self, mapper, old, new):
        mapper['masks@{}'.format(old).encode('utf8')] = 'masks@{}'.format(new)
        mapper['columns@{}'.format(old).encode('utf8')] = 'columns@{}'.format(new)
        mapper['lib@values@{}'.format(old).encode('utf8')] = 'lib@values@{}'.format(new)
        mapper[old.encode('utf8')] = new
        mapper['masks@{}'.format(old).decode('utf8')] = 'masks@{}'.format(new)
        mapper['columns@{}'.format(old).decode('utf8')] = 'columns@{}'.format(new)
        mapper['lib@values@{}'.format(old).decode('utf8')] = 'lib@values@{}'.format(new)
        mapper[old.decode('utf8')] = new
        return mapper

    def _check_against_weak_dupes(self, name):
        included = self.resolve_name(name)
        if included and self._verbose_infos:
            w = "weak duplicate is created, {} found in DataSet. Please rename."
            warnings.warn(w.format(included))

    def _clean_codes_against_meta(self, name, codes):
        valid = [c for c in codes if c in self._get_valuemap(name, 'codes')]
        deduped_valid = []
        for v in valid:
            if v not in deduped_valid: deduped_valid.append(v)
        return deduped_valid

    def _clean_items_against_meta(self, name, items):
        return [i for i in items if i in self.sources(name)]

    @staticmethod
    def _value(value, text_key, text):
        """
        Return a well-formed Quantipy value object from the given arguments.

        Parameters
        ----------
        value : int
            The numeric value to be given to the returned value object.
        text_key : str
            The text key to be used when generating the returned value
            object's text object.
        text : str
            The label to be given to the returned value object.
        """
        return {'value': value, 'text': {text_key: text}}

    def _verify_same_value_codes_meta(self, name_a, name_b):
        value_codes_a = self._get_valuemap(name_a, non_mapped='codes')
        value_codes_b = self._get_valuemap(name_b, non_mapped='codes')
        if not set(value_codes_a) == set(value_codes_b):
            msg = "'{}' and '{}' do not share the same code values!"
            raise ValueError(msg.format(name_a, name_b))
        return None

    @classmethod
    def _remove_code(cls, x, code):
        if x is np.NaN:
            return np.NaN
        elif ';' in str(x):
            x = str(x).split(';')
            x = [y for y in x if not (y == str(code))]
            x = ';'.join(x)
            if x =='':
                x = np.NaN
        elif x == code:
            x = np.NaN
        return x

    @classmethod
    def _consecutive_codes(cls, codes):
        return sorted(codes) == range(min(codes), max(codes)+1)

    @classmethod
    def _highest_code(cls, codes):
        return max(codes)

    @classmethod
    def _lowest_code(cls, codes):
        return min(codes)

    def _code_from_text(self, valuemap, text):
        check = dict(valuemap)
        for c, t in check.items():
            t = t.replace(' ', '').lower()
            if t == text: return c

    def _get_missing_map(self, var):
        if self.is_array(var):
            var = self._get_itemmap(var, non_mapped='items')
        else:
            if not isinstance(var, list): var = [var]
        for v in var:
            if self._has_missings(v):
                return self._meta['columns'][v]['missings']
            else:
                return None

    def _get_missing_list(self, var, globally=True):
        if self._has_missings(var):
            miss = self._get_missing_map(var)
            if globally:
                return miss['exclude']
            else:
                miss_list = []
                for miss_type in miss.keys():
                    miss_list.extend(miss[miss_type])
                return miss_list
        else:
            return None

    def _maskname_from_item(self, item_name):
        return self.parents(item_name)[0].split('@')[-1]

    def _verify_data_vs_meta_codes(self, name, raiseError=True):
        data_codes = self.codes_in_data(name)
        meta_codes = self.codes(name)
        wild_codes = [code for code in data_codes if code not in meta_codes]
        if wild_codes:
            if self._verbose_errors:
                msg = "Warning: Meta not consistent with case data for '{}'!"
                print '*' * 60
                print msg.format(name)
                if raiseError: print '*' * 60
                print 'Found in data: {}'.format(data_codes)
                print 'Defined as per meta: {}'.format(meta_codes)
            if raiseError:
                raise ValueError('Please review your data processing!')
        return None

    def _verify_old_vs_new_codes(self, name, new_codes):
        org_codes = self.codes(name)
        equal = set(org_codes) == set(new_codes)
        if not equal:
            missing_codes = [c for c in org_codes if c not in new_codes]
            wild_codes = [c for c in new_codes if c not in org_codes]
            if self._verbose_errors:
                print '*' * 60
                if missing_codes:
                    msg = "Warning: Code order is incomplete for '{}'!"
                    print msg.format(name)
                if wild_codes:
                    msg = "Warning: Order contains unknown codes for '{}'!"
                    print msg.format(name)
                print '*' * 60
                if missing_codes: print 'Missing: {}'.format(missing_codes)
                if wild_codes: print 'Unknown: {}'.format(wild_codes)
            raise ValueError('Please review your data processing!')
        return None

    def _get_value_loc(self, var):
        if self._is_numeric(var):
            raise TypeError("Numerical columns do not have 'values' meta.")
        if not self._has_categorical_data(var):
            raise TypeError("Variable '{}' is not categorical!".format(var))
        loc = self._get_meta_loc(var)
        if not self.is_array(var):
            return emulate_meta(self._meta, loc[var].get('values', None))
        else:
            return emulate_meta(self._meta, loc[var])

    def _get_valuemap(self, var, non_mapped=None, text_key=None, axis_edit=None):
        if text_key is None: text_key = self.text_key
        vals = self._get_value_loc(var)
        if non_mapped in ['codes', 'lists', None]:
            codes = [int(v['value']) for v in vals]
            if non_mapped == 'codes':
                return codes
        if non_mapped in ['texts', 'lists', None]:
            if axis_edit:
                a_edit = '{} edits'.format(axis_edit)
                texts = [v['text'][a_edit][text_key]
                         if text_key in v['text'].get(a_edit, []) else None
                         for v in vals]
            else:
                texts = [v['text'][text_key] if text_key in v['text'] else None
                         for v in vals]
            if non_mapped == 'texts':
                return texts
        if non_mapped == 'lists':
            return codes, texts
        else:
            return zip(codes, texts)

    def _get_itemmap(self, var, non_mapped=None, text_key=None, axis_edit=None):
        if text_key is None: text_key = self.text_key
        if non_mapped in ['items', 'lists', None]:
            items = [i['source'].split('@')[-1]
                     for i in self._meta['masks'][var]['items']]
            if non_mapped == 'items':
                return items
        if non_mapped in ['texts', 'lists', None]:
            if axis_edit:
                a_edit = '{} edits'.format(axis_edit)
                items_texts = [i['text'][a_edit][text_key]
                               if text_key in i['text'].get(a_edit, []) else None
                               for i in self._meta['masks'][var]['items']]
            else:
                items_texts = [i['text'][text_key]
                               if text_key in i['text'] else None
                               for i in self._meta['masks'][var]['items']]
            if non_mapped == 'texts':
                return items_texts
        if non_mapped == 'lists':
            return items, items_texts
        else:
            return zip(items, items_texts)

    def _get_source_ref(self, var):
        if self.is_array(var):
            return [i['source'] for i in self._meta['masks'][var]['items']]
        else:
            return []

    def enumerator(self, name):
        x = 1
        n = name
        while n in self:
            x += 1
            n = '{}_{}'.format(name, x)
        return n
    # ------------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------------
    def validate(self, spss_limits=False, verbose=True):
        """
        Identify and report inconsistencies in the ``DataSet`` instance.

        name:
            column/mask name and ``meta[collection][var]['name']`` are not identical
        q_label:
            text object is badly formatted or has empty text mapping
        values:
            categorical variable does not contain values, value text is badly
            formatted or has empty text mapping
        text_keys:
            dataset.text_key is not included or existing text keys are not
            consistent (also for parents)
        source:
            parents or items do not exist
        codes:
            codes in data component are not included in meta component
        spss limit name:
            length of name is greater than spss limit (64 characters)
            (only shown if spss_limits=True)
        spss limit q_label:
            length of q_label is greater than spss limit (256 characters)
            (only shown if spss_limits=True)
        spss limit values:
            length of any value text is greater than spss limit (120 characters)
            (only shown if spss_limits=True)
        """
        def validate_text_obj(text_obj):
            edits = ['x edits', 'y edits']
            if not isinstance(text_obj, dict):
                return False
            else:
                for tk, text in text_obj.items():
                    if ((tk in edits and not validate_text_obj(text_obj[tk]))
                        or text in [None, '', ' ']):
                        return False
            return True

        def validate_value_obj(value_obj):
            if not value_obj:
                return False
            else:
                for val in value_obj:
                    if not 'value' in val or not validate_text_obj(val.get('text')):
                        return False
                return True

        def validate_limits(text_obj, limit):
            if isinstance(text_obj, dict):
                for text in text_obj.values():
                    if isinstance(text, (str, unicode)):
                        if len(text) > limit:
                            return False
                    elif not validate_limits(text.values(), limit):
                        return False
                return True

        def collect_and_validate_tks(all_text_obj):
            edits = ['x edits', 'y edits']
            tks = []
            for obj in all_text_obj:
                if not isinstance(obj, dict): continue
                for tk in obj.keys():
                    if tk in ['x edits', 'y edits']: continue
                    if not tk in tks: tks.append(tk)
            if not self.text_key in tks: return False
            for obj in all_text_obj:
                if not isinstance(obj, dict): continue
                if not all(tk in obj for tk in tks): return False
            return True

        msg = 'Please check the following variables, metadata is inconsistent.'
        err_columns = ['name', 'q_label', 'values', 'text keys', 'source', 'codes',
                       'spss limit name', 'spss limit q_label', 'spss limit values']
        if not spss_limits: err_columns = err_columns[:6]
        err_df = pd.DataFrame(columns=err_columns)

        skip = [v for v in self.masks() + self.columns() if v.startswith('qualityControl_')]
        skip += ['@1', 'id_L1.1', 'id_L1']

        for v in self.columns() + self.masks():
            if v in skip: continue
            collection = 'masks' if self.is_array(v) else 'columns'
            var = self._meta[collection][v]
            err_var = ['' for x in range(9)]
            # check name
            if not var.get('name') == v: err_var[0] = 'x'
            if len(var.get('name', '')) > 64: err_var[6] = 'x'
            # check q_label
            if not validate_text_obj(var.get('text')):
                err_var[1] = 'x'
            elif not validate_limits(var.get('text', {}), 256):
                err_var[7] = 'x'
            # check values
            if self._has_categorical_data(v):
                values = self._get_value_loc(v)
                if not validate_value_obj(values):
                    err_var[2] = 'x'
                    values = []
                elif not all(validate_limits(c.get('text', {}), 120) for c in values):
                    err_var[8] = 'x'
            else:
                values = []
            # check sources
            if self._is_array_item(v):
                source = self._maskname_from_item(v)
                s = self._meta['masks'][source]
                s_tks = [s.get('text')]
                if not self.var_exists(source): err_var[4] = 'x'
            elif self.is_array(v):
                source = self.sources(v)
                s_tks = []
                if not all(self.var_exists(i) for i in source): err_var[4] = 'x'
            else:
                s_tks = []
            # check text_keys
            all_text_obj = [var.get('text', {})] + [val.get('text', {}) for val in values] + s_tks
            if not collect_and_validate_tks(all_text_obj): err_var[3] = 'x'
            # check codes
            if not self.is_array(v) and self._has_categorical_data(v):
                data_c = self.codes_in_data(v)
                meta_c = self.codes(v)
                if [c for c in data_c if not c in meta_c]: err_var[5] = 'x'
            if not spss_limits:
                err_var = err_var[:6]
                err_columns = err_columns[:6]
            if any(x=='x' for x in err_var):
                new_err = pd.DataFrame([err_var], index=[v], columns=err_columns)
                err_df = err_df.append(new_err)

        for c in [c for c in self._data.columns if not c in self._meta['columns']
                  and not c in skip]:
            err_var = ['' for x in range(9)]
            err_var[5] = 'x'
            if not spss_limits:
                err_var = err_var[:6]
                err_columns = err_columns[:6]
            new_err = pd.DataFrame([err_var], index=[c], columns=err_columns)
            err_df = err_df.append(new_err)

        if not all(self.var_exists(v.split('@')[-1])
                   for v in self._meta['sets']['data file']['items']) and verbose:
            print "'dataset._meta['sets']['data file']['items']' is not consistent!"
        if not len(err_df) == 0:
            if verbose:
                print msg
                print self.validate.__doc__
            return err_df.sort_index()
        else:
            if verbose: print 'No issues found in the dataset!'
            return None

    @modify(to_list=['variables', 'text_key'])
    @verify(text_keys='text_key')
    def compare(self, dataset, variables=None, strict=False, text_key=None):
        """
        Compares types, codes, values, question labels of two datasets.

        Parameters
        ----------
        dataset : quantipy.DataSet instance
            Test if all variables in the provided ``dataset`` are also in
            ``self`` and compare their metadata definitions.
        variables : str, list of str
            Check only these variables
        strict : bool, default False
            If True lower/ upper cases and spaces are taken into account.
        text_key : str, list of str
            The textkeys for which texts are compared.

        Returns
        -------
        None
        """
        def _comp_texts(text1, text2, strict):
            equal = True
            if strict:
                if not text1 == text2: equal = False
            else:
                if not text1:
                    text1 = ' '
                else:
                    text1 = text1.encode('cp1252').decode('ascii', errors='ignore').replace(' ', '').lower()
                if not text2:
                    text2 = ' '
                else:
                    text2 = text2.encode('cp1252').decode('ascii', errors='ignore').replace(' ', '').lower()
                if not (text1 in text2 or text2 in text1): equal = False
            return equal

        columns = ['type', 'q_label', 'codes', 'value texts']
        df = pd.DataFrame(columns=columns)

        if not text_key: text_key = self.valid_tks
        vars1 = self.masks() + self.columns()
        vars2 = dataset.masks() + dataset.columns()
        if not variables: variables = vars2
        comp = [key for key in vars2 if key in vars1 and key in variables]
        no_comp = [key for key in vars2 if not key in vars1 and key in variables]
        if no_comp:
            print '{} are not included in main DataSet.\n'.format(no_comp)
        for var in comp:
            if var == '@1': continue
            row = ['' for x in range(4)]
            if not self._get_type(var) == dataset._get_type(var):
                row[0] = 'x'
            if self._has_categorical_data(var):
                codes1 = self.codes(var)
                codes2 = dataset.codes(var)
                if not codes1 == codes2:
                    row[2] = 'x'
                else:
                    val_texts = {c: '' for c in codes1}
                    for tk in text_key:
                        for values, text2 in zip(self.values(var, tk),
                                                 dataset.value_texts(var, tk)):
                            c, text1 = values
                            if not _comp_texts(text1, text2, strict):
                                val_texts[c] += '{}, '.format(tk)
                    if not all(text=='' for text in val_texts.values()):
                        for c, tk in val_texts.items():
                            if not tk == '':
                                row[3] += '{}: {}'.format(c, tk)
            for tk in text_key:
                text1 = self.text(var, True, tk)
                text2 = dataset.text(var, True, tk)
                if not _comp_texts(text1, text2, strict):
                    row[1] += '{}, '.format(tk)
            if not all(x=='' for x in row):
                new_row = pd.DataFrame([row], index=[var], columns=columns)
                df = df.append(new_row)
        if not len(df) == 0: return df.sort_index()

    def weight(self, weight_scheme, weight_name='weight', unique_key='identity',
               subset=None, report=True, path_report=None, inplace=True, verbose=True):
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
        inplace : bool, default True
            If True, the weight factors are merged back into the ``DataSet``
            instance. Will otherwise return the ``pandas.DataFrame`` that
            contains the weight factors, the ``unique_key`` and all variables
            that have been used to compute the weights (filters, target
            variables, etc.).

        Returns
        -------
        None or ``pandas.DataFrame``
            Will either create a new column called ``'weight'`` in the
            ``DataSet`` instance or return a ``DataFrame`` that contains
            the weight factors.
        """
        if subset:
            if isinstance(subset, basestring):
                if self.is_filter(subset):
                    subset = {subset: 0}
                else:
                    raise ValueError('{} is not a valid filter_var'.format(subset))
            ds = self.filter('subset', subset, False)
            meta, data = ds.split()
        else:
            meta, data = self.split()
        engine = qp.WeightEngine(data, meta=meta)
        engine.add_scheme(weight_scheme, key=unique_key, verbose=verbose)
        engine.run()

        org_wname = weight_name
        if report:
            print engine.get_report()
            print
        if path_report:
            df = engine.get_report()
            full_file_path = '{} ({}).xlsx'.format(path_report, weight_name)
            df.to_excel(full_file_path)
            print 'Weight report saved to:\n{}'.format(full_file_path)
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
    # lists/ sets of variables/ data file items
    # ------------------------------------------------------------------------
    def _apply_order(self, variables):
        # set order of 'data file' items listing
        datafile_items = self._variables_to_set_format(variables)
        self._meta['sets']['data file']['items'] = datafile_items
        # set pd.DataFrame column order
        column_order = self.unroll(variables)
        self._data = self._data[column_order]
        return None

    def _mapped_by_substring(self):
        suffixed = {}
        suffixed_variables = self.find()
        if suffixed_variables:
            for sv in suffixed_variables:
                for suffix in VAR_SUFFIXES:
                    if suffix in sv:
                        origin = sv.split(suffix)[0]

                        # test name...
                        origin_res = self.resolve_name(origin)
                        if not origin_res:
                            origin_res = origin
                        if isinstance(origin_res, list):
                            if len(origin_res) > 1:
                                msg = "Unable to regroup to {}, ".format(origin)
                                msg += "found weak duplicate derived names:\n"
                                msg += "{}".format(origin_res)
                                warnings.warn(msg)
                                origin_res = origin
                            else:
                                origin_res = origin_res[0]

                        if not origin_res in suffixed:
                            suffixed[origin_res] = [sv]
                        else:
                            suffixed[origin_res].append(sv)
        return suffixed

    def _mapped_by_meta(self):
        rec_views = {}
        for v in self.variables():
            origin = self.get_property(v, 'recoded_net')
            if origin:
                if not origin in rec_views:
                    rec_views[origin] = [v]
                else:
                    rec_views[origin].append(v)
        return rec_views

    def _map_to_origins(self):
        by_origins = self._mapped_by_substring()
        recoded_views = self._mapped_by_meta()
        varlist = self.variables()
        for var in varlist:
            if var in recoded_views:
                if not var in by_origins:
                    by_origins[var] = recoded_views[var]
                else:
                    for recoded_view in recoded_views[var]:
                        if recoded_view not in by_origins[var]:
                            by_origins[var].append(recoded_view)
        for k, v in by_origins.items():
            if not k in varlist:
                del by_origins[k]
                if not v[0] in varlist:
                    by_origins[v[0]] = v[1:]
        sort_them = []
        for k, v in by_origins.items():
            sort_them.append(k)
            sort_them.extend(v)
        grouped = []
        for v in varlist:
            if v in by_origins:
                grouped.append(v)
                grouped.extend(by_origins[v])
            else:
                if not v in sort_them: grouped.append(v)
        return grouped

    @modify(to_list=['vlist', 'fix'])
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
            align_against = self._variables_from_set("data file")
        elif isinstance(align_against, basestring):
            align_against = self._variables_from_set(align_against)

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

    @modify(to_list='reposition')
    def order(self, new_order=None, reposition=None, regroup=False):
        """
        Set the global order of the DataSet variables collection.

        The global order of the DataSet is reflected in the data component's
        pd.DataFrame.columns order and the variable references in the meta
        component's 'data file' items.

        Parameters
        ----------
        new_order : list
            A list of all DataSet variables in the desired order.
        reposition : (List of) dict
            Each dict maps one or a list of variables to a reference variable
            name key. The mapped variables are moved before the reference key.
        regroup : bool, default False
            Attempt to regroup non-native variables (i.e. created either
            manually with ``add_meta()``, ``recode()``, ``derive()``, etc.
            or automatically by manifesting ``qp.View`` objects) with their
            originating variables.

        Returns
        -------
        None
        """
        if (bool(new_order) + bool(reposition) + regroup) > 1:
            err = "Can only either apply ``new_order``, ``reposition`` or "
            err += "``regroup`` variables, not perform multiple operations at once."
            raise ValueError(err)
        if new_order:
            if not sorted(self._variables_from_set('data file')) == sorted(new_order):
                err = "'new_order' must contain all DataSet variables."
                raise ValueError(err)
            check = new_order
        elif reposition:
            check = []
            for r in reposition:
                check.extend(list(r.keys() + r.values()))
        elif regroup:
            new_order = self._map_to_origins()
            check = new_order
        else:
            err = "No ``order`` operation provided, select one of "
            err += "``new_order``, ``regroup``, ``reposition``."
            raise ValueError(err)
        if not all(self.var_exists(v) for v in check):
            err = "At least one variable named in ordering does not exist."
            raise ValueError(err)
        if reposition:
            new_order = self._variables_from_set('data file')
            for repos in reposition:
                before_var = repos.keys()[0]
                repos_vars = repos.values()[0]
                if not isinstance(repos_vars, list): repos_vars = [repos_vars]
                repos_vars = list(reversed(repos_vars))
                idx = new_order.index(before_var)
                for repos_var in repos_vars:
                    new_order.remove(repos_var)
                    new_order.insert(idx, repos_var)
        self._apply_order(new_order)
        return None

    @modify(to_list=['logic'])
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
            the new filter-variable and all cases are kept that satify all
            conditions (intersection)

        overwrite: bool, default False
            Overwrite an already existing filter-variable.
        """
        name = self._verify_filter_name(name, None)
        if name in self:
            if overwrite and not self.is_filter(name):
                msg = "Cannot add filter-variable '{}', a non-filter"
                msg +=" variable is already included"
                raise ValueError(msg.format(name))
            elif not overwrite:
                msg = "Cannot add filter-variable '{}', it's already included."
                raise ValueError(msg.format(name))
            else:
                self.drop(name)
                if self._verbose_infos:
                    print 'Overwriting {}'.format(name)
        values = [(0, 'keep', None)]
        values += self._transform_filter_logics(logic, 1)
        self.add_meta(name, 'delimited set', name, [(x, y) for x, y, z in values])
        self.recode(name, {x: z for x, y, z in values[1:]})
        self.recode(name, {0: {name: has_count(len(values)-1)}}, append=True)
        self._set_property(name, 'recoded_filter', True)
        return None

    @modify(to_list=['logic'])
    def extend_filter_var(self, name, logic, extend_as=None):
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
        extend_as: str, default None
            Addition to the filter-name to create a new filter. If it is None
            the existing filter-variable is overwritten.
        """
        if not self.is_filter(name):
            raise KeyError('{} is no valid filter-variable.'.format(name))
        name = self._verify_filter_name(name, None)
        if extend_as:
            extend_as = self._verify_filter_name(extend_as, None)
            f_name = '{}_{}'.format(name, extend_as)
            if f_name in self:
                msg = "Please change 'extend_as': '{}' is already in dataset."
                raise KeyError(msg.format(f_name))
            self.copy(name, extend_as)
            self._meta['columns'][f_name]['properties']['recoded_filter'] = True
        else:
            f_name = name
        self.uncode(f_name, {0: {f_name: 0}})
        values = self._transform_filter_logics(logic, max(self.codes(f_name))+1)
        self.extend_values(f_name, values)
        self.recode(f_name, {x: z for x, y, z in values}, append=True)
        self.recode(f_name, {0: {f_name: has_count(len(self.codes(f_name))-1)}}, append=True)
        text = '{} _ {}'.format(self.text(f_name), extend_as)
        self._meta['columns'][f_name]['text'][self.text_key] = text
        return None

    def _transform_filter_logics(self, logic, start):
        if not logic: logic = ['@1']
        values = []
        for x, l in enumerate(logic, start):
            if isinstance(l, basestring):
                if not l in self:
                    raise KeyError("{} is not included in Dataset".format(l))
                val = (x, '{} not empty'.format(l), {l: not_count(0)})
            elif isinstance(l, dict):
                if not ('label' in l and 'logic' in l):
                    l = {'label': str(x), 'logic': l}
                    if self._verbose_infos:
                        msg = "Filter logic must contain 'label' and 'logic'"
                        warnings.warn(msg)
                val = (x, l['label'], l['logic'])
            else:
                try:
                    l[0].__name__ in ['_intersection', '_union']
                    val = (x, str(x), l)
                except:
                    msg = 'Included logic must be (list of) str or dict/complex logic.'
                    raise TypeError(msg)
            values.append(val)
        return values

    def _verify_filter_name(self, name, suf='f', number=False):
        f = '{}_{}'.format(name, suf) if suf else name
        f = f.encode('utf8')
        repl = [(' ', '_'), ('~', '_'), ('(', ''), (')', ''), ('&', '_')]
        for r in repl:
            f = f.replace(r[0], r[1])
        if number:
            f = self.enumerator(f)
        return f

    @modify(to_list=['values'])
    def reduce_filter_var(self, name, values):
        """
        Remove values from filter-variables and recalculate the filter.
        """
        name = self._verify_filter_name(name, None)
        if not self.is_filter(name):
            raise KeyError('{} is no valid filter-variable.'.format(name))
        if 0 in values:
            raise ValueError('Cannot remove the 0-keep value from filter var')
        elif len([x for x in self.codes(name) if not x in values]) <= 1:
            raise ValueError('Cannot remove all values from filter var.')
        self.uncode(name, {0: {name: 0}})
        self.remove_values(name, values)
        self.recode(name, {0: {name: has_count(len(self.codes(name))-1)}}, append=True)
        return None

    def manifest_filter(self, name):
        """
        Get index slicer from filter-variables.

        Parameters
        ----------
        name: str
            Name of the filter_variable.
        """
        if not name:
            return self._data.index
        else:
            name = self._verify_filter_name(name, None)
        if not self.is_filter(name):
            raise KeyError('{} is no valid filter-variable.'.format(name))
        return self.take({name: 0})

    @modify(to_list="filters")
    def merge_filter(self, name, filters):
        if not all(f in self.filters() for f in filters):
            raise KeyError("Not all included names are valid filters.")
        logic = {
            'label': 'merged filter logics',
            'logic': union([{f: 0} for f in filters])
            }
        self.add_filter_var(name, logic, True)
        return None

    @modify(to_list=['name2'])
    @verify(variables={'name1': 'both', 'name2': 'both'})
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

    @modify(to_list=["name2"])
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
    # extending / merging
    # ------------------------------------------------------------------------

    @modify(to_list=['dataset'])
    @verify(variables={'on': 'columns', 'left_on': 'columns'})
    def hmerge(self, dataset, on=None, left_on=None, right_on=None,
               overwrite_text=False, from_set=None, inplace=True,
               merge_existing=None, verbose=True):

        """
        Merge Quantipy datasets together using an index-wise identifer.

        This function merges two Quantipy datasets together, updating variables
        that exist in the left dataset and appending others. New variables
        will be appended in the order indicated by the 'data file' set if
        found, otherwise they will be appended in alphanumeric order.
        This merge happend horizontally (column-wise). Packed kwargs will be
        passed on to the pandas.DataFrame.merge() method call, but that merge
        will always happen using how='left'.

        Parameters
        ----------
        dataset : ``quantipy.DataSet``
            The dataset to merge into the current ``DataSet``.
        on : str, default=None
            The column to use as a join key for both datasets.
        left_on : str, default=None
            The column to use as a join key for the left dataset.
        right_on : str, default=None
            The column to use as a join key for the right dataset.
        overwrite_text : bool, default=False
            If True, text_keys in the left meta that also exist in right
            meta will be overwritten instead of ignored.
        from_set : str, default=None
            Use a set defined in the right meta to control which columns are
            merged from the right dataset.
        inplace : bool, default True
            If True, the ``DataSet`` will be modified inplace with new/updated
            columns. Will return a new ``DataSet`` instance if False.
        verbose : bool, default=True
            Echo progress feedback to the output pane.

        Returns
        -------
        None or new_dataset : ``quantipy.DataSet``
            If the merge is not applied ``inplace``, a ``DataSet`` instance
            is returned.
        """
        new_tks = []
        for d in dataset:
            for tk in d.valid_tks:
                if not d in self.valid_tks and not d in new_tks:
                    new_tks.append(tk)
        self.extend_valid_tks(new_tks)
        ds_left = (self._meta, self._data)
        ds_right = [(ds._meta, ds._data) for ds in dataset]
        if on is None and right_on in self.columns():
            id_backup = self._data[right_on].copy()
        else:
            id_backup = None
        merged_meta, merged_data = _hmerge(
            ds_left, ds_right, on=on, left_on=left_on, right_on=right_on,
            overwrite_text=overwrite_text, from_set=from_set, verbose=verbose,
            merge_existing=merge_existing)
        if id_backup is not None:
            merged_data[right_on] = id_backup
        if inplace:
            self._data = merged_data
            self._meta = merged_meta
            return None
        else:
            new_dataset = self.clone()
            new_dataset._data = merged_data
            new_dataset._meta = merged_meta
            return new_dataset
        return None

    def update(self, data, on='identity'):
        """
        Update the ``DataSet`` with the case data entries found in ``data``.

        Parameters
        ----------
        data : ``pandas.DataFrame``
            A dataframe that contains a subset of columns from the ``DataSet``
            case data component.
        on : str, default 'identity'
            The column to use as a join key.

        Returns
        -------
        None
            DataSet is modified inplace.
        """
        ds_left = (self._meta, self._data)
        update_meta = self._meta.copy()
        update_items = ['columns@{}'.format(name) for name
                        in data.columns.tolist()]
        update_meta['sets']['update'] = {'items': update_items}
        ds_right = (update_meta, data)
        merged_meta, merged_data = _hmerge(
            ds_left, ds_right, on=on, from_set='update', verbose=False)
        self._meta, self._data = merged_meta, merged_data
        del self._meta['sets']['update']
        return None

    @modify(to_list=['dataset'])
    @verify(variables={'on': 'columns', 'left_on': 'columns'})
    def vmerge(self, dataset, on=None, left_on=None, right_on=None,
               row_id_name=None, left_id=None, right_id=None, row_ids=None,
               overwrite_text=False, from_set=None, uniquify_key=None,
               reset_index=True, inplace=True, verbose=True):
        """
        Merge Quantipy datasets together by appending rows.

        This function merges two Quantipy datasets together, updating variables
        that exist in the left dataset and appending others. New variables
        will be appended in the order indicated by the 'data file' set if
        found, otherwise they will be appended in alphanumeric order. This
        merge happens vertically (row-wise).

        Parameters
        ----------
        dataset : (A list of multiple) ``quantipy.DataSet``
            One or multiple datasets to merge into the current ``DataSet``.
        on : str, default=None
            The column to use to identify unique rows in both datasets.
        left_on : str, default=None
            The column to use to identify unique in the left dataset.
        right_on : str, default=None
            The column to use to identify unique in the right dataset.
        row_id_name : str, default=None
            The named column will be filled with the ids indicated for each
            dataset, as per left_id/right_id/row_ids. If meta for the named
            column doesn't already exist a new column definition will be
            added and assigned a reductive-appropriate type.
        left_id : str/int/float, default=None
            Where the row_id_name column is not already populated for the
            dataset_left, this value will be populated.
        right_id : str/int/float, default=None
            Where the row_id_name column is not already populated for the
            dataset_right, this value will be populated.
        row_ids : list of str/int/float, default=None
            When datasets has been used, this list provides the row ids
            that will be populated in the row_id_name column for each of
            those datasets, respectively.
        overwrite_text : bool, default=False
            If True, text_keys in the left meta that also exist in right
            meta will be overwritten instead of ignored.
        from_set : str, default=None
            Use a set defined in the right meta to control which columns are
            merged from the right dataset.
        uniquify_key : str, default None
            A int-like column name found in all the passed ``DataSet`` objects
            that will be protected from having duplicates. The original version
            of the column will be kept under its name prefixed with 'original'.
        reset_index : bool, default=True
            If True pandas.DataFrame.reindex() will be applied to the merged
            dataframe.
        inplace : bool, default True
            If True, the ``DataSet`` will be modified inplace with new/updated
            rows. Will return a new ``DataSet`` instance if False.
        verbose : bool, default=True
            Echo progress feedback to the output pane.

        Returns
        -------
        None or new_dataset : ``quantipy.DataSet``
            If the merge is not applied ``inplace``, a ``DataSet`` instance
            is returned.
        """
        datasets = [(self._meta, self._data)]
        merge_ds = [(ds._meta, ds._data) for ds in dataset]
        datasets.extend(merge_ds)
        merged_meta, merged_data = _vmerge(
            None, None, datasets, on=on, left_on=left_on,
            right_on=right_on, row_id_name=row_id_name, left_id=left_id,
            right_id=right_id, row_ids=row_ids, overwrite_text=overwrite_text,
            from_set=from_set, reset_index=reset_index, verbose=verbose)
        if inplace:
            self._data = merged_data
            self._meta = merged_meta
            if uniquify_key:
                self._make_unique_key(uniquify_key, row_id_name)
            return None
        else:
            new_dataset = self.clone()
            new_dataset._data = merged_data
            new_dataset._meta = merged_meta
            if uniquify_key:
                new_dataset._make_unique_key(uniquify_key, row_id_name)
            return new_dataset


    @verify(variables={'id_key_name': 'columns', 'multiplier': 'columns'})
    def _make_unique_key(self, id_key_name, multiplier):
        """
        """
        columns = self._meta['columns']
        if columns[id_key_name]['type'] not in ['int', 'float']:
            raise TypeError("'id_key_name' must be of type int, float, single!")
        elif columns[multiplier]['type'] not in ['single', 'int', 'float']:
            raise TypeError("'multiplier' must be of type int, float, single!")
        org_key_col = self._data.copy()[id_key_name]
        new_name = 'original_{}'.format(id_key_name)
        name, qtype, lab = new_name, 'int', 'Original ID'
        self.add_meta(name, qtype, lab)
        self[new_name] = org_key_col
        self[id_key_name] += self[multiplier].astype(int) * 100000000
        return None

    @modify(to_list='dataset')
    def merge_texts(self, dataset):
        """
        Add additional ``text`` versions from other ``text_key`` meta.

        Case data will be ignored during the merging process.

        Parameters
        ----------
        dataset : (A list of multiple) ``quantipy.DataSet``
            One or multiple datasets that provide new ``text_key`` meta.

        Returns
        -------
        None
        """
        for ds in dataset:
            empty_data = ds._data.copy()
            ds._data = ds._data[ds._data.index < 0]
        self.vmerge(dataset, verbose=False, overwrite_text=True)
        return None

    # ------------------------------------------------------------------------
    # Recoding
    # ------------------------------------------------------------------------
    def _add_data_column(self, name, qtype, replace=True):
        if replace or name not in self._data.columns:
            if qtype == "delimited set":
                self._data[name] = ''
            else:
                self._data[name] = np.NaN

    @modify(to_list=['categories', 'items'])
    def add_meta(self, name, qtype, label, categories=None, items=None,
        text_key=None, replace=True):
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
        self._meta.add_meta(name, qtype, label, categories, items, text_key)
        name = self.unroll(name)
        for n in name:
            self._add_data_column(name, qtype, replace)

    @verify(variables={'name': 'columns'})
    def categorize(self, name, categorized_name=None):
        """
        Categorize an ``int``/``string``/``text`` variable to ``single``.

        The ``values`` object of the categorized variable is populated with the
        unique values found in the originating variable (ignoring np.NaN /
        empty row entries).

        Parameters
        ----------
        name : str
            The column variable name keyed in ``meta['columns']`` that will
            be categorized.
        categorized_name : str
            If provided, the categorized variable's new name will be drawn
            from here, otherwise a default name in form of ``'name#'`` will be
            used.

        Returns
        -------
        None
            DataSet is modified inplace, adding the categorized variable to it.
        """
        org_type = self._get_type(name)
        valid_types = ['int', 'string', 'date']
        if org_type not in valid_types:
            raise TypeError('Can only categorize {}!'.format(valid_types))
        new_var_name = categorized_name or '{}#'.format(name)
        self.copy(name)
        self.convert('{}_rec'.format(name), 'single')
        self.rename('{}_rec'.format(name), new_var_name)
        return None

    @modify(to_list='ignore')
    @verify(variables={'name': 'columns'}, text_keys='text_key')
    def dichotomize(self, name, value_texts=None, keep_variable_text=True,
                    ignore=None, replace=False, text_key=None):
        """
        """
        if not text_key: text_key = self.text_key
        if not value_texts: value_texts = ('Yes', 'No')
        if not isinstance(value_texts, (list, tuple)):
            err = "'value_texts' must be list-like."
            raise TypeError(err)
        elif len(value_texts) != 2:
            err = "'value_texts' must contain exactly two elements."
            raise ValueError(err)
        elif value_texts[0] == value_texts[1]:
            err = "'value_texts' must contain two different elements."
            raise ValueError(err)
        values = self.values(name, text_key)
        if ignore: values = [v for v in values if v[0] not in ignore]
        new_vars = []
        for value in values:
            code, text = value[0], value[1]
            dicho_name = '{}_{}'.format(name, code)
            new_vars.append(dicho_name)
            if keep_variable_text:
                dicho_label = '{}: {}'.format(self.text(name, text_key), text)
            else:
                dicho_label = text
            cond = [(1, value_texts[0],  {name: [code]})]
            self.derive(dicho_name, 'single', dicho_label, cond)
            self.extend_values(dicho_name, (0, value_texts[1]), text_key=text_key)
            self[self.is_nan(dicho_name), dicho_name] = 0
        if self._verbose_infos:
            print 'created: {}'.format(new_vars)
        if replace:
            new_order = {name: new_vars}
            self.order(reposition=new_order)
            self.drop(name)
        return None

    @verify(variables={'name': 'columns'})
    def first_responses(self, name, n=3, others='others', reduce_values=False):
        """
        Create n-first mentions from the set of responses of a delimited set.

        Parameters
        ----------
        name : str
            The column variable name of a delimited set keyed in
            ``meta['columns']``.
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

        Returns
        -------
        None
            DataSet is modified inplace.
        """
        if self._get_type(name) != 'delimited set' or self.empty(name):
            return None
        created = []
        values = self.values(name)
        for _n in frange('1-{}'.format(n)):
            n_name = '{}_{}'.format(name, _n)
            n_label = '{} ({})'.format(self.text(name), _n)
            self.add_meta(n_name, 'single', n_label, values)
            n_vector = self[name].str.split(';', n=_n, expand=True)[_n-1]
            self[n_name] = n_vector.replace(('', None), np.NaN).astype(float)
            created.append(n_name)
        if others:
            o_name = '{}_{}'.format(name, others)
            o_label = '{} ({})'.format(self.text(name), others)
            self.add_meta(o_name, 'delimited set', o_label, values)
            o_string = self[name].str.split(';', n=n, expand=True)[n]
            self[o_name] = o_string.replace(('', None), np.NaN)
            created.append(o_name)
        if reduce_values:
            for v in created:
                reduce_codes = [value[0] for value in values
                                if value[0] not in self.codes_in_data(v)]
                self.remove_values(v, reduce_codes)
        return None

    @modify(to_list='codes')
    @verify(variables={'name': 'masks'}, text_keys='text_key')
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
        text_key : str, default None
            Text key for text-based label information. Uses the
            ``DataSet.text_key`` information if not provided.
        Returns
        -------
        None
            The DataSet is modified inplace, delimited set variable is added.
        """
        if not new_name:
            if '.' in name:
                new_name = '{}_rec'.format(name.split('.')[0])
            else:
                new_name = '{}_rec'.format(name)
        if not text_key: text_key = self.text_key
        label = self._meta['masks'][name]['text'][text_key]
        cats = self.item_texts(name)
        self.add_meta(new_name, 'delimited set', label, cats)
        for x, source in enumerate(self.sources(name), 1):
            self.recode(new_name, {x: {source: codes}}, append=True)
        return None

    @modify(to_list='name')
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
        self._meta.drop(name, ignore_items)
        data_drop = [self.unroll(n) for n in names if not ignore_items]
        data.drop(data_drop, 1, inplace=True)

    @modify(to_list=['name'])
    def unbind(self, name):
        """
        Remove mask-structure for arrays
        """
        if not all(self.is_array(n) for n in name):
            err = "Can only unbind arrays!"
            logger.error(err); raise ValueError(err)
        for n in name:
            self.drop(n, ignore_items=True)

    @modify(to_list=['copy_only', 'copy_not'])
    def copy(self, name, suffix='rec', copy_data=True, slicer=None, copy_only=None,
             copy_not=None):
        """
        Copy meta and case data of the variable defintion given per ``name``.

        Parameters
        ----------
        name : str
            The originating column variable name keyed in ``meta['columns']``
            or ``meta['masks']``.
        suffix : str, default 'rec'
            The new variable name will be constructed by suffixing the original
            ``name`` with ``_suffix``, e.g. ``'age_rec``.
        copy_data : bool, default True
            The new variable assumes the ``data`` of the original variable.
        slicer : dict
            If the data is copied it is possible to filter the data with a
            complex logic. Example: slicer = {'q1': not_any([99])}
        copy_only: int or list of int, default None
            If provided, the copied version of the variable will only contain
            (data and) meta for the specified codes.
        copy_not: int or list of int, default None
            If provided, the copied version of the variable will contain
            (data and) meta for the all codes, except of the indicated.

        Returns
        -------
        None
            DataSet is modified inplace, adding a copy to both the data and meta
            component.
        """
        if copy_only and copy_not:
            raise ValueError("Must pass either 'copy_only' or 'copy_not', not both!")
        verify_name = name[0] if isinstance(name, tuple) else name
        is_array = self.is_array(verify_name)

        array_item_copied = isinstance(name, tuple)
        if not array_item_copied and self._is_array_item(verify_name):
            err = ("Cannot make isolated copy of array item '{}'. "
                   "Please copy array variable '{}' instead!")
            err = err.format(verify_name, self.parents(verify_name)[0].split('@')[-1])
            raise NotImplementedError(err)

        meta = self._meta
        if not 'renames' in meta['sets']: meta['sets']['renames'] = {}
        renames = meta['sets']['renames']
        # are we dealing with an recursive array item copy?
        if array_item_copied:
            copy_name = '{}_{}'.format(name[1], suffix)
            name = name[0]
        else:
            copy_name = '{}_{}'.format(self._dims_free_arr_name(name), suffix)
        # force stripped names...
        if not renames:
            self.undimensionize([name] + self.sources(name))
            name = self._dims_free_arr_name(name)

        check_name = self._dims_compat_arr_name(copy_name)
        if self.var_exists(check_name): self.drop(check_name)
        self._check_against_weak_dupes(check_name)

        if is_array:
            # copy meta and create rename mapper for array items
            renames = self._add_all_renames_to_mapper(renames, name, copy_name)
            meta['masks'][copy_name] = org_copy.deepcopy(meta['masks'][name])
            meta['sets'][copy_name] = org_copy.deepcopy(meta['sets'][name])
            self._add_to_datafile_items_set(copy_name)
            for item in self.sources(name):
                item_name_split = item.split('_')
                element_name = '_'.join(item_name_split[:-1])
                element_no = item_name_split[-1]
                new_item_name = '{}_{}_{}'.format(element_name, suffix, element_no)
                self.copy((item, element_name), '{}_{}'.format(suffix, element_no),
                          copy_data, slicer=slicer, copy_only=copy_only)
                renames[item] = new_item_name
        else:
            # copy regular 'columns' meta data
            renames = self._add_all_renames_to_mapper(renames, name, copy_name)
            meta['columns'][copy_name] = org_copy.deepcopy(meta['columns'][name])
            meta['columns'][copy_name]['name'] = copy_name
            self._add_to_datafile_items_set(copy_name)
            # handle the case data copy for columns (incl.slicing)
            if copy_data:
                if slicer:
                    self._data[copy_name] = np.NaN
                    take = self.take(slicer)
                    self[take, copy_name] = self._data[name].copy()
                else:
                    self._data[copy_name] = self._data[name].copy()
            else:
                self._data[copy_name] = np.NaN

        # run the renaming for the copied variable
        self.rename_from_mapper(renames, keep_original=True, ignore_batch_props=True)
        # set type 'created'
        if is_array:
            for s in self.sources(copy_name):
                if meta['columns'][s].get('properties'):
                    for q_type in ['survey', 'open', 'system', 'merged']:
                        meta['columns'][s]['properties'][q_type] = False
                    meta['columns'][s]['properties']['created'] = True
        elif not self._is_array_item(copy_name):
            if meta['columns'][copy_name].get('properties'):
                for q_type in ['survey', 'open', 'system', 'merged']:
                    meta['columns'][copy_name]['properties'][q_type] = False
                meta['columns'][copy_name]['properties']['created'] = True
        # finished, i.e. not any longer inside a recursive array item copy?
        if is_array:
            finalized = len(self.sources(name)) == len(self.sources(copy_name))
        elif self._is_array_item(name):
            finalized = False
        else:
            finalized = True
        if finalized:
            # reduce the meta/data?
            if copy_not:
                remove = [c for c in self.codes(copy_name) if c in copy_not]
                self.remove_values(copy_name, remove)
            if copy_only:
                remove = [c for c in self.codes(copy_name) if not c in copy_only]
                self.remove_values(copy_name, remove)
            del meta['sets']['renames']
            # restore Dimensions-like names if in compatibility mode
            if self._dimensions_comp:
                self.dimensionize(copy_name)
                self.dimensionize(name)
        return None

    def copy_array_data(self, source, target, source_items=None,
                        target_items=None, slicer=None):
        """
        """
        self._verify_same_value_codes_meta(source, target)
        all_source_items = self._get_itemmap(source, non_mapped='items')
        all_target_items = self._get_itemmap(target, non_mapped='items')
        if slicer: mask = self.take(slicer)
        if source_items:
            source_items = [all_source_items[i-1] for i in source_items]
        else:
            source_items = all_source_items
        if target_items:
            target_items = [all_target_items[i-1] for i in target_items]
        else:
            target_items = all_target_items
        for s, t in zip(source_items, target_items):
                if slicer:
                    self._data.loc[mask, t] = self._data.loc[mask, s]
                else:
                    self[t] = self[s]
        return None

    @modify(to_list=['ignore_items', 'ignore_values'])
    @verify(variables={'name': 'masks'}, text_keys='text_key')
    def transpose(self, name, new_name=None, ignore_items=None,
                  ignore_values=None, copy_data=True, text_key=None,
                  overwrite=False):
        """
        Create a new array mask with transposed items / values structure.

        This method will automatically create meta and case data additions in
        the ``DataSet`` instance.

        Parameters
        ----------
        name : str
            The originating mask variable name keyed in ``meta['masks']``.
        new_name : str, default None
            The name of the new mask. If not provided explicitly, the new_name
            will be constructed constructed by suffixing the original
            ``name`` with '_trans', e.g. ``'Q2Array_trans``.
        ignore_items : int or list of int, default None
            If provided, the items listed by their order number in the
            ``_meta['masks'][name]['items']`` object will not be part of the
            transposed array. This means they will be ignored while creating
            the new value codes meta.
        ignore_codes : int or list of int, default None
            If provided, the listed code values will not be part of the
            transposed array. This means they will not be part of the new
            item meta.
        text_key : str
            The text key to be used when generating text objects, i.e.
            item and value labels.
        overwrite: bool, default False
            Overwrite variable if `new_name` is already included.

        Returns
        -------
        None
            DataSet is modified inplace.
        """
        if (new_name and self._dims_compat_arr_name(new_name) in self and
            not overwrite):
            raise ValueError("'{}' is already included.".format(new_name))
        if not new_name:
            new_name = '{}_trans'.format(name)
        if new_name == name:
            tname = '{}_trans'.format(new_name)
        else:
            tname = new_name

        # input item_map
        item_map = self._get_itemmap(name)
        item_labels = [(x, i[1]) for x, i in enumerate(item_map, 1)
                       if not x in ignore_items]
        item_vars = [(x, i[0]) for x, i in enumerate(item_map, 1)
                     if not x in ignore_items]
        # input value_map
        value_map = self._get_valuemap(name)
        value_map = [v for x, v in enumerate(value_map, 1)
                     if not x in ignore_values]
        # input label
        label = self.text(name, False, text_key)
        # add transposed meta
        self.add_meta(tname, "delimited set", label, item_labels, value_map, text_key)
        tname = self._dims_compat_arr_name(tname)

        # transpose the data
        tsources = self.sources(tname)
        mapper = {
            tsource: {} for tsource in tsources
        }
        for code, source in zip([val[0] for val in value_map], tsources):
            mapper = {}
            for x, item in item_vars:
                mapper[x] = {item: code}
            self.recode(source, mapper, append=True)
        if new_name == name:
            print("Overwrite '{}'.".format(name))
            self.drop(name)
            self.rename(tname, new_name)

    @verify(variables={'target': 'columns'})
    def recode(self, target, mapper, default=None, append=False,
               intersect=None, initialize=None, fillna=None, inplace=True):
        """
        Create a new or copied series from data, recoded using a mapper.

        This function takes a mapper of {key: logic} entries and injects the
        key into the target column where its paired logic is True. The logic
        may be arbitrarily complex and may refer to any other variable or
        variables in data. Where a pre-existing column has been used to
        start the recode, the injected values can replace or be appended to
        any data found there to begin with. Note that this function does
        not edit the target column, it returns a recoded copy of the target
        column. The recoded data will always comply with the column type
        indicated for the target column according to the meta.

        Parameters
        ----------
        target : str
            The column variable name keyed in ``_meta['columns']`` that is the
            target of the recode. If not found in ``_meta`` this will fail
            with an error. If ``target`` is not found in data.columns the
            recode will start from an empty series with the same index as
            ``_data``. If ``target`` is found in data.columns the recode will
            start from a copy of that column.
        mapper : dict
            A mapper of {key: logic} entries.
        default : str, default None
            The column name to default to in cases where unattended lists
            are given in your logic, where an auto-transformation of
            {key: list} to {key: {default: list}} is provided. Note that
            lists in logical statements are themselves a form of shorthand
            and this will ultimately be interpreted as:
            {key: {default: has_any(list)}}.
        append : bool, default False
            Should the new recoded data be appended to values already found
            in the series? If False, data from series (where found) will
            overwrite whatever was found for that item instead.
        intersect : logical statement, default None
            If a logical statement is given here then it will be used as an
            implied intersection of all logical conditions given in the
            mapper.
        initialize : str or np.NaN, default None
            If not None, a copy of the data named column will be used to
            populate the target column before the recode is performed.
            Alternatively, initialize can be used to populate the target
            column with np.NaNs (overwriting whatever may be there) prior
            to the recode.
        fillna : int, default=None
            If not None, the value passed to fillna will be used on the
            recoded series as per pandas.Series.fillna().
        inplace : bool, default True
            If True, the ``DataSet`` will be modified inplace with new/updated
            columns. Will return a new recoded ``pandas.Series`` instance if
            False.

        Returns
        -------
        None or recode_series
            Either the ``DataSet._data`` is modified inplace or a new
            ``pandas.Series`` is returned.
        """
        meta = self._meta
        data = self._data
        recode_series = _recode(meta, data, target, mapper,
                                default, append, intersect, initialize, fillna)
        if inplace:
            self._data[target] = recode_series
            if not self._is_numeric(target):
                self._verify_data_vs_meta_codes(target)
            return None
        else:
            return recode_series

    @verify(variables={'target': 'both'})
    def uncode(self, target, mapper, default=None, intersect=None, inplace=True):
        """
        Create a new or copied series from data, recoded using a mapper.

        Parameters
        ----------
        target : str
            The variable name that is the target of the uncode. If it is keyed
            in ``_meta['masks']`` the uncode is done for all mask items.
            If not found in ``_meta`` this will fail with an error.
        mapper : dict
            A mapper of {key: logic} entries.
        default : str, default None
            The column name to default to in cases where unattended lists
            are given in your logic, where an auto-transformation of
            {key: list} to {key: {default: list}} is provided. Note that
            lists in logical statements are themselves a form of shorthand
            and this will ultimately be interpreted as:
            {key: {default: has_any(list)}}.
        intersect : logical statement, default None
            If a logical statement is given here then it will be used as an
            implied intersection of all logical conditions given in the
            mapper.
        inplace : bool, default True
            If True, the ``DataSet`` will be modified inplace with new/updated
            columns. Will return a new recoded ``pandas.Series`` instance if
            False.

        Returns
        -------
        None or uncode_series
            Either the ``DataSet._data`` is modified inplace or a new
            ``pandas.Series`` is returned.
        """
        meta = self._meta
        data = self._data
        if self.is_array(target):
            targets = self.sources(target)
            if inplace:
                for t in targets:
                    self.uncode(t, mapper, default, intersect, inplace)
                return None
            else:
                uncode_series = []
                for t in targets:
                    uncode_series.append(self.uncode(t, mapper, default,
                                                     intersect, inplace))
                return uncode_series
        else:
            if not target in meta['columns']:
                raise ValueError("{} not found in meta['columns'].".format(target))

            if not isinstance(mapper, dict):
                raise ValueError("'mapper' must be a dictionary.")

            if not (default is None or default in meta['columns']):
                raise ValueError("'{}' not found in meta['columns'].".format(default))

            index_map = index_mapper(meta, data, mapper, default, intersect)

            uncode_series = self[target].copy()
            for code, index in index_map.items():
                uncode_series[index] = uncode_series[index].apply(lambda x:
                                                    self._remove_code(x, code))

            if inplace:
                self._data[target] = uncode_series
                if not self._is_numeric(target):
                    self._verify_data_vs_meta_codes(target)
                return None
            else:
                return uncode_series

    def interlock(self, name, label, variables, val_text_sep = '/'):
        """
        Build a new category-intersected variable from >=2 incoming variables.

        Parameters
        ----------
        name : str
            The new column variable name keyed in ``_meta['columns']``.
        label : str
            The new text label for the created variable.
        variables : list of >= 2 str or dict (mapper)
            The column names of the variables that are feeding into the
            intersecting recode operation. Or dicts/mapper to create temporary
            variables for interlock. Can also be a mix of str and dict. Example:

            >>> ['gender',
            ...  {'agegrp': [(1, '18-34', {'age': frange('18-34')}),
            ...              (2, '35-54', {'age': frange('35-54')}),
            ...              (3, '55+', {'age': is_ge(55)})]},
            ...  'region']
        val_text_sep : str, default '/'
            The passed character (or any other str value) wil be used to
            separate the incoming individual value texts to make up the inter-
            sected category value texts, e.g.: 'Female/18-30/London'.

        Returns
        -------
        None
        """
        if not isinstance(variables, list) or len(variables) < 2:
            raise ValueError("'variables' must be a list of at least two items!")

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

        codes = [self._get_valuemap(v, 'codes') for v in i_variables]
        texts = [self._get_valuemap(v, 'texts') for v in i_variables]
        zipped = zip(list(product(*codes)), list(product(*texts)))
        categories = []
        cat_id = 0
        for codes, texts in zipped:
            cat_id += 1
            cat_label = val_text_sep.join(texts)
            rec = [{v: [c]} for v, c in zip(i_variables, codes)]
            rec = intersection(rec)
            categories.append((cat_id, cat_label, rec))
        self.derive(name, qtype, label, categories)
        for var in new_variables:
            self.drop(var)
        return None

    @verify(variables={'name': 'masks'})
    def _level(self, name):
        """
        """
        self.copy(name, 'level')
        if self._dimensions_comp:
            temp = self._dims_free_arr_name(name)
            lvlname = self._dims_compat_arr_name('{}_level'.format(temp))
        else:
            lvlname = '{}_level'.format(name)
        items = self.items(name)
        sources = enumerate(self.sources(lvlname), 1)
        codes = self.codes(lvlname)
        max_code = len(codes)
        replace_codes = {}
        mapped_codes = {c: [] for c in self.codes(name)}

        for no, source in sources:
            offset = (no-1) * max_code
            new_codes = frange('{}-{}'.format((offset + 1), (offset + max_code)))
            replace_codes[source] = dict(zip(codes, new_codes))

        for source, codes in replace_codes.items():
            self[source].replace(codes, inplace=True)
            self[source].replace(np.NaN, '', inplace=True)
            for org, new in codes.items():
                mapped_codes[org].append(new)

        code_range = frange('1-{}'.format(max_code * len(items)))
        labels = self.value_texts(name) * len(items)
        cats = zip(code_range, labels)
        new_sources = self.sources(lvlname)
        self.unbind(lvlname)
        self.add_meta(lvlname, 'delimited set', self.text(name), cats)
        self[lvlname] = self[new_sources].astype('str').apply(
            lambda x: ';'.join(x).replace('.0', ''), axis=1)
        self.drop(new_sources)
        self._meta['columns'][lvlname]['properties']['level'] = {
            'source': name,
            'level_codes': mapped_codes,
            'item_look': self.sources(name)[0]}
        return None

    @verify(text_keys='text_key')
    def derive(self, name, qtype, label, cond_map, text_key=None):
        """
        Create meta and recode case data by specifying derived category logics.

        Parameters
        ----------
        name : str
            The column variable name keyed in ``meta['columns']``.
        qtype : [``int``, ``float``, ``single``, ``delimited set``]
            The structural type of the data the meta describes.
        label : str
            The ``text`` label information.
        cond_map : list of tuples
            Tuples of either two or three elements of following structures:

            2 elements, no labels provided:
            (code, <qp logic expression here>), e.g.:
            ``(1, intersection([{'gender': [1]}, {'age': frange('30-40')}]))``

            2 elements, no codes provided:
            ('text label', <qp logic expression here>), e.g.:
            ``('Cat 1', intersection([{'gender': [1]}, {'age': frange('30-40')}]))``

            3 elements, with codes + labels:
            (code, 'Label goes here', <qp logic expression here>), e.g.:
            ``(1, 'Men, 30 to 40', intersection([{'gender': [1]}, {'age': frange('30-40')}]))``

        text_key : str, default None
            Text key for text-based label information. Will automatically fall
            back to the instance's text_key property information if not provided.

        Returns
        -------
        None
            ``DataSet`` is modified inplace.
        """
        if not text_key: text_key = self.text_key
        append = qtype == 'delimited set'
        err_msg = ("'cond_map' structure not understood. Must pass a list "
                   "of 2 (code, logic) / (text, logic) or 3 (code, text label, "
                   "logic) element tuples!")
        if all(len(cond) == 3 for cond in cond_map):
            categories = [(cond[0], cond[1]) for cond in cond_map]
            idx_mapper = {cond[0]: cond[-1] for cond in cond_map}
        elif all(len(cond) == 2 for cond in cond_map):
            all_int = all(isinstance(cond[0], int) for cond in cond_map)
            all_str = all(isinstance(cond[0], (str, unicode)) for cond in cond_map)
            if not (all_str or all_int):
                raise TypeError(err_msg)
            categories = [cond[0] for cond in cond_map]
            if all_int:
                idx_mapper = {cond[0]: cond[-1] for cond in cond_map}
            if all_str:
                idx_mapper = {c: cond[-1] for c, cond in enumerate(cond_map, start=1)}
        else:
            raise TypeError(err_msg)
        self.add_meta(name, qtype, label, categories, items=None, text_key=text_key)
        self.recode(name, idx_mapper, append=append)
        return None

    @verify(variables={'name': 'columns'}, text_keys='text_key')
    def band(self, name, bands, new_name=None, label=None, text_key=None):
        """
        Group numeric data with band definitions treated as group text labels.

        Wrapper around ``derive()`` for quick banding of numeric
        data.

        Parameters
        ----------
        name : str
            The column variable name keyed in ``_meta['columns']`` that will
            be banded into summarized categories.
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
        text_key : str, default None
            Text key for text-based label information. Uses the
            ``DataSet.text_key`` information if not provided.

        Returns
        -------
        None
            ``DataSet`` is modified inplace.
        """
        if not self._is_numeric(name):
            msg = "Can only band numeric typed data! {} is {}."
            msg = msg.format(name, self._get_type(name))
            raise TypeError(msg)
        if not text_key: text_key = self.text_key
        if not new_name: new_name = '{}_banded'.format(name)
        if not label: label = self.text(name, False, text_key)
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

        self.derive(new_name, 'single', label, franges,
                                text_key=text_key)

        return None

    @modify(to_list='variables')
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
        variables: list of str or list of tuples
            variables that get combined into the new delimited set. If they are
            dichotomous (from_dichotomous=True), the labels of the variables
            are used as category texts or if tuples are included, the second
            items will be used for the category texts.
            If the variables are categorical (from_dichotomous=False) the values
            of the variables need to be eqaul and are taken for the delimited set.
        from_dichotomous: bool, default True
            Define if the input variables are dichotomous or categorical.
        codes_from_name: bool, default True
            If from_dichotomous=True, the codes can be taken from the Variable
            names, if they are in form of 'q01_1', 'q01_3', ...
            In this case the codes will be 1, 3, ....

        Returns
        -------
        None
        """
        if self.var_exists(name):
            raise ValueError('{} does already exist.'.format(name))
        elif not all(isinstance(c, (str, unicode, tuple)) for c in variables):
            raise ValueError('Input of variables must be string or tuple.')
        cols = [c if isinstance(c, (str, unicode)) else c[0] for c in variables]
        if not all(self.var_exists(c) for c in cols):
            not_in_ds = [c for c in cols if not self.var_exists(c)]
            raise KeyError('{} not found in dataset!'.format(not_in_ds))
        elif not all(self._has_categorical_data(c) for c in cols):
            not_cat = [c for c in cols if not self._has_categorical_data(c)]
            raise ValueError('Variables must have categorical data: {}'.format(not_cat))
        if from_dichotomous:
            if not all(x in [0, 1] for c in cols for x in self.codes_in_data(c)):
                non_d = [c for c in cols
                         if not all(x in [0, 1] for x in self.codes_in_data(c))]
                raise ValueError('Variables are not dichotomous: {}'.format(non_d))
            mapper = []
            for x, col in enumerate(variables, 1):
                if codes_from_name:
                    x = int(col.split('_')[-1])
                if isinstance(col, tuple):
                    text = col[1]
                else:
                    text = self.text(col)
                mapper.append((x, text, {col: [1]}))
        else:
            values = self.values(cols[0])
            if not all(self.values(c) == values for c in cols):
                not_eq = [c for c in cols if not self.values(c) == values]
                raise ValueError('Variables must have eqaul values: {}'.format(not_eq))
            mapper = []
            for v in values:
                mapper.append((v[0], v[1], union([{c: v[0]} for c in cols])))

        self.derive(name, 'delimited set', label, mapper)

        return None

    def to_array(self, name, variables, label, safe=True):
        """
        Combines column variables with same ``values`` meta into an array.

        Parameters
        ----------
        name : str
            Name of new grid.
        variables : list of str or list of dicts
            Variable names that become items of the array. New item labels can
            be added as dict. Example:
            variables = ['q1_1', {'q1_2': 'shop 2'}, {'q1_3': 'shop 3'}]
        label : str
            Text label for the mask itself.
        safe : bool, default True
            If True, the method will raise a ``ValueError`` if the provided
            variable name is already present in self. Select ``False`` to
            forcefully overwrite an existing variable with the same name
            (independent of its type).

        Returns
        -------
        None
        """
        meta = self._meta
        newname = self._dims_compat_arr_name(name)
        if self.var_exists(newname):
            if safe:
                raise ValueError('{} does already exist.'.format(name))
            self.drop(newname, ignore_items=True)
        var_list = [v.keys()[0] if isinstance(v, dict)
                     else v for v in variables]
        if not all(self.var_exists(v) for v in var_list):
            raise KeyError("'variables' must be included in DataSet.")
        elif not len(set(var_list)) == len(var_list):
            raise ValueError("'variables' contains duplicates!")
        to_comb = {v.keys()[0]: v.values()[0] for v in variables if isinstance(v, dict)}
        for var in var_list:
            to_comb[var] = self.text(var) if var in variables else to_comb[var]
        first = var_list[0]
        subtype = self._get_type(var_list[0])
        if self._has_categorical_data(var_list[0]):
            categorical = True
            if not all(self.codes(var) == self.codes(first) for var in var_list):
                raise ValueError("Variables must have same 'codes' in meta.")
            elif not all(self.values(var) == self.values(first) for var in var_list):
                msg = 'Not all variables have the same value texts. Assume valuemap'
                msg += ' of {} for the mask'.format(first)
                warnings.warn(msg)
            val_map = self._get_value_loc(first)
        else:
            categorical = False
        items = []
        name_set = []
        for v in var_list:
            item = {'properties': {},
                    'source': 'columns@{}'.format(v),
                    'text': {self.text_key: to_comb[v]}}
            if categorical:
                meta['columns'][v]['values'] = 'lib@values@{}'.format(name)
            meta['columns'][v]['parent'] = {'masks@{}'.format(name): {'type': 'array'}}
            name_set.append('columns@{}'.format(v))
            items.append(item)
        meta['masks'][name] = {'name': name,
                               'items': items,
                               'properties': {},
                               'text': {self.text_key: label},
                               'type': 'array',
                               'subtype': subtype}
        if categorical:
            meta['masks'][name]['values'] = 'lib@values@{}'.format(name)
            meta['lib']['values'][name] = val_map
        meta['sets'][name] = {'items': name_set}
        meta['sets']['data file']['items'].append('masks@{}'.format(name))
        meta['sets']['data file']['items'] = [v for v in meta['sets']['data file']['items']
                                                if not v in name_set]

        if self._dimensions_comp and not self._dimensions_comp == 'ignore':
            self.dimensionize(name)
        return None


    # renaming
    # ------------------------------------------------------------------------

    @verify(variables={'name': 'both'})
    def rename(self, name, new_name):
        """
        Change meta and data column name references of the variable defintion.

        Parameters
        ----------
        name : str
            The originating column variable name keyed in ``meta['columns']``
            or ``meta['masks']``.
        new_name : str
            The new variable name.

        Returns
        -------
        None
            DataSet is modified inplace. The new name reference replaces the
            original one.
        """
        renames = {}
        if new_name in self._data.columns:
            msg = "Cannot rename '{}' into '{}'. Column name already exists!"
            raise ValueError(msg.format(name, new_name))

        self._in_blacklist(new_name)
        self._check_against_weak_dupes(new_name)

        if not self._dimensions_comp == 'ignore':
            self.undimensionize([name] + self.sources(name))
            name = self._dims_free_arr_name(name)

        for no, s in enumerate(self.sources(name), start=1):
            if '_' in s and s.split('_')[-1].isdigit():
                new_s_name = '{}_{}'.format(new_name, s.split('_')[-1])
            else:
                new_s_name = '{}_{}'.format(new_name, no)
            self._add_all_renames_to_mapper(renames, s, new_s_name)

        self._add_all_renames_to_mapper(renames, name, new_name)

        self.rename_from_mapper(renames)

        if self._dimensions_comp and not self._dimensions_comp == 'ignore':
            self.dimensionize(new_name)

        return None

    def rename_from_mapper(self, mapper, keep_original=False,
                           ignore_batch_props=False):
        """
        Rename meta objects and data columns using mapper.

        Parameters
        ----------
        mapper : dict
            A renaming mapper in the form of a dict of {old: new} that
            will be used to rename columns throughout the meta and data.


        Returns
        -------
        None
            DataSet is modified inplace.
        """
        def rename_properties(mapper):
            """
            Rename variable properties that reference other variables, i.e.
            'recoded_net', 'recoded_stat' meta objects.
            """
            net_recs = self._net_view_recodes()
            stat_recs = self._stat_view_recodes()
            all_recs = set([r for r in net_recs + stat_recs if r in mapper])
            for rec in all_recs:
                if self._is_array_item(rec): continue
                rn = self.get_property(rec, 'recoded_net')
                if rn: self._set_property(rec, 'recoded_net', mapper[rn])
                rs = self.get_property(rec, 'recoded_stat')
                if rs: self._set_property(rec, 'recoded_stat', mapper[rs])
            return None

        def rename_meta(meta, mapper, ignore_batch_props):
            """
            Rename lib@values, masks, set items and columns using mapper.
            """
            rename_properties(mapper)
            rename_lib_values(meta['lib']['values'], mapper)
            rename_masks(meta['masks'], mapper, keep_original)
            rename_columns(meta['columns'], mapper, keep_original)
            rename_sets(meta['sets'], mapper, keep_original)
            if 'batches' in meta['sets'] and not ignore_batch_props:
                rename_batch_properties(meta['sets']['batches'], mapper)
            if not keep_original:
                rename_set_items(meta['sets'], mapper)

        def rename_lib_values(lib_values, mapper):
            """
            Rename lib@values objects using mapper.
            """
            for name, rename in mapper.iteritems():
                if name in lib_values:
                    lib_values[rename] = org_copy.deepcopy(lib_values[name])
                    if not keep_original: del lib_values[name]

        def rename_masks(masks, mapper, keep_original):
            """
            Rename mask objects using mapper.
            """
            for name, rename in mapper.iteritems():
                if name in masks:
                    masks[rename] = org_copy.deepcopy(masks[name])
                    if not keep_original: del masks[name]
                    masks[rename]['name'] = rename

                    if masks[rename].get('values'):
                        values = masks[rename]['values']
                        if isinstance(values, (str, unicode)):
                            if values in mapper:
                                masks[rename]['values'] = mapper[values]

                    items = masks[rename]['items']
                    for i, item in enumerate(items):
                        for key in ['source', 'values']:
                            if item.get(key):
                                if item[key] in mapper:
                                    items[i][key] = mapper[item[key]]

        def rename_columns(columns, mapper, keep_original):
            """
            Rename column objects using mapper.
            """
            for name, rename in mapper.iteritems():
                if name in columns:
                    columns[rename] = org_copy.deepcopy(columns[name])
                    if 'parent' in columns[name]:
                        parents = columns[name]['parent']
                    else:
                        parents = {}
                    if not keep_original: del columns[name]
                    columns[rename]['name'] = rename
                    for parent_name, parent_spec in parents.items():
                        new_parent_map = {}
                        if parent_name in mapper:
                            new_name = mapper[parent_name]
                            new_parent_map[new_name] = parent_spec
                            columns[rename]['parent'] = new_parent_map
                    if columns[rename].get('values'):
                        values = columns[rename]['values']
                        if isinstance(values, (str, unicode)):
                            if values in mapper:
                                columns[rename]['values'] = mapper[values]

        def rename_sets(sets, mapper, keep_original):
            """
            Rename set object items using mapper.
            """
            for name, rename in mapper.iteritems():
                if name in sets:
                    sets[rename] = org_copy.deepcopy(sets[name])
                    if not keep_original: del sets[name]
                    sets[rename]['name'] = rename
                    # copied from 'rename_set_items'
                    items = sets[rename].get('items', False)
                    if items:
                        for i, item in enumerate(items):
                            if item in mapper:
                                items[i] = mapper[item]

        def rename_batch_properties(batches, mapper):

            def _iterate_props(obj, mapper):
                if isinstance(obj, bool):
                    pass
                elif isinstance(obj, basestring):
                    return mapper.get(obj)
                elif isinstance(obj, dict):
                    for k, v in obj.items():
                        if _iterate_props(k, mapper):
                            obj[_iterate_props(k, mapper)] = _iterate_props(v, mapper) or v
                            del obj[k]
                        else:
                            obj[k] = _iterate_props(v, mapper) or v
                elif isinstance(obj, list):
                    return [_iterate_props(a, mapper) or a for a in obj]
                elif isinstance(obj, tuple):
                    return tuple(_iterate_props(a, mapper) or a for a in obj)

            for batch, defs in batches.items():
                _iterate_props(defs, mapper)


        def rename_set_items(sets, mapper):
            """
            Rename standard set object items using mapper.
            """
            for set_name in sets.keys():
                try:
                    items = sets[set_name].get('items', False)
                    if items:
                        for i, item in enumerate(items):
                            if item in mapper:
                                items[i] = mapper[item]
                except (AttributeError, KeyError, TypeError, ValueError):
                    pass

        rename_meta(self._meta, mapper, ignore_batch_props)
        if not keep_original: self._data.rename(columns=mapper, inplace=True)

    def dimensionizing_mapper(self, names=None):
        """
        Return a renaming dataset mapper for dimensionizing names.

        Parameters
        ----------
        None


        Returns
        -------
        mapper : dict
            A renaming mapper in the form of a dict of {old: new} that
            maps non-Dimensions naming conventions to Dimensions naming
            conventions.
        """

        def fix(string):
            tags = [
                "'", '"', ' ', '&', '.', '/', '-',
                '(', ')', '[', ']', '{', '}'
            ]
            for tag in tags:
                string = string.replace(tag, '_')
            return string

        masks = self._meta['masks']
        columns = self._meta['columns']
        suffix = self._dimensions_suffix

        if not names: names = self.variables()
        mapper = {}
        for org_mn, mask in masks.iteritems():
            if org_mn in names:
                mask_name = fix(org_mn)
                new_mask_name = '{mn}.{mn}{s}'.format(mn=mask_name, s=suffix)
                mapper[org_mn] = new_mask_name

                mask_mapper = 'masks@{mn}'.format(mn=org_mn)
                new_mask_mapper = 'masks@{nmn}'.format(nmn=new_mask_name)
                mapper[mask_mapper] = new_mask_mapper

                values_mapper = 'lib@values@{mn}'.format(mn=org_mn)
                new_values_mapper = 'lib@values@{nmn}'.format(nmn=new_mask_name)
                mapper[values_mapper] = new_values_mapper

                items = masks[org_mn]['items']
                for i, item in enumerate(items):
                    org_cn = item['source'].split('@')[-1]
                    col_name = fix(org_cn)
                    new_col_name = '{mn}[{{{cn}}}].{mn}{s}'.format(
                        mn=mask_name, cn=col_name, s=suffix
                    )
                    mapper[org_cn] = new_col_name

                    col_mapper = 'columns@{cn}'.format(cn=org_cn)
                    new_col_mapper = 'columns@{ncn}'.format(ncn=new_col_name)
                    mapper[col_mapper] = new_col_mapper

        for col_name, col in columns.iteritems():
            if col_name in names and not self._is_array_item(col_name):
                new_col_name = fix(col_name)
                if new_col_name == col_name: continue
                mapper[col_name] = new_col_name

                col_mapper = 'columns@{cn}'.format(cn=col_name)
                new_col_mapper = 'columns@{ncn}'.format(ncn=new_col_name)
                mapper[col_mapper] = new_col_mapper

        return mapper

    def undimensionizing_mapper(self, names=None):
        """
        Return a renaming dataset mapper for un-dimensionizing names.

        Parameters
        ----------
        None


        Returns
        -------
        mapper : dict
            A renaming mapper in the form of a dict of {old: new} that
            maps Dimensions naming conventions to non-Dimensions naming
            conventions.
        """

        masks = self._meta['masks']
        columns = self._meta['columns']

        mask_pattern = '(^.+)\..+$'
        column_pattern = '(?<=\[{)(.*?)(?=}\])'

        mapper = {}
        if not names:
            names = masks.keys() + columns.keys()
        for mask_name in masks.keys():
            if mask_name in names:
                matches = re.findall(mask_pattern, mask_name)
                if matches:
                    new_mask_name = matches[0]
                    mapper[mask_name] = new_mask_name

                    mask_mapper = 'masks@{mn}'.format(mn=mask_name)
                    new_mask_mapper = 'masks@{nmn}'.format(nmn=new_mask_name)
                    mapper[mask_mapper] = new_mask_mapper

                    values_mapper = 'lib@values@{mn}'.format(mn=mask_name)
                    new_values_mapper = 'lib@values@{nmn}'.format(nmn=new_mask_name)
                    mapper[values_mapper] = new_values_mapper

        for col_name in columns.keys():
            if col_name in names:
                matches = re.findall(column_pattern, col_name)
                if matches:
                    new_col_name = matches[0]
                    mapper[col_name] = new_col_name
                    col_mapper = 'columns@{mn}'.format(mn=col_name)
                    new_col_mapper = 'columns@{nmn}'.format(nmn=new_col_name)
                    mapper[col_mapper] = new_col_mapper
        return mapper

    @modify(to_list='names')
    @verify(variables={'names': 'both'})
    def dimensionize(self, names=None):
        """
        Rename the dataset columns for Dimensions compatibility.
        """
        if not names and self._dimensions_comp:
            raise ValueError('File is already dimensionized.')
        mapper = self.dimensionizing_mapper(names)
        self.rename_from_mapper(mapper)
        if not names:
            self.set_dim_comp(True)
            if 'type' in self:
                self.rename('type', '_type')
        return None

    @modify(to_list='names')
    @verify(variables={'names': 'both'})
    def undimensionize(self, names=None, mapper_to_meta=False):
        """
        Rename the dataset columns to remove Dimensions compatibility.
        """
        mapper = self.undimensionizing_mapper(names)
        self.rename_from_mapper(mapper)
        if mapper_to_meta: self._meta['sets']['rename_mapper'] = mapper
        if not names: self.set_dim_comp(False)

    # value manipulation
    # ------------------------------------------------------------------------

    @verify(variables={'name': 'both'})
    def reorder_values(self, name, new_order=None):
        """
        Apply a new order to the value codes defined by the meta data component.

        Parameters
        ----------
        name : str
            The column variable name keyed in ``_meta['columns']`` or
            ``_meta['masks']``.
        new_order : list of int, default None
            The new code order of the DataSet variable. If no order is given,
            the ``values`` object is sorted ascending.

        Returns
        -------
        None
            DataSet is modified inplace.
        """
        values = self._get_value_loc(name)
        if not new_order:
            new_order = list(sorted(self._get_valuemap(name, 'codes')))
        else:
            self._verify_old_vs_new_codes(name, new_order)
        new_values = [value for i in new_order for value in values
                      if value['value'] == i]
        if self._get_type(name) == 'array':
            self._meta['lib']['values'][name] = new_values
        else:
            self._meta['columns'][name]['values'] = new_values
        return None

    @modify(to_list='remove')
    @verify(variables={'name': 'both'})
    def remove_values(self, name, remove):
        """
        Erase value codes safely from both meta and case data components.

        Attempting to remove all value codes from the variable's value object
        will raise a ``ValueError``!

        Parameters
        ----------
        name : str
            The originating column variable name keyed in ``meta['columns']``
            or ``meta['masks']``.
        remove : int or list of int
            The codes to be removed from the ``DataSet`` variable.
        Returns
        -------
        None
            DataSet is modified inplace.
        """
        # Do we need to modify a mask's lib def.?
        if not self.is_array(name) and self._is_array_item(name):
            name = self._maskname_from_item(name)
        # Are any meta undefined codes provided? - Warn user!
        values = self._get_value_loc(name)
        codes = self.codes(name)
        ignore_codes = [r for r in remove if r not in codes]
        if ignore_codes:
            print 'Warning: Cannot remove values...'
            print '*' * 60
            msg = "Codes {} not found in values object of '{}'!"
            print msg.format(ignore_codes, name)
            print '*' * 60
            remove = [x for x in remove if x not in ignore_codes]
        # Would be remove all defined values? - Prevent user from doing this!
        new_values = [value for value in values
                      if value['value'] not in remove]
        if not new_values:
            msg = "Cannot remove all codes from the value object of '{}'!"
            raise ValueError(msg.format(name))
        # Apply new ``values`` definition
        if self.is_array(name):
            self._meta['lib']['values'][name] = new_values
        else:
            self._meta['columns'][name]['values'] = new_values
        # Remove values in ``data``
        if self.is_array(name):
            items = self._get_itemmap(name, 'items')
            for i in items:
                self.uncode(i, {x: {i: x} for x in remove})
                self._verify_data_vs_meta_codes(i)
        else:
            self.uncode(name, {x: {name: x} for x in remove})
            self._verify_data_vs_meta_codes(name)
        # convert delimited set to single if only one cat is left
        self._prevent_one_cat_set(name)
        return None

    @modify(to_list='ext_values')
    @verify(variables={'name': 'both'}, categorical='name', text_keys='text_key')
    def extend_values(self, name, ext_values, text_key=None, safe=True):
        """
        Add to the 'values' object of existing column or mask meta data.

        Attempting to add already existing value codes or providing already
        present value texts will both raise a ``ValueError``!

        Parameters
        ----------
        name : str
            The column variable name keyed in ``_meta['columns']`` or
            ``_meta['masks']``.
        ext_values : list of str or tuples in form of (int, str), default None
            When a list of str is given, the categorical values will simply be
            enumerated and mapped to the category labels. Alternatively codes can
            mapped to categorical labels, e.g.:
            [(1, 'Elephant'), (2, 'Mouse'), (999, 'No animal')]
        text_key : str, default None
            Text key for text-based label information. Will automatically fall
            back to the instance's text_key property information if not provided.
        safe : bool, default True
            If set to False, duplicate value texts are allowed when extending
            the ``values`` object.

        Returns
        -------
        None
            The ``DataSet`` is modified inplace.
        """
        # Do we need to modify a mask's lib def.?
        if not self.is_array(name) and self._is_array_item(name):
            name = self._maskname_from_item(name)
        use_array = self.is_array(name)
        if not text_key: text_key = self.text_key
        value_obj = self._get_valuemap(name, text_key=text_key)
        codes = self.codes(name)
        texts = self.value_texts(name)
        if not isinstance(ext_values[0], tuple):
            start_here = self._highest_code(codes) + 1
        else:
            start_here = None
        ext_values = self._make_values_list(ext_values, text_key, start_here)
        dupes = []
        for ext_value in ext_values:
            code, text = ext_value['value'], ext_value['text'][text_key]
            if code in codes or (text in texts and safe):
                dupes.append((code, text))
        if dupes:
            msg = 'Cannot add values since code and/or text already exists: {}'
            raise ValueError(msg.format(dupes))
        if use_array:
            self._meta['lib']['values'][name].extend(ext_values)
        else:
            self._meta['columns'][name]['values'].extend(ext_values)
        return None

    # array item manipulation
    # ------------------------------------------------------------------------

    @verify(variables={'name': 'masks'})
    def reorder_items(self, name, new_order):
        """
        Apply a new order to mask items.

        Parameters
        ----------
        name : str
            The variable name keyed in ``_meta['masks']``.
        new_order : list of int, default None
            The new order of the mask items. The included ints match up to
            the number of the items (``DataSet.item_no('item_name')``).

        Returns
        -------
        None
            DataSet is modified inplace.
        """
        sources = self.sources(name)
        s_ref = self._get_source_ref(name)
        org_i = OrderedDict([(self.item_no(s), ref)
                             for s, ref in zip(sources, s_ref)])
        if not set(org_i.keys()) == set(new_order):
            msg = "Only these item numbers are valid for 'new_order': {}"
            raise ValueError(msg.format(org_i.keys()))
        n_set = []
        n_items = []
        for i in new_order:
            ref = org_i[i]
            n_set.append(ref)
            for item in self._meta['masks'][name]['items']:
                if item['source'] == ref:
                    n_items.append(item)
        self._meta['masks'][name]['items'] = n_items
        self._meta['sets'][name]['items'] = n_set
        return None

    @modify(to_list='remove')
    @verify(variables={'name': 'masks'})
    def remove_items(self, name, remove):
        """
        Erase array mask items safely from both meta and case data components.

        Parameters
        ----------
        name : str
            The originating column variable name keyed in ``meta['masks']``.
        remove : int or list of int
            The items listed by their order number in the
            ``_meta['masks'][name]['items']`` object will be droped from the
            ``mask`` definition.

        Returns
        -------
        None
            DataSet is modified inplace.
        """
        items = self._get_itemmap(name, 'items')
        drop_item_names = [item for idx, item in enumerate(items, start=1)
                        if idx in remove]
        keep_item_idxs = [idx for idx, item in enumerate(items, start=1)
                          if idx not in remove]
        new_items = self._meta['masks'][name]['items']
        new_items = [item for idx, item in enumerate(new_items, start=1)
                     if idx in keep_item_idxs]
        self._meta['masks'][name]['items'] = new_items
        for drop_item_name in drop_item_names:
            self._data.drop(drop_item_name, axis=1, inplace=True)
            del self._meta['columns'][drop_item_name]
            col_ref = 'columns@{}'.format(drop_item_name)
            if col_ref in self._meta['sets']['data file']['items']:
                self._meta['sets']['data file']['items'].remove(col_ref)
            self._meta['sets'][name]['items'].remove(col_ref)
        return None

    @modify(to_list=['ext_items'])
    @verify(variables={'name': 'masks'}, text_keys='text_key')
    def extend_items(self, name, ext_items, text_key=None):
        """
        Extend mask items of an existing array.

        Parameters
        ----------
        name: str
            The originating column variable name keyed in ``meta['masks']``.
        ext_items: list of str/ list of dict
            The label of the new item. It can be provided as str, then the new
            column is named by the grid and the item_no, or as dict
            {'new_column': 'label'}.
        text_key: str/ list of str, default None
            Text key for text-based label information. Will automatically fall
            back to the instance's text_key property information if not provided.
        """
        if not text_key: text_key = self.text_key
        self.undimensionize()
        name = self._dims_free_arr_name(name)
        cat = self._has_categorical_data(name)
        source0 = self._meta['columns'][self.sources(name)[0]]
        for n_item in ext_items:
            if isinstance(n_item, dict):
                col = n_item.keys()[0]
                label = n_item.values()[0]
            else:
                col = '{}_{}'.format(name, len(self.sources(name))+1)
                label = n_item
            if self.var_exists(col):
                raise ValueError("Cannot add '{}', as it already exists.".format(col))
            # add column meta
            column = {'name':   col,
                      'text':   {text_key: ''},
                      'type':   source0['type'],
                      'parent': source0['parent'],
                      'properties': {'created': True}}
            if cat:
                column['values'] = source0['values']
            self._meta['columns'][col] = column
            # modify mask meta
            self._meta['masks'][name]['items'].append(
                {'properties': {'created': True},
                 'source':     'columns@{}'.format(col),
                 'text':       {text_key: ''}})
            self._meta['sets'][name]['items'].append('columns@{}'.format(col))
            self.set_variable_text(col, label, text_key)
            self._data[col] = '' if source0['type'] == 'delimited set' else np.NaN
        if self._dimensions_comp and not self._dimensions_comp == 'ignore':
            self.dimensionize()
        return None

    # rules and properties
    # ------------------------------------------------------------------------
    @modify(to_list='name')
    @verify(variables={'name': 'columns'}, axis='axis')
    def slicing(self, name, slicer, axis='y'):
        """
        Set or update ``rules[axis]['slicex']`` meta for the named column.

        Quantipy builds will respect the kept codes and *show them exclusively*
        in results.

        .. note:: This is not a replacement for ``DataSet.set_missings()`` as
            missing values are respected also in computations.

        Parameters
        ----------
        name : str or list of str
            The column variable(s) name keyed in ``_meta['columns']``.
        slice : int or list of int
            Values indicated by their ``int`` codes will be shown in
            ``Quantipy.View.dataframe``\s, respecting the provided order.
        axis : {'x', 'y'}, default 'y'
            The axis to slice the values on.

        Returns
        -------
        None
        """
        for n in name:
            if self._is_array_item(n):
                raise ValueError('Cannot slice on array items.')
            if 'rules' not in self._meta['columns'][n]:
                self._meta['columns'][n]['rules'] = {'x': {}, 'y': {}}
            if not isinstance(slicer, list): slicer = [slicer]
            sl = self._clean_codes_against_meta(n, slicer)
            rule_update = {'slicex': {'values': sl}}
            for ax in axis:
                self._meta['columns'][n]['rules'][ax].update(rule_update)
        return None

    def empty(self, name, condition=None):
        """
        Check variables for emptiness (opt. restricted by a condition).

        Parameters
        ----------
        name : (list of) str
            The mask variable name keyed in ``_meta['columns']``.
        condition : Quantipy logic expression, default None
            A logical condition expressed as Quantipy logic that determines
            which subset of the case data rows to be considered.

        Returns
        -------
        empty : bool
        """
        empty = []
        if not isinstance(name, list): name = [name]
        return_bool = len(name) == 1
        if condition:
            df = pd.DataFrame(self[self.take(condition), name])
        else:
            df = self._data
        for n in name:
            if df[n].count() == 0:
                empty.append(n)
        if return_bool:
            return bool(empty)
        else:
            return empty

    @modify(to_list='name')
    @verify(variables={'name': 'masks'})
    def empty_items(self, name, condition=None, by_name=True):
        """
        Test arrays for item emptiness (opt. restricted by a condition).

        Parameters
        ----------
        name : (list of) str
            The mask variable name keyed in ``_meta['masks']``.
        condition : Quantipy logic expression, default None
            A logical condition expressed as Quantipy logic that determines
            which subset of the case data rows to be considered.
        by_name : bool, default True
            Return array items by their name or their index.

        Returns
        -------
        empty : list
            The list of empty items by their source names or positional index
            (starting from 1!, mapped to their parent mask name if more than
            one).
        """
        empty = {}
        if condition:
            df = self[self.take(condition), name].copy()
        else:
            df = self._data.copy()
        for n in name:
            empty_items = [i for i in self.unroll(n)
                if df[i].value_counts().sum() == 0]
            if not by_name: empty_items = [self.item_no(i) for i in empty_items]
            if empty_items: empty[n] = empty_items
        if empty:
            return empty[name[0]] if len(name) == 1 else empty
        else:
            return None

    @verify(variables={'arrays': 'masks'})
    def hide_empty_items(self, condition=None, arrays=None):
        """
        Apply ``rules`` meta to automatically hide empty array items.

        Parameters
        ----------
        name : (list of) str, default None
            The array mask variable names keyed in ``_meta['masks']``. If not
            explicitly provided will test all array mask definitions.
        condition : Quantipy logic expression
            A logical condition expressed as Quantipy logic that determines
            which subset of the case data rows to be considered.

        Returns
        -------
        None
        """
        if not arrays: arrays = self.masks()
        if arrays and not isinstance(arrays, list): arrays = [arrays]
        empty_items = self.empty_items(arrays, condition, False)
        if empty_items:
            if isinstance(empty_items, list):
                empty_items = {arrays[0]: empty_items}
            for arr, items in empty_items.items():
                if len(items) == len(self.sources(arr)):
                    self.set_property(arr, '_no_valid_items', True, True)
                self.hiding(arr, items, axis='x', hide_values=False)
        return None

    def fully_hidden_arrays(self):
        """
        Get all array definitions that contain only hidden items.

        Returns
        -------
        hidden : list
            The list of array mask names.
        """
        hidden = []
        for m in self.masks():
            invalid = self.get_property(m, '_no_valid_items')
            if invalid: hidden.append(m)
        return hidden

    @modify(to_list='name')
    @verify(variables={'name': 'columns'}, axis='axis')
    def min_value_count(self, name, min=50, weight=None, condition=None,
                        axis='y', verbose=True):
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
        for v in name:
            df = self.crosstab(v, w=weight, text=False, f=condition)[v]['@'][v]
            hide = []
            for i, c in zip(df.index, df.values):
                if c < min:
                    hide.append(i)
            if hide:
                codes = self.codes(v)
                if verbose:
                    if 'All' in hide or all(c in hide for c in codes):
                        msg = '{}: All values have less counts than {}.'
                        print msg.format(v, min)
                    else:
                        print '{}: Hide values {}'.format(v, hide)
                hide = [h for h in hide if not h == 'All']
                self.hiding(v, hide, axis)
        return None

    @modify(to_list='name')
    @verify(variables={'name': 'both'}, axis='axis')
    def hiding(self, name, hide, axis='y', hide_values=True):
        """
        Set or update ``rules[axis]['dropx']`` meta for the named column.

        Quantipy builds will respect the hidden codes and *cut* them from
        results.

        .. note:: This is not equivalent to ``DataSet.set_missings()`` as
            missing values are respected also in computations.

        Parameters
        ----------
        name : str or list of str
            The column variable(s) name keyed in ``_meta['columns']``.
        hide : int or list of int
            Values indicated by their ``int`` codes will be dropped from
            ``Quantipy.View.dataframe``\s.
        axis : {'x', 'y'}, default 'y'
            The axis to drop the values from.
        hide_values : bool, default True
            Only considered if ``name`` refers to a mask. If True, values are
            hidden on all mask items. If False, mask items are hidden by position
            (only for array summaries).

        Returns
        -------
        None
        """
        for n in name:
            collection = 'columns' if not self.is_array(n) else 'masks'
            if 'rules' not in self._meta[collection][n]:
                self._meta[collection][n]['rules'] = {'x': {}, 'y': {}}
            if not isinstance(hide, list): hide = [hide]

            if collection == 'masks' and 'y' in axis and not hide_values:
                raise ValueError('Cannot hide mask items on y axis!')
            for ax in axis:
                if collection == 'masks' and ax == 'x' and not hide_values:
                    sources = self.sources(n)
                    h = [sources[idx-1]
                            for idx, s in enumerate(sources, start=1) if idx in hide]
                else:
                    h = self._clean_codes_against_meta(n, hide)
                    if set(h) == set(self._get_valuemap(n, 'codes')):
                        msg = "Cannot hide all values of '{}'' on '{}'-axis"
                        raise ValueError(msg.format(n, ax))
                if collection == 'masks' and ax == 'x' and hide_values:
                    for s in self.sources(n):
                        self.hiding(s, h, 'x')
                else:
                    rule_update = {'dropx': {'values': h}}
                    self._meta[collection][n]['rules'][ax].update(rule_update)
        return None

    @modify(to_list=['name', 'fix'])
    @verify(variables={'name': 'both'})
    def sorting(self, name, on='@', within=False, between=False, fix=None,
                ascending=False, sort_by_weight='auto'):
        """
        Set or update ``rules['x']['sortx']`` meta for the named column.

        Parameters
        ----------
        name : str or list of str
            The column variable(s) name keyed in ``_meta['columns']``.
        within : bool, default True
            Applies only to variables that have been aggregated by creating a
            an ``expand`` grouping / overcode-style ``View``:
            If True, will sort frequencies inside each group.
        between : bool, default True
            Applies only to variables that have been aggregated by creating a
            an ``expand`` grouping / overcode-style ``View``:
            If True, will sort group and regular code frequencies with regard
            to each other.
        fix : int or list of int, default None
            Values indicated by their ``int`` codes will be ignored in
            the sorting operation.
        ascending : bool, default False
            By default frequencies are sorted in descending order. Specify
            ``True`` to sort ascending.

        Returns
        -------
        None
        """
        for n in name:
            is_array = self.is_array(n)
            collection = 'masks' if is_array else 'columns'
            if on != '@' and not is_array:
                msg = "Column to sort on can only be changed for array summaries!"
                raise NotImplementedError(msg)
            if on == '@' and is_array:
                for source in self.sources(n):
                    self.sorting(source, fix=fix, within=within,
                                 between=between, ascending=ascending,
                                 sort_by_weight=sort_by_weight)
            else:
                if 'rules' not in self._meta[collection][n]:
                    self._meta[collection][n]['rules'] = {'x': {}, 'y': {}}
                if not is_array:
                    n_fix = self._clean_codes_against_meta(n, fix)
                else:
                    n_fix = self._clean_items_against_meta(n, fix)
                rule_update = {'ascending': ascending,
                               'within': within,
                               'between': between,
                               'fixed': n_fix,
                               'sort_on': on,
                               'with_weight': sort_by_weight}
                self._meta[collection][n]['rules']['x']['sortx'] = rule_update
        return None
    # ------------------------------------------------------------------------
    # derotate the dataset
    # ------------------------------------------------------------------------

    def _derotate_df(self, mapper, levels, other=None, dropna=True):
        """
        Returns derotated ``dataframe``.
        """
        data = self._data
        dfs = []
        level = levels.keys()[0]
        for question_group in mapper:
            new_var = question_group.keys()[0]
            q_group = question_group.values()[0]

            df = data[q_group]
            df = df.stack().reset_index([1])
            df.columns = [level, new_var]
            df[level] = df[level].map({el: ind for ind, el in enumerate(
                                           q_group, 1)})
            df.set_index([level], append=True, drop=True, inplace=True)
            dfs.append(df)

        new_df = pd.concat(dfs, axis=1)
        new_df = new_df.reset_index(1)

        new_df = new_df.join(data[other])

        new_df.index = list(xrange(0, len(new_df.index)))

        return new_df

    def _derotate_meta(self, mapper, other):
        """
        Returns derotated ``meta``.
        """
        meta = self._meta
        new_meta = self.start_meta(self.text_key)
        n = meta['info'].get('dataset', meta['info']).get('name')
        dname = '{}_derotate'.format(n)
        new_meta['info']['dataset'] = {'name': dname}
        for var in other:
            new_meta = self._assume_meta(new_meta, var, var)

        for question_group in mapper:
            new_var = question_group.keys()[0]
            old_var = question_group.values()[0][0]
            new_meta = self._assume_meta(new_meta, new_var, old_var)
        return new_meta

    def _assume_meta(self, new_meta, new_var, old_var):
        """
        Assumes meta information for variables to other meta object.
        """
        meta = self._meta
        n_masks = new_meta['masks']
        n_cols = new_meta['columns']
        n_sets = new_meta['sets']
        n_lib_v = new_meta['lib']['values']

        if self.is_array(old_var):
            n_masks[new_var] = org_copy.deepcopy(meta['masks'][old_var])
            n_masks[new_var]['name'] = new_var
            if self._has_categorical_data(old_var):
                n_lib_v[new_var] = meta['lib']['values'][old_var]
            n_sets[new_var] = org_copy.deepcopy(meta['sets'][old_var])
            n_sets['data file']['items'].append('masks@{}'.format(new_var))
            for var in self.sources(old_var):
                new_meta = self._assume_meta(new_meta, var, var)
        else:
            n_cols[new_var] = org_copy.deepcopy(meta['columns'][old_var])
            n_cols[new_var]['name'] = new_var
            if self._is_array_item(old_var):
                if not self._maskname_from_item(old_var) in new_meta['masks']:
                    n_cols[new_var]['parent'] = {}
                    n_cols[new_var]['values'] = self._get_value_loc(old_var)
                    n_sets['data file']['items'].append('columns@{}'.format(new_var))
            else:
                n_sets['data file']['items'].append('columns@{}'.format(new_var))

        return new_meta

    @modify(to_list='other')
    def derotate(self, levels, mapper, other=None, unique_key='identity',
                 dropna=True):
        """
        Derotate data and meta using the given mapper, and appending others.

        This function derotates data using the specification defined in
        mapper, which is a list of dicts of lists, describing how
        columns from data can be read as a heirarchical structure.

        Returns derotated DataSet instance and saves data and meta as json
        and csv.

        Parameters
        ----------
        levels : dict
            The name and values of a new column variable to identify cases.

        mapper : list of dicts of lists
            A list of dicts matching where the new column names are keys to
            to lists of source columns. Example:

            >>> mapper = [{'q14_1': ['q14_1_1', 'q14_1_2', 'q14_1_3']},
            ...           {'q14_2': ['q14_2_1', 'q14_2_2', 'q14_2_3']},
            ...           {'q14_3': ['q14_3_1', 'q14_3_2', 'q14_3_3']}]

        unique_key: str
            Name of column variable that will be copied to new dataset.

        other: list (optional; default=None)
            A list of additional columns from the source data to be appended
            to the end of the resulting stacked dataframe.

        dropna: boolean (optional; default=True)
            Passed through to the pandas.DataFrame.stack() operation.

        Returns
        -------
        new ``qp.DataSet`` instance
        """
        data = self._data
        meta = self._meta

        if not (isinstance(levels.values()[0], list) and isinstance(levels, dict)):
            raise ValueError('``levels`` must be a ``dict`` of ``lists``.')
        if not all(isinstance(e, dict) and isinstance(e.values()[0], list) and
                   isinstance(mapper, list) for e in mapper):
            msg = '``mapper`` must be ``list`` of ``dicts`` of ``lists``.'
            raise ValueError(msg)
        for q_group in mapper:
            if not len(levels.values()[0]) == len(q_group.values()[0]):
                raise ValueError('``lists`` of source ``columns`` and level '
                                 'variables must have same length.')
        level = levels.keys()[0]
        if other:
            exist_vars = [unique_key] + other + levels[level]
        else:
            exist_vars = [unique_key] + levels[level]
        for var in exist_vars:
            if not (var in meta['columns'] or var in meta['masks']):
                msg = "{} not found in dataset.".format(var)
                raise KeyError(msg)

        # derotated data
        add_cols = self.unroll(exist_vars)
        new_df = self._derotate_df(mapper, levels, add_cols, dropna)

        # new meta
        new_meta = self._derotate_meta(mapper, exist_vars)

        ds = DataSet('{}_derotated'.format(self.name))
        ds.from_components(new_df, new_meta)
        ds.path = self.path

        # some recodes/edits
        lev = ds._data[level]
        ds.add_meta(level, 'single', level, levels[level])
        ds._data[level] = lev

        ds.add_meta('{}_levelled'.format(level), 'single', level,
                    self.values(levels[level][0]))

        for x, lev in enumerate(levels[level], 1):
            rec = {y: {lev: y} for y in ds.codes('{}_levelled'.format(level))}
            ds.recode('{}_levelled'.format(level), rec, intersect={level: x})

        cols = (['@1', unique_key, level, '{}_levelled'.format(level)] +
                levels[level] + [new_var.keys()[0] for new_var in mapper] +
                self.unroll(other))
        ds._data = ds._data[cols]

        # save ``DataSet`` instance as json and csv
        path_json = os.path.join(ds.path, ''.join([ds.name, '.json']))
        path_csv = os.path.join(ds.path, ''.join([ds.name,  '.csv']))
        ds.write_quantipy(path_json, path_csv)

        return ds

    # ------------------------------------------------------------------------
    # DATA MANIPULATION/HANDLING
    # ------------------------------------------------------------------------

    def _logic_as_pd_expr(self, logic, prefix='default'):
        """
        """
        varname = '{}__logic_dummy__'.format(prefix).replace(' ', '_')
        category = [(1, 'select', logic)]
        meta = (varname, 'single', '', category)
        self.derive(*meta)
        return '{}==1'.format(varname)

    def make_dummy(self, var, partitioned=False):
        if not self.is_array(var):
            vartype = self._get_type(var)
            if vartype == 'delimited set':
                try:
                    dummy_data = self[var].str.get_dummies(';')
                except:
                    dummy_data = self._data[[var]]
                    dummy_data.columns = [0]
                if self.meta is not None:
                    var_codes = self._get_valuemap(var, non_mapped='codes')
                    dummy_data.columns = [int(col) for col in dummy_data.columns]
                    dummy_data = dummy_data.reindex(columns=var_codes)
                    dummy_data.replace(np.NaN, 0, inplace=True)
                if not self.meta:
                    dummy_data.sort_values(axis=1, inplace=True)
            else: # single, int, float data
                dummy_data = pd.get_dummies(self[var])
                if self.meta and not self._is_numeric(var):
                    var_codes = self._get_valuemap(var, non_mapped='codes')
                    dummy_data = dummy_data.reindex(columns=var_codes)
                    dummy_data.replace(np.NaN, 0, inplace=True)
                dummy_data.rename(
                    columns={
                        col: int(col)
                        if float(col).is_integer()
                        else col
                        for col in dummy_data.columns
                    },
                    inplace=True)
            if not partitioned:
                return dummy_data
            else:
                return dummy_data.values, dummy_data.columns.tolist()
        else: # array-type data
            items = self.sources(var)
            # items = self._get_itemmap(var, non_mapped='items')
            if self._has_categorical_data(var):
                codes = self._get_valuemap(var, non_mapped='codes')
            else:
                codes = []
                for i in items:
                    codes.extend(self._data[i].dropna().unique().tolist())
                codes = sorted(list(set(codes)))
            dummy_data = []
            if any(self[i].dtype == 'object' for i in items):
                for i in items:
                    try:
                        i_dummy = self[i].str.get_dummies(';')
                        i_dummy.columns = [int(col) for col in i_dummy.columns]
                        # dummy_data.append(i_dummy.reindex(columns=codes))
                    except:
                        i_dummy = self._data[[i]]
                        i_dummy.columns = [0]
                    dummy_data.append(i_dummy.reindex(columns=codes))
            else:
                for i in items:
                    if codes:
                        dummy_data.append(
                            pd.get_dummies(self[i]).reindex(columns=codes))
                    else:
                        dummy_data.append(pd.get_dummies(self[i]))
            dummy_data = pd.concat(dummy_data, axis=1)
            if not partitioned:
                return dummy_data
            else:
                return dummy_data.values, codes, items


    # ------------------------------------------------------------------------
    # BATCH HANDLERS
    # ------------------------------------------------------------------------

    @modify(to_list=['ci', 'weights', 'tests'])
    def add_batch(self, name, ci=['c', 'p'], weights=None, tests=None):
        return qp.Batch(self, name, ci, weights, tests)

    def get_batch(self, name=None):
        """
        Get existing Batch instance from DataSet meta information.

        Parameters
        ----------
        name: str
            Name of existing Batch instance.
        """
        if not name:
            return [qp.Batch(self, b) for b in self.batches]
        elif name in self.batches:
            return qp.Batch(self, name)
        else:
            err = "'{}' is not a valid batch.".format(name)
            logger.error(err); raise KeyError(name)

    @modify(to_list='batches')
    def populate(self, batches='all', verbose=True):
        """
        Create a ``qp.Stack`` based on all available ``qp.Batch`` definitions.

        Parameters
        ----------
        batches: str/ list of str
            Name(s) of ``qp.Batch`` instances that are used to populate the
            ``qp.Stack``.

        Returns
        -------
        qp.Stack
        """
        dk = self.name
        meta = self._meta
        data = self._data
        stack = Stack(name='aggregations', add_data={dk: (data, meta)})
        batches = stack._check_batches(dk, batches)
        for name in batches:
            batch = meta['sets']['batches'][name]
            xys = batch['x_y_map']
            fs = batch['x_filter_map']
            fy = batch['y_filter_map']
            my  = batch['yks']
            total_len = len(xys) + len(batch['y_on_y'])
            for idx, xy in enumerate(xys, start=1):
                x, y = xy
                if x == '@':
                    if fs[y[0]] is None:
                        fi = 'no_filter'
                    else:
                        fi = {fs[y[0]]: {fs[y[0]]: 0}}
                    stack.add_link(dk, fi, x='@', y=y)
                else:
                    if fs[x] is None:
                        fi = 'no_filter'
                    else:
                        fi = {fs[x]: {fs[x]: 0}}
                    stack.add_link(dk, fi, x=x, y=y)
                if verbose:
                    done = float(idx) / float(total_len) *100
                    print '\r',
                    time.sleep(0.01)
                    print  'Batch [{}]: {} %'.format(name, round(done, 1)),
                    sys.stdout.flush()
            for idx, y_on_y in enumerate(batch['y_on_y'], len(xys)+1):
                if fy[y_on_y] is None:
                    fi = 'no_filter'
                else:
                    fi = {fy[y_on_y]: {fy[y_on_y]: 1}}
                stack.add_link(dk, fi, x=my[1:], y=my)
                if verbose:
                    done = float(idx) / float(total_len) *100
                    print '\r',
                    time.sleep(0.01)
                    print  'Batch [{}]: {} %'.format(name, round(done, 1)),
                    sys.stdout.flush()
            if verbose:
                print '\n'
        return stack
