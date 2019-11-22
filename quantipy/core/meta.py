
from ..__imports__ import *  # noqa

logger = get_logger(__name__)


class Meta(dict):

    def __init__(self, json_dict={}):
        super(dict, self).__init__()
        if not json_dict:
            json_dict = self.start_meta()
        self.update(json_dict)
        self._repair_structure()

    def __getitem__(self, item):
        if isinstance(item, str) and "@" in item:
            item = item.split("@")
            i = item.pop(0)
            obj = dict.__getitem__(self, i)
            while item:
                i = item.pop(0)
                if isinstance(obj, list):
                    i = int(i)
                obj = obj[i]
        else:
            obj = dict.__getitem__(self, item)
        return obj

    def __contains__(self, name):
        return self.var_exists(name)

    # -------------------------------------------------------------------------
    # properties
    # -------------------------------------------------------------------------
    @property
    def text_key(self):
        return self["lib"].get("default text", "en-GB")

    @text_key.setter
    def text_key(self, tk):
        if tk not in self.valid_tks:
            err = "'{}' is not a valid textkey!".format(tk)
            logger.error(err); raise ValueError(err)
        self["lib"]["default text"] = tk

    @property
    def valid_tks(self):
        return self["lib"].get("valid text", VALID_TKS)

    @valid_tks.setter
    def valid_tks(self, valids):
        self["lib"]["valid text"] = ensure_list(valids)

    @property
    def dimensions_comp(self):
        return self["info"].get("dimensions_comp", False)

    @dimensions_comp.setter
    def dimensions_comp(self, value):
        self["info"]["dimensions_comp"] = value

    @property
    def dimensions_suffix(self):
        return self["info"].get("dimensions_suffix", "_grid")

    @dimensions_suffix.setter
    def dimensions_suffix(self, suffix):
        self["info"]["dimensions_suffix"] = suffix

    @property
    def columns(self):
        return list(self["columns"].keys())

    @property
    def masks(self):
        return list(self["masks"].keys())

    @property
    def sets(self):
        return list(self["sets"].keys())

    @property
    def singles(self):
        return self._get_columns("single")

    @property
    def delimited_sets(self):
        return self._get_columns("delimited set")

    @property
    def ints(self):
        return self._get_columns("int")

    @property
    def floats(self):
        return self._get_columns("float")

    @property
    def dates(self):
        return self._get_columns("date")

    @property
    def strings(self):
        return self._get_columns("string")

    def _get_columns(self, vtype):
        return [
            col for col in self.columns
            if self.get_type(col) == vtype]

    @property
    def filters(self):
        return self.by_property("recoded_filter")

    @property
    def hidden_arrays(self):
        return self.by_property("_no_valid_items")

    # -------------------------------------------------------------------------
    # i/o
    # -------------------------------------------------------------------------
    def clone(self):
        """
        Get a deep copy of the instance.
        """
        return copy.deepcopy(self)

    @params(is_var=["variables"], to_list=["variables"])
    def subset(self, variables=[], from_set=None, inplace=False):
        """
        Create a version of self with a reduced collection of variables.

        Parameters
        ----------
        variables : (list of) str
            Names of the variables to keep in the (new) instance.
        from_set : str
            The set name to base the (new) instance on.
        inplace : bool, default False
            *  True: Delete all vars that are not in "variables" or "from_set".
            *  False: Return a new (reduced) instance.
        """
        if not any([variables, from_set]):
            err = "Must either pass 'variables' or 'from_set'!"
            logger.error(err); raise ValueError(err)
        elif all([variables, from_set]):
            err = "Must either pass 'variables' or 'from_set', not both!"
            logger.error(err); raise ValueError(err)
        meta = self if inplace else self.clone()
        if variables:
            variables = meta.roll_up(variables)
        else:
            variables = meta.variables_from_set(from_set)
        for v in meta.variables():
            if v not in variables:
                meta.drop(v)
        if not inplace:
            return meta

    @classmethod
    def from_json(cls, path, reset=True):
        """
        Load json file into a new Meta instance.

        Parameters
        ----------
        path : str
            Full path, where the .json file is located.
        reset : bool, default True
            *  True: Custom sets and libs are removed.
        """
        meta = cls(load_json(path))
        if reset:
            meta._clean_custom_sets_and_libs()
        return meta

    def to_json(self, path, key=None, emulate_meta=False):
        """
        Save instance (or extract) into a json file.

        Parameters
        ----------
        path : str
            Full path, where the .json file is saved to.
        key : str
            Pointer like "collection@variable@values@index@..." to define
            extract.
        emulate_meta : bool
            *  True: Fill pointers with their dependent values.
        """
        obj = self[key] if key else self
        if emulate_meta:
            obj = self.emulate_meta(obj)
        save_json(obj, path)

    @classmethod
    def inferred_from_df(cls, df, text_key=None):
        """
        Create a new Meta instance derived by data information.

        Parameters
        ----------
        df : pd.DataFrame
            The data object.
        text_key : str, default None (== ``self.text_key``)
            Text key for label information.
        """
        meta = cls()
        if text_key:
            meta.text_key = text_key
        for name in df.columns:
            s = df[name]
            try:
                if all(s.dropna().astype(int) == s.dropna()):
                    pdtype = "int"
                else:
                    pdtype = str(s.dtype)
            except TypeError:
                pdtype = str(s.dtype)
            if "int" in pdtype:
                qptype = "int"
            elif "float" in pdtype:
                qptype = "float"
            else:
                qptype = "string"
            meta.add_meta(name, qptype, name)
        return meta

    # -------------------------------------------------------------------------
    # start meta
    # -------------------------------------------------------------------------
    def start_meta(self):
        """
        Get dict with basic meta structure.
        """
        meta = {
            "info": {
                "text": "",
                "dimensions_comp": False,
                "dimensions_suffix": "_grid"
            },
            "lib": {
                "default text": "en-GB",
                "valid text": VALID_TKS,
                "values": {}
            },
            "columns": {},
            "masks": {},
            "sets": {
                "data file": {
                    "text": {"en-GB": "Variable order in source file"},
                    "items": []
                },
                "batches": OrderedDict()
            },
            "type": "pandas.DataFrame"
        }
        return meta

    @params(text_key=["text_key"])
    def start_column(self, name, qtype, label, text_key=None, values=[],
                     parent=None, prop={}):
        """
        Get dict with basic column structure.

        Parameters
        ----------
        name : str
            The variable name.
        qtype : str
            The variable type.
        label : str
            The variable label.
        text_key : str, default None (== ``self.text_key``)
            Text key for label information.
        values : list
            A list of mappings with following structure:
            ``[{"value": code, "text": label}, ...]``
        parent : str, default None
            The name of a parent array (only for array items).
        prop : dict
            A mapping with the following structure:
            ``{"property name": "property value"}``
        """
        if parent:
            p_obj = {"masks@{}".format(parent): {"type": "array"}}
            if values:
                values = "lib@values@{}".format(parent)
        else:
            p_obj = {}
        column = {
            "text": {text_key: label},
            "type": qtype,
            "name": name,
            "parent": p_obj,
            "properties": prop
        }
        if values:
            column["values"] = values
        return column

    @params(text_key=["text_key"])
    def start_mask(self, name, subtype, label, text_key=None, prop={}):
        """
        Get dict with basic mask structure.

        Parameters
        ----------
        name : str
            The variable name.
        subtype : str
            The variable subtype.
        label : str
            The variable label.
        text_key : str, default None (== ``self.text_key``)
            Text key for label information.
        prop : dict
            A mapping with the following structure:
            ``{"property name": "property value"}``
        """
        mask = {
            "items": [],
            "type": "array",
            "subtype": subtype,
            "text": {text_key: label},
            "name": name,
            "properties": prop
        }
        if subtype in CATEGORICAL:
            mask["values"] = "lib@values@{}".format(name)
        return mask

    @params(text_key=["text_key"])
    def start_values(self, categories, text_key=None, start_at=1, safe=True):
        """
        Get dict with basic values structure.

        Parameters
        ----------
        categories : list of str, int, or tuples (int, str)
            A list with items with same type:
            *  all ints -> used as codes, labels are empty
            *  all strings -> used as labels, codes are created by enumeration
            *  all tuples -> ``[(code, label), ...]``
        text_key : str, default None (== ``self.text_key``)
            Text key for label information.
        start_at : int
            Starting number for code enumeration (all cats are strings).
        safe : bool, default True
            *  True: Duplicated texts will lead to an error raise.
        """
        all_int = all(isinstance(v, int) for v in categories)
        all_str = all(isinstance(v, str) for v in categories)
        all_tuple = all(isinstance(v, tuple) for v in categories)
        if not any([all_int, all_str, all_tuple]):
            err = (
                "All included categories must be of same type (int, str or "
                "tuples).")
            logger.error(err); raise TypeError(err)
        if all_int:
            safe = False
            values = [
                {"value": cat, "text": {text_key: ""}} for cat in categories]
        elif all_str:
            values = [
                {"value": idx, "text": {text_key: cat}}
                for idx, cat in enumerate(categories, start_at)]
        else:
            values = [
                {"value": code, "text": {text_key: lab}}
                for code, lab in categories]
        self._check_dupes_in_values(values, text_key, safe)
        return values

    @params(text_key=["text_key"])
    def start_item(self, item, text, text_key=None):
        """
        Get dict with basic item structure.

        Parameters
        ----------
        item : str
            Name of the item.
        text : str
            The item text.
        text_key : str, default None (== ``self.text_key``)
            Text key for label information.
        """
        return {
            "source": "columns@{}".format(item),
            "text": {text_key: text}}

    def emulate_meta(self, item):
        """
        Get full meta info. Pointers are replaced with their spotted location.
        """
        if isinstance(item, (list, tuple, set)):
            nitem = [self.emulate_meta(i) for i in item]
            if isinstance(item, tuple):
                nitem = tuple(nitem)
            if isinstance(item, set):
                nitem = set(nitem)
            return nitem
        elif isinstance(item, dict):
            for k, i in item.items():
                item[k] = self.emulate_meta(i)
            return item
        elif not isinstance(item, (float, int)) and "@" in item:
            item = self[item]
            item = self.emulate_meta(item)
            return item
        else:
            return item

    @params(to_list=["categories", "items"], text_key=["text_key"])
    def add_meta(self, name, qtype, label="", categories=[], items=[],
                 text_key=None, properties={}):
        """
        Add a new variable to the instance.

        Parameters
        ----------
        name : str
            The variable name.
        qtype : str
            The variable type (subtype for arrays).
        label : str, default ""
            The variable label.
        categories : list of str, int, or tuples in form of (int, str)
            A list with items of the same type:
            *  all ints -> used as codes, labels are empty
            *  all strings -> used as labels, codes are created by enumeration
            *  all tuples -> ``[(code, label), ...]``
        items : list of str or tuples (int, str)
            A list with items of the same type:
            *  all strings -> used as labels, number are created by enumeration
            *  all tuples -> ``[(number, label), ...]``
        text_key : str, default None == self.text_key
            Text key for text-based label information.
        """
        weak_dupes = self.get_weak_dupes(name)
        err = ""
        if name in BLACKLIST_VARIABLES:
            err = "Cannot add '{}', this name is blacklisted.".format(name)
        elif qtype not in QP_TYPES:
            err = "'{}' is not a valid type.".format(qtype)
        elif qtype in CATEGORICAL and not categories:
            err = "Expect categories for type '{}'".format(qtype)
        elif qtype == "delimited set" and len(categories) == 1:
            err = "Expect more than one category for delimited sets."
        elif qtype not in CATEGORICAL and categories:
            err = "None categorical type does not support categories."
        elif weak_dupes and not weak_dupes == name:
            err = "Cannot create '{}'. '{}' already exist".format(
                name, weak_dupes)
        elif items and len(items) == 1:
            err = "Cannot create an array with only one item."
        if err:
            logger.error(err); raise ValueError(err)

        for batch in self.batches:
            meta = Meta(self["sets"]["batches"][batch]["meta"])
            meta.add_meta(name, qtype, label, categories, items, text_key,
                          properties)

        if self.var_exists(name):
            msg = "Overwriting meta for '{}'".format(name)
            logger.info(msg)
            self.drop(name)

        if items:
            self._add_array(name, qtype, label, items, categories, text_key)
            return None

        values = self.start_values(categories, text_key) if categories else []
        column = self.start_column(name, qtype, label, text_key, values,
                                   prop={'created': True})
        self["columns"][name] = column
        if not name == "@1":
            self.extend_set(name)
        for k, v in properties.items():
            self.set_property(name, k, v)

    def _add_array(self, name, qtype, label, items, categories, text_key):
        """
        Add a new mask to the instance.
        """
        all_str = all(isinstance(v, str) for v in items)
        all_tuple = all(isinstance(v, tuple) for v in items)
        if not any([all_str, all_tuple]):
            err = "All included items must be of same type (str or tuples)."
            logger.error(err); raise ValueError(err)
        item_obj = []
        item_set = []
        for idx, item in enumerate(items, 1):
            if all_str:
                i = "{}_{}".format(name, idx)
                lab = item
            elif all_tuple:
                i = "{}_{}".format(name, item[0])
                lab = item[1]
            item_set.append(i)
            item_obj.append(self.start_item(i, lab, text_key))
            column_lab = '{} - {}'.format(label, lab)
            self["columns"][i] = self.start_column(
                i, qtype, column_lab, text_key, categories, name,
                {'created': True})
        self["masks"][name] = self.start_mask(name, qtype, label, text_key)
        self["masks"][name]["items"] = item_obj
        values = self.start_values(categories, text_key) if categories else []
        if values:
            self["lib"]["values"][name] = values
        self.create_set(name, item_set, overwrite=True)
        self.extend_set(name)
        if self.dimensions_comp:
            mapper = {}
            mapper[name] = self.dims_comp_array_name(name)
            for source in self.get_sources(name):
                mapper[source] = self.dims_comp_array_item_name(source, name)
            self._rename_from_mapper(self, mapper)

    @params(repeat=["name"])
    def drop(self, name, ignore_items=False):
        """
        Drop variable safely from meta dict.

        Parameters
        ----------
        name : str
            The variable name (mask or column).
        ignore_items: bool
            If False source variables for arrays in ``_meta["columns"]``
            are dropped, otherwise kept.
        """
        def remove_loop(obj, var):
            if isinstance(obj, dict):
                obj.pop(var, None)
                for key in obj:
                    remove_loop(obj[key], var)

        for b in self.batches:
            meta = Meta(self["sets"]["batches"][b]["meta"])
            meta.drop(name, ignore_items)

        if name == "@1":
            return None
        if self.is_array_item(name):
            err = "Cannot drop isolated array items!"
            logger.error(err); raise ValueError(err)

        data_file = self.get_set()[:]
        if self.is_array(name):
            m_ref = "masks@{}".format(name)
            if ignore_items:
                items = self.get_set(n)
                idx = data_file.index(m_ref)
                data_file = data_file[:idx] + items + data_file[idx + 1:]
                if self.is_categorical(name):
                    values = self["lib"]["values"][name][:]
                for source in self.get_sources(n):
                    if self.is_categorical(source):
                        self["columns"][source]["values"] = values
                    self["columns"][source]["parent"] = {}
            else:
                data_file.remove(m_ref)
        else:
            c_ref = "columns@{}".format(name)
            data_file.remove(c_ref)
        self["sets"]["data file"]["items"] = data_file
        remove_loop(self, name)

    @params(is_var=["name"])
    def copy(self, name, new_name=None, copy_only=None, copy_not=None):
        """
        Copy variable and all depending references.

        Parameters
        ----------
        name : str
            The variable name (mask or column)
        new_name : str, default None
            The new variable name, is None is provided, the initial name is
            extended by the suffix ``_rec``.
        copy_only: int or list of int, default None
            If provided, the copied version of the variable will only contain
            meta for the specified codes.
        copy_not: int or list of int, default None
            If provided, the copied version of the variable will contain
            meta for the all codes, except of the indicated.
        """
        if not new_name:
            new_name = "{}_rec".format(name)
        if copy_only and copy_not:
            err = "Must pass either 'copy_only' or 'copy_not', not both!"
            logger.error(err); raise ValueError(err)
        if self.is_array_item(name):
            err = "Cannot copy a single array item."
            logger.error(err); raise ValueError(err)
        if not self._verify_new_name(new_name):
            err = "Cannot create '{}'. Weak duplicates exist: {}"
            err = err.format(new_name, self.get_weak_dupes(new_name))
            logger.error(err); raise ValueError(err)
        dims_comp = self.dimensions_comp
        if dims_comp:
            self.undimensionize()
        mapper = {name: new_name}
        if self.is_array(name):
            sources, new_sources = self._new_array_items(name, new_name)
            mapper.update(dict(zip(sources, new_sources)))
            self["masks"][new_name] = self._rename_from_mapper(
                copy.deepcopy(self["masks"][name]), mapper)
            self["sets"][new_name] = self._rename_from_mapper(
                copy.deepcopy(self["sets"][name]), mapper)
            if self.is_categorical(name):
                self["lib"]["values"][new_name] = self._rename_from_mapper(
                    self["lib"]["values"][name][:], mapper)
            for source, new_source in zip(sources, new_sources):
                self["columns"][new_source] = self._rename_from_mapper(
                    copy.deepcopy(self["columns"][source]), mapper)
        else:
            self["columns"][new_name] = self._rename_from_mapper(
                copy.deepcopy(self["columns"][name]), mapper)
        self.extend_set(new_name)
        if self.is_categorical and (copy_only or copy_not):
            remove = [
                code for code in self.get_codes(new_name)
                if (code not in copy_only) or (code in copy_not)]
            self.remove_values(new_name, remove)
        if dims_comp:
            self.dimensionize()

    def merge(self, meta, overwrite=True):
        for tk in dataset.valid_tks:
            if tk not in self.valid_tks:
                self.valid_tks.append(tk)
        skip = []
        for v in meta.variables():
            try:
                if self.is_array(v):
                    self._merge_mask(v, meta, overwrite)
                else:
                    self._merge_column(v, meta["columns"][v], overwrite)
            except ValueError:
                skip.append(v)
        if skip:
            logger.info("Could not merge: {}".format(skip))
        return skip

    def _merge_mask(self, name, meta, overwrite):
        if name in self:
            mask = meta["masks"][name]
            self._merge_texts(
                self["masks"][name]["text"], mask["text"], overwrite)
            self._merge_properties(name, mask.get("properties", {}), overwrite)
            if self.is_categorical(name):
                self._merge_values(name, meta[mask["values"]], overwrite)
            self._merge_items(name, mask["items"], overwrite)
        else:
            if not self._verify_new_name(new_name):
                raise ValueError("Weak dupes")
            self["masks"][name] = meta["masks"][name]
            self["sets"][name] = meta["sets"][name]
            if self.is_categorical(name):
                self["lib"]["values"][name] = meta["lib"]["values"][name]
            for source in meta.get_sources(name):
                self["columns"][source] = meta["columns"][source]
            self.extend_set(name)

    def _merge_column(self, name, col, overwrite):
        if name in self:
            self._merge_texts(
                self["columns"][name]["text"], col["text"], overwrite)
            self._merge_properties(name, col.get("properties", {}), overwrite)
            if self.is_categorical(name) and not self.is_array_item(name):
                self._merge_values(name, col["values"], overwrite)
        else:
            if not self._verify_new_name(new_name):
                raise ValueError("Weak dupes")
            self["columns"][name] = col
            if not self.is_array_item(name):
                self.extend_set(name)

    # ------------------------------------------------------------------------
    # inspect
    # ------------------------------------------------------------------------
    def _check_type(self, name, checktype):
        if isinstance(name, list):
            return all(self._check_type(n, checktype) for n in name)
        if self.is_array(name):
            return self.get_subtype(name) in ensure_list(checktype)
        else:
            return self.get_type(name) in ensure_list(checktype)

    @params(is_var=["name"])
    def is_single(self, name):
        return self._check_type(name, "single")

    @params(is_var=["name"])
    def is_delimited_set(self, name):
        return self._check_type(name, "delimited set")

    @params(is_var=["name"])
    def is_int(self, name):
        return self._check_type(name, "int")

    @params(is_var=["name"])
    def is_float(self, name):
        return self._check_type(name, "float")

    @params(is_var=["name"])
    def is_string(self, name):
        return self._check_type(name, "string")

    @params(is_var=["name"])
    def is_date(self, name):
        return self._check_type(name, "date")

    @params(is_var=["name"])
    def is_array(self, name):
        return self.get_type(name) == "array"

    @params(is_var=["name"])
    def is_array_item(self, name):
        if self.is_array(name) or not self["columns"][name].get("parent"):
            return False
        else:
            return True

    @params(is_var=["name"])
    def is_numeric(self, name):
        return self._check_type(name, NUMERIC)

    @params(is_var=["name"])
    def is_filter(self, name):
        if self.get_property(name, "recoded_filter"):
            return True
        else:
            return False

    @params(is_var=["name"])
    def is_categorical(self, name):
        return self._check_type(name, CATEGORICAL)

    # -------------------------------------------------------------------------
    @params(to_list=["name"])
    def var_exists(self, name):
        variables = self.masks + self.columns
        return all(n in variables for n in name)

    @params(to_list=["blacklist"])
    def variables(self, setname="data file", numeric=True, string=True,
                  date=True, blacklist=[]):
        """
        View all variables listed in their global order.

        Parameters
        ----------
        setname : str, default "data file"
            The set name to query.
        numeric : bool, default True
            *  False: Skip numeric variables.
        string : bool, default True
            *  False: Skip string variables.
        date : bool, default True
            *  False: Skip date variables.
        blacklist : list, default None
            A list of variables names to exclude from the variable listing.

        Returns
        -------
        varlist : list
            The list of variables registered in the queried ``set``.
        """
        if not numeric:
            blacklist.extend(self.ints + self.floats)
        if not string:
            blacklist.extend(self.strings)
        if not date:
            blacklist.extend(self.dates)
        variables = self.variables_from_set(setname)
        return [var for var in variables if var not in blacklist]

    @params(is_var=["name"], to_list=["only_type"], text_key=["text_key"],
            axis=[("edits", "axis")])
    def describe(self, name=None, only_type=None, text_key=None, axis=None):
        """
        Inspect the instance's global or variable level structure.

        Parameters
        ----------
        name : str, default None
            The variable name (mask or column).
        only_type : (list of) str, default None
            Display specific types of global structure.
        text_key : str, default None == self.text_key
            Text key for text-based label information.
        axis : {"x", "y"}, default None
            Axis for text-based label information.
        """
        if name:
            return self._describe(name, text_key, axis)
        types = {qptype: [] for qptype in QP_TYPES}
        for var in self.variables():
            qptype = self.get_type(var)
            types[qptype].append(var)
        max_types = max(len(values) for values in types.values())
        for k, v in types.items():
            types[k] = self._pad_list(v, max_types)
        types = pd.DataFrame(types)
        if only_type:
            types = types[only_type]
            types = types.replace("", np.NaN).dropna(how="all")
        else:
            types = types[QP_TYPES]
        return types

    def _describe(self, name, text_key, axis):
        """
        Return the meta data of a variable in a well formated DataFrame.
        """
        if self.is_array(name):
            qtype = self.get_subtype(name)
        else:
            qtype = self.get_type(name)
        label = self.get_text(name, False, text_key, axis)
        if self.is_array(name) or self.is_categorical(name):
            codes = texts = []
            missings = []
            if self.is_categorical(name):
                codes = self.get_codes(name)
                texts = self.get_value_texts(name, text_key, axis)
                if self.get_missings(name):
                    for code in codes:
                        has_missing = False
                        for miss_t, miss_c in self.get_missings(name).items():
                            if code in miss_c:
                                missings.append(miss_t)
                                has_missing = True
                        if not has_missing:
                            missings.append(None)
            if self.is_array(name):
                sources = self.get_sources(name)
                item_t = self.get_item_texts(name)
                max_len = max(len(obj) for obj in [codes, sources])
                codes = self._pad_list(codes, max_len)
                texts = self._pad_list(texts, max_len)
                missings = self._pad_list(missings, max_len)
                sources = self._pad_list(sources, max_len)
                item_t = self._pad_list(item_t, max_len)
                elements = [items, items_texts, codes, texts, missings]
                columns = ["items", "item texts", "codes", "texts", "missing"]
            else:
                max_len = len(codes)
                missings = self._pad_list(missings, max_len)
                elements = [codes, texts, missings]
                columns = ["codes", "texts", "missing"]
            s = [pd.Series(el, index=range(0, max_len))
                 for el in elements]
            df = pd.concat(s, axis=1)
            df.columns = columns
            df.columns.name = qtype
            df.index.name = "{}: {}".format(name, label)
        else:
            df = pd.DataFrame(["N/A"])
            df.columns = [qtype]
            df.index = ["{}: {}".format(name, label)]
        return df

    def by_type(self, qtype=None):
        """
        View all variables that own the requested type.
        """
        return self.describe(only_type=qtype)

    def by_property(self, prop, **kwargs):
        """
        View all variables that own the requested property.

        Parameters
        ----------
        prop : str
            The property name.
        kwargs : dict
            kwargs for ``Meta.variables()``.
        """
        return [
            v for v in self.variables(**kwargs) if self.get_property(v, prop)]

    def variables_from_set(self, setname="data file"):
        items = self.get_set(setname)
        return self._dissect_setlist(items)

    # -------------------------------------------------------------------------
    # names
    # -------------------------------------------------------------------------
    @params(to_list=["str_tags"])
    def find(self, str_tags=None, suffixed=False):
        """
        Find variables by searching their names for substrings.

        Parameters
        ----------
        str_tags : (list of) str
            The strings tags to look for in the variable names.
            If not provided, the modules" default global list of substrings
            from VAR_SUFFIXES will be used.
        suffixed : bool, default False
            *  True: Only variable names that end with a given string sequence
            will qualify.
        """
        if not str_tags:
            str_tags = VAR_SUFFIXES
        found = []
        for v in self.variables():
            for str_tag in str_tags:
                if suffixed and v.endswith(str_tag):
                    found.append(v)
                elif str_tag in v:
                    found.append(v)
        return found

    def names(self, ignore_items=True, as_df=True):
        """
        Find all weak-duplicate variable names that are different only by case.

        Parameters
        ----------
        ignore_items : bool, default True
            *  True: array items are not taken into account.
        as_df : bool, default True
            *  True: Return weak duplicates in a pandas DataFrame.
            *  False: Return weak duplicates in an OrderedDict.

        Note
        ----
        Returns self.variables() if no weak-duplicates are found.
        """
        all_names = self.variables()
        if not ignore_items:
            all_names = self.unroll(all_names, both="all")
        lower_names = [n.lower() for n in all_names]
        weak_dupes = OrderedDict()
        for name in all_names:
            lower = name.lower()
            if lower_names.count(lower) > 1:
                if lower not in weak_dupes:
                    weak_dupes[lower] = [name]
                else:
                    weak_dupes[lower].append(name)
        if not weak_dupes:
            return all_names
        if not as_df:
            return weak_dupes
        else:
            max_dupes = max(len(values) for values in weak_dupes.values())
            for k, v in weak_dupes.items():
                weak_dupes[k] = _pad_list(v, max_dupes)
            return pd.DataFrame(weak_dupes)

    def valid_var_name(self, name, prefix=True, extend="_"):
        """
        Find a valid variable name. Check against weak_dupes, removes invalid
        chars, extends if necessary.

        Parameters
        ----------
        name : str
            The desired name.
        prefix : bool, default True
            If modifications are needed:
            *  True: a string is added as prefix (see extend)
            *  False: the name is enumerated (add number as suffix)
        extend : str, default "_"
            This string is used as prefix.
        """
        for char in INVALID_CHARS_IN_NAMES:
            name = name.replace(char, "")
        if self.get_weak_dupes(name) or name in BLACKLIST_VARIABLES:
            if prefix:
                valid = "{}{}".format(prefix, name)
                while self.get_weak_dupes(name):
                    name = "{}{}".format(prefix, name)
            else:
                suf = 2
                valid = "{}{}".format(name, suf)
                while self.get_weak_dupes(valid):
                    suf += 1
                    valid = "{}{}".format(name, suf)
                name = valid
        return name

    def _verify_new_name(self, new_name):
        new = self.valid_var_name(new_name)
        return new == new_name

    def get_weak_dupes(self, name):
        """
        Get all weak dupes (different spellings) of an input string.
        """
        resolved = []
        dupes = self.names(ignore_items=False, as_df=False)
        lower = OrderedDict([
            (v.lower(), v) for v in self.unroll(self.variables(), both="all")])
        if isinstance(dupes, dict) and name.lower() in dupes:
            resolved = dupes[name.lower()]
        elif name in self:
            resolved.append(name)
        elif name.lower() in lower:
            resolved.append(lower[name.lower()])
        return resolved[0] if len(resolved) == 1 else resolved

    @params(is_var=["name"])
    def rename(self, name, new_name, ignore_items=False):
        """
        Change the name of a variable in all keys and references.

        Parameters
        ----------
        name : str
            The variable name (mask or column).
        new_name : str
            The new variable name.
        ignore_items : bool, default False
            *  False : Array items inherit mask's name. Suffix is either the
            initial digit (``_x``) or the item_no.
        """
        if not self._verify_new_name(new_name):
            err = "Cannot rename '{}' into '{}'. Weak duplicates exist: {}"
            err = err.format(name, new_name, self.get_weak_dupes(new_name))
            logger.error(err); raise ValueError(err)
        dims_comp = self.dimensions_comp
        if dims_comp:
            self.undimensionize()
        mapper = {name: new_name}
        if self.is_array(name) and not ignore_items:
            sources, new_sources = self._new_array_items(name, new_name)
            mapper.update(dict(zip(sources, new_sources)))
        self._rename_from_mapper(self, mapper)
        if dims_comp:
            self.dimensionize()

    def _new_array_items(self, name, new_name):
        sources = self.get_sources(name)
        suffixes = [source.rsplit("_", 1)[-1] for source in sources]
        unique, dupes = dupes_and_unique(suffixes)
        if dupes or not all(u.isdigit() for u in unique):
            new_sources = [
                "{}_{}".format(new_name, self.get_item_no(source))
                for source in sources]
        else:
            new_sources = [
                "{}_{}".format(new_name, source.rsplit("_", 1)[-1])
                for source in sources]
        if not all(self._verify_new_name(n_s) for n_s in new_sources):
            invalids = [
                (s, n_s) for s, n_s in zip(sources, new_sources)
                if not self._verify_new_name(n_s)]
            err = "Cannot rename/ create, weak duplicates exist:\n {}"
            err = err.format(
                "\n".join([" -> ".join(invalid)] for invalid in invalids))
            logger.error(err); raise ValueError(err)
        return sources, new_sources

    @staticmethod
    def _rename_from_mapper(obj, mapper):
        if isinstance(obj, dict):
            for key in list(obj.keys()):
                k = Meta._rename_from_mapper(key, mapper)
                obj[k] = Meta._rename_from_mapper(obj.pop(key), mapper)
            return obj
        elif isinstance(obj, list):
            return [Meta._rename_from_mapper(item, mapper) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(
                Meta._rename_from_mapper(item, mapper) for item in obj)

        elif isinstance(obj, str) and not (obj == "@1" or "@" not in obj):
            new_obj = obj.split("@")
            if any(item in mapper for item in new_obj):
                for x, item in enumerate(new_obj):
                    if item in mapper:
                        new_obj[x] = mapper[item]
            return "@".join(new_obj)
        elif isinstance(obj, str):
            return mapper.get(obj, obj)
        else:
            return obj

    def dimensionize(self):
        """
        Rename the variables for Dimensions compatibility.
        """
        if self.dimensions_comp:
            err = "Instance is already dimensionized."
            logger.error(err); raise ValueError(err)

        self.dimensions_comp = True

        mapper = {}
        for mask in self.masks:
            mapper[mask] = self.dims_comp_array_name(mask)
            for source in self.get_sources(mask):
                mapper[source] = self.dims_comp_array_item_name(source, mask)
        self._rename_from_mapper(self, mapper)

    def undimensionize(self):
        """
        Rename the variables for Dimensions compatibility.
        """
        if not self.dimensions_comp:
            err = "Instance is already undimensionized."
            logger.error(err); raise ValueError(err)

        self.dimensions_comp = False

        mapper = {}
        for mask in self.masks:
            mapper[mask] = self.dims_comp_array_name(mask)
            for source in self.get_sources(mask):
                mapper[source] = self.dims_comp_array_item_name(source, mask)
        self._rename_from_mapper(self, mapper)

    @classmethod
    def dims_free_array_name(cls, name):
        return name.split(".")[0]

    @classmethod
    def dims_free_array_item_name(cls, name):
        pattern = '\[\{.*?\}\]\.'  # noqa
        found = re.search(pattern, name)
        if found:
            return found.group()[2: -3]
        else:
            return name

    @params(is_var=["name"])
    def dims_comp_array_name(self, name):
        name = self.dims_free_array_name(name)
        if self.dimensions_comp:
            return "{}.{}{}".format(name, name, self.dimensions_suffix)
        else:
            return name

    def dims_comp_array_item_name(self, name, parent=None):
        if not parent:
            parent = self.get_parent(name)
        name = self.dims_free_array_item_name(name)
        if self.dimensions_comp:
            return "{parent}[{{{name}}}].{parent}{suffix}".format(
                parent=parent, name=name, suffix=self.dimensions_suffix)
        else:
            return name

    # -------------------------------------------------------------------------
    # sets
    # -------------------------------------------------------------------------
    def get_set(self, setname="data file"):
        if setname not in self.sets:
            err = "'{}' is not a valid setname.".format(setname)
            logger.error(err); raise KeyError(err)
        return self["sets"][setname]["items"]

    @params(to_list=["include", "exclude"])
    def create_set(self, setname, include=[], exclude=[], overwrite=False):
        """
        Create a new set.

        Parameters
        ----------
        setname : str
            Name of the new set.
        included : (list of) str
            Names of the variables to be included in the new set.
            If None, all variables in ``data file`` are taken.
        excluded : (list of) str
            Names of the variables to be excluded in the new set.
        overwrite : bool, default False
            *  True: Overwrite existing set with same name.
            *  False: Raises if set with same name exists.
        """
        if setname in self.sets and not overwrite:
            err = "'{}' already exists.".format(setname)
            logger.error(err); raise ValueError(err)
        self["sets"][setname] = {"items": []}
        if not include:
            include = self.variables_from_set()
        for v in include:
            if v not in exclude:
                self.extend_set(v, setname)

    @params(is_var=["name"], repeat=["name"])
    def extend_set(self, name, setname="data file", idx=-1):
        collection = "masks" if self.is_array(name) else "columns"
        ref = "{}@{}".format(collection, name)
        if idx == -1:
            self["sets"][setname]["items"].append(ref)
        else:
            items = self.get_set(setname)
            nitems = items[:idx] + [ref] + items[idx + 1:]
            self["sets"][setname]["items"] = nitems

    @params(is_var=["name"], repeat=["name"])
    def _reduce_set(self, name, setname="data file"):
        collection = "masks" if self.is_array(name) else "columns"
        ref = "{}@{}".format(collection, name)
        try:
            self["sets"][setname]["items"].remove(ref)
        except ValueError:
            pass

    # -------------------------------------------------------------------------
    # types
    # -------------------------------------------------------------------------
    @params(is_var=["name"])
    def get_type(self, name):
        for collection in ["columns", "masks"]:
            if name in self[collection]:
                return self[collection][name]["type"]

    @params(is_mask=["name"])
    def get_subtype(self, name):
        return self["masks"][name]["subtype"]

    @params(is_var=["name"])
    def _set_type(self, name, qtype):
        if qtype not in QP_TYPES:
            err = "'{}' is not a valid variable type.".format(qtype)
            logger.error(err); raise ValueError(err)
        if self.is_array(name):
            self["masks"][name]["subtype"] = qtype
            for source in self.get_sources(name):
                self._set_type(source, qtype)
        else:
            self["columns"][name]["type"] = qtype

    # -------------------------------------------------------------------------
    # properties
    # -------------------------------------------------------------------------
    @params(is_var=["name"])
    def get_property(self, name, prop):
        if self.is_array(name):
            return self["masks"][name]["properties"].get(prop)
        else:
            return self["columns"][name]["properties"].get(prop)

    @params(is_var=["name"])
    def set_property(self, name, prop, value, ignore_items=False,
                     text_key=None):
        prop_dict = {prop: {text_key: value} if prop == "base_text" else value}
        if self.is_array(name):
            self["masks"][name]["properties"].update(prop_dict)
            if not ignore_items:
                for source in self.get_sources(name):
                    self.set_property(source, prop, value)
        else:
            self["columns"][name]["properties"].update(prop_dict)

    @params(is_var=["name"])
    def del_property(self, name, prop, ignore_items=False):
        if self.is_array(name):
            self["masks"][name]["properties"].pop(prop, None)
            if not ignore_items:
                for source in self.get_sources(name):
                    self.del_property(source, prop)
        else:
            self["columns"][name]["properties"].pop(prop, None)

    def _merge_properties(self, name, new_prop, overwrite):
        for prop, value in new_prop.items():
            has_prop = self.get_property(name, prop)
            if overwrite or not has_prop:
                self.set_property(name, prop, value)

    @params(is_var=["name"], text_key=["text_key"])
    def get_basetext(self, name, text_key=None):
        base_text = self.get_property(name, "base_text")
        return self._extract_text(base_text, text_key)

    # -------------------------------------------------------------------------
    # missings
    # -------------------------------------------------------------------------
    @params(is_var=["name"])
    def get_missings(self, name):
        if self.is_array(name):
            return self["columns"][self.get_sources(name)[0]].get("missings")
        else:
            return self["columns"][name].get("missings")

    @params(is_var=["name"], is_cat=["name"], repeat=["name"],
            to_list=["missings"])
    def set_missings(self, name, missings, flag="exclude", hide_on_y=True):
        """
        Flag category definitions for exclusion in aggregations.

        Parameters
        ----------
        name : str
            The variable name (column or mask).
        missings : (list of) int or str
            The codes or value labels to flag.
        flag : str, default "exclude" {"exclude", d.exclude}
            * "exclude": ignore flagged values globally
            * "d.exclude": ignore flagged values only in descriptive statistics
        hide_on_y : bool, default True
            * True: Hide flagged values in crossbreaks.
        """
        if self.is_array(name):
            for source in self.get_sources(name):
                self.set_missings(source, missings, flag, hide_on_y)
        else:
            if all(isinstance(missings, str)):
                missings = self.get_codes_from_label(name, missings)
            codes = self.get_codes(name)
            m_codes = [c for c in codes if c in missings]
            m_map = {flag: m_codes}
            self["columns"][name]["missings"] = m_map
            if hide_on_y:
                self.hiding(name, m_codes, 'y', True)

    @params(is_var=["name"], repeat=["name"])
    def del_missings(self, name):
        if self.is_array(name):
            for source in self.get_sources(name):
                self.del_missings(source)
        else:
            self["columns"][name].pop("missings", None)

    # -------------------------------------------------------------------------
    # rules
    # -------------------------------------------------------------------------
    @params(is_var=["name"], axis=["axis"])
    def get_rules(self, name, axis="x"):
        if self.is_array(name):
            rules = self["masks"][name].get("rules", {}).get(axis, {})
        else:
            rules = self["columns"][name].get("rules", {}).get(axis, {})
        return rules

    @params(is_var=["name"], axis=["axis"], repeat=["name", "axis"])
    def _set_rules(self, name, rule, axis="x"):
        collection = "masks" if self.is_array(name) else "columns"
        if "rules" not in self[collection][name]:
            self[collection][name]["rules"] = {"x": {}, "y": {}}
        self[collection][name]["rules"][axis].update(rule)

    @params(is_var=["name"], repeat=["name"], to_list=["fix"])
    def set_sorting(self, name, on='@', within=False, between=False, fix=[],
                    ascending=False, sort_by_weight='auto'):
        """
        Set or update ``rules['x']['sortx']`` meta for the named column.

        Parameters
        ----------
        name : str
            The variable name (column or mask).
        within : bool, default True
            Applies only to variables that have been aggregated by creating a
            an ``expand`` grouping / overcode-style ``View``:
            *  True: sort frequencies inside each group.
        between : bool, default True
            Applies only to variables that have been aggregated by creating a
            an ``expand`` grouping / overcode-style ``View``:
            *  True: sort group and regular code frequencies with regard
                to each other.
        fix : (list of) int/ str
            Codes/ items to ignore while sorting
        ascending : bool, default False
            By default frequencies are sorted in descending order. Specify
            ``True`` to sort ascending.
        sort_by_weight : str, default "auto"
            Variable name, which is used to weight the data.
        """
        if on == "@" and self.is_array(name):
            for source in self.get_sources(name):
                self.set_sorting(source, on, within, between, fix, ascending,
                                 sort_by_weight)
        else:
            if self.is_array(name):
                n_fix = [f for f in fix if f in self.get_sources(name)]
            else:
                n_fix = [f for f in fix if f in self.get_codes(name)]
            rule = {
                "sortx": {
                    "ascending": ascending,
                    "within": within,
                    "between": between,
                    "fixed": n_fix,
                    "sort_on": on,
                    "with_weight": sort_by_weight}}
            self._set_rules(name, rule, "x")

    @params(is_var=["name"], repeat=["name", "axis"], to_list=["hide"],
            axis=["axis"])
    def set_hiding(self, name, hide, axis="y", hide_values=True):
        """
        Set or update ``rules[axis]['dropx']`` meta for the named column.

        Parameters
        ----------
        name : str
            The variable name (column or mask).
        hide : (list of) int
            Codes/ items to hide.
        axis : {'x', 'y'}, default 'y'
            The axis to drop the values from.
        hide_values : bool, default True
            Only considered if ``name`` refers to a mask.
            *  True: values are hidden on all mask items
            *  False: mask items are hidden by item_no
        """
        if self.is_array(name):
            sources = self.get_sources(name)
            if not hide_values:
                if axis == "y":
                    err = "Cannot hide mask items on y axis!"
                    logger.error(err); raise ValueError(err)
                n_hide = [
                    source for idx, source in enumerate(sources, 1)
                    if idx in hide]
                if len(sources) == len(n_hide):
                    err = "Cannot hide all array items."
                    logger.error(err); raise ValueError(err)
            else:
                self.set_hiding(sources, hide, axis, hide_values)
                return None
        else:
            codes = self.get_codes(name)
            n_hide = [h for h in hide if h in codes]
            if len(codes) == len(n_hide):
                err = "Cannot hide all codes."
                logger.error(err); raise ValueError(err)
        rule = {"dropx": {"values": n_hide}}
        self._set_rules(name, rule, axis)

    @params(is_column=["name"], repeat=["name", "axis"], to_list=["slicer"],
            axis=["axis"])
    def set_slicing(self, name, slicer, axis="y"):
        """
        Set or update ``rules[axis]['slicex']`` meta for the named column.

        Parameters
        ----------
        name : str
            The variable name (column).
        slicer : (list of) int
            Codes to keep.
        axis : {'x', 'y'}, default 'y'
            The axis to keep the values.
        """
        if self.is_array_item(name):
            err = "Cannot slice on single array items."
            logger.error(err); raise ValueError(err)
        n_slice = [h for h in hide if h in codes]
        if len(n_slice) == 0:
            err = "Cannot hide all codes."
            logger.error(err); raise ValueError(err)
        rule = {"slicex": {"values": n_slice}}
        self._set_rules(name, rule, axis)

    # -------------------------------------------------------------------------
    # texts (variable labels)
    # -------------------------------------------------------------------------
    @params(is_var=["name"], text_key=["text_key"], axis=[("edits", "axis")])
    def get_text(self, name, shorten=True, text_key=None, axis=None):
        """
        Get text information of a variable.

        Parameters
        ----------
        name : str
            The variable name (mask or column).
        shorten : bool, default True
            * True: Get the short label for an array item.
        text_key : str, default None == self.text_key
            Text key for text-based label information.
        axis : str, default None {"x", "y"}
            Axis for text-based label information.
        """
        if self.is_array(name):
            text = self["masks"][name]["text"]
        elif self.is_array_item(name) and shorten:
            parent = self.get_parent(name)
            item_no = self.get_item_no(name)
            text = self["masks"][parent]["items"][item_no - 1]["text"]
        else:
            text = self["columns"][name]["text"]
        return self._extract_text(text, text_key, axis)

    @params(is_var=["name"], text_key=["text_key"], axis=[("edits", "axis")],
            repeat=["name", "text_key", "axis"])
    def set_text(self, name, text, text_key=None, axis=None):
        """
        Apply a new variable text.

        Parameters
        ----------
        name : str
            The variable name (mask or column).
        text : str
            The label to be set.
        text_key : str, default None == self.text_key
            Text key for text-based label information.
        axis: str, default None {'x', 'y', ['x', 'y']}
            If the ``new_text`` of the variable should only be considered temp.
            for build exports, the axes on that the edited text should appear
            can be provided.
        """
        if self.is_array(name):
            text_obj = self["masks"][name]["text"]
            self._update_text(text, text_obj, text_key, axis)
            for source in self.get_sources(name):
                text_obj = self["columns"][source]["text"]
                itext = self.get_text(source, True, text_key, axis)
                ftext = "{} - {}".format(text, itext)
                self._update_text(ftext, text_obj, text_key, axis)
        elif self.is_array_item(name):
            parent = self.get_parent(name)
            ptext = self.get_text(parent, False, text_key, axis)
            if text.startswith("{} - ".format(ptext)):
                ftext = text
                itext = text[(len(ptext) + 3):]
            else:
                itext = text
                ftext = "{} - {}".format(ptext, text)
            idx = self.get_item_no(name) - 1
            text_obj = self["masks"][parent]["items"][idx]["text"]
            self._update_text(itext, text_obj, text_key, axis)
            text_obj = self["columns"][name]["text"]
            self._update_text(ftext, text_obj, text_key, axis)
        else:
            text_obj = self["columns"][name]["text"]
            self._update_text(text, text_obj, text_key, axis)

    # -------------------------------------------------------------------------
    # values
    # -------------------------------------------------------------------------
    @params(is_var=["name"], is_cat=["name"], text_key=["text_key"],
            axis=[("edits", "axis")])
    def get_values(self, name, text_key=None, axis=None):
        """
        Get categorical data's paired code and texts information from the meta.

        Parameters
        ----------
        name : str
            The variable name (mask or column).
        text_key : str, default None == self.text_key
            Text key for text-based label information.
        axis : str, default None {"x", "y"}
            Axis for text-based label information.
        """
        values = self[self._get_value_ref(name)]
        return [
            (val["value"], self._extract_text(val["text"], text_key, axis))
            for val in values]

    @params(is_var=["name"])
    def _del_values(self, name):
        if self.is_array(name):
            self["masks"][name].pop("values", None)
            self["lib"]["values"].pop(name, None)
            for source in self.get_sources(name):
                self._del_values(source)
        elif not self.is_array_item(name):
            self["columns"][name].pop("values", None)

    @params(is_var=["name"], is_cat=["name"])
    def _set_values(self, name, values):
        if self.is_array(name):
            lib_ref = self._get_value_ref(name)
            self["masks"][name]["values"] = lib_ref
            self["lib"]["values"][name] = values
            for source in self.get_sources(name):
                self["columns"][source]["values"] = lib_ref
        elif not self.is_array_item(name):
            self["columns"][name]["values"] = values

    @params(is_var=["name"], is_cat=["name"], to_list=["new_values"],
            text_key=["text_key"], repeat=["name"])
    def extend_values(self, name, new_values, text_key=None, safe=True):
        """
        Extend the existing values object of a variable.

        Parameters
        ----------
        name : str
            The variable name (mask or column)
        new_values : (list of) str / int/ tuples (int, str)
            The values to add, all item list must be if the same type.
        text_key : str, default None == self.text_key
            Text key for text-based label information.
        safe : bool, default True
            *  True: Duplicated texts will lead to an error raise.
        """
        if self.is_array_item(name):
            err = "Cannot change values object of a single array item."
            logger.error(err); raise ValueError(err)
        start_at = max(self.get_codes(name)) + 1
        new_val_obj = self.start_values(new_values, text_key, start_at, safe)
        values = self[self._get_value_ref(name)][:]
        values.extend(new_val_obj)

        self._check_dupes_in_values(values, text_key, safe)
        self._set_values(name, values)

    @params(is_var=["name"], is_cat=["name"], to_list=["remove"],
            repeat=["name"])
    def remove_values(self, name, remove):
        """
        Remove values from the existing values object of a variable.

        Parameters
        ----------
        name : str
            The variable name (mask or column)
        remove : (list of) int
            The codes to remove from the values object.
        """
        if self.is_array_item(name):
            err = "Cannot change values object of a single array item."
            logger.error(err); raise ValueError(err)
        codes = self.get_codes(name)
        remove = [r for r in remove if r in codes]
        if set(remove) == set(codes):
            err = "Cannot remove all codes of a categorical variable."
            logger.error(err); raise ValueError(err)
        value_ref = self._get_value_ref(name)
        new_values = [
            value for value in self[value_ref] if value["value"] not in remove]
        self._set_values(name, new_values)

    @params(is_var=["name"], is_cat=["name"], to_list=["order"],
            repeat=["name"])
    def reorder_values(self, name, order):
        """
        Apply a new order to the value codes.

        Parameters
        ----------
        name : str
            The variable name (mask or column)
        order : (list of) int
            The order to apply to the values object.
        """
        if self.is_array_item(name):
            err = "Cannot change values object of a single array item."
            logger.error(err); raise ValueError(err)
        codes = self.get_codes(name)
        order = [r for r in order if r in codes]
        if not set(order) == set(codes):
            err = "The new order does not take all variable codes into account"
            logger.error(err); raise ValueError(err)
        value_ref = self._get_value_ref(name)
        new_values = [
            value for idx in order for value in self[value_ref]
            if value["value"] == idx]
        self._set_values(name, new_values)

    def _merge_values(self, name, new_values, overwrite):
        codes = self.get_codes(name)
        values = self[self._get_value_ref(name)]
        for value in new_values:
            if value["value"] in codes:
                for v in values:
                    if v["value"] == value["value"]:
                        self._merge_texts(v["text"], value["text"], overwrite)
            else:
                values.append(value)
        self._set_values(name, values)

    @params(is_var=["name"], is_cat=["name"])
    def _get_value_ref(self, name):
        if self.is_array(name):
            return "lib@values@{}".format(name)
        elif self.is_array_item(name):
            return "lib@values@{}".format(self.get_parent(name))
        else:
            return "columns@{}@values".format(name)

    @params(is_var=["name"], is_cat=["name"], text_key=["text_key"],
            axis=["axis"])
    def get_value_texts(self, name, text_key=None, axis=None):
        """
        Get categorical data's text information.

        Parameters
        ----------
        name : str
            The variable name (mask or column).
        text_key : str, default None == self.text_key
            Text key for text-based label information.
        axis : str, default None {"x", "y"}
            Axis for text-based label information.
        """
        values = self.get_values(name, text_key, axis)
        return [text for code, text in values]

    @params(is_var=["name"], is_cat=["name"], text_key=["text_key"],
            axis=[("edits", "axis")], to_list=["axis"])
    def set_value_texts(self, name, texts, text_key=None, axis=None):
        """
        Apply new value texts.

        Parameters
        ----------
        name : str
            The variable name (mask or column). lalalala
        texts : dict
            A mapping with following structure: `{code: new text}`
        text_key : str, default None == self.text_key
            Text key for text-based label information.
        axis: str, default None {'x', 'y', ['x', 'y']}
            If the ``new_text`` of the variable should only be considered temp.
            for build exports, the axes on that the edited text should appear
            can be provided.
        """
        if self.is_array_item(name):
            err = "Cannot modify values for single array items."
            logger.error(err); raise ValueError(err)
        values = self[self._get_value_ref(name)]
        new_values = []
        for value in values:
            code = value["value"]
            if code in list(texts.keys()):
                self._update_text(texts[code], value["text"], text_key, axis)
            new_values.append(value)
        self._set_values(name, new_values)

    @params(is_var=["name"], is_cat=["name"])
    def get_codes(self, name):
        """
        Get codes of categorical variables.

        Parameters
        ----------
        name : str
            The variable name (mask or column).
        """
        values = self.get_values(name)
        return [code for code, text in values]

    @params(is_var=["name"], is_cat=["name"], to_list=["text_label"],
            text_key=["text_key"], axis=[("edits", "axis")])
    def get_codes_from_label(self, name, text_label, text_key=None,
                             axis=None, exact=True, flat=True):
        """
        Return the code belonging to the passed ``text`` label (if present).

        Parameters
        ----------
        name : str
            The variable name (mask or column).
        text_label : (list of) str
            The value text(s) to search for.
        text_key : str, default None == self.text_key
            Text key for text-based label information.
        axis : str, default None {"x", "y"}
            Axis for text-based label information.
        exact : bool, default True
            * True: returns only codes with exact same label
            * False: return codes whose label contains ``text_label``
        flat : bool, default True
            If a list is passed for ``text_label``:
            *  True: return all found codes as a regular list.
            *  False: return a list of lists matching the order of text_label
        """
        values = self.get_values(name, text_key, axis)
        codes = []
        for text in text_label:
            sub_codes = []
            for code, label in values:
                if not label:
                    continue
                if (not exact and label in text) or label == text:
                    sub_codes.append(code)
            if flat:
                codes.extend(sub_codes)
            else:
                codes.append(sub_codes)
        if len(codes) == 1 and isinstance(codes[0], list):
            return codes[0]
        else:
            return codes

    @params(is_var=["name"], is_cat=["name"])
    def get_factors(self, name):
        """
        Get categorical data"s stat. factor values.

        Parameters
        ----------
        name : str
            The variable name (mask or column).

        Returns
        -------
        factors : OrderedDict
            A ``{value: factor}`` mapping.
        """
        values = self[self._get_value_ref(name)]
        return OrderedDict([
            (val["value"], val.get("factor"))
            for val in values if val.get("factor")])

    @params(is_var=["name"], is_cat=["name"], repeat=["name"])
    def set_factors(self, name, factors):
        """
        Apply numerical factors to (``single``-type categorical) variables.

        Factors can be read while aggregating descrp. stat. ``qp.Views``.

        Parameters
        ----------
        name : str
            The variable name (mask or column).
        factors : dict
            A mapping with following structure: `{code: factor}`
        """
        if self.is_single(name):
            values = self[self._get_value_ref(name)]
            for value in values:
                code = value["value"]
                if code in factors:
                    value["factor"] = factors[code]
            self._set_values(name, values)

    @params(is_var=["name"], is_cat=["name"], repeat=["name"])
    def del_factors(self, name):
        """
        Remove all factors set in the variable's ``'values'`` object.

        Parameters
        ----------
        name : str
            The variable name (mask or column).
        """
        values = self[self._get_value_ref(name)]
        for value in values:
            value.pop("factor", None)

    # -------------------------------------------------------------------------
    # items
    # -------------------------------------------------------------------------
    @params(is_mask=["name"], text_key=["text_key"], axis=[("edits", "axis")])
    def get_items(self, name, text_key=None, axis=None):
        """
        Get the array"s paired item names and texts information from the meta.

        Parameters
        ----------
        name : str
            The variable name keyed in masks.
        text_key : str, default None == self.text_key
            Text key for text-based label information.
        axis : {"x", "y"}, default None
            If provided the text_key is taken from the x/y edits dict.
        """
        if not self.is_array(name):
            err = "'{}' is not an array.".format(name)
            logger.error(err); raise TypeError(err)
        items = self.variables_from_set(name)
        return [
            (item, self.get_text(item, True, text_key, axis))
            for item in items]

    @params(is_column=["name"])
    def get_item_no(self, name):
        """
        Return the position/ index of passed array item variable name.

        Parameters
        ----------
        name : str
            The variable name keyed in columns.
        """
        if not self.is_array_item(name):
            err = "'{}'' is not an array item.".format(name)
            logger.error(err); raise TypeError(err)
        items = self.variables_from_set(self.get_parent(name))
        return items.index(name) + 1

    @params(is_mask=["name"], text_key=["text_key"], axis=[("edits", "axis")])
    def get_item_texts(self, name, text_key=None, axis=None):
        """
        Get the ``text`` meta data for the items of the passed array mask name.

        Parameters
        ----------
        name : str
            The variable name keyed in columns or masks.
        text_key : str, default None == self.text_key
            Text key for text-based label information.
        axis : {"x", "y"}, default None
            If provided the text_key is taken from the x/ y edits dict.
        """
        items = self.get_items(name, text_key, axis)
        return [text for item, text in items]

    @params(is_mask=["name"], text_key=["text_key"], axis=[("edits", "axis")],
            repeat=["name", "text_key", "axis"])
    def set_item_texts(self, name, texts, text_key=None, axis=None):
        """
        Apply new item texts.

        Parameters
        ----------
        name : str
            The variable name (mask).
        texts : dict
            A mapping with following structure: `{item_no: new text}`
        text_key : str, default None == self.text_key
            Text key for text-based label information.
        axis: str, default None {'x', 'y', ['x', 'y']}
            If the new texts of the variable should only be considered temp.
            for build exports, the axes on that the edited text should appear
            can be provided.
        """
        for source in self.get_sources(name):
            item_no = self.get_item_no(source)
            if item_no in texts:
                self.set_text(source, texts[item_no], text_key, axis)

    @params(is_mask=["name"], repeat=["name"])
    def reorder_items(self, name, order):
        """
        Apply a new order to mask items.

        Parameters
        ----------
        name : str
            The variable name (mask).
        order : list of int
            The new order of the mask items. The included ints match up to
            the number of the items (``self.get_item_no('item_name')``).
        """
        sources = self.get_sources(name)
        s_mapping = dict(enumerate(sources, 1))
        if not set(order) == set(s_mapping.keys()):
            err = "The new order does not take all item numbers into account"
            logger.error(err); raise ValueError(err)
        new_sources = [s_mapping[no] for no in order]
        items = [
            item
            for new_source in new_sources
            for item in self["masks"][name]["items"]
            if item["source"] == "columns@{}".format(new_source)]
        self["masks"][name]["items"] = items
        self.create_set(name, new_sources, overwrite=True)

    @params(is_mask=["name"], repeat=["name"], text_key=["text_key"])
    def extend_items(self, name, new_items, text_key=None):
        """
        Create new item variables for an existing mask.

        Parameters
        ----------
        name : str
            The variable name (mask)
        new_items : (list of str) or tuples (int, str)
            A list with items of the same type:
            *  all strings -> used as labels, number are created by enumeration
            *  all tuples -> ``[(number, label), ...]``
        text_key : str, default None == self.text_key
            Text key for text-based label information.
        """
        all_str = all(isinstance(v, str) for v in new_items)
        all_tuple = all(isinstance(v, tuple) for v in new_items)
        if not any([all_str, all_tuple]):
            err = "All included items must be of same type (str or tuples)."
            logger.error(err); raise ValueError(err)
        item_obj = []
        new_items = []
        for idx, item in enumerate(new_items, 1):
            if all_str:
                i = "{}_{}".format(name, idx)
                lab = item
            elif all_tuple:
                i = "{}_{}".format(name, item[0])
                lab = item[1]
            i = self.dims_comp_array_item_name(i, name)
            item_obj.append(self.start_item(i, lab, text_key))
            column_lab = '{} - {}'.format(label, lab)
            new_items.append(i, column_lab)
        if not all(self._verify_new_name(n_i[0]) for n_i in new_items):
            invalids = [
                n_i[0] for n_i in new_items
                if not self._verify_new_name(n_i[0])]
            err = "Cannot create items, weak duplicates exist:\n {}".format(
                invalids)
            logger.error(err); raise ValueError(err)
        self["masks"][name]["items"].extend(item_obj)
        for n_i in new_items:
            self["columns"][n_i[0]] = self.start_column(
                n_i[0], self.get_subtype(name), n_i[1], text_key,
                self.is_categorical(name), name, {'created': True})
            self.extend_set(name, n_i[0])

    @params(is_mask=["name"], repeat=["name"], to_list=["remove"])
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
        r_items = [
            source for source in sources if self.get_item_no(source) in remove]
        if set(r_items) == set(sources):
            err = "Cannot remove all items or an array."
            logger.error(err); raise ValueError(err)
        items = [
            item for no, item in enumerate(self["masks"][name]["items"], 1)
            if no not in remove]
        self["masks"][name]["items"] = items
        self._reduce_set(r_items)

    def _merge_items(self, name, meta, overwrite):
        for item in meta["masks"][name]["items"]:
            source = item["source"].split("@")[-1]
            if source in self:
                for i in self["masks"][name]["items"]:
                    if source == i["source"].split("@")[-1]:
                        self._merge_texts(i["text"], item["text"], overwrite)
            else:
                self["masks"][name]["items"].append(item)
                self.extend_set(source, name)
            self._merge_column(source, meta["columns"][source], overwrite)

    @params(is_mask=["name"])
    def get_sources(self, name):
        """
        Get the ``columns`` elements for the passed array mask name.

        Parameters
        ----------
        name : str
            The variable name (mask).
        """
        return self.variables_from_set(name)

    @params(is_column=["name"])
    def get_parent(self, name):
        if not self.is_array_item(name):
            return None
        else:
            parents = list(self["columns"][name]["parent"].keys())
            return parents[0].split("@")[-1]

    @params(is_column=["variables"])
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
        if not self._verify_new_name(name):
            err = "Cannot add '{}'. Weak duplicates exist: {}"
            err = err.format(name, name, self.get_weak_dupes(name))
            logger.error(err); raise ValueError(err)
        name = self.dims_free_array_name(name)
        variables = uniquify_list(variables)

        if self.is_categorical(variables[0]):
            all_values = uniquify_list([
                self[self._get_value_ref(var)] for var in variables])
            if len(all_values) > 1:
                err = "Cannot combine variables to array with different values"
                logger.error(err); raise ValueError(err)
            else:
                values = all_values[0]
            if any(self.is_delimited_set(var) for var in variables):
                qtype = "delimited set"
            else:
                qtype = "single"
        else:
            values = False
            qtype = self.get_type(variables[0])
        texts = []
        items = []
        parent = {'masks@{}'.format(name): {'type': 'array'}}
        for var in variables:
            text = self.get_text(var)
            texts.append(text)
            items.append(self.start_item(var, text))
            self["columns"][var]["parent"] = parent
        self["masks"][name] = self.start_mask(name, qtype, label)
        self["masks"][name]["items"] = items
        self.create_set(name, variables)
        idx = self.variables_from_set().index(variables[0])
        self.extend_set(name, idx=idx)
        self._reduce_set(variables)
        if values:
            self._set_values(name, values)
        self.set_item_texts(name, texts)
        if self.dimensions_comp:
            mapper = {}
            mapper[name] = self.dims_comp_array_name(name)
            for source in self.get_sources(name):
                mapper[source] = self.dims_comp_array_item_name(source, name)
            self._rename_from_mapper(self, mapper)

    @params(repeat=["name"], is_mask=["name"])
    def unbind(self, name):
        """
        Remove mask-structure for arrays.
        """
        self.drop(name, ignore_items=True)

    # -------------------------------------------------------------------------
    # batches
    # -------------------------------------------------------------------------
    @property
    def batches(self):
        return list(self["sets"].get("batches", {}).keys())

    def get_batches(self, main=True, add=True):
        """
        View all names of included batches, depending if they are main or add.
        """
        batches = []
        for b in self.batches:
            if main and self.main_batch(b):
                batches.append(b)
            if add and not self.main_batch(b):
                batches.append(b)
        return batches

    def main_batch(self, batchname):
        return not self["sets"]["batches"][batchname]["mains"]

    def adds_per_mains(self, reverse=False):
        """
        Return a dictionary that maps all additional batches to the main batch.
        """
        apm = {
            name: bdef["additions"]
            for name, bdef in self["sets"]["batches"].items()
            if self.main_batch(name)
        }
        if reverse:
            rev = {}
            for name in self.batches(main=False, add=True):
                rev[name] = []
                for main, adds in apm.items():
                    if name in adds:
                        rev[name].append(main)
            return rev
        else:
            return apm

    # -------------------------------------------------------------------------
    # texts and text_keys
    # -------------------------------------------------------------------------
    @classmethod
    def _merge_texts(cls, text_obj, new_texts, overwrite):
        if overwrite:
            text_obj.update(new_texts)
        else:
            for tk, text in new_texts.items():
                if tk not in text_obj:
                    text_obj[tk] = text

    @classmethod
    def _extract_text(cls, text_obj, text_key, axis=None):
        if axis:
            return text_obj.get(axis, {}).get(text_key, "")
        else:
            return text_obj.get(text_key, "")

    @classmethod
    def _update_text(cls, text, text_obj, text_key, axis=None):
        if isinstance(axis, list):
            for ax in axis:
                cls._update_text(text, text_obj, text_key, ax)
        if axis:
            if axis not in text_obj:
                text_obj[axis] = {}
            text_obj[axis][text_key] = text
        else:
            text_obj[text_key] = text

    @staticmethod
    def _apply_to_texts(func, obj, kwargs={}):
        """
        Running recursively through the meta dict and apply func.
        """
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "text" and isinstance(v, dict):
                    func(v, **kwargs)
                elif k not in ["sets", "ddf"]:
                    Meta._apply_to_texts(func, v, kwargs)
        elif isinstance(obj, list):
            for i in obj:
                Meta._apply_to_texts(func, i, kwargs)

    @staticmethod
    def _used_text_keys(obj, tks, ignore=["x edits", "y edits"]):
        new = [tk for tk in obj.keys() if tk not in ignore + tks['tks']]
        tks["tks"].extend(new)

    def used_text_keys(self):
        """
        Get a list of all used textkeys in the dataset instance.
        """
        func = self._used_text_keys
        kwargs = {'tks': {'tks': []}}
        Meta._apply_to_texts(func, self, kwargs)
        return kwargs['tks']['tks']

    @staticmethod
    def _force_texts(obj, copy_to, copy_from, update_existing):
        new_text_key = None
        for new_tk in reversed(copy_from):
            if new_tk in obj.keys():
                if new_tk in ['x edits', 'y edits']:
                    if obj[new_tk].get(copy_to):
                        new_text_key = new_tk
                else:
                    new_text_key = new_tk
        if not new_text_key:
            raise ValueError('{} is no existing text_key'.format(copy_from))
        if not obj.get(copy_to) or update_existing:
            if new_text_key in ['x edits', 'y edits']:
                text = obj[new_text_key][copy_to]
            else:
                text = obj[new_text_key]
            obj.update({copy_to: text})

    @params(to_list=['copy_from'], text_key=["copy_to", "copy_from"])
    def force_texts(self, copy_to=None, copy_from="all",
                    update_existing=False):
        """
        Copy info from existing text_key to a new or update the existing.

        Parameters
        ----------
        copy_to : str, {self.valid_tks}
            None -> self.text_key
            The text key that will be filled.
        copy_from : str / list {self.valid_tks}
            You can also enter a list with text_keys, if the first text_key
            doesn't exist, it takes the next one
        update_existing : bool
            True : copy_to will be filled in any case
            False: copy_to will be filled if it's empty/ not existing
        """
        copy_from.append(copy_to)
        func = self._force_texts
        kwargs = {
            'copy_to': copy_to,
            'copy_from': copy_from,
            'update_existing': update_existing}
        Meta._apply_to_texts(func, self, kwargs)

    @staticmethod
    def _select_text_keys(obj, text_key):
        if not any(tk in obj for tk in text_key):
            msg = 'Cannot select {}. A variable does not contain any of it.'
            raise ValueError(msg.format(text_key))
        for tk in list(obj.keys()):
            if tk not in ['x edits', 'y edits']:
                if tk not in text_key:
                    obj.pop(tk)
            else:
                for etk in list(obj[tk].keys()):
                    if etk not in text_key:
                        obj[tk].pop(etk)

    @params(to_list=['text_key'], text_key=["text_key"])
    def select_text_keys(self, text_key="all"):
        """
        Cycle through all meta ``text`` objects keep only selected text_key.

        Parameters
        ----------
        text_key : str / list of str, default None {self.valid_tks}
            The text_keys which should be kept.
        """
        text_key = text_key or self.valid_tks
        func = self._select_text_keys
        kwargs = {'text_key': text_key}
        Meta._apply_to_texts(func, self, kwargs)

    @staticmethod
    def _remove_html(obj):
        htmls = ['_', '**', '*']
        for tk, text in obj.items():
            if tk not in ['x edits', 'y edits']:
                for html in htmls:
                    text = text.replace(html, '')
                text = text.replace('<br/>', '\n')
                remove = re.compile('<.*?>')
                text = re.sub(remove, '', text)
                remove = '(<|\$)(.|\n)+?(>|.raw |.raw)'  # noqa
                obj[tk] = re.sub(remove, '', text)
            else:
                for etk, etext in obj[tk].items():
                    for html in htmls:
                        etext = etext.replace(html, '')
                    remove = re.compile('<.*?>')
                    etext = re.sub(remove, '', etext)
                    remove = '(<|\$)(.|\n)+?(>|.raw |.raw)'  # noqa
                    obj[tk][etk] = re.sub(remove, '', etext)

    def remove_html(self):
        """
        Cycle through all meta ``text`` objects removing html tags.
        """
        func = self._remove_html
        Meta._apply_to_texts(func, self)

    @staticmethod
    def _replace_from_dict(obj, replace_map, text_key):
        for tk, text in obj.items():
            if tk in text_key:
                for k, v in replace_map.items():
                    obj[tk] = obj[tk].replace(k, v)
            elif tk in ['x edits', 'y edits']:
                for etk, etext in obj[tk].items():
                    if etk in text_key:

                        for k, v in replace_map.items():
                            obj[tk][etk] = obj[tk][etk].replace(k, v)

    @params(to_list=['text_key'], text_key=["text_key"])
    def replace_texts(self, replace, text_key="all"):
        """
        Cycle through all meta ``text`` objects replacing unwanted strings.

        Parameters
        ----------
        replace : dict, default None
            A dictionary mapping {unwanted string: replacement string}.
        text_key : str / list of str, default None {self.valid_tks}
            The text_keys for which unwanted strings are replaced.
        """
        func = self._replace_from_dict
        kwargs = {'replace_map': replace, 'text_key': text_key}
        Meta._apply_to_texts(func, self, kwargs)

    @staticmethod
    def _repair_text_edits(obj, text_key):
        for ax in ['x edits', 'y edits']:
            if not isinstance(obj.get(ax, {}), dict):
                obj[ax] = {tk: obj[ax] for tk in obj.keys() if tk in text_key}

    @params(to_list=['text_key'], text_key=["text_key"])
    def repair_text_edits(self, text_key="all"):
        """
        Cycle through all meta ``text`` objects repairing axis edits.

        Parameters
        ----------
        text_key : str / list of str, default None {self.valid_tks}
            The text_keys for which text edits should be included.
        """
        func = self._repair_text_edits
        kwargs = {'text_key': text_key}
        Meta._apply_to_texts(func, self, kwargs)

    # -------------------------------------------------------------------------
    # misc
    # -------------------------------------------------------------------------
    @params(to_list=["varlist", "keep", "both"], is_var=["varlist"])
    def unroll(self, varlist, keep=None, both=None):
        """
        Replace mask with its items, optionally excluding/keeping certain ones.

        Parameters
        ----------
        varlist : list
           A list of meta ``"columns"`` and/or ``"masks"`` names.
        keep : str or list, default None
            The names of masks that will not be replaced with their items.
        both : "all", str or list of str, default None
            The names of masks that will be included both as themselves and as
            collections of their items.

        Note
        ----
        varlist can also contain nesting `var1 > var2`. The variables which are
        included in the nesting can also be controlled by keep and both, even
        if the variables are also included as a "normal" variable.

        Example::
            meta.unroll(varlist = ["q1", "q1 > gender"], both="all")
            ["q1",
             "q1_1",
             "q1_2",
             "q1 > gender",
             "q1_1 > gender",
             "q1_2 > gender"]

        Returns
        -------
        unrolled : list
            The modified ``varlist``.
        """
        if both and both[0] == "all":
            both = self.masks
        unrolled = []
        for var in varlist:
            if " > " in var:
                nested = var.replace(" ", "").split(">")
                n_list = []
                for n in nested:
                    if n in keep:
                        to_keep = "mask"
                    elif n in both:
                        to_keep = "both"
                    else:
                        to_keep = "items"
                    n_list.append(self._array_and_item_list(n, to_keep))
                for ur in [" > ".join(list(un)) for un in product(*n_list)]:
                    if ur not in unrolled:
                        unrolled.append(ur)
            else:
                if var in keep:
                    to_keep = "mask"
                elif var in both:
                    to_keep = "both"
                else:
                    to_keep = "items"
                for ur in self._array_and_item_list(var, to_keep):
                    if ur not in unrolled:
                        unrolled.append(ur)
        return unrolled

    @params(to_list=["varlist", "ignore_arrays"], is_var=["varlist"])
    def roll_up(self, varlist, ignore_arrays=None):
        """
        Replace any array items with its parent mask variable definition name.

        Parameters
        ----------
        varlist : list
           A list of meta ``"columns"`` and/or ``"masks"`` names.
        ignore_arrays : (list of) str
            A list of array mask names that should not be rolled up if their
            items are found inside ``varlist``.

        Note
        ----
        varlist can also contain nesting `var1 > var2`. The variables which are
        included in the nesting can also be controlled by keep and both, even
        if the variables are also included as a "normal" variable.

        Returns
        -------
        rolled_up : list
            The modified ``varlist``.
        """
        def _var_to_keep(var, ignore):
            if self.is_array(var):
                to_keep = "mask"
            else:
                to_keep = "items"
                if self.is_array_item(var):
                    parent = self.get_parent(var)
                    if parent not in ignore_arrays:
                        var = parent
                        to_keep = "mask"
            return var, to_keep

        rolled_up = []
        for var in varlist:
            if " > " in var:
                nested = var.replace(" ", "").split(">")
                n_list = []
                for n in nested:
                    n, to_keep = _var_to_keep(n, ignore_arrays)
                    n_list.append(self._array_and_item_list(n, to_keep))
                for ru in [" > ".join(list(un)) for un in product(*n_list)]:
                    if ru not in rolled_up:
                        rolled_up.append(ru)
            else:
                var, to_keep = _var_to_keep(var, ignore_arrays)
                for ru in self._array_and_item_list(var, to_keep):
                    if ru not in rolled_up:
                        rolled_up.append(ru)
        return rolled_up

    # -------------------------------------------------------------------------
    # repair
    # -------------------------------------------------------------------------
    def _clean_custom_sets_and_libs(self, clean_batches=False):
        if clean_batches:
            valid_set = self.masks + ["data file"]
        else:
            valid_set = self.masks + ["data file", "batches"]
        valid_lib = ["default text", "valid text", "values"]
        for mset in self.sets:
            if mset not in valid_set:
                self["sets"].pop(mset)
        for mlib in self["lib"].keys():
            if mlib not in valid_lib:
                self["lib"].pop(mlib)

    def _repair_structure(self):
        parents = {}
        for mask in self.masks:
            # verify mask name
            self["masks"][mask]["name"] = mask
            # verify properties
            if "properties" not in self["masks"][mask]:
                self["masks"][mask]["properties"] = {}
            # verify mask set
            sources = [item["source"] for item in self["masks"][mask]["items"]]
            if not self["sets"][mask]["items"] == sources:
                self["sets"][mask]["items"] = sources
                logger.info("Fixed set for '{}'".format(mask))
            items = self._dissect_setlist(sources)
            parents.update({i: mask for i in items})
            # verify mask subtype
            if "subtype" not in self["masks"][mask]:
                self["masks"][mask]["subtype"] = self.get_type(items[0])
                logger.info("Fixed subtype for '{}'".format(mask))
            # verify mask values
            if self.is_categorical(mask):
                lib_ref = self._get_value_ref(mask)
                if not isinstance(self["masks"][mask]["values"], str):
                    self["masks"][mask]["values"] = lib_ref
                    self["lib"]["values"][mask] = self[lib_ref]
                    logger.info("Fixed value ref for '{}'".format(mask))
            else:
                self._del_values(mask)

        for col in self.columns:
            # verify column name
            self["columns"][col]["name"] = col
            # verify properties
            if "properties" not in self["columns"][col]:
                self["columns"][col]["properties"] = {}
            if col in parents:
                # verify parent and values for array items
                mask = parents[col]
                parent_def = {'masks@{}'.format(mask): {'type': 'array'}}
                self["columns"][col]["parent"] = parent_def
                if self.is_categorical(col):
                    lib_def = self._get_value_ref(mask)
                    self["columns"][col]["values"] = lib_def
            else:
                # verify parent and values for non-array items
                self["columns"][col]["parent"] = {}
                if self.is_categorical(col):
                    values = self["columns"][col]["values"]
                    if isinstance(values, str):
                        self["columns"][col]["values"] = self[values]
                else:
                    self._del_values(col)

        # verify data file set
        data_file = []
        for var in self.variables_from_set():
            if self.var_exists(var):
                if self.is_array_item(var):
                    data_file.append(self.parent(var))
                else:
                    data_file.append(var)
        self.create_set(
            "data file", include=uniquify_list(data_file), overwrite=True)

        # verify text edits
        self.repair_text_edits()

    # -------------------------------------------------------------------------
    # helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _pad_list(pad_list, pad_to_len):
        to_add = pad_to_len - len(pad_list)
        pad_list.extend([""] * to_add)
        return pad_list

    @params(text_key=["text_key"])
    def _check_dupes_in_values(self, values, text_key=None, safe=True):
        c_dupes = dupes_in_list([value["value"] for value in values])
        if c_dupes:
            err = "Cannot add duplicated codes: {}".format(c_dupes)
            logger.error(err); raise ValueError(err)
        if safe:
            t_dupes = dupes_in_list([
                self._extract_text(value["text"], text_key)
                for value in values])
            if t_dupes:
                err = "Cannot add duplicated texts: {}".format(t_dupes)
                logger.error(err); raise ValueError(err)

    @params(to_list=["items"])
    def _dissect_setlist(self, items, collection=False, name=True):
        new_list = []
        for item in items:
            c, n = item.split("@")
            if collection and name:
                new_list.append((c, n))
            elif collection:
                new_list.append(c)
            elif name:
                new_list.append(n)
        return new_list

    def _array_and_item_list(self, v, keep):
        new_list = []
        if not self.is_array(v):
            # columns
            if keep in ["both", "items"]:
                new_list.append(v)
        else:
            # masks
            if keep in ["both", "mask"]:
                new_list.append(v)
            if keep in ["both", "items"]:
                new_list.extend(self.get_sources(v))
        return new_list
