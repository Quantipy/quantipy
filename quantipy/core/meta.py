
from ..__imports__ import *  # noqa


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

    @property
    def text_key(self):
        return self["lib"].get("default text", "en-GB")

    @text_key.setter
    def text_key(self, tk):
        if tk not in self.valid_tks:
            err = "'{}' is not a valid textkey!".format(tk)
            logger.error(err); ValueError(err)
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
        if not isinstance(value, bool):
            err = "'dimensions_comp' must be of type bool."
            logger.error(err); raise TypeError(err)
        self["info"]["dimensions_comp"] = value

    @property
    def dimensions_suffix(self):
        return self["info"].get("dimensions_suffix", "_grid")

    @dimensions_suffix.setter
    def dimensions_suffix(self, suffix):
        if not isinstance(suffix, str):
            err = "'dimensions_suffix' must be of type str."
            logger.error(err); raise TypeError(err)
        self["info"]["dimensions_suffix"] = suffix

    def start_meta(self):
        """
        Fills meta dict with basic structure.
        """
        meta = {
            "info": {
                "text": "",
                "dimensions_comp": self.dimensions_comp,
                "dimensions_suffix": self.dimensions_suffix
            },
            "lib": {
                "default text": self.text_key,
                "valid text": self.valid_tks,
                "values": {}
            },
            "columns": {},
            "masks": {},
            "sets": {
                "data file": {
                    "text": {self.text_key: "Variable order in source file"},
                    "items": []
                }
            },
            "type": "pandas.DataFrame"
        }
        return meta

    def start_column(self, name, qtype, label, text_key=None, values=[],
                     parent=None, prop={}):
        text_key = text_key or self.text_key
        if parent:
            p_obj = {"masks@{}".format(parent) : {"type": "array"}}
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

    def start_mask(self, name, subtype, label, text_key=None, prop={}):
        text_key = text_key or self.text_key
        mask = {
            "items": [],
            "type": "array",
            "subtype": subtype,
            "text": {text_key: label},
            "name": array_name,
            "properties": prop
        }
        if subtype in CATEGORICAL:
            mask["values"] = "lib@values@{}".format(name)
        return mask

    def start_values(self, categories, text_key=None, start_at=1):
        all_int = all(isinstance(v, int) for v in categories)
        all_str = all(isinstance(v, str) for v in categories)
        all_tuple = all(isinstance(v, tuple) for v in categories)
        if not any([all_int, all_str, all_tuple]):
            err = (
                "All included categories must be of same type (int, str or "
                "tuples).")
            logger.error(err); raise TypeError(err)
        text_key = text_key or self.text_key
        if all_int:
            values = [
                {"value": cat, "text": {text_key: ""}} for cat in categories]
        elif all_str:
            values = [
                {"value": idx, "text": {text_key: cat}}
                for idx, cat in enumerate(categories, start_at)]
        else:
            codes = [cat[0] for cat in categories]
            if not len(codes) == len(set(codes)):
                err = "Cannot add duplicated codes."
                logger.error(err); raise ValueError(err)
            values = [
                {"value": code, "text": {text_key: lab}}
                for code, lab in categories]
        return values

    def start_item(self, item, text, text_key=None):
        text_key = text_key or self.text_key
        return {
            "source": "columns@{}".format(item),
            "text": {text_key: text}}

    def emulate_meta(self, item):
        if isinstance(item, (list, tuple, set)):
            for x, i in enumerate(item):
                item[x] = self.emulate_meta(i)
            return item
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

    def clone(self):
        """
        Get a deep copy of the ``Meta`` instance.
        """
        return copy.deepcopy(self)

    @classmethod
    def from_json(cls, path, reset=True):
        """
        Load json file into a Meta instance.
        """
        meta = cls(load_json(path))
        if reset:
            meta._clean_custom_sets_and_libs()
        return meta

    def to_json(self, path):
        """
        Save instance into a json file.
        """
        save_json(self, path)

    @classmethod
    def inferred_from_df(cls, df, text_key=None):
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

    def valid_var_name(self, name, prefix=True, extend="_"):
        """
        Find a valid variable name. Check against weak_dupes, removes invalid
        chars, extends if necessary.

        Parameters
        ----------
        name : str
            The desired name, which is checked.
        prefix : bool, default True
            If modifications are needed, a string is added as prefix (see
            extend) or the name is enumerated.
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

    @modify(to_list=["categories", "items"])
    def add_meta(self, name, qtype, label="", categories=[], items=[],
                 text_key=None):
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

        if self.var_exists(name):
            msg = "Overwriting meta for '{}'".format(name)
            logger.info(msg)
            self.drop(name)

        text_key = text_key or self.text_key

        if items:
            self._add_array(name, qtype, label, items, categories, text_key)
            return None

        values = self.start_values(categories, text_key) if categories else []
        column = self.start_column(name, qtype, label, text_key, values,
                                   prop={'created': True})
        self["columns"][name] = column
        self.extend_set(name)

    def _add_array(name, qtype, label, items, categories, text_key):
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
            item_set.append("columns@{}".format(i))
            item_obj.append(self.start_item(i, lab, text_key))
            column_lab = '{} - {}'.format(label, lab)
            self["columns"][i] = self.start_column(
                i, qtype, column_lab, text_key, True, name, {'created': True})
        self["masks"][name] = self.start_mask(name, qtype, label, text_key)
        values = self.start_values(categories, text_key) if categories else []
        if values:
            self["lib"]["values"][name] = values
        self.create_set(name, item_set)
        self.extend_set(name)

    @modify(to_list=["include", "exclude"])
    def create_set(self, setname, include=[], exclude=[], overwrite=False):
        """
        Create a new set in ``self['sets']``.

        Parameters
        ----------
        setname : str
            Name of the new set.
        included : str or list/set/tuple of str
            Names of the variables to be included in the new set. If None all
            variables in ``data file`` are taken.
        excluded : str or list/set/tuple of str
            Names of the variables to be excluded in the new set.
        overwrite : bool, default False
            Overwrite if setname already exist.
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

    @modify(to_list=["variables"])
    def subset(self, variables=[], from_set=None, inplace=False):
        """
        Create a version of self with a reduced collection of variables.

        Parameters
        ----------
        variables : str or list of str, default None
            A list of variable names to include in the new DataSet instance.
        from_set : str
            The name of an already existing set to base the new Meta on.
        inplace : bool, default False
            Return a new instance or modify the instance inplace.
        """
        err = None
        if not any([variables, from_set]):
            err = "Must either pass 'variables' or 'from_set'!"
        elif all([variables, from_set]):
            err = "Must either pass 'variables' or 'from_set', not both!"
        if err:
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

    # -------------------------------------------------------------------------
    # properties
    # -------------------------------------------------------------------------
    @property
    def columns(self):
        return self["columns"].keys()

    @property
    def masks(self):
        return self["masks"].keys()

    @property
    def sets(self):
        return self["sets"].keys()

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

    def _get_columns(self, vtype=None):
        if vtype:
            return [
                col for col in self.columns
                if self.get_type(col) == vtype]
        else:
            return self.columns

    @property
    def filters(self):
        return self.by_property("recoded_filter")

    # ------------------------------------------------------------------------
    # is ...
    # ------------------------------------------------------------------------
    @modify(to_list=["checktype"])
    def _check_type(self, name, checktype):
        if self.is_array(name):
            return self.get_subtype(name) in [checktype]
        else:
            return self.get_type(name) in [checktype]

    def is_single(self, name):
        return self._check_type(name, "single")

    def is_delimited_set(self, name):
        return self._check_type(name, "delimited set")

    def is_int(self, name):
        return self._check_type(name, "int")

    def is_float(self, name):
        return self._check_type(name, "float")

    def is_string(self, name):
        return self._check_type(name, "string")

    def is_date(self, name):
        return self._check_type(name, "date")

    def is_array(self, name):
        return self.get_type(name) == "array"

    def is_array_item(self, name):
        if self.is_array or not self["columns"][name].get("parent"):
            return False
        else:
            return True

    def is_numeric(self, name):
        return self._check_type(name, NUMERIC)

    def is_filter(self, name):
        if self.get_property(name, "recoded_filter"):
            return True
        else:
            return False

    def is_categorical(self, name):
        return self._check_type(name, CATEGORICAL)

    # -------------------------------------------------------------------------
    @modify(to_list=["name"])
    def var_exists(self, name):
        variables = self.variables()
        return all(n in variables for n in name)

    @modify(to_list=["blacklist"])
    def variables(self, setname="data file", numeric=True, string=True,
                  date=True, boolean=True, blacklist=[]):
        """
        View all variables listed in their global order.

        Parameters
        ----------
        setname : str, default "data file"
            The name of the variable set to query. Defaults to the main
            variable collection stored via "data file".
        numeric : bool, default True
            Include ``int`` and ``float`` type variables?
        string : bool, default True
            Include ``string`` type variables?
        date : bool, default True
            Include ``date`` type variables?
        boolean : bool, default True
            Include ``boolean`` type variables?
        blacklist : list, default None
            A list of variables names to exclude from the variable listing.

        Returns
        -------
        varlist : list
            The list of variables registered in the queried ``set``.
        """
        varlist = []
        except_list = []
        dsvars = self.variables_from_set(setname)
        if not numeric:
            except_list.extend(NUMERIC)
        if not string:
            except_list.extend(STRING)
        if not date:
            except_list.extend(DATE)
        if not boolean:
            except_list.extend(BOOLEAN)
        for dsvar in dsvars:
            vtype = self.get_type(dsvar)
            if not (vtype in except_list or dsvar in blacklist):
                varlist.append(dsvar)
        return varlist

    @modify(to_list=["only_type"])
    def describe(self, name=None, only_type=None, text_key=None, axis=None):
        """
        Inspect the DataSet"s global or variable level structure.
        """
        text_key = text_key or self.text_key
        if name:
            return self._get_meta(var, text_key, axis)
        types = {qptype: [] for qptype in QP_TYPES}
        for var in self.variables():
            qptype = self.get_type(var)
            types[qptype].append(var)
        max_types = max(len(values) for values in types.values())
        for k, v in types.items():
            types[k] = _pad_list(v, max_types)
        types = pd.DataFrame(types)
        if only_type:
            types = types[only_type]
            types = types.replace("", np.NaN).dropna(how="all")
        else:
            types = types[QP_TYPES]
        return types

    def _describe(self, name, text_key=None, axis=None):
        """
        Return the meta data of a variable in a well formated DataFrame.
        """
        text_key = text_key or self.text_key
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
                texts = self.get_value_texts(text_key, axis)
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
                codes = self._pad_meta_list(codes, max_len)
                texts = self._pad_meta_list(texts, max_len)
                missings = self._pad_meta_list(missings, max_len)
                sources = self._pad_meta_list(sources, max_len)
                item_t = self._pad_meta_list(item_t, max_len)
                elements = [items, items_texts, codes, texts, missings]
                columns = ["items", "item texts", "codes", "texts", "missing"]
            else:
                max_len = len(codes)
                elements = [codes, texts, missings]
                columns = ["codes", "texts", "missing"]
            s = [pd.Series(el, index=range(0, idx_len)) for el in elements]
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
        View all DataSet variables that own the requested type.
        """
        return self.describe(only_type=qtype)

    def by_property(self, prop, **kwargs):
        """
        View all variables that own the requested property.

        Note:
            The list can be additionally filtered by adding kwargs for
            ``Meta.variables()``.
        """
        return [
            v for v in self.variables(**kwargs) if self.get_property(v, prop)]

    @modify(to_list=["str_tags"])
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
            If set to True, only variable names that end with a given string
            sequence will qualify.
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

        .. note:: Will return self.variables() if no weak-duplicates are found.
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

    def variables_from_set(self, setname="data file"):
        items = self.get_set(setname)
        return self._dissect_setlist(items)

    def get_type(self, name):
        for collection in ["columns", "masks"]:
            if name in self[collection]:
                return self[collection][name]["type"]

    def get_subtype(self, name):
        if self.is_array(name):
            return self["masks"][name]["subtype"]

    def get_property(self, name, prop):
        if self.is_array(name):
            return self["masks"][name]["properties"].get(prop)
        else:
            return self["columns"][name]["properties"].get(prop)

    def get_basetext(self, name, text_key=None):
        bt = self.get_property(name, "base_text")
        if bt:
            if not text_key:
                text_key = self.text_key
            return bt.get(text_key)

    def get_missings(self, name):
        if self.is_array(name):
            name = self.sources(name)[0]
        return self["columns"][name].get("missings")

    def get_rules(self, name, axis="x"):
        if self.is_array(name):
            rules = self["masks"][name].get("rules", {}).get(axis, {})
        else:
            rules = self["columns"][name].get("rules", {}).get(axis, {})
        return rules

    def get_set(self, setname="data file"):
        if setname not in self.sets:
            err = "'{}' is not a valid setname.".format(setname)
            logger.error(err); raise KeyError(err)
        return self["sets"][setname]["items"]

    def get_weak_dupes(self, name):
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

    def get_text(self, name, shorten=True, text_key=None, axis=None):
        """
        Get text information of a variable.

        Parameters
        ----------
        name : str
            The variable name keyed in columns or masks.
        shorten : bool
            If True, the short label for array items is taken.
        text_key : str, default None
            The text_key that should be used when taking the label.
        axis : {"x", "y"}, default None
            If provided the text_key is taken from the x/ y edits dict.
        """
        text_key = text_key or self.text_key
        if self.is_array(name):
            text = self["masks"][name]["text"]
        elif self.is_array_item(name) and shorten:
            parent = self.get_parent(name)
            item_no = self.get_item_no(name)
            text = self["masks"][parent]["items"][item_no-1]["text"]
        else:
            text = self["columns"][name]["text"]
        return self._extract_text(text, text_key, axis)

    def get_values(self, name, text_key=None, axis=None):
        """
        Get categorical data"s paired code and texts information from the meta.

        Parameters
        ----------
        name : str
            The variable name keyed in columns or masks.
        text_key : str, default None
            The text_key that should be used when taking labels.
        axis : {"x", "y"}, default None
            If provided the text_key is taken from the x/y edits dict.
        """
        if not self.is_categorical(name):
            err = "'{}' is not categorical.".format(name)
            logger.error(err); raise TypeError(err)
        text_key = text_key or self.text_key
        if self.is_array(name):
            values = self["masks"][name]["values"]
        else:
            values = self["columns"][name]["values"]
        values = self.emulate_meta(values)
        return [
            (val["value"], self._extract_text(val["text"], text_key, axis))
            for val in values]

    def get_value_texts(self, name, text_key=None, axis=None):
        """
        Get categorical data"s text information.

        Parameters
        ----------
        name : str
            The variable name keyed in columns or masks.
        text_key : str, default None
            The text_key that should be used when taking labels.
        axis : {"x", "y"}, default None
            If provided the text_key is taken from the x/ y edits dict.
        """
        values = self.get_values(name, text_key, axis)
        return [text for code, text in values]

    def get_codes(self, name):
        """
        Get categorical data"s text information.

        Parameters
        ----------
        name : str
            The variable name keyed in columns or masks.
        """
        values = self.get_values(name)
        return [code for code, text in values]

    @modify(to_list="text_label")
    def get_codes_from_label(self, name, text_label, text_key=None,
                             axis=None, exact=True, flat=True):
        """
        Return the code belonging to the passed ``text`` label (if present).

        Parameters
        ----------
        name : str
            The variable name keyed in columns or masks.
        text_label : str or list of str
            The value text(s) to search for.
        text_key : str, default None
            The desired ``text_key`` to search through. Uses ``self.text_key``
            if not provided.
        axis : {"x", "y"}, default None
            If provided the text_key is taken from the x/y edits dict.
        exact : bool, default True
            ``text_label`` must exactly match a categorical value"s ``text``.
            If False, it is enough that the category *contains* ``text_label``.
        flat : If a list is passed for ``text_label``, return all found codes
            as a regular list. If False, return a list of lists matching the
            order of the ``text_label`` list.
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

    def get_factors(self, name):
        """
        Get categorical data"s stat. factor values.

        Parameters
        ----------
        name : str
            The variable name keyed in columns or masks.

        Returns
        -------
        factors : OrderedDict
            A ``{value: factor}`` mapping.
        """
        if not self.is_categorical(name):
            err = "'{}' is not categorical.".format(name)
            logger.error(err); raise TypeError(err)
        if self.is_array(name):
            values = self["masks"][name]["values"]
        else:
            values = self["columns"][name]["values"]
        values = self.emulate_meta(values)
        return OrderedDict([
            (val["value"], val.get("factor"))
            for val in values if "factor" in val])

    def get_items(self, name, text_key=None, axis=None):
        """
        Get the array"s paired item names and texts information from the meta.

        Parameters
        ----------
        name : str
            The variable name keyed in masks.
        text_key : str, default None
            The text_key that should be used when taking labels.
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

    def get_item_texts(self, name, text_key=None, axis=None):
        """
        Get the ``text`` meta data for the items of the passed array mask name.

        Parameters
        ----------
        name : str
            The variable name keyed in columns or masks.
        text_key : str, default None
            The text_key that should be used when taking labels.
        axis : {"x", "y"}, default None
            If provided the text_key is taken from the x/ y edits dict.
        """
        items = self.get_values(name, text_key, axis)
        return [text for item, text in items]

    def get_sources(self, name):
        """
        Get the ``columns`` elements for the passed array mask name.

        Parameters
        ----------
        name : str
            The variable name keyed in masks.
        """
        if not self.is_array(name):
            err = "'{}' is not an array.".format(name)
            logger.error(err); raise TypeError(err)
        return self.variables_from_set(name)

    def get_parent(self, name):
        if not is_array_item(name):
            return None
        else:
            return self["columns"][name]["parent"][0].split("@")[-1]

    # -------------------------------------------------------------------------
    # helpers
    # -------------------------------------------------------------------------
    @classmethod
    def dims_free_array_name(cls, name):
        return name.split(".")[0]

    @classmethod
    def dims_free_array_item_name(cls, name):
        pattern = '\[\{.*?\}\]\.'
        found = re.search(pattern, name)
        if found:
            return found.group()[2: -3]
        else:
            return name

    def dims_comp_array_name(self, name):
        name = self.dims_free_array_name(name)
        if self.dimensions_comp:
            return "{}.{}{}".format(name, name, self.dimensions_suffix)
        else:
            return name

    def dims_comp_array_item_name(self, name):
        parent = self.get_parent(name)
        name = self.dims_free_array_item_name(name)
        if self.dimensions_comp:
            return "{parent}[{{name}}].{parent}{suffix}".format(
                parent=parent, name=name, suffix=self.dimensions_suffix)

    def extend_set(self, item, setname="data file", idx=-1):
        collection = "masks" if self.is_array(item) else "columns"
        item = "{}@{}".format(collection, item)
        if idx == -1:
            self["sets"][setname]["items"].append(item)
        else:
            items = self.get_set(setname)
            nitems = items[:idx] + [item] + items[idx + 1:]
            self["sets"][setname]["items"] = nitems

    @staticmethod
    def _pad_list(pad_list, pad_to_len):
        to_add = pad_to_len - len(pad_list)
        pad_list.extend([""] * to_add)
        return pad_list

    @modify(to_list=["items"])
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
                new_list.extend(self.sources(v))
        return new_list

    # -------------------------------------------------------------------------
    @modify(to_list=["varlist", "keep", "both"])
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

    @modify(to_list=["varlist", "ignore_arrays"])
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
                if self._is_array_item(var):
                    parent = self._maskname_from_item(var)
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

    @modify(to_list=["name"])
    def drop(self, name, ignore_items=False):
        """
        Drop variable safely from meta dict.

        Parameters
        ----------
        name : str or list of str
            The column variable name keyed in ``_meta["columns"]`` or
            ``_meta["masks"]``.
        ignore_items: bool
            If False source variables for arrays in ``_meta["columns"]``
            are dropped, otherwise kept.
        """
        def remove_loop(obj, var):
            if isinstance(obj, dict):
                obj.pop(var, None)
                for key in obj:
                    remove_loop(obj[key], var)

        if any(self.is_array_item(n) for n in name):
            err = "Cannot drop isolated array items!"
            logger.error(err); raise ValueError(err)
        to_drop = []
        data_file = self.get_set()[:]
        for n in name:
            to_drop.append(n)
            if self.is_array(n):
                if not ignore_items:
                    to_drop.extend(self.sources(n))
                else:
                    items = self.get_set(n)
                    idx = data_file.index("masks@{}".format(n))
                    data_file = data_file[:idx] + items + data_file[idx + 1:]
                    if self.is_categorical(n):
                        values = self["lib"]["values"][n][:]
                    for item in self.sources(n):
                        if self.is_categorical(n)
                            self["columns"][item]["values"] = value
                        self["columns"][item]["parent"] = {}
        self["sets"]["data file"]["items"] = data_file
        for n in to_drop:
            remove_loop(self, n)

    # -------------------------------------------------------------------------
    # batches
    # -------------------------------------------------------------------------
    @property
    def batches(self):
        return self["sets"].get("batches", {}).keys()

    def batches(self, main=True, add=True):
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
        return not self["sets"]["batches"][batchname]["additional"]

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
    # text_keys
    # -------------------------------------------------------------------------
    def _extract_text(self, text_obj, text_key, axis):
        if axis:
            return text_obj.get(text_key, {}).get(axis, "")
        else:
            return text_obj.get(text_key, "")

    # -------------------------------------------------------------------------
    # repair
    # -------------------------------------------------------------------------
    def _clean_custom_sets_and_libs(self):
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
                values = self["masks"][mask]["values"]
                if not isinstance(values, str):
                    lib_def = "lib@values@{}".format(mask)
                    self["masks"][mask]["values"] = lib_def
                    self["lib"]["values"][mask] = values
                    logger.info("Fixed value ref for '{}'".format(mask))
            elif "values" in self["masks"][mask]:
                self["masks"][mask].pop("values")

        for col in self.columns:
            # verify column name
            self["columns"][col]["name"] = col
            if col in parents:
                # verify parent and values for array items
                mask = parents[col]
                parent_def = {'masks@{}'.format(mask): {'type': 'array'}}
                self["columns"][col]["parent"] = parent_def
                if self.is_categorical(col):
                    lib_def = "lib@values@{}".format(mask)
                    self["columns"][col]["values"] = lib_def
            else:
                # verify parent and values for non-array items
                self["columns"][col]["parent"] = {}
                values = self["masks"][col]["values"]
                if self.is_categorical(col) and isinstance(values, str):
                    self["masks"][col]["values"] = self.emulate_meta(values)

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
