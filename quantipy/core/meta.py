
from ..__imports__ import *  # noqa


class Meta(dict):

    def __init__(self, json_dict={}):
        super(dict, self).__init__()
        lib = json_dict.get("lib", {})
        info = json_dict.get("info", {})
        self.text_key = lib.get("default text", "en-GB")
        self.valid_tks = lib.get("valid text", VALID_TKS)
        self._dimensions_comp = info.get("dimensions_comp", False)
        self._dimensions_suffix = info.get("dimensions_suffix", "_grid")
        if not json_dict:
            json_dict = self.start_meta()
        self.update(json_dict)

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
    def dimensions_comp(self):
        return self._dimensions_comp

    @dimensions_comp.setter
    def dimensions_comp(self, value):
        self['info']['dimensions_comp'] = value
        self._dimensions_comp = value

    @property
    def dimensions_suffix(self):
        return self._dimensions_suffix

    @dimensions_suffix.setter
    def dimensions_suffix(self, suffix):
        if not suffix:
            suffix = self['info'].get(
                'dimensions_suffix',
                self._dimensions_suffix
            )
        if not suffix == self._dimensions_suffix:
            self._dimensions_suffix = suffix
        self['info']['dimensions_suffix'] = suffix

    def start_meta(self):
        """
        Fills meta dict with basic structure.
        """
        meta = {
            'info': {
                'text': '',
                'dimensions_comp': self.dimensions_comp,
                'dimensions_suffix': self.dimensions_suffix
            },
            'lib': {
                'default text': self.text_key,
                'valid text': self.valid_tks,
                'values': {}
            },
            'columns': {},
            'masks': {},
            'sets': {
                'data file': {
                    'text': {self.text_key: 'Variable order in source file'},
                    'items': []
                }
            },
            'type': 'pandas.DataFrame'
        }
        return meta

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

    @classmethod
    def load(cls, path):
        """
        Load json file into a Meta object.
        """
        if not path.endswith(".json"):
            path += ".json"
        meta = cls(load_json(path))
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

    def by_property(self, prop, **kwargs):
        """
        View all variables that own the requested property.

        Note:
            The list can be additionally filtered by adding kwargs for
            ``self.variables()``.
        """
        return [
            v for v in self.variables(**kwargs)
            if self.get_property(v, prop)]

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
        return self.get_type(name) == 'array'

    def is_array_item(self, name):
        if self.is_array or not self['columns'][name].get('parent'):
            return False
        else:
            return True

    def is_numeric(self, name):
        return self._check_type(name, NUMERIC)

    def is_filter(self, name):
        if self.get_property(name, 'recoded_filter'):
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
        setname : str, default 'data file'
            The name of the variable set to query. Defaults to the main
            variable collection stored via 'data file'.
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
        dsvars = self._dissect_setlist(self.get_set(setname))
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

    def get_set(self, setname="data file"):
        if setname not in self.sets:
            err = "'{}' is not a valid setname.".format(setname)
            logger.error(err); raise KeyError(err)
        return self["sets"][setname]["items"]

    # -------------------------------------------------------------------------
    # helpers
    # -------------------------------------------------------------------------
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
            if keep in ['both', 'items']:
                new_list.append(v)
        else:
            # masks
            if keep in ['both', 'mask']:
                new_list.append(v)
            if keep in ['both', 'items']:
                new_list.extend(self.sources(v))
        return new_list

    @modify(to_list=['varlist', 'keep', 'both'])
    def unroll(self, varlist, keep=None, both=None):
        """
        Replace mask with its items, optionally excluding/keeping certain ones.

        Parameters
        ----------
        varlist : list
           A list of meta ``'columns'`` and/or ``'masks'`` names.
        keep : str or list, default None
            The names of masks that will not be replaced with their items.
        both : 'all', str or list of str, default None
            The names of masks that will be included both as themselves and as
            collections of their items.

        Note
        ----
        varlist can also contain nesting `var1 > var2`. The variables which are
        included in the nesting can also be controlled by keep and both, even
        if the variables are also included as a "normal" variable.

        Example::
            meta.unroll(varlist = ['q1', 'q1 > gender'], both='all')
            ['q1',
             'q1_1',
             'q1_2',
             'q1 > gender',
             'q1_1 > gender',
             'q1_2 > gender']

        Returns
        -------
        unrolled : list
            The modified ``varlist``.
        """
        if both and both[0] == 'all':
            both = self.masks
        unrolled = []
        for var in varlist:
            if ' > ' in var:
                nested = var.replace(' ', '').split('>')
                n_list = []
                for n in nested:
                    if n in keep:
                        to_keep = 'mask'
                    elif n in both:
                        to_keep = 'both'
                    else:
                        to_keep = 'items'
                    n_list.append(self._array_and_item_list(n, to_keep))
                for ur in [' > '.join(list(un)) for un in product(*n_list)]:
                    if ur not in unrolled:
                        unrolled.append(ur)
            else:
                if var in keep:
                    to_keep = 'mask'
                elif var in both:
                    to_keep = 'both'
                else:
                    to_keep = 'items'
                for ur in self._array_and_item_list(var, to_keep):
                    if ur not in unrolled:
                        unrolled.append(ur)
        return unrolled

    @modify(to_list=['varlist', 'ignore_arrays'])
    @verify(variables={'varlist': 'both_nested', 'ignore_arrays': 'masks'})
    def roll_up(self, varlist, ignore_arrays=None):
        """
        Replace any array items with its parent mask variable definition name.

        Parameters
        ----------
        varlist : list
           A list of meta ``'columns'`` and/or ``'masks'`` names.
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
                to_keep = 'mask'
            else:
                to_keep = 'items'
                if self._is_array_item(var):
                    parent = self._maskname_from_item(var)
                    if parent not in ignore_arrays:
                        var = parent
                        to_keep = 'mask'
            return var, to_keep

        rolled_up = []
        for var in varlist:
            if ' > ' in var:
                nested = var.replace(' ', '').split('>')
                n_list = []
                for n in nested:
                    n, to_keep = _var_to_keep(n, ignore_arrays)
                    n_list.append(self._array_and_item_list(n, to_keep))
                for ru in [' > '.join(list(un)) for un in product(*n_list)]:
                    if ru not in rolled_up:
                        rolled_up.append(ru)
            else:
                var, to_keep = _var_to_keep(var, ignore_arrays)
                for ru in self._array_and_item_list(var, to_keep):
                    if ru not in rolled_up:
                        rolled_up.append(ru)
        return rolled_up

    # -------------------------------------------------------------------------
    @modify(to_list=["name"])
    def drop(self, name, ignore_items=False):
        """
        Drop variable safely from meta dict.

        Parameters
        ----------
        name : str or list of str
            The column variable name keyed in ``_meta['columns']`` or
            ``_meta['masks']``.
        ignore_items: bool
            If False source variables for arrays in ``_meta['columns']``
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


