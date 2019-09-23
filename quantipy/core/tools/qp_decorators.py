
from decorator import decorator
from inspect import getfullargspec as getargspec

# ------------------------------------------------------------------------
# decorators
# ------------------------------------------------------------------------

def lazy_property(func):
    """Decorator that makes a property lazy-evaluated.
    """
    attr_name = '_lazy_' + func.__name__
    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))
        return getattr(self, attr_name)
    return _lazy_property

def verify(variables=None, categorical=None, text_keys=None, axis=None, is_str=None):
    """
    Decorator to verify arguments.
    """
    @decorator
    def _var_in_ds(func, *args, **kwargs):
        all_args = getargspec(func)[0]
        ds = args[0]
        for variable, collection in list(variables.items()):
            nested = False
            if collection.endswith('_nested'):
                nested = True
                collection = collection.split('_')[0]
            # get collection for argument
            if collection == 'both':
                collection = ['columns', 'masks']
            else:
                collection = [collection]
            c = [key for col in collection for key in list(list(ds._meta[col].keys()))]
            # get the variable argument to check
            v_index = all_args.index(variable)
            var = kwargs.get(variable, args[v_index])
            if var is None:
                return func(*args, **kwargs)
            if not isinstance(var, list):
                var = [var]
            if nested:
                valid = []
                for v in var:
                    if ' > ' in v:
                        valid.extend(v.replace(' ', '').split('>'))
                    else:
                        valid.append(v)
            else:
                valid = var
            # check the variable
            not_valid = [v for v in valid if not v in c + ['@']]
            if not_valid:
                msg = "'{}' argument for {}() must be in {}.\n"
                msg += '{} is not in {}.'
                msg = msg.format(variable, func.__name__, collection,
                                 not_valid, collection)
                raise KeyError(msg)
        return func(*args, **kwargs)

    @decorator
    def _var_is_cat(func, *args, **kwargs):
        all_args = getargspec(func)[0]
        ds = args[0]
        for cat in categorical:
            # get the variable argument to check if it is categorical
            v_index = all_args.index(cat)
            var = kwargs.get(cat, args[v_index])
            if var is None: return func(*args, **kwargs)
            if not isinstance(var, list): var = [var]
            valid = []
            for v in var:
                if ' > ' in v:
                    valid.extend(v.replace(' ', '').split('>'))
                elif not '@' == v:
                    valid.append(v)
            # check if varaibles are categorical
            not_cat = [v for v in valid if not ds._has_categorical_data(v)]
            if not_cat:
                msg = "'{}' argument for {}() must reference categorical "
                msg += 'variable.\n {} is not categorical.'
                msg = msg.format(cat, func.__name__, not_cat)
                raise ValueError(msg)
        return func(*args, **kwargs)

    @decorator
    def _verify_text_key(func, *args, **kwargs):
        all_args = getargspec(func)[0]
        ds = args[0]
        for text_key in text_keys:
            # get the text_key argument to check
            tk_index = all_args.index(text_key)
            tks = kwargs.get(text_key, args[tk_index])
            if tks is None: return func(*args, **kwargs)
            if not isinstance(tks, list): tks = [tks]
            # ckeck the text_key
            valid_tks = ds.valid_tks
            not_supported = [tk for tk in tks if not tk in valid_tks]
            if not_supported:
                msg = "{} is not a valid text_key! Supported are: \n {}"
                raise ValueError(msg.format(not_supported, valid_tks))
        return func(*args, **kwargs)

    @decorator
    def _verify_axis(func, *args, **kwargs):
        # get the axis argument to check
        all_args = getargspec(func)[0]
        ax_index = all_args.index(axis)
        a_edit = kwargs.get(axis, args[ax_index])
        if a_edit is None: return func(*args, **kwargs)
        if not isinstance(a_edit, list): a_edit = [a_edit]
        # ckeck the axis
        valid_ax = ['x', 'y']
        not_supported = [ax for ax in a_edit if not ax in valid_ax]
        if not_supported:
            msg = "{} is not a valid axis! Supported are: {}"
            raise ValueError(msg.format(not_supported, valid_ax))
        return func(*args, **kwargs)

    @decorator
    def _is_str(func, *args, **kwargs):
        all_args = getargspec(func)[0]
        for val in is_str:
            # get the arguments to modify
            val_index = all_args.index(val)
            v = kwargs.get(val, args[val_index])
            if not isinstance(v, (list, tuple)): v = [v]
            if not all(isinstance(text, str) for text in v):
                raise ValueError('Included value must be str or list of str.')
        return func(*args, **kwargs)

    @decorator
    def _deco(func, *args, **kwargs):
        p = [variables, categorical, text_keys, axis, is_str]
        d = [_var_in_ds, _var_is_cat, _verify_text_key, _verify_axis, _is_str]
        for arg, dec in reversed(list(zip(p, d))):
            if arg is None: continue
            func = dec(func)
        return func(*args, **kwargs)

    if categorical and not isinstance(categorical, list): categorical = [categorical]
    if text_keys and not isinstance(text_keys, list): text_keys = [text_keys]
    if is_str and not isinstance(is_str, list): is_str = [is_str]

    return _deco

def modify(to_list=None):
    """
    Decorator to modify arguments.
    """
    @decorator
    def _to_list(func, *args, **kwargs):
        all_args = getargspec(func)[0]
        for val in to_list:
            # get the arguments to modify
            val_index = all_args.index(val)
            v = kwargs.get(val, args[val_index])
            if v is None: v = []
            if not isinstance(v, list): v = [v]
            if kwargs.get(val):
                kwargs[val] = v
            else:
                args = tuple(a if not x == val_index else v
                             for x, a in enumerate(args))
        return func(*args, **kwargs)

    if to_list:
        if not isinstance(to_list, list): to_list = [to_list]
        return _to_list
