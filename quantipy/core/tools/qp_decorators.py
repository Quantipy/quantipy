
from decorator import decorator
from inspect import getargspec
from itertools import product
from .functions import ensure_list
from .logger import get_logger

logger = get_logger(__name__)


def _update_args(all_args, args, kwargs, new):
    new_args = []
    for idx, arg in enumerate(all_args):
        new_arg = new.get(arg, kwargs.get(arg, args[idx]))
        new_args.append(new_arg)
    return new_args


def _collect_args(all_args, keys, args, kwargs, nested=False, listed=True):
    if nested:
        pre_collect = []
        for key in keys:
            if isinstance(key, tuple):
                values = ensure_list(
                    kwargs.get(key[1], args[all_args.index(key[1])]))
            else:
                values = ensure_list(
                    kwargs.get(key, args[all_args.index(key)]))
            pre_collect.append([{key: value} for value in values])
        collect = [
            {k: v for p in prod for k, v in p.items()}
            for prod in product(*pre_collect)]
    else:
        collect = {}
        for key in keys:
            if isinstance(key, tuple):
                values = kwargs.get(key[1], args[all_args.index(key[1])])
            else:
                values = kwargs.get(key, args[all_args.index(key)])
            if listed:
                values = ensure_list(values)
            collect[key] = values
    return collect


def params(repeat=[], to_list=[], is_column=[], is_mask=[], is_var=[],
           is_cat=[], text_key=[], axis=[]):

    @decorator
    def _repeat(func, *args, **kwargs):
        """
        Repeat a method for each item (combination) if specified param is list.
        """
        all_args = getargspec(func)[0]
        repeat_args = []
        for new in _collect_args(all_args, repeat, args, kwargs, True):
            repeat_args.append(_update_args(all_args, args, kwargs, new))

        def wrapper_repeat(func, repeat_args):
            for n_args in repeat_args:
                func(*n_args)

        return wrapper_repeat(func, repeat_args)

    @decorator
    def _is_var(func, *args, **kwargs):
        """
        Verify that defined parameters are included in object (self).
        """
        all_args = getargspec(func)[0]
        collect = is_var + ["self"]

        collected = _collect_args(all_args, collect, args, kwargs, False, True)
        obj = collected.pop("self")[0]
        err = "Not found in '{}': '{}'"
        invalid = []
        for variables in collected.values():
            for var in variables:
                for v in var.split(">"):
                    v = v.replace(" ", "")
                    if not(v in obj or v == "@"):
                        invalid.append(v)
        if invalid:
            err = err.format(obj.__class__.__name__, "', '".join(invalid))
            logger.error(err); raise KeyError(err)
        return func(*args, **kwargs)

    @decorator
    def _is_column(func, *args, **kwargs):
        """
        Verify that defined parameters are included columns in object (self).
        """
        all_args = getargspec(func)[0]
        collect = is_column + ["self"]
        collected = _collect_args(all_args, collect, args, kwargs, False, True)
        obj = collected.pop("self")[0]
        columns = obj.columns
        err = "Not found in columns: '{}'"
        invalid = []
        for variables in collected.values():
            for var in variables:
                for v in var.split(">"):
                    v = v.replace(" ", "")
                    if not(v in columns or v == "@"):
                        invalid.append(v)
        if invalid:
            err = err.format("', '".join(invalid))
            logger.error(err); raise KeyError(err)
        return func(*args, **kwargs)

    @decorator
    def _is_mask(func, *args, **kwargs):
        """
        Verify that defined parameters are included columns in object (self).
        """
        all_args = getargspec(func)[0]
        collect = is_mask + ["self"]
        collected = _collect_args(all_args, collect, args, kwargs, False, True)
        obj = collected.pop("self")[0]
        masks = obj.masks
        err = "Not found in masks: '{}'"
        invalid = []
        for variables in collected.values():
            for var in variables:
                if var not in masks:
                    invalid.append(var)
        if invalid:
            err = err.format("', '".join(invalid))
            logger.error(err); raise KeyError(err)
        return func(*args, **kwargs)

    @decorator
    def _is_cat(func, *args, **kwargs):
        """
        Verify that defined parameters are categorical variables.
        """
        all_args = getargspec(func)[0]
        collect = is_cat + ["self"]
        collected = _collect_args(all_args, collect, args, kwargs, False, True)
        obj = collected.pop("self")[0]
        err = "Not categorical: '{}'"
        invalid = []
        for variables in collected.values():
            for var in variables:
                if not obj.is_categorical(var):
                    invalid.append(var)
        if invalid:
            err = err.format("', '".join(invalid))
            logger.error(err); raise TypeError(err)
        return func(*args, **kwargs)

    @decorator
    def _to_list(func, *args, **kwargs):
        """
        Verify that defined parameters are lists.
        """
        all_args = getargspec(func)[0]
        collected = _collect_args(all_args, to_list, args, kwargs, False, True)
        args = _update_args(all_args, args, kwargs, collected)
        return func(*args)

    @decorator
    def _axis(func, *args, **kwargs):
        """
        Verify that defined parameters are {"x", "y", ["x", "y"]}.
        """
        all_args = getargspec(func)[0]
        collected = _collect_args(all_args, axis, args, kwargs, False, False)
        invalid = []
        for k, ax in list(collected.items()):
            collected.pop(k)
            suffix = None
            if not ax:
                continue
            if isinstance(k, tuple):
                suffix = k[0]
                n_axis = "{{}} {b}".format(b=suffix).format
                k = k[1]
            else:
                n_axis = "{}".format
            new_ax = []
            for a in ensure_list(ax):
                if a not in ["x", "y", n_axis("x"), n_axis("y")]:
                    invalid.append(a)
                elif suffix and suffix in a:
                    new_ax.append(a)
                else:
                    new_ax.append(n_axis(a))
            if isinstance(ax, list):
                collected[k] = new_ax
            else:
                collected[k] = new_ax[0]
        if invalid:
            err = "Not valid axis: '{}'".format("', '".join(invalid))
            logger.error(err); raise KeyError(err)
        args = _update_args(all_args, args, kwargs, collected)
        return func(*args)

    @decorator
    def _text_key(func, *args, **kwargs):
        """
        Verify that defined parameters are valid text_keys.
        """
        all_args = getargspec(func)[0]
        collect = text_key + ["self"]
        collected = _collect_args(all_args, collect, args, kwargs, False,
                                  False)
        obj = collected.pop("self")
        invalid = []
        for k, tks in collected.items():
            if tks == ["all"]:
                collected[k] = obj.valid_tks
            elif not tks:
                collected[k] = obj.text_key
            elif isinstance(tks, list):
                for tk in collected[k]:
                    if tk not in obj.valid_tks:
                        invalid.append(tk)
            elif tks not in obj.valid_tks:
                invalid.append(tks)
        if invalid:
            err = "Not found in '{}.valid_tks': '{}'".format(
                obj.__class__.__name__, "', '".join(invalid))
            logger.error(err); raise ValueError(err)
        args = _update_args(all_args, args, kwargs, collected)
        return func(*args)

    @decorator
    def _deco(func, *args, **kwargs):
        if repeat:
            func = _repeat(func)
        if is_mask:
            func = _is_mask(func)
        if is_column:
            func = _is_column(func)
        if is_cat:
            func = _is_cat(func)
        if is_var:
            func = _is_var(func)
        if axis:
            func = _axis(func)
        if text_key:
            func = _text_key(func)
        if to_list:
            func = _to_list(func)
        return func(*args, **kwargs)

    repeat = ensure_list(repeat)
    to_list = ensure_list(to_list)
    is_column = ensure_list(is_column)
    is_mask = ensure_list(is_mask)
    is_var = ensure_list(is_var)
    is_cat = ensure_list(is_cat)
    text_key = ensure_list(text_key)
    axis = ensure_list(axis)
    return _deco


def lazy_property(func):
    """
    Decorator that makes a property lazy-evaluated.
    """
    attr_name = '_lazy_' + func.__name__
    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))
        return getattr(self, attr_name)
    return _lazy_property
