#!/usr/bin/python
# -*- coding: utf-8 -*-
from ...__imports__ import *  # noqa

logger = get_logger(__name__)


class View(object):
    def __init__(self, link, name, method=None, kwargs=None):
        self.name = name.split("|")[-1]
        self.link = link
        self._method = method
        self.kwargs = kwargs if kwargs else {}
        self.condition = kwargs.get("condition")
        self._logic = kwargs.get("logic")

        self.dataframe = pd.DataFrame()
        self._link_meta()

        self._rbases = None
        self._cbases = None
        self._custom_txt = ''

    def __repr__(self):
        return '{}'.format(self.dataframe)

    def _link_meta(self,):
        meta = self.link.meta
        xk = self.link.xk
        nested = ">" in xk
        self._xk = {
            "name": xk,
            "is_multi": meta.is_delimited_set(xk),
            "is_nested": nested,
            "is_array": meta.is_array(xk) if not nested else False
        }
        yk = self.link.yk
        nested = ">" in yk
        self._yk = {
            "name": yk,
            "is_multi": meta.is_delimited_set(yk),
            "is_nested": ">" in yk,
            "is_array": meta.is_array(yk) if not nested else False
        }

    def meta(self):
        """
        Get a summary on a View's meta information.

        Returns
        -------
        viewmeta: dict
            A dictionary that contains global aggregation information.
        """
        viewmeta = {
            'agg': {
                'is_weighted': self.is_weighted,
                'weights': self.weight,
                'method': self.method,
                'name': self.name,
                'fullname': self.notation,
                'text': self.text,
                'grp_text_map': self.grp_text_map,
                'is_block': self.is_block},
            'x': self._xk,
            'y': self._yk,
            'shape': self.dataframe.shape}
        return viewmeta


    # -------------------------------------------------------------------------
    # kwargs
    # -------------------------------------------------------------------------
    @property
    def notation(self):
        return "x|{}|{}|{}|{}|{}".format(
            self._method, self.condition, self.rel_to, self.weight, self.name)

    @property
    def method(self):
        if callable(self._method):
            logger.debug("Use custom method '{}'".format(self.name))
            return self._method
        if self._method.startswith("d."):
            return "descriptives"
        elif self._method.startswith("f"):
            return "frequency"
        elif self._method.startswith("t."):
            return "coltests"
        else:
            return self._method

    @property
    def rel_to(self):
        return self.kwargs.get("rel_to", "")

    @property
    def weight(self):
        return self.kwargs.get("weights") or ""

    @property
    def condition(self):
        return self.kwargs.get("condition", ":")

    @condition.setter
    def condition(self, condition):
        if not condition:
            condition = ':'
        elif condition not in ['x:', ':']:
            if 't.' not in self._method:
                colon_form = '*:' if self._complete else ':'
                if self.axis == "x" or not self.axis:
                    if not condition.endswith(colon_form):
                        condition = condition + colon_form
                else:
                    if not condition.startswith(colon_form):
                        condition = colon_form + condition
        self.kwargs["condition"] = condition

    @property
    def axis(self):
        return self.kwargs.get("axis", "x")

    @property
    def text(self):
        return self.kwargs.get("text", "")

    # net properties
    @property
    def _calc(self):
        return self.kwargs.get("calc", {})

    @property
    def _calc_only(self):
        return self.kwargs.get("calc_only", False)

    @property
    def _logic(self):
        return self.kwargs.get("logic", [])

    @_logic.setter
    def _logic(self, logic):
        if not logic:
            self.kwargs["logic"] = None
        elif isinstance(logic, list) and not all(isinstance(log, dict)
                                                 for log in logic):
            self.kwargs["logic"] = [{self.name: logic}]
        else:
            self.kwargs["logic"] = logic

    @property
    def _expand(self):
        return self.kwargs.get("expand", None)

    @property
    def _expanded_net_groups(self):
        groups = {"net": OrderedDict()}
        normal = []
        expanded = []
        for idx, item in zip([self.dataframe.index.levels[1], self._logic]):
            if any([exp in ["before", "after"]
                    for exp in [self._expand, item.get("expand")]]):
                for key in item.keys():
                    if key not in ["text", "expand"]:
                        break
                groups["net"][key] = item[key]
                expanded.extend(item[key])
            else:
                normal.append(idx)
        groups["normal"] = [idx for idx in normal if idx not in expanded]
        return groups

    @property
    def _complete(self):
        return self.kwargs.get("complete", False) or self._expand is not None

    @property
    def grp_text_map(self):
        if self._calc_only and self._logic:
            calc_text = self._calc.get("text")
            calc_key = [
                key for key in self._calc.keys()
                if key not in ["text", "expand"]]
            return {calc_key: calc_text}
        elif self._logic:
            net_texts = []
            net_names = []
            for item in self._logic:
                net_texts.append(item.get("text"))
                net_names.extend([
                    key for key in item.keys()
                    if key not in ["text", "expand"]])
            grp_text_map = dict(zip(net_names, net_texts))
            if self._calc:
                calc_text = self._calc.get("text")
                calc_key = [
                    key for key in self._calc.keys() if not key == "text"][0]
                grp_text_map[calc_key] = calc_text
            return grp_text_map
        else:
            return None

    # descriptive properties
    @property
    def _stats(self):
        return self.kwargs.get("stats", None)

    @property
    def _exclude(self):
        return self.kwargs.get("exclude", None)

    @property
    def _drop(self):
        return self.kwargs.get("drop", None)

    @property
    def _rescale(self):
        return self.kwargs.get("rescale", None)

    @property
    def _source(self):
        return self.kwargs.get("source", None)

    # coltest properties
    @property
    def _metric(self):
        return self.kwargs.get("metric", None)

    @property
    def _mimic(self):
        return self.kwargs.get("mimic", None)

    @property
    def _level(self):
        return self.kwargs.get("level", None)

    @property
    def _flag_bases(self):
        return self.kwargs.get("flag_bases", None)

    @property
    def _test_total(self):
        return self.kwargs.get("test_total", None)

    # -------------------------------------------------------------------------
    # properties
    # -------------------------------------------------------------------------
    @property
    def is_weighted(self):
        return len(self.weight) > 0

    @property
    def is_pct(self):
        return self.method == "frequency" and len(self.rel_to) > 0

    @property
    def is_counts(self):
        return self.method == "frequency" and len(self.rel_to) == 0

    @property
    def is_base(self):
        return self._method == "f" and len(self.condition) == 2

    @property
    def is_sum(self):
        return self._method.startswith("f.c") and len(self.condition) == 2

    @property
    def is_cumulative(self):
        return self.condition == "x++:"

    @property
    def is_stat(self):
        return self.method == "descriptives"

    @property
    def is_net(self):
        return all([
            self.method == "frequency",
            len(self.condition) > 3,
            not self.is_cumulative])

    @property
    def is_expanded_net(self):
        return self.is_net and any([
            x in self.condition for x in ["}+]", "[+{"]])

    @property
    def has_calc(self):
        return self._method.startswith("f.c") and not self.is_cumulative

    @property
    def is_block(self):
        return self.method == "frequency" and any([
            len(self.condition.split("[")) > 2, self._expand, self._complete])

    @property
    def is_test(self):
        return self.method == "coltests"

    @property
    def is_meanstest(self):
        if self.is_test and self._method.split(".")[1] == "means":
            return float(self._method.split(".")[3].split("+")[0])/100
        else:
            return False

    @property
    def is_propstest(self):
        if self.is_test and self._method.split(".")[1] == "props":
            return float(self._method.split(".")[3].split("+")[0])/100
        else:
            return False

    # -------------------------------------------------------------------------
    # conditions
    # -------------------------------------------------------------------------
    def spec_condition(self):
        """
        Update the condition component based on agg details.
        """
        if self._logic:
            condition = self._frequency_condition()
        elif self._stats:
            condition = self._descriptives_condition()
        else:
            condition = self.axis or self.condition
        if self._calc:
            calc_cond = self._calc_condition(condition)
            if self._calc_only:
                condition = calc_cond
            else:
                if self._logic:
                    condition.append(calc_cond)
                else:
                    condition = [condition, calc_cond]
        if isinstance(condition, list):
            condition = ",".join(condition)
        self.condition = condition

    def _frequency_condition(self):
        logic_codes = []
        for item in self._logic:
            codes = "{{{}}}".format(
                ",".join([
                    str(v) for k, val in item.items()
                    for v in val if k not in ["text", "expand"]]))
            if self._expand == 'after' or item.get("expand") == "after":
                logic_codes.append("{}[{}+]".format(self.axis, codes))
            elif self._expand == 'before' or item.get("expand") == "before":
                logic_codes.append("{}[+{}]".format(self.axis, codes))
            else:
                logic_codes.append("{}[{}]".format(self.axis, codes))
        return logic_codes

    def _descriptives_condition(self):
        if self._source:
            return self._source
        if not (self._exclude or self._rescale):
            return self.axis

        var = self.link.yk if self.link.xk == "@" else self.link.xk
        if self.link.meta.is_categorical(var):
            codes = self.link.meta.get_codes(var)
            if self._exclude:
                codes = [code for code in codes if not code in self._exclude]
            if self._rescale:
                codes = [self._rescale.get(code, code) for code in codes]
            return 'x[{{{}}}]'.format(','.join(map(str, codes)))
        else:
            if self._exclude:
                return 'x~{{{}}}'.format(",".join(map(str, self._exclude)))
            else:
                return self.axis

    def _calc_condition(self, conditions):
        for k, v in self._calc.items():
            if k not in ["text", "calc_only"]:
                val1, op, val2 = v
                break
        symbol_map = {add: '+', sub: '-', mul: '*', div: '/'}
        cond_names = [
            key for log in self._logic for key in log.keys()
            if key not in ["text", "expand"]]
        cond_map = dict(zip(cond_names, conditions))
        v1 = cond_map.get(val1, val1)
        v2 = cond_map.get(val2, val2)
        calc_string = "{}{}{}".format(v1, symbol_map[op], v2)
        repl = [("+{", "{"), ("}+", "}"), ("x", ""), ("[", ""), ("]", "")]
        for rep in repl:
            calc_string = calc_string.replace(*rep)
        calc_string = 'x[{}]'.format(calc_string)
        return calc_string

    # -------------------------------------------------------------------------
    # metric
    # -------------------------------------------------------------------------
    def translate_metric(self):
        if self.is_stat or self.is_base or self.is_sum:
            add_custom_text = True
            if not self._custom_txt:
                invalid = [
                    'Total', 'Lower quartile', 'Max', 'Min', 'Mean',
                    'Upper quartile', 'Unweighted base', 'Total Sum',
                    'Std. err. of mean', 'Base', 'Median', 'Std. dev',
                    'Sample variance', 'Coefficient of variance',
                    'Gross base', 'Unweighted gross base', '']
                if self.text not in invalid:
                    self._custom_txt = self.text
                else:
                    add_custom_text = False
            tk = self.link.meta.text_key
            transl = METRIC_NAME_MAP.get(tk, "en-GB")
            try:
                old_val = self.dataframe.index.get_level_values(1)[0]
                if self.is_base:
                    if "_gross" in self.name:
                        if self.is_weighted:
                            old_val = 'gross All'
                        else:
                            old_val = 'no_w_gross_' + old_val
                    if not self.is_weighted:
                        old_val = 'no_w_' + old_val
                new_val = transl[old_val]
                if add_custom_text:
                    new_val = new_val + ' ' + self._custom_txt
                if not self.text == new_val:
                    self.kwargs["text"] = new_val
            except (TypeError, KeyError):
                return self.text or old_val
