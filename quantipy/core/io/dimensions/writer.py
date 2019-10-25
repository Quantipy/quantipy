#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Created on 20 June 2016

@author: AlexBuchhammer

Cleanup: KMUE, Oct 2019
"""

from ....__imports__ import *  # noqa

from .dimlabels import (
    qp_dim_languages,
    DimLabels)

logger = get_logger(__name__)


QTYPES = {
    'single': 'mr.Categorical',
    'delimited set': 'mr.Categorical',
    'int': 'mr.Long',
    'float': 'mr.Double',
    'string': 'mr.Text',
    'date': 'mr.Date',
    'boolean': 'mr.Boolean'
}


def tab(tabs):
    return '' if tabs == 0 else '\t' * tabs


def AddProp(prop, content):
    return 'MDM.{}.Add("{}")'.format(prop, content.replace(' ', ''))


def SetCurrent(prop, content):
    return 'MDM.{}.Current = "{}"'.format(prop, content)


def Dim(*args):
    return 'Dim {}'.format(', '.join(*args))


def SetMDM():
    return u'Set MDM = CreateObject("MDM.Document")'


def section_break(n):
    return u"\n'{}".format('#' * n)


def comment(tabs, text):
    return u"{t}' {tx}".format(t=tab(tabs), tx=text)


def CreateVariable(tabs, name):
    return u'{t}Set newVar = MDM.CreateVariable("{n}")'.format(
        t=tab(tabs), n=name)


def DataType(tabs, parent, dtype):
    return u'{t}{p}.DataType = {dt}'.format(t=tab(tabs), p=parent, dt=dtype)


def MaxValue(tabs, parent, mval):
    return u'{t}{p}.MaxValue = {mv}'.format(t=tab(tabs), p=parent, mv=mval)


def CreateElement(tabs, name):
    return u'{t}Set newElement = MDM.CreateElement("{n}")'.format(
        t=tab(tabs), n=name)


def ElementType(tabs):
    return u'{t}newElement.Type = 0'.format(t=tab(tabs))


def ElementExpression(tabs, expression):
    return u'{t}newElement.Expression = {e}'.format(t=tab(tabs), e=expression)


def AddLabel(tabs, element, labeltype, language, text):
    return u'{ta}{e}.Labels["{lt}"].Text["Analysis"]["{lang}"] = "{t}"'.format(
        ta=tab(tabs), e=element, lt=labeltype.replace(' ', ''), lang=language,
        t=text)


def AddElement(tabs, parent, child):
    return u'{t}{p}.Elements.Add({c})'.format(t=tab(tabs), p=parent, c=child)


def AddField(tabs, parent, child):
    return u'{t}{p}.Fields.Add({c})'.format(t=tab(tabs), p=parent, c=child)


def CreateGrid(tabs, name):
    return u'{t}Set newGrid = MDM.CreateGrid("{n}")'.format(
        t=tab(tabs), n=name)


def MDMSave(tabs, path_mdd):
    return u'{t}MDM.Save("{p}")'.format(t=tab(tabs), p=path_mdd)


class DimensionsWriter(object):

    def __init__(self, data, meta, text_key=None):
        self.data = data
        self.meta = meta
        self.text_key = text_key or meta.text_key
        self._crlf = "CR"

    def run(self, name, path=".", execute=True, clean_up=True):
        self._name = name
        self._path_mdd = os.path.join(path, u"{}.mdd".format(name))
        self._path_ddf = os.path.join(path, u"{}.ddf".format(name))
        self._path_mrs = os.path.join(path, u"{}_create_mdd.mrs".format(name))
        self._path_dms = os.path.join(path, u"{}_create_ddf.dms".format(name))
        self._path_paired = os.path.join(path, u"{}_paired.csv".format(name))
        self._path_datastore = os.path.join(
            path, u"{}_datastore.csv".format(name))
        self.create_mdd()
        self.create_ddf()
        self.get_case_data_inputs()
        logger.info('Case and meta data validated and transformed.')
        if execute:
            feedback = None
            try:
                logger.info('Converting to .ddf/.mdd...')
                command = 'mrscriptcl "{}"'.format(self._path_mrs)
                check_output(command, stderr=STDOUT, shell=True)
                logger.info('.mdd file generated successfully.')
                command = 'DMSRun "{}"'.format(self._path_dms)
                check_output(command, stderr=STDOUT, shell=True)
                logger.info('.ddf file generated successfully.')
            except CalledProcessError as exc:
                logger.error("ERROR:\n{}'".format(exc.output))
                feedback = exc.returncode
            if clean_up:
                for file_loc in [self._path_mrs, self._path_dms,
                                 self._path_paired, self._path_datastore]:
                    os.remove(file_loc)
            return feedback

    # -------------------------------------------------------------------------
    # mdd / mrs
    # -------------------------------------------------------------------------
    def create_mdd(self):
        mrs = [Dim(['MDM', 'newVar', 'newElement', 'newGrid']), SetMDM()]

        variables = []
        all_languages = []
        all_labeltypes = []
        for var in self.meta.variables():
            if self.meta.is_array(var):
                var_mrs, lang, ltype = self.mask_to_mrs(var)
            elif not self.meta.is_array_item(var):
                var_mrs, lang, ltype = self.col_to_mrs(var)
            variables.extend(var_mrs)
            all_languages.extend(lang)
            all_labeltypes.extend(ltype)

        all_languages = uniquify_list(all_languages)
        all_labeltypes = uniquify_list(all_labeltypes)
        for lt in all_labeltypes:
            mrs.append(AddProp('LabelTypes', lt))
        for l in all_languages:
            mrs.append(AddProp('Languages', l))
        mrs.append(SetCurrent(
            'languages', qp_dim_languages.get(self.text_key, 'ENG')))
        mrs.extend(variables)
        mrs.extend([
            section_break(20), comment(0, 'Save MDD'),
            MDMSave(0, self._path_mdd)])
        with open(self._path_mrs, 'w') as f:
            f.write("\n".join(mrs))

    def col_to_mrs(self, name):
        mrs = [
            section_break(20),
            comment(0, name),
            CreateVariable(0, name),
            DataType(0, 'newVar', QTYPES[self.meta.get_type(name)])]

        labels = DimLabels(name, self.text_key)
        labels.add_text(self.meta["columns"][name]['text'])
        lab_mrs = self.get_lab_mrs(0, 'newVar', labels)
        mrs.append(lab_mrs)

        lang = labels.incl_languages
        ltype = labels.incl_labeltypes
        if self.meta.is_categorical(name):
            val_mrs, val_lan, val_lt = self.values_to_mrs(name, "newVar", name)
            mrs.extend(val_mrs)
            lang.extend(val_lan)
            ltype.extend(val_lt)
        mrs.append(AddField(0, 'MDM', 'newVar'))
        return mrs, uniquify_list(lang), uniquify_list(ltype)

    def mask_to_mrs(self, name):
        mask, field = name.split(".")
        mrs = [section_break(20), comment(0, mask), CreateGrid(0, mask)]

        labels = DimLabels(name, self.text_key)
        labels.add_text(self.meta["masks"][name]['text'])
        lab_mrs = self.get_lab_mrs(0, 'newGrid', labels)
        mrs.append(lab_mrs)
        lang = labels.incl_languages
        ltype = labels.incl_labeltypes

        for source in self.meta.get_sources(name):
            mrs.append(CreateElement(0, source))
            i_lab = DimLabels(source, self.text_key)
            i_lab.add_text(self.meta["columns"][source]['text'])
            ilab_mrs = self.get_lab_mrs(0, 'newElement', i_lab)
            mrs.append(ilab_mrs)
            mrs.append(AddElement(0, 'newGrid', 'newElement'))
            lang.extend(i_lab.incl_languages)
            ltype.extend(i_lab.incl_labeltypes)

        subtype = QTYPES[self.meta.get_subtype(name)]
        mrs.extend([CreateVariable(0, field), DataType(0, 'newVar', subtype)])

        if self.meta.is_categorical(name):
            val_mrs, val_lan, val_lt = self.values_to_mrs(name, "newVar", mask)
            mrs.extend(val_mrs)
            lang.extend(val_lan)
            ltype.extend(val_lt)
        mrs.extend([
            AddField(0, 'newGrid', 'newVar'),
            AddField(0, 'MDM', 'newGrid')])
        return mrs, uniquify_list(lang), uniquify_list(ltype)

    def values_to_mrs(self, name, child, child_name):
        values = self.meta[self.meta._get_value_ref(name)][:]
        max_val = 1 if self.meta.is_single(name) else len(values)
        mrs = [MaxValue(0, child, max_val)]
        lang = []
        ltype = []
        for value in values:
            if value["value"] < 0:
                val = "{}aminus{}".format(child_name, -1 * value["value"])
            else:
                val = "{}a{}".format(child_name, value["value"])
            labels = DimLabels(name, self.text_key)
            labels.add_text(value['text'])
            mrs.extend([
                CreateElement(0, val),
                self.get_lab_mrs(0, 'newElement', labels),
                ElementType(0), AddElement(0, child, 'newElement')])
            lang.extend(labels.incl_languages)
            ltype.extend(labels.incl_labeltypes)
        return mrs, uniquify_list(lang), uniquify_list(ltype)

    def get_lab_mrs(self, tab, element, dimlabels):
        lab_mrs = []
        for dimlabel in dimlabels.labels:
            lt = dimlabel.labeltype or 'Label'
            lang = dimlabel.language
            text = dimlabel.text
            lab_mrs.append(AddLabel(tab, element, lt, lang, text))
        return '\n'.join(lab_mrs)

    # -------------------------------------------------------------------------
    # ddf / dms
    # -------------------------------------------------------------------------
    def create_ddf(self):
        header = [
            '#define MASTER_INPUT "{}"'.format(self._name),
            '#define CRLF "{}"'.format(self._crlf)]
        dms = open(
            os.path.join(os.path.dirname(__file__), '_create_ddf.dms'), 'r')
        full_dms = header + [line.replace('\n', '') for line in dms]
        del full_dms[2]
        with open(self._path_dms, 'w') as f:
            f.write("\n".join(full_dms))

    def get_case_data_inputs(self):
        empty_csv = self._paired_empty_csv()
        paired_cols = empty_csv.columns
        datastore_csv = self._datastore_csv(paired_cols)
        empty_csv.to_csv(self._path_paired, index=False, sep='\t')
        datastore_csv.to_csv(self._path_datastore, index=False)

    def _paired_empty_csv(self):
        empty_csv = self.data.copy()
        paired_cols = self.meta.unroll(self.meta.variables())
        empty_csv = empty_csv[paired_cols]
        empty_csv[paired_cols] = np.NaN
        return empty_csv

    def _datastore_csv(self, columns):
        datastore = self.data.copy()
        for col in columns:
            if self.meta.is_categorical(col):
                datastore[col] = self.convert_categorical(datastore[col])
            elif self.meta.is_int(col):
                datastore[col].replace(np.NaN, 'NULL', inplace=True)
                try:
                    datastore[col] = datastore[col].astype('int32')
                except TypeError:
                    pass
            elif self.meta.is_float(col):
                datastore[col].replace(np.NaN, 'NULL', inplace=True)
            elif self.meta.is_string(col):
                datastore[col] = self.clean_string(datastore[col])
        return datastore

    def clean_string(self, series):
        def _replace(x):
            x = str(x)
            repl = [(',', '>_>_>'), ('\r\n', ''), ('\n', ''), ('nan', '')]
            for r in repl:
                x = x.replace(*r)
            return x
        return series.apply(lambda x: _replace(x))

    def convert_categorical(self, series):
        def _convert_ds(x, prefix):
            x = str(x)[:-1]
            if not x:
                return np.NaN
            return ";".join(
                ["{}{}".format(prefix, y.replace('-', 'minus')) for y in x])

        def _convert_single(x, prefix):
            if np.isnan(x):
                return np.NaN
            elif x < 0:
                x = "{}minus{}".format(prefix, -x)
            else:
                x = "{}{}".format(prefix, x)

        name = self.meta.dims_free_array_item_name(series.name)
        prefix = "{}a".format(name)
        if not series.dtype == 'object':
            series = series.apply(lambda x: _convert_single(x, prefix))
        else:
            series = series.apply(lambda x: _convert_ds(x, prefix))
        return series
