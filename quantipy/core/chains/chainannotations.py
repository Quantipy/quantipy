#!/usr/bin/python
# -*- coding: utf-8 -*-

from quantipy.core.tools.logger import get_logger
logger = get_logger(__name__)


HEADERS = [
    "header-title",
    "header-left",
    "header-center",
    "header-right"]

FOOTERS = [
    "footer-title",
    "footer-left",
    "footer-center",
    "footer-right"]

VALID_ANNOT_TYPES = HEADERS + FOOTERS + ["notes"]

VALID_ANNOT_CATS = [
    "header",
    "footer",
    "notes"]

VALID_ANNOT_POS = [
    "title",
    "left",
    "center",
    "right"]


class ChainAnnotations(dict):

    def __init__(self):
        super(ChainAnnotations, self).__init__()
        self.header_title = []
        self.header_left = []
        self.header_center = []
        self.header_right = []
        self.footer_title = []
        self.footer_left = []
        self.footer_center = []
        self.footer_right = []
        self.notes = []
        for v in VALID_ANNOT_TYPES:
            self[v] = []

    def __setitem__(self, key, value):
        self._test_valid_key(key)
        return super(ChainAnnotations, self).__setitem__(key, value)

    def __getitem__(self, key):
        self._test_valid_key(key)
        return super(ChainAnnotations, self).__getitem__(key)

    def __repr__(self):
        headers = [
            (h.split("-")[1], self[h])
            for h in self.populated
            if h.split("-")[0] == "header"]
        footers = [
            (f.split("-")[1], self[f])
            for f in self.populated
            if f.split("-")[0] == "footer"]
        notes = self["notes"] if self["notes"] else []
        if notes:
            ar = "Notes\n"
            ar += "-{:>16}\n".format(str(notes))
        else:
            ar = "Notes: None\n"
        if headers:
            ar += "Headers\n"
            for pos, text in dict(headers).items():
                ar += "  {:>5}: {:>5}\n".format(str(pos), str(text))
        else:
            ar += "Headers: None\n"
        if footers:
            ar += "Footers\n"
            for pos, text in dict(footers).items():
                ar += "  {:>5}: {:>5}\n".format(str(pos), str(text))
        else:
            ar += "Footers: None"
        return ar

    def _test_valid_key(self, key):
        msg = "'{}' is not a valid annotation {}!"
        if key not in VALID_ANNOT_TYPES:
            splitted = key.split("-")
            if not len(splitted) == 2:
                err = msg.format(key, "type")
            else:
                acat, apos = splitted
                if acat == "notes":
                    err = "'notes' annotation type does not support positions!"
                elif not (acat in VALID_ANNOT_CATS or apos in VALID_ANNOT_POS):
                    err = msg.format(key, "type")
                elif acat not in VALID_ANNOT_CATS:
                    err = msg.format(acat, "category")
                elif apos not in VALID_ANNOT_POS:
                    err = msg.format(apos, "position")
            logger.error(err); raise KeyError(err)

    @property
    def header(self):
        h_dict = {}
        for h in HEADERS:
            if self[h]:
                h_dict[h.split("-")[1]] = self[h]
        return h_dict

    @property
    def footer(self):
        f_dict = {}
        for f in FOOTERS:
            if self[f]:
                f_dict[f.split("-")[1]] = self[f]
        return f_dict

    @property
    def populated(self):
        """
        The annotation fields that are defined.
        """
        return sorted([k for k, v in self.items() if v])

    @staticmethod
    def _annot_key(a_type, a_pos):
        if a_pos:
            return "{}-{}".format(a_type, a_pos)
        else:
            return a_type

    def set(self, text, category="header", position="title"):
        """
        Add annotation texts defined by their category and position.

        Parameters
        ----------
        category : {"header", "footer", "notes"}, default "header"
            Defines if the annotation is treated as a *header*, *footer* or
            *note*.
        position : {"title", "left", "center", "right"}, default "title"
            Sets the placement of the annotation within its category.

        Returns
        -------
        None
        """
        if not category:
            category = "header"
        if not position and not category == "notes":
            position = "title"
        if category == "notes":
            position = None
        akey = self._annot_key(category, position)
        self[akey].append(text)
        self.__dict__[akey.replace("-", "_")].append(text)
        return None
