#!/usr/bin/python
# -*- coding: utf-8 -*-

VALID_TKS = [
    "en-GB", "da-DK", "fi-FI", "nb-NO", "sv-SE", "de-DE", "fr-FR", "ar-AR",
    "es-ES", "it-IT", "pl-PL"]

VAR_SUFFIXES = [
    "_rc", "_net", " (categories", " (NET", "_rec"]

BLACKLIST_VARIABLES = [
    "batches", "columns", "info", "items", "lib", "masks", "name", "parent",
    "properties", "text", "type", "sets", "subtype", "values", "filter"]

INVALID_VARS = ["id_L1", "id_L1.1"]

INVALID_CHARS_IN_NAMES = [' ', '~', '(', ')', '&']

DEFAULT_MISSINGS = [
    u'weißnicht',
    u'keineangabe',
    u'weißnicht/keineangabe',
    u'keineangabe/weißnicht',
    u'kannmichnichterinnern',
    u'weißichnicht',
    u'nichtindeutschland']

# variable types
QP_TYPES = [
    "single", "delimited set", "array", "int", "float", "string", "date"]

NUMERIC = ["int", "float"]

STRING = ["string"]

DATE = ["date"]

BOOLEAN = ["boolean"]

CATEGORICAL = ["single", "delimited set"]

# key can be converted into values (astype/ categorized)
COMPATIBLE_TYPES = {
    "single":
        ["single", "delimited set", "int", "float", "string"],
    "delimited set":
        ["delimited set", "single", "string"],
    "int":
        ["int", "single", "float", "string"],
    "float":
        ["float", "single", "int", "string"],
    "string":
        ["string", "single"],
    "date":
        ["date", "single", "string"]
}
