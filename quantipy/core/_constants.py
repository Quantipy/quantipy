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

INVALID_CHARS_IN_NAMES = [" ", "~", "(", ")", "&"]

DEFAULT_MISSINGS = [
    u"don'tknow",
    u"noanswer",
    u"skip",
    u"don'tknow/noanswer",
    u"weißnicht",
    u"keineangabe",
    u"weißnicht/keineangabe",
    u"keineangabe/weißnicht",
    u"kannmichnichterinnern",
    u"weißichnicht",
    u"nichtindeutschland"]

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

STAT_VIEWS = [
    'median', 'stddev', 'sem', 'max', 'min', 'mean', 'upper_q', 'lower_q']

METRIC_NAME_MAP = {
    # English
    u"en-GB": {
        u"@": u"Total",
        u"All": u"Base",
        u"no_w_All": u"Unweighted base",
        u"gross All": u"Gross base",
        u"no_w_gross_All": u"Unweighted gross base",
        u"mean": u"Mean",
        u"min": u"Min",
        u"max": u"Max",
        u"median": u"Median",
        u"var": u"Sample variance",
        u"varcoeff": u"Coefficient of variation",
        u"stddev": u"Std. dev",
        u"sem": u"Std. err. of mean",
        u"sum": u"Total Sum",
        u"lower_q": u"Lower quartile",
        u"upper_q": u"Upper quartile"},
    # Danish
    u"da-DK": {
        u"@": u"Total",
        u"All": u"Base",
        u"no_w_All": u"Unweighted base",
        u"gross All": u"Gross base",
        u"no_w_gross_All": u"Unweighted gross base",
        u"mean": u"Gennemsnit",
        u"min": u"Min",
        u"max": u"Max",
        u"median": u"Median",
        u"var": u"Sample variance",
        u"varcoeff": u"Coefficient of variation",
        u"stddev": u"Std.afv",
        u"sem": u"StdErr",
        u"sum": u"Totalsum",
        u"lower_q": u"Nedre kvartil",
        u"upper_q": u"Øvre kvartil"},
    # Swedish
    u"sv-SE": {
        u"@": u"Total",
        u"All": u"Bas",
        u"no_w_All": u"ovägd bas",
        u"gross All": u"Gross base",
        u"no_w_gross_All": u"Unweighted gross base",
        u"mean": u"Medelvärde",
        u"min": u"Min",
        u"max": u"Max",
        u"median": u"Median",
        u"var": u"Sample variance",
        u"varcoeff": u"Coefficient of variation",
        u"stddev": u"Std. av.",
        u"sem": u"StdErr",
        u"sum": u"Summa",
        u"lower_q": u"Undre kvartilen",
        u"upper_q": u"Övre kvartilen"},
    # Norwegian
    u"nb-NO": {
        u"@": u"Total",
        u"All": u"Base",
        u"no_w_All": u"Unweighted base",
        u"gross All": u"Gross base",
        u"no_w_gross_All": u"Unweighted gross base",
        u"mean": u"Gjennomsnitt",
        u"min": u"Min",
        u"max": u"Max",
        u"median": u"Median",
        u"var": u"Sample variance",
        u"varcoeff": u"Coefficient of variation",
        u"stddev": u"Standardavvik",
        u"sem": u"StdErr",
        u"sum": u"Totalsum",
        u"lower_q": u"Nedre kvartil",
        u"upper_q": u"Øvre kvartil"},
    # Finnish
    u"fi-FI": {
        u"@": u"Total",
        u"All": u"Base",
        u"no_w_All": u"Unweighted base",
        u"gross All": u"Gross base",
        u"no_w_gross_All": u"Unweighted gross base",
        u"mean": u"Mean",
        u"min": u"Min",
        u"max": u"Max",
        u"median": u"Median",
        u"var": u"Sample variance",
        u"varcoeff": u"Coefficient of variation",
        u"stddev": u"Std.dev.",
        u"sem": u"StdErr",
        u"sum": u"Totalsum",
        u"lower_q": u"Alakvartiili",
        u"upper_q": u"Yläkvartiili"},
    # French
    u"fr-FR": {
        u"@": u"Total",
        u"All": u"Base",
        u"no_w_All": u"Base brute",
        u"gross All": u"Gross base",
        u"no_w_gross_All": u"Unweighted gross base",
        u"mean": u"Moyenne",
        u"min": u"Min",
        u"max": u"Max",
        u"median": u"Médiane",
        u"var": u"Sample variance",
        u"varcoeff": u"Coefficient of variation",
        u"stddev": u"Ecart-type",
        u"sem": u"StdErr",
        u"sum": u"Totalsum",
        u"lower_q": u"Quartile inférieur",
        u"upper_q": u"Quartile supérieur"},
    # Italian
    u"it-IT": {
        u"@": u"Total",
        u"All": u"Base",
        u"no_w_All": u"Unweighted base",
        u"gross All": u"Gross base",
        u"no_w_gross_All": u"Unweighted gross base",
        u"mean": u"Mean",
        u"min": u"Min",
        u"max": u"Max",
        u"median": u"Median",
        u"var": u"Sample variance",
        u"varcoeff": u"Coefficient of variation",
        u"stddev": u"Std. dev",
        u"sem": u"Std. err. of mean",
        u"sum": u"Total Sum",
        u"lower_q": u"Lower quartile",
        u"upper_q": u"Upper quartile"},
    # Spanish
    u"es-ES": {
        u"@": u"Total",
        u"All": u"Base",
        u"no_w_All": u"Unweighted base",
        u"gross All": u"Gross base",
        u"no_w_gross_All": u"Unweighted gross base",
        u"mean": u"Mean",
        u"min": u"Min",
        u"max": u"Max",
        u"median": u"Median",
        u"var": u"Sample variance",
        u"varcoeff": u"Coefficient of variation",
        u"stddev": u"Std. dev",
        u"sem": u"Std. err. of mean",
        u"sum": u"Total Sum",
        u"lower_q": u"Lower quartile",
        u"upper_q": u"Upper quartile"},
    # German
    u"de-DE": {
        u"@": u"Gesamt",
        u"All": u"Basis Netto",
        u"no_w_All": u"Ungewichtete Basis Netto",
        u"gross All": u"Basis Brutto",
        u"no_w_gross_All": u"Ungewichtete Basis Brutto",
        u"mean": u"Mittelwert",
        u"min": u"Min",
        u"max": u"Max",
        u"median": u"Median",
        u"var": u"Stichprobenvarianz",
        u"varcoeff": u"Variationskoeffizient",
        u"stddev": u"StdDev",
        u"sem": u"StdErr",
        u"sum": u"Summe",
        u"lower_q": u"25% Perzentil",
        u"upper_q": u"75% Perzentil"}
}


KNOWN_METHODS = {
    "default": {
        "method": "default",
        "kwargs": {
            "text": ""
        },
    },
    # BASES
    "cbase": {
        "method": "frequency",
        "kwargs": {
            "text": "Base",
            "axis": "x",
            "condition": "x"
        }
    },
    "cbase_gross": {
        "method": "frequency",
        "kwargs": {
            "text": "Gross base",
            "axis": "x",
            "condition": "x",
            "ignore_flags": True
        }
    },
    "rbase": {
        "method": "frequency",
        "kwargs": {
            "text": "Base",
            "axis": "y",
            "condition": "y"
        }
    },
    "ebase": {
        "method": "frequency",
        "kwargs": {
            "text": "Effective Base",
            "axis": "x",
            "condition": "x",
            "effective": True
        }
    },
    # COUNTS
    "counts": {
        "method": "frequency",
        "kwargs": {
            "text": "",
            "axis": None
        }
    },
    # PERCENTAGES
    "c%": {
        "method": "frequency",
        "kwargs": {
            "text": "",
            "axis": None,
            "rel_to": "y"
        }
    },
    "r%": {
        "method": "frequency",
        "kwargs": {
            "text": "",
            "axis": None,
            "rel_to": "x"
        }
    },
    "res_c%": {
        "method": "frequency",
        "kwargs": {
            "text": "",
            "axis": None,
            "rel_to": "counts_sum"
        }
    },
    # SUMS
    "counts_sum": {
        "method": "frequency",
        "kwargs": {
            "text": "Total Sum",
            "axis": "x",
            "condition": "x",
            "raw_sum": True
        }
    },
    "c%_sum": {
        "method": "frequency",
        "kwargs": {
            "text": "Total Sum",
            "axis": "x",
            "condition": "x",
            "rel_to": "y",
            "raw_sum": True
        }
    },
    # CUMULATIVE SUMS
    "counts_cumsum": {
        "method": "frequency",
        "kwargs": {
            "text": "",
            "axis": None,
            "condition": "x++",
            "cum_sum": True
        }
    },
    "c%_cumsum": {
        "method": "frequency",
        "kwargs": {
            "text": "",
            "axis": None,
            "condition": "x++",
            "rel_to": "y",
            "cum_sum": True
        }
    },
    # DESCRIPTIVES
    "mean": {
        "method": "descriptives",
        "kwargs": {
            "text": "",
            "axis": "x"
        }
    },
    "stddev": {
        "method": "descriptives",
        "kwargs": {
            "text": "",
            "axis": "x",
            "stats": "stddev"
        }
    },
    "min": {
        "method": "descriptives",
        "kwargs": {
            "text": "",
            "axis": "x",
            "stats": "min"
        }
    },
    "max": {
        "method": "descriptives",
        "kwargs": {
            "text": "",
            "axis": "x",
            "stats": "max"
        }
    },
    "median": {
        "method": "descriptives",
        "kwargs": {
            "text": "",
            "axis": "x",
            "stats": "median"
        }
    },
}
