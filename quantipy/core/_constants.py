
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

# variable types
QP_TYPES = [
    "single", "delimited set", "array", "int", "float", "string", "date",
    "time"]

NUMERIC = ["int", "float"]

STRING = ["string"]

DATE = ["date"]

BOOLEAN = ["boolean"]

CATEGORICAL = ["single", "delimited set"]

