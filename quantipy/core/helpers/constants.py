#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
''' Constant mapping appropriate quantipy types to pandas dtypes
'''
DTYPE_MAP = {
    "float": ["float64", "float32", "float16"],
    "int": ["int64", "int32", "int16", "int8", "int0", "float64", "float32", "float16"],
    "string": ["object"],
    "date": ["datetime64"],
    "time": ["timedelta64"],
    "bool": ["bool"],
    "single": ["int64", "int32", "int16", "int8", "int0", "float64", "float32", "float16"],
    "dichotomous set": [],
    "categorical set": [],
    "delimited set": ["object"],
    "grid": []
}

MAPPED_PATTERN = "^[^@].*[@].*[^@]$"