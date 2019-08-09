# -* coding: utf-8 -*-

from ..core.builds.excel_formats import ExcelFormats as ExcelFormatsCore
import warnings
warnings.simplefilter('always')


class ExcelFormats(ExcelFormatsCore):
    def __init__(self, *args, **kwargs):
        msg = (
            "Please use 'quantipy.core.builds.xlsx.excel_formats.ExcelFormats'"
            " instead!")
        warnings.warn(msg, DeprecationWarning)
        super(ExcelFormats, self).__init__(*args, **kwargs)
