# -* coding: utf-8 -*-

from ..core.builds.excel import Excel as ExcelCore
import warnings
warnings.simplefilter('always')


class Excel(ExcelCore):
    def __init__(self, *args, **kwargs):
        msg = "Please use 'quantipy.core.builds.xlsx.excel.Excel' instead!"
        warnings.warn(msg, DeprecationWarning)
        super(Excel, self).__init__(*args, **kwargs)
