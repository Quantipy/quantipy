def cWriterow(self, record):
    cdef int varType
    cdef float sysmis
    cdef Py_ssize_t i
    sysmis = self.sysmis_
    for i, varName in enumerate(self.varNames):
        varName = self.varNames[i]
        varType = self.varTypes[varName]
        if varType == 0:
            try:
                numValue = float(record[i])
            except ValueError:
                numValue = sysmis
            except TypeError:
                numValue = sysmis
            record[i] = numValue
        else:
            value = record[i]
            if self.ioUtf8_ and isinstance(value, unicode):
                valueX = (<unicode>value).encode("utf-8")
                strValue = valueX
            else:
                strValue = self.pad_8_lookup[varType] % value
            record[i] = strValue
    self.record = record
