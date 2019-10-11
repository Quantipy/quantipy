

class Meta(dict):

    def __init__(self, meta=None, text_key="main", dimensions_comp=False):
        super(dict, self).__init__()
        self.path = None
        self.name = name
        self.text_key = None
        self.valid_tks = VALID_TKS
        self._dimensions_comp = dimensions_comp
        self._dimensions_suffix = '_grid'

        if not meta:
            meta = self.start_meta()
        self.update(meta)

    def __getitem__(self, item):
        if isinstance(item, str) and "@" in item:
            item = item.split("@")
            i = item.pop(0)
            obj = dict.__getitem__(self, i)
            while item:
                i = item.pop(0)
                if isinstance(obj, list):
                    i = int(i)
                obj = obj[i]
        else:
            obj = dict.__getitem__(self, item)
        return obj

    def start_meta(self):
        """
        Fills meta dict with basic structure.
        """
        meta = {
            'info': {
                'text': ''
            },
            'lib': {
                'default text': self.text_key,
                'values': {}
            },
            'columns': {},
            'masks': {},
            'sets': {
                'data file': {
                    'text': {self.text_key: 'Variable order in source file'},
                    'items': []
                }
            },
            'type': 'pandas.DataFrame'
        }
        return meta

    def emulate_meta(self, item):
        if isinstance(item, (list, tuple, set)):
            for x, i in enumerate(item):
                item[x] = self.emulate_meta(i)
            return item
        elif isinstance(item, dict):
            for k, i in item.items():
                item[k] = self.emulate_meta(i)
            return item
        elif not isinstance(item, (float, int)) and "@" in item:
            item = self[item]
            item = self.emulate_meta(item)
            return item
        else:
            return item
