class Labels(object):
    """
    This class represent the labels including different types labels.
    """

    def __init__(self):
        self._fields = {}
    
    def add_field(self, field, field_data):
        self._fields[field] = field_data

    def get_field(self, field):
        return self._fields[field]

    def has_field(self, field):
        return field in self._fields

    def list_field(self):
        return list(self._fields.keys())