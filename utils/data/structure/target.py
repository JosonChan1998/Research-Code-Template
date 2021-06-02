from .base_label import BaseLabel

__all__ = ["Target"]

class Target(object):
    def __init__(self):
        self.fields = {}
    
    def add_field(self, field, field_data):
        if isinstance(field_data, BaseLabel):
            self.fields[field] = field_data
        else:
            raise ValueError("No BaseLabel class!")
    
    def get_field(self, field):
        return self.fields[field]
    
    def has_field(self, field):
        return field in self.fields

    def fields(self):
        return list(self.fields.keys())

    def set_size(self, size):
        for k, v in self.fields.items():
            self.fields[k] = v.set_size(size)
        return self

    def move(self, gap):
        for k, v in self.fields.items():
            self.fields[k] = v.move(gap)
        return self

    def resize(self, size):
        for k, v in self.fields.items():
            self.fields[k] = v.resize(size)
        return self

    def corp(self, box):
        for k, v in self.fields.items():
            self.fields[k] = v.corp(box)
        return self
    
    def transpose(self, method):
        for k, v in self.fields.items():
            self.fields[k] = v.transpose(method)
        return self