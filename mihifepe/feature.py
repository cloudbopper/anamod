"""Feature class"""

import anytree

from mihifepe import constants


class Feature(anytree.Node):
    """Class representing feature/feature group"""
    def __init__(self, name, **kwargs):
        super().__init__(name)
        self.parent_name = kwargs.get(constants.PARENT_NAME, "")
        self.description = kwargs.get(constants.DESCRIPTION, "")
        self.static_indices = kwargs.get(constants.STATIC_INDICES, [])
        self.temporal_indices = kwargs.get(constants.TEMPORAL_INDICES, [])

    @staticmethod
    def unpack_indices(str_indices):
        """Converts tab-separated string of indices to int list"""
        if not str_indices:
            return []
        return [int(idx) for idx in str_indices.split("\t")]

    @staticmethod
    def pack_indices(int_indices):
        """Converts int list of indices to tab-separated string"""
        return "\t".join([str(idx) for idx in int_indices])
