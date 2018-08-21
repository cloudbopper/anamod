"""Feature class"""

import anytree

import constants

class Feature(anytree.Node):
    """Class representing feature/feature group"""
    def __init__(self, name, **kwargs):
        super().__init__(name)
        self.category = kwargs.get(constants.CATEGORY, "")
        self.parent_name = kwargs.get(constants.PARENT_NAME, "")
        self.description = kwargs.get(constants.DESCRIPTION, "")
        self.static_indices = kwargs.get(constants.STATIC_INDICES, [])
        self.temporal_indices = kwargs.get(constants.TEMPORAL_INDICES, [])
        self.identifier = self.get_identifier(self.category, name)

    @staticmethod
    def get_identifier(category, name):
        """
        Construct feature identifier for given (category, name) tuple

        Args:
            category: feature category, acts as namespace
            name: feature name

        Returns:
            identifier
        """
        return "%s:%s" % (category, name)

    @staticmethod
    def unpack_indices(str_indices):
        """Converts tab-separated string of indices to int list"""
        return [int(idx) for idx in "\t".split(str_indices)]

    @staticmethod
    def pack_indices(int_indices):
        """Converts int list of indices to tab-separated string"""
        return "\t".join([str(idx) for idx in int_indices])
