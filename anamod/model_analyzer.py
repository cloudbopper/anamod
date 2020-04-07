"""Python API to analyze temporal models"""
from abc import ABC
import csv
import tempfile

import anytree
import cloudpickle
import h5py

from anamod import master, constants


class ModelAnalyzer(ABC):
    """Analyzes properties of learned models"""
    def __init__(self, model, data, targets, **kwargs):
        """
        Model:   must provide required interface
        Data:    feature matrix/tensor
        Targets: vector of targets/labels
        """
        # FIXME: Temporary directory may not be in shared location on shared FS
        self.output_dir = kwargs.pop("output_dir") if "output_dir" in kwargs else tempfile.mkdtemp()
        self.model_filename = self.gen_model_file(model)
        self.data_filename = self.gen_data_file(data, targets)
        # TODO: Add optional feature names
        # TODO: Add -condor 1 -shared_filesystem 0 iff htcondor available
        self.cmd = f"python -m anamod -model_filename {self.model_filename} -data_filename {self.data_filename} -output_dir {self.output_dir}"
        self.add_options(**kwargs)

    def add_options(self, **kwargs):
        """Add options to model analysis"""
        for key, value in kwargs.items():
            self.cmd += f" -{key} {value}"

    def analyze(self):
        """Analyze model"""
        strargs = " ".join(self.cmd.split()[3:])
        features = master.main(strargs)
        return features

    def gen_model_file(self, model):
        """Generate model file"""
        model_filename = f"{self.output_dir}/{constants.MODEL_FILENAME}"
        with open(model_filename, "wb") as model_file:
            cloudpickle.dump(model, model_file)
        return model_filename

    def gen_data_file(self, data, targets):
        """Generate data file"""
        data_filename = f"{self.output_dir}/{constants.DATA_FILENAME}"
        root = h5py.File(data_filename, "w")
        num_instances = data.shape[0]
        record_ids = [str(idx).encode("utf8") for idx in range(num_instances)]
        root.create_dataset(constants.RECORD_IDS, data=record_ids)
        root.create_dataset(constants.DATA, data=data)
        root.create_dataset(constants.TARGETS, data=targets)
        root.close()
        return data_filename


class TemporalModelAnalyzer(ModelAnalyzer):
    """Analyzes properties of temporal model"""
    def __init__(self, model, data, targets, **kwargs):
        super().__init__(model, data, targets, **kwargs)
        analysis_type = kwargs.get("analysis_type", constants.TEMPORAL)
        assert analysis_type == constants.TEMPORAL


class HierarchicalModelAnalyzer(ModelAnalyzer):
    """Analyzes hierarchical feature importance"""
    def __init__(self, model, data, targets, hierarchy, **kwargs):
        """Hierarchy in the form of an anytree root node"""
        super().__init__(model, data, targets, **kwargs)
        analysis_type = kwargs.get("analysis_type", constants.HIERARCHICAL)
        assert analysis_type == constants.HIERARCHICAL
        self.hierarchy_filename = self.gen_hierarchy_file(hierarchy)
        self.cmd += f" -hierarchy_filename {self.hierarchy_filename}"

    def gen_hierarchy_file(self, hierarchy):
        """
        Generate hierarchy CSV file (hierarchical feature importance analysis only)

        Columns:
                *name*:             feature name, must be unique across features
                *parent_name*:      name of parent if it exists, else '' (root node)
                *description*:      node description
                *idx*:              [only required for leaf nodes] list of tab-separated indices corresponding to the indices
                                    of these features in the data
        """
        hierarchy_filename = f"{self.output_dir}/hierarchy.csv"
        with open(hierarchy_filename, "w", newline="") as hierarchy_file:
            writer = csv.writer(hierarchy_file, delimiter=",")
            writer.writerow([constants.NODE_NAME, constants.PARENT_NAME,
                             constants.DESCRIPTION, constants.INDICES])
            for node in anytree.PreOrderIter(hierarchy):
                idx = node.idx if node.is_leaf else ""
                parent_name = node.parent.name if node.parent else ""
                writer.writerow([node.name, parent_name, node.description, idx])
        return hierarchy_filename
