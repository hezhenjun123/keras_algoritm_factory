from experiment.experiment_classification import ExperimentClassification
from experiment.experiment_segmentation_unet import ExperimentSegmentationUnet


class ExperimentFactory:

    experiment_registry = {
        "ExperimentClassification": ExperimentClassification,
        "ExperimentSegmentationUnet": ExperimentSegmentationUnet
    }

    def __init__(self, config):
        self.config = config

    def create_experiment(self, name):
        if name not in self.experiment_registry:
            raise Exception(f"transform type is not supported: {name}")
        return self.experiment_registry[name](self.config)
