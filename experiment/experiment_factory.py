from experiment.experiment_classification import ExperimentClassification
from experiment.experiment_segmentation_tf1unet import ExperimentSegmentationTF1Unet
from experiment.experiment_segmentation_tf2unet import ExperimentSegmentationTF2Unet


class ExperimentFactory:

    experiment_registry = {
        "ExperimentClassification": ExperimentClassification,
        "ExperimentSegmentationTF1Unet": ExperimentSegmentationTF1Unet,
        "ExperimentSegmentationTF2Unet": ExperimentSegmentationTF2Unet
    }

    def __init__(self, config):
        self.config = config

    def create_experiment(self, name):
        if name not in self.experiment_registry:
            raise Exception(f"transform type is not supported: {name}")
        return self.experiment_registry[name](self.config)
