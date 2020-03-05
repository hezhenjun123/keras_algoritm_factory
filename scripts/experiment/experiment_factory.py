import platform
from experiment.experiment_small_segmentation_unet import ExperimentSmallSegmentationUnet
if platform.machine() != 'aarch64':
    from experiment.experiment_classification import ExperimentClassification
    from experiment.experiment_regression import ExperimentRegression
    from experiment.experiment_bbox import ExperimentBbox
    from experiment.experiment_segmentation_unet import ExperimentSegmentationUnet

# We won't do formal any training on xavier, just in order to quickly generate and run small models on xavier.

class ExperimentFactory:

    if platform.machine() != 'aarch64':
        experiment_registry = {
            "ExperimentClassification": ExperimentClassification,
            "ExperimentRegression": ExperimentRegression,
            "ExperimentSegmentationUnet": ExperimentSegmentationUnet,
            "ExperimentBbox":ExperimentBbox,
            "ExperimentSmallSegmentationUnet": ExperimentSmallSegmentationUnet,
        }
    else:
        experiment_registry = {
            "ExperimentSmallSegmentationUnet": ExperimentSmallSegmentationUnet,
        }

    def __init__(self, config):
        self.config = config

    def create_experiment(self, name):
        if name not in self.experiment_registry:
            raise Exception(f"transform type is not supported: {name}")
        return self.experiment_registry[name](self.config)
