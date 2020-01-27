import platform
from model.model_resnet_regression import ModelResnetRegression
from model.model_unet_segmentation import ModelUnetSegmentation
from model.model_small_unet_segmentation import ModelSmallUnetSegmentation

if platform.machine() != 'aarch64':
    from model.model_resnet_classification import ModelResnetClassification
    from model.model_resnet_classification_yield import ModelResnetClassificationYield
    from model.model_skip_unet_segmentation import ModelSkipUnetSegmentation
    from model.model_retinanet import ModelRetinaNet

class ModelFactory:

    if platform.machine() != 'aarch64':
        model_registry = {
            "ModelResnetClassification": ModelResnetClassification,
            "ModelResnetRegression": ModelResnetRegression,
            "ModelResnetClassificationYield": ModelResnetClassificationYield,
            "ModelUnetSegmentation": ModelUnetSegmentation,
            "ModelSkipUnetSegmentation": ModelSkipUnetSegmentation,
            "ModelRetinaNet":ModelRetinaNet,
            "ModelSmallUnetSegmentation": ModelSmallUnetSegmentation,
        }
    else:
        model_registry = {
            "ModelResnetRegression": ModelResnetRegression,
            "ModelUnetSegmentation": ModelUnetSegmentation,
            "ModelSmallUnetSegmentation": ModelSmallUnetSegmentation,
        }

    def __init__(self, config):
        self.config = config

    def create_model(self, name):
        if name not in self.model_registry:
            raise Exception("model type is not supported: {}".format(name))
        return self.model_registry[name](self.config)
