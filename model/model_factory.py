import platform
from model.model_resnet_regression import ModelResnetRegression
from model.model_unet_segmentation import ModelUnetSegmentation

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
            "ModelRetinaNet":ModelRetinaNet
        }
    else:
        model_registry = {
            "ModelResnetRegression": ModelResnetRegression,
            "ModelUnetSegmentation": ModelUnetSegmentation,
        }

    def __init__(self, config):
        self.config = config

    def create_model(self, name):
        if name not in self.model_registry:
            raise Exception(f"model type is not supported: {name}")
        return self.model_registry[name](self.config)
