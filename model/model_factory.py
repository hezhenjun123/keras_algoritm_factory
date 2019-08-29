from model.model_resnet_classification import ModelResnetClassification
from model.model_resnet_regression import ModelResnetRegression
from model.model_resnet_classification_yield import ModelResnetClassificationYield
from model.model_unet_segmentation import ModelUnetSegmentation


class ModelFactory:

    model_registry = {
        "ModelResnetClassification": ModelResnetClassification,
        "ModelResnetRegression": ModelResnetRegression,
        "ModelResnetClassificationYield": ModelResnetClassificationYield,
        "ModelUnetSegmentation": ModelUnetSegmentation
    }

    def __init__(self, config):
        self.config = config

    def create_model(self, name):
        if name not in self.model_registry:
            raise Exception(f"model type is not supported: {name}")
        return self.model_registry[name](self.config)
