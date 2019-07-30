from model.model_resnet_classification import ModelResnetClassification
from model.model_unet_segmentation_uv import ModelUnetSegmentationUV
from model.model_unet_tf1_segmentation import ModelUnetTF1Segmentation
from model.model_unet_backbone_segmentation import BackboneUNetModel

class ModelFactory:

    model_registry = {
        "ModelResnetClassification": ModelResnetClassification,
        "ModelUnetSegmentationUV": ModelUnetSegmentationUV,
        "ModelUnetTF1Segmentation": ModelUnetTF1Segmentation,
        "BackboneUNetModel": BackboneUNetModel
    }

    def __init__(self, config):
        self.config = config

    def create_model(self, name):
        if name not in self.model_registry:
            raise Exception(f"model type is not supported: {name}")
        return self.model_registry[name](self.config)
