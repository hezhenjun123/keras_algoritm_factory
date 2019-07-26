from transforms.transform_simple_classiication import TransformSimpleClassification
from transforms.transform_segmentation_uv import TransformSegmentationUV
from transforms.transform_segmentation_chaff import TransformSegmentationChaff


class TransformFactory:

    transform_registry = {
        "TransformSimpleClassification": TransformSimpleClassification,
        "TransformSegmentationUV": TransformSegmentationUV,
        "TransformSegmentationChaff": TransformSegmentationChaff
    }

    def __init__(self, config):
        self.config = config

    def create_transform(self, name):
        if name not in self.transform_registry:
            raise Exception(f"transform type is not supported: {name}")
        return self.transform_registry[name](self.config)
