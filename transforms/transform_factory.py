from transforms.transform_simple_classiication import TransformSimpleClassification
from transforms.transform_uv_segmentation import TransformUVSegmentation


class TransformFactory:

    transform_registry = {
        "TransformSimpleClassification": TransformSimpleClassification,
        "TransformUVSegmentation": TransformUVSegmentation
    }

    def __init__(self, config):
        self.config = config

    def create_transform(self, name):
        if name not in self.transform_registry:
            raise Exception(f"transform type is not supported: {name}")
        return self.transform_registry[name](self.config)
