from transforms.transform_classiication import TransformClassification
from transforms.transform_segmentation import TransformSegmentation


class TransformFactory:

    transform_registry = {
        "TransformClassification": TransformClassification,
        "TransformSegmentation": TransformSegmentation
    }

    def __init__(self, config):
        self.config = config

    def create_transform(self, name):
        if name not in self.transform_registry:
            raise Exception(f"transform type is not supported: {name}")
        return self.transform_registry[name](self.config)
