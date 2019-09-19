from transforms.transform_classification import TransformClassification
from transforms.transform_segmentation import TransformSegmentation
from transforms.transform_bbox import TransformBbox
from transforms.transform_classification_yield_delta import TransformClassificationYieldDelta
from transforms.transform_yield_absolute import TransformYieldAbsolute


class TransformFactory:

    transform_registry = {
        "TransformClassification": TransformClassification,
        "TransformSegmentation": TransformSegmentation,
        "TransformBbox": TransformBbox,
        "TransformClassificationYieldDelta": TransformClassificationYieldDelta,
        "TransformYieldAbsolute": TransformYieldAbsolute
    }

    def __init__(self, config):
        self.config = config

    def create_transform(self, name):
        if name not in self.transform_registry:
            raise Exception(f"transform type is not supported: {name}")
        return self.transform_registry[name](self.config)
