import platform
from transforms.transform_yield_absolute import TransformYieldAbsolute
from transforms.transform_segmentation import TransformSegmentation

if platform.machine() != 'aarch64':
    from transforms.transform_classification import TransformClassification
    from transforms.transform_bbox import TransformBbox
    from transforms.transform_classification_yield_delta import TransformClassificationYieldDelta

class TransformFactory:
    if platform.machine() != 'aarch64':
        transform_registry = {
            "TransformClassification": TransformClassification,
            "TransformSegmentation": TransformSegmentation,
            "TransformBbox": TransformBbox,
            "TransformClassificationYieldDelta": TransformClassificationYieldDelta,
            "TransformYieldAbsolute": TransformYieldAbsolute
        }
    else:
        transform_registry = {
            "TransformYieldAbsolute": TransformYieldAbsolute,
            "TransformSegmentation": TransformSegmentation,
        }

    def __init__(self, config):
        self.config = config

    def create_transform(self, name):
        if name not in self.transform_registry:
            raise Exception("transform type is not supported: {}".format(name))
        return self.transform_registry[name](self.config)
