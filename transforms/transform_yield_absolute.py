import albumentations as A
from transforms.transform_base import TransformBase


class TransformYieldAbsolute(TransformBase):

    def __init__(self, config):
        super().__init__(config)
        resize = self.config["TRANSFORM"]["RESIZE"]
        normalize = self.config["TRANSFORM"]["LABEL_NORMALIZE"]
        self.transforms = [TransformBase.Image(A.Resize(resize[0], resize[1])),
                           TransformBase.Image(A.Normalize()),
                           TransformBase.Label(lambda mask: {'mask': mask/normalize})
                          ]
