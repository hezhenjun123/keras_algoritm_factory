import albumentations as A
from transforms.transform_base import TransformBase

# TODO(yuanzhedong): check if albumentation is effient.
class TransformYieldAbsolute(TransformBase):

    def __init__(self, config):
        super().__init__(config)
        resize = self.config["TRANSFORM"]["RESIZE"]
        normalize = self.config["TRANSFORM"]["LABEL_NORMALIZE"]
        self.transforms = [TransformBase.Image(A.Resize(resize[0], resize[1])),
                           TransformBase.Image(A.Normalize()),
                           # TODO(yuanzhedong): may need to optimize since label is not needed for inference
                           TransformBase.Label(lambda mask: {'mask': mask/normalize})
                          ]
