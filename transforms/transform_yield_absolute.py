import platform
import numpy as np
from transforms.transform_base import TransformBase

if platform.machine() != 'aarch64':
    import albumentations as A
else:
    import cv2

class TransformYieldAbsolute(TransformBase):

    def __init__(self, config):
        super().__init__(config)
        resize = self.config["TRANSFORM"]["RESIZE"]
        normalize = self.config["TRANSFORM"]["LABEL_NORMALIZE"]
        if platform.machine() != 'aarch64':
            self.transforms = [TransformBase.Image(A.Resize(resize[0], resize[1])),
                               TransformBase.Image(A.Normalize()),
                               TransformBase.Label(lambda mask: {'mask': mask/normalize})
            ]
        else:
            # TODO: check whether cv2 output is consistant with albumentations
            self.transforms = [TransformBase.Image(lambda image: {'image' : cv2.resize(image, (resize[0], resize[1]))}),
                               TransformBase.Image(lambda image: {'image': cv2.normalize(image, image).astype(np.float32)})]

            # self.transforms = [TransformBase.Image(lambda image: {'image' : cv2.resize(image, (resize[0], resize[1])).astype(np.float32)}),
            #                    TransformBase.Label(lambda mask: {'mask' : cv2.resize(mask, (resize[0], resize[1])).astype(np.float32)})]
