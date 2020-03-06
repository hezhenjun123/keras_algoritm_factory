import platform
from transforms.transform_base import TransformBase
if platform.machine() != 'aarch64':
    import albumentations as A
else:
    import cv2
    import numpy as np

class TransformSegmentation(TransformBase):

    def __init__(self, config):
        super().__init__(config)
        resize_size = config["TRANSFORM"]["RESIZE"]
        if platform.machine() != 'aarch64':
            self.transform["IMAGE_ONLY"] = [A.Normalize()]
            self.transform["IMAGE_LABEL"] = [A.Resize(*resize_size)]
            self.transform["LABEL_ONLY"] = [self.squash_labels]
        else:
            self.transforms = [TransformBase.Image(lambda image: {'image' : cv2.resize(image, (resize_size[0], resize_size[1]))}),
                               TransformBase.Image(lambda image: {'image': cv2.normalize(image, None).astype(np.float32)}),
                               TransformBase.Label(lambda mask: {'mask': cv2.resize(mask, (resize_size[0], resize_size[1]))}),
                               TransformBase.Label(self.squash_labels)]
         
    def squash_labels(self, mask=[]):
        mask[mask > 0] = 1
        return {"mask": mask}
