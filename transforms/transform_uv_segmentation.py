import albumentations as A
import numpy as np
from transforms.transform_base import TransformBase


class TransformUVSegmentation(TransformBase):
    def __init__(self, config):
        super().__init__(config)
        tfunc = A.Compose([A.Normalize()])
        self.transform["IMAGE_ONLY"] = tfunc
        self.transform["LABEL_ONLY"] = self.threshold_label

    def threshold_label(self, mask=[]):
        mask = mask[:, :, [2]]
        mask = (mask > 255 * 0.66).astype(np.uint8)
        return {"mask": mask}
