import albumentations as A
import numpy as np
from transforms.transform_base import TransformBase


class TransformSegmentationChaff(TransformBase):
    def __init__(self, config):
        super().__init__(config)
        resize_size = config["TRANSFORM"]["RESIZE"]
        self.transform["IMAGE_ONLY"] = A.Normalize()
        self.transform["IMAGE_LABEL"] = A.Resize(*resize_size)
        self.transform["LABEL_ONLY"] = self.squash_labels

    def squash_labels(self,mask=[]):
        mask[mask>0]=1
        return {"mask":mask}

    