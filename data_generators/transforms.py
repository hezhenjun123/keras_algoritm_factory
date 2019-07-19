import albumentations as A
import numpy as np


class TransformBase:
    def __init__(self, config):
        self.transform = {"ImageTransform": None,
                          "ImageAndSegMapTransform": None,
                          "SegMapTransform": None}
        self.config = config


class TransformSimpleClassification(TransformBase):
    def __init__(self, config):
        super().__init__(config)
        random_crop_size = self.config["TRANSFORM"]["RANDOMCROP"]
        resize = self.config["TRANSFORM"]["RESIZE"]
        tfunc = A.Compose([A.RandomCrop(random_crop_size[0], random_crop_size[1]),
                           A.Resize(resize[0], resize[1])])
        self.transform["ImageTransform"] = tfunc


class TransformUVSegmentation(TransformBase):
    def __init__(self, config):
        super().__init__(config)
        tfunc = A.Compose([A.Normalize()])
        self.transform["ImageTransfor"] = tfunc
        self.transform["SegMapTransform"] = self.threshold_label


    def threshold_label(self, mask=[]):
        mask = mask[:, :, [2]]
        mask = (mask > 255*0.66).astype(np.uint8)
        return {"mask": mask}