import albumentations as A
from transforms.transform_base import TransformBase


class TransformSimpleClassification(TransformBase):
    def __init__(self, config):
        super().__init__(config)
        random_crop_size = self.config["TRANSFORM"]["RANDOMCROP"]
        resize = self.config["TRANSFORM"]["RESIZE"]
        tfunc = A.Compose([
            A.RandomCrop(random_crop_size[0], random_crop_size[1]),
            A.Resize(resize[0], resize[1])
        ])
        self.transform[0] = tfunc

    def apply_transforms(self, image, segmentation_labels):
        return super().apply_transforms(image, segmentation_labels)