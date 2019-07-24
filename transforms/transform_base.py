import numpy as np


class TransformBase:
    def __init__(self, config):
        # transform[0]: Transform only image
        # transform[1]: Transform both image and label
        # transform[2]: Transform only label
        self.transform = [None, None, None]
        self.config = config

    def apply_transforms(self, image, segmentation_labels):
        image = image
        seg_label = segmentation_labels
        if self.transform[0] is not None:
            image = self.transform[0](image=image)["image"]
        if self.transform[1] is not None:
            augmented = self.transform[1](image=image, mask=seg_label)
            image = augmented["image"]
            seg_label = augmented["mask"]
        if self.transform[2] is not None:
            seg_label = self.transform[2](mask=seg_label)["mask"]

        image = image
        seg_label = seg_label
        return [image, seg_label]
