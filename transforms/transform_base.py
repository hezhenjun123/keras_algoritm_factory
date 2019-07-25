import numpy as np


class TransformBase:
    def __init__(self, config):
        # transform[0]: Transform only image
        # transform[1]: Transform both image and label
        # transform[2]: Transform only label
        self.transform = {
            "IMAGE_ONLY": None,
            "IMAGE_LABEL": None,
            "LABEL_ONLY": None
        }
        self.config = config

    def apply_transforms(self, image, segmentation_labels):
        image = image
        seg_label = segmentation_labels
        if self.transform["IMAGE_ONLY"] is not None:
            image = self.transform["IMAGE_ONLY"](image=image)["image"]
        if self.transform["IMAGE_LABEL"] is not None:
            augmented = self.transform["IMAGE_LABEL"](image=image,
                                                      mask=seg_label)
            image = augmented["image"]
            seg_label = augmented["mask"]
        if self.transform["LABEL_ONLY"] is not None:
            seg_label = self.transform["LABEL_ONLY"](mask=seg_label)["mask"]

        image = image
        seg_label = seg_label
        return [image, seg_label]

    def has_transform(self):
        if self.transform is not None and \
                (self.transform["IMAGE_ONLY"] is not None or
                 self.transform["IMAGE_LABEL"] is not None or
                 self.transform["LABEL_ONLY"] is not None):
            return True
        else:
            return False

    def has_image_transform(self):
        if self.transform is not None and self.transform[
                "IMAGE_ONLY"] is not None:
            return True
        else:
            return False
