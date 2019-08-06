class TransformBase:

    def __init__(self, config):
        self.transform = {"IMAGE_ONLY": None, "IMAGE_LABEL": None, "LABEL_ONLY": None}
        self.config = config

    def apply_transforms(self, image, label):
        if self.transform["IMAGE_ONLY"] is not None:
            for transform_func in self.transform["IMAGE_ONLY"]:
                image = transform_func(image=image)["image"]
        if self.transform["IMAGE_LABEL"] is not None:
            for transform_func in self.transform["IMAGE_LABEL"]:
                augmented = transform_func(image=image, mask=label)
                image = augmented["image"]
                label = augmented["mask"]
        if self.transform["LABEL_ONLY"] is not None:
            for transform_func in self.transform["LABEL_ONLY"]:
                label = transform_func(mask=label)["mask"]
        return [image, label]

    def has_transform(self):
        if self.transform is not None and \
                (self.transform["IMAGE_ONLY"] is not None or
                 self.transform["IMAGE_LABEL"] is not None or
                 self.transform["LABEL_ONLY"] is not None):
            return True
        else:
            return False

    def has_image_transform(self):
        if self.transform is not None and self.transform["IMAGE_ONLY"] is not None:
            return True
        else:
            return False
