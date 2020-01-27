import numpy as np

def imagenet_preprocess(image):
    # from imagenet
    MEAN_RGB = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
    STDDEV_RGB = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))
    image = image / 255.0
    image -= MEAN_RGB
    image /= STDDEV_RGB
    return image

class TransformBase:

    class Image:

        def __init__(self, transform):
            self.transform = transform

        def __call__(self, image, mask):
            augmented = self.transform(image=image)
            return {"image": augmented["image"], "mask": mask}

    class ImageLabel:

        def __init__(self, transform):
            self.transform = transform

        def __call__(self, image, mask):
            augmented = self.transform(image=image, mask=mask)
            return {"image": augmented["image"], "mask": augmented["mask"]}

    class Label:

        def __init__(self, transform):
            self.transform = transform

        def __call__(self, image, mask):
            augmented = self.transform(mask=mask)
            return {"image": image, "mask": augmented["mask"]}

    def __init__(self, config):
        self.config = config
        self.transforms = []

        #FIXME: remove this once all the transform classes migrated to use self.transforms.
        self.transform = {"IMAGE_ONLY": None, "IMAGE_LABEL": None, "LABEL_ONLY": None}

    def apply_transforms(self, image, label):
        if self.transforms:
            for t in self.transforms:
                augmented = t(image=image, mask=label)
                image = augmented["image"]
                label = augmented["mask"]
            return [image, label]
        else:
            return self._apply_transforms_deprecated(image, label)

    def has_transform(self):
        if self._has_transform_deprecated():
            return True

        return len(self.transforms) > 0

    def _apply_transforms_deprecated(self, image, label):
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

    def _has_transform_deprecated(self):
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
