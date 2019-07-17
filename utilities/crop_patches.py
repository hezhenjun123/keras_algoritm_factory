import tensorflow as tf
import logging


logging.getLogger().setLevel(logging.INFO)
class CropPatches(tf.keras.layers.Layer):
    r"""Crop the input image and label into batch of patches"""
    def __init__(self, resize, boxes):
        r"""Initialize the CropPatches
        Parameters
        ----------
        resize : tuple of int
            The final size the output images should be, in the format of (H, W)
        boxes : list of lists
            The coordinates for cropping, in relative value
            For example:
            boxes = [
                [0.00, 0.00, 0.25, 0.25],
                [0.25, 0.00, 0.50, 0.25],
            ]
        """
        super(CropPatches, self).__init__()
        self.resize = resize
        self.boxes = boxes
        self.n = len(boxes)

    def _crop_and_resize(self, image):
        r"""Crop the image into patches and resize
        Parameters
        ----------
        image : tensor of image
            Should be in shape [1, H, W, D] or [H, W, D]
        Returns
        -------
        Batches of cropped patches of the image in shape [N, H, W, D]
        """
        # Convert to [1, h, w, d] to feed into crop_and_resize
        if image.shape.ndims == 3:
            image = tf.expand_dims(image, 0)
        # image = tf.expand_dims(image, 0)
        cropped_images = tf.image.crop_and_resize(
            image,
            boxes=self.boxes,
            box_ind=[0] * self.n,
            crop_size=self.resize,
            method="nearest"
        )
        cropped_images = tf.cast(cropped_images, image.dtype)
        return cropped_images

    def call(self, image, label=None):
        r"""Run the CropPatches
        Parameters
        ----------
        image : tensor of image
            Should be in shape [1, H, W, D] or [H, W, D]
        label : tensor of label
            Should be in shape [1, H, W, D] or [H, W, D]
        Returns
        -------
        Batches of cropped patches of the image and label
        """
        cropped_images = self._crop_and_resize(image)
        if label is not None:
            cropped_labels = self._crop_and_resize(label)
        else:
            cropped_labels = None
        return cropped_images, cropped_labels


class CropAndExpand(tf.keras.layers.Layer):
    r"""Function to Crop the images and labels into patches and expand the
    input data records"""
    def __init__(self, resize, boxes):
        r"""Initialize CropAndExpand
        Parameters
        ----------
        resize : tuple of int
            The final size the output images should be, in the format of (H, W)
        boxes : list of lists
            The coordinates for cropping, in relative value
            For example:
            boxes = [
                [0.00, 0.00, 0.25, 0.25],
                [0.25, 0.00, 0.50, 0.25],
            ]
        """
        super(CropAndExpand, self).__init__()
        self.resize = resize
        self.boxes = boxes
        self.n = len(boxes)
        self.crop_patches = CropPatches(resize, boxes)

    def call(self, row):
        r"""Execute the crop and expand
        Parameters
        ----------
        row : data record with values `image`, `label`, `name` and `defects`
        Returns
        -------
        The expanded data records
        """
        image = row["image"]
        label = row["segmentation_labels"]
        cropped_images, cropped_labels = self.crop_patches(image, label)
        row["image"] = cropped_images
        row["segmentation_labels"] = cropped_labels

        return row