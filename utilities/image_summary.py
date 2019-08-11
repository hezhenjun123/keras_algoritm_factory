import cv2
import io
import matplotlib.cm
import numpy as np
from scipy.special import expit as sigmoid

from PIL import Image
import logging
logging.getLogger().setLevel(logging.INFO)

import tensorflow as tf


class ImageSummary(tf.keras.callbacks.Callback):
    """Callback that adds iamge summaries to an existing tensorboard calback."""

    def __init__(
            self,
            tensorboard_callback,
            data,
            update_freq=10,
            transforms=[None, None, None],
            from_logits=False,
            cmap="viridis",
            **kwargs,
    ):
        """Initializes the callback.

        Parameters
        ----------
        tensorboard_callback: tf.keras.callbacks.Tensorboard
            Instance of the tensorboard callback that's going to be used during training.
        data: list of tuples (np.ndarray, np.ndarray, str)
            Data to plot. It's expected to be a list of triplets. The first element is the image, in uint8 format;
            the second one is the groud truth segmentation as indices (not one-hot); and the third is an unique name.
            The image and the label should have the same size; and include both the batch and the channel dimensions,
            that is, they should be 4-D arrays.
        update_freq: int
            Write the image summaries after this many epochs.
        transforms: function or None
            Function to be applied to the images and labels in `data` before feeding them to the model. Useful when
            normalizing inputs. Please note that these will only be applied once when calling __init__ of this
            callback.
        from_logits: bool
            Whether the model outputs logits.
        cmap: str or np.ndarray
            Colormap to be used when colorizing predictions and labels. If the problem is binary, then a string
            can be provided to use one of matplotlib's colormaps. In the multi-class scenario, provide an array with
            shape (num_classes, 3) mapping from class index to color (as np.uint8). Note that for the model's
            predictions, in the binary case the probabilities are displayed; in the multi-class case the class with
            highest probability is shown.
        """
        super(ImageSummary, self).__init__(**kwargs)
        self.tensorboard_callback = tensorboard_callback
        self.transforms = transforms
        self.from_logits = from_logits
        self.update_freq = update_freq
        if isinstance(cmap, str):
            self.colormap = (matplotlib.cm.get_cmap(cmap)(np.arange(256))[:, :3] * 255).astype(
                np.uint8)
        else:
            self.colormap = cmap
        self._apply_transforms(data, transforms)

    def _apply_transforms(self, data, transforms):
        if not hasattr(data, "__len__"):
            raise ValueError("expected data to be a list.")
        self.data = []
        for i, elem in enumerate(data):
            if not (hasattr(elem, "__len__") and len(elem) == 3):
                raise ValueError(f"({i}-th elem) expected the elements of data to be triplets.")
            image, label, name = elem
            if not isinstance(image, np.ndarray):
                raise ValueError(
                    f"({i}-th elem) the first element of the triplet should be a numpy array.")
            if not isinstance(label, np.ndarray):
                raise ValueError(
                    f"({i}-th elem) the second element of the triplet should be a numpy array.")
            if not isinstance(name, str):
                raise ValueError(
                    f"({i}-th elem) the third element of the triplet should be a string.")
            if image.ndim != 4 or label.ndim != 4:
                raise ValueError(f"({i}-th elem) expected 4-dim arrays.")
            if image.shape[0] != 1 or label.shape[0] != 1:
                raise ValueError(
                    f"({i}-th elem) expected only one image/label, that is first dimension of 1.")
            if image.shape[:3] != label.shape[:3]:
                raise ValueError(
                    f"({i}-th elem) image and label should have the same size (first 3 dims). Got "
                    f"{image.shape} and {label.shape}.")
            if image.dtype != np.uint8:
                raise ValueError(f"({i}-th elem) expected uint8 image, dtype is {image.dtype}.")
            image = image[0, ...]
            label = label[0]

            tr_image, _ = transforms.apply_transforms(image, label)
            label = label[:, :, 0]
            tr_image = np.expand_dims(tr_image, axis=0)
            self.data.append((image, tr_image, label, name))

    def _make_image(self, array):
        """Converts an image array to the protobuf representation neeeded for image
        summaries."""
        height, width, channels = array.shape
        image = Image.fromarray(array)

        output = io.BytesIO()
        image.save(output, format="PNG")
        image_string = output.getvalue()
        output.close()
        return tf.Summary.Image(
            height=height,
            width=width,
            colorspace=channels,
            encoded_image_string=image_string,
        )

    def _colorize(self, indices):
        """Converts an matrix consisting of numbers in the range [0...1] to a color
        image for easier visualiations."""
        assert (indices.ndim == 2), f"expected matrix, got {indices.ndim}-dim array"
        indices = np.round(indices).astype(int)
        return self.colormap[indices]

    def _process_pred(self, pred):
        """Standarizes pred to be an integer array."""
        assert pred.ndim == 3, f"expected 3-dim array, got {pred.ndim}-dim"
        if pred.shape[-1] == 1:  # binary case
            if self.from_logits:
                pred = sigmoid(pred)
            pred = pred[..., 0] * 255
        else:  # multiclass
            pred = np.argmax(pred, axis=-1)
        return pred

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.update_freq != 0:
            return
        summary_values = []
        for i, (image, tr_image, label, name) in enumerate(self.data):
            height, width, _ = image.shape
            separator = np.full(fill_value=255, shape=(height, 5, 3), dtype=np.uint8)
            pred = self.model.predict(tr_image)[0, ...]
            if pred.shape[:2] != image[:2]:
                pred = cv2.resize(pred, (width, height))
                if pred.ndim == 2:
                    pred = np.expand_dims(pred, axis=-1)
            if pred.shape[-1] == 1:
                label = 255 * label
            to_show = np.concatenate(
                [
                    image,
                    separator,
                    self._colorize(self._process_pred(pred)),
                    separator,
                    self._colorize(label),
                ],
                axis=1,  # concat side by side
            )
            summary_values.append(
                tf.Summary.Value(
                    tag=f"Image_Pred_Label/{i}_{name}",
                    image=self._make_image(to_show),
                ))
        summary = tf.Summary(value=summary_values)
        self.tensorboard_callback.writer.add_summary(summary, epoch)
