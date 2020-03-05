import tensorflow as tf


class FocalLoss:
    """Special focal loss for classification outputs in RetinaNet.
    From paper: Focal Loss for Dense Object Detection
    (https://arxiv.org/pdf/1708.02002.pdf)
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        """Initializes the loss.
        Parameters
        ----------
        alpha: float
            Scale the focal weight with alpha.
        gamma: float
            Take the power of the focal weight with gamma.
        """
        self.alpha = alpha
        self.gamma = gamma
        self.__name__ = "FocalLoss"

    def __call__(self, y_true, y_pred):
        """Compute the focal loss
        Parameters
        ----------
        y_true: tf.Tensor
            Tensor of target data with shape (B, N, num_classes + 1).
            num_classes + 1 = (object_class1, object_class2,..., object_classN,
                , anchor_state)
            anchor_state: -1 for ignore, 0 for background, 1 for object
        y_pred: tf.Tensor
            Tensor of predicted data with shape (B, N, num_classes). The
            values are sigmoid outputs, which is in the range [0,1]
        Returns
        -------
        tf.Tensor
            The focal loss of y_pred w.r.t. y_true.
        """
        labels = y_true[:, :, :-1]
        anchor_state = y_true[:, :, -1]
        classification = y_pred

        # filter out "ignore" anchors
        indices = tf.where(tf.not_equal(anchor_state, -1))
        labels = tf.gather_nd(labels, indices)
        classification = tf.gather_nd(classification, indices)

        # compute the focal loss
        alpha_factor = tf.ones_like(labels) * self.alpha
        alpha_factor = tf.where(tf.equal(labels, 1), alpha_factor, 1 - alpha_factor)
        focal_weight = tf.where(tf.equal(labels, 1), 1 - classification, classification)
        focal_weight = alpha_factor * focal_weight ** self.gamma

        cls_loss = focal_weight * tf.keras.backend.binary_crossentropy(
            labels, classification
        )

        # compute the normalizer: the number of positive anchors
        normalizer = tf.where(tf.equal(anchor_state, 1))
        normalizer = tf.maximum(1, tf.shape(normalizer)[0])
        normalizer = tf.cast(normalizer, tf.float32)

        return tf.reduce_sum(cls_loss) / normalizer