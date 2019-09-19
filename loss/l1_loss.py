import tensorflow as tf


class SmoothL1Loss:
    """Special Smooth L1 Loss for regression outputs in RetinaNet.
    Description of the loss can be found in page 3 here:
    https://mohitjainweb.files.wordpress.com/2018/03/smoothl1loss.pdf
    """

    def __init__(self, sigma=3.0):
        """Initializes the loss.
        Parameters
        ----------
        sigma: float
            defines the point where the loss changes from L2 to L1.
        """
        self.sigma_squared = tf.constant(sigma ** 2, tf.float32)
        self.__name__ = "SmoothL1Loss"

    def __call__(self, y_true, y_pred):
        """Compute the smooth L1 loss
        Parameters
        ----------
        y_true: tf.Tensor
            Tensor of target data with shape (B, N, 4 + 1).
            4 + 1 = (x1, y1, x2, y2, anchor_state)
            anchor_state: -1 for ignore, 0 for background, 1 for object
        y_pred: tf.Tensor
            Tensor of predicted data with shape (B, N, 4).
        Returns
        -------
        tf.Tensor
            The smooth L1 loss of y_pred w.r.t. y_true.
        """
        # separate target and state
        regression = y_pred
        regression_target = y_true[:, :, :-1]
        anchor_state = y_true[:, :, -1]

        # filter to keep only positive anchors
        # (overlaped with ground-truth boxes)
        indices = tf.where(tf.equal(anchor_state, 1))
        regression = tf.gather_nd(regression, indices)
        regression_target = tf.gather_nd(regression_target, indices)

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = regression - regression_target
        regression_diff = tf.math.abs(regression_diff)
        regression_loss = tf.where(
            tf.less(regression_diff, 1.0 / self.sigma_squared),
            0.5 * self.sigma_squared * tf.math.pow(regression_diff, 2),
            regression_diff - 0.5 / self.sigma_squared,
        )

        # compute the normalizer: the number of positive anchors
        normalizer = tf.maximum(1, tf.shape(indices)[0])
        normalizer = tf.cast(normalizer, dtype=tf.float32)
        return tf.reduce_sum(regression_loss) / normalizer