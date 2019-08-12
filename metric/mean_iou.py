import tensorflow as tf

# def MeanIOU(num_classes, from_logits=False):
#     def mean_iou(y_true, y_pred):
#         y_true = tf.cast(y_true, dtype=tf.float32)
#         y_pred = tf.cast(y_pred, dtype=tf.float32)
#         meaniou = tf.metrics.mean_iou(y_true, y_pred, num_classes=num_classes)
#         return meaniou
#     return mean_iou


def MeanIOU(num_classes, from_logits=False):
    """Mean intersection-over-union as a metric for semantic segmentation. This returns
    a function which takes as input the ground-truth labels and the model's predictions,
    both as 4-D volumes with the classes encoded in one-hot form. In the binary case,
    the "depth" of the voilumes should be 1.

    Parameters
    ----------
    num_classes : int
        Number of classes
    from_logits : bool
        Whether the loss is computed from the logits of the network or from
        softmax/sigmoid activations.

    Returns
    -------
    mean_iou : function
        Function to compute the mean IoU given labels and predictions.
    """

    def mean_iou(y_true, y_pred):
        """Function to be used in Keras' training loop.

        Parameters
        ----------
        y_true : tf.Tensor
            Ground truth labels with shape (batch, height, width, num_classes
            or 1). In the binary case the 4-th dimension is 1.
        y_pred : tf.Tensor
            Model's predictions with shape (batch, height, width, num_classes
            or 1). In the binary case the 4-th dimension is 1.

        Returns
        -------
        miou : tf.Tensor
            Scalar tensor with the mean intersection-over-union metric.
        """

        if y_pred.shape[-1] == 1:
            if from_logits:
                y_pred = tf.sigmoid(y_pred)
            y_pred = tf.round(y_pred)
        else:
            if from_logits:
                y_pred = tf.nn.softmax(y_pred)
            y_true = tf.argmax(y_true, axis=-1)
            y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=num_classes, dtype=tf.float64)

        sum_over_row = tf.reduce_sum(cm, 0)
        sum_over_col = tf.reduce_sum(cm, 1)
        cm_diag = tf.compat.v1.diag_part(cm)
        denominator = sum_over_row + sum_over_col - cm_diag

        # The mean is only computed over classes that appear in the label or
        # prediction tensor. If the denominator is
        # 0, we need to ignore the class.
        num_valid_entries = tf.reduce_sum(tf.cast(tf.not_equal(denominator, 0), dtype=tf.float64))

        # If the value of the denominator is 0, set it to 1 to avoid zero
        # division.
        denominator = tf.where(denominator > 0, denominator, tf.ones_like(denominator))
        iou = cm_diag / denominator

        # If the number of valid entries is 0 (no classes) we return 0.
        result = tf.where(num_valid_entries > 0, tf.reduce_sum(iou) / num_valid_entries, 0)
        return result

    return mean_iou
