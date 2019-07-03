import tensorflow as tf


def Dice(from_logits=False, eps=1e-6):
    """Sørensen–Dice coefficient as a loss function for semantic
    segmentation.

    Parameters
    ----------
    from_logits : bool
        Whether the loss is computed from the logits of the network or from
        softmax/sigmoid activations.
    eps : float
        Small value to avoid numerical errors.

    Returns
    -------
    dice_loss : function
        Function that computes the loss given labels and predictions.
    """

    def dice_loss(y_true, y_pred):
        """Function to be used in Keras' training loop.

        Parameters
        ----------
        y_true : tf.Tensor
            Ground truth labels with shape (batch, height, width, num_classes or 1). In
            the binary case the 4-th dimension is 1.
        y_pred : tf.Tensor
            Model's predictions with shape (batch, height, width, num_classes). In the
            binary case the 4-th dimension is 1.

        Returns
        -------
        loss : tf.Tensor
            Scalar tensor with the dice loss.
        """
        if from_logits:
            if y_pred.shape[-1] == 1:
                y_pred = tf.sigmoid(y_pred)
            else:
                y_pred = tf.nn.softmax(y_pred)
        numerator = 2.0 * tf.reduce_sum(y_pred * y_true, axis=[1, 2, 3])
        denominator = tf.reduce_sum(tf.square(y_true), axis=[1, 2, 3]) + tf.reduce_sum(
            tf.square(y_pred), axis=[1, 2, 3]
        )
        loss = 1 - (numerator + eps) / (denominator + eps)
        return tf.reduce_mean(loss)

    return dice_loss