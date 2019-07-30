import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class LayerNormalization(tf.keras.layers.Layer):
    """Normalize a layer's activation through the channels. From 'Layer Normalization'
    (Lei Ba et al). https://arxiv.org/abs/1607.06450."""

    def __init__(self, eps=1e-6, **kwargs):
        """Initializes the layer.

        Parameters
        ----------
        eps : float
            Small value to avoid numerical errors.
        """
        super().__init__(**kwargs)
        self.eps = eps

    def build(self, input_shape):
        channels = int(input_shape[-1])
        self.gamma = self.add_weight(name="gamma",
                                     shape=(1, 1, 1, channels),
                                     initializer="ones",
                                     trainable=True)
        self.beta = self.add_weight(name="beta",
                                    shape=(1, 1, 1, channels),
                                    initializer="zeros",
                                    trainable=True)

    def call(self, x):
        mean, var = tf.nn.moments(x, axes=-1, keep_dims=True)
        x_norm = (x - mean) / tf.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {'eps': self.eps}


class ResizeBilinear(tf.keras.layers.Layer):
    """Layer to resize images using bilinear interpolation."""

    def __init__(self, factor=None, target_shape=None, **kwargs):
        """Initializes the resize layer. You should only provide one of factor or
        target_shape.

        Parameters
        ----------
        factor : float
            Scale factor such that the new image size is factor times that of the
            original.
        target_shape : tuple of ints
            Desired shape (height, width) of the output.
        """
        super().__init__(**kwargs)
        if (factor is None) and (target_shape is None):
            raise ValueError("Please provide either factor or target_shape.")
        if (factor is not None) and (target_shape is not None):
            raise ValueError(
                "Please provide only one of factor or target_shape.")
        self.factor = factor
        self.target_shape = target_shape

    def build(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError(
                f"expected image input (4 dims), got {len(input_shape)}.")
        height, width = map(int, input_shape[1:3])
        if self.factor is not None:
            self.target_shape = (
                round(self.factor * height),
                round(self.factor * width),
            )

    def call(self, inputs):
        return tf.image.resize_bilinear(inputs,
                                        size=self.target_shape,
                                        align_corners=True)

    def compute_output_shape(self, input_shape):
        if self.factor is not None:
            height = (None if input_shape[1] is None else int(
                round(self.factor * input_shape[1])))
            width = (None if input_shape[2] is None else int(
                round(self.factor * input_shape[2])))
        else:
            height, width = self.target_shape
        return [input_shape[0], height, width, input_shape[3]]

    def get_config(self):
        return {'factor': self.factor, 'target_shape': self.target_shape}


def UNet(
        input_shape,
        channel_list=None,
        num_classes=2,
        return_logits=False,
        activation="relu",
        dropout_prob=0.1,
        dropout_type="spatial",
        name="unet",
        input_name="images",
        output_name="seg_map",
        conv_block="default",
        normalization="layernorm",
):
    """Initializes the U-Net model. From 'U-Net: Convolutional Networks for
    Biomedical Image Segmentation' (Ronneberger et al).
    https://arxiv.org/abs/1505.04597.

    Parameters
    ----------
    input_shape : tuple of ints
        Shape of the images to segment. Must have 4 elements (batch, height,
        width, channels).
    channel_list : list of ints
        Number of channels per convolutional block. Defaults to [64, 128, 256,
        512, 1024] as in the U-Net paper.
    num_classes : int
        Number of classes of the segmentation problem.
    return_logis : bool
        Whether the last layer outputs logits or class predictions.
    activation : str
        Non-linearity applied after each convolution.
    dropout_prob : float
        Dropout rate to use. Use False if you don't want dropout. This parameter is only
        considered when conv_block == "default" and dropout_type != None.
    dropout_type : str or None
        Type of dropout to use. Can be "standard" (https://arxiv.org/abs/1207.0580),
        "spatial" (https://arxiv.org/abs/1411.4280), or None for no dropout.
    name : str
        Name of the model.
    input_name : str
        Name of the input layer.
    output_name : str
        Name of the output layer.
    conv_block : str or tf.keras.models.Model
        Convolutional block to use when building U-Net, defaults to the one used by the
        authors, which is two 3x3 convolutions. Otherwise it should be a Keras model
        that takes the number of filters as its sole parameter.
    normalization : str or None
        Type of normalization to use, can be "layernorm"
        (https://arxiv.org/abs/1607.06450), "batchnorm"
        (https://arxiv.org/abs/1502.03167), or None for no normalization. This parameter
        is only considered when conv_block == "default".

    Returns
    -------
    model : tf.keras.models.Model
        U-Net model instance.
    """
    if channel_list is None:
        channel_list = [64, 128, 256, 512, 1024]
    if conv_block == "default":
        ConvBlock = lambda channels: DefaultConvBlock(
            channels,
            activation=activation,
            dropout_prob=dropout_prob,
            normalization=normalization,
            dropout_type=dropout_type,
        )
    else:
        ConvBlock = conv_block

    assert len(input_shape) == 3, "expected image input (3 dims)"
    assert len(
        channel_list) > 0, "expected at least one channel in channel_list"
    assert num_classes >= 2, "expected 2 or more classes"

    inputs = tf.keras.layers.Input(shape=input_shape, name=input_name)
    x = _build_unet(inputs, channel_list, ConvBlock, activation=activation)
    last_activation = "sigmoid" if num_classes == 2 else "softmax"
    x = tf.keras.layers.Conv2D(
        filters=1 if num_classes == 2 else num_classes,
        kernel_size=1,
        padding="same",
        activation=None if return_logits else last_activation,
        name=output_name,
    )(x)
    return tf.keras.models.Model(inputs=inputs, outputs=x, name=name)


def _build_unet(x, channel_list, ConvBlock, activation="relu"):
    """Recursively build U-Net.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor.
    channel_list : list of ints
        Number of channels per convolutional block.
    ConvBlock : tf.keras.models.Model
        Convolutional block to use when building U-Net.

    Returns
    -------
    output : tf.Tensor
        Output of the convolutional block defined by the first channel and it's
        descendants.
    """
    if len(channel_list) == 1:
        return ConvBlock(channel_list[0])(x)
    skip = ConvBlock(channel_list[0])(x)
    x = tf.keras.layers.MaxPool2D(pool_size=2)(skip)
    x = _build_unet(x, channel_list[1:], ConvBlock, activation)
    x = tf.keras.layers.Conv2DTranspose(
        filters=channel_list[0],
        kernel_size=3,
        strides=2,
        activation=activation,
        padding="same",
    )(x)
    x = ResizeBilinear(target_shape=skip.shape[1:3])(x)
    x = tf.keras.layers.Concatenate()([skip, x])
    x = ConvBlock(channel_list[0])(x)
    return x


def DefaultConvBlock(
        channels,
        activation="relu",
        dropout_prob=0.2,
        dropout_type="spatial",
        normalization="layernorm",
        **kwargs,
):
    """Initializes the convolutional block functional.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor.
    channel : int
        Number of channels.
    activation : str
        Non-linearity applied after each convolution.
    dropout_prob : float
        Dropout rate to use. Use False if you don't want dropout. This parameter is
        only considered when conv_block == "default" and dropout_type != None.
    dropout_type : str or None
        Type of dropout to use. Can be "spatial" (https://arxiv.org/abs/1411.4280),
        "standard" (https://arxiv.org/abs/1207.0580), or None for no dropout.
    normalization : str
        Type of normalization to use, can be "layernorm" or "batchnorm".
    """
    if dropout_type == "spatial":
        Dropout = tf.keras.layers.SpatialDropout2D
    elif dropout_type == "standard":
        Dropout = tf.keras.layers.Dropout
    elif dropout_type is None:
        Dropout = None
    else:
        raise ValueError(f"Unknown dropout type: {dropout_type}.")

    if normalization == "batchnorm":
        Norm = tf.keras.layers.BatchNormalization
    elif normalization == "layernorm":
        Norm = LayerNormalization
    elif normalization is None:
        Norm = None
    else:
        raise ValueError(f"Unknown normalization: {normalization}.")

    net_layers = []
    for _ in range(2):
        net_layers.append(
            tf.keras.layers.Conv2D(filters=channels,
                                   kernel_size=(3, 3),
                                   padding="same",
                                   use_bias=False))
        if Norm is not None:
            net_layers.append(Norm())
        net_layers.append(tf.keras.layers.Activation(activation))
        if Dropout is not None:
            net_layers.append(Dropout(dropout_prob))

    def net(x):
        for layer in net_layers:
            x = layer(x)
        return x

    return net
