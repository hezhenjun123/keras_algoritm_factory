import tensorflow as tf
import logging

from model.model_base import ModelBase

logger = logging.getLogger(__name__)


class ModelUnetSegmentation(ModelBase):

    def __init__(self, config):
        super().__init__(config)
        self.is_backbone_trainable = self.config["MODEL"]["IS_BACKBONE_TRAINABLE"]
        self.backbone = self.config["MODEL"]["BACKBONE"]
        self.layer_size = self.config["MODEL"]["LAYER_SIZE"]
        self.layer_count = self.config["MODEL"]["LAYER_COUNT"]
        output_shape = self.config["DATA_GENERATOR"]["OUTPUT_SHAPE"]
        output_image_channels = self.config["DATA_GENERATOR"]["OUTPUT_IMAGE_CHANNELS"]
        self.image_shape = (output_shape[0], output_shape[1], output_image_channels)
        self.model = self.get_or_load_model()

    def create_model(self):
        inputs = tf.keras.Input(shape=self.image_shape, name='input_image')
        if self.is_backbone_mobilenet_v2():
            mobilenet_v2 = tf.keras.applications.MobileNetV2(input_shape=self.image_shape,
                                                             include_top=False,
                                                             weights='imagenet')

            mobilenet_v2.trainable = self.is_backbone_trainable
            hidden = mobilenet_v2(inputs)
        else:
            raise Exception("Backbone didn't setup")
        hidden = tf.keras.layers.Conv2D(filters=self.layer_size,
                                        kernel_size=3,
                                        activation='relu',
                                        padding='same')(hidden)
        processed_inputs = tf.keras.layers.Conv2D(filters=self.layer_size,
                                                  kernel_size=3,
                                                  activation='relu',
                                                  padding='same')(inputs)
        downsampled = [processed_inputs]
        for layer_resize_factor in self.get_resize_factors():
            downsampled.append(self.downsample_layer(downsampled[-1], layer_resize_factor))
        for layer_resize_factor, downsampled_input in zip(reversed(self.get_resize_factors()),
                                                          reversed(downsampled[1:])):
            hidden = self.upsample_layer(hidden, downsampled_input, layer_resize_factor)
        hidden = tf.keras.layers.Conv2D(filters=1,
                                        padding='same',
                                        activation='sigmoid',
                                        name="probabilities",
                                        kernel_size=3)(hidden)
        upsampled = hidden
        model = tf.keras.Model(inputs=inputs, outputs=[upsampled], name="segmentation_model")
        logging.info(model.summary())
        return model

    def downsample_layer(self, inputs, resize_factor):
        residual = tf.keras.layers.MaxPool2D(pool_size=(resize_factor, resize_factor))(inputs)
        hidden = tf.keras.layers.Conv2D(filters=self.layer_size,
                                        kernel_size=3,
                                        activation='relu',
                                        padding='same')(inputs)
        hidden = tf.keras.layers.MaxPool2D(pool_size=(resize_factor, resize_factor))(hidden)
        hidden = tf.keras.layers.BatchNormalization()(hidden)
        hidden = tf.keras.layers.Add()([hidden, residual])
        return hidden

    def upsample_layer(self, hidden, inputs, resize_factor):
        residual = tf.keras.layers.UpSampling2D(size=(resize_factor, resize_factor))(hidden)
        hidden = tf.keras.layers.concatenate([inputs, hidden])
        hidden = tf.keras.layers.UpSampling2D(size=(resize_factor, resize_factor))(hidden)
        hidden = tf.keras.layers.Conv2D(filters=self.layer_size,
                                        kernel_size=3,
                                        activation='relu',
                                        padding='same')(hidden)
        hidden = tf.keras.layers.BatchNormalization()(hidden)
        hidden = tf.keras.layers.Add()([hidden, residual])
        return hidden

    def get_resize_factors(self):
        prime_factors = [2, 4, 4]
        return prime_factors

    def is_backbone_mobilenet_v2(self):
        return self.backbone == "mobilenet-v2"
