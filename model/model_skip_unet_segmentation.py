import tensorflow as tf
import logging
import segmentation_models as sm
sm.set_framework('tf.keras')

from model.model_base import ModelBase

logger = logging.getLogger(__name__)


class ModelSkipUnetSegmentation(ModelBase):

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
        if self.backbone == "mobilenet-v2":
            if self.layer_count!=3:
                raise ValueError('Layer Count must equal 3 for Mobilenet Backend due to downsampling sizes')
            
            mobilenet_v2 = tf.keras.applications.MobileNetV2(input_shape=self.image_shape,
                                                            include_top=False,
                                                            weights='imagenet')

            mobilenet_v2.trainable = self.is_backbone_trainable
            hidden = mobilenet_v2(inputs)
            hidden = tf.keras.layers.Conv2D(filters=self.layer_size,
                                    kernel_size=3,
                                    activation='relu',
                                    padding='same')(hidden)
        elif self.backbone == 'nobackbone':
            hidden = None
        else:
            raise Exception("Backbone didn't setup")
        
        processed_inputs = tf.keras.layers.Conv2D(filters=self.layer_size,
                                                kernel_size=3,
                                                activation='relu',
                                                padding='same')(inputs)
        downsampled = [processed_inputs]
        for layer_resize_factor in self.get_resize_factors():
            downsampled.append(self.downsample_layer(downsampled[-1],self.layer_size, layer_resize_factor))
        for layer_resize_factor, downsampled_input in zip(reversed(self.get_resize_factors()),
                                                                    reversed(downsampled[1:])):
            hidden = self.upsample_layer(hidden, downsampled_input, self.layer_size, layer_resize_factor)
        hidden = tf.keras.layers.Conv2D(filters=1,
                                        padding='same',
                                        activation='sigmoid',
                                        name="probabilities",
                                        kernel_size=3)(hidden)
        upsampled = hidden
        model = tf.keras.Model(inputs=inputs, outputs=[upsampled], name="segmentation_model")
        logging.info(model.summary())
        return model

    def downsample_layer(self, vertical, layer_size, resize_factor):
        residual = tf.keras.layers.MaxPool2D(pool_size=(resize_factor, resize_factor))(vertical)
        hidden = tf.keras.layers.Conv2D(filters=layer_size,
                                        kernel_size=3,
                                        activation='relu',
                                        padding='same')(vertical)
        hidden = tf.keras.layers.MaxPool2D(pool_size=(resize_factor, resize_factor))(hidden)
        hidden = tf.keras.layers.BatchNormalization()(hidden)
        hidden = tf.keras.layers.Add()([hidden, residual])
        return hidden

    def upsample_layer(self, vertical, horizontal, layer_size, resize_factor):
        if vertical is not None: 
            residual = tf.keras.layers.UpSampling2D(size=(resize_factor, resize_factor))(vertical)
            hidden = tf.keras.layers.concatenate([vertical, horizontal])
        else: 
            hidden=horizontal
        hidden = tf.keras.layers.UpSampling2D(size=(resize_factor, resize_factor))(hidden)
        hidden = tf.keras.layers.Conv2D(filters=layer_size,
                                        kernel_size=3,
                                        activation='relu',
                                        padding='same')(hidden)
        hidden = tf.keras.layers.BatchNormalization()(hidden)
        if vertical is not None:
            hidden = tf.keras.layers.Add()([hidden, residual])
        return hidden

    def get_resize_factors(self):
        prime_factors = [2]+[4]*(self.layer_count-1)
        return prime_factors

