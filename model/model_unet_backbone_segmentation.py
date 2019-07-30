import tensorflow as tf
import os
import logging
import math

from model.model_base import ModelBase

logger = logging.getLogger(__name__)

class BackboneUNetModel(ModelBase):
    def __init__(self, config):
        self.config = config
        super().__init__(config)
        self.is_backbone_trainable = self.config["MODEL"]["IS_BACKBONE_TRAINABLE"]
        self.backbone = self.config["MODEL"]["BACKBONE"]
        self.layer_size = self.config["MODEL"]["LAYER_SIZE"]
        self.model_directory = self.config["MODEL"]["MODEL_DIRECTORY"]
        self.checkpoint_directory = self.config["MODEL"]["CHECKPOINT_DIRECTORY"]
        self.layer_count = self.config["MODEL"]["LAYER_COUNT"]
        output_shape = self.config["DATA_GENERATOR"]["OUTPUT_SHAPE"]
        output_image_channels = self.config["DATA_GENERATOR"]["OUTPUT_IMAGE_CHANNELS"]
        self.image_shape = (output_shape[0], output_shape[1], output_image_channels)
        self.validation_steps = self.config["MODEL"]["VALIDATION_STEPS"]
        self.model = self.get_or_load_model()

    def get_or_load_model(self):
        if self.model_exists():
            return self.load_model()

        return self.create_model()

    def model_exists(self):
        if os.path.basename(self.model_directory) != 'checkpoint':
            return False

        return os.path.exists(self.model_directory)

    def load_model(self):
        logger.debug("Loading model from " + self.model_directory)
        return tf.keras.models.load_model(self.model_directory)

    def create_model(self):
        inputs = tf.keras.Input(shape=self.image_shape, name='input_image')

        if self.is_backbone_mobilenet_v2():
            mobilenet_v2 = tf.keras.applications.MobileNetV2(input_shape=self.image_shape,
                include_top=False, weights='imagenet')

            mobilenet_v2.trainable=self.is_backbone_trainable

            hidden = mobilenet_v2(inputs)
        else:
            hidden = tf.keras.layers.MaxPool2D(pool_size=(32, 32))(inputs)

        hidden = tf.keras.layers.Conv2D(
            filters=self.layer_size,
            kernel_size=3,
            activation='relu',
            padding='same')(hidden)

        processed_inputs = tf.keras.layers.Conv2D(
            filters=self.layer_size,
            kernel_size=3,
            activation='relu',
            padding='same')(inputs)

        downsampled = [processed_inputs]

        for layer_resize_factor in self.get_resize_factors():
            #print("Downsample by " + str(layer_resize_factor))
            downsampled.append(self.downsample_layer(downsampled[-1], layer_resize_factor))

        for layer_resize_factor, downsampled_input in zip(reversed(self.get_resize_factors()), reversed(downsampled[1:])):
            hidden = self.upsample_layer(hidden, downsampled_input, layer_resize_factor)

        hidden = tf.keras.layers.Conv2D(
            filters=1,
            padding='same',
            activation='sigmoid',
            name="probabilities",
            kernel_size=3)(hidden)

        upsampled=hidden

        predictions = tf.keras.layers.Lambda(lambda x : tf.keras.backend.greater(x, 0.5), name="predictions")(upsampled)

        return tf.keras.Model(inputs=inputs, outputs=[upsampled, predictions], name="segmentation_model")


    def downsample_layer(self, inputs, resize_factor):
        residual = tf.keras.layers.MaxPool2D(pool_size=(resize_factor, resize_factor))(inputs)

        hidden = tf.keras.layers.Conv2D(
            filters=self.layer_size,
            kernel_size=3,
            activation='relu',
            padding='same')(inputs)

        hidden = tf.keras.layers.MaxPool2D(pool_size=(resize_factor, resize_factor))(hidden)

        hidden = tf.keras.layers.BatchNormalization()(hidden)

        hidden = tf.keras.layers.Add()([hidden, residual])

        #print(hidden.shape)

        return hidden

    def upsample_layer(self, hidden, inputs, resize_factor):

        residual = tf.keras.layers.UpSampling2D(size=(resize_factor, resize_factor))(hidden)

        hidden = tf.keras.layers.concatenate([inputs, hidden])

        hidden = tf.keras.layers.UpSampling2D(size=(resize_factor, resize_factor))(hidden)

        hidden = tf.keras.layers.Conv2D(
            filters=self.layer_size,
            kernel_size=3,
            activation='relu',
            padding='same')(hidden)

        hidden = tf.keras.layers.BatchNormalization()(hidden)

        hidden = tf.keras.layers.Add()([hidden, residual])

        return hidden

    def get_resize_factors(self):
        total_resize = self.image_shape[0] // 4

        prime_factors = list(sorted(self.get_prime_factors(total_resize)))

        #print(prime_factors)

        assert len(prime_factors) >= self.layer_count, \
            "Not possible to resize by " + str(total_resize) + "x using " + \
            str(self.layer_count) + " layers"

        while len(prime_factors) > self.layer_count:
            prime_factors = list(sorted([prime_factors[0] * prime_factors[1]] + prime_factors[2:]))

        return prime_factors

    # a given number n
    def get_prime_factors(self, n):

        factors = []

        # Print the number of two's that divide n
        while n % 2 == 0:
            factors.append(2)
            n = n / 2

        # n must be odd at this point
        # so a skip of 2 ( i = i + 2) can be used
        for i in range(3,int(math.sqrt(n))+1,2):

            # while i divides n , print i ad divide n
            while n % i== 0:
                factors.append(i)
                n = n / i

        # Condition if n is a prime
        # number greater than 2
        if n > 2:
            factors.append(n)

        return factors


    def fit_model(self, training_data_source, validation_data_source, callbacks, **kwargs):
        print(self.model.summary())
        self.model.fit(
            x=training_data_source,
            epochs=self.epochs,
            callbacks=callbacks,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=validation_data_source)

        self.model.save(self.checkpoint_directory)

    def predict(self, *, x, steps):
        return self.model.predict(x=x, steps=steps)


    def is_backbone_mobilenet_v2(self):
        return self.backbone == "mobilenet-v2"
