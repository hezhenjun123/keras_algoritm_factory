import tensorflow as tf
import logging
import segmentation_models as sm
sm.set_framework('tf.keras')
tf.keras.backend.set_image_data_format('channels_last')
from loss.dice import Dice 
from metric.mean_iou import MeanIOU 

from model.model_base import ModelBase

logger = logging.getLogger(__name__)


class ModelUnetSegmentation(ModelBase):

    def __init__(self, config):
        super().__init__(config)
        self.is_backbone_trainable = self.config["MODEL"]["IS_BACKBONE_TRAINABLE"]
        self.backbone = self.config["MODEL"]["BACKBONE"]
        self.layer_sizes = self.config["MODEL"]["LAYER_SIZE"]
        self.layer_count = self.config["MODEL"]["LAYER_COUNT"]
        if not isinstance(self.layer_sizes,list):
            self.layer_sizes = [self.layer_sizes]*self.layer_count
        else:
            self.layer_count = len(self.layer_sizes)
        output_shape = self.config["DATA_GENERATOR"]["OUTPUT_SHAPE"]
        output_image_channels = self.config["DATA_GENERATOR"]["OUTPUT_IMAGE_CHANNELS"]
        self.image_shape = (output_shape[0], output_shape[1], output_image_channels)
        self.model = self.get_or_load_model()

    def create_model(self):
        model = sm.Unet(self.backbone,
                        input_shape =self.image_shape,
                        decoder_filters = list(reversed(self.layer_sizes)),
                        encoder_freeze = not self.is_backbone_trainable,
                        classes = 1 if self.num_classes==2 else self.num_classes,
                        activation = 'sigmoid' if self.num_classes==2 else "softmax",
                        decoder_block_type = "transpose")
        logging.info(model.summary())
        return model
   
    def generate_custom_objects(self):
        self.custom_objects={"dice_loss":Dice(), "mean_iou": MeanIOU(num_classes=self.num_classes)}
