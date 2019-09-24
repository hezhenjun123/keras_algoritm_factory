import os
import logging
from experiment.experiment_base import ExperimentBase
from data_generators.generator_factory import DataGeneratorFactory
from utilities.smart_checkpoint import SmartCheckpoint
from utilities.bbox_eval_callbacks import EvaluateBboxCallback,RedirectModel
from utilities.cos_anneal import CosineAnnealingScheduler
import tensorflow as tf
from loss.focal_loss import FocalLoss
from loss.l1_loss import SmoothL1Loss
logging.getLogger().setLevel(logging.INFO)


class ExperimentBbox(ExperimentBase):

    def __init__(self, config):
        super().__init__(config)
        self.learning_rate = config["LEARNING_RATE"]

    def run_experiment(self):
        train_transform, valid_transform = self.generate_transform()
        data_train_split, data_valid_split = self.read_train_csv()
        train_dataset, valid_dataset = self.generate_dataset(data_train_split, data_valid_split,
                                                             train_transform, valid_transform)
        model = self.generate_model()

        callbacks = self.__compile_callback(model.num_classes,
                                            len(data_valid_split),
                                            valid_dataset,
                                            model.RetinaNetBbox())
        kwarg_para = {
            "num_train_data": len(data_train_split),
            "num_valid_data": len(data_valid_split)
        }
        model.fit_model(train_dataset, None, callbacks, **kwarg_para)
        # for batch in train_dataset:
        #     print(tf.reduce_sum(batch[1][1]))
        #     break
            
    def __compile_callback(self,num_classes,num_steps,valid_dataset,inference_model):
        summaries_dir = os.path.join(self.save_dir, "summaries")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=summaries_dir)
        checkpoints_dir = os.path.join(self.save_dir, "checkpoints")
        checkpoint_callback = RedirectModel(SmartCheckpoint(destination_path=checkpoints_dir,
                                              file_format='epoch_{epoch:04d}/cp.hdf5',
                                              save_weights_only=False,
                                              verbose=1,
                                              monitor='val_mean_average_precision',
                                              mode='max',
                                              save_best_only=True),inference_model)
        # create bbox metrics backback
        eval_callback = EvaluateBboxCallback(
            num_classes =  num_classes,
            num_steps = num_steps,
            generator=valid_dataset,
            eval_interval=1,
        )
        eval_callback = RedirectModel(eval_callback, inference_model)
        callbacks = [
            # CosineAnnealingScheduler(20, self.learning_rate),
            eval_callback,
            checkpoint_callback,
            tensorboard_callback,
            ]
        return callbacks

    def model_compile_para(self):
        compile_para = dict()
        compile_para["optimizer"] = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        compile_para["loss"] = {"regression": SmoothL1Loss(), "classification": FocalLoss()}
        return compile_para


    def generate_dataset(self, data_train_split, data_valid_split, train_transform,
                         valid_transform):
        #override base method to have validation generator output original images and annotations
        generator_factory = DataGeneratorFactory(self.config)
        train_generator = generator_factory.create_generator(self.train_generator_name)
        valid_generator = generator_factory.create_generator(self.valid_generator_name)
        train_dataset = train_generator.create_dataset(df=data_train_split,
                                                       transforms=train_transform)
        valid_dataset = valid_generator.create_inference_dataset(df=data_valid_split,
                                                       transforms=valid_transform)
        return [train_dataset, valid_dataset]