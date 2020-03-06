import numpy as np
import pandas as pd
import logging
import tensorflow as tf
from data_generators.generator_base import DataGeneratorBase

logging.getLogger().setLevel(logging.INFO)


class GeneratorClassification(DataGeneratorBase):

    def create_dataset(self, df, transforms):
        if isinstance(df, pd.DataFrame) == False:
            raise ValueError("ERROR: Dataset is not DataFrame. DataFrame is required")
        df = df.copy()
        df[self.image_path] = df[self.image_path].apply(self.get_join_root_dir_map(self.data_dir))
        # df[self.label_name] = df[self.label_name].apply(str)
        df[self.image_level_label] = df[self.image_level_label].apply(self.__multi_hot_encode,
                                                                      args=(self.n_classes,))
        dataset = tf.data.Dataset.from_tensor_slices(
            dict(image_path=df.image_path.values,
                 label=np.array(list(df[self.image_level_label].values))))
        dataset = dataset.map(self.__load_data, num_parallel_calls=self.num_parallel_calls)
        dataset = dataset.cache(self.cache_file_location(self.cache_dir))
        if self.repeat:
            dataset = dataset.repeat()

        if transforms.has_transform():
            transforms_map = self.__get_transform_map(transforms, self.output_shape,
                                                      self.output_image_channels,
                                                      self.output_image_type)
            dataset = dataset.map(transforms_map)
            logging.info(dataset)
        dataset = dataset.map(lambda row: ([row["image"], row['label']]))
        dataset = dataset.batch(self.batch_size, drop_remainder=self.drop_remainder)
        dataset = dataset.prefetch(4)
        return dataset

    

    def __get_transform_map(self, transforms, output_shape, output_image_channels,
                            output_image_type):

        def transform_map(row):
            original_image = row["image"]
            original_labels = row["label"]
            augmented = tf.compat.v1.py_func(transforms.apply_transforms, [row["image"], row["label"]],
                                       [output_image_type, row["label"].dtype])
            logging.info(augmented)
            image = augmented[0]
            image.set_shape(output_shape + (output_image_channels,))
            logging.info(image)
            label = augmented[1]
            logging.info(label)
            row["image"] = image
            row["label"] = label
            row["original_image"] = original_image
            row["original_labels"] = original_labels
            return row

        return transform_map

    def __load_data(self, row):
        image_path = row["image_path"]
        image = self.load_image(image_path)
        label = row["label"]
        new_row = dict(image=image, label=label)
        return new_row

    def __multi_hot_encode(self, label_indicies, n_classes):
        encoded = np.zeros((n_classes,))
        encoded[label_indicies] = 1
        return encoded
