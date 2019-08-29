import tensorflow as tf
import logging
from data_generators.generator_base import DataGeneratorBase

logging.getLogger().setLevel(logging.INFO)


class GeneratorSegmentation(DataGeneratorBase):

    def __init__(self, config):
        super().__init__(config)
        self.batch_size = config["BATCH_SIZE"]

    def create_dataset_dict(self, df, transforms=None):
        df = df.copy()
        join_root_dir = self.get_join_root_dir_map(self.data_dir)
        df[self.segmentation_path] = df[self.segmentation_path] \
                                        .fillna("") \
                                        .apply(join_root_dir)
        df[self.image_path] = df[self.image_path].apply(join_root_dir)

        dataset = tf.data.Dataset.from_tensor_slices(
            dict(image_path=df[self.image_path].values,
                 segmentation_path=df[self.segmentation_path].values))
        dataset = dataset.map(self.__load_data, num_parallel_calls=self.num_parallel_calls)
        dataset = dataset.cache(self.cache_file_location(self.cache_dir))
        if self.repeat:
            dataset = dataset.repeat()
        if transforms.has_transform():
            transform_map = self.__get_transform_map(transforms, self.output_shape,
                                                     self.output_image_channels,
                                                     self.output_image_type)
            dataset = dataset.map(transform_map)
        dataset = dataset.batch(self.batch_size, drop_remainder=self.drop_remainder)
        return dataset

    def create_dataset(self, df, transforms=None):
        dataset = self.create_dataset_dict(df, transforms)
        # FIXME: Also may wantn ot have a final transform to make the schema of data generator flexible
        dataset = dataset.map(lambda row: (row["image"], (row["segmentation_labels"])))
        dataset = dataset.prefetch(4)
        logging.info("==========================dataset=====================")
        logging.info(dataset)
        return dataset

    def create_inference_dataset(self, df, transforms=None):
        dataset = self.create_dataset_dict(df, transforms)
        # dataset = dataset.map(lambda row: (row["image"], row["segmentation_labels"], row[
        # "original_image"], row["original_segmentation_labels"]))
        dataset = dataset.map(lambda row: (row["image"], row["original_image"]))
        dataset = dataset.prefetch(4)
        logging.info("====================inference dataset=====================")
        logging.info(dataset)
        return dataset

    def __get_transform_map(self, transforms, output_shape, output_image_channels,
                            output_image_type):

        def transform_map(row):
            original_image = row["image"]
            original_segmentation_labels = row["segmentation_labels"]
            augmented = tf.compat.v1.py_func(transforms.apply_transforms,
                                             [row["image"], row["segmentation_labels"]],
                                             [output_image_type, tf.uint8])
            logging.info(augmented)
            image = augmented[0]
            image.set_shape(output_shape + (output_image_channels,))
            logging.info(image)
            label = augmented[1]

            label.set_shape(output_shape + (1,))
            logging.info(label)
            row["image"] = image
            row["segmentation_labels"] = label
            row["original_image"] = original_image
            row["original_segmentation_labels"] = original_segmentation_labels
            return row

        return transform_map

    def __load_data(self, row):
        image = self.load_image(row["image_path"])
        label = self.load_image(row["segmentation_path"])
        new_row = dict(
            image=image,
            segmentation_labels=label,
        )
        return new_row
