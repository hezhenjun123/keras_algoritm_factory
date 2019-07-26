import tensorflow as tf
import logging
from data_generators.generator_base import DataGeneratorBase
from utilities.crop_patches import CropAndExpand

logging.getLogger().setLevel(logging.INFO)


class GeneratorSegmentationUV(DataGeneratorBase):
    def __init__(self, config):
        super().__init__(config)
        self.batch_size = config["BATCH_SIZE"]
        resize_config = config["TRANSFORM"]["RESIZE"]
        self.resize = (resize_config[0], resize_config[1])

    def create_dataset(self, df, transforms=None):
        df = df.copy()
        df[self.segmentation_path] = df[self.segmentation_path].fillna("")
        df[self.segmentation_path] = df[self.segmentation_path].apply(
            self.get_join_root_dir_map(self.data_dir))
        df[self.image_path] = df[self.image_path].apply(
            self.get_join_root_dir_map(self.data_dir))

        dataset = tf.data.Dataset.from_tensor_slices(
            dict(image_path=df[self.image_path].values,
                 segmentation_path=df[self.segmentation_path].values))
        dataset = dataset.map(self.__load_data,
                              num_parallel_calls=self.num_parallel_calls)
        dataset = dataset.cache(self.cache_file(self.cache_dir))
        if self.repeat is True:
            dataset = dataset.repeat()
        if transforms is not None and transforms.has_transform() is True:
            transform_map = self.__get_transform_map(
                transforms, self.output_shape, self.output_image_channels,
                self.output_image_type)
            dataset = dataset.map(transform_map)
        dataset = dataset.batch(self.batch_size,
                                drop_remainder=self.drop_remainder)
        dataset = dataset.prefetch(4)
        logging.info("==========================dataset=====================")
        logging.info(dataset)
        if transforms is not None:
            boxes = [[x / 4, y / 4, (x + 1) / 4,
                  min((y + 1) / 4, .95)] for x in range(4) for y in range(4)]
            crop_and_expand = CropAndExpand(resize=self.resize, boxes=boxes)
            dataset = dataset. \
                apply(tf.data.experimental.unbatch()). \
                map(crop_and_expand, num_parallel_calls=4). \
                shuffle(buffer_size=100, seed=1). \
                apply(tf.data.experimental.unbatch()). \
                batch(self.batch_size).map(
                lambda row: (row["image"], row["segmentation_labels"])
            )
        else:
            dataset= dataset.map(lambda row: (row["image"], row["segmentation_labels"]))

        return dataset

    def __get_transform_map(self, transforms, output_shape,
                            output_image_channels, output_image_type):
        def transform_map(row):
            logging.info(tf.py_func(transforms.apply_transforms, [row["image"], row["segmentation_labels"]], \
                                    [output_image_type, tf.uint8]))
            augmented = tf.py_func(transforms.apply_transforms, [row["image"], row["segmentation_labels"]], \
                                   [output_image_type, tf.uint8])
            image = augmented[0]
            image.set_shape(output_shape + (output_image_channels, ))
            logging.info(image)
            label = augmented[1]

            label.set_shape(output_shape + (1, ))
            logging.info(label)
            row["image"] = image
            row["segmentation_labels"] = label
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
