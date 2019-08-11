import logging
from data_generators.generator_segmentation import GeneratorSegmentation

logging.getLogger().setLevel(logging.INFO)


class GeneratorSegmentationTF2Unet(GeneratorSegmentation):

    def create_dataset(self, df, transforms=None):
        dataset = self.create_dataset_dict(df, transforms)
        # FIXME: change the style to use one segmenatation labels
        # FIXME: Also may wantn ot have a final transform to make the schema of data generator flexible
        dataset = dataset.map(
            lambda row: (row["image"], (row["segmentation_labels"])))
        dataset = dataset.prefetch(4)
        logging.info("==========================dataset=====================")
        logging.info(dataset)
        return dataset

    def create_inference_dataset(self, df, transforms=None):
        dataset = self.create_dataset_dict(df, transforms)
        dataset = dataset.map(lambda row: (row["image"], row["segmentation_labels"], row[
            "original_image"], row["original_segmentation_labels"]))
        dataset = dataset.prefetch(4)
        logging.info("====================inference dataset=====================")
        logging.info(dataset)
        return dataset
