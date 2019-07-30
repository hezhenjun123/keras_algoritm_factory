import logging
from data_generators.generator_segmentation import GeneratorSegmentation

logging.getLogger().setLevel(logging.INFO)


class GeneratorSegmentationTF2Unet(GeneratorSegmentation):

    def create_dataset(self, df, transforms=None):
        dataset = self.create_dataset_dict(df, transforms)
        dataset = dataset.map(lambda row: (row["image"], (row[
            "segmentation_labels"], row["segmentation_labels"])))
        dataset = dataset.prefetch(4)
        logging.info("==========================dataset=====================")
        logging.info(dataset)
        return dataset
