from data_generators.generator_classification_vanilla import GeneratorClassificationVanilla
from data_generators.generator_segmentation_vanilla import GeneratorSegmentationVanilla
from data_generators.generator_segmentation_uv import GeneratorSegmentationUV


class DataGeneratorFactory:

    generator_registry = {
        "GeneratorClassificationVanilla": GeneratorClassificationVanilla,
        "GeneratorSegmentationVanilla": GeneratorSegmentationVanilla,
        "GeneratorSegmentationUV": GeneratorSegmentationUV,
    }

    def __init__(self, config):
        self.config = config

    def create_generator(self, name):
        if name not in self.generator_registry:
            raise Exception(f"generator type is not supported: {name}")
        return self.generator_registry[name](self.config)
