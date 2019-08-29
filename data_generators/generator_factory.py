from data_generators.generator_classification import GeneratorClassification
from data_generators.generator_regression import GeneratorRegression
from data_generators.generator_segmentation import GeneratorSegmentation
from data_generators.generator_video import GeneratorVideo


class DataGeneratorFactory:

    generator_registry = {
        "GeneratorClassification": GeneratorClassification,
        "GeneratorRegression": GeneratorRegression,
        "GeneratorSegmentation": GeneratorSegmentation,
        "GeneratorVideo": GeneratorVideo
    }

    def __init__(self, config):
        self.config = config

    def create_generator(self, name):
        if name not in self.generator_registry:
            raise Exception(f"generator type is not supported: {name}")
        return self.generator_registry[name](self.config)
