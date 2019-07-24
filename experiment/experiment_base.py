class ExperimentBase:
    def __init__(self, config):
        self.config = config

    def generate_transform(self):
        train_transform = None
        valid_transform = None
        return [train_transform, valid_transform]

    def generate_dataset(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError
