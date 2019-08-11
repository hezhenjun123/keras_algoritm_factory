from inference.inference_lodging_tf2 import InferenceLodgingTF2


class InferenceFactory:

    inference_registry = {"InferenceLodgingTF2": InferenceLodgingTF2}

    def __init__(self, config):
        self.config = config

    def create_inference(self, name):
        if name not in self.inference_registry:
            raise Exception(f"inference type is not supported: {name}")
        return self.inference_registry[name](self.config)
