from inference.inference_lodging import InferenceLodging
from inference.inference_chaff import InferenceChaff
from inference.inference_chaff_video import InferenceChaffVideo
from inference.inference_lodging_video import InferenceLodgingVideo


class InferenceFactory:

    inference_registry = {
        "InferenceLodging": InferenceLodging,
        "InferenceChaffVideo": InferenceChaffVideo,
        "InferenceChaff": InferenceChaff,
        "InferenceLodgingVideo": InferenceLodgingVideo
    }

    def __init__(self, config):
        self.config = config

    def create_inference(self, name):
        if name not in self.inference_registry:
            raise Exception(f"inference type is not supported: {name}")
        return self.inference_registry[name](self.config)
