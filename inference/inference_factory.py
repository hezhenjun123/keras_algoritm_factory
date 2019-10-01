from inference.inference_lodging import InferenceLodging
from inference.inference_chaff import InferenceChaff
from inference.inference_chaff_video import InferenceChaffVideo
from inference.inference_lodging_video import InferenceLodgingVideo
from inference.inference_yield_absolute_video import InferenceYieldAbsoluteVideo
from inference.inference_bbox_breakage import InferenceBboxBreakage
from inference.inference_sprayer_video import InferenceSprayerVideo

class InferenceFactory:

    inference_registry = {
        "InferenceLodging": InferenceLodging,
        "InferenceChaffVideo": InferenceChaffVideo,
        "InferenceChaff": InferenceChaff,
        "InferenceLodgingVideo": InferenceLodgingVideo,
        "InferenceYieldAbsoluteVideo": InferenceYieldAbsoluteVideo,
        "InferenceBboxBreakage": InferenceBboxBreakage,
        "InferenceSprayerVideo": InferenceSprayerVideo

    }

    def __init__(self, config):
        self.config = config

    def create_inference(self, name):
        if name not in self.inference_registry:
            raise Exception(f"inference type is not supported: {name}")
        return self.inference_registry[name](self.config)
