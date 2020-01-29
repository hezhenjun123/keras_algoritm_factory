import platform
from inference.yield_estimate.inference_yield_absolute_newview_trt import InferenceYieldAbsoluteNewViewTRT
from inference.chaff.inference_chaff_video import InferenceChaffVideo
from inference.chaff.inference_chaff_video_trt import InferenceChaffVideoTRT
from inference.yield_estimate.inference_yield_absolute_newview_video import InferenceYieldAbsoluteNewViewVideo
from inference.lodging.inference_lodging_video import InferenceLodgingVideo
from inference.lodging.inference_lodging_trt import InferenceLodgingTRT
from inference.yield_estimate.inference_yield_absolute_newview import InferenceYieldAbsoluteNewView


if platform.machine() != 'aarch64':
    from inference.chaff.inference_chaff import InferenceChaff
    from inference.chaff.inference_chaff_raw_video import InferenceChaffRawVideo
    from inference.lodging.inference_lodging import InferenceLodging
    from inference.yield_estimate.inference_yield_absolute_video import InferenceYieldAbsoluteVideo
    from inference.breakage.inference_bbox_breakage import InferenceBboxBreakage
    from inference.sprayer.inference_sprayer_video import InferenceSprayerVideo

class InferenceFactory:
    if platform.machine() != 'aarch64':
        inference_registry = {
            "InferenceChaff": InferenceChaff,
            "InferenceChaffVideo": InferenceChaffVideo,
            "InferenceChaffVideoTRT": InferenceChaffVideoTRT,
            "InferenceChaffRawVideo": InferenceChaffRawVideo,
            "InferenceLodging": InferenceLodging,
            "InferenceLodgingTRT": InferenceLodgingTRT,
            "InferenceLodgingVideo": InferenceLodgingVideo,
            "InferenceYieldAbsoluteVideo": InferenceYieldAbsoluteVideo,
            "InferenceYieldAbsoluteNewView": InferenceYieldAbsoluteNewView,
            "InferenceYieldAbsoluteNewViewVideo": InferenceYieldAbsoluteNewViewVideo,
            "InferenceBboxBreakage": InferenceBboxBreakage,
            "InferenceSprayerVideo": InferenceSprayerVideo,
            "InferenceYieldAbsoluteNewViewTRT": InferenceYieldAbsoluteNewViewTRT,
        }
    else:
        inference_registry = {
            "InferenceYieldAbsoluteNewView": InferenceYieldAbsoluteNewView,
            "InferenceYieldAbsoluteNewViewTRT": InferenceYieldAbsoluteNewViewTRT,
            "InferenceYieldAbsoluteNewViewVideo": InferenceYieldAbsoluteNewViewVideo,
            "InferenceChaffVideo": InferenceChaffVideo,
            "InferenceLodgingVideo": InferenceLodgingVideo,
            "InferenceChaffVideoTRT": InferenceChaffVideoTRT,
            "InferenceLodgingTRT": InferenceLodgingTRT,
        }

    def __init__(self, config):
        self.config = config

    def create_inference(self, name):
        if name not in self.inference_registry:
            raise Exception("inference type is not supported: {}".format(name))
        return self.inference_registry[name](self.config)
