import platform
from transforms.transform_base import TransformBase

if platform.machine() != 'aarch64':
    import albumentations as A
else:
    import cv2
    
class TransformClassification(TransformBase):

    def __init__(self, config):
        super().__init__(config)
        random_crop_size = self.config["TRANSFORM"]["RANDOMCROP"]
        resize = self.config["TRANSFORM"]["RESIZE"]
        
        if platform.machine() != 'aarch64':
            tfunc = [
                A.RandomCrop(random_crop_size[0], random_crop_size[1]),
                A.Resize(resize[0], resize[1])
            ]
            self.transform["IMAGE_ONLY"] = tfunc                
