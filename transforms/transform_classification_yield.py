import albumentations as A
from transforms.transform_base import TransformBase
import numpy as np


class TransformClassificationYield(TransformBase):

    def __init__(self, config):
        super().__init__(config)
        buffer_size = self.config["TRANSFORM"]["BUFFER_SIZE"]
        resize = self.config["TRANSFORM"]["RESIZE"]
        use_splicer = self.config["TRANSFORM"]["USE_SPLICER"]
        tfunc = [self.splice_on_buffer(buffer_size)] if use_splicer else []
        tfunc.extend([
                      A.Resize(resize[0], resize[1]),
                      self.extraction_bug_fix(),
                      self.floatify()
                    ])
        self.transform["IMAGE_ONLY"] = tfunc


    def splice_on_buffer(self,buffer_size):
        def transform_func(image):
            image_center = image.shape[0]/2
            offset = buffer_size/2
            top_image = image[:int(image_center-offset)]
            bottom_image = image[int(image_center+offset):]
            stacked_image = np.concatenate((top_image,bottom_image),axis=2)
            return {"image":stacked_image}
        return transform_func
    
    def extraction_bug_fix(self):
        def transform_func(image):
            #during extraction there was a bug multiplying images by 255
            # causing weird looking images
            # multiplying by 255 again fixes it
            image = (image*255).astype(np.uint8)
            return {"image":image}
        return transform_func

    def floatify(self):
        def transform_func(image):
            image = (image/255).astype(np.float32)
            return {"image":image}
        return transform_func