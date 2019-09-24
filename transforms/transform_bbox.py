import albumentations as A
from transforms.transform_base import TransformBase
import numpy as np

class TransformBbox(TransformBase):

    def __init__(self, config):
        super().__init__(config)
        resize_size = config["TRANSFORM"]["RESIZE"]
        self.transforms= [TransformBase.Image(A.Normalize())]
        #masks are np.array with shape (N,5) where 0th value encode label
        if resize_size is not None: 
            self.transforms.append(TransformBase.ImageLabel(self.resize_with_bbox(resize_size)))

    def resize_with_bbox(self,resize_size):
        def func(image,mask):
            old_shape = image.shape[:2]
            #rescale mask
            scale = [resize_size[1-i]/old_shape[i] for i in range(2)]
            annotation = mask
            annotation[:,1::2] *= scale[1]
            annotation[:,2::2] *= scale[0]
            annotation = np.round(annotation) 
            image = A.Resize(*resize_size)(image=image)['image']
            return {'image': image, 'mask': annotation}
        return func