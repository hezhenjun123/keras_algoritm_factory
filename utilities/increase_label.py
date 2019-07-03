from scipy import signal
import numpy as np

from avi.exp.transforms.basic.Transform import Transform
from avi.exp.transforms import parser
from avi.exp.transforms.row import Row

@parser.register_transform
class IncreaseBinarySegmap(Transform):
    r"""Increases the size of NG labels on binary segmap
    ----------
    segmentation_labels : numpy.ndarray
        The segmentation labels. 
    """
    def __init__(self,kernel_size):
        self.kernel_size = kernel_size
        super().__init__()

    def call(self, row):
        orig_label = row["segmentation_labels"]
        self.check_binary_label(orig_label)
        label =  orig_label[...,0].astype(np.float32)
        label = signal.convolve2d(label,
                                  np.ones((self.kernel_size,self.kernel_size)),
                                  mode='same')
        label = np.clip(label,0,1)[:,:,None].astype(orig_label.dtype)
        row["segmentation_labels"] = label
        return row

    def check_binary_label(self,label):
        if  not(np.max(label)==1 and np.min(label)==0 and len(label.shape)==3):
            raise ValueError('Segmap needs to be binary')
            
        
    @property
    def required_keys(self):
        return Row(segmentation_labels=True)

    @property
    def added_keys(self):
        pass