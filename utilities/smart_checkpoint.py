import os
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
import subprocess

class SmartCheckpoint(Callback):
    r"Checkpoint class that automatically handles non existing paths and s3 synchronization"
    def __init__(self,destination_path,file_format,**kwargs):
        self.local_dir = None
        self.destination_path =  destination_path
        self.file_format = file_format
        self.__create_local_folder__()
        self.checkpoint_path = os.path.join(
                                            self.local_dir if self.local_dir is not None else self.destination_path,
                                            self.file_format
                                            )
        self.checkpoint_callback = ModelCheckpoint(self.checkpoint_path,
                                                   **kwargs)

    def __create_local_folder__(self):
        #only use local temp directory if destionation is s3
        if 's3://' in self.destination_path:
            self.local_dir = 'temporary_checkpoints/'
            os.makedirs(self.local_dir)

    def on_epoch_end(self,epoch,logs={}):
        #can't move to init due to how self.model is assigned
        if epoch==0:
            self.checkpoint_callback.model = self.model

        ckpt_path_formatted = self.checkpoint_path.format(epoch=epoch+1,**logs)
        ckpt_path_directory= os.path.dirname(ckpt_path_formatted)
        os.makedirs(ckpt_path_directory,exist_ok=True)
        self.checkpoint_callback.on_epoch_end(epoch,logs)
        if self.local_dir is not None:
            checkpoint_created = len(os.listdir(ckpt_path_directory))
            if checkpoint_created:
                #move all of the contents of local directory to respect directory structure
                files_or_dirs = os.listdir(self.local_dir)
                for file_or_dir in files_or_dirs:
                    command = 'aws s3 mv --recursive {} {}'.format(os.path.join(self.local_dir,file_or_dir),
                                                                   os.path.join(self.destination_path,file_or_dir))
                    subprocess.Popen(command.split(' '),stdout=subprocess.DEVNULL)