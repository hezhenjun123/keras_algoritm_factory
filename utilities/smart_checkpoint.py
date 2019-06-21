import os
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
import subprocess

class SmartCheckpoint(Callback):
    def __init__(self,destination_path,file_format,**kwargs):
        self.temp_dir = None
        self.destination_path =  destination_path
        self.file_format = file_format
        self.period = kwargs.pop('period',1)
        self.__create_local_folder__()
        self.local_checkpoint_path = os.path.join(self.temp_dir if self.temp_dir is not None else self.destination_path,
                                                 self.file_format)
        self.checkpoint_callback = ModelCheckpoint(self.local_checkpoint_path,
                                                     save_weights_only=False,
                                                     verbose=1,
                                                     period=1,
                                                     **kwargs)

    def __create_local_folder__(self):
        if 's3://' in self.destination_path:
            self.temp_dir = 'temporary_checkpoints/'
            os.makedirs(self.temp_dir)

    def on_epoch_end(self,epoch,logs={}):
        if epoch==0:
            self.checkpoint_callback.model = self.model
        if (epoch+1)%self.period==0:
            path_formatted = self.local_checkpoint_path.format(epoch=epoch+1,**logs)
            os.makedirs(os.path.dirname(path_formatted),exist_ok=True)
            self.checkpoint_callback.on_epoch_end(epoch,logs)
            if self.temp_dir is not None:
                current_files = os.listdir(self.temp_dir)
                if current_files:
                    for sub_file in current_files:
                        command = 'aws s3 mv --recursive {} {}'.format(os.path.join(self.temp_dir,sub_file),
                                                        os.path.join(self.destination_path,sub_file))
                        subprocess.Popen(command.split(' '))