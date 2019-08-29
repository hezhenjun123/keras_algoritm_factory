import logging
import cv2
import os
from inference.inference_base import InferenceBase
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from utilities.file_system_manipulation import directory_to_file_list

logging.getLogger().setLevel(logging.INFO)


# FIXME: I recommend not going through matplotlib interface and doing the numpy concatenations directly. Matplotlib is very slow
# FIXME: Instead we should make a utility function that we can use here and in ImageSummary
class InferenceYieldAbsoluteVideo(InferenceBase):

    def __init__(self, config):
        super().__init__(config)
        self.pred_image_dir = config["INFERENCE"]["PRED_IMAGE_DIR"]
        self.num_process_image = config["INFERENCE"]["NUM_PROCESS_IMAGE"]
        self.video_path = config["INFERENCE"]["VIDEO_PATH"]
        self.warmup = 300
        self.buffer_length = 300
        self.buffer = [0]*self.buffer_length
        self.hl_absolute, = plt.plot([], []) 
        self.hl, = plt.plot([], []) 
        self.offset=3800
        self.image_size = (960,640)
        self.log_size = (self.image_size[0]//2,self.image_size[1])

        if config["RUN_ENV"] == 'local':
            matplotlib.use('TkAgg')

    def run_inference(self):
        model = self.load_model()
        inference_transform = self.generate_transform()

        file_list = sorted(directory_to_file_list(self.video_path))
        file_count = 1
        total_count = len(file_list)
        for file_path in file_list:
            logging.info(f"process video num: {file_count}/{total_count}")
            logging.info(f"file name: {file_path}")
            inference_dataset = self.generate_dataset(file_path, inference_transform)
            input_video_name = os.path.splitext(os.path.basename(file_path))[0]
            self.output_video_name = f"inference_{input_video_name}.avi"
            self.__produce_video(model, inference_dataset)
            file_count += 1
        logging.info("================Inference Complete=============")

    def setup_writer(self,log_shape):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_shape = (self.image_size[0]+log_shape[1],log_shape[0])
        self.writer = cv2.VideoWriter(os.path.join(self.save_dir, self.output_video_name),
                                    fourcc, 5, video_shape, True)

    def resize(self, img, shape):
        return cv2.resize(img, shape, interpolation=cv2.INTER_NEAREST)
        
    def get_image_pred(self,model,elem):
        pred = model.predict(elem)
        if pred.shape[1]!=1:
            pred = (np.argmax(pred[0])+1)/pred.shape[1]
        else:
            pred = np.squeeze(pred)
        original_image = np.squeeze(elem[1], axis=0)
        original_image = self.resize(original_image, self.image_size)
        return original_image, pred

    def create_log_image(self, shape,data, log_type):
        if log_type == "absolute": 
            plt.ylabel('Fullness Percentage')
            plt.title('Hopper fullness',{'fontsize':18})
            plt.ylim(0,1)
            line = self.hl_absolute
        elif log_type == "delta":
            plt.ylabel('Percent_Fullness/Second')
            plt.title('Speed of harvesting',{'fontsize':18})
            line = self.hl
        line.set_xdata(np.append(line.get_xdata(),data[0]))
        line.set_ydata(np.append(line.get_ydata(),data[1]))
        plt.plot(line.get_xdata(),
                 line.get_ydata(),
                 color='steelblue')
        locs = plt.yticks()[0]
        plt.yticks(*(zip(*[[x,f'{x:.1%}'] for x in locs])))
        log_file = os.path.join(self.save_dir, 'log.png')
        if os.path.isfile(log_file): os.remove(log_file)
        plt.savefig(log_file,bbox_inches='tight', pad_inches=0.2)
        plt.clf()
        log = cv2.imread(log_file)
        log = self.resize(log, shape).astype(np.uint8)
        plt.clf()
        return log

    def concatenate_video_images(self, img, log):
        img = img[:, :, ::-1].astype(np.uint8)
        log = log.astype(np.uint8)
        out = np.concatenate((img, log), axis=1)
        return out

    def overlay_text(self,img,fullness,speed): 
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        fullness_text = f"Wheat Level: {fullness:.0%}"
        if speed <.008:
            speed_text = 'Slow'
        elif speed >= .008 and speed<.012: 
            speed_text = 'Medium'
        else: 
            speed_text = 'Fast'
        speed_text = f"Harvest Speed: {speed_text}"
        #add outlines 
        img =cv2.putText(img,fullness_text, (int(img.shape[1]*.02),int(img.shape[0]*.1)), 
                         font, fontScale,[0]*3,thickness=4)
        img =cv2.putText(img,speed_text, (int(img.shape[1]*.02),int(img.shape[0]*.25)), 
                         font, fontScale,[0]*3,thickness=4)
        # Write some Text
        img =cv2.putText(img,fullness_text, (int(img.shape[1]*.02),int(img.shape[0]*.1)), 
                         font, fontScale,[255]*3,thickness=2)
        img =cv2.putText(img,speed_text, (int(img.shape[1]*.02),int(img.shape[0]*.25)), 
                         font, fontScale,[255]*3,thickness=2)
        return img

    def update_log(self,count,old_log):
        old_label = np.mean(self.buffer[0:self.buffer_length//2])
        pred = np.mean(self.buffer[self.buffer_length//2:])
        speed = max(pred-old_label,0)/10
        if (count - self.buffer_length-self.offset) % 15 == 0:
            log_absolute = self.create_log_image(self.log_size,(count,pred),"absolute")
            log_delta = self.create_log_image(self.log_size,(count,speed),"delta")
            log = np.concatenate((log_absolute,log_delta),axis=1) 
        else: 
            log = old_log
        return pred,speed,log
    
    def create_video_image(self,image,count,old_log):
        fullness, speed, log = self.update_log(count,old_log)
        out = self.concatenate_video_images(image,log)
        out = self.overlay_text(out,fullness,speed)
        return log,out 
        
    def __produce_video(self, model, dataset):
        #modify parameters of class and update_log methods to change 
        # how visualizations are created
        count = 0
        inference_dataset = dataset.unbatch().batch(1)
        log = np.zeros((self.log_size[1],self.log_size[0]*2,3))+255
        self.setup_writer(log.shape)
        for elem in inference_dataset:
            if count >= self.offset:
                image, pred = self.get_image_pred(model,elem)
                if len(self.buffer) < self.buffer_length:
                    self.buffer.append(pred)
                else:
                    self.buffer.pop(0)
                    self.buffer.append(pred)
                    log,out = self.create_video_image(image,count,log)
                    if (count - self.buffer_length-self.offset) % 3 == 0:
                        self.writer.write(out)

            count += 1
            if ((count - self.buffer_length-self.offset) >= self.num_process_image and 
                 self.num_process_image != -1):
                break
        plt.clf()
        self.writer.release()
