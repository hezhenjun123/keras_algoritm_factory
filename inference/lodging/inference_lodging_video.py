import platform
import logging
import cv2
import os
from inference.inference_base import InferenceBase
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import multiprocessing as mp
from utilities import file_system_manipulation as fsm
logging.getLogger().setLevel(logging.INFO)


# FIXME: I recommend not going through matplotlib interface and doing the numpy concatenations directly. Matplotlib is very slow
# FIXME: Instead we should make a utility function that we can use here and in ImageSummary
class InferenceLodgingVideo(InferenceBase):

    def __init__(self, config):
        super().__init__(config)
        self.pred_image_dir = config["INFERENCE"]["PRED_IMAGE_DIR"]
        self.num_process_image = config["INFERENCE"]["NUM_PROCESS_IMAGE"]
        self.video_path = config["INFERENCE"]["VIDEO_PATH"]

    def run_inference(self):
        model = self.load_model()
        inference_transform = self.generate_transform()

        file_list = sorted(fsm.directory_to_file_list(self.video_path))
        file_count = 1
        total_count = len(file_list)
        for file_path in file_list:
            logging.info(f"process video num: {file_count}/{total_count}")
            logging.info(f"file name: {file_path}")
            # inference_dataset = self.generate_dataset(file_path, inference_transform)
            inference_dataset = self.generate_dataset(file_path,inference_transform)
            input_video_name = os.path.splitext(os.path.basename(file_path))[0]
            self.output_video_name = f"inference_{input_video_name}.avi"
            self.__produce_segmentation_image(model, inference_dataset)
            file_count += 1
        logging.info("================Inference Complete=============")

    def make_plot_for_video(self, img, preds, log):
        img = img[:, :, ::-1]

        _ ,contours, _ = cv2.findContours(preds, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        newimg = np.copy(img)
        for contour in contours:
            cv2.drawContours(newimg, contour, -1, (0, 0, 255), 4)
        out = np.concatenate((newimg, log), axis=1)
        return out

    def resize(self, img, shape):
        return cv2.resize(img, shape, interpolation=cv2.INTER_NEAREST)

    def update_line(self, hl, new_data):
        hl.set_xdata(np.append(hl.get_xdata(), new_data[0]))
        hl.set_ydata(np.append(hl.get_ydata(), new_data[1]))

    def create_log_array(self, hl, shape):
        plt.plot(hl.get_xdata(), hl.get_ydata(), color='steelblue')
        plt.ylabel('Lodging Percent')
        plt.title('Amount of Lodging',{"fontsize":18})
        locs = plt.yticks()[0]
        plt.yticks(*(zip(*[[x,f'{x:.1%}'] for x in locs])))
        log_file = os.path.join(self.save_dir, 'log.png')
        if os.path.isfile(log_file):
            os.remove(log_file)
        plt.savefig(log_file)
        log = cv2.imread(log_file)

        log = self.resize(log, shape).astype(np.uint8)
        plt.clf()
        return log

    def __produce_segmentation_image(self, model, dataset):
        buffer_length = 5
        buffer = []
        inference_dataset = dataset
        count = 0
        writer = None
        hl, = plt.plot([], [])
        for elem in inference_dataset:
            logging.info("======================print elem=================================")
            logging.info(elem)
            logging.info(count)
            # logging.info(elem[0])
            # logging.info(elem[1])
            # elem[0].reshape((512, 512, 3))
            # elem[1].reshape(((1080, 1920, 3)))
            pred_res = model.predict(elem)
            original_image = np.squeeze(elem[1], axis=0)
            original_image = self.resize(original_image, (960, 640))
            resize_shape = (original_image.shape[1], original_image.shape[0])

            pred_seg = np.round(pred_res[0])
            resized_pred_seg = self.resize(pred_seg, resize_shape)
            log_shape = (int(resize_shape[0] / 2), resize_shape[1])
            if len(buffer) < buffer_length:
                buffer.append((original_image, resized_pred_seg))
            else:
                buffer.pop(0)
                buffer.append((original_image, resized_pred_seg))
                image = buffer[buffer_length // 2][0]
                pred = np.max(np.array([x[1] for x in buffer]), axis=0).astype(np.uint8)
                self.update_line(hl, (count, resized_pred_seg.sum() / resized_pred_seg.size))
                if (count - buffer_length) % 15 == 0: log = self.create_log_array(hl, log_shape)
                out = self.make_plot_for_video(image.astype(np.uint8), pred.astype(np.uint8),
                                               log.astype(np.uint8))
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    video_shape = (out.shape[1], out.shape[0])
                    writer = cv2.VideoWriter(os.path.join(self.save_dir, self.output_video_name),
                                             fourcc, 5, video_shape, True)
                writer.write(out)

            count += 1
            if count >= self.num_process_image and self.num_process_image != -1: break
        plt.clf()
        writer.release()


    def generate_dataset(self,file_path,transforms):

        if platform.machine() != 'aarch64':
            mp.set_start_method('spawn')

        queue = mp.Manager().Queue(maxsize=100) 

        def generator(queue):
            elem = [0,0]
            while elem[1] is not None:
                elem = queue.get()
                yield elem

        mp.Process(target=read_data,args=(file_path,transforms,queue)).start()

        return generator(queue)

def read_data(video_path,transforms,queue):
    if fsm.is_s3_path(video_path):
        video_path = fsm.s3_to_local(video_path, './video.avi')[0]
    if not os.path.exists(video_path):
        raise ValueError("Incorrect video path")
    video = cv2.VideoCapture(video_path)
    frame = 0
    while 1:
        frame = video.read()[1]
        if frame is None:  
            queue.put(None,None)
            video.release()
            break
        else:
            frame = frame[:,:,::-1]
            transformed_frame = transforms.apply_transforms(image=frame,label=np.zeros_like(frame))[0]
            queue.put((transformed_frame[None,:],frame[None,:]))
