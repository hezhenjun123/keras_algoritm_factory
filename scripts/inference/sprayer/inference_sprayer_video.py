import logging
import cv2
import os
from inference.inference_base import InferenceBase
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
tf.enable_eager_execution()

logging.getLogger().setLevel(logging.INFO)


#FIXME: I recommend not going through matplotlib interface and doing the numpy concatenations directly. Matplotlib is very slow
#FIXME: Instead we should make a utility function that we can use here and in ImageSummary
class InferenceSprayerVideo(InferenceBase):

    def __init__(self, config):
        super().__init__(config)
        self.pred_image_dir = config["INFERENCE"]["PRED_IMAGE_DIR"]
        self.num_process_image = config["INFERENCE"]["NUM_PROCESS_IMAGE"]
        self.video_path = config["INFERENCE"]["VIDEO_PATH"]
        if config["RUN_ENV"] == 'local':
            matplotlib.use('TkAgg')

    def run_inference(self):
        print("Generating transform")
        inference_transform = self.generate_transform()
        print("Generating dataset")
        inference_dataset = self.generate_dataset(self.video_path, inference_transform)
        print("Loading model")
        model = self.load_model()
        print("Producing segmentation video")
        self.__produce_segmentation_image(model, inference_dataset)
        logging.info("================Inference Complete=============")

    def make_2plot(self, img, preds):
        img = img[:, :, ::-1]
        mask = preds == 1
        preds = preds[:, :, None].repeat(3, 2)
        preds[mask] = [0, 0, 255]
        out = np.concatenate((img, cv2.addWeighted(img, .7, preds, .3, 0)), axis=1)
        return out

    def resize(self, img, shape):
        return cv2.resize(img, shape, interpolation=cv2.INTER_NEAREST)

    def update_line(self, hl, new_data):
        hl.set_xdata(np.append(hl.get_xdata(), new_data[0]))
        hl.set_ydata(np.append(hl.get_ydata(), new_data[1]))

    def create_log_array(self, hl, shape):
        plt.plot(hl.get_xdata(), hl.get_ydata(), color='steelblue')
        plt.ylabel('Chaff Percent')
        locs = plt.yticks()[0]
        plt.yticks(*(zip(*[[x,f'{x:.1%}'] for x in locs])))
        plt.title('Chaff Content',{"fontsize":18})
        log_file = os.path.join(self.save_dir, 'log.png')
        if os.path.isfile(log_file):
            os.remove(log_file)
        plt.savefig(log_file)
        plt.clf()
        log = cv2.imread(log_file)
        log = self.resize(log, shape).astype(np.uint8)
        return log

    def __produce_segmentation_image(self, model, dataset):
        buffer_length = 1
        buffer = []
        inference_dataset = dataset
        count = 0
        writer = None
        hl, = plt.plot([], [])
        ei = 0
        for elem in inference_dataset:
            ei += 1
            print("Video element: ", ei)
            pred_res = model.predict(elem)
            original_image = np.squeeze(elem[1], axis=0)
            resize_shape = (original_image.shape[1], original_image.shape[0])

            pred_seg = np.round(pred_res[0])
            resized_pred_seg = self.resize(pred_seg, resize_shape)

            if len(buffer) < buffer_length:
                buffer.append((original_image, resized_pred_seg))
            else:
                buffer.pop(0)
                buffer.append((original_image, resized_pred_seg))
                image = buffer[buffer_length // 2][0]
                pred = np.max(np.array([x[1] for x in buffer]), axis=0).astype(np.uint8)
                self.update_line(hl, (count, resized_pred_seg.sum() / resized_pred_seg.size))
                #if (count - buffer_length) % 15 == 0: log = self.create_log_array(hl, resize_shape)
                out = self.make_2plot(image, pred)
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    # fourcc=0
                    video_shape = (out.shape[1], out.shape[0])
                    writer = cv2.VideoWriter(os.path.join(self.save_dir, 'inference.avi'), fourcc,
                                             5, video_shape, True)
                writer.write(out)

            count += 1
            if count >= self.num_process_image and self.num_process_image != -1: break
        writer.release()
        print("video complete")