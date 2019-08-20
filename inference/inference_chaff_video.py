import logging
import cv2
import os
from inference.inference_base import InferenceBase
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

logging.getLogger().setLevel(logging.INFO)


#FIXME: I recommend not going through matplotlib interface and doing the numpy concatenations directly. Matplotlib is very slow
#FIXME: Instead we should make a utility function that we can use here and in ImageSummary
class InferenceChaffVideo(InferenceBase):

    def __init__(self, config):
        super().__init__(config)
        self.pred_image_dir = config["INFERENCE"]["PRED_IMAGE_DIR"]
        self.num_process_image = config["INFERENCE"]["NUM_PROCESS_IMAGE"]
        self.video_path = config["INFERENCE"]["VIDEO_PATH"]
        if config["RUN_ENV"] == 'local':
            matplotlib.use('TkAgg')

    def run_inference(self):
        inference_transform = self.generate_transform()
        inference_dataset = self.generate_dataset(self.video_path, inference_transform)
        model = self.load_model()
        self.__produce_segmentation_image(model, inference_dataset)
        logging.info("================Inference Complete=============")

    def make_triplot(self, img, preds, log):
        img = img[:, :, ::-1]
        mask = preds == 1
        preds = preds[:, :, None].repeat(3, 2)
        preds[mask] = [255, 0, 0]
        out = np.concatenate((img, cv2.addWeighted(img, .7, preds, .3, 0), log), axis=1)
        return out

    def resize(self, img, shape):
        return cv2.resize(img, shape, interpolation=cv2.INTER_NEAREST)

    def update_line(self, hl, new_data):
        hl.set_xdata(np.append(hl.get_xdata(), new_data[0]))
        hl.set_ydata(np.append(hl.get_ydata(), new_data[1]))

    def create_log_array(self, hl, shape):
        plt.plot(hl.get_xdata(), hl.get_ydata(), color='steelblue')
        log_file = os.path.join(self.save_dir, 'log.png')
        if os.path.isfile(log_file):
            os.remove(log_file)
        plt.savefig(log_file)
        log = cv2.imread(log_file)
        log = self.resize(log, shape).astype(np.uint8)
        return log

    def __produce_segmentation_image(self, model, dataset):
        buffer_length = 5
        buffer = []
        inference_dataset = dataset.unbatch().batch(1)
        count = 0
        writer = None
        hl, = plt.plot([], [])
        for elem in inference_dataset:
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
                self.update_line(hl, (count, np.log(1 + resized_pred_seg.sum() / 255)))
                if (count - buffer_length) % 15 == 0: log = self.create_log_array(hl, resize_shape)
                out = self.make_triplot(image, pred, log)
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    video_shape = (out.shape[1], out.shape[0])
                    writer = cv2.VideoWriter(os.path.join(self.save_dir, 'inference.avi'), fourcc,
                                             5, video_shape, True)
                writer.write(out)

            count += 1
            if count >= self.num_process_image and self.num_process_image != -1: break
        writer.release()
