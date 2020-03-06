import logging
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from inference.inference_base import InferenceBase
from utilities.file_system_manipulation import directory_to_file_list
import tensorflow as tf
tf.enable_eager_execution()


logging.getLogger().setLevel(logging.INFO)


#FIXME: I recommend not going through matplotlib interface and doing the numpy concatenations directly. Matplotlib is very slow
#FIXME: Instead we should make a utility function that we can use here and in ImageSummary
class InferenceChaffVideo(InferenceBase):

    def __init__(self, config):
        super().__init__(config)
        self.pred_image_dir = config["INFERENCE"]["PRED_IMAGE_DIR"]
        self.num_process_image = config["INFERENCE"]["NUM_PROCESS_IMAGE"]
        self.video_path = config["INFERENCE"]["VIDEO_PATH"]
        self.maximizing_buffer_length = config["INFERENCE"]["MAXIMIZING_BUFFER_LENGTH"]
        self.output_fps = config["INFERENCE"]["OUTPUT_FPS"]
        self.make_plots  = config["INFERENCE"]["MAKE_PLOTS"]

    def run_inference(self):
        inference_transform = self.generate_transform()
        model = self.load_model()

        file_list = sorted(directory_to_file_list(self.video_path))
        total_count = len(file_list)
        for file_count, file_path in enumerate(file_list, start=1):
            logging.info(f"process video num: {file_count}/{total_count}")
            logging.info(f"file name: {file_path}")
            inference_dataset = self.generate_dataset(file_path, inference_transform)
            input_video_name = file_path.split('/')[-1]
            self.output_video_name = f"{input_video_name}.avi"
            self.__produce_segmentation_image(model, inference_dataset)
        logging.info("================Inference Complete=============")



    def overlay_preds(self,img,preds):
        img = img[:, :, ::-1]
        mask = preds == 1
        preds = preds[:, :, None].repeat(3, 2)
        preds[mask] = [0, 0, 255]
        overlayed_preds =  cv2.addWeighted(img, .7, preds, .3, 0)
        return overlayed_preds

    def make_dualplot(self, img, preds):
        overlayed_preds = self.overlay_preds(img,preds)
        img = img[:, :, ::-1]
        out = np.concatenate((img,overlayed_preds), axis=1)
        return out

    def make_triplot(self, img, preds, log):
        dual_plot = self.make_dualplot(img,preds)
        out = np.concatenate((dual_plot, log), axis=1)
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
        log = cv2.imread(log_file)
        log = self.resize(log, shape).astype(np.uint8)
        plt.clf()
        return log

    def __produce_segmentation_image(self, model, dataset):
        buffer_length = self.maximizing_buffer_length
        buffer = []
        inference_dataset = dataset
        count = 0
        writer = None
        if self.make_plots: hl, = plt.plot([], [])
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
                # self.update_line(hl, (count, np.log(1 + resized_pred_seg.sum() / 255)))
                if self.make_plots:
                    self.update_line(hl, (count, resized_pred_seg.sum() / (resized_pred_seg.shape[0]*resized_pred_seg.shape[1])))
                    if (count - buffer_length) % (self.output_fps*2) == 0:
                        log = self.create_log_array(hl, resize_shape)
                    out = self.make_triplot(image, pred, log)
                else:
                    # out = self.make_dualplot(image,pred)
                    out = self.overlay_preds(image,pred)
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    video_shape = (out.shape[1], out.shape[0])
                    writer = cv2.VideoWriter(os.path.join(self.save_dir, self.output_video_name), fourcc,
                                             self.output_fps, video_shape, True)
                writer.write(out)

            count += 1
            if count >= self.num_process_image and self.num_process_image != -1: break
        if self.make_plots: plt.clf()
        writer.release()
