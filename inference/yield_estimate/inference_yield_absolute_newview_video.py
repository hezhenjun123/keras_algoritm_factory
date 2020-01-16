import platform
import logging
import cv2
import os
from inference.inference_base import InferenceBase
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from utilities.file_system_manipulation import directory_to_file_list
import tensorflow as tf
import shutil
from tensorflow.python.tools import freeze_graph
tf.enable_eager_execution()

# FIXME: assign 2G to yield model, should be configurable in yaml
# WARNING: set gpu fraction will increase the latency.
if platform.machine() == 'aarch64':
    from utilities.helper import config_gpu_memory
    config_gpu_memory(2048)

logging.getLogger().setLevel(logging.INFO)


# FIXME: I recommend not going through matplotlib interface and doing the numpy concatenations directly. Matplotlib is very slow
# FIXME: Instead we should make a utility function that we can use here and in ImageSummary
class InferenceYieldAbsoluteNewViewVideo(InferenceBase):

    def __init__(self, config):
        super().__init__(config)
        self.pred_image_dir = config["INFERENCE"]["PRED_IMAGE_DIR"]
        self.num_process_image = config["INFERENCE"]["NUM_PROCESS_IMAGE"]
        self.video_path = config["INFERENCE"]["VIDEO_PATH"]
        self.warmup = 300
        self.buffer_length = config["INFERENCE"]["MAXIMIZING_BUFFER_LENGTH"]
        self.hl_absolute, = plt.plot([], []) 
        self.hl, = plt.plot([], []) 
        self.offset = config["INFERENCE"]["OUTPUT_FRAME_OFFSET"]
        self.output_fps = config["INFERENCE"]["OUTPUT_FPS"]
        self.output_frame_keep = config["INFERENCE"]["OUTPUT_FRAME_KEEP"]
        self.pred_calib = config["INFERENCE"]["PREDICTION_CALIBRATION"]
        self.image_size = (960,640)

    def freeze_to_pb(self, save_dir):
        tf.compat.v1.disable_eager_execution()
        shutil.rmtree(save_dir) if os.path.exists(save_dir) else None
        os.makedirs(save_dir)
        model = self.load_model()
        model.model.save_weights(save_dir + "/tmp_resnet_weights.h5")
        tf.keras.backend.clear_session()
        tf.keras.backend.set_learning_phase(0)
        model = self.load_model(False)
        model.model.load_weights(save_dir + "/tmp_resnet_weights.h5")
        shutil.rmtree(save_dir) if os.path.exists(save_dir) else None
        tf.saved_model.simple_save(tf.keras.backend.get_session(),
                                   save_dir,
                                   inputs={"input": model.model.inputs[0]},
                                   outputs={"output": model.model.outputs[0]})

        freeze_graph.freeze_graph(None,
                                  None,
                                  None,
                                  None,
                                  model.model.outputs[0].op.name,
                                  None,
                                  None,
                                  os.path.join(save_dir + "/frozen_model.pb"),
                                  False,
                                  "",
                                  input_saved_model_dir=save_dir)
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

    def setup_writer(self):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.writer = cv2.VideoWriter(os.path.join(self.save_dir, self.output_video_name),
                                    fourcc, self.output_fps, self.image_size, True)

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

    def concatenate_video_images(self, img):
        return img[:, :, ::-1].astype(np.uint8)

    def overlay_text(self,img,fullness):
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 3
        fullness_text = f"Wheat Level: {fullness:.0%}"
        #add outlines 
        #img =cv2.putText(img,fullness_text, (int(img.shape[1]*.02),int(img.shape[0]*.1)),
       #                  font, fontScale,[0]*3,thickness=4)

       # img =cv2.putText(img,fullness_text, (int(img.shape[1]*.02),int(img.shape[0]*.1)),
       #                  font, fontScale,[255]*3,thickness=2)

        img = cv2.putText(img, fullness_text, (int(img.shape[1] * .1), int(img.shape[0] * .45)),
                          font, fontScale, [0] * 3, thickness=8)

        # Write some Text
        img = cv2.putText(img, fullness_text, (int(img.shape[1] * .1), int(img.shape[0] * .45)),
                          font, fontScale, [255, 0, 0], thickness=8)

        return img
        
    def __produce_video(self, model, dataset):
        #modify parameters of class and update_log methods to change 
        # how visualizations are created
        count = 0
        frames_written = 0
        candidate_frames = 0
        inference_dataset = dataset
        raw_preds = []
        adj_preds = []

        self.setup_writer()
        for elem in inference_dataset:
            if count >= self.offset:
                candidate_frames += 1

                if candidate_frames % self.output_frame_keep == 0:
                    image, pred = self.get_image_pred(model, elem)
                    pred = max(pred - self.pred_calib, 0)

                    raw_preds.append(pred)

                    adj_pred = np.median(raw_preds[-self.buffer_length:])

                    if len(adj_preds) > 0 and adj_pred > 0:
                        last_adj_pred = adj_preds[-1]
                        if last_adj_pred > 0 and last_adj_pred > adj_pred:
                            delta = last_adj_pred - adj_pred
                            if delta < .05:
                                adj_pred = max(adj_pred, last_adj_pred)

                    adj_preds.append(adj_pred)

                    frames_written += 1

                    out = self.concatenate_video_images(image)
                    out = self.overlay_text(out, adj_pred)

                    print("Frame: ", frames_written, int(pred*100), int(adj_pred*100))
                    self.writer.write(out)
                    if self.num_process_image >= 0 and frames_written >= self.num_process_image:
                        break

            count += 1

        plt.clf()
        self.writer.release()
