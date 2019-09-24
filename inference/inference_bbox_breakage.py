import logging
import cv2
import os
from inference.inference_base import InferenceBase
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

logging.getLogger().setLevel(logging.INFO)


#FIXME: I recommend not going through matplotlib interface and doing the numpy concatenations directly. Matplotlib is very slow
#FIXME: Instead we should make a utility function that we can use here and in ImageSummary
class InferenceBboxBreakage(InferenceBase):

    def __init__(self, config):
        super().__init__(config)
        if config["RUN_ENV"] == 'local':
            matplotlib.use('TkAgg')
        self.pred_image_dir = config["INFERENCE"]["PRED_IMAGE_DIR"]
        self.num_process_image = config["INFERENCE"]["NUM_PROCESS_IMAGE"]
        if self.num_process_image >= 99999:
            raise Exception("cannot process more than 99999 images ")

    def run_inference(self):
        inference_transform = self.generate_transform()
        inference_data_split = self.read_train_csv()
        inference_dataset = self.generate_dataset(inference_data_split, inference_transform)
        model = self.load_model()
        self.__produce_segmentation_image(model, inference_dataset)
        logging.info("================Inference Complete=============")

    def _overlay_boxes(self,image,boxes,labels,color=(255,0,0)):
        image = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        for box,label in zip(boxes,labels):
            image = cv2.rectangle(image,tuple(box[[0,1]]),tuple(box[[2,3]]),color,3,cv2.LINE_AA)
            # image =cv2.putText(image,str(label), tuple(box[0:2]+10), 
            #                    font, fontScale,[255]*3,thickness=2)
            # image =cv2.putText(image,str(label), tuple(box[0:2]+10), 
            #                    font, fontScale,[0]*3,thickness=4)            
        return image

    def __produce_segmentation_image(self, model, dataset):
        inference_dataset = dataset.unbatch().batch(1)
        save_dir = os.path.join(self.save_dir, self.pred_image_dir)
        if os.path.exists(save_dir) is False:
            os.makedirs(save_dir)
        count = 0
        for elem in inference_dataset:
            transformed_image = elem[0]
            preds = model.predict(transformed_image)
            original_image = np.squeeze(elem[2], axis=0).astype(np.uint8)
            original_bboxes, original_labels = map(lambda x : np.squeeze(x, axis=0).astype(int),elem[3])

            pred_bboxes,_,pred_labels = map(lambda x : np.squeeze(x, axis=0),preds)
            resize_scale = (original_image.shape[0]/transformed_image.shape[1], original_image.shape[1]/transformed_image.shape[2])            
            pred_bboxes[:,0::2] *= resize_scale[1]
            pred_bboxes[:,1::2] *= resize_scale[0]
            pred_bboxes = pred_bboxes.astype(int)
            pred_bboxes = pred_bboxes[pred_labels!=-1]
            pred_labels = pred_labels[pred_labels!=-1]

            fig1 = plt.figure()
            fig1.add_subplot(1, 2, 1)
            plt.imshow(self._overlay_boxes(original_image,original_bboxes,original_labels))
            plt.title('Original Bboxes')
            fig1.add_subplot(1, 2, 2)
            preds_overlayed = self._overlay_boxes(original_image,pred_bboxes,pred_labels)
            plt.imshow(preds_overlayed)
            plt.title('Predicted Bboxes')
            plt.savefig(os.path.join(save_dir, f"image{count:05d}_model.png"),dpi=300)
            plt.clf()

            fig2 = plt.figure()
            plt.imshow(self._overlay_boxes(preds_overlayed,original_bboxes,original_labels,color=(0,0,255)))
            plt.title('Combined Bboxes (Blue original, Red predicted)')

            plt.savefig(os.path.join(save_dir, f"image{count:05d}_original.png"),dpi=300)
            plt.clf()
            
            count += 1
            if count >= self.num_process_image and self.num_process_image!=-1: break
