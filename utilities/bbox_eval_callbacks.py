import tensorflow as tf
import pandas as pd
import numpy as np
from utilities.bbox_overlap import compute_overlap

class EvaluateBboxCallback(tf.keras.callbacks.Callback):
    """A tf.keras callback that shows bounding box evalution on Tensorboard
    during training.
    """

    def __init__(self, num_classes, num_steps, generator, eval_interval=1):
        """
        Parameters
        ----------
        generator: tf.keras.utils.Sequence
            the data generator that feed model inputs, e.g.
            e.g. avi.models.retinanet.preprocess.Generator
        tensorboard: tf.keras.callbacks.TensorBoard
            the tensorboord callback used to send the evaluation metrics
        eval_interval: int
            the interval of epochs to perform evaluation
        """
        self.num_classes = num_classes
        self.num_steps = num_steps
        self.generator = generator
        self.eval_interval = eval_interval

        super(EvaluateBboxCallback, self).__init__()

    def on_epoch_end(self, epoch, logs={}):
        if epoch == 0 or (epoch + 1) % self.eval_interval == 0:
            metrics = self.evaluate()
            for metric,value in metrics.items():
                logs['val_'+metric]=value
        print(logs)

    def evaluate(
        self,
    ):
        """ Evaluate a given dataset using a given model.
        # Returns
            A dict mapping class names to mAP scores.
        """
        # gather all annotations and detections
        all_ground_truth, all_detections= self._get_detections()
        metrics = {}
        metrics['mean_average_precision'] = self._compute_aps(all_ground_truth,all_detections)
        return metrics

    def _get_detections(self):
        """ Get the detections from the model using the generator.
        The result is a list of lists such that the size is:
            all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]
        # Returns
            Two lists of lists containing the ground truth and detections for each image in the generator.
        """
        all_detections,all_ground_truth = [],[]
        for i,batch in enumerate(self.generator):
            if i>=self.num_steps: break
            detections, ground_truth = [],[]
            image, annotations = batch
            annotations = [x.numpy() for x in annotations]
            ground_bboxes, ground_labels = annotations
            # run network
            boxes, scores, labels = self.model.predict_on_batch(image)[:3]
            image_detections = np.concatenate([boxes[0],
                                               np.expand_dims(scores[0], axis=1),
                                               np.expand_dims(labels[0], axis=1)], axis=1)
            # copy detections to all_detections
            for label in range(self.num_classes):
                detections.append(image_detections[image_detections[:, -1] == label, :-1])
                ground_truth.append(ground_bboxes[ground_labels == label, :].copy())
            all_detections.append(detections)
            all_ground_truth.append(ground_truth)
            
        return all_ground_truth, all_detections

    def _compute_aps(self, all_ground_truth, all_detections):
        """ Compute the average precisions , given the bounding boxes.
        Code originally from https://github.com/rbgirshick/py-faster-rcnn.
        # Returns
            The average precision as computed in py-faster-rcnn.
        """
        average_precisions = {}
        for label in range(self.num_classes):
            false_positives = np.zeros((0,))
            true_positives  = np.zeros((0,))
            scores          = np.zeros((0,))
            num_annotations = 0.0
            for i in range(len(all_ground_truth)):
                detections           = all_detections[i][label]
                annotations          = all_ground_truth[i][label]
                num_annotations     += annotations.shape[0]
                detected_annotations = []
                for d in detections:
                    scores = np.append(scores, d[4])
                    if annotations.shape[0] == 0:
                        false_positives = np.append(false_positives, 1)
                        true_positives  = np.append(true_positives, 0)
                        continue
                    overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
                    assigned_annotation = np.argmax(overlaps, axis=1)
                    if assigned_annotation not in detected_annotations:
                        false_positives = np.append(false_positives, 0)
                        true_positives  = np.append(true_positives, 1)
                        detected_annotations.append(assigned_annotation)
                    else:
                        false_positives = np.append(false_positives, 1)
                        true_positives  = np.append(true_positives, 0)

            # no annotations -> AP for this class is 0 (is this correct?)
            if num_annotations == 0:
                average_precisions[label] = 0, 0
                continue

            # sort by score
            indices         = np.argsort(-scores)
            false_positives = false_positives[indices]
            true_positives  = true_positives[indices]

            # compute false positives and true positives
            false_positives = np.cumsum(false_positives)
            true_positives  = np.cumsum(true_positives)

            # compute recall and precision
            recall    = true_positives / num_annotations
            precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

            # compute average precision
            # first append sentinel values at the end
            mrec = np.concatenate(([0.], recall, [1.]))
            mpre = np.concatenate(([0.], precision, [0.]))

            # compute the precision envelope
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            average_precision = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
            average_precisions[label] = average_precision, num_annotations
        mean_average_precision = np.mean([x[0] for x in average_precisions.values()])
        return mean_average_precision



class RedirectModel(tf.keras.callbacks.Callback):
    """Callback which wraps another callback, but executed on a different model.
    Example
    -------
    ```python
    model = tf.keras.models.load_model('model.h5')
    model_checkpoint = ModelCheckpoint(filepath='snapshot.h5')
    parallel_model = multi_gpu_model(model, gpus=2)
    parallel_model.fit(X_train, Y_train, callbacks=[RedirectModel(model_checkpoint, model)])
    ```
    """

    def __init__(self, callback, model):
        """
        Parameters
        ----------
        model: tf.keras.models.Model
            model to use when executing callbacks.
        callback: tf.keras.callbacks.Callback
            callback to wrap.
        """
        super(RedirectModel, self).__init__()

        self.callback = callback
        self.redirect_model = model

    def on_epoch_begin(self, epoch, logs=None):
        self.callback.on_epoch_begin(epoch, logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        self.callback.on_epoch_end(epoch, logs=logs)

    def on_batch_begin(self, batch, logs=None):
        self.callback.on_batch_begin(batch, logs=logs)

    def on_batch_end(self, batch, logs=None):
        self.callback.on_batch_end(batch, logs=logs)

    def on_train_begin(self, logs=None):
        # overwrite the model with our custom model
        self.callback.set_model(self.redirect_model)
        self.callback.on_train_begin(logs=logs)

    def on_train_end(self, logs=None):
        self.callback.on_train_end(logs=logs)
