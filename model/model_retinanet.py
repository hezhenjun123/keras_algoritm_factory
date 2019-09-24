import tensorflow as tf
import numpy as np
from model.model_base import ModelBase
from data_generators.generator_retinanet import generate_anchors, AnchorParameters


class ModelRetinaNet(ModelBase):
    """ Construct a RetinaNet model on top of a backbone.
    This model is the minimum model necessary for training. The model outputs
    raw regression and classification predictions for each anchor box
    and each pyramid level from the input image.
    Parameters
    ----------
    backbone: Backbone class instance
        The Backbone class instance that contains the info about backbone and
        its Keras.Model.
    num_classes: int
        Number of classes to classify.
    num_anchors: int
        Number of base anchors in each pixel location in each pyramid level.
    name: str
        Name of the model.
    Returns
    -------
    tf.keras.models.Model
        the RetinaNet training model that takes an image as input and outputs
        generated anchors and the result from each submodel on every pyramid
        level. The order of the outputs is as defined in submodels:
        [regression, classification], each with shape:
        regression: (Batch_size, (H3*W3+...+H7*W7)*num_anchors, 4),
        classification:(Batch_size, (H3*W3+...+H7*W7)*num_anchors, num_classes)
        where H3, W3, ..., H7, W7 are the height and width shape from pyramid
        features P3,...,P7
    """
    def __init__(self,config):
        super().__init__(config)
        self.backbone_name = config["MODEL"]["BACKBONE"]
        self.num_anchors=AnchorParameters(**config["MODEL"].get("ANCHORPARAMS",None)).num_anchors()
        self.model =  self.get_or_load_model()

    def create_model(self):
        inputs, outputs = Backbone(self.backbone_name,weights='imagenet').get_pyramid_inputs()
        C3, C4, C5 = outputs

        # compute pyramid features as per https://arxiv.org/abs/1708.02002
        features = self.create_pyramid_features(C3, C4, C5)

        # for all pyramid levels, run available submodels
        model_regression = self.DefaultRegressionModel()
        pyramid_regression = [model_regression(f) for f in features]
        pyramid_regression = tf.keras.layers.Concatenate(axis=1, name="regression")(
            pyramid_regression
        )

        model_classification = self.DefaultClassificationModel()
        pyramid_classification = [model_classification(f) for f in features]
        pyramid_classification = tf.keras.layers.Concatenate(axis=1, name="classification")(
            pyramid_classification
        )

        pyramids = [pyramid_regression, pyramid_classification]

        return tf.keras.models.Model(inputs=inputs, outputs=pyramids, name='retinanet')

    def generate_custom_objects(self):
        custom_objects = {'UpsampleLike':UpsampleLike,
                        'Anchors':Anchors,
                        'RegressBoxes':RegressBoxes,
                        'FilterDetections':FilterDetections,
                        'ClipBoxes':ClipBoxes,
                        'PriorProbability':PriorProbability}
        self.custom_objects = custom_objects

    def RetinaNetBbox(
        self,
        class_specific_filter=False,
        anchor_params=None,
        score_threshold=0.05,
        nms_threshold=0.5,
        max_detections=100,
        name="retinanet-bbox",
        **kwargs
    ):
        """Construct a RetinaNet Bounding box model for inference on top of
        regular RetinaNet training modela to output boxes directly.
        This model uses the minimum retinanet model and appends a few layers to
        compute boxes within the graph. These layers include applying the
        regression values to the anchors and performing non_max_suppression.
        Parameters
        ----------
        model: tf.keras.models.Model
            Minimum RetinaNet training model created from retinanet().
        class_specific_filter: bool
            Whether to use class specific filtering or filter for the best scoring
            class only.
        anchor_params: AnchorParameters
            Anchor parameters including base_size, strides, ratio, scales.
            If None, default values are used.
        score_threshold: float
            Threshold used to prefilter the boxes with.
        nms_threshold: float
                Threshold for the IoU value to determine when a box should be
                suppressed in non_max_suppression for bbox filtering
        max_detections: int
            Maximum number of detections to keep.
        name: str
            Name of the model.
        Returns
        -------
        tf.keras.models.Model
            The which takes an image as input and outputs the bbox detections.
            the outputs is a list of tf.Tensor: [boxes, scores, labels]
            boxes: shape (batch_size, max_detections, 4) and contains the
                (x1, y1, x2, y2) of the non-suppressed boxes.
            scores: shape (batch_size, max_detections) and contains the
                probability scores(0-1) of the predicted class.
            labels: shape (batch_size, max_detections) and contains the
                predicted label.
            In case there are less than max_detections detections, the tensors
            are padded with -1's.
        """
        # if no anchor parameters are passed, use default values
        if anchor_params is None:
            anchor_params = AnchorParameters()

        # compute the anchors
        features = [
            self.model.get_layer(p_name).output for p_name in ["P3", "P4", "P5", "P6", "P7"]
        ]
        anchors = [
            Anchors(
                # the height/width anchor if it square
                size=anchor_params.sizes[i],
                # if input image is 512x512 with feature map 128x128, then stride=4
                stride=anchor_params.strides[i],
                # height to width aspect ratio for more boxes in one location
                ratios=anchor_params.ratios,
                # scale the anchor size to produce more boxes in one location
                scales=anchor_params.scales,
                name="anchors_{}".format(i),
            )(f)
            for i, f in enumerate(features)
        ]
        anchors = tf.keras.layers.Concatenate(axis=1, name="anchors")(anchors)

        # obtain model regression and classification outputs
        regression = self.model.outputs[0]
        classification = self.model.outputs[1]

        # apply predicted regression to anchors
        # adjust the anchor box locations with regression model output
        boxes = RegressBoxes(name="boxes")([anchors, regression])
        # constraint the predicted box locations with [0,H], [0,W] of input image
        boxes = ClipBoxes(name="clipped_boxes")([self.model.inputs[0], boxes])

        # filter detections (apply NMS / score threshold / select top-k)
        detections = FilterDetections(
            class_specific_filter=class_specific_filter,
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            max_detections=max_detections,
            name="filtered_detections",
        )([boxes, classification])

        # construct the model
        return tf.keras.models.Model(inputs=self.model.inputs, outputs=detections, name=name)


    def DefaultClassificationModel(
        self,
        pyramid_feature_size=256,
        prior_probability=0.01,
        classification_feature_size=256,
        name="classification_submodel",
    ):
        """ Creates the default regression submodel.
        Parameters
        ----------
        num_classes: int
            Number of classes to predict a score for at each feature level.
        num_anchors: int
            Number of anchors to predict classification scores for at each
            feature level.
        pyramid_feature_size: float
            The number of filters to expect from the feature pyramid levels.
        classification_feature_size : int
            The number of filters to use in the layers in the classification
            submodel.
        name: str
            The name of the submodel.
        Returns
        -------
        tf.keras.models.Model
            A model that predicts classes for each anchor.
        """
        options = {"kernel_size": 3, "strides": 1, "padding": "same"}

        inputs = tf.keras.layers.Input(shape=(None, None, pyramid_feature_size))

        outputs = inputs
        # 4 conv layers for feature extraction
        for i in range(4):
            outputs = tf.keras.layers.Conv2D(
                filters=classification_feature_size,
                activation="relu",
                name="pyramid_classification_{}".format(i),
                kernel_initializer=tf.keras.initializers.RandomNormal(
                    mean=0.0, stddev=0.01, seed=None
                ),
                bias_initializer="zeros",
                **options
            )(outputs)

        # last conv layer to logits ouput for each class and each anchor
        outputs = tf.keras.layers.Conv2D(
            filters=self.num_classes * self.num_anchors,
            kernel_initializer=tf.keras.initializers.RandomNormal(
                mean=0.0, stddev=0.01, seed=None
            ),
            bias_initializer=PriorProbability(probability=prior_probability),
            name="pyramid_classification",
            **options
        )(outputs)

        # reshape output and apply sigmoid
        outputs = tf.keras.layers.Reshape(
            (-1, self.num_classes), name="pyramid_classification_reshape"
        )(outputs)
        outputs = tf.keras.layers.Activation(
            "sigmoid", name="pyramid_classification_sigmoid"
        )(outputs)

        return tf.keras.models.Model(inputs=inputs, outputs=outputs, name=name)


    def DefaultRegressionModel(
        self,
        num_values=4,
        pyramid_feature_size=256,
        regression_feature_size=256,
        name="regression_submodel",
    ):
        """ Creates the default regression submodel.
        Parameters
        ----------
        num_anchors: int
            Number of anchors to regress for each feature level.
        num_values: int
            Number of values to regress. By default, they are 4 box coordinates
            offset (dx1, dy1, dx2, dy2) form anchor boxes.
        pyramid_feature_size: int
            The number of filters to expect from the feature pyramid levels.
        regression_feature_size: int
            The number of filters to use in the layers in the regression submodel.
        name: str
            The name of the submodel.
        Returns
        -------
        tf.keras.models.Model
            A model that predicts regression values for each anchor.
        """
        # All new conv layers except the final one in the
        # RetinaNet (classification) subnets are initialized
        # with bias b = 0 and a Gaussian weight fill with stddev = 0.01.
        options = {
            "kernel_size": 3,
            "strides": 1,
            "padding": "same",
            "kernel_initializer": tf.keras.initializers.RandomNormal(
                mean=0.0, stddev=0.01, seed=None
            ),
            "bias_initializer": "zeros",
        }

        inputs = tf.keras.layers.Input(shape=(None, None, pyramid_feature_size))

        # 4 conv layers for feature extraction
        outputs = inputs
        for i in range(4):
            outputs = tf.keras.layers.Conv2D(
                filters=regression_feature_size,
                activation="relu",
                name="pyramid_regression_{}".format(i),
                **options
            )(outputs)

        # last conv layer to 4 box coordinates offset for each anchor
        outputs = tf.keras.layers.Conv2D(
            self.num_anchors * num_values, name="pyramid_regression", **options
        )(outputs)

        outputs = tf.keras.layers.Reshape(
            (-1, num_values), name="pyramid_regression_reshape"
        )(outputs)

        return tf.keras.models.Model(inputs=inputs, outputs=outputs, name=name)


    def create_pyramid_features(self,C3, C4, C5, feature_size=256):
        """ Creates the FPN layers on top of the backbone features layers.
        If the image input shape has (B, H, W, ?), the input feature layers
        contains:
        C3: the last layer output with shape (B, H/8, W/8, ?),
        C4: the last layer output with shape (B, H/16, W/16, ?),
        C5: the last layer output with shape (B, H/32, W/32, ?).
        Parameters
        ----------
        C3: tf.Tensor
            Feature stage C3 from the backbone.
        C4: tf.Tensor
            Feature stage C4 from the backbone.
        C5: tf.Tensor
            Feature stage C5 from the backbone.
        feature_size: int
            The feature size to use for the resulting feature levels.
        Returns
        -------
        list of tf.Tensor
            A list of feature levels [P3, P4, P5, P6, P7].
        """
        # upsample C5 to get P5 from the FPN paper
        P5 = tf.keras.layers.Conv2D(
            feature_size, kernel_size=1, strides=1, padding="same", name="C5_reduced"
        )(C5)
        P5_upsampled = UpsampleLike(name="P5_upsampled")([P5, C4])
        P5 = tf.keras.layers.Conv2D(
            feature_size, kernel_size=3, strides=1, padding="same", name="P5"
        )(P5)

        # add P5 elementwise to C4
        P4 = tf.keras.layers.Conv2D(
            feature_size, kernel_size=1, strides=1, padding="same", name="C4_reduced"
        )(C4)
        P4 = tf.keras.layers.Add(name="P4_merged")([P5_upsampled, P4])
        P4_upsampled = UpsampleLike(name="P4_upsampled")([P4, C3])
        P4 = tf.keras.layers.Conv2D(
            feature_size, kernel_size=3, strides=1, padding="same", name="P4"
        )(P4)

        # add P4 elementwise to C3
        P3 = tf.keras.layers.Conv2D(
            feature_size, kernel_size=1, strides=1, padding="same", name="C3_reduced"
        )(C3)
        P3 = tf.keras.layers.Add(name="P3_merged")([P4_upsampled, P3])
        P3 = tf.keras.layers.Conv2D(
            feature_size, kernel_size=3, strides=1, padding="same", name="P3"
        )(P3)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        P6 = tf.keras.layers.Conv2D(
            feature_size, kernel_size=3, strides=2, padding="same", name="P6"
        )(C5)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        P7 = tf.keras.layers.Activation("relu", name="C6_relu")(P6)
        P7 = tf.keras.layers.Conv2D(
            feature_size, kernel_size=3, strides=2, padding="same", name="P7"
        )(P7)

        return [P3, P4, P5, P6, P7]



class UpsampleLike(tf.keras.layers.Layer):
    """Layer for upsampling a Tensor to be the same shape as another Tensor.
    """

    def call(self, inputs, **kwargs):
        source, target = inputs
        target_shape = tf.shape(target)
        return tf.image.resize(
            images=source,
            size=(target_shape[1], target_shape[2]),
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        )

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],) + input_shape[1][1:3] + (input_shape[0][-1],)


class PriorProbability(tf.keras.initializers.Initializer):
    """ Apply a prior probability to the weights.
    """

    def __init__(self, probability=0.01):
        self.probability = probability

    def get_config(self):
        return {"probability": self.probability}

    def __call__(self, shape, dtype=None, partition_info=None):
        # set bias to -log((1 - p)/p) for foreground
        result = tf.ones(shape, dtype=dtype) * -tf.math.log(
            (1 - self.probability) / self.probability
        )

        return result


class Anchors(tf.keras.layers.Layer):
    """ Keras layer for generating achors for a given shape in one pyramid
    freature level.
    """

    def __init__(self, size, stride, ratios, scales, *args, **kwargs):
        """ Initializer for an Anchors layer.
        Parameters
        ----------
        sizes: int
            Anchor size corresponds to one feature level.
        strides: int
            Anchor bix stride correspond to one feature level.
        ratios: np.array
            List of ratios to use per location in a feature map.
        scales: np.array
            List of scales to use per location in a feature map.
        """
        self.size = size
        self.stride = stride
        self.ratios = ratios
        self.scales = scales

        self.num_anchors = len(ratios) * len(scales)
        self.anchors = tf.constant(
            generate_anchors(base_size=size, ratios=ratios, scales=scales),
            dtype=tf.float32,
        )

        super(Anchors, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        features = inputs
        features_shape = tf.shape(features)
        # generate proposals from bbox deltas and shifted anchors
        anchors = self.shift(features_shape[1:3], self.stride, self.anchors)
        anchors = tf.tile(tf.expand_dims(anchors, axis=0), (features_shape[0], 1, 1))
        return anchors

    def shift(self, shape, stride, anchors):
        """ Produce shifted anchors based on shape of the map and stride size.
        This function might be removed later by using tf.py_func from
        preprocess.shift()
        Parameters
        ----------
        shape : tf.Tensor
            Shape(H,W) to shift the anchors over. Each pyramid level would have
            different down-scaled shape by the factor of 2.
        stride : int
            Stride to shift the anchors with over the shape.
        anchors: tf.Tensor
            shape (nums_ratios * nums_scales, 4), the anchors to apply at
            each location.
        Returns
        -------
        tf.Tensor
            shape (H * W * nums_ratios * nums_scales, 4), anchor boxes
            coordinates (x1, y1, x2, y2) for all pixel locations in a pyramid
            level.
        """
        shift_x = (
            tf.cast(tf.range(0, shape[1]), tf.float32) + tf.constant(0.5)
        ) * stride
        shift_y = (
            tf.cast(tf.range(0, shape[0]), tf.float32) + tf.constant(0.5)
        ) * stride

        shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
        shift_x = tf.reshape(shift_x, [-1])
        shift_y = tf.reshape(shift_y, [-1])

        shifts = tf.stack([shift_x, shift_y, shift_x, shift_y], axis=0)
        shifts = tf.transpose(shifts)
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)s
        # reshape to (K*A, 4) shifted anchors
        A = tf.shape(anchors)[0]  # number of anchors at one pixel point
        K = tf.shape(shifts)[0]  # number of base points = feat_h * feat_w

        shifted_anchors = tf.reshape(anchors, [1, A, 4]) + tf.cast(
            tf.reshape(shifts, [K, 1, 4]), tf.float32
        )
        shifted_anchors = tf.reshape(shifted_anchors, [K * A, 4])

        return shifted_anchors

    def compute_output_shape(self, input_shape):
        if None not in input_shape[1:]:
            total = np.prod(input_shape[1:3]) * self.num_anchors
            return (input_shape[0], total, 4)
        else:
            return (input_shape[0], None, 4)

    def get_config(self):
        config = super(Anchors, self).get_config()
        config.update(
            {
                "size": self.size,
                "stride": self.stride,
                "ratios": self.ratios,
                "scales": self.scales,
            }
        )

        return config


class RegressBoxes(tf.keras.layers.Layer):
    """ Keras layer for applying regression values to boxes.
    """

    def __init__(self, mean=None, std=None, *args, **kwargs):
        """ Initializer for the RegressBoxes layer.
        Parameters
        ----------
        mean: np.array
            The mean value of the regression values used for normalization.
        std: np.array
            The standard value of the regression values used for normalization.
        """
        if mean is None:
            mean = np.array([0, 0, 0, 0])
        if std is None:
            std = np.array([0.2, 0.2, 0.2, 0.2])

        self.mean = mean
        self.std = std
        super(RegressBoxes, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        anchors, regression = inputs

        width = anchors[:, :, 2] - anchors[:, :, 0]
        height = anchors[:, :, 3] - anchors[:, :, 1]

        x1 = (
            anchors[:, :, 0]
            + (regression[:, :, 0] * self.std[0] + self.mean[0]) * width
        )
        y1 = (
            anchors[:, :, 1]
            + (regression[:, :, 1] * self.std[1] + self.mean[1]) * height
        )
        x2 = (
            anchors[:, :, 2]
            + (regression[:, :, 2] * self.std[2] + self.mean[2]) * width
        )
        y2 = (
            anchors[:, :, 3]
            + (regression[:, :, 3] * self.std[3] + self.mean[3]) * height
        )

        pred_boxes = tf.stack([x1, y1, x2, y2], axis=2)

        return pred_boxes

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super(RegressBoxes, self).get_config()
        config.update({"mean": self.mean, "std": self.std})

        return config


class ClipBoxes(tf.keras.layers.Layer):
    """ Keras layer to clip box values to lie inside a given shape.
    """

    def call(self, inputs, **kwargs):
        image, boxes = inputs
        shape = tf.cast(tf.shape(image), tf.float32)
        height = shape[1]
        width = shape[2]

        x1 = tf.clip_by_value(boxes[:, :, 0], 0, width)
        y1 = tf.clip_by_value(boxes[:, :, 1], 0, height)
        x2 = tf.clip_by_value(boxes[:, :, 2], 0, width)
        y2 = tf.clip_by_value(boxes[:, :, 3], 0, height)

        return tf.stack([x1, y1, x2, y2], axis=2)

    def compute_output_shape(self, input_shape):
        return input_shape[1]


class FilterDetections(tf.keras.layers.Layer):
    """ Keras layer for filtering detections using score threshold and NMS.
    """

    def __init__(
        self,
        class_specific_filter=False,
        nms_threshold=0.5,
        score_threshold=0.05,
        max_detections=100,
        parallel_iterations=32,
        **kwargs
    ):
        """ Filters detections using score threshold, non_max_suppression and
        selecting the top-k detections.
        Parameters
        ----------
        class_specific_filter: bool
            Whether to perform filtering per class, or take the best scoring
            class and filter those.
        nms_threshold: float
            Threshold for the IoU value to determine when a box should be
            suppressed in non_max_suppression for bbox filtering
        score_threshold: float
            Threshold used to prefilter the boxes with.
        max_detections: int
            Maximum number of detections to keep.
        parallel_iterations: int
            Number of batch items to process in parallel.
        """
        self.class_specific_filter = class_specific_filter
        self.nms_threshold = nms_threshold
        self.score_threshold = score_threshold
        self.max_detections = max_detections
        self.parallel_iterations = parallel_iterations
        super(FilterDetections, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """ Constructs the NMS graph.
        Parameters
        ----------
        inputs : List of tf.Tensors
            [boxes, classification] model outputs from retinanet().
            boxes: tf.Tensor
                shape (batch_size, num_boxes, 4), boxes in (x1, y1, x2, y2)
                format.
            classification: tf.Tensor
                shape (batch_size, num_boxes, num_classes), the classification
                scores.
        Returns
        -------
        list of tf.Tensor
            A list of [boxes, scores, labels].
            boxes: shape (batch_size, max_detections, 4) and contains the
                (x1, y1, x2, y2) of the non-suppressed boxes.
            scores: shape (batch_size, max_detections) and contains the
                probability scores(0-1) of the predicted class.
            labels: shape (batch_size, max_detections) and contains the
                predicted label.
            In case there are less than max_detections detections, the tensors
            are padded with -1's.
        """

        def filter_detections(inputs):
            """ Filter detections using the boxes and classification values.
            Parameters
            ----------
            inputs: list of tf.Tensors
                boxes: tf.Tensor
                    shape (num_boxes, 4), boxes in (x1, y1, x2, y2) format.
                classification: tf.Tensor
                    shape (num_boxes, num_classes), the classification scores.
            list of tf.Tensor
                A list of [boxes, scores, labels].
                boxes: shape (max_detections, 4)
                scores: shape (max_detections,)
                labels: shape (max_detections,)
            """

            def _filter_detections(scores, labels):
                # threshold based on score
                indices = tf.where(tf.greater(scores, self.score_threshold))

                filtered_boxes = tf.gather_nd(boxes, indices)
                filtered_scores = tf.gather(scores, indices)[:, 0]

                # perform NMS
                nms_indices = tf.image.non_max_suppression(
                    filtered_boxes,
                    filtered_scores,
                    max_output_size=self.max_detections,
                    iou_threshold=self.nms_threshold,
                )

                # filter indices based on NMS
                indices = tf.gather(indices, nms_indices)

                # add indices to list of all indices
                labels = tf.gather_nd(labels, indices)
                indices = tf.stack([indices[:, 0], labels], axis=1)

                return indices

            boxes = inputs[0]
            classification = inputs[1]

            if self.class_specific_filter:
                all_indices = []
                # perform per class filtering
                for c in range(int(classification.shape[1])):
                    scores = classification[:, c]
                    labels = c * tf.ones((tf.shape(scores)[0],), dtype=tf.int64)
                    all_indices.append(_filter_detections(scores, labels))

                # concatenate indices to single tensor
                indices = tf.concat(all_indices, axis=0)
            else:
                scores = tf.reduce_max(classification, axis=1)
                labels = tf.argmax(classification, axis=1)
                indices = _filter_detections(scores, labels)

            # select top k
            scores = tf.gather_nd(classification, indices)
            labels = indices[:, 1]
            scores, top_indices = tf.math.top_k(
                scores, k=tf.minimum(self.max_detections, tf.shape(scores)[0])
            )

            # filter input using the final set of indices
            indices = tf.gather(indices[:, 0], top_indices)
            boxes = tf.gather(boxes, indices)
            labels = tf.gather(labels, top_indices)

            # zero pad the outputs
            pad_size = tf.maximum(0, self.max_detections - tf.shape(scores)[0])
            boxes = tf.pad(boxes, [[0, pad_size], [0, 0]], constant_values=-1)
            scores = tf.pad(scores, [[0, pad_size]], constant_values=-1)
            labels = tf.pad(labels, [[0, pad_size]], constant_values=-1)
            labels = tf.cast(labels, tf.int32)

            # set shapes, since we know what they are
            boxes.set_shape([self.max_detections, 4])
            scores.set_shape([self.max_detections])
            labels.set_shape([self.max_detections])

            return [boxes, scores, labels]

        # call filter_detections on each batch
        outputs = tf.map_fn(
            filter_detections,
            elems=inputs,
            dtype=[tf.float32, tf.float32, tf.int32],
            parallel_iterations=self.parallel_iterations,
        )
        return outputs

    def compute_output_shape(self, input_shape):
        return [
            (input_shape[0][0], self.max_detections, 4),
            (input_shape[1][0], self.max_detections),
            (input_shape[1][0], self.max_detections),
        ]

    def compute_mask(self, inputs, mask=None):
        """ This is required in Keras when there is more than 1 output.
        """
        return (len(inputs) + 1) * [None]

    def get_config(self):
        config = super(FilterDetections, self).get_config()
        config.update(
            {
                "class_specific_filter": self.class_specific_filter,
                "nms_threshold": self.nms_threshold,
                "score_threshold": self.score_threshold,
                "max_detections": self.max_detections,
                "parallel_iterations": self.parallel_iterations,
            }
        )

        return config


# from .classification_models.resnet import ResNet18, ResNet34
# from .classification_models.weights import weights_collection
# from .classification_models.utils import load_model_weights


class Backbone:
    """ The backbone class for RetinaNet.
    It includes network atchitecture imported from tf.keras.applications and
    describes how it is connected to the rest of RestinaNet.
    """

    def __init__(
        self, backbone_name, input_shape=(None, None, 3), weights=None, trainable=True
    ):
        """Initialize the backbone model.
        The Keras.Model can be accessed with self.model
        Parameters
        ----------
        backbone_name: str
            Name of the backbone layer
        input_shape: tuple of int
            Inputs to Network (Height, Width, channels). Defaults to (None, None, 3)).
        weights: str
            Pretained weights to be used for example imagenet.  Defaults to None.
        trainable: bool
            Set backbone layer as trainable.  Default is False
        """
        if backbone_name == "resnet50":
            self.model = tf.keras.applications.ResNet50(
                input_shape=input_shape, include_top=False, weights=weights
            )
            self.pyramid_layers = ["activation_21", "activation_39", "activation_48"]
        # elif backbone_name == "resnet34":
        #     self.model = ResNet34(input_shape=input_shape, include_top=False)
        #     self.pyramid_layers = ["stage3_unit1_relu1", "stage4_unit1_relu1", "relu1"]
        # elif backbone_name == "resnet18":
        #     self.model = ResNet18(input_shape=input_shape, include_top=False)
        #     self.pyramid_layers = ["stage3_unit1_relu1", "stage4_unit1_relu1", "relu1"]
        elif backbone_name == "vgg16":
            self.model = tf.keras.applications.VGG16(
                input_shape=input_shape, include_top=False, weights=weights
            )
            self.pyramid_layers = ["block3_pool", "block4_pool", "block5_pool"]
        elif backbone_name == "mobilenet-v2":
            self.model = tf.keras.applications.MobileNetV2(
                input_shape=input_shape, include_top=False, weights=weights, alpha=1.0
            )
            self.pyramid_layers = [
                "block_6_expand_relu",
                "block_13_expand_relu",
                "out_relu",
            ]
        else:
            raise ValueError(f"Unknown Backbone provided: {backbone_name}")
        # Load weights for Resent18 and Resnet34
        # if backbone_name in ["resnet18", "resnet34"] and weights:
        #     load_model_weights(
        #         weights_collection,
        #         self.model,
        #         backbone_name,
        #         weights,
        #         classes=None,
        #         include_top=False,
        #     )
        for layer in self.model.layers:
            layer.trainable = trainable

    def get_pyramid_inputs(self, modifier=None, **kwargs):
        """Provide the pyramid inputs to constructs a retinanet model.
        Notes
        -----
        This function also modify the Keras.Model (self.model) stored in the
        class instance, which is changed by the modifier function, and it
        outputs will contain 3 layer outputs (C3, C4, C5) used for RetinaNet.
        Parameters
        ----------
        modifier: A function handler which can modify the backbone before
            using it in retinanet (can be used to freeze backbone layers)
        Returns
        -------
        inputs: tf.Tensors
            input to the model.
        outputs: tf.Tensors
            C3, C4, C5 layer outpus from model to contruct  pyramid features
        """
        # invoke modifier if given
        if modifier:
            self.model = modifier(self.model)

        # one default modifer to add last activation output of each residual
        # stage in of C3, C4, C5 to model ouput
        outputs = []
        for name in self.pyramid_layers:
            outputs.append(self.model.get_layer(name).output)
        self.model.outputs = outputs

        # create the full model
        return self.model.inputs, self.model.outputs