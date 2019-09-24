  
import tensorflow as tf
import numpy as np
import s3fs
import xmltodict
import logging
from utilities.bbox_overlap import compute_overlap
from data_generators.generator_base import DataGeneratorBase
import ast

logging.getLogger().setLevel(logging.INFO)

class GeneratorRetinaNet(DataGeneratorBase):
    """ The generator class that perform the preprocessing for the inputs of
    RetinaNet. It takes the data annotation table and defect map
    definition, and feed batch inputs and targets for training in RetinaNet
    """

    def __init__(
        self,
        config,
    ):  
        super().__init__(config)
        self.anchor_params = AnchorParameters(**config["MODEL"].get("ANCHORPARAMS",None))
        #retinanet uses anchors instead of raw bounding boxes 
        self.anchors = generate_pyramids_anchors(image_shape=self.output_shape, 
                                                 anchor_params=self.anchor_params)
        self.fs = s3fs.S3FileSystem(anon=False).open if "s3://" in self.data_dir else open

    def create_dataset_dict(self, df, transforms=None):
        df = df.copy()
        join_root_dir = self.get_join_root_dir_map(self.data_dir)
        self.defect_map = ast.literal_eval(df['label_names'].values[0])
        df['bbox_label_path'] = df['bbox_label_path'] \
                                        .fillna("") \
                                        .apply(join_root_dir)
        df[self.image_path] = df[self.image_path].apply(join_root_dir)

        dataset = tf.data.Dataset.from_tensor_slices(
            dict(image_path=df[self.image_path].values,
                 bbox_label_path=df['bbox_label_path'].values))
        dataset = dataset.map(self.__load_data, num_parallel_calls=self.num_parallel_calls)
        dataset = dataset.cache(self.cache_file_location(self.cache_dir))
        if self.repeat:
            dataset = dataset.repeat()
        if transforms is not None and transforms.has_transform():
            transform_map = self.__get_transform_map(transforms, self.output_shape,
                                                     self.output_image_channels,
                                                     self.output_image_type)
            dataset = dataset.map(transform_map)
        return dataset


    def create_dataset(self, df, transforms=None):
        dataset = self.create_dataset_dict(df, transforms)
        # FIXME: Also may wantn ot have a final transform to make the schema of data generator flexible
        #assinging bboxes to anchors and one hot encode classes
        dataset = dataset.map(self.__get_preprocess_targets(self.anchors,self.defect_map,self.batch_size))
        dataset = dataset.map(lambda row: (row["image"], (row["bbox_labels_bboxes"],row["bbox_labels_labels"])))
        dataset = dataset.batch(self.batch_size, drop_remainder=self.drop_remainder)
        dataset = dataset.prefetch(4)
        logging.info("==========================dataset=====================")
        logging.info(dataset)
        return dataset

    def create_inference_dataset(self, df, transforms=None):
        dataset = self.create_dataset_dict(df, transforms)
        # FIXME: Also may wantn ot have a final transform to make the schema of data generator flexible
        dataset = dataset.map(lambda row: (row["image"], (row["bbox_labels_bboxes"],row["bbox_labels_labels"]),
                                          row['original_image'],(row["original_bbox_labels_bboxes"],row["original_bbox_labels_labels"])))
        dataset = dataset.batch(1, drop_remainder=self.drop_remainder)
        dataset = dataset.prefetch(4)
        logging.info("==========================dataset=====================")
        logging.info(dataset)
        return dataset



    # def create_inference_dataset(self, df, transforms=None):
    #     dataset = self.create_dataset_dict(df, transforms)
    #     dataset = dataset.map(lambda row: (row["image"], row["segmentation_labels"], row[
    #     "original_image"], row["original_segmentation_labels"]))
    #     dataset = dataset.prefetch(4)
    #     logging.info("====================inference dataset=====================")
    #     logging.info(dataset)
    #     return dataset


    def __load_data(self, row):
        #Load the image and its label annotations
        image = self.load_image(row["image_path"])
        load_bboxes_func = self.__get_load_annotations(self.defect_map,self.fs)
        annos = tf.compat.v1.py_func(load_bboxes_func,
                                    [row["bbox_label_path"]],
                                    [tf.float64])[0]
        new_row = dict(
            image=image,
            bbox_labels=annos,
        )
        return new_row

    def __get_load_annotations(self,defect_map,fs):
        #construct the annotation loading function 
        # (it must not use arguments other than those in data itself)
        # unlike other constructed functions in this generator it is used inside another loader method and
        # thus takes explicit argument rather than `row`
        def load_bboxes_func(label_path):
            #returns annotations as np.array(n_Annotations,5) where 1 value is class and 4 are corners of bbox
            label_path = label_path.decode()
            annotations = [np.empty((0,1)),np.empty((0, 4))]
            if not label_path:
                return annotations
            with fs(label_path, "rb") as f:
                lbl = xmltodict.parse(f.read())
            # Read bounding box
            labels = []
            bboxes = []
            if "object" in lbl["annotation"].keys():  # If there is (are) bbox(es)
                object_annotation = lbl["annotation"]["object"]
                if not type(object_annotation) == list:
                    object_annotation = [object_annotation]
                for bbox_dict in object_annotation:
                    if isinstance(bbox_dict,str):
                        print(label_path,bbox_dict)
                    label = bbox_dict["name"]
                    if label not in defect_map:
                        continue
                    labels.append(defect_map[label])
                    xmin = int(float(bbox_dict["bndbox"]["xmin"]))
                    ymin = int(float(bbox_dict["bndbox"]["ymin"]))
                    xmax = int(float(bbox_dict["bndbox"]["xmax"]))
                    ymax = int(float(bbox_dict["bndbox"]["ymax"]))
                    bboxes.append([xmin, ymin, xmax, ymax])
            if len(labels) > 0:
                annotations = [np.array(labels).reshape((-1,1)),np.array(bboxes)]
            annotations = np.concatenate(annotations,axis=1).astype(float)
            return annotations
            
        return load_bboxes_func

    def __get_transform_map(self, transforms, output_shape, output_image_channels,
                            output_image_type):

        def transform_map(row):
            original_image = row["image"]
            original_labels = row["bbox_labels"]
            augmented = tf.compat.v1.py_func(transforms.apply_transforms,
                                             [row["image"], row['bbox_labels']],
                                             [output_image_type,tf.float64])
            logging.info(augmented)
            image = augmented[0]
            image.set_shape(output_shape + (output_image_channels,))
            logging.info(image)
            label = augmented[1]
            logging.info(label)
            row = {}
            row["image"] = image
            row["bbox_labels_bboxes"] = label[:,1:]
            row["bbox_labels_labels"] = label[:,0]
            row["original_image"] = original_image
            row["original_bbox_labels_bboxes"] = original_labels[:,1:]
            row["original_bbox_labels_labels"] = original_labels[:,0]

            return row

        return transform_map

    def __get_preprocess_targets(self, anchors,defect_map,batch_size):
        #construct preprocessing function for annotations
        def preprocess_targets_func(row):
            image, bboxes,labels = row['image'],row["bbox_labels_bboxes"],row["bbox_labels_labels"]
            # get model output targets for loss calculation, for each anchor box
            # in each pytamid level, we have regress_target for the box offset,
            # and class_target for classification probs
            regress_target, class_target = tf.compat.v1.py_func(
                lambda image,bboxes,labels: anchor_targets_bbox(anchors, image, [bboxes,labels], len(defect_map)),
                [image,bboxes,labels],
                [tf.float64,tf.float64]
            )
            row['bbox_labels_bboxes'] = regress_target
            row['bbox_labels_labels'] = class_target
            return row

        return preprocess_targets_func


def anchor_targets_bbox(
    anchors,
    image,
    annotations,
    num_classes,
    negative_overlap=0.4,
    positive_overlap=0.5,
):
    """ Generate anchor targets for bbox detection.
    Parameters
    ----------
    anchors: np.array
        shape (N, 4), anchor box coordinates (x1, y1, x2, y2) in all
        pyramid_levels for a image with given shape.
    image_group: list of np.array
        list of input images with shape (H, W, C) in a batch
    annotations_group: np.array
        bounding box information with class and corner coordinates
    num_classes: int
        Number of classes (defects) to predict.
    negative_overlap: float, optional
        IoU overlap threshold for negative anchors for backgroud class, i.e.
        anchors with IOU < negative_overlap for any bbox objects are considered
        background, and it's not associated with any bbox objects.
    positive_overlap: float, optional
        IoU overlap threshold for positive anchors i.e.
        anchors with IOU > positive_overlap for any bbox objects are considered
        to be associated with some bbox objects.
    Returns
    -------
    regression_batch: np.array
        shape (batch_size, N, 4 + 1), a batch that contains bounding-box
        regression targets for an image.
        where N is the number of anchors for an image, the first 4 columns
        define regression targets for (x1, y1, x2, y2) and the
        last column defines anchor states (-1 for ignore, 0 for bg, 1 for fg).
    labels_batch: np.array
        shape (batch_size, N, num_classes + 1), a batch that contains
        classification targets for an image. where
        num_classes + 1 = (class1, class2,..., classN, anchor_state),
        N is the number of anchors for an image and the
        last column defines anchor states (-1 for ignore, 0 for bg, 1 for fg).
    """
    regression_batch = np.zeros((anchors.shape[0], 4 + 1), dtype=float)
    labels_batch = np.zeros((anchors.shape[0], num_classes + 1), dtype=float)
    # compute labels and regression target
    #for some reason annotations have an extra dimension to them after transforms
    bboxes, labels = annotations
    if bboxes.shape[0]:
        # obtain indices of gt annotations with the greatest overlap
        positive_indices, ignore_indices, argmax_overlaps_inds = compute_gt_annotations(
            anchors, bboxes, negative_overlap, positive_overlap
        )

        # set the anchor state to ignore(-1)
        labels_batch[ignore_indices, -1] = -1
        # set the anchor state to foreground class (1)
        labels_batch[positive_indices, -1] = 1
        # otherwise, keep anchor state to background class (0)
        regression_batch[ignore_indices, -1] = -1
        regression_batch[positive_indices, -1] = 1

        # compute target class labels
        # assign positive label 1 to the corresponding class only for
        # anchors that reach positive threshold 0.5
        label_indices = labels[
            argmax_overlaps_inds[positive_indices]
        ].astype(int)
        labels_batch[positive_indices, label_indices] = 1
        # assign bounding box offsets to all anchors, even if the overlap
        # doesn't reach the threshold
        regression_batch[:, :-1] = bbox_transform(
            anchors, bboxes[argmax_overlaps_inds, :]
        )

    # ignore annotations outside of image
    if image.shape:
        anchors_centers = np.vstack(
            [
                (anchors[:, 0] + anchors[:, 2]) / 2,
                (anchors[:, 1] + anchors[:, 3]) / 2,
            ]
        ).T
        indices = np.logical_or(
            anchors_centers[:, 0] >= image.shape[1],
            anchors_centers[:, 1] >= image.shape[0],
        )

        labels_batch[indices, -1] = -1
        regression_batch[indices, -1] = -1
    return regression_batch, labels_batch


def compute_gt_annotations(
    anchors, bboxes, negative_overlap=0.4, positive_overlap=0.5
):
    """ Obtain indices of ground-truth annotations with the greatest overlap.
    Parameters
    ----------
    anchors: np.array
        shape (N, 4), anchor box coordinates (x1, y1, x2, y2) of a image.
    annotations: np.array
        shape (M, 4), grouth-truth bouding box annotations (x1, y1, x2, y2)
        of a image.
    negative_overlap: float, optional
        IoU overlap for negative anchors
        (all anchors with overlap < negative_overlap are negative).
    positive_overlap: float, optional
        IoU overlap or positive anchors
        (all anchors with overlap > positive_overlap are positive).
    Returns
    -------
    positive_indices: np.array
        shape (N, ) type bool, indices of positive anchors (>0.5)
    ignore_indices: np.array
        shape (N, ) type bool, indices of ignored anchors (0.4-0.5)
    argmax_overlaps_inds: np.array
        shape (N, ) type int, ordered overlaps indices for each anchor,
        which annotation has highest overlap.
        if the anchor has no overlap to any bonding box, assign 0 for
        the first bounding box
    """
    # overlaps in shape (number of total anchors, number of annotations)
    overlaps = compute_overlap(
        anchors.astype(np.float64), bboxes.astype(np.float64)
    )
    # for each anchor, which annotation has highest overlap,
    # if all overlaps for one anchor are 0, this anchor is background, assign 0
    argmax_overlaps_inds = np.argmax(overlaps, axis=1)
    max_overlaps = overlaps[np.arange(overlaps.shape[0]), argmax_overlaps_inds]
    # assign "dont care" labels
    positive_indices = max_overlaps >= positive_overlap
    ignore_indices = (max_overlaps > negative_overlap) & ~positive_indices

    return positive_indices, ignore_indices, argmax_overlaps_inds


def bbox_transform(anchors, gt_boxes, mean=None, std=None):
    """Compute bounding-box regression targets for an image.
    Parameters
    ----------
    anchors: np.array
        shape (N, 4), anchor box coordinates (x1, y1, x2, y2) in all
        pyramid_levels for a image with given shape.
    gt_boxes: np.array
        shape (N, 4) coordinates of ground truth bounding box that has highest
        IOU overlap for each anchor box
    mean: np.array, optional
        shape (4,) the mean for the normalization of regression targets for
        each coordinates (x1, y1, x2, y2)
    mean: np.array, optional
        shape (4,) the mean for the normalization of regression targets for
        each coordinates (x1, y1, x2, y2)
    """

    if mean is None:
        mean = np.array([0, 0, 0, 0])
    if std is None:
        std = np.array([0.2, 0.2, 0.2, 0.2])

    anchor_widths = anchors[:, 2] - anchors[:, 0]
    anchor_heights = anchors[:, 3] - anchors[:, 1]

    # get regression target from difference between anchors box coordinates
    # and ground-truth box coordinates
    anchor_widths_and_heights = np.stack(
        [anchor_widths, anchor_heights, anchor_widths, anchor_heights], axis=1
    )
    targets = (gt_boxes - anchors) / anchor_widths_and_heights

    targets = (targets - mean) / std

    return targets
    


class AnchorParameters:
    """ The parameteres that define how anchors are generated.
    Parameters
    ----------
    sizes: list of int, optional
        List of sizes to use. Each size corresponds to one feature level.
    strides: list of int, optional
        List of strides to use. Each stride correspond to one feature level.
    ratios: np.array of float, optional
        List of ratios to use per location in a feature map.
    scales: np.array of float, optional
        List of scales to use per location in a feature map.
    """

    def __init__(self, sizes=None, strides=None, ratios=None, scales=None):
        self.sizes = [16, 32, 64, 128, 256] if sizes is None else sizes
        self.strides = [8, 16, 32, 64, 128] if strides is None else strides
        self.ratios = np.array([0.5, 1, 2]) if ratios is None else ratios
        self.scales = (
            np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
            if scales is None
            else scales
        )

    def num_anchors(self):
        return len(self.ratios) * len(self.scales)


def generate_pyramids_anchors(image_shape, anchor_params, pyramid_levels=None):
    """ generate anchor boxes coordinates of all pyramids levels for a image
    with given shape.
    Parameters
    ----------
    image_shape: tuple of int
        The shape (H, W, C) of the image.
    anchor_params: AnchorParameters
        Anchor parameters including base_size, strides, ratio, scales.
    pyramid_levels: list of int, optional
        sepcify which pyramid level to use (defaults to [3, 4, 5, 6, 7]).
    Returns
    -------
    np.array
        shape (N, 4), all anchor box information in specified pyramid_levels
        for a image with given shape, containing the (x1, y1, x2, y2)
        coordinates for the anchors.
        where N = (H3*W3 + ... + H7*W7) * num_anchors,
        num_anchors = nums_ratios * nums_scales
    """

    if pyramid_levels is None:
        pyramid_levels = [3, 4, 5, 6, 7]

    # Guess shapes based on pyramid levels.
    image_shapes = [
        (np.array(image_shape[:2]) + 2 ** x - 1) // (2 ** x) for x in pyramid_levels
    ]

    # compute anchors over all pyramid levels
    all_anchors = np.zeros((0, 4))
    for idx, p in enumerate(pyramid_levels):
        anchors = generate_anchors(
            base_size=anchor_params.sizes[idx],
            ratios=anchor_params.ratios,
            scales=anchor_params.scales,
        )
        shifted_anchors = shift(image_shapes[idx], anchor_params.strides[idx], anchors)
        all_anchors = np.append(all_anchors, shifted_anchors, axis=0)

    return all_anchors


def generate_anchors(base_size, ratios, scales):
    """
    Generate anchor boxes coordinates (x1, y1, x2, y2) for one pixel location
    in a pyramid level based on specified base_size and multiple aspect ratios
    and multiple scales
    Parameters
    ----------
    base_size: list of int
        the base shape size for achor boxes in specified pyramid level.
        The area of the anchor box without scaling is (base_size * base_size).
    ratios: list of float
        the aspect ratios (width / height) for different anchor boxes in one
        pixel location
    scales: list of float
        the scaling factor for different anchor boxes in one pixel location.
    Returns
    -------
    np.ndarray
        shape (num_ratios * num_sclaes, 4), anchor boxes coordinates
        (x1, y1, x2, y2) for one pixel location in a pyramid level.
    """

    num_anchors = len(ratios) * len(scales)

    # initialize output anchors
    anchors = np.zeros((num_anchors, 4))

    # scale base_size
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

    # compute areas of anchors
    areas = anchors[:, 2] * anchors[:, 3]

    # correct for ratios
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors


def shift(shape, stride, anchors):
    """ Produce shifted anchors based on shape of the map and stride size.
    Parameters
    ----------
    shape : tuple of int
        Shape (H, W) to shift the anchors over. Each pyramid level would have
        different down-scaled shape by the factor of 2.
    stride : int
        Stride to shift the anchors with over the shape.
    anchors: np.array
        shape (nums_ratios * nums_scales, 4), the anchors to apply at each
        location.
    Returns
    -------
    np.array
        shape (H * W * nums_ratios * nums_scales, 4), anchor boxes coordinates
        (x1, y1, x2, y2) for all pixel locations in a pyramid level.
    """

    # create a grid starting from half stride from the top left corner
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack(
        (shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())
    ).transpose()

    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose(
        (1, 0, 2)
    )
    all_anchors = all_anchors.reshape((K * A, 4))

    return all_anchors

