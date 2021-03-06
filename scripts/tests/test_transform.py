from transforms.transform_base import TransformBase, imagenet_preprocess
import albumentations as A
import cv2
import numpy as np

#helper functions
def transform_plus_1():

    def func(image=None, mask=None):
        if image:
            image[0] = image[0] + 1
        if mask:
            mask[0] = mask[0] + 1
        return {"image": image, "mask": mask}

    return func


def transform_mul_2():

    def func(image=None, mask=None):
        if image:
            image[0] = image[0] * 2
        if mask:
            mask[0] = mask[0] * 2
        return {"image": image, "mask": mask}

    return func


#tests begin here
def test_transform_image():
    image = [0, 0, 0]
    label = [1, 1, 1]
    augmented = TransformBase.Image(transform_plus_1())(image, label)
    assert len(augmented.keys()) == 2
    assert augmented["image"] == [1, 0, 0]
    assert augmented["mask"] == [1, 1, 1]


def test_transform_image_label():
    image = [0, 0, 0]
    label = [1, 1, 1]
    augmented = TransformBase.ImageLabel(transform_plus_1())(image, label)
    assert len(augmented.keys()) == 2
    assert augmented["image"] == [1, 0, 0]
    assert augmented["mask"] == [2, 1, 1]


def test_transform_label():
    image = [0, 0, 0]
    label = [1, 1, 1]
    augmented = TransformBase.Label(transform_plus_1())(image, label)
    assert len(augmented.keys()) == 2
    assert augmented["image"] == [0, 0, 0]
    assert augmented["mask"] == [2, 1, 1]


def test_apply_transform():
    transform = TransformBase({})
    transform.transforms = [
        TransformBase.Image(transform_plus_1()),
        TransformBase.ImageLabel(transform_plus_1()),
        TransformBase.Label(transform_mul_2()),
        #Image should be applied after the above transforms.
        TransformBase.Image(transform_mul_2()),
    ]

    image = [10, 10, 10]
    label = [11, 11, 11]
    augmented = transform.apply_transforms(image, label)
    assert augmented[0] == [24, 10, 10]
    assert augmented[1] == [24, 11, 11]


def test_backward_compatibilty():
    transform = TransformBase({})
    transform.transform["IMAGE_ONLY"] = [transform_plus_1(), transform_mul_2()]
    transform.transform["IMAGE_LABEL"] = [transform_plus_1()]
    transform.transform["LABEL_ONLY"] = [transform_mul_2(), transform_plus_1()]

    image = [10, 10, 10]
    label = [11, 11, 11]
    augmented = transform.apply_transforms(image, label)
    assert augmented[0] == [23, 10, 10]
    assert augmented[1] == [25, 11, 11]

def test_albumentation_cv2_resize():
    resize = (256, 256)
    transform_A = TransformBase({})
    transform_cv2 = TransformBase({})
    transform_A.transforms = [TransformBase.Image(A.Resize(resize[0], resize[1]))]
    transform_cv2.transforms = [TransformBase.Image(lambda image: {'image' : cv2.resize(image, (resize[0], resize[1]))})]
    input = np.random.randint(256, size=(512, 512, 3)).astype('float32')
    augmented_A = transform_A.apply_transforms(input, input)
    augmented_cv2 = transform_cv2.apply_transforms(input, input)
    np.testing.assert_array_almost_equal(augmented_A[0], augmented_cv2[0], decimal=3)

def test_albumentation_cv2_normalize():
    input = np.random.randint(255, size=(512, 512, 3)).astype('float32')
    transform_A = TransformBase({})
    transform_cv2 = TransformBase({})
    transform_A.transforms = [TransformBase.Image(A.Normalize())]
    transform_cv2.transforms = [TransformBase.Image(lambda image: {'image': imagenet_preprocess(image).astype(np.float32)})]
    augmented_A = transform_A.apply_transforms(input, input)
    augmented_cv2 = transform_cv2.apply_transforms(input, input)
    np.testing.assert_array_almost_equal(augmented_A[0], augmented_cv2[0], decimal=3)

