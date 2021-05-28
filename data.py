from typing import Callable, Union, Tuple, Dict, Collection, List
from numbers import Number
import tensorflow as tf
import math

NUM_TRAIN_DATA = {
    'COCO': 4803,
    'OpenImages': 4312
}


def get_train_steps(dataset: Collection[str], batch_size: int, drop_remainder: bool = True):
    num = 0
    for _dataset in dataset:
        num += NUM_TRAIN_DATA[_dataset]

    steps = num / batch_size
    if drop_remainder:
        return math.ceil(steps)
    return math.floor(steps)


def pad_to_fixed_size(
        data: tf.Tensor,
        pad_value: Number,
        output_shape: Union[list, tuple]) -> tf.Tensor:
    max_instances_per_image = output_shape[0]
    dimension = output_shape[1]
    data = tf.reshape(data, [-1, dimension])
    num_instances = tf.shape(data)[0]
    pad_length = max_instances_per_image - num_instances
    paddings = pad_value * tf.ones(tf.convert_to_tensor([pad_length, dimension]))
    padded_data = tf.concat([data, paddings], axis=0)
    padded_data = tf.reshape(padded_data, output_shape)
    return padded_data


def dataset_parser(
        input_processor: Callable,
        example_proto: bytes) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    image_feature_description = {
        'image/source_id': tf.io.FixedLenFeature([], tf.string, ''),
        'image/height': tf.io.FixedLenFeature([], tf.int64, -1),
        'image/width': tf.io.FixedLenFeature([], tf.int64, -1),
        'image/encoded': tf.io.FixedLenFeature([], tf.string, ''),
        'image/object/bbox/y_min': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/x_min': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/y_max': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/x_max': tf.io.VarLenFeature(tf.float32),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    }
    parsed_tensors = tf.io.parse_single_example(example_proto, image_feature_description)
    for k in parsed_tensors:
        if isinstance(parsed_tensors[k], tf.SparseTensor):
            if parsed_tensors[k].dtype == tf.string:
                parsed_tensors[k] = tf.sparse.to_dense(
                    parsed_tensors[k], default_value='')
            else:
                parsed_tensors[k] = tf.sparse.to_dense(
                    parsed_tensors[k], default_value=0)

    image = tf.io.decode_jpeg(parsed_tensors['image/encoded'], channels=3)
    boxes = tf.stack([
        parsed_tensors['image/object/bbox/y_min'],
        parsed_tensors['image/object/bbox/x_min'],
        parsed_tensors['image/object/bbox/y_max'],
        parsed_tensors['image/object/bbox/x_max']], axis=1)

    decoded_tensors = {
        'source_id': parsed_tensors['image/source_id'],
        'image': image,
        'height': parsed_tensors['image/height'],
        'width': parsed_tensors['image/width'],
        'groundtruth_classes': parsed_tensors['image/object/class/label'],
        'groundtruth_boxes': boxes
    }

    max_instances_per_image = None
    _max_instances_per_image = max_instances_per_image or 100

    source_id = decoded_tensors['source_id']
    # TODO: Data augmentations must be here!
    image = input_processor(tf.cast(decoded_tensors['image'][None, ...], dtype=tf.float32))[0][0]
    boxes = decoded_tensors['groundtruth_boxes']
    classes = decoded_tensors['groundtruth_classes']

    boxes = pad_to_fixed_size(boxes, -1, [_max_instances_per_image, 4])
    classes = pad_to_fixed_size(tf.cast(classes, dtype=tf.float32)[:, None], -1, [_max_instances_per_image, 1])[:, 0]
    return source_id, image, boxes, classes


def process_example(
        num_classes: int,
        source_ids: tf.Tensor,
        images: tf.Tensor,
        boxes: tf.Tensor,
        classes: tf.Tensor) -> Tuple[tf.Tensor, Dict[str, Union[tf.Tensor, tf.RaggedTensor]]]:
    mask = tf.not_equal(classes, -1)

    boxes: tf.RaggedTensor = tf.ragged.boolean_mask(boxes, mask)
    classes: tf.RaggedTensor = tf.ragged.boolean_mask(classes, mask)
    classes = tf.one_hot(tf.cast(classes - 1, dtype=tf.int64), num_classes)

    labels = {'source_ids': source_ids, 'groundtruth_boxes': boxes, 'groundtruth_classes': classes}
    return images, labels


def build_dataset(
        files: Union[str, List[str]],
        batch_size: int,
        num_classes: int,
        input_processor: Callable,
        cache: bool = False,
        shuffle: bool = False,
        repeat: bool = False,
        drop_remainder: bool = False) -> tf.data.Dataset:
    dataset = tf.data.TFRecordDataset(files)
    if cache:
        dataset = dataset.cache()
    if shuffle:
        dataset = dataset.shuffle(batch_size * 10)
    if repeat:
        dataset = dataset.repeat(-1)
    dataset = dataset.map(
        lambda *args: dataset_parser(input_processor, *args), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    dataset = dataset.map(
        lambda *args: process_example(num_classes, *args), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(2)
    return dataset


if __name__ == '__main__':
    ds = build_dataset(
        files=['openimages_train.tfrecords', 'coco_validation.tfrecords'],
        batch_size=32,
        num_classes=1,
        input_processor=lambda image: tf.image.resize(image, [320, 320]),
        cache=False,
        shuffle=False,
        repeat=False,
        drop_remainder=False)
