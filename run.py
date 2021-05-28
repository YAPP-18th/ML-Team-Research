import os
from os.path import join
from typing import List, Optional, Union
import wandb
import numpy as np
import tensorflow as tf
from models.research.object_detection.builders import model_builder
from object_detection.core.model import DetectionModel
import data as data_lib
import model as model_lib

from absl import app
from absl import flags
from absl import logging


FLAGS = flags.FLAGS

flags.DEFINE_float(
    'learning_rate', 0.01,
    'Initial learning rate.')

flags.DEFINE_integer(
    'train_batch_size', 32,
    'Batch size for training.')

flags.DEFINE_integer(
    'eval_batch_size', 32,
    'Batch size for eval.')

flags.DEFINE_integer(
    'train_epochs', 100,
    'Number of epochs to train for.')

flags.DEFINE_string(
    'pretrained_model_dir', None,
    'Pretrained model directory from tensorflow object api model zoo.')

flags.DEFINE_string(
    'model_dir', None,
    'Model directory for saving during training.')

flags.DEFINE_string(
    'data_dir', None,
    'Directory where dataset is stored.')

flags.DEFINE_enum(
    'optimizer', 'momentum', ['momentum', 'adam'],
    'Optimizer to use.')

flags.DEFINE_float(
    'momentum', 0.9,
    'Momentum parameter.')

flags.DEFINE_integer(
    'keep_checkpoint_max', 5,
    'Maximum number of checkpoints to keep.')

flags.DEFINE_string(
    'project_name', None,
    'Project name for Weights & Biases.')


def log_metrics(
        all_metrics: List[tf.keras.metrics.Metric],
        global_step: Union[int, np.integer],
        commit: Optional[bool] = None,
        **kwargs):
    metric_values = {}
    for metric in all_metrics:
        metric_value = metric.result().numpy().astype(float)
        logging.info(f'Step: [{global_step}] {metric.name} = {metric_value}')
        metric_values[metric.name] = metric_value
    metric_values.update(kwargs)
    wandb.log(metric_values, step=int(global_step), commit=commit)


def perform_evaluation(
        model: DetectionModel,
        files: Union[str, List[str]],
        prefix: Optional[str] = None) -> List[tf.keras.metrics.Metric]:
    ds = data_lib.build_dataset(
        files=files,
        batch_size=FLAGS.eval_batch_size,
        num_classes=1,
        input_processor=model.preprocess,
        cache=False,
        shuffle=False,
        repeat=False,
        drop_remainder=False)

    if prefix:
        prefix += '/'
    else:
        prefix = ''

    total_loss_metrics = tf.keras.metrics.Mean(f'val/{prefix}total_loss')
    localization_loss_metrics = tf.keras.metrics.Mean(f'val/{prefix}localization_loss')
    classification_loss_metrics = tf.keras.metrics.Mean(f'val/{prefix}classification_loss')
    all_metrics = [total_loss_metrics, localization_loss_metrics, classification_loss_metrics]

    @tf.function(experimental_relax_shapes=False)
    def single_step(images: tf.Tensor, ground_truth_boxes: List[tf.Tensor], groundtruth_classes: List[tf.Tensor]):
        shapes = tf.constant(FLAGS.eval_batch_size * [[320, 320, 3]], dtype=tf.int32)
        model.provide_groundtruth(
            groundtruth_boxes_list=ground_truth_boxes,
            groundtruth_classes_list=groundtruth_classes)
        prediction_dict = model.predict(images, shapes)
        losses_dict = model.loss(prediction_dict, shapes)
        total_loss = losses_dict['Loss/localization_loss'] + losses_dict['Loss/classification_loss']

        localization_loss_metrics.update_state(losses_dict['Loss/localization_loss'])
        classification_loss_metrics.update_state(losses_dict['Loss/classification_loss'])
        total_loss_metrics.update_state(total_loss)

    iterator = iter(ds)
    for _images, labels in iterator:
        single_step(_images, list(labels['groundtruth_boxes']), list(labels['groundtruth_classes']))
    return all_metrics


def build_checkpoint_manager(model: DetectionModel):
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        directory=FLAGS.model_dir,
        max_to_keep=FLAGS.keep_checkpoint_max)
    return checkpoint_manager


def main(argv):
    wandb.init(project=FLAGS.project_name, config=FLAGS)

    train_data_files = [
        join(FLAGS.data_dir, 'coco_train.tfrecords'),
        join(FLAGS.data_dir, 'openimages_train.tfrecords')
    ]
    val_data_files = {
        'COCO': join(FLAGS.data_dir, 'coco_validation.tfrecords'),
        'OpenImages': join(FLAGS.data_dir, 'openimages_validation.tfrecords')
    }
    train_steps = data_lib.get_train_steps(['COCO', 'OpenImages'], FLAGS.train_batch_size, drop_remainder=True)
    print(train_steps)
    train_steps = 1

    # Metrics
    total_loss_metrics = tf.keras.metrics.Mean('train/total_loss')
    localization_loss_metrics = tf.keras.metrics.Mean('train/localization_loss')
    classification_loss_metrics = tf.keras.metrics.Mean('train/classification_loss')
    all_metrics = [total_loss_metrics, localization_loss_metrics, classification_loss_metrics]

    model_config_manager = model_lib.ModelConfigManager(
        model_dir=FLAGS.pretrained_model_dir,
        model_type='ssd',
        num_classes=1,
        freeze_batchnorm=True)
    model = model_builder.build(model_config=model_config_manager.model_config, is_training=True)
    model_manager = model_lib.ModelManager(
        model=model,
        model_config_manager=model_config_manager,
        dimension=320)
    model_manager.restore('finetune')

    # Select variables in top layers to fine-tune.
    variables_to_finetune = model_manager.variables_to_finetune

    # Build optimizer
    optimizer = model_lib.build_optimizer(FLAGS.learning_rate)

    # Build dataloader
    ds = data_lib.build_dataset(
        files=train_data_files,
        batch_size=FLAGS.train_batch_size,
        num_classes=1,
        input_processor=model.preprocess,
        cache=True,
        shuffle=True,
        repeat=True,
        drop_remainder=True)

    @tf.function(experimental_relax_shapes=False)
    def single_step(images: tf.Tensor, ground_truth_boxes: List[tf.Tensor], groundtruth_classes: List[tf.Tensor]):
        shapes = tf.constant(FLAGS.train_batch_size * [[320, 320, 3]], dtype=tf.int32)
        model.provide_groundtruth(
            groundtruth_boxes_list=ground_truth_boxes,
            groundtruth_classes_list=groundtruth_classes)
        with tf.GradientTape() as tape:
            prediction_dict = model.predict(images, shapes)
            losses_dict = model.loss(prediction_dict, shapes)
            total_loss = losses_dict['Loss/localization_loss'] + losses_dict['Loss/classification_loss']
            gradients = tape.gradient(total_loss, variables_to_finetune)
            optimizer.apply_gradients(zip(gradients, variables_to_finetune))

            localization_loss_metrics.update_state(losses_dict['Loss/localization_loss'])
            classification_loss_metrics.update_state(losses_dict['Loss/classification_loss'])
            total_loss_metrics.update_state(total_loss)

    checkpoint_manager = build_checkpoint_manager(model)
    global_step = optimizer.iterations
    iterator = iter(ds)

    for epoch in range(FLAGS.train_epochs):
        logging.info(f'Epoch [{epoch}/{FLAGS.train_epochs}]')
        for step in range(train_steps):
            _images, labels = next(iterator)
            single_step(_images, list(labels['groundtruth_boxes']), list(labels['groundtruth_classes']))
            log_metrics(all_metrics, global_step.numpy(), epoch=epoch)

        checkpoint_manager.save(epoch)
        for metric in all_metrics:
            metric.reset_states()

        eval_metrics = []
        for prefix, val_data_file in val_data_files.items():
            eval_metrics.extend(perform_evaluation(model, val_data_file, prefix))
        log_metrics(eval_metrics, global_step.numpy(), epoch=epoch)

    logging.info('Training complete...')


def test():
    pretrained_model_dir = './pretrained-models/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
    model_config_manager = model_lib.ModelConfigManager(
        model_dir=pretrained_model_dir,
        model_type='ssd',
        num_classes=1,
        freeze_batchnorm=True)
    model = model_builder.build(model_config=model_config_manager.model_config, is_training=False)
    model_manager = model_lib.ModelManager(
        model=model,
        model_config_manager=model_config_manager.model_config,
        dimension=320)
    ds = data_lib.build_dataset(
        files='openimages_validation.tfrecords',
        batch_size=32,
        num_classes=1,
        input_processor=model_manager.input_processor,
        drop_remainder=True)
    images, labels = next(iter(ds))


if __name__ == '__main__':
    app.run(main)

