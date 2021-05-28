import os
from typing import Optional, Union, Dict
import tensorflow as tf
import numpy as np
from models.research.object_detection.utils import config_util
from object_detection.core.model import DetectionModel
from absl import flags
from absl import logging

FLAGS = flags.FLAGS


def build_optimizer(learning_rate: float) -> tf.keras.optimizers.Optimizer:
    if FLAGS.optimizer == 'momentum':
        return tf.keras.optimizers.SGD(learning_rate, momentum=FLAGS.momentum, nesterov=True)
    elif FLAGS.optimizer == 'adam':
        return tf.keras.optimizers.Adam(learning_rate)
    else:
        ValueError(FLAGS.optimizer)


class ModelConfigManager:
    def __init__(self,
                 model_dir: str,
                 model_type: str,
                 num_classes: Optional[int] = None,
                 freeze_batchnorm: Optional[bool] = None):
        self.model_dir = model_dir

        self.pipeline_config_path = os.path.join(model_dir, 'pipeline.config')
        self.checkpoint_path = os.path.join(model_dir, 'checkpoint', 'ckpt-0')
        self.model_type = model_type
        self.num_classes = num_classes
        self.freeze_batchnorm = freeze_batchnorm

        self.configs = config_util.get_configs_from_pipeline_file(self.pipeline_config_path)
        self.model_config = self.configs['model']

        self._model = getattr(self.model_config, model_type)

        if num_classes:
            setattr(self._model, 'num_classes', num_classes)

        if freeze_batchnorm:
            setattr(self._model, 'freeze_batchnorm', freeze_batchnorm)


class ModelManager:
    def __init__(self,
                 model: DetectionModel,
                 model_config_manager: ModelConfigManager,
                 dimension: int):
        self.model = model
        self.model_config_manager = model_config_manager
        self.dimension = dimension

    @property
    def variables_to_finetune(self):
        trainable_variables = self.model.trainable_variables
        variables = []
        prefixes_to_train = [
            'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalBoxHead',
            'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalClassHead'
        ]
        for var in trainable_variables:
            if any([var.name.startswith(prefix) for prefix in prefixes_to_train]):
                variables.append(var)
        return variables

    def restore(self, mode: str):
        checkpoint_path = self.model_config_manager.checkpoint_path
        if mode == 'pretrain':
            ckpt = tf.train.Checkpoint(model=self.model)
            ckpt.restore(checkpoint_path).expect_partial()
        elif mode == 'finetune':
            fake_box_predictor = tf.train.Checkpoint(
                _base_tower_layers_for_heads=self.model._box_predictor._base_tower_layers_for_heads,
                # _prediction_heads=detection_model._box_predictor._prediction_heads,
                #    (i.e., the classification head that we *will not* restore)
                _box_prediction_head=self.model._box_predictor._box_prediction_head
            )
            fake_model = tf.train.Checkpoint(
                _feature_extractor=self.model._feature_extractor,
                _box_predictor=fake_box_predictor
            )
            ckpt = tf.train.Checkpoint(model=fake_model)
            ckpt.restore(checkpoint_path).expect_partial()
        else:
            raise NotImplementedError

        self.predict(tf.zeros([1, self.dimension, self.dimension, 3]))
        logging.info('Weights restored!')

    def predict(self, images: Union[np.ndarray, tf.Tensor]) -> Dict[str, tf.Tensor]:
        if isinstance(images, np.ndarray):
            images = tf.convert_to_tensor(images, dtype=tf.float32)

        assert tf.is_tensor(images)

        if images.ndim != 4:
            images = tf.expand_dims(images, 0)

        assert images.ndim == 4

        if images.dtype != tf.float32:
            images = tf.cast(images, tf.float32)

        preprocessed_images, shapes = self.model.preprocess(images)
        prediction_dict = self.model.predict(preprocessed_images, shapes)
        return self.model.postprocess(prediction_dict, shapes)






