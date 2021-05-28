import os
from os.path import join
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'pipeline_config_path', None,
    'Path of pipeline config for pretrained model from tensorflow object detection api model zoo.')

flags.DEFINE_string(
    'trained_checkpoint_dir', None,
    'Path to trained checkpoint directory')

flags.DEFINE_string(
    'output_directory', None,
    'Output directory for the models to be saved: saved model, frozen model, tfjs graph model')

flags.mark_flag_as_required('pipeline_config_path')
flags.mark_flag_as_required('trained_checkpoint_dir')
flags.mark_flag_as_required('output_directory')


def _read_pb(file):
    gf = tf.compat.v1.GraphDef()
    gf.ParseFromString(open(file, 'rb').read())

    print([f'{n.name} => {n.op}' for n in gf.node if n.op])


def export_tf_saved_model(pipeline_config_path: str, trained_checkpoint_dir: str, output_dir: str):
    if not os.path.exists('./models/research/object_detection/exporter_main_v2.py'):
        raise FileNotFoundError

    os.system(f"python ./models/research/object_detection/exporter_main_v2.py "
              f"--input_type image_tensor "
              f"--pipeline_config_path {pipeline_config_path} "
              f"--trained_checkpoint_dir {trained_checkpoint_dir} "
              f"--output_directory {output_dir}")


def tf_saved_model2tf_frozen_model(saved_model_dir: str, output_dir: str):
    model = tf.saved_model.load(saved_model_dir)
    graph_func = model.signatures['serving_default']
    frozen_func = convert_variables_to_constants_v2(graph_func)
    frozen_func.graph.as_graph_def()

    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir=output_dir,
                      name='frozen_model.pb',
                      as_text=False)


def tf_frozen_model2tfjs_graph_model(frozen_model_dir: str, output_dir: str):
    os.system(f"tensorflowjs_converter --input_format=tf_frozen_model "
              f"--output_format=tfjs_graph_model "
              f"--output_node_names='StatefulPartitionedCall/Postprocessor/ExpandDims_1,StatefulPartitionedCall/Postprocessor/Slice' "
              f"{frozen_model_dir}/frozen_model.pb "
              f"{output_dir}")


def main(argv):
    star_format = '\U0001F31F' * 3 + '{}' + '\U0001F31F' * 3
    saved_model_dir = join(FLAGS.output_directory, 'saved_model')
    frozen_model_dir = join(FLAGS.output_directory, 'frozen_model')
    tfjs_graph_model_dir = join(FLAGS.output_directory, 'tfjs_graph_model')

    export_tf_saved_model(
        pipeline_config_path=FLAGS.pipeline_config_path,
        trained_checkpoint_dir=FLAGS.trained_checkpoint_dir,
        output_dir=FLAGS.output_directory)
    logging.info(star_format.format('Checkpoint ==> SavedModel complete...'))
    tf_saved_model2tf_frozen_model(saved_model_dir=saved_model_dir, output_dir=frozen_model_dir)
    logging.info(star_format.format('SavedModel ==> FrozenModel complete...'))
    tf_frozen_model2tfjs_graph_model(frozen_model_dir=frozen_model_dir, output_dir=tfjs_graph_model_dir)
    logging.info(star_format.format('FrozenModel ==> TFJSGraphModel complete...'))


if __name__ == '__main__':
    app.run(main)
