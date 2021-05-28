import matplotlib.pyplot as plt
from object_detection.utils import visualization_utils


def plot_detections(image_np,
                    boxes,
                    classes,
                    scores,
                    category_index,
                    figsize=(12, 16),
                    min_score_threshold=0.8,
                    image_name=None, ):
    """Wrapper function to visualize detections.

    Args:
      image_np: uint8 numpy array with shape (img_height, img_width, 3)
      boxes: a numpy array of shape [N, 4]
      classes: a numpy array of shape [N]. Note that class indices are 1-based,
        and match the keys in the label map.
      scores: a numpy array of shape [N] or None.  If scores=None, then
        this function assumes that the boxes to be plotted are groundtruth
        boxes and plot all boxes as black with no classes or scores.
      category_index: a dict containing category dictionaries (each holding
        category index `id` and category name `name`) keyed by category indices.
      figsize: size for the figure.
      image_name: a name for the image file.
    """
    image_np_with_annotations = image_np.copy()
    visualization_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_annotations,
        boxes,
        classes,
        scores,
        category_index,
        use_normalized_coordinates=True,
        min_score_thresh=min_score_threshold)
    if image_name:
        plt.imsave(image_name, image_np_with_annotations)
    else:
        plt.imshow(image_np_with_annotations)


if __name__ == '__main__':
    from PIL import Image
    import numpy as np
    from models.research.object_detection.builders import model_builder
    import model as model_lib

    model_dir = './pretrained-models/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
    model_config_manager = model_lib.ModelConfigManager(
        model_dir=model_dir,
        model_type='ssd',
        freeze_batchnorm=True)
    model = model_builder.build(model_config=model_config_manager.model_config, is_training=False)
    model_manager = model_lib.ModelManager(
        model=model,
        model_config_manager=model_config_manager,
        dimension=320)
    model_manager.restore('pretrain')
    image = np.asarray(Image.open('./samples/crop.png').convert('RGB'))
    detections = model_manager.predict(image)
    plot_detections(
        image,
        detections['detection_boxes'][0].numpy(),
        detections['detection_classes'][0].numpy().astype(np.int32) + 1,
        detections['detection_scores'][0].numpy(),
        {i: {'id': i, 'name': str(i)} for i in range(1, 1 + 100)},
        min_score_threshold=0.3,
        image_name='test.jpg')
