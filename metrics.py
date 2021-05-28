from typing import Collection, Dict, Any
import numpy as np


def box_area(box: Collection) -> float:
    y1, x1, y2, x2 = box
    return (y2 - y1 + 1) * (x2 - x1 + 1)


def intersection_area(box1: Collection, box2: Collection) -> float:
    y1 = max(box1[0], box2[0])
    x1 = max(box1[1], box2[1])
    y2 = min(box1[2], box2[2])
    x2 = min(box1[3], box2[2])
    return max(0., box_area([y1, x1, y2, x2]))


def union_area(box1: Collection, box2: Collection) -> float:
    area1, area2 = box_area(box1), box_area(box2)
    if area1 < 0 or area2 < 0:
        raise ValueError
    inter_area = intersection_area(box1, box2)
    return area1 + area2 - inter_area


def box_iou(box1: Collection, box2: Collection) -> float:
    return intersection_area(box1, box2) / union_area(box1, box2)


def predict_detection_boxes_per_image(
        groundtruth_boxes: np.ndarray,
        detection_boxes: np.ndarray,
        iou_threshold: float) -> np.ndarray:
    if len(detection_boxes) == 0:
        return np.asarray([])

    if len(groundtruth_boxes) == 0:
        return np.full(len(detection_boxes), False)

    predictions = np.full(len(detection_boxes), False)
    matched = np.full(len(groundtruth_boxes), False)

    for i, detection_box in enumerate(detection_boxes):
        iou_list = [box_iou(groundtruth_box, detection_box) for groundtruth_box in groundtruth_boxes]
        max_iou_value, max_iou_index = np.max(iou_list), np.argmax(iou_list)

        if max_iou_value < iou_threshold:
            predictions[i] = False
        else:
            if matched[max_iou_index]:
                predictions[i] = False
            else:
                predictions[i] = True
                matched[max_iou_index] = True
    return predictions


def average_precision(
        groundtruth_boxes: np.ndarray,
        detection_boxes: np.ndarray,
        detection_scores: np.ndarray,
        iou_threshold: float,
        interpolation: str = '101') -> Dict[str, Any]:
    num_positive = sum([len(groundtruth_box) for groundtruth_box in groundtruth_boxes])

    predictions = np.concatenate([predict_detection_boxes_per_image(groundtruth_box, detection_box, iou_threshold)
                                  for groundtruth_box, detection_box in zip(groundtruth_boxes, detection_boxes)])
    detection_scores = np.concatenate([detection_score for detection_score in detection_scores])

    idx = np.argsort(detection_scores)[::-1]
    predictions = predictions[idx]

    # average precision top k
    true_positive = predictions.astype(np.int64)
    false_positive = 1 - true_positive

    acc_true_positive = np.cumsum(true_positive)
    acc_false_positive = np.cumsum(false_positive)

    precision = acc_true_positive / (acc_true_positive + acc_false_positive)
    recall = acc_true_positive / num_positive

    if interpolation == '101':
        xs = np.linspace(0., 1., int(np.round((1.00 - .0) / .01) + 1))
        max_precisions = [np.max(precision[x <= recall], initial=0.) for x in xs]
        ap = np.mean(max_precisions)
    else:
        raise NotImplementedError

    result = {
        'precision': precision,
        'recall': recall,
        'AP': ap,
        'interpolated_precision': max_precisions,
        'interpolated_recall': xs,
        'total_TP': acc_true_positive[-1],
        'total_FP': acc_false_positive[-1],
    }
    return result


def mean_average_precision(
        groundtruth_boxes: np.ndarray,
        groundtruth_classes: np.ndarray,
        detection_boxes: np.ndarray,
        detection_scores: np.ndarray,
        detection_classes: np.ndarray,
        iou_threshold: float = 0.5) -> Dict[str, Any]:
    result = {}
    for detection_class in np.unique([76]).astype(int):
        detection_idx = detection_classes == detection_class
        groundtruth_idx = [groundtruth_class == detection_class for groundtruth_class in groundtruth_classes]

        _groundtruth_boxes = np.asarray(
            [groundtruth_box[idx] for idx, groundtruth_box in zip(groundtruth_idx, groundtruth_boxes)])
        _detection_boxes = np.asarray(
            [detection_box[idx] for idx, detection_box in zip(detection_idx, detection_boxes)])
        _detection_scores = np.asarray(
            [detection_score[idx] for idx, detection_score in zip(detection_idx, detection_scores)])

        result[detection_class] = average_precision(
            _groundtruth_boxes,
            _detection_boxes,
            _detection_scores,
            iou_threshold)
    result['mAP'] = np.mean([_result['AP'] for _result in result.values()])
    return result


def main():
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from models.research.object_detection.builders import model_builder
    import data as data_lib
    import model as model_lib

    model_dir = './pretrained-models/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
    batch_size = 32

    model_config_manager = model_lib.ModelConfigManager(
        model_dir=model_dir,
        model_type='ssd',
        num_classes=1,
        freeze_batchnorm=True)
    model = model_builder.build(model_config=model_config_manager.model_config, is_training=False)
    model_manager = model_lib.ModelManager(
        model=model,
        model_config_manager=model_config_manager.model_config,
        dimension=320)
    model_manager.restore('finetune')

    for data in data_lib.build_dataset(
            files='coco_validation.tfrecords',
            batch_size=batch_size,
            num_classes=1,
            input_processor=model.preprocess,
            drop_remainder=False):
        images = data[0]
        groundtruth_boxes = data[1]['groundtruth_boxes']
        groundtruth_classes = np.asarray([len(groundtruth_box) * [77 - 1] for groundtruth_box in groundtruth_boxes])

        shapes = tf.constant(batch_size * [[320, 320, 3]], dtype=tf.int32)
        prediction_dict = model.predict(images, shapes)
        detection = model.postprocess(prediction_dict, shapes)

        detection_boxes = detection['detection_boxes']
        detection_scores = detection['detection_scores']
        detection_classes = detection['detection_classes']
        result = mean_average_precision(
            groundtruth_boxes.numpy(),
            groundtruth_classes,
            detection_boxes.numpy(),
            detection_scores.numpy(),
            detection_classes.numpy())
        break

    result = result[77 - 1]
    plt.plot(result['interpolated_recall'], result['interpolated_precision'])
    plt.plot(result['recall'], result['precision'])
    plt.savefig('precision_recall_graph.png')


if __name__ == '__main__':
    main()
