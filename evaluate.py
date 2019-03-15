# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 17:58:14 2018

@author: shirhe-lyh
"""

import cv2
import json
import os
import tensorflow as tf

import data_provider
import predictor

flags = tf.app.flags

flags.DEFINE_string('frozen_inference_graph_path',
                    './training/frozen_inference_graph_pb/'+
                    'frozen_inference_graph.pb',
                    'Path to frozen inference graph.')
flags.DEFINE_string('images_dir', 
                    '/data2/raycloud/jingxiong_datasets/six_classes/images', 
                    'Path to images (directory).')
flags.DEFINE_string('annotation_path', 
                    '/data2/raycloud/jingxiong_datasets/six_classes/' +
                    'val_joint.json',
                    'Path to annotation`s .json file.')
flags.DEFINE_string('output_path', './val_result.json', 'Path to output file.')
flags.DEFINE_integer('resize_size', 368, 'The smallest side size to resized.')

FLAGS = flags.FLAGS


if __name__ == '__main__':
    # Specify which gpu to be used
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    
    frozen_inference_graph_path = FLAGS.frozen_inference_graph_path
    images_dir = FLAGS.images_dir
    annotation_path = FLAGS.annotation_path
    output_path = FLAGS.output_path
    
    model = predictor.Predictor(frozen_inference_graph_path)
    
    _, annotation_dict = data_provider.provide(annotation_path, images_dir)

    val_results_dict = {}
    predicted_count = 0
    correct_count_sc = 0
    correct_count_bt = 0
    correct_count_ot = 0
    total_count_bt = 0
    total_count_ot = 0
    num_samples = len(annotation_dict)
    for image_path, labels in annotation_dict.items():
        predicted_count += 1
        if predicted_count % 100 == 0:
            print('Predict {}/{}.'.format(predicted_count, num_samples))
        
        image_name = image_path.split('/')[-1]
        image = cv2.imread(image_path)
        if image is None:
            print('image %s does not exist.' % image_name)
            continue
        height, width, _ = image.shape
        if height > width:
            width = FLAGS.resize_size
            height = int(height * 368 / width)
        else:
            width = int(width * 368 / height)
            height = FLAGS.resize_size
        image = cv2.resize(image, (width, height))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        pred_labels = model.predict([image])
        label_sc, label_bt, label_ot = labels
        pred_label_sc, pred_label_bt, pred_label_ot = pred_labels
        if pred_label_sc[0] == label_sc:
            correct_count_sc += 1
        if label_sc in [4, 5, 6]:
            total_count_bt += 1
            if pred_label_bt[0] == label_bt:
                correct_count_bt += 1
        if label_sc in [5, 6]:
            total_count_ot += 1
            if pred_label_ot[0] == label_ot:
                correct_count_ot += 1
        val_results_dict[image_name] = [int(pred_label_sc),
                                        int(pred_label_bt),
                                        int(pred_label_ot)]
        
    print('Accuracy of Seven Classes: ', correct_count_sc*1.0/num_samples)
    print('Accuracy of Body Types: ', correct_count_bt*1.0/total_count_bt)
    print('Accuracy of Orientations: ', correct_count_ot*1.0/total_count_ot)
        
    val_results_json = json.dumps(val_results_dict)
    file = open('./val_result_1030.json', 'w')
    json.dump(val_results_json, file)
    file.close()
