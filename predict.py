# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 17:58:14 2018

@author: shirhe-lyh
"""

import cv2
import glob
import os
import tensorflow as tf

import predictor

flags = tf.app.flags

flags.DEFINE_string('frozen_inference_graph_path',
                    './training/frozen_inference_graph_pb/'+
                    'frozen_inference_graph.pb',
                    'Path to frozen inference graph.')
flags.DEFINE_string('images_dir', 
                    './test_images', 
                    'Path to images (directory).')

FLAGS = flags.FLAGS


if __name__ == '__main__':
    frozen_inference_graph_path = FLAGS.frozen_inference_graph_path
    images_dir = FLAGS.images_dir
    
    model = predictor.Predictor(frozen_inference_graph_path)
    
    image_files = glob.glob(os.path.join(images_dir, '*.*'))

    val_results = []
    predicted_count = 0
    num_samples = len(image_files)
    for image_path in image_files:
        predicted_count += 1
        if predicted_count % 100 == 0:
            print('Predict {}/{}.'.format(predicted_count, num_samples))
        
        image_name = image_path.split('/')[-1]
        image = cv2.imread(image_path)
        if image is None:
            print('image %s does not exist.' % image_name)
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        pred_label = int(model.predict([image])[0])

        print('Image Name: %s' % image_name)
        print('Pred Label: %d' %pred_label)