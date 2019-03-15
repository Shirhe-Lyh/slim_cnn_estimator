#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 09:02:10 2018

@author: shirhe-lyh
"""

"""Generate tfrecord file from images.

Example Usage:
---------------
python3 train.py \
    --images_dir: Path to images (directory).
    --annotation_path: Path to annotatio's .txt file.
    --output_path: Path to .record.
    --resize_side_size: Resize images to fixed size.
"""

import io
import tensorflow as tf
import time

from PIL import Image

import data_provider

flags = tf.app.flags

flags.DEFINE_string('images_dir', 
                    '/data1/jingxiong_datasets/cat_dog_kaggle/train',
                    'Path to images (directory).')
flags.DEFINE_string('train_annotation_path', 
                    './datasets/train.json',
                    'Path to annotation`s .json file.')
flags.DEFINE_string('train_output_path', 
                    './datasets/train.record',
                    'Path to output tfrecord file.')
flags.DEFINE_string('val_annotation_path', 
                    './datasets/val.json',
                    'Path to annotation`s .json file.')
flags.DEFINE_string('val_output_path', 
                    './datasets/val.record',
                    'Path to output tfrecord file.')
flags.DEFINE_integer('resize_side_size', 256, 'Resize images to fixed size.')

FLAGS = flags.FLAGS


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_tf_example(image_path, label, resize_size=None):
    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    
    width, height = image.size
    
    # Resize
    if resize_size is not None:
        if width > height:
            width = int(width * resize_size / height)
            height = resize_size
        else:
            width = resize_size
            height = int(height * resize_size / width)
        image = image.resize((width, height), Image.ANTIALIAS)
        bytes_io = io.BytesIO()
        image.save(bytes_io, format='JPEG')
        encoded_jpg = bytes_io.getvalue()
    
    tf_example = tf.train.Example(
        features=tf.train.Features(feature={
            'image/encoded': bytes_feature(encoded_jpg),
            'image/format': bytes_feature('jpg'.encode()),
            'image/class/label': int64_feature(label),
            'image/height': int64_feature(height),
            'image/width': int64_feature(width)}))
    return tf_example


def generate_tfrecord(annotation_dict, output_path, resize_size=None):
    num_valid_tf_example = 0
    writer = tf.python_io.TFRecordWriter(output_path)
    for image_path, label in annotation_dict.items():
        if not tf.gfile.GFile(image_path):
            print('%s does not exist.' % image_path)
            continue
        tf_example = create_tf_example(image_path, label, resize_size)
        writer.write(tf_example.SerializeToString())
        num_valid_tf_example += 1
        
        if num_valid_tf_example % 100 == 0:
            print('Create %d TF_Example.' % num_valid_tf_example)
    writer.close()
    print('Total create TF_Example: %d' % num_valid_tf_example)
    
    
def main(_):
    images_dir = FLAGS.images_dir
    train_annotation_path = FLAGS.train_annotation_path
    train_record_path = FLAGS.train_output_path
    val_annotation_path = FLAGS.val_annotation_path
    val_record_path = FLAGS.val_output_path
    resize_size = FLAGS.resize_side_size
    
    # Write json
    data_provider.write_annotation_json(images_dir, train_annotation_path,
                                        val_annotation_path)
    
    time.sleep(5)
    
    _, train_annotation_dict = data_provider.provide(train_annotation_path, 
                                                     None)
    _, val_annotation_dict = data_provider.provide(val_annotation_path, 
                                                   None)
    
    generate_tfrecord(train_annotation_dict, train_record_path, resize_size)
    generate_tfrecord(val_annotation_dict, val_record_path, resize_size)
    
    
if __name__ == '__main__':
    tf.app.run()