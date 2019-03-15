# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 21:43:47 2018

@author: shirhe-lyh


Read a .txt file to provide annotation class labels.
"""

import glob
import json
import os


def split_train_val_sets(images_dir, val_ratio=0.02):
    """Split image files to training and validation."""
    if not os.path.exists(images_dir):
        raise ValueError('`images_dir` does not exist.')
        
    image_files = glob.glob(os.path.join(images_dir, '*.jpg'))
    num_val_samples = int(len(image_files) * val_ratio)
    val_files = image_files[:num_val_samples]
    train_files = image_files[num_val_samples:]
    
    train_dict = _get_labling_dict(train_files)
    val_dict = _get_labling_dict(val_files)
    return train_dict, val_dict


def _get_labling_dict(image_files=None):
    if image_files is None:
        return None
    
    labling_dict = {}
    for image_file in image_files:
        image_name = image_file.split('/')[-1]
        if image_name.startswith('cat'):
            labling_dict[image_file] = 0
        elif image_name.startswith('dog'):
            labling_dict[image_file] = 1
    return labling_dict


def write_annotation_json(images_dir, train_json_output_path, 
                          val_json_output_path):
    """Save training and validation annotations."""
    train_files_dict, val_files_dict = split_train_val_sets(images_dir)
    train_json = json.dumps(train_files_dict)
    
    if train_json_output_path.startswith('./datasets'):
        if not os.path.exists('./datasets'):
            os.mkdir('./datasets')
    
    with open(train_json_output_path, 'w') as writer:
        json.dump(train_json, writer)
    val_json = json.dumps(val_files_dict)
    with open(val_json_output_path, 'w') as writer:
        json.dump(val_json, writer)


def provide(annotation_path=None, images_dir=None):
    """Return image_paths and class labels.
    
    Args:
        annotation_path: Path to an anotation's .json file.
        images_dir: Path to images directory.
            
    Returns:
        image_files: A list containing the paths of images.
        annotation_dict: A dictionary containing the class labels of each 
            image.
            
    Raises:
        ValueError: If annotation_path does not exist.
    """
    if not os.path.exists(annotation_path):
        raise ValueError('`annotation_path` does not exist.')
        
    with open(annotation_path, 'r') as reader:
        annotation_str = json.load(reader)
        annotation_d = json.loads(annotation_str)
    image_files = []
    annotation_dict = {}
    for image_name, labels in annotation_d.items():
        if images_dir is not None:
            image_name = os.path.join(images_dir, image_name)
        image_files.append(image_name)
        annotation_dict[image_name] = labels
    return image_files, annotation_dict

