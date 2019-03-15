#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 19:27:44 2018

@author: shirhe-lyh


Train a CNN model to classifying 10 digits.

Example Usage:
---------------
python3 train.py \
    --train_record_path: Path to training tfrecord file.
    --val_record_path: Path to validation tfrecord file.
    --model_dir: Path to log directory.
"""

import functools
import logging
import os
import tensorflow as tf

import exporter
import model

slim = tf.contrib.slim
flags = tf.app.flags

flags.DEFINE_string('gpu_indices', '0', 'The index of gpus to used.')
flags.DEFINE_string('train_record_path', 
                    './datasets/train.record', 
                    'Path to training tfrecord file.')
flags.DEFINE_string('val_record_path', 
                    './datasets/val.record', 
                    'Path to validation tfrecord file.')
flags.DEFINE_string('checkpoint_path',
                    '/data1/model_zoo/resnet_v1_50.ckpt',
                    'Path to a pretrained model.')
flags.DEFINE_string('model_dir', './training', 'Path to log directory.')
flags.DEFINE_float('keep_checkpoint_every_n_hours', 
                   0.2,
                   'Save model checkpoint every n hours.')
flags.DEFINE_string('learning_rate_decay_type',
                    'exponential',
                    'Specifies how the learning rate is decayed. One of '
                    '"fixed", "exponential", or "polynomial"')
flags.DEFINE_float('learning_rate', 
                   0.0001, 
                   'Initial learning rate.')
flags.DEFINE_float('end_learning_rate', 
                   0.000001,
                   'The minimal end learning rate used by a polynomial decay '
                   'learning rate.')
flags.DEFINE_float('decay_steps',
                   1000,
                   'Number of epochs after which learning rate decays. '
                   'Note: this flag counts epochs per clone but aggregates '
                   'per sync replicas. So 1.0 means that each clone will go '
                   'over full epoch individually, but replicas will go once '
                   'across all replicas.')
flags.DEFINE_float('learning_rate_decay_factor',
                   0.5,
                   'Learning rate decay factor.')
flags.DEFINE_integer('num_classes', 2, 'Number of classes.')
flags.DEFINE_integer('batch_size', 64, 'Batch size.')
flags.DEFINE_integer('num_steps', 5000, 'Number of steps.')
flags.DEFINE_integer('input_size', 224, 'Number of steps.')

FLAGS = flags.FLAGS


def get_decoder():
    """Returns a TFExampleDecoder."""
    keys_to_features = {
        'image/encoded': 
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': 
            tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/class/label': 
            tf.FixedLenFeature([1], tf.int64, default_value=tf.zeros([1], 
                               dtype=tf.int64))}
        
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(image_key='image/encoded',
                                              format_key='image/format',
                                              channels=3),
        'label': slim.tfexample_decoder.Tensor('image/class/label', shape=[])}
    
    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)
    return decoder
    
    
def transform_data(image):
    size = FLAGS.input_size + 32
    image = tf.squeeze(tf.image.resize_bilinear([image], size=[size, size]))
    image = tf.to_float(image)
    return image


def read_dataset(file_read_fun, input_files, num_readers=1, shuffle=False,
                 num_epochs=0, read_block_length=32, shuffle_buffer_size=2048):
    """Reads a dataset, and handles repeatition and shuffling.
    
    This function and the following are modified from:
        https://github.com/tensorflow/models/blob/master/research/
            object_detection/builders/dataset_builder.py
    
    Args:
        file_read_fun: Function to use in tf.contrib.data.parallel_iterleave,
            to read every individual file into a tf.data.Dataset.
        input_files: A list of file paths to read.
        
    Returns:
        A tf.data.Dataset of (undecoded) tf-records.
    """
    # Shard, shuffle, and read files
    filenames = tf.gfile.Glob(input_files)
    if num_readers > len(filenames):
        num_readers = len(filenames)
        tf.logging.warning('num_readers has been reduced to %d to match input '
                           'file shards.' % num_readers)
    filename_dataset = tf.data.Dataset.from_tensor_slices(filenames)
    if shuffle:
        filename_dataset = filename_dataset.shuffle(100)
    elif num_readers > 1:
        tf.logging.warning('`shuffle` is false, but the input data stream is '
                           'still slightly shuffled since `num_readers` > 1.')
    filename_dataset = filename_dataset.repeat(num_epochs or None)
    records_dataset = filename_dataset.apply(
        tf.contrib.data.parallel_interleave(
            file_read_fun,
            cycle_length=num_readers,
            block_length=read_block_length,
            sloppy=shuffle))
    if shuffle:
        records_dataset = records_dataset.shuffle(shuffle_buffer_size)
    return records_dataset  


def create_input_fn(record_paths, batch_size=64,
                    num_epochs=0, num_parallel_batches=8, 
                    num_prefetch_batches=2):
    """Create a train or eval `input` function for `Estimator`.
    
    Args:
        record_paths: A list contains the paths of tfrecords.
    
    Returns:
        `input_fn` for `Estimator` in TRAIN/EVAL mode.
    """
    def _input_fn():
        decoder = get_decoder()
        
        def decode(value):
            keys = decoder.list_items()
            tensors = decoder.decode(value)
            tensor_dict = dict(zip(keys, tensors))
            image = tensor_dict.get('image')
            image = transform_data(image)
            features_dict = {'image': image}
            return features_dict, tensor_dict.get('label')
        
        dataset = read_dataset(
            functools.partial(tf.data.TFRecordDataset, 
                              buffer_size=8 * 1000 * 1000),
            input_files=record_paths,
            num_epochs=num_epochs)
        if batch_size:
            num_parallel_calles = batch_size * num_parallel_batches
        else:
            num_parallel_calles = num_parallel_batches
        dataset = dataset.map(decode, num_parallel_calls=num_parallel_calles)
        if batch_size:
            dataset = dataset.apply(
                tf.contrib.data.batch_and_drop_remainder(batch_size))
        dataset = dataset.prefetch(num_prefetch_batches)
        return dataset
    
    return _input_fn


def create_predict_input_fn():
    """Creates a predict `input` function for `Estimator`.
    
    Modified from:
        https://github.com/tensorflow/models/blob/master/research/
            object_detection/inputs.py
    
    Returns:
        `input_fn` for `Estimator` in PREDICT mode.
    """
    def _predict_input_fn():
        """Decodes serialized tf.Examples and returns `ServingInputReceiver`.
        
        Returns:
            `ServingInputReceiver`.
        """
        example = tf.placeholder(dtype=tf.string, shape=[], name='tf_example')
        
        decoder = get_decoder()
        keys = decoder.list_items()
        tensors = decoder.decode(example, items=keys)
        tensor_dict = dict(zip(keys, tensors))
        image = tensor_dict.get('image')
        image = transform_data(image)
        images = tf.expand_dims(image, axis=0)
        return tf.estimator.export.ServingInputReceiver(
            features={'image': images},
            receiver_tensors={'serialized_example': example})
        
    return _predict_input_fn


def create_model_fn(features, labels, mode, params=None):
    """Constructs the classification model.
    
    Modifed from:
        https://github.com/tensorflow/models/blob/master/research/
            object_detection/model_lib.py.
    
    Args:
        features: A 4-D float32 tensor with shape [batch_size, height,
            width, channels] representing a batch of images. (Support dict)
        labels: A 1-D int32 tensor with shape [batch_size] representing
             the labels of each image. (Support dict)
        mode: Mode key for tf.estimator.ModeKeys.
        params: Parameter dictionary passed from the estimator.
        
    Returns:
        An `EstimatorSpec` the encapsulates the model and its serving
        configurations.
    """
    params = params or {}
    loss, acc, train_op, export_outputs = None, None, None, None
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    
    cls_model = model.Model(is_training=is_training, 
                            num_classes=FLAGS.num_classes)
    preprocessed_inputs = cls_model.preprocess(features.get('image'))
    prediction_dict = cls_model.predict(preprocessed_inputs)
    postprocessed_dict = cls_model.postprocess(prediction_dict)
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        if FLAGS.checkpoint_path:
            init_variables_from_checkpoint()
    
    if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
        loss_dict = cls_model.loss(prediction_dict, labels)
        loss = loss_dict['loss']
        classes = postprocessed_dict['classes']
        acc = tf.reduce_mean(tf.cast(tf.equal(classes, labels), 'float'))
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', acc)
    
    scaffold = None
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()
        learning_rate = configure_learning_rate(FLAGS.decay_steps,
                                                global_step)
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                               momentum=0.9)
        train_op = slim.learning.create_train_op(loss, optimizer,
                                                 summarize_gradients=True)
        
        keep_checkpoint_every_n_hours = FLAGS.keep_checkpoint_every_n_hours
        saver = tf.train.Saver(
            sharded=True,
            keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours,
            save_relative_paths=True)
        tf.add_to_collection(tf.GraphKeys.SAVERS, saver)
        scaffold = tf.train.Scaffold(saver=saver)
        
    eval_metric_ops = None
    if mode == tf.estimator.ModeKeys.EVAL:
        accuracy = tf.metrics.accuracy(labels=labels, predictions=classes)
        eval_metric_ops = {'Accuracy': accuracy}
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        export_output = exporter._add_output_tensor_nodes(postprocessed_dict)
        export_outputs = {
            tf.saved_model.signature_constants.PREDICT_METHOD_NAME:
                tf.estimator.export.PredictOutput(export_output)}
    
        
    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=prediction_dict,
                                      loss=loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops,
                                      export_outputs=export_outputs,
                                      scaffold=scaffold)
    
    
def configure_learning_rate(decay_steps, global_step):
    """Configures the learning rate.
    
    Modified from:
        https://github.com/tensorflow/models/blob/master/research/slim/
        train_image_classifier.py
    
    Args:
        decay_steps: The step to decay learning rate.
        global_step: The global_step tensor.
        
    Returns:
        A `Tensor` representing the learning rate.
    """ 
    if FLAGS.learning_rate_decay_type == 'exponential':
        return tf.train.exponential_decay(FLAGS.learning_rate,
                                          global_step,
                                          decay_steps,
                                          FLAGS.learning_rate_decay_factor,
                                          staircase=True,
                                          name='exponential_decay_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'fixed':
        return tf.constant(FLAGS.learning_rate, name='fixed_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'polynomial':
        return tf.train.polynomial_decay(FLAGS.learning_rate,
                                         global_step,
                                         decay_steps,
                                         FLAGS.end_learning_rate,
                                         power=1.0,
                                         cycle=False,
                                         name='polynomial_decay_learning_rate')
    else:
        raise ValueError('learning_rate_decay_type [%s] was not recognized' %
                         FLAGS.learning_rate_decay_type)
        
        
def init_variables_from_checkpoint(checkpoint_exclude_scopes=None):
    """Variable initialization form a given checkpoint path.
    
    Modified from:
        https://github.com/tensorflow/models/blob/master/research/
        object_detection/model_lib.py
    
    Note that the init_fn is only run when initializing the model during the 
    very first global step.
    
    Args:
        checkpoint_exclude_scopes: Comma-separated list of scopes of variables
            to exclude when restoring from a checkpoint.
    """
    exclude_patterns = None
    if checkpoint_exclude_scopes:
        exclude_patterns = [scope.strip() for scope in 
                            checkpoint_exclude_scopes.split(',')]
    variables_to_restore = tf.global_variables()
    variables_to_restore.append(slim.get_or_create_global_step())
    variables_to_init = tf.contrib.framework.filter_variables(
        variables_to_restore, exclude_patterns=exclude_patterns)
    variables_to_init_dict = {var.op.name: var for var in variables_to_init}
    
    available_var_map = get_variables_available_in_checkpoint(
        variables_to_init_dict, FLAGS.checkpoint_path, 
        include_global_step=False)
    tf.train.init_from_checkpoint(FLAGS.checkpoint_path, available_var_map)
    
    
def get_variables_available_in_checkpoint(variables,
                                          checkpoint_path,
                                          include_global_step=True):
    """Returns the subset of variables in the checkpoint.
    
    Inspects given checkpoint and returns the subset of variables that are
    available in it.
    
    Args:
        variables: A dictionary of variables to find in checkpoint.
        checkpoint_path: Path to the checkpoint to restore variables from.
        include_global_step: Whether to include `global_step` variable, if it
            exists. Default True.
            
    Returns:
        A dictionary of variables.
        
    Raises:
        ValueError: If `variables` is not a dict.
    """
    if not isinstance(variables, dict):
        raise ValueError('`variables` is expected to be a dict.')
    
    # Available variables
    ckpt_reader = tf.train.NewCheckpointReader(checkpoint_path)
    ckpt_vars_to_shape_map = ckpt_reader.get_variable_to_shape_map()
    if not include_global_step:
        ckpt_vars_to_shape_map.pop(tf.GraphKeys.GLOBAL_STEP, None)
    vars_in_ckpt = {}
    for variable_name, variable in sorted(variables.items()):
        if variable_name in ckpt_vars_to_shape_map:
            if ckpt_vars_to_shape_map[variable_name] == variable.shape.as_list():
                vars_in_ckpt[variable_name] = variable
            else:
                logging.warning('Variable [%s] is avaible in checkpoint, but '
                                'has an incompatible shape with model '
                                'variable. Checkpoint shape: [%s], model '
                                'variable shape: [%s]. This variable will not '
                                'be initialized from the checkpoint.',
                                variable_name, 
                                ckpt_vars_to_shape_map[variable_name],
                                variable.shape.as_list())
        else:
            logging.warning('Variable [%s] is not available in checkpoint',
                            variable_name)
    return vars_in_ckpt


def main(_):
    # Specify which gpu to be used
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_indices
    
    estimator = tf.estimator.Estimator(model_fn=create_model_fn, 
                                       model_dir=FLAGS.model_dir)
    train_input_fn = create_input_fn([FLAGS.train_record_path], 
                                     batch_size=FLAGS.batch_size)
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                        max_steps=FLAGS.num_steps)
    eval_input_fn = create_input_fn([FLAGS.val_record_path], 
                                    batch_size=FLAGS.batch_size,
                                    num_epochs=1)
    predict_input_fn = create_predict_input_fn()
    eval_exporter = tf.estimator.FinalExporter(
        name='servo', serving_input_receiver_fn=predict_input_fn)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=None,
                                      exporters=eval_exporter)
    
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    
if __name__ == '__main__':
    tf.app.run()