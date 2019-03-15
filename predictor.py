# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 11:49:09 2018

@author: shirhe-lyh
"""

import os
import tensorflow as tf

# Note: We need to import addditional module to fix the following bug:
# tensorflow.python.framework.errors_impl.NotFoundError: Op type not 
# registered 'ImageProjectiveTransform' in binary running on BJGS-SF-81. 
# Make sure the Op and Kernel are registered in the binary running in this 
# process. Note that if you are loading a saved graph which used ops from 
# tf.contrib, accessing (e.g.) `tf.contrib.resampler` should be done before 
# importing the graph, as contrib ops are lazily registered when the module 
# is first accessed.
import tensorflow.contrib.image

#from timeit import default_timer as timer


class Predictor(object):
    """Classify images to predifined classes."""
    
    def __init__(self,
                 frozen_inference_graph_path,
                 gpu_index=None):
        """Constructor.
        
        Args:
            frozen_inference_graph_path: Path to frozen inference graph.
            gpu_index: The GPU index to be used. Default None.
        """
        self._gpu_index = gpu_index
        
        self._graph, self._sess = self._load_model(frozen_inference_graph_path)
        self._inputs = self._graph.get_tensor_by_name('image_tensor:0')
#        self._logits = self._graph.get_tensor_by_name('logits:0')
        self._classes = self._graph.get_tensor_by_name('classes:0')
        
    def _load_model(self, frozen_inference_graph_path):
        """Load a (frozen) Tensorflow model into memory.
        
        Args:
            frozen_inference_graph_path: Path to frozen inference graph.
            
        Returns:
            graph: A tensorflow Graph object.
            sess: A tensorflow Session object.
        
        Raises:
            ValueError: If frozen_inference_graph_path does not exist.
        """
        if not tf.gfile.Exists(frozen_inference_graph_path):
            raise ValueError('`frozen_inference_graph_path` does not exist.')
            
        # Specify which gpu to be used.
        if self._gpu_index is not None:
            if not isinstance(self._gpu_index, str):
                self._gpu_index = str(self._gpu_index)
            os.environ["CUDA_VISIBLE_DEVICES"] = self._gpu_index
            
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(frozen_inference_graph_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            
        config = tf.ConfigProto(allow_soft_placement = True) 
        config.gpu_options.per_process_gpu_memory_fraction = 0.50
        sess = tf.Session(graph=graph, config=config)
        return graph, sess
        
    def predict(self, inputs):
        """Predict prediction tensors from inputs tensor.
        
        Args:
            preprocessed_inputs: A 4D float32 tensor with shape [batch_size, 
                height, width, channels] representing a batch of images.
            
        Returns:
            classes: A 1D integer tensor with shape [batch_size].
        """
        feed_dict = {self._inputs: inputs}
        classes = self._sess.run(self._classes, feed_dict=feed_dict)
        return classes
    
        