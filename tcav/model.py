from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from abc import ABCMeta
from abc import abstractmethod
from six.moves import zip
import numpy as np
import six
import torch

class ModelWrapper(six.with_metaclass(ABCMeta, object)):
    @abstractmethod
    def __init__(self, model_path=None, node_dict=None):
        pass

    def _try_loading_model(self, model_path):
        pass

    def _find_ends_and_bottleneck_tensors(self, node_dict):
        pass

    def _make_gradient_tensors(self):
        pass

    def get_gradient(self, acts, y, bottleneck_name, example):
        pass

    def get_predictions(self, examples):
        pass

    def adjust_prediction(self, pred_t):
        pass

    def reshape_activation(self, layer_acts):
        pass

    def label_to_id(self, label):
        pass

    def id_to_label(self, idx):
        pass

    def run_examples(self, examples, bottleneck_name):
        pass