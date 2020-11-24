from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six
import torch
import torch.nn as nn
from abc import ABCMeta
from abc import abstractmethod


class ModelWrapper(six.with_metaclass(ABCMeta, object)):
    @abstractmethod
    def __init__(self, model, state_dict_path=None):
        self.bottleneck_tensors = None
        self.ends = None
        self.model_name = None
        self.y_input = None

        def dummy_loss(out, y):
            return torch.mean(out[:, y])

        self.loss = dummy_loss
        self.import_prefix = False
        self.model = model
        if state_dict_path:
            self._try_loading_model(state_dict_path)
            print('Loaded model {}'.format(state_dict_path))

    def _try_loading_model(self, state_dict_path):
        try:
            self.model.load_state_dict(torch.load(state_dict_path))
        except Exception as e:
            template = f'An exception of type {type(e)} occurred ' \
                       'when trying to load model from {model_path} ' \
                       'Arguments:\n{e.args}'
            print(template)

    @staticmethod
    def process_result(output):
        tmp = []
        for i in range(output.shape[0]):
            tmp.append(output[i].detach().cpu().numpy())
        return tmp

    @staticmethod
    def hook_fw_fn(fw_buffer):
        def hook_fw(module, input, output):
            fw_buffer.extend(ModelWrapper.process_result(output))

        return hook_fw

    @staticmethod
    def hook_bw_fn(bw_buffer):
        def hook_bw(module, input, output):
            bw_buffer.extend(ModelWrapper.process_result(output[0]))

        return hook_bw

    def _setting_fw_hook_bottleneck(self, bottlenecks):
        """
        Finds and setting up forward hooks for calculating activation
        @param bottlenecks: list of target bottlenecks' names
        @type bottlenecks: list
        @return: a dictionary that will be used to save the result
        """
        fw_buffer = {}
        for name, module in self.model.named_modules():
            if type(module) != nn.Sequential and type(module) != self.model.__class__:
                for layer in bottlenecks:
                    if name == layer:
                        fw_buffer[layer] = []
                        module.register_forward_hook(ModelWrapper.hook_fw_fn(fw_buffer[layer]))
        return fw_buffer

    def _setting_bw_hook_bottleneck(self, bottlenecks):
        """
        Finds and setting up backward hooks for calculating gradient
        @param bottlenecks: list of target bottlenecks' names
        @type bottlenecks: list
        @return: a dictionary that will be used to save the result
        """
        bw_buffer = {}
        for name, module in self.model.named_modules():
            if type(module) != nn.Sequential and type(module) != self.model.__class__:
                for layer in bottlenecks:
                    if name == layer:
                        bw_buffer[layer] = []
                        module.register_backward_hook(ModelWrapper.hook_bw_fn(bw_buffer[layer]))
        return bw_buffer

    def get_gradient(self, examples, bottlenecks, y):
        """
        Get gradient for target label y at layer in node_dict with specified examples
        @param bottlenecks: list of target bottlenecks' names
        @param y: target class
        @param examples: examples to use
        @return: a dictionary buffer that contains the result
        """
        bw_buffer = self._setting_bw_hook_bottleneck(bottlenecks)
        for example in examples:
            example = torch.tensor(example) if isinstance(example, np.ndarray) else example
            example.requires_grad = True
            out = self.adjust_prediction(self.model(example))
            loss = -torch.sum(out[:, y])
            loss.backward()
        return bw_buffer

    def get_acts(self, examples, bottlenecks):
        """
        @param bottlenecks: list of target bottlenecks' names
        @param examples: list of examples to use
        @return: a dictionary buffer that contains the result
        """
        fw_buffer = self._setting_fw_hook_bottleneck(bottlenecks)
        for example in examples:
            example = torch.tensor(example).float() if isinstance(example, np.ndarray) else example.float()
            _ = self.model(example)
        return fw_buffer

    def get_predictions(self, example):
        example = torch.tensor(example) if isinstance(example, np.ndarray) else example
        out = self.adjust_prediction(self.model(example))
        return ModelWrapper.process_result(out)

    def adjust_prediction(self, pred_t):
        return pred_t

    def reshape_activation(self, layer_acts):
        return torch.reshape(layer_acts, -1)

    def label_to_id(self, label):
        print('Warning: label_to_id undefined. Defaults to returning 0.')
        return 0

    def id_to_label(self, idx):
        return str(idx)


class ImageModelWrapper(ModelWrapper):
    def __init__(self, model, state_dict_path,
                 image_shape, labels_path,
                 criterion=nn.CrossEntropyLoss()):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.image_shape = image_shape

        with open(labels_path, 'r') as f:
            self.labels = [s[:-1] for s in f.readlines()]
        if state_dict_path:
            self._try_loading_model(state_dict_path)
            print('Loaded model {}'.format(state_dict_path))

        def loss(out, y):
            target = torch.tensor([y] * out.size()[0])
            return criterion(out, target)

        self.loss = loss

    def get_image_shape(self):
        return self.image_shape

    def id_to_label(self, idx):
        return self.labels[idx]

    def label_to_id(self, label):
        return self.labels.index(label)
