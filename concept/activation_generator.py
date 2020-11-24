from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import os.path
import six
from abc import ABCMeta
from abc import abstractmethod

from concept import utils


class ActivationGeneratorInterface(six.with_metaclass(ABCMeta, object)):
    """ Interface for an activation generator for a model"""

    @abstractmethod
    def process_and_load_activations(self, bottlenecks, concepts):
        pass

    @abstractmethod
    def process_and_load_grads(self, bottlenecks, target_classes):
        pass

    @abstractmethod
    def get_model(self):
        pass


class ActivationGeneratorBase(ActivationGeneratorInterface):
    """ Basic abstract activation generator for a model """

    def __init__(self, model, working_dir, max_examples=500, resume=True):
        self.model = model
        self.acts_dir = os.path.join(working_dir, 'acts')
        self.grads_dir = os.path.join(working_dir, 'grads')
        self.max_examples = max_examples
        if resume is False:
            utils.rm_tree(self.grads_dir)
            utils.rm_tree(self.acts_dir)

    @staticmethod
    def process_and_load(bottlenecks, targets, save_dir, calc_func):
        results = {}
        for target in targets:
            if target not in results:
                results[target] = {}
            remain_bottlenecks = []
            # First load already processed grads
            for bottleneck in bottlenecks:
                grads_path = os.path.join(save_dir,
                                          'grads_{}_{}'.format(target,
                                                               bottleneck))
                if grads_path and os.path.exists(grads_path):
                    with open(grads_path, 'rb') as f:
                        results[target][bottleneck] = np.load(f, allow_pickle=True).squeeze()
                    print('Loaded {} shape {}'.format(
                        grads_path, results[target][bottleneck].shape))
                else:
                    remain_bottlenecks.append(bottleneck)

            # Then take care of the rest, and save them
            results[target].update(calc_func(target, remain_bottlenecks))
            for bottleneck in remain_bottlenecks:
                grads_path = os.path.join(save_dir,
                                          'grads_{}_{}'.format(target,
                                                               bottleneck))
                if bottleneck in results[target]:
                    with open(grads_path, 'wb') as f:
                        np.save(f, results[target][bottleneck], allow_pickle=False)
                else:
                    print('Ignore gradients of {}-{}'.format(target, bottleneck))

        return results

    def get_model(self):
        return self.model

    @abstractmethod
    def get_examples_for_concept(self, concept):
        pass

    @abstractmethod
    def get_examples_for_class(self, target_class):
        pass

    def get_activations_for_concept(self, concept, bottlenecks):
        examples = self.get_examples_for_concept(concept)
        return self.get_activations_for_examples(examples, bottlenecks)

    def get_activations_for_examples(self, examples, bottlenecks):
        return self.model.get_acts(examples, bottlenecks)

    def get_grads_for_class(self, target_class, bottlenecks):
        examples = self.get_examples_for_class(target_class)
        return self.get_grads_for_examples(examples, bottlenecks, target_class)

    def get_grads_for_examples(self, examples, bottlenecks, target_class):
        target_id = self.model.label_to_id(target_class)
        return self.model.get_gradient(examples, bottlenecks, target_id)

    def process_and_load_grads(self, bottlenecks, target_classes):
        return ActivationGeneratorBase.process_and_load(bottlenecks,
                                                        target_classes,
                                                        self.grads_dir,
                                                        self.get_grads_for_class)

    def process_and_load_activations(self, bottlenecks, concepts):
        return ActivationGeneratorBase.process_and_load(bottlenecks,
                                                        concepts,
                                                        self.acts_dir,
                                                        self.get_activations_for_concept)


class ImageActivationGenerator(ActivationGeneratorBase):
    def __init__(self,
                 model,
                 source_dir,
                 working_dir,
                 max_examples=10,
                 transform=None):
        self.source_dir = source_dir
        self.concept_dir = os.path.join(working_dir, 'concepts')
        self.transform = transform
        super(ImageActivationGenerator, self).__init__(model,
                                                       working_dir,
                                                       max_examples)

    def get_examples_for_class(self, target_class):
        paths = utils.get_paths_dir_subdir(self.source_dir, target_class)
        return utils.load_images_from_files(paths,
                                            max_imgs=self.max_examples,
                                            shape=self.model.get_image_shape()[:2],
                                            batch_size=32)

    def get_examples_for_concept(self, concept):
        paths = utils.get_paths_dir_subdir(self.concept_dir, concept)
        return utils.load_images_from_files(paths,
                                            max_imgs=self.max_examples,
                                            shape=self.model.get_image_shape()[:2],
                                            batch_size=32)
