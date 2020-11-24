from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numbers
from multiprocessing import dummy as multiprocessing

import PIL.Image
import numpy as np
import os
import shutil
import six
from scipy.stats import ttest_ind


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


def get_paths_dir_subdir(source_dir, subdir):
    img_dir = os.path.join(source_dir, subdir)
    img_paths = [os.path.join(img_dir, d)
                 for d in os.listdir(img_dir)]
    return img_paths


def load_image_from_file(path, shape, return_raw, normalize):
    if not os.path.exists(path):
        print('Cannot find file: {}'.format(path))
        return None
    try:
        with open(path, 'rb') as f:
            img = PIL.Image.open(f).convert('RGB').resize(shape, PIL.Image.BILINEAR)
            img = np.array(img, dtype='float32')
            if normalize:
                img /= 255.0
            if not return_raw:
                img = np.transpose(img, (2, 0, 1))
                if not (len(img.shape) == 3 and img.shape[0] == 3):
                    print('Invalid image shape: {}'.format(path))
                    return None

    except Exception as e:
        print(e)
        return None

    return img


def load_images_from_files(paths,
                           max_imgs=500,
                           do_shuffle=True,
                           run_parallel=True,
                           shape=(299, 299),
                           num_workers=10,
                           batch_size=32,
                           return_raw=False,
                           normalize=False):
    imgs = []
    if do_shuffle:
        np.random.shuffle(paths)

    if run_parallel:
        pool = multiprocessing.Pool(num_workers)
        imgs = pool.map(
            lambda path: load_image_from_file(path, shape, return_raw, normalize),
            paths[:max_imgs]
        )
    else:
        imgs = [load_image_from_file(path, shape, return_raw, normalize)
                for path in paths]
    imgs = [img for img in imgs if img is not None]
    if len(imgs) < 1:
        raise ValueError('Must have more than one image in each class/concept to run TCAV')

    if return_raw:
        return imgs

    chunked_imgs = chunks(imgs, batch_size)
    return [np.array(chunked_img) for chunked_img in chunked_imgs]


def _cast_to_type_if_compatible(name, param_type, value):
    fail_msg = (
            "Could not cast hparam '%s' of type '%s' from value %r" %
            (name, param_type, value))

    # Some callers use None, for which we can't do any casting/checking. :(
    if issubclass(param_type, type(None)):
        return value

    # Avoid converting a non-string type to a string.
    if (issubclass(param_type, (six.string_types, six.binary_type)) and
            not isinstance(value, (six.string_types, six.binary_type))):
        raise ValueError(fail_msg)

    # Avoid converting a number or string type to a boolean or vice versa.
    if issubclass(param_type, bool) != isinstance(value, bool):
        raise ValueError(fail_msg)

    # Avoid converting float to an integer (the reverse is fine).
    if (issubclass(param_type, numbers.Integral) and
            not isinstance(value, numbers.Integral)):
        raise ValueError(fail_msg)

    # Avoid converting a non-numeric type to a numeric type.
    if (issubclass(param_type, numbers.Number) and
            not isinstance(value, numbers.Number)):
        raise ValueError(fail_msg)

    return param_type(value)


class HParams(object):
    def __init__(self, **kwargs):
        self._hparam_types = {}
        for name, value in six.iteritems(kwargs):
            self.add_hparam(name, value)

    def add_hparam(self, name, value):
        if getattr(self, name, None) is not None:
            raise ValueError('Hyperparameter name is reversed: %s' % name)
        if isinstance(value, (list, tuple)):
            if not value:
                raise ValueError(
                    'Multi-valued hyperparameters cannot be empty: %s' % name)
            self._hparam_types[name] = (type(value[0], True))
        else:
            self._hparam_types[name] = (type(value), False)
        setattr(self, name, value)

    def set_hparam(self, name, value):
        param_type, is_list = self._hparam_types[name]
        if isinstance(value, list):
            if not is_list:
                raise ValueError(
                    'Must not past a list for single-valued parameter: %s' % name)
            setattr(self, name, [
                _cast_to_type_if_compatible(name, param_type, v) for v in value])
        else:
            if is_list:
                raise ValueError(
                    'Must pass a list for multi-valued parameters: %s.' % name)
            setattr(self, name, value)


def flatten(nested_list):
    """Flatten a nested list."""
    return [item for a_list in nested_list for item in a_list]


def make_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def rm_tree(directory):
    if os.path.exists(directory):
        print('Reset', directory)
        shutil.rmtree(directory)


def process_what_to_run_concepts(pairs_to_test):
    """Process concepts and pairs to test.

    Args:
      pairs_to_test: a list of concepts to be tested and a target (e.g,
       [ ("target1",  ["concept1", "concept2", "concept3"]),...])

    Returns:
      return pairs to test:
         target1, concept1
         target1, concept2
         ...
         target2, concept1
         target2, concept2
         ...

    """
    pairs_for_testing = []
    # prepare pairs for concept vs random.
    for pair in pairs_to_test:
        for concept in pair[1]:
            pairs_for_testing.append([pair[0], [concept]])
    return pairs_for_testing


def process_what_to_run_randoms(pairs_to_test, random_counterpart):
    """Process concepts and pairs to test.

    Args:
      pairs_to_test: a list of concepts to be tested and a target (e.g,
       [ ("target1",  ["concept1", "concept2", "concept3"]),...])
      random_counterpart: a random concept that will be compared to the concept.

    Returns:
      return pairs to test:
            target1, random_counterpart,
            target2, random_counterpart,
            ...
    """
    # prepare pairs for random vs random.
    pairs_for_testing_random = []
    targets = list(set([pair[0] for pair in pairs_to_test]))
    for target in targets:
        pairs_for_testing_random.append([target, [random_counterpart]])
    return pairs_for_testing_random


def process_what_to_run_expand(pairs_to_test,
                               random_counterpart=None,
                               num_random_exp=100,
                               random_concepts=None):
    """Get concept vs. random or random vs. random pairs to run.

      Given set of target, list of concept pairs, expand them to include
       random pairs. For instance [(t1, [c1, c2])...] becomes
       [(t1, [c1, random1],
        (t1, [c1, random2],...
        (t1, [c2, random1],
        (t1, [c2, random2],...]

    Args:
      pairs_to_test: [(target1, concept1), (target1, concept2), ...,
                      (target2, concept1), (target2, concept2), ...]
      random_counterpart: random concept that will be compared to the concept.
      num_random_exp: number of random experiments to run against each concept.
      random_concepts: A list of names of random concepts for the random
                       experiments to draw from. Optional, if not provided, the
                       names will be random500_{i} for i in num_random_exp.

    Returns:
      all_concepts: unique set of targets/concepts
      new_pairs_to_test: expanded
    """

    def get_random_concept(i):
        return (random_concepts[i] if random_concepts
                else 'random500_{}'.format(i))

    new_pairs_to_test = []
    for (target, concept_set) in pairs_to_test:
        new_pairs_to_test_t = []
        # if only one element was given, this is to test with random.
        if len(concept_set) == 1:
            i = 0
            while len(new_pairs_to_test_t) < min(100, num_random_exp):
                # make sure that we are not comparing the same thing to each other.
                if concept_set[0] != get_random_concept(i) \
                        and random_counterpart != get_random_concept(i):
                    new_pairs_to_test_t.append((target,
                                                [concept_set[0], get_random_concept(i)]))
                i += 1
        # if there are two concepts, this is to test with non-random concepts
        elif len(concept_set) > 1:
            new_pairs_to_test_t.append((target, concept_set))
        else:
            print('PAIR NOT PROCESSED')
        new_pairs_to_test.extend(new_pairs_to_test_t)

    all_concepts = list(set(flatten([cs for tc, cs in new_pairs_to_test])))
    return all_concepts, new_pairs_to_test


def print_results(results, random_counterpart=None,
                  random_concepts=None,
                  min_p_val=0.05):
    """Helper function to organize results.
    If you ran TCAV with a random_counterpart, supply it here, otherwise supply random_concepts.
    If you get unexpected output, make sure you are using the correct keywords.

    Args:
      results: dictionary of results from TCAV runs.
      random_counterpart: name of the random_counterpart used, if it was used.
      random_concepts: list of random experiments that were run.
      min_p_val: minimum p value for statistical significance
    """

    # helper function, returns if this is a random concept
    def is_random_concept(concept):
        if random_counterpart:
            return random_counterpart == concept

        elif random_concepts:
            return concept in random_concepts

        else:
            return 'random500_' in concept

    # print class, it will be the same for all
    print("Class =", results[0]['target_class'])

    # prepare data
    # dict with keys of concepts containing dict with bottlenecks
    result_summary = {}

    # random
    random_i_ups = {}

    for result in results:
        concept = result['cav_concept']
        bottleneck = result['bottleneck']
        if concept not in result_summary:
            result_summary[concept] = {}

        if bottleneck not in result_summary[concept]:
            result_summary[concept][bottleneck] = []

        result_summary[concept][bottleneck].append(result)

        # store random
        if is_random_concept(concept):
            if bottleneck not in random_i_ups:
                random_i_ups[bottleneck] = []

            random_i_ups[bottleneck].append(result['i_up'])

    # print concepts and classes with indentation
    for concept in result_summary:

        # if not random
        if not is_random_concept(concept):
            print(" ", "Concept =", concept)

            for bottleneck in result_summary[concept]:
                i_ups = [item['i_up'] for item in result_summary[concept][bottleneck]]

                # Calculate statistical significance
                _, p_val = ttest_ind(random_i_ups[bottleneck], i_ups)

                print(3 * " ", "Bottleneck =", ("%s. TCAV Score = %.2f (+- %.2f), "
                                                "random was %.2f (+- %.2f). p-val = %.3f (%s)") % (
                          bottleneck, np.mean(i_ups), np.std(i_ups),
                          np.mean(random_i_ups[bottleneck]),
                          np.std(random_i_ups[bottleneck]), p_val,
                          "undefined" if np.isnan(
                              p_val) else "not significant" if p_val > min_p_val else "significant"))
