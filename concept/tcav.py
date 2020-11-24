from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from multiprocessing import dummy as multiprocessing
from six.moves import range
from concept.cav import CAV
from concept.cav import get_or_train_cav
from concept import run_params
from concept import utils
import numpy as np
import time
import gc
import os


class TCAV(object):
    @staticmethod
    def get_direction_dir(cav, concept, grad):
        """Get the sign of directional derivative.

        Args:
            cav: an instance of cav
            concept: one concept
            grad: target gradient
        Returns:
            sign of the directional derivative
        """
        # Grad points in the direction which DECREASES probability of class
        grad = np.reshape(grad, -1)
        cav_dir = np.reshape(cav.get_direction(concept), -1)
        dot_prod = np.dot(grad, cav_dir)
        return dot_prod

    @staticmethod
    def get_direction_dir_sign(cav, concept, grad):
        return TCAV.get_direction_dir(cav, concept, grad) < 0

    @staticmethod
    def compute_tcav_score(cav,
                           concept,
                           grads,
                           num_workers=10):
        """Compute TCAV score.
        Args:
          cav: an instance of cav
          concept: one concept
          grads: grads of the examples in the target class to use
          run_parallel: run this parallel fashion
          num_workers: number of workers if we run in parallel.

        Returns:
            TCAV score (i.e., ratio of pictures that returns negative
            dot product wrt loss).
        """

        count = 0
        if num_workers > 0:
            pool = multiprocessing.Pool(num_workers)
            directions = pool.map(
                lambda i: TCAV.get_direction_dir_sign(
                    cav, concept, grads[i]),
                range(len(grads)))
            return sum(directions) / float(len(grads))
        else:
            for i in range(len(grads)):
                if TCAV.get_direction_dir_sign(
                        cav, concept, grads[i]):
                    count += 1
            return float(count) / float(len(grads))

    @staticmethod
    def get_directional_dir_vals(cav,
                                 concept,
                                 grads):

        directional_dir_vals = []
        for i in range(len(grads)):
            val = TCAV.get_direction_dir(cav, concept, grads[i])
            directional_dir_vals.append(val)
        return directional_dir_vals

    def __init__(self,
                 target,
                 concepts,
                 bottlenecks,
                 activation_generator,
                 alphas,
                 random_counterpart=None,
                 working_dir=None,
                 num_random_exp=5,
                 random_concepts=None):
        """
        Args:
          target: one target class
          concepts: A list of names of positive concept sets.
          bottlenecks: the name of a bottleneck of interest.
          activation_generator: an ActivationGeneratorInterface instance to return
                                activations.
          alphas: list of hyper parameters to run
          working_dir: the path to store CAVs
          random_counterpart: the random concept to run against the concepts for
                      statistical testing. If supplied, only this set will be
                      used as a positive set for calculating random TCAVs
          num_random_exp: number of random experiments to compare against.
          random_concepts: A list of names of random concepts for the random
                           experiments to draw from. Optional, if not provided, the
                           names will be random500_{i} for i in num_random_exp.
                           Relative TCAV can be performed by passing in the same
                           value for both concepts and random_concepts.
        """
        self.target = target
        self.concepts = concepts
        self.bottlenecks = bottlenecks
        self.activation_generator = activation_generator
        self.cav_dir = os.path.join(working_dir, 'cavs')
        self.alphas = alphas
        self.random_counterpart = random_counterpart
        self.relative_tcav = (random_concepts is not None) and (set(concepts) == set(random_concepts))

        if num_random_exp < 2:
            print('The number of random concepts has to be at least 2')
        if random_concepts:
            num_random_exp = len(random_concepts)

        self._process_what_to_run_expand(num_random_exp=num_random_exp,
                                         random_concepts=random_concepts)

        # param_format: (bottleneck, concepts_in_test, target_in_test, alpha)
        self.params = self.get_params()
        print('TCAV have %s params' % len(self.params))

    def run(self, num_workers=10, overwrite=False):
        if overwrite:
            utils.rm_tree(self.cav_dir)
            utils.rm_tree(self.activation_generator.acts_dir)
            utils.rm_tree(self.activation_generator.grads_dir)

        utils.make_dir_if_not_exists(self.cav_dir)
        utils.make_dir_if_not_exists(self.activation_generator.grads_dir)
        utils.make_dir_if_not_exists(self.activation_generator.acts_dir)

        print('running %s params' % len(self.params))
        now = time.time()
        results = []

        # TODO: Make parallel working here
        # if num_workers > 0:
        #     pool = multiprocessing.Pool(num_workers)
        #     for i, res in enumerate(pool.imap(
        #             lambda p: self._run_single_test(param=p,
        #                                             num_workers=num_workers),
        #             self.params), 1):
        #         print('Finished running param %s of %s' % (i, len(self.params)))
        #         print('TCAV score:', res['i_up'])
        #         print('=' * 10)
        #         results.append(res)
        # else:

        for i, param in enumerate(self.params):
            print('Running param %s of %s' % (i, len(self.params)))
            res = self._run_single_test(param=param,
                                        num_workers=num_workers)
            print('TCAV score:', res['i_up'])
            print('=' * 10)
            results.append(res)

        print('Done running %s params. Took %s seconds...' % (len(self.params), time.time() - now))
        return results

    def _run_single_test(self, param, num_workers=10):
        """Run TCAV with provided for one set of (target, concepts).

        Args:
          param: parameters to run
          overwrite: if True, overwrite any saved CAV files.
          run_parallel: run this parallel.

        Returns:
          a dictionary of results (panda frame)
        """
        bottleneck = param.bottleneck
        concepts = param.concepts
        target_class = param.target_class
        activation_generator = param.activation_generator
        alpha = param.alpha
        cav_dir = param.cav_dir

        print('running %s %s' % (target_class, concepts))

        acts = activation_generator.process_and_load_activations([bottleneck], concepts)

        cav_hparams = CAV.default_hparams()
        cav_hparams.alpha = alpha
        cav_instance = get_or_train_cav(
            concepts,
            bottleneck,
            acts,
            cav_dir=cav_dir,
            cav_hparams=cav_hparams
        )

        for c in concepts:
            del acts[c]

        a_cav_key = CAV.cav_key(concepts, bottleneck,
                                cav_hparams.model_type,
                                cav_hparams.alpha)
        cav_concept = concepts[0]

        grads = activation_generator.process_and_load_grads([bottleneck], [target_class])

        i_up = self.compute_tcav_score(cav_instance,
                                       cav_concept,
                                       grads[target_class][bottleneck],
                                       num_workers=num_workers)
        val_directional_dirs = self.get_directional_dir_vals(cav_instance,
                                                             cav_concept,
                                                             grads[target_class][bottleneck])

        result = {
            'cav_key': a_cav_key,
            'cav_concept': cav_concept,
            'negative_concept': concepts[1],
            'target_class': target_class,
            'cav_accuracies': cav_instance.accuracies,
            'i_up': i_up,
            'val_directional_dirs_abs_mean': np.mean(np.abs(val_directional_dirs)),
            'val_directional_dirs_mean': np.mean(val_directional_dirs),
            'val_directional_dirs_std': np.std(val_directional_dirs),
            'val_directional_dirs': val_directional_dirs,
            'alpha': alpha,
            'bottleneck': bottleneck
        }

        del grads[target_class]
        del acts
        del grads
        gc.collect()

        return result

    def _process_what_to_run_expand(self, num_random_exp=100, random_concepts=None):
        """Get tuples of parameters to run TCAV with.
        TCAV builds random concept to conduct statistical significance testing
        againts the concept. To do this, we build many concept vectors, and many
        random vectors. This function prepares runs by expanding parameters.

        Args:
          num_random_exp: number of random experiments to run to compare.
          random_concepts: A list of names of random concepts for the random experiments
                       to draw from. Optional, if not provided, the names will be
                       random500_{i} for i in num_random_exp.
        """
        # Build pairs for target concepts
        target_concept_pairs = [(self.target, self.concepts)]
        all_concepts_concepts, pairs_to_run_concepts = (
            utils.process_what_to_run_expand(
                utils.process_what_to_run_concepts(target_concept_pairs),
                self.random_counterpart,
                num_random_exp=num_random_exp
                               - (1 if random_concepts and self.random_counterpart in random_concepts else 0)
                               - (1 if self.relative_tcav else 0),
                random_concepts=random_concepts
            )
        )

        def get_random_concept(i):
            return random_concepts[i] if random_concepts else 'random500_{}'.format(i)

        # Build pairs for random concepts
        pairs_to_run_randoms = []
        all_concepts_randoms = []
        if self.random_counterpart is None:
            for i in range(num_random_exp):
                all_concepts_randoms_tmp, pairs_to_run_randoms_tmp = (
                    utils.process_what_to_run_expand(
                        utils.process_what_to_run_randoms(target_concept_pairs,
                                                          get_random_concept(i)),
                        num_random_exp=num_random_exp - 1,
                        random_concepts=random_concepts)
                )
                pairs_to_run_randoms.extend(pairs_to_run_randoms_tmp)
                all_concepts_randoms.extend(all_concepts_randoms_tmp)
        else:
            all_concepts_randoms_tmp, pairs_to_run_randoms_tmp = (
                utils.process_what_to_run_expand(
                    utils.process_what_to_run_randoms(target_concept_pairs,
                                                      self.random_counterpart),
                    self.random_counterpart,
                    num_random_exp=num_random_exp -
                                   (1 if random_concepts and
                                         self.random_counterpart in random_concepts else 0),
                    random_concepts=random_concepts)
            )

            pairs_to_run_randoms.extend(pairs_to_run_randoms_tmp)
            all_concepts_randoms.extend(all_concepts_randoms_tmp)

        self.all_concepts = list(set(all_concepts_concepts + all_concepts_randoms))
        self.pairs_to_test = pairs_to_run_concepts \
            if self.relative_tcav \
            else pairs_to_run_concepts + pairs_to_run_randoms

    def get_params(self):
        """Enumerate parameters for the run function.

        Returns:
          parameters
        """
        params = []
        for bottleneck in self.bottlenecks:
            for target_in_test, concepts_in_test in self.pairs_to_test:
                for alpha in self.alphas:
                    print('%s %s %s %s' % (bottleneck, concepts_in_test,
                                           target_in_test, alpha))
                    params.append(run_params.RunParams(bottleneck, concepts_in_test,
                                                       target_in_test, self.activation_generator,
                                                       self.cav_dir, alpha))
        return params
