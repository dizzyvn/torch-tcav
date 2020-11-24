from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from multiprocessing import dummy as multiprocessing
import sys
import os
import numpy as np
from PIL import Image
import scipy.stats as stats
import skimage.segmentation as segmentation
import sklearn.cluster as cluster
import sklearn.metrics.pairwise as metrics
import torch
import torchvision
from concept import cav
from concept import tcav
from concept import utils
from matplotlib import pyplot as plt


class ACE(object):
    def __init__(self,
                 activation_generator,
                 target_class,
                 random_concept,
                 bottlenecks,
                 source_dir,
                 working_dir,
                 num_random_exp=2,
                 channel_mean=True,
                 max_imgs=40,
                 min_imgs=20,
                 num_discovery_imgs=40,
                 num_workers=10,
                 average_image_value=117):
        self.activation_generator = activation_generator
        self.model = activation_generator.model
        self.target_class = target_class
        self.random_concept = random_concept
        self.num_random_exp = num_random_exp
        if isinstance(bottlenecks, str):
            bottlenecks = [bottlenecks]
        self.bottlenecks = bottlenecks
        self.source_dir = source_dir
        self.concept_dir = os.path.join(working_dir, 'concepts')
        self.act_dir = os.path.join(working_dir, 'acts')
        self.cav_dir = os.path.join(working_dir, 'cavs')
        self.channel_mean = channel_mean
        self.image_shape = self.model.get_image_shape()[:2]
        self.max_imgs = max_imgs
        self.min_imgs = min_imgs
        if num_discovery_imgs is None:
            num_discovery_imgs = max_imgs
        self.num_discovery_imgs = num_discovery_imgs
        self.num_workers = num_workers
        self.average_image_value = average_image_value

    def load_class_imgs(self, target_class, max_imgs=1000):
        img_paths = utils.get_paths_dir_subdir(self.source_dir, target_class)
        return utils.load_images_from_files(
            paths=img_paths,
            max_imgs=max_imgs,
            shape=self.image_shape,
            return_raw=True,
            normalize=True)

    def load_concept_imgs(self, concept, max_imgs=1000):
        img_paths = utils.get_paths_dir_subdir(self.concept_dir, concept)
        return utils.load_images_from_files(
            paths=img_paths,
            max_imgs=max_imgs,
            shape=self.image_shape,
            return_raw=True,
            normalize=True)

    def create_patches(self, method='slic', discovery_images=None, param_dict=None):
        param_dict = {} if param_dict is None else param_dict
        dataset, image_idxs, patches = [], [], []
        if discovery_images is None:
            self.discovery_images = self.load_class_imgs(self.target_class,
                                                         self.num_discovery_imgs)
        else:
            self.discovery_images = discovery_images
        if self.num_workers:
            pool = multiprocessing.Pool(self.num_workers)
            outputs = pool.map(
                lambda img: self._return_superpixels(img, method, param_dict),
                self.discovery_images)
        else:
            outputs = [self._return_superpixels(img, method, param_dict)
                       for img in self.discovery_images]

        for idx, sp_outputs in enumerate(outputs):
            img_superpixels, img_patches = sp_outputs
            for superpixel, patch in zip(img_superpixels, img_patches):
                dataset.append(superpixel)
                patches.append(patch)
                image_idxs.append(idx)

        print('Created {} patches using {} segmentation method'.format(len(dataset), method))

        self.dataset = np.array(dataset)
        self.patches = np.array(patches)
        self.image_idxs = np.array(image_idxs)

    def _return_superpixels(self, img, method='slic', param_dict=None):
        param_dict = {} if param_dict is None else param_dict
        if method == 'slic':
            n_segmentss = param_dict.pop('n_segments', [15, 50, 80])
            n_params = len(n_segmentss)
            compactnesses = param_dict.pop('compactness', [20] * n_params)
            sigmas = param_dict.pop('sigma', [1.] * n_params)
        elif method == 'watershed':
            markerss = param_dict.pop('marker', [15, 50, 80])
            n_params = len(markerss)
            compactnesses = param_dict.pop('compactness', [0.] * n_params)
        elif method == 'quickshift':
            max_dists = param_dict.pop('max_dist', [20, 15, 10])
            n_params = len(max_dists)
            ratios = param_dict.pop('ratio', [1.0] * n_params)
            kernel_sizes = param_dict.pop('kernel_size', [10] * n_params)
        elif method == 'felzenszwalb':
            scales = param_dict.pop('scale', [1200, 500, 250])
            n_params = len(scales)
            sigmas = param_dict.pop('sigma', [0.8] * n_params)
            min_sizes = param_dict.pop('min_size', [20] * n_params)
        else:
            raise ValueError('Invalid superpixel method!')

        unique_masks = []
        for i in range(n_params):
            param_masks = []
            if method == 'slic':
                segments = segmentation.slic(
                    img,
                    n_segments=n_segmentss[i],
                    compactness=compactnesses[i],
                    sigma=sigmas[i])
            elif method == 'watershed':
                segments = segmentation.watershed(
                    img,
                    markers=markerss[i],
                    compactness=compactnesses[i])
            elif method == 'quickshift':
                segments = segmentation.quickshift(
                    img,
                    kernel_size=kernel_sizes[i],
                    max_dist=max_dists[i],
                    ratio=ratios[i])
            elif method == 'felzenszwalb':
                segments = segmentation.felzenszwalb(
                    img,
                    scale=scales[i],
                    sigma=sigmas[i],
                    min_size=min_sizes[i])

            for s in range(segments.max()):
                mask = (segments == s).astype(float)
                if np.mean(mask) > 0.001:
                    unique = True
                    for seen_mask in unique_masks:
                        jaccard = np.sum(seen_mask * mask) / np.sum((seen_mask + mask) > 0)
                        if jaccard > 0.5:
                            unique = False
                            break
                    if unique:
                        param_masks.append(mask)
            unique_masks.extend(param_masks)
        superpixels, patches = [], []

        while unique_masks:
            superpixel, patch = self._extract_patch(img, unique_masks.pop())
            superpixels.append(superpixel)
            patches.append(patch)

        return superpixels, patches

    def _extract_patch(self, image, mask):
        mask_expanded = np.expand_dims(mask, -1)
        patch = (mask_expanded * image +
                 (1 - mask_expanded) * float(self.average_image_value) / 255)

        ones = np.where(mask == 1)
        h1, h2, w1, w2 = ones[0].min(), ones[0].max(), ones[1].min(), ones[1].max()
        image = Image.fromarray((patch[h1:h2 + 1, w1:w2 + 1] * 255).astype(np.uint8))
        image_resized = np.array(image.resize(self.image_shape,
                                              Image.BICUBIC)).astype(float) / 255.
        patch = np.transpose(patch, (2, 0, 1))
        image_resized = np.transpose(image_resized, (2, 0, 1))
        return image_resized, patch

    def _patch_activations(self, imgs, bottleneck, batch_size=100):
        imgs = [np.array(chunked_img)
                for chunked_img
                in utils.chunks(imgs, batch_size)]
        output_dict = self.model.get_acts(imgs, [bottleneck])
        output = output_dict[bottleneck]
        if self.channel_mean:
            output = [np.mean(act, axis=2) for act in output]
        n_output = len(output)
        output = np.array(output).reshape(n_output, -1)
        return output

    def _cluster(self, acts, method='KM', param_dict=None):
        print('Starting clustering with {} for {} activations'.format(method, acts.shape[0]))
        if param_dict is None:
            param_dict = {}
        centers = None
        if method == 'KM':
            n_clusters = param_dict.pop('n_clusters', 25)
            km = cluster.KMeans(n_clusters)
            d = km.fit(acts)
            centers = km.cluster_centers_
            d = np.linalg.norm(
                np.expand_dims(acts, 1) - np.expand_dims(centers, 0), ord=2, axis=-1)
            asg, cost = np.argmin(d, -1), np.min(d, -1)
        elif method == 'AP':
            damping = param_dict.pop('damping', 0.5)
            ca = cluster.AffinityPropagation(damping)
            ca.fit(acts)
            centers = ca.cluster_centers_
            d = np.linalg.norm(
                np.expand_dims(acts, 1) - np.expand_dims(centers, 0), ord=2, axis=-1)
            asg, cost = np.argmin(d, -1), np.min(d, -1)
        elif method == 'MS':
            ms = cluster.MeanShift(n_jobs=self.num_workers)
            asg = ms.fit_predict(acts)
        elif method == 'SC':
            n_clusters = param_dict.pop('n_clusters', 25)
            sc = cluster.SpectralClustering(
                n_clusters=n_clusters, n_jobs=self.num_workers)
            asg = sc.fit_predict(acts)
        elif method == 'DB':
            eps = param_dict.pop('eps', 0.5)
            min_samples = param_dict.pop('min_samples', 20)
            sc = cluster.DBSCAN(eps, min_samples, n_jobs=self.num_workers)
            asg = sc.fit_predict(acts)
        else:
            raise ValueError('Invalid Clustering Method!')
        if centers is None:  ## If clustering returned cluster centers, use medoids
            centers = np.zeros((asg.max() + 1, acts.shape[1]))
            cost = np.zeros(len(acts))
            for cluster_label in range(asg.max() + 1):
                cluster_idxs = np.where(asg == cluster_label)[0]
                cluster_points = acts[cluster_idxs]
                pw_distances = metrics.euclidean_distances(cluster_points)
                centers[cluster_label] = cluster_points[np.argmin(
                    np.sum(pw_distances, -1))]
                cost[cluster_idxs] = np.linalg.norm(
                    acts[cluster_idxs] - np.expand_dims(centers[cluster_label], 0),
                    ord=2,
                    axis=-1)
        print('Created {} clusters'.format(len(np.unique(asg))))
        return asg, cost, centers

    def discover_concepts(self,
                          seg_method='slic',
                          seg_param_dicts=None,
                          cluster_method='KM',
                          cluster_param_dicts=None):
        if cluster_param_dicts is None:
            cluster_param_dicts = {}
        if set(cluster_param_dicts.keys()) != set(self.bottlenecks):
            for bn in self.bottlenecks:
                if bn not in cluster_param_dicts:
                    cluster_param_dicts[bn] = {}

        # First: create patches using segmentation
        self.create_patches(method=seg_method, param_dict=seg_param_dicts)

        # Then cluster and filtering activations for each bottleneck
        self.result = {}
        for bn in self.bottlenecks:
            bn_result = {}
            # Get activations
            bn_activations = self._patch_activations(self.dataset, bottleneck=bn)

            # Clustering
            bn_result['label'], bn_result['cost'], centers = self._cluster(
                bn_activations,
                cluster_method,
                cluster_param_dicts[bn])

            # Post-process clustering result
            concept_number, bn_result['concepts'] = 0, []

            # Consider each cluster
            for i in range(bn_result['label'].max() + 1):
                label_idxs = np.where(bn_result['label'] == i)[0]
                print(label_idxs)
                if len(label_idxs) > self.min_imgs:
                    # Comparing the diversity of the whole set with the closest patches
                    concept_costs = bn_result['cost'][label_idxs]
                    concept_idxs = label_idxs[np.argsort(concept_costs)[:self.max_imgs]]
                    concept_image_idxs = set(self.image_idxs[label_idxs])

                    discovery_size = len(self.discovery_images)
                    # Diversity of the closest patches is high compared to whole cluster
                    highly_common_concept = len(concept_image_idxs) > 0.5 * len(label_idxs)
                    # Diversity of the closest patches is mildly high compared to whole cluster
                    mildly_common_concept = len(concept_image_idxs) > 0.25 * len(label_idxs)
                    # Diversity of the closest patches is low compared to whole cluster
                    non_common_concept = len(concept_image_idxs) > 0.1 * len(label_idxs)
                    # The closest patches comes from more than 25% of images
                    mildly_populated_concept = len(concept_image_idxs) > 0.25 * discovery_size
                    # The closest patches comes from more than 50% of images
                    highly_populated_concept = len(concept_image_idxs) > 0.5 * discovery_size

                    cond2 = mildly_populated_concept and mildly_common_concept
                    cond3 = non_common_concept and highly_populated_concept
                    if highly_common_concept or cond2 or cond3:
                        concept_number += 1
                        concept = '{}_concept{}'.format(self.target_class, concept_number)
                        bn_result['concepts'].append(concept)
                        bn_result[concept] = {
                            'images': self.dataset[concept_idxs],
                            'patches': self.patches[concept_idxs],
                            'image_numbers': self.image_idxs[concept_idxs]
                        }
                        bn_result[concept + '_center'] = centers[i]
            bn_result.pop('label', None)
            bn_result.pop('cost', None)
            self.result[bn] = bn_result

    # def generate_tcav(self):
    #     self.tcavs = {}
    #     for bn in self.bottlenecks:
    #         self.tcavs



    def _random_concept_activation(self, bottleneck, random_concept):
        pass

    def _calculate_cav(self, c, r, bn, act_c, ow, directory=None):
        pass

    def _concept_cavs(self, bn, concept, activations, randoms=None, ow=True):
        pass

    def cavs(self, min_acc=0., ow=True):
        pass

    def load_cav_direction(self, c, r, bn, directory=None):
        pass

    def _sort_concepts(self, scores):
        pass

    def _return_gradients(self, images):
        pass

    def _tcav_score(self, bn, concept, rnd, gradients):
        pass

    def tcavs(self, test=False, sort=True, tcav_score_images=None):
        pass

    def do_statistical_testings(self, i_ups_concept, i_ups_random):
        pass

    def test_and_remove_concepts(self, tcav_scores):
        pass

    def delete_concept(self, bn, concept):
        pass

    def _concept_profile(self, bn, activations, concept, randoms):
        pass

    def fine_profile(self, bn, images, mean=True):
        pass
