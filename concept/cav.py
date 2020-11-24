from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path
import pickle
import numpy as np
from six.moves import range
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from concept import utils


class CAV(object):
    """CAV class contains methods for concept activation vector (CAV).

    CAV represents semantically meaningful vector directions in
    network's embeddings (bottlenecks).
    """

    @staticmethod
    def default_hparams():
        return utils.HParams(model_type='linear', alpha=.01, max_iter=1000, tol=1e-3)

    @staticmethod
    def load_cav(cav_path):
        with open(cav_path, 'rb') as pkl_file:
            save_dict = pickle.load(pkl_file)

        cav = CAV(save_dict['concepts'], save_dict['bottleneck'],
                  save_dict['hparams'], save_dict['saved_path'])
        cav.accuracies = save_dict['accuracies']
        cav.cavs = save_dict['cavs']
        return cav

    @staticmethod
    def cav_key(concepts, bottleneck, model_type, alpha):
        return '-' \
               ''.join([str(c) for c in concepts
                         ]) + '-' + bottleneck + '-' + model_type + '-' + str(alpha)

    @staticmethod
    def check_cav_exists(cav_dir, concepts, bottleneck, cav_hparams):
        cav_path = os.path.join(
            cav_dir,
            CAV.cav_key(concepts, bottleneck, cav_hparams.model_type,
                        cav_hparams.alpha) + '.pkl')
        return os.path.exists(cav_path)

    @staticmethod
    def _create_cav_training_set(concepts, bottleneck, acts):
        x = []
        labels = []
        labels2text = {}
        min_data_point = np.min(
            [len(acts[concept][bottleneck]) for concept in acts.keys()]
        )

        for i, concept in enumerate(concepts):
            x.extend(np.array(acts[concept][bottleneck][:min_data_point])
                     .reshape(min_data_point, -1))
            labels.extend([i] * min_data_point)
            labels2text[i] = concept

        x = np.array(x)
        labels = np.array(labels)

        return x, labels, labels2text

    @staticmethod
    def _train_lm(lm, x, y, labels2text):
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.33, stratify=y
        )
        lm.fit(x_train, y_train)
        y_pred = lm.predict(x_test)
        num_classes = max(y) + 1
        acc = {}
        num_correct = 0
        for class_id in range(num_classes):
            idx = (y_test == class_id)
            acc[labels2text[class_id]] = metrics.accuracy_score(
                y_pred[idx], y_test[idx])
            num_correct += (sum(idx) * acc[labels2text[class_id]])
        acc['overall'] = float(num_correct) / float(len(y_test))
        print('acc per class %s' % (str(acc)))
        return acc

    def __init__(self, concepts, bottleneck, hparams, save_path=None):
        self.concepts = concepts
        self.bottleneck = bottleneck
        self.hparams = hparams
        self.save_path = save_path

    def train(self, acts):
        print('training with alpha={}'.format(self.hparams.alpha))
        x, labels, labels2text = CAV._create_cav_training_set(
            self.concepts, self.bottleneck, acts
        )

        if self.hparams.model_type == 'linear':
            lm = linear_model.SGDClassifier(alpha=self.hparams.alpha,
                                            max_iter=self.hparams.max_iter,
                                            tol=self.hparams.tol)
        elif self.hparams.model_type == 'logistic':
            lm = linear_model.LogisticRegression()
        else:
            raise ValueError('Invalid hparams.model_type: {}'.format(
                self.hparams.model_type))

        self.accuracies = CAV._train_lm(lm, x, labels, labels2text)
        if len(lm.coef_) == 1:
            # if there were only two labels, the concept is assigned to label 0 by
            # default. So we flip the coef_ to reflect this.
            self.cavs = [-1 * lm.coef_[0], lm.coef_[0]]
        else:
            self.cavs = [c for c in lm.coef_]
        self._save_cavs()

    def perturb_act(self, act, concept, operation=np.add, alpha=1.0):
        flat_act = np.reshape(act, -1)
        pert = operation(flat_act, alpha * self.get_direction(concept))
        return np.reshape(pert, act.shape)

    def get_key(self):
        return CAV.cav_key(self.concepts, self.bottleneck, self.hparams.model_type,
                           self.hparams.alpha)

    def get_direction(self, concept):
        return self.cavs[self.concepts.index(concept)]

    def _save_cavs(self):
        save_dict = {
            'concepts': self.concepts,
            'bottleneck': self.bottleneck,
            'hparams': self.hparams,
            'accuracies': self.accuracies,
            'cavs': self.cavs,
            'saved_path': self.save_path
        }
        if self.save_path is not None:
            with open(self.save_path, 'wb') as pkl_file:
                pickle.dump(save_dict, pkl_file)
        else:
            print('save_path is None. Not saving anything')


def get_or_train_cav(concepts,
                     bottleneck,
                     acts,
                     cav_dir=None,
                     cav_hparams=None,
                     overwrite=False):
    if cav_hparams is None:
        cav_hparams = CAV.default_hparams()

    if cav_dir is not None:
        utils.make_dir_if_not_exists(cav_dir)
        cav_path = os.path.join(
            cav_dir,
            CAV.cav_key(concepts, bottleneck, cav_hparams.model_type,
                        cav_hparams.alpha).replace('/', '.') + '.pkl')
        if not overwrite and os.path.exists(cav_path):
            print('CAV already exists: {}'.format(cav_path))
            cav_instance = CAV.load_cav(cav_path)
            print('CAV accuracies: {}'.format(cav_instance.accuracies))
            return cav_instance
    else:
        raise ValueError('cav_dir must be specififed')

    print('Training CAV {} - {} alpha {}'.format(
        concepts, bottleneck, cav_hparams))
    cav_instance = CAV(concepts, bottleneck, cav_hparams, cav_path)
    cav_instance.train({c: acts[c] for c in concepts})
    print('CAV accuracies: {}'.format(cav_instance.accuracies))
    return cav_instance