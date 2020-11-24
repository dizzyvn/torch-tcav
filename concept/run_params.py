from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class RunParams(object):
    def __init__(self,
                 bottleneck,
                 concepts,
                 target_class,
                 activation_generator,
                 cav_dir,
                 alpha,
                 overwrite=True):
        self.bottleneck = bottleneck
        self.concepts = concepts
        self.target_class = target_class
        self.activation_generator = activation_generator
        self.cav_dir = cav_dir
        self.alpha = alpha
        self.overwrite = overwrite

    def get_key(self):
        return '_'.join([
            str(self.bottleneck), '_'.join(self.concepts), 'target_' + self.target_class, 'alpha_' + str(self.alpha)])
