from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import six
import numbers

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


def make_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)